//! Agent sandboxes as RL environments — the paper's core use case.
//!
//! [`TerminalTaskEnv`] gives each env its own sandbox on a shared node.
//! The observation is a fixed-size encoding of the last command output;
//! the action space selects shell commands (`ls`, `cat <file>`,
//! `grep FLAG`). Solving the task (finding the flag hidden in the
//! sandbox's filesystem) exercises the full Chronus data plane per step.
//!
//! A dedicated multi-thread tokio runtime hosts the Aether server; env
//! steps bridge with `Handle::block_on` from the pool's worker threads.
//!
//! Two execution paths share one pure state machine ([`TerminalCore`]):
//!
//! * `EnvPool` + [`TerminalTaskEnv`] — the reference path: each env
//!   steps through its own `call` round trip.
//! * [`BatchedSandboxEnvPool`] — the throughput path: one vectorized
//!   step issues every env's exec as ONE pipelined
//!   [`AetherClient::call_batch`] transmission (one wakeup chain, one
//!   timer), folding results locally. Per-env command order is
//!   preserved; staging resets run phase-barrier pipelined. Both paths
//!   produce byte-identical command/observation/reward sequences
//!   (regression-tested).

use std::sync::Arc;

use dsec_protocol::frame::Channel;
use dsec_protocol::message::{Request, Response};
use dsec_runtime::backend::SandboxSpec as RuntimeSpec;
use dsec_runtime::{AetherClient, EdgeNode};
use dsec_storage::latency::NodeLatencyProfile;

use crate::envpool::{Env, StepResult, Stepping};
use crate::spaces::{Action, ActionSpace, ObsSpace};
use dsec_protocol::rng::Rng;

/// Fixed observation size (feature encoding of last stdout).
pub const TERM_OBS_SIZE: usize = 16;
/// Number of files staged in /task.
pub const TASK_FILES: usize = 4;

/// Builds a pool of sandbox envs sharing one node + Aether connection.
pub struct SandboxEnvBuilder {
    node: Arc<EdgeNode>,
    runtime: Arc<tokio::runtime::Runtime>,
    client: Arc<AetherClient>,
    _serve: tokio::task::JoinHandle<()>,
}

impl SandboxEnvBuilder {
    /// Creates the node, the runtime and the data-plane connection.
    pub fn new(seed: u64) -> Self {
        let image = Arc::new(dsec_storage::erofs::ErofsImageBuilder::agent_base().build());
        let mut registry = dsec_storage::erofs::ImageRegistry::default();
        registry.register(
            image,
            Arc::new(dsec_storage::cache::LruBlockCache::new(1024)),
            NodeLatencyProfile::zero().block_fetch,
        );
        let node = EdgeNode::new(
            "rl-node".to_string(),
            Arc::new(registry),
            NodeLatencyProfile::zero(),
            1_000_000,
            1_000_000,
            100_000,
            seed,
        );
        node.factory.prewarm("dsec/agent-base", 16);
        let node = Arc::new(node);

        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap(),
        );
        let (client_conn, server_conn) = dsec_runtime::aether::channel_pair(1024);
        // Enter the runtime context so the client's reader task attaches
        // to THIS runtime (we are called from a plain thread).
        let _guard = runtime.enter();
        let serve = {
            let node = node.clone();
            runtime.spawn(async move {
                dsec_runtime::aether::serve_connection(node, server_conn).await;
            })
        };
        let client = AetherClient::new(client_conn);
        drop(_guard);
        SandboxEnvBuilder {
            node,
            runtime,
            client: Arc::new(client),
            _serve: serve,
        }
    }

    /// The shared data-plane client (diagnostics: `call_stats`).
    pub fn client(&self) -> &Arc<AetherClient> {
        &self.client
    }

    async fn create_sandbox(&self) -> crate::error::Result<u64> {
        self.node
            .create(RuntimeSpec::default())
            .await
            .map_err(|e| crate::error::Error::Sandbox(e.to_string()))
            .map(|e| e.sid)
    }

    /// Creates N sandbox envs (each with its own sandbox) — the
    /// reference per-env stepping path.
    pub fn build_envs(&self, n: usize, seed: u64) -> crate::error::Result<Vec<Box<dyn Env>>> {
        let mut envs: Vec<Box<dyn Env>> = Vec::with_capacity(n);
        for i in 0..n {
            let sid = self.runtime.block_on(self.create_sandbox())?;
            envs.push(Box::new(TerminalTaskEnv {
                handle: self.runtime.handle().clone(),
                client: self.client.clone(),
                core: TerminalCore::new(sid, seed + i as u64),
            }));
        }
        Ok(envs)
    }

    /// Creates N sandbox envs stepped as ONE pipelined data-plane tick —
    /// the throughput path ([`BatchedSandboxEnvPool`]).
    pub fn build_batch_pool(
        &self,
        n: usize,
        seed: u64,
    ) -> crate::error::Result<BatchedSandboxEnvPool> {
        let mut cores = Vec::with_capacity(n);
        for i in 0..n {
            let sid = self.runtime.block_on(self.create_sandbox())?;
            cores.push(TerminalCore::new(sid, seed + i as u64));
        }
        Ok(BatchedSandboxEnvPool {
            handle: self.runtime.handle().clone(),
            client: self.client.clone(),
            cores,
        })
    }
}

// ---------------------------------------------------------------------------
// Shared pure state machine
// ---------------------------------------------------------------------------

/// Terminal-task state machine — shared verbatim by both execution paths
/// so command, reward, and observation sequences are identical.
struct TerminalCore {
    sid: u64,
    rng: Rng,
    /// Index of the file hiding the flag.
    flag_file: usize,
    flag: String,
    t: u64,
    last_stdout: Vec<u8>,
    solved: bool,
    max_steps: u64,
    episode_return: f32,
}

impl TerminalCore {
    fn new(sid: u64, seed: u64) -> Self {
        TerminalCore {
            sid,
            rng: Rng::new(seed),
            flag_file: 0,
            flag: String::new(),
            t: 0,
            last_stdout: Vec::new(),
            solved: false,
            max_steps: 24,
            episode_return: 0.0,
        }
    }

    /// The exec issued for one action (pure).
    fn step_command(&self, idx: usize) -> String {
        if idx == 0 {
            "ls /task".to_string()
        } else if idx <= TASK_FILES {
            format!("cat /task/f{}.txt", idx - 1)
        } else {
            "grep FLAG /task/f0.txt /task/f1.txt /task/f2.txt /task/f3.txt".to_string()
        }
    }

    /// Starts a fresh episode and returns the staging exec lines in
    /// per-env execution order (RNG consumption order is fixed, so both
    /// paths stage identical episodes).
    ///
    /// Staging is ONE compound `&&` line (the interpreter chains
    /// segments with POSIX `sh -c` semantics) plus a readiness probe —
    /// two execs instead of eight per reset.
    fn begin_reset(&mut self) -> Vec<String> {
        self.flag_file = self.rng.below(TASK_FILES);
        self.flag = format!("FLAG-{}", self.rng.next_u64());
        self.t = 0;
        self.solved = false;
        self.last_stdout.clear();
        self.episode_return = 0.0;
        let staging = format!(
            "mkdir -p /task && rm -r /task && mkdir -p /task && echo decoy-{} > /task/f0.txt && echo decoy-{} > /task/f1.txt && echo decoy-{} > /task/f2.txt && echo {} > /task/f{}.txt",
            self.rng.next_u64() % 1000,
            self.rng.next_u64() % 1000,
            self.rng.next_u64() % 1000,
            self.flag,
            self.flag_file
        );
        vec![staging, "echo ready".to_string()]
    }

    /// Fresh-episode observation (after staging completes).
    fn reset_obs(&self) -> Vec<f32> {
        encode_obs(b"ready\n", self.t, self.solved)
    }

    /// Folds one exec output into a step result (pure except the
    /// deterministic state update).
    fn apply_step(&mut self, idx: usize, stdout: Vec<u8>) -> StepResult {
        self.t += 1;
        self.last_stdout = stdout;
        let found = String::from_utf8_lossy(&self.last_stdout).contains(&self.flag);
        if found {
            self.solved = true;
        }
        let done = self.solved || self.t >= self.max_steps;
        // Reward: solve fast; small step penalty; grep hint if flag
        // content leaked partially.
        let reward = if found {
            1.0 - 0.02 * (self.t as f32 - 1.0)
        } else if idx > TASK_FILES && String::from_utf8_lossy(&self.last_stdout).contains("FLAG-") {
            0.1
        } else {
            -0.01
        };
        self.episode_return += reward;
        let obs = encode_obs(&self.last_stdout, self.t, self.solved);
        StepResult {
            obs,
            reward,
            done,
            episode_len: self.t,
            episode_return: self.episode_return,
        }
    }
}

/// One exec round trip (shared by the per-env path).
async fn exec_line(client: &Arc<AetherClient>, sid: u64, cmd: &str) -> (Vec<u8>, i32) {
    match client
        .call(
            sid,
            Channel::Exec,
            Request::Exec {
                cmd: cmd.to_string(),
                timeout_ms: Some(5000),
            },
        )
        .await
    {
        Ok(Response::Exec {
            exit_code, stdout, ..
        }) => (stdout, exit_code),
        _ => (Vec::new(), -1),
    }
}

// ---------------------------------------------------------------------------
// Reference path: one sandbox env behind the sync `Env` contract
// ---------------------------------------------------------------------------

/// One sandbox-backed terminal task env.
pub struct TerminalTaskEnv {
    handle: tokio::runtime::Handle,
    client: Arc<AetherClient>,
    core: TerminalCore,
}

impl TerminalTaskEnv {
    async fn exec(&self, cmd: &str) -> (Vec<u8>, i32) {
        exec_line(&self.client, self.core.sid, cmd).await
    }
}

impl Env for TerminalTaskEnv {
    fn obs_space(&self) -> ObsSpace {
        ObsSpace::Flat {
            size: TERM_OBS_SIZE,
        }
    }

    fn action_space(&self) -> ActionSpace {
        // 0 = ls /task, 1..=TASK_FILES = cat file, TASK_FILES+1 = grep.
        ActionSpace::discrete(TASK_FILES + 2)
    }

    fn reset(&mut self) -> Vec<f32> {
        for line in self.core.begin_reset() {
            let _ = self.handle.block_on(self.exec(&line));
        }
        self.core.reset_obs()
    }

    fn step(&mut self, action: &Action) -> StepResult {
        let idx = action.as_index().max(0) as usize;
        let cmd = self.core.step_command(idx);
        let (stdout, _code) = self.handle.block_on(self.exec(&cmd));
        self.core.apply_step(idx, stdout)
    }
}

// ---------------------------------------------------------------------------
// Throughput path: the whole vectorized step as one pipelined tick
// ---------------------------------------------------------------------------

/// A vectorized sandbox env pool that steps every env in ONE pipelined
/// data-plane round trip ([`AetherClient::call_batch`]).
///
/// Per tick the client issues every env's exec as a single coalesced
/// transmission and awaits all replies under one deadline — one wakeup
/// chain and one timer per tick instead of per env. This is exactly the
/// workload Aether's per-connection multiplexing was designed for (the
/// paper multiplexes every sandbox of a client over one connection), so
/// the wire behavior is unchanged; only the client-side issue pattern
/// becomes pipelined.
///
/// Semantics are identical to stepping the same envs through [`EnvPool`]
/// (same commands, same per-env ordering, same rewards/observations) —
/// enforced by a cross-validation regression test. Staging resets run
/// phase-barrier pipelined: env i's line k completes before any env's
/// line k+1 is issued, so per-env ordering is preserved while different
/// sandboxes execute concurrently.
pub struct BatchedSandboxEnvPool {
    handle: tokio::runtime::Handle,
    client: Arc<AetherClient>,
    cores: Vec<TerminalCore>,
}

impl BatchedSandboxEnvPool {
    pub fn num_envs(&self) -> usize {
        self.cores.len()
    }

    pub fn obs_space(&self) -> ObsSpace {
        ObsSpace::Flat {
            size: TERM_OBS_SIZE,
        }
    }

    pub fn action_space(&self) -> ActionSpace {
        ActionSpace::discrete(TASK_FILES + 2)
    }

    /// Resets every env; returns initial observations (env order).
    pub fn reset_all(&mut self) -> Vec<Vec<f32>> {
        let lists: Vec<Vec<String>> = self.cores.iter_mut().map(|c| c.begin_reset()).collect();
        let sids: Vec<u64> = self.cores.iter().map(|c| c.sid).collect();
        self.handle
            .block_on(run_phased(&self.client, &sids, &lists));
        self.cores.iter().map(|c| c.reset_obs()).collect()
    }

    /// Steps every env with one action vector — the pipelined fast path.
    pub fn step_all(&mut self, actions: &[Action]) -> crate::error::Result<Vec<StepResult>> {
        if actions.len() != self.cores.len() {
            return Err(crate::error::Error::Dim {
                what: "actions",
                expected: self.cores.len(),
                actual: actions.len(),
            });
        }
        // Phase 1 (pure): command per env.
        let idxs: Vec<usize> = actions
            .iter()
            .map(|a| a.as_index().max(0) as usize)
            .collect();
        let cmds: Vec<String> = self
            .cores
            .iter()
            .zip(idxs.iter())
            .map(|(c, &i)| c.step_command(i))
            .collect();
        // Phase 2: ONE pipelined batch — one block_on, one lock pass,
        // one transmission, one deadline for every env in the tick.
        let responses = self.handle.block_on(async {
            let calls: Vec<(u64, Channel, Request)> = self
                .cores
                .iter()
                .zip(cmds)
                .map(|(c, cmd)| {
                    (
                        c.sid,
                        Channel::Exec,
                        Request::Exec {
                            cmd,
                            timeout_ms: Some(5000),
                        },
                    )
                })
                .collect();
            self.client.call_batch(calls).await
        });
        // Phase 3 (pure): fold results, remembering done envs.
        let mut results: Vec<StepResult> = Vec::with_capacity(self.cores.len());
        let mut done: Vec<usize> = Vec::new();
        for (i, r) in responses.into_iter().enumerate() {
            let (stdout, _code) = match r {
                Ok(Response::Exec {
                    exit_code, stdout, ..
                }) => (stdout, exit_code),
                _ => (Vec::new(), -1),
            };
            let res = self.cores[i].apply_step(idxs[i], stdout);
            if res.done {
                done.push(i);
            }
            results.push(res);
        }
        // Phase 4 — `reset_if_done`: staged resets for every done env,
        // pipelined together (phase-barrier preserves per-env order).
        if !done.is_empty() {
            let lists: Vec<Vec<String>> =
                done.iter().map(|&i| self.cores[i].begin_reset()).collect();
            let sids: Vec<u64> = done.iter().map(|&i| self.cores[i].sid).collect();
            self.handle
                .block_on(run_phased(&self.client, &sids, &lists));
            for &i in &done {
                results[i].obs = self.cores[i].reset_obs();
            }
        }
        Ok(results)
    }
}

impl Stepping for BatchedSandboxEnvPool {
    fn num_envs(&self) -> usize {
        BatchedSandboxEnvPool::num_envs(self)
    }

    fn reset_all(&mut self) -> Vec<Vec<f32>> {
        BatchedSandboxEnvPool::reset_all(self)
    }

    fn step_parallel(&mut self, actions: &[Action]) -> crate::error::Result<Vec<StepResult>> {
        self.step_all(actions)
    }
}

/// Executes per-env command lists with a phase barrier: env i's line k
/// completes before any env's line k+1 is issued. Per-env ordering is
/// therefore exact while distinct sandboxes run concurrently. Output is
/// discarded (staging results are not used by either path).
async fn run_phased(client: &Arc<AetherClient>, sids: &[u64], lists: &[Vec<String>]) {
    let max_lines = lists.iter().map(Vec::len).max().unwrap_or(0);
    for phase in 0..max_lines {
        let mut calls: Vec<(u64, Channel, Request)> = Vec::with_capacity(sids.len());
        for (sid, list) in sids.iter().zip(lists.iter()) {
            if let Some(line) = list.get(phase) {
                calls.push((
                    *sid,
                    Channel::Exec,
                    Request::Exec {
                        cmd: line.clone(),
                        timeout_ms: Some(5000),
                    },
                ));
            }
        }
        if calls.is_empty() {
            continue;
        }
        let _ = client.call_batch(calls).await;
    }
}

/// Deterministic fixed-size observation encoding of the last output.
fn encode_obs(stdout: &[u8], t: u64, solved: bool) -> Vec<f32> {
    let mut obs = vec![0.0f32; TERM_OBS_SIZE];
    if !stdout.is_empty() {
        // Feature 0: output length (log-scaled, clamped).
        obs[0] = ((stdout.len() as f32) + 1.0).ln() / 8.0;
        // Features 1..8: first bytes normalized.
        for (i, &b) in stdout.iter().take(8).enumerate() {
            obs[1 + i] = (b as f32) / 255.0;
        }
        // Features 9..15: byte histogram sketch (checksum-ish).
        let mut acc: u32 = 0;
        for &b in stdout.iter().take(64) {
            acc = acc.rotate_left(3) ^ b as u32;
        }
        for j in 0..7 {
            obs[9 + j] = ((acc >> (j * 4)) & 0xF) as f32 / 15.0;
        }
    }
    obs[15] = if solved { 1.0 } else { 0.0 };
    let _ = t;
    obs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_obs_deterministic_and_bounded() {
        let a = encode_obs(b"hello world", 3, false);
        let b = encode_obs(b"hello world", 4, false);
        assert_eq!(a, b);
        assert!(a.iter().all(|&v| v.abs() <= 2.0));
        let solved = encode_obs(b"FLAG", 1, true);
        assert_eq!(solved[15], 1.0);
        let empty = encode_obs(b"", 0, false);
        assert_eq!(empty[0], 0.0);
    }

    #[test]
    fn sandbox_env_solves_task() {
        let builder = SandboxEnvBuilder::new(7);
        let mut envs = builder.build_envs(2, 7).unwrap();
        let mut env = envs.remove(0);
        let _ = env.reset();
        // Walk all files; one holds the flag.
        let mut steps = 0;
        let mut solved = false;
        for idx in 1..=TASK_FILES {
            let r = env.step(&Action::Discrete(idx as i64));
            steps += 1;
            if r.done {
                solved = true;
                assert!(r.reward > 0.5, "reward {}", r.reward);
                break;
            }
        }
        assert!(solved, "did not solve in {} steps", steps);
    }

    #[test]
    fn sandbox_env_timeout_ends_episode() {
        let builder = SandboxEnvBuilder::new(11);
        let mut envs = builder.build_envs(1, 11).unwrap();
        let mut env = envs.remove(0);
        let _ = env.reset();
        let mut dones = 0;
        for t in 0..30 {
            let r = env.step(&Action::Discrete(0)); // ls forever
            if r.done {
                dones += 1;
                assert_eq!(r.episode_len, 24);
                break;
            }
            let _ = t;
        }
        assert_eq!(dones, 1);
    }

    /// Cross-validation: the pipelined batch pool must produce
    /// byte-identical step sequences to the reference per-env path,
    /// including resets triggered by `done` (episodes run to completion
    /// and past it, so resets fire).
    #[test]
    fn batch_pool_matches_reference_path() {
        let n = 6usize;
        let mut rng_actions = dsec_protocol::rng::Rng::new(999);
        let mut reference = {
            let builder = SandboxEnvBuilder::new(5);
            let envs = builder.build_envs(n, 5).unwrap();
            crate::envpool::EnvPool::new(envs).with_workers(3)
        };
        let mut batched = {
            let builder = SandboxEnvBuilder::new(5);
            builder.build_batch_pool(n, 5).unwrap()
        };
        let a = reference.reset_all();
        let b = batched.reset_all();
        assert_eq!(a, b, "initial observations diverge");
        for t in 0..60 {
            // Mixed action schedule: searches, cats, greps, ls.
            let actions: Vec<Action> = (0..n)
                .map(|i| {
                    Action::Discrete(((i as u64 + t + rng_actions.below(3) as u64) % 7) as i64)
                })
                .collect();
            let ra = reference.step_parallel(&actions).unwrap();
            let rb = batched.step_all(&actions).unwrap();
            for i in 0..n {
                assert_eq!(ra[i], rb[i], "tick {t} env {i} diverged");
            }
        }
    }

    #[test]
    fn batch_pool_solves_task() {
        let builder = SandboxEnvBuilder::new(17);
        let mut pool = builder.build_batch_pool(4, 17).unwrap();
        let _ = pool.reset_all();
        let mut solved_any = false;
        for t in 0..24 {
            let actions: Vec<Action> = (0..4)
                .map(|i| Action::Discrete((1 + ((i + t as usize) % TASK_FILES)) as i64))
                .collect();
            let results = pool.step_all(&actions).unwrap();
            for (i, r) in results.iter().enumerate() {
                if r.done {
                    // reset_if_done: terminal obs replaced by fresh obs.
                    assert!(!r.obs.is_empty(), "env {i}");
                    solved_any = true;
                }
            }
        }
        assert!(solved_any, "no episode terminated");
    }

    #[test]
    fn batch_pool_dim_mismatch_rejected() {
        let builder = SandboxEnvBuilder::new(3);
        let mut pool = builder.build_batch_pool(2, 3).unwrap();
        assert!(pool.step_all(&[Action::Discrete(0)]).is_err());
    }
}
