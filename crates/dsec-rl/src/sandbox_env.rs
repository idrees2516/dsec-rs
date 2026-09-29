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

use std::sync::Arc;

use dsec_protocol::frame::Channel;
use dsec_protocol::message::{Request, Response};
use dsec_runtime::backend::SandboxSpec as RuntimeSpec;
use dsec_runtime::{AetherClient, EdgeNode};
use dsec_storage::latency::NodeLatencyProfile;

use crate::envpool::{Env, StepResult};
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

    /// Creates N sandbox envs (each with its own sandbox).
    pub fn build_envs(&self, n: usize, seed: u64) -> crate::error::Result<Vec<Box<dyn Env>>> {
        let mut envs: Vec<Box<dyn Env>> = Vec::with_capacity(n);
        for i in 0..n {
            let sid = self
                .runtime
                .block_on(async { self.node.create(RuntimeSpec::default()).await })
                .map_err(|e| crate::error::Error::Sandbox(e.to_string()))?
                .sid;
            envs.push(Box::new(TerminalTaskEnv {
                handle: self.runtime.handle().clone(),
                client: self.client.clone(),
                sid,
                rng: Rng::new(seed + i as u64),
                flag_file: 0,
                flag: String::new(),
                t: 0,
                last_stdout: Vec::new(),
                solved: false,
                max_steps: 24,
                episode_return: 0.0,
            }));
        }
        Ok(envs)
    }
}

/// One sandbox-backed terminal task env.
pub struct TerminalTaskEnv {
    handle: tokio::runtime::Handle,
    client: Arc<AetherClient>,
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

impl TerminalTaskEnv {
    async fn exec(&self, cmd: &str) -> (Vec<u8>, i32) {
        match self
            .client
            .call(
                self.sid,
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
        // Re-stage the task: TASK_FILES files, one hides a fresh flag.
        self.flag_file = self.rng.below(TASK_FILES);
        self.flag = format!("FLAG-{}", self.rng.next_u64());
        self.t = 0;
        self.solved = false;
        self.last_stdout.clear();
        self.episode_return = 0.0;
        let cmd = format!(
            "mkdir -p /task && rm -r /task && mkdir -p /task && echo decoy-{} > /task/f0.txt && echo decoy-{} > /task/f1.txt && echo decoy-{} > /task/f2.txt && echo {} > /task/f{}.txt",
            self.rng.next_u64() % 1000,
            self.rng.next_u64() % 1000,
            self.rng.next_u64() % 1000,
            self.flag,
            self.flag_file
        );
        // Staging runs as a single compound line; our interpreter runs
        // commands per line, so split on " && ".
        for line in cmd.split(" && ") {
            let _ = self.handle.block_on(self.exec(line.trim()));
        }
        let _ = self.handle.block_on(self.exec("echo ready"));
        encode_obs(b"ready\n", self.t, self.solved)
    }

    fn step(&mut self, action: &Action) -> StepResult {
        self.t += 1;
        let idx = action.as_index().max(0) as usize;
        let (stdout, _code) = if idx == 0 {
            self.handle.block_on(self.exec("ls /task"))
        } else if idx <= TASK_FILES {
            self.handle
                .block_on(self.exec(&format!("cat /task/f{}.txt", idx - 1)))
        } else {
            self.handle.block_on(
                self.exec("grep FLAG /task/f0.txt /task/f1.txt /task/f2.txt /task/f3.txt"),
            )
        };
        self.last_stdout = stdout.clone();

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
}
