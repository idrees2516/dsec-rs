//! Vectorized environment pools — the port of PufferLib's `PufferEnv`
//! layer.
//!
//! `EnvPool` steps N environments with one action vector. Semantics
//! follow pufferlib's `reset_if_done`: when an env returns `done`, the
//! result carries the terminal observation AND the env is immediately
//! reset, so the *next* step belongs to a fresh episode.
//!
//! `step_parallel` distributes envs across **persistent** worker threads
//! (spawned once on first use, parked between steps) instead of spawning
//! scoped threads per step — per-step OS thread creation dominated the
//! stepping cost (~100us per step on a small box; the persistent pool
//! amortizes thread creation to zero). Chunk assignment is fixed and each
//! env's result is computed independently, so results stay in env order
//! regardless of thread scheduling (deterministic, equal to the
//! sequential path — regression-tested).

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::error::{Error, Result};
use crate::spaces::{Action, ActionSpace, ObsSpace};

/// Per-step result for one env.
#[derive(Debug, Clone, PartialEq)]
pub struct StepResult {
    pub obs: Vec<f32>,
    pub reward: f32,
    pub done: bool,
    /// Info: episode length and return at termination.
    pub episode_len: u64,
    pub episode_return: f32,
}

/// The environment contract (sync — sandbox envs bridge with `block_on`).
pub trait Env: Send {
    fn obs_space(&self) -> ObsSpace;
    fn action_space(&self) -> ActionSpace;
    fn reset(&mut self) -> Vec<f32>;
    fn step(&mut self, action: &Action) -> StepResult;
}

/// What a worker produced for its chunk during one dispatch.
enum TaskOutput {
    Steps(Vec<StepResult>),
    Resets(Vec<Vec<f32>>),
    Empty,
}

/// One broadcast task: actions for every env (workers slice their own
/// range), or a reset pass.
#[derive(Clone)]
struct Command {
    actions: Arc<Vec<Action>>,
    reset: bool,
}

/// Broadcast coordination: `task` holds the current generation, its
/// payload, and the completion count under ONE mutex, so a worker
/// always reads a (generation, payload) pair that belongs together
/// (observing generation g while reading payload g+1 is impossible).
/// Workers park on `idle_cv`; the dispatcher parks on `progress`.
struct TaskState {
    generation: u64,
    cmd: Command,
    /// Number of workers that finished the in-flight task.
    done_count: u64,
    poisoned: bool,
}

struct Coord {
    task: Mutex<TaskState>,
    /// Workers park here while no new generation is pending.
    idle_cv: Condvar,
    /// Main parks here until `done_count` reaches the worker count.
    progress: Condvar,
    shutdown: AtomicBool,
}

impl Coord {
    fn new() -> Self {
        Coord {
            task: Mutex::new(TaskState {
                generation: 0,
                cmd: Command {
                    actions: Arc::new(Vec::new()),
                    reset: false,
                },
                done_count: 0,
                poisoned: false,
            }),
            idle_cv: Condvar::new(),
            progress: Condvar::new(),
            shutdown: AtomicBool::new(false),
        }
    }
}

fn process_chunk(envs: &mut [Box<dyn Env>], base: usize, cmd: &Command) -> TaskOutput {
    if cmd.reset {
        TaskOutput::Resets(envs.iter_mut().map(|e| e.reset()).collect())
    } else {
        // Bind the action slice before the mutable env iteration so the
        // two borrows never overlap.
        let my_actions = &cmd.actions[base..base + envs.len()];
        let steps = envs
            .iter_mut()
            .zip(my_actions.iter())
            .map(|(e, a)| {
                let mut r = e.step(a);
                if r.done {
                    // reset_if_done: the terminal observation is replaced
                    // by the fresh episode's first observation.
                    r.obs = e.reset();
                }
                r
            })
            .collect();
        TaskOutput::Steps(steps)
    }
}

fn worker_loop(
    coord: Arc<Coord>,
    mut envs: Vec<Box<dyn Env>>,
    out: Arc<Mutex<TaskOutput>>,
    base: usize,
) {
    let mut seen = 0u64;
    loop {
        // Atomically observe (generation, payload): the pair is read under
        // the task lock, so it can never straddle a dispatch.
        let cmd = {
            let mut task = coord.task.lock().expect("pool task poisoned");
            loop {
                if coord.shutdown.load(Ordering::Relaxed) {
                    return;
                }
                if task.generation > seen {
                    seen = task.generation;
                    break task.cmd.clone();
                }
                task = coord.idle_cv.wait(task).expect("pool wait poisoned");
            }
        };
        // Run the chunk; a panicking env must not kill the whole pool
        // silently — it flags the pool poisoned and stops that worker.
        let outcome = catch_unwind(AssertUnwindSafe(|| process_chunk(&mut envs, base, &cmd)));
        let failed = outcome.is_err();
        *out.lock().expect("pool output poisoned") = outcome.unwrap_or(TaskOutput::Empty);
        {
            let mut task = coord.task.lock().expect("pool task poisoned");
            task.done_count += 1;
            if failed {
                task.poisoned = true;
            }
        }
        coord.progress.notify_all();
        if failed {
            return;
        }
    }
}

struct Worker {
    /// Env chunk; empty once ownership moved into the worker thread.
    envs: Vec<Box<dyn Env>>,
    out: Arc<Mutex<TaskOutput>>,
    handle: Option<JoinHandle<()>>,
}

/// A vectorized pool.
pub struct EnvPool {
    workers: Vec<Worker>,
    n_envs: usize,
    n_workers: usize,
    coord: Arc<Coord>,
    /// Cached before envs are handed to worker threads.
    obs_space: Option<ObsSpace>,
    action_space: Option<ActionSpace>,
}

impl EnvPool {
    pub fn new(envs: Vec<Box<dyn Env>>) -> Self {
        let n_envs = envs.len();
        let (obs_space, action_space) = match envs.first() {
            Some(e) => (Some(e.obs_space()), Some(e.action_space())),
            None => (None, None),
        };
        EnvPool {
            workers: vec![Worker {
                envs,
                out: Arc::new(Mutex::new(TaskOutput::Empty)),
                handle: None,
            }],
            n_envs,
            n_workers: 1,
            coord: Arc::new(Coord::new()),
            obs_space,
            action_space,
        }
    }

    pub fn with_workers(mut self, workers: usize) -> Self {
        self.n_workers = workers.max(1);
        self
    }

    pub fn num_envs(&self) -> usize {
        self.n_envs
    }

    pub fn obs_space(&self) -> ObsSpace {
        self.obs_space.unwrap_or(ObsSpace::Flat { size: 0 })
    }

    pub fn action_space(&self) -> ActionSpace {
        self.action_space
            .clone()
            .unwrap_or_else(|| ActionSpace::discrete(1))
    }

    fn is_parallel(&self) -> bool {
        self.workers[0].handle.is_some()
    }

    /// Splits the sequential chunk into one chunk per worker and spawns
    /// the persistent workers. No-op when already parallel or when
    /// parallelism cannot help (single worker, fewer than two envs).
    fn ensure_parallel(&mut self) {
        if self.n_workers <= 1 || self.n_envs < 2 || self.is_parallel() {
            return;
        }
        let n = self.n_workers.min(self.n_envs);
        let chunk = self.n_envs.div_ceil(n);
        let mut all = std::mem::take(&mut self.workers[0].envs);
        let mut workers = Vec::with_capacity(n);
        let mut base = 0usize;
        let mut idx = 0usize;
        while !all.is_empty() {
            let take = chunk.min(all.len());
            // split_off returns the tail; the head stays in `all` — swap so
            // `chunk_envs` holds this worker's slice and `all` the rest.
            let rest = all.split_off(take);
            let chunk_envs = std::mem::replace(&mut all, rest);
            let out = Arc::new(Mutex::new(TaskOutput::Empty));
            let handle = std::thread::Builder::new()
                .name(format!("dsec-envpool-{}", idx))
                .spawn({
                    let coord = self.coord.clone();
                    let out = out.clone();
                    move || worker_loop(coord, chunk_envs, out, base)
                })
                .expect("spawn envpool worker");
            workers.push(Worker {
                envs: Vec::new(),
                out,
                handle: Some(handle),
            });
            base += take;
            idx += 1;
        }
        self.workers = workers;
    }

    /// Resets every env; returns initial observations.
    pub fn reset_all(&mut self) -> Vec<Vec<f32>> {
        match self.run_command(Command {
            actions: Arc::new(Vec::new()),
            reset: true,
        }) {
            Ok(TaskOutput::Resets(v)) => v,
            Ok(_) => Vec::new(),
            Err(_) => Vec::new(),
        }
    }

    /// Stepping (deterministic env order).
    ///
    /// Pufferlib `reset_if_done`: when an env reports `done`, the env is
    /// reset immediately and the result carries the NEW episode's first
    /// observation (reward/done describe the finished episode).
    pub fn step(&mut self, actions: &[Action]) -> Result<Vec<StepResult>> {
        if actions.len() != self.n_envs {
            return Err(Error::Dim {
                what: "actions",
                expected: self.n_envs,
                actual: actions.len(),
            });
        }
        match self.run_command(Command {
            actions: Arc::new(actions.to_vec()),
            reset: false,
        })? {
            TaskOutput::Steps(v) => Ok(v),
            _ => Err(Error::Pool("worker returned wrong output type".into())),
        }
    }

    /// Parallel stepping across persistent workers. Results preserve env
    /// order.
    pub fn step_parallel(&mut self, actions: &[Action]) -> Result<Vec<StepResult>> {
        self.ensure_parallel();
        self.step(actions)
    }

    fn run_command(&mut self, cmd: Command) -> Result<TaskOutput> {
        if self.is_parallel() {
            self.run_parallel(cmd)
        } else {
            self.run_sequential(cmd)
        }
    }

    fn run_sequential(&mut self, cmd: Command) -> Result<TaskOutput> {
        let w = &mut self.workers[0];
        // The sequential pool keeps envs on this thread; no coordination.
        Ok(process_chunk(&mut w.envs, 0, &cmd))
    }

    fn run_parallel(&mut self, cmd: Command) -> Result<TaskOutput> {
        let n = self.workers.len() as u64;
        // Publish the new generation and reset the completion counter
        // atomically: workers can only observe the pair together.
        {
            let mut task = self.coord.task.lock().expect("pool task poisoned");
            if task.poisoned {
                return Err(Error::Pool("a worker panicked earlier".into()));
            }
            task.done_count = 0;
            task.generation += 1;
            task.cmd = cmd;
        }
        self.coord.idle_cv.notify_all();
        {
            let mut task = self.coord.task.lock().expect("pool task poisoned");
            while task.done_count < n {
                if task.poisoned {
                    return Err(Error::Pool("worker panicked during step".into()));
                }
                task = self
                    .coord
                    .progress
                    .wait(task)
                    .expect("pool progress poisoned");
            }
            if task.poisoned {
                return Err(Error::Pool("worker panicked during step".into()));
            }
        }
        // Collect chunk outputs in worker (= env) order.
        let mut steps: Option<Vec<StepResult>> = None;
        let mut resets: Option<Vec<Vec<f32>>> = None;
        for w in &self.workers {
            let taken = std::mem::replace(
                &mut *w.out.lock().expect("pool output poisoned"),
                TaskOutput::Empty,
            );
            match (taken, &mut steps, &mut resets) {
                (TaskOutput::Steps(v), Some(acc), _) => acc.extend(v),
                (TaskOutput::Steps(v), None, None) => steps = Some(v),
                (TaskOutput::Resets(v), _, Some(acc)) => acc.extend(v),
                (TaskOutput::Resets(v), None, None) => resets = Some(v),
                (TaskOutput::Empty, _, _) => {
                    return Err(Error::Pool("worker produced no output".into()))
                }
                _ => return Err(Error::Pool("mixed worker outputs".into())),
            }
        }
        Ok(match (steps, resets) {
            (Some(s), None) => TaskOutput::Steps(s),
            (None, Some(r)) => TaskOutput::Resets(r),
            _ => return Err(Error::Pool("incomplete worker outputs".into())),
        })
    }
}

impl Drop for EnvPool {
    fn drop(&mut self) {
        // Flip shutdown while holding the task lock (workers check it
        // under the same lock), then wake every parked worker so they exit.
        {
            let _task = self.coord.task.lock().expect("pool task poisoned");
            self.coord.shutdown.store(true, Ordering::Relaxed);
        }
        self.coord.idle_cv.notify_all();
        for w in &mut self.workers {
            if let Some(h) = w.handle.take() {
                let _ = h.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic counter env: done every `horizon` steps.
    struct CounterEnv {
        t: u64,
        horizon: u64,
        seed: f32,
    }

    impl Env for CounterEnv {
        fn obs_space(&self) -> ObsSpace {
            ObsSpace::Flat { size: 4 }
        }
        fn action_space(&self) -> ActionSpace {
            ActionSpace::discrete(4)
        }
        fn reset(&mut self) -> Vec<f32> {
            self.t = 0;
            vec![self.seed, 0.0, 0.0, 0.0]
        }
        fn step(&mut self, action: &Action) -> StepResult {
            self.t += 1;
            let done = self.t >= self.horizon;
            let reward = if action.as_index() == (self.t % 4) as i64 {
                1.0
            } else {
                0.0
            };
            StepResult {
                obs: vec![
                    self.seed,
                    self.t as f32,
                    action.as_index() as f32,
                    done as u8 as f32,
                ],
                reward,
                done,
                episode_len: self.t,
                episode_return: reward,
            }
        }
    }

    fn pool(n: usize, workers: usize) -> EnvPool {
        let envs: Vec<Box<dyn Env>> = (0..n)
            .map(|i| {
                Box::new(CounterEnv {
                    t: 0,
                    horizon: 5,
                    seed: i as f32,
                }) as Box<dyn Env>
            })
            .collect();
        EnvPool::new(envs).with_workers(workers)
    }

    #[test]
    fn reset_all_and_step() {
        let mut p = pool(4, 1);
        let obs = p.reset_all();
        assert_eq!(obs.len(), 4);
        assert_eq!(obs[2], vec![2.0, 0.0, 0.0, 0.0]);
        let actions: Vec<Action> = (0..4).map(Action::Discrete).collect();
        let rs = p.step(&actions).unwrap();
        assert_eq!(rs[3].obs[1], 1.0);
        assert!(!rs[0].done);
    }

    #[test]
    fn parallel_matches_sequential() {
        let mut seq = pool(16, 1);
        let mut par = pool(16, 4);
        seq.reset_all();
        par.reset_all();
        for t in 0..12 {
            let actions: Vec<Action> = ((t * 3 + 1) % 4..).take(16).map(Action::Discrete).collect();
            let a = seq.step(&actions).unwrap();
            let b = par.step_parallel(&actions).unwrap();
            assert_eq!(a, b, "step {}", t);
        }
    }

    #[test]
    fn parallel_reset_matches_sequential() {
        let mut seq = pool(10, 1);
        let mut par = pool(10, 3);
        let a = seq.reset_all();
        let b = par.reset_all();
        assert_eq!(a, b);
    }

    #[test]
    fn many_steps_stay_correct() {
        // Longer soak: catches dispatch/done races and ordering bugs.
        // Horizon 1000 keeps episodes un-terminated so per-step invariants
        // (seed + step counter + action echo) hold on every iteration;
        // the done/reset path is covered by the tests above.
        let envs: Vec<Box<dyn Env>> = (0..32)
            .map(|i| {
                Box::new(CounterEnv {
                    t: 0,
                    horizon: 1000,
                    seed: i as f32,
                }) as Box<dyn Env>
            })
            .collect();
        let mut p = EnvPool::new(envs).with_workers(4);
        p.reset_all();
        for t in 0..200u64 {
            let actions: Vec<Action> = (0..32)
                .map(|i| Action::Discrete(((i as u64 + t) % 4) as i64))
                .collect();
            let rs = p.step_parallel(&actions).unwrap();
            for (i, r) in rs.iter().enumerate() {
                assert_eq!(r.obs[0], i as f32, "env {} seed", i);
                assert_eq!(r.obs[1], (t + 1) as f32, "env {} step counter", i);
                assert_eq!(r.obs[2], ((i as u64 + t) % 4) as f32, "env {} action", i);
            }
        }
    }

    #[test]
    fn done_flags_after_horizon() {
        let mut p = pool(2, 1);
        p.reset_all();
        let actions = vec![Action::Discrete(0), Action::Discrete(1)];
        for _ in 0..4 {
            let rs = p.step(&actions).unwrap();
            assert!(!rs[0].done);
        }
        let rs = p.step(&actions).unwrap();
        assert!(rs[0].done && rs[1].done);
        assert_eq!(rs[0].episode_len, 5);
    }

    #[test]
    fn dim_mismatch_rejected() {
        let mut p = pool(3, 1);
        assert!(p.step(&[Action::Discrete(0)]).is_err());
        assert!(p.step_parallel(&[Action::Discrete(0)]).is_err());
    }
}
