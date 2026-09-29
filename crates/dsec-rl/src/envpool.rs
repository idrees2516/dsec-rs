//! Vectorized environment pools — the port of PufferLib's `PufferEnv`
//! layer.
//!
//! `EnvPool` steps N environments with one action vector. Semantics
//! follow pufferlib's `reset_if_done`: when an env returns `done`, the
//! result carries the terminal observation AND the env is immediately
//! reset, so the *next* step belongs to a fresh episode.
//!
//! `step_parallel` splits envs across worker threads
//! (`std::thread::scope`, chunked — chunk order is deterministic, so
//! results stay in env order regardless of thread scheduling).

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

/// A vectorized pool.
pub struct EnvPool {
    envs: Vec<Box<dyn Env>>,
    workers: usize,
}

impl EnvPool {
    pub fn new(envs: Vec<Box<dyn Env>>) -> Self {
        EnvPool { envs, workers: 1 }
    }

    pub fn with_workers(mut self, workers: usize) -> Self {
        self.workers = workers.max(1);
        self
    }

    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    pub fn obs_space(&self) -> ObsSpace {
        self.envs
            .first()
            .map(|e| e.obs_space())
            .unwrap_or(ObsSpace::Flat { size: 0 })
    }

    pub fn action_space(&self) -> ActionSpace {
        self.envs
            .first()
            .map(|e| e.action_space())
            .unwrap_or_else(|| ActionSpace::discrete(1))
    }

    /// Resets every env; returns initial observations.
    pub fn reset_all(&mut self) -> Vec<Vec<f32>> {
        self.envs.iter_mut().map(|e| e.reset()).collect()
    }

    /// Sequential stepping (deterministic order).
    ///
    /// Pufferlib `reset_if_done`: when an env reports `done`, the env is
    /// reset immediately and the result carries the NEW episode's first
    /// observation (reward/done describe the finished episode).
    pub fn step(&mut self, actions: &[Action]) -> Result<Vec<StepResult>> {
        if actions.len() != self.envs.len() {
            return Err(Error::Dim {
                what: "actions",
                expected: self.envs.len(),
                actual: actions.len(),
            });
        }
        Ok(self
            .envs
            .iter_mut()
            .zip(actions)
            .map(|(env, a)| {
                let mut r = env.step(a);
                if r.done {
                    r.obs = env.reset();
                }
                r
            })
            .collect())
    }

    /// Parallel stepping across scoped worker threads. Results preserve
    /// env order.
    pub fn step_parallel(&mut self, actions: &[Action]) -> Result<Vec<StepResult>> {
        let n = self.envs.len();
        if actions.len() != n {
            return Err(Error::Dim {
                what: "actions",
                expected: n,
                actual: actions.len(),
            });
        }
        if self.workers <= 1 || n < 2 {
            return self.step(actions);
        }
        let workers = self.workers.min(n);
        let chunk = n.div_ceil(workers);
        let mut results: Vec<StepResult> = Vec::with_capacity(n);
        {
            let env_chunks: Vec<&mut [Box<dyn Env>]> = self.envs.chunks_mut(chunk).collect();
            let act_chunks: Vec<&[Action]> = actions.chunks(chunk).collect();
            std::thread::scope(|scope| {
                let handles: Vec<_> = env_chunks
                    .into_iter()
                    .zip(act_chunks)
                    .map(|(envs, acts)| {
                        scope.spawn(move || {
                            envs.iter_mut()
                                .zip(acts)
                                .map(|(e, a)| {
                                    let mut r = e.step(a);
                                    if r.done {
                                        r.obs = e.reset();
                                    }
                                    r
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                for h in handles {
                    match h.join() {
                        Ok(mut v) => results.append(&mut v),
                        Err(_) => return Err(Error::Pool("worker thread panicked".into())),
                    }
                }
                Ok(())
            })?;
        }
        Ok(results)
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
