//! Pure-compute benchmark environment (no sandbox I/O) — the RL-layer
//! equivalent of pufferlib's pure-Python speed floors: measures how fast
//! the buffer + driver can ingest transitions without data-plane cost.

use crate::envpool::{Env, StepResult};
use crate::spaces::{Action, ActionSpace, ObsSpace};

/// Deterministic counter bandit: reward 1 when action matches a hidden
/// target that rotates every `horizon` steps; episodes of `horizon`.
pub struct FastCounterEnv {
    t: u64,
    horizon: u64,
    seed: u64,
    obs_buf: Vec<f32>,
}

impl FastCounterEnv {
    pub fn new(seed: u64, horizon: u64) -> Self {
        FastCounterEnv {
            t: 0,
            horizon,
            seed,
            obs_buf: vec![0.0; 8],
        }
    }
}

impl Env for FastCounterEnv {
    fn obs_space(&self) -> ObsSpace {
        ObsSpace::Flat { size: 8 }
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::discrete(8)
    }

    fn reset(&mut self) -> Vec<f32> {
        self.t = 0;
        self.obs_buf.iter_mut().for_each(|v| *v = 0.0);
        self.obs_buf[0] = (self.seed % 97) as f32 / 97.0;
        self.obs_buf.clone()
    }

    fn step(&mut self, action: &Action) -> StepResult {
        self.t += 1;
        let target = (self.seed.wrapping_mul(self.t) % 8) as i64;
        let hit = action.as_index() == target;
        let done = self.t >= self.horizon;
        self.obs_buf[1] = self.t as f32 / self.horizon as f32;
        self.obs_buf[2] = action.as_index() as f32 / 8.0;
        self.obs_buf[3] = hit as u8 as f32;
        let reward = if hit { 1.0 } else { 0.0 };
        StepResult {
            obs: self.obs_buf.clone(),
            reward,
            done,
            episode_len: self.t,
            episode_return: reward,
        }
    }
}
