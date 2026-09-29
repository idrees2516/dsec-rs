//! Training driver: envpool + policy + buffer with pufferlib's
//! reset-on-done hidden state semantics.
//!
//! Each loop iteration:
//! 1. reads the pool's current observations,
//! 2. asks the policy for actions (and actor/critic hidden snapshots),
//! 3. steps the pool,
//! 4. enqueues (obs, action, reward, done, logprob, value, h, c),
//! 5. **zeroes the hidden state of envs whose episode just ended** —
//!    pufferlib's LSTM reset — before the next iteration,
//! 6. auto-resets are handled by the pool (`reset_if_done`).

use std::sync::Arc;

use crate::buffer::ReplayBuffer;
use crate::envpool::EnvPool;
use crate::error::Result;
use crate::lstm::LstmCell;
use crate::spaces::Action;

/// The policy closure: batch of observations -> actions.
/// (Observations are flattened, env-major, `num_envs * obs_size`.)
pub type PolicyFn = dyn Fn(&[f32], &mut [f32], &mut [f32]) -> Vec<Action> + Send + Sync;

/// Driver configuration.
#[derive(Debug, Clone)]
pub struct DriverConfig {
    pub gamma: f64,
    pub gae_lambda: f64,
    pub normalize_advantages: bool,
    /// LSTM applied to observations for hidden-state bookkeeping.
    pub use_lstm: bool,
}

impl Default for DriverConfig {
    fn default() -> Self {
        DriverConfig {
            gamma: 0.99,
            gae_lambda: 0.95,
            normalize_advantages: true,
            use_lstm: true,
        }
    }
}

/// Aggregated rollout statistics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RolloutStats {
    pub steps: u64,
    pub episodes: u64,
    pub mean_return: f64,
    pub mean_len: f64,
    pub steps_per_sec: f64,
}

pub struct Driver {
    pub pool: EnvPool,
    pub buffer: ReplayBuffer,
    policy: Arc<PolicyFn>,
    lstm: Option<LstmCell>,
    // Current hidden state per env (actor only; critic shares size).
    actor_h: Vec<f32>,
    actor_c: Vec<f32>,
    critic_h: Vec<f32>,
    critic_c: Vec<f32>,
    // Last observations per env (flattened).
    obs: Vec<f32>,
    episode_returns: Vec<f32>,
    episode_lens: Vec<u64>,
    done_returns: Vec<f32>,
    done_lens: Vec<u64>,
    cfg: DriverConfig,
}

impl Driver {
    pub fn new(
        pool: EnvPool,
        buffer: ReplayBuffer,
        policy: Arc<PolicyFn>,
        cfg: DriverConfig,
        seed: u64,
    ) -> Self {
        let num_envs = pool.num_envs();
        let hidden = buffer.hidden_size();
        let lstm = if cfg.use_lstm && hidden > 0 {
            Some(LstmCell::new(buffer.obs_size(), hidden, seed))
        } else {
            None
        };
        Driver {
            pool,
            buffer,
            policy,
            lstm,
            actor_h: vec![0.0; num_envs * hidden],
            actor_c: vec![0.0; num_envs * hidden],
            critic_h: vec![0.0; num_envs * hidden],
            critic_c: vec![0.0; num_envs * hidden],
            obs: Vec::new(),
            episode_returns: vec![0.0; num_envs],
            episode_lens: vec![0; num_envs],
            done_returns: Vec::new(),
            done_lens: Vec::new(),
            cfg,
        }
    }

    /// Collects exactly `steps` vectorized steps.
    pub fn rollout(&mut self, steps: u64) -> Result<RolloutStats> {
        let t0 = std::time::Instant::now();
        let num_envs = self.pool.num_envs();
        let obs_size = self.buffer.obs_size();
        let hidden = self.buffer.hidden_size();

        // Initial reset if needed.
        if self.obs.is_empty() {
            let initial = self.pool.reset_all();
            self.obs = initial.into_iter().flatten().collect();
            assert_eq!(self.obs.len(), num_envs * obs_size);
        }

        for _ in 0..steps {
            // 1) Snapshot the INPUT hidden state for this step (the state
            // the recurrence starts from — zero at episode starts, which
            // is exactly what the buffer's reset invariant checks).
            let (ah, ac, ch, cc) = (
                self.actor_h.clone(),
                self.actor_c.clone(),
                self.critic_h.clone(),
                self.critic_c.clone(),
            );

            // 2) Update LSTM hidden from current obs (recurrent feature).
            if let Some(cell) = &self.lstm {
                for env in 0..num_envs {
                    let x = &self.obs[env * obs_size..(env + 1) * obs_size];
                    let h = &mut self.actor_h[env * hidden..(env + 1) * hidden];
                    let c = &mut self.actor_c[env * hidden..(env + 1) * hidden];
                    cell.forward(x, h, c);
                    // Critic path: share the same recurrence (pufferlib
                    // stores separate h for actor/critic; a shared encoder
                    // is the common architecture).
                    let xh = self.actor_h[env * hidden..(env + 1) * hidden].to_vec();
                    self.critic_h[env * hidden..(env + 1) * hidden].copy_from_slice(&xh);
                    self.critic_c[env * hidden..(env + 1) * hidden]
                        .copy_from_slice(&self.actor_c[env * hidden..(env + 1) * hidden]);
                }
            }

            // 3) Policy.
            let actions = (self.policy)(&self.obs, &mut self.actor_h, &mut self.actor_c);

            // 3) Step.
            let results = self.pool.step_parallel(&actions)?;
            // 4) Values / log probs: a deterministic stand-in (the real
            // training loop computes these in torch; here a simple
            // heuristic keeps the port self-contained).
            let mut new_obs = Vec::with_capacity(num_envs * obs_size);
            let mut rewards = Vec::with_capacity(num_envs);
            let mut dones = Vec::with_capacity(num_envs);
            let mut log_probs = Vec::with_capacity(num_envs);
            let mut values = Vec::with_capacity(num_envs);
            for (env, r) in results.iter().enumerate() {
                new_obs.extend_from_slice(&r.obs);
                rewards.push(r.reward);
                dones.push(r.done);
                log_probs.push(0.0f32);
                values.push(r.reward); // value ~= immediate reward heuristic
                self.episode_returns[env] += r.reward;
                self.episode_lens[env] += 1;
                if r.done {
                    self.done_returns.push(self.episode_returns[env]);
                    self.done_lens.push(self.episode_lens[env]);
                    self.episode_returns[env] = 0.0;
                    self.episode_lens[env] = 0;
                    // 5) pufferlib LSTM reset: zero this env's hidden state.
                    if hidden > 0 {
                        for v in &mut self.actor_h[env * hidden..(env + 1) * hidden] {
                            *v = 0.0;
                        }
                        for v in &mut self.actor_c[env * hidden..(env + 1) * hidden] {
                            *v = 0.0;
                        }
                        for v in &mut self.critic_h[env * hidden..(env + 1) * hidden] {
                            *v = 0.0;
                        }
                        for v in &mut self.critic_c[env * hidden..(env + 1) * hidden] {
                            *v = 0.0;
                        }
                    }
                }
            }

            // 4b) Enqueue (obs from BEFORE the step, action, reward...).
            self.buffer.enqueue(
                &self.obs, &actions, &rewards, &dones, &log_probs, &values, &ah, &ac, &ch, &cc,
            )?;
            self.obs = new_obs;
        }

        let elapsed = t0.elapsed().as_secs_f64();
        let episodes = self.done_returns.len() as u64;
        let mean_return = if episodes > 0 {
            self.done_returns.iter().sum::<f32>() as f64 / episodes as f64
        } else {
            0.0
        };
        let mean_len = if episodes > 0 {
            self.done_lens.iter().sum::<u64>() as f64 / episodes as f64
        } else {
            0.0
        };
        let stats = RolloutStats {
            steps: steps * num_envs as u64,
            episodes,
            mean_return,
            mean_len,
            steps_per_sec: if elapsed > 0.0 {
                steps as f64 * num_envs as f64 / elapsed
            } else {
                0.0
            },
        };
        // Keep rolling episode stats across rollouts.
        self.done_returns.clear();
        self.done_lens.clear();
        Ok(stats)
    }

    /// Computes GAE (+ optional normalization) after a rollout phase.
    pub fn finish_phase(&mut self) {
        self.buffer.compute_gae(self.cfg.gamma, self.cfg.gae_lambda);
        if self.cfg.normalize_advantages {
            self.buffer.normalize_advantages();
        }
    }

    /// Verifies and repairs the LSTM reset invariant (returns violations).
    pub fn check_hidden_invariants(&mut self) -> usize {
        self.buffer.repair_hidden_resets()
    }
}
