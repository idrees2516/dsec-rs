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
use crate::envpool::Stepping;
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
    /// The env pool (any [`Stepping`] implementation — the sync
    /// [`EnvPool`] or the pipelined sandbox batch pool).
    pub pool: Box<dyn Stepping>,
    pub buffer: ReplayBuffer,
    policy: Arc<PolicyFn>,
    lstm: Option<LstmCell>,
    // Current hidden state per env (actor only; critic shares size).
    actor_h: Vec<f32>,
    actor_c: Vec<f32>,
    critic_h: Vec<f32>,
    critic_c: Vec<f32>,
    // Reusable INPUT hidden snapshot (what enqueue stores): the buffers
    // are allocated once and refilled per step instead of four fresh
    // clones per iteration.
    snap_h: Vec<f32>,
    snap_c: Vec<f32>,
    snap_ch: Vec<f32>,
    snap_cc: Vec<f32>,
    // Reusable per-step scratch (obs / scalars).
    new_obs: Vec<f32>,
    rewards: Vec<f32>,
    dones: Vec<bool>,
    log_probs: Vec<f32>,
    values: Vec<f32>,
    // Last observations per env (flattened).
    obs: Vec<f32>,
    episode_returns: Vec<f32>,
    episode_lens: Vec<u64>,
    done_returns: Vec<f32>,
    done_lens: Vec<u64>,
    cfg: DriverConfig,
}

impl Driver {
    /// `pool` accepts any [`Stepping`] pool — pass the sync [`EnvPool`]
    /// or the pipelined sandbox batch pool (`Box::new(pool)`).
    pub fn new(
        pool: Box<dyn Stepping>,
        buffer: ReplayBuffer,
        policy: Arc<PolicyFn>,
        cfg: DriverConfig,
        seed: u64,
    ) -> Self {
        let num_envs = pool.num_envs();
        let hidden = buffer.hidden_size();
        let obs_size = buffer.obs_size();
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
            snap_h: vec![0.0; num_envs * hidden],
            snap_c: vec![0.0; num_envs * hidden],
            snap_ch: vec![0.0; num_envs * hidden],
            snap_cc: vec![0.0; num_envs * hidden],
            new_obs: Vec::with_capacity(num_envs * obs_size),
            rewards: Vec::with_capacity(num_envs),
            dones: Vec::with_capacity(num_envs),
            log_probs: Vec::with_capacity(num_envs),
            values: Vec::with_capacity(num_envs),
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
            // Reused buffers: copy in place, no per-step allocation.
            self.snap_h.copy_from_slice(&self.actor_h);
            self.snap_c.copy_from_slice(&self.actor_c);
            self.snap_ch.copy_from_slice(&self.critic_h);
            self.snap_cc.copy_from_slice(&self.critic_c);

            // 2) Update LSTM hidden from current obs (recurrent feature).
            if let Some(cell) = &self.lstm {
                for env in 0..num_envs {
                    let x = &self.obs[env * obs_size..(env + 1) * obs_size];
                    let h = &mut self.actor_h[env * hidden..(env + 1) * hidden];
                    let c = &mut self.actor_c[env * hidden..(env + 1) * hidden];
                    cell.forward(x, h, c);
                }
                // Critic path: share the same recurrence (pufferlib
                // stores separate h for actor/critic; a shared encoder
                // is the common architecture).
                self.critic_h.copy_from_slice(&self.actor_h);
                self.critic_c.copy_from_slice(&self.actor_c);
            }

            // 3) Policy.
            let actions = (self.policy)(&self.obs, &mut self.actor_h, &mut self.actor_c);

            // 3) Step.
            let results = self.pool.step_parallel(&actions)?;
            // 4) Values / log probs: a deterministic stand-in (the real
            // training loop computes these in torch; here a simple
            // heuristic keeps the port self-contained). Scratch vectors
            // are reused across steps.
            self.new_obs.clear();
            self.rewards.clear();
            self.dones.clear();
            self.log_probs.clear();
            self.values.clear();
            for r in &results {
                self.new_obs.extend_from_slice(&r.obs);
                self.rewards.push(r.reward);
                self.dones.push(r.done);
                self.log_probs.push(0.0f32);
                self.values.push(r.reward); // value ~= immediate reward heuristic
            }
            for (env, r) in results.iter().enumerate() {
                self.episode_returns[env] += r.reward;
                self.episode_lens[env] += 1;
                if r.done {
                    self.done_returns.push(self.episode_returns[env]);
                    self.done_lens.push(self.episode_lens[env]);
                    self.episode_returns[env] = 0.0;
                    self.episode_lens[env] = 0;
                    // 5) pufferlib LSTM reset: zero this env's hidden state.
                    if hidden > 0 {
                        for buf in [
                            &mut self.actor_h,
                            &mut self.actor_c,
                            &mut self.critic_h,
                            &mut self.critic_c,
                        ] {
                            buf[env * hidden..(env + 1) * hidden].fill(0.0);
                        }
                    }
                }
            }

            // 4b) Enqueue (obs from BEFORE the step, action, reward...).
            self.buffer.enqueue(
                &self.obs,
                &actions,
                &self.rewards,
                &self.dones,
                &self.log_probs,
                &self.values,
                &self.snap_h,
                &self.snap_c,
                &self.snap_ch,
                &self.snap_cc,
            )?;
            std::mem::swap(&mut self.obs, &mut self.new_obs);
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
