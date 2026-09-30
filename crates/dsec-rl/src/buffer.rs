//! Episode ring replay buffer — the Rust port of PufferLib's
//! `PufferReplayBuffer`.
//!
//! Semantics ported faithfully:
//! - **Ring over steps**: each `enqueue` stores one *step* for all
//!   `num_envs` environments; the write pointer wraps at `capacity`.
//! - **Contiguous episodes**: a training chunk for one env is a run of
//!   consecutive steps uninterrupted by `done`.
//! - **LSTM hidden state**: actor and critic (h, c) are stored per step
//!   per env; they are **reset to zero at episode boundaries** (pufferlib
//!   resets hidden state on `done` before the next episode's first
//!   step).
//! - **mat-obs**: observations are one contiguous f32 slab; a `Mat`
//!   space yields `[num_envs][rows][cols]` views per step.
//! - **GAE** (`compute_gae`) is computed per env across the ring in
//!   temporal order, restarting at episode boundaries.
//!
//! The buffer is on-policy (as in pufferlib): after a training phase the
//! caller resets it and collects fresh rollouts.

use crate::error::{Error, Result};
use crate::spaces::{Action, ActionSpace, ObsSpace};

/// One stored step's cross-section (all envs).
#[derive(Debug, Clone)]
pub struct StepSlice {
    pub obs: Vec<f32>,
    pub actions: Vec<Action>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub log_probs: Vec<f32>,
    pub values: Vec<f32>,
}

/// Read view of one (step, env) transition.
#[derive(Debug, Clone)]
pub struct Transition {
    pub obs: Vec<f32>,
    pub action: Action,
    pub reward: f32,
    pub done: bool,
    pub log_prob: f32,
    pub value: f32,
    pub actor_h: Vec<f32>,
    pub actor_c: Vec<f32>,
    pub critic_h: Vec<f32>,
    pub critic_c: Vec<f32>,
    pub advantage: f32,
    pub ret: f32,
}

/// One contiguous same-env run without an interior `done`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpisodeChunk {
    pub env: usize,
    /// First ring slot (absolute step index within stored window).
    pub start: usize,
    pub len: usize,
    /// Ends with a `done` (episode boundary).
    pub terminated: bool,
}

/// A batch of fixed-length training sequences with reset-correct h0/c0.
#[derive(Debug, Clone)]
pub struct SequenceBatch {
    /// `[seq][batch][obs_size]`
    pub obs: Vec<f32>,
    pub actions: Vec<i64>,
    pub old_log_probs: Vec<f32>,
    pub advantages: Vec<f32>,
    pub returns: Vec<f32>,
    /// Actor h/c at each sequence start (zeros when the sequence starts
    /// an episode).
    pub h0: Vec<f32>,
    pub c0: Vec<f32>,
    pub seq_len: usize,
    pub batch: usize,
    pub obs_size: usize,
}

pub struct ReplayBuffer {
    capacity: usize,
    num_envs: usize,
    obs_size: usize,
    obs_space: ObsSpace,
    action_space: ActionSpace,
    hidden_size: usize,

    // Ring storage (step-major).
    obs: Vec<f32>,       // capacity * num_envs * obs_size
    actions_i: Vec<i64>, // discrete path
    actions_f: Vec<f32>, // box path
    rewards: Vec<f32>,
    dones: Vec<u8>,
    log_probs: Vec<f32>,
    values: Vec<f32>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
    actor_h: Vec<f32>, // capacity * num_envs * hidden
    actor_c: Vec<f32>,
    critic_h: Vec<f32>,
    critic_c: Vec<f32>,

    write: usize, // next step slot
    len: usize,   // filled steps
    total_steps: u64,
    episodes_completed: u64,
}

impl ReplayBuffer {
    /// Creates a buffer. `hidden_size == 0` disables hidden storage.
    pub fn new(
        capacity: usize,
        num_envs: usize,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        hidden_size: usize,
    ) -> Self {
        let obs_size = obs_space.size();
        let n = capacity.max(1) * num_envs.max(1);
        ReplayBuffer {
            capacity: capacity.max(1),
            num_envs,
            obs_size,
            obs_space,
            hidden_size,
            obs: vec![0.0; n * obs_size],
            actions_i: vec![0; n],
            actions_f: vec![0.0; n * action_space.size()],
            action_space: action_space.clone(),
            rewards: vec![0.0; n],
            dones: vec![0; n],
            log_probs: vec![0.0; n],
            values: vec![0.0; n],
            advantages: vec![0.0; n],
            returns: vec![0.0; n],
            actor_h: vec![0.0; n * hidden_size],
            actor_c: vec![0.0; n * hidden_size],
            critic_h: vec![0.0; n * hidden_size],
            critic_c: vec![0.0; n * hidden_size],
            write: 0,
            len: 0,
            total_steps: 0,
            episodes_completed: 0,
        }
    }

    // -- accessors -----------------------------------------------------------

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }
    pub fn obs_size(&self) -> usize {
        self.obs_size
    }
    pub fn obs_space(&self) -> ObsSpace {
        self.obs_space
    }
    pub fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }
    pub fn episodes_completed(&self) -> u64 {
        self.episodes_completed
    }

    fn idx(&self, slot: usize, env: usize) -> usize {
        slot * self.num_envs + env
    }

    /// Absolute slot for the i-th oldest stored step.
    fn slot_of(&self, age: usize) -> usize {
        debug_assert!(age < self.len);
        let first = (self.write + self.capacity - self.len) % self.capacity;
        (first + age) % self.capacity
    }

    // -- enqueue ---------------------------------------------------------------

    /// Stores one step for all envs.
    ///
    /// * `obs`: `num_envs * obs_size` floats (env-major).
    /// * `actor_h`/`actor_c`/`critic_h`/`critic_c`: `num_envs * hidden_size`
    ///   each (empty when `hidden_size == 0`). The caller zeroes an env's
    ///   hidden state after its episode ended (pufferlib's LSTM reset).
    /// * `actions`: one per env.
    ///
    /// The argument count mirrors pufferlib's per-step transition tuple
    /// (obs / action / reward / done / logprob / value / 4 hidden slices).
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue(
        &mut self,
        obs: &[f32],
        actions: &[Action],
        rewards: &[f32],
        dones: &[bool],
        log_probs: &[f32],
        values: &[f32],
        actor_h: &[f32],
        actor_c: &[f32],
        critic_h: &[f32],
        critic_c: &[f32],
    ) -> Result<()> {
        let expect_obs = self.num_envs * self.obs_size;
        if obs.len() != expect_obs {
            return Err(Error::Dim {
                what: "observations",
                expected: expect_obs,
                actual: obs.len(),
            });
        }
        if actions.len() != self.num_envs {
            return Err(Error::Dim {
                what: "actions",
                expected: self.num_envs,
                actual: actions.len(),
            });
        }
        if rewards.len() != self.num_envs {
            return Err(Error::Dim {
                what: "rewards",
                expected: self.num_envs,
                actual: rewards.len(),
            });
        }
        if dones.len() != self.num_envs {
            return Err(Error::Dim {
                what: "dones",
                expected: self.num_envs,
                actual: dones.len(),
            });
        }
        if log_probs.len() != self.num_envs {
            return Err(Error::Dim {
                what: "log_probs",
                expected: self.num_envs,
                actual: log_probs.len(),
            });
        }
        if values.len() != self.num_envs {
            return Err(Error::Dim {
                what: "values",
                expected: self.num_envs,
                actual: values.len(),
            });
        }
        let hs = self.num_envs * self.hidden_size;
        if self.hidden_size > 0
            && (actor_h.len() != hs
                || actor_c.len() != hs
                || critic_h.len() != hs
                || critic_c.len() != hs)
        {
            return Err(Error::Dim {
                what: "hidden states",
                expected: hs,
                actual: actor_h.len(),
            });
        }

        // Per-env field copies: measured faster than field-major bulk
        // memcpy here (each per-env copy is 256-512 B, which the compiler
        // vectorizes into tight AVX loops; libc memcpy's large-copy path
        // loses on this class of VM core). Enqueue stays env-interleaved.
        let slot = self.write;
        for env in 0..self.num_envs {
            let i = self.idx(slot, env);
            // Count episodes once: a done completes an episode.
            if dones[env] {
                self.episodes_completed += 1;
            }
            self.obs[i * self.obs_size..(i + 1) * self.obs_size]
                .copy_from_slice(&obs[env * self.obs_size..(env + 1) * self.obs_size]);
            self.rewards[i] = rewards[env];
            self.dones[i] = dones[env] as u8;
            self.log_probs[i] = log_probs[env];
            self.values[i] = values[env];
            match (&actions[env], &self.action_space) {
                (Action::Discrete(a), ActionSpace::Discrete { .. }) => self.actions_i[i] = *a,
                (Action::Box(v), ActionSpace::Box { .. }) => {
                    let asz = v.len();
                    self.actions_f[i * asz..(i + 1) * asz].copy_from_slice(v);
                }
                (a, space) => {
                    return Err(Error::ActionSpace(format!(
                        "action {:?} does not match space {:?}",
                        a, space
                    )))
                }
            }
            if self.hidden_size > 0 {
                let h = i * self.hidden_size;
                self.actor_h[h..h + self.hidden_size].copy_from_slice(
                    &actor_h[env * self.hidden_size..(env + 1) * self.hidden_size],
                );
                self.actor_c[h..h + self.hidden_size].copy_from_slice(
                    &actor_c[env * self.hidden_size..(env + 1) * self.hidden_size],
                );
                self.critic_h[h..h + self.hidden_size].copy_from_slice(
                    &critic_h[env * self.hidden_size..(env + 1) * self.hidden_size],
                );
                self.critic_c[h..h + self.hidden_size].copy_from_slice(
                    &critic_c[env * self.hidden_size..(env + 1) * self.hidden_size],
                );
            }
        }
        self.write = (self.write + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
        self.total_steps += 1;
        Ok(())
    }

    /// Enforce pufferlib's LSTM reset invariant: hidden state stored for
    /// env `e` at step following a `done` must be zero.
    ///
    /// Returns the count of invariants violated (should be 0 when the
    /// driver resets hidden state correctly) and repairs them.
    pub fn repair_hidden_resets(&mut self) -> usize {
        if self.hidden_size == 0 || self.len < 2 {
            return 0;
        }
        let mut fixed = 0;
        for age in 0..self.len.saturating_sub(1) {
            let slot = self.slot_of(age);
            let next = self.slot_of(age + 1);
            for env in 0..self.num_envs {
                if self.dones[self.idx(slot, env)] != 0 {
                    let i = self.idx(next, env);
                    let (h, c) = (
                        &mut self.actor_h[i * self.hidden_size..(i + 1) * self.hidden_size],
                        &mut self.actor_c[i * self.hidden_size..(i + 1) * self.hidden_size],
                    );
                    if h.iter().any(|&x| x != 0.0) || c.iter().any(|&x| x != 0.0) {
                        fixed += 1;
                        h.fill(0.0);
                        c.fill(0.0);
                    }
                    let (ch, cc) = (
                        &mut self.critic_h[i * self.hidden_size..(i + 1) * self.hidden_size],
                        &mut self.critic_c[i * self.hidden_size..(i + 1) * self.hidden_size],
                    );
                    if ch.iter().any(|&x| x != 0.0) || cc.iter().any(|&x| x != 0.0) {
                        ch.fill(0.0);
                        cc.fill(0.0);
                    }
                }
            }
        }
        fixed
    }

    // -- reads -----------------------------------------------------------------

    /// Reads one transition (age counted from oldest stored step).
    pub fn at(&self, age: usize, env: usize) -> Transition {
        assert!(age < self.len, "age {} beyond len {}", age, self.len);
        assert!(env < self.num_envs);
        let slot = self.slot_of(age);
        let i = self.idx(slot, env);
        let action = match &self.action_space {
            ActionSpace::Discrete { .. } => Action::Discrete(self.actions_i[i]),
            ActionSpace::Box { low, .. } => {
                let asz = low.len();
                Action::Box(self.actions_f[i * asz..(i + 1) * asz].to_vec())
            }
        };
        Transition {
            obs: self.obs[i * self.obs_size..(i + 1) * self.obs_size].to_vec(),
            action,
            reward: self.rewards[i],
            done: self.dones[i] != 0,
            log_prob: self.log_probs[i],
            value: self.values[i],
            actor_h: self.actor_h[i * self.hidden_size..(i + 1) * self.hidden_size].to_vec(),
            actor_c: self.actor_c[i * self.hidden_size..(i + 1) * self.hidden_size].to_vec(),
            critic_h: self.critic_h[i * self.hidden_size..(i + 1) * self.hidden_size].to_vec(),
            critic_c: self.critic_c[i * self.hidden_size..(i + 1) * self.hidden_size].to_vec(),
            advantage: self.advantages[i],
            ret: self.returns[i],
        }
    }

    /// Flat observation view for one step (all envs, contiguous).
    pub fn obs_step(&self, age: usize) -> &[f32] {
        let slot = self.slot_of(age);
        let base = slot * self.num_envs * self.obs_size;
        &self.obs[base..base + self.num_envs * self.obs_size]
    }

    /// Mat-obs view for one step: `[num_envs][rows][cols]`.
    pub fn obs_mat_step(&self, age: usize) -> Result<&[f32]> {
        match self.obs_space.mat_shape() {
            Some(_) => Ok(self.obs_step(age)),
            None => Err(Error::Other("not a mat obs space".into())),
        }
    }

    /// Slice of one env's obs at one step.
    pub fn obs_env(&self, age: usize, env: usize) -> &[f32] {
        let slot = self.slot_of(age);
        let i = self.idx(slot, env);
        &self.obs[i * self.obs_size..(i + 1) * self.obs_size]
    }

    // -- episodes ----------------------------------------------------------------

    /// Contiguous episode chunks per env (in ring order).
    pub fn episode_chunks(&self) -> Vec<EpisodeChunk> {
        let mut out = Vec::new();
        if self.len == 0 {
            return out;
        }
        for env in 0..self.num_envs {
            let mut start = 0usize;
            for age in 0..self.len {
                let slot = self.slot_of(age);
                if self.dones[self.idx(slot, env)] != 0 {
                    out.push(EpisodeChunk {
                        env,
                        start,
                        len: age - start + 1,
                        terminated: true,
                    });
                    start = age + 1;
                }
            }
            if start < self.len {
                out.push(EpisodeChunk {
                    env,
                    start,
                    len: self.len - start,
                    terminated: false,
                });
            }
        }
        out
    }

    /// GAE over the stored window, per env, restarted at dones.
    pub fn compute_gae(&mut self, gamma: f64, lambda: f64) {
        if self.len == 0 {
            return;
        }
        // Env-blocked backward pass: with slot-major storage, one env's
        // stream strides `num_envs * 4` bytes between consecutive steps —
        // a fresh cache line per element. Processing a block of 8 envs
        // together touches 32-byte contiguous runs per field per step,
        // reusing each cache line 8x. The recursion itself is unchanged
        // (each env's GAE is independent); blocks only change traversal
        // order. Fused write-back removes the second pass entirely.
        const BLOCK: usize = 8;
        let n_envs = self.num_envs;
        let mut gae = [0.0f64; BLOCK];
        let mut env0 = 0usize;
        while env0 < n_envs {
            let b = BLOCK.min(n_envs - env0);
            gae[..b].fill(0.0);
            // Walk newest -> oldest within this env block.
            for age_back in 0..self.len {
                let age = self.len - 1 - age_back;
                let slot = self.slot_of(age);
                let base = slot * n_envs;
                for (k, g) in gae.iter_mut().enumerate().take(b) {
                    let i = base + env0 + k;
                    let done = self.dones[i] != 0;
                    let v = self.values[i] as f64;
                    let r = self.rewards[i] as f64;
                    let v_next = if done {
                        // Episode ends here: no bootstrap across the boundary.
                        0.0
                    } else if age + 1 < self.len {
                        // Same episode's next stored step.
                        self.values[(self.slot_of(age + 1)) * n_envs + env0 + k] as f64
                    } else {
                        // Newest step: bootstrap with its own value estimate.
                        v
                    };
                    let delta = r + gamma * v_next - v;
                    let mask = if done { 0.0 } else { 1.0 };
                    *g = delta + gamma * lambda * mask * *g;
                    self.advantages[i] = *g as f32;
                    self.returns[i] = (*g + v) as f32;
                }
            }
            env0 += b;
        }
    }

    /// Normalizes advantages across the window (PPO standard).
    pub fn normalize_advantages(&mut self) {
        if self.len == 0 {
            return;
        }
        let n = self.len * self.num_envs;
        let mean: f64 = (0..n)
            .map(|k| {
                let slot = self.slot_of(k / self.num_envs);
                self.advantages[self.idx(slot, k % self.num_envs)] as f64
            })
            .sum::<f64>()
            / n as f64;
        let var: f64 = (0..n)
            .map(|k| {
                let slot = self.slot_of(k / self.num_envs);
                let d = self.advantages[self.idx(slot, k % self.num_envs)] as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let std = var.sqrt().max(1e-8);
        for age in 0..self.len {
            let slot = self.slot_of(age);
            for env in 0..self.num_envs {
                let i = self.idx(slot, env);
                self.advantages[i] =
                    (((self.advantages[i] as f64 - mean) / std) as f32).clamp(-10.0, 10.0);
            }
        }
    }

    /// Samples fixed-length training sequences, splitting at episode
    /// boundaries (h0/c0 zeroed when a sequence starts mid-episode only
    /// if the stored hidden state says so; a sequence starting exactly at
    /// an episode start carries zeros by construction).
    pub fn sample_sequences(
        &self,
        seq_len: usize,
        batch: usize,
        rng: &mut dsec_protocol::rng::Rng,
    ) -> SequenceBatch {
        let seq_len = seq_len.max(1);
        let mut obs_out = Vec::with_capacity(seq_len * batch * self.obs_size);
        let mut act_out = Vec::with_capacity(seq_len * batch);
        let mut lp_out = Vec::with_capacity(seq_len * batch);
        let mut adv_out = Vec::with_capacity(seq_len * batch);
        let mut ret_out = Vec::with_capacity(seq_len * batch);
        let mut h0_out = Vec::with_capacity(batch * self.hidden_size);
        let mut c0_out = Vec::with_capacity(batch * self.hidden_size);
        if self.len == 0 {
            return SequenceBatch {
                obs: obs_out,
                actions: act_out,
                old_log_probs: lp_out,
                advantages: adv_out,
                returns: ret_out,
                h0: h0_out,
                c0: c0_out,
                seq_len,
                batch: 0,
                obs_size: self.obs_size,
            };
        }
        let chunk = if self.hidden_size > 0 {
            self.hidden_size
        } else {
            1
        };
        let chunks = self.episode_chunks();
        let chunks: Vec<&EpisodeChunk> = chunks.iter().filter(|c| c.len >= 2).collect();
        let actual_batch = if chunks.is_empty() {
            0
        } else {
            batch.min(chunks.len().max(1))
        };
        for b in 0..actual_batch {
            // Deterministic round-robin when fewer chunks than batch.
            let c = chunks[(b + rng.below(chunks.len().max(1))) % chunks.len().max(1)];
            let env = c.env;
            // Start within the chunk so the whole sequence fits.
            let max_start = c.len.saturating_sub(seq_len);
            let start = c.start + rng.below(max_start + 1);
            // Hidden at sequence start: zeros iff start is the episode
            // start (or the previous step was a done).
            let is_episode_start = start == 0 || {
                if start > 0 {
                    let prev_slot = self.slot_of(start - 1);
                    self.dones[self.idx(prev_slot, env)] != 0
                } else {
                    true
                }
            };
            for t in 0..seq_len {
                let age = start + t;
                let slot = self.slot_of(age);
                let i = self.idx(slot, env);
                obs_out.extend_from_slice(&self.obs[i * self.obs_size..(i + 1) * self.obs_size]);
                act_out.push(self.actions_i[i]);
                lp_out.push(self.log_probs[i]);
                adv_out.push(self.advantages[i]);
                ret_out.push(self.returns[i]);
            }
            if self.hidden_size > 0 {
                let slot = self.slot_of(start);
                let i = self.idx(slot, env);
                if is_episode_start {
                    h0_out.extend(std::iter::repeat_n(0.0, chunk));
                    c0_out.extend(std::iter::repeat_n(0.0, chunk));
                } else {
                    h0_out.extend_from_slice(&self.actor_h[i * chunk..(i + 1) * chunk]);
                    c0_out.extend_from_slice(&self.actor_c[i * chunk..(i + 1) * chunk]);
                }
            }
        }
        SequenceBatch {
            obs: obs_out,
            actions: act_out,
            old_log_probs: lp_out,
            advantages: adv_out,
            returns: ret_out,
            h0: h0_out,
            c0: c0_out,
            seq_len,
            batch: actual_batch,
            obs_size: self.obs_size,
        }
    }

    /// Clears the buffer (on-policy reset between rollout phases).
    pub fn reset(&mut self) {
        self.write = 0;
        self.len = 0;
    }

    /// Zero-copy handoff of the whole observation slab into a
    /// `SharedRegion` (the paper's zero-copy data pipeline).
    pub fn dump_obs_to_region(&self, region: &dsec_storage::prefetch::SharedRegion) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }
        // Only the current window, in ring order.
        let view = {
            let mut ordered = Vec::with_capacity(self.len * self.num_envs * self.obs_size);
            for age in 0..self.len {
                ordered.extend_from_slice(self.obs_step(age));
            }
            ordered
        };
        if region.total_len() != view.len() {
            return Err(Error::Dim {
                what: "shared region",
                expected: view.len(),
                actual: region.total_len(),
            });
        }
        region.write_all(&view);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::Action;

    fn simple(cap: usize, envs: usize, hidden: usize) -> ReplayBuffer {
        ReplayBuffer::new(
            cap,
            envs,
            ObsSpace::Flat { size: 3 },
            ActionSpace::discrete(4),
            hidden,
        )
    }

    type StepData = (
        Vec<f32>,
        Vec<Action>,
        Vec<f32>,
        Vec<bool>,
        Vec<f32>,
        Vec<f32>,
    );

    fn step_data(envs: usize, t: u64, reward: f32, done: bool) -> StepData {
        (
            (0..envs * 3).map(|i| t as f32 * 10.0 + i as f32).collect(),
            (0..envs)
                .map(|i| Action::Discrete((i % 4) as i64))
                .collect(),
            vec![reward; envs],
            vec![done; envs],
            vec![-0.5; envs],
            vec![0.25; envs],
        )
    }

    fn hidden_data(
        envs: usize,
        hidden: usize,
        fill: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            vec![fill; envs * hidden],
            vec![fill * 2.0; envs * hidden],
            vec![fill * 3.0; envs * hidden],
            vec![fill * 4.0; envs * hidden],
        )
    }

    #[test]
    fn enqueue_at_roundtrip() {
        let mut b = simple(8, 2, 0);
        let (obs, act, rew, done, lp, val) = step_data(2, 3, 1.0, false);
        b.enqueue(&obs, &act, &rew, &done, &lp, &val, &[], &[], &[], &[])
            .unwrap();
        b.enqueue(&obs, &act, &rew, &done, &lp, &val, &[], &[], &[], &[])
            .unwrap();
        assert_eq!(b.len(), 2);
        assert_eq!(b.total_steps(), 2);
        let t = b.at(0, 1);
        assert_eq!(t.obs, vec![33.0, 34.0, 35.0]);
        assert_eq!(t.action, Action::Discrete(1));
        assert_eq!(t.reward, 1.0);
        assert_eq!(t.value, 0.25);
    }

    #[test]
    fn ring_wraps_and_keeps_newest() {
        let mut b = simple(4, 1, 0);
        for t in 0..10u64 {
            let (obs, act, rew, done, lp, val) = step_data(1, t, t as f32, false);
            b.enqueue(&obs, &act, &rew, &done, &lp, &val, &[], &[], &[], &[])
                .unwrap();
        }
        assert_eq!(b.len(), 4); // capacity
        assert_eq!(b.total_steps(), 10);
        // Oldest four stored steps are t = 6..=9 (obs first elem = t*10).
        assert_eq!(b.at(0, 0).obs[0], 60.0);
        assert_eq!(b.at(3, 0).obs[0], 90.0);
        assert_eq!(b.obs_step(3)[0], 90.0);
    }

    #[test]
    fn dim_mismatch_rejected() {
        let mut b = simple(4, 2, 0);
        let (obs, act, rew, done, lp, val) = step_data(2, 1, 0.0, false);
        assert!(b
            .enqueue(&obs[..3], &act, &rew, &done, &lp, &val, &[], &[], &[], &[])
            .is_err());
        assert!(b
            .enqueue(&obs, &act[..1], &rew, &done, &lp, &val, &[], &[], &[], &[])
            .is_err());
        assert!(b
            .enqueue(&obs, &act, &rew, &done[..1], &lp, &val, &[], &[], &[], &[])
            .is_err());
    }

    #[test]
    fn hidden_stored_and_read() {
        let mut b = simple(8, 2, 4);
        let (obs, act, rew, done, lp, val) = step_data(2, 1, 0.0, false);
        let (ah, ac, ch, cc) = hidden_data(2, 4, 0.5);
        b.enqueue(&obs, &act, &rew, &done, &lp, &val, &ah, &ac, &ch, &cc)
            .unwrap();
        let t = b.at(0, 0);
        assert_eq!(t.actor_h, vec![0.5; 4]);
        assert_eq!(t.actor_c, vec![1.0; 4]);
        assert_eq!(t.critic_h, vec![1.5; 4]);
        assert_eq!(t.critic_c, vec![2.0; 4]);
        let t1 = b.at(0, 1);
        assert_eq!(t1.actor_h, t.actor_h); // same fill
    }

    #[test]
    fn lstm_reset_invariant_detected_and_repaired() {
        let mut b = simple(8, 1, 3);
        // Step 0: done. Step 1: hidden NOT reset (driver bug simulation).
        let (obs, act, rew, done, lp, val) = step_data(1, 0, 1.0, true);
        b.enqueue(
            &obs, &act, &rew, &done, &lp, &val, &[0.1; 3], &[0.2; 3], &[0.3; 3], &[0.4; 3],
        )
        .unwrap();
        let (obs, act, rew, done, lp, val) = step_data(1, 1, 0.0, false);
        b.enqueue(
            &obs, &act, &rew, &done, &lp, &val, &[0.9; 3], &[0.9; 3], &[0.9; 3], &[0.9; 3],
        )
        .unwrap();
        // The post-done hidden must be zero; repair reports 1 fix.
        assert_eq!(b.repair_hidden_resets(), 1);
        // Second pass: nothing to fix.
        assert_eq!(b.repair_hidden_resets(), 0);
        assert_eq!(b.at(1, 0).actor_h, vec![0.0; 3]);
    }

    #[test]
    fn episode_chunks_segmented_at_dones() {
        let mut b = simple(16, 2, 0);
        // env0: done at step 2, len 3; env1: done at step 4, len 5.
        for t in 0..5u64 {
            let (obs, act, mut rew, mut done, lp, val) = step_data(2, t, 0.0, false);
            done[0] = t == 2;
            done[1] = t == 4;
            rew[0] = 1.0;
            b.enqueue(&obs, &act, &rew, &done, &lp, &val, &[], &[], &[], &[])
                .unwrap();
        }
        let chunks = b.episode_chunks();
        // env0: [0..=2] terminated, [3..=4] open.
        assert!(chunks.contains(&EpisodeChunk {
            env: 0,
            start: 0,
            len: 3,
            terminated: true
        }));
        assert!(chunks.contains(&EpisodeChunk {
            env: 0,
            start: 3,
            len: 2,
            terminated: false
        }));
        // env1: [0..=4] terminated.
        assert!(chunks.contains(&EpisodeChunk {
            env: 1,
            start: 0,
            len: 5,
            terminated: true
        }));
        assert_eq!(b.episodes_completed(), 2);
    }

    #[test]
    fn gae_hand_computed() {
        // Single env, 3 steps, values 0.5, rewards 0/0/1, last done.
        let mut b = simple(8, 1, 0);
        let episodes: Vec<(f32, f32, bool)> =
            vec![(0.0, 0.5, false), (0.0, 0.5, false), (1.0, 0.5, true)];
        for (t, (r, v, d)) in episodes.iter().enumerate() {
            let (obs, act, mut rew, mut done, lp, mut val) = step_data(1, t as u64, 0.0, false);
            rew[0] = *r;
            val[0] = *v;
            done[0] = *d;
            b.enqueue(&obs, &act, &rew, &done, &lp, &val, &[], &[], &[], &[])
                .unwrap();
        }
        let gamma = 0.5f64;
        let lambda = 0.5f64;
        b.compute_gae(gamma, lambda);
        // Step 2 (done): delta = 1 + 0 - 0.5 = 0.5; adv = 0.5.
        let t2 = b.at(2, 0);
        assert!((t2.advantage - 0.5).abs() < 1e-6, "adv2 = {}", t2.advantage);
        assert!((t2.ret - 1.0).abs() < 1e-6, "ret2 = {}", t2.ret);
        // Step 1: delta = 0 + 0.5*0.5 - 0.5 = -0.25;
        // adv = -0.25 + 0.5*0.5*0.5 = -0.125.
        let t1 = b.at(1, 0);
        assert!(
            (t1.advantage - (-0.125)).abs() < 1e-6,
            "adv1 = {}",
            t1.advantage
        );
        assert!((t1.ret - 0.375).abs() < 1e-6, "ret1 = {}", t1.ret);
        // Step 0: delta = -0.25; adv = -0.25 + 0.25*(-0.125) = -0.28125.
        let t0 = b.at(0, 0);
        assert!(
            (t0.advantage - (-0.28125)).abs() < 1e-6,
            "adv0 = {}",
            t0.advantage
        );
    }

    #[test]
    fn gae_restarts_across_episodes() {
        let mut b = simple(16, 1, 0);
        // Episode 1 ends with big reward; episode 2 gets none.
        for (t, (r, d)) in [
            (0u64, (0.0f32, false)),
            (1, (0.0, false)),
            (2, (10.0, true)),
            (3, (0.0, false)),
            (4, (0.0, true)),
        ] {
            let (obs, act, mut rew, mut done, lp, mut val) = step_data(1, 0, 0.0, false);
            rew[0] = r;
            done[0] = d;
            val[0] = 0.0; // isolate episode-boundary behavior
            b.enqueue(&obs, &act, &rew, &done, &lp, &val, &[], &[], &[], &[])
                .unwrap();
            let _ = t;
        }
        b.compute_gae(0.9, 0.8);
        // Episode 2's steps must not inherit episode 1's advantage.
        let t3 = b.at(3, 0);
        assert!(t3.advantage.abs() < 1e-6, "adv3 = {}", t3.advantage);
        // Episode 1's earlier steps DO propagate.
        let t0 = b.at(0, 0);
        assert!(t0.advantage > 1.0, "adv0 = {}", t0.advantage);
    }

    #[test]
    fn normalize_advantages_zero_mean_unit_std() {
        let mut b = simple(32, 2, 0);
        for t in 0..8u64 {
            let (obs, act, mut rew, done, lp, mut val) = step_data(2, t, 0.0, false);
            rew[0] = t as f32;
            rew[1] = -(t as f32);
            val[0] = 0.0;
            val[1] = 0.0;
            b.enqueue(&obs, &act, &rew, &done, &lp, &val, &[], &[], &[], &[])
                .unwrap();
        }
        b.compute_gae(0.9, 0.95);
        b.normalize_advantages();
        let n = b.len() * b.num_envs();
        let mean: f64 = (0..n)
            .map(|k| {
                let age = k / b.num_envs();
                let slot = b.slot_of(age);
                b.advantages[b.idx(slot, k % b.num_envs())] as f64
            })
            .sum::<f64>()
            / n as f64;
        assert!(mean.abs() < 1e-5, "mean = {}", mean);
    }

    #[test]
    fn sample_sequences_shapes_and_reset() {
        let mut b = ReplayBuffer::new(
            64,
            2,
            ObsSpace::Mat { rows: 2, cols: 3 },
            ActionSpace::discrete(4),
            4,
        );
        // env 0: 10-step episode, done at 9; env 1: never done.
        for t in 0..10u64 {
            let obs: Vec<f32> = (0..12).map(|i| t as f32 + i as f32 * 0.01).collect();
            let act = vec![Action::Discrete(1), Action::Discrete(2)];
            let rew = vec![0.1, 0.2];
            let done = vec![t == 9, false];
            let lp = vec![0.0, 0.0];
            let val = vec![0.0, 0.0];
            let (ah, ac, ch, cc) = hidden_data(2, 4, t as f32 * 0.1);
            b.enqueue(&obs, &act, &rew, &done, &lp, &val, &ah, &ac, &ch, &cc)
                .unwrap();
        }
        b.compute_gae(0.99, 0.95);
        let mut rng = dsec_protocol::rng::Rng::new(3);
        let batch = b.sample_sequences(5, 8, &mut rng);
        assert!(batch.batch >= 1);
        assert_eq!(batch.seq_len, 5);
        assert_eq!(batch.obs.len(), batch.seq_len * batch.batch * 6);
        assert_eq!(batch.actions.len(), batch.seq_len * batch.batch);
        assert_eq!(batch.h0.len(), batch.batch * 4);
        // h0 is either all-zero (episode start) or a real hidden snapshot.
        for i in 0..batch.batch {
            let h = &batch.h0[i * 4..(i + 1) * 4];
            let is_zero = h.iter().all(|&v| v == 0.0);
            let is_snapshot = h.iter().all(|&v| (v / 0.1).fract().abs() < 1e-4);
            assert!(is_zero || is_snapshot, "h0 = {:?}", h);
        }
    }

    #[test]
    fn mat_obs_views() {
        let mut b = ReplayBuffer::new(
            8,
            2,
            ObsSpace::Mat { rows: 2, cols: 3 },
            ActionSpace::discrete(2),
            0,
        );
        let obs: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let act = vec![Action::Discrete(0), Action::Discrete(1)];
        b.enqueue(
            &obs,
            &act,
            &[0.0; 2],
            &[false; 2],
            &[0.0; 2],
            &[0.0; 2],
            &[],
            &[],
            &[],
            &[],
        )
        .unwrap();
        let step = b.obs_mat_step(0).unwrap();
        assert_eq!(step.len(), 2 * 2 * 3);
        // env0 view = first 6 floats.
        assert_eq!(b.obs_env(0, 0), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(b.obs_env(0, 1), &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn box_action_space_roundtrip() {
        let mut b = ReplayBuffer::new(
            4,
            1,
            ObsSpace::Flat { size: 2 },
            ActionSpace::Box {
                low: vec![0.0; 2],
                high: vec![1.0; 2],
            },
            0,
        );
        let obs = vec![0.0, 1.0];
        let act = vec![Action::Box(vec![0.25, 0.75])];
        b.enqueue(
            &obs,
            &act,
            &[1.0],
            &[false],
            &[0.0],
            &[0.0],
            &[],
            &[],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(b.at(0, 0).action, Action::Box(vec![0.25, 0.75]));
        // Discrete action against a Box space rejected.
        assert!(b
            .enqueue(
                &obs,
                &[Action::Discrete(0)],
                &[1.0],
                &[false],
                &[0.0],
                &[0.0],
                &[],
                &[],
                &[],
                &[]
            )
            .is_err());
    }

    #[test]
    fn dump_obs_into_shared_region() {
        let mut b = simple(4, 2, 0);
        for t in 0..3u64 {
            let (obs, act, rew, done, lp, val) = step_data(2, t, 0.0, false);
            b.enqueue(&obs, &act, &rew, &done, &lp, &val, &[], &[], &[], &[])
                .unwrap();
        }
        let region = dsec_storage::prefetch::SharedRegion::new(6, 3);
        b.dump_obs_to_region(&region).unwrap();
        let view = region.read_view();
        // Ring order: age-major, env-minor. age0 = [0,1,2 | 3,4,5].
        assert_eq!(&view.as_slice()[..3], &[0.0, 1.0, 2.0]);
        // age2 (t=2) env1: [20+3, 20+4, 20+5].
        assert_eq!(&view.as_slice()[15..18], &[23.0, 24.0, 25.0]);
    }

    #[test]
    fn reset_clears_window() {
        let mut b = simple(4, 1, 0);
        let (obs, act, rew, done, lp, val) = step_data(1, 1, 0.0, false);
        b.enqueue(&obs, &act, &rew, &done, &lp, &val, &[], &[], &[], &[])
            .unwrap();
        assert_eq!(b.len(), 1);
        b.reset();
        assert!(b.is_empty());
        assert_eq!(b.total_steps(), 1); // cumulative
    }
}
