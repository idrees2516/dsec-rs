//! DPPO — Divergence Proximal Policy Optimization (Section 3.3, App. G.1).
//!
//! "We use Divergence Proximal Policy Optimization (DPPO with separate
//! reward groups for the two roles." The two DPPO ingredients the paper
//! pins down are:
//!
//! * a **total-variation trust region**: "DPPO total-variation threshold
//!   0.1" (Table 12) — a policy update is admissible only while the TV
//!   distance between the new and old policy stays below the threshold;
//! * standard PPO mechanics on top: importance ratios against stored
//!   sampling log-probabilities, a clipped surrogate, and
//!   "KL / entropy coefficients: 0/0".
//!
//! Policies here are tabular softmax distributions over token/action ids,
//! keyed by context (one table per skill for the solver, one per design
//! decision for the proposer). That is the smallest faithful stand-in for
//! the token-level softmax of an LLM: every update step is a real gradient
//! step on the clipped surrogate with a real TV projection, so the math —
//! ratios, clipping, trust region, carry-over importance sampling — is
//! exercised end-to-end and deterministically.
//!
//! Carry-over (Appendix G.1): "Unfinished terminal trajectories are carried
//! across updates, preserving their interaction history and live sandbox
//! state. They resume with the updated policy, while stored sampling
//! log-probabilities are retained for previously generated tokens."

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// DPPO hyper-parameters. Defaults follow Table 12 where they apply; the
/// learning rate for the tabular demo policies is configurable because
/// `1e-6` would need millions of updates to move a tabular softmax.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DppoConfig {
    /// Learning rate (Table 12: 1e-6 for the 35B-A3B run).
    pub lr: f64,
    /// DPPO total-variation threshold (Table 12: 0.1).
    pub tv_threshold: f64,
    /// PPO clip epsilon.
    pub clip_eps: f64,
    /// KL coefficient (Table 12: 0).
    pub kl_coef: f64,
    /// Entropy coefficient (Table 12: 0).
    pub entropy_coef: f64,
}

impl Default for DppoConfig {
    fn default() -> Self {
        DppoConfig {
            lr: 0.05,
            tv_threshold: 0.1,
            clip_eps: 0.2,
            kl_coef: 0.0,
            entropy_coef: 0.0,
        }
    }
}

impl DppoConfig {
    /// The paper's exact Table 12 settings.
    pub fn paper() -> Self {
        DppoConfig {
            lr: 1e-6,
            tv_threshold: 0.1,
            clip_eps: 0.2,
            kl_coef: 0.0,
            entropy_coef: 0.0,
        }
    }
}

/// A tabular softmax policy: logits over a fixed action/token vocabulary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoftmaxPolicy {
    /// Logits, one per action.
    pub logits: Vec<f64>,
}

impl SoftmaxPolicy {
    /// Uniform-initialize a policy over `n` actions.
    pub fn uniform(n: usize) -> Self {
        SoftmaxPolicy {
            logits: vec![0.0; n],
        }
    }

    /// Softmax probabilities.
    pub fn probs(&self) -> Vec<f64> {
        let max = self
            .logits
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = self.logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    /// Log-probability of an action.
    pub fn logprob(&self, action: usize) -> f64 {
        let probs = self.probs();
        probs[action].ln()
    }

    /// Sample an action.
    pub fn sample(&self, rng: &mut impl rand::Rng) -> usize {
        let probs = self.probs();
        let r: f64 = rng.gen();
        let mut acc = 0.0;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if r < acc {
                return i;
            }
        }
        probs.len() - 1
    }

    /// Total-variation distance to another policy:
    /// `TV(p, q) = 0.5 * Σ |p_i - q_i|`.
    pub fn tv(&self, other: &SoftmaxPolicy) -> f64 {
        let p = self.probs();
        let q = other.probs();
        let n = p.len().max(q.len());
        let mut d = 0.0;
        for i in 0..n {
            let a = p.get(i).copied().unwrap_or(0.0);
            let b = q.get(i).copied().unwrap_or(0.0);
            d += (a - b).abs();
        }
        0.5 * d
    }
}

/// A bank of tabular policies keyed by context id (skill tag, design
/// decision, ...). Each key's table is updated only from the steps sampled
/// under that key.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PolicyBank {
    /// Context-keyed policies.
    pub tables: BTreeMap<String, SoftmaxPolicy>,
}

impl PolicyBank {
    /// Fetch or create the policy for a context (uniform over `n` actions).
    pub fn table(&mut self, context: &str, n_actions: usize) -> &SoftmaxPolicy {
        self.tables
            .entry(context.to_string())
            .or_insert_with(|| SoftmaxPolicy::uniform(n_actions))
    }

    /// Immutable lookup.
    pub fn get(&self, context: &str) -> Option<&SoftmaxPolicy> {
        self.tables.get(context)
    }

    /// Sample an action for a context, returning (action, logprob).
    pub fn sample(
        &mut self,
        context: &str,
        n_actions: usize,
        rng: &mut impl rand::Rng,
    ) -> (usize, f64) {
        let policy = self
            .tables
            .entry(context.to_string())
            .or_insert_with(|| SoftmaxPolicy::uniform(n_actions));
        let a = policy.sample(rng);
        let lp = policy.logprob(a);
        (a, lp)
    }
}

/// One training step: a sampled token/action with its stored sampling
/// log-probability and its (already computed, Eq. 3) advantage.
///
/// The stored `old_logprob` is what makes carry-over correct: tokens
/// generated by a previous policy version keep their original sampling
/// log-probabilities ("stored sampling log-probabilities are retained for
/// previously generated tokens"), and the importance ratio
/// `exp(logprob_new - old_logprob)` corrects for the policy change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenStep {
    /// Context key of the policy table.
    pub context: String,
    /// The sampled action id.
    pub action: usize,
    /// Stored sampling log-probability (from the policy that sampled it).
    pub old_logprob: f64,
    /// Group-relative advantage (Eq. 3).
    pub advantage: f64,
}

/// Summary of one DPPO update.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpdateSummary {
    /// TV distance before the trust-region projection.
    pub tv_raw: f64,
    /// TV distance after the projection (<= threshold).
    pub tv_final: f64,
    /// Scale applied to the raw gradient step by the line search.
    pub accepted_scale: f64,
    /// Fraction of steps whose ratio left the clip range.
    pub clipped_frac: f64,
    /// Mean importance ratio.
    pub mean_ratio: f64,
    /// Number of steps trained on.
    pub steps: usize,
}

/// The DPPO optimizer: clipped-surrogate gradient ascent with a
/// total-variation trust region enforced by exact line search.
#[derive(Debug, Clone, PartialEq)]
pub struct DppoOptimizer {
    /// Configuration.
    pub config: DppoConfig,
}

impl DppoOptimizer {
    /// Construct with a configuration.
    pub fn new(config: DppoConfig) -> Self {
        DppoOptimizer { config }
    }

    /// Importance ratio `ρ = exp(logπ_θ(a) - logπ_old(a))`.
    pub fn ratio(&self, new_logprob: f64, old_logprob: f64) -> f64 {
        (new_logprob - old_logprob).exp()
    }

    /// The clipped surrogate objective for one policy table:
    /// `L = mean_t min(ρ_t Â_t, clip(ρ_t, 1-ε, 1+ε) Â_t)`.
    pub fn surrogate(&self, policy: &SoftmaxPolicy, steps: &[TokenStep]) -> f64 {
        if steps.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        for s in steps {
            let rho = self.ratio(policy.logprob(s.action), s.old_logprob);
            let clipped = rho.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps);
            total += rho.min(clipped) * s.advantage;
        }
        total / steps.len() as f64
    }

    /// Analytic gradient of the surrogate w.r.t. the policy logits.
    ///
    /// Only unclipped terms contribute; `dlogπ(a)/dz_j = 1[j=a] − π(j)`.
    fn surrogate_grad(&self, policy: &SoftmaxPolicy, steps: &[TokenStep]) -> Vec<f64> {
        let probs = policy.probs();
        let mut grad = vec![0.0; probs.len()];
        if steps.is_empty() {
            return grad;
        }
        for s in steps {
            if s.action >= probs.len() {
                continue;
            }
            let rho = self.ratio(policy.logprob(s.action), s.old_logprob);
            let clipped = rho.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps);
            // If the clipped branch is the active min, gradient is zero.
            if rho.min(clipped) == clipped && rho != clipped {
                continue;
            }
            for j in 0..probs.len() {
                let indicator = if j == s.action { 1.0 } else { 0.0 };
                grad[j] += rho * s.advantage * (indicator - probs[j]);
            }
        }
        for g in &mut grad {
            *g /= steps.len() as f64;
        }
        grad
    }

    /// Update one policy table from its steps under the TV trust region.
    ///
    /// Procedure: take the raw gradient-ascent step; if the resulting TV
    /// distance from the old policy exceeds `tv_threshold`, binary-search
    /// the largest step scale `α ∈ (0, 1]` that respects the region (TV is
    /// continuous and TV(α=0) = 0, so the search always succeeds).
    pub fn update_policy(&self, policy: &mut SoftmaxPolicy, steps: &[TokenStep]) -> UpdateSummary {
        let old = policy.clone();
        let grad = self.surrogate_grad(policy, steps);
        let raw_step: Vec<f64> = grad.iter().map(|g| self.config.lr * g).collect();

        let mut candidate = policy.clone();
        for (i, d) in raw_step.iter().enumerate() {
            if i < candidate.logits.len() {
                candidate.logits[i] += d;
            }
        }
        let tv_raw = old.tv(&candidate);

        // Trust-region projection.
        let mut alpha = 1.0;
        if tv_raw > self.config.tv_threshold {
            let mut lo = 0.0f64;
            let mut hi = 1.0f64;
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                let mut interp = old.clone();
                for i in 0..interp.logits.len() {
                    if i < raw_step.len() {
                        interp.logits[i] += mid * raw_step[i];
                    }
                }
                if old.tv(&interp) <= self.config.tv_threshold {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            alpha = lo;
        }

        for i in 0..policy.logits.len() {
            if i < raw_step.len() {
                policy.logits[i] += alpha * raw_step[i];
            }
        }

        // Diagnostics.
        let mut clipped = 0usize;
        let mut ratio_sum = 0.0;
        for s in steps {
            let rho = self.ratio(
                policy.logprob(s.action.min(policy.logits.len() - 1)),
                s.old_logprob,
            );
            let cl = rho.clamp(1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps);
            if rho != cl {
                clipped += 1;
            }
            ratio_sum += rho;
        }
        UpdateSummary {
            tv_raw,
            tv_final: old.tv(policy),
            accepted_scale: alpha,
            clipped_frac: if steps.is_empty() {
                0.0
            } else {
                clipped as f64 / steps.len() as f64
            },
            mean_ratio: if steps.is_empty() {
                1.0
            } else {
                ratio_sum / steps.len() as f64
            },
            steps: steps.len(),
        }
    }

    /// Update every policy table in a bank from the given steps (grouped by
    /// context).
    pub fn update_bank(
        &self,
        bank: &mut PolicyBank,
        steps: &[TokenStep],
    ) -> Vec<(String, UpdateSummary)> {
        let mut by_context: BTreeMap<String, Vec<TokenStep>> = BTreeMap::new();
        for s in steps {
            by_context
                .entry(s.context.clone())
                .or_default()
                .push(s.clone());
        }
        let mut summaries = Vec::new();
        for (ctx, ctx_steps) in by_context {
            let mut policy = bank
                .tables
                .get(&ctx)
                .cloned()
                .unwrap_or_else(|| SoftmaxPolicy::uniform(2));
            let summary = self.update_policy(&mut policy, &ctx_steps);
            bank.tables.insert(ctx.clone(), policy);
            summaries.push((ctx, summary));
        }
        summaries
    }
}

/// A trajectory carried across updates because it had not finished when the
/// update landed (Appendix G.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CarryTrajectory {
    /// Role of the trajectory.
    pub role: crate::trajectory::Role,
    /// Group key (environment for solvers, assignment for proposers).
    pub group_key: String,
    /// The prompt the episode started from.
    pub prompt: String,
    /// Tokens generated so far, with their stored sampling log-probs.
    pub steps: Vec<TokenStep>,
    /// Whether the episode has since completed.
    pub done: bool,
}

/// The buffer of unfinished trajectories.
///
/// "Unfinished terminal trajectories are carried across updates, preserving
/// their interaction history and live sandbox state. They resume with the
/// updated policy, while stored sampling log-probabilities are retained for
/// previously generated tokens."
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CarryOverBuffer {
    /// The unfinished trajectories.
    pub pending: Vec<CarryTrajectory>,
}

impl CarryOverBuffer {
    /// Park an unfinished trajectory.
    pub fn park(&mut self, traj: CarryTrajectory) {
        self.pending.push(traj);
    }

    /// Number of parked trajectories.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Take back all parked trajectories for resumption with the updated
    /// policy. Old steps keep their stored log-probabilities — the
    /// importance ratios in the next update correct for the policy change.
    pub fn drain(&mut self) -> Vec<CarryTrajectory> {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_steps(policy: &SoftmaxPolicy, choices: &[(usize, f64)]) -> Vec<TokenStep> {
        choices
            .iter()
            .map(|(a, adv)| TokenStep {
                context: "ctx".into(),
                action: *a,
                old_logprob: policy.logprob(*a),
                advantage: *adv,
            })
            .collect()
    }

    #[test]
    fn uniform_policy_tv_zero() {
        let p = SoftmaxPolicy::uniform(3);
        assert!(p.tv(&p.clone()) < 1e-12);
    }

    #[test]
    fn tv_symmetric_bounded() {
        let p = SoftmaxPolicy {
            logits: vec![2.0, 0.0],
        };
        let q = SoftmaxPolicy {
            logits: vec![0.0, 2.0],
        };
        assert!((p.tv(&q) - q.tv(&p)).abs() < 1e-12);
        assert!(p.tv(&q) <= 1.0 + 1e-12);
    }

    #[test]
    fn advantage_follows_gradient() {
        // Steps that all prefer action 0 with positive advantage should
        // raise P(0).
        let opt = DppoOptimizer::new(DppoConfig {
            lr: 0.5,
            ..Default::default()
        });
        let mut policy = SoftmaxPolicy::uniform(2);
        let steps = make_steps(&policy, &[(0, 1.0), (0, 1.0)]);
        let before = policy.probs()[0];
        opt.update_policy(&mut policy, &steps);
        assert!(policy.probs()[0] > before);
    }

    #[test]
    fn tv_trust_region_respected() {
        // A huge learning rate must still be projected back inside the
        // trust region.
        let opt = DppoOptimizer::new(DppoConfig {
            lr: 50.0,
            tv_threshold: 0.1,
            ..Default::default()
        });
        let mut policy = SoftmaxPolicy::uniform(4);
        let old = policy.clone();
        let steps = make_steps(&policy, &[(0, 1.0), (0, 1.0), (1, -1.0)]);
        let summary = opt.update_policy(&mut policy, &steps);
        assert!(
            summary.tv_raw > 0.1,
            "raw step should have breached the region"
        );
        assert!(summary.tv_final <= 0.1 + 1e-9);
        assert!(summary.accepted_scale < 1.0);
        assert!(old.tv(&policy) <= 0.1 + 1e-9);
    }

    #[test]
    fn clipping_zeroes_extreme_ratios() {
        let opt = DppoOptimizer::new(DppoConfig::default());
        let policy = SoftmaxPolicy {
            logits: vec![5.0, 0.0],
        };
        // Old logprob from a very different policy -> huge ratio.
        let steps = vec![TokenStep {
            context: "c".into(),
            action: 0,
            old_logprob: -20.0,
            advantage: 1.0,
        }];
        // Surrogate stays bounded by the clip range.
        let s = opt.surrogate(&policy, &steps);
        assert!(s <= (1.0 + opt.config.clip_eps) + 1e-9);
    }

    #[test]
    fn paper_config_values() {
        let c = DppoConfig::paper();
        assert_eq!(c.tv_threshold, 0.1);
        assert_eq!(c.kl_coef, 0.0);
        assert_eq!(c.entropy_coef, 0.0);
        assert!((c.lr - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn carry_over_keeps_stored_logprobs() {
        let mut buf = CarryOverBuffer::default();
        buf.park(CarryTrajectory {
            role: crate::trajectory::Role::Solver,
            group_key: "env_1".into(),
            prompt: "solve".into(),
            steps: vec![TokenStep {
                context: "csv".into(),
                action: 1,
                old_logprob: -0.9,
                advantage: 0.0,
            }],
            done: false,
        });
        let drained = buf.drain();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].steps[0].old_logprob, -0.9);
        assert!(buf.is_empty());
    }

    #[test]
    fn bank_update_groups_by_context() {
        let opt = DppoOptimizer::new(DppoConfig {
            lr: 0.5,
            ..Default::default()
        });
        let mut bank = PolicyBank::default();
        let p_csv = bank.table("csv", 2).clone();
        let p_py = bank.table("py", 2).clone();
        let steps = vec![
            TokenStep {
                context: "csv".into(),
                action: 0,
                old_logprob: p_csv.logprob(0),
                advantage: 1.0,
            },
            TokenStep {
                context: "py".into(),
                action: 1,
                old_logprob: p_py.logprob(1),
                advantage: -1.0,
            },
        ];
        let summaries = opt.update_bank(&mut bank, &steps);
        assert_eq!(summaries.len(), 2);
        assert!(bank.get("csv").unwrap().probs()[0] > 0.5);
        assert!(bank.get("py").unwrap().probs()[1] < 0.5);
    }
}
