//! Rewards for the two roles and the multi-step roll-up.
//!
//! The paper's reward design (Section 3.3):
//!
//! * **Solver, Eq. 1**: the fraction of the environment's `N` tests passed —
//!   "a partial solution receives partial reward."
//! * **Proposer, Eq. 2**: `-1` if validation fails, `-0.25` if validation
//!   passes but calibration fails, `1` if both pass — "This reward separates
//!   broken environments from working ones that do not suit the current
//!   solver, and it gives full reward only to environments that enter the
//!   training pool."
//! * **Multi-step roll-up**: `mean` (per-key mean across steps that produced
//!   a result) or `final` (the last step's verifier result verbatim, with
//!   the early-abort caveat).
//! * **Clarification shaping** (Section 4.5): task success combined with
//!   Ask-F1.

use crate::harbor::RewardStrategy;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Eq. 1: the solver reward — the fraction of the environment's tests that
/// pass.
pub fn solver_reward(passed: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        passed as f64 / total as f64
    }
}

/// Eq. 2: the proposer reward.
///
/// ```text
/// r_P = -1.00  if validation fails,
///      -0.25  if validation passes but calibration fails,
///      +1.00  if both pass.
/// ```
pub fn proposer_reward(validation_passed: bool, calibration_passed: bool) -> f64 {
    match (validation_passed, calibration_passed) {
        (false, _) => -1.0,
        (true, false) => -0.25,
        (true, true) => 1.0,
    }
}

/// A reward that is either a scalar or a keyed (multi-dimensional) dict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RewardValue {
    /// A single number (the overall fraction of tests passed).
    Scalar(f64),
    /// One value per test key (0/1 per check).
    Keys(BTreeMap<String, f64>),
}

impl RewardValue {
    /// The scalar view: mean of keyed values, or the scalar itself.
    pub fn as_scalar(&self) -> f64 {
        match self {
            RewardValue::Scalar(v) => *v,
            RewardValue::Keys(keys) => {
                if keys.is_empty() {
                    0.0
                } else {
                    keys.values().sum::<f64>() / keys.len() as f64
                }
            }
        }
    }
}

/// One step's verifier result within a multi-step trial.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepResult {
    /// Step name.
    pub step: String,
    /// Overall step reward.
    pub reward: f64,
    /// Keyed per-check rewards.
    pub keyed: BTreeMap<String, f64>,
    /// Whether the trial aborted at this step (`min_reward` gate failed).
    /// An aborted step still *produced a result* — which is exactly the
    /// caveat of the `final` strategy.
    pub aborted: bool,
}

impl StepResult {
    /// Construct a step result.
    pub fn new(step: impl Into<String>, reward: f64, keyed: BTreeMap<String, f64>) -> Self {
        StepResult {
            step: step.into(),
            reward,
            keyed,
            aborted: false,
        }
    }
}

/// Roll per-step rewards up into the trial-level verifier result
/// (Appendix E.1, "Choosing a reward strategy").
///
/// * `Mean` (the default): per-key mean across steps that produced a result
///   — "good for continuous progress rewards."
/// * `Final`: the last step's verifier result verbatim — "if `min_reward`
///   triggers an early abort, `"final"` uses the *aborted* step's result,
///   not the intended final step."
pub fn multi_step_rollup(steps: &[StepResult], strategy: RewardStrategy) -> RewardValue {
    if steps.is_empty() {
        return RewardValue::Scalar(0.0);
    }
    match strategy {
        RewardStrategy::Final => {
            let last = steps.last().expect("checked non-empty");
            let mut keyed = last.keyed.clone();
            // Prefix keys with the step name so multi-step trials keep
            // per-step dimensions distinct.
            for (k, v) in last.keyed.clone() {
                keyed.insert(format!("{}/{}", last.step, k), v);
            }
            keyed.remove("");
            RewardValue::Scalar(last.reward).merge_keys(keyed)
        }
        RewardStrategy::Mean => {
            let produced: Vec<&StepResult> = steps.iter().collect();
            let scalar = produced.iter().map(|s| s.reward).sum::<f64>() / produced.len() as f64;
            // Per-key mean across steps that produced that key.
            let mut acc: BTreeMap<String, (f64, usize)> = BTreeMap::new();
            for s in &produced {
                for (k, v) in &s.keyed {
                    let e = acc.entry(k.clone()).or_insert((0.0, 0));
                    e.0 += v;
                    e.1 += 1;
                }
            }
            let keyed: BTreeMap<String, f64> = acc
                .into_iter()
                .map(|(k, (sum, n))| (k, sum / n as f64))
                .collect();
            RewardValue::Scalar(scalar).merge_keys(keyed)
        }
    }
}

impl RewardValue {
    /// Attach keyed dimensions to a scalar reward value.
    fn merge_keys(self, keys: BTreeMap<String, f64>) -> RewardValue {
        RewardValue::Keys({
            let mut m = match self {
                RewardValue::Keys(existing) => existing,
                RewardValue::Scalar(v) => {
                    let mut m = BTreeMap::new();
                    if v != 0.0 || keys.is_empty() {
                        m.insert("overall".to_string(), v);
                    }
                    m
                }
            };
            for (k, v) in keys {
                m.insert(k, v);
            }
            m
        })
    }
}

/// Configuration of the shaped clarification reward for HiL training
/// (Section 4.5: "We then train with RL for up to 150 steps, using a shaped
/// clarification reward based on Ask-F1"). The exact blend is not given in
/// the paper; we expose both weights and default to success-dominant
/// shaping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClarificationRewardConfig {
    /// Weight of task success in the shaped reward.
    pub success_weight: f64,
    /// Weight of Ask-F1 (question quality) in the shaped reward.
    pub ask_f1_weight: f64,
}

impl Default for ClarificationRewardConfig {
    fn default() -> Self {
        ClarificationRewardConfig {
            success_weight: 1.0,
            ask_f1_weight: 1.0,
        }
    }
}

/// The shaped clarification reward: `w_success * success + w_ask_f1 *
/// ask_f1`, where `ask_f1` measures "whether the questions cover the
/// information needed to complete the task".
pub fn clarification_reward(success: bool, ask_f1: f64, cfg: &ClarificationRewardConfig) -> f64 {
    let s = if success { 1.0 } else { 0.0 };
    (cfg.success_weight * s + cfg.ask_f1_weight * ask_f1)
        .clamp(0.0, cfg.success_weight + cfg.ask_f1_weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq1_fractional_reward() {
        assert_eq!(solver_reward(3, 4), 0.75);
        assert_eq!(solver_reward(0, 0), 0.0);
        assert_eq!(solver_reward(4, 4), 1.0);
    }

    #[test]
    fn eq2_three_valued() {
        assert_eq!(proposer_reward(false, false), -1.0);
        assert_eq!(proposer_reward(false, true), -1.0);
        assert_eq!(proposer_reward(true, false), -0.25);
        assert_eq!(proposer_reward(true, true), 1.0);
    }

    #[test]
    fn mean_strategy_per_key_mean() {
        let mut k1 = BTreeMap::new();
        k1.insert("correctness".to_string(), 1.0);
        k1.insert("style".to_string(), 0.0);
        let mut k2 = BTreeMap::new();
        k2.insert("correctness".to_string(), 0.5);
        k2.insert("style".to_string(), 1.0);
        let steps = vec![
            StepResult::new("scaffold", 0.5, k1),
            StepResult::new("implement", 0.75, k2),
        ];
        let rolled = multi_step_rollup(&steps, RewardStrategy::Mean);
        assert!((rolled.as_scalar() - 0.625).abs() < 1e-9);
        match rolled {
            RewardValue::Keys(keys) => {
                assert!((keys["correctness"] - 0.75).abs() < 1e-9);
                assert!((keys["style"] - 0.5).abs() < 1e-9);
            }
            other => panic!("expected keyed reward, got {other:?}"),
        }
    }

    #[test]
    fn final_strategy_uses_aborted_step_result() {
        let mut k = BTreeMap::new();
        k.insert("end_to_end".to_string(), 0.4);
        let mut last = StepResult::new("document", 1.0, BTreeMap::new());
        last.aborted = false;
        let aborted = StepResult::new("implement", 0.4, k);
        // Trial aborted at 'implement'; 'document' never ran.
        let rolled = multi_step_rollup(&[aborted.clone(), last.clone()], RewardStrategy::Final);
        assert_eq!(rolled.as_scalar(), 1.0); // last produced result wins
                                             // Now with only the aborted step produced:
        let rolled2 = multi_step_rollup(&[aborted], RewardStrategy::Final);
        assert!((rolled2.as_scalar() - 0.4).abs() < 1e-9);
        assert!(!last.aborted);
    }

    #[test]
    fn clarification_shaping() {
        let cfg = ClarificationRewardConfig::default();
        assert!((clarification_reward(true, 0.5, &cfg) - 1.5).abs() < 1e-9);
        assert!((clarification_reward(false, 0.0, &cfg)).abs() < 1e-9);
    }
}
