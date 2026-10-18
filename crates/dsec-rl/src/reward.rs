//! Reward validity and group-relative advantage.
//!
//! The single most damaging silent bug in agentic RL is a **false-zero
//! reward**: the evaluator crashed, the verifier testbed was corrupted,
//! the judge was unreachable — and the pipeline recorded reward `0.0`
//! for a perfectly fine rollout. The model then learns that the
//! behavior was *bad*, which is worse than learning nothing.
//!
//! This module therefore keeps three states strictly separate:
//!
//! - [`RewardValidity::Valid`] — a real, usable score;
//! - [`RewardValidity::Invalid`] — the rollout genuinely failed the
//!   task (score is real, typically 0);
//! - [`RewardValidity::EvaluatorFailure`] — **infrastructure** failed;
//!   there is no score at all, the sample is untrainable, and it is
//!   excluded from the group statistics rather than counted as zero.
//!
//! Group advantages follow the GRPO/QWENGYRE shape: computed from the
//! group's scalar rewards over valid members only, no learned value
//! function.

use std::collections::BTreeMap;

/// Reward validity tri-state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewardValidity {
    /// Verifier ran and produced a trustworthy score.
    Valid,
    /// Verifier ran; the task genuinely failed (score stands).
    Invalid,
    /// Evaluator/infrastructure failure — no score exists. Never to be
    /// converted to reward zero.
    EvaluatorFailure,
}

impl RewardValidity {
    /// Whether this reward can be used for training.
    pub fn trainable(&self) -> bool {
        !matches!(self, RewardValidity::EvaluatorFailure)
    }
}

/// One execution's reward with validity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewardRecord {
    /// Execution id.
    pub execution_id: u64,
    /// Group id (advantages are computed within a group).
    pub group_id: u64,
    /// Raw verifier score (meaningless when validity is
    /// `EvaluatorFailure`).
    pub raw: f32,
    /// Validity classification.
    pub validity: RewardValidity,
}

impl RewardRecord {
    /// A valid reward.
    pub fn valid(execution_id: u64, group_id: u64, raw: f32) -> Self {
        Self {
            execution_id,
            group_id,
            raw,
            validity: RewardValidity::Valid,
        }
    }

    /// An evaluator failure (no score — never zero).
    pub fn evaluator_failure(execution_id: u64, group_id: u64) -> Self {
        Self {
            execution_id,
            group_id,
            raw: f32::NAN,
            validity: RewardValidity::EvaluatorFailure,
        }
    }

    /// The score, if valid or a genuine failure; `None` for evaluator
    /// failures (callers must not substitute 0.0).
    pub fn score(&self) -> Option<f32> {
        match self.validity {
            RewardValidity::EvaluatorFailure => None,
            _ => Some(self.raw),
        }
    }
}

/// Group advantage configuration.
#[derive(Debug, Clone)]
pub struct AdvantageConfig {
    /// Std stabilization epsilon (GRPO style).
    pub eps: f32,
}

impl Default for AdvantageConfig {
    fn default() -> Self {
        Self { eps: 1e-4 }
    }
}

/// Group-relative advantage: `A_i = (r_i - mean) / (std + eps)` over the
/// group's **valid** members. Evaluator failures get `None` (excluded),
/// never a numeric advantage — and never counted as zero in the
/// statistics.
///
/// A group with fewer than two scorable members yields all-`None`
/// (no baseline exists).
pub fn group_advantage(
    rewards: &[RewardRecord],
    config: &AdvantageConfig,
) -> BTreeMap<u64, Option<f32>> {
    let mut groups: BTreeMap<u64, Vec<&RewardRecord>> = BTreeMap::new();
    for r in rewards {
        groups.entry(r.group_id).or_default().push(r);
    }
    let mut out: BTreeMap<u64, Option<f32>> = BTreeMap::new();
    for (_gid, members) in groups {
        let scored: Vec<f32> = members.iter().filter_map(|m| m.score()).collect();
        if scored.len() < 2 {
            for m in &members {
                out.insert(m.execution_id, None);
            }
            continue;
        }
        let mean = scored.iter().sum::<f32>() / scored.len() as f32;
        let var = scored.iter().map(|r| (r - mean) * (r - mean)).sum::<f32>() / scored.len() as f32;
        let std = var.sqrt();
        for m in &members {
            let adv = m.score().map(|r| (r - mean) / (std + config.eps));
            out.insert(m.execution_id, adv);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluator_failure_is_never_reward_zero() {
        let r = RewardRecord::evaluator_failure(1, 7);
        // no score exists to be zero
        assert_eq!(r.score(), None);
        assert!(!r.validity.trainable());
        // and the raw is NaN, not 0
        assert!(r.raw.is_nan());
    }

    #[test]
    fn group_advantage_excludes_evaluator_failures() {
        // rewards: 1.0, 0.0 valid; one evaluator failure
        let rewards = vec![
            RewardRecord::valid(1, 1, 1.0),
            RewardRecord::valid(2, 1, 0.0),
            RewardRecord::evaluator_failure(3, 1),
        ];
        let adv = group_advantage(&rewards, &AdvantageConfig::default());
        // stats over the two valid members only: mean 0.5, std 0.5
        assert!((adv[&1].unwrap() - 1.0).abs() < 1e-3);
        assert!((adv[&2].unwrap() + 1.0).abs() < 1e-3);
        // the failed sample is excluded — None, NOT advantage 0
        assert_eq!(adv[&3], None);
    }

    #[test]
    fn if_failure_were_counted_as_zero_the_stats_would_differ() {
        // proves the exclusion matters: counting the failure as 0 changes
        // the mean
        let rewards = vec![
            RewardRecord::valid(1, 1, 1.0),
            RewardRecord::valid(2, 1, 0.6),
            RewardRecord::evaluator_failure(3, 1),
        ];
        let adv = group_advantage(&rewards, &AdvantageConfig::default());
        let a1 = adv[&1].unwrap();
        let a2 = adv[&2].unwrap();
        // mean over {1.0, 0.6} = 0.8
        assert!((a1 + 0.2f32).abs() < 1e-3 || (a1 - (0.2 / 0.2 + 1e-4)).abs() < 1e-3);
        assert!(a1 > 0.0 && a2 < 0.0);
        // with a zero-counted failure the mean would be 0.533 and the
        // sign pattern of a2 would flip for some rewards — the assertion
        // above pins the correct exclusion semantics.
    }

    #[test]
    fn singleton_group_has_no_baseline() {
        let rewards = vec![RewardRecord::valid(1, 9, 0.7)];
        let adv = group_advantage(&rewards, &AdvantageConfig::default());
        assert_eq!(adv[&1], None);
    }

    #[test]
    fn all_failures_group_is_fully_masked() {
        let rewards = vec![
            RewardRecord::evaluator_failure(1, 4),
            RewardRecord::evaluator_failure(2, 4),
        ];
        let adv = group_advantage(&rewards, &AdvantageConfig::default());
        assert!(adv.values().all(|v| v.is_none()));
    }
}
