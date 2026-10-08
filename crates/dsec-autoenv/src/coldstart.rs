//! Cold-Start fine-tuning data and paired controls (Section 4.3, App. H).
//!
//! "The base model generates valid environments only 32.3% of the time,
//! too rarely to start training from. We therefore cold-start it by
//! fine-tuning on 600 high-quality proposer trajectories." (Appendix H:
//! "Cold-Start fine-tunes Qwen3.6-35B-A3B on 600 high-quality proposer
//! trajectories from Claude Opus 5, DeepSeek-V4-Flash, and Kimi-K3.")
//!
//! Quality selection: a trajectory is high-quality when its episode was
//! **admitted** (validation and calibration both passed — Eq. 2 reward
//! `+1`). Selection deduplicates by assignment so the fine-tuning set
//! covers distinct generation problems, then caps at the configured size.
//!
//! The paired controls reproduce Appendix H's protocol: the same 60 (or
//! 136) generation assignments run against both models, disagreements are
//! counted, and the paired sign test plus a bootstrap interval on the
//! solving difference decide whether environment-generation skill moved
//! without a detected change in solving.

use crate::trajectory::{Provenance, Trajectory};
use serde::{Deserialize, Serialize};

/// Cold-Start configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColdStartConfig {
    /// Number of fine-tuning trajectories (paper: 600).
    pub n_trajectories: usize,
    /// The frontier proposers the trajectories come from.
    pub sources: Vec<String>,
}

impl Default for ColdStartConfig {
    fn default() -> Self {
        ColdStartConfig {
            n_trajectories: 600,
            sources: vec![
                "Claude Opus 5".into(),
                "DeepSeek-V4-Flash".into(),
                "Kimi-K3".into(),
            ],
        }
    }
}

/// One frontier proposer episode offered as Cold-Start material.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateEpisode {
    /// The assignment id (deduplication key).
    pub assignment_id: String,
    /// The frontier model that produced it.
    pub source: String,
    /// Whether the episode's environment was admitted (Eq. 2 reward `+1`).
    pub admitted: bool,
    /// The trajectory itself.
    pub trajectory: Trajectory,
}

impl CandidateEpisode {
    /// Eq. 2-style reward of the episode.
    pub fn reward(&self) -> f64 {
        if self.admitted {
            1.0
        } else {
            -1.0
        }
    }
}

/// The selected Cold-Start fine-tuning set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColdStartSet {
    /// The selected trajectories.
    pub trajectories: Vec<Trajectory>,
    /// Their sources.
    pub sources: Vec<String>,
    /// Distinct assignments covered.
    pub distinct_assignments: usize,
}

impl ColdStartSet {
    /// Size of the set.
    pub fn len(&self) -> usize {
        self.trajectories.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.trajectories.is_empty()
    }
}

/// Select the Cold-Start set: admitted trajectories only, deduplicated by
/// assignment, capped at `config.n_trajectories`.
pub fn select_cold_start(
    config: &ColdStartConfig,
    candidates: &[CandidateEpisode],
) -> ColdStartSet {
    let mut seen_assignments = std::collections::BTreeSet::new();
    let mut trajectories = Vec::new();
    let mut sources = Vec::new();
    for cand in candidates.iter().filter(|c| c.admitted) {
        if trajectories.len() >= config.n_trajectories {
            break;
        }
        if seen_assignments.insert(cand.assignment_id.clone()) {
            trajectories.push(cand.trajectory.clone());
            sources.push(cand.source.clone());
        }
    }
    ColdStartSet {
        distinct_assignments: seen_assignments.len(),
        trajectories,
        sources,
    }
}

/// Annotate a trajectory with frontier provenance (for the archive's
/// `provenance` field).
pub fn frontier_provenance(source: &str) -> Provenance {
    Provenance::Frontier(source.to_string())
}

/// One paired evaluation over the same assignments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedEvaluation {
    /// Per-assignment outcomes for the base model (`true` = accepted
    /// environment).
    pub base: Vec<bool>,
    /// Per-assignment outcomes for the Cold-Start model.
    pub treatment: Vec<bool>,
}

impl PairedEvaluation {
    /// Build from outcome pairs.
    pub fn new(base: Vec<bool>, treatment: Vec<bool>) -> Self {
        PairedEvaluation { base, treatment }
    }

    /// Acceptance rates of both arms.
    pub fn acceptance_rates(&self) -> (f64, f64) {
        let rate = |v: &[bool]| {
            if v.is_empty() {
                0.0
            } else {
                v.iter().filter(|b| **b).count() as f64 / v.len() as f64
            }
        };
        (rate(&self.base), rate(&self.treatment))
    }

    /// Disagreements split into (treatment-wins, base-wins). Ties are
    /// excluded from the sign test.
    pub fn disagreements(&self) -> (usize, usize) {
        let mut treatment_wins = 0;
        let mut base_wins = 0;
        for (b, t) in self.base.iter().zip(&self.treatment) {
            if *t && !*b {
                treatment_wins += 1;
            } else if *b && !*t {
                base_wins += 1;
            }
        }
        (treatment_wins, base_wins)
    }

    /// Two-sided paired sign test on the acceptance disagreements.
    pub fn sign_test_p(&self) -> f64 {
        let (wins, losses) = self.disagreements();
        crate::metrics::paired_sign_test(wins, losses)
    }
}

/// Per-task solving scores of two arms over the same tasks (for the
/// solving-difference control: Table 13's "mean paired change in solving
/// performance is 1.1 percentage points (95% bootstrap interval:
/// [-4.9, 7.4]; paired sign test p = 0.88)").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolvingPairedScores {
    /// Per-task scores of the base model.
    pub base: Vec<f64>,
    /// Per-task scores of the Cold-Start model.
    pub treatment: Vec<f64>,
}

impl SolvingPairedScores {
    /// Per-task paired differences (treatment - base).
    pub fn differences(&self) -> Vec<f64> {
        self.base
            .iter()
            .zip(&self.treatment)
            .map(|(b, t)| t - b)
            .collect()
    }

    /// Mean paired change.
    pub fn mean_change(&self) -> f64 {
        crate::metrics::mean(&self.differences())
    }

    /// 95% bootstrap interval of the mean paired change.
    pub fn bootstrap_interval(&self, iterations: usize, seed: u64) -> (f64, f64) {
        crate::metrics::bootstrap_ci_mean(&self.differences(), iterations, 0.95, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::Outcome;
    use crate::trajectory::Role;

    fn traj() -> Trajectory {
        Trajectory {
            role: Role::Proposer,
            prompt: "assignment".into(),
            turns: vec![],
            outcome: Some(Outcome {
                reward: 1.0,
                verifier_stdout: String::new(),
                passed_tests: 4,
                total_tests: 4,
                failed_checks: vec![],
            }),
        }
    }

    fn candidate(id: &str, source: &str, admitted: bool) -> CandidateEpisode {
        CandidateEpisode {
            assignment_id: id.into(),
            source: source.into(),
            admitted,
            trajectory: traj(),
        }
    }

    #[test]
    fn config_defaults() {
        let c = ColdStartConfig::default();
        assert_eq!(c.n_trajectories, 600);
        assert_eq!(
            c.sources,
            vec![
                "Claude Opus 5".to_string(),
                "DeepSeek-V4-Flash".to_string(),
                "Kimi-K3".to_string()
            ]
        );
    }

    #[test]
    fn selection_filters_admits_and_dedupes() {
        let candidates: Vec<CandidateEpisode> = vec![
            candidate("a1", "Claude Opus 5", true),
            candidate("a2", "DeepSeek-V4-Flash", false),
            candidate("a1", "Kimi-K3", true), // duplicate assignment
            candidate("a3", "Kimi-K3", true),
            candidate("a4", "Claude Opus 5", false),
            candidate("a5", "Claude Opus 5", true),
        ];
        let set = select_cold_start(&ColdStartConfig::default(), &candidates);
        assert_eq!(set.len(), 3);
        assert_eq!(set.distinct_assignments, 3);
        assert_eq!(
            set.sources,
            vec![
                "Claude Opus 5".to_string(),
                "Kimi-K3".to_string(),
                "Claude Opus 5".to_string()
            ]
        );
    }

    #[test]
    fn selection_caps_at_configured_size() {
        let candidates: Vec<CandidateEpisode> = (0..50)
            .map(|i| candidate(&format!("a{i}"), "Kimi-K3", true))
            .collect();
        let config = ColdStartConfig {
            n_trajectories: 10,
            ..Default::default()
        };
        let set = select_cold_start(&config, &candidates);
        assert_eq!(set.len(), 10);
    }

    #[test]
    fn paired_sign_test_on_disagreements() {
        // Appendix H: 32 favor Cold-Start, 6 favor base, of 38
        // disagreements.
        let mut base = vec![false; 38];
        let mut treatment = vec![false; 38];
        for (i, t) in treatment.iter_mut().enumerate().take(32) {
            *t = true;
            let _ = i;
        }
        for b in base.iter_mut().skip(32).take(6) {
            *b = true;
        }
        let paired = PairedEvaluation::new(base, treatment);
        assert_eq!(paired.disagreements(), (32, 6));
        let p = paired.sign_test_p();
        assert!(p > 0.0 && p <= 1.0);
        // Strong asymmetry -> small p.
        assert!(p < 0.01, "p={p}");
    }

    #[test]
    fn acceptance_rates() {
        let paired = PairedEvaluation::new(
            vec![true, false, true, false],
            vec![true, true, false, false],
        );
        let (base, treatment) = paired.acceptance_rates();
        assert!((base - 0.5).abs() < 1e-9);
        assert!((treatment - 0.5).abs() < 1e-9);
        assert_eq!(paired.disagreements(), (1, 1));
        assert!(paired.sign_test_p() >= 0.99);
    }

    #[test]
    fn solving_paired_scores_table13_shape() {
        // Table 13: base 40.0 vs Cold-Start 41.1 avg@5 with a not
        // statistically significant difference (per-task variation kept
        // so the bootstrap interval is meaningful).
        let base: Vec<f64> = (0..89)
            .map(|i| 0.40 + ((i % 7) as f64 - 3.0) * 0.01)
            .collect();
        let treatment: Vec<f64> = (0..89)
            .map(|i| 0.40 + ((i % 5) as f64 - 2.0) * 0.012 + 0.011)
            .collect();
        let scores = SolvingPairedScores { base, treatment };
        assert!(
            (scores.mean_change() - 0.011).abs() < 5e-4,
            "mean={}",
            scores.mean_change()
        );
        let (lo, hi) = scores.bootstrap_interval(500, 3);
        assert!(lo < 0.011 && hi > 0.011);
    }

    #[test]
    fn frontier_provenance_tags_source() {
        assert_eq!(
            frontier_provenance("Claude Opus 5"),
            Provenance::Frontier("Claude Opus 5".into())
        );
    }
}
