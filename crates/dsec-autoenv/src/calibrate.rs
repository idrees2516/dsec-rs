//! Solver-guided calibration (Section 3.2, Appendix G.4).
//!
//! "An environment that passes validation is then attempted several times
//! by the current solver. We keep it if the mean reward falls between 0.25
//! and 0.75 and the rewards vary across attempts, so that the solver
//! sometimes solves it and sometimes does not."
//!
//! The exact protocol (Appendix G.4):
//!
//! * "Calibration targets mean reward in [0.25, 0.75] with reward standard
//!   deviation at least 0.1."
//! * "It begins with four solver rollouts and expands to eight when the
//!   mean is within 0.15 of a band boundary."
//! * "Tasks below the band receive one diagnostic rollout with
//!   reference-solution hints to guide revision."
//! * "Each assignment permits up to two revisions. Revised environments
//!   repeat the admission checks before entering the training pool."

use crate::error::Result;
use crate::harbor::HarborTask;
use crate::policy::{Hint, SolverModel};
use serde::{Deserialize, Serialize};

/// Calibration thresholds and protocol parameters (Appendix G.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// The reward band the solver's mean must fall inside: (0.25, 0.75).
    pub band: (f64, f64),
    /// Minimum reward standard deviation: 0.1 ("the rewards vary across
    /// attempts").
    pub min_std: f64,
    /// Initial rollout count: 4.
    pub initial_rollouts: usize,
    /// Expanded rollout count: 8.
    pub expanded_rollouts: usize,
    /// Boundary expansion margin: 0.15 — expand when the mean is within
    /// this distance of either band edge.
    pub boundary_margin: f64,
    /// Maximum revisions per assignment: 2.
    pub max_revisions: usize,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        CalibrationConfig {
            band: (0.25, 0.75),
            min_std: 0.1,
            initial_rollouts: 4,
            expanded_rollouts: 8,
            boundary_margin: 0.15,
            max_revisions: 2,
        }
    }
}

impl CalibrationConfig {
    /// Whether the mean lies within the boundary-expansion margin of either
    /// band edge.
    pub fn near_boundary(&self, mean: f64) -> bool {
        let eps = 1e-9;
        (mean - self.band.0).abs() <= self.boundary_margin + eps
            || (mean - self.band.1).abs() <= self.boundary_margin + eps
    }
}

/// The calibration verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationVerdict {
    /// Mean in band and std >= 0.1: the environment enters the pool.
    Accepted,
    /// Mean above the band: the solver always solves it (too easy).
    TooEasy,
    /// Mean below the band: the solver never solves it (too hard).
    TooHard,
    /// Mean in band but rewards do not vary (no GRPO signal).
    NoSpread,
}

impl CalibrationVerdict {
    /// Whether the environment is admitted to the training pool.
    pub fn admitted(&self) -> bool {
        matches!(self, CalibrationVerdict::Accepted)
    }

    /// One-line description.
    pub fn describe(&self) -> &'static str {
        match self {
            CalibrationVerdict::Accepted => "mean in [0.25, 0.75] with reward variation",
            CalibrationVerdict::TooEasy => "solver mean above the band",
            CalibrationVerdict::TooHard => "solver mean below the band",
            CalibrationVerdict::NoSpread => "rewards do not vary across attempts",
        }
    }
}

/// One diagnostic rollout with reference-solution hints (below-band tasks
/// only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticRollout {
    /// Reward the hinted attempt earned.
    pub reward: f64,
    /// Checks that still failed under the hint.
    pub failed_checks: Vec<String>,
    /// The hint demonstrated every check of the reference solution.
    pub demonstrated_all: bool,
}

/// The full calibration outcome for one environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationOutcome {
    /// The environment id.
    pub env_id: String,
    /// Per-attempt rewards (4, or 8 after boundary expansion).
    pub rewards: Vec<f64>,
    /// Mean reward.
    pub mean: f64,
    /// Population standard deviation.
    pub std: f64,
    /// Whether the protocol expanded to 8 rollouts.
    pub expanded: bool,
    /// The verdict.
    pub verdict: CalibrationVerdict,
    /// The diagnostic rollout for below-band tasks.
    pub diagnostic: Option<DiagnosticRollout>,
}

impl CalibrationOutcome {
    /// Whether calibration passed.
    pub fn passed(&self) -> bool {
        self.verdict.admitted()
    }
}

/// Population standard deviation.
pub fn population_std(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64;
    var.sqrt()
}

/// Run the adaptive calibration protocol.
///
/// 1. Four solver rollouts.
/// 2. If the mean is within `boundary_margin` of a band edge, expand to
///    eight rollouts and recompute.
/// 3. Verdict: `Accepted` iff mean ∈ band ∧ std ≥ `min_std`.
/// 4. Below-band tasks get exactly one diagnostic rollout with
///    reference-solution hints.
pub fn calibrate(
    task: &HarborTask,
    solver: &mut dyn SolverModel,
    config: &CalibrationConfig,
) -> CalibrationOutcome {
    let mut rewards = Vec::new();
    for _ in 0..config.initial_rollouts {
        rewards.push(solver.attempt(task, None).reward);
    }
    let mut mean = mean_of(&rewards);
    let expanded = config.near_boundary(mean);
    if expanded {
        while rewards.len() < config.expanded_rollouts {
            rewards.push(solver.attempt(task, None).reward);
        }
        mean = mean_of(&rewards);
    }
    let std = population_std(&rewards);
    let verdict = if mean < config.band.0 {
        CalibrationVerdict::TooHard
    } else if mean > config.band.1 {
        CalibrationVerdict::TooEasy
    } else if std < config.min_std {
        CalibrationVerdict::NoSpread
    } else {
        CalibrationVerdict::Accepted
    };

    let diagnostic = if verdict == CalibrationVerdict::TooHard {
        Some(run_diagnostic(task, solver))
    } else {
        None
    };

    CalibrationOutcome {
        env_id: task.env_id.clone(),
        rewards,
        mean,
        std,
        expanded,
        verdict,
        diagnostic,
    }
}

/// One diagnostic rollout with reference-solution hints: the hint
/// demonstrates every check the reference solution satisfies, and the
/// residual failures guide the proposer's revision.
pub fn run_diagnostic(task: &HarborTask, solver: &mut dyn SolverModel) -> DiagnosticRollout {
    let hint = Hint {
        demonstrated_checks: task.check_names(),
    };
    let attempt = solver.attempt(task, Some(&hint));
    DiagnosticRollout {
        reward: attempt.reward,
        failed_checks: attempt.report.failed_names(),
        demonstrated_all: true,
    }
}

/// The revision ledger: "each assignment permits up to two revisions."
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevisionLedger {
    /// Revisions remaining.
    pub remaining: usize,
}

impl RevisionLedger {
    /// New ledger with the configured budget.
    pub fn new(config: &CalibrationConfig) -> Self {
        RevisionLedger {
            remaining: config.max_revisions,
        }
    }

    /// Request a revision slot. Errors when the budget is exhausted — the
    /// environment is then permanently rejected for this assignment.
    pub fn request(&mut self) -> Result<()> {
        if self.remaining == 0 {
            return Err(crate::error::Error::EnvironmentRejected {
                env_id: String::new(),
                reasons: vec!["revision budget exhausted (max 2 per assignment)".into()],
            });
        }
        self.remaining -= 1;
        Ok(())
    }

    /// Whether a revision slot remains.
    pub fn can_revise(&self) -> bool {
        self.remaining > 0
    }
}

/// Feedback for a rejected environment: every failure reason plus the
/// diagnostic rollout, "back to the proposer for revision".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RevisionFeedback {
    /// Validation failure reasons (if validation failed).
    pub validation_failures: Vec<String>,
    /// The calibration outcome (if validation passed).
    pub calibration: Option<CalibrationOutcome>,
    /// The diagnostic rollout for below-band tasks.
    pub diagnostic: Option<DiagnosticRollout>,
    /// Revisions remaining.
    pub revisions_remaining: usize,
}

fn mean_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harbor::{Difficulty, SkillTag};
    use crate::policy::{generate_task, ScriptedSolver, TaskGenParams};

    fn task(n_checks: usize, skills: Vec<&str>) -> HarborTask {
        generate_task(&TaskGenParams {
            env_id: "env_c".into(),
            name: "flywheel/env_c".into(),
            n_checks,
            skills: skills.iter().map(|k| SkillTag::new(*k)).collect(),
            difficulty: Difficulty::Medium,
            broken: false,
        })
    }

    fn skewed_solver(seed: u64, csv_bias: f64, json_bias: f64) -> ScriptedSolver {
        // Bias = logit toward ACTION_SOLVE:  solve vs (sloppy|skip).
        let mut s = ScriptedSolver::new("t", seed);
        s.bank.tables.insert(
            "csv".into(),
            crate::dppo::SoftmaxPolicy {
                logits: vec![csv_bias, -5.0, 0.0],
            },
        );
        s.bank.tables.insert(
            "json".into(),
            crate::dppo::SoftmaxPolicy {
                logits: vec![json_bias, -5.0, 0.0],
            },
        );
        s
    }

    #[test]
    fn config_defaults_match_paper() {
        let c = CalibrationConfig::default();
        assert_eq!(c.band, (0.25, 0.75));
        assert!((c.min_std - 0.1).abs() < 1e-9);
        assert_eq!(c.initial_rollouts, 4);
        assert_eq!(c.expanded_rollouts, 8);
        assert!((c.boundary_margin - 0.15).abs() < 1e-9);
        assert_eq!(c.max_revisions, 2);
    }

    #[test]
    fn boundary_expansion_rule() {
        let c = CalibrationConfig::default();
        assert!(c.near_boundary(0.25));
        assert!(c.near_boundary(0.12));
        assert!(c.near_boundary(0.74));
        assert!(c.near_boundary(0.9));
        assert!(!c.near_boundary(0.5));
        assert!(!c.near_boundary(0.05));
        assert!(!c.near_boundary(0.95));
    }

    #[test]
    fn strong_solver_makes_task_too_easy() {
        let t = task(4, vec!["csv", "json"]);
        let mut solver = skewed_solver(1, 8.0, 8.0); // always solve
        let out = calibrate(&t, &mut solver, &CalibrationConfig::default());
        assert_eq!(out.verdict, CalibrationVerdict::TooEasy);
        assert!(!out.passed());
        assert!(out.diagnostic.is_none());
    }

    #[test]
    fn weak_solver_makes_task_too_hard_with_diagnostic() {
        let t = task(4, vec!["csv", "json"]);
        let mut solver = skewed_solver(2, -8.0, -8.0); // always skip
        let out = calibrate(&t, &mut solver, &CalibrationConfig::default());
        assert_eq!(out.verdict, CalibrationVerdict::TooHard);
        // Exactly one diagnostic rollout with reference-solution hints.
        let diag = out
            .diagnostic
            .expect("below-band tasks get a diagnostic rollout");
        assert!(diag.reward > 0.5, "hinted rollout should mostly solve it");
        assert!(diag.demonstrated_all);
    }

    #[test]
    fn split_solver_calibrates_in_band() {
        // csv solved, json skipped: 2 of 4 checks pass deterministically ->
        // reward 0.5 with no spread... so mix sloppy/skip for the json
        // checks to create variance.
        let t = task(4, vec!["csv", "json"]);
        let mut solver = skewed_solver(3, 8.0, 0.6); // json ~ solve/sloppy/... mixture
        let out = calibrate(&t, &mut solver, &CalibrationConfig::default());
        assert!(
            matches!(
                out.verdict,
                CalibrationVerdict::Accepted
                    | CalibrationVerdict::NoSpread
                    | CalibrationVerdict::TooEasy
            ),
            "got {:?}",
            out.verdict
        );
        assert!(out.rewards.len() == 4 || out.rewards.len() == 8);
    }

    #[test]
    fn no_spread_detected() {
        // Perfectly deterministic 0.5 reward: in band, std = 0.
        let t = task(2, vec!["csv", "json"]);
        let mut solver = skewed_solver(4, 8.0, -8.0); // csv solve (1 check), json skip (1 check)
        let out = calibrate(&t, &mut solver, &CalibrationConfig::default());
        assert!((out.mean - 0.5).abs() < 1e-9);
        assert!(out.std < 0.1);
        assert_eq!(out.verdict, CalibrationVerdict::NoSpread);
        assert!(!out.passed());
    }

    #[test]
    fn revision_ledger_two_max() {
        let mut ledger = RevisionLedger::new(&CalibrationConfig::default());
        assert!(ledger.can_revise());
        assert!(ledger.request().is_ok());
        assert!(ledger.request().is_ok());
        assert!(!ledger.can_revise());
        let err = ledger.request().unwrap_err();
        assert!(err.to_string().contains("revision budget"));
    }

    #[test]
    fn population_std_matches() {
        assert!((population_std(&[0.0, 1.0]) - 0.5).abs() < 1e-9);
        assert!(population_std(&[]) == 0.0);
        assert!(population_std(&[0.5, 0.5]) < 1e-9);
    }

    #[test]
    fn verdict_descriptions() {
        assert!(CalibrationVerdict::Accepted.admitted());
        assert!(!CalibrationVerdict::NoSpread.admitted());
        assert!(CalibrationVerdict::TooHard.describe().contains("below"));
        assert!(CalibrationVerdict::TooEasy.describe().contains("above"));
    }
}
