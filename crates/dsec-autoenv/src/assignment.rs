//! Assignments: how the flywheel tells the proposer what to build next
//! (Section 3.1).
//!
//! "Each episode starts from an assignment that names a parent environment
//! from the training pool and gives the solver's pass rate and most common
//! failure on it. The proposer reads the solver's trajectories and verifier
//! outputs on the parent to find what went wrong or what the solver already
//! does reliably. It then builds a simpler variant when the solver keeps
//! failing, or a harder one with a longer horizon or an additional skill
//! when the solver succeeds."

use crate::harbor::SkillTag;
use crate::trajectory::RolloutRecord;
use serde::{Deserialize, Serialize};

/// The curriculum direction derived from the solver's pass rate on the
/// parent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "direction", rename_all = "snake_case")]
pub enum Direction {
    /// The solver keeps failing on the parent (pass rate below the band):
    /// build a simpler variant.
    Simplify {
        /// How many fewer checks the variant should carry.
        reduce_checks_by: u32,
    },
    /// The solver succeeds on the parent (pass rate above the band): build
    /// a harder variant with a longer horizon and/or an additional skill.
    Harden {
        /// How many more checks the variant should carry (longer horizon).
        add_checks: u32,
        /// A skill the parent does not exercise.
        additional_skill: Option<SkillTag>,
    },
    /// Pool growth: build a fresh environment grounded in the solver's
    /// failure modes (used when the pool is under capacity).
    Fresh,
}

impl Direction {
    /// Short tag used as the proposer's policy context key.
    pub fn tag(&self) -> &'static str {
        match self {
            Direction::Simplify { .. } => "simplify",
            Direction::Harden { .. } => "harden",
            Direction::Fresh => "fresh",
        }
    }
}

/// The most common failure across the solver's failed attempts on the
/// parent, extracted from verifier stdout / failed-check lists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailureMode {
    /// The failing check (or verifier signature) that occurred most often.
    pub check: String,
    /// How many failed attempts hit it.
    pub occurrences: usize,
    /// Share of failed attempts that hit it.
    pub share: f64,
}

/// Extract the most common failure mode from the parent's rollout records:
/// the check that failed on the largest number of attempts, aggregated from
/// the per-check failure counts each record carries (the recorder derives
/// them from verifier stdout — the proposer reads the same signal).
pub fn most_common_failure(records: &[RolloutRecord]) -> Option<FailureMode> {
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut failed_attempts = 0usize;
    for r in records {
        for reward in &r.rewards {
            if *reward < 1.0 {
                failed_attempts += 1;
            }
        }
        for (check, n) in &r.check_failures {
            *counts.entry(check.clone()).or_insert(0) += n;
        }
    }
    if failed_attempts == 0 || counts.is_empty() {
        return None;
    }
    let (check, occurrences) = counts.into_iter().max_by_key(|(_, c)| *c)?;
    Some(FailureMode {
        check,
        occurrences,
        share: (occurrences as f64 / failed_attempts as f64).min(1.0),
    })
}

/// A reference to the parent environment, as seen by the proposer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParentRef {
    /// Parent environment id.
    pub env_id: String,
    /// Number of checks in the parent (drives variant sizing).
    pub checks: usize,
    /// Skills the parent exercises (drives additional-skill selection).
    pub skills: Vec<SkillTag>,
}

/// One generation assignment handed to the proposer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assignment {
    /// Assignment id (`asg_0007`).
    pub assignment_id: String,
    /// The environment slot the new task will occupy (env id).
    pub target_env_id: String,
    /// The parent environment, when the assignment is grounded in one.
    pub parent: Option<ParentRef>,
    /// The solver's pass rate on the parent.
    pub parent_pass_rate: Option<f64>,
    /// The parent's most common failure.
    pub most_common_failure: Option<FailureMode>,
    /// The curriculum direction.
    pub direction: Direction,
}

impl Assignment {
    /// Choose the direction from a pass rate and the calibration band
    /// (Section 3.1's rule).
    pub fn direction_for(pass_rate: f64, band: (f64, f64)) -> Direction {
        if pass_rate < band.0 {
            Direction::Simplify {
                reduce_checks_by: 2,
            }
        } else if pass_rate > band.1 {
            Direction::Harden {
                add_checks: 2,
                additional_skill: None,
            }
        } else {
            Direction::Fresh
        }
    }

    /// Render the assignment text the proposer reads (`assignment/`).
    pub fn render(&self) -> String {
        let mut s = String::from("# Generation assignment\n\n");
        match &self.parent {
            Some(parent) => {
                s.push_str(&format!("Parent environment: {}\n", parent.env_id));
                s.push_str(&format!("Parent checks: {}\n", parent.checks));
                s.push_str(&format!(
                    "Parent skills: {}\n",
                    parent
                        .skills
                        .iter()
                        .map(|k| k.name().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                if let Some(pr) = self.parent_pass_rate {
                    s.push_str(&format!("Solver pass rate on parent: {pr:.2}\n"));
                }
                if let Some(f) = &self.most_common_failure {
                    s.push_str(&format!(
                        "Most common failure: check '{}' on {:.0}% of failed attempts\n",
                        f.check,
                        f.share * 100.0
                    ));
                }
            }
            None => s.push_str("Parent environment: none (fresh environment)\n"),
        }
        match &self.direction {
            Direction::Simplify { reduce_checks_by } => s.push_str(&format!(
                "\nThe solver keeps failing on the parent. Build a SIMPLER variant (about {reduce_checks_by} fewer checks), keeping the same skill focus.\n"
            )),
            Direction::Harden { add_checks, additional_skill } => {
                s.push_str(&format!(
                    "\nThe solver succeeds on the parent. Build a HARDER variant with a longer horizon (about {add_checks} more checks)"
                ));
                if let Some(sk) = additional_skill {
                    s.push_str(&format!(" and an additional skill ({})", sk.name()));
                }
                s.push_str(".\n");
            }
            Direction::Fresh => s.push_str("\nBuild a fresh environment grounded in the solver's failure modes from the archive.\n"),
        }
        s.push_str("\nRead memory/past_trajectories and memory/past_environments first, then follow the create-task and auto_env_scaling skills. Validate (oracle-passes AND no-op-fails) before submitting.\n");
        s
    }
}

/// Build one assignment from a parent pool entry and its rollouts.
pub fn build_assignment(
    assignment_id: impl Into<String>,
    target_env_id: impl Into<String>,
    parent: ParentRef,
    rollouts: &[RolloutRecord],
    band: (f64, f64),
) -> Assignment {
    let parent_records: Vec<RolloutRecord> = rollouts
        .iter()
        .filter(|r| r.env_id == parent.env_id)
        .cloned()
        .collect();
    let pass_rate = pass_rate(&parent_records);
    let failure = most_common_failure(&parent_records);
    let mut direction = Assignment::direction_for(pass_rate, band);
    if let Direction::Harden {
        additional_skill, ..
    } = &mut direction
    {
        *additional_skill = pick_additional_skill(&parent.skills);
    }
    Assignment {
        assignment_id: assignment_id.into(),
        target_env_id: target_env_id.into(),
        parent: Some(parent),
        parent_pass_rate: Some(pass_rate),
        most_common_failure: failure,
        direction,
    }
}

/// Mean reward across the parent's attempts (the paper's "pass rate").
pub fn pass_rate(records: &[RolloutRecord]) -> f64 {
    let rewards: Vec<f64> = records.iter().flat_map(|r| r.rewards.clone()).collect();
    if rewards.is_empty() {
        0.0
    } else {
        rewards.iter().sum::<f64>() / rewards.len() as f64
    }
}

/// Pick a skill the parent does not exercise (for hardening).
fn pick_additional_skill(parent_skills: &[SkillTag]) -> Option<SkillTag> {
    let candidates = [
        "json", "regex", "sql", "docker", "git", "crypto", "csv", "network",
    ];
    for c in candidates {
        let tag = SkillTag::new(c);
        if !parent_skills.contains(&tag) {
            return Some(tag);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::Provenance;

    fn record(env: &str, rewards: Vec<f64>, failed_checks: &str) -> RolloutRecord {
        let mut check_failures = std::collections::BTreeMap::new();
        let failed_attempts = rewards.iter().filter(|r| **r < 1.0).count();
        for check in failed_checks
            .split(',')
            .map(|c| c.trim())
            .filter(|c| !c.is_empty() && *c != "none")
        {
            *check_failures.entry(check.to_string()).or_insert(0) += failed_attempts;
        }
        RolloutRecord {
            env_id: env.into(),
            solver: "solver".into(),
            round: 1,
            provenance: Provenance::SelfRound(1),
            rewards: rewards.clone(),
            reward_std: 0.3,
            check_failures,
            path: format!("memory/past_trajectories/failed/{env}.json"),
        }
    }

    #[test]
    fn direction_follows_pass_rate() {
        assert!(matches!(
            Assignment::direction_for(0.10, (0.25, 0.75)),
            Direction::Simplify { .. }
        ));
        assert!(matches!(
            Assignment::direction_for(0.90, (0.25, 0.75)),
            Direction::Harden { .. }
        ));
        assert!(matches!(
            Assignment::direction_for(0.50, (0.25, 0.75)),
            Direction::Fresh
        ));
    }

    #[test]
    fn most_common_failure_aggregates() {
        let records = vec![
            record("env_1", vec![0.0, 0.25], "test_tolerance, test_parse"),
            record("env_1", vec![0.5, 0.0], "test_tolerance"),
        ];
        let f = most_common_failure(&records).unwrap();
        assert_eq!(f.check, "test_tolerance");
        // 2 failed attempts on each record attribute their failures.
        assert_eq!(f.occurrences, 4);
        assert!(f.share > 0.0 && f.share <= 1.0);
    }

    #[test]
    fn no_failure_when_all_pass() {
        let records = vec![record("env_1", vec![1.0, 1.0], "none")];
        assert!(most_common_failure(&records).is_none());
    }

    #[test]
    fn build_assignment_harden_picks_new_skill() {
        let parent = ParentRef {
            env_id: "env_003".into(),
            checks: 4,
            skills: vec![SkillTag::new("csv"), SkillTag::new("python")],
        };
        let rollouts = vec![record("env_003", vec![1.0, 1.0, 0.75, 1.0], "test_edge")];
        let a = build_assignment("asg_1", "env_010", parent, &rollouts, (0.25, 0.75));
        match &a.direction {
            Direction::Harden {
                additional_skill,
                add_checks,
            } => {
                assert_eq!(*add_checks, 2);
                assert!(additional_skill.is_some());
                assert_ne!(additional_skill.as_ref().unwrap().name(), "csv");
            }
            other => panic!("expected harden, got {other:?}"),
        }
        let text = a.render();
        assert!(text.contains("env_003"));
        assert!(text.contains("HARDER"));
        assert!(text.contains("longer horizon"));
    }

    #[test]
    fn build_assignment_simplify_when_failing() {
        let parent = ParentRef {
            env_id: "env_005".into(),
            checks: 6,
            skills: vec![SkillTag::new("json")],
        };
        let rollouts = vec![record("env_005", vec![0.0, 0.0, 0.1, 0.0], "test_parse")];
        let a = build_assignment("asg_2", "env_011", parent, &rollouts, (0.25, 0.75));
        assert!(matches!(a.direction, Direction::Simplify { .. }));
        assert!(a.render().contains("SIMPLER"));
        assert!(a.render().contains("test_parse"));
    }

    #[test]
    fn pass_rate_over_all_attempts() {
        let records = vec![record("e", vec![1.0, 0.5], "x")];
        assert!((pass_rate(&records) - 0.75).abs() < 1e-9);
        assert_eq!(pass_rate(&[]), 0.0);
    }
}
