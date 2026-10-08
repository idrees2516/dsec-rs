//! The `harbor check` task-review rubric (Section 3.2, Appendix G.4).
//!
//! "For these cases we run Harbor's task review (`harbor check`), in which
//! the proposer's model reviews the task against the official Terminal-Bench
//! rubric." And from G.4: "We apply every criterion except `binary_reward`,
//! which requires rewards of exactly 0 or 1, because our tests give partial
//! credit. A task that fails a criterion returns to the proposer for
//! revision."
//!
//! The rubric below encodes the task-implementation criteria the paper
//! engages: instruction quality, no test leakage, solvability,
//! determinism (offline at solve time), no shortcuts, minimal environment,
//! discoverable metadata, and populated docs. Each criterion is evaluated
//! deterministically over the task structure (the frontier models grade
//! free-text judgment; we grade the machine-checkable subset).

use crate::harbor::{HarborTask, NetworkMode};
use serde::{Deserialize, Serialize};

/// Rubric configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RubricConfig {
    /// Whether to exclude the `binary_reward` criterion. The paper always
    /// excludes it: "our tests give partial credit."
    pub exclude_binary_reward: bool,
}

impl Default for RubricConfig {
    fn default() -> Self {
        RubricConfig {
            exclude_binary_reward: true,
        }
    }
}

/// Outcome of one rubric criterion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RubricStatus {
    /// The criterion is satisfied.
    Pass,
    /// The criterion is violated; the task returns to the proposer.
    Fail,
    /// The criterion is excluded by configuration (`binary_reward`).
    Excluded,
}

/// One graded criterion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CriterionResult {
    /// Criterion identifier.
    pub name: String,
    /// The grade.
    pub status: RubricStatus,
    /// Why (failure detail or exclusion note).
    pub detail: String,
}

impl CriterionResult {
    /// Whether this criterion blocks admission.
    pub fn blocks(&self) -> bool {
        self.status == RubricStatus::Fail
    }
}

/// The full rubric report for one task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RubricReport {
    /// All graded criteria, in rubric order.
    pub criteria: Vec<CriterionResult>,
}

impl RubricReport {
    /// Whether the task passes the review (no failing criterion).
    pub fn passed(&self) -> bool {
        !self.criteria.iter().any(|c| c.blocks())
    }

    /// The failing criteria.
    pub fn failures(&self) -> Vec<&CriterionResult> {
        self.criteria.iter().filter(|c| c.blocks()).collect()
    }

    /// Names of failing criteria.
    pub fn failure_names(&self) -> Vec<String> {
        self.failures().iter().map(|c| c.name.clone()).collect()
    }
}

/// Whole-token containment (see `harbor::contains_word`).
fn contains_word(haystack: &str, needle: &str) -> bool {
    haystack
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .any(|w| w == needle)
}

/// Inputs the rubric grades that the task structure alone cannot determine:
/// the Oracle reward and the do-nothing reward, computed by the host.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReviewInputs {
    /// Reward the reference solution earned (should be 1.0).
    pub reference_reward: f64,
    /// Reward the empty solution earned (should be < 0.5).
    pub noop_reward: f64,
}

/// Review a task against the rubric.
///
/// Every criterion except `binary_reward` is applied; `binary_reward` is
/// `Excluded` when configured (the paper's setting, because managed
/// verifiers write fractional rewards).
pub fn review_task(
    task: &HarborTask,
    inputs: &ReviewInputs,
    config: &RubricConfig,
) -> RubricReport {
    let mut criteria = Vec::new();

    // 1. instruction_clarity: concrete goal, expected outputs, constraints.
    let body = task.instruction.body.trim();
    let mentions_output = task.tests.checks.iter().any(|c| {
        let path = match c {
            crate::world::TestCheck::FileExists { path, .. }
            | crate::world::TestCheck::FileContains { path, .. }
            | crate::world::TestCheck::FileJsonKey { path, .. }
            | crate::world::TestCheck::FileLineCount { path, .. } => Some(path.clone()),
            _ => None,
        };
        path.map(|p| body.contains(&p)).unwrap_or(false)
    });
    criteria.push(CriterionResult {
        name: "instruction_clarity".into(),
        status: if body.len() >= 60 && mentions_output {
            RubricStatus::Pass
        } else {
            RubricStatus::Fail
        },
        detail: if body.len() < 60 {
            "instruction too thin: state the goal, expected outputs, and constraints".into()
        } else if !mentions_output {
            "instruction does not name the expected output paths".into()
        } else {
            "states goal, outputs, constraints".into()
        },
    });

    // 2. instruction_no_test_leak: describe what done looks like, not how
    //    it is checked.
    let leaked: Vec<String> = task
        .check_names()
        .into_iter()
        .filter(|name| contains_word(body, name))
        .collect();
    criteria.push(CriterionResult {
        name: "instruction_no_test_leak".into(),
        status: if leaked.is_empty() {
            RubricStatus::Pass
        } else {
            RubricStatus::Fail
        },
        detail: if leaked.is_empty() {
            "no test names leak into the instruction".into()
        } else {
            format!("instruction leaks test names: {leaked:?}")
        },
    });

    // 3. solvable: the reference solution must earn full reward.
    criteria.push(CriterionResult {
        name: "solvable".into(),
        status: if (inputs.reference_reward - 1.0).abs() < 1e-9 {
            RubricStatus::Pass
        } else {
            RubricStatus::Fail
        },
        detail: format!("reference solution reward = {}", inputs.reference_reward),
    });

    // 4. deterministic: self-contained, no network at solve time.
    let offline = matches!(task.effective_agent_baseline(), NetworkMode::NoNetwork);
    criteria.push(CriterionResult {
        name: "deterministic".into(),
        status: if offline {
            RubricStatus::Pass
        } else {
            RubricStatus::Fail
        },
        detail: if offline {
            "network_mode = no-network at solve time".into()
        } else {
            "environment baseline is not offline; spread could come from network flakiness".into()
        },
    });

    // 5. no_shortcut: the no-op must fail (not trivially gameable).
    criteria.push(CriterionResult {
        name: "no_shortcut".into(),
        status: if inputs.noop_reward < 0.5 {
            RubricStatus::Pass
        } else {
            RubricStatus::Fail
        },
        detail: format!("do-nothing reward = {}", inputs.noop_reward),
    });

    // 6. verifier_checks_artifacts: checks must inspect real artifacts.
    let artifact_checks = task
        .tests
        .checks
        .iter()
        .any(|c| !matches!(c, crate::world::TestCheck::CommandSucceeds { .. }));
    criteria.push(CriterionResult {
        name: "verifier_checks_artifacts".into(),
        status: if artifact_checks {
            RubricStatus::Pass
        } else {
            RubricStatus::Fail
        },
        detail: if artifact_checks {
            "verifier inspects produced files".into()
        } else {
            "verifier only runs commands; check the actual artifact".into()
        },
    });

    // 7. environment_minimal: the Dockerfile installs what the task needs,
    //    not the solution.
    let preplanted: Vec<&String> = task
        .solution
        .ops
        .iter()
        .filter_map(|op| match op {
            crate::world::ShellOp::WriteFile { path, .. } => Some(path),
            _ => None,
        })
        .filter(|p| task.environment.files.contains_key(p.as_str()))
        .collect();
    criteria.push(CriterionResult {
        name: "environment_minimal".into(),
        status: if preplanted.is_empty() {
            RubricStatus::Pass
        } else {
            RubricStatus::Fail
        },
        detail: if preplanted.is_empty() {
            "environment plants only task state".into()
        } else {
            format!("environment pre-plants solution outputs: {preplanted:?}")
        },
    });

    // 8. keywords_populated: 3-8 lowercase tokens (registry discovery).
    let kw_ok = (3..=8).contains(&task.meta.keywords.len())
        && task.meta.keywords.iter().all(|k| {
            k.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        });
    criteria.push(CriterionResult {
        name: "keywords_populated".into(),
        status: if kw_ok {
            RubricStatus::Pass
        } else {
            RubricStatus::Fail
        },
        detail: format!("keywords = {:?}", task.meta.keywords),
    });

    // 9. readme_populated: docs, not marketing.
    let readme_ok = task.readme.trim().len() >= 80 && !task.readme.contains("# TODO");
    criteria.push(CriterionResult {
        name: "readme_populated".into(),
        status: if readme_ok {
            RubricStatus::Pass
        } else {
            RubricStatus::Fail
        },
        detail: if readme_ok {
            "README describes the task".into()
        } else {
            "README is a stub".into()
        },
    });

    // 10. binary_reward: excluded by the paper's setting.
    criteria.push(CriterionResult {
        name: "binary_reward".into(),
        status: if config.exclude_binary_reward {
            RubricStatus::Excluded
        } else {
            // Applied: rewards of exactly 0 or 1 only.
            let fractional = task.total_checks() > 1;
            if fractional {
                RubricStatus::Fail
            } else {
                RubricStatus::Pass
            }
        },
        detail: if config.exclude_binary_reward {
            "excluded: managed verifiers give partial credit (paper setting)".into()
        } else {
            "applied: rewards must be exactly 0 or 1".into()
        },
    });

    RubricReport { criteria }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harbor::{
        AgentConfig, Difficulty, Instruction, RewardStrategy, SkillTag, TaskEnvironment, TaskMeta,
        TaskMetadata, TaskToml, VerifierConfig,
    };
    use crate::world::{RewardFile, ShellOp, TestCheck, TestSuite};

    fn task() -> HarborTask {
        let checks = vec![
            TestCheck::FileExists {
                name: "test_report_exists".into(),
                path: "/app/out/report.csv".into(),
            },
            TestCheck::FileContains {
                name: "test_report_rows".into(),
                path: "/app/out/report.csv".into(),
                needle: "id,total".into(),
            },
            TestCheck::FileJsonKey {
                name: "test_summary_json".into(),
                path: "/app/out/summary.json".into(),
                key: "total".into(),
                expected: "42".into(),
            },
        ];
        HarborTask {
            env_id: "env_001".into(),
            meta: TaskMeta {
                name: "acme/task".into(),
                version: "1.0.0".into(),
                description: "demo".into(),
                keywords: vec!["csv".into(), "python".into(), "pytest".into()],
            },
            instruction: Instruction::new(
                "# Reconcile\n\nProduce `/app/out/report.csv` (header id,total) and `/app/out/summary.json` with key total = 42. Python 3 is available; do not modify the inputs.\n",
            ),
            environment: TaskEnvironment {
                network_mode: NetworkMode::NoNetwork,
                ..Default::default()
            },
            solution: SolutionScript {
                ops: vec![
                    ShellOp::WriteFile { path: "/app/out/report.csv".into(), content: "id,total\n".into() },
                    ShellOp::WriteFile { path: "/app/out/summary.json".into(), content: "{\"total\": 42}".into() },
                ],
                executable: true,
            },
            tests: TestSuite {
                test_sh: "#!/bin/bash\nuvx --with pytest==8.4.1 pytest /tests/test_outputs.py\n".into(),
                checks,
                reward_file: RewardFile::Txt,
            },
            readme: "# Task\n\nA complete description of the task, its environment, verifier, directory layout, and the harbor run commands used to verify it end to end.\n".into(),
            steps: Vec::new(),
            toml: TaskToml {
                schema_version: "1.0".into(),
                task: TaskMeta {
                    name: "acme/task".into(),
                    version: "1.0.0".into(),
                    description: "demo".into(),
                    keywords: vec!["csv".into(), "python".into(), "pytest".into()],
                },
                metadata: TaskMetadata { difficulty: Difficulty::Easy, category: "programming".into(), tags: vec![] },
                agent: AgentConfig::default(),
                verifier: VerifierConfig::default(),
                reward_strategy: RewardStrategy::Mean,
            },
            skills: vec![SkillTag::new("csv")],
        }
    }
    use crate::world::SolutionScript;

    #[test]
    fn clean_task_passes_rubric() {
        let inputs = ReviewInputs {
            reference_reward: 1.0,
            noop_reward: 0.0,
        };
        let report = review_task(&task(), &inputs, &RubricConfig::default());
        assert!(report.passed(), "{:?}", report.failure_names());
        // binary_reward excluded by default.
        assert!(report
            .criteria
            .iter()
            .any(|c| c.status == RubricStatus::Excluded));
    }

    #[test]
    fn broken_reference_fails_solvable() {
        let inputs = ReviewInputs {
            reference_reward: 0.25,
            noop_reward: 0.0,
        };
        let report = review_task(&task(), &inputs, &RubricConfig::default());
        assert!(report.failure_names().contains(&"solvable".to_string()));
    }

    #[test]
    fn gameable_task_fails_no_shortcut() {
        let inputs = ReviewInputs {
            reference_reward: 1.0,
            noop_reward: 1.0,
        };
        let report = review_task(&task(), &inputs, &RubricConfig::default());
        assert!(report.failure_names().contains(&"no_shortcut".to_string()));
    }

    #[test]
    fn online_task_fails_deterministic() {
        let mut t = task();
        t.environment.network_mode = NetworkMode::Public;
        let inputs = ReviewInputs {
            reference_reward: 1.0,
            noop_reward: 0.0,
        };
        let report = review_task(&t, &inputs, &RubricConfig::default());
        assert!(report
            .failure_names()
            .contains(&"deterministic".to_string()));
    }

    #[test]
    fn test_leak_fails() {
        let mut t = task();
        t.instruction
            .body
            .push_str("\nMake sure the check named `test_summary_json` passes.\n");
        let inputs = ReviewInputs {
            reference_reward: 1.0,
            noop_reward: 0.0,
        };
        let report = review_task(&t, &inputs, &RubricConfig::default());
        assert!(report
            .failure_names()
            .contains(&"instruction_no_test_leak".to_string()));
    }

    #[test]
    fn binary_reward_can_be_applied() {
        let inputs = ReviewInputs {
            reference_reward: 1.0,
            noop_reward: 0.0,
        };
        let report = review_task(
            &task(),
            &inputs,
            &RubricConfig {
                exclude_binary_reward: false,
            },
        );
        // A 3-check suite is fractional -> binary_reward fails when applied.
        assert!(report
            .failure_names()
            .contains(&"binary_reward".to_string()));
    }
}
