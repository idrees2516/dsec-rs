//! Host-side admission: validation (Section 3.2).
//!
//! "Before a new environment enters the training pool, the host checks that
//! it works and that it suits the current solver."
//!
//! Validation runs *outside* the proposer's sandbox (Section 3.1: "the
//! proposer can edit its memory and tools but not the instructions or the
//! admission checks, which run outside the sandbox"), and per Appendix G.4:
//!
//! * "Admission requires reference reward r = 1 and do-nothing reward r <
//!   0.5, in addition to valid files and a successful container build."
//! * the 13-gram contamination check (Appendix F.2);
//! * the `harbor check` task review against the Terminal-Bench rubric
//!   ([`crate::rubric`]);
//! * "A task that fails a criterion returns to the proposer for revision."
//!
//! The proposer runs the same two solution checks itself *before*
//! submitting, using the editable `tools/validate.py` — the no-op swap
//! dance of the auto_env_scaling skill:
//!
//! ```text
//! cp <task>/solution/solve.sh /tmp/solve.bak
//! printf '#!/bin/sh\nexit 0\n' > <task>/solution/solve.sh
//! python3 /opt/tools/validate.py <task> oracle   # expect reward < 1
//! cp /tmp/solve.bak <task>/solution/solve.sh
//! ```

use crate::decontaminate::ContaminationIndex;
use crate::error::Result;
use crate::harbor::HarborTask;
use crate::rubric::{review_task, ReviewInputs, RubricConfig, RubricReport};
use crate::world::{self, ImageRegistry, SolutionScript};
use serde::{Deserialize, Serialize};

/// Validation thresholds (Appendix G.4). The empty solution must "receive
/// less than half"; the reference must earn exactly `1`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Maximum reward the do-nothing solution may earn (paper: 0.5,
    /// exclusive).
    pub max_noop_reward: f64,
    /// Rubric configuration (excludes `binary_reward` by default).
    pub rubric: RubricConfig,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        ValidationConfig {
            max_noop_reward: 0.5,
            rubric: RubricConfig::default(),
        }
    }
}

/// The full outcome of host validation for one submitted environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationOutcome {
    /// The environment id.
    pub env_id: String,
    /// Structural lint passed (files, layout, keywords, executable solution).
    pub layout_ok: bool,
    /// The container image built successfully.
    pub build_ok: bool,
    /// Reward earned by the reference solution (must be 1.0).
    pub reference_reward: f64,
    /// Reward earned by the empty solution (must be < 0.5).
    pub noop_reward: f64,
    /// Contamination matches against held-out benchmarks (must be empty).
    pub contamination: Vec<crate::decontaminate::ContaminationMatch>,
    /// The `harbor check` review.
    pub review: RubricReport,
    /// Whether every validation check passed.
    pub passed: bool,
    /// Failure reasons, one per failed check (fed back to the proposer for
    /// revision).
    pub failures: Vec<String>,
}

impl ValidationOutcome {
    /// All failure reasons in admission order.
    pub fn collect_failures(&mut self, lint: &[String]) {
        if !self.layout_ok {
            self.failures.push("invalid task files/layout".into());
        }
        for l in lint {
            self.failures.push(l.clone());
        }
        if !self.build_ok {
            self.failures.push("container build failed".into());
        }
        if (self.reference_reward - 1.0).abs() > 1e-9 {
            self.failures.push(format!(
                "reference solution earned {} (must be 1.0)",
                self.reference_reward
            ));
        }
        if self.noop_reward >= 0.5 {
            self.failures.push(format!(
                "do-nothing solution earned {} (must be < 0.5)",
                self.noop_reward
            ));
        }
        if !self.contamination.is_empty() {
            self.failures.push(format!(
                "instruction shares a 13-gram with held-out task {:?}",
                self.contamination[0].benchmark_task
            ));
        }
        for f in self.review.failures() {
            self.failures
                .push(format!("harbor check: {} — {}", f.name, f.detail));
        }
    }
}

/// Run the full host validation over a submitted task.
///
/// Order (mirroring Appendix G.4): structural lint → build → reference
/// solution → no-op solution → decontamination → rubric review. Every
/// failure reason is collected so the proposer can revise; "each assignment
/// permits up to two revisions" (handled by [`crate::calibrate`]).
pub fn validate(
    task: &HarborTask,
    registry: &ImageRegistry,
    heldout: &ContaminationIndex,
    config: &ValidationConfig,
) -> ValidationOutcome {
    // 1. Structural lint (the Appendix E.1 pitfalls).
    let lint = task.lint();
    let layout_ok = lint.is_empty();

    // 2. Build.
    let build_ok = registry.build_check(task).is_empty();

    // 3+4. The two solution arms. If the build fails we still record the
    // arms as 0 (the oracle cannot run).
    let (reference_reward, noop_reward) = if build_ok {
        match (
            world::oracle_run(task, registry),
            world::noop_run(task, registry),
        ) {
            (Ok(r), Ok(n)) => (r.reward, n.reward),
            (Err(_), Ok(n)) => (0.0, n.reward),
            _ => (0.0, 0.0),
        }
    } else {
        (0.0, 0.0)
    };

    // 5. Decontamination.
    let contamination = heldout.check(&task.instruction.body);

    // 6. Rubric review.
    let inputs = ReviewInputs {
        reference_reward,
        noop_reward,
    };
    let review = review_task(task, &inputs, &config.rubric);

    let mut outcome = ValidationOutcome {
        env_id: task.env_id.clone(),
        layout_ok,
        build_ok,
        reference_reward,
        noop_reward,
        contamination,
        review,
        passed: false,
        failures: Vec::new(),
    };
    outcome.collect_failures(&lint);
    outcome.passed = outcome.failures.is_empty();
    outcome
}

/// The proposer-side self-check before submission (the skill's validation
/// block): oracle must pass and the no-op must fail. The proposer runs this
/// with its *editable* copy of the tools; the host re-runs everything
/// outside the sandbox regardless.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposerSelfCheck {
    /// Whether the reference solution earned full reward.
    pub oracle_passed: bool,
    /// Whether the no-op solution failed (reward < 1 with non-zero exit).
    pub noop_failed: bool,
    /// Rewards observed during the swap dance.
    pub oracle_reward: f64,
    /// No-op reward observed during the swap dance.
    pub noop_reward: f64,
}

impl ProposerSelfCheck {
    /// Both submission conditions: "Submit only when oracle-passes AND
    /// no-op-fails."
    pub fn ready_to_submit(&self) -> bool {
        self.oracle_passed && self.noop_failed
    }
}

/// Run the proposer's pre-submission validation, including the exact no-op
/// swap dance: back up `solve.sh`, overwrite it with `#!/bin/sh\nexit 0`,
/// re-run the oracle, and restore the real solution.
pub fn proposer_self_check(
    task: &HarborTask,
    registry: &ImageRegistry,
) -> Result<ProposerSelfCheck> {
    // 1. Oracle MUST pass (env is genuinely solvable) — reward = 1.0.
    let oracle = world::oracle_run(task, registry)?;
    let oracle_passed = (oracle.reward - 1.0).abs() < 1e-9 && oracle.exit_code == 0;

    // 2. No-op MUST fail (env is not gameable): the swap dance.
    let backup = task.solution.clone(); // cp <task>/solution/solve.sh /tmp/solve.bak
    let mut noop_task = task.clone();
    noop_task.solution = SolutionScript::noop(); // printf '#!/bin/sh\nexit 0\n' > solve.sh
    let noop = world::oracle_run(&noop_task, registry)?; // validate.py <task> oracle
    let noop_failed = noop.reward < 1.0 || noop.exit_code != 0;
    // cp /tmp/solve.bak <task>/solution/solve.sh — the backup is restored by
    // returning it; `task` itself was never mutated.
    let _ = backup;

    Ok(ProposerSelfCheck {
        oracle_passed,
        noop_failed,
        oracle_reward: oracle.reward,
        noop_reward: noop.reward,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harbor::{
        AgentConfig, Difficulty, Instruction, NetworkMode, RewardStrategy, SkillTag,
        TaskEnvironment, TaskMeta, TaskMetadata, TaskToml, VerifierConfig,
    };
    use crate::world::{RewardFile, ShellOp, TestCheck, TestSuite};

    fn clean_task() -> HarborTask {
        let checks = vec![
            TestCheck::FileExists {
                name: "test_a_exists".into(),
                path: "/app/out/a.txt".into(),
            },
            TestCheck::FileContains {
                name: "test_a_content".into(),
                path: "/app/out/a.txt".into(),
                needle: "42".into(),
            },
            TestCheck::FileJsonKey {
                name: "test_b_json".into(),
                path: "/app/out/b.json".into(),
                key: "total".into(),
                expected: "7".into(),
            },
        ];
        HarborTask {
            env_id: "env_010".into(),
            meta: TaskMeta {
                name: "acme/sums".into(),
                version: "1.0.0".into(),
                description: "sum two exports".into(),
                keywords: vec!["csv".into(), "python".into(), "pytest".into()],
            },
            instruction: Instruction::new(
                "# Sums\n\nRead the inputs under /app/in and write `/app/out/a.txt` containing 42 and `/app/out/b.json` with total = 7.\n",
            ),
            environment: TaskEnvironment {
                network_mode: NetworkMode::NoNetwork,
                files: [("/app/in/rows.csv".to_string(), "1,2\n".to_string())].into_iter().collect(),
                ..Default::default()
            },
            solution: SolutionScript {
                ops: vec![
                    ShellOp::WriteFile { path: "/app/out/a.txt".into(), content: "42".into() },
                    ShellOp::WriteFile { path: "/app/out/b.json".into(), content: "{\"total\": 7}".into() },
                ],
                executable: true,
            },
            tests: TestSuite {
                test_sh: "#!/bin/bash\nuvx --with pytest==8.4.1 pytest /tests/test_outputs.py\n".into(),
                checks,
                reward_file: RewardFile::Txt,
            },
            readme: "# Sums\n\nEnvironment, verifier, directory layout, and the harbor run commands (oracle plus a real agent) for this task.\n".into(),
            steps: Vec::new(),
            toml: TaskToml {
                schema_version: "1.0".into(),
                task: TaskMeta {
                    name: "acme/sums".into(),
                    version: "1.0.0".into(),
                    description: "sum two exports".into(),
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

    #[test]
    fn clean_task_passes_admission() {
        let heldout = ContaminationIndex::build(&[]);
        let out = validate(
            &clean_task(),
            &ImageRegistry::default_world(),
            &heldout,
            &ValidationConfig::default(),
        );
        assert!(out.passed, "{:?}", out.failures);
        assert_eq!(out.reference_reward, 1.0);
        assert_eq!(out.noop_reward, 0.0);
    }

    #[test]
    fn broken_reference_is_rejected() {
        let mut t = clean_task();
        t.solution.ops.pop(); // reference stops writing b.json
        let heldout = ContaminationIndex::build(&[]);
        let out = validate(
            &t,
            &ImageRegistry::default_world(),
            &heldout,
            &ValidationConfig::default(),
        );
        assert!(!out.passed);
        assert!(out
            .failures
            .iter()
            .any(|f| f.contains("reference solution earned")));
    }

    #[test]
    fn contaminated_instruction_is_rejected() {
        let mut t = clean_task();
        t.instruction.body = format!(
            "{}\nAlso: rebuild the pytorch model from its saved weights and verify the outputs match the original within tolerance\n",
            t.instruction.body
        );
        let heldout = ContaminationIndex::build(&[(
            "tb2.1/pytorch-restore".to_string(),
            "Rebuild the PyTorch model from its saved weights and verify the outputs match the original within tolerance".to_string(),
        )]);
        let out = validate(
            &t,
            &ImageRegistry::default_world(),
            &heldout,
            &ValidationConfig::default(),
        );
        assert!(!out.passed);
        assert!(out.failures.iter().any(|f| f.contains("13-gram")));
    }

    #[test]
    fn build_failure_blocks_everything() {
        let mut t = clean_task();
        t.environment.base_image = "nope:1".into();
        let heldout = ContaminationIndex::build(&[]);
        let out = validate(
            &t,
            &ImageRegistry::default_world(),
            &heldout,
            &ValidationConfig::default(),
        );
        assert!(!out.passed);
        assert!(!out.build_ok);
        assert_eq!(out.reference_reward, 0.0);
    }

    #[test]
    fn proposer_self_check_swap_dance() {
        let t = clean_task();
        let check = proposer_self_check(&t, &ImageRegistry::default_world()).unwrap();
        assert!(check.ready_to_submit());
        assert_eq!(check.oracle_reward, 1.0);
        assert!(check.noop_reward < 0.5);
        // The original solution is untouched after the dance.
        assert!(t.solution.executable);
    }

    #[test]
    fn gameable_task_self_check_blocks_submission() {
        let mut t = clean_task();
        // A verifier that passes on nothing: every check trivially true.
        t.tests.checks = vec![TestCheck::CommandSucceeds {
            name: "always".into(),
            command: "echo done".into(),
        }];
        let check = proposer_self_check(&t, &ImageRegistry::default_world()).unwrap();
        // noop also earns 1.0 -> noop_failed = false -> not ready.
        assert!(!check.ready_to_submit());
    }
}
