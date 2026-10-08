//! The proposer workspace (Figure 18) with editable / host-managed / fixed
//! regions and per-round refresh.
//!
//! ```text
//! workspace/
//! ├─ skills/                    authoring instructions      (fixed in-episode)
//! │   ├─ create-task/           Harbor task format
//! │   └─ auto_env_scaling/      sandbox use + authoring
//! ├─ tools/                     authoring tools             (editable)
//! │   ├─ README.md
//! │   ├─ validate.py            build + reference + no-op
//! │   ├─ decontaminate.py       benchmark overlap
//! │   ├─ calibrate.py           solver rollouts + rewards
//! │   ├─ selftest.sh            test tool edits
//! │   └─ snippets/              reusable templates
//! ├─ cookbook/                  search results and domain notes (editable)
//! ├─ memory/                    past experience
//! │   ├─ lessons.md                                     (editable)
//! │   ├─ past_environments/                              (host-managed)
//! │   └─ past_trajectories/
//! │       ├─ failed/                                     (host-managed)
//! │       └─ successful/                                  (host-managed)
//! ├─ output/tasks/              new Harbor tasks            (editable)
//! │   └─ task_001/
//! │       ├─ instruction.md
//! │       ├─ environment/
//! │       ├─ solution/
//! │       └─ tests/
//! ├─ logs/                      execution results           (editable)
//! ├─ assignment/                current assignment          (fixed)
//! └─ SANDBOX_CONTRACT.md        sandbox rules               (fixed)
//! ```
//!
//! Region rules (Section 3.1 + D.1): "Tools and retained experience can
//! change across rounds; the assignment and sandbox rules remain fixed
//! during generation. Final admission checks are enforced outside the
//! editable workspace." Between rounds (harness optimization, Section 3.3)
//! the proposer "updates its own memory, skills, and tools"; during
//! generation it can edit memory and tools but not the authoring
//! instructions, the assignment, or the contract.

use crate::error::{Error, Result};
use crate::harbor::HarborTask;
use crate::memory::Memory;
use crate::skills;
use std::collections::BTreeMap;

/// Which phase the workspace is in — the edit permissions differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// An episode is running: skills, assignment, and the contract are
    /// immutable; past_* is host-managed read-only.
    DuringEpisode,
    /// Between rounds: harness optimization may edit skills too.
    BetweenRounds,
}

/// The region classification of a workspace path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// The proposer may edit it in both phases.
    Editable,
    /// Refreshed by the host each round; read-only for the proposer.
    HostManaged,
    /// Fixed during the current phase.
    Fixed,
}

/// Classify a workspace path under a phase.
pub fn region_of(path: &str, phase: Phase) -> Region {
    let p = path.trim_start_matches("./").trim_matches('/');
    // Host-managed memory regions (read-only for the proposer).
    if p.starts_with("memory/past_environments") || p.starts_with("memory/past_trajectories") {
        return Region::HostManaged;
    }
    // Fixed during episodes; skills open up between rounds.
    if p.starts_with("skills/") {
        return match phase {
            Phase::DuringEpisode => Region::Fixed,
            Phase::BetweenRounds => Region::Editable,
        };
    }
    // Always fixed: the assignment and the sandbox contract.
    if p.starts_with("assignment/") || p == "SANDBOX_CONTRACT.md" {
        return Region::Fixed;
    }
    // Everything else (tools, cookbook, memory notes, output, logs) is
    // editable.
    Region::Editable
}

/// The proposer workspace.
#[derive(Debug, Clone, PartialEq)]
pub struct Workspace {
    /// File contents keyed by workspace-relative path.
    pub files: BTreeMap<String, String>,
    /// Current phase.
    pub phase: Phase,
    /// The persistent memory (lessons + archive snapshot).
    pub memory: Memory,
}

impl Default for Workspace {
    fn default() -> Self {
        Workspace::new()
    }
}

impl Workspace {
    /// Build a fresh workspace with the Figure-18 layout: skills seeded
    /// from the skill texts, tools seeded with the editable authoring tools,
    /// and the sandbox contract.
    pub fn new() -> Self {
        let mut ws = Workspace {
            files: BTreeMap::new(),
            phase: Phase::DuringEpisode,
            memory: Memory::default(),
        };
        for skill in skills::proposer_skills() {
            ws.files.insert(
                format!("skills/{}/SKILL.md", skill.name),
                skill.body.to_string(),
            );
        }
        ws.files
            .insert("tools/README.md".into(), TOOLS_README.into());
        ws.files
            .insert("tools/validate.py".into(), VALIDATE_PY.into());
        ws.files
            .insert("tools/decontaminate.py".into(), DECONTAMINATE_PY.into());
        ws.files
            .insert("tools/calibrate.py".into(), CALIBRATE_PY.into());
        ws.files
            .insert("tools/selftest.sh".into(), SELFTEST_SH.into());
        ws.files.insert(
            "tools/snippets/verifier_template.sh".into(),
            skills::MANAGED_VERIFIER_TEMPLATE.into(),
        );
        ws.files.insert(
            "cookbook/README.md".into(),
            "# Cookbook\n\nSearch results and domain notes land here.\n".into(),
        );
        ws.files.insert("memory/lessons.md".into(), String::new());
        ws.files.insert(
            "SANDBOX_CONTRACT.md".into(),
            crate::sandbox::sandbox_contract_md(),
        );
        ws
    }

    /// Switch phase (host-controlled).
    pub fn set_phase(&mut self, phase: Phase) {
        self.phase = phase;
    }

    /// Read a file.
    pub fn read(&self, path: &str) -> Option<&str> {
        self.files.get(path).map(|s| s.as_str())
    }

    /// The region of a path under the current phase.
    pub fn region(&self, path: &str) -> Region {
        region_of(path, self.phase)
    }

    /// Edit (or create) a file, enforcing the region rules. Attempts to
    /// edit fixed or host-managed regions return
    /// [`Error::SandboxViolation`].
    pub fn edit(&mut self, path: &str, content: impl Into<String>) -> Result<()> {
        match self.region(path) {
            Region::Editable => {
                self.files.insert(path.to_string(), content.into());
                Ok(())
            }
            Region::HostManaged => Err(Error::sandbox_violation(
                path,
                "host-managed (refreshed from the training pool each round)",
                "proposer attempted to edit host-managed data",
            )),
            Region::Fixed => Err(Error::sandbox_violation(
                path,
                "fixed during generation",
                "proposer attempted to edit a fixed region",
            )),
        }
    }

    /// Append to a log under `logs/`.
    pub fn log(&mut self, name: &str, line: impl Into<String>) -> Result<()> {
        let path = format!("logs/{name}");
        let existing = self.files.get(&path).cloned().unwrap_or_default();
        let mut content = existing;
        content.push_str(&line.into());
        content.push('\n');
        self.edit(&path, content)
    }

    /// Install the current assignment (host-side; the proposer reads it).
    pub fn set_assignment(&mut self, assignment_text: &str) {
        self.files
            .insert("assignment/current.md".into(), assignment_text.to_string());
    }

    /// Host-side per-round refresh: replace `memory/past_trajectories/`
    /// (index + buckets) and `memory/past_environments/` from the training
    /// pool. Bypasses the edit guard — the host owns these regions.
    pub fn refresh_from_pool(
        &mut self,
        records: Vec<crate::trajectory::RolloutRecord>,
        attempts: Vec<(String, crate::trajectory::Trajectory)>,
        envs: &[&HarborTask],
    ) {
        self.memory.refresh_from_pool(records, attempts, envs);
        // Project the memory into the visible file tree.
        self.files.insert(
            "memory/past_trajectories/index.jsonl".into(),
            self.memory.archive.to_index_jsonl(),
        );
        for (key, traj) in &self.memory.archive.attempts {
            let body = serde_json::to_string_pretty(traj).unwrap_or_default();
            self.files
                .insert(format!("memory/past_trajectories/{key}.json"), body);
        }
        for (env_id, snapshot) in &self.memory.past_environments {
            self.files.insert(
                format!("memory/past_environments/{env_id}/notes.md"),
                snapshot.clone(),
            );
        }
    }

    /// Write the finished canonical task into `output/tasks/<env_id>/`
    /// (the skill's "Where to write" section: "env_NNN/ itself must contain
    /// environment/, solution/, tests/, instruction.md, task.toml").
    pub fn write_task(&mut self, task: &HarborTask) -> Result<()> {
        let root = format!("output/tasks/{}", task.env_id);
        let files: Vec<(String, String)> = vec![
            (
                format!("{root}/instruction.md"),
                task.instruction.body.clone(),
            ),
            (format!("{root}/task.toml"), task.render_task_toml()),
            (
                format!("{root}/environment/Dockerfile"),
                task.environment.render_dockerfile(),
            ),
            (
                format!("{root}/solution/solve.sh"),
                task.solution.render_sh(),
            ),
            (format!("{root}/tests/test.sh"), task.tests.test_sh.clone()),
            (
                format!("{root}/tests/test_outputs.py"),
                render_test_outputs_py(task),
            ),
            (format!("{root}/README.md"), task.readme.clone()),
        ];
        for (path, content) in files {
            self.edit(&path, content)?;
        }
        Ok(())
    }

    /// The set of task directories under `output/tasks/`.
    pub fn output_tasks(&self) -> Vec<String> {
        self.files
            .keys()
            .filter(|p| p.starts_with("output/tasks/") && p.ends_with("/instruction.md"))
            .map(|p| {
                p.trim_start_matches("output/tasks/")
                    .trim_end_matches("/instruction.md")
                    .to_string()
            })
            .collect()
    }

    /// The harness state that persists to the next round: tools, cookbook,
    /// lessons, and the (edited) skills. Assignment, contract, host-managed
    /// regions, outputs, and logs are excluded.
    pub fn persist_harness_state(&self) -> BTreeMap<String, String> {
        let mut state = BTreeMap::new();
        for (path, content) in &self.files {
            let keep = match region_of(path, Phase::BetweenRounds) {
                Region::Editable => {
                    path.starts_with("tools/")
                        || path.starts_with("cookbook/")
                        || path.starts_with("skills/")
                        || path == "memory/lessons.md"
                }
                _ => false,
            };
            if keep {
                state.insert(path.clone(), content.clone());
            }
        }
        state
    }
}

/// Render `tests/test_outputs.py` from the suite's checks.
fn render_test_outputs_py(task: &HarborTask) -> String {
    use crate::world::TestCheck;
    let mut s = String::from("from pathlib import Path\nimport json\n\n");
    for check in &task.tests.checks {
        match check {
            TestCheck::FileExists { name, path } => {
                s.push_str(&format!(
                    "\ndef {name}():\n    assert Path(\"{path}\").exists()\n"
                ));
            }
            TestCheck::FileContains { name, path, needle } => {
                s.push_str(&format!(
                    "\ndef {name}():\n    assert \"{needle}\" in Path(\"{path}\").read_text()\n"
                ));
            }
            TestCheck::FileJsonKey {
                name,
                path,
                key,
                expected,
            } => {
                s.push_str(&format!(
                    "\ndef {name}():\n    data = json.loads(Path(\"{path}\").read_text())\n    assert data[\"{key}\"] == \"{expected}\"\n"
                ));
            }
            TestCheck::FileLineCount { name, path, count } => {
                s.push_str(&format!(
                    "\ndef {name}():\n    assert len(Path(\"{path}\").read_text().splitlines()) == {count}\n"
                ));
            }
            TestCheck::CommandSucceeds { name, command } => {
                s.push_str(&format!(
                    "\ndef {name}():\n    import subprocess\n    assert subprocess.run(\"{command}\", shell=True).returncode == 0\n"
                ));
            }
        }
    }
    s
}

/// `tools/README.md`.
pub const TOOLS_README: &str = r#"# Authoring tools

Editable by the proposer (the validation/calibration checks the host runs
stay fixed and live outside this sandbox).

- `validate.py <task> oracle` — build + reference solution + no-op check.
- `decontaminate.py <task>` — 13-gram benchmark overlap check.
- `calibrate.py <task>` — run the solver model over the task; reports
  mean/std of attempt rewards.
- `selftest.sh` — run after editing any tool, before using it.
- `snippets/` — reusable templates (verifier, Dockerfile, instruction).
"#;

/// `tools/validate.py` — the editable mirror of the host's build +
/// reference + no-op check.
pub const VALIDATE_PY: &str = r#"#!/usr/bin/env python3
"""Validate a Harbor task inside the sandbox (editable tool).

Usage: python3 validate.py <task> oracle
Runs the reference solution then the verifier; prints the reward.
The no-op swap dance is documented in the auto_env_scaling skill.
"""
import sys

def main() -> int:
    task = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "oracle"
    if mode != "oracle":
        print(f"unknown mode {mode}", file=sys.stderr)
        return 2
    reward = run_oracle(task)
    print(reward)
    return 0 if reward == 1.0 else 1

def run_oracle(task: str) -> float:
    # Builds the environment, runs solution/solve.sh, runs tests/test.sh,
    # and reads /logs/verifier/reward.txt.
    raise NotImplementedError("wired to the sandbox_client at runtime")

if __name__ == "__main__":
    sys.exit(main())
"#;

/// `tools/decontaminate.py`.
pub const DECONTAMINATE_PY: &str = r#"#!/usr/bin/env python3
"""Check a task's instruction for 13-gram overlap with held-out benchmarks."""
import sys

def main() -> int:
    instruction = open(sys.argv[1] + "/instruction.md").read()
    hits = overlap_13gram(instruction)
    print("clean" if not hits else f"CONTAMINATED: {hits}")
    return 0 if not hits else 1

def overlap_13gram(text: str) -> list:
    raise NotImplementedError("wired to the decontamination service at runtime")

if __name__ == "__main__":
    sys.exit(main())
"#;

/// `tools/calibrate.py` — the solver_model front-end.
pub const CALIBRATE_PY: &str = r#"#!/usr/bin/env python3
"""Estimate a task's difficulty with the current solver (solver_model tool).

Reports mean and std of attempt rewards; target mean in [0.25, 0.75] with
std >= 0.1 before submitting.
"""
import sys

def main() -> int:
    task = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rewards = solver_model(task, n)
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    print(f"mean={mean:.3f} std={var ** 0.5:.3f}")
    return 0

def solver_model(task: str, n: int) -> list:
    raise NotImplementedError("wired to the calibration service at runtime")

if __name__ == "__main__":
    sys.exit(main())
"#;

/// `tools/selftest.sh` — run after editing any tool.
pub const SELFTEST_SH: &str = r#"#!/bin/bash
# Self-test: run after editing any tool in this directory, before using it.
set -e
for tool in validate.py decontaminate.py calibrate.py; do
  python3 -m py_compile "$tool"
done
echo "tools OK"
"#;

/// `tools/validate_fast.py` — the merged single-pass validation script the
/// harness optimizer installs after its first round ("it merged several
/// validation steps into a faster script").
pub const VALIDATE_FAST_PY: &str = r#"#!/usr/bin/env python3
"""Merged validation: oracle + no-op in one pass (faster than validate.py)."""
import sys

def main() -> int:
    task = sys.argv[1]
    oracle, noop = run_both(task)
    print(f"oracle={oracle} noop={noop}")
    ok = oracle == 1.0 and noop < 1.0
    return 0 if ok else 1

def run_both(task: str):
    raise NotImplementedError("wired to the sandbox_client at runtime")

if __name__ == "__main__":
    sys.exit(main())
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harbor::{
        AgentConfig, Difficulty, Instruction, RewardStrategy, SkillTag, TaskEnvironment, TaskMeta,
        TaskMetadata, TaskToml, VerifierConfig,
    };
    use crate::trajectory::{Outcome, Role, Trajectory};
    use crate::world::{RewardFile, ShellOp, TestCheck, TestSuite};

    fn task() -> HarborTask {
        HarborTask {
            env_id: "env_002".into(),
            meta: TaskMeta {
                name: "acme/demo".into(),
                version: "1.0.0".into(),
                description: "demo".into(),
                keywords: vec!["csv".into(), "python".into(), "pytest".into()],
            },
            instruction: Instruction::new("# Demo task\n\nWrite `/app/out/report.csv` with the reconciled totals.\n"),
            environment: TaskEnvironment::default(),
            solution: SolutionScript {
                ops: vec![ShellOp::WriteFile { path: "/app/out/report.csv".into(), content: "id,total\n".into() }],
                executable: true,
            },
            tests: TestSuite {
                test_sh: "#!/bin/bash\nuvx --with pytest==8.4.1 pytest /tests/test_outputs.py\n".into(),
                checks: vec![TestCheck::FileExists { name: "test_report".into(), path: "/app/out/report.csv".into() }],
                reward_file: RewardFile::Txt,
            },
            readme: "# Demo\n\nFull environment, verifier, layout, and running instructions for the demo task.\n".into(),
            steps: Vec::new(),
            toml: TaskToml {
                schema_version: "1.0".into(),
                task: TaskMeta {
                    name: "acme/demo".into(),
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
    fn fresh_workspace_matches_figure18() {
        let ws = Workspace::new();
        for path in [
            "skills/create-task/SKILL.md",
            "skills/auto_env_scaling/SKILL.md",
            "tools/README.md",
            "tools/validate.py",
            "tools/decontaminate.py",
            "tools/calibrate.py",
            "tools/selftest.sh",
            "tools/snippets/verifier_template.sh",
            "cookbook/README.md",
            "memory/lessons.md",
            "SANDBOX_CONTRACT.md",
        ] {
            assert!(ws.files.contains_key(path), "missing {path}");
        }
    }

    #[test]
    fn regions_follow_phase() {
        assert_eq!(
            region_of("tools/validate.py", Phase::DuringEpisode),
            Region::Editable
        );
        assert_eq!(
            region_of("memory/lessons.md", Phase::DuringEpisode),
            Region::Editable
        );
        assert_eq!(
            region_of("memory/past_trajectories/index.jsonl", Phase::DuringEpisode),
            Region::HostManaged
        );
        assert_eq!(
            region_of("skills/create-task/SKILL.md", Phase::DuringEpisode),
            Region::Fixed
        );
        assert_eq!(
            region_of("skills/auto_env_scaling/SKILL.md", Phase::BetweenRounds),
            Region::Editable
        );
        assert_eq!(
            region_of("assignment/current.md", Phase::BetweenRounds),
            Region::Fixed
        );
        assert_eq!(
            region_of("SANDBOX_CONTRACT.md", Phase::BetweenRounds),
            Region::Fixed
        );
    }

    #[test]
    fn edits_enforced() {
        let mut ws = Workspace::new();
        ws.phase = Phase::DuringEpisode;
        assert!(ws.edit("tools/validate.py", "new content").is_ok());
        assert!(ws.edit("memory/lessons.md", "- lesson").is_ok());
        let err = ws
            .edit("skills/create-task/SKILL.md", "tampered")
            .unwrap_err();
        assert!(matches!(err, Error::SandboxViolation { .. }));
        let err2 = ws.edit("assignment/current.md", "faked").unwrap_err();
        assert!(matches!(err2, Error::SandboxViolation { .. }));
        let err3 = ws
            .edit("memory/past_trajectories/index.jsonl", "[]")
            .unwrap_err();
        assert!(matches!(err3, Error::SandboxViolation { .. }));
        let err4 = ws.edit("SANDBOX_CONTRACT.md", "no rules").unwrap_err();
        assert!(matches!(err4, Error::SandboxViolation { .. }));
        // Between rounds, skills open up; assignment/contract stay fixed.
        ws.phase = Phase::BetweenRounds;
        assert!(ws.edit("skills/web-cache/SKILL.md", "# web-cache").is_ok());
        assert!(ws.edit("assignment/current.md", "faked").is_err());
    }

    #[test]
    fn refresh_writes_archive_tree() {
        let mut ws = Workspace::new();
        let records = vec![crate::trajectory::RolloutRecord {
            env_id: "env_001".into(),
            solver: "s".into(),
            round: 1,
            provenance: crate::trajectory::Provenance::SelfRound(1),
            rewards: vec![0.0, 1.0],
            reward_std: 0.5,
            check_failures: Default::default(),
            path: "memory/past_trajectories/failed/env_001.json".into(),
        }];
        let attempts = vec![(
            "env_001".to_string(),
            Trajectory {
                role: Role::Solver,
                prompt: "solve".into(),
                turns: vec![],
                outcome: Some(Outcome {
                    reward: 0.0,
                    verifier_stdout: "FAILED test_a".into(),
                    passed_tests: 0,
                    total_tests: 2,
                    failed_checks: vec!["test_a".into()],
                }),
            },
        )];
        let t = task();
        ws.refresh_from_pool(records, attempts, &[&t]);
        assert!(ws
            .files
            .contains_key("memory/past_trajectories/index.jsonl"));
        assert!(ws
            .files
            .contains_key("memory/past_trajectories/failed/env_001.json"));
        assert!(ws
            .files
            .contains_key("memory/past_environments/env_002/notes.md"));
        let idx = ws.read("memory/past_trajectories/index.jsonl").unwrap();
        assert!(idx.contains("\"env_id\":\"env_001\"") || idx.contains("\"env_id\": \"env_001\""));
    }

    #[test]
    fn write_task_lays_out_canonical_files() {
        let mut ws = Workspace::new();
        let t = task();
        ws.write_task(&t).unwrap();
        for path in [
            "output/tasks/env_002/instruction.md",
            "output/tasks/env_002/task.toml",
            "output/tasks/env_002/environment/Dockerfile",
            "output/tasks/env_002/solution/solve.sh",
            "output/tasks/env_002/tests/test.sh",
            "output/tasks/env_002/tests/test_outputs.py",
            "output/tasks/env_002/README.md",
        ] {
            assert!(ws.files.contains_key(path), "missing {path}");
        }
        assert_eq!(ws.output_tasks(), vec!["env_002".to_string()]);
        let py = ws
            .read("output/tasks/env_002/tests/test_outputs.py")
            .unwrap();
        assert!(py.contains("def test_report():"));
        let sh = ws.read("output/tasks/env_002/solution/solve.sh").unwrap();
        assert!(sh.contains("/app/out/report.csv"));
    }

    #[test]
    fn persist_harness_state_excludes_fixed_and_transient() {
        let mut ws = Workspace::new();
        ws.phase = Phase::BetweenRounds;
        ws.edit("memory/lessons.md", "- keep").unwrap();
        ws.edit("tools/snippets/note.md", "note").unwrap();
        let t = task();
        ws.write_task(&t).unwrap();
        ws.set_assignment("assignment text");
        let persisted = ws.persist_harness_state();
        assert!(persisted.contains_key("memory/lessons.md"));
        assert!(persisted.contains_key("tools/snippets/note.md"));
        assert!(persisted.contains_key("tools/validate.py"));
        assert!(!persisted.contains_key("assignment/current.md"));
        assert!(!persisted.contains_key("SANDBOX_CONTRACT.md"));
        assert!(!persisted.contains_key("output/tasks/env_002/instruction.md"));
    }

    #[test]
    fn logging_appends() {
        let mut ws = Workspace::new();
        ws.log("episode.log", "first").unwrap();
        ws.log("episode.log", "second").unwrap();
        assert_eq!(ws.read("logs/episode.log").unwrap(), "first\nsecond\n");
    }
}
