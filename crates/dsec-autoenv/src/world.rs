//! The simulated container: deterministic execution semantics for Harbor
//! tasks, so the whole flywheel (validation, calibration, RL rollouts) runs
//! and is testable without Docker or GPUs.
//!
//! The paper executes tasks in real containers (Harbor builds the image, the
//! agent acts inside it, the verifier writes `/logs/verifier/reward.txt`).
//! This module reproduces the *observable contract* of that execution as
//! pure logic:
//!
//! * building the environment = resolving the base image and packages
//!   against a registry;
//! * running a solution = applying its script ops to a [`WorldState`];
//! * running the verifier = evaluating the suite's checks and producing a
//!   CTRF report whose reward is *tests passed / total* — exactly the
//!   semantics of the managed verifier template
//!   (`/opt/verifier_template.sh`, Appendix E.2: "reward = pytest tests
//!   passed / total, via CTRF ... writes a fractional reward 0..1 to
//!   `/logs/verifier/reward.txt`");
//! * the Oracle agent = reference solution then verifier (reward should be
//!   1.0);
//! * the no-op solution = `#!/bin/sh\nexit 0` (should fail);
//! * reset reproducibility = re-initialization must reproduce the identical
//!   initial state (Appendix C.1).

use crate::harbor::HarborTask;
use crate::trajectory::Outcome;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// One operation of a shell script (`solve.sh`, `setup.sh`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShellOp {
    /// `mkdir -p <dir>`.
    MkDir(String),
    /// Write a file with the given content (heredoc / redirect).
    WriteFile {
        /// Destination path.
        path: String,
        /// File content.
        content: String,
    },
    /// Run a command; a non-zero expected exit fails the script.
    Run {
        /// The command line.
        cmd: String,
        /// Expected exit code (None = don't check).
        expect_exit: Option<i32>,
    },
    /// `rm -- "$0"` — the reserved self-removal line of
    /// `steps/{name}/workdir/setup.sh`.
    RmSelf,
    /// `exit <code>` — terminates the script.
    Exit(i32),
}

/// A shell script: the reference `solution/solve.sh`, a solver attempt, or a
/// step's `setup.sh`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolutionScript {
    /// Operations in order.
    pub ops: Vec<ShellOp>,
    /// Whether the script is executable (`chmod +x`). The Oracle agent fails
    /// on a non-executable `solve.sh` (pitfall 5).
    pub executable: bool,
}

impl SolutionScript {
    /// The empty solution the host runs as the do-nothing baseline
    /// (Section 3.2: "an empty solution, which should receive less than
    /// half"). The skill's exact text: `#!/bin/sh\nexit 0`.
    pub fn noop() -> Self {
        SolutionScript {
            ops: vec![ShellOp::Exit(0)],
            executable: true,
        }
    }

    /// Render the script as shell text.
    pub fn render_sh(&self) -> String {
        let mut s = String::from("#!/bin/bash\n");
        for op in &self.ops {
            match op {
                ShellOp::MkDir(d) => s.push_str(&format!("mkdir -p {d}\n")),
                ShellOp::WriteFile { path, content } => {
                    s.push_str(&format!("cat > {path} <<'EOF'\n{content}\nEOF\n"));
                }
                ShellOp::Run { cmd, .. } => s.push_str(&format!("{cmd}\n")),
                ShellOp::RmSelf => s.push_str("rm -- \"$0\"\n"),
                ShellOp::Exit(c) => s.push_str(&format!("exit {c}\n")),
            }
        }
        s
    }
}

/// One verifier check. Every variant carries a test name; the name doubles
/// as the key of the per-check reward dict and the failure signature the
/// assignment extractor reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TestCheck {
    /// A file must exist.
    FileExists {
        /// Test name.
        name: String,
        /// Absolute path to check.
        path: String,
    },
    /// A file must contain a needle.
    FileContains {
        /// Test name.
        name: String,
        /// Absolute path to check.
        path: String,
        /// Substring that must appear.
        needle: String,
    },
    /// A JSON file must have a key equal to an expected value.
    FileJsonKey {
        /// Test name.
        name: String,
        /// Absolute path to check.
        path: String,
        /// JSON key (dot-separated for nesting).
        key: String,
        /// Expected value (compared as text).
        expected: String,
    },
    /// A file must have an exact line count.
    FileLineCount {
        /// Test name.
        name: String,
        /// Absolute path to check.
        path: String,
        /// Expected number of lines.
        count: usize,
    },
    /// A shell command must succeed. Supported forms: `test -f <path>`,
    /// `diff <a> <b>`, `python3 <script>`.
    CommandSucceeds {
        /// Test name.
        name: String,
        /// The command to run.
        command: String,
    },
}

impl TestCheck {
    /// The test's name (its identity in the CTRF report and keyed rewards).
    pub fn name(&self) -> &str {
        match self {
            TestCheck::FileExists { name, .. }
            | TestCheck::FileContains { name, .. }
            | TestCheck::FileJsonKey { name, .. }
            | TestCheck::FileLineCount { name, .. }
            | TestCheck::CommandSucceeds { name, .. } => name,
        }
    }

    /// Evaluate the check against a world state.
    pub fn eval(&self, state: &WorldState) -> bool {
        match self {
            TestCheck::FileExists { path, .. } => state.files.contains_key(path),
            TestCheck::FileContains { path, needle, .. } => state
                .files
                .get(path)
                .map(|c| c.contains(needle.as_str()))
                .unwrap_or(false),
            TestCheck::FileJsonKey {
                path,
                key,
                expected,
                ..
            } => state
                .files
                .get(path)
                .and_then(|c| serde_json::from_str::<serde_json::Value>(c).ok())
                .map(|v| json_lookup(&v, key) == Some(expected.clone()))
                .unwrap_or(false),
            TestCheck::FileLineCount { path, count, .. } => state
                .files
                .get(path)
                .map(|c| c.lines().count() == *count)
                .unwrap_or(false),
            TestCheck::CommandSucceeds { command, .. } => command_succeeds(command, state),
        }
    }
}

/// Dot-separated JSON lookup, returning scalar values as text.
fn json_lookup(v: &serde_json::Value, key: &str) -> Option<String> {
    let mut cur = v;
    for part in key.split('.') {
        cur = cur.get(part)?;
    }
    match cur {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Deterministic command semantics for `CommandSucceeds`.
fn command_succeeds(cmd: &str, state: &WorldState) -> bool {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    match parts.as_slice() {
        ["test", "-f", path] => state.files.contains_key(*path),
        ["diff", a, b] => match (state.files.get(*a), state.files.get(*b)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        },
        ["python3", script] => {
            // A solver script "succeeds" when it exists in the state and
            // declares success (a trailing `exit 0` convention in content).
            state
                .files
                .get(*script)
                .map(|c| c.contains("exit 0") || !c.contains("raise"))
                .unwrap_or(false)
        }
        _ => false,
    }
}

/// Where the verifier writes its reward (Appendix E.1: all verifier options).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewardFile {
    /// `/logs/verifier/reward.txt` — a single number (usually 0 or 1; the
    /// managed template writes the fractional tests-passed/total).
    Txt,
    /// `/logs/verifier/reward.json` — `{"accuracy": 0.95, ...}` for multiple
    /// metrics.
    Json,
}

impl RewardFile {
    /// The in-container path of the reward file.
    pub fn path(&self) -> &'static str {
        match self {
            RewardFile::Txt => "/logs/verifier/reward.txt",
            RewardFile::Json => "/logs/verifier/reward.json",
        }
    }
}

/// The verifier suite: `tests/test.sh` plus the checks it runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TestSuite {
    /// The rendered `test.sh` (always absolute paths — pitfall 2).
    pub test_sh: String,
    /// The individual checks (pytest test functions in the real format).
    pub checks: Vec<TestCheck>,
    /// Which reward file the script writes.
    pub reward_file: RewardFile,
}

impl TestSuite {
    /// Run the suite against a world state, producing a CTRF report.
    pub fn run(&self, state: &WorldState) -> CtrfReport {
        let tests = self
            .checks
            .iter()
            .map(|c| TestResult {
                name: c.name().to_string(),
                status: if c.eval(state) {
                    TestStatus::Passed
                } else {
                    TestStatus::Failed
                },
                duration_ms: 1,
            })
            .collect();
        CtrfReport { tests }
    }

    /// The stdout the solver's attempt would capture: a pytest-style
    /// summary plus the reward the managed template writes.
    pub fn verifier_stdout(&self, report: &CtrfReport) -> String {
        let mut s = format!(
            "pytest /tests/test_outputs.py\n{} passed, {} failed in 0.01s\n",
            report.passed_count(),
            report.failed_count()
        );
        if !report.failed_names().is_empty() {
            s.push_str(&format!("FAILED {}\n", report.failed_names().join(", ")));
        }
        s.push_str(&format!(
            "echo {} > {}\n",
            report.reward(),
            self.reward_file.path()
        ));
        s
    }
}

/// A single CTRF test result (Common Test Report Format).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TestResult {
    /// Test name.
    pub name: String,
    /// Outcome.
    pub status: TestStatus,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

/// CTRF test status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TestStatus {
    /// The check passed.
    Passed,
    /// The check failed.
    Failed,
}

impl TestStatus {
    /// CTRF tag.
    pub fn as_str(&self) -> &'static str {
        match self {
            TestStatus::Passed => "passed",
            TestStatus::Failed => "failed",
        }
    }
}

/// A CTRF run report: the artifact the managed verifier template produces
/// and the host parses to compute Eq. 1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CtrfReport {
    /// Results in suite order.
    pub tests: Vec<TestResult>,
}

impl CtrfReport {
    /// Number of passed tests.
    pub fn passed_count(&self) -> usize {
        self.tests
            .iter()
            .filter(|t| t.status == TestStatus::Passed)
            .count()
    }

    /// Number of failed tests.
    pub fn failed_count(&self) -> usize {
        self.tests
            .iter()
            .filter(|t| t.status == TestStatus::Failed)
            .count()
    }

    /// Names of failed tests, in order.
    pub fn failed_names(&self) -> Vec<String> {
        self.tests
            .iter()
            .filter(|t| t.status == TestStatus::Failed)
            .map(|t| t.name.clone())
            .collect()
    }

    /// **Eq. 1**: the solver reward — the fraction of the environment's `N`
    /// tests that pass. "A partial solution receives partial reward."
    pub fn reward(&self) -> f64 {
        if self.tests.is_empty() {
            // Pitfall 1: a suite with no checks never writes the reward file
            // — the task "passes" silently with reward 0.
            return 0.0;
        }
        self.passed_count() as f64 / self.tests.len() as f64
    }

    /// Keyed per-check rewards (0/1 per check), used by the dict form of
    /// multi-step `min_reward` gating and the `mean` roll-up.
    pub fn keyed_rewards(&self) -> BTreeMap<String, f64> {
        self.tests
            .iter()
            .map(|t| {
                (
                    t.name.clone(),
                    if t.status == TestStatus::Passed {
                        1.0
                    } else {
                        0.0
                    },
                )
            })
            .collect()
    }

    /// Render as CTRF JSON (`{"results":{"tests":[...]}}`).
    pub fn render_json(&self) -> String {
        let mut tests = Vec::new();
        for t in &self.tests {
            tests.push(serde_json::json!({
                "name": t.name,
                "status": t.status.as_str(),
                "duration": t.duration_ms,
            }));
        }
        serde_json::json!({
            "results": {
                "summary": {
                    "tests": self.tests.len(),
                    "passed": self.passed_count(),
                    "failed": self.failed_count(),
                },
                "tests": tests,
            }
        })
        .to_string()
    }

    /// Parse a CTRF JSON report.
    pub fn parse_json(raw: &str) -> crate::error::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(raw)?;
        let arr = v
            .pointer("/results/tests")
            .and_then(|t| t.as_array())
            .ok_or_else(|| crate::error::Error::invalid_task("CTRF JSON missing results.tests"))?;
        let mut tests = Vec::new();
        for t in arr {
            let name = t
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or_default()
                .to_string();
            let status = match t.get("status").and_then(|s| s.as_str()) {
                Some("passed") => TestStatus::Passed,
                _ => TestStatus::Failed,
            };
            let duration_ms = t.get("duration").and_then(|d| d.as_u64()).unwrap_or(0);
            tests.push(TestResult {
                name,
                status,
                duration_ms,
            });
        }
        Ok(CtrfReport { tests })
    }
}

/// The state of one task container: the files that exist in the workdir.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldState {
    /// Files planted by the environment plus files written by the agent /
    /// solution.
    pub files: BTreeMap<String, String>,
    /// Packages installed in the container.
    pub installed: BTreeSet<String>,
    /// Exit code of the last run script.
    pub last_exit: i32,
    /// Artifacts collected for the verifier environment (separate verifier
    /// mode).
    pub artifacts: Vec<String>,
}

impl WorldState {
    /// A stable content hash used by the reset-reproducibility check.
    pub fn content_hash(&self) -> u64 {
        let mut acc: u64 = 1469598103934665603;
        for (path, content) in &self.files {
            for byte in path.bytes().chain(content.bytes()) {
                acc ^= byte as u64;
                acc = acc.wrapping_mul(1099511628211);
            }
        }
        acc
    }
}

/// The registry of buildable base images and installable packages. Builds
/// resolve against this (a task that references an unknown image or an
/// uninstallable package fails validation).
#[derive(Debug, Clone, Default)]
pub struct ImageRegistry {
    /// Known base images.
    pub images: BTreeSet<String>,
    /// Known apt packages.
    pub packages: BTreeSet<String>,
}

impl ImageRegistry {
    /// A permissive default registry covering the base images and packages
    /// the paper's environments use.
    pub fn default_world() -> Self {
        let mut images = BTreeSet::new();
        for img in [
            "ubuntu:24.04",
            "python:3.12-slim",
            "pytorch/pytorch:2.6",
            "nvidia/cuda:12.4",
        ] {
            images.insert(img.to_string());
        }
        let mut packages = BTreeSet::new();
        for pkg in [
            "python3",
            "python3-pip",
            "curl",
            "openssh-client",
            "jq",
            "ripgrep",
            "build-essential",
            "git",
            "fasttext",
            "primer3",
            "libreoffice-calc",
        ] {
            packages.insert(pkg.to_string());
        }
        ImageRegistry { images, packages }
    }

    /// Check the build of a task environment. Returns one string per build
    /// failure.
    pub fn build_check(&self, task: &HarborTask) -> Vec<String> {
        let mut failures = Vec::new();
        if !self.images.contains(&task.environment.base_image) {
            failures.push(format!(
                "image build failed: unknown base image {}",
                task.environment.base_image
            ));
        }
        for pkg in &task.environment.packages {
            if !self.packages.contains(pkg) {
                failures.push(format!("image build failed: package {pkg} not installable"));
            }
        }
        failures
    }
}

/// A running task container: the task plus its current world state.
#[derive(Debug, Clone)]
pub struct TaskWorld {
    /// The environment's task.
    pub task: HarborTask,
    /// Current container state.
    pub state: WorldState,
    /// Build registry reference snapshot.
    pub registry: ImageRegistry,
    /// Hash of the freshly-initialized state, for reset checks.
    pub initial_hash: u64,
}

impl TaskWorld {
    /// Build the environment and produce the initial state. Fails if the
    /// base image or packages do not resolve.
    pub fn build(task: HarborTask, registry: &ImageRegistry) -> crate::error::Result<Self> {
        let failures = registry.build_check(&task);
        if !failures.is_empty() {
            return Err(crate::error::Error::EnvironmentRejected {
                env_id: task.env_id.clone(),
                reasons: failures,
            });
        }
        let state = Self::init_state(&task);
        let initial_hash = state.content_hash();
        Ok(TaskWorld {
            task,
            state,
            registry: registry.clone(),
            initial_hash,
        })
    }

    /// The freshly-initialized container state: environment files planted,
    /// packages installed, nothing else.
    pub fn init_state(task: &HarborTask) -> WorldState {
        WorldState {
            files: task.environment.files.clone(),
            installed: task.environment.packages.iter().cloned().collect(),
            last_exit: 0,
            artifacts: Vec::new(),
        }
    }

    /// Reset the container to its initial state. Returns whether the reset
    /// reproduced the initial state bit-for-bit (Appendix C.1's admission
    /// requirement: "Resetting must reproduce the initial state and a
    /// passing reference execution").
    pub fn reset(&mut self) -> bool {
        self.state = Self::init_state(&self.task);
        self.state.content_hash() == self.initial_hash
    }

    /// Run a script inside the container. A non-executable script fails
    /// immediately (the Oracle agent cannot run it). Returns the exit code.
    pub fn run_script(&mut self, script: &SolutionScript) -> i32 {
        if !script.executable {
            self.state.last_exit = 126;
            return 126;
        }
        for op in &script.ops {
            match op {
                ShellOp::MkDir(dir) => {
                    // Directories are implicit in the file map.
                    self.state.files.entry(format!("{dir}/.keep")).or_default();
                }
                ShellOp::WriteFile { path, content } => {
                    self.state.files.insert(path.clone(), content.clone());
                }
                ShellOp::Run { cmd, expect_exit } => {
                    // A `Run` inside a solution succeeds when its command's
                    // target semantics hold; we approximate with the same
                    // deterministic evaluator the tests use.
                    let ok = command_succeeds(cmd, &self.state) || cmd.starts_with("echo ");
                    let code = if ok { 0 } else { 1 };
                    if let Some(expected) = expect_exit {
                        if code != *expected {
                            self.state.last_exit = code;
                            return code;
                        }
                    }
                }
                ShellOp::RmSelf => {
                    // The script removes itself; the harness convention only.
                }
                ShellOp::Exit(code) => {
                    self.state.last_exit = *code;
                    return *code;
                }
            }
        }
        self.state.last_exit = 0;
        0
    }

    /// Run the verifier suite against the current state.
    pub fn run_tests(&self) -> CtrfReport {
        self.task.tests.run(&self.state)
    }

    /// Run one step's suite (multi-step tasks).
    pub fn run_step_tests(&self, step: &str) -> CtrfReport {
        match self.task.steps.iter().find(|s| s.name == step) {
            Some(spec) => spec.tests.run(&self.state),
            None => CtrfReport { tests: Vec::new() },
        }
    }
}

/// The result of running one full attempt (solution then verifier) in a
/// fresh container.
#[derive(Debug, Clone, PartialEq)]
pub struct AttemptResult {
    /// Script exit code.
    pub exit_code: i32,
    /// The CTRF report.
    pub report: CtrfReport,
    /// Eq. 1 reward (fraction of tests passed).
    pub reward: f64,
}

/// Run one attempt of `script` against a fresh world for `task`: plant the
/// environment, run the script, run the verifier. This is the entire
/// observable semantics of "the solver sees only the instruction and works
/// inside the environment's container".
pub fn run_attempt(
    task: &HarborTask,
    script: &SolutionScript,
    registry: &ImageRegistry,
) -> crate::error::Result<AttemptResult> {
    let mut world = TaskWorld::build(task.clone(), registry)?;
    let exit_code = world.run_script(script);
    let report = world.run_tests();
    let reward = if exit_code == 0 {
        report.reward()
    } else {
        report.reward() * 0.0
    };
    Ok(AttemptResult {
        exit_code,
        report,
        reward,
    })
}

/// The Oracle agent (`harbor run -a oracle`): runs the reference solution
/// and then the verifier. "Reward should be 1.0" (Appendix E.1 Step 7).
pub fn oracle_run(
    task: &HarborTask,
    registry: &ImageRegistry,
) -> crate::error::Result<AttemptResult> {
    run_attempt(task, &task.solution, registry)
}

/// The do-nothing solution run: Section 3.2's second validation arm ("an
/// empty solution, which should receive less than half").
pub fn noop_run(
    task: &HarborTask,
    registry: &ImageRegistry,
) -> crate::error::Result<AttemptResult> {
    run_attempt(task, &SolutionScript::noop(), registry)
}

/// Convert an attempt into the trajectory outcome the rollout archive
/// stores (with `verifier_stdout` for failure-signature extraction).
pub fn attempt_outcome(task: &HarborTask, attempt: &AttemptResult) -> Outcome {
    Outcome {
        reward: attempt.reward,
        verifier_stdout: task.tests.verifier_stdout(&attempt.report),
        passed_tests: attempt.report.passed_count(),
        total_tests: attempt.report.tests.len(),
        failed_checks: attempt.report.failed_names(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harbor::{
        AgentConfig, Difficulty, Instruction, RewardStrategy, SkillTag, TaskEnvironment, TaskMeta,
        TaskMetadata, TaskToml, VerifierConfig,
    };

    fn demo_task() -> HarborTask {
        let checks = vec![
            TestCheck::FileExists {
                name: "report_exists".into(),
                path: "/app/out/report.csv".into(),
            },
            TestCheck::FileContains {
                name: "report_header".into(),
                path: "/app/out/report.csv".into(),
                needle: "id,total".into(),
            },
            TestCheck::FileJsonKey {
                name: "summary_json".into(),
                path: "/app/out/summary.json".into(),
                key: "matched_count".into(),
                expected: "3".into(),
            },
            TestCheck::CommandSucceeds {
                name: "pipeline_runs".into(),
                command: "python3 /app/reconcile.py".into(),
            },
        ];
        let mut env_files = BTreeMap::new();
        env_files.insert(
            "/app/exports/billing.csv".into(),
            "invoice_id,net,tax\nA1,10,2\n".into(),
        );
        HarborTask {
            env_id: "env_001".into(),
            meta: TaskMeta {
                name: "acme/reconcile".into(),
                version: "1.0.0".into(),
                description: "reconcile exports".into(),
                keywords: vec!["csv".into(), "python".into(), "pytest".into()],
            },
            instruction: Instruction::new("# Reconcile\nWrite /app/out/report.csv.\n"),
            environment: TaskEnvironment {
                files: env_files,
                network_mode: crate::harbor::NetworkMode::NoNetwork,
                ..Default::default()
            },
            solution: SolutionScript {
                ops: vec![
                    ShellOp::MkDir("/app/out".into()),
                    ShellOp::WriteFile { path: "/app/reconcile.py".into(), content: "print('ok')\nexit 0".into() },
                    ShellOp::WriteFile {
                        path: "/app/out/report.csv".into(),
                        content: "id,total\nA1,12\n".into(),
                    },
                    ShellOp::WriteFile {
                        path: "/app/out/summary.json".into(),
                        content: "{\"matched_count\": 3}".into(),
                    },
                ],
                executable: true,
            },
            tests: TestSuite {
                test_sh: "#!/bin/bash\nuvx --with pytest==8.4.1 pytest /tests/test_outputs.py\n".into(),
                checks,
                reward_file: RewardFile::Txt,
            },
            readme: "# Reconcile\n\nFull description of the reconciliation task with running instructions.\n".into(),
            steps: Vec::new(),
            toml: TaskToml {
                schema_version: "1.0".into(),
                task: TaskMeta {
                    name: "acme/reconcile".into(),
                    version: "1.0.0".into(),
                    description: "reconcile exports".into(),
                    keywords: vec!["csv".into(), "python".into(), "pytest".into()],
                },
                metadata: TaskMetadata { difficulty: Difficulty::Medium, category: "programming".into(), tags: vec![] },
                agent: AgentConfig::default(),
                verifier: VerifierConfig::default(),
                reward_strategy: RewardStrategy::Mean,
            },
            skills: vec![SkillTag::new("csv"), SkillTag::new("python")],
        }
    }

    #[test]
    fn oracle_passes_with_full_reward() {
        let task = demo_task();
        let run = oracle_run(&task, &ImageRegistry::default_world()).unwrap();
        assert_eq!(run.exit_code, 0);
        assert_eq!(run.reward, 1.0);
        assert_eq!(run.report.passed_count(), 4);
    }

    #[test]
    fn noop_fails() {
        let task = demo_task();
        let run = noop_run(&task, &ImageRegistry::default_world()).unwrap();
        assert!(run.reward < 0.5, "no-op should fail, got {}", run.reward);
    }

    #[test]
    fn partial_solution_partial_reward() {
        let task = demo_task();
        // Only two of four checks satisfied.
        let partial = SolutionScript {
            ops: vec![
                ShellOp::WriteFile {
                    path: "/app/out/report.csv".into(),
                    content: "id,total\n".into(),
                },
                ShellOp::WriteFile {
                    path: "/app/out/summary.json".into(),
                    content: "{\"matched_count\": 2}".into(),
                },
            ],
            executable: true,
        };
        let run = run_attempt(&task, &partial, &ImageRegistry::default_world()).unwrap();
        assert!((run.reward - 0.5).abs() < 1e-9);
        assert_eq!(
            run.report.failed_names(),
            vec!["summary_json", "pipeline_runs"]
        );
    }

    #[test]
    fn non_executable_solution_fails_oracle() {
        let mut task = demo_task();
        task.solution.executable = false;
        let run = oracle_run(&task, &ImageRegistry::default_world()).unwrap();
        assert_eq!(run.exit_code, 126);
        assert_eq!(run.reward, 0.0);
    }

    #[test]
    fn build_failure_rejects_task() {
        let mut task = demo_task();
        task.environment.base_image = "not-a-real-image:9".into();
        let err = TaskWorld::build(task, &ImageRegistry::default_world()).unwrap_err();
        assert!(err.to_string().contains("image build failed"));
    }

    #[test]
    fn reset_reproduces_initial_state() {
        let task = demo_task();
        let mut world = TaskWorld::build(task, &ImageRegistry::default_world()).unwrap();
        world.run_script(&world.task.solution.clone());
        assert!(world.reset());
        assert_eq!(world.state.files.len(), 1); // only the planted export
    }

    #[test]
    fn ctrf_round_trip_and_reward() {
        let mut report = CtrfReport { tests: Vec::new() };
        for (i, ok) in [true, true, false].into_iter().enumerate() {
            report.tests.push(TestResult {
                name: format!("t{i}"),
                status: if ok {
                    TestStatus::Passed
                } else {
                    TestStatus::Failed
                },
                duration_ms: 2,
            });
        }
        let json = report.render_json();
        let parsed = CtrfReport::parse_json(&json).unwrap();
        assert_eq!(parsed, report);
        assert!((report.reward() - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(report.failed_names(), vec!["t2".to_string()]);
    }

    #[test]
    fn empty_suite_silent_zero() {
        let report = CtrfReport { tests: Vec::new() };
        assert_eq!(report.reward(), 0.0);
    }

    #[test]
    fn verifier_stdout_includes_reward_write() {
        let task = demo_task();
        let run = oracle_run(&task, &ImageRegistry::default_world()).unwrap();
        let out = task.tests.verifier_stdout(&run.report);
        assert!(out.contains("4 passed"));
        assert!(out.contains("echo 1 > /logs/verifier/reward.txt"));
    }

    #[test]
    fn solution_script_renders_shell() {
        let s = SolutionScript {
            ops: vec![ShellOp::MkDir("/app/out".into()), ShellOp::Exit(0)],
            executable: true,
        };
        let sh = s.render_sh();
        assert!(sh.contains("mkdir -p /app/out"));
        assert!(sh.contains("exit 0"));
    }
}
