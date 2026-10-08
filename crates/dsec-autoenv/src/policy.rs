//! Pluggable policies for the two roles, and the `solver_model` tool.
//!
//! Both roles act in a terminal; the policies here are the deterministic,
//! tabular stand-ins for the LLM policies the paper trains. Each policy
//! decision is a *sampled token* from a [`crate::dppo::PolicyBank`] table —
//! so every attempt produces a real trajectory of sampled actions with
//! stored log-probabilities, and DPPO can train exactly the way the paper
//! describes.
//!
//! * [`ScriptedSolver`] — the solver: for every verifier check it samples a
//!   strategy (solve-carefully / sloppy / skip) from the policy table of the
//!   skill that check exercises. `SolveCorrect` always passes; `Skip` always
//!   fails; `Sloppy` passes the coarse checks but fails the precision checks
//!   (`json_key`, `line_count`) — a real skill gap, not noise.
//! * [`ScriptedProposer`] — the proposer: reads the workspace (rollout
//!   index, past environments, assignment), samples a design move
//!   (conservative / balanced / ambitious) keyed by the assignment's
//!   direction, generates a complete Harbor task, runs the pre-submission
//!   validation (oracle-passes AND no-op-fails), optionally runs the
//!   `solver_model` pre-calibration, and revises up to the assignment's
//!   budget.
//! * [`SolverModelTool`] — "The proposer can run this check itself before
//!   submitting, using a solver_model tool similar to the ones in AutoData
//!   and GPT-Red."

use crate::assignment::{Assignment, Direction};
use crate::dppo::{PolicyBank, TokenStep};
use crate::error::Result;
use crate::harbor::{
    AgentConfig, Difficulty, HarborTask, Instruction, NetworkMode, RewardStrategy, SkillTag,
    TaskEnvironment, TaskMeta, TaskMetadata, TaskToml, VerifierConfig, VerifierMode,
};
use crate::trajectory::{Action, Observation, Outcome, Turn};
use crate::validate::{proposer_self_check, ProposerSelfCheck};
use crate::workspace::Workspace;
use crate::world::{self, AttemptResult, ImageRegistry, RewardFile, ShellOp, TestCheck, TestSuite};
use rand::Rng;
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Solver action vocabulary: one token per verifier check.
pub const SOLVER_ACTIONS: usize = 3;
/// Action 0: solve the check correctly.
pub const ACTION_SOLVE: usize = 0;
/// Action 1: a sloppy attempt — passes coarse checks, fails precision
/// checks.
pub const ACTION_SLOPPY: usize = 1;
/// Action 2: skip the check.
pub const ACTION_SKIP: usize = 2;

/// A reference-solution hint for diagnostic rollouts (Appendix G.4: "Tasks
/// below the band receive one diagnostic rollout with reference-solution
/// hints to guide revision").
#[derive(Debug, Clone, PartialEq)]
pub struct Hint {
    /// The checks the reference solution demonstrates.
    pub demonstrated_checks: Vec<String>,
}

/// One full solver attempt: the world result, the sampled token steps, and
/// the emulated turns.
#[derive(Debug, Clone)]
pub struct AttemptTrace {
    /// The attempt result (reward, CTRF report).
    pub result: AttemptResult,
    /// Sampled actions with stored log-probs (DPPO training data).
    pub steps: Vec<TokenStep>,
    /// The terminal turns of the attempt.
    pub turns: Vec<Turn>,
    /// The archive outcome (reward + verifier stdout + failed checks).
    pub outcome: Outcome,
    /// The world state after this invocation (for carry-over parking).
    pub world_state: Option<crate::world::WorldState>,
}

/// A partial (possibly unfinished) attempt: the trace plus the live
/// sandbox state and the resume position.
#[derive(Debug, Clone)]
pub struct PartialAttempt {
    /// The trace of this invocation.
    pub trace: AttemptTrace,
    /// The next un-attempted check index.
    pub next_check: usize,
    /// Whether the attempt completed.
    pub done: bool,
    /// The live sandbox state (files produced so far) — carried across
    /// updates for unfinished attempts.
    pub world: crate::world::WorldState,
}

/// The solver-side policy interface. The host calibrates environments with
/// it; the proposer calls it through [`SolverModelTool`].
pub trait SolverModel {
    /// Model tag.
    fn solver_id(&self) -> &str;

    /// One attempt, with the full trace.
    fn attempt_with_trace(&mut self, task: &HarborTask, hint: Option<&Hint>) -> AttemptTrace;

    /// One partial attempt bounded by a check budget (the agent-timeout
    /// mapping). The default is a full, unbounded attempt.
    fn attempt_partial(
        &mut self,
        task: &HarborTask,
        hint: Option<&Hint>,
        world: Option<crate::world::WorldState>,
        from_check: usize,
        check_budget: Option<usize>,
    ) -> PartialAttempt;

    /// One attempt, result only.
    fn attempt(&mut self, task: &HarborTask, hint: Option<&Hint>) -> AttemptResult {
        self.attempt_with_trace(task, hint).result
    }
}

/// A tabular, deterministic solver. Capability emerges from the policy
/// bank: as DPPO concentrates each skill's table on `ACTION_SOLVE`, the
/// solver's rewards on tasks exercising that skill rise.
#[derive(Debug, Clone)]
pub struct ScriptedSolver {
    /// Model tag (e.g. "qwen3.6-35b-a3b").
    pub solver_id: String,
    /// Policy bank keyed by skill tag.
    pub bank: PolicyBank,
    /// Deterministic RNG.
    pub rng: StdRng,
}

impl ScriptedSolver {
    /// Create with a seed.
    pub fn new(solver_id: impl Into<String>, seed: u64) -> Self {
        ScriptedSolver {
            solver_id: solver_id.into(),
            bank: PolicyBank::default(),
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl SolverModel for ScriptedSolver {
    fn solver_id(&self) -> &str {
        &self.solver_id
    }

    fn attempt_with_trace(&mut self, task: &HarborTask, hint: Option<&Hint>) -> AttemptTrace {
        self.attempt_partial(task, hint, None, 0, None).trace
    }

    fn attempt_partial(
        &mut self,
        task: &HarborTask,
        hint: Option<&Hint>,
        world_state: Option<crate::world::WorldState>,
        from_check: usize,
        check_budget: Option<usize>,
    ) -> PartialAttempt {
        let (trace, next_check, done) =
            Self::attempt_partial_inner(self, task, hint, world_state, from_check, check_budget);
        let world = trace.world_state.clone().unwrap_or_default();
        PartialAttempt {
            trace,
            next_check,
            done,
            world,
        }
    }
}

impl ScriptedSolver {
    /// Attempt a task starting from check `from_check`, resuming the given
    /// world state (carry-over), bounded by an optional check budget. A
    /// check budget mirrors the agent timeout: long-horizon tasks do not
    /// finish within one update and are carried across updates (Appendix
    /// G.1: "Unfinished terminal trajectories are carried across updates,
    /// preserving their interaction history and live sandbox state. They
    /// resume with the updated policy, while stored sampling
    /// log-probabilities are retained for previously generated tokens").
    ///
    /// Returns the trace of *this* invocation, the next un-attempted check
    /// index, and whether the attempt completed.
    pub(crate) fn attempt_partial_inner(
        &mut self,
        task: &HarborTask,
        hint: Option<&Hint>,
        world_state: Option<crate::world::WorldState>,
        from_check: usize,
        check_budget: Option<usize>,
    ) -> (AttemptTrace, usize, bool) {
        let mut steps = Vec::new();
        let mut turns = if from_check == 0 {
            vec![
                Turn::new(
                    Action::Command("cat /app/instruction.md".into()),
                    Observation {
                        stdout: task.instruction.body.clone(),
                        exit_code: 0,
                    },
                ),
                Turn::new(
                    Action::Command("ls /app && ls /app/data".into()),
                    Observation {
                        stdout: task
                            .environment
                            .files
                            .keys()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("\n"),
                        exit_code: 0,
                    },
                ),
            ]
        } else {
            vec![Turn::new(
                Action::Command(format!(
                    "resume from check {from_check} with preserved workspace"
                )),
                Observation::ok(),
            )]
        };
        let mut world = match world_state {
            Some(state) => crate::world::TaskWorld {
                task: task.clone(),
                state,
                registry: ImageRegistry::default_world(),
                initial_hash: 0,
            },
            None => world::TaskWorld::build(task.clone(), &ImageRegistry::default_world())
                .expect("calibration attempts run against a validated (buildable) task"),
        };

        // One sampled action per remaining check, up to the budget.
        let total = task.tests.checks.len();
        let end = match check_budget {
            Some(budget) => (from_check + budget).min(total),
            None => total,
        };
        let mut synthesized_ops = Vec::new();
        for i in from_check..end {
            let check = &task.tests.checks[i];
            let skill = task
                .skills
                .get(i % task.skills.len().max(1))
                .map(|s| s.name().to_string())
                .unwrap_or_else(|| "general".to_string());
            let (action, logprob) = self.bank.sample(&skill, SOLVER_ACTIONS, &mut self.rng);
            // Hints force the demonstrated checks to a correct solve.
            let effective = match hint {
                Some(h) if h.demonstrated_checks.iter().any(|c| c == check.name()) => ACTION_SOLVE,
                _ => action,
            };
            steps.push(TokenStep {
                context: skill,
                action,
                old_logprob: logprob,
                advantage: 0.0,
            });
            turns.push(Turn::new(
                Action::Command(format!(
                    "python3 - <<'PY'\n# attempt check {} ({})\nPY",
                    check.name(),
                    action_name(effective)
                )),
                Observation::ok(),
            ));
            if passes_with(effective, check) {
                synthesized_ops.push(synthesize_op(i, check));
            }
        }
        for op in synthesized_ops {
            world.run_script(&crate::world::SolutionScript {
                ops: vec![op],
                executable: true,
            });
        }
        let next_check = end;
        let done = next_check >= total;
        let report = world.run_tests();
        let result = AttemptResult {
            exit_code: 0,
            report: report.clone(),
            reward: report.reward(),
        };
        if done {
            turns.push(Turn::new(
                Action::Command("harbor verify --phase agent".into()),
                Observation {
                    stdout: task.tests.verifier_stdout(&report),
                    exit_code: 0,
                },
            ));
        }
        let outcome = if done {
            world::attempt_outcome(task, &result)
        } else {
            Outcome {
                reward: 0.0,
                verifier_stdout: String::new(),
                passed_tests: report.passed_count(),
                total_tests: report.tests.len(),
                failed_checks: Vec::new(),
            }
        };
        (
            AttemptTrace {
                result,
                steps,
                turns,
                outcome,
                world_state: Some(world.state.clone()),
            },
            next_check,
            done,
        )
    }
}

/// Name of a solver action for turn rendering.
fn action_name(a: usize) -> &'static str {
    match a {
        ACTION_SOLVE => "solve",
        ACTION_SLOPPY => "sloppy",
        _ => "skip",
    }
}

/// Whether `action` passes `check` under the deterministic semantics.
fn passes_with(action: usize, check: &TestCheck) -> bool {
    match action {
        ACTION_SOLVE => true,
        ACTION_SKIP => false,
        _ => {
            // Sloppy passes the coarse checks, fails the precision checks.
            !matches!(
                check,
                TestCheck::FileJsonKey { .. } | TestCheck::FileLineCount { .. }
            )
        }
    }
}

/// The solution op that satisfies check `i` (used to synthesize the
/// solver's partial solution).
fn synthesize_op(i: usize, check: &TestCheck) -> ShellOp {
    match check {
        TestCheck::FileExists { path, .. } => ShellOp::WriteFile {
            path: path.clone(),
            content: "done\n".into(),
        },
        TestCheck::FileContains { path, needle, .. } => ShellOp::WriteFile {
            path: path.clone(),
            content: format!("{needle}\n"),
        },
        TestCheck::FileJsonKey {
            path,
            key,
            expected,
            ..
        } => ShellOp::WriteFile {
            path: path.clone(),
            content: format!("{{\"{key}\": {expected}}}"),
        },
        TestCheck::FileLineCount { path, count, .. } => {
            let content: String = "row\n".repeat(*count);
            ShellOp::WriteFile {
                path: path.clone(),
                content,
            }
        }
        TestCheck::CommandSucceeds { .. } => ShellOp::WriteFile {
            path: format!("/app/step_{i}.py"),
            content: "print('ok')\nexit 0\n".into(),
        },
    }
}

/// The `solver_model` tool: the proposer's pre-submission calibration
/// ("using a solver_model tool similar to the ones in AutoData and
/// GPT-Red").
pub struct SolverModelTool<'a> {
    /// The wrapped solver.
    pub solver: &'a mut dyn SolverModel,
}

impl<'a> SolverModelTool<'a> {
    /// Wrap a solver.
    pub fn new(solver: &'a mut dyn SolverModel) -> Self {
        SolverModelTool { solver }
    }

    /// Run `n` attempts of the current solver on a task and summarize the
    /// rewards.
    pub fn evaluate(&mut self, task: &HarborTask, n: usize) -> SolverModelSample {
        let mut rewards = Vec::new();
        for _ in 0..n {
            rewards.push(self.solver.attempt(task, None).reward);
        }
        SolverModelSample::from_rewards(rewards)
    }
}

/// A summary of solver_model attempts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverModelSample {
    /// Per-attempt rewards.
    pub rewards: Vec<f64>,
    /// Mean reward.
    pub mean: f64,
    /// Population standard deviation.
    pub std: f64,
}

impl SolverModelSample {
    /// Summarize a raw reward vector.
    pub fn from_rewards(rewards: Vec<f64>) -> Self {
        let mean = if rewards.is_empty() {
            0.0
        } else {
            rewards.iter().sum::<f64>() / rewards.len() as f64
        };
        let var = if rewards.is_empty() {
            0.0
        } else {
            rewards.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / rewards.len() as f64
        };
        SolverModelSample {
            rewards,
            mean,
            std: var.sqrt(),
        }
    }
}

/// Proposer design-move vocabulary: one token per episode, keyed by the
/// assignment direction.
pub const PROPOSER_ACTIONS: usize = 3;
/// Move 0: conservative (smaller, easier variant).
pub const MOVE_CONSERVATIVE: usize = 0;
/// Move 1: balanced (variant matching the direction's default sizing).
pub const MOVE_BALANCED: usize = 1;
/// Move 2: ambitious (bigger, riskier variant).
pub const MOVE_AMBITIOUS: usize = 2;

/// Parameters for generated tasks (the design move's realization).
#[derive(Debug, Clone, PartialEq)]
pub struct TaskGenParams {
    /// Env id for the new task.
    pub env_id: String,
    /// Harbor task name.
    pub name: String,
    /// Number of verifier checks.
    pub n_checks: usize,
    /// Skills the task exercises.
    pub skills: Vec<SkillTag>,
    /// Difficulty metadata.
    pub difficulty: Difficulty,
    /// Deliberately ship an incomplete reference solution (drives the
    /// proposer's own revision loop).
    pub broken: bool,
}

/// Generate a complete, self-consistent Harbor task from parameters: the
/// instruction, the planted data, the reference solution, and a verifier
/// suite whose checks diagnose real stages (the skill's "meaningful pytest
/// checks that diagnose real stages" — coarse and precision checks mixed).
pub fn generate_task(params: &TaskGenParams) -> HarborTask {
    let n = params.n_checks.max(1);
    let mut checks = Vec::new();
    for i in 0..n {
        if i == n - 1 {
            // The final milestone is a precision check on the summary.
            checks.push(TestCheck::FileJsonKey {
                name: format!("test_summary_json_{i}"),
                path: "/app/out/summary.json".into(),
                key: "records".into(),
                expected: n.to_string(),
            });
        } else if i % 2 == 0 {
            checks.push(TestCheck::FileExists {
                name: format!("test_out_{i}_exists"),
                path: format!("/app/out/out_{i}.txt"),
            });
        } else {
            checks.push(TestCheck::FileContains {
                name: format!("test_out_{i}_content"),
                path: format!("/app/out/out_{i}.txt"),
                needle: format!("row-{i}"),
            });
        }
    }

    // The reference solution writes every artifact.
    let mut ops = Vec::new();
    ops.push(ShellOp::MkDir("/app/out".into()));
    for (i, check) in checks.iter().enumerate() {
        let op = match check {
            TestCheck::FileJsonKey { .. } => None,
            TestCheck::FileExists { path, .. } => Some(ShellOp::WriteFile {
                path: path.clone(),
                content: format!("row-{i}\n"),
            }),
            TestCheck::FileContains { path, needle, .. } => Some(ShellOp::WriteFile {
                path: path.clone(),
                content: format!("{needle}\n"),
            }),
            _ => None,
        };
        if let Some(o) = op {
            ops.push(o);
        }
    }
    ops.push(ShellOp::WriteFile {
        path: "/app/out/summary.json".into(),
        content: format!("{{\"records\": {n}}}"),
    });
    if params.broken {
        // The broken variant forgets the summary (oracle reward < 1).
        ops.pop();
    }

    let mut files = BTreeMap::new();
    files.insert(
        "/app/data/input.csv".to_string(),
        (0..n)
            .map(|i| format!("id,stage\n{i},{i}"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    );

    // The instruction enumerates every expected output path explicitly
    // (the rubric's instruction_clarity criterion: "Specify expected
    // outputs — paths, formats, content").
    let mut outputs = String::new();
    for (i, check) in checks.iter().enumerate() {
        match check {
            TestCheck::FileJsonKey {
                path,
                key,
                expected,
                ..
            } => {
                outputs.push_str(&format!(
                    "- `{path}`: a JSON object with key `{key}` equal to {expected}.\n"
                ));
            }
            TestCheck::FileExists { path, .. } => {
                outputs.push_str(&format!(
                    "- `{path}`: exists, containing the single line `row-{i}`.\n"
                ));
            }
            TestCheck::FileContains { path, needle, .. } => {
                outputs.push_str(&format!(
                    "- `{path}`: exists, containing the exact text `{needle}`.\n"
                ));
            }
            _ => {}
        }
    }
    let instruction_body = format!(
        "# Process the staged records\n\nRead `/app/data/input.csv` (columns `id,stage`) and produce the following outputs:\n\n{outputs}\nPython 3 and its standard library are available and are all you need. The box is offline. Leave the input file exactly as it is.\n"
    );

    let keywords: Vec<String> = params
        .skills
        .iter()
        .map(|s| s.name().to_string())
        .chain(["pytest".to_string()])
        .collect();

    let task = HarborTask {
        env_id: params.env_id.clone(),
        meta: TaskMeta {
            name: params.name.clone(),
            version: "1.0.0".into(),
            description: format!("process {} staged records", params.n_checks),
            keywords,
        },
        instruction: Instruction::new(instruction_body),
        environment: TaskEnvironment {
            base_image: "python:3.12-slim".into(),
            packages: vec!["python3".into()],
            files,
            workdir: "/app".into(),
            network_mode: NetworkMode::NoNetwork,
            cpus: 1,
            memory_mb: 2048,
            storage_mb: 10240,
            gpus: None,
            gpu_types: None,
            extra_allowed_hosts: Vec::new(),
        },
        solution: crate::world::SolutionScript { ops, executable: true },
        tests: TestSuite {
            test_sh: "#!/bin/bash\nuvx --with pytest==8.4.1 pytest /tests/test_outputs.py\n".into(),
            checks,
            reward_file: RewardFile::Txt,
        },
        readme: format!(
            "# {}\n\nThe agent processes `/app/data/input.csv` into per-stage outputs under `/app/out` and a summary JSON.\n\n## Environment\n\npython:3.12-slim with python3; offline (network_mode = no-network).\n\n## Verifier\n\npytest fraction via CTRF: {} checks (existence, exact content, summary JSON).\n\n## Running\n\nharbor run -p . -a oracle\n",
            params.name, params.n_checks
        ),
        steps: Vec::new(),
        toml: TaskToml {
            schema_version: "1.0".into(),
            task: TaskMeta {
                name: params.name.clone(),
                version: "1.0.0".into(),
                description: format!("process {} staged records", params.n_checks),
                keywords: params.skills.iter().map(|s| s.name().to_string()).chain(["pytest".to_string()]).collect(),
            },
            metadata: TaskMetadata { difficulty: params.difficulty, category: "programming".into(), tags: vec![] },
            agent: AgentConfig::default(),
            verifier: VerifierConfig { timeout_sec: 600.0, environment_mode: VerifierMode::Shared, env: BTreeMap::new(), network: Default::default() },
            reward_strategy: RewardStrategy::Mean,
        },
        skills: params.skills.clone(),
    };
    task
}

/// Sizing derived from the assignment direction and the sampled design
/// move.
pub fn size_for(
    direction: &Direction,
    parent_checks: Option<usize>,
    design_move: usize,
) -> TaskGenParamsBuilder {
    let base = parent_checks.unwrap_or(4);
    let (n, difficulty) = match direction {
        Direction::Simplify { reduce_checks_by } => {
            let n = base.saturating_sub(*reduce_checks_by as usize) + design_move.saturating_sub(1);
            (n.max(2), Difficulty::Easy)
        }
        Direction::Harden { add_checks, .. } => {
            let n = base + *add_checks as usize + design_move;
            (n, Difficulty::Hard)
        }
        Direction::Fresh => (4 + design_move, Difficulty::Medium),
    };
    TaskGenParamsBuilder {
        n_checks: n,
        difficulty,
    }
}

/// Intermediate sizing returned by [`size_for`].
#[derive(Debug, Clone, PartialEq)]
pub struct TaskGenParamsBuilder {
    /// Check count.
    pub n_checks: usize,
    /// Difficulty.
    pub difficulty: Difficulty,
}

/// The result of one proposer episode.
#[derive(Debug, Clone)]
pub struct ProposedEpisode {
    /// The (possibly revised) task the proposer ships.
    pub task: HarborTask,
    /// Terminal turns of the episode (Figure 22's episode depth).
    pub turns: Vec<Turn>,
    /// The design-move token step (DPPO training data).
    pub steps: Vec<TokenStep>,
    /// The final pre-submission self-check.
    pub self_check: ProposerSelfCheck,
    /// Whether the proposer actually submitted (oracle-passes ∧
    /// no-op-fails; otherwise it keeps the task for revision feedback).
    pub submitted: bool,
    /// Whether a web tool was used.
    pub used_web: bool,
    /// solver_model pre-calibration sample, when used.
    pub pre_calibration: Option<SolverModelSample>,
    /// How many revision rounds the episode consumed.
    pub revisions_used: u32,
}

/// Everything a proposer episode needs from the host: the workspace, the
/// sandbox, the build registry, the revision budget, and the `solver_model`
/// the proposer may call before submitting.
pub struct EpisodeContext<'a> {
    /// The proposer's workspace.
    pub workspace: &'a mut Workspace,
    /// The episode's sandbox.
    pub sandbox: &'a mut crate::sandbox::Sandbox,
    /// The build registry.
    pub registry: &'a ImageRegistry,
    /// Maximum revisions the episode may consume.
    pub max_revisions: u32,
    /// The current solver, exposed through the `solver_model` tool.
    pub use_solver_model: &'a mut dyn SolverModel,
}

/// A tabular, deterministic proposer. Grounded in the assignment: it reads
/// the workspace archive, adapts the parent's sizing to the direction, and
/// validates before submitting.
#[derive(Debug, Clone)]
pub struct ScriptedProposer {
    /// Model tag.
    pub proposer_id: String,
    /// Policy bank keyed by direction tag.
    pub bank: PolicyBank,
    /// Deterministic RNG.
    pub rng: StdRng,
}

impl ScriptedProposer {
    /// Create with a seed.
    pub fn new(proposer_id: impl Into<String>, seed: u64) -> Self {
        ScriptedProposer {
            proposer_id: proposer_id.into(),
            bank: PolicyBank::default(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Run one proposer episode: read → design → build → validate → (revise)
    /// → submit. `max_revisions` follows Appendix G.4 ("each assignment
    /// permits up to two revisions"); the episode context carries the
    /// workspace, sandbox, registry, and the `solver_model` used for the
    /// pre-submission calibration.
    pub fn propose(
        &mut self,
        assignment: &Assignment,
        parent: Option<&HarborTask>,
        ctx: &mut EpisodeContext<'_>,
    ) -> Result<ProposedEpisode> {
        let EpisodeContext {
            workspace,
            sandbox,
            registry,
            max_revisions,
            use_solver_model,
        } = ctx;
        let mut turns = Vec::new();
        let mut steps = Vec::new();

        // --- Read: ground the task in real signal (the skill's mandate).
        let index = workspace
            .read("memory/past_trajectories/index.jsonl")
            .unwrap_or("")
            .to_string();
        turns.push(Turn::new(
            Action::Command("cat memory/past_trajectories/index.jsonl".into()),
            Observation {
                stdout: index.clone(),
                exit_code: 0,
            },
        ));
        if let Some(parent) = parent {
            let notes = workspace
                .read(&format!(
                    "memory/past_environments/{}/notes.md",
                    parent.env_id
                ))
                .unwrap_or("");
            turns.push(Turn::new(
                Action::Command(format!(
                    "cat memory/past_environments/{}/notes.md",
                    parent.env_id
                )),
                Observation {
                    stdout: notes.to_string(),
                    exit_code: 0,
                },
            ));
        }
        let assignment_text = workspace
            .read("assignment/current.md")
            .unwrap_or("")
            .to_string();
        turns.push(Turn::new(
            Action::Command("cat assignment/current.md".into()),
            Observation {
                stdout: assignment_text,
                exit_code: 0,
            },
        ));

        // --- Web: only when the runtime preamble lists a web tool.
        let mut used_web = false;
        if crate::skills::web_tools_allowed(&sandbox.available_tools) {
            sandbox.check_tool("web_search")?;
            sandbox.check_egress("www.google.com")?;
            used_web = true;
            turns.push(Turn::new(
                Action::Tool(crate::trajectory::ToolCall {
                    name: "web_search".into(),
                    args: format!("real problems: {}", assignment.direction.tag()),
                }),
                Observation {
                    stdout: "3 results cached into cookbook/".into(),
                    exit_code: 0,
                },
            ));
        }

        // --- Design: sample the design move from the direction's policy.
        let (design_move, logprob) =
            self.bank
                .sample(assignment.direction.tag(), PROPOSER_ACTIONS, &mut self.rng);
        steps.push(TokenStep {
            context: assignment.direction.tag().to_string(),
            action: design_move,
            old_logprob: logprob,
            advantage: 0.0,
        });

        // --- Build: scaffold then write the canonical task.
        turns.push(Turn::new(
            Action::Command(format!(
                "harbor task init flywheel/{}",
                assignment.target_env_id
            )),
            Observation {
                stdout: "scaffolded".into(),
                exit_code: 0,
            },
        ));
        let mut rng = StdRng::seed_from_u64(self.rng.gen::<u64>());
        let mut params = self.params_for(assignment, parent, design_move, &mut rng);
        let mut task = generate_task(&params);
        workspace.write_task(&task)?;
        turns.push(Turn::new(
            Action::Command(format!(
                "write output/tasks/{}/... ({} checks)",
                assignment.target_env_id,
                task.total_checks()
            )),
            Observation::ok(),
        ));

        // --- Validate: oracle + no-op swap dance, then optionally
        //     solver_model, with revisions.
        let mut revisions = 0u32;
        let mut self_check = proposer_self_check(&task, registry)?;
        let mut pre_calibration: Option<SolverModelSample> = None;
        loop {
            turns.push(Turn::new(
                Action::Command(format!(
                    "python3 /opt/tools/validate.py output/tasks/{} oracle",
                    assignment.target_env_id
                )),
                Observation {
                    stdout: format!("{:.3}", self_check.oracle_reward),
                    exit_code: if self_check.oracle_passed { 0 } else { 1 },
                },
            ));
            turns.push(Turn::new(
                Action::Command("printf '#!/bin/sh\\nexit 0\\n' > solution/solve.sh && python3 /opt/tools/validate.py ... oracle".into()),
                Observation { stdout: format!("{:.3}", self_check.noop_reward), exit_code: if self_check.noop_failed { 0 } else { 1 } },
            ));

            if self_check.ready_to_submit() {
                break;
            }
            if revisions >= *max_revisions {
                break;
            }
            // Revise: repair the reference solution (the common failure).
            revisions += 1;
            params.broken = false;
            task = generate_task(&params);
            workspace.write_task(&task)?;
            self_check = proposer_self_check(&task, registry)?;
            turns.push(Turn::new(
                Action::Command(format!(
                    "revise #{revisions}: fix solution/solve.sh ({} checks)",
                    task.total_checks()
                )),
                Observation::ok(),
            ));
        }

        // --- Pre-calibration with the solver_model tool (optional).
        if self_check.ready_to_submit() && use_solver_model_enabled(assignment) {
            let mut tool = SolverModelTool::new(*use_solver_model);
            let sample = tool.evaluate(&task, 4);
            turns.push(Turn::new(
                Action::Tool(crate::trajectory::ToolCall {
                    name: "solver_model".into(),
                    args: format!("evaluate {}", assignment.target_env_id),
                }),
                Observation {
                    stdout: format!("mean={:.3} std={:.3}", sample.mean, sample.std),
                    exit_code: 0,
                },
            ));
            pre_calibration = Some(sample);
        }

        sandbox.tick(300.0)?;
        let submitted = self_check.ready_to_submit();
        Ok(ProposedEpisode {
            task,
            turns,
            steps,
            self_check,
            submitted,
            used_web,
            pre_calibration,
            revisions_used: revisions,
        })
    }

    /// Task parameters for an assignment + design move.
    fn params_for(
        &self,
        assignment: &Assignment,
        parent: Option<&HarborTask>,
        design_move: usize,
        rng: &mut StdRng,
    ) -> TaskGenParams {
        let builder = size_for(
            &assignment.direction,
            parent.map(|p| p.total_checks()),
            design_move,
        );
        let mut skills: Vec<SkillTag> = parent
            .map(|p| p.skills.clone())
            .unwrap_or_else(|| vec![SkillTag::new("csv"), SkillTag::new("json")]);
        if let Direction::Harden {
            additional_skill: Some(extra),
            ..
        } = &assignment.direction
        {
            if !skills.contains(extra) {
                skills.push(extra.clone());
            }
        }
        if skills.is_empty() {
            skills.push(SkillTag::new("python"));
        }
        // The ambitious move sometimes ships a broken reference solution —
        // caught by its own validation and repaired in revision.
        let broken = design_move == MOVE_AMBITIOUS && rng.gen::<f64>() < 0.3;
        TaskGenParams {
            env_id: assignment.target_env_id.clone(),
            name: format!("flywheel/{}", assignment.target_env_id),
            n_checks: builder.n_checks,
            skills,
            difficulty: builder.difficulty,
            broken,
        }
    }
}

/// Whether the proposer's policy configuration enables solver_model
/// pre-calibration for this assignment. Enabled for harden/fresh
/// assignments (difficulty estimation matters most there).
fn use_solver_model_enabled(assignment: &Assignment) -> bool {
    !matches!(assignment.direction, Direction::Simplify { .. })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assignment::{build_assignment, ParentRef};
    use crate::sandbox::{Sandbox, SandboxResources};
    use crate::trajectory::RolloutRecord;

    fn solver() -> ScriptedSolver {
        ScriptedSolver::new("qwen-test", 42)
    }

    fn sample_task() -> HarborTask {
        generate_task(&TaskGenParams {
            env_id: "env_001".into(),
            name: "flywheel/env_001".into(),
            n_checks: 4,
            skills: vec![SkillTag::new("csv"), SkillTag::new("json")],
            difficulty: Difficulty::Medium,
            broken: false,
        })
    }

    #[test]
    fn generated_task_is_valid_and_oracle_passes() {
        let t = sample_task();
        assert_eq!(t.total_checks(), 4);
        let run = world::oracle_run(&t, &ImageRegistry::default_world()).unwrap();
        assert_eq!(run.reward, 1.0);
        let noop = world::noop_run(&t, &ImageRegistry::default_world()).unwrap();
        assert!(noop.reward < 0.5);
        assert!(t.lint().is_empty());
    }

    #[test]
    fn broken_task_fails_oracle_until_repaired() {
        let mut params = TaskGenParams {
            env_id: "env_x".into(),
            name: "flywheel/env_x".into(),
            n_checks: 3,
            skills: vec![SkillTag::new("csv")],
            difficulty: Difficulty::Medium,
            broken: true,
        };
        let broken = generate_task(&params);
        let run = world::oracle_run(&broken, &ImageRegistry::default_world()).unwrap();
        assert!(run.reward < 1.0);
        params.broken = false;
        let fixed = generate_task(&params);
        assert_eq!(
            world::oracle_run(&fixed, &ImageRegistry::default_world())
                .unwrap()
                .reward,
            1.0
        );
    }

    #[test]
    fn solver_attempts_have_spread_and_traces() {
        let t = sample_task();
        let mut s = solver();
        let mut rewards = Vec::new();
        for _ in 0..12 {
            let trace = s.attempt_with_trace(&t, None);
            rewards.push(trace.result.reward);
            assert_eq!(trace.steps.len(), 4);
            assert!(!trace.turns.is_empty());
            assert!(trace.outcome.total_tests == 4);
        }
        // Uniform policy over 3 strategies: rewards should vary.
        assert!(rewards.iter().any(|r| *r < 1.0));
        assert!(rewards.iter().any(|r| *r > 0.0));
    }

    #[test]
    fn hint_forces_demonstrated_checks() {
        let t = sample_task();
        let mut s = solver();
        // One attempt first so the skill tables exist, then skew to always
        // skip.
        let _ = s.attempt(&t, None);
        for policy in s.bank.tables.values_mut() {
            policy.logits = vec![-20.0, 0.0, 20.0];
        }
        let bare = s.attempt(&t, None);
        assert_eq!(bare.reward, 0.0);
        let hinted = s.attempt(
            &t,
            Some(&Hint {
                demonstrated_checks: t.check_names(),
            }),
        );
        assert_eq!(hinted.reward, 1.0);
    }

    #[test]
    fn solver_model_tool_summarizes() {
        let t = sample_task();
        let mut s = solver();
        let mut tool = SolverModelTool::new(&mut s);
        let sample = tool.evaluate(&t, 4);
        assert_eq!(sample.rewards.len(), 4);
        assert!(sample.std >= 0.0);
    }

    #[test]
    fn proposer_episode_full_flow() {
        let parent = sample_task();
        let rollouts = vec![RolloutRecord {
            env_id: "env_001".into(),
            solver: "qwen-test".into(),
            round: 1,
            provenance: crate::trajectory::Provenance::SelfRound(1),
            rewards: vec![1.0, 1.0, 0.75],
            reward_std: 0.14,
            check_failures: Default::default(),
            path: "memory/past_trajectories/successful/env_001.json".into(),
        }];
        let assignment = build_assignment(
            "asg_001",
            "env_010",
            ParentRef {
                env_id: "env_001".into(),
                checks: parent.total_checks(),
                skills: parent.skills.clone(),
            },
            &rollouts,
            (0.25, 0.75),
        );
        assert!(matches!(assignment.direction, Direction::Harden { .. }));

        let mut ws = Workspace::new();
        ws.set_assignment(&assignment.render());
        ws.refresh_from_pool(rollouts, vec![], &[&parent]);
        let mut sandbox = Sandbox::new(
            crate::sandbox::SandboxResources::proposer(crate::sandbox::CodingAgent::Pi),
            vec!["bash".into()],
        );
        let mut proposer = ScriptedProposer::new("gpt-test", 7);
        let mut solver = solver();

        let mut ctx = EpisodeContext {
            workspace: &mut ws,
            sandbox: &mut sandbox,
            registry: &ImageRegistry::default_world(),
            max_revisions: 2,
            use_solver_model: &mut solver,
        };
        let episode = proposer
            .propose(&assignment, Some(&parent), &mut ctx)
            .unwrap();
        assert!(episode.submitted, "self-check: {:?}", episode.self_check);
        assert!(episode.self_check.oracle_passed);
        assert!(episode.self_check.noop_failed);
        assert_eq!(episode.steps.len(), 1);
        assert_eq!(episode.steps[0].context, "harden");
        // The generated task is bigger than the parent (harder variant).
        assert!(episode.task.total_checks() > parent.total_checks());
        // Workspace carries the canonical output tree.
        assert!(ws.files.contains_key(&format!(
            "output/tasks/{}/instruction.md",
            assignment.target_env_id
        )));
        // Harden assignments invoke solver_model.
        assert!(episode.pre_calibration.is_some());
        assert!(!episode.used_web);
    }

    #[test]
    fn proposer_uses_web_only_when_preamble_lists_it() {
        let parent = sample_task();
        let assignment = Assignment {
            assignment_id: "asg_002".into(),
            target_env_id: "env_011".into(),
            parent: Some(ParentRef {
                env_id: "env_001".into(),
                checks: 4,
                skills: parent.skills.clone(),
            }),
            parent_pass_rate: Some(0.1),
            most_common_failure: None,
            direction: Direction::Simplify {
                reduce_checks_by: 2,
            },
        };
        let mut ws = Workspace::new();
        ws.set_assignment(&assignment.render());
        let mut sandbox_with_web = Sandbox::new(
            SandboxResources::proposer(crate::sandbox::CodingAgent::Pi),
            vec!["bash".into(), "web_search".into()],
        );
        let mut sandbox_no_web = Sandbox::new(
            SandboxResources::proposer(crate::sandbox::CodingAgent::Pi),
            vec!["bash".into()],
        );
        let mut proposer = ScriptedProposer::new("p", 3);
        let mut solver = solver();

        let mut ctx_web = EpisodeContext {
            workspace: &mut ws,
            sandbox: &mut sandbox_with_web,
            registry: &ImageRegistry::default_world(),
            max_revisions: 2,
            use_solver_model: &mut solver,
        };
        let ep = proposer
            .propose(&assignment, Some(&parent), &mut ctx_web)
            .unwrap();
        assert!(ep.used_web);
        let mut ctx_no_web = EpisodeContext {
            workspace: &mut ws,
            sandbox: &mut sandbox_no_web,
            registry: &ImageRegistry::default_world(),
            max_revisions: 2,
            use_solver_model: &mut solver,
        };
        let ep2 = proposer
            .propose(&assignment, Some(&parent), &mut ctx_no_web)
            .unwrap();
        assert!(!ep2.used_web);
    }

    #[test]
    fn sizing_follows_direction() {
        let simplify = size_for(
            &Direction::Simplify {
                reduce_checks_by: 2,
            },
            Some(6),
            MOVE_BALANCED,
        );
        assert_eq!(simplify.n_checks, 4);
        assert_eq!(simplify.difficulty, Difficulty::Easy);

        let harden = size_for(
            &Direction::Harden {
                add_checks: 2,
                additional_skill: None,
            },
            Some(4),
            MOVE_AMBITIOUS,
        );
        assert_eq!(harden.n_checks, 8);
        assert_eq!(harden.difficulty, Difficulty::Hard);

        let fresh = size_for(&Direction::Fresh, None, MOVE_CONSERVATIVE);
        assert_eq!(fresh.n_checks, 4);
    }
}
