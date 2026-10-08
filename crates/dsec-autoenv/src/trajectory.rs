//! Trajectories of terminal agents, the two roles, and the rollout archive.
//!
//! Both roles act in a terminal in the same way (Section 3.3): given a prompt
//! `x` and a container `C`, the policy issues a command or tool call
//! `a_t ~ pi_theta(. | x, a_<t, o_<t)` and reads its output `o_t` from `C`,
//! producing a trajectory `tau = (x, a_1, o_1, ..., a_T, o_T)`. "The two roles
//! differ only in the prompt, the container, and how the trajectory is
//! scored."

use serde::{Deserialize, Serialize};

/// The two flywheel roles. Rewards, advantage baselines, and training
/// cadence are all separated by role (Section 3.3, Appendix G.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    /// The environment proposer: builds new RL environments inside a sandbox
    /// and is scored by [`crate::reward::proposer_reward`] (Eq. 2).
    Proposer,
    /// The solver: attempts environments and is scored by
    /// [`crate::reward::solver_reward`] (Eq. 1, fraction of tests passed).
    Solver,
}

impl Role {
    /// Lowercase role tag used in serialized records.
    pub fn tag(&self) -> &'static str {
        match self {
            Role::Proposer => "proposer",
            Role::Solver => "solver",
        }
    }
}

/// A tool call issued by an agent. Tool surface visible in the paper:
/// `bash`, `web_search` / `web_fetch`, `solver_model`, `ask_human`, and the
/// Harbor CLI (`harbor task init`, `harbor run`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool name, e.g. `bash`, `solver_model`, `ask_human`.
    pub name: String,
    /// Free-form arguments (command line, query text, task path).
    pub args: String,
}

/// One action of a terminal agent: either a shell command or a tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Action {
    /// A shell command typed into the terminal.
    Command(String),
    /// A structured tool call.
    Tool(ToolCall),
}

impl Action {
    /// Short label used in metrics and episode renderings.
    pub fn label(&self) -> String {
        match self {
            Action::Command(c) => {
                let first = c.split_whitespace().next().unwrap_or("");
                format!("bash:{first}")
            }
            Action::Tool(t) => format!("tool:{}", t.name),
        }
    }

    /// The raw text of the action (command or tool argument).
    pub fn text(&self) -> &str {
        match self {
            Action::Command(c) => c,
            Action::Tool(t) => &t.args,
        }
    }
}

/// The observation an agent reads back from the container.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    /// Standard output captured from the command or tool call.
    pub stdout: String,
    /// Exit status (`0` = success).
    pub exit_code: i32,
}

impl Observation {
    /// A successful, silent observation.
    pub fn ok() -> Self {
        Observation {
            stdout: String::new(),
            exit_code: 0,
        }
    }

    /// A failed observation with the given stderr-as-stdout text.
    pub fn fail(stdout: impl Into<String>) -> Self {
        Observation {
            stdout: stdout.into(),
            exit_code: 1,
        }
    }
}

/// One turn: an action and its observation. Turn counts are the paper's
/// episode-depth metric (Figure 22: "A turn is a public text step or tool
/// call. Native thinking is excluded.").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    /// The action taken.
    pub action: Action,
    /// What the container returned.
    pub observation: Observation,
}

impl Turn {
    /// Construct a turn.
    pub fn new(action: Action, observation: Observation) -> Self {
        Turn {
            action,
            observation,
        }
    }

    /// Classify a turn for the Read/Build/Write/Test plots of Figure 8.
    ///
    /// The paper hand-labels proposer episodes into Read / Build / Write /
    /// Test actions; we use the same four buckets with a keyword classifier
    /// in the spirit of (and with the caveats of) Appendix I.1.
    pub fn phase(&self) -> TurnPhase {
        let text = match &self.action {
            Action::Command(c) => c.to_ascii_lowercase(),
            Action::Tool(t) => format!("{} {}", t.name, t.args).to_ascii_lowercase(),
        };
        if text.contains("harbor task init")
            || text.starts_with("docker build")
            || text.contains("docker build")
            || text.contains("build ")
        {
            TurnPhase::Build
        } else if text.contains("validate")
            || text.contains("pytest")
            || text.contains(" test")
            || text.contains("test.sh")
        {
            if text.contains("cat ") || text.contains("read") {
                TurnPhase::Read
            } else {
                TurnPhase::Test
            }
        } else if text.contains("cat ")
            || text.contains("ls")
            || text.contains("read")
            || text.contains("index.jsonl")
        {
            TurnPhase::Read
        } else {
            TurnPhase::Write
        }
    }
}

/// Phase buckets for episode analysis (Figure 8's Read/Build/Write/Test).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TurnPhase {
    /// Inspecting workspace state: rollouts, memory, past environments.
    Read,
    /// Scaffolding and building: `harbor task init`, image builds.
    Build,
    /// Writing task files: instruction, Dockerfile, solution, tests.
    Write,
    /// Running validation, solutions, and verifiers.
    Test,
}

/// Where a stored trajectory came from. Rounds generated by the policy being
/// trained carry `Self` provenance; trajectories distilled from frontier
/// models (Cold-Start, Appendix H) carry the model name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "provenance", content = "source", rename_all = "snake_case")]
pub enum Provenance {
    /// Produced by the flywheel itself in the given round.
    SelfRound(u64),
    /// Produced by a frontier model (e.g. "Claude Opus 5").
    Frontier(String),
}

/// The verifier outcome attached to a completed attempt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    /// Reward in `[0, 1]`: for the solver, the fraction of the environment's
    /// tests passed (Eq. 1).
    pub reward: f64,
    /// Captured verifier stdout, used to extract failure signatures for
    /// assignments (the skill: "read ... `outcome.verifier_stdout`").
    pub verifier_stdout: String,
    /// Number of individual tests that passed.
    pub passed_tests: usize,
    /// Total number of tests.
    pub total_tests: usize,
    /// Names of the checks that failed, in order.
    pub failed_checks: Vec<String>,
}

impl Outcome {
    /// An outcome for a fully passing attempt.
    pub fn full_pass(total: usize) -> Self {
        Outcome {
            reward: 1.0,
            verifier_stdout: format!("{total} passed, 0 failed"),
            passed_tests: total,
            total_tests: total,
            failed_checks: Vec::new(),
        }
    }
}

/// A full agent trajectory: prompt, turns, and scored outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trajectory {
    /// Which role produced this trajectory.
    pub role: Role,
    /// The prompt the episode started from (an environment instruction for
    /// the solver, an assignment for the proposer).
    pub prompt: String,
    /// The turns of the episode.
    pub turns: Vec<Turn>,
    /// Verifier outcome (None while the episode is unfinished — see
    /// [`crate::dppo::CarryOverBuffer`]).
    pub outcome: Option<Outcome>,
}

impl Trajectory {
    /// Number of public turns (Figure 22's episode-depth metric).
    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    /// The episode's reward, if it completed.
    pub fn reward(&self) -> Option<f64> {
        self.outcome.as_ref().map(|o| o.reward)
    }

    /// Phase histogram over turns (used by the Figure-8 style plots).
    pub fn phase_counts(&self) -> [usize; 4] {
        let mut counts = [0usize; 4];
        for t in &self.turns {
            let idx = match t.phase() {
                TurnPhase::Read => 0,
                TurnPhase::Build => 1,
                TurnPhase::Write => 2,
                TurnPhase::Test => 3,
            };
            counts[idx] += 1;
        }
        counts
    }
}

/// One row of `past_trajectories/index.jsonl` (Appendix E.2: "use shell /
/// tool calls to find the latest `provenance:\"self\"` round and inspect the
/// solver's `reward_std` plus raw `rewards`").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RolloutRecord {
    /// Environment the attempt ran in.
    pub env_id: String,
    /// Solver that produced it (model tag).
    pub solver: String,
    /// Round of the flywheel.
    pub round: u64,
    /// Where this trajectory came from.
    pub provenance: Provenance,
    /// Per-attempt rewards on this environment.
    pub rewards: Vec<f64>,
    /// Standard deviation of `rewards` (the primary calibration signal the
    /// proposer reads).
    pub reward_std: f64,
    /// Per-check failure counts: how many attempts failed each check
    /// (aggregated by the recorder from verifier stdout).
    pub check_failures: std::collections::BTreeMap<String, usize>,
    /// Path of the full trajectory inside the archive.
    pub path: String,
}

impl RolloutRecord {
    /// Mean of the recorded rewards.
    pub fn reward_mean(&self) -> f64 {
        if self.rewards.is_empty() {
            return 0.0;
        }
        self.rewards.iter().sum::<f64>() / self.rewards.len() as f64
    }

    /// Bucket for the archive layout: `past_trajectories/failed/` vs
    /// `past_trajectories/successful/` (Figure 18).
    pub fn bucket(&self) -> &'static str {
        if self.reward_mean() >= 0.5 {
            "successful"
        } else {
            "failed"
        }
    }
}

/// A group of trajectories sampled for one training unit, with their rewards.
///
/// Solver groups share one environment (16 trajectories, Appendix G.1);
/// proposer groups share one generation assignment (128 trajectories).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryGroup {
    /// The role of every trajectory in the group.
    pub role: Role,
    /// Group key: environment id for solver groups, assignment id for
    /// proposer groups.
    pub group_key: String,
    /// The trajectories.
    pub trajectories: Vec<Trajectory>,
    /// Rewards, aligned with `trajectories`.
    pub rewards: Vec<f64>,
}

impl TrajectoryGroup {
    /// Construct a group from trajectories, extracting rewards.
    pub fn new(role: Role, group_key: impl Into<String>, trajectories: Vec<Trajectory>) -> Self {
        let rewards = trajectories
            .iter()
            .map(|t| t.reward().unwrap_or(0.0))
            .collect();
        TrajectoryGroup {
            role,
            group_key: group_key.into(),
            trajectories,
            rewards,
        }
    }

    /// Mean reward of the group.
    pub fn mean_reward(&self) -> f64 {
        if self.rewards.is_empty() {
            return 0.0;
        }
        self.rewards.iter().sum::<f64>() / self.rewards.len() as f64
    }

    /// Whether rewards vary across the group (DAPO filter, Section 3.3:
    /// "we drop groups whose rewards do not vary").
    pub fn rewards_vary(&self) -> bool {
        let Some(first) = self.rewards.first() else {
            return false;
        };
        self.rewards
            .iter()
            .any(|r| (r - first).abs() > f64::EPSILON)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_phase_classifier() {
        let read = Turn::new(
            Action::Command("cat memory/past_trajectories/index.jsonl".into()),
            Observation::ok(),
        );
        assert_eq!(read.phase(), TurnPhase::Read);

        let build = Turn::new(
            Action::Command("harbor task init org/task-a".into()),
            Observation::ok(),
        );
        assert_eq!(build.phase(), TurnPhase::Build);

        let test = Turn::new(
            Action::Command("python3 /opt/tools/validate.py output/tasks/task_001 oracle".into()),
            Observation::ok(),
        );
        assert_eq!(test.phase(), TurnPhase::Test);

        let write = Turn::new(
            Action::Command("echo '# Task' > instruction.md".into()),
            Observation::ok(),
        );
        assert_eq!(write.phase(), TurnPhase::Write);
    }

    #[test]
    fn rollout_record_bucket() {
        let r = RolloutRecord {
            env_id: "env_001".into(),
            solver: "qwen3.6-35b".into(),
            round: 3,
            provenance: Provenance::SelfRound(3),
            rewards: vec![0.0, 0.25, 0.5],
            reward_std: 0.25,
            check_failures: std::collections::BTreeMap::new(),
            path: "memory/past_trajectories/failed/env_001.json".into(),
        };
        assert_eq!(r.bucket(), "failed");
        assert!((r.reward_mean() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn group_rewards_vary_dapo_filter_input() {
        let mk = |reward: f64| Trajectory {
            role: Role::Solver,
            prompt: "p".into(),
            turns: vec![],
            outcome: Some(Outcome {
                reward,
                verifier_stdout: String::new(),
                passed_tests: 0,
                total_tests: 4,
                failed_checks: vec![],
            }),
        };
        let varying = TrajectoryGroup::new(Role::Solver, "env_1", vec![mk(0.0), mk(1.0)]);
        let flat = TrajectoryGroup::new(Role::Solver, "env_2", vec![mk(0.5), mk(0.5)]);
        assert!(varying.rewards_vary());
        assert!(!flat.rewards_vary());
    }
}
