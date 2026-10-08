//! Continual-Harness optimization between rounds (Section 3.3).
//!
//! "Following Continual Harness, the proposer updates its own memory,
//! skills, and tools between rounds based on its past rewards and logs.
//! For example, it merged several validation steps into a faster script
//! and saved useful GitHub tasks and Docker Hub images as a web-cache
//! skill. The validation and calibration checks stay fixed."
//!
//! The optimizer encodes those two concrete evolutions (plus lesson
//! writing) as deterministic, logged actions applied to the workspace in
//! the `BetweenRounds` phase — never touching the assignment, the sandbox
//! contract, or anything the admission checks depend on.

use crate::workspace::{Phase, Workspace};
use serde::{Deserialize, Serialize};

/// A summary of one round's proposer experience, the input to harness
/// optimization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarnessRoundLog {
    /// The round that just finished.
    pub round: u64,
    /// Proposer episodes launched.
    pub episodes: usize,
    /// Environments admitted to the pool.
    pub admitted: usize,
    /// Environments rejected (validation or calibration).
    pub rejected: usize,
    /// Proposer rewards (Eq. 2) of the round's episodes.
    pub proposer_rewards: Vec<f64>,
    /// Common failure reasons across rejected episodes.
    pub common_failures: Vec<String>,
    /// Whether any episode used web access.
    pub used_web: bool,
}

impl HarnessRoundLog {
    /// Mean proposer reward of the round.
    pub fn mean_reward(&self) -> f64 {
        if self.proposer_rewards.is_empty() {
            0.0
        } else {
            self.proposer_rewards.iter().sum::<f64>() / self.proposer_rewards.len() as f64
        }
    }
}

/// One harness evolution action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum HarnessAction {
    /// "Merged several validation steps into a faster script": the
    /// oracle + no-op pair becomes one `tools/validate_fast.py` pass.
    MergeValidationSteps {
        /// Tool invocations before the merge (2 per validation: oracle,
        /// no-op).
        steps_before: usize,
        /// Tool invocations after the merge (1).
        steps_after: usize,
    },
    /// "Saved useful GitHub tasks ... as a web-cache skill".
    CacheGithubTask {
        /// The cached repository reference.
        repo: String,
    },
    /// "Saved ... Docker Hub images as a web-cache skill".
    CacheDockerImage {
        /// The cached image reference.
        image: String,
    },
    /// A lesson appended to `memory/lessons.md`.
    AddLesson {
        /// The lesson text.
        lesson: String,
    },
}

impl HarnessAction {
    /// Human-readable rendering for logs.
    pub fn describe(&self) -> String {
        match self {
            HarnessAction::MergeValidationSteps {
                steps_before,
                steps_after,
            } => {
                format!("merged validation steps: {steps_before} -> {steps_after} tool calls")
            }
            HarnessAction::CacheGithubTask { repo } => format!("web-cache: github task {repo}"),
            HarnessAction::CacheDockerImage { image } => format!("web-cache: docker image {image}"),
            HarnessAction::AddLesson { lesson } => format!("lesson: {lesson}"),
        }
    }
}

/// The harness optimizer: deterministic rules derived from the round log.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessOptimizer {
    /// Whether the merged validation script has been installed yet.
    pub merged_validation: bool,
    /// GitHub tasks cached so far.
    pub cached_tasks: Vec<String>,
    /// Docker images cached so far.
    pub cached_images: Vec<String>,
}

impl HarnessOptimizer {
    /// Run one optimization pass on the workspace (which must be in the
    /// `BetweenRounds` phase) and return the actions applied.
    ///
    /// Rules (paper-faithful, deterministic):
    ///
    /// 1. After the first round with rejections, merge the validation
    ///    steps into a single faster script.
    /// 2. When the round used web access, cache one GitHub task and one
    ///    Docker Hub image per round into the web-cache skill.
    /// 3. Reject-heavy rounds append the dominant failure as a lesson.
    ///
    /// The validation and calibration *checks* stay fixed — only the
    /// proposer's own tools change, and the host keeps re-running its own
    /// copies outside the sandbox.
    pub fn optimize(
        &mut self,
        workspace: &mut Workspace,
        log: &HarnessRoundLog,
    ) -> Vec<HarnessAction> {
        workspace.set_phase(Phase::BetweenRounds);
        let mut actions = Vec::new();

        // 1. Merge validation steps into a faster script.
        if !self.merged_validation && log.round >= 1 && log.episodes > 0 {
            let steps_before = log.episodes * 2; // oracle + no-op per episode
            let steps_after = log.episodes; // one merged pass per episode
            let _ = workspace.edit("tools/validate_fast.py", crate::workspace::VALIDATE_FAST_PY);
            actions.push(HarnessAction::MergeValidationSteps {
                steps_before,
                steps_after,
            });
            self.merged_validation = true;
        }

        // 2. Grow the web-cache skill.
        if log.used_web {
            let repo = format!("github.com/example/task-{}", log.round);
            let image = format!("docker.io/library/python:3.{}", 9 + (log.round % 10));
            workspace.memory.web_cache.cache_task(&repo);
            workspace.memory.web_cache.cache_image(&image);
            let _ = workspace.edit(
                "skills/web-cache/SKILL.md",
                workspace.memory.web_cache.render_skill(),
            );
            actions.push(HarnessAction::CacheGithubTask { repo });
            actions.push(HarnessAction::CacheDockerImage { image });
            self.cached_tasks
                .push(format!("github.com/example/task-{}", log.round));
            self.cached_images.push(format!(
                "docker.io/library/python:3.{}",
                9 + (log.round % 10)
            ));
        }

        // 3. Learn from the dominant failure.
        if log.rejected > 0 {
            if let Some(dom) = log.common_failures.first() {
                let lesson = format!("round {}: dominant failure — {}", log.round, dom);
                workspace.memory.add_lesson(&lesson);
                let lessons = workspace.memory.lessons.clone();
                let _ = workspace.edit("memory/lessons.md", lessons);
                actions.push(HarnessAction::AddLesson { lesson });
            }
        } else if log.admitted > 0 {
            let lesson = format!(
                "round {}: all {} submissions admitted",
                log.round, log.admitted
            );
            workspace.memory.add_lesson(&lesson);
            let lessons = workspace.memory.lessons.clone();
            let _ = workspace.edit("memory/lessons.md", lessons);
            actions.push(HarnessAction::AddLesson { lesson });
        }

        workspace.set_phase(Phase::DuringEpisode);
        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log(
        round: u64,
        admitted: usize,
        rejected: usize,
        used_web: bool,
        failures: Vec<&str>,
    ) -> HarnessRoundLog {
        HarnessRoundLog {
            round,
            episodes: admitted + rejected,
            admitted,
            rejected,
            proposer_rewards: vec![if rejected > 0 { -0.25 } else { 1.0 }; admitted + rejected],
            common_failures: failures.into_iter().map(|f| f.to_string()).collect(),
            used_web,
        }
    }

    #[test]
    fn merges_validation_after_first_round() {
        let mut ws = Workspace::new();
        let mut opt = HarnessOptimizer::default();
        let actions = opt.optimize(
            &mut ws,
            &log(
                1,
                2,
                1,
                false,
                vec!["reference solution earned 0.75 (must be 1.0)"],
            ),
        );
        assert!(actions
            .iter()
            .any(|a| matches!(a, HarnessAction::MergeValidationSteps { .. })));
        assert!(ws.files.contains_key("tools/validate_fast.py"));
        // Idempotent: the second round does not re-merge.
        let actions2 = opt.optimize(&mut ws, &log(2, 2, 0, false, vec![]));
        assert!(!actions2
            .iter()
            .any(|a| matches!(a, HarnessAction::MergeValidationSteps { .. })));
    }

    #[test]
    fn web_cache_grows_when_web_used() {
        let mut ws = Workspace::new();
        let mut opt = HarnessOptimizer::default();
        let actions = opt.optimize(&mut ws, &log(1, 1, 0, true, vec![]));
        assert!(actions
            .iter()
            .any(|a| matches!(a, HarnessAction::CacheGithubTask { .. })));
        assert!(actions
            .iter()
            .any(|a| matches!(a, HarnessAction::CacheDockerImage { .. })));
        assert!(ws.files.contains_key("skills/web-cache/SKILL.md"));
        assert!(ws
            .read("skills/web-cache/SKILL.md")
            .unwrap()
            .contains("github.com/example/task-1"));
        let actions2 = opt.optimize(&mut ws, &log(2, 1, 0, true, vec![]));
        assert!(actions2.len() >= 2);
        assert!(ws
            .read("skills/web-cache/SKILL.md")
            .unwrap()
            .contains("task-2"));
    }

    #[test]
    fn lessons_from_dominant_failure() {
        let mut ws = Workspace::new();
        let mut opt = HarnessOptimizer::default();
        opt.optimize(
            &mut ws,
            &log(
                1,
                0,
                2,
                false,
                vec!["do-nothing solution earned 1.0 (must be < 0.5)"],
            ),
        );
        assert!(ws
            .read("memory/lessons.md")
            .unwrap()
            .contains("do-nothing solution earned 1.0"));
    }

    #[test]
    fn harness_never_touches_fixed_regions() {
        let mut ws = Workspace::new();
        let contract_before = ws.read("SANDBOX_CONTRACT.md").unwrap().to_string();
        let assignment_before = "assignment text".to_string();
        ws.set_assignment(&assignment_before);
        let mut opt = HarnessOptimizer::default();
        opt.optimize(&mut ws, &log(1, 1, 1, true, vec!["x"]));
        // Fixed regions untouched.
        assert_eq!(ws.read("SANDBOX_CONTRACT.md").unwrap(), &contract_before);
        assert_eq!(
            ws.read("assignment/current.md").unwrap(),
            &assignment_before
        );
        // Phase restored to during-episode.
        assert_eq!(ws.phase, crate::workspace::Phase::DuringEpisode);
        // And the proposer now CANNOT edit skills (fixed during episodes).
        assert!(ws.edit("skills/web-cache/SKILL.md", "tamper").is_err());
    }

    #[test]
    fn mean_reward_helper() {
        // 1 admitted (+1) and 1 calibration-rejected (-0.25).
        let mut l = log(1, 1, 1, false, vec![]);
        l.proposer_rewards = vec![1.0, -0.25];
        assert!((l.mean_reward() - 0.375).abs() < 1e-9);
        assert!(log(2, 0, 2, false, vec![]).mean_reward() < 0.0);
    }
}
