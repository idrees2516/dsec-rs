//! Persistent proposer memory: lessons, past environments, the trajectory
//! archive with `index.jsonl`, and the web cache (Section 3.1, App. D.1).
//!
//! "The workspace contains the solver's past rollouts with their verifier
//! outputs, previously generated environments, instructions for writing
//! Harbor tasks, and the proposer's own memory and tools. The rollouts and
//! past environments are refreshed from the training pool each round, while
//! the memory and tools persist across rounds."
//!
//! The archive index is the skill's first grounding read: "use shell/tool
//! calls to find the latest `provenance:\"self\"` round and inspect the
//! solver's `reward_std` plus raw `rewards`."

use crate::harbor::HarborTask;
use crate::trajectory::{Provenance, RolloutRecord, Trajectory};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// The trajectory archive: `memory/past_trajectories/` with its
/// `index.jsonl` and the full attempt files.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryArchive {
    /// Rows of `index.jsonl`.
    pub index: Vec<RolloutRecord>,
    /// Full trajectories keyed by `failed|successful/<env_id>/<n>.json`.
    pub attempts: BTreeMap<String, Trajectory>,
}

impl TrajectoryArchive {
    /// Render `index.jsonl` (one JSON object per line).
    pub fn to_index_jsonl(&self) -> String {
        let mut s = String::new();
        for row in &self.index {
            if let Ok(v) = serde_json::to_string(row) {
                s.push_str(&v);
                s.push('\n');
            }
        }
        s
    }

    /// The latest round with `provenance: "self"` — the grounding read the
    /// skill mandates.
    pub fn latest_self_round(&self) -> Option<u64> {
        self.index
            .iter()
            .filter_map(|r| match &r.provenance {
                Provenance::SelfRound(round) => Some(*round),
                Provenance::Frontier(_) => None,
            })
            .max()
    }

    /// Records for one environment.
    pub fn records_for(&self, env_id: &str) -> Vec<&RolloutRecord> {
        self.index.iter().filter(|r| r.env_id == env_id).collect()
    }

    /// Replace the archive with this round's host refresh. Attempt files go
    /// under `failed/` or `successful/` by mean reward (Figure 18).
    pub fn refresh(&mut self, records: Vec<RolloutRecord>, attempts: Vec<(String, Trajectory)>) {
        self.index = records;
        self.attempts.clear();
        for (env_id, traj) in attempts {
            let bucket = match traj.reward() {
                Some(r) if r >= 0.5 => "successful",
                _ => "failed",
            };
            self.attempts.insert(format!("{bucket}/{env_id}"), traj);
        }
    }

    /// Count of archived attempt files.
    pub fn attempt_count(&self) -> usize {
        self.attempts.len()
    }
}

/// The web cache the proposer's harness optimization built: "it ... saved
/// useful GitHub tasks and Docker Hub images as a web-cache skill."
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WebCache {
    /// Cached GitHub task references.
    pub github_tasks: BTreeSet<String>,
    /// Cached Docker Hub image references.
    pub docker_images: BTreeSet<String>,
}

impl WebCache {
    /// Record a cached GitHub task.
    pub fn cache_task(&mut self, repo: impl Into<String>) {
        self.github_tasks.insert(repo.into());
    }

    /// Record a cached Docker image.
    pub fn cache_image(&mut self, image: impl Into<String>) {
        self.docker_images.insert(image.into());
    }

    /// Render the web-cache skill body.
    pub fn render_skill(&self) -> String {
        let mut s = String::from("# web-cache\n\nCached during earlier rounds; reuse before searching.\n\n## GitHub tasks\n\n");
        for t in &self.github_tasks {
            s.push_str(&format!("- {t}\n"));
        }
        s.push_str("\n## Docker Hub images\n\n");
        for i in &self.docker_images {
            s.push_str(&format!("- {i}\n"));
        }
        s
    }
}

/// The proposer's persistent memory (what survives rounds).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    /// `memory/lessons.md` — accumulated lessons, editable by the proposer.
    pub lessons: String,
    /// `memory/past_environments/<env_id>/` — rendered canonical layouts of
    /// the parent environments, refreshed each round by the host.
    pub past_environments: BTreeMap<String, String>,
    /// `memory/past_trajectories/` — the archive.
    pub archive: TrajectoryArchive,
    /// The web cache (a harness skill).
    pub web_cache: WebCache,
}

impl Memory {
    /// Append a lesson (the proposer edits its own memory between or during
    /// episodes).
    pub fn add_lesson(&mut self, lesson: impl Into<String>) {
        if !self.lessons.is_empty() {
            self.lessons.push('\n');
        }
        self.lessons.push_str(&format!("- {}", lesson.into()));
    }

    /// Host-side refresh: replace rollouts and past environments from the
    /// training pool (kept read-only for the proposer).
    pub fn refresh_from_pool(
        &mut self,
        records: Vec<RolloutRecord>,
        attempts: Vec<(String, Trajectory)>,
        envs: &[&HarborTask],
    ) {
        self.archive.refresh(records, attempts);
        self.past_environments.clear();
        for env in envs {
            self.past_environments
                .insert(env.env_id.clone(), render_environment_snapshot(env));
        }
    }

    /// Whether the memory is empty (round 0, before any refresh).
    pub fn is_empty(&self) -> bool {
        self.lessons.is_empty()
            && self.archive.index.is_empty()
            && self.past_environments.is_empty()
    }
}

/// Render a past environment snapshot: the canonical Harbor layout notes
/// the skill tells the proposer to read ("how each attempted env is BUILT:
/// instruction.md, tests/test_outputs.py = real verifier design,
/// environment/Dockerfile + build script = how state is planted, task.toml
/// = taxonomy").
pub fn render_environment_snapshot(env: &HarborTask) -> String {
    let mut s = String::new();
    s.push_str(&format!("# environment {}\n\n", env.env_id));
    s.push_str("## instruction.md\n\n");
    s.push_str(&env.instruction.body);
    s.push_str("\n\n## task.toml\n\n");
    s.push_str(&env.render_task_toml());
    s.push_str("\n\n## environment/Dockerfile\n\n");
    s.push_str(&env.environment.render_dockerfile());
    s.push_str("\n\n## tests/test_outputs.py (verifier design)\n\n");
    for check in env.check_names() {
        s.push_str(&format!("def {check}(): ...\n"));
    }
    s.push_str("\n\n## taxonomy\n\n");
    s.push_str(&format!(
        "skills: {}\nchecks: {}\ndifficulty: {}\n",
        env.skills
            .iter()
            .map(|k| k.name().to_string())
            .collect::<Vec<_>>()
            .join(", "),
        env.total_checks(),
        env.toml.metadata.difficulty.as_str()
    ));
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{Outcome, Role};

    fn traj(reward: f64) -> Trajectory {
        Trajectory {
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
        }
    }

    fn record(env: &str, round: u64, rewards: Vec<f64>) -> RolloutRecord {
        RolloutRecord {
            env_id: env.into(),
            solver: "s".into(),
            round,
            provenance: Provenance::SelfRound(round),
            rewards,
            reward_std: 0.2,
            check_failures: BTreeMap::new(),
            path: format!("memory/past_trajectories/failed/{env}.json"),
        }
    }

    #[test]
    fn index_jsonl_round_trips() {
        let mut archive = TrajectoryArchive::default();
        archive.index.push(record("env_1", 2, vec![0.0, 1.0]));
        archive.index.push(record("env_1", 3, vec![0.5, 0.5]));
        let jsonl = archive.to_index_jsonl();
        assert_eq!(jsonl.lines().count(), 2);
        let first: RolloutRecord = serde_json::from_str(jsonl.lines().next().unwrap()).unwrap();
        assert_eq!(first.env_id, "env_1");
        assert_eq!(first.round, 2);
    }

    #[test]
    fn latest_self_round_ignores_frontier() {
        let mut archive = TrajectoryArchive::default();
        archive.index.push(record("a", 1, vec![0.0]));
        archive.index.push(record("b", 5, vec![0.0]));
        archive.index.push(RolloutRecord {
            env_id: "c".into(),
            solver: "s".into(),
            round: 9,
            provenance: Provenance::Frontier("Claude Opus 5".into()),
            rewards: vec![0.0],
            reward_std: 0.0,
            check_failures: BTreeMap::new(),
            path: String::new(),
        });
        assert_eq!(archive.latest_self_round(), Some(5));
    }

    #[test]
    fn archive_buckets_attempts() {
        let mut archive = TrajectoryArchive::default();
        archive.refresh(
            vec![record("env_1", 1, vec![0.0])],
            vec![
                ("env_1".to_string(), traj(0.25)),
                ("env_2".to_string(), traj(0.75)),
            ],
        );
        assert!(archive.attempts.contains_key("failed/env_1"));
        assert!(archive.attempts.contains_key("successful/env_2"));
        assert_eq!(archive.attempt_count(), 2);
    }

    #[test]
    fn memory_lessons_accumulate() {
        let mut m = Memory::default();
        m.add_lesson("no-op must fail before submit");
        m.add_lesson("prefer json over csv checks for spread");
        assert!(m.lessons.contains("- no-op must fail"));
        assert!(m.lessons.lines().count() == 2);
        assert!(!m.is_empty());
    }

    #[test]
    fn web_cache_skill_renders() {
        let mut wc = WebCache::default();
        wc.cache_task("github.com/foo/bar");
        wc.cache_image("docker.io/library/postgres:16");
        let skill = wc.render_skill();
        assert!(skill.contains("# web-cache"));
        assert!(skill.contains("github.com/foo/bar"));
        assert!(skill.contains("docker.io/library/postgres:16"));
    }
}
