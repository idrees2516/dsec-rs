//! The solver's training pool (Section 3.3, Appendix G.1).
//!
//! "Admitted environments form the solver's training pool. At each
//! refresh, the host reruns the updated solver on the pool, removes
//! environments that fall outside the calibration band, and asks the
//! proposer for replacements based on the latest rollouts."
//!
//! Table 12: "Active environment pool: 256 tasks. Pool review interval: 16
//! solver updates."

use crate::calibrate::{CalibrationConfig, CalibrationVerdict};
use crate::error::Result;
use crate::harbor::HarborTask;
use crate::policy::SolverModel;
use crate::trajectory::RolloutRecord;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Running statistics of one pool environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvStats {
    /// Environment id.
    pub env_id: String,
    /// Mean reward of the latest review attempts (the solver's pass rate).
    pub pass_rate: f64,
    /// Reward standard deviation of the latest attempts.
    pub reward_std: f64,
    /// Number of attempts in the latest review.
    pub attempts: usize,
    /// The update index at which the environment was last reviewed.
    pub last_reviewed: u64,
}

/// One admitted environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoolEntry {
    /// The task.
    pub task: HarborTask,
    /// The latest solver statistics on it.
    pub stats: EnvStats,
    /// The solver-update index at which it was admitted.
    pub admitted_at: u64,
}

/// An eviction during pool review.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Eviction {
    /// Environment id.
    pub env_id: String,
    /// Why it was removed.
    pub reason: String,
    /// Its last pass rate.
    pub pass_rate: f64,
    /// Its last reward std.
    pub reward_std: f64,
    /// The evicted task itself — the parent of the replacement assignment
    /// ("asks the proposer for replacements based on the latest rollouts").
    pub task: HarborTask,
}

/// The outcome of one pool review.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReviewReport {
    /// The update index the review ran at.
    pub update: u64,
    /// Environments removed this review.
    pub evicted: Vec<Eviction>,
    /// Environments rechecked.
    pub rechecked: usize,
    /// Environments retained.
    pub retained: usize,
}

/// The training pool.
#[derive(Debug, Clone, Default)]
pub struct TrainingPool {
    /// Admitted entries keyed by env id.
    pub entries: BTreeMap<String, PoolEntry>,
    /// Maximum pool size (Table 12: 256).
    pub capacity: usize,
}

impl TrainingPool {
    /// Create a pool with the given capacity.
    pub fn new(capacity: usize) -> Self {
        TrainingPool {
            entries: BTreeMap::new(),
            capacity,
        }
    }

    /// Number of environments.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the pool is at capacity.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    /// Admit an environment (admission already decided upstream).
    pub fn admit(&mut self, task: HarborTask, update: u64) -> Result<()> {
        if self.entries.len() >= self.capacity {
            return Err(crate::error::Error::Invariant(format!(
                "pool at capacity ({})",
                self.capacity
            )));
        }
        let env_id = task.env_id.clone();
        let stats = EnvStats {
            env_id: env_id.clone(),
            pass_rate: 0.0,
            reward_std: 0.0,
            attempts: 0,
            last_reviewed: update,
        };
        self.entries.insert(
            env_id,
            PoolEntry {
                task,
                stats,
                admitted_at: update,
            },
        );
        Ok(())
    }

    /// All tasks, in env-id order.
    pub fn tasks(&self) -> Vec<&HarborTask> {
        self.entries.values().map(|e| &e.task).collect()
    }

    /// One task by id.
    pub fn task(&self, env_id: &str) -> Option<&HarborTask> {
        self.entries.get(env_id).map(|e| &e.task)
    }

    /// Update an environment's stats after solver attempts.
    pub fn update_stats(&mut self, env_id: &str, rewards: &[f64], update: u64) {
        if let Some(entry) = self.entries.get_mut(env_id) {
            entry.stats.pass_rate = if rewards.is_empty() {
                0.0
            } else {
                rewards.iter().sum::<f64>() / rewards.len() as f64
            };
            entry.stats.reward_std = crate::calibrate::population_std(rewards);
            entry.stats.attempts = rewards.len();
            entry.stats.last_reviewed = update;
        }
    }

    /// Environments whose stats are out of the calibration band (or have no
    /// spread) — eviction candidates.
    pub fn out_of_band(&self, config: &CalibrationConfig) -> Vec<(String, f64, f64)> {
        let mut out = Vec::new();
        for entry in self.entries.values() {
            if entry.stats.attempts == 0 {
                continue; // never reviewed yet
            }
            let verdict = if entry.stats.pass_rate < config.band.0 {
                CalibrationVerdict::TooHard
            } else if entry.stats.pass_rate > config.band.1 {
                CalibrationVerdict::TooEasy
            } else if entry.stats.reward_std < config.min_std {
                CalibrationVerdict::NoSpread
            } else {
                CalibrationVerdict::Accepted
            };
            if !verdict.admitted() {
                out.push((
                    entry.stats.env_id.clone(),
                    entry.stats.pass_rate,
                    entry.stats.reward_std,
                ));
            }
        }
        out
    }

    /// The pool review (Section 3.3): "the host reruns the updated solver
    /// on the pool, removes environments that fall outside the calibration
    /// band, and asks the proposer for replacements."
    ///
    /// Each environment gets `initial_rollouts` fresh attempts from the
    /// current solver; out-of-band or no-spread environments are evicted.
    pub fn review(
        &mut self,
        solver: &mut dyn SolverModel,
        config: &CalibrationConfig,
        update: u64,
    ) -> ReviewReport {
        let env_ids: Vec<String> = self.entries.keys().cloned().collect();
        let mut evicted = Vec::new();
        let mut per_env: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for env_id in &env_ids {
            let task = self.entries[env_id].task.clone();
            let mut rewards = Vec::new();
            for _ in 0..config.initial_rollouts {
                rewards.push(solver.attempt(&task, None).reward);
            }
            per_env.insert(env_id.clone(), rewards);
        }
        for (env_id, rewards) in &per_env {
            self.update_stats(env_id, rewards, update);
        }
        for (env_id, pass_rate, reward_std) in self.out_of_band(config) {
            let reason = if pass_rate > config.band.1 {
                format!("solver mean {pass_rate:.2} above band: too easy for the current solver")
            } else if pass_rate < config.band.0 {
                format!("solver mean {pass_rate:.2} below band: too hard for the current solver")
            } else {
                format!("reward std {reward_std:.2} below minimum: no learning signal")
            };
            let task = self.entries.get(&env_id).map(|e| e.task.clone());
            self.entries.remove(&env_id);
            if let Some(task) = task {
                evicted.push(Eviction {
                    env_id: env_id.clone(),
                    reason,
                    pass_rate,
                    reward_std,
                    task,
                });
            }
        }
        ReviewReport {
            update,
            rechecked: env_ids.len(),
            retained: self.entries.len(),
            evicted,
        }
    }

    /// How many replacements the pool needs: evicted slots plus capacity
    /// shortfall.
    pub fn replacements_needed(&self) -> usize {
        self.capacity - self.entries.len()
    }

    /// Sample the environments for one solver rollout batch (Table 12:
    /// 16 environments per update; "both methods retain 16 groups").
    pub fn sample_solver_batch(&self, n: usize, offset: usize) -> Vec<String> {
        let ids: Vec<String> = self.entries.keys().cloned().collect();
        if ids.is_empty() {
            return Vec::new();
        }
        (0..n.min(ids.len()))
            .map(|i| ids[(offset + i) % ids.len()].clone())
            .collect()
    }

    /// Build rollout records for the archive refresh from the latest stats.
    pub fn rollout_records(&self, round: u64, solver_id: &str) -> Vec<RolloutRecord> {
        self.entries
            .values()
            .map(|e| RolloutRecord {
                env_id: e.stats.env_id.clone(),
                solver: solver_id.to_string(),
                round,
                provenance: crate::trajectory::Provenance::SelfRound(round),
                rewards: vec![e.stats.pass_rate; e.stats.attempts.max(1)],
                reward_std: e.stats.reward_std,
                check_failures: BTreeMap::new(),
                path: format!(
                    "memory/past_trajectories/{}/{}.json",
                    if e.stats.pass_rate >= 0.5 {
                        "successful"
                    } else {
                        "failed"
                    },
                    e.stats.env_id
                ),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::calibrate;
    use crate::harbor::{Difficulty, SkillTag};
    use crate::policy::{generate_task, ScriptedSolver, TaskGenParams};

    fn task(id: &str, n: usize) -> HarborTask {
        generate_task(&TaskGenParams {
            env_id: id.into(),
            name: format!("flywheel/{id}"),
            n_checks: n,
            skills: vec![SkillTag::new("csv"), SkillTag::new("json")],
            difficulty: Difficulty::Medium,
            broken: false,
        })
    }

    fn strong_solver() -> ScriptedSolver {
        let mut s = ScriptedSolver::new("strong", 9);
        s.bank.tables.insert(
            "csv".into(),
            crate::dppo::SoftmaxPolicy {
                logits: vec![8.0, -5.0, 0.0],
            },
        );
        s.bank.tables.insert(
            "json".into(),
            crate::dppo::SoftmaxPolicy {
                logits: vec![8.0, -5.0, 0.0],
            },
        );
        s
    }

    #[test]
    fn admission_and_capacity() {
        let mut pool = TrainingPool::new(3);
        pool.admit(task("env_1", 4), 0).unwrap();
        pool.admit(task("env_2", 4), 0).unwrap();
        assert_eq!(pool.len(), 2);
        assert!(!pool.is_full());
        pool.admit(task("env_3", 4), 0).unwrap();
        assert!(pool.is_full());
        let err = pool.admit(task("env_4", 4), 0).unwrap_err();
        assert!(err.to_string().contains("capacity"));
    }

    #[test]
    fn review_evicts_too_easy_for_strong_solver() {
        let mut pool = TrainingPool::new(8);
        pool.admit(task("env_1", 4), 0).unwrap();
        pool.admit(task("env_2", 4), 0).unwrap();
        let mut solver = strong_solver();
        let report = pool.review(&mut solver, &CalibrationConfig::default(), 16);
        // The strong solver fully solves 4-check tasks -> too easy -> evict.
        assert_eq!(report.evicted.len(), 2);
        assert_eq!(pool.len(), 0);
        assert!(report.evicted[0].reason.contains("too easy"));
        // The evicted task is carried for the replacement assignment.
        assert_eq!(report.evicted[0].task.env_id, "env_1");
    }

    #[test]
    fn replacements_needed_after_eviction() {
        let mut pool = TrainingPool::new(8);
        for i in 1..=3 {
            pool.admit(task(&format!("env_{i}"), 4), 0).unwrap();
        }
        assert_eq!(pool.replacements_needed(), 5);
        let mut solver = strong_solver();
        let _ = pool.review(&mut solver, &CalibrationConfig::default(), 16);
        // All evicted -> pool empty -> all 8 slots need replacements.
        assert_eq!(pool.replacements_needed(), 8);
    }

    #[test]
    fn stats_track_attempts() {
        let mut pool = TrainingPool::new(4);
        pool.admit(task("env_1", 4), 0).unwrap();
        pool.update_stats("env_1", &[0.5, 0.5, 0.5, 0.5], 5);
        let e = &pool.entries["env_1"];
        assert!((e.stats.pass_rate - 0.5).abs() < 1e-9);
        assert!(e.stats.reward_std < 1e-9);
        assert_eq!(e.stats.attempts, 4);
    }

    #[test]
    fn solver_batch_wraps() {
        let mut pool = TrainingPool::new(8);
        for i in 1..=3 {
            pool.admit(task(&format!("env_{i}"), 4), 0).unwrap();
        }
        let batch = pool.sample_solver_batch(16, 0);
        assert_eq!(batch.len(), 3); // only 3 distinct envs
        assert_eq!(pool.sample_solver_batch(2, 1), vec!["env_2", "env_3"]);
    }

    #[test]
    fn rollout_records_for_archive() {
        let mut pool = TrainingPool::new(8);
        pool.admit(task("env_1", 4), 0).unwrap();
        pool.update_stats("env_1", &[0.2, 0.4], 3);
        let records = pool.rollout_records(7, "solver-a");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].round, 7);
        assert_eq!(records[0].env_id, "env_1");
        assert_eq!(
            records[0].provenance,
            crate::trajectory::Provenance::SelfRound(7)
        );
        assert!(records[0].path.contains("failed"));
    }

    #[test]
    fn calibrate_integration_with_pool_task() {
        // A pool task that a weak solver cannot solve calibrates TooHard.
        let t = task("env_9", 6);
        let mut solver = ScriptedSolver::new("weak", 4);
        // Populate the skill tables with one calibration, then skew to
        // always-skip.
        let _ = calibrate(&t, &mut solver, &CalibrationConfig::default());
        for policy in solver.bank.tables.values_mut() {
            policy.logits = vec![-8.0, -5.0, 0.0];
        }
        let out = calibrate(&t, &mut solver, &CalibrationConfig::default());
        assert_eq!(out.verdict, CalibrationVerdict::TooHard);
    }
}
