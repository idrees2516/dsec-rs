//! Branch-safe subagents (Phase 3).
//!
//! A subagent is a *fork* of the parent's workspace plus its own
//! worker container, its own journal file, and its own bounded budget.
//! The isolation rules follow the paper's pack_diff semantics:
//!
//! - the **fork** deep-copies writable state (volumes, DB rows, the
//!   idempotency ledger) and shares only the immutable tool surface;
//!   neither side sees the other's writes;
//! - the subagent runs to completion (bounded concurrency, bounded
//!   steps) and produces a [`WorkspacePatch`] — a three-way-merge-ready
//!   diff against the branch-point state;
//! - the **merge validator** checks every op against the parent's
//!   *current* state and previously merged sibling patches: a file the
//!   parent also changed (or another subagent already patched)
//!   conflicts; a merge that would break source conservation is
//!   rejected and rolled back;
//! - **abandoned branches** are destroyed without merging, and their
//!   journals record the abandonment.
//!
//! The most damaging bug this design forbids: two agents silently
//! sharing writable state. The fork guarantees isolation; the
//! three-way base hashes guarantee detection of overlapping edits.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::json;

use crate::error::{Error, Result};
use crate::journal::{event_kind, Journal};
use crate::taskenv::{fnv1a, BranchBase, PatchOp, TaskEnvironment, WorkspacePatch};
use crate::worker::{TaskOutcome, WorkerConfig, WorkerContainer};

/// A subagent's configuration.
#[derive(Debug, Clone)]
pub struct SubagentSpec {
    /// Branch (subagent task) id.
    pub id: String,
    /// The sub-task instruction.
    pub instruction: String,
    /// Step budget for the subagent's rollout.
    pub budget_steps: u64,
}

impl SubagentSpec {
    /// New spec.
    pub fn new(id: &str, instruction: &str, budget_steps: u64) -> Self {
        Self {
            id: id.to_string(),
            instruction: instruction.to_string(),
            budget_steps,
        }
    }
}

/// Merge outcome.
#[derive(Debug, Clone, PartialEq)]
pub enum MergeOutcome {
    /// The patch applied cleanly.
    Applied {
        /// Number of ops applied.
        ops: usize,
    },
    /// Overlapping edits — nothing was applied.
    Conflict {
        /// Human-readable conflict descriptions.
        conflicts: Vec<String>,
    },
    /// Applied then rolled back (protection violation).
    Rejected {
        /// Why.
        reason: String,
    },
}

/// What a finished branch produced.
struct BranchRun {
    outcome: TaskOutcome,
    env: Box<dyn TaskEnvironment>,
}

struct BranchRecord {
    journal_path: PathBuf,
    base: BranchBase,
    handle: tokio::task::JoinHandle<Result<BranchRun>>,
}

/// Spawns, joins, merges, and reaps subagent branches for one parent
/// task.
pub struct SubagentManager {
    journal_dir: PathBuf,
    parent_task: String,
    model: Arc<dyn crate::model::ModelClient>,
    branches: BTreeMap<String, BranchRecord>,
    applied: Vec<WorkspacePatch>,
    permits: Arc<tokio::sync::Semaphore>,
    inflight: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}

impl SubagentManager {
    /// New manager: subagent journals live next to the parent's
    /// (`{dir}/{branch_id}.jsonl`).
    pub fn new(
        journal_dir: impl Into<PathBuf>,
        parent_task: &str,
        model: Arc<dyn crate::model::ModelClient>,
        max_concurrency: usize,
    ) -> Self {
        Self {
            journal_dir: journal_dir.into(),
            parent_task: parent_task.to_string(),
            model,
            branches: BTreeMap::new(),
            applied: Vec::new(),
            permits: Arc::new(tokio::sync::Semaphore::new(max_concurrency.max(1))),
            inflight: Arc::new(AtomicUsize::new(0)),
            peak: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Peak simultaneous subagent executions (concurrency tests).
    pub fn peak_concurrency(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }

    /// Patches merged so far (sibling-conflict input).
    pub fn applied_patches(&self) -> &[WorkspacePatch] {
        &self.applied
    }

    /// Active branch ids.
    pub fn branches(&self) -> Vec<String> {
        self.branches.keys().cloned().collect()
    }

    /// Spawns one branch: captures the branch base, forks the parent
    /// workspace, and runs the subagent worker under the concurrency
    /// semaphore.
    pub async fn spawn(
        &mut self,
        parent_env: &mut dyn TaskEnvironment,
        spec: SubagentSpec,
        worker_cfg: WorkerConfig,
    ) -> Result<String> {
        if self.branches.contains_key(&spec.id) {
            return Err(Error::Config(format!("branch {} already exists", spec.id)));
        }
        // 1. branch base = parent state at fork time
        let base = parent_env.capture_state().await?;
        // 2. isolated fork (in-memory pack_diff)
        let mut child = parent_env.fork(&spec.id).await?;
        // the fork inherited the PARENT's task marker: re-target it to
        // the branch's sub-task
        child.set_instruction(&spec.instruction);
        // 3. its own journal — the spawn lineage is recorded BEFORE the
        //    worker starts, so only one handle appends at a time and the
        //    hash chain stays intact
        let journal_path = self
            .journal_dir
            .join(format!("{}.jsonl", sanitize(&spec.id)));
        let mut journal = Journal::create(&journal_path)?;
        journal.append(
            &spec.id,
            event_kind::SUBAGENT_SPAWNED,
            json!({"parent": self.parent_task, "base_files": base.files.len(), "base_rows": base.rows.len()}),
        )?;
        let cfg = WorkerConfig {
            step_limit: spec.budget_steps,
            ..worker_cfg
        };
        let worker =
            WorkerContainer::new(spec.id.clone(), child, self.model.clone(), journal, cfg)?;
        // 4. spawn under the semaphore
        let permits = self.permits.clone();
        let inflight = self.inflight.clone();
        let peak = self.peak.clone();
        let handle = tokio::spawn(async move {
            let _permit = permits.acquire().await.expect("semaphore never closed");
            let n = inflight.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(n, Ordering::SeqCst);
            let mut worker = worker;
            let result = worker.run_task().await;
            inflight.fetch_sub(1, Ordering::SeqCst);
            let env = worker.into_env();
            result.map(|outcome| BranchRun {
                outcome,
                env: env.expect("env survives the worker"),
            })
        });
        self.branches.insert(
            spec.id.clone(),
            BranchRecord {
                journal_path,
                base,
                handle,
            },
        );
        Ok(spec.id)
    }

    /// Joins a finished branch and merges its patch into the parent:
    /// validate → apply → check protection → (rollback if violated).
    pub async fn join(
        &mut self,
        branch_id: &str,
        parent_env: &mut dyn TaskEnvironment,
    ) -> Result<MergeOutcome> {
        let record = self
            .branches
            .remove(branch_id)
            .ok_or_else(|| Error::Config(format!("unknown branch {branch_id}")))?;
        let run = record.handle.await.map_err(|e| Error::Task {
            task: branch_id.to_string(),
            message: format!("branch task panicked: {e}"),
        })??;
        // the branch's rollout result (reward) is the subagent's own
        // outcome; it is journaled with the patch for provenance
        let branch_outcome = match &run.outcome {
            TaskOutcome::Completed { reward, .. } => json!({
                "kind": "completed",
                "score": reward.score().map(|s| s as f64),
                "validity": format!("{:?}", reward.validity),
            }),
            TaskOutcome::Failed { error } => json!({"kind": "failed", "error": error}),
        };
        let mut branch_env = run.env;
        let patch = branch_env.patch_since(branch_id, &record.base).await?;
        let mut journal = Journal::open_append(&record.journal_path)?;
        journal.append(
            branch_id,
            event_kind::PATCH_RETURNED,
            json!({"ops": patch.ops.len(), "outcome": branch_outcome}),
        )?;
        let outcome = merge_into_parent(parent_env, &patch, &self.applied).await?;
        journal.append(
            branch_id,
            event_kind::MERGE_APPLIED,
            json!({"outcome": format!("{:?}", outcome)}),
        )?;
        if let MergeOutcome::Applied { .. } = outcome {
            self.applied.push(patch);
        }
        branch_env.destroy().await?;
        Ok(outcome)
    }

    /// Abandons a branch: no merge, no patch, workspace destroyed.
    pub async fn abandon(&mut self, branch_id: &str) -> Result<()> {
        let record = self
            .branches
            .remove(branch_id)
            .ok_or_else(|| Error::Config(format!("unknown branch {branch_id}")))?;
        let run = record.handle.await.map_err(|e| Error::Task {
            task: branch_id.to_string(),
            message: format!("branch task panicked: {e}"),
        })??;
        let mut branch_env = run.env;
        let mut journal = Journal::open_append(&record.journal_path)?;
        journal.append(
            branch_id,
            event_kind::BRANCH_ABANDONED,
            json!({"parent": self.parent_task}),
        )?;
        branch_env.destroy().await?;
        Ok(())
    }
}

/// The three-way merge: patch ops vs the parent's current state vs
/// already-applied sibling patches. Nothing is mutated on conflict.
pub async fn merge_into_parent(
    parent_env: &mut dyn TaskEnvironment,
    patch: &WorkspacePatch,
    applied: &[WorkspacePatch],
) -> Result<MergeOutcome> {
    let pre = parent_env.capture_state().await?;
    let mut conflicts = Vec::new();
    for op in &patch.ops {
        let path = match op {
            PatchOp::FileWrite {
                path, base_hash, ..
            } => {
                // parent's current state vs the branch-point base
                let cur_hash = pre.files.get(path).map(|b| fnv1a(b));
                match (cur_hash, base_hash) {
                    (Some(cur), Some(base)) if cur != *base => {
                        conflicts.push(format!(
                            "{path}: parent modified since branch (parent {cur:#x} vs base {base:#x})"
                        ));
                    }
                    (Some(_), None) => {
                        conflicts.push(format!(
                            "{path}: file created independently by parent and branch"
                        ));
                    }
                    _ => {}
                }
                path.clone()
            }
            PatchOp::FileDelete { path, base_hash } => {
                let cur_hash = pre.files.get(path).map(|b| fnv1a(b));
                if cur_hash != Some(*base_hash) {
                    conflicts.push(format!(
                        "{path}: parent modified since branch (deleted in branch)"
                    ));
                }
                path.clone()
            }
            PatchOp::RowUpsert {
                system,
                table,
                pk,
                base,
                ..
            } => {
                let key = (system.clone(), table.clone(), pk.clone());
                let cur = pre
                    .rows
                    .get(&key)
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                if cur != *base {
                    conflicts.push(format!("{system}.{table}:{pk}: row modified since branch"));
                }
                format!("{system}.{table}:{pk}")
            }
        };
        // sibling patches that already touched the same target
        for sibling in applied {
            for sop in &sibling.ops {
                let spath = match sop {
                    PatchOp::FileWrite { path, .. } | PatchOp::FileDelete { path, .. } => {
                        path.clone()
                    }
                    PatchOp::RowUpsert {
                        system, table, pk, ..
                    } => {
                        format!("{system}.{table}:{pk}")
                    }
                };
                if spath == *path {
                    conflicts.push(format!(
                        "{path}: conflicting subagent patches ({} also edited it)",
                        sibling.subagent
                    ));
                }
            }
        }
    }
    if !conflicts.is_empty() {
        return Ok(MergeOutcome::Conflict { conflicts });
    }
    parent_env.apply_patch(patch).await?;
    if !parent_env.sources_conserved() {
        // rollback: restore the captured pre-merge state
        parent_env.restore_state(&pre).await?;
        return Ok(MergeOutcome::Rejected {
            reason: "merge would violate source conservation; rolled back".into(),
        });
    }
    Ok(MergeOutcome::Applied {
        ops: patch.ops.len(),
    })
}

fn sanitize(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_ids() {
        assert_eq!(sanitize("task/with slashes"), "task_with_slashes");
        assert_eq!(sanitize("ok-id_1"), "ok-id_1");
    }
}
