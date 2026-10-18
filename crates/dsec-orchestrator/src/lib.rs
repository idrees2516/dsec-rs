//! # dsec-orchestrator
//!
//! The **worker container** of DSec §6.2 — the scaffold-agnostic
//! control layer that manages agent sandboxes and drives rollouts —
//! implemented end to end:
//!
//! - [`model`] — a real OpenAI-compatible HTTP client with egress
//!   policy, retries, and idempotency keys (plus a deterministic
//!   scripted client and, behind the `server` feature, a mock model
//!   server for tests and drills).
//! - [`taskenv`] — the [`TaskEnvironment`] contract with two real
//!   backends: `dsec-agentenv` pods and `dsec-sdk` sandboxes on a local
//!   cluster. Both implement **idempotent tool calls** through a
//!   durable in-sandbox ledger, real pause/resume, isolated forks
//!   (in-memory pack_diff), and three-way-mergeable workspace patches.
//! - [`journal`] — the external orchestrator journal: fsync'd JSONL,
//!   SHA-256 hash-chained, tamper-evident, with transcript-rebuilding
//!   replay.
//! - [`worker`] — [`WorkerContainer`]: the durable rollout loop with
//!   crash-injection points at every dangerous boundary, pause/resume,
//!   failure handling, and recovery that never repeats side effects.
//! - [`subagent`] — [`SubagentManager`]: branch-safe subagents with
//!   isolated workspaces, patch return, merge validation with conflict
//!   and rollback semantics, bounded concurrency, and abandoned-branch
//!   reaping.
//!
//! Phased build (see `docs/phased-build.md`):
//! - Phase 1 — single agent, real environment, full lifecycle;
//! - Phase 2 — durable long-lived rollouts, journal, idempotent tools,
//!   snapshot/recovery;
//! - Phase 3 — branch-safe subagents;
//! - Phase 7 — journal tamper evidence, egress allowlist, attacker
//!   tests.
//!
//! The RL data path (TITO records, trajectory trees, execution-level
//! loss) lives in `dsec-rl`; the trainer, policy server, and elastic
//! GPU scheduler live in `dsec-train`.

pub mod error;
pub mod journal;
pub mod model;
pub mod subagent;
pub mod taskenv;
pub mod worker;

#[cfg(feature = "server")]
pub mod mockserver;

pub use error::{Error, Result};
pub use journal::{event_kind, replay, Journal, JournalEvent, ReplayReport, TaskReplayState};
pub use model::{
    ChatMessage, ChatRequest, ChatResponse, ModelClient, NetPolicy, OpenAiCompatClient,
    ScriptedModelClient, ToolCallOut,
};
pub use subagent::{MergeOutcome, SubagentManager, SubagentSpec};
pub use taskenv::{
    AgentEnvBackend, AgentEnvFactory, BranchBase, DsecSandboxBackend, EnvFactory, FileCheck,
    FileTask, PatchOp, TaskEnvironment, TaskSpec, WorkspacePatch,
};
pub use worker::{
    Crashpoint, TaskOutcome, WorkerConfig, WorkerContainer, WorkerRollout, WorkerState,
};
