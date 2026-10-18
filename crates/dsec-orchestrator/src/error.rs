//! Errors for the orchestrator crate.

use thiserror::Error;

/// Result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Orchestrator errors.
#[derive(Debug, Error)]
pub enum Error {
    /// Journal I/O or integrity failure.
    #[error("journal: {0}")]
    Journal(String),
    /// Journal hash-chain verification failed.
    #[error("journal tampered at seq {seq}: {detail}")]
    JournalTampered {
        /// First sequence number that failed verification.
        seq: u64,
        /// What failed.
        detail: String,
    },
    /// Model endpoint failure (after retries).
    #[error("model {model}: {message}")]
    Model {
        /// Model name / endpoint label.
        model: String,
        /// Failure description.
        message: String,
    },
    /// Egress blocked by the network policy.
    #[error("net policy denied {host}:{port}: {reason}")]
    NetPolicyDenied {
        /// Host that was attempted.
        host: String,
        /// Port that was attempted.
        port: u16,
        /// Why it was denied.
        reason: String,
    },
    /// Environment (sandbox) failure.
    #[error("env: {0}")]
    Env(String),
    /// Task-level failure (task failed, not infra).
    #[error("task {task}: {message}")]
    Task {
        /// Task id.
        task: String,
        /// Failure description.
        message: String,
    },
    /// Subagent merge conflict.
    #[error("merge conflict: {0}")]
    MergeConflict(String),
    /// A crashpoint was armed and fired — the worker "crashed".
    #[error("crash injected at {point:?}")]
    CrashInjected {
        /// Where the crash fired.
        point: crate::worker::Crashpoint,
    },
    /// Bad configuration or usage.
    #[error("config: {0}")]
    Config(String),
    /// JSON (de)serialization failure.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// Underlying dsec-sdk failure.
    #[error("sdk: {0}")]
    Sdk(#[from] dsec_sdk::Error),
    /// Underlying dsec-agentenv failure.
    #[error("agentenv: {0}")]
    AgentEnv(#[from] dsec_agentenv::error::Error),
    /// I/O failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
