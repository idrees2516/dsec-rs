//! Error taxonomy for the agentic environment harness.
//!
//! The single most important distinction in this crate mirrors the MiMo
//! Live RL reward contract: a **task failure** (the agent did badly — a
//! legitimate zero) is *never* confused with a **testbed failure** (the
//! environment or verifier broke — the rollout must be masked out of
//! training instead of trained as a false zero). See
//! [`RewardErrorKind`] and [`crate::verifier`] for the masking contract.

use thiserror::Error;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Library-level errors.
#[derive(Debug, Error)]
pub enum Error {
    /// A task row or manifest failed schema validation.
    #[error("invalid task spec: {0}")]
    InvalidSpec(String),

    /// A manifest referenced a source that does not exist.
    #[error("manifest source missing: {src:?} (resolved {resolved:?})")]
    MissingSource {
        /// Manifest-relative source path as written.
        src: String,
        /// Resolved absolute path.
        resolved: String,
    },

    /// The pod refused an operation in the wrong lifecycle state.
    #[error("pod {pod} is {state}; operation {op} not allowed")]
    BadLifecycle {
        /// Pod identifier.
        pod: String,
        /// Current state name.
        state: &'static str,
        /// Rejected operation.
        op: &'static str,
    },

    /// A command executed in a container failed.
    #[error("exec in {container} failed (rc={rc}): {output}")]
    ExecFailed {
        /// Container name.
        container: String,
        /// Return code.
        rc: i32,
        /// Captured output (truncated by the caller when huge).
        output: String,
    },

    /// A command exceeded its wall-clock budget.
    #[error("exec in {container} timed out after {timeout_sec}s: {command}")]
    ExecTimeout {
        /// Container name.
        container: String,
        /// Timeout budget, seconds.
        timeout_sec: u64,
        /// Command that timed out.
        command: String,
    },

    /// An MCP tool call failed.
    #[error("mcp tool {tool} failed: {message}")]
    ToolFailed {
        /// Tool name (`server.tool`).
        tool: String,
        /// Failure detail.
        message: String,
    },

    /// An MCP JSON-RPC exchange returned an error response.
    #[error("mcp rpc error {code}: {message}")]
    RpcError {
        /// JSON-RPC error code.
        code: i64,
        /// JSON-RPC error message.
        message: String,
    },

    /// The requested MCP server is not registered on the pod.
    #[error("unknown mcp server {0}")]
    UnknownServer(String),

    /// I/O error on a dataset file.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// JSON (de)serialization error.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    /// The HTTP server feature is not enabled.
    #[error("the axum-based AgentEnv server requires the `server` feature")]
    ServerFeatureMissing,
}

/// The reward-phase error taxonomy — a direct port of the MiMo/verl
/// `REWARD_TESTBED_CORRUPTED` categories.
///
/// Every variant means: *the zero reward is an artifact of broken
/// infrastructure, not of the policy*. The trainer must mask the whole
/// sequence instead of training on it. The variants and their names follow
/// the upstream harness (`reward/testbed_corrupted`, `judge_key_missing`,
/// `mcp_backend_down`, `missing_or_invalid_reward_json`,
/// `reward_out_of_range`, `verify_crashed`) so operators can grep the same
/// strings across the Python and Rust stacks.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardErrorKind {
    /// The LLM-judge credential was missing at reward time.
    JudgeKeyMissing,
    /// An MCP backend port the verifier depends on is dead (liveness probe
    /// failed before verification ran).
    McpBackendDead,
    /// The verifier entry script was not uploaded / does not exist.
    MissingRunVerify,
    /// `reward.json` is missing, unparseable, or has no `reward` key.
    MissingOrInvalidRewardJson,
    /// The parsed reward is not finite or outside `[0, 1]`.
    RewardOutOfRange,
    /// The task verifier crashed (traceback captured).
    VerifyCrashed,
    /// The judge endpoint was unreachable / errored mid-verification.
    JudgeUnavailable,
    /// Reward-phase transport to the pod died.
    RewardTransportDied,
}

impl RewardErrorKind {
    /// The upstream string identifier (`reward/<kind>` / the sentinel
    /// `reward/testbed_corrupted` when a raw category is unavailable).
    pub fn as_category(&self) -> &'static str {
        match self {
            Self::JudgeKeyMissing => "reward/testbed_corrupted",
            Self::McpBackendDead => "reward/testbed_corrupted",
            Self::MissingRunVerify => "reward/testbed_corrupted",
            Self::MissingOrInvalidRewardJson => "reward/testbed_corrupted",
            Self::RewardOutOfRange => "reward/testbed_corrupted",
            Self::VerifyCrashed => "reward/testbed_corrupted",
            Self::JudgeUnavailable => "reward/testbed_corrupted",
            Self::RewardTransportDied => "reward/testbed_corrupted",
        }
    }

    /// Fine-grained identifier used inside `reward_detail.json`.
    pub fn as_reward_error(&self) -> &'static str {
        match self {
            Self::JudgeKeyMissing => "judge_key_missing",
            Self::McpBackendDead => "mcp_backend_down",
            Self::MissingRunVerify => "missing_run_verify",
            Self::MissingOrInvalidRewardJson => "missing_or_invalid_reward_json",
            Self::RewardOutOfRange => "reward_out_of_range",
            Self::VerifyCrashed => "verify_crashed",
            Self::JudgeUnavailable => "judge_unavailable",
            Self::RewardTransportDied => "reward_transport_died",
        }
    }
}

/// The rollout-phase error taxonomy (verl `AgentLoop` infra categories).
///
/// These distinguish *why* a rollout produced no trainable trajectory:
/// environment setup failed, the pod connection broke, the sequence
/// wall-clock expired, or the reward phase itself failed.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InfraErrorKind {
    /// Env/pod creation failed or timed out (upstream `setup/failed`).
    SetupFailed,
    /// The exec stream broke mid-trajectory.
    PodConnTimeout,
    /// The trajectory wall-clock budget expired.
    SeqTimeout,
    /// `calculate_reward` raised.
    RewardException,
    /// Env-actor-side reward failure.
    RewardEnvError,
    /// Reward-phase transport died (the masking contract).
    RewardTestbedCorrupted,
}

impl InfraErrorKind {
    /// Upstream category string.
    pub fn as_category(&self) -> &'static str {
        match self {
            Self::SetupFailed => "setup/failed",
            Self::PodConnTimeout => "rollout/pod_conn_timeout",
            Self::SeqTimeout => "rollout/seq_timeout",
            Self::RewardException => "reward/exception",
            Self::RewardEnvError => "reward/env_error",
            Self::RewardTestbedCorrupted => "reward/testbed_corrupted",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_error_kinds_roundtrip_and_name_stably() {
        let kinds = [
            RewardErrorKind::JudgeKeyMissing,
            RewardErrorKind::McpBackendDead,
            RewardErrorKind::MissingRunVerify,
            RewardErrorKind::MissingOrInvalidRewardJson,
            RewardErrorKind::RewardOutOfRange,
            RewardErrorKind::VerifyCrashed,
            RewardErrorKind::JudgeUnavailable,
            RewardErrorKind::RewardTransportDied,
        ];
        for k in &kinds {
            let s = serde_json::to_string(k).unwrap();
            let back: RewardErrorKind = serde_json::from_str(&s).unwrap();
            assert_eq!(&back, k);
            assert_eq!(k.as_category(), "reward/testbed_corrupted");
            assert!(!k.as_reward_error().is_empty());
        }
    }

    #[test]
    fn infra_categories_match_upstream_strings() {
        assert_eq!(InfraErrorKind::SetupFailed.as_category(), "setup/failed");
        assert_eq!(
            InfraErrorKind::RewardTestbedCorrupted.as_category(),
            "reward/testbed_corrupted"
        );
    }
}
