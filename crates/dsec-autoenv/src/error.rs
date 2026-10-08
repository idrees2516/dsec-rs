//! Error taxonomy for the AutoEnvScaling flywheel.
//!
//! The paper separates three kinds of failure that must never be conflated
//! (Section 3.2): a *broken environment* (validation fails, proposer reward
//! `-1`), a *working environment that does not suit the current solver*
//! (calibration fails, proposer reward `-0.25`), and *host misconfiguration*
//! (our own bug — surfaced as [`enum@Error`]). See [`crate::reward`] for the
//! reward-side encoding of the first two.

use thiserror::Error;

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced by the flywheel machinery.
#[derive(Debug, Error)]
pub enum Error {
    /// An environment failed admission. Carries every failed check so the
    /// proposer can revise (Appendix G.4: failed environments "go back to
    /// the proposer for revision").
    #[error("environment {env_id} rejected: {reasons:?}")]
    EnvironmentRejected {
        /// Identifier of the rejected environment.
        env_id: String,
        /// Human-readable failure reasons, one per failed check.
        reasons: Vec<String>,
    },

    /// The proposer attempted to modify a region of its workspace that is
    /// fixed during generation (assignment, sandbox rules, authoring
    /// instructions), or to touch the admission checks (which live outside
    /// the sandbox entirely). Section 3.1: "The proposer can edit its memory
    /// and tools but not the instructions or the admission checks, which run
    /// outside the sandbox."
    #[error("sandbox contract violated: {path} is {region}; {detail}")]
    SandboxViolation {
        /// The path that was targeted.
        path: String,
        /// The region classification of that path.
        region: String,
        /// What the proposer tried to do.
        detail: String,
    },

    /// An action attempted to use a tool that the runtime preamble does not
    /// list. The auto_env_scaling skill: "Web search — ONLY when the runtime
    /// preamble explicitly lists a web tool (`web_search`/`web_fetch` or
    /// `web_search.py`). If no such preamble is present, you have no web
    /// access; do not attempt searches."
    #[error("tool {tool} is not available in this sandbox: {detail}")]
    ToolUnavailable {
        /// The tool name that was invoked.
        tool: String,
        /// Why it is unavailable.
        detail: String,
    },

    /// An egress rule blocked a network access (Terminal-Bench and other
    /// held-out benchmark pages). Section 3.1: "egress rules block
    /// Terminal-Bench and other benchmark pages."
    #[error("egress denied to {host}: {rule}")]
    EgressDenied {
        /// Host that was contacted.
        host: String,
        /// The blocking rule.
        rule: String,
    },

    /// A Harbor task file failed schema validation.
    #[error("invalid harbor task: {0}")]
    InvalidTask(String),

    /// A serialized structure failed to round-trip.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// An invariant of the flywheel state machine was violated.
    #[error("flywheel invariant violated: {0}")]
    Invariant(String),
}

impl Error {
    /// Convenience constructor for [`Error::Invariant`].
    pub fn invariant(msg: impl Into<String>) -> Self {
        Error::Invariant(msg.into())
    }

    /// Convenience constructor for [`Error::InvalidTask`].
    pub fn invalid_task(msg: impl Into<String>) -> Self {
        Error::InvalidTask(msg.into())
    }

    /// Convenience constructor for [`Error::SandboxViolation`].
    pub fn sandbox_violation(
        path: impl Into<String>,
        region: &str,
        detail: impl Into<String>,
    ) -> Self {
        Error::SandboxViolation {
            path: path.into(),
            region: region.to_string(),
            detail: detail.into(),
        }
    }
}
