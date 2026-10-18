//! Errors for the training crate.

use thiserror::Error;

/// Result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Training errors.
#[derive(Debug, Error)]
pub enum Error {
    /// Bad configuration or usage.
    #[error("config: {0}")]
    Config(String),
    /// Policy version contract violation.
    #[error("policy: {0}")]
    Policy(String),
    /// A transition was refused (draining, barrier closed, ...).
    #[error("transition: {0}")]
    Transition(String),
    /// Serialization failure.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// I/O failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
