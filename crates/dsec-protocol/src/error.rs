use std::io;

/// Error taxonomy shared by every dsec-rs crate, aligned with the
/// error-code table exposed by the apiserver and Aether.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("protocol violation: {message} (code {code})")]
    Protocol { code: u32, message: String },

    #[error("resource not found: {0}")]
    NotFound(String),

    #[error("invalid sandbox state transition: {0}")]
    InvalidState(String),

    #[error("quota exceeded: {0}")]
    QuotaExceeded(String),

    #[error("unauthorized: {0}")]
    Unauthorized(String),

    #[error("rate limited: {0}")]
    RateLimited(String),

    #[error("operation timed out: {0}")]
    Timeout(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("service unavailable: {0}")]
    Unavailable(String),

    #[error("internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Stable wire error code, matching `message::ErrorCode`.
    pub fn code(&self) -> u32 {
        use crate::message::ErrorCode as E;
        match self {
            Error::Io(_) | Error::Serde(_) => E::Internal as u32,
            Error::Protocol { code, .. } => *code,
            Error::NotFound(_) => E::NotFound as u32,
            Error::InvalidState(_) => E::InvalidState as u32,
            Error::QuotaExceeded(_) => E::QuotaExceeded as u32,
            Error::Unauthorized(_) => E::Unauthorized as u32,
            Error::RateLimited(_) => E::RateLimited as u32,
            Error::Timeout(_) => E::Timeout as u32,
            Error::InvalidArgument(_) => E::InvalidArgument as u32,
            Error::Unavailable(_) => E::Unavailable as u32,
            Error::Internal(_) => E::Internal as u32,
        }
    }
}
