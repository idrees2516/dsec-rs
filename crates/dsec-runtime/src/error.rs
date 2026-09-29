use std::io;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error(transparent)]
    Protocol(#[from] dsec_protocol::Error),

    #[error(transparent)]
    Storage(#[from] dsec_storage::Error),

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("sandbox {0} not found on this node")]
    SandboxNotFound(u64),

    #[error("invalid state transition: {from} -> {to} for sandbox {sid}")]
    InvalidTransition {
        sid: u64,
        from: &'static str,
        to: &'static str,
    },

    #[error("node at capacity: {needed:?} needed, {available:?} available")]
    AtCapacity { needed: String, available: String },

    #[error("image {0} not present on node")]
    ImageNotLocal(String),

    #[error("sandbox {0} is paused; operation rejected")]
    SandboxPaused(u64),

    #[error("project {0} not admitted to node")]
    ProjectNotAdmitted(String),

    #[error("session {0} not found")]
    SessionNotFound(u32),

    #[error("command timed out after {0} ms")]
    ExecTimeout(u64),

    #[error("stream {0} not found or closed")]
    StreamNotFound(u32),

    #[error("runtime error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
