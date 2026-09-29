use std::future::Future;
use std::pin::Pin;

/// Boxed future used by the `Provisioner` trait (object-safe async).
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("project {0} already exists")]
    ProjectExists(String),

    #[error("project {0} not found")]
    ProjectNotFound(String),

    #[error("parent project {0} not found")]
    ParentNotFound(String),

    #[error("quota exceeded in {project}: {what}")]
    QuotaExceeded { project: String, what: String },

    #[error("sandbox {0} not found")]
    SandboxNotFound(u64),

    #[error("node {0} not found")]
    NodeNotFound(String),

    #[error("no node satisfies the request (spec: {0})")]
    NoCandidate(String),

    #[error("unauthorized: {0}")]
    Unauthorized(String),

    #[error("rate limited: retry after {0} ms")]
    RateLimited(u64),

    #[error("provisioning failed on {node}: {message}")]
    ProvisionFailed { node: String, message: String },

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("node {node} is {status}")]
    NodeUnhealthy { node: String, status: String },

    #[error("control plane error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, Error>;
