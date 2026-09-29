#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("dimension mismatch: {what} (expected {expected}, got {actual})")]
    Dim {
        what: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("index {0} out of range ({1} slots)")]
    Index(usize, usize),

    #[error("action space mismatch: {0}")]
    ActionSpace(String),

    #[error("env pool error: {0}")]
    Pool(String),

    #[error("sandbox env error: {0}")]
    Sandbox(String),

    #[error("rl error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
