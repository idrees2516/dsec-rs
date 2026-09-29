use std::io;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("block {0} out of range (device has {1} blocks)")]
    BlockOutOfRange(u64, u64),

    #[error("path {0:?} not found")]
    PathNotFound(String),

    #[error("path {0:?} is not a directory")]
    NotADirectory(String),

    #[error("path {0:?} is not a file")]
    NotAFile(String),

    #[error("path {0:?} already exists")]
    PathExists(String),

    #[error("invalid path {0:?}: must be absolute and normalized")]
    InvalidPath(String),

    #[error("checksum mismatch for image {0}: expected {1}, got {2}")]
    DigestMismatch(String, String, String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("storage error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
