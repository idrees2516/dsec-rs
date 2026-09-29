use std::io;

use dsec_control::Error as ControlError;
use dsec_protocol::Error as ProtocolError;
use dsec_runtime::Error as RuntimeError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("http client error: {status} {message}")]
    Http { status: u16, message: String },

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error(transparent)]
    Protocol(#[from] ProtocolError),

    #[error(transparent)]
    Control(#[from] ControlError),

    #[error(transparent)]
    Runtime(#[from] RuntimeError),

    #[error("sandbox {0} is not usable: {1}")]
    SandboxUnusable(u64, String),

    #[error("pool exhausted: {0}")]
    PoolExhausted(String),

    #[error("retry budget exhausted after {attempts} attempts: {message}")]
    RetriesExhausted { attempts: u32, message: String },
}

pub type Result<T> = std::result::Result<T, Error>;
