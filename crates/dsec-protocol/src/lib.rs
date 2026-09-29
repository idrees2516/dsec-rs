//! # dsec-protocol
//!
//! Wire protocol spoken between the `dsec-sdk` client library and the
//! node-local runtime (`Edge` -> `Aether` -> `Chronus`). It mirrors the
//! Aether per-sandbox proxy described in the DSec paper: a single
//! multiplexed connection per node carrying framed requests for many
//! sandboxes, each demultiplexed into per-sandbox shell sessions.
//!
//! Layout:
//! - [`frame`] — binary framing (21-byte header, length-prefixed payload, CRC32)
//! - [`message`] — request/response message types (serde)
//! - [`codec`] — async encode/decode over any `AsyncRead`/`AsyncWrite`
//! - [`rng`] — shared deterministic splitmix64 PRNG for reproducible simulation
//!
//! The transport itself is intentionally abstract: the same codec runs over
//! an in-process channel pair (deterministic simulation / tests) or a real
//! UDS socket (production data plane), exactly as Aether switches between
//! UDS and vsock in the paper.

pub mod codec;
pub mod error;
pub mod frame;
pub mod message;
pub mod rng;

pub use error::{Error, Result};
pub use frame::{Frame, FrameHeader, CRC32};
pub use message::{Request, Response};

/// Maximum payload accepted by the codec (16 MiB), guards against
/// malformed or hostile peers allocating unbounded memory.
pub const MAX_PAYLOAD: usize = 16 * 1024 * 1024;

/// Wire magic byte.
pub const MAGIC: u8 = 0xD5;
/// Wire protocol version.
pub const VERSION: u8 = 1;
