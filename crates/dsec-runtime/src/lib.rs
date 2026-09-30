//! # dsec-runtime
//!
//! Node-local sandbox runtime, the Rust re-creation of the DSec paper's
//! Edge + Aether + Chronus stack:
//!
//! - [`edge::EdgeNode`] — admission, lifecycle (create/pause/resume/
//!   destroy), resource governance, stats; serves the data plane.
//! - [`aether`] — per-sandbox multiplexing proxy over in-process channels
//!   or real Unix domain sockets.
//! - [`chronus`] — in-sandbox sessions: exec, filesystem, proxied HTTP,
//!   streaming I/O.
//! - [`backend`] — four sandbox backends (FnCall warm pool / Container /
//!   MicroVM / FullVM) with paper-calibrated latency profiles.
//! - [`state`] — CAS-guarded lifecycle state machine.
//! - [`resource`] — capacity accounting, SCHED_IDLE class, pause-time
//!   memory reclaim (balloon + `memory.reclaim` analogue).
//! - [`stats`] — latency histograms and counters for `/metrics`.

pub mod aether;
pub mod backend;
pub mod chronus;
pub mod edge;
pub mod error;
pub mod microvm;
pub mod resource;
pub mod state;
pub mod stats;

pub use aether::{AetherClient, StreamEvent};
pub use backend::{BackendFactory, BackendInstance, BackendKind, SandboxSpec};
pub use edge::{EdgeNode, NodeSummary, PauseOutcome, SandboxEntry};
pub use error::{Error, Result};
pub use microvm::{BoxFut, MicrovmDriver, MicrovmHandle, SimulatedMicrovmDriver, SnapshotPaths};
pub use resource::{CpuClass, NodeUsage, ResourcePool, ResourceRequest};
pub use state::SandboxState;
pub use stats::{EdgeStats, HistSnapshot};

// Re-export the storage pieces the SDK and RL layers consume.
pub use dsec_storage::latency::NodeLatencyProfile;
pub use dsec_storage::packdiff::DiffPack;
