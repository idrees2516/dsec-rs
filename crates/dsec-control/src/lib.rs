//! # dsec-control
//!
//! The DSec control plane, re-created in Rust:
//!
//! - [`iam`] — nested projects (`root/team/sub`) with hierarchical quota
//!   enforcement, tokens and RBAC roles.
//! - [`registry`] — versioned cluster state with an event log (the
//!   in-process stand-in for the paper's etcd-backed store).
//! - [`placement`] — k-choice (power-of-two-choices) placement engine
//!   with hard filters, packing/fragmentation/locality/cost scoring and
//!   cloud bursting.
//! - [`watcher`] — heartbeat TTL health tracking, node eviction and
//!   preemption policy (paused-first, lowest-priority-first).
//! - [`control::ControlPlane`] — the service core every API call flows
//!   through; provisioner attachment via the [`control::Provisioner`]
//!   trait.
//! - [`apiserver`] — stateless REST surface (axum) with bearer auth and
//!   rate limiting.
//! - [`metrics`] / [`ratelimit`] — Prometheus text exposition and
//!   token-bucket limiting.

pub mod apiserver;
pub mod control;
pub mod error;
pub mod iam;
pub mod metrics;
pub mod model;
pub mod placement;
pub mod ratelimit;
pub mod registry;
pub mod watcher;

pub use control::{ClusterSummary, ControlPlane, ProvisionedSandbox, Provisioner};
pub use error::{BoxFuture, Error, Result};
pub use watcher::{Watcher, WatcherConfig};
