//! # dsec-firecracker
//!
//! A **real Firecracker backend** for [dsec-rs](https://docs.rs/dsec-runtime)
//! sandboxes, implementing dsec-runtime's
//! [`MicrovmDriver`](dsec_runtime::microvm::MicrovmDriver) trait.
//!
//! What it does behind the paper's backend abstractions:
//!
//! * **boot** — launches a `firecracker` process with a per-sandbox API
//!   socket, configures the machine shape from the sandbox spec
//!   (`cpu_millicores` → vCPUs, `mem_mib` → guest RAM), attaches the
//!   node's EROFS image as a **shared read-only rootfs drive** (the
//!   host page cache dedups it across VMs — the paper's shared image
//!   cache semantics) plus a per-VM writable scratch drive, and starts
//!   the instance.
//! * **pause / resume** — the real VMM state machine
//!   (`PATCH /vm {"state":"Paused"|"Resumed"}`).
//! * **snapshot / restore** — `PUT /snapshots/create` (diff mode is the
//!   `pack_diff` analogue: base + dirty delta) and
//!   `PUT /snapshots/load` for replicas / fast resume.
//! * **destroy** — graceful `SendCtrlAltDel` (paired with
//!   `reboot=t panic=1` boot args) then process reaping.
//!
//! The wire behavior is exercised end-to-end against an in-process fake
//! VMM ([`fake::FakeVmm`]) over real unix sockets, so the request
//! sequencing, error propagation and lifecycle wiring are fully tested
//! without KVM; driving a real `firecracker` additionally requires the
//! binary, a kernel image and `/dev/kvm` (probed by
//! [`config::detect`], reported clearly otherwise). Set
//! `DSEC_FIRECRACKER_PATH` and `DSEC_FC_KERNEL` to override locations.
//!
//! ## Wire into an EdgeNode
//!
//! ```no_run
//! use std::sync::Arc;
//! use dsec_firecracker::{FirecrackerConfig, FirecrackerDriver};
//! use dsec_runtime::{BackendFactory, EdgeNode};
//! use dsec_storage::erofs::ImageRegistry;
//! use dsec_storage::latency::NodeLatencyProfile;
//!
//! # fn main() {
//! let registry = Arc::new(ImageRegistry::default());
//! let driver = Arc::new(FirecrackerDriver::new(
//!     FirecrackerConfig::default(),
//!     registry.clone(),
//! ));
//! let factory = BackendFactory::new(
//!     registry,
//!     NodeLatencyProfile::zero(),
//!     8,
//! )
//! .with_microvm_driver(driver);
//! // EdgeNode::new(...) with this factory boots real microVMs for
//! // BackendKind::Microvm specs and routes pause/resume/destroy
//! // through the Firecracker API.
//! # }
//! ```

pub mod client;
pub mod config;
pub mod driver;
pub mod fake;
pub mod rootfs;

pub use client::{FcClient, FcResponse};
pub use config::{detect, Capability, FirecrackerConfig, FIRECRACKER_PATH_ENV, KERNEL_PATH_ENV};
pub use driver::{FakeLauncher, FirecrackerDriver, ProcessLauncher, VmmLauncher};
pub use fake::{FakeVmm, RequestLog};
pub use rootfs::{materialize_image, RootfsCache};
