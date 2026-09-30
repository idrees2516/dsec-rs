# dsec-firecracker

[![CI](https://github.com/idrees2516/dsec-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/idrees2516/dsec-rs/actions)
[![crates.io](https://img.shields.io/crates/v/dsec-firecracker.svg)](https://crates.io/crates/dsec-firecracker)

A **real Firecracker backend** for dsec-rs sandboxes — implementing
dsec-runtime's `MicrovmDriver` trait against the actual Firecracker
VMM Unix-socket API.

## What it does

- **boot** — launches `firecracker --api-sock <per-sandbox>`,
  configures machine shape from the sandbox spec
  (`cpu_millicores` → vCPUs, `mem_mib` → guest RAM), attaches the
  node's EROFS image as a **shared read-only rootfs drive** (the host
  page cache dedups it across VMs — the paper's shared image cache
  semantics) plus a per-VM writable scratch drive, then
  `InstanceStart`.
- **pause / resume** — `PATCH /vm {"state":"Paused"|"Resumed"}` —
  the paper's checkpoint windows, measured for real.
- **snapshot / restore** — `PUT /snapshots/create` with diff mode
  (the `pack_diff` analogue: base + dirty delta) and
  `PUT /snapshots/load` for replica fast-resume.
- **destroy** — graceful `SendCtrlAltDel` (paired with
  `reboot=t panic=1` boot args), then process reaping.

The HTTP/1.1 API client is dependency-free (`FcClient`); the whole
request sequencing, error propagation and EdgeNode integration are
tested against an in-process fake VMM (`FakeVmm`) over **real unix
sockets** — no KVM needed for CI. Real boots additionally require the
`firecracker` binary, a kernel image and `/dev/kvm`; `detect()`
reports what is missing. Override locations with
`DSEC_FIRECRACKER_PATH` and `DSEC_FC_KERNEL`.

## Example

```rust,no_run
use std::sync::Arc;
use dsec_firecracker::{FirecrackerConfig, FirecrackerDriver};
use dsec_runtime::{BackendFactory, EdgeNode};
use dsec_storage::erofs::ImageRegistry;
use dsec_storage::latency::NodeLatencyProfile;

# fn main() {
let registry = Arc::new(ImageRegistry::default());
let driver = Arc::new(FirecrackerDriver::new(
    FirecrackerConfig::default(),
    registry.clone(),
));
let factory = BackendFactory::new(registry, NodeLatencyProfile::zero(), 8)
    .with_microvm_driver(driver);
// EdgeNode::with_factory(...) boots real microVMs for
// BackendKind::Microvm specs; pause/resume/destroy route through the
// Firecracker API.
# }
```

Run the opt-in real e2e on a KVM host:
`DSEC_FIRECRACKER_E2E=1 cargo test -p dsec-firecracker --test integration`
