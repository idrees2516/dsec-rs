# dsec-control

[![CI](https://github.com/idrees2516/dsec-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/idrees2516/dsec-rs/actions)
[![crates.io](https://img.shields.io/crates/v/dsec-control.svg)](https://crates.io/crates/dsec-control)

The management plane of the DSec reimplementation: a stateless REST
apiserver over nested-project IAM, the k-selection placement engine,
and the Watcher heartbeat/eviction/preemption loop.

## What's inside

- **IAM** — nested projects with quotas (CPU/memory/sandbox slots),
  per-project tokens, zero-allocation authorize checks; usage tracking
  through an O(1) incremental index (cross-validated against full
  scans in tests).
- **ControlPlane** — `create_sandbox` (quota check → placement → node
  admission → Aether bootstrap), `destroy`, `pause`, `resume`,
  snapshots. Runs on any axum 0.8 router via `router()`.
- **Placement** — the paper's k-candidates selection: score every
  admissible node (affinity/anti-affinity, load, image locality),
  sample k via Floyd's algorithm, pick the best — 124k decisions/s at
  1,024 nodes on one core.
- **Watcher** — node heartbeats, dead-node eviction (sandboxes
  rescheduled), preemption of low-priority work when capacity is
  contended.

## Example

```rust,no_run
use dsec_control::control_plane::ControlPlaneBuilder;
# fn main() {} // node registration omitted; see the repo's examples
```

`dsec-sdk` is the client side of this plane; `dsec-bench` measures it
(20k+ RPS keep-alive on one core).
