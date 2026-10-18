# dsec-storage

[![CI](https://github.com/idrees2516/dsec-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/idrees2516/dsec-rs/actions)
[![crates.io](https://img.shields.io/crates/v/dsec-storage.svg)](https://crates.io/crates/dsec-storage)

Storage layer of the DSec reimplementation: the EROFS-style image
model, on-demand block loading, copy-on-write overlays, LRU block
caches and `pack_diff` working-state snapshots.

## What's inside

- **ErofsImage / ImageRegistry** — immutable, content-addressed images
  as 4 KiB block vectors; `ErofsImageBuilder::agent_base()` builds the
  shared agent rootfs the paper's sandboxes boot from.
- **OnDemandLoader** — fault-driven block fetch with a swappable
  latency model (zero for tests, paper-shaped for simulation) and an
  LRU cache mirroring the DAX page-cache dedup the paper measures.
- **OverlayDev** — copy-on-write dirty block tracking per sandbox.
- **LayeredImage** — the guest-visible filesystem (files, dirs,
  atomic-ish writes) over base + overlay.
- **pack_diff** — snapshots of dirty blocks + FS metadata + allocator
  cursor; `replica_from` materializes an identical working state
  without a full image copy (the paper's ~20-50x size reduction).
- **ArtifactStore** (`artifact`) — the hardened real-filesystem store
  (Phase 7): atomic temp-fsync-rename writes, SHA-256-verified reads,
  traversal-proof ids (component checks + canonical containment),
  hash-covered manifests. Attacker-tested: path traversal, symlink
  escape, content tampering, torn writes, manifest forgery and
  editing.

## Example

```rust
use dsec_storage::erofs::{ErofsImageBuilder, OnDemandLoader};
use dsec_storage::cache::LruBlockCache;
use dsec_storage::latency::LatencyModel;
use std::sync::Arc;
use std::time::Duration;

let image = Arc::new(ErofsImageBuilder::agent_base().build());
let loader = Arc::new(OnDemandLoader::new(
    image,
    Arc::new(LruBlockCache::new(1024)),
    LatencyModel::fixed(Duration::ZERO),
));
```

See `dsec-runtime` for the overlay + session layers built on top.
