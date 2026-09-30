# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Inline-first-poll fast path for Aether request handling (poll each
  request future once in the reader task; spawn only futures that
  actually suspend) — the next structural step past §13 of
  `docs/performance.md`.
- io_uring transport for the Aether UDS data plane.
- Jailer integration (chroot/uid/cgroups) for `dsec-firecracker`.
- containerd backend and EROFS loop-mount images behind the same
  `MicrovmDriver`-style traits.
- 3FS-style distributed storage backend behind the `dsec-storage` traits.
- Fuzz targets for the Aether codec (`cargo fuzz`).

## [0.3.0] - 2026-09-30

### Added

- **`dsec-firecracker` crate — a real Firecracker microVM backend.**
  Implements the new `MicrovmDriver` trait: launches
  `firecracker --api-sock`, configures machine shape from the sandbox
  spec, attaches the EROFS image as a shared read-only rootfs drive
  (page-cache dedup across VMs = the paper's shared image cache) plus a
  per-VM scratch drive, and starts the instance; pause/resume use the
  real VMM state machine; snapshots map to `PUT /snapshots/create`
  (diff mode = the pack_diff analogue) and restore to
  `PUT /snapshots/load`; destroy is graceful `SendCtrlAltDel` + reap.
  Dependency-free HTTP/1.1 API client, host capability detection with
  honest blockers (`DSEC_FIRECRACKER_PATH`, `DSEC_FC_KERNEL`), and a
  fake VMM (`FakeVmm`) so the full request sequencing and `EdgeNode`
  integration are tested over real unix sockets without KVM. Opt-in
  real e2e: `DSEC_FIRECRACKER_E2E=1`.
- **Pipelined batch data plane.** `AetherClient::call_batch` issues N
  requests as ONE coalesced transmission (`AetherWriter::send_batch`:
  one `write_all` for the UDS transport) awaited under ONE deadline;
  `AetherReader::recv_many` drains whole batches per wake with a
  persistent UDS read buffer; both reader loops complete calls under
  one lock pass. `AetherClient::call_stats` exposes aggregate call
  timing for I/O-boundness diagnostics.
- **`BatchedSandboxEnvPool`** — sandbox envs stepped as one pipelined
  tick (pure command selection → one `call_batch` → pure folding;
  phase-barrier pipelined `reset_if_done` staging). Byte-identical
  sequences to the per-env path, enforced by cross-validation tests.
  The training driver now accepts any `Stepping` pool
  (`Box<dyn Stepping>`).
- `dsec-profiling` crate (unpublished): pprof/SIGPROF flamegraph
  harness + server-side exec micro-bench for the sandbox-env path.
- Chronus interpreter supports `&&` compound commands with POSIX
  `sh -c` semantics (episode staging is one round trip, not eight).
- `EdgeNode::with_factory` — injection point for real VMM drivers.

### Changed

- **Sandbox-env stepping: 14.1k (v0.1.0) / ~19k (v0.2.0) → ~35k
  steps/s pipelined** (2.5x the base case; the per-env reference path
  itself improved 1.45x from `&&` staging). Flamegraph evidence and the
  full diagnosis→fix→result chain in `docs/performance.md` §10-13:
  the path was I/O-bound (per-step wakeup chains, ~90% transport
  overhead vs. a measured 2.12us server-side exec cost).
- All 8 public crates are `crates.io`-publish-ready (per-crate
  metadata, READMEs, versioned path deps) with a release workflow
  (`.github/workflows/publish.yml`) and `docs/publishing.md`.

## [0.2.0] - 2026-09-30

### Changed

- **Performance campaign** (full methodology and A/B evidence in
  `docs/performance.md`; all claims from interleaved runs against the
  v0.1.0 binary):
  - `EnvPool` persistent worker threads replace per-step scoped-thread
    spawning: vectorized stepping 599k -> 1.42M env steps/s (2.4x), now
    above PufferLib's ~1M reference. Generation/payload atomicity lives
    under one mutex; a 200-step soak test pins per-env ordering.
  - GAE processed in 8-env blocks with fused write-back (strided cache
    misses eliminated): 22.6M -> 117M transitions/s (5.2x).
  - CRC-32 via `crc32fast` (PCLMULQDQ folding; the SSE4.2 `crc32`
    instruction is CRC-32C and was deliberately NOT used — wrong
    polynomial), copy-free `Frame::decode_parts`, single-allocation
    `read_frame`: codec 419/310 -> 888/892 MB/s (2.1-2.9x).
  - `DiffPack.files` shared as `Arc<[FileWire]>` with `Arc<str>` paths;
    `LayeredImage` caches the wire snapshot by version and `apply_diff`
    adopts the pack's table; `replica_from` constructs replicas without
    re-interning base paths; `take_dirty_blocks` snapshots without cloning
    the CoW map: pack_diff snapshot 48k -> 8.9M packs/s (183x), apply
    8.6k -> 34.9k/s (4.1x).
  - Control plane: incremental per-project usage index (O(1)
    `project_usage`, cross-validated against the full scan), in-place
    `adjust_node_resources` (no NodeInfo clone, identical events),
    `sandbox_count()`, allocation-free IAM subtree checks: creation
    +35-123%, burst lifecycle +18-22% at 100k (0 errors).
  - Placement: Floyd's k-sample (O(k^2), no 0..n materialization) and a
    single RNG critical section; throughput par, k-choice quality
    unchanged at 67.65%.
  - `apiserver_rps_keepalive` benchmark (pipelined persistent connections,
    the real DsecClient transport): 129k RPS, 6.1x the cold-connection
    path. Creation latency now returns via join handles; burst peak
    tracking is O(1).
  - `ReplayBuffer::enqueue` bulk-memcpy experiment measured ~20% slower
    than the compiler-vectorized per-env loop on this hardware class and
    was reverted (documented so it is not retried blind).
  - Paper boundary conditions preserved: latency anchors (pause ~4 s,
    FnCall ~5 ms) untouched; placement, quota, event, and RL semantics
    regression-tested.

### Added

- `Rng::sample_k_floyd`, `Frame::decode_parts`, `CRC32::compute_parts`,
  `Registry::sandbox_count` / `adjust_node_resources`, `Iam::project_exists`,
  `LayeredImage::replica_from`, `OverlayDev::take_dirty_blocks`.
- Tests: Floyd sampling invariants, env-pool soak (200 steps x 4 workers),
  usage-index vs full-scan cross-validation, CRC split-equals-contiguous.

## [0.1.0] - 2026-09-29

### Added

- **dsec-protocol**: Aether wire framing (25-byte header + CRC32),
  request/response codec over `AsyncRead`/`AsyncWrite`, seeded splitmix64
  PRNG for deterministic lossy-transport simulation, multiplexing channels.
- **dsec-storage**: EROFS-style immutable images with on-demand fault-driven
  loading; node-level LRU block cache (Arc-shared blocks model the DAX page
  cache dedup across sandboxes); overlay CoW devices; layered guest
  filesystems; `pack_diff` incremental snapshots (dirty blocks + FS metadata
  + allocator cursor); semaphore-bounded async data pipeline; zero-copy
  shared regions for torch-style shared-memory handoff.
- **dsec-runtime**: Edge node admission control and CAS-based lifecycle
  state machine with pause (60% memory reclaim, balloon/DAMON model) and
  resume (refetch + prefetch); Aether server/client over in-process channels
  **and real Unix domain sockets**, with pipelined request dispatch and
  paced stream frames; Chronus session layer (command interpreter with
  redirection, filesystem ops, fake egress HTTP router, streaming I/O);
  four sandbox backends (FnCall warm pool, Container, MicroVM, FullVM) with
  paper-anchored latency profiles.
- **dsec-control**: IAM with nested projects and hierarchical quota rollup,
  RBAC tokens; versioned registry with broadcast event log; k-choice
  placement engine (packing / fragmentation / locality / topology / cost
  scoring, cloud burst, seeded tie-break jitter); heartbeat watcher with
  TTL eviction and preemption (paused-first, low-priority-first); axum REST
  apiserver (12 routes); Prometheus metrics endpoint; token-bucket rate
  limiting.
- **dsec-sdk**: full libdsec port — `DsecClient`, `Sandbox`
  (execute/streams/fs/http/pause/resume/destroy), `SandboxPool` with reset
  semantics, hand-rolled HTTP/1.1 client with backoff retry, channel + UDS
  transports, and `LocalCluster` one-call full-stack assembly.
- **dsec-rl**: PufferLib core port — step-major episode ring `ReplayBuffer`
  with per-step LSTM (h, c) storage and reset-on-done invariant (with
  repair), mat-obs/flat-obs views, GAE with episode-boundary restart,
  sequence sampling with reset-correct h0/c0, zero-copy dump;
  `EnvPool` with `reset_if_done` and scoped-thread parallel stepping;
  rollout `Driver` with input-state snapshot and zero-on-done; `LstmCell`;
  terminal-task environments that execute real sandbox sessions per step.
- **dsec-bench**: 17 benchmarks with JSON + markdown reporting.
- Examples: `control_plane_demo` (IAM → quota → REST → exec → cross-node
  packdiff → pause → stream → pool), `training_loop` (sandbox envs + buffer
  + LSTM reset verification), `burst_simulation` (100k lifecycles).
- CI: rustfmt check, clippy with `-D warnings`, full test matrix on stable
  and MSRV (1.85), docs build.

### Fixed

- Admission-control race: `ResourcePool::try_acquire` used a
  read-check-then-add pattern that allowed transient oversubscription under
  full contention; replaced with lock-free optimistic
  reserve-verify-rollback admission (regression test included).
