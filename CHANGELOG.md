# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Optional real-backend feature flags: Firecracker MicroVM (via `firecracker`
  API), containerd, and EROFS loop-mount images.
- 3FS-style distributed storage backend behind the `dsec-storage` trait.
- Cloud-burst provider simulation with cost accounting dashboards.
- Fuzz targets for the Aether codec (`cargo fuzz`).

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
