# dsec-rs

**A deep Rust reimplementation of [DeepSeek Elastic Compute (DSec)](docs/paper-mapping.md) —
the sandbox infrastructure for agentic RL training at scale — together with a
Rust port of PufferLib's core.**

Built from the paper *"DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure
for Effective Agentic Training at Scale"* (DeepSeek-AI + Tsinghua, 2026):
every component the paper describes is re-created in Rust as a deterministic
userspace simulation that builds and runs anywhere — no root, no containers,
no Firecracker, fully reproducible (seeded latency + PRNG).

[![CI](https://github.com/idrees2516/dsec-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/idrees2516/dsec-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust)](Cargo.toml)
[![Tests](https://img.shields.io/badge/tests-161%20passing-brightgreen)](#running-the-tests)
[![LOC](https://img.shields.io/badge/lines%20of%20Rust-~13k-blueviolet)](crates)

---

## Why

DSec runs **100k+ concurrent sandboxes** and **~5,000 sandbox creations per
second** to train agents that interact with real tooling. The paper's stack is
mixed-language: 3FS (C++), EROFS (kernel C), Firecracker (Rust), the
ublk/OverlayBD user-space block layer (Rust, open-sourced at
[kvcache-ai/AgentENV](https://github.com/kvcache-ai/AgentENV)) — and Python
for everything user-facing (libdsec SDK, control plane glue, PufferLib RL
integration).

**This workspace re-implements the Python-side architecture in Rust** — the
same role PufferLib plays for the training loop — while faithfully modeling
the storage, isolation, and scheduling layers in userspace. The result is a
single-language, dependency-light, fully deterministic test bed for the
paper's ideas, suitable for research, teaching, and as a reference for
production ports.

## Architecture

![dsec-rs architecture](docs/architecture.png)

Two planes, exactly as in the paper: a **management plane** (stateless REST
apiserver, IAM, placement, watcher) and a **data plane** (Aether frame
multiplexing over UDS/vsock-style transports into per-node Edge runtimes).

| crate            | role                                                                                                        | paper component                                        |
|------------------|-------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| [`dsec-protocol`](crates/dsec-protocol) | Aether wire protocol: frames, CRC32, request/response codec, multiplexing channels                           | Aether data plane                                      |
| [`dsec-storage`](crates/dsec-storage)   | EROFS-style images, on-demand block loading, node LRU page cache, overlay CoW devices, `pack_diff` snapshots, async prefetch pipeline, zero-copy shared regions | storage layer (EROFS / 3FS / OverlayBD / ublk)         |
| [`dsec-runtime`](crates/dsec-runtime)   | Edge lifecycle state machine, Aether server/client (channel + UDS transports), Chronus sessions (exec / fs / http / streaming), four backends (FnCall pool, Container, MicroVM, FullVM), resource governance with pause-time reclaim | Edge, Aether, Chronus, sandbox backends                |
| [`dsec-control`](crates/dsec-control)   | IAM with nested project quotas, versioned registry + event log, k-choice placement engine with cloud bursting, heartbeat watcher with eviction + preemption, REST apiserver (axum), Prometheus metrics, rate limiting | apiserver, IAM, Placement Engine, Watcher              |
| [`dsec-sdk`](crates/dsec-sdk)           | **libdsec port**: `DsecClient`, `Sandbox` (execute/filesystem/http/streams/pause/resume), `SandboxPool`, retry with backoff, channel+UDS transports, full-stack `LocalCluster` assembly | libdsec                                                |
| [`dsec-rl`](crates/dsec-rl)             | **PufferLib core port**: episode ring replay buffer with LSTM hidden-state reset, vectorized env pools (parallel stepping), mat-obs/flat-obs, GAE, LSTM cell, training driver, agent sandboxes as RL envs | RL co-design layer                                     |
| [`dsec-bench`](crates/dsec-bench)       | Benchmark suite: creation rate vs the paper's ~5k/s, 100k burst, pause/resume latency, RL steps/sec vs PufferLib, pack_diff savings, placement throughput/quality, HTTP RPS, codec throughput | —                                                      |

## Quick start

```bash
git clone https://github.com/idrees2516/dsec-rs.git
cd dsec-rs
cargo build --release          # ~2 min on 2 cores
cargo test --workspace         # 161 tests: unit + integration + e2e
```

Run the examples (each exercises a full stack slice):

```bash
cargo run --release -p dsec-sdk --example control_plane_demo   # IAM -> quota -> REST -> exec -> cross-node packdiff -> pause -> stream -> pool
cargo run --release -p dsec-rl --example training_loop         # sandbox envs + replay buffer + LSTM reset (0 violations)
cargo run --release -p dsec-sdk --example burst_simulation     # 100k sandbox lifecycles
```

### Running the tests

The suite covers protocol round-trips, storage invariants (CoW isolation,
pack_diff fidelity, cache dedup), lifecycle state machines, admission
concurrency, IAM quota rollup, placement quality, e2e SDK flows over real
UDS transports, and RL invariants (episode contiguity, LSTM reset-on-done,
GAE correctness vs a naive reference):

```bash
cargo test --workspace                       # everything
cargo test -p dsec-storage packdiff          # one area
cargo nextest run                            # if you prefer nextest
```

### Benchmarks

```bash
cargo run --release -p dsec-bench            # full suite, writes bench-results.{json,md}
cargo run --release -p dsec-bench -- --quick # smoke-sized
cargo run --release -p dsec-bench -- creation --count 5000 --concurrency 256 --paper-latency
cargo run --release -p dsec-bench -- burst --count 100000
```

## Results vs the paper

2-core CI-class VM, release profile, **interleaved A/B against the `v0.1.0`
baseline** (per-optimization deep-dive: [`docs/performance.md`](docs/performance.md);
raw logs: [`docs/benchmarks.md`](docs/benchmarks.md)):

| metric | dsec-rs | paper / reference |
|---|---|---|
| creation rate (paper latency model, 256 concurrent) | **6,306 /s** | ~5,000/s cluster-wide |
| creation rate (software path, A/B: +35-123%) | **13,400 /s** | — |
| burst lifecycle (create + exec + destroy, A/B: +22%) | **56.7k /s, 0 errors @ 100k** | 100k+ concurrent sandboxes |
| pause latency (p50) | 4,634 ms (injected model, paper-anchored) | ~4 s (kernel checkpoint + reclaim) |
| resume latency (p50) | 1,738 ms | ~1.5 s class |
| apiserver throughput (keep-alive, pipelined) | **129,000 RPS** (6.1x vs cold) | must sustain ~5k creates/s |
| RL vectorized envs (A/B: 2.4x) | **1.42M steps/s** | PufferLib ~1M+/s (Cython) |
| GAE (A/B: 5.2x) | **117M transitions/s** | C-path in pufferlib train() |
| replay buffer enqueue | 6.3M transitions/s | PufferReplayBuffer ~10M+/s |
| sandbox-env RL (real exec per step, A/B: 1.12x) | 18.9k steps/s | — |
| pack_diff snapshot (A/B: 183x) | **8.9M packs/s** | replicate working state w/o full copy |
| pack_diff apply (A/B: 4.1x) | **34.9k applies/s** | — |
| pack_diff size ratio | **0.05x** full image | replicate working state w/o full copy |
| placement decisions | 128k /s @ 1,024 nodes; k-choice quality 67.65% (unchanged) | must not bottleneck ~5k creates/s |
| protocol codec (A/B: 2.1-2.9x) | 888 / 892 MB/s enc/dec | — |

## libdsec API mapping

| libdsec (paper, Python)                  | dsec-sdk (Rust)                               |
|------------------------------------------|-----------------------------------------------|
| `dsec.DsecClient(token, endpoint)`       | `DsecClient::new(token, endpoint, transport)` |
| `client.sandboxes.create(spec)`          | `client.create_sandbox(spec).await`           |
| `sandbox.execute(cmd)`                   | `sandbox.execute(cmd).await`                  |
| `sandbox.execute_stream(cmd)`            | `sandbox.execute_stream(cmd).await`           |
| `sandbox.filesystem.read_file(p)`        | `sandbox.read_file(p).await`                  |
| `sandbox.filesystem.write_file(p, data)` | `sandbox.write_file(p, data).await`           |
| `sandbox.http.get(url)`                  | `sandbox.http_get(url).await`                 |
| `sandbox.pause()` / `resume()`           | `sandbox.pause()` / `resume().await`          |
| `sandbox.pool(min, max)`                 | `SandboxPool::new(client, cfg)`               |

REST surface (wire-compatible with the paper's description):

```
POST   /v1/projects            GET /v1/projects/{name}
POST   /v1/tokens
POST   /v1/sandboxes           GET /v1/sandboxes[?project=]
GET    /v1/sandboxes/{id}      DELETE /v1/sandboxes/{id}
POST   /v1/sandboxes/{id}/pause   POST /v1/sandboxes/{id}/resume
GET    /v1/nodes               POST /v1/nodes/{id}/heartbeat
GET    /v1/cluster             GET /healthz   GET /metrics
```

## Design notes

- **Two planes**, as in the paper: management (REST) and data (Aether
  frames). The data plane is transport-agnostic: in-process channels for
  deterministic simulation, real Unix domain sockets for production-shaped
  runs.
- **Deterministic simulation**: splitmix64 PRNG + injectable latency models
  (`NodeLatencyProfile::paper()` replays the paper's timing anchors — ~5 ms
  FnCall handout, ~900 ms microVM, ~4 s pause; `::zero()` for pure logic).
- **Lock-free admission**: node resource pools use optimistic
  reserve-verify-rollback on atomics — under full contention exactly
  `max_sandboxes` admissions succeed (regression-tested).
- **Faithful PufferLib semantics**: step-major episode ring, contiguous
  episodes, LSTM (h, c) stored per step and zeroed at episode boundaries,
  mat-obs views, GAE across the ring, `reset_if_done` vectorized stepping.
- **pack_diff**: dirty-block snapshots carry the layered-FS file table and
  allocator cursor, so replicas observe identical filesystems (mirrors the
  real system where FS metadata lives in the replicated blocks).

## Relation to the real DeepSeek stack

The paper's production stack is mixed-language: 3FS (C++), EROFS (kernel C),
Firecracker (Rust), the ublk/OverlayBD user-space block layer (Rust, open
sourced by DeepSeek at [kvcache-ai/AgentENV](https://github.com/kvcache-ai/AgentENV)),
and the Python layers (libdsec, RL integration, PufferLib). This workspace
re-implements the Python-side architecture in Rust — the same role PufferLib
plays for the training loop — while modeling the storage, isolation and
scheduling layers in userspace. See the
[research report](docs/DSec_Rust_Reimplementation_Report.pdf) for the full
language mapping and the integration points a real deployment would use.

## Repository layout

```
crates/
  dsec-protocol/  dsec-storage/  dsec-runtime/  dsec-control/
  dsec-sdk/       dsec-rl/       dsec-bench/
docs/
  architecture.md            component deep-dive
  paper-mapping.md           paper claim -> code artifact map
  benchmarks.md              full benchmark logs
  DSec_Rust_Reimplementation_Report.pdf   ~8k-word research report
.github/
  workflows/ci.yml           fmt + clippy (-D warnings) + tests + MSRV
  dependabot.yml             weekly cargo dependency bumps
```

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). All CI checks
(fmt, clippy with `-D warnings`, tests, MSRV) must pass. Report
vulnerabilities per [SECURITY.md](SECURITY.md).

## License

MIT — see [LICENSE](LICENSE). This project is an independent reimplementation
for research purposes and is not affiliated with or endorsed by DeepSeek.
