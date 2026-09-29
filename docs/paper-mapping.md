# Paper mapping: DSec → dsec-rs

Reference: *"DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for
Effective Agentic Training at Scale"* (DeepSeek-AI + Tsinghua, 2026,
arXiv 2609.22978). This map links each paper claim/system to the artifact
that re-implements or models it, and marks how literal the port is.

Legend — **L** literal reimplementation of described behavior,
**M** modeled in userspace (real mechanism is kernel/hypervisor/C++),
**P** ported from the Python original (libdsec / PufferLib).

## System overview

| paper concept | artifact | mode |
|---|---|---|
| libdsec Python SDK | `dsec-sdk` (`DsecClient`, `Sandbox`, `SandboxPool`, `LocalCluster`) | **P** |
| Stateless apiserver (REST) | `dsec-control::apiserver` (axum, 12 routes) | **L** |
| IAM nested projects + quotas | `dsec-control::iam` (hierarchical rollup, RBAC tokens) | **L** |
| Placement Engine (k-choices) | `dsec-control::placement` (filter → sample k → score → pick, cloud burst) | **L** |
| Watcher (health, eviction, preemption) | `dsec-control::watcher` (heartbeat TTL, paused-first preemption) | **L** |
| Edge node runtime | `dsec-runtime::edge` (admission, CAS lifecycle) | **L** |
| Aether data plane | `dsec-protocol` (frames, CRC32, mux) + `dsec-runtime::aether` (server/client, UDS) | **L** |
| Chronus in-sandbox runtime | `dsec-runtime::chronus` (sessions, exec, fs, http, streams) | **M** |
| 4 backends (FnCall / Container / MicroVM / FullVM) | `dsec-runtime::backends` + `NodeLatencyProfile` | **M** |

## Storage

| paper concept | artifact | mode |
|---|---|---|
| EROFS read-only root over 3FS | `dsec-storage::image` (immutable block images, content-hashed) | **M** |
| On-demand block fetch | `dsec-storage::loader` (fault-driven) + `pipeline` (bounded async prefetch) | **L** (userspace) |
| DAX page-cache sharing across sandboxes | node LRU cache of `Arc<Block>` (dedup by digest) | **M** |
| OverlayBD-style writable layer | `dsec-storage::overlay` (CoW) + layered FS | **M** |
| pack_diff working-state replication | `dsec-storage::packdiff` (dirty blocks + FS metadata + allocator cursor; apply = identical FS) | **L** (userspace) |
| Zero-copy to trainer | `dsec-storage::shared` (`SharedRegion`) + buffer `dump` | **M** |

## CPU / memory optimization

| paper concept | artifact | mode |
|---|---|---|
| SCHED_IDLE for agent sandboxes | `CpuClass::Idle` in `SandboxGovernor` | **M** |
| Pause ≈ 4 s (checkpoint + reclaim) | `EdgeNode::pause` + `ResourcePool::pause_reclaim` (60% reclaim model) | **M** |
| MADV_WILLNEED resume prefetch | `ResourcePool::resume_refetch` | **M** |
| vfork/CLONE_VM creation speed | software-path creation benchmark (36.8k/s) | **M** |
| FnCall warm pool (~1 s create) | `backends::FnCall` pool + paper latency profile | **M** |

## RL co-design

| paper concept | artifact | mode |
|---|---|---|
| PufferLib episode ring buffer | `dsec-rl::ReplayBuffer` (step-major ring, contiguous episodes) | **P** |
| LSTM hidden reset on episode end | per-step (h, c) storage, reset-on-done invariant + repair | **P** |
| mat-obs / flat-obs layouts | buffer views | **P** |
| GAE | `compute_gae` (boundary-restarted, validated vs naive) | **P** |
| Vectorized env stepping | `EnvPool` (scoped threads, `reset_if_done`) | **P** |
| Agent loop = sandbox session | `TerminalTaskEnv` (real Chronus exec per step) | **L** |
| Preemptible rollouts (pause/resume) | sandbox pause/resume through SDK | **M** |
| Malicious behavior mitigation | policy profiles (`SandboxGovernor::policy_profile`) + fake egress deny lists | **M** |

## Scale anchors (paper numbers → benchmark)

| paper | benchmark (2-core VM) |
|---|---|
| ~5,000 creations/s | 5,806/s (paper latency model, 256 concurrent) |
| 100k+ concurrent sandboxes | 100k burst, 54.7k lifecycles/s, 0 errors |
| ~4 s average pause | pause p50 4,634 ms (injected model) |
| ~768 edge nodes / unit | placement: 1,024-node candidate sets, 124.7k decisions/s |
| cloud bursting | placement cloud-burst path + cost scoring |

## What is intentionally *not* real here

1. **Isolation** — no containers/microVMs/namespaces are created; backends
   are latency-profiled simulations. Real deployment: Firecracker API,
   containerd, real EROFS mounts.
2. **Networking** — egress routing is a fake router with deny lists; the
   paper uses eBPF programs per sandbox.
3. **Distributed storage** — single-process block store stands in for 3FS
   (C++); `pack_diff` apply models replication, not network transfer.
4. **Kernel mechanisms** — SCHED_IDLE, balloon/DAMON, core scheduling,
   MADV_WILLNEED, vfork are modeled as accounting + latency profiles.

See [architecture.md](architecture.md) for how each model is structured so
it can be swapped for a real backend behind the same traits.
