# Architecture

This is a component deep-dive; for the paper claim → code artifact map see
[paper-mapping.md](paper-mapping.md), and for the full research report see
[`DSec_Rust_Reimplementation_Report.pdf`](DSec_Rust_Reimplementation_Report.pdf).

![architecture](architecture.png)

## The two planes

DSec separates a **management plane** (control decisions) from a **data
plane** (sandbox I/O). dsec-rs mirrors this split at the crate level:

- Management plane: `dsec-control` (apiserver, IAM, placement, watcher) —
  REST over HTTP, stateless, horizontally scalable as in the paper.
- Data plane: `dsec-protocol` (Aether framing) + `dsec-runtime` (Edge node
  servers) — every sandbox operation is a multiplexed Aether request, never
  a management-plane call.

## Management plane (`dsec-control`)

### IAM — nested projects & quotas

Projects form a tree; quotas roll up hierarchically (a child's usage counts
against every ancestor's budget). Tokens carry role-scoped permissions
(admin / operator / observer) checked at the apiserver. Admission checks
`project_quota_ok` at sandbox creation — the paper's mechanism for
multi-team fairness on shared clusters.

### Registry — versioned state + events

A watch-style registry: every mutation bumps a version and broadcasts an
event to subscribers. The watcher consumes node heartbeats through it;
the SDK's `LocalCluster` uses it to observe cluster membership.

### Placement Engine — k-choice power scheduling

For each creation request the engine:

1. **Filters** candidates (capacity, image affinity, labels, health,
   local-vs-cloud policy),
2. **Samples k** nodes (paper: k-choices to keep decisions O(k) not O(n) at
   ~5k creations/s),
3. **Scores** them on packing / fragmentation / locality / topology / cost,
4. **Picks the best of k**, breaking ties with seeded jitter (prevents
   positive-feedback herd effects — regression-tested).

Cloud nodes are eligible only when the spec allows bursting, and take a
cost penalty in scoring.

### Watcher — liveness, eviction, preemption

Heartbeats carry node usage; the watcher expires nodes past a TTL and
evicts their sandboxes. Preemption picks victims paused-first (cheapest —
paper: paused sandboxes are already swapped), then lowest priority.

### apiserver

Stateless axum service exposing the 12-route REST surface, Prometheus
metrics, and token-bucket rate limiting (protects the ~5k/s creation path).

## Data plane (`dsec-protocol` + `dsec-runtime`)

### Aether framing

25-byte header (magic, version, stream id, type, length, flags) + payload,
CRC32-verified. `Connection` implements request/response multiplexing over
any `AsyncRead + AsyncWrite` — in-process channels (determinism) or real
UDS (production-shaped). Stream frames are paced to model bounded pipe
throughput.

### Edge node

Owns the sandbox lifecycle state machine (Creating → Ready → Pausing →
Paused → Resuming → Destroying) driven by CAS transitions, admission
control against the node `ResourcePool`, and per-sandbox governors
(SCHED_IDLE class, policy profile). Pause reclaims ~60% of a sandbox's
memory (the paper's balloon + `memory.reclaim` model); resume refetches
cold pages (MADV_WILLNEED model).

**Admission is lock-free**: optimistic reserve-verify-rollback on atomics.
Under full contention exactly `max_sandboxes` admissions succeed — see
`resource::tests::concurrent_admission_never_oversubscribes`.

### Chronus — session layer

Command interpreter (with `>`/`>>` redirection and `&&` chaining),
filesystem ops, fake egress HTTP router (per-sandbox routing with deny
lists), and streaming I/O — the "standard runtime interface" the paper
puts inside every sandbox, fronted by the SDK.

### Backends

Four pluggable backends with injectable latency profiles:
`NodeLatencyProfile::paper()` replays the paper's anchors (FnCall ~5 ms
handout, MicroVM ~900 ms cold, pause ~4 s); `::zero()` for pure logic.

| backend | paper role | dsec-rs model |
|---|---|---|
| FnCall | pre-created warm pool, ~1 s create | pool of pre-admitted sessions, ~5 ms handout |
| Container | Docker-shaped | simulated image + cgroup accounting |
| MicroVM | Firecracker | simulated boot with ~900 ms latency |
| FullVM | QEMU-class | simulated, slowest |

## Storage (`dsec-storage`)

- **EROFS-style images**: immutable block-addressed images, content-hashed
  blocks; **on-demand loading** faults blocks in as the guest reads them
  (paper: EROFS over 3FS with on-demand fetch).
- **Node block cache**: LRU cache of `Arc<Block>` — blocks shared between
  sandboxes on the same image are loaded **once** (models virtio-pmem DAX
  page-cache dedup; benchmarked as cache-hit ratio).
- **Overlay devices**: CoW write layer over a base image; blocks are
  copy-on-first-write.
- **Layered FS**: file table + free-block allocator on top of an overlay —
  a tiny read-only-root + writable-upper guest filesystem.
- **pack_diff**: incremental snapshot of an overlay's dirty blocks + FS
  metadata + allocator cursor. Applying a pack to a fresh overlay of the
  same base yields an **identical filesystem** (tested property). This is
  the paper's working-state replication primitive (sends ~5% of image bytes
  when 5% of files are mutated).
- **AsyncDataPipeline**: semaphore-bounded prefetch of next blocks — the
  paper's async data fetch overlap.
- **SharedRegion**: zero-copy handoff of packed transitions to a trainer
  process (models the torch shared-memory zero-copy channel).

## RL layer (`dsec-rl`)

PufferLib's core semantics, in Rust:

- **ReplayBuffer**: step-major ring (all envs' step t contiguous — matches
  pufferlib's episode-major layout for training), storing obs / action /
  reward / done / logprob / value / **per-step LSTM (h, c)** for actor and
  critic. Episodes are carved contiguously; done flags restart GAE; hidden
  states are **zeroed at episode starts** (pufferlib's LSTM reset), with an
  invariant checker + repair.
- **EnvPool**: vectorized stepping across scoped threads with
  `reset_if_done` semantics (auto-reset finished envs, matching pufferlib).
- **Driver**: rollout loop that snapshots model input state (obs mat +
  hidden states), steps envs, and zeroes h/c on done before the next
  forward — the ordering bug pufferlib's design guards against.
- **GAE**: γ/λ advantage estimation restarted at episode boundaries,
  validated against a naive per-episode reference.
- **Sandbox envs**: `TerminalTaskEnv` runs a real Chronus session per step
  — RL environments that are actual sandboxes, the paper's core premise.

## Benchmarks (`dsec-bench`)

17 suites: creation rate (software + paper-latency), 100k burst lifecycle,
pause/resume latency distributions, apiserver RPS, RL throughput (envs /
buffer / GAE / driver / sandbox-envs), pack_diff size + apply + snapshot,
placement throughput + k-choice quality, codec encode/decode. Output:
`bench-results.json` + `bench-results.md`. See [benchmarks.md](benchmarks.md).
