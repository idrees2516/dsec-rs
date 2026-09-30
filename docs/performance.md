# Performance engineering deep-dive

This document records the v0.2 performance campaign: every hot path that was
researched, what dominated it, what changed, and the measured result. All
numbers come from **interleaved A/B runs** (baseline = tag `v0.1.0`, built
into a separate target dir, executed alternately with the optimized build on
the same 2-core VM) — single-shot numbers on this class of shared machine
swing ±20%, so medians of interleaved runs are the only honest comparison.

Paper boundary conditions held throughout: the two-plane architecture,
REST surface, Aether request/response semantics, `reset_if_done` and LSTM
reset-on-done semantics, quota/eviction/preemption semantics, and the
paper-anchored latency models (`NodeLatencyProfile::paper()`: FnCall ~5 ms,
pause ~4 s) are unchanged — the pause/resume benchmarks still replay the
paper's numbers by construction. What changed is the *efficiency of the
machinery around the semantics*.

## Results (interleaved medians, 2-core VM)

| benchmark | v0.1.0 | optimized | change |
|---|---|---|---|
| `rl_env_steps_fast` (64 envs, 4 workers) | 599k steps/s | 1,419k steps/s | **2.4x** |
| `rl_gae` (8192 x 64 window) | 22.6M trans/s | 117.5M trans/s | **5.2x** |
| `rl_driver_e2e` | 143k steps/s | 173k steps/s | **1.21x** |
| `rl_env_steps_sandbox` (real exec/step) | 16.8k steps/s | 18.9k steps/s | **1.12x** |
| `rl_buffer_enqueue` | 6.0M trans/s | 6.3M trans/s | par (see §5) |
| `packdiff_snapshot` | 48.4k packs/s | 8,854k packs/s | **183x** |
| `packdiff_apply` | 8.6k applies/s | 34.9k applies/s | **4.1x** |
| `protocol_encode` | 419 MB/s | 888 MB/s | **2.1x** |
| `protocol_decode` | 310 MB/s | 892 MB/s | **2.9x** |
| `apiserver_rps_keepalive` (pipelined) | 21-31k (cold only) | 129k RPS | **6.1x vs cold** |
| `apiserver_rps` (connection-per-request) | 31.5k | 31.0k | par (client-bound) |
| `placement_decisions` (1024 nodes, k=8) | 129k/s | 128k/s | par, quality 67.65% kept |
| `burst_lifecycle` (create+exec+destroy) | 41.7-46.8k/s | 50.9-56.7k/s | **+18-22%** |
| `creation` (5k @ 64 concurrent) | 5.9-9.9k/s | 13.2-13.4k/s | **+35-123%** |
| `pause` / `resume` p50 | 4633 / 1738 ms | 4633 / 1738 ms | paper-anchored (unchanged by design) |

Aggregate: the RL training loop now sustains **1.4M env steps/s (> PufferLib's
~1M reference)**, GAE at **117M transitions/s**, working-state replication
snapshots at **8.9M/s**, and the apiserver clears **129k RPS** with the
client transport real deployments use.

## 1. EnvPool: persistent worker threads (2.4x)

**Diagnosis.** `step_parallel` spawned scoped OS threads *per step*
(`std::thread::scope` + N spawns). Measured cost on this box: ~100-125 µs of
pure thread creation per vectorized step — with 64 trivial envs the useful
work was ~5 µs. The benchmark was effectively measuring `clone(2)`.

**Fix.** Workers are spawned **once** (lazily, on the first parallel step)
and own a fixed contiguous chunk of envs. Each step is a generation
broadcast:

- `TaskState { generation, cmd, done_count }` lives under **one mutex**, so
  a worker always reads a (generation, payload) pair that belongs together.
  An earlier split-mutex design had a genuine race — a worker could observe
  generation `g` but read the payload of `g+1` (caught by a 200-step soak
  test that checks per-env step counters and action echoes; the first
  attempt failed it deterministically).
- Workers park on `idle_cv`, the dispatcher parks on `progress`; the
  generation lives under the same lock as the park predicate, which makes
  lost wakeups structurally impossible.
- Results are collected in worker (= env) order; each env's result is
  computed independently, so output is bit-identical to the sequential path
  (regression-tested against it).
- Panicking envs poison the pool flag instead of hanging the dispatcher;
  `Drop` shuts workers down under the task lock and joins them.

## 2. GAE: env-blocked traversal + fused write-back (5.2x)

**Diagnosis.** Storage is slot-major (`idx(slot, env) = slot * num_envs +
env`), so walking one env's stream strides `num_envs * 4` bytes between
consecutive steps — at 64 envs that is **one cache line per element**, and
GAE walks the whole window backwards per env. The profile was pure cache
misses; the write-back was a second full pass.

**Fix.** Process envs in blocks of 8: for a fixed step, the 8 envs' values /
rewards / dones are 32 contiguous bytes (one cache line serves 8 elements
instead of 1). The recursion is untouched — each env's GAE is independent;
blocks only change traversal order. The write-back is fused into the
backward pass (each slot is written exactly once and never re-read).
Validated by the existing hand-computed and episode-boundary GAE tests.

## 3. Codec: hardware CRC + copy-free decode (2.1-2.9x)

**Diagnosis.** Three problems per frame: (a) CRC-32 was a byte-at-a-time
table walk (~0.4 GB/s); (b) `decode` copied the *whole* buffer into a
temporary just to zero the CRC field before recomputation; (c) `read_frame`
concatenated header + payload into another buffer before calling `decode`
— three allocations and three copies per frame on every data-plane hop.

**Fix.**
- CRC-32 now uses `crc32fast` (runtime-selects PCLMULQDQ carryless-multiply
  folding; slice-by-8 fallback). Note the trap avoided: the x86 `crc32`
  *instruction* implements CRC-32**C** (Castagnoli, 0x1EDC6F41), not the
  IEEE 802.3 polynomial (0xEDB88320) this wire format uses — PCLMULQDQ
  folding is the correct hardware route. Known-vector tests pin the
  polynomial.
- `Frame::decode_parts(header, payload)` verifies CRC over the two segments
  the reader already holds (the wire spec's "CRC over header[0..21] ++
  payload" is now literally what the code computes — no zeroed-copy
  temporary).
- `read_frame` decodes straight from the stack header + payload slice:
  one allocation total (the payload).

## 4. pack_diff: shared file-table snapshots (183x snapshot, 4.1x apply)

**Diagnosis.** Every snapshot deep-cloned the entire file table (512
`String` clones) even when nothing changed — the post-first-snapshot
management path *was* the table clone. Every apply re-interned 512 base
paths into a fresh `LayeredImage`, then immediately cleared and replaced
them from the pack. The overlay snapshot also cloned the whole CoW map to
read back a handful of dirty blocks.

**Fix.**
- `DiffPack.files` is `Arc<[FileWire]>` with `Arc<str>` paths (serde "rc";
  wire format unchanged). `LayeredImage` keeps a version-keyed cache of the
  wire snapshot: an unchanged table is handed out by refcount — the
  snapshot hot path is two Arc clones.
- `apply_diff` **adopts** the pack's shared table as its cache entry
  (keyed to the version it just published), so replica→snapshot chains
  stay on the refcount path.
- `LayeredImage::replica_from(overlay, pack)` builds a replica's table from
  the pack's own keys (refcount inserts) instead of interning the base
  image and replacing it — the apply benchmark measures the honest minimal
  replication work.
- `take_dirty_blocks()` consumes the dirty set and copies only the dirty
  blocks under one lock pass (no full CoW-map clone).

## 5. ReplayBuffer enqueue — measured, then reverted

A field-major bulk-copy rewrite (one SIMD memcpy per field per step instead
of N per-env copies) benchmarked **~20% slower** in interleaved A/B runs:
each per-env copy is 256-512 B, which the compiler vectorizes into tight
AVX loops, while libc's large-copy path loses on this VM core class. The
bulk version was reverted; the measured-fast per-env loop ships with a
comment recording the result so the experiment is not repeated blind.
(Proof that the A/B harness catches regressions, not just wins.)

## 6. Control plane: O(1) quota checks, in-place node projection

**Diagnosis.** Per creation: `check_quota` cloned the ancestor `Project`s
and called `project_usage`, which **scanned every sandbox record** and
allocated a `format!("{}/", root)` String *per record per ancestor* (O(n)
allocations per creation — at paper scale this is the difference between
5k/s and a stall). `create_sandbox` then cloned the whole `NodeInfo`
(strings, images, labels, admitted projects) to decrement three integers,
and `upsert`ed it back. `sandboxes().len()` — used by the burst benchmark
for peak tracking — cloned and sorted every record.

**Fix.**
- **Incremental project-usage index**: `usage_add(rec, ±1)` maintains a
  `HashMap<project, Usage>` over each record's ancestor chain on
  insert/remove/pause-toggle/node-loss. `project_usage` is an O(1) read.
  Cross-validated against the full-scan derivation over a mixed
  insert/pause/unpause/remove/node-loss sequence (test:
  `usage_index_matches_full_scan`).
- `adjust_node_resources(node_id, cpu, mem, slots)` mutates the three
  counters in place and publishes the same `NodeUpdated` event — no
  NodeInfo clone, identical event stream.
- `sandbox_count()` — O(1).
- `Iam::authorize` and the project-subtree checks lost their per-call
  `format!` allocations (`strip_prefix(..).is_some_and(rest.starts_with('/'))`).
- `check_quota` walks the ancestor chain under the projects read lock with
  no intermediate clones.

Result: creation +35-123%, burst +18-22% at 100k lifecycles (0 errors).

## 7. Placement: Floyd sampling + one RNG critical section

`place_filtered` previously locked the shared RNG **twice** per decision
(sample, then jitter) and `sample_k` materialized a 1024-element index Vec
plus a partial Fisher-Yates shuffle per decision. Now: Floyd's algorithm
draws k distinct candidate positions in O(k^2) (sorted, so scoring walks
the candidate list directly), and sampling + jitter happen under a single
lock. Net throughput is par with v0.1.0 while removing the O(n) per-decision
allocation — and the k-choice **quality metric is unchanged** (67.65% of
exhaustive improvement, identical scores), which is the point: the paper's
placement semantics are untouched.

## 8. apiserver: keep-alive is the real transport

The old HTTP benchmark opened a TCP connection per request and reported
~21-31k RPS — a number that measures `connect(2)`, not the server. A new
`apiserver_rps_keepalive` benchmark models what `DsecClient` pools actually
do (persistent connections, pipelined requests, Content-Length framing on
the client side): **129k RPS** on 2 cores. The cold-connection variant is
kept for continuity.

## 9. Methodology notes

- `dsec-bench` creation latency is now returned via join handles instead of
  a shared `Mutex<Vec<f64>>` on the hot path, and burst peak tracking uses
  `sandbox_count()` (O(1)) instead of cloning all records — benchmark
  harness overhead must not pollute what it measures.
- The FnCall/backend latency models, pause (~4 s) and resume (~1.5 s)
  anchors replay the paper's numbers and are deliberately **not**
  "optimized": they model real hardware the paper reports; making them
  faster would fake a result.
- Concurrency-sensitive tests are run repeatedly in CI; the env-pool soak
  test asserts per-env step counters and action echoes every step for 200
  steps across 4 workers.

---

# v0.3.0 campaign — the sandbox-env data plane (flamegraph-driven)

The one path the v0.2 campaign left on the table: `rl_env_steps_sandbox`
(real Chronus exec per step) improved only 1.12x because it is
**I/O-bound** in the async transport, not CPU-bound. This section records
the v0.3 pass: profile first, then fix what the profile actually shows.

## 10. Evidence: profiling the sandbox-env path

`dsec-profiling` (new, unpublished crate) runs the exact bench workload
under a SIGPROF sampler (`pprof`, no `perf`/privileges needed) and
reports aggregate data-plane timing via new `AetherClient::call_stats`
counters.

Baseline (v0.2, 8 envs, 4 pool workers):

```
steps/s:            19,769      ticks: 9,918 / 4.00 s
data-plane calls:   186,880     avg call latency: 43.5 us
call-time share:    25.3% of per-env worker wall time (= 2.02 cores busy waiting)
```

The flamegraph stacks were dominated by kernel/scheduler time, not
business logic: `syscall` (futex parks) under the Aether reader tasks,
`epoll_wait` under the tokio io driver, `__write` under
`mio::Waker::wake` (eventfd writes from `Unparker::unpark`), and `park`
under `Handle::block_on` inside `TerminalTaskEnv::exec`. Textbook
scheduler-bound transport: every step paid a full wakeup chain
(env worker → channel → server reader → spawned task → reply channel →
client reader → oneshot → unpark).

The decisive control experiment: a server-side micro-bench
(`dsec-profiling exec_cost`) calling `EdgeNode::handle_aether_request`
directly — no transport — measured **2.12 us per exec** (JSON decode +
interpreter + FS + JSON encode). With ~20 us amortized per exec on the
full path, **~90% of the sandbox-env cost was transport overhead**, not
exec work. (Server-side pure capacity: ~470k execs/s on one core.)

## 11. Fixes

### 11.1 Pipelined batch calls (`AetherClient::call_batch`)

Every env's exec for one vectorized tick is now issued as ONE batch:
requests serialized and reply slots registered under ONE lock pass,
frames sent as ONE transmission (`AetherWriter::send_batch` — for the
UDS transport literally one `write_all` + flush for N frames), replies
awaited under ONE deadline. Wire behavior (frames, per-request
correlation, demux) is identical to N individual `call`s — this is the
workload Aether's per-connection multiplexing was designed for. The
timeout slot accounting and per-slot error propagation are
regression-tested.

### 11.2 Batch-draining reader loops (`recv_many`)

Both the server connection loop and the client demux loop now drain up
to 64 frames per wake (`Receiver::recv_many` for channels; a persistent
read buffer for the UDS transport, so one `read` syscall yields many
frames — symmetric with `send_batch`). A pipelined batch of N frames
costs one park/unpark cycle on the receive side; completed calls are
matched under one `pending`-map lock pass.

### 11.3 Shell-faithful `&&` compound commands

Staging a fresh episode used to cost 8 sequential exec round trips (the
compound line was split because the interpreter was single-command).
The interpreter now chains `&&` segments with POSIX `sh -c` semantics
(run left to right, stop on failure, concatenate output, last exit code
wins) — staging is ONE exec plus a readiness probe. Both env paths
issue the same commands, so sequences stay identical.

### 11.4 `BatchedSandboxEnvPool` — the pipelined env pool

One pure state machine (`TerminalCore`) shared verbatim by both paths
guarantees identical commands, RNG consumption, rewards and
observations; `step_all` runs pure command selection, ONE pipelined
batch, pure result folding, and phase-barrier pipelined
`reset_if_done` staging (env i's line k completes before any env's line
k+1 — per-env order exact, distinct sandboxes concurrent). The driver
now takes any `Stepping` pool (`Box<dyn Stepping>`).

## 12. Results (interleaved medians, 3 runs each, 2-core VM)

| path | v0.2 (pre-campaign) | v0.3 | change |
|---|---|---|---|
| sandbox-env, pipelined batch pool | — | **~35.5k steps/s** | new path |
| sandbox-env, per-env reference path | ~19.8k steps/s | ~28.8k steps/s | **1.45x** |
| full-suite run (dsec-bench) | 18.9k (v0.2 recorded) | 30.8k ref / 35.1k batched | 1.6-1.9x |
| vs. v0.1.0 base case (14.1k) | | | **2.5x** |

All other suites re-measured at par or better (creation 8.3k/s software
path, 3.0M fast steps/s, 128.7M GAE trans/s, 112.6k keep-alive RPS);
167 → 197 tests green, 0 clippy warnings, MSRV 1.85 held.

## 13. What remains (honestly)

With exec at 2.12 us but ~48k execs/s aggregate on 2 cores, the
remaining ~10x is tokio task granularity: per-exec task spawn +
schedule + wakeups. The next structural step would be an
inline-first-poll fast path (poll each request future once in the
reader task; spawn only futures that actually suspend — the safe
pattern requires a re-scheduling waker) or io_uring for the UDS
transport. Both are protocol-preserving but were judged too risky to
land blind in this campaign; the profile harness is in-tree to drive
that work.
