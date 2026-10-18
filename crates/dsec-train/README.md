# dsec-train

**Fixed-pool and elastic agentic RL training** (Phases 5–6): explicit
policy versions, publication barriers, bounded staleness, cell-role
scheduling with safe rerouting, waterlevel tracking, and the
Async/Colocate/Elastic benchmark.

## What's in the box

| Module | Role |
|---|---|
| `lm` | `TinyLm` — a deliberately tiny but *genuinely trainable* LM: hand-written forward + backward policy gradients over the Phase 4 loss terms, deterministic sampling, SHA-256 weight digests. |
| `policy` | `PolicyStore` — versioned weights, exactly one **published** version, staged checkpoints, and the **publication barrier**: a crash mid-flip leaves the OLD version served and the barrier closed until recovery. |
| `trainer` | `Trainer` — the fixed-pool trainer: **bounded staleness** (samples older than `published − staleness_bound` are rejected, never trained), the Phase 4 loss path, real gradient steps, barrier publication per step. |
| `scheduler` | `ElasticPool` — cell roles with **draining transitions** (in-flight model calls always finish; queued work reroutes), rollout **waterlevel** with hysteresis, **joinable streaming batches**, barrier-gated role flips, bounded per-cell concurrency, long-tailed job durations. |
| `bench` | Async (fixed split) vs Colocate (phase-serialized) vs Elastic on a simulated, calibrated workload. Typical shape: elastic ≈ 1.2–1.3× over Async, ≈ 1.8–1.9× over Colocate (simulation; QWENGYRE reports 1.38–1.78× / 1.21–1.85× on real hardware). |
| `server` *(feature `server`)* | The policy model server: a real axum HTTP endpoint speaking the OpenAI chat shape, serving the published policy, reporting `policy-v{major}` on every response. |

## The version contract (Phase 5 in one test)

```
rollout  → POST /v1/chat/completions → response "model": "policy-v3"
sample  → RolloutSample { execution, policy_version: 3 }
trainer → step(samples):
            age = published(4) − 3 = 1 ≤ bound(1) → accepted
            gradients → staged v5 → barrier → published v5
next request is served by v5 — immediately.
```

A crash at the barrier: `stage → ✖ (before flip)` — v4 stays served,
the barrier stays closed, recovery completes or discards.

## Quick start

```rust
use dsec_train::{Trainer, TrainerConfig, RolloutSample};

let mut trainer = Trainer::new(TrainerConfig { staleness_bound: 1, ..Default::default() });
let result = trainer.step(&stamped_samples)?;      // gradients + publication
assert_eq!(result.published.unwrap().major, 1);
```

Serving: `dsec_train::server::serve(trainer.store(), addr, 0.2, 8).await`
— then point `dsec-orchestrator`'s `OpenAiCompatClient` at it and the
full loop (rollout → stamp → train → publish → serve) runs over real
HTTP.

## Honest scope

Real: the version/staleness/barrier contracts, the gradient math, the
HTTP serving path, the scheduler invariants (drain safety, no-dropped
calls, barrier blocking, bounded concurrency). Toy by design: the model
size (swap in vLLM/SGLang + a verl-style trainer in production) and
the benchmark's logical clocks (calibrated simulation, not GPU
wall-clock).

## Examples

```bash
cargo run --release --example phase5_fixed_pool --features server -p dsec-train
cargo run --release --example phase6_elastic -p dsec-train
```

## Tests

```bash
cargo test -p dsec-train --features server
```

13 tests: learning-signal proofs (rewarded tokens gain probability,
punished tokens lose it), barrier crash/recovery, staleness rejection,
evaluator-failure exclusion, drain safety, benchmark assertions, and
the full HTTP e2e loop.
