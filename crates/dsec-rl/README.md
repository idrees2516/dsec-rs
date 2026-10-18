# dsec-rl

[![CI](https://github.com/idrees2516/dsec-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/idrees2516/dsec-rs/actions)
[![crates.io](https://img.shields.io/crates/v/dsec-rl.svg)](https://crates.io/crates/dsec-rl)

PufferLib's core, ported to Rust — plus the paper's flagship use case:
**agent sandboxes as RL environments**.

## What's inside

- **ReplayBuffer** — step-major episode ring (PufferLib layout) with
  per-step LSTM (h, c) storage and done-time hidden-state resets;
  12M+ transitions/s enqueue, batched sampling.
- **GAE** — blocked 8-env windows with fused advantage write-back,
  117M transitions/s on one core.
- **EnvPool / Stepping** — vectorized envs across persistent worker
  threads; `Stepping` lets the driver accept any pool (sync or
  pipelined). Deterministic env-order results.
- **BatchedSandboxEnvPool** — the throughput path: one vectorized step
  = ONE pipelined Aether transmission (`call_batch`), phase-barrier
  resets, `reset_if_done` semantics — byte-identical sequences to the
  per-env reference path (cross-validated in tests).
- **Driver** — the rollout loop with recurrent hidden-state
  bookkeeping, policy hooks, episode stats.
- **TerminalTaskEnv** — the flag-hunting terminal task from the paper:
  each env gets its own sandbox, actions are shell commands, the full
  Chronus data plane runs per step.

## The agentic-RL data path (QWENGYRE contracts)

- **TITO records** (`tito`) — token-level origin classification:
  model-generated tokens are the *only* training targets; tool results,
  harness text, and system templates are masked out. Includes a
  deterministic shared `SimpleTokenizer`.
- **Trajectory trees** (`trajectory`) — executions of one task group
  share prefix nodes; **shared-prefix targets are counted once**
  (deduplicated for training cost), each node preserving its original
  conditioning context.
- **Reward validity** (`reward`) — a tri-state (valid / genuinely
  failed / **evaluator failure**): infra failures carry no score, are
  excluded from group statistics, and are **never converted to reward
  zero**. GRPO-style group advantages over scorable members only.
- **Execution-level loss** (`loss`) — `compute_loss_terms` emits the
  trainer-facing `LossTerm`s: per-execution token averaging (an
  execution with more turns carries no extra mass), merged
  shared-target terms with mean advantages, and hard exclusion of
  evaluator failures.

These four modules are the data contract `dsec-train`'s trainer
consumes; `dsec-orchestrator` maps its rollouts onto them via
`WorkerRollout::to_transcript_turns`.

## Example

```rust,no_run
use dsec_rl::sandbox_env::SandboxEnvBuilder;
use dsec_rl::spaces::Action;

# fn main() {} // async plumbing omitted; see examples/training_loop.rs
# #[allow(dead_code)]
fn demo() {
    let builder = SandboxEnvBuilder::new(7);
    let mut pool = builder.build_batch_pool(8, 7).unwrap();
    pool.reset_all();
    let actions: Vec<Action> = (0..8).map(|i| Action::Discrete((i % 6) as i64)).collect();
    let results = pool.step_all(&actions).unwrap(); // one pipelined tick
    let _ = results;
}
```

`examples/training_loop.rs` in the repository runs a full training
rollout through the driver, buffer, and GAE.
