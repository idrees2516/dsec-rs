# dsec-orchestrator

**The DSec worker container** (paper §6.2): the scaffold-agnostic
control layer that manages agent sandboxes and drives rollouts —
implemented end to end, with crash recovery that never repeats side
effects.

## What's in the box

| Module | Role |
|---|---|
| `model` | A real OpenAI-compatible HTTP client (`OpenAiCompatClient`) — tokio TCP, bearer auth, idempotency keys, retries, an **egress allowlist enforced before any socket opens** — plus a deterministic `ScriptedModelClient` for drills. |
| `taskenv` | The `TaskEnvironment` contract with two real backends: `dsec-agentenv` pods and `dsec-sdk` sandboxes on a local cluster. Both implement **idempotent tool calls** through a durable in-sandbox ledger, pause/resume, isolated forks (in-memory pack_diff), and three-way-mergeable workspace patches. |
| `journal` | The external orchestrator journal: fsync'd JSONL, **SHA-256 hash-chained**, tamper-evident, with transcript-rebuilding replay. |
| `worker` | `WorkerContainer` — the durable rollout loop. Four `Crashpoint`s fire at the exact boundaries where the worst bugs hide; `recover()` re-attaches a new worker to the **surviving environment** and continues. |
| `subagent` | `SubagentManager` — branch-safe subagents: isolated forks, per-branch journals, patch return, merge validation with conflict/rollback semantics, bounded concurrency, abandoned-branch reaping. |
| `mockserver` *(feature `server`)* | An OpenAI-compatible mock model server (real axum + TCP) with a transcript-driven rule responder. |

## The recovery contract (Phase 2 in one test)

```
worker 1: task_created → model_call_completed → tool_call_started
          → [ tool EXECUTES in the sandbox ]
          → ✖ crash before tool_call_completed
sandbox : survives (it outlives the worker) — ledger has the key
worker 2: recover(journal, surviving env)
          → re-issues the call → ledger answers from cache
          → side effects: exactly 1
```

## Quick start

```rust
use dsec_orchestrator::{WorkerContainer, Journal, WorkerConfig, taskenv::AgentEnvBackend};

let env = AgentEnvBackend::new(spec.compile()?.boot()?);
let mut worker = WorkerContainer::new(
    "task-1".into(),
    Box::new(env),
    model,                       // Arc<dyn ModelClient>
    Journal::create("/run/task-1.jsonl")?,
    WorkerConfig::default(),
)?;
match worker.run_task().await? {
    TaskOutcome::Completed { reward, rollout } => { /* verify + collect */ }
    TaskOutcome::Failed { error } => { /* journaled failure */ }
}
```

Real endpoints: `OpenAiCompatClient::from_env()` reads
`DSEC_MODEL_URL`, `DSEC_MODEL_KEY`, `DSEC_MODEL_NAME`.

## Examples

```bash
cargo run --release --example phase1_task --features server -p dsec-orchestrator
cargo run --release --example phase1_task --features server -- --dsec     # real local cluster + UDS
cargo run --release --example phase2_crash_recovery --features server
cargo run --release --example phase3_subagents --features server
```

## Tests

```bash
cargo test -p dsec-orchestrator --features server
```

19 integration tests across `phase1_lifecycle`, `phase2_recovery`,
`phase3_subagents`, and `security` (attacker-style), plus the unit
suites. The full build order and its phase map: `docs/phased-build.md`.
