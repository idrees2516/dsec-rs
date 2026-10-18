# The Phased Build Map (Phases 1–7)

This document maps the seven-phase production build order onto the
workspace's crates, tests, and examples — and states, phase by phase,
**what is real, what is simulated, and what needs a deployment**.

Primary sources:

| Source | What this repo takes from it |
|---|---|
| DeepSeek Elastic Compute (DSec), arXiv 2609.22978 | worker container / agent sandbox split (§6.2), command-log replay with idempotent side effects (§6.2), pause/resume with resource reclaim (§6.3), pack_diff environments (§6.1), agent misbehavior + access controls (§6.4–6.5) |
| QWENGYRE, arXiv 2609.33848 | trajectory trees with shared prefixes, shared targets counted once, execution-level loss normalization, group advantages without a value function, elastic GPU reallocation by unfinished rollout work, streaming data-parallel training, 1.38–1.78× over Async / 1.21–1.85× over Colocate (paper results on their hardware) |
| AutoEnvScaling + the repo's `dsec-autoenv` | environment proposer / admission / curriculum flywheel |
| MiMo Live RL + the repo's `dsec-agentenv` | dataset schema, agent-loop/tool/verifier contracts, the Live RL mechanisms |
| Karotte + the repo's `dsec-karotte` | threat model, judge semantics, failure classification, confinement |

---

## Phase 1 — single agent, real environment, no training

**Scope**: run one task through a real model endpoint and sandbox
backend; record the full lifecycle; verify the outcome; test
pause/resume and failure handling.

**Where**: `dsec-orchestrator`
([model](../crates/dsec-orchestrator/src/model.rs),
[taskenv](../crates/dsec-orchestrator/src/taskenv.rs),
[worker](../crates/dsec-orchestrator/src/worker.rs)).

**What is real**
- The model client is a real HTTP/1.1 client over `tokio::net` —
  OpenAI-compatible request/response shape, bearer auth, idempotency
  keys, retries with backoff. Point it at any in-cluster vLLM/SGLang
  server (`OpenAiCompatClient::from_env`).
- The `dsec-sandbox` backend runs on a real `dsec-sdk` local cluster:
  control plane (IAM + placement + watcher), edge node admission,
  Aether transport over real Unix sockets, Chronus exec/fs, real
  node-level pause/resume with memory reclaim, real cross-sandbox
  `pack_diff` forks.
- Every lifecycle boundary is journaled and fsync'd (Phase 2's journal).

**What is a stand-in**
- The in-repo mock model server (`mockserver.rs`) stands in for a real
  LLM endpoint in tests/examples — but it is a real axum server on a
  real TCP socket, so the client exercises its full HTTP path.
- The `agentenv` backend is an in-process pod (simulated DBs, MCP
  tools, rubric verifier) — the harness contract of MiMo Live RL.

**Tests**: `crates/dsec-orchestrator/tests/phase1_lifecycle.rs` —
lifecycle over real HTTP, pause/resume, dead-endpoint failure,
crash-during-pause, dsec UDS file task.
**Example**: `cargo run --release --example phase1_task --features server -p dsec-orchestrator [-- --dsec | --real]`.

---

## Phase 2 — durable long-lived rollout

**Scope**: external orchestrator journal, idempotent tool-call handling,
transcript persistence, snapshot ids, recovery that never repeats
completed tool side effects.

**Where**: `dsec-orchestrator`
([journal](../crates/dsec-orchestrator/src/journal.rs), the worker's
recovery path).

**What is real**
- The journal is a real file: append-only JSONL, `sync_data` per event,
  SHA-256 hash-chained. Torn tails (crash mid-append) are detected and
  truncated; tampering (rewrites, reordering, mid-file deletion,
  forged appends) breaks the chain and replay refuses to continue.
- Tool-call idempotency keys are enforced by a **durable ledger inside
  the sandbox** (`/tmp/.dsec-idem-ledger.jsonl` on the dsec backend's
  real filesystem, or the pod's `/tmp` volume) — the sandbox outlives
  the worker, exactly as DSec §6.2 specifies. The crash window *after
  execution but before journaling* is covered by the ledger: recovery
  re-issues the call, the ledger answers from cache, the side effect
  happens exactly once.
- Recovery rebuilds the transcript from the journal, finishes
  interrupted dispatches, reconciles pause state with the surviving
  environment (the env is the source of truth), and continues.

**Tests**: `tests/phase2_recovery.rs` — crash-after-tool-exec → exactly
one side effect; model-call idempotent replay; eval-pending re-verify;
tampered journal refusal; double recovery.
**Example**: `--example phase2_crash_recovery --features server`.

---

## Phase 3 — branch-safe subagents

**Scope**: isolated workspaces/snapshots, parent-child relationships,
patch return, merge validation, bounded concurrency, conflicting edits,
abandoned branches.

**Where**: `dsec-orchestrator`
([subagent](../crates/dsec-orchestrator/src/subagent.rs)) + the
`SimPod::fork` primitive in `dsec-agentenv`.

**What is real**
- Forks deep-copy writable state (volumes, DB rows, the idempotency
  ledger) and share only the immutable tool surface — the in-memory
  analogue of pack_diff checkpoint/restore. Neither side sees the
  other's writes.
- Subagent journals are per-branch files under the parent's directory.
- Patches are three-way-merge-ready: every op carries the branch-point
  base hash. The validator checks the parent's *current* state and
  previously merged sibling patches; conflicts refuse (nothing
  applied); a merge that would break source conservation is rolled
  back.
- Bounded concurrency is a real semaphore; peak concurrency is
  observable.

**Tests**: `tests/phase3_subagents.rs` — isolation + clean merge,
parent-modification conflict, sibling sequencing, abandoned branch,
bounded concurrency.
**Example**: `--example phase3_subagents --features server`.

---

## Phase 4 — correct RL data before scale

**Scope**: TITO records, trajectory trees, reward validity, masks,
group advantage, execution-level loss normalization; shared-prefix
targets contribute once; harness/tool tokens never become targets;
evaluator failures are never converted to reward zero.

**Where**: `dsec-rl`
([tito](../crates/dsec-rl/src/tito.rs),
[trajectory](../crates/dsec-rl/src/trajectory.rs),
[reward](../crates/dsec-rl/src/reward.rs),
[loss](../crates/dsec-rl/src/loss.rs)).

**What is real**
- Token-level origin classification and masks; a deterministic shared
  tokenizer so ids line up across executions and with the trainer.
- Trajectory trees keyed by `(parent, digest)`; shared prefixes reuse
  nodes; shared targets are counted once; every node preserves its
  conditioning context.
- Group advantages over scorable members only; evaluator failures carry
  `None` — no score, no advantage, no terms (a test proves the group
  statistics would differ if failures were counted as zero).
- Execution-level normalization: every valid execution carries exactly
  `1/|E|` weight regardless of token count; shared terms merge with
  combined coefficients and mean advantages.

**Honest limits**: the tokenizer is word-level; the "harness-injected
tokens" are the system prompt + task text (not a byte-level chat
template). The contracts under test are the masks and the
normalization, which are tokenization-agnostic.

**Tests**: inline in the four modules (18 tests).
The loss terms feed Phase 5's trainer directly.

---

## Phase 5 — fixed-pool online training

**Scope**: a real trainer and model server with explicit policy
versions and bounded staleness; static GPU split until the data and
policy-version contracts are reliable.

**Where**: `dsec-train`
([lm](../crates/dsec-train/src/lm.rs),
[policy](../crates/dsec-train/src/policy.rs),
[trainer](../crates/dsec-train/src/trainer.rs),
[server](../crates/dsec-train/src/server.rs)).

**What is real**
- The trainer performs actual gradient updates (hand-written forward +
  backward for a mean-context linear LM) on the Phase 4 loss terms —
  the "rewarded token gains probability" test proves real learning.
- Policy versions are `(major, sha256-of-weights)`; exactly one version
  is published; publication goes through a barrier (stage → flip →
  open). A crash mid-publication leaves the OLD version served and the
  barrier closed until recovery completes or discards.
- Bounded staleness: samples stamped with a policy version older than
  `published − staleness_bound` are rejected, per execution, before
  training.
- The model server is a real axum HTTP server speaking the
  OpenAI-compatible chat shape; every response reports its served
  version, which is where rollout stamps come from.

**Honest limits**: the model is deliberately tiny (production swaps in
vLLM/SGLang + a verl-style trainer); the contracts — versions,
staleness, barrier, loss plumbing — are what is tested end to end.
**Tests**: `crates/dsec-train/tests/contracts.rs` (barrier, staleness,
learning signal) + `tests/e2e_server.rs` (full loop over real HTTP).
**Example**: `--example phase5_fixed_pool --features server -p dsec-train`.

---

## Phase 6 — elastic GPU scheduling

**Scope**: cell role transitions, safe rerouting, training-batch
streaming, rollout-waterlevel tracking, policy publication barriers;
benchmark against fixed-pool Async and Colocate baselines.

**Where**: `dsec-train`
([scheduler](../crates/dsec-train/src/scheduler.rs),
[bench](../crates/dsec-train/src/bench.rs)).

**What is real (invariants)**
- A `Rollout → Train` transition first *drains*: in-flight model calls
  run to completion (never dropped — tested), queued work is rerouted
  to remaining rollout cells, then the role flips.
- Waterlevel-driven rebalancing with hysteresis (cadence-gated, and the
  two rules are mutually exclusive — no thrashing).
- Training batches stream: slices hand out to train cells by progress,
  and cells that join mid-batch pick up unassigned slices.
- Role flips are refused while the publication barrier is closed.
- Bounded per-cell serving concurrency; long-tailed per-job durations.

**What is simulated**: time — logical clocks with calibrated service
times, not GPU wall-clock. The benchmark reports makespan/utilization
comparisons (typical shape: elastic ≈ 1.2–1.3× over fixed-split Async
and ≈ 1.8–1.9× over phase-serialized Colocate on a bursty 16-group
workload). These are **simulation** numbers; QWENGYRE reports
1.38–1.78× / 1.21–1.85× on real hardware. The structural claim —
elasticity removes the serialization and starvation phases — is what
the benchmark tests.

**Tests**: `tests/contracts.rs` (drain safety, barrier blocking, full
workload drain, benchmark assertions).
**Example**: `--example phase6_elastic -p dsec-train`.

---

## Phase 7 — production security and storage

**Scope**: replace modeled isolation/storage/network paths with real
backends; attacker-style tests against the deployed kernel, hypervisor,
firewall, and artifact pipeline.

**What this repo does in-environment**
- `dsec-storage::artifact` — a real-filesystem artifact store: atomic
  temp-fsync-rename writes, SHA-256-verified reads, traversal-proof ids
  (component checks + canonical containment through the deepest
  existing ancestor), and hash-covered manifests. Attacker tests:
  path traversal, symlink escape, content tampering, torn writes,
  manifest forgery, manifest editing.
- Journal tamper evidence (Phase 2) — attacker tests: forged
  completions, mid-file truncation, reordering.
- Egress policy on the model client — allowlist enforced *before any
  socket opens*; the client is pinned to its own endpoint.
- Tool-call replay protection via the durable ledger (Phase 2).
- The real Firecracker microVM driver exists in `dsec-firecracker`
  (real VMM process + API socket + pause/resume/snapshot/restore) and
  is tested against a fake VMM over a real UDS; `dsec-karotte` carries
  the confinement/firewall/cgroups model with its own test suite.

**What needs a deployment** (documented, not faked): real
AppArmor/eBPF enforcement, a real hypervisor/host kernel under attack,
and 3FS-backed image distribution. This environment has no GPUs, no
Firecracker binary, and no kernel privileges — claiming those would be
dishonest, so the repo keeps them as clearly-labeled integration
points (`DSEC_FIRECRACKER_PATH`, the driver's `Capability::detect()`).

**Tests**: `dsec-storage` artifact suite (7),
`dsec-orchestrator/tests/security.rs` (4),
`dsec-firecracker` suite (13), `dsec-karotte` suite (183).

---

## Crash-injection matrix

| Boundary | Crashpoint | Test |
|---|---|---|
| After tool execution, before response persistence | `AfterToolExecBeforePersist` | `phase2_recovery::crash_after_tool_exec_never_repeats_side_effect` |
| After the model responded, before journaling | `AfterModelCallBeforePersist` | `phase2_recovery::crash_after_model_response_replays_same_response` |
| During a sandbox pause | `DuringSandboxPause` | `phase1_lifecycle::crash_during_pause_is_recoverable_state` |
| During policy publication | `publish(true)` / `step_with_publication_crash` | `contracts::publication_barrier_blocks_and_recovers`, `e2e_server::publication_crash_leaves_old_version_served` |
| While evaluation is pending | `WhileEvaluationPending` | `phase2_recovery::crash_while_evaluation_pending_reverifies_and_completes` |
| During branch merge | merge rollback path | `phase3_subagents` conflict tests (rollback on protection violation is covered by `merge_into_parent`) |

The four bug classes the user called out are all covered by dedicated
assertions: silent duplicate side effects (`real_exec_count`),
false-zero rewards (`EvaluatorFailure` + group statistics), stale-policy
training (staleness rejection), and shared writable state between agents
(fork isolation + conflict detection).
