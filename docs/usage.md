# Using dsec-rs — a practical adoption guide

This guide answers the question every new reader has after the README:
**"OK, so what do I actually *do* with this?"** It maps the twelve crates
to four concrete engineering personas, gives copy-paste entry points for
each, and names the exact integration points you would replace to move
from the deterministic userspace simulation to a production deployment.

The one-sentence orientation: **dsec-rs is a complete, deterministic,
single-language model of the full agentic-RL training stack** — the
sandbox cloud (DSec), the RL training core (PufferLib), the agentic
environment harness (MiMo Live RL), the robustness layer (Karotte), and
the environment-generation flywheel (AutoEnvScaling) — where every layer
runs anywhere Rust runs, with zero infrastructure requirements.

---

## 1. Who uses which layer

| persona | primary crates | what they build with it |
|---|---|---|
| **RL researcher** — needs thousands of parallel environments under a training loop | `dsec-rl`, `dsec-agentenv` | vectorized sandbox-backed env pools, PufferLib-style replay + GAE, GRPO-style Live-RL training with reward masking |
| **Eval / environments engineer** — needs reproducible, cheat-proof eval tasks | `dsec-agentenv`, `dsec-karotte`, `dsec-autoenv` | rubric verifiers, confinement + resource governance, adversarial screening, environment proposal pipelines |
| **Platform / infra engineer** — needs the sandbox cloud itself | `dsec-protocol`, `dsec-storage`, `dsec-runtime`, `dsec-control`, `dsec-sdk`, `dsec-firecracker` | the two-plane architecture: REST management plane, Aether data plane, microVM backends, quota/placement/eviction |
| **Performance / systems engineer** — needs numbers, not vibes | `dsec-bench`, `dsec-profiling` | A/B'd throughput evidence, flamegraphs, per-subsystem rates vs the paper's anchors |

None of these personas needs root, Docker, GPUs, or a cluster: the
default simulation mode is a seeded userspace model (splitmix64 PRNG,
injectable latency profiles). That is the point — you can develop and
test sandbox-cloud *logic* on a laptop and port it to real isolation
later.

## 2. Choose your entry point

- "I want to train an agent **today**" → §3 (training loop).
- "I want to serve environments to an external trainer over HTTP" → §4 (AgentEnv REST server).
- "I want to run evals that agents can't game" → §5 (verifiers + confinement).
- "I want environments to *generate themselves*" → §6 (the flywheel).
- "I want to build/operate the sandbox cloud" → §7 (SDK + cluster).
- "I want to reproduce the paper's numbers" → §8 (bench + profiling).

Every snippet below compiles against the workspace — the full runnable
versions live in the `examples/` directories named in each section.

## 3. Train an agent over real sandbox environments

The paper's core workload, in ~15 lines (full version:
`crates/dsec-rl/examples/training_loop.rs`):

```rust
use dsec_rl::driver::{Driver, DriverConfig};
use dsec_rl::envpool::EnvPool;
use dsec_rl::sandbox_env::SandboxEnvBuilder;

let builder = SandboxEnvBuilder::new(42);          // seed
let envs = builder.build_envs(8, 42)?;             // 8 sandbox-backed envs
let pool = EnvPool::new(envs).with_workers(4);     // vectorized stepping

let buffer = ReplayBuffer::new(4096, 8,            // capacity, num_envs
    pool.obs_space(), pool.action_space(),
    32);                                           // LSTM hidden size

let policy = /* your model — or the hash policy in the example */;
let mut driver = Driver::new(Box::new(pool), buffer, policy,
    DriverConfig::default(), 42);

driver.run_phase(1600);   // steps; check driver.stats()
```

What you get for free that naive loops get wrong: **episode
contiguity** in the step-major ring, **LSTM (h, c) zeroing exactly at
episode boundaries** (regression-tested to 0 violations), GAE computed
across the ring with a naive-reference test, and the **pipelined batch
data plane** (~35k steps/s, 2.5× the per-env baseline in
`docs/performance.md`).

For **Live-RL** (multi-turn agents, tool calls, rubric rewards, GRPO
with group-advantage and adversarial screening), the same loop shape
lives in `dsec-agentenv`'s trainer — see §4 for the environment side.

## 4. Serve environments over HTTP (the AgentEnv server)

Run the Scale-AgentEnv-style REST server and drive it like a remote
trainer (full version: `crates/dsec-agentenv/examples/live_pipeline.rs`):

```bash
cargo run -p dsec-agentenv --features server --example live_pipeline
```

```rust
use dsec_agentenv::server::EnvServer;
use dsec_agentenv::{envgen::templates, verifier::{AnchorJudge, RewardConfig}};

let srv = Arc::new(EnvServer::new(
    Arc::new(AnchorJudge::new()),      // swap in your LLM judge here
    RewardConfig::default(),
));
let app = srv.router();                 // axum router: mount anywhere
// tokio + axum::serve(listener, app) — see the example
```

The wire surface a trainer sees:

```
POST /envs                    create (from a spec or dataset row)
GET  /envs/{id}/obs           observation
POST /envs/{id}/tools/call    MCP-shaped tool call (fs, shell, ...)
POST /envs/{id}/verify        grade -> reward (+ masking)
GET  /envs/{id}/metrics       counters (created/masked/verifications)
DELETE /envs/{id}             destroy
```

To train on the **real HuggingFace dataset** instead of templates:

```bash
bash scripts/research/mimo-samples/download.sh     # ~20 MB, git-ignored
cargo run -p dsec-agentenv --features parquet \
    --example real_dataset scripts/research/mimo-samples
```

The `parquet` feature loads the actual `XiaomiMiMo/MiMo-V2.6-RL-oss`
shards (code 2698 / cyber 1000 / music 1000 / webdev 2093 / general
989 rows) through the verl-row schema with `instance_json` expansion.
New environments are declared in the **EnvGen DSL** — a `WorldSpec` +
tool + rubric JSON that the factory compiles, boots, rolls out, and
grades (see `examples/envgen_zoo.rs` for one per domain).

## 5. Run evals agents cannot game

Two layers, composable:

**Verifiers (dsec-agentenv).** `verifier_meta.json`-shaped rubrics:
rule + llm items, weights, gates, `VERIFY_DETERMINISTIC` /
`VERIFY_AGENT_JUDGE` filtering — and the
`REWARD_TESTBED_CORRUPTED` masking contract: when the *testbed* is
broken, the reward is masked (excluded from training), never zero,
never silently passed.

**Confinement (dsec-karotte).** The robustness spine from
preferencemodel/karotte, for when the agent itself is adversarial:

```rust
use dsec_karotte::confinement::{
    NetworkPolicy, ResourceLimits, Sandbox as SandboxKind};

// sandbox type: runc | gvisor (fail-open, distrust!) | vm
let kind = SandboxKind::Vm;
// upstream arithmetic: VM memory minus the 1 GiB harness reserve
let limits = ResourceLimits::defaults(5 << 30, 100 << 30);
// egress: localhost + own addresses + allowlist only
let policy = NetworkPolicy::Strict;
// then: firewall_plan(policy, ...) -> nft rules in upstream order,
// firewall_canaries(...) -> probes, firewall_effective(...) ->
//   Contract::Prevented | Reaped | Unsupported (the tri-state)
```

Plus cgroup v1/v2 student groups (`memory.max` + `swap.max=0` +
`oom.group` + `cgroup.kill`), the RSS/PSS/tmpfs/SysV memory watchdog,
the fd-safe reclaim sweep, and defensive submission custody
(symlink/FIFO/capability refusals).

Judges — `RegexJudge`, `RubricJudge` (strict threshold), 
`ExecutableJudge` — compose with short-circuit `And`/`Or`, and the
three-class failure separation holds everywhere: *task failure* scores
zero, *misbehavior* scores zero **and is labeled**, *infra failure* is
masked rather than scored. Run `cargo run -p dsec-karotte --example
full_run` for the whole pipeline (confinement → cgroups → firewall →
MCP → agent loop → judges → durable transcript → websocket replay).

## 6. Make environments generate themselves

The AutoEnvScaling flywheel treats environment design as a terminal
task: a **proposer** writes Harbor tasks, a **solver** attempts them,
reward spread across the solver's attempts drives admission
(calibration band [0.25, 0.75], std ≥ 0.1, 13-gram decontamination,
no-op screening), and a reviewed pool evolves with the solver:

```rust
use dsec_autoenv::flywheel::{Flywheel, FlywheelConfig};

let mut fw = Flywheel::new(FlywheelConfig::demo(), seed_pool);
for _ in 0..6 {
    let report = fw.run_round();
    println!("{}: reward {:.2}, pool {}, useful {:.0}%",
        report.round, report.mean_solver_reward, report.pool_size,
        100.0 * report.useful_group_rate());
}
```

`FlywheelConfig::paper()` gives the paper's full hyperparameters;
`::demo()` shrinks them for laptops. `cargo run -p dsec-autoenv --
example flywheel` shows the full six-round dynamics (solver reward
climbing while pool review evicts outgrown environments and
harden-grounded replacements are admitted — 67% valid-env rate,
Wilson CI [57.8, 75.0]).

## 7. Build on the sandbox cloud itself

Assemble a full local cluster — apiserver, IAM, placement, watcher,
N Edge runtimes, Aether data plane — in one call
(`crates/dsec-sdk/examples/control_plane_demo.rs`):

```rust
use dsec_sdk::integration::{spawn_local_cluster, ClusterConfig};
use dsec_sdk::{DsecClient, Endpoint};

let cluster = spawn_local_cluster(
    ClusterConfig { node_count: 3, ..Default::default() }).await;

cluster.plane.create_project("root", "lab",
    Quota::limited(4000, 4096, 8))?;          // nested quotas
let token = cluster.plane.create_token("root/lab/rl-team",
    Role::Writer)?.token;

let client = DsecClient::new(token,
    Endpoint::localhost(cluster.api_addr.port()),
    Arc::new(ChannelTransport::with_nodes(cluster.nodes.clone())));

let sbx = client.create_sandbox(SandboxSpec { /* ... */ }).await?;
let out = sbx.execute("make test").await?;     // Chronus session
sbx.pause().await?;  /* ... */ sbx.resume().await?;
```

The REST surface (`POST /v1/sandboxes`, `/pause`, `/resume`,
`GET /v1/nodes`, `/healthz`, `/metrics`) is wire-compatible with the
paper's description — your existing control-plane tooling can drive it.
For real microVMs, install the Firecracker driver behind the same
`MicrovmDriver` trait the runtime already speaks (`dsec-firecracker`:
VMM launch, drives, pause/resume, diff snapshots; a fake-VMM harness
covers the protocol without a hypervisor).

## 8. Reproduce the numbers

```bash
cargo run --release -p dsec-bench                # full suite -> json + md
cargo run --release -p dsec-bench -- --quick     # smoke-sized
cargo run --release -p dsec-bench -- burst --count 100000
cargo run -p dsec-profiling --release -- batch 8 4 profile.svg
```

Headline rates on a 2-core CI-class VM (methodology + A/B evidence in
[`performance.md`](performance.md), logs in [`benchmarks.md`](benchmarks.md)):
6.3k creations/s under the paper's latency model (paper: ~5k/s
cluster-wide), 56.7k full lifecycles/s at 100k scale with 0 errors,
129k apiserver RPS keep-alive, 1.42M vectorized RL steps/s, 8.9M
pack_diff snapshots/s.

## 9. The production integration points

The simulation is a *model* with real seams. Swapping to production
means replacing exactly five things — everything else (IAM, quotas,
placement, the Aether codec, the env harness, the verifiers, the
flywheel) carries over unchanged:

| seam | simulation today | replace with |
|---|---|---|
| transport | in-process channels | real UDS / vsock (`dsec-sdk` ships both; `ChannelTransport` ↔ `UdsTransport`) |
| sandbox backend | userspace `FnCall`/simulated containers | Firecracker microVMs via `dsec-firecracker`, or your own `MicrovmDriver` |
| storage | in-memory EROFS-style images | real EROFS/3FS/OverlayBD behind the `dsec-storage` traits |
| judge | `AnchorJudge` / regex / rubric / executable | your LLM-judge API behind the same verifier interface |
| solver/proposer policy | scripted deterministic policies | your model endpoint behind `SolverModel` (`dsec-autoenv`) / the trainer hooks |

This table is the honest boundary of the reimplementation — and the
checklist a production port would work through.

## 10. Day-to-day commands

```bash
cargo test --workspace --all-features    # 620 tests, ~40 s warm
cargo fmt --all && cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo doc --workspace --no-deps --open
```

- Determinism: everything takes a seed; identical seeds ⇒ identical
  runs (the examples assert on it).
- Features: `server` (axum REST) and `parquet` (arrow dataset loading)
  on `dsec-agentenv` are opt-in to keep the core dependency-light.
- `dsec-profiling` is internal (`publish = false`); the other eleven
  crates are publishable (see [`publishing.md`](publishing.md)).
- MSRV 1.85; CI runs fmt + clippy `-D warnings` + tests + MSRV on every
  PR.
