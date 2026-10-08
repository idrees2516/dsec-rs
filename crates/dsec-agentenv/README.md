# dsec-agentenv

[![CI](https://github.com/idrees2516/dsec-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/idrees2516/dsec-rs)
[![crates.io](https://img.shields.io/crates/v/dsec-agentenv.svg)](https://crates.io/crates/dsec-agentenv)

**Agentic RL environment harness** — a Rust port of the MiMo Live RL
training environments (the HuggingFace dataset
[`XiaomiMiMo/MiMo-V2.6-RL-oss`](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss)
and its verl-side harness), plus the Scale AgentEnv server contract, the
Repo2RL conversion pipeline, and a generic factory for building new
environments of the same shape.

Everything runs as a **deterministic userspace simulation**: no Docker,
no root, no network — reproducible on any machine, seeded where
randomness matters. The real HuggingFace parquet shards load through the
`parquet` feature (row counts cross-checked against the dataset card:
code 2698, cyber 1000, music 1000, webdev 2093, general 989).

## What's inside

- **Task rows** (`task`) — the verl/HuggingFace dataset schema:
  `prompt` / `data_source` / `ability` / `agent_name` / `reward_model` /
  `extra_info` + the inner `instance_json`; JSONL shards; the five
  domain presets; a builder mirroring the upstream `build_parquet.py`.
- **Manifests** (`manifest`) — the `manifest.json` contract: uploads,
  setup, `wait_ports`, `mcp_servers`, the verifier bundle; default
  synthesis for manifest-less task dirs; the same hard invariants
  (setup and verifier never run in the agent's container).
- **Pod topology** (`topology`) — the two-container environment
  (`main` + `sidecar`) with RW/RO volumes, longest-prefix mounts, the
  simulated shell subset agents use, MCP port lifecycle, session logs,
  and workspace source protection. DB isolation is physical: the main
  container has no mount for `/work/system`, so tools are the only
  mutation path — same observable behavior as the real pod.
- **State DBs** (`state`) — the `system/<mcp>/state.db` layer as
  deterministic in-process relational stores (schema, CRUD, mutation
  accounting, `post_state` snapshots).
- **MCP** (`mcp`) — JSON-RPC 2.0 envelopes, `tools/list` discovery,
  `tools/call`, OpenAI function-schema conversion, and an in-process
  transport whose wire bytes match the streamable-http sidecar.
- **Verifier** (`verifier`) — the `verifier_meta.json` rubric schema,
  the rubric engine (`VERIFY_DETERMINISTIC` / `VERIFY_AGENT_JUDGE`
  filtering, weighted scoring, the auto `src_protect` gate), the
  LLM-judge contract, `reward.json` emission, and the full
  **REWARD_TESTBED_CORRUPTED masking pipeline** end to end — task
  failures and testbed failures are never confused.
- **Agent loop** (`agentloop`) — multi-turn rollouts: append-only
  transcript, ordered tool dispatch, observation truncation, step limit,
  response budget, the `tool_exception` vs `transport_error` taxonomy.
- **Live RL trainer** (`live`) — the MiMo-V2.6 Live RL recipe:
  fully-asynchronous GRPO group rollouts (tokio tasks, bounded
  concurrency), group-relative advantages with masked rollouts excluded
  from the statistics, **GAR** (advantage redistribution toward
  higher-quality passing solutions), **GRS** (offline rubric synthesis
  from contrasting rollouts), adversarial screening (zero-tool lucky
  guesses, verifier cross-checks with an independent judge), and the
  aligned-RL self-correction cold start.
- **Environment factory** (`envgen`) — one declarative `EnvSpec` (plain
  JSON) compiles into a bootable environment: systems, tools, workspace,
  rubric, manifest. Seeded variant generation (scaled money literals,
  distractor rows) and domain templates for knowledge work, terminal /
  computer use, and webdev.
- **Repo2RL** (`repo2rl`) — repositories → environments: commit mining
  (bots/merges filtered, test+source required), task synthesis, the
  test-driven verifier (per-case items + golden-patch gate), verl row
  emission. `parse_git_log` covers the `git log --name-status` shape.
- **AgentEnv server** (`server`, feature `server`) — the Scale
  AgentEnv REST contract over the deterministic stack: create / list /
  info / observe / tool-call / reset / verify / destroy, with masked
  verifications reporting the category instead of a false zero.

## The contract that matters

```rust,no_run
use dsec_agentenv::verifier::{
    AnchorJudge, RewardConfig, RewardOutcome, Rubric, RubricItem, RuleCheck,
    VerifierHarness,
# , AgentRolloutResult
};

# fn main() {}
# #[allow(dead_code)]
fn demo() {
    let rubric = Rubric::new().item(RubricItem::rule(
        "answer",
        1.0,
        RuleCheck::AllText { needles: vec!["42".into()] },
    ));
    let judge = AnchorJudge::new();
    let harness = VerifierHarness::new(&rubric, &judge, RewardConfig::default());
    # let mut pod = dsec_agentenv::SimPod::builder("t").build();
    # let manifest = dsec_agentenv::manifest::Manifest::default();
    match harness.calculate_reward(&mut pod, &manifest, &AgentRolloutResult::default()) {
        RewardOutcome::Valid { score, .. } => {
            let _ = score; // train on it
        }
        RewardOutcome::TestbedCorrupted { kind, .. } => {
            let _ = kind; // mask the sequence — never train on the false zero
        }
    }
}
```

## Examples

```bash
# the environment zoo: compile + boot + rollout + grade every template
cargo run -p dsec-agentenv --example envgen_zoo

# the Live RL pipeline behind the AgentEnv REST server (real HTTP)
cargo run -p dsec-agentenv --features server --example live_pipeline

# load the REAL HuggingFace parquet shards (all 5 splits, ~20 MB)
bash scripts/research/mimo-samples/download.sh
cargo run -p dsec-agentenv --features parquet --example real_dataset \
    scripts/research/mimo-samples
# code 2698 / cyber 1000 / music 1000 / webdev 2093 / general_train 989 rows
# (the general split is fetched from general/train.parquet upstream and
#  saved as general_train.parquet — the name the example probes)
```

## Features

| feature   | enables                                    | cost                    |
|-----------|--------------------------------------------|-------------------------|
| (default) | the whole harness library                  | tokio/serde/thiserror   |
| `server`  | the axum REST server (`server` module)     | + axum                  |
| `parquet` | real HuggingFace parquet loading           | + arrow / parquet       |

## Tests

72 unit tests + 4 integration tests, all deterministic, covering the
masking contract, GRPO/GAR/GRS, the pod topology isolation, MCP
discovery, the REST surface, and the repo pipeline. Run with:

```bash
cargo test -p dsec-agentenv --features server,parquet
```
