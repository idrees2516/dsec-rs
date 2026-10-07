# MiMo Live RL → dsec-agentenv

Reference: the HuggingFace dataset
[`XiaomiMiMo/MiMo-V2.6-RL-oss`](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss)
("Agentic RL Environments", Apache-2.0), its harness in
[`XiaomiMiMo/verl`](https://github.com/XiaomiMiMo/verl) (recipes:
`general/`, `code/`, `arvo/`, `design/{music,webdev}/`), and the
Live RL methodology described on the
[Flash-RL model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)
(fully-async GRPO, groupwise agentic grading, aligned RL with
environment hardening / adversarial screening / verifier cross-checks).
Also folded in: the [Scale AgentEnv](https://github.com/scaleapi/agentenv-framework)
server contract and the [HF Repo2RLEnv](https://github.com/huggingface/Repo2RLEnv)
conversion recipe.

This map links each upstream component to the artifact that
re-implements it, with the same mode legend as `paper-mapping.md`:
**P** ported from the Python original, **L** literal reimplementation of
described behavior, **M** modeled in userspace (the real mechanism is
Docker/K8s/a hosted LLM judge).

## Dataset (all five released configs)

| upstream concept | artifact | mode |
|---|---|---|
| verl row schema (`prompt`/`data_source`/`ability`/`agent_name`/`reward_model`/`extra_info`) | `task::TaskRow` (+ `ChatMessage`, `RewardModel`, `ExtraInfo`) | **P** |
| `extra_info.instance_json` kept as a string (Arrow struct-union corruption note) | `task::TaskInstance::parse` (expanded only at rollout time) | **P** |
| legacy SWE rows: single user turn with no `role` key | `ChatMessage` role deserializes optionally, defaults `user`, skips on serialize | **P** |
| `RLHFDataset` parquet loading | `parquet::load_parquet` (feature `parquet`; real shards load, row counts 2698/1000/1000/2093/989) | **L** |
| `build_parquet.py` row assembly | `task::TaskRowBuilder` | **P** |
| domain tags (`opensource-code`, `arvo`, `music`, `blackbox/webdev`, `mimoagent/terminal_bench`) | `task::Domain` presets + `envgen::DomainSpec` | **P** |
| `MimoAgentSWEDataset` adapter (instance_json → tools_kwargs) | `task::TaskRow::instance()` | **P** |
| JSONL shards | `task::TaskDataset::{load_jsonl, to_jsonl}` | **L** |

## Environment spec & topology

| upstream concept | artifact | mode |
|---|---|---|
| `manifest.json` (uploads / setup / wait_ports / mcp_servers / verifier) | `manifest::Manifest` (+ `Upload`, `SetupSpec`, `McpServerSpec`, `VerifierSpec`) | **P** |
| `_default_manifest` synthesis (workspace→main, system/tools→sidecar, ports 39101+i, late verifier upload) | `manifest::Manifest::synthesize_default` | **P** |
| `_normalize_manifest` invariants (setup/verifier never in `main`, source existence) | `manifest::Manifest::validate` | **P** |
| tool enumeration order (sorted `.py` stems, `_`-prefixed and `tools_test.py` excluded — the silent tool/port mix-up guard) | `manifest::enumerate_mcp_tools` | **P** |
| `MCP_PORT_BASE = 39101`, `MCP_STARTUP_TIMEOUT = 300`, `DEFAULT_VERIFIER_TIMEOUT = 900` | `manifest::{MCP_PORT_BASE, MCP_STARTUP_TIMEOUT, DEFAULT_VERIFIER_TIMEOUT}` | **P** |
| two-container pod (`main` agent / `sidecar` MCP + verifier) | `topology::{SimPod, Container, Volume}` | **M** |
| physical DB isolation (main never mounts `/work/system`) | `topology` mount table — the path does not resolve in `main`'s namespace | **M** |
| sidecar mounts the workspace read-only | `Access::Ro` on the sidecar's `/work/workspace` mount | **M** |
| `base_env.execute` / `copy_to` / `copy_out` | `topology::SimPod::{execute, copy_to, copy_out}` | **M** |
| the shell subset (`cat`, `ls`, `test`, `mkdir`, `rm`, `echo >`, `tail -c`, env prefixes, `python3 -c` probe) | `topology::SimPod::execute` command router | **M** |
| `/proc/net/tcp` liveness probe (connect would be REJECTed) | `topology::SimPod::{port_probe, probe_port}` | **M** |
| session logs (`/tmp/mimo-claude-logs/sessions`, injected to `/tmp/agent_output/sessions`) | `topology::SimPod::{append_session}` + `verifier::inject_agent_sessions` | **P** |
| `system/<mcp>/state.db` (+ `schema.sql`) | `state::{StateDb, Schema, Table}` | **M** |
| `build_post_state` (every table of every DB) | `state::PostState::snapshot` | **P** |

## MCP

| upstream concept | artifact | mode |
|---|---|---|
| streamable-http JSON-RPC 2.0 (`initialize`, `tools/list`, `tools/call`) | `mcp::{RpcRequest, RpcResponse, McpClient}` | **P** |
| transport resolution (`streamable-http` default, `http`/`sse` URL entries, `stdio` command entries) | `manifest::McpServerSpec::sdk_entry` | **P** |
| `discover_mcp_tools` → OpenAI function-tool schemas | `mcp::{ToolInfo::to_function_tool, discover_pod_tools}` | **P** |
| `tools/call` result union (`content` blocks, `isError`) | `mcp::ToolCallOutcome` | **P** |
| in-pod stdio bridge / out-of-pod dialing | `mcp::McpTransport` in-process impl (wire-identical envelopes) | **M** |
| privilege isolation (agent uid, bridge user, sudo launcher) | modeled by the mount-table isolation (the URL and DB are unreachable from `main`) | **M** |

## Verifier & the reward contract

| upstream concept | artifact | mode |
|---|---|---|
| `verifier_meta.json` (id / tier / method `rule|llm` / weight / gate / files / question / pass_anchor) | `verifier::RubricItem` | **P** |
| `rubrics.json` / `answer_key.json` (ground truth) | `envgen::RubricSpec` / `envgen::CheckSpec` | **M** |
| `run_verify.py` (extract agent output, build post_state, run verify, write reward.json) | `verifier::RubricEngine` + `verifier::VerifierHarness` | **P** |
| `extract_agent_output` (last assistant message from session JSONL; answer.md fallback) | `verifier::extract_agent_output` | **P** |
| `filter_rubrics` (`VERIFY_DETERMINISTIC` / `VERIFY_AGENT_JUDGE`) | `verifier::VerifyToggles` + `RubricEngine::filter` | **P** |
| weighted score, `strict_pass`, `deterministic_score`, `judge_score`, `judge_all_passed` | `verifier::EngineOutput` | **P** |
| `src_protect` auto gate (source conservation) | `topology::SimPod::{capture_protection, source_conserved}` + the auto-appended rubric result | **P** |
| LLM judge (`GA_JUDGE_*` env, `method: "llm"` items) | `verifier::{Judge, AnchorJudge, UnavailableJudge}` (deterministic anchor matching; injectable failure modes) | **M** |
| **the masking contract** — any testbed failure writes `{"reward_error": ...}` with NO `reward` key, the runner maps it to `REWARD_TESTBED_CORRUPTED`, and the trainer masks the sequence | `verifier::RewardOutcome::{Valid, TestbedCorrupted}` + `error::RewardErrorKind` (judge_key_missing, mcp_backend_down, missing_run_verify, missing_or_invalid_reward_json, reward_out_of_range, verify_crashed, judge_unavailable, reward_transport_died) | **P** |
| reward range check (finite, `[0,1]`) | `VerifierHarness::calculate_reward` stage 10 | **P** |
| `reward_binary_fullscore` binarization | `RewardConfig::binary_fullscore` | **P** |
| `_annotate_rubric_kinds` (join results back to meta) | results carry method/weight/gate inline | **P** |
| late verifier upload (ground truth invisible during rollout) | stage 4 of the harness (local task dir → pod staging volume → sidecar `/work`) | **P** |
| `pull_post_state` (offline re-verification) | `live::reverify_with_judge` | **M** |
| verifier cross-check (Live RL anti-reward-hacking) | `verifier::cross_check` + `live::cross_check_disagreements` | **L** |

## Agent loop (verl AgentLoop / mimoagent DefaultAgent)

| upstream concept | artifact | mode |
|---|---|---|
| append-only message list, `[system, instance]` priming | `agentloop::Message` + `AgentLoop::run` | **P** |
| `execute_action` dispatch (parallel, ordered fold-back) | `agentloop::AgentLoop::dispatch` | **P** |
| `ToolException` → user-turn observation vs infra error | `agentloop::ToolDispatch::{kind}` (`ok` / `tool_exception` / `transport_error`) | **P** |
| infra error categories (`setup/failed`, `rollout/pod_conn_timeout`, `rollout/seq_timeout`, `reward/*`) | `error::InfraErrorKind::as_category` | **P** |
| observation truncation | `agentloop::truncate_observation` | **P** |
| `step_limit` + forced final turn | `LoopConfig::step_limit` | **P** |
| `TokenTrace` response budget / `ResponseBudgetExhausted` | `LoopConfig::response_budget` + forced wrap-up | **M** |
| the RL bridge (`_VerlRolloutModel.query` re-entering the training engine) | `agentloop::Policy` trait (scripted policies in tests; the model plugs in here) | **P** |
| session log recording | `SimPod::append_session` on every assistant turn | **P** |

## Live RL trainer (the MiMo-V2.6 recipe)

| upstream concept | artifact | mode |
|---|---|---|
| fully asynchronous GRPO on large batches (1,568 prompts × 16 rollouts upstream) | `live::LiveTrainer::step` (tokio `JoinSet` + `spawn_blocking`, bounded `concurrency`, one fresh pod per rollout) | **L** |
| GRPO group-relative advantage `(r - mean)/std` | `live::grpo_advantages` (masked rollouts excluded from stats — never averaged, never trained) | **L** |
| **GAR** — Groupwise Advantage Redistribution (rank passing trajectories, move advantage toward higher quality: shorter paths, fewer tokens) | `live::gar` (mass-conservative rank-weighted redistribution, `gar_strength`) | **L** |
| **GRS** — Groupwise Reward Synthesis (task-specific rubrics offline from contrasting rollouts, fusing rubric quality with test outcomes) | `live::grs` (contrastive needle synthesis from pass/fail answer sets) | **L** |
| adversarial screening + environment hardening | `live::adversarial_screening` (zero-tool full scores, cross-check disagreements) | **L** |
| verifier cross-checks (independent judge) | `live::cross_check_disagreements` | **L** |
| aligned RL cold start (self-correction pairs) | `live::self_correction_pairs` (misaligned answer ↔ grounded rewrite from the group's best pass) | **L** |
| rollout determinism per (row, member) regardless of scheduling | `live::rollout_seed` (FNV over instance id + member) | **L** |

## AgentEnv server (Scale agentenv-framework)

| upstream concept | artifact | mode |
|---|---|---|
| env-server / env-client separation, environments over REST | `server::EnvServer` (axum, feature `server`) | **L** |
| env lifecycle (create / observe / step / reset / destroy) | `POST /envs`, `GET /envs[/{id},/{id}/obs]`, `POST /envs/{id}/tools/call`, `POST /envs/{id}/reset`, `DELETE /envs/{id}` | **L** |
| reward over the API with the masking contract intact | `POST /envs/{id}/verify` → `reward` or `masked + category + reward_error` | **L** |
| server counters | `GET /metrics` | **L** |

## Repo2RLEnv (HuggingFace)

| upstream concept | artifact | mode |
|---|---|---|
| mine commits that change source + tests | `repo2rl::RepoMiner::mine` (merge/bot filtering, diff-size caps, min test cases) | **P** |
| task synthesis (problem statement from the commit, context rendering) | `repo2rl::RepoMiner::render_statement` | **P** |
| golden patch vs test patch separation | `RepoTask::{golden_patch, test_patch}` | **P** |
| test-driven reward (per-case items + apply-cleanly gate) | `RepoTask::verifier` | **P** |
| verl row emission (`cwd = /testbed`, `docker_image`) | `repo2rl::emit_dataset` | **P** |
| git history ingestion | `repo2rl::parse_git_log` (`git log --name-status` shape) | **M** |

## The generic factory (this repo's own contribution)

| concept | artifact |
|---|---|
| declarative environment spec (systems / tools / workspace / rubric, plain JSON) | `envgen::EnvSpec` |
| compiled CRUD tool surface (`get`/`list`/`update`/`insert`/`delete` per table) | `envgen::ToolOp` + `ToolSpec::compile` |
| bootable bundle for the loop + trainer | `envgen::EnvSpec::compile` → `live::EnvBundle` |
| seeded instance generation (scaled money literals propagated to rows/needles/workspace, distractor rows) | `envgen::Variants::generate` |
| domain templates (knowledge work, terminal/computer use, webdev) | `envgen::templates` |
| provider for the trainer | `envgen::SpecProvider` |

## What is deliberately simulated

The port is a **deterministic userspace test bed**, not a Docker/K8s
replacement. Concretely:

- containers are volume mounts + a command router, not real namespaces
  (the observable isolation — path resolution, read-only mounts, port
  liveness — matches the real pod's behavior);
- the LLM judge is an anchor matcher with injectable failure modes, so
  the reward pipeline (including the masking contract) is replayable
  without a hosted model;
- the "python" entry scripts route to native Rust handlers — the
  *contract* (who may run what, where, and what files appear) is
  preserved byte-for-byte in the JSON payloads the tests pin;
- the policy is a trait with scripted implementations; an actual model
  plugs in at `agentloop::Policy` the way verl's rollout engine does.

Everything else — schemas, port numbers, path constants, error strings,
timeout defaults, the manifest invariants, the reward-file contract,
the GRPO/GAR/GRS math — is a direct port.
