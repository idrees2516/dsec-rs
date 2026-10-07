//! # dsec-agentenv
//!
//! **Agentic RL environment harness** — a Rust port of the MiMo Live RL
//! training environments (the HuggingFace dataset
//! `XiaomiMiMo/MiMo-V2.6-RL-oss` and its verl-side harness), plus the
//! Scale AgentEnv server contract, the Repo2RL conversion pipeline, and
//! a generic factory for building new environments of the same shape.
//!
//! Everything runs as a deterministic userspace simulation: no Docker,
//! no root, no network — reproducible on any machine, seeded where
//! randomness matters.
//!
//! ## The stack, layer by layer
//!
//! | module | role | upstream counterpart |
//! |--------|------|----------------------|
//! | [`task`] | dataset rows: `prompt`/`data_source`/`ability`/`agent_name`/`reward_model`/`extra_info` (+ inner `instance_json`), JSONL shards, the five domain presets | `RLHFDataset`, `MimoAgentSWEDataset`, `build_parquet.py` |
//! | [`manifest`] | `manifest.json`: uploads, setup, `wait_ports`, `mcp_servers`, verifier bundle; default synthesis | `GeneralAgentEnvironment._load_manifest` / `_default_manifest` |
//! | [`topology`] | the two-container pod (`main` + `sidecar`), volumes with RW/RO mounts, the simulated shell, MCP port lifecycle, session logs, source protection | the k8s pod topology + `base_env.execute`/`copy_to`/`copy_out` |
//! | [`state`] | in-memory relational `state.db`s with schema, CRUD, and mutation accounting | `/work/system/<mcp>/state.db` |
//! | [`mcp`] | MCP JSON-RPC 2.0 envelopes, `tools/list` discovery, `tools/call`, OpenAI function-schema conversion | `mcp_proxy.discover_mcp_tools`, the streamable-http sidecar |
//! | [`verifier`] | rubric items (`verifier_meta.json`), the rubric engine, the LLM-judge contract, `reward.json` emission, and the **REWARD_TESTBED_CORRUPTED masking** pipeline end to end | `run_verify.py`, `verify.py`, `_do_calculate_reward` |
//! | [`agentloop`] | multi-turn rollout: append-only transcript, tool dispatch, observation truncation, step limit, response budget, error taxonomy | mimoagent `DefaultAgent`, verl `AgentLoop` |
//! | [`live`] | the Live RL trainer: fully-async group rollouts, GRPO advantages, **GAR**, **GRS**, adversarial screening, verifier cross-checks, self-correction cold start | the MiMo-V2.6 Live RL recipe |
//! | [`envgen`] | the generic environment factory: declarative [`EnvSpec`](envgen::EnvSpec) → bootable env; seeded variant generation; domain templates | — (this repo's own contribution) |
//! | [`repo2rl`] | repository → environments: commit mining, task synthesis, test-driven verifiers, row emission | HuggingFace Repo2RLEnv |
//! | server | the AgentEnv REST server (create / obs / tool-call / reset / verify / destroy) | Scale `agentenv-framework` env-server |
//!
//! ## The one contract that matters
//!
//! A **task failure** (agent did badly → reward 0) and a **testbed
//! failure** (environment broke → rollout masked) are never confused:
//!
//! ```
//! use dsec_agentenv::verifier::{RewardOutcome, VerifierHarness, Rubric,
//!     RubricItem, RuleCheck, AnchorJudge, RewardConfig};
//!
//! let rubric = Rubric::new().item(RubricItem::rule(
//!     "answer", 1.0, RuleCheck::AllText { needles: vec!["42".into()] }));
//! # let mut pod = dsec_agentenv::topology::SimPod::builder("t").build();
//! # let manifest = dsec_agentenv::manifest::Manifest::default();
//! let judge = AnchorJudge::new();
//! let harness = VerifierHarness::new(&rubric, &judge, RewardConfig::default());
//! let rollout = Default::default();
//! match harness.calculate_reward(&mut pod, &manifest, &rollout) {
//!     RewardOutcome::Valid { score, .. } => { /* train on it */ let _ = score; }
//!     RewardOutcome::TestbedCorrupted { kind, .. } => {
//!         // mask the sequence — never train on the false zero
//!         let _ = kind;
//!     }
//! }
//! ```
//!
//! ## Feature flags
//!
//! * `server` — the axum REST server (the `server` module).
//! * `parquet` — read the real HuggingFace parquet files (arrow-based).
//!
//! The default build needs neither.

#![deny(missing_docs)]

pub mod agentloop;
pub mod envgen;
pub mod error;
pub mod live;
pub mod manifest;
pub mod mcp;
#[cfg(feature = "parquet")]
pub mod parquet;
pub mod repo2rl;
#[cfg(feature = "server")]
pub mod server;
pub mod state;
pub mod task;
pub mod topology;
pub mod verifier;

pub use error::{Error, InfraErrorKind, Result, RewardErrorKind};
pub use task::{
    ChatMessage, Domain, ExtraInfo, RewardModel, TaskDataset, TaskInstance, TaskRow, TaskRowBuilder,
};
pub use topology::{SimPod, ToolCtx, ToolDef};
pub use verifier::{Judge, RewardOutcome, Rubric, RubricItem, RuleCheck};
