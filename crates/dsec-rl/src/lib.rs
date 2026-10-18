//! # dsec-rl
//!
//! The core of **PufferLib**, ported from Python/Cython to Rust, wired to
//! agent sandboxes through the DSec data plane.
//!
//! Ported components:
//! - [`buffer::ReplayBuffer`] — `PufferReplayBuffer`: step-major episode
//!   ring, per-step LSTM hidden storage with **reset-on-done**, mat-obs
//!   and flat-obs views, GAE, sequence sampling, zero-copy dump into a
//!   `SharedRegion` (the paper's torch shared-memory handoff).
//! - [`envpool::EnvPool`] — `PufferEnv`: vectorized stepping with
//!   `reset_if_done` semantics and scoped-thread parallelism.
//! - [`driver::Driver`] — the rollout loop that maintains hidden state,
//!   zeroes it at episode ends, enqueues transitions and computes GAE.
//! - [`lstm::LstmCell`] — deterministic recurrent cell used by example
//!   policies and the hidden-reset tests.
//! - [`sandbox_env::TerminalTaskEnv`] — **agent sandboxes as RL
//!   environments**: observations encode real command output from the
//!   Chronus session; actions are shell commands; the reward is solving
//!   a file-search task — the paper's RL-for-agents workload in
//!   miniature.
//! - [`fastenv::FastCounterEnv`] — pure-compute env for throughput
//!   floors (pufferlib's "release the GIL" analogue: here, no I/O).
//! - [`tito`] — **TITO records** (Tool-Interleaved Trajectory
//!   Optimization): token-level origin classification (model-generated /
//!   tool-result / harness-injected / system-template) with target
//!   masks, plus the deterministic shared tokenizer.
//! - [`trajectory`] — trajectory trees with shared prefixes: executions
//!   of one task group share prefix nodes; shared targets are counted
//!   once (QWENGYRE's trajectory processor semantics).
//! - [`reward`] — reward validity tri-state (valid / invalid /
//!   evaluator-failure — never a false zero) and GRPO-style group
//!   advantages over scorable members only.
//! - [`loss`] — execution-level loss normalization: per-execution token
//!   averaging (an execution with more turns carries no extra mass),
//!   merged shared-target terms, and the [`loss::LossTerm`] contract the
//!   trainer consumes.

pub mod buffer;
pub mod driver;
pub mod envpool;
pub mod error;
pub mod fastenv;
pub mod loss;
pub mod lstm;
pub mod reward;
pub mod sandbox_env;
pub mod spaces;
pub mod tito;
pub mod trajectory;

pub use buffer::{EpisodeChunk, ReplayBuffer, SequenceBatch, StepSlice, Transition};
pub use driver::{Driver, DriverConfig, PolicyFn, RolloutStats};
pub use envpool::{Env, EnvPool, StepResult};
pub use loss::{compute_loss_terms, compute_loss_terms_with, ExecutionInput, LossOutput, LossTerm};
pub use reward::{group_advantage, AdvantageConfig, RewardRecord, RewardValidity};
pub use spaces::{Action, ActionSpace, ObsSpace};
pub use tito::{SimpleTokenizer, TitoRecord, TokenOrigin, TranscriptTurn};
pub use trajectory::{TrajectoryNode, TrajectoryTree};
