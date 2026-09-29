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

pub mod buffer;
pub mod driver;
pub mod envpool;
pub mod error;
pub mod fastenv;
pub mod lstm;
pub mod sandbox_env;
pub mod spaces;

pub use buffer::{EpisodeChunk, ReplayBuffer, SequenceBatch, StepSlice, Transition};
pub use driver::{Driver, DriverConfig, PolicyFn, RolloutStats};
pub use envpool::{Env, EnvPool, StepResult};
pub use spaces::{Action, ActionSpace, ObsSpace};
