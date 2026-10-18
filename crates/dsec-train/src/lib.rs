//! # dsec-train
//!
//! Fixed-pool and elastic agentic RL training (Phases 5–6 of the
//! phased build):
//!
//! - [`lm`] — a tiny, genuinely-trainable language model with
//!   hand-written forward/backward policy gradients over the Phase 4
//!   loss terms, deterministic sampling, and SHA-256 weight digests.
//! - [`policy`] — explicit policy versions + the publication barrier:
//!   exactly one published version, staged checkpoints, crash-safe
//!   publication (an interrupted flip leaves the OLD version served
//!   and the barrier closed until recovery).
//! - [`trainer`] — the fixed-pool trainer: bounded staleness (samples
//!   older than `published − staleness_bound` are rejected, never
//!   trained), the Phase 4 loss path, real gradient steps, and
//!   barrier publication per step.
//! - [`scheduler`] — the elastic GPU scheduler: cell roles, draining
//!   transitions with safe rerouting (in-flight model calls always
//!   finish), rollout waterlevel, joinable streaming training batches,
//!   and publication-barrier-gated role flips.
//! - [`bench`] — Async (fixed split) vs Colocate (phase-serialized) vs
//!   Elastic on a simulated, calibrated workload.
//! - [`server`] (feature `server`) — an OpenAI-compatible policy model
//!   server over real HTTP, reporting the served policy version on
//!   every response.
//!
//! What is real here: the version/staleness/barrier contracts, the
//! gradient math, the HTTP serving path, and the scheduler's
//! invariant set. What is deliberately toy: the model size (production
//! swaps in vLLM/SGLang + a verl-style trainer) and the benchmark's
//! logical clocks (calibrated simulation, not GPU wall-clock).

pub mod bench;
pub mod error;
pub mod lm;
pub mod policy;
pub mod scheduler;
pub mod trainer;

#[cfg(feature = "server")]
pub mod server;

pub use error::{Error, Result};
pub use lm::TinyLm;
pub use policy::{Checkpoint, PolicyStore, PolicyVersion, PublicationBarrier};
pub use scheduler::{Cell, CellRole, ElasticPool, SchedulerConfig, TickSummary, Waterlevel};
pub use trainer::{GradTerm, RolloutSample, TrainStepResult, Trainer, TrainerConfig};
