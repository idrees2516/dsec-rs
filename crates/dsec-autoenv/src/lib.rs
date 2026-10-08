//! # dsec-autoenv
//!
//! **A deep Rust implementation of the AutoEnvScaling data flywheel**
//! (Yu et al., 2026, *"AutoEnvScaling: Automating the Data Flywheel with
//! Terminal Agents"*) — the pipeline that turns environment **design** into
//! a terminal task so agents can build their own RL training environments.
//!
//! ## The idea being implemented
//!
//! Terminal agents are trained by RL, which needs diverse, verifiable
//! environments — but environments are designed by human researchers, "so
//! their supply grows only as fast as researchers can work." AutoEnvScaling
//! closes the loop:
//!
//! 1. a **proposer** agent, inside a sandbox with a coding harness, reads
//!    the solver's rollouts and builds complete Harbor environments
//!    (instruction + container + reference solution + tests) as terminal
//!    work;
//! 2. the host **validates** (build; reference solution must earn reward 1;
//!    empty solution must earn < 0.5; 13-gram decontamination; rubric
//!    review) and **calibrates** each environment against the *current*
//!    solver (mean reward in [0.25, 0.75] with reward variation);
//! 3. the **solver** trains on the admitted pool (Eq. 1: fraction of tests
//!    passed), and its new rollouts guide the next round;
//! 4. the **proposer** is trained too (Eq. 2: −1 / −0.25 / +1) with
//!    role-separated group advantages (Eq. 3) under DPPO, and improves its
//!    **harness** (memory, skills, tools) between rounds;
//! 5. every k solver updates the pool is **reviewed**: out-of-band
//!    environments are evicted and the proposer is asked for replacements.
//!
//! The same policy can play both roles — recursive self-improvement.
//!
//! ## Module map
//!
//! | module | role | paper section |
//! |--------|------|---------------|
//! | [`harbor`] | the complete Harbor task model: scaffold, `task.toml`, multi-step `steps/`, three-layer network policy, verifier modes, pitfalls-as-lints | §3.1, App. E.1 |
//! | [`world`] | simulated container execution: builds, solution scripts, CTRF fraction rewards, Oracle, no-op, reset reproducibility | §3.1–3.2, App. E.2 |
//! | [`reward`] | Eq. 1 / Eq. 2 rewards, multi-step roll-up, shaped clarification reward | §3.3, §4.5 |
//! | [`decontaminate`] | 13-gram sliding-window contamination check vs. held-out benchmarks | §3.2, App. F.2 |
//! | [`rubric`] | the `harbor check` task-review rubric (Terminal-Bench criteria, `binary_reward` excluded) | §3.2, App. G.4 |
//! | [`validate`] | host admission: build + reference-1.0 + no-op-fails + decontamination + review | §3.2 |
//! | [`calibrate`] | adaptive solver calibration: 4→8 rollouts, band, std, diagnostics, revisions | §3.2, App. G.4 |
//! | [`assignment`] | assignments from solver rollouts: parent env, pass rate, most common failure, simplify/harden direction | §3.1 |
//! | [`workspace`] | the Figure-18 proposer workspace: editable / retained / fixed regions, per-round refresh | §3.1, App. D.1 |
//! | [`sandbox`] | sandbox resources (Table 10), egress rules, web-tool gating, `sandbox.toml`, `SANDBOX_CONTRACT.md` | §3.1, App. D.2–D.3 |
//! | [`skills`] | the two proposer skills (Harbor `create-task` + the `auto_env_scaling` overlay) and the managed verifier template | App. E |
//! | [`memory`] | persistent memory: lessons, past environments, trajectory archive with `index.jsonl` provenance | §3.1, App. D.1 |
//! | [`harness`] | Continual-Harness optimization between rounds (memory / skills / tools evolve; checks stay fixed) | §3.3 |
//! | [`policy`] | pluggable policies for the two roles plus the `solver_model` pre-submission calibration tool | §3.1–3.3 |
//! | [`pool`] | the 256-task training pool with 16-update review and eviction | §3.3, App. G.1 |
//! | [`advantage`] | Eq. 3 role-separated group advantages, DAPO filtering, oversampled batch filling | §3.3 |
//! | [`dppo`] | DPPO updates under a total-variation trust region, with carry-over of unfinished trajectories | §3.3, App. G.1 |
//! | [`flywheel`] | the orchestrator: rounds, events, the full loop | §3, App. G |
//! | [`coldstart`] | Cold-Start fine-tuning data selection and paired controls | §4.3, App. H |
//! | [`metrics`] | Wilson intervals, cost per valid env, avg@5 / pass@5, bootstrap, sign test | §4, App. H |
//! | [`hil`] | clarification tasks: blocker registry, `ask_human`, Ask-F1, three instruction settings | §4.5, App. B |
//! | [`domains`] | cross-domain generation and admission across the seven agentic domains | App. C |
//! | [`trajectory`] | turns, actions, outcomes, rollout records, trajectory groups | §3.3 |
//!
//! ## The one contract that matters
//!
//! An environment's value to the flywheel is *reward spread across the
//! current solver's attempts*, not average difficulty. The skill says it
//! directly: "`[0, 1]` and `[[0.5, 0.5]]` both average 0.5, but only the
//! first has group-relative learning signal; the second has zero GRPO
//! signal." Everything in [`calibrate`], [`reward`], and [`advantage`]
//! exists to keep that signal alive.
//!
//! ```
//! use dsec_autoenv::reward::proposer_reward;
//!
//! // Eq. 2: separate broken environments from mis-calibrated ones.
//! assert_eq!(proposer_reward(false, false), -1.0);
//! assert_eq!(proposer_reward(true, false), -0.25);
//! assert_eq!(proposer_reward(true, true), 1.0);
//! ```

#![deny(missing_docs)]

pub mod advantage;
pub mod assignment;
pub mod calibrate;
pub mod coldstart;
pub mod decontaminate;
pub mod domains;
pub mod dppo;
pub mod error;
pub mod flywheel;
pub mod memory;
pub mod metrics;
pub mod policy;
pub mod pool;
pub mod sandbox;
pub mod skills;
pub mod workspace;

pub mod harbor;
pub mod harness;
pub mod hil;
pub mod reward;
pub mod rubric;
pub mod validate;

pub mod trajectory;

pub mod world;

pub use error::{Error, Result};
