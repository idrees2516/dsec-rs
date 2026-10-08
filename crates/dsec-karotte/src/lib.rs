//! # dsec-karotte
//!
//! **A Rust port of the Karotte robust RL environment framework**
//! ([preferencemodel/karotte](https://github.com/preferencemodel/karotte))
//! — the harness that trains agents inside VMs they can attack, and grades
//! them with judges they cannot reach.
//!
//! Karotte's design premise: the RL student is a *trained adversary*.
//! Whatever raises its score gets reinforced — including reading the
//! answer off disk, killing the grader, or planting symlinks for it to
//! follow. The framework's answer is layered confinement (cgroups,
//! uid demotion, iptables owner rules, loop-mounted disk quotas, fd-safe
//! reaping) plus a scoring model that separates **student misbehavior**
//! (score 0) from **infrastructure failure** (mask the run).
//!
//! This crate ports the framework's core in Rust, as deterministic
//! userspace logic: rule builders, protocol models, and algorithms that
//! produce and consume the *same artifacts* as upstream (iptables
//! commands, cgroup file contents, event JSON, judge trees), so behavior
//! is testable without root or VMs.
//!
//! ## The stack, layer by layer
//!
//! | module | role | upstream counterpart |
//! |--------|------|----------------------|
//! | [`schemas`] | chat messages, scorings, the 14-event union, transcript, run state, run config | `karotte/schemas/` |
//! | [`text`] | surrogate escaping, middle truncation, JSON-byte-aware heads, metadata caps | `text.py`, `truncation.py` |
//! | [`task`] | `Task`/`Step` contracts, the declarative task factory, registry | `task.py`, `step.py`, `task_factory.py`, `load_tasks.py` |
//! | [`judges`] | regex / rubric (LLM) / executable judges, `And`/`Or` short-circuit composites with the Unicode tree, rubric contexts | `karotte/judges/` |
//! | [`model_spec`] | the model catalog tables, reasoning-effort ladders, provider request shaping | `model_spec.py`, `model_catalog.py`, `providers.py` |
//! | [`message_loop`] | the agent loop: run-wide turn limit, between-turn time/context checks, empty-turn nudges, tool-call projection, counter injection, retry policy | `agents/message_loop.py`, `agents/builtin_source.py` |
//! | [`mcp`] | the MCP tool server (JSON-RPC subset: initialize, tools/list, tools/call, register_tools), tool discovery validation, resource sampling | `mcp_servers/` |
//! | [`tools`] | `bash`, `view_lines_in_file`, `replace_in_file` (with the hand-rolled unified diff), `view_image_file`, and the demoted-subprocess discipline | `karotte/tools/`, `demoted.py` |
//! | [`confinement`] | the `Contract` tri-state, resource limits, the iptables firewall builder, canary self-tests, cohort reaping | `confinement.py`, `process_utils.py` |
//! | [`cgroups`] | cgroup v1/v2 mount parsing and the student-group model (memory+swap, pids, oom.group, cgroup.kill) | `cgroups.py` |
//! | [`memory_watch`] | the watchdog weigher: RSS/PSS, tmpfs scratch, SysV shared memory; reap loops | `memory_watch.py` |
//! | [`reclaim`] | the fd-safe DFS sweep that deletes everything the uid owns; SysV segment cleanup | `reclaim.py` |
//! | [`submission`] | defensive custody copy of student output (no symlinks, hard caps); `untrusted_paths` fd primitives | `save_submission.py`, `untrusted_paths.py` |
//! | [`runner`] | the evaluation runner: the exact event flow, scoring order, durable transcript | `evaluation_runner.py` |
//! | [`streaming`] | event fan-out: the stdout renderer, the websocket broadcaster model, the backend client | `transcript_streaming/` |
//! | [`runtime`] | runtime selection, the Firecracker VM plan (pinned artifacts, drives, boot args, network, watchdogs) | `runtime.py`, `firecracker/`, `apple_container.py` |
//!
//! ## The one contract that matters
//!
//! A **task failure** (agent did badly → legitimate zero) and a **student
//! misbehavior** (agent cheated → also zero, but *labeled*) and an
//! **infrastructure failure** (grader broke → mask, never train) are three
//! different things:
//!
//! ```
//! use dsec_karotte::error::StudentMisbehaviorError;
//! use dsec_karotte::schemas::Scoring;
//!
//! let s = Scoring::misbehavior(&StudentMisbehaviorError::Symlink {
//!     path: "/workdir/answer.txt".into(),
//! });
//! assert_eq!(s.score, 0.0);
//! assert!(!s.continue_task);
//! assert!(s.metadata.contains_key("misbehavior"));
//! ```
//!
//! Every security decision in [`confinement`] returns a
//! [`Contract`](confinement::Contract) saying whether the kernel *refused*
//! the operation, the watchdog *reaped* it, or the host simply *does not
//! support* it — a "successfully written" limit on gVisor enforces nothing,
//! and Karotte says so out loud.

#![deny(missing_docs)]

pub mod cgroups;
pub mod confinement;
pub mod error;
pub mod judges;
pub mod mcp;
pub mod memory_watch;
pub mod message_loop;
pub mod model_spec;
pub mod reclaim;
pub mod runner;
pub mod runtime;
pub mod schemas;
pub mod streaming;
pub mod submission;
pub mod task;
pub mod text;
pub mod tools;

pub use error::{Error, Result, StudentMisbehaviorError};
pub use schemas::{Event, Message, RunStatus, Scoring, Transcript};
