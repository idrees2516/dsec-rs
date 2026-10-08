# dsec-autoenv

**A deep Rust implementation of the AutoEnvScaling data flywheel for
terminal-agent RL** — [Yu et al., 2026, *AutoEnvScaling: Automating the Data
Flywheel with Terminal Agents*](https://simonucl.github.io/assets/pdf/autoenvscaling.pdf).

Terminal agents are trained by RL, which needs diverse, verifiable
environments — but environments are designed by human researchers, "so their
supply grows only as fast as researchers can work." AutoEnvScaling turns
environment **design** into a terminal task and closes the loop:

```
        ┌────────────────────────────────────────────────────────┐
        │                        ROUND r                          │
        └────────────────────────────────────────────────────────┘
 pool review (every 16 updates)          workspace refresh
   evict out-of-band envs        ─▶  rollouts + past envs in,
   ask for replacements              memory + tools persist
        │                                       │
        ▼                                       ▼
   assignments ──▶ proposer episodes ──▶ host admission
   (parent, pass      (sandbox: read,      (validation: build +
    rate, failure,     design, build,      reference = 1, no-op < 0.5,
    simplify/harden)   validate, submit)   13-gram, rubric review)
        │                                       │
        │                              calibrate (4→8 rollouts,
        │                              [0.25, 0.75], std ≥ 0.1)
        │                                       │
        ▼                                       ▼
   r_P (Eq. 2: -1 / -0.25 / +1)  ──▶  admitted envs enter the pool
        │                                       │
        ▼                                       ▼
  proposer DPPO (every k updates)     solver DPPO (every step)
                                            │
                                            ▼
                              new rollouts → next round  (the flywheel)
```

The same policy can play both roles — recursive self-improvement.

## What is implemented

Every core mechanism of the paper, as deterministic, testable Rust:

| concept | implementation |
|---|---|
| Environment design as a terminal task (§3.1) | [`policy::ScriptedProposer`] episodes: read workspace → sample design move → build → validate → submit, inside a [`sandbox::Sandbox`] with Table-10 resources, egress rules, and web-tool gating |
| Harbor task model (§3.1, App. E.1) | [`harbor`]: scaffold, `task.toml`, multi-step `steps/` with `mean`/`final` roll-up and `min_reward` gates, three-layer network policy (baselines / phase overrides / run-time merges), shared vs. separate verifiers, the nine authoring steps, all common pitfalls as lints |
| Proposer workspace (Fig. 18, App. D.1) | [`workspace`]: the exact layout with editable / host-managed / fixed regions, per-round refresh, persistent harness state |
| The two skills (App. E) | [`skills`]: Harbor's `create-task` (verbatim) and the `auto_env_scaling` overlay (verbatim), plus the managed fraction-scoring verifier template |
| Validation (§3.2, App. G.4) | [`validate`]: build + reference-reward-1 + no-op-<0.5 + decontamination + rubric review, run outside the sandbox; the proposer's own pre-submission check with the no-op swap dance |
| 13-gram decontamination (App. F.2) | [`decontaminate`]: sliding window n=13, stride 1, against held-out benchmarks |
| `harbor check` rubric (App. G.4) | [`rubric`]: the Terminal-Bench task-implementation criteria, `binary_reward` excluded |
| Solver-guided calibration (§3.2, App. G.4) | [`calibrate`]: 4 → 8 rollouts (within 0.15 of a band edge), mean ∈ [0.25, 0.75], std ≥ 0.1, one hinted diagnostic rollout for below-band tasks, ≤ 2 revisions |
| Assignments from rollouts (§3.1) | [`assignment`]: parent env, pass rate, most common failure, simplify-when-failing / harden-when-succeeding (longer horizon, additional skill) |
| Eq. 1 solver reward | [`world`]: CTRF fraction-of-tests-passed, partial credit |
| Eq. 2 proposer reward | [`reward`]: −1 / −0.25 / +1 |
| Eq. 3 + DAPO (§3.3) | [`advantage`]: role-separated group-relative advantages, mean-centered without std division, zero-variance group dropping, oversampled batch filling |
| DPPO (§3.3, Table 12) | [`dppo`]: clipped surrogate with analytic gradients, total-variation trust region (0.1) enforced by exact line search, carry-over of unfinished trajectories with stored log-probabilities |
| Training pool (Table 12) | [`pool`]: 256-capacity pool, 16-update review, eviction + replacement flow |
| The full loop | [`flywheel`]: rounds, event stream, all Table-12 settings ([`FlywheelConfig::paper`]) |
| Harness optimization (§3.3) | [`harness`]: Continual-Harness evolution between rounds (merged validation script, web-cache skill, lessons); the checks stay fixed |
| Cold-Start (§4.3, App. H) | [`coldstart`]: 600-trajectory selection from frontier episodes, paired sign test, bootstrap intervals |
| Metrics (§4) | [`metrics`]: Wilson 95% intervals, cost per valid env, avg@5 / pass@5, useful-group rate, collection time |
| HiL clarification (§4.5, App. B) | [`hil`]: the Figure-9 billing task with Table-6's five blocker types, `ask_human`, Ask-F1, shaped reward, derived full-info / no-tool variants |
| Cross-domain (App. C) | [`domains`]: the seven domains of Table 7, Table-8 acceptance data, admission with reset reproducibility |

## The one contract that matters

An environment's value is **reward spread across the current solver's
attempts**, not average difficulty. "`[0, 1]` and `[0.5, 0.5]` both average
0.5, but only the first has group-relative learning signal; the second has
zero GRPO signal." Everything in `calibrate`, `reward`, and `advantage`
exists to keep that signal alive.

```rust
use dsec_autoenv::reward::proposer_reward;

// Eq. 2 separates broken environments from mis-calibrated ones.
assert_eq!(proposer_reward(false, false), -1.0);   // validation failed
assert_eq!(proposer_reward(true, false), -0.25);   // calibrated out of band
assert_eq!(proposer_reward(true, true), 1.0);      // entered the pool
```

## Quick start

```bash
cargo run -p dsec-autoenv --example flywheel
cargo test  -p dsec-autoenv
```

The example runs six full rounds of the flywheel with a deterministic
simulated solver and proposer: pool reviews evict environments the solver
outgrows, replacements are issued (harden-when-succeeding), admission
validates and calibrates every submission, DPPO trains both roles, harness
optimization grows the web-cache skill — and the closing metrics report the
valid-environment rate (with a 95% Wilson interval), cost per valid
environment, useful-group rate, and collection time, followed by the HiL,
Cold-Start, and cross-domain pipelines.

## Paper configuration

```rust
use dsec_autoenv::flywheel::FlywheelConfig;

let config = FlywheelConfig::paper();
// Table 12:
//   16 environments x 16 trajectories per solver update
//   128 proposer trajectories per assignment, every 16 updates
//   active pool of 256 tasks, reviewed every 16 updates
//   DPPO TV threshold 0.1, KL/entropy coefficients 0/0, LR 1e-6
//   calibration band [0.25, 0.75], std >= 0.1, 4 -> 8 rollouts
```

See `docs/autoenvscaling.md` for the complete paper → artifact mapping.

## Design note

The paper executes tasks in real containers with real LLMs. This crate
reproduces the *observable contracts* of that execution as pure,
deterministic logic (file-based worlds, CTRF reports, tabular softmax
policies with real token log-probabilities), so the entire flywheel —
admission, calibration, advantage computation, DPPO with trust regions,
carry-over, harness evolution — runs and is testable without Docker or
GPUs. The `SolverModel` / proposer-policy surfaces are traits: swap in a
live backend without touching the loop.
