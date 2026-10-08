# AutoEnvScaling in Rust — paper → artifact map

Source: *AutoEnvScaling: Automating the Data Flywheel with Terminal Agents*
(Yu, Peng, Liu, Shi, Wang, Cheng, Wu, Yao, Jaques, Shi, Gao — Northeastern
University / Microsoft / Stanford / UW / UNC, 2026).
Crate: `crates/dsec-autoenv`.

The paper's core claim: environment design can be a terminal task. A
proposer agent with a coding harness reads the solver's rollouts, builds
complete Harbor environments (instruction + container + reference solution
+ tests) inside a sandbox, and the host validates and calibrates each one
against the *current* solver before it enters the training pool. The
solver's new rollouts guide the next round. Both roles are trained (DPPO,
role-separated group advantages), and the proposer's harness (memory,
skills, tools) also evolves between rounds.

## Section-by-section map

### §3.1 — Turning environment design into a terminal task

| paper element | artifact |
|---|---|
| "a complete Harbor task ... with an instruction, an execution environment, a reference solution, and tests" | `harbor::HarborTask` (+ `world::TestSuite`, `world::SolutionScript`) |
| "The solver sees only the instruction ... reference solution and tests stay hidden" | `world::run_attempt` plants environment state, runs a script, runs the verifier; the solver policies never see `task.solution` (hints are explicit `Hint` objects for diagnostic rollouts only) |
| "The proposer works in a sandbox, an isolated container with a workspace, child containers ... restricted web access" | `sandbox::Sandbox` (+ `spawn_child`, egress classes, web-tool gating) |
| Workspace (Figure 18) | `workspace::Workspace` — `skills/`, `tools/` (validate.py, decontaminate.py, calibrate.py, selftest.sh, snippets/), `cookbook/`, `memory/` (lessons.md, past_environments/, past_trajectories/{failed,successful}), `output/tasks/`, `logs/`, `assignment/`, `SANDBOX_CONTRACT.md` |
| "The rollouts and past environments are refreshed from the training pool each round, while the memory and tools persist" | `Workspace::refresh_from_pool` (host-side) vs `Workspace::persist_harness_state` |
| "can edit its memory and tools but not the instructions or the admission checks, which run outside the sandbox" | `workspace::region_of` with `Phase::DuringEpisode` / `Phase::BetweenRounds`; admission checks in `validate.rs` never read the workspace |
| "egress rules block Terminal-Bench and other benchmark pages" | `sandbox::BLOCKED_BENCHMARK_HOSTS` + `EgressPolicy::check` |
| Assignment "names a parent environment ... and gives the solver's pass rate and most common failure" | `assignment::build_assignment`, `assignment::most_common_failure` |
| "builds a simpler variant when the solver keeps failing, or a harder one with a longer horizon or an additional skill when the solver succeeds" | `assignment::Direction` + `policy::size_for` + `policy::generate_task` |
| "It can also run the current solver on a new environment to estimate its difficulty" | `policy::SolverModelTool` (the `solver_model` tool) |
| Harbor's create-task skill, end-to-end | `skills::CREATE_TASK_SKILL` (verbatim from App. E.1) + `harbor::scaffold_task` |
| Appendix D.2 sandbox resources (Table 10) | `sandbox::SandboxResources::proposer` (16 CPU / 32 GiB / 2 h / no turn limit) and `::solver` (4 CPU / 8 GiB) |
| Appendix D.3 image + `sandbox.toml` (Figure 19) | `sandbox::SandboxImage::render_dockerfile`, `sandbox::SandboxConfig::render_toml` |

### §3.2 — Environment validation and calibration

| paper element | artifact |
|---|---|
| "the host checks that it works and that it suits the current solver" | `validate::validate` then `calibrate::calibrate` |
| reference solution "should receive full reward, and an empty solution ... less than half" | `world::oracle_run`, `world::noop_run`, `validate::ValidationConfig::max_noop_reward = 0.5` |
| "flag any instruction that shares a 13-gram with a held-out task, following Tmax" | `decontaminate::ContaminationIndex` (n = 13, stride = 1) |
| "we run Harbor's task review (harbor check) ... against the official Terminal-Bench rubric" | `rubric::review_task` |
| "We apply every criterion except binary_reward ... because our tests give partial credit" | `rubric::RubricConfig::exclude_binary_reward = true` |
| "attempted several times by the current solver ... mean reward falls between 0.25 and 0.75 and the rewards vary" | `calibrate::CalibrationConfig` (band, `min_std`) |
| "begins with four solver rollouts and expands to eight when the mean is within 0.15 of a band boundary" | `calibrate::calibrate` (`near_boundary`, `expanded_rollouts`) |
| "Tasks below the band receive one diagnostic rollout with reference-solution hints" | `calibrate::run_diagnostic` + `policy::Hint` |
| "Each assignment permits up to two revisions" | `calibrate::RevisionLedger` + the revision loop in `flywheel::generation_phase` |
| "Environments that fail validation or calibration go back to the proposer for revision" | `flywheel::Flywheel::revise_task` |

### §3.3 — Training the proposer and solver

| paper element | artifact |
|---|---|
| Eq. 1: r_S = fraction of tests passed | `world::CtrfReport::reward`, `reward::solver_reward` |
| Eq. 2: r_P = −1 / −0.25 / +1 | `reward::proposer_reward` |
| Eq. 3: Â = r − group mean (per role) | `advantage::group_advantage` (mean-centered, no std division) |
| "Proposer rewards are centered within proposer groups, solver rewards ... across attempts on the same environment" | `advantage::role_batch` over assignment-keyed vs. env-keyed groups |
| "Following DAPO, we drop groups whose rewards do not vary" | `advantage::dapo_filter` + the flywheel's sampler |
| "We oversample and discard groups with no reward variation, collecting additional groups until the batch is filled" | `advantage::fill_batch` / the in-round sampler with `making_progress` |
| DPPO with TV threshold 0.1, KL/entropy 0/0, LR 1e-6 | `dppo::DppoConfig::paper`, `dppo::DppoOptimizer` (clipped surrogate + exact line-search TV projection) |
| "Unfinished terminal trajectories are carried across updates, preserving ... live sandbox state. They resume with the updated policy, while stored sampling log-probabilities are retained" | `flywheel::CarriedAttempt` (steps + `WorldState` + `next_check`), `flywheel::PendingGroup`, `dppo::CarryOverBuffer`; agent timeout → check budget mapping |
| "At each refresh, the host reruns the updated solver on the pool, removes environments that fall outside the calibration band, and asks the proposer for replacements" | `pool::TrainingPool::review` + `flywheel::run_round` step 1 |
| "The validation and calibration checks stay fixed" | harness optimization (`harness.rs`) can never touch them; they are host-side |
| "the proposer updates its own memory, skills, and tools between rounds ... merged several validation steps into a faster script and saved useful GitHub tasks and Docker Hub images as a web-cache skill" | `harness::HarnessOptimizer` (`MergeValidationSteps`, `CacheGithubTask`, `CacheDockerImage`, `AddLesson`) |
| Table 12 settings | `flywheel::FlywheelConfig::paper` |

### §4.1 — Does environment design benefit from a terminal?

Valid-environment rate and cost per valid environment: `metrics::RunMetrics`
(`valid_env_rate`, `valid_env_rate_ci`, `cost_per_valid_env`). Figure 5's
richness metrics (files per task, solver turns, pass rate):
`harbor::HarborTask::total_checks` (data files ≈ checks),
`trajectory::Trajectory::turn_count`, `world` rewards.

### §4.2 — Model-designed curricula outperform static pools

The matched comparison (static Tmax pool vs. flywheel) is the seed-pool vs.
generated-pool split in `Flywheel::new` / `run_round`: `flywheel::tests`
assert the pool never drains and the curriculum issues `harden`
assignments as the solver improves.

### §4.3 — Recursive self-improvement

Cold-Start: `coldstart::select_cold_start` (600 admitted, deduplicated
trajectories from Claude Opus 5 / DeepSeek-V4-Flash / Kimi-K3). Paired
controls: `coldstart::PairedEvaluation::sign_test_p`,
`coldstart::SolvingPairedScores::bootstrap_interval` (Table 13's protocol).
Useful-group rate and collection time (Figure 7): `metrics::RunMetrics`.

### §4.5 — Learning to ask for help

`hil`: the Figure-9 billing/ledger instruction, Table-6's five blockers
(`amount_agreement_tolerance` Missing parameters, `invoice_status_scope`
Business information, `duplicate_invoice_row_precedence` Ambiguous
requirements, `discrepancy_report_ordering` Question,
`total_variance_definition` Schema), the `ask_human` tool
(`BlockerRegistry::ask`), Ask-F1 (`hil::ask_f1`), the shaped clarification
reward (`reward::clarification_reward`), and the derived full-info /
no-tool variants ("without further model calls").

### Appendix C — Cross-domain generation

`domains::Domain::all()` — the seven domains of Table 7 with their
environment interfaces and verification styles, Table 8's acceptance data
as reference constants, and the admission protocol including "Resetting
must reproduce the initial state and a passing reference execution"
(`domains::cross_domain_admission`, `world::TaskWorld::reset`).

### Appendix E — The proposer skills

`skills::CREATE_TASK_SKILL` (Harbor's create-task, verbatim: nine steps,
multi-step layout, network policy, pitfalls) and
`skills::AUTO_ENV_SCALING_SKILL` (the overlay, verbatim: reward spread as
the primary objective, grounding reads of `index.jsonl` /
`past_environments`, the managed verifier template, oracle-passes ∧
no-op-fails, canonical output layout) plus
`skills::MANAGED_VERIFIER_TEMPLATE` (`/opt/verifier_template.sh` semantics:
CTRF fraction → `/logs/verifier/reward.txt`).

## Modes of use

| mode | what runs | example |
|---|---|---|
| Library | embed the flywheel, the admission checks, or single components | `Flywheel`, `validate::validate`, `calibrate::calibrate` |
| End-to-end demo | 6 rounds + HiL + Cold-Start + cross-domain + metrics | `cargo run -p dsec-autoenv --example flywheel` |
| Paper configuration | Table 12 settings verbatim | `FlywheelConfig::paper()` |

## Test surface

* **155 unit tests** across all 24 modules, including the Eq. 1/2/3 values,
  the band/σ/expansion/revision numbers, the 13-gram window semantics, the
  TV trust region, the workspace region guards, the sandbox egress and
  web-gating rules, and the Table-8 reference data.
* **11 integration tests** (`tests/flywheel.rs`): the golden event order of
  a round, multi-round stability with curriculum tracking, carry-over
  resume-and-complete, contaminated and gameable environments rejected
  through the correct arms, sandbox-contract violations, calibration
  boundary behavior, DPPO trust-region endurance, HiL Ask-F1, Cold-Start
  selection, cross-domain admission, and harness evolution.

## What is deliberately out of scope

Real container builds, real LLM rollouts, and real GPU training. The crate
replaces them with deterministic equivalents that preserve every observable
contract the flywheel depends on (rewards, verdicts, orders, budgets,
regions), so the algorithms — not the infrastructure — are what gets
exercised. Live backends plug in through `policy::SolverModel` and the
proposer policy surface.
