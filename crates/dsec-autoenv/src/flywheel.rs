//! The flywheel orchestrator: the complete AutoEnvScaling loop (Section 3).
//!
//! One round:
//!
//! 1. **Pool review** (every `review_interval` solver updates, Table 12):
//!    "the host reruns the updated solver on the pool, removes environments
//!    that fall outside the calibration band, and asks the proposer for
//!    replacements based on the latest rollouts."
//! 2. **Workspace refresh**: rollouts and past environments flow into the
//!    proposer's workspace; memory and tools persist.
//! 3. **Assignments**: parent environment, pass rate, most common failure,
//!    simplify/harden direction.
//! 4. **Proposer episodes** (a group of trajectories per assignment, Table
//!    12: 128): read workspace → design → build → validate (oracle ∧
//!    no-op) → optionally solver_model → submit.
//! 5. **Admission**: host validation (Eq. 2's `-1` arm), then calibration
//!    (the `-0.25` / `+1` arms); up to two revisions per assignment.
//! 6. **Solver updates** (every step): 16 environments × 16 trajectories,
//!    oversampled and DAPO-filtered, Eq. 1 rewards, Eq. 3 advantages, DPPO
//!    under the TV trust region. Unfinished trajectories carry across
//!    updates with their stored log-probabilities and live sandbox state.
//! 7. **Proposer training** (every k = 16 solver updates): Eq. 2 rewards,
//!    Eq. 3 advantages within the assignment group, DPPO.
//! 8. **Harness optimization** between rounds (memory / skills / tools
//!    evolve; the checks stay fixed).
//!
//! Every event lands in the round report; running metrics (valid-env rate
//! with Wilson intervals, cost per valid environment, useful-group rate,
//! collection time) accumulate across rounds.

use crate::advantage::{group_advantage, useful_group_rate};
use crate::assignment::{build_assignment, Assignment, Direction, ParentRef};
use crate::calibrate::{calibrate, CalibrationConfig, CalibrationOutcome, RevisionLedger};
use crate::decontaminate::ContaminationIndex;
use crate::dppo::{CarryOverBuffer, CarryTrajectory, DppoConfig, DppoOptimizer, TokenStep};
use crate::harbor::HarborTask;
use crate::harness::{HarnessAction, HarnessOptimizer, HarnessRoundLog};
use crate::metrics::RunMetrics;
use crate::policy::{generate_task, ScriptedProposer, ScriptedSolver, SolverModel, TaskGenParams};
use crate::pool::{Eviction, TrainingPool};
use crate::reward::proposer_reward;
use crate::trajectory::{Role, Trajectory, TrajectoryGroup};
use crate::validate::{validate, ValidationConfig, ValidationOutcome};
use crate::workspace::Workspace;
use crate::world::ImageRegistry;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Flywheel configuration. [`FlywheelConfig::paper`] is Table 12;
/// [`FlywheelConfig::demo`] is a small deterministic instantiation for
/// examples and tests (same protocol, smaller groups).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlywheelConfig {
    /// Solver rollout batch: environments per update (Table 12: 16).
    pub solver_batch_envs: usize,
    /// Solver rollout batch: trajectories per environment (Table 12: 16).
    pub solver_group_trajs: usize,
    /// Proposer trajectories per generation assignment (Table 12: 128).
    pub proposer_group: usize,
    /// Proposer generation / training interval k (Table 12: 16 solver
    /// updates).
    pub proposer_interval: usize,
    /// Pool review interval (Table 12: 16 solver updates).
    pub review_interval: usize,
    /// Active environment pool capacity (Table 12: 256).
    pub pool_capacity: usize,
    /// Solver updates executed per flywheel round.
    pub updates_per_round: usize,
    /// Assignments issued per round (replacement demand; the paper fills
    /// the pool).
    pub assignments_per_round: usize,
    /// Calibration protocol.
    pub calibration: CalibrationConfig,
    /// Validation thresholds.
    pub validation: ValidationConfig,
    /// DPPO hyper-parameters.
    pub dppo: DppoConfig,
    /// Virtual seconds one verifier check takes (drives the agent-timeout
    /// carry-over and the collection-time metric).
    pub check_time_sec: f64,
    /// Solver seed.
    pub solver_seed: u64,
    /// Proposer seed.
    pub proposer_seed: u64,
    /// Virtual cost of one proposer episode (for the cost-per-valid-env
    /// metric; the paper reports USD per valid environment).
    pub cost_per_episode: f64,
}

impl Default for FlywheelConfig {
    fn default() -> Self {
        FlywheelConfig::demo()
    }
}

impl FlywheelConfig {
    /// The paper's Table 12 settings.
    pub fn paper() -> Self {
        FlywheelConfig {
            solver_batch_envs: 16,
            solver_group_trajs: 16,
            proposer_group: 128,
            proposer_interval: 16,
            review_interval: 16,
            pool_capacity: 256,
            updates_per_round: 16,
            assignments_per_round: 16,
            calibration: CalibrationConfig::default(),
            validation: ValidationConfig::default(),
            dppo: DppoConfig::paper(),
            check_time_sec: 30.0,
            solver_seed: 0,
            proposer_seed: 0,
            cost_per_episode: 1.0,
        }
    }

    /// A small deterministic instantiation (same protocol, smaller groups)
    /// for examples and tests.
    pub fn demo() -> Self {
        FlywheelConfig {
            solver_batch_envs: 4,
            solver_group_trajs: 4,
            proposer_group: 4,
            proposer_interval: 4,
            review_interval: 4,
            pool_capacity: 12,
            updates_per_round: 4,
            assignments_per_round: 3,
            calibration: CalibrationConfig::default(),
            validation: ValidationConfig::default(),
            dppo: DppoConfig {
                lr: 1.0,
                tv_threshold: 0.1,
                clip_eps: 0.2,
                kl_coef: 0.0,
                entropy_coef: 0.0,
            },
            check_time_sec: 30.0,
            solver_seed: 42,
            proposer_seed: 7,
            cost_per_episode: 1.0,
        }
    }
}

/// One admission decision for a submitted environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Admission {
    /// The environment id.
    pub env_id: String,
    /// Host validation outcome (`None` if the episode never submitted).
    pub validation: Option<ValidationOutcome>,
    /// Host calibration outcome (`None` if validation failed).
    pub calibration: Option<CalibrationOutcome>,
    /// Eq. 2 proposer reward.
    pub proposer_reward: f64,
    /// Whether the environment entered the pool.
    pub admitted: bool,
    /// Revisions consumed.
    pub revisions: usize,
}

/// Summary of one proposer episode within a round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeSummary {
    /// Assignment id.
    pub assignment_id: String,
    /// Environment id targeted.
    pub env_id: String,
    /// Direction tag.
    pub direction: String,
    /// Whether the proposer submitted a task.
    pub submitted: bool,
    /// Turns the episode took (Figure 22's episode depth).
    pub turns: usize,
    /// The admission decision.
    pub admission: Option<Admission>,
}

/// One carried (unfinished) solver attempt: preserved interaction history
/// and live sandbox state (Appendix G.1).
#[derive(Debug, Clone)]
pub struct CarriedAttempt {
    /// The environment the attempt belongs to.
    pub env_id: String,
    /// Tokens generated so far, with their stored sampling
    /// log-probabilities.
    pub steps: Vec<TokenStep>,
    /// The live sandbox state (files produced so far).
    pub world: crate::world::WorldState,
    /// The next un-attempted check.
    pub next_check: usize,
    /// Turns taken so far.
    pub turns: Vec<crate::trajectory::Turn>,
}

/// One collected solver group: the trajectories plus their token steps,
/// aligned by index (the DPPO wiring).
#[derive(Debug, Clone)]
pub struct CollectedGroup {
    /// The group.
    pub group: TrajectoryGroup,
    /// Per-trajectory token steps.
    pub steps: Vec<Vec<TokenStep>>,
}

/// Completed trajectories accumulating toward a full group. Long-horizon
/// environments complete one trajectory every few updates; the group
/// trains once it has `group_trajs` of them.
#[derive(Debug, Clone, Default)]
pub struct PendingGroup {
    /// Completed trajectories so far.
    pub trajectories: Vec<Trajectory>,
    /// Their token steps, aligned.
    pub steps: Vec<Vec<TokenStep>>,
}

/// The report of one flywheel round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundReport {
    /// Round index.
    pub round: u64,
    /// The solver-update index the round ended at.
    pub end_update: u64,
    /// Assignments issued.
    pub assignments: Vec<String>,
    /// Episode summaries.
    pub episodes: Vec<EpisodeSummary>,
    /// Environments admitted.
    pub admitted: usize,
    /// Environments rejected (after revisions).
    pub rejected: usize,
    /// Pool review evictions this round.
    pub evicted: Vec<Eviction>,
    /// Solver updates executed.
    pub solver_updates: usize,
    /// Proposer updates executed.
    pub proposer_updates: usize,
    /// Pool size at round end.
    pub pool_size: usize,
    /// Groups retained this round (Figure 7).
    pub groups_retained: usize,
    /// Groups collected this round.
    pub groups_collected: usize,
    /// Virtual minutes of rollout collection this round (Figure 7, right).
    pub collection_minutes: f64,
    /// Mean solver reward across the round's rollouts.
    pub mean_solver_reward: f64,
    /// Proposer rewards (Eq. 2) this round.
    pub proposer_rewards: Vec<f64>,
    /// Harness actions applied between rounds.
    pub harness_actions: Vec<HarnessAction>,
    /// Carried (unfinished) attempts pending resume.
    pub carried_attempts: usize,
}

impl RoundReport {
    /// The useful-group rate of this round (Figure 7, left).
    pub fn useful_group_rate(&self) -> f64 {
        useful_group_rate(self.groups_retained, self.groups_collected)
    }
}

/// Flywheel events, in order, for golden-flow tests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlywheelEvent {
    /// Pool review ran.
    PoolReview {
        /// Update index.
        update: u64,
        /// Environments evicted.
        evicted: usize,
    },
    /// Workspace refreshed with rollouts and environments.
    WorkspaceRefreshed {
        /// Rollout records installed.
        rollouts: usize,
        /// Past environments installed.
        environments: usize,
    },
    /// An assignment was issued.
    Assignment {
        /// Assignment id.
        assignment_id: String,
        /// Parent environment, when grounded.
        parent: Option<String>,
        /// Direction tag.
        direction: String,
    },
    /// A proposer episode submitted (or gave up).
    EpisodeSubmitted {
        /// Assignment id.
        assignment_id: String,
        /// Environment id.
        env_id: String,
        /// Whether a task was submitted.
        submitted: bool,
    },
    /// Host validation finished.
    Validated {
        /// Environment id.
        env_id: String,
        /// Whether validation passed.
        passed: bool,
    },
    /// Host calibration finished.
    Calibrated {
        /// Environment id.
        env_id: String,
        /// Verdict name.
        verdict: String,
        /// Mean reward.
        mean: f64,
        /// Reward std.
        std: f64,
    },
    /// An environment entered the pool.
    Admitted {
        /// Environment id.
        env_id: String,
    },
    /// An environment was rejected.
    Rejected {
        /// Environment id.
        env_id: String,
        /// Eq. 2 reward.
        reward: f64,
    },
    /// A solver DPPO update landed.
    SolverUpdate {
        /// Update index.
        update: u64,
        /// Groups trained on.
        groups: usize,
        /// Mean reward of the batch.
        mean_reward: f64,
    },
    /// A proposer DPPO update landed.
    ProposerUpdate {
        /// Update index.
        update: u64,
        /// Trajectories trained on.
        trajectories: usize,
    },
    /// Harness optimization ran.
    HarnessOptimized {
        /// Actions applied.
        actions: usize,
    },
    /// Unfinished solver attempts were carried into the next update.
    Carried {
        /// Pending attempts.
        attempts: usize,
    },
}

/// The flywheel state machine.
pub struct Flywheel {
    /// Configuration.
    pub config: FlywheelConfig,
    /// The training pool.
    pub pool: TrainingPool,
    /// The solver (tabular policy over skills).
    pub solver: ScriptedSolver,
    /// The proposer (tabular policy over design moves).
    pub proposer: ScriptedProposer,
    /// The harness optimizer.
    pub optimizer: HarnessOptimizer,
    /// The proposer's workspace.
    pub workspace: Workspace,
    /// The build registry.
    pub registry: ImageRegistry,
    /// The decontamination index over held-out benchmarks.
    pub heldout: ContaminationIndex,
    /// DPPO optimizer.
    pub dppo: DppoOptimizer,
    /// Carry-over buffer (the token-level view of unfinished episodes).
    pub carry: CarryOverBuffer,
    /// Carried unfinished solver attempts (with live world state).
    pub unfinished: Vec<CarriedAttempt>,
    /// Completed trajectories waiting for their group to fill.
    pub pending: BTreeMap<String, PendingGroup>,
    /// Monotonic solver-update counter.
    pub update_counter: u64,
    /// Round counter.
    pub round: u64,
    /// Proposer token steps accumulated since the last proposer update.
    pub proposer_steps: Vec<TokenStep>,
    /// Proposer rewards accumulated since the last proposer update.
    pub proposer_rewards_pending: Vec<f64>,
    /// The event stream of the most recent round.
    pub last_events: Vec<FlywheelEvent>,
    /// Running metrics across rounds.
    pub metrics: RunMetrics,
    /// Total virtual seconds spent collecting rollouts.
    collection_sec: f64,
}

impl Flywheel {
    /// Build a flywheel with a seed pool (the Tmax-style static pool the
    /// paper's comparisons start from).
    pub fn new(config: FlywheelConfig, seed_pool: Vec<HarborTask>) -> Self {
        let mut pool = TrainingPool::new(config.pool_capacity);
        for task in seed_pool {
            let _ = pool.admit(task, 0);
        }
        let solver = ScriptedSolver::new("solver", config.solver_seed);
        let proposer = ScriptedProposer::new("proposer", config.proposer_seed);
        let dppo = DppoOptimizer::new(config.dppo.clone());
        Flywheel {
            config,
            pool,
            solver,
            proposer,
            optimizer: HarnessOptimizer::default(),
            workspace: Workspace::new(),
            registry: ImageRegistry::default_world(),
            heldout: ContaminationIndex::build(&[]),
            dppo,
            carry: CarryOverBuffer::default(),
            unfinished: Vec::new(),
            pending: BTreeMap::new(),
            update_counter: 0,
            round: 0,
            proposer_steps: Vec::new(),
            proposer_rewards_pending: Vec::new(),
            last_events: Vec::new(),
            metrics: RunMetrics::default(),
            collection_sec: 0.0,
        }
    }

    /// Set the held-out benchmark index for decontamination.
    pub fn with_heldout(mut self, heldout: ContaminationIndex) -> Self {
        self.heldout = heldout;
        self
    }

    /// Run one full round; returns the round report.
    pub fn run_round(&mut self) -> RoundReport {
        let mut events: Vec<FlywheelEvent> = Vec::new();
        let round = self.round;
        self.round += 1;

        // --- 1. Pool review (every review_interval updates).
        let mut evicted = Vec::new();
        if self.update_counter > 0
            && self.update_counter % (self.config.review_interval as u64) == 0
        {
            let report = self.pool.review(
                &mut self.solver,
                &self.config.calibration,
                self.update_counter,
            );
            events.push(FlywheelEvent::PoolReview {
                update: report.update,
                evicted: report.evicted.len(),
            });
            evicted = report.evicted;
        }

        // --- 2. Workspace refresh: rollouts + past environments in;
        //     memory and tools persist.
        let records = self
            .pool
            .rollout_records(round, self.solver.solver_id.as_str());
        let envs: Vec<HarborTask> = self.pool.tasks().into_iter().cloned().collect();
        let env_count = envs.len();
        let env_refs: Vec<&HarborTask> = envs.iter().collect();
        let attempts = self.solver_attempts_snapshot(&envs);
        self.workspace
            .refresh_from_pool(records.clone(), attempts, &env_refs);
        events.push(FlywheelEvent::WorkspaceRefreshed {
            rollouts: records.len(),
            environments: env_count,
        });

        // --- 3. Assignments: replacements for evictions + pool growth.
        let assignments = self.build_assignments(&evicted, &records);
        for a in &assignments {
            events.push(FlywheelEvent::Assignment {
                assignment_id: a.assignment_id.clone(),
                parent: a.parent.as_ref().map(|p| p.env_id.clone()),
                direction: a.direction.tag().to_string(),
            });
        }

        // --- 4+5. Proposer episodes + host admission (with revisions).
        let (episode_summaries, admissions, proposer_rewards) =
            self.generation_phase(&assignments, &mut events);

        // --- 6+7. Solver updates (every step) + proposer updates (every k).
        let (
            solver_updates,
            groups_retained,
            groups_collected,
            mean_solver_reward,
            proposer_updates,
        ) = self.training_phase(&mut events);

        // --- 8. Harness optimization between rounds.
        let admitted_count = admissions.iter().filter(|a| a.admitted).count();
        let rejected_count = admissions.iter().filter(|a| !a.admitted).count();
        let log = HarnessRoundLog {
            round,
            episodes: episode_summaries.len(),
            admitted: admitted_count,
            rejected: rejected_count,
            proposer_rewards: proposer_rewards.clone(),
            common_failures: self.collect_common_failures(&admissions),
            used_web: true, // the demo proposer's preamble lists web_search
        };
        let harness_actions = self.optimizer.optimize(&mut self.workspace, &log);
        events.push(FlywheelEvent::HarnessOptimized {
            actions: harness_actions.len(),
        });

        // --- Metrics. Every group member is one proposer episode.
        let episodes_total = admissions.len();
        self.metrics
            .record_round(crate::metrics::RoundMetricsInput {
                episodes: episodes_total,
                admitted: admitted_count,
                cost: episodes_total as f64 * self.config.cost_per_episode,
                groups_retained,
                groups_collected,
                collection_minutes: self.collection_sec / 60.0,
                solver_updates,
            });

        let report = RoundReport {
            round,
            end_update: self.update_counter,
            assignments: assignments
                .iter()
                .map(|a| a.assignment_id.clone())
                .collect(),
            episodes: episode_summaries,
            admitted: admitted_count,
            rejected: rejected_count,
            evicted,
            solver_updates,
            proposer_updates,
            pool_size: self.pool.len(),
            groups_retained,
            groups_collected,
            collection_minutes: self.collection_sec / 60.0,
            mean_solver_reward,
            proposer_rewards,
            harness_actions,
            carried_attempts: self.unfinished.len(),
        };
        self.last_events = events;
        report
    }

    /// Build this round's assignments: one per eviction (replacement, with
    /// the evicted env as parent), then growth up to
    /// `assignments_per_round`.
    fn build_assignments(
        &self,
        evicted: &[Eviction],
        records: &[crate::trajectory::RolloutRecord],
    ) -> Vec<Assignment> {
        let mut assignments = Vec::new();
        let mut seq = self.round * 100;
        for ev in evicted {
            let parent_ref = ParentRef {
                env_id: ev.env_id.clone(),
                checks: ev.task.total_checks(),
                skills: ev.task.skills.clone(),
            };
            // Ground the replacement in the review's own attempts on the
            // evicted env ("asks the proposer for replacements based on
            // the latest rollouts"): the eviction carries the final pass
            // rate and std.
            let mut grounding = records.to_vec();
            grounding.push(crate::trajectory::RolloutRecord {
                env_id: ev.env_id.clone(),
                solver: self.solver.solver_id.clone(),
                round: self.round.saturating_sub(1),
                provenance: crate::trajectory::Provenance::SelfRound(self.round.saturating_sub(1)),
                rewards: vec![ev.pass_rate; 4],
                reward_std: ev.reward_std,
                check_failures: BTreeMap::new(),
                path: format!("memory/past_trajectories/failed/{}.json", ev.env_id),
            });
            assignments.push(build_assignment(
                format!("asg_{seq:04}"),
                format!("env_{seq:04}"),
                parent_ref,
                &grounding,
                self.config.calibration.band,
            ));
            seq += 1;
        }
        // Growth assignments: fresh environments grounded in the solver's
        // weakest environments.
        let shortfall = self
            .config
            .assignments_per_round
            .saturating_sub(assignments.len());
        for _ in 0..shortfall {
            let parent = records
                .iter()
                .min_by(|a, b| {
                    a.reward_mean()
                        .partial_cmp(&b.reward_mean())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .and_then(|r| self.pool.task(&r.env_id))
                .map(|t| ParentRef {
                    env_id: t.env_id.clone(),
                    checks: t.total_checks(),
                    skills: t.skills.clone(),
                })
                .or_else(|| {
                    self.pool.tasks().first().map(|t| ParentRef {
                        env_id: t.env_id.clone(),
                        checks: t.total_checks(),
                        skills: t.skills.clone(),
                    })
                });
            let assignment = match parent {
                Some(p) => build_assignment(
                    format!("asg_{seq:04}"),
                    format!("env_{seq:04}"),
                    p,
                    records,
                    self.config.calibration.band,
                ),
                None => Assignment {
                    assignment_id: format!("asg_{seq:04}"),
                    target_env_id: format!("env_{seq:04}"),
                    parent: None,
                    parent_pass_rate: None,
                    most_common_failure: None,
                    direction: Direction::Fresh,
                },
            };
            assignments.push(assignment);
            seq += 1;
        }
        assignments
    }

    /// The generation phase: proposer episode groups, host admission, and
    /// the revision loop.
    fn generation_phase(
        &mut self,
        assignments: &[Assignment],
        events: &mut Vec<FlywheelEvent>,
    ) -> (Vec<EpisodeSummary>, Vec<Admission>, Vec<f64>) {
        let mut summaries = Vec::new();
        let mut admissions = Vec::new();
        let mut all_rewards = Vec::new();
        let group_size = self.config.proposer_group.max(1);

        for assignment in assignments {
            let parent_task = assignment
                .parent
                .as_ref()
                .and_then(|p| self.pool.task(&p.env_id).cloned());
            let mut ledger = RevisionLedger::new(&self.config.calibration);

            // The proposer group: `group_size` trajectories sharing this
            // generation assignment (Table 12: 128).
            let mut episode_steps: Vec<TokenStep> = Vec::new();
            let mut episode_rewards: Vec<f64> = Vec::new();
            let mut last_summary: Option<EpisodeSummary> = None;

            for member in 0..group_size {
                let mut assignment = assignment.clone();
                if member > 0 {
                    assignment.assignment_id = format!("{}#{}", assignment.assignment_id, member);
                }
                self.workspace
                    .set_phase(crate::workspace::Phase::DuringEpisode);
                self.workspace.set_assignment(&assignment.render());
                let mut sandbox = crate::sandbox::Sandbox::new(
                    crate::sandbox::SandboxResources::proposer(crate::sandbox::CodingAgent::Pi),
                    vec!["bash".into(), "web_search".into()],
                );

                let mut ctx = crate::policy::EpisodeContext {
                    workspace: &mut self.workspace,
                    sandbox: &mut sandbox,
                    registry: &self.registry,
                    max_revisions: self.config.calibration.max_revisions as u32,
                    use_solver_model: &mut self.solver,
                };
                let episode = self
                    .proposer
                    .propose(&assignment, parent_task.as_ref(), &mut ctx);
                // Sandbox violations surface as failed episodes (the
                // proposer submitted nothing).
                let episode = match episode {
                    Ok(ep) => ep,
                    Err(_) => {
                        episode_rewards.push(-1.0);
                        admissions.push(Admission {
                            env_id: assignment.target_env_id.clone(),
                            validation: None,
                            calibration: None,
                            proposer_reward: -1.0,
                            admitted: false,
                            revisions: 0,
                        });
                        last_summary = Some(EpisodeSummary {
                            assignment_id: assignment.assignment_id.clone(),
                            env_id: assignment.target_env_id.clone(),
                            direction: assignment.direction.tag().to_string(),
                            submitted: false,
                            turns: 0,
                            admission: None,
                        });
                        continue;
                    }
                };
                let turns = episode.turns.len();
                events.push(FlywheelEvent::EpisodeSubmitted {
                    assignment_id: assignment.assignment_id.clone(),
                    env_id: assignment.target_env_id.clone(),
                    submitted: episode.submitted,
                });
                if let Some(step) = episode.steps.first() {
                    episode_steps.push(step.clone());
                }

                // --- Host admission, outside the sandbox ---
                let mut current_task = episode.task.clone();
                let mut validation = if episode.submitted {
                    Some(validate(
                        &current_task,
                        &self.registry,
                        &self.heldout,
                        &self.config.validation,
                    ))
                } else {
                    None
                };
                let mut calibration: Option<CalibrationOutcome> = None;
                let mut admitted = false;
                let mut proposer_r = -1.0;
                let mut revisions_used: usize = episode.revisions_used as usize;

                if episode.submitted {
                    let (ok, r) =
                        self.admit_task(&current_task, &mut validation, &mut calibration, events);
                    admitted = ok;
                    proposer_r = r;

                    // Revision loop: "each assignment permits up to two
                    // revisions. Revised environments repeat the admission
                    // checks."
                    let mut guard = 0;
                    while !admitted && ledger.can_revise() && guard < 3 {
                        let _ = ledger.request();
                        revisions_used += 1;
                        guard += 1;
                        current_task = self.revise_task(&current_task, &validation, &calibration);
                        let (ok, r) = self.admit_task(
                            &current_task,
                            &mut validation,
                            &mut calibration,
                            events,
                        );
                        admitted = ok;
                        proposer_r = r;
                    }

                    if admitted {
                        let _ = self.pool.admit(current_task.clone(), self.update_counter);
                        events.push(FlywheelEvent::Admitted {
                            env_id: current_task.env_id.clone(),
                        });
                    } else {
                        events.push(FlywheelEvent::Rejected {
                            env_id: current_task.env_id.clone(),
                            reward: proposer_r,
                        });
                    }
                }

                episode_rewards.push(proposer_r);
                let admission = Admission {
                    env_id: current_task.env_id.clone(),
                    validation: validation.clone(),
                    calibration: calibration.clone(),
                    proposer_reward: proposer_r,
                    admitted,
                    revisions: revisions_used,
                };
                last_summary = Some(EpisodeSummary {
                    assignment_id: assignment.assignment_id.clone(),
                    env_id: assignment.target_env_id.clone(),
                    direction: assignment.direction.tag().to_string(),
                    submitted: episode.submitted,
                    turns,
                    admission: Some(admission.clone()),
                });
                admissions.push(admission);
            }

            // Eq. 3 within the assignment group: advantages attach to the
            // design-move steps.
            if !episode_steps.is_empty() {
                let adv = group_advantage(&episode_rewards);
                for (step, a) in episode_steps.iter_mut().zip(adv) {
                    step.advantage = a;
                }
                self.proposer_steps.extend(episode_steps);
                self.proposer_rewards_pending
                    .extend(episode_rewards.iter().cloned());
                all_rewards.extend(episode_rewards.iter().cloned());
            }
            if let Some(summary) = last_summary {
                summaries.push(summary);
            }
        }
        (summaries, admissions, all_rewards)
    }

    /// Run the host admission checks (validation then calibration) on one
    /// submitted task, recording the events.
    fn admit_task(
        &mut self,
        task: &HarborTask,
        validation: &mut Option<ValidationOutcome>,
        calibration: &mut Option<CalibrationOutcome>,
        events: &mut Vec<FlywheelEvent>,
    ) -> (bool, f64) {
        let v = validate(task, &self.registry, &self.heldout, &self.config.validation);
        events.push(FlywheelEvent::Validated {
            env_id: task.env_id.clone(),
            passed: v.passed,
        });
        *validation = Some(v);
        if !validation.as_ref().expect("just set").passed {
            return (false, proposer_reward(false, false));
        }
        let out = calibrate(task, &mut self.solver, &self.config.calibration);
        events.push(FlywheelEvent::Calibrated {
            env_id: task.env_id.clone(),
            verdict: format!("{:?}", out.verdict),
            mean: out.mean,
            std: out.std,
        });
        let r = proposer_reward(true, out.passed());
        let passed = out.passed();
        *calibration = Some(out);
        (passed, r)
    }

    /// A revision: ease the task toward the calibration band (fewer checks
    /// when too hard, more when too easy), regenerating from the same env
    /// id and skill set with a repaired reference solution.
    fn revise_task(
        &self,
        task: &HarborTask,
        validation: &Option<ValidationOutcome>,
        calibration: &Option<CalibrationOutcome>,
    ) -> HarborTask {
        let too_hard = calibration
            .as_ref()
            .map(|c| c.mean < self.config.calibration.band.0)
            .unwrap_or(false);
        let too_easy = calibration
            .as_ref()
            .map(|c| c.mean > self.config.calibration.band.1)
            .unwrap_or(false);
        let broken_reference = validation
            .as_ref()
            .map(|v| (v.reference_reward - 1.0).abs() > 1e-9)
            .unwrap_or(true);
        let n = task.total_checks();
        let new_n = if broken_reference {
            n
        } else if too_hard {
            n.saturating_sub(1).max(2)
        } else if too_easy {
            n + 1
        } else {
            n
        };
        generate_task(&TaskGenParams {
            env_id: task.env_id.clone(),
            name: task.meta.name.clone(),
            n_checks: new_n,
            skills: task.skills.clone(),
            difficulty: task.toml.metadata.difficulty,
            broken: false,
        })
    }

    /// The training phase: solver updates every step, proposer updates
    /// every k steps, carry-over of unfinished attempts.
    #[allow(clippy::type_complexity)]
    fn training_phase(
        &mut self,
        events: &mut Vec<FlywheelEvent>,
    ) -> (usize, usize, usize, f64, usize) {
        let mut solver_updates = 0usize;
        let mut proposer_updates = 0usize;
        let mut retained_total = 0usize;
        let mut collected_total = 0usize;
        let mut reward_sum = 0.0;
        let mut reward_count = 0.0f64;
        let batch_offset = (self.round as usize) * self.config.solver_batch_envs;

        for _ in 0..self.config.updates_per_round {
            // Rollout collection: oversample and discard groups with no
            // reward variation until the batch is filled (Appendix G.1).
            let env_ids = self
                .pool
                .sample_solver_batch(self.config.solver_batch_envs, batch_offset);
            let need = env_ids.len();
            let (collected, discarded) = {
                let solver = &mut self.solver;
                let pool = &self.pool;
                let unfinished = &mut self.unfinished;
                let collection_sec = &mut self.collection_sec;
                let config = &self.config;
                let pending = &mut self.pending;
                let mut rounds = 0usize;
                let mut collected: Vec<CollectedGroup> = Vec::new();
                let mut discarded = 0usize;
                // Oversample and discard groups with no reward variation,
                // collecting additional groups until the batch is filled
                // (Appendix G.1) — bounded to 3 sampler rounds, and never
                // breaking early while carried attempts keep making
                // progress.
                while collected.len() < need && rounds < 3 {
                    rounds += 1;
                    let batch: Vec<CollectedGroup> = {
                        let mut batch = Vec::new();
                        for env_id in &env_ids {
                            if let Some(cg) = collect_group(
                                env_id,
                                pool,
                                solver,
                                unfinished,
                                pending,
                                config,
                                collection_sec,
                            ) {
                                batch.push(cg);
                            }
                        }
                        batch
                    };
                    let making_progress = !unfinished.is_empty() || !pending.is_empty();
                    if batch.is_empty() && !making_progress {
                        break;
                    }
                    for cg in batch {
                        if cg.group.rewards_vary() {
                            collected.push(cg);
                        } else {
                            discarded += 1;
                        }
                    }
                }
                (collected, discarded)
            };
            // Collected groups (before DAPO filtering) for the metrics.
            let collected_count = collected.len() + discarded;
            collected_total += collected_count;
            retained_total += collected.len();

            // Eq. 3 advantages per environment group; DPPO update over all
            // steps of all retained groups.
            let mut steps: Vec<TokenStep> = Vec::new();
            let mut batch_rewards = Vec::new();
            for cg in &collected {
                let advs = group_advantage(&cg.group.rewards);
                for (traj, adv, traj_steps) in cg
                    .group
                    .trajectories
                    .iter()
                    .zip(&advs)
                    .zip(&cg.steps)
                    .map(|((t, a), s)| (t, a, s))
                {
                    let r = traj.reward().unwrap_or(0.0);
                    batch_rewards.push(r);
                    reward_sum += r;
                    reward_count += 1.0;
                    for s in traj_steps {
                        let mut step = s.clone();
                        step.advantage = *adv;
                        steps.push(step);
                    }
                }
            }
            let _ = self.dppo.update_bank(&mut self.solver.bank, &steps);
            self.update_counter += 1;
            solver_updates += 1;
            let mean_r = if batch_rewards.is_empty() {
                0.0
            } else {
                batch_rewards.iter().sum::<f64>() / batch_rewards.len() as f64
            };
            events.push(FlywheelEvent::SolverUpdate {
                update: self.update_counter,
                groups: collected.len(),
                mean_reward: mean_r,
            });

            // Proposer training every k solver updates.
            if self.update_counter % (self.config.proposer_interval as u64) == 0
                && !self.proposer_steps.is_empty()
            {
                let adv = group_advantage(&self.proposer_rewards_pending);
                let mut steps = self.proposer_steps.clone();
                for (s, a) in steps.iter_mut().zip(adv) {
                    s.advantage = a;
                }
                let n_traj = steps.len();
                let _ = self.dppo.update_bank(&mut self.proposer.bank, &steps);
                proposer_updates += 1;
                events.push(FlywheelEvent::ProposerUpdate {
                    update: self.update_counter,
                    trajectories: n_traj,
                });
                self.proposer_steps.clear();
                self.proposer_rewards_pending.clear();
            }

            // Park still-unfinished attempts in the carry-over buffer
            // (the token-level view mirrors the live attempts: rebuild it
            // from the current set each update).
            if !self.unfinished.is_empty() {
                self.carry = CarryOverBuffer::default();
                for carried in &self.unfinished {
                    self.carry.park(CarryTrajectory {
                        role: Role::Solver,
                        group_key: carried.env_id.clone(),
                        prompt: carried.env_id.clone(),
                        steps: carried.steps.clone(),
                        done: false,
                    });
                }
                events.push(FlywheelEvent::Carried {
                    attempts: self.unfinished.len(),
                });
            }
        }
        let mean_solver = if reward_count > 0.0 {
            reward_sum / reward_count
        } else {
            0.0
        };
        (
            solver_updates,
            retained_total,
            collected_total,
            mean_solver,
            proposer_updates,
        )
    }

    /// Snapshot of one representative solver attempt per env (for the
    /// workspace archive).
    fn solver_attempts_snapshot(&self, envs: &[HarborTask]) -> Vec<(String, Trajectory)> {
        // A fresh solver cannot be sampled through an immutable `self`, so
        // the snapshot stores a synthetic solved attempt per environment:
        // the archive's role is grounding for the proposer, and the
        // reference-attempt shape is what it reads.
        let mut out = Vec::new();
        for env in envs {
            let total = env.total_checks().max(1);
            let traj = Trajectory {
                role: Role::Solver,
                prompt: env.instruction.body.clone(),
                turns: vec![crate::trajectory::Turn::new(
                    crate::trajectory::Action::Command("harbor verify --phase agent".into()),
                    crate::trajectory::Observation {
                        stdout: env.tests.verifier_stdout(&crate::world::CtrfReport {
                            tests: vec![
                                crate::world::TestResult {
                                    name: "reference".into(),
                                    status: crate::world::TestStatus::Passed,
                                    duration_ms: 1,
                                };
                                total
                            ],
                        }),
                        exit_code: 0,
                    },
                )],
                outcome: Some(crate::trajectory::Outcome::full_pass(total)),
            };
            out.push((env.env_id.clone(), traj));
        }
        out
    }

    /// Collect the round's common admission failures for harness lessons.
    fn collect_common_failures(&self, admissions: &[Admission]) -> Vec<String> {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for a in admissions {
            if a.admitted {
                continue;
            }
            if let Some(v) = &a.validation {
                for f in &v.failures {
                    *counts.entry(f.clone()).or_insert(0) += 1;
                }
            }
        }
        counts
            .into_iter()
            .map(|(f, c)| format!("{f} (x{c})"))
            .collect()
    }
}

/// Collect one environment group: `group_size` trajectories (resuming
/// carried attempts with the updated policy). Long-horizon tasks bounded
/// by the agent timeout park as carried attempts and complete on later
/// updates.
fn collect_group(
    env_id: &str,
    pool: &TrainingPool,
    solver: &mut dyn SolverModel,
    unfinished: &mut Vec<CarriedAttempt>,
    pending: &mut BTreeMap<String, PendingGroup>,
    config: &FlywheelConfig,
    collection_sec: &mut f64,
) -> Option<CollectedGroup> {
    let task = pool.task(env_id)?;
    // Agent timeout -> check budget: a 120 s timeout over 30 s checks
    // bounds each invocation to 4 checks; longer tasks carry over.
    let check_budget = (task.toml.agent.timeout_sec / config.check_time_sec)
        .floor()
        .max(1.0) as usize;
    let entry = pending.entry(env_id.to_string()).or_default();

    // Resume carried attempts first ("They resume with the updated policy,
    // while stored sampling log-probabilities are retained").
    let carried: Vec<CarriedAttempt> = unfinished
        .iter()
        .filter(|c| c.env_id == env_id)
        .cloned()
        .collect();
    unfinished.retain(|c| c.env_id != env_id);
    for mut c in carried {
        let partial = solver.attempt_partial(
            task,
            None,
            Some(c.world.clone()),
            c.next_check,
            Some(check_budget),
        );
        c.steps.extend(partial.trace.steps.clone());
        c.turns.extend(partial.trace.turns.clone());
        *collection_sec += partial.trace.steps.len() as f64 * config.check_time_sec;
        if partial.done {
            entry.trajectories.push(Trajectory {
                role: Role::Solver,
                prompt: task.instruction.body.clone(),
                turns: c.turns.clone(),
                outcome: Some(partial.trace.outcome.clone()),
            });
            entry.steps.push(c.steps);
        } else {
            c.world = partial.world;
            c.next_check = partial.next_check;
            unfinished.push(c);
        }
    }

    // Fresh attempts up to the group size.
    while entry.trajectories.len() < config.solver_group_trajs {
        let partial = solver.attempt_partial(task, None, None, 0, Some(check_budget));
        *collection_sec += partial.trace.steps.len() as f64 * config.check_time_sec;
        if partial.done {
            entry.trajectories.push(Trajectory {
                role: Role::Solver,
                prompt: task.instruction.body.clone(),
                turns: partial.trace.turns.clone(),
                outcome: Some(partial.trace.outcome.clone()),
            });
            entry.steps.push(partial.trace.steps.clone());
        } else {
            // The attempt is unfinished: park it with its live sandbox
            // state and stored sampling log-probabilities.
            unfinished.push(CarriedAttempt {
                env_id: env_id.to_string(),
                steps: partial.trace.steps.clone(),
                world: partial.world,
                next_check: partial.next_check,
                turns: partial.trace.turns.clone(),
            });
            break;
        }
    }

    // Release the group once it has filled (Table 12: 16 trajectories).
    if entry.trajectories.len() >= config.solver_group_trajs {
        let released = pending.remove(env_id).expect("entry exists");
        return Some(CollectedGroup {
            group: TrajectoryGroup::new(Role::Solver, env_id, released.trajectories),
            steps: released.steps,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harbor::{Difficulty, SkillTag};
    use crate::metrics::wilson_interval;

    fn seed_tasks() -> Vec<HarborTask> {
        (0..3)
            .map(|i| {
                generate_task(&TaskGenParams {
                    env_id: format!("seed_{i:03}"),
                    name: format!("flywheel/seed_{i:03}"),
                    n_checks: 4 + i,
                    skills: vec![SkillTag::new("csv"), SkillTag::new("json")],
                    difficulty: Difficulty::Medium,
                    broken: false,
                })
            })
            .collect()
    }

    #[test]
    fn demo_config_matches_protocol_shape() {
        let c = FlywheelConfig::demo();
        assert_eq!(c.calibration.band, (0.25, 0.75));
        assert_eq!(c.calibration.initial_rollouts, 4);
        assert_eq!(c.proposer_interval, c.review_interval);
        let p = FlywheelConfig::paper();
        assert_eq!(p.solver_batch_envs, 16);
        assert_eq!(p.solver_group_trajs, 16);
        assert_eq!(p.proposer_group, 128);
        assert_eq!(p.proposer_interval, 16);
        assert_eq!(p.pool_capacity, 256);
        assert!((p.dppo.tv_threshold - 0.1).abs() < 1e-9);
        assert_eq!(p.dppo.kl_coef, 0.0);
    }

    #[test]
    fn one_round_runs_the_full_loop() {
        let mut fw = Flywheel::new(FlywheelConfig::demo(), seed_tasks());
        let report = fw.run_round();
        assert_eq!(report.round, 0);
        assert_eq!(report.solver_updates, fw.config.updates_per_round);
        assert!(!report.assignments.is_empty());
        assert!(report.pool_size >= 3);
        let events = fw.last_events.clone();
        let first_assignment = events
            .iter()
            .position(|e| matches!(e, FlywheelEvent::Assignment { .. }))
            .expect("assignments issued");
        let first_update = events
            .iter()
            .position(|e| matches!(e, FlywheelEvent::SolverUpdate { .. }))
            .expect("solver updates ran");
        assert!(first_assignment < first_update);
        assert!(events
            .iter()
            .any(|e| matches!(e, FlywheelEvent::WorkspaceRefreshed { .. })));
        assert!(events
            .iter()
            .any(|e| matches!(e, FlywheelEvent::HarnessOptimized { .. })));
    }

    #[test]
    fn multiple_rounds_train_the_solver() {
        let mut fw = Flywheel::new(FlywheelConfig::demo(), seed_tasks());
        for _ in 0..6 {
            fw.run_round();
        }
        // The solver's skill policies concentrate on ACTION_SOLVE over
        // training (DPPO on positive-advantage solving).
        let csv = fw.solver.bank.get("csv").cloned();
        let json = fw.solver.bank.get("json").cloned();
        assert!(csv.is_some() || json.is_some());
        if let Some(p) = csv {
            assert!(
                p.probs()[0] > 0.34,
                "csv solve prob should rise above uniform, got {}",
                p.probs()[0]
            );
        }
        assert!(fw.metrics.rounds() >= 6);
    }

    #[test]
    fn carry_over_exercises_with_long_tasks() {
        // Agent timeout 60s / 30s per check -> budget 2 checks; tasks with
        // 5+ checks cannot finish in one invocation.
        let seeds = vec![generate_task(&TaskGenParams {
            env_id: "seed_long".into(),
            name: "flywheel/seed_long".into(),
            n_checks: 6,
            skills: vec![SkillTag::new("csv"), SkillTag::new("json")],
            difficulty: Difficulty::Hard,
            broken: false,
        })];
        let mut fw = Flywheel::new(FlywheelConfig::demo(), seeds);
        for entry in fw.pool.entries.values_mut() {
            entry.task.toml.agent.timeout_sec = 60.0;
        }
        let r1 = fw.run_round();
        assert!(
            r1.carried_attempts > 0,
            "expected carry-over, got {}",
            r1.carried_attempts
        );
        // Carried attempts resume with the updated policy and eventually
        // complete.
        for _ in 0..3 {
            let _ = fw.run_round();
        }
        assert!(!fw.pool.is_empty());
    }

    #[test]
    fn round_report_useful_group_rate_bounded() {
        let mut fw = Flywheel::new(FlywheelConfig::demo(), seed_tasks());
        let r = fw.run_round();
        let rate = r.useful_group_rate();
        assert!((0.0..=1.0).contains(&rate));
    }

    #[test]
    fn wilson_interval_on_valid_env_rate() {
        let mut fw = Flywheel::new(FlywheelConfig::demo(), seed_tasks());
        fw.run_round();
        let (lo, hi) = wilson_interval(fw.metrics.admitted_total, fw.metrics.episodes_total);
        assert!(lo <= hi);
        assert!(lo >= 0.0 && hi <= 1.0);
    }
}
