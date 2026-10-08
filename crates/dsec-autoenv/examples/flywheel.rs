//! End-to-end AutoEnvScaling data-flywheel demo.
//!
//! Runs the complete loop from the paper over several rounds with a
//! deterministic simulated solver and proposer, then reports the metrics
//! the paper tracks (Figures 4–7): valid-environment rate with a 95%
//! Wilson interval, cost per valid environment, useful-group rate,
//! collection time, pool evolution, and proposer rewards (Eq. 2). It also
//! demonstrates the three auxiliary pipelines: the HiL clarification task
//! (Section 4.5) with Ask-F1, Cold-Start selection (Appendix H), and
//! cross-domain admission (Appendix C).
//!
//! ```bash
//! cargo run -p dsec-autoenv --example flywheel
//! ```

use dsec_autoenv::assignment::Direction;
use dsec_autoenv::coldstart::{select_cold_start, CandidateEpisode, ColdStartConfig};
use dsec_autoenv::decontaminate::ContaminationIndex;
use dsec_autoenv::domains::{cross_domain_admission, Domain};
use dsec_autoenv::dppo::DppoConfig;
use dsec_autoenv::flywheel::{Flywheel, FlywheelConfig, FlywheelEvent};
use dsec_autoenv::harbor::{Difficulty, SkillTag};
use dsec_autoenv::hil::{ask_f1, example_billing_task, HilAttempt};
use dsec_autoenv::metrics::RunMetrics;
use dsec_autoenv::policy::{generate_task, TaskGenParams};
use dsec_autoenv::trajectory::{Outcome, Role, Trajectory};

/// The seed pool: a small Tmax-style static environment set.
fn seed_pool() -> Vec<dsec_autoenv::harbor::HarborTask> {
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

/// Held-out benchmark descriptions for decontamination (Terminal-Bench
/// style).
fn heldout_index() -> ContaminationIndex {
    ContaminationIndex::build(&[
        (
            "tb2.1/pytorch-restore".to_string(),
            "Rebuild the PyTorch model from its saved weights and verify the outputs match the original within tolerance".to_string(),
        ),
        (
            "tb2.1/ssh-key-pair".to_string(),
            "Generate an SSH key pair in the files id_rsa and id_rsa.pub without a password".to_string(),
        ),
    ])
}

fn main() {
    println!("AutoEnvScaling: automating the data flywheel with terminal agents");
    println!("====================================================================\n");

    // --- Configuration: the paper's protocol at demo scale.
    let mut config = FlywheelConfig::demo();
    config.dppo = DppoConfig {
        lr: 1.0,
        tv_threshold: 0.1,
        clip_eps: 0.2,
        kl_coef: 0.0,
        entropy_coef: 0.0,
    };
    println!(
        "protocol: band [{:.2}, {:.2}], std >= {:.1}, {}->{} rollouts, k = {}, pool <= {}, DPPO TV <= {:.1}\n",
        config.calibration.band.0,
        config.calibration.band.1,
        config.calibration.min_std,
        config.calibration.initial_rollouts,
        config.calibration.expanded_rollouts,
        config.proposer_interval,
        config.pool_capacity,
        config.dppo.tv_threshold
    );

    let mut flywheel = Flywheel::new(config, seed_pool()).with_heldout(heldout_index());

    // --- The flywheel rounds.
    let rounds = 6;
    let mut last_report = None;
    for r in 0..rounds {
        let report = flywheel.run_round();
        println!(
            "round {r}: pool {:>2} | admitted {:>2} rejected {:>2} | evicted {:>1} | solver-reward {:.3} | groups {}/{} | carried {} | {:.1} min",
            report.pool_size,
            report.admitted,
            report.rejected,
            report.evicted.len(),
            report.mean_solver_reward,
            report.groups_retained,
            report.groups_collected,
            report.carried_attempts,
            report.collection_minutes
        );
        for ev in &report.evicted {
            println!("           evict {}: {}", ev.env_id, ev.reason);
        }
        for ep in &report.episodes {
            if let Some(a) = &ep.admission {
                let cal = a
                    .calibration
                    .as_ref()
                    .map(|c| format!("mean {:.2} std {:.2} ({:?})", c.mean, c.std, c.verdict))
                    .unwrap_or_else(|| "validation failed".into());
                println!(
                    "           {:<10} {:<9} r_P {:+.2}  {}",
                    ep.env_id, ep.direction, a.proposer_reward, cal
                );
            }
        }
        if !report.harness_actions.is_empty() {
            for a in &report.harness_actions {
                println!("           harness: {}", a.describe());
            }
        }
        last_report = Some(report);
    }
    let _ = last_report;

    // --- Running metrics (Figures 4-7).
    let m: &RunMetrics = &flywheel.metrics;
    let (lo, hi) = m.valid_env_rate_ci();
    println!("\nmetrics over {} rounds:", m.rounds());
    println!(
        "  valid-environment rate: {:.1}% (95% Wilson CI [{:.1}%, {:.1}%])",
        m.valid_env_rate() * 100.0,
        lo * 100.0,
        hi * 100.0
    );
    println!(
        "  cost per valid environment: {:.2} units",
        m.cost_per_valid_env()
    );
    println!(
        "  useful-group rate: {:.1}% (retained {}/{} collected)",
        m.useful_group_rate() * 100.0,
        m.groups_retained_total,
        m.groups_collected_total
    );
    println!(
        "  mean collection time: {:.1} min/update",
        m.mean_collection_minutes()
    );

    // --- The event stream of the final round (the golden flow).
    println!("\nfinal-round event stream:");
    for e in &flywheel.last_events.iter().take(14).collect::<Vec<_>>() {
        println!("  {}", describe_event(e));
    }
    println!("  ... ({} events total)", flywheel.last_events.len());

    // --- Proposer training moved the design policy.
    println!("\nproposer design policies after training:");
    for (ctx, policy) in flywheel.proposer.bank.tables.iter() {
        let probs = policy.probs();
        println!(
            "proposer policy [{ctx}]: conservative {:.2} / balanced {:.2} / ambitious {:.2}",
            probs[0], probs[1], probs[2]
        );
    }

    // --- Solver skill policies after training.
    println!("\nsolver skill policies after training:");
    for (skill, policy) in flywheel.solver.bank.tables.iter() {
        let probs = policy.probs();
        println!(
            "  {skill:<8} solve {:.2} / sloppy {:.2} / skip {:.2}",
            probs[0], probs[1], probs[2]
        );
    }

    // --- Section 4.5: the HiL clarification task (Figure 9 / Table 6).
    println!("\nHiL clarification task (billing/ledger reconciliation):");
    let hil = example_billing_task();
    println!(
        "  instruction: {} words; {} withheld blockers",
        hil.instruction.split_whitespace().count(),
        hil.registry.blockers.len()
    );
    let attempts = [
        HilAttempt {
            questions: vec![
                "How much tolerance counts as agreement between the two amounts?".to_string(),
                "Which invoice statuses are in scope?".to_string(),
                "How is the total variance defined?".to_string(),
            ],
            had_tool: true,
            task_success: true,
        },
        HilAttempt {
            questions: vec![],
            had_tool: false,
            task_success: false,
        },
    ];
    for (i, a) in attempts.iter().enumerate() {
        let f1 = a.ask_f1(&hil.registry);
        println!(
            "  attempt {i}: asked {} questions, success {} -> Ask-F1 {:.2} (P {:.2} / R {:.2})",
            a.questions.len(),
            a.task_success,
            f1.f1,
            f1.precision,
            f1.recall
        );
    }
    let full = hil.derive_full_info();
    println!(
        "  full-info variant derived: {} chars (answers inlined)",
        full.len()
    );
    let f1_all = ask_f1(
        &[
            "How much tolerance counts as agreement between the two amounts?".to_string(),
            "Which invoice statuses are in scope?".to_string(),
            "What about a duplicate invoice id?".to_string(),
            "How should the discrepancies report be ordered?".to_string(),
            "How is the total variance defined?".to_string(),
        ],
        &hil.registry,
    );
    println!("  perfect questioning: Ask-F1 {:.2}", f1_all.f1);

    // --- Appendix H: Cold-Start selection.
    println!("\nCold-Start selection (600 high-quality trajectories):");
    let candidates: Vec<CandidateEpisode> = (0..1200)
        .map(|i| {
            let admitted = i % 3 != 0; // ~66% admitted
            CandidateEpisode {
                assignment_id: format!("asg_{i:04}"),
                source: ["Claude Opus 5", "DeepSeek-V4-Flash", "Kimi-K3"][i % 3].to_string(),
                admitted,
                trajectory: Trajectory {
                    role: Role::Proposer,
                    prompt: format!("assignment {i}"),
                    turns: vec![],
                    outcome: Some(Outcome::full_pass(4)),
                },
            }
        })
        .collect();
    let set = select_cold_start(&ColdStartConfig::default(), &candidates);
    println!(
        "  {} candidates -> {} selected ({} distinct assignments, cap 600)",
        candidates.len(),
        set.len(),
        set.distinct_assignments
    );

    // --- Appendix C: cross-domain admission.
    println!("\ncross-domain admission (7 domains):");
    for domain in Domain::all() {
        let task = generate_task(&TaskGenParams {
            env_id: format!("cross_{}", domain.name().to_lowercase().replace(' ', "_")),
            name: format!(
                "flywheel/cross_{}",
                domain.name().to_lowercase().replace(' ', "_")
            ),
            n_checks: 4,
            skills: vec![SkillTag::new("csv"), SkillTag::new("json")],
            difficulty: Difficulty::Medium,
            broken: false,
        });
        let report = cross_domain_admission(
            &task,
            domain,
            &dsec_autoenv::world::ImageRegistry::default_world(),
        )
        .unwrap();
        println!(
            "  {:<22} accepted {} ({})",
            domain.name(),
            report.accepted(),
            if report.simulated {
                "file-simulated"
            } else {
                "ui-declared, file contract"
            }
        );
    }

    // --- Directions observed this run.
    println!("\ncurriculum directions: simplify (solver keeps failing), harden (longer horizon / additional skill), fresh (pool growth)");
    let directions: Vec<&str> = flywheel
        .last_events
        .iter()
        .filter_map(|e| match e {
            FlywheelEvent::Assignment { direction, .. } => Some(direction.as_str()),
            _ => None,
        })
        .collect();
    println!("  final round issued: {}", directions.join(", "));
    let _ = Direction::Fresh;
    println!("\nflywheel complete.");
}

/// One-line rendering of a flywheel event.
fn describe_event(e: &FlywheelEvent) -> String {
    match e {
        FlywheelEvent::PoolReview { update, evicted } => {
            format!("pool-review @ update {update} (evicted {evicted})")
        }
        FlywheelEvent::WorkspaceRefreshed {
            rollouts,
            environments,
        } => {
            format!("workspace-refreshed (rollouts {rollouts}, environments {environments})")
        }
        FlywheelEvent::Assignment {
            assignment_id,
            parent,
            direction,
        } => {
            format!(
                "assignment {assignment_id} <- parent {:?} ({direction})",
                parent.as_deref().unwrap_or("<fresh>")
            )
        }
        FlywheelEvent::EpisodeSubmitted {
            assignment_id,
            env_id,
            submitted,
        } => {
            format!("episode {assignment_id} -> {env_id} (submitted {submitted})")
        }
        FlywheelEvent::Validated { env_id, passed } => {
            format!("validated {env_id} (passed {passed})")
        }
        FlywheelEvent::Calibrated {
            env_id,
            verdict,
            mean,
            std,
        } => {
            format!("calibrated {env_id}: {verdict} (mean {mean:.2}, std {std:.2})")
        }
        FlywheelEvent::Admitted { env_id } => format!("admitted {env_id} -> training pool"),
        FlywheelEvent::Rejected { env_id, reward } => {
            format!("rejected {env_id} (r_P {reward:+.2})")
        }
        FlywheelEvent::SolverUpdate {
            update,
            groups,
            mean_reward,
        } => {
            format!("solver-update #{update} ({groups} groups, mean reward {mean_reward:.2})")
        }
        FlywheelEvent::ProposerUpdate {
            update,
            trajectories,
        } => {
            format!("proposer-update @ {update} ({trajectories} trajectories)")
        }
        FlywheelEvent::HarnessOptimized { actions } => {
            format!("harness-optimized ({actions} actions)")
        }
        FlywheelEvent::Carried { attempts } => {
            format!("carried {attempts} unfinished attempts into the next update")
        }
    }
}
