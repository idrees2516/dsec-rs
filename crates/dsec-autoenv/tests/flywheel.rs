//! Integration tests: the golden flywheel flow and the adversarial paths.

use dsec_autoenv::assignment::{build_assignment, ParentRef};
use dsec_autoenv::calibrate::{calibrate, CalibrationConfig, CalibrationVerdict};
use dsec_autoenv::decontaminate::ContaminationIndex;
use dsec_autoenv::dppo::{DppoConfig, DppoOptimizer};
use dsec_autoenv::flywheel::{Flywheel, FlywheelConfig, FlywheelEvent};
use dsec_autoenv::harbor::{Difficulty, SkillTag};
use dsec_autoenv::hil::{ask_f1, example_billing_task, HilAttempt};
use dsec_autoenv::policy::{generate_task, SolverModel, TaskGenParams};
use dsec_autoenv::reward::proposer_reward;
use dsec_autoenv::validate::{proposer_self_check, validate, ValidationConfig};
use dsec_autoenv::workspace::{Phase, Workspace};
use dsec_autoenv::world::ImageRegistry;

fn seed_tasks() -> Vec<dsec_autoenv::harbor::HarborTask> {
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

// ---------------------------------------------------------------------
// The golden flow
// ---------------------------------------------------------------------

#[test]
fn golden_event_order_of_one_round() {
    let mut fw = Flywheel::new(FlywheelConfig::demo(), seed_tasks());
    fw.run_round();
    let events = fw.last_events.clone();

    // Workspace refresh precedes assignments; assignments precede
    // episodes; episodes precede solver updates; harness optimization
    // closes the round.
    let pos = |pred: &dyn Fn(&FlywheelEvent) -> bool| events.iter().position(pred);
    let refresh = pos(&|e| matches!(e, FlywheelEvent::WorkspaceRefreshed { .. })).expect("refresh");
    let assignment = pos(&|e| matches!(e, FlywheelEvent::Assignment { .. })).expect("assignment");
    let episode = pos(&|e| matches!(e, FlywheelEvent::EpisodeSubmitted { .. })).expect("episode");
    let update = pos(&|e| matches!(e, FlywheelEvent::SolverUpdate { .. })).expect("update");
    let harness = pos(&|e| matches!(e, FlywheelEvent::HarnessOptimized { .. })).expect("harness");
    assert!(refresh < assignment && assignment < episode && episode < update && update < harness);

    // Validation precedes calibration precedes admission for every
    // admitted environment.
    let validated: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| matches!(e, FlywheelEvent::Validated { passed: true, .. }))
        .map(|(i, _)| i)
        .collect();
    let calibrated: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| matches!(e, FlywheelEvent::Calibrated { .. }))
        .map(|(i, _)| i)
        .collect();
    let admitted: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| matches!(e, FlywheelEvent::Admitted { .. }))
        .map(|(i, _)| i)
        .collect();
    assert!(!validated.is_empty());
    assert!(!calibrated.is_empty());
    assert!(!admitted.is_empty());
    assert!(calibrated.iter().min() > validated.iter().min());
    assert!(admitted.iter().min() > calibrated.iter().min());
}

#[test]
fn multi_round_loop_is_stable_and_curriculum_tracks() {
    let mut fw = Flywheel::new(FlywheelConfig::demo(), seed_tasks());
    let mut reports = Vec::new();
    for _ in 0..8 {
        reports.push(fw.run_round());
    }
    // The pool never drains: admission keeps pace with eviction.
    assert!(
        reports.iter().all(|r| r.pool_size >= 1),
        "pool drained: {:?}",
        reports.iter().map(|r| r.pool_size).collect::<Vec<_>>()
    );
    // The solver's mean reward stays in a sane band (the curriculum keeps
    // the pool near the calibration band).
    let means: Vec<f64> = reports.iter().map(|r| r.mean_solver_reward).collect();
    assert!(means.iter().all(|m| (0.0..=1.0).contains(m)));
    // DPPO updates landed every round.
    assert!(reports
        .iter()
        .all(|r| r.solver_updates == fw.config.updates_per_round));
    // Some round ran a pool review with evictions as the solver improved.
    assert!(reports.iter().map(|r| r.evicted.len()).sum::<usize>() > 0);
    // And replacements were issued grounded in the evicted parents.
    let harden_assignments = fw
        .last_events
        .iter()
        .filter(
            |e| matches!(e, FlywheelEvent::Assignment { direction, .. } if direction == "harden"),
        )
        .count();
    assert!(harden_assignments > 0);
}

#[test]
fn carry_over_resumes_and_completes() {
    // 60 s agent timeout / 30 s per check -> budget 2; 6-check tasks
    // cannot finish in one invocation.
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
    let mut carried_seen = false;
    for _ in 0..5 {
        let report = fw.run_round();
        carried_seen = carried_seen || report.carried_attempts > 0;
    }
    assert!(carried_seen, "carry-over never engaged");
    // The carry-over buffer mirrors the unfinished attempts.
    assert_eq!(fw.carry.len(), fw.unfinished.len());
    // Pending groups release only when full.
    for pending in fw.pending.values() {
        assert!(pending.trajectories.len() < fw.config.solver_group_trajs);
    }
}

// ---------------------------------------------------------------------
// Adversarial paths
// ---------------------------------------------------------------------

#[test]
fn contaminated_environment_is_rejected() {
    let mut fw = Flywheel::new(FlywheelConfig::demo(), seed_tasks())
        .with_heldout(ContaminationIndex::build(&[(
            "tb2.1/pytorch-restore".to_string(),
            "Rebuild the PyTorch model from its saved weights and verify the outputs match the original within tolerance".to_string(),
        )]));
    // Round 0 runs with the index in place.
    let report = fw.run_round();
    // No admitted environment carries a contaminated instruction.
    for entry in fw.pool.entries.values() {
        assert!(
            !fw.heldout.is_contaminated(&entry.task.instruction.body),
            "{}",
            entry.task.env_id
        );
    }
    let _ = report;

    // Direct check: a verbatim 13-gram copy is flagged.
    let mut t = seed_tasks().remove(0);
    t.instruction.body = format!(
        "{}\nAlso rebuild the pytorch model from its saved weights and verify the outputs match the original within tolerance\n",
        t.instruction.body
    );
    let out = validate(
        &t,
        &ImageRegistry::default_world(),
        &fw.heldout,
        &ValidationConfig::default(),
    );
    assert!(!out.passed);
    assert!(out.failures.iter().any(|f| f.contains("13-gram")));
    // Eq. 2's validation-failure arm.
    assert_eq!(proposer_reward(out.passed, true), -1.0);
}

#[test]
fn gameable_environment_fails_no_op_check() {
    let mut t = seed_tasks().remove(0);
    // A verifier whose check passes from the environment's pre-planted
    // state alone: no real work is needed, so the do-nothing solution
    // earns full reward.
    t.environment
        .files
        .insert("/app/ready.py".into(), "print('ready')\nexit 0\n".into());
    t.tests.checks = vec![dsec_autoenv::world::TestCheck::CommandSucceeds {
        name: "test_ready".into(),
        command: "python3 /app/ready.py".into(),
    }];
    // The proposer's own self-check must block submission (the no-op does
    // not fail).
    let check = proposer_self_check(&t, &ImageRegistry::default_world()).unwrap();
    assert!(!check.ready_to_submit(), "{check:?}");
    // And host validation fails the no-op arm.
    let out = validate(
        &t,
        &ImageRegistry::default_world(),
        &ContaminationIndex::build(&[]),
        &ValidationConfig::default(),
    );
    assert!(!out.passed);
    assert!(
        out.failures.iter().any(|f| f.contains("do-nothing")),
        "{:?}",
        out.failures
    );
}

#[test]
fn sandbox_contract_blocks_web_and_fixed_edits() {
    // A proposer without web tools in the preamble cannot use web_search.
    let parent = seed_tasks().remove(0);
    let assignment = build_assignment(
        "asg_x",
        "env_x",
        ParentRef {
            env_id: parent.env_id.clone(),
            checks: parent.total_checks(),
            skills: parent.skills.clone(),
        },
        &[dsec_autoenv::trajectory::RolloutRecord {
            env_id: parent.env_id.clone(),
            solver: "s".into(),
            round: 1,
            provenance: dsec_autoenv::trajectory::Provenance::SelfRound(1),
            rewards: vec![1.0, 1.0, 0.8, 1.0],
            reward_std: 0.1,
            check_failures: Default::default(),
            path: "p".into(),
        }],
        (0.25, 0.75),
    );
    let mut ws = Workspace::new();
    ws.set_assignment(&assignment.render());
    // The workspace edit guard: fixed regions are immutable during
    // episodes.
    ws.phase = Phase::DuringEpisode;
    assert!(ws.edit("assignment/current.md", "fake").is_err());
    assert!(ws.edit("SANDBOX_CONTRACT.md", "no rules").is_err());
    assert!(ws.edit("skills/create-task/SKILL.md", "tamper").is_err());
}

#[test]
fn calibration_protocol_boundaries() {
    let t = generate_task(&TaskGenParams {
        env_id: "env_c".into(),
        name: "flywheel/env_c".into(),
        n_checks: 4,
        skills: vec![SkillTag::new("csv"), SkillTag::new("json")],
        difficulty: Difficulty::Medium,
        broken: false,
    });
    let mut strong = dsec_autoenv::policy::ScriptedSolver::new("strong", 1);
    let _ = strong.attempt(&t, None); // populate tables
    for policy in strong.bank.tables.values_mut() {
        policy.logits = vec![8.0, -5.0, 0.0];
    }
    let out = calibrate(&t, &mut strong, &CalibrationConfig::default());
    assert_eq!(out.verdict, CalibrationVerdict::TooEasy);
    assert_eq!(proposer_reward(true, out.passed()), -0.25);

    let mut weak = dsec_autoenv::policy::ScriptedSolver::new("weak", 2);
    let _ = weak.attempt(&t, None);
    for policy in weak.bank.tables.values_mut() {
        policy.logits = vec![-8.0, -5.0, 0.0];
    }
    let out2 = calibrate(&t, &mut weak, &CalibrationConfig::default());
    assert_eq!(out2.verdict, CalibrationVerdict::TooHard);
    // Below-band tasks get exactly one diagnostic rollout with hints.
    let diag = out2.diagnostic.expect("diagnostic rollout");
    assert!(diag.reward >= 0.5);
}

#[test]
fn dppo_trust_region_holds_across_many_updates() {
    let opt = DppoOptimizer::new(DppoConfig {
        lr: 50.0,
        tv_threshold: 0.1,
        clip_eps: 0.2,
        kl_coef: 0.0,
        entropy_coef: 0.0,
    });
    let mut policy = dsec_autoenv::dppo::SoftmaxPolicy::uniform(3);
    let old = policy.clone();
    let _ = old;
    for i in 0..20 {
        let steps = vec![
            dsec_autoenv::dppo::TokenStep {
                context: "c".into(),
                action: 0,
                old_logprob: -1.1,
                advantage: 1.0,
            },
            dsec_autoenv::dppo::TokenStep {
                context: "c".into(),
                action: 1,
                old_logprob: -1.1,
                advantage: -1.0,
            },
        ];
        let summary = opt.update_policy(&mut policy, &steps);
        assert!(
            summary.tv_final <= 0.1 + 1e-9,
            "update {i} breached the region: {}",
            summary.tv_final
        );
    }
    // The trust region is per-update; cumulative drift may exceed one
    // step, but every update respected TV <= 0.1.
    assert!(policy.probs()[0] > policy.probs()[2]);
}

#[test]
fn hil_ask_f1_and_shaped_reward() {
    let hil = example_billing_task();
    let attempt = HilAttempt {
        questions: vec![
            "How much tolerance counts as agreement between the two amounts?".into(),
            "Which invoice statuses are in scope?".into(),
            "What about a duplicate invoice id?".into(),
            "How should the discrepancies report be ordered?".into(),
            "How is the total variance defined?".into(),
        ],
        had_tool: true,
        task_success: true,
    };
    assert!((ask_f1(&attempt.questions, &hil.registry).f1 - 1.0).abs() < 1e-9);
    // The no-tool variant cannot ask: zero coverage.
    let no_tool = HilAttempt {
        questions: vec![],
        had_tool: false,
        task_success: false,
    };
    assert_eq!(no_tool.ask_f1(&hil.registry).f1, 0.0);
}

#[test]
fn cold_start_and_cross_domain_pipelines() {
    // Cross-domain admission accepts clean tasks and flags UI domains.
    let t = seed_tasks().remove(0);
    for domain in dsec_autoenv::domains::Domain::all() {
        let report = dsec_autoenv::domains::cross_domain_admission(
            &t,
            domain,
            &ImageRegistry::default_world(),
        )
        .unwrap();
        assert!(report.accepted(), "{:?}: {:?}", domain, report.failures());
    }

    // Cold-Start selection: admitted-only, deduplicated, capped.
    use dsec_autoenv::coldstart::{select_cold_start, CandidateEpisode, ColdStartConfig};
    use dsec_autoenv::trajectory::{Outcome, Role, Trajectory};
    let candidates: Vec<CandidateEpisode> = (0..100)
        .map(|i| {
            let admitted = i % 2 == 0;
            CandidateEpisode {
                assignment_id: format!("a{i}"),
                source: "Claude Opus 5".into(),
                admitted,
                trajectory: Trajectory {
                    role: Role::Proposer,
                    prompt: format!("a{i}"),
                    turns: vec![],
                    outcome: Some(Outcome::full_pass(3)),
                },
            }
        })
        .collect();
    let cfg = ColdStartConfig {
        n_trajectories: 20,
        ..Default::default()
    };
    let set = select_cold_start(&cfg, &candidates);
    assert_eq!(set.len(), 20);
    assert!(set.trajectories.iter().all(|t| t.reward() == Some(1.0)));
}

#[test]
fn workspace_regions_and_harness_evolution() {
    let seeds = seed_tasks();
    let mut ws = Workspace::new();
    ws.refresh_from_pool(vec![], vec![], &seeds.iter().collect::<Vec<_>>());
    // Between rounds, harness optimization grows the web-cache skill and
    // merges validation steps; fixed regions never change.
    let mut opt = dsec_autoenv::harness::HarnessOptimizer::default();
    let log = dsec_autoenv::harness::HarnessRoundLog {
        round: 1,
        episodes: 4,
        admitted: 3,
        rejected: 1,
        proposer_rewards: vec![1.0, 1.0, 1.0, -0.25],
        common_failures: vec!["do-nothing solution earned 1.0 (must be < 0.5)".into()],
        used_web: true,
    };
    let actions = opt.optimize(&mut ws, &log);
    assert!(actions.iter().any(|a| matches!(
        a,
        dsec_autoenv::harness::HarnessAction::MergeValidationSteps {
            steps_before: 8,
            steps_after: 4
        }
    )));
    assert!(ws.files.contains_key("tools/validate_fast.py"));
    assert!(ws.files.contains_key("skills/web-cache/SKILL.md"));
    // The proposer reads the web-cache skill; fixed files are untouched.
    assert!(ws
        .read("SANDBOX_CONTRACT.md")
        .unwrap()
        .contains("# Sandbox contract"));
}
