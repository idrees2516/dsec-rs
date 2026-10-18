//! Integration-grade tests for the training contracts (Phases 5–6).

use dsec_rl::reward::RewardRecord;
use dsec_rl::tito::TranscriptTurn;

use dsec_train::policy::PolicyVersion;
use dsec_train::scheduler::{CellRole, ElasticPool, SchedulerConfig, Waterlevel};
use dsec_train::trainer::{RolloutSample, Trainer, TrainerConfig};

fn sample(exec: u64, group: u64, reward: f32, version: u64) -> RolloutSample {
    RolloutSample {
        execution: dsec_rl::loss::ExecutionInput {
            execution_id: exec,
            group_id: group,
            turns: vec![
                TranscriptTurn::System("sys words".into()),
                TranscriptTurn::User("task words".into()),
                TranscriptTurn::Assistant {
                    text: if reward > 0.5 {
                        "good answer words".into()
                    } else {
                        "bad answer words".into()
                    },
                    tool_calls: vec![],
                },
            ],
            reward: RewardRecord::valid(exec, group, reward),
        },
        policy_version: version,
    }
}

fn samples_group(group: u64, rewards: &[f32], version: u64) -> Vec<RolloutSample> {
    rewards
        .iter()
        .enumerate()
        .map(|(i, r)| sample(100 * group + i as u64, group, *r, version))
        .collect()
}

// ---------------------------------------------------------------------------
// Phase 5: policy versions + publication barrier
// ---------------------------------------------------------------------------

#[test]
fn publication_barrier_blocks_and_recovers() {
    let v0 = PolicyVersion {
        major: 0,
        digest: "d0".into(),
    };
    let store = dsec_train::policy::PolicyStore::new(v0, vec![1, 2, 3]);

    // stage a new version, then crash before the flip
    let v1 = PolicyVersion {
        major: 1,
        digest: "d1".into(),
    };
    store.stage(v1.clone(), vec![4, 5, 6]).unwrap();
    let crashed = store.publish(true);
    assert!(crashed.is_err(), "publication crash");
    assert!(store.barrier_closed(), "barrier stays closed");
    assert_eq!(store.published().major, 0, "OLD version still served");

    // a second stage attempt is refused while the barrier is closed
    let v2 = PolicyVersion {
        major: 2,
        digest: "d2".into(),
    };
    assert!(store.stage(v2, vec![]).is_err());

    // recovery: discard the staged flip — rollouts keep v0
    let still = store.recover_publication(false).unwrap();
    assert_eq!(still.major, 0);
    assert!(!store.barrier_closed(), "barrier re-opened");

    // publication works after recovery
    store.stage(v1, vec![4, 5, 6]).unwrap();
    let published = store.publish(false).unwrap();
    assert_eq!(published.major, 1);
    assert_eq!(store.published().major, 1);
    assert!(!store.barrier_closed());
}

// ---------------------------------------------------------------------------
// Phase 5: bounded staleness
// ---------------------------------------------------------------------------

#[test]
fn stale_samples_are_rejected_not_trained() {
    let mut trainer = Trainer::new(TrainerConfig {
        staleness_bound: 1,
        ..Default::default()
    });
    assert_eq!(trainer.published().major, 0);

    // step 1: fresh samples (version 0, published 0) → accepted
    let r1 = trainer
        .step(&samples_group(1, &[1.0, 0.0, 0.5, 0.25], 0))
        .unwrap();
    assert!(r1.rejected_stale.is_empty());
    assert_eq!(r1.published.as_ref().unwrap().major, 1);

    // step 2: samples from version 0 while published is 1 → age 1,
    // bound 1 → still accepted
    let r2 = trainer
        .step(&samples_group(2, &[1.0, 0.0, 0.5, 0.25], 0))
        .unwrap();
    assert!(r2.rejected_stale.is_empty());

    // step 3: version-0 samples with published at 2 → age 2 > bound 1
    // → REJECTED (never trained, reported)
    let r3 = trainer
        .step(&samples_group(3, &[1.0, 0.0, 0.5, 0.25], 0))
        .unwrap();
    assert_eq!(r3.rejected_stale.len(), 4);
    assert!(r3.accepted.is_empty());
    assert!(r3.published.is_none(), "no publication without data");

    // fresh samples still train
    let r4 = trainer
        .step(&samples_group(4, &[1.0, 0.0], trainer.published().major))
        .unwrap();
    assert!(r4.rejected_stale.is_empty());
    assert!(r4.published.is_some());
}

#[test]
fn evaluator_failures_are_excluded_never_zero_trained() {
    let mut trainer = Trainer::new(TrainerConfig::default());
    let mut group = samples_group(1, &[1.0, 0.0, 0.5, 0.25], 0);
    // one member's verifier died: EvaluatorFailure
    group[2].execution.reward = RewardRecord::evaluator_failure(group[2].execution.execution_id, 1);
    let result = trainer.step(&group).unwrap();
    assert!(
        result.excluded.contains(&group[2].execution.execution_id),
        "the failed sample is excluded, not trained as zero"
    );
    // the remaining members formed a baseline (terms > 0)
    assert!(result.terms > 0);
}

#[test]
fn real_learning_signal_moves_the_policy() {
    // reward "good words", punish "bad words" across many steps: the
    // model's preference must actually shift
    let mut trainer = Trainer::new(TrainerConfig {
        lr: 0.2,
        hidden: 24,
        ..Default::default()
    });
    // build vocabulary coverage first
    let mut all = samples_group(1, &[1.0, 0.0], 0);
    all.extend(samples_group(2, &[1.0, 0.0], 0));
    trainer.ensure_vocab(&all);
    let model = trainer.model().clone();
    let tokenizer = trainer.tokenizer().clone();
    let good = tokenizer.tokenize_known("good answer")[0];
    let bad = tokenizer.tokenize_known("bad answer")[0];
    let ctx = tokenizer.tokenize_known("sys words task words");
    let before = model.log_prob(&ctx, good) - model.log_prob(&ctx, bad);

    for step in 0..40 {
        let v = trainer.published().major;
        let group = samples_group(step, &[1.0, 0.0], v);
        trainer.step(&group).unwrap();
    }
    let after = trainer.model().log_prob(&ctx, good) - trainer.model().log_prob(&ctx, bad);
    assert!(
        after > before,
        "rewarding 'good' and punishing 'bad' must widen the gap: {before} -> {after}"
    );
}

// ---------------------------------------------------------------------------
// Phase 6: scheduler invariants
// ---------------------------------------------------------------------------

#[test]
fn draining_never_drops_in_flight_calls() {
    let v0 = PolicyVersion {
        major: 0,
        digest: "d0".into(),
    };
    let store = dsec_train::policy::PolicyStore::new(v0, vec![]);
    let mut pool = ElasticPool::new(
        SchedulerConfig {
            cells: 4,
            initial_rollout: 3,
            waterlevel: Waterlevel {
                low: 2,
                high: 64,
                batch_size: 8,
                ..Default::default()
            },
            gen_ms: 20,
            train_slice_ms: 5,
            concurrency_per_cell: 4,
            tail: 3,
            seed: 7,
            rebalance_ticks: 30,
        },
        store,
    );
    // submit rollouts and let work start
    for id in 0..12u64 {
        pool.submit_rollout(id, 2);
    }
    for _ in 0..3 {
        pool.tick();
    }
    let in_flight: usize = pool.cells().iter().map(|c| c.in_flight.len()).sum();
    assert!(in_flight > 0, "work is in flight");

    // drain one rollout cell
    let draining_id = pool
        .cells()
        .iter()
        .find(|c| c.role == CellRole::Rollout)
        .map(|c| c.id)
        .unwrap();
    pool.request_transition(draining_id, CellRole::Train)
        .unwrap();
    let cell = &pool.cells()[draining_id];
    assert!(cell.draining);
    assert!(cell.queue.is_empty(), "queued work was rerouted");

    // tick until the transition completes: in-flight calls must all
    // finish (none dropped)
    let before: usize = pool.cells()[draining_id].in_flight.len();
    for _ in 0..50 {
        pool.tick();
        let cell = &pool.cells()[draining_id];
        if !cell.draining {
            assert_eq!(cell.role, CellRole::Train);
            break;
        }
        // invariant: the cell keeps serving until its in-flight calls
        // complete — the count only decreases
        assert!(cell.in_flight.len() <= before);
    }
    assert!(!pool.cells()[draining_id].draining, "transition completed");
}

#[test]
fn barrier_blocks_transitions_until_recovered() {
    let v0 = PolicyVersion {
        major: 0,
        digest: "d0".into(),
    };
    let store = dsec_train::policy::PolicyStore::new(v0, vec![]);
    let mut pool = ElasticPool::new(SchedulerConfig::default(), store.clone());

    // simulate a crashed publication: barrier closed, staged checkpoint
    let v1 = PolicyVersion {
        major: 1,
        digest: "d1".into(),
    };
    store.stage(v1, vec![9]).unwrap();
    let _ = store.publish(true); // crash: barrier stays closed
    assert!(store.barrier_closed());

    // transitions are refused while the barrier is closed
    let err = pool.request_transition(0, CellRole::Train);
    assert!(err.is_err(), "barrier blocks role flips");

    // recovery re-opens the world
    store.recover_publication(true).unwrap();
    assert!(!store.barrier_closed());
    assert!(pool.request_transition(0, CellRole::Train).is_ok());
}

#[test]
fn elastic_pool_drains_full_workload() {
    let v0 = PolicyVersion {
        major: 0,
        digest: "d0".into(),
    };
    let store = dsec_train::policy::PolicyStore::new(v0, vec![]);
    let mut pool = ElasticPool::new(SchedulerConfig::default(), store);
    for id in 0..16u64 {
        pool.submit_rollout(id, 4);
    }
    let mut total_gens = 0u64;
    let mut total_batches = 0u64;
    for _ in 0..5_000 {
        let summary = pool.tick();
        total_gens += summary
            .completions
            .iter()
            .filter(|(_, k)| *k == "generation")
            .count() as u64;
        total_batches += summary.batches_completed as u64;
        if total_gens >= 64 && total_batches >= 2 {
            break;
        }
    }
    assert_eq!(total_gens, 64, "every model call was served");
    assert!(total_batches >= 2, "batches trained, got {total_batches}");
    assert_eq!(pool.cfg_waterlevel_batch(), 8);
}

// ---------------------------------------------------------------------------
// Phase 6: benchmark
// ---------------------------------------------------------------------------

#[test]
fn benchmark_elastic_beats_colocate_and_async() {
    let workload = dsec_train::bench::Workload {
        groups: 16,
        group_size: 8,
        gens_per_rollout: 8,
        burst_ms: 200,
    };
    let cfg = SchedulerConfig {
        cells: 8,
        initial_rollout: 5,
        waterlevel: Waterlevel {
            low: 2,
            high: 8,
            batch_size: 8,
            ..Default::default()
        },
        gen_ms: 40,
        train_slice_ms: 12,
        concurrency_per_cell: 4,
        tail: 3,
        seed: 42,
        rebalance_ticks: 30,
    };
    let results = dsec_train::bench::run_comparison(workload, cfg).unwrap();
    let async_r = &results[0];
    let colocate = &results[1];
    let elastic = &results[2];

    // same work served everywhere (correctness of the comparison)
    assert_eq!(async_r.generations_served, elastic.generations_served);
    assert_eq!(colocate.generations_served, elastic.generations_served);
    assert!(elastic.batches_completed >= colocate.batches_completed);

    // the structural claim: elasticity beats phase-serialization and
    // the fixed split on makespan (simulated clocks)
    let over_colocate = elastic.speedup_over(colocate);
    let over_async = elastic.speedup_over(async_r);
    assert!(
        over_colocate > 1.0,
        "elastic must beat colocate (simulated), got {over_colocate:.2}x"
    );
    assert!(
        over_async > 1.0,
        "elastic must beat async (simulated), got {over_async:.2}x"
    );
}
