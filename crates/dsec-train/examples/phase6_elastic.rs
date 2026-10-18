//! Phase 6 demo: elastic GPU scheduling vs the Async and Colocate
//! baselines (simulated, calibrated logical clocks).
//!
//! Also demonstrates a draining role transition with safe rerouting.
//!
//! Run: `cargo run --release --example phase6_elastic -p dsec-train`

use dsec_train::bench::{run_comparison, Workload};
use dsec_train::policy::{PolicyStore, PolicyVersion};
use dsec_train::scheduler::{CellRole, ElasticPool, SchedulerConfig, Waterlevel};

fn main() {
    println!("== phase 6 demo: elastic GPU scheduling ==");
    let workload = Workload {
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

    let results = run_comparison(workload, cfg.clone()).unwrap();
    println!();
    for r in &results {
        println!(
            "{:<28} makespan={:>6}ms  batches={:<3} rollout_util={:.2}",
            r.strategy, r.makespan_ms, r.batches_completed, r.rollout_utilization
        );
    }
    let elastic = &results[2];
    println!();
    println!(
        "elastic vs Async    : {:.2}x",
        elastic.speedup_over(&results[0])
    );
    println!(
        "elastic vs Colocate : {:.2}x",
        elastic.speedup_over(&results[1])
    );
    println!("(simulated clocks — QWENGYRE reports 1.38-1.78x / 1.21-1.85x on real hardware)");

    // a draining transition with safe rerouting, live
    println!();
    println!("-- draining transition demo --");
    let v0 = PolicyVersion {
        major: 0,
        digest: "demo".into(),
    };
    let store = PolicyStore::new(v0, vec![]);
    let mut pool = ElasticPool::new(cfg, store);
    for id in 0..24u64 {
        pool.submit_rollout(id, 6);
    }
    for _ in 0..3 {
        pool.tick();
    }
    let victim = pool
        .cells()
        .iter()
        .find(|c| c.role == CellRole::Rollout)
        .map(|c| c.id)
        .unwrap();
    let (in_flight, queued) = {
        let c = &pool.cells()[victim];
        (c.in_flight.len(), c.queue.len())
    };
    pool.request_transition(victim, CellRole::Train).unwrap();
    println!(
        "cell {victim} draining: {in_flight} in-flight calls must finish, {queued} queued jobs rerouted"
    );
    for _ in 0..100 {
        let s = pool.tick();
        let c = &pool.cells()[victim];
        if !c.draining {
            println!(
                "cell {victim} is now {:?} after {} completions this drain (no call dropped)",
                c.role,
                s.completions.len()
            );
            break;
        }
    }
    assert_eq!(pool.cells()[victim].role, CellRole::Train);
    println!("== phase 6 demo complete ==");
}
