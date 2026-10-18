//! The Phase 6 benchmark: fixed-pool **Async** vs **Colocate** vs
//! **Elastic**, on a simulated workload with calibrated per-model-call
//! and per-sample training latencies.
//!
//! Honest framing: these are **simulation** numbers — logical clocks
//! on the same scheduler core, not wall-clock GPU measurements. The
//! claim the benchmark supports is structural: elasticity removes the
//! serialization and starvation phases that the two classical layouts
//! suffer, under the same calibrated service times. QWENGYRE reports
//! 1.38–1.78× over Async and 1.21–1.85× over Colocate on real
//! hardware; this simulation typically lands in the same regime but
//! makes no hardware claim.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::policy::{PolicyStore, PolicyVersion};
use crate::scheduler::{CellRole, ElasticPool, SchedulerConfig};

/// Workload shape.
#[derive(Debug, Clone)]
pub struct Workload {
    /// Task groups.
    pub groups: usize,
    /// Rollouts per group (the GRPO group size).
    pub group_size: usize,
    /// Model calls per rollout.
    pub gens_per_rollout: usize,
    /// Burst spacing: group `g` is submitted at `g * burst_ms`
    /// (bursty arrivals are the realistic RL shape — the framework
    /// dispatches groups as data construction completes).
    pub burst_ms: u64,
}

impl Default for Workload {
    fn default() -> Self {
        Self {
            groups: 32,
            group_size: 8,
            gens_per_rollout: 12,
            burst_ms: 120,
        }
    }
}

/// Feeds groups to the pool as their arrival times pass the pool
/// clock; returns the number of rollouts submitted so far.
struct ArrivalFeed {
    next_group: usize,
    submitted: usize,
}

impl ArrivalFeed {
    fn new() -> Self {
        Self {
            next_group: 0,
            submitted: 0,
        }
    }

    fn pump(&mut self, pool: &mut ElasticPool, w: &Workload) {
        let clock = pool.clock_ms();
        while self.next_group < w.groups && clock >= self.next_group as u64 * w.burst_ms {
            let base = (self.next_group * w.group_size) as u64;
            for i in 0..w.group_size as u64 {
                pool.submit_rollout(base + i, w.gens_per_rollout);
                self.submitted += 1;
            }
            self.next_group += 1;
        }
    }

    fn done(&self, w: &Workload) -> bool {
        self.submitted >= (w.groups * w.group_size)
    }
}

/// One strategy's measured result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyResult {
    /// Strategy name.
    pub strategy: &'static str,
    /// Simulated makespan, ms.
    pub makespan_ms: u64,
    /// Model calls served.
    pub generations_served: u64,
    /// Batches completed.
    pub batches_completed: u64,
    /// Average rollout cells busy (utilization proxy).
    pub rollout_utilization: f64,
    /// Average train cells busy.
    pub train_utilization: f64,
}

impl StrategyResult {
    /// Speedup of `self` over `baseline`.
    pub fn speedup_over(&self, baseline: &StrategyResult) -> f64 {
        baseline.makespan_ms as f64 / self.makespan_ms.max(1) as f64
    }
}

/// Runs all three strategies on the same workload and store seed.
pub fn run_comparison(workload: Workload, cfg: SchedulerConfig) -> Result<Vec<StrategyResult>> {
    let v0 = PolicyVersion {
        major: 0,
        digest: "bench-seed".into(),
    };
    let async_r = run_async(&workload, &cfg, v0.clone())?;
    let colocate_r = run_colocate(&workload, &cfg, v0.clone())?;
    let elastic_r = run_elastic(&workload, &cfg, v0)?;
    Ok(vec![async_r, colocate_r, elastic_r])
}

/// Fixed-pool Async: a static rollout/train split, both sides always
/// on; rollouts stream samples into the batch as they finish.
fn run_async(w: &Workload, cfg: &SchedulerConfig, v0: PolicyVersion) -> Result<StrategyResult> {
    let store = PolicyStore::new(v0, vec![]);
    let mut pool = ElasticPool::new(
        SchedulerConfig {
            initial_rollout: cfg.initial_rollout,
            ..cfg.clone()
        },
        store,
    );
    pool.set_auto(false); // fixed split: no waterlevel-driven moves
                          // static split: never request transitions
    let (ms, gens, batches, rollout_util, train_util) = drain_to_completion(&mut pool, w)?;
    Ok(StrategyResult {
        strategy: "Async (fixed split)",
        makespan_ms: ms,
        generations_served: gens,
        batches_completed: batches,
        rollout_utilization: rollout_util,
        train_utilization: train_util,
    })
}

/// Colocate: one shared pool; ALL cells roll out, then ALL cells train,
/// phase by phase (the classic colocated loop).
fn run_colocate(w: &Workload, cfg: &SchedulerConfig, v0: PolicyVersion) -> Result<StrategyResult> {
    let store = PolicyStore::new(v0, vec![]);
    let mut pool = ElasticPool::new(
        SchedulerConfig {
            initial_rollout: cfg.cells, // everything rolls out first
            ..cfg.clone()
        },
        store,
    );
    pool.set_auto(false); // phase-serialized: no moves during rollout
    let (mut ms, mut gens, mut batches, _rollout_util, _train_util) =
        (0u64, 0u64, 0u64, 0.0f64, 0.0f64);
    // phase 1: rollout everything
    let (ms1, gens1, _b1, ru, _tu) = drain_rollout_only(&mut pool, w)?;
    ms += ms1;
    gens += gens1;
    let rollout_util = ru;
    // phase 2: flip every cell to train and drain the batches
    for i in 0..pool.cells().len() {
        let _ = pool.request_transition(i, CellRole::Train);
    }
    let (ms2, _g2, b2, _ru2, tu) = drain_train_only(&mut pool, w)?;
    ms += ms2;
    batches += b2;
    let train_util = tu;
    Ok(StrategyResult {
        strategy: "Colocate (phase-serialized)",
        makespan_ms: ms,
        generations_served: gens,
        batches_completed: batches,
        rollout_utilization: rollout_util,
        train_utilization: train_util,
    })
}

/// Elastic: the full scheduler — waterlevel-driven transitions with
/// safe rerouting and streaming batches.
fn run_elastic(w: &Workload, cfg: &SchedulerConfig, v0: PolicyVersion) -> Result<StrategyResult> {
    let store = PolicyStore::new(v0, vec![]);
    let mut pool = ElasticPool::new(cfg.clone(), store);
    let (ms, gens, batches, rollout_util, train_util) = drain_to_completion(&mut pool, w)?;
    Ok(StrategyResult {
        strategy: "Elastic (this crate)",
        makespan_ms: ms,
        generations_served: gens,
        batches_completed: batches,
        rollout_utilization: rollout_util,
        train_utilization: train_util,
    })
}

/// Runs ticks until all rollouts are served AND all possible batches
/// completed; returns (elapsed, generations, batches, utilizations).
fn drain_to_completion(pool: &mut ElasticPool, w: &Workload) -> Result<(u64, u64, u64, f64, f64)> {
    let total_gens = (w.groups * w.group_size * w.gens_per_rollout) as u64;
    let target_batches = (w.groups * w.group_size / w.group_size.max(1)) as u64;
    let mut feed = ArrivalFeed::new();
    let mut gens = 0u64;
    let mut rollout_busy_sum = 0usize;
    let mut train_busy_sum = 0usize;
    let mut ticks = 0usize;
    loop {
        feed.pump(pool, w);
        let summary = pool.tick();
        gens += summary
            .completions
            .iter()
            .filter(|(_, k)| *k == "generation")
            .count() as u64;
        rollout_busy_sum += pool
            .cells()
            .iter()
            .filter(|c| {
                c.role == CellRole::Rollout && (!c.in_flight.is_empty() || !c.queue.is_empty())
            })
            .count();
        train_busy_sum += pool
            .cells()
            .iter()
            .filter(|c| c.role == CellRole::Train && c.train_remaining > 0)
            .count();
        ticks += 1;
        if gens >= total_gens && pool.batches_completed() as u64 >= target_batches && feed.done(w) {
            break;
        }
        if ticks > 200_000 {
            return Err(crate::error::Error::Config(
                "simulation did not converge".into(),
            ));
        }
    }
    let rollout_util = rollout_busy_sum as f64 / (ticks * pool.cells().len().max(1)) as f64;
    let train_util = train_busy_sum as f64 / (ticks * pool.cells().len().max(1)) as f64;
    Ok((
        pool.clock_ms(),
        gens,
        pool.batches_completed() as u64,
        rollout_util,
        train_util,
    ))
}

fn drain_rollout_only(pool: &mut ElasticPool, w: &Workload) -> Result<(u64, u64, u64, f64, f64)> {
    let total_gens = (w.groups * w.group_size * w.gens_per_rollout) as u64;
    let mut feed = ArrivalFeed::new();
    let mut gens = 0u64;
    let mut rollout_busy_sum = 0usize;
    let mut ticks = 0usize;
    loop {
        feed.pump(pool, w);
        let summary = pool.tick();
        gens += summary
            .completions
            .iter()
            .filter(|(_, k)| *k == "generation")
            .count() as u64;
        rollout_busy_sum += pool
            .cells()
            .iter()
            .filter(|c| {
                c.role == CellRole::Rollout && (!c.in_flight.is_empty() || !c.queue.is_empty())
            })
            .count();
        ticks += 1;
        if gens >= total_gens && feed.done(w) {
            break;
        }
        if ticks > 200_000 {
            return Err(crate::error::Error::Config("rollout phase stalled".into()));
        }
    }
    let rollout_util = rollout_busy_sum as f64 / (ticks * pool.cells().len().max(1)) as f64;
    Ok((pool.clock_ms(), gens, 0, rollout_util, 0.0))
}

fn drain_train_only(pool: &mut ElasticPool, w: &Workload) -> Result<(u64, u64, u64, f64, f64)> {
    let total_samples = (w.groups * w.group_size) as u64;
    let target = (total_samples / pool.cfg_waterlevel_batch() as u64).max(1);
    let mut batches = 0u64;
    let mut train_busy_sum = 0usize;
    let mut ticks = 0usize;
    loop {
        let summary = pool.tick();
        batches += summary.batches_completed as u64;
        train_busy_sum += pool
            .cells()
            .iter()
            .filter(|c| c.role == CellRole::Train && c.train_remaining > 0)
            .count();
        ticks += 1;
        if batches >= target && summary.batches_completed == 0 && pool.train_idle() {
            break;
        }
        if ticks > 200_000 {
            return Err(crate::error::Error::Config("train phase stalled".into()));
        }
    }
    let train_util = train_busy_sum as f64 / (ticks * pool.cells().len().max(1)) as f64;
    Ok((pool.clock_ms(), 0, batches, 0.0, train_util))
}
