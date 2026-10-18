//! The elastic GPU scheduler (Phase 6): cell roles, safe rerouting,
//! rollout waterlevel, training-batch streaming, publication barriers.
//!
//! QWENGYRE's elasticity, as operationalized here:
//!
//! - the GPU pool is partitioned into **cells**, each holding a role —
//!   `Rollout` (serving model calls for in-flight harness executions)
//!   or `Train` (data-parallel training shards);
//! - cells change roles only through a **draining transition**: model
//!   calls already in flight on the cell run to completion (QWENGYRE:
//!   "ensuring that model calls from ongoing harness executions
//!   continue to be served"), pending work is **rerouted** to
//!   remaining rollout cells, and only then does the role flip;
//! - the **waterlevel** tracks unfinished rollout work (pending +
//!   in-flight) against ready training samples: below the low mark the
//!   pool shifts cells to rollout; above the high mark (a full batch
//!   is ready) it shifts cells to train;
//! - **training-batch streaming**: a batch is split across train
//!   cells by progress, and a cell that joins mid-batch picks up the
//!   unprocessed slices ("newly available nodes can join an ongoing
//!   batch");
//! - the **publication barrier** gates role flips: no transition while
//!   a policy publication is in flight.
//!
//! Timing is simulated with cell-local logical clocks (millisecond
//! ticks) — the benchmark harness (`bench`) reports wall-clock-shaped
//! comparisons, honestly labeled as simulation.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::policy::PolicyStore;

/// A cell's role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellRole {
    /// Serving model calls for rollouts.
    Rollout,
    /// Data-parallel training shard.
    Train,
}

/// One model call (queued or in flight) on a rollout cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenJob {
    /// Rollout (task) id.
    pub rollout_id: u64,
    /// Service duration, simulated ms (long-tailed: drawn per job).
    pub duration_ms: u64,
    /// Remaining service time, simulated ms.
    pub remaining_ms: u64,
}

/// A cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    /// Cell id.
    pub id: usize,
    /// Current role.
    pub role: CellRole,
    /// Transition state: draining before a role flip.
    pub draining: bool,
    /// Target role of an in-progress transition.
    pub target_role: Option<CellRole>,
    /// In-flight model calls (rollout role).
    pub in_flight: Vec<GenJob>,
    /// Queue of model calls not yet started.
    pub queue: VecDeque<GenJob>,
    /// Remaining training work, sample-slices.
    pub train_remaining: usize,
}

impl Cell {
    /// Whether the cell is still serving (not draining or idle-drained).
    pub fn is_serving(&self) -> bool {
        !self.draining || !self.in_flight.is_empty()
    }
}

/// The waterlevel tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Waterlevel {
    /// Rollouts waiting to start.
    pub pending: usize,
    /// Rollouts with model calls in flight.
    pub in_flight: usize,
    /// Completed samples ready for training.
    pub ready: usize,
    /// Below this pending level → shift cells to rollout.
    pub low: usize,
    /// Above this ready level → a batch is starving the trainer: shift
    /// cells to train.
    pub high: usize,
    /// Samples per training batch.
    pub batch_size: usize,
}

impl Default for Waterlevel {
    fn default() -> Self {
        Self {
            pending: 0,
            in_flight: 0,
            ready: 0,
            low: 2,
            high: 8,
            batch_size: 8,
        }
    }
}

/// One simulated scheduler tick's summary.
#[derive(Debug, Clone, Default)]
pub struct TickSummary {
    /// Simulated milliseconds elapsed this tick.
    pub elapsed_ms: u64,
    /// Completions this tick (rollout id, kind).
    pub completions: Vec<(u64, &'static str)>,
    /// Role transitions completed.
    pub transitions: Vec<(usize, CellRole)>,
    /// Transitions blocked by the publication barrier.
    pub barrier_blocked: usize,
    /// Samples streamed to train cells this tick.
    pub streamed: usize,
    /// Batches completed this tick.
    pub batches_completed: usize,
}

/// Scheduler configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Total cells.
    pub cells: usize,
    /// Initial rollout cells (the rest start as train).
    pub initial_rollout: usize,
    /// Waterlevel marks.
    pub waterlevel: Waterlevel,
    /// Mean model-call service time, ms (simulated).
    pub gen_ms: u64,
    /// Training time per sample slice, ms (simulated).
    pub train_slice_ms: u64,
    /// Concurrent model calls per rollout cell (bounded serving
    /// capacity — the GPU's batch window).
    pub concurrency_per_cell: usize,
    /// Long-tail multiplier: job durations are drawn in
    /// `[gen_ms/2, gen_ms/2 + gen_ms*tail]` (agent rollouts have
    /// long-tailed durations — QWENGYRE's core motivation).
    pub tail: u64,
    /// RNG seed for duration draws (deterministic runs).
    pub seed: u64,
    /// Rebalance interval: waterlevel-driven transitions are considered
    /// at most once per this many ticks (hysteresis — constant draining
    /// would cost more than it gains).
    pub rebalance_ticks: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            cells: 8,
            initial_rollout: 5,
            waterlevel: Waterlevel {
                pending: 0,
                in_flight: 0,
                ready: 0,
                low: 2,
                high: 8,
                batch_size: 8,
            },
            gen_ms: 40,
            train_slice_ms: 12,
            concurrency_per_cell: 4,
            tail: 3,
            seed: 42,
            rebalance_ticks: 30,
        }
    }
}

/// The elastic pool.
pub struct ElasticPool {
    cells: Vec<Cell>,
    cfg: SchedulerConfig,
    store: PolicyStore,
    /// FIFO of pending rollouts (their remaining generation count).
    pending: VecDeque<u64>,
    /// Samples ready for training.
    ready: Vec<serde_json::Value>,
    /// Remaining model calls per rollout (sample = completed rollout).
    rollout_gens: std::collections::BTreeMap<u64, usize>,
    /// In-progress training batch: slices not yet handed to a cell.
    batch_remaining: usize,
    /// Slices handed out but not yet trained.
    batch_outstanding: usize,
    batches_done: usize,
    clock_ms: u64,
    /// Deterministic duration draws.
    rng: dsec_protocol::rng::Rng,
    /// Whether tick() applies waterlevel-driven transitions (the
    /// elastic strategies enable it; the fixed baselines disable it).
    auto: bool,
    /// Ticks until the next rebalance consideration.
    rebalance_in: u64,
}

impl ElasticPool {
    /// New pool over a policy store (the publication barrier gates its
    /// transitions).
    pub fn new(cfg: SchedulerConfig, store: PolicyStore) -> Self {
        let cells = (0..cfg.cells)
            .map(|i| Cell {
                id: i,
                role: if i < cfg.initial_rollout {
                    CellRole::Rollout
                } else {
                    CellRole::Train
                },
                draining: false,
                target_role: None,
                in_flight: vec![],
                queue: VecDeque::new(),
                train_remaining: 0,
            })
            .collect();
        let seed = cfg.seed;
        Self {
            cells,
            cfg,
            store,
            pending: VecDeque::new(),
            ready: Vec::new(),
            rollout_gens: std::collections::BTreeMap::new(),
            batch_remaining: 0,
            batch_outstanding: 0,
            batches_done: 0,
            clock_ms: 0,
            rng: dsec_protocol::rng::Rng::new(seed),
            auto: true,
            rebalance_in: 0,
        }
    }

    /// Current clock (simulated ms since start).
    pub fn clock_ms(&self) -> u64 {
        self.clock_ms
    }

    /// Read-only cell view.
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    /// Batches completed so far.
    pub fn batches_completed(&self) -> usize {
        self.batches_done
    }

    /// Enables / disables waterlevel-driven automatic transitions.
    /// The fixed-layout baselines (Async, Colocate) run with auto off;
    /// the elastic strategy runs with it on.
    pub fn set_auto(&mut self, auto: bool) {
        self.auto = auto;
    }

    /// Queues a rollout: `id` with `gens` model calls to serve. A
    /// **sample** becomes ready only when the rollout's last model call
    /// finishes (RL samples are completed rollouts, not generations).
    pub fn submit_rollout(&mut self, id: u64, gens: usize) {
        for _ in 0..gens {
            self.pending.push_back(id);
        }
        self.rollout_gens.insert(id, gens);
        self.cfg.waterlevel.pending = self.pending.len();
    }

    /// Requests a role transition for one cell — the safe path:
    ///
    /// 1. refused while the publication barrier is closed;
    /// 2. `Rollout → Train` sets `draining`: in-flight calls finish,
    ///    queued calls are **rerouted** to other rollout cells, and the
    ///    flip completes when the last in-flight call does;
    /// 3. `Train → Rollout` completes only when the cell has no
    ///    remaining training slices (it finishes its current work).
    pub fn request_transition(&mut self, cell_id: usize, to: CellRole) -> Result<()> {
        if self.store.barrier_closed() {
            return Err(Error::Transition(
                "publication barrier closed — transitions deferred".into(),
            ));
        }
        let cell = self
            .cells
            .get_mut(cell_id)
            .ok_or_else(|| Error::Config(format!("no cell {cell_id}")))?;
        if cell.role == to {
            return Ok(());
        }
        match (cell.role, to) {
            (CellRole::Rollout, CellRole::Rollout) | (CellRole::Train, CellRole::Train) => Ok(()),
            (CellRole::Rollout, CellRole::Train) => {
                cell.draining = true;
                cell.target_role = Some(CellRole::Train);
                // safe rerouting: move queued work to other rollout cells
                let rerouted: Vec<GenJob> = cell.queue.drain(..).collect();
                for job in rerouted {
                    if let Some(target) = self
                        .cells
                        .iter_mut()
                        .find(|c| c.role == CellRole::Rollout && !c.draining && c.id != cell_id)
                    {
                        target.queue.push_back(job);
                    } else {
                        // no other rollout cell: park back (the caller
                        // must not drain the last rollout cell)
                        self.pending.push_front(job.rollout_id);
                        // refund the rollout's gen bookkeeping? the job
                        // is parked, not completed — bookkeeping counts
                        // completions, so nothing to refund here
                    }
                }
                Ok(())
            }
            (CellRole::Train, CellRole::Rollout) => {
                cell.draining = true;
                cell.target_role = Some(CellRole::Rollout);
                // completes when train_remaining hits 0 (in tick)
                Ok(())
            }
        }
    }

    /// Advances the simulation by one scheduling quantum.
    ///
    /// Order: finish in-flight generations → dispatch queued work →
    /// stream ready samples to train cells (progress-based, joinable
    /// mid-batch) → complete draining transitions → refill a batch →
    /// waterlevel-driven transition requests.
    pub fn tick(&mut self) -> TickSummary {
        let mut summary = TickSummary {
            elapsed_ms: self.quantum(),
            ..Default::default()
        };
        self.clock_ms += self.quantum();

        // 1. progress in-flight generations
        let quantum = self.quantum();
        let mut finished: Vec<u64> = Vec::new();
        for cell in self.cells.iter_mut() {
            let step = quantum;
            let mut done = Vec::new();
            for (i, job) in cell.in_flight.iter_mut().enumerate() {
                job.remaining_ms = job.remaining_ms.saturating_sub(step);
                if job.remaining_ms == 0 {
                    done.push(i);
                }
            }
            for i in done.into_iter().rev() {
                let job = cell.in_flight.swap_remove(i);
                finished.push(job.rollout_id);
            }
        }
        for id in finished {
            summary.completions.push((id, "generation"));
            // a sample becomes ready only when the ROLLOUT completes
            // (its last model call finished)
            let remaining = self
                .rollout_gens
                .get_mut(&id)
                .map(|g| {
                    *g -= 1;
                    *g
                })
                .unwrap_or(0);
            if remaining == 0 {
                self.ready.push(serde_json::json!({"rollout": id}));
                summary.completions.push((id, "rollout"));
                self.rollout_gens.remove(&id);
            }
        }

        // 2. dispatch queued work to rollout cells up to the cell's
        //    bounded concurrency (never to draining ones); each job's
        //    duration is a long-tailed draw
        let gen_ms = self.cfg.gen_ms;
        let tail = self.cfg.tail;
        let capacity = self.cfg.concurrency_per_cell.max(1);
        let mut draws = Vec::new();
        for _ in 0..self.cells.iter().map(|c| c.queue.len()).sum::<usize>() {
            let base = gen_ms / 2;
            let span = gen_ms.saturating_mul(tail);
            draws.push(base + self.rng.below(span.max(1) as usize) as u64);
        }
        let mut next_draw = 0usize;
        for cell in self.cells.iter_mut() {
            if cell.role == CellRole::Rollout && !cell.draining {
                while !cell.queue.is_empty() && cell.in_flight.len() < capacity {
                    let job = cell.queue.pop_front().unwrap();
                    let duration = draws.get(next_draw).copied().unwrap_or(gen_ms);
                    next_draw += 1;
                    cell.in_flight.push(GenJob {
                        rollout_id: job.rollout_id,
                        duration_ms: duration,
                        remaining_ms: duration,
                    });
                }
            }
        }

        // 3. training-batch streaming: start a batch when the high
        //    waterlevel is reached, then hand slices to train cells —
        //    including cells that JOIN mid-batch (QWENGYRE streaming)
        if self.batch_remaining == 0
            && self.batch_outstanding == 0
            && self.ready.len() >= self.cfg.waterlevel.batch_size
        {
            self.batch_remaining = self.cfg.waterlevel.batch_size;
            self.ready.drain(..self.cfg.waterlevel.batch_size);
        }
        if self.batch_remaining > 0 {
            for cell in self.cells.iter_mut() {
                if cell.role == CellRole::Train
                    && cell.train_remaining == 0
                    && self.batch_remaining > 0
                {
                    cell.train_remaining += 1;
                    self.batch_remaining -= 1;
                    self.batch_outstanding += 1;
                    summary.streamed += 1;
                }
            }
        }

        // 4. train slices progress; a batch completes when every slice
        //    has been trained (none outstanding, none unassigned)
        let mut trained_slices = 0usize;
        for cell in self.cells.iter_mut() {
            if cell.role == CellRole::Train && cell.train_remaining > 0 {
                cell.train_remaining -= 1;
                trained_slices += 1;
            }
        }
        self.batch_outstanding = self.batch_outstanding.saturating_sub(trained_slices);
        if trained_slices > 0 && self.batch_remaining == 0 && self.batch_outstanding == 0 {
            self.batches_done += 1;
            summary.batches_completed += 1;
        }

        // 5. complete draining transitions
        for cell in self.cells.iter_mut() {
            if cell.draining {
                let can_flip = match cell.target_role {
                    Some(CellRole::Train) => cell.in_flight.is_empty(),
                    Some(CellRole::Rollout) => cell.train_remaining == 0,
                    None => false,
                };
                if can_flip {
                    if let Some(target) = cell.target_role.take() {
                        cell.role = target;
                        cell.draining = false;
                        summary.transitions.push((cell.id, target));
                    }
                }
            }
        }

        // 6. waterlevel-driven elasticity (auto mode, cadence-gated:
        //    transitions are considered once per rebalance_ticks —
        //    hysteresis, so cells are not perpetually draining)
        let mut rebalance_due = false;
        if self.auto {
            if self.rebalance_in == 0 {
                rebalance_due = true;
                self.rebalance_in = self.cfg.rebalance_ticks.max(1);
            } else {
                self.rebalance_in -= 1;
            }
        }
        let in_flight: usize = self
            .cells
            .iter()
            .map(|c| c.in_flight.len() + c.queue.len())
            .sum();
        let pending = self.pending.len();
        let ready = self.ready.len();
        self.cfg.waterlevel.in_flight = in_flight;
        self.cfg.waterlevel.pending = pending;
        self.cfg.waterlevel.ready = ready;
        // 6a. too much unfinished rollout work AND no batch ready →
        //     train cells to rollout (mutually exclusive with 6b: no
        //     thrashing)
        if rebalance_due
            && pending + in_flight > self.cfg.waterlevel.high
            && ready < self.cfg.waterlevel.batch_size
        {
            if let Some(idx) = self
                .cells
                .iter()
                .position(|c| c.role == CellRole::Train && !c.draining && c.train_remaining == 0)
            {
                if self.store.barrier_closed() {
                    summary.barrier_blocked += 1;
                } else if self.request_transition(idx, CellRole::Rollout).is_ok() {
                    summary.transitions.push((idx, CellRole::Rollout));
                }
            }
        }
        // 6b. a batch is ready and trainer is starved → rollout to train
        if rebalance_due
            && ready >= self.cfg.waterlevel.batch_size
            && !self
                .cells
                .iter()
                .any(|c| c.role == CellRole::Train && c.train_remaining > 0)
        {
            // only if we still have rollout capacity to spare (>= 2
            // non-draining rollout cells)
            let rollout_live = self
                .cells
                .iter()
                .filter(|c| c.role == CellRole::Rollout && !c.draining)
                .count();
            if rollout_live >= 2 {
                if let Some(idx) = self
                    .cells
                    .iter()
                    .position(|c| c.role == CellRole::Rollout && !c.draining)
                {
                    if self.store.barrier_closed() {
                        summary.barrier_blocked += 1;
                    } else if self.request_transition(idx, CellRole::Train).is_ok() {
                        summary.transitions.push((idx, CellRole::Train));
                    }
                }
            }
        }

        // 7. feed pending work into rollout queues — least-loaded cell
        //    first (a naive `find` would pile everything onto cell 0)
        while !self.pending.is_empty() {
            let id = self.pending.front().copied().unwrap();
            let gen_ms = self.cfg.gen_ms;
            let target = self
                .cells
                .iter_mut()
                .filter(|c| c.role == CellRole::Rollout && !c.draining)
                .min_by_key(|c| c.queue.len() + c.in_flight.len());
            match target {
                Some(cell) => {
                    cell.queue.push_back(GenJob {
                        rollout_id: id,
                        duration_ms: gen_ms,
                        remaining_ms: 0,
                    });
                }
                None => break,
            }
            self.pending.pop_front();
        }

        summary
    }

    fn quantum(&self) -> u64 {
        10
    }

    /// The configured batch size.
    pub fn cfg_waterlevel_batch(&self) -> usize {
        self.cfg.waterlevel.batch_size
    }

    /// Whether every train cell is idle (no outstanding or unassigned
    /// slices).
    pub fn train_idle(&self) -> bool {
        self.cells
            .iter()
            .filter(|c| c.role == CellRole::Train)
            .all(|c| c.train_remaining == 0)
            && self.batch_remaining == 0
            && self.batch_outstanding == 0
    }
}
