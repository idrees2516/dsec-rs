//! Resource governance: node capacity accounting, per-sandbox CPU class
//! (SCHED_IDLE simulation) and memory limits with pause-time reclaim.
//!
//! The paper frees memory by combining the kernel balloon (idle page
//! reporting via DAMON + virtio-balloon) with `memory.reclaim` writes;
//! the userspace model returns a configured fraction of a paused
//! sandbox's memory to the node pool and refetches "cold" pages on
//! resume (`MADV_WILLNEED` prefetch in the real system).

use std::sync::atomic::{AtomicI64, Ordering};

use serde::{Deserialize, Serialize};

/// CPU scheduling class for a sandbox (paper: SCHED_IDLE for agent
/// sandboxes so training workers keep the foreground).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuClass {
    Normal,
    Idle,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceRequest {
    pub cpu_millicores: i64,
    pub mem_mib: i64,
}

impl Default for ResourceRequest {
    fn default() -> Self {
        ResourceRequest {
            cpu_millicores: 500,
            mem_mib: 256,
        }
    }
}

/// Snapshot of node-level resource usage.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NodeUsage {
    pub cpu_millicores: i64,
    pub mem_mib: i64,
    pub sandboxes: i64,
    pub paused_sandboxes: i64,
}

/// Node capacity with atomic usage counters.
#[derive(Debug)]
pub struct ResourcePool {
    pub capacity: ResourceRequest,
    pub max_sandboxes: i64,
    cpu_used: AtomicI64,
    mem_used: AtomicI64,
    sandboxes: AtomicI64,
    paused: AtomicI64,
    /// Fraction of a paused sandbox's memory returned to the pool.
    pub reclaim_fraction: f64,
}

impl ResourcePool {
    pub fn new(cpu_millicores: i64, mem_mib: i64, max_sandboxes: i64) -> Self {
        ResourcePool {
            capacity: ResourceRequest {
                cpu_millicores,
                mem_mib,
            },
            max_sandboxes,
            cpu_used: AtomicI64::new(0),
            mem_used: AtomicI64::new(0),
            sandboxes: AtomicI64::new(0),
            paused: AtomicI64::new(0),
            reclaim_fraction: 0.6,
        }
    }

    pub fn usage(&self) -> NodeUsage {
        NodeUsage {
            cpu_millicores: self.cpu_used.load(Ordering::Relaxed),
            mem_mib: self.mem_used.load(Ordering::Relaxed),
            sandboxes: self.sandboxes.load(Ordering::Relaxed),
            paused_sandboxes: self.paused.load(Ordering::Relaxed),
        }
    }

    pub fn available(&self) -> ResourceRequest {
        let u = self.usage();
        ResourceRequest {
            cpu_millicores: self.capacity.cpu_millicores - u.cpu_millicores,
            mem_mib: self.capacity.mem_mib - u.mem_mib,
        }
    }

    pub fn sandbox_slots_free(&self) -> i64 {
        self.max_sandboxes - self.usage().sandboxes
    }

    /// Admission-time reservation (fail-fast, no waiting queue: batch
    /// sandbox creation must reject rather than block, as in the paper).
    ///
    /// Lock-free optimistic admission: reserve first with `fetch_add`
    /// (every caller receives a unique "old" counter value, so at most
    /// `max_sandboxes` slot admissions can ever succeed), verify the
    /// capacity invariants, and roll the reservations back on failure.
    /// This closes the read-check-add window that allowed transient
    /// oversubscription under full contention.
    pub fn try_acquire(&self, req: &ResourceRequest) -> Result<(), (i64, i64, i64)> {
        let slots_old = self.sandboxes.fetch_add(1, Ordering::Relaxed);
        if slots_old >= self.max_sandboxes {
            self.sandboxes.fetch_sub(1, Ordering::Relaxed);
            return Err(self.deficit());
        }
        let cpu_old = self
            .cpu_used
            .fetch_add(req.cpu_millicores, Ordering::Relaxed);
        if cpu_old + req.cpu_millicores > self.capacity.cpu_millicores {
            self.cpu_used
                .fetch_sub(req.cpu_millicores, Ordering::Relaxed);
            self.sandboxes.fetch_sub(1, Ordering::Relaxed);
            return Err(self.deficit());
        }
        let mem_old = self.mem_used.fetch_add(req.mem_mib, Ordering::Relaxed);
        if mem_old + req.mem_mib > self.capacity.mem_mib {
            self.mem_used.fetch_sub(req.mem_mib, Ordering::Relaxed);
            self.cpu_used
                .fetch_sub(req.cpu_millicores, Ordering::Relaxed);
            self.sandboxes.fetch_sub(1, Ordering::Relaxed);
            return Err(self.deficit());
        }
        Ok(())
    }

    /// Current headroom snapshot used in admission error reports.
    fn deficit(&self) -> (i64, i64, i64) {
        let u = self.usage();
        (
            self.capacity.cpu_millicores - u.cpu_millicores,
            self.capacity.mem_mib - u.mem_mib,
            self.max_sandboxes - u.sandboxes,
        )
    }

    /// Full release at destroy.
    pub fn release(&self, req: &ResourceRequest) {
        self.cpu_used
            .fetch_sub(req.cpu_millicores, Ordering::Relaxed);
        self.mem_used.fetch_sub(req.mem_mib, Ordering::Relaxed);
        self.sandboxes.fetch_sub(1, Ordering::Relaxed);
    }

    /// Pause: returns reclaimed MiB to the pool (balloon + reclaim model).
    pub fn pause_reclaim(&self, req: &ResourceRequest) -> i64 {
        self.paused.fetch_add(1, Ordering::Relaxed);
        let reclaimed = (req.mem_mib as f64 * self.reclaim_fraction).round() as i64;
        self.mem_used.fetch_sub(reclaimed, Ordering::Relaxed);
        reclaimed
    }

    /// Resume: refetch the reclaimed pages (charged back).
    pub fn resume_refetch(&self, _req: &ResourceRequest, reclaimed_mib: i64) {
        self.paused.fetch_sub(1, Ordering::Relaxed);
        self.mem_used.fetch_add(reclaimed_mib, Ordering::Relaxed);
    }

    /// Pause without reclaim accounting (state-only).
    pub fn unpause(&self) {
        self.paused.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Per-sandbox governance state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SandboxGovernor {
    pub request: ResourceRequest,
    pub cpu_class: CpuClass,
    /// AppArmor/eBPF policy profile (paper: malicious behavior mitigation).
    pub policy_profile: String,
}

impl Default for SandboxGovernor {
    fn default() -> Self {
        SandboxGovernor {
            request: ResourceRequest::default(),
            cpu_class: CpuClass::Idle,
            policy_profile: "dsec-agent-default".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_and_release() {
        let pool = ResourcePool::new(4000, 4096, 100);
        let req = ResourceRequest {
            cpu_millicores: 500,
            mem_mib: 256,
        };
        assert!(pool.try_acquire(&req).is_ok());
        assert_eq!(pool.usage().sandboxes, 1);
        assert_eq!(pool.usage().mem_mib, 256);
        pool.release(&req);
        assert_eq!(pool.usage().cpu_millicores, 0);
        assert_eq!(pool.usage().sandboxes, 0);
    }

    #[test]
    fn rejects_over_capacity() {
        let pool = ResourcePool::new(1000, 1024, 2);
        let req = ResourceRequest {
            cpu_millicores: 600,
            mem_mib: 600,
        };
        assert!(pool.try_acquire(&req).is_ok());
        // 600 + 600 = 1200 millicores > 1000 capacity: rejected.
        assert!(pool.try_acquire(&req).is_err());
        // Slot limit: two tiny sandboxes fill the remaining slot.
        let tiny = ResourceRequest {
            cpu_millicores: 1,
            mem_mib: 1,
        };
        assert!(pool.try_acquire(&tiny).is_ok());
        assert!(pool.try_acquire(&tiny).is_err());
    }

    #[test]
    fn pause_reclaim_and_resume() {
        let pool = ResourcePool::new(4000, 4096, 10);
        let req = ResourceRequest {
            cpu_millicores: 500,
            mem_mib: 1000,
        };
        pool.try_acquire(&req).unwrap();
        let reclaimed = pool.pause_reclaim(&req);
        assert_eq!(reclaimed, 600); // 0.6 fraction
        assert_eq!(pool.usage().mem_mib, 400);
        assert_eq!(pool.usage().paused_sandboxes, 1);
        pool.resume_refetch(&req, reclaimed);
        assert_eq!(pool.usage().mem_mib, 1000);
        assert_eq!(pool.usage().paused_sandboxes, 0);
    }

    #[test]
    fn concurrent_admission_never_oversubscribes() {
        // cpu/mem large enough that the slot cap binds first.
        let pool = std::sync::Arc::new(ResourcePool::new(10_000_000, 10_000_000, 5000));
        let req = ResourceRequest {
            cpu_millicores: 100,
            mem_mib: 100,
        };
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let p = pool.clone();
                let r = req;
                std::thread::spawn(move || (0..1000).filter(|_| p.try_acquire(&r).is_ok()).count())
            })
            .collect();
        let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(total, 5000); // exactly the slot cap
        assert_eq!(pool.usage().sandboxes, 5000);
    }
}
