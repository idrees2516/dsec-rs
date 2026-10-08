//! The memory/process/file watchdog: weighing a uid's real footprint and
//! reaping when it exceeds the budget.
//!
//! Ports of upstream `karotte/memory_watch.py`. The weigher counts what
//! the kernel charges the uid: process memory (RSS, refined to PSS on the
//! precise pass), RAM-backed scratch files in the temp dirs (`st_blocks`
//! of owned files), and SysV shared-memory segments created by the uid
//! (charged to the *creator* — `shmctl(IPC_SET)` lets the creator rewrite
//! the owner). On violation: re-weigh precisely, then `kill_processes` —
//! and the watch *keeps running*, so tmpfs files and SysV segments that
//! survive SIGKILL keep reaping whatever respawns.

use crate::confinement::kill_processes;
use crate::confinement::{CohortBackend, KillPass};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Upstream `TEMP_DIRS` — RAM-backed scratch the student can fill.
pub const TEMP_DIRS: &[&str] = &["/tmp", "/var/tmp", "/dev/shm"];
/// Upstream `POLL_INTERVAL_S` (memory/process watch).
pub const POLL_INTERVAL_S: f64 = 0.1;
/// Upstream `FILE_POLL_INTERVAL_S`.
pub const FILE_POLL_INTERVAL_S: f64 = 1.0;
/// Upstream `MAX_UNCAPPED_PROCESSES`.
pub const MAX_UNCAPPED_PROCESSES: usize = 10_000;
/// Upstream `MAX_TMPFS_ENTRIES`.
pub const MAX_TMPFS_ENTRIES: usize = 10_000;
/// Upstream `MAX_UNCAPPED_FILES`.
pub const MAX_UNCAPPED_FILES: usize = 1_000_000;
/// Upstream `_VISITS_PER_BUDGET` — walk bound multiple of the budget.
pub const VISITS_PER_BUDGET: usize = 2;
/// Upstream `PAGE_SIZE` default when unknown.
pub const PAGE_SIZE: u64 = 4096;
/// Upstream `_RAM_FSTYPES` — filesystems backed by RAM.
pub const RAM_FSTYPES: &[&str] = &["tmpfs", "ramfs"];
/// Upstream `/proc/sysvipc/shm`.
pub const SYSVIPC_SHM_PATH: &str = "/proc/sysvipc/shm";

/// One process of the student (a `/proc/<pid>` entry).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcEntry {
    /// Pid.
    pub pid: u32,
    /// Owner uid.
    pub uid: u32,
    /// `statm` field 1 (resident pages).
    pub resident_pages: u64,
    /// `smaps_rollup` Pss in kB (None when unavailable — gVisor).
    pub pss_kib: Option<u64>,
}

/// One RAM-backed scratch file the uid owns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScratchFile {
    /// Path.
    pub path: String,
    /// Owner uid.
    pub uid: u32,
    /// `st_blocks` (512-byte units — RAM-backed files report their
    /// allocated size there).
    pub blocks: u64,
}

/// One SysV shared-memory segment (a `/proc/sysvipc/shm` row).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShmSegment {
    /// shmid (field 1).
    pub shmid: u32,
    /// Size in bytes (field 3).
    pub bytes: u64,
    /// Creator uid (field 9 — segments charge to the creator because
    /// `shmctl(IPC_SET)` lets the creator rewrite the owner).
    pub creator_uid: u32,
}

/// Parse `/proc/sysvipc/shm` text: a header line then rows; fields of
/// interest are 1 (shmid), 3 (size), 9 (cuid).
pub fn parse_sysvipc_shm(text: &str) -> Vec<ShmSegment> {
    let mut out = Vec::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split_whitespace().collect();
        // key(0) shmid(1) perms(2) size(3) cpid(4) lpid(5) nattch(6)
        // uid(7) gid(8) cuid(9) — segments charge to the creator (cuid).
        if f.len() < 10 {
            continue;
        }
        let (Ok(shmid), Ok(bytes), Ok(creator)) = (
            f[1].parse::<u32>(),
            f[3].parse::<u64>(),
            f[9].parse::<u32>(),
        ) else {
            continue;
        };
        out.push(ShmSegment {
            shmid,
            bytes,
            creator_uid: creator,
        });
    }
    out
}

/// The result of one weighing pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Weighing {
    /// Sum of process memory (bytes, RSS or PSS depending on precision).
    pub process_bytes: u64,
    /// RAM-backed scratch files, bytes.
    pub ram_scratch_bytes: u64,
    /// SysV segments created by the uid, bytes.
    pub sysv_bytes: u64,
    /// Processes counted.
    pub processes: usize,
    /// Scratch entries counted.
    pub scratch_entries: usize,
    /// Whether any count hit its walk budget (an *over*-count trigger
    /// upstream: "left more than N processes running / entries ...").
    pub budget_exceeded: bool,
}

impl Weighing {
    /// Total bytes charged.
    pub fn total_bytes(&self) -> u64 {
        self.process_bytes + self.ram_scratch_bytes + self.sysv_bytes
    }
}

/// Whether a path is disk-backed given a mount table (upstream
/// `disk_backed`): find the longest matching mount prefix; RAM
/// filesystems are tmpfs/ramfs; unclassifiable paths err toward *not*
/// disk-backed (toward watching).
pub fn disk_backed(path: &str, mounts: &[(String, String)]) -> bool {
    let mut best: Option<&(String, String)> = None;
    for m in mounts {
        if path.starts_with(&m.0) {
            match best {
                Some(b) if b.0.len() >= m.0.len() => {}
                _ => best = Some(m),
            }
        }
    }
    match best {
        Some((_, fstype)) => !RAM_FSTYPES.contains(&fstype.as_str()),
        None => false,
    }
}

/// The /proc + filesystem view the weigher reads.
pub trait ProcView: Send + Sync {
    /// Enumerate `/proc/<pid>` entries for a uid (bounded by
    /// `process_budget`).
    fn processes(&self, uid: u32, budget: usize) -> Vec<ProcEntry>;
    /// Enumerate owned files under the RAM-backed temp dirs (bounded by
    /// `entry_budget`).
    fn ram_scratch(&self, uid: u32, budget: usize) -> Vec<ScratchFile>;
    /// The SysV shm rows.
    fn sysv_shm(&self) -> Vec<ShmSegment>;
}

/// Weigh the uid's footprint (upstream `_weigh`).
pub fn weigh(
    view: &dyn ProcView,
    uid: u32,
    precise: bool,
    max_processes: Option<usize>,
    page_size: u64,
) -> Weighing {
    let process_budget = max_processes
        .map(|p| p.saturating_mul(VISITS_PER_BUDGET))
        .unwrap_or(MAX_UNCAPPED_PROCESSES);
    let procs = view.processes(uid, process_budget);
    let budget_exceeded = procs.len() >= process_budget;
    let process_bytes: u64 = procs
        .iter()
        .map(|p| {
            if precise {
                // PSS when available (smaps_rollup; gVisor has none).
                p.pss_kib.unwrap_or(p.resident_pages * page_size) * 1024
            } else {
                p.resident_pages * page_size
            }
        })
        .sum();
    let scratch = view.ram_scratch(uid, MAX_TMPFS_ENTRIES);
    let scratch_exceeded = scratch.len() >= MAX_TMPFS_ENTRIES;
    let ram_scratch_bytes: u64 = scratch.iter().map(|f| f.blocks * 512).sum();
    let sysv_bytes: u64 = view
        .sysv_shm()
        .iter()
        .filter(|s| s.creator_uid == uid)
        .map(|s| s.bytes)
        .sum();
    Weighing {
        process_bytes,
        ram_scratch_bytes,
        sysv_bytes,
        processes: procs.len(),
        scratch_entries: scratch.len(),
        budget_exceeded: budget_exceeded || scratch_exceeded,
    }
}

/// The watch state machine (upstream `start_watch`'s thread): poll →
/// weigh → on over-limit, re-weigh precisely → reap → keep watching.
pub struct MemoryWatch {
    uid: u32,
    max_bytes: u64,
    max_processes: Option<usize>,
    view: Arc<dyn ProcView>,
    cohort: Arc<dyn CohortBackend>,
    peak_bytes: AtomicU64,
    violations: AtomicU64,
    running: AtomicBool,
    /// Injected reap hook (tests observe kills).
    on_violation: Arc<dyn Fn() + Send + Sync>,
}

impl MemoryWatch {
    /// A watch over `view` reaping via `cohort`.
    pub fn new(
        uid: u32,
        max_bytes: u64,
        max_processes: Option<usize>,
        view: Arc<dyn ProcView>,
        cohort: Arc<dyn CohortBackend>,
    ) -> Self {
        Self {
            uid,
            max_bytes,
            max_processes,
            view,
            cohort,
            peak_bytes: AtomicU64::new(0),
            violations: AtomicU64::new(0),
            running: AtomicBool::new(true),
            on_violation: Arc::new(|| {}),
        }
    }

    /// Attach a violation observer.
    pub fn with_on_violation(mut self, f: impl Fn() + Send + Sync + 'static) -> Self {
        self.on_violation = Arc::new(f);
        self
    }

    /// One poll iteration (upstream's loop body every `POLL_INTERVAL_S`).
    pub fn tick(&self) -> WatchOutcome {
        let w = weigh(
            self.view.as_ref(),
            self.uid,
            false,
            self.max_processes,
            PAGE_SIZE,
        );
        let total = w.total_bytes();
        self.peak_bytes.fetch_max(total, Ordering::SeqCst);
        let over = total > self.max_bytes || w.budget_exceeded;
        if over {
            // Re-weigh precisely first: RSS over-counts shared pages; the
            // kill only fires when the precise pass agrees.
            let precise = weigh(
                self.view.as_ref(),
                self.uid,
                true,
                self.max_processes,
                PAGE_SIZE,
            );
            let precise_total = precise.total_bytes();
            self.peak_bytes.fetch_max(precise_total, Ordering::SeqCst);
            if precise_total > self.max_bytes || precise.budget_exceeded {
                self.violations.fetch_add(1, Ordering::SeqCst);
                (self.on_violation)();
                // Reap: cgroup kill + pidns + cohort + sweep. The watch
                // itself keeps running — tmpfs/SysV state that survives
                // SIGKILL keeps reaping respawns.
                let _ = kill_processes(self.cohort.as_ref(), &mut || f64::MAX);
                return WatchOutcome::Reaped {
                    bytes: precise_total,
                    budget_exceeded: precise.budget_exceeded,
                };
            }
            return WatchOutcome::UnderAfterPrecise {
                coarse_bytes: total,
                precise_bytes: precise_total,
            };
        }
        WatchOutcome::Within { bytes: total }
    }

    /// The peak footprint observed so far.
    pub fn peak_bytes(&self) -> u64 {
        self.peak_bytes.load(Ordering::SeqCst)
    }

    /// Violations reaped so far.
    pub fn violations(&self) -> u64 {
        self.violations.load(Ordering::SeqCst)
    }

    /// Whether the watch is still running (it never stops on its own).
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

/// One tick's outcome.
#[derive(Debug, Clone, PartialEq)]
pub enum WatchOutcome {
    /// Within budget.
    Within {
        /// Current total.
        bytes: u64,
    },
    /// Over on the coarse pass, within on the precise pass (shared-page
    /// over-count; no reap).
    UnderAfterPrecise {
        /// Coarse (RSS) total.
        coarse_bytes: u64,
        /// Precise (PSS) total.
        precise_bytes: u64,
    },
    /// Over on both passes: reaped.
    Reaped {
        /// Precise total at reap time.
        bytes: u64,
        /// A walk budget tripped (over-count trigger).
        budget_exceeded: bool,
    },
}

/// A static proc view for tests.
pub struct StaticView {
    /// Processes.
    pub procs: Vec<ProcEntry>,
    /// Scratch files.
    pub scratch: Vec<ScratchFile>,
    /// SysV segments.
    pub shm: Vec<ShmSegment>,
}

impl ProcView for StaticView {
    fn processes(&self, uid: u32, budget: usize) -> Vec<ProcEntry> {
        self.procs
            .iter()
            .filter(|p| p.uid == uid)
            .take(budget.max(1))
            .cloned()
            .collect()
    }
    fn ram_scratch(&self, uid: u32, budget: usize) -> Vec<ScratchFile> {
        self.scratch
            .iter()
            .filter(|f| f.uid == uid)
            .take(budget.max(1))
            .cloned()
            .collect()
    }
    fn sysv_shm(&self) -> Vec<ShmSegment> {
        self.shm.clone()
    }
}

/// A cohort backend whose passes always clean up.
pub struct CleaningCohort;

impl CohortBackend for CleaningCohort {
    fn pass(&self) -> KillPass {
        KillPass {
            cgroup_kill: true,
            pidns_kill: true,
            cohort_kill_ran: true,
            remaining: 0,
        }
    }
    fn cohort_alive(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proc(uid: u32, pages: u64, pss_kib: Option<u64>) -> ProcEntry {
        ProcEntry {
            pid: rand_pid(),
            uid,
            resident_pages: pages,
            pss_kib,
        }
    }

    fn rand_pid() -> u32 {
        use std::sync::atomic::AtomicU32;
        static N: AtomicU32 = AtomicU32::new(1);
        N.fetch_add(1, Ordering::SeqCst)
    }

    #[test]
    fn sysvipc_parsing_fields_1_3_9() {
        let text = "       key      shmid perms       size      cpid      lpid nattch   uid   gid  cuid  cgid      atime      dtime      ctime\n\
0x00000000 327680   600        1048576     1000        1000      1     0     0    1000  1000 ...\n";
        let rows = parse_sysvipc_shm(text);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].shmid, 327680);
        assert_eq!(rows[0].bytes, 1_048_576);
        assert_eq!(rows[0].creator_uid, 1000);
    }

    #[test]
    fn disk_backed_classification() {
        let mounts = vec![
            ("/".to_string(), "ext4".to_string()),
            ("/tmp".to_string(), "tmpfs".to_string()),
            ("/dev/shm".to_string(), "tmpfs".to_string()),
            ("/var/tmp".to_string(), "ext4".to_string()),
        ];
        assert!(!disk_backed("/tmp/foo", &mounts));
        assert!(!disk_backed("/dev/shm/x", &mounts));
        assert!(disk_backed("/var/tmp/x", &mounts));
        assert!(disk_backed("/home/student/x", &mounts));
        // unclassifiable (no matching prefix) → treated as RAM-backed
        assert!(!disk_backed("/weird", &[]));
        // longest prefix wins
        let nested = vec![
            ("/a".to_string(), "ext4".to_string()),
            ("/a/b".to_string(), "tmpfs".to_string()),
        ];
        assert!(!disk_backed("/a/b/f", &nested));
        assert!(disk_backed("/a/c/f", &nested));
    }

    #[test]
    fn weighing_counts_all_three_sources() {
        let view = StaticView {
            procs: vec![proc(1000, 1000, Some(500)), proc(0, 9999, None)],
            scratch: vec![ScratchFile {
                path: "/tmp/blob".into(),
                uid: 1000,
                blocks: 2048, // 1 MiB
            }],
            shm: vec![
                ShmSegment {
                    shmid: 1,
                    bytes: 2 << 20,
                    creator_uid: 1000,
                },
                ShmSegment {
                    shmid: 2,
                    bytes: 9 << 20,
                    creator_uid: 0,
                },
            ],
        };
        let w = weigh(&view, 1000, false, None, 4096);
        // RSS: 1000 pages * 4096 = 4 MiB (other uid ignored)
        assert_eq!(w.process_bytes, 1000 * 4096);
        assert_eq!(w.ram_scratch_bytes, 2048 * 512);
        // Only the segment the uid CREATED.
        assert_eq!(w.sysv_bytes, 2 << 20);
        assert_eq!(w.processes, 1);
        assert!(!w.budget_exceeded);
    }

    #[test]
    fn precise_pass_uses_pss() {
        let view = StaticView {
            procs: vec![proc(1000, 10_000, Some(2_000))],
            scratch: vec![],
            shm: vec![],
        };
        let coarse = weigh(&view, 1000, false, None, 4096);
        let precise = weigh(&view, 1000, true, None, 4096);
        assert_eq!(coarse.process_bytes, 10_000 * 4096);
        assert_eq!(precise.process_bytes, 2_000 * 1024);
    }

    #[test]
    fn watch_reaps_on_precise_overage_only() {
        // RSS way over, PSS under → no reap.
        let view = Arc::new(StaticView {
            procs: vec![proc(1000, 10_000_000, Some(1))],
            scratch: vec![],
            shm: vec![],
        });
        let watch = MemoryWatch::new(1000, 1 << 30, None, view, Arc::new(CleaningCohort));
        match watch.tick() {
            WatchOutcome::UnderAfterPrecise {
                coarse_bytes,
                precise_bytes,
            } => {
                assert!(coarse_bytes > precise_bytes);
            }
            other => panic!("expected UnderAfterPrecise, got {other:?}"),
        }
        assert_eq!(watch.violations(), 0);

        // Both over → reap.
        let view2 = Arc::new(StaticView {
            procs: vec![proc(1000, 10_000_000, Some(2_000_000))],
            scratch: vec![],
            shm: vec![],
        });
        let watch2 = MemoryWatch::new(1000, 1 << 30, None, view2, Arc::new(CleaningCohort))
            .with_on_violation(|| {});
        match watch2.tick() {
            WatchOutcome::Reaped { .. } => {}
            other => panic!("expected Reaped, got {other:?}"),
        }
        assert_eq!(watch2.violations(), 1);
        // The watch keeps running after a reap.
        assert!(watch2.is_running());
    }

    #[test]
    fn watch_counts_sysv_that_survives_sigkill() {
        // SysV segments survive SIGKILL; the next tick still sees them.
        let view = Arc::new(StaticView {
            procs: vec![],
            scratch: vec![],
            shm: vec![ShmSegment {
                shmid: 7,
                bytes: 6 << 30,
                creator_uid: 1000,
            }],
        });
        let watch = MemoryWatch::new(1000, 1 << 30, None, view, Arc::new(CleaningCohort));
        match watch.tick() {
            WatchOutcome::Reaped { bytes, .. } => assert_eq!(bytes, 6 << 30),
            other => panic!("expected Reaped, got {other:?}"),
        }
    }

    #[test]
    fn peak_tracking() {
        let view = Arc::new(StaticView {
            procs: vec![proc(1000, 1024, None)],
            scratch: vec![],
            shm: vec![],
        });
        let watch = MemoryWatch::new(1000, 1 << 30, None, view, Arc::new(CleaningCohort));
        watch.tick();
        watch.tick();
        assert!(watch.peak_bytes() >= 1024 * 4096);
    }

    #[test]
    fn budget_exceeded_is_violation() {
        // "left more than N processes running" — an over-count trigger.
        let view = StaticView {
            procs: (0..10).map(|_| proc(1000, 1, None)).collect(),
            scratch: vec![],
            shm: vec![],
        };
        // Budget 4 (=> walk bound 8), 10 procs → budget exceeded.
        let w = weigh(&view, 1000, false, Some(4), 4096);
        assert!(w.budget_exceeded);
        // the precise pass re-uses the same bound
        let precise = weigh(&view, 1000, true, Some(4), 4096);
        assert!(precise.budget_exceeded);
    }
}
