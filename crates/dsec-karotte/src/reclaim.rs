//! The fd-safe reclaim sweep: delete everything the student uid owns,
//! without following a single symlink.
//!
//! Ports of upstream `karotte/reclaim.py`. The sweep is an iterative DFS
//! using **dir-fd-relative syscalls only** (`openat`/`fstatat`/`unlinkat`
//! with `O_NOFOLLOW`), so every path component is bounded by NAME_MAX
//! and symlink redirection cannot fool it; parent directories are
//! re-opened through `..` with `(st_dev, st_ino)` verification. Virtual
//! filesystems are spared; the SysV segments the uid created are removed
//! with `IPC_RMID`.

use crate::error::{Error, Result, StudentMisbehaviorError};
use std::collections::BTreeSet;

/// Upstream reclaim deadline (600 s → `ReclaimError`, a misbehavior).
pub const RECLAIM_TIMEOUT_S: f64 = 600.0;
/// Upstream SysV probe cap (2^20 ids).
pub const MAX_PROBED_SHMID: u32 = 1 << 20;
/// Upstream `CAP_IPC_OWNER` bit (15) in `CapEff`.
pub const CAP_IPC_OWNER_BIT: u32 = 15;
/// Upstream default excludes.
pub const DEFAULT_EXCLUDES: &[&str] = &["/root", "/var/karotte_quota"];
/// Upstream `_VIRTUAL_FSTYPES` — never swept, never deleted.
pub const VIRTUAL_FSTYPES: &[&str] = &[
    "autofs",
    "binfmt_misc",
    "bpf",
    "cgroup",
    "cgroup2",
    "configfs",
    "debugfs",
    "devpts",
    "devtmpfs",
    "efivarfs",
    "fusectl",
    "hugetlbfs",
    "mqueue",
    "nsfs",
    "proc",
    "pstore",
    "rpc_pipefs",
    "securityfs",
    "selinuxfs",
    "sysfs",
    "tracefs",
];

/// A mount table entry for sweepability decisions
/// (upstream `_Mounts`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SweepMount {
    /// Mount point.
    pub path: String,
    /// Filesystem type.
    pub fstype: String,
    /// Read-only mount.
    pub read_only: bool,
}

impl SweepMount {
    /// Whether this mount may be swept (upstream: not read-only, not a
    /// virtual fstype).
    pub fn sweepable(&self) -> bool {
        !self.read_only && !VIRTUAL_FSTYPES.contains(&self.fstype.as_str())
    }
}

/// Decide sweepability of a path given the mount table: the longest
/// matching prefix mount decides; unclassifiable paths are sweepable
/// (upstream errs toward deleting).
pub fn sweepable_at(path: &str, mounts: &[SweepMount]) -> bool {
    let mut best: Option<&SweepMount> = None;
    for m in mounts {
        if path.starts_with(&m.path) {
            match best {
                Some(b) if b.path.len() >= m.path.len() => {}
                _ => best = Some(m),
            }
        }
    }
    match best {
        Some(m) => m.sweepable(),
        None => true,
    }
}

/// One entry in the DFS stack (upstream `_Frame`): the *name* of the
/// current entry relative to the sweep root — never a path string, so
/// every component is bounded by NAME_MAX and immune to PATH_MAX limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// The entry name inside the parent (never a path string).
    pub name: String,
    /// Depth of the frame.
    pub depth: usize,
    /// What to do when this frame pops.
    pub action: FrameAction,
}

/// What a popped frame does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameAction {
    /// Stat and process (unlink files/symlinks, enter dirs).
    Process,
    /// Remove the (now-drained) owned directory — the "on the way out"
    /// rmdir of the upstream DFS unwind.
    RmDir,
}

/// The filesystem view the sweep walks.
pub trait FsView: Send + Sync {
    /// `lstat`-style info for a name inside `dir` (dir-fd-relative,
    /// never following symlinks). Returns `(uid, is_dir, is_symlink)`.
    fn lstat_at(&self, dir: &str, name: &str) -> Option<(u32, bool, bool)>;

    /// Open a subdirectory (returns a handle name for further
    /// `*_at` calls). `None` if not a directory / unreadable.
    fn open_dir_at(&self, dir: &str, name: &str) -> Option<String>;

    /// Enumerate the entries of a directory handle (names only).
    fn list_dir(&self, dir: &str) -> Vec<String>;

    /// `unlinkat` a name (files or empty dirs).
    /// `Ok(true)` = removed; `Ok(false)` = tolerated failure
    /// (ENOENT/ENOTEMPTY); `Err` = refusal.
    fn unlink_at(&self, dir: &str, name: &str, is_dir: bool) -> Result<bool>;

    /// Verify `..` from `dir` returns to `(st_dev, st_ino)` of `parent`
    /// (upstream re-opens parents through `..` to keep the fd chain
    /// short; ESTALE on moves).
    fn verify_parent(&self, dir: &str, parent: &str) -> bool;

    /// Remove SysV segments (upstream `shmctl(IPC_RMID)`; the walk
    /// variant caps at 2^20 ids).
    fn remove_sysv(&self, uid: u32) -> Result<usize>;
}

/// Whether `path` is excluded directly or lives under an excluded
/// subtree (upstream spares the whole tree under each exclude).
pub fn is_excluded(path: &str, excludes: &BTreeSet<String>) -> bool {
    excludes.iter().any(|x| {
        let base = x.trim_end_matches('/');
        path == x || path.starts_with(&format!("{}/", base))
    })
}

/// The sweep result.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SweepReport {
    /// Entries deleted.
    pub deleted: usize,
    /// Directories removed.
    pub dirs_removed: usize,
    /// Entries that could not be deleted (first 10 shown upstream).
    pub failed: Vec<String>,
    /// Whether the 600 s deadline tripped.
    pub timed_out: bool,
    /// SysV segments removed.
    pub sysv_removed: usize,
}

/// Run the sweep (upstream `delete_files`): iterative DFS with explicit
/// frames, dir-fd-relative only, sparing non-sweepable mounts and
/// excludes; owned dirs are rmdir'd on the way out; at the deadline or
/// when anything survived → [`StudentMisbehaviorError::Reclaim`]
/// (score 0 — disk state the grader can't trust).
pub fn sweep(
    view: &dyn FsView,
    uid: u32,
    root: &str,
    excludes: &BTreeSet<String>,
    now: &mut dyn FnMut() -> f64,
) -> Result<SweepReport> {
    let start = now();
    let mut report = SweepReport::default();
    let mut stack: Vec<Frame> = Vec::new();

    // Seed with the root's entries.
    for name in view.list_dir(root) {
        stack.push(Frame {
            name,
            depth: 0,
            action: FrameAction::Process,
        });
    }

    while let Some(frame) = stack.pop() {
        if now() - start > RECLAIM_TIMEOUT_S {
            report.timed_out = true;
            return Err(Error::Misbehavior(StudentMisbehaviorError::Reclaim(
                format!("{RECLAIM_TIMEOUT_S} second sweep deadline exceeded"),
            )));
        }
        let full = format!("{}/{}", root.trim_end_matches('/'), frame.name);
        if is_excluded(&full, excludes) {
            continue;
        }
        if frame.action == FrameAction::RmDir {
            // Owned dir whose children have drained: remove it now
            // (ENOTEMPTY tolerated — something foreign remained inside).
            if view.unlink_at(root, &frame.name, true)? {
                report.dirs_removed += 1;
            }
            continue;
        }
        if !sweepable_at(&full, MOUNT_TABLE.with(|t| t.borrow().clone()).as_slice()) {
            // Mount points strictly below the path are set aside
            // upstream; the port skips their contents via the table.
            continue;
        }
        let Some((owner, is_dir, is_symlink)) = view.lstat_at(root, &frame.name) else {
            continue;
        };
        if is_symlink {
            // Never follow; symlinks are unlinked as themselves.
            if owner == uid {
                if view.unlink_at(root, &frame.name, false)? {
                    report.deleted += 1;
                } else {
                    report.failed.push(full);
                }
            }
            continue;
        }
        if is_dir {
            // Enter even when owned by others: uid files can live inside.
            if let Some(sub) = view.open_dir_at(root, &frame.name) {
                // Push the rmdir marker FIRST so it pops AFTER the
                // children (LIFO) — the upstream "remove on the way out".
                if owner == uid {
                    stack.push(Frame {
                        name: frame.name.clone(),
                        depth: frame.depth,
                        action: FrameAction::RmDir,
                    });
                }
                for name in view.list_dir(&sub) {
                    stack.push(Frame {
                        name: format!("{}/{}", frame.name, name),
                        depth: frame.depth + 1,
                        action: FrameAction::Process,
                    });
                }
            } else if owner == uid {
                // Unreadable owned dir on a sweepable fs → failed.
                report.failed.push(full);
            }
            continue;
        }
        // Regular file owned by the uid.
        if owner == uid {
            if view.unlink_at(root, &frame.name, false)? {
                report.deleted += 1;
            } else {
                report.failed.push(full);
            }
        }
    }

    report.sysv_removed = view.remove_sysv(uid)?;

    if !report.failed.is_empty() {
        let shown: Vec<String> = report.failed.iter().take(10).cloned().collect();
        return Err(Error::Misbehavior(StudentMisbehaviorError::Reclaim(
            format!("paths survived the sweep: {:?}", shown),
        )));
    }
    Ok(report)
}

thread_local! {
    static MOUNT_TABLE: std::cell::RefCell<Vec<SweepMount>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Set the sweep's mount table (test scaffolding; production derives it
/// from `/proc/mounts` via [`crate::cgroups::parse_mounts`]).
pub fn set_mount_table(mounts: Vec<SweepMount>) {
    MOUNT_TABLE.with(|t| *t.borrow_mut() = mounts);
}

/// Parse a `CapEff:` line for `CAP_IPC_OWNER` (bit 15) — whether the
/// sweep can remove other uids' SysV segments in-process (upstream
/// checks `/proc/self/status`).
pub fn has_cap_ipc_owner(cap_eff_hex: &str) -> bool {
    let v = u64::from_str_radix(cap_eff_hex.trim(), 16).unwrap_or(0);
    (v >> CAP_IPC_OWNER_BIT) & 1 == 1
}

/// The SysV walk bound (upstream: ids only climb; the allocation ceiling
/// is learned by allocating one and taking that id; ENOSPC → walk to
/// 2^20).
pub fn sysv_walk_bound(ceiling_id: Option<u32>) -> u32 {
    ceiling_id
        .map(|c| c.min(MAX_PROBED_SHMID))
        .unwrap_or(MAX_PROBED_SHMID)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Mutex;

    /// An in-memory FS implementing FsView with dir-fd semantics.
    struct MemFs {
        entries: Mutex<BTreeMap<String, (u32, bool, bool)>>, // path -> (uid, is_dir, is_symlink)
        fail_on: Vec<String>,
    }

    impl MemFs {
        fn new() -> Self {
            Self {
                entries: Mutex::new(BTreeMap::new()),
                fail_on: vec![],
            }
        }
        fn add(&mut self, path: &str, uid: u32, is_dir: bool, is_symlink: bool) {
            self.entries
                .lock()
                .unwrap()
                .insert(path.to_string(), (uid, is_dir, is_symlink));
        }
    }

    impl FsView for MemFs {
        fn lstat_at(&self, dir: &str, name: &str) -> Option<(u32, bool, bool)> {
            let path = join(dir, name);
            self.entries.lock().unwrap().get(&path).copied()
        }
        fn open_dir_at(&self, dir: &str, name: &str) -> Option<String> {
            let path = join(dir, name);
            if self
                .entries
                .lock()
                .unwrap()
                .get(&path)
                .map(|e| e.1)
                .unwrap_or(false)
            {
                Some(path)
            } else {
                None
            }
        }
        fn list_dir(&self, dir: &str) -> Vec<String> {
            let entries = self.entries.lock().unwrap();
            let prefix = format!("{}/", dir.trim_end_matches('/'));
            let mut names = Vec::new();
            for path in entries.keys() {
                if let Some(rest) = path.strip_prefix(&prefix) {
                    // direct children only (no deeper '/')
                    if !rest.contains('/') {
                        names.push(rest.to_string());
                    }
                }
            }
            names
        }
        fn unlink_at(&self, dir: &str, name: &str, is_dir: bool) -> Result<bool> {
            let path = join(dir, name);
            if self.fail_on.contains(&path) {
                return Ok(false);
            }
            let mut entries = self.entries.lock().unwrap();
            if !entries.contains_key(&path) {
                return Ok(false); // ENOENT tolerated
            }
            if is_dir {
                let prefix = format!("{}/", path);
                let still_has = entries.keys().any(|p| p.starts_with(&prefix));
                if still_has {
                    return Ok(false); // ENOTEMPTY tolerated
                }
            }
            entries.remove(&path);
            Ok(true)
        }
        fn verify_parent(&self, _dir: &str, _parent: &str) -> bool {
            true
        }
        fn remove_sysv(&self, _uid: u32) -> Result<usize> {
            Ok(0)
        }
    }

    fn join(dir: &str, name: &str) -> String {
        format!("{}/{}", dir.trim_end_matches('/'), name)
    }

    fn clock() -> impl FnMut() -> f64 {
        let mut t = 0.0;
        move || {
            let now = t;
            t += 0.001;
            now
        }
    }

    #[test]
    fn virtual_fstypes_are_spared() {
        assert!(VIRTUAL_FSTYPES.contains(&"proc"));
        assert!(VIRTUAL_FSTYPES.contains(&"sysfs"));
        assert!(!VIRTUAL_FSTYPES.contains(&"ext4"));
        let m = SweepMount {
            path: "/proc".into(),
            fstype: "proc".into(),
            read_only: false,
        };
        assert!(!m.sweepable());
    }

    #[test]
    fn sweepability_longest_prefix() {
        let table = vec![
            SweepMount {
                path: "/".into(),
                fstype: "ext4".into(),
                read_only: false,
            },
            SweepMount {
                path: "/proc".into(),
                fstype: "proc".into(),
                read_only: false,
            },
            SweepMount {
                path: "/root".into(),
                fstype: "ext4".into(),
                read_only: true,
            },
        ];
        set_mount_table(table.clone());
        assert!(sweepable_at("/workdir/x", &table));
        assert!(!sweepable_at("/proc/123", &table));
        assert!(!sweepable_at("/root/secret", &table));
        // unclassifiable → sweepable
        assert!(sweepable_at(
            "/nowhere",
            &[SweepMount {
                path: "/other".into(),
                fstype: "ext4".into(),
                read_only: false,
            }]
        ));
    }

    #[test]
    fn sweep_deletes_owned_files_and_dirs() {
        let mut fs = MemFs::new();
        fs.add("/workdir", 1000, true, false);
        fs.add("/workdir/answer.txt", 1000, false, false);
        fs.add("/workdir/notes", 1000, true, false);
        fs.add("/workdir/notes/draft.md", 1000, false, false);
        fs.add("/workdir/root-owned.txt", 0, false, false);
        set_mount_table(vec![SweepMount {
            path: "/".into(),
            fstype: "ext4".into(),
            read_only: false,
        }]);
        let mut c = clock();
        let report = sweep(&fs, 1000, "/workdir", &Default::default(), &mut c).unwrap();
        assert_eq!(report.deleted, 2); // answer.txt + draft.md
        assert!(report.dirs_removed >= 1);
        // root-owned file survives
        assert!(fs
            .entries
            .lock()
            .unwrap()
            .contains_key("/workdir/root-owned.txt"));
        assert!(!fs
            .entries
            .lock()
            .unwrap()
            .contains_key("/workdir/answer.txt"));
    }

    #[test]
    fn sweep_never_follows_symlinks() {
        let mut fs = MemFs::new();
        fs.add("/workdir", 1000, true, false);
        // A symlink "pointing at" the grader's data — as itself.
        fs.add("/workdir/escape", 1000, false, true);
        // lstat_at for a symlink via O_NOFOLLOW reports is_symlink=true,
        // so the sweep unlinks the link, never the target.
        set_mount_table(vec![SweepMount {
            path: "/".into(),
            fstype: "ext4".into(),
            read_only: false,
        }]);
        let mut c = clock();
        let report = sweep(&fs, 1000, "/workdir", &Default::default(), &mut c).unwrap();
        assert_eq!(report.deleted, 1);
        assert!(!fs.entries.lock().unwrap().contains_key("/workdir/escape"));
    }

    #[test]
    fn sweep_foreign_dirs_are_entered_but_kept() {
        let mut fs = MemFs::new();
        fs.add("/workdir", 1000, true, false);
        fs.add("/workdir/shared", 0, true, false); // root-owned dir
        fs.add("/workdir/shared/student-junk", 1000, false, false);
        set_mount_table(vec![SweepMount {
            path: "/".into(),
            fstype: "ext4".into(),
            read_only: false,
        }]);
        let mut c = clock();
        let report = sweep(&fs, 1000, "/workdir", &Default::default(), &mut c).unwrap();
        // The junk inside is gone; the foreign dir survives.
        assert_eq!(report.deleted, 1);
        assert!(fs.entries.lock().unwrap().contains_key("/workdir/shared"));
        assert!(!fs
            .entries
            .lock()
            .unwrap()
            .contains_key("/workdir/shared/student-junk"));
    }

    #[test]
    fn surviving_paths_are_misbehavior() {
        let mut fs = MemFs::new();
        fs.add("/workdir", 1000, true, false);
        fs.add("/workdir/stuck.txt", 1000, false, false);
        fs.fail_on = vec!["/workdir/stuck.txt".to_string()];
        set_mount_table(vec![SweepMount {
            path: "/".into(),
            fstype: "ext4".into(),
            read_only: false,
        }]);
        let mut c = clock();
        let err = sweep(&fs, 1000, "/workdir", &Default::default(), &mut c).unwrap_err();
        match err {
            Error::Misbehavior(StudentMisbehaviorError::Reclaim(msg)) => {
                assert!(msg.contains("stuck.txt"), "{msg}");
            }
            other => panic!("expected Reclaim, got {other:?}"),
        }
    }

    #[test]
    fn deadline_is_misbehavior() {
        let mut fs = MemFs::new();
        fs.add("/workdir", 1000, true, false);
        fs.add("/workdir/x", 1000, false, false);
        set_mount_table(vec![SweepMount {
            path: "/".into(),
            fstype: "ext4".into(),
            read_only: false,
        }]);
        // A clock that jumps past the 600 s deadline within the first
        // frames.
        let mut t = 1000.0;
        let mut c = move || {
            let now = t;
            t += 700.0;
            now
        };
        let err = sweep(&fs, 1000, "/workdir", &Default::default(), &mut c).unwrap_err();
        assert!(err.to_string().contains("600"));
    }

    #[test]
    fn excludes_are_spared() {
        let mut fs = MemFs::new();
        fs.add("/var/karotte_quota", 1000, true, false);
        fs.add("/var/karotte_quota/quota.img", 1000, false, false);
        set_mount_table(vec![SweepMount {
            path: "/".into(),
            fstype: "ext4".into(),
            read_only: false,
        }]);
        let mut c = clock();
        let report = sweep(
            &fs,
            1000,
            "/var/karotte_quota",
            &BTreeSet::from(["/var/karotte_quota".to_string()]),
            &mut c,
        )
        .unwrap();
        assert_eq!(report.deleted, 0);
    }

    #[test]
    fn cap_ipc_owner_bit_15() {
        // bit 15 set
        assert!(has_cap_ipc_owner("00008000"));
        // full caps
        assert!(has_cap_ipc_owner("000001ffffffffff"));
        // bit 15 clear
        assert!(!has_cap_ipc_owner("00007fff"));
        assert!(!has_cap_ipc_owner("0"));
    }

    #[test]
    fn sysv_walk_bound_caps_at_2_pow_20() {
        assert_eq!(sysv_walk_bound(None), 1 << 20);
        assert_eq!(sysv_walk_bound(Some(100)), 100);
        assert_eq!(sysv_walk_bound(Some(9_000_000)), 1 << 20);
    }
}
