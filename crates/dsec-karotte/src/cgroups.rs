//! cgroup v1/v2 mount parsing and the student cgroup model.
//!
//! Ports of upstream `karotte/cgroups.py`. The model parses `/proc/mounts`
//! text (v2 unified and v1 hierarchies), delegates the harness leaf,
//! creates per-uid student groups, and exposes the exact file contents
//! upstream writes: `memory.max` + `memory.swap.max=0` (swap would let
//! the group page out past its cap), `pids.max`,
//! `memory.oom.group=1` (an OOM kill kills the whole group), and the
//! atomic `cgroup.kill` subtree kill.

use std::collections::BTreeMap;

/// Upstream `PROC_MOUNTS`.
pub const PROC_MOUNTS: &str = "/proc/mounts";
/// Upstream `HARNESS_LEAF` — the group the harness pids move into.
pub const HARNESS_LEAF: &str = "karotte_harness";
/// Upstream `_FREEZE_TIMEOUT` (v1 freezer).
pub const FREEZE_TIMEOUT_S: f64 = 5.0;
/// Upstream `_NO_LIMIT_THRESHOLD` (v1 reads `-1` back as PAGE_COUNTER_MAX).
pub const NO_LIMIT_THRESHOLD: u64 = 1 << 60;

/// One parsed mount line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mount {
    /// Mount point.
    pub path: String,
    /// Filesystem type.
    pub fstype: String,
    /// Non-flag options (v1 controllers).
    pub controllers: Vec<String>,
    /// Read-only mount.
    pub read_only: bool,
}

/// The mount flags to strip when deriving v1 controllers
/// (upstream `_MOUNT_FLAGS`).
pub const MOUNT_FLAGS: &[&str] = &[
    "rw", "ro", "nosuid", "nodev", "noexec", "relatime", "noatime", "seclabel",
];

/// Parse `/proc/mounts` text (upstream `parse_mounts`). v2 controllers are
/// `cgroup2` mounts (no controller options); v1 mounts carry controller
/// names among the options, minus flags and `name=`/`mode=`/`size=`
/// prefixes.
pub fn parse_mounts(text: &str) -> Vec<Mount> {
    let mut out = Vec::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let (Some(_dev), Some(path), Some(fstype)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let Some(options) = parts.next() else {
            continue;
        };
        let opts: Vec<&str> = options.split(',').collect();
        let read_only = opts.contains(&"ro");
        let controllers: Vec<String> = opts
            .iter()
            .filter(|o| {
                !MOUNT_FLAGS.contains(o)
                    && !o.starts_with("name=")
                    && !o.starts_with("mode=")
                    && !o.starts_with("size=")
            })
            .map(|s| s.to_string())
            .collect();
        // Only cgroup-ish mounts matter upstream; keep everything so
        // callers can reuse the parse (reclaim uses the same table).
        out.push(Mount {
            path: unescape_path(path),
            fstype: fstype.to_string(),
            controllers,
            read_only,
        });
    }
    out
}

/// Unescape the octal escapes `/proc/mounts` uses (e.g. `\040` for
/// space).
fn unescape_path(p: &str) -> String {
    let mut out = String::new();
    let mut chars = p.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            let mut digits = String::new();
            for _ in 0..3 {
                if let Some(d) = chars.peek().filter(|c| c.is_ascii_digit()) {
                    digits.push(*d);
                    chars.next();
                }
            }
            if let Ok(v) = u8::from_str_radix(&digits, 8) {
                out.push(v as char);
            } else {
                out.push(c);
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Which cgroup version the host exposes (upstream `detect_cgroups`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CgroupLayout {
    /// cgroup v2 unified hierarchy.
    V2,
    /// cgroup v1 split hierarchies.
    V1,
    /// No writable cgroup mount.
    None,
}

/// The harness's own cgroup dir from a `/proc/self/cgroup` `0::` line,
/// unwrapping `karotte_harness` leaves (upstream `own_cgroup_dir`).
pub fn own_cgroup_dir(line: &str) -> String {
    let path = line.trim_start_matches("0::");
    let trimmed = path.trim_start_matches('/');
    let stripped = trimmed
        .split('/')
        .filter(|seg| *seg != HARNESS_LEAF)
        .collect::<Vec<_>>()
        .join("/");
    format!("/{stripped}")
}

/// Whether a v2 mount exposes the required controllers
/// (`memory` + `pids`, upstream `_REQUIRED`).
pub fn v2_has_required_controllers(controllers_line: &str) -> bool {
    let have: Vec<&str> = controllers_line.split_whitespace().collect();
    have.contains(&"memory") && have.contains(&"pids")
}

/// The student cgroup (v2 or v1), as file contents and operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StudentCgroup {
    /// The group's directory path.
    pub path: String,
    /// The layout it lives under.
    pub layout: CgroupLayout,
}

impl StudentCgroup {
    /// A v2 student group at `path`.
    pub fn v2(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            layout: CgroupLayout::V2,
        }
    }

    /// A v1 student group (path = the memory-root dir).
    pub fn v1(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            layout: CgroupLayout::V1,
        }
    }

    /// The writes that create + delegate this group (upstream
    /// `V2Cgroup.create` + `_delegate`): move harness pids into the leaf,
    /// enable `+memory +pids` on subtree control, arm oom.group.
    pub fn delegation_writes(&self) -> Vec<(String, String)> {
        match self.layout {
            CgroupLayout::V2 => vec![
                (
                    format!("{}/cgroup.subtree_control", self.parent()),
                    "+memory +pids".to_string(),
                ),
                (format!("{}/memory.oom.group", self.path), "1".to_string()),
            ],
            _ => vec![],
        }
    }

    fn parent(&self) -> String {
        let p = self.path.trim_end_matches('/');
        match p.rfind('/') {
            Some(i) => p[..i].to_string(),
            None => p.to_string(),
        }
    }

    /// Set the memory limit (upstream `set_memory_limit`): writes
    /// `memory.max` **and** `memory.swap.max = 0` when a limit is set
    /// (`"max"` when lifted) — otherwise the group pages out past its cap
    /// on hosts with swap.
    pub fn set_memory_limit_writes(&self, limit: Option<u64>) -> Vec<(String, String)> {
        match self.layout {
            CgroupLayout::V2 => {
                let max = match limit {
                    Some(n) => n.to_string(),
                    None => "max".to_string(),
                };
                let swap = if limit.is_some() { "0" } else { "max" };
                vec![
                    (format!("{}/memory.max", self.path), max),
                    (format!("{}/memory.swap.max", self.path), swap.to_string()),
                ]
            }
            CgroupLayout::V1 => {
                let v = match limit {
                    Some(n) => n.to_string(),
                    None => "-1".to_string(),
                };
                vec![(format!("{}/memory.limit_in_bytes", self.path), v)]
            }
            CgroupLayout::None => vec![],
        }
    }

    /// Set the process limit (upstream `set_process_limit`).
    pub fn set_process_limit_writes(&self, limit: Option<u32>) -> Vec<(String, String)> {
        let v = match limit {
            Some(n) => n.to_string(),
            None => "max".to_string(),
        };
        match self.layout {
            CgroupLayout::V2 | CgroupLayout::None => {
                vec![(format!("{}/pids.max", self.path), v)]
            }
            CgroupLayout::V1 => vec![(format!("{}/pids.max", self.path), v)],
        }
    }

    /// Join this group (write own pid to `cgroup.procs` — done in the
    /// forked child while still root, upstream `join_self`).
    pub fn join_self_write(&self, pid: u32) -> (String, String) {
        (format!("{}/cgroup.procs", self.path), pid.to_string())
    }

    /// The subtree kill write (upstream `kill_all`): `cgroup.kill = 1`
    /// is an atomic kill of everything in the subtree.
    pub fn kill_all_write(&self) -> (String, String) {
        (format!("{}/cgroup.kill", self.path), "1".to_string())
    }

    /// Parse a read-back limit (`"max"`, negative, or ≥ 2^60 → None;
    /// v1 `-1` reads back as PAGE_COUNTER_MAX).
    pub fn parse_limit_value(v: &str) -> Option<u64> {
        if v == "max" || v == "-1" || v.is_empty() {
            return None;
        }
        match v.parse::<u64>() {
            Ok(n) if n >= NO_LIMIT_THRESHOLD => None,
            Ok(n) => Some(n),
            Err(_) => None,
        }
    }

    /// The tightest inherited ancestor limit (upstream
    /// `inherited_memory_limit` walks ancestors taking the min).
    pub fn inherited_memory_limit(ancestor_limits: &[u64]) -> Option<u64> {
        ancestor_limits.iter().copied().min()
    }

    /// The kernel floors `memory.max` to a page (upstream
    /// `_page_floor`, used when readback must equal what was asked).
    pub fn page_floor(nbytes: u64, page_size: u64) -> u64 {
        nbytes / page_size * page_size
    }
}

/// Detect the cgroup layout from mounts and the harness's own cgroup
/// (upstream `detect_cgroups`): prefer a writable cgroup2 mount whose
/// controllers include memory+pids; else v1 roots with memory+pids; else
/// none.
pub fn detect_layout(mounts: &[Mount], own_dir: &str, controllers_line: &str) -> CgroupLayout {
    for m in mounts {
        if m.fstype == "cgroup2" && !m.read_only {
            // controllers are read at the harness's own dir (unwrapping
            // harness leaves); the caller passes the line already.
            if v2_has_required_controllers(controllers_line) {
                let _ = own_dir;
                let _ = &m.path;
                return CgroupLayout::V2;
            }
        }
    }
    let mut memory = false;
    let mut pids = false;
    for m in mounts {
        if m.fstype == "cgroup" && !m.read_only {
            if m.controllers.iter().any(|c| c == "memory") {
                memory = true;
            }
            if m.controllers.iter().any(|c| c == "pids") {
                pids = true;
            }
        }
    }
    if memory && pids {
        CgroupLayout::V1
    } else {
        CgroupLayout::None
    }
}

/// v1 controller roots (memory/pids) from the mounts (upstream
/// `V1Cgroup(roots)`).
pub fn v1_roots(mounts: &[Mount]) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for m in mounts {
        if m.fstype == "cgroup" && !m.read_only {
            for c in &m.controllers {
                if c == "memory" || c == "pids" || c == "freezer" {
                    out.insert(c.clone(), m.path.clone());
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_MOUNTS: &str = "\
sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0
proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0
udev /dev devtmpfs rw,nosuid,relatime 0 0
tmpfs /dev/shm tmpfs rw,nosuid,nodev,noexec 0 0
cgroup2 /sys/fs/cgroup cgroup2 rw,nosuid,nodev,noexec,relatime 0 0
/dev/vda1 /workdir ext4 rw,relatime 0 0
tmpfs /tmp/tmp\\040space tmpfs rw 0 0
";

    #[test]
    fn parse_mounts_v2_and_flags() {
        let mounts = parse_mounts(SAMPLE_MOUNTS);
        let c2 = mounts.iter().find(|m| m.fstype == "cgroup2").unwrap();
        assert_eq!(c2.path, "/sys/fs/cgroup");
        assert!(!c2.read_only);
        assert!(c2.controllers.is_empty(), "v2 has no controller options");
        // ext4 mount parsed with no controllers
        let ext = mounts.iter().find(|m| m.fstype == "ext4").unwrap();
        assert!(ext.controllers.is_empty());
        // octal escape unescaped
        let tmp = mounts
            .iter()
            .find(|m| m.fstype == "tmpfs" && m.path.contains("space"))
            .unwrap();
        assert_eq!(tmp.path, "/tmp/tmp space");
    }

    #[test]
    fn parse_mounts_v1_controllers() {
        let text = "cgroup /sys/fs/cgroup/memory cgroup rw,memory 0 0\ncgroup /sys/fs/cgroup/pids cgroup rw,pids 0 0\n";
        let mounts = parse_mounts(text);
        assert!(mounts[0].controllers.contains(&"memory".to_string()));
        assert!(mounts[1].controllers.contains(&"pids".to_string()));
        // flags stripped
        assert!(!mounts[0].controllers.contains(&"rw".to_string()));
    }

    #[test]
    fn v1_controller_prefixes_stripped() {
        let text = "cgroup /sys/fs/cgroup/name=systemd cgroup rw,name=systemd,xattr 0 0\n";
        let mounts = parse_mounts(text);
        assert!(mounts[0]
            .controllers
            .iter()
            .all(|c| !c.starts_with("name=")));
        assert!(mounts[0].controllers.contains(&"xattr".to_string()));
    }

    #[test]
    fn own_cgroup_dir_unwraps_harness_leaf() {
        assert_eq!(
            own_cgroup_dir("0::/session.slice/karotte_harness"),
            "/session.slice"
        );
        assert_eq!(own_cgroup_dir("0::/"), "/");
        assert_eq!(own_cgroup_dir("0::/a/b"), "/a/b");
    }

    #[test]
    fn v2_required_controllers() {
        assert!(v2_has_required_controllers("cpuset cpu io memory pids"));
        assert!(!v2_has_required_controllers("cpuset cpu io"));
    }

    #[test]
    fn memory_limit_writes_pair_with_swap() {
        let g = StudentCgroup::v2("/sys/fs/cgroup/karotte_uid_1000");
        let writes = g.set_memory_limit_writes(Some(3u64 << 30));
        assert_eq!(
            writes,
            vec![
                (
                    "/sys/fs/cgroup/karotte_uid_1000/memory.max".to_string(),
                    (3u64 << 30).to_string()
                ),
                (
                    "/sys/fs/cgroup/karotte_uid_1000/memory.swap.max".to_string(),
                    "0".to_string()
                ),
            ]
        );
        let lifted = g.set_memory_limit_writes(None);
        assert_eq!(lifted[0].1, "max");
        assert_eq!(lifted[1].1, "max");
    }

    #[test]
    fn v1_memory_limit_uses_limit_in_bytes() {
        let g = StudentCgroup::v1("/sys/fs/cgroup/memory/karotte_uid_1000");
        let writes = g.set_memory_limit_writes(Some(1024));
        assert_eq!(
            writes,
            vec![(
                "/sys/fs/cgroup/memory/karotte_uid_1000/memory.limit_in_bytes".to_string(),
                "1024".to_string()
            )]
        );
        let lifted = g.set_memory_limit_writes(None);
        assert_eq!(lifted[0].1, "-1");
    }

    #[test]
    fn delegation_arms_oom_group() {
        let g = StudentCgroup::v2("/sys/fs/cgroup/karotte_uid_1000");
        let writes = g.delegation_writes();
        assert_eq!(
            writes[0],
            (
                "/sys/fs/cgroup/cgroup.subtree_control".to_string(),
                "+memory +pids".to_string()
            )
        );
        assert_eq!(
            writes[1],
            (
                "/sys/fs/cgroup/karotte_uid_1000/memory.oom.group".to_string(),
                "1".to_string()
            )
        );
    }

    #[test]
    fn kill_all_is_atomic_subtree_kill() {
        let g = StudentCgroup::v2("/sys/fs/cgroup/karotte_uid_1000");
        assert_eq!(
            g.kill_all_write(),
            (
                "/sys/fs/cgroup/karotte_uid_1000/cgroup.kill".to_string(),
                "1".to_string()
            )
        );
    }

    #[test]
    fn join_self_write() {
        let g = StudentCgroup::v2("/sys/fs/cgroup/karotte_uid_1000");
        assert_eq!(
            g.join_self_write(4242),
            (
                "/sys/fs/cgroup/karotte_uid_1000/cgroup.procs".to_string(),
                "4242".to_string()
            )
        );
    }

    #[test]
    fn limit_readback_parsing() {
        assert_eq!(StudentCgroup::parse_limit_value("max"), None);
        assert_eq!(StudentCgroup::parse_limit_value("-1"), None);
        assert_eq!(StudentCgroup::parse_limit_value(""), None);
        assert_eq!(
            StudentCgroup::parse_limit_value("3221225472"),
            Some(3221225472)
        );
        // PAGE_COUNTER_MAX reads back as "no limit"
        assert_eq!(
            StudentCgroup::parse_limit_value(&(1u64 << 60).to_string()),
            None
        );
        assert_eq!(StudentCgroup::parse_limit_value("garbage"), None);
    }

    #[test]
    fn page_floor() {
        assert_eq!(StudentCgroup::page_floor(3221225473, 4096), 3221225472);
        assert_eq!(StudentCgroup::page_floor(4095, 4096), 0);
    }

    #[test]
    fn inherited_limit_takes_min_of_ancestors() {
        assert_eq!(
            StudentCgroup::inherited_memory_limit(&[16 << 30, 8 << 30, 64 << 30]),
            Some(8 << 30)
        );
    }

    #[test]
    fn detect_layout_prefers_v2_with_controllers() {
        let mounts = parse_mounts(SAMPLE_MOUNTS);
        assert_eq!(
            detect_layout(&mounts, "/session.slice", "cpuset cpu io memory pids"),
            CgroupLayout::V2
        );
        assert_eq!(
            detect_layout(&mounts, "/", "cpuset cpu io"),
            CgroupLayout::None
        );
    }

    #[test]
    fn detect_layout_falls_back_to_v1() {
        let text = "cgroup /sys/fs/cgroup/memory cgroup rw,memory 0 0\ncgroup /sys/fs/cgroup/pids cgroup rw,pids 0 0\n";
        let mounts = parse_mounts(text);
        assert_eq!(detect_layout(&mounts, "/", ""), CgroupLayout::V1);
        let roots = v1_roots(&mounts);
        assert_eq!(
            roots.get("memory").map(String::as_str),
            Some("/sys/fs/cgroup/memory")
        );
        assert_eq!(
            roots.get("pids").map(String::as_str),
            Some("/sys/fs/cgroup/pids")
        );
    }

    #[test]
    fn read_only_cgroups_are_not_usable() {
        let text = "cgroup2 /sys/fs/cgroup cgroup2 ro,nosuid 0 0\n";
        let mounts = parse_mounts(text);
        assert_eq!(
            detect_layout(&mounts, "/", "memory pids"),
            CgroupLayout::None
        );
    }
}
