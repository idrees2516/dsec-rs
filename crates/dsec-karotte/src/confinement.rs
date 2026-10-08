//! The confinement layer: the `Contract` tri-state, resource limits, the
//! iptables owner-match firewall builder, canary self-tests, and the
//! process-cohort reaping model.
//!
//! Ports of upstream `karotte/confinement.py` and
//! `karotte/process_utils.py`. Rule construction and limit arithmetic are
//! real and byte-compatible with upstream (the same iptables commands,
//! the same cgroup writes, the same ordering); execution is behind the
//! [`FirewallBackend`]/[`CohortBackend`] traits so the model is testable
//! without root. The design spine: **a write that "succeeded" is not
//! evidence of enforcement** — gVisor accepts cgroup writes and enforces
//! nothing, so every operation reports
//! [`Contract::Prevented`]/[`Contract::Reaped`]/[`Contract::Unsupported`].

use crate::error::{Error, Result, StudentMisbehaviorError};
use std::net::Ipv4Addr;

/// What a confinement mechanism actually guarantees
/// (upstream `confinement.Contract`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Contract {
    /// The kernel refused the operation at the boundary — hard limit.
    Prevented,
    /// A watchdog detected the violation and reaped it — soft limit.
    Reaped,
    /// No mechanism exists on this host; said out loud.
    Unsupported,
}

impl Contract {
    /// The upstream string values.
    pub fn as_str(&self) -> &'static str {
        match self {
            Contract::Prevented => "prevented",
            Contract::Reaped => "detected_and_reaped",
            Contract::Unsupported => "not_supported",
        }
    }
}

/// The sandbox type (upstream `Sandbox`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sandbox {
    /// runc containers.
    Runc,
    /// gVisor (runsc) — accepts rules without enforcing them.
    Gvisor,
    /// A real VM (Firecracker / apple-container).
    Vm,
}

impl Sandbox {
    /// Parse `KAROTTE_SANDBOX` (`runc`/`gvisor`/`vm`; `firecracker` is a
    /// legacy alias for `vm` via `_missing_`).
    pub fn from_env_value(v: &str) -> Self {
        match v {
            "gvisor" => Sandbox::Gvisor,
            "vm" | "firecracker" => Sandbox::Vm,
            _ => Sandbox::Runc,
        }
    }

    /// The env string.
    pub fn as_env_value(&self) -> &'static str {
        match self {
            Sandbox::Runc => "runc",
            Sandbox::Gvisor => "gvisor",
            Sandbox::Vm => "vm",
        }
    }

    /// gVisor is excluded from the cgroup path: it accepts writes and
    /// enforces nothing (upstream `build_confinement`).
    pub fn uses_cgroups(&self) -> bool {
        !matches!(self, Sandbox::Gvisor)
    }

    /// gVisor rejects uid_map writes (upstream `--map-current-user`
    /// omission) — its procfs rejects them, children see overflow uid.
    pub fn maps_userns(&self) -> bool {
        !matches!(self, Sandbox::Gvisor)
    }
}

/// Upstream `HARNESS_RESERVE_BYTES` — 1 GiB withheld from the student.
pub const HARNESS_RESERVE_BYTES: u64 = 1 << 30;
/// Upstream `STUDENT_PROCESS_LIMIT`.
pub const STUDENT_PROCESS_LIMIT: u32 = 2048;
/// Upstream `STUDENT_FILE_COUNT_LIMIT`.
pub const STUDENT_FILE_COUNT_LIMIT: u64 = 1_000_000;
/// Upstream `_FREE_DISK_FRACTION`.
pub const FREE_DISK_FRACTION: f64 = 0.8;
/// Upstream `STUDENT_CGROUP_NAME`.
pub const STUDENT_CGROUP_NAME: &str = "karotte_student";
/// Per-uid group name (upstream `karotte_uid_{uid}`).
pub fn uid_cgroup_name(uid: u32) -> String {
    format!("karotte_uid_{uid}")
}
/// Upstream `_LOOP_DEVICES` — loop nodes for the quota mounts.
pub const LOOP_DEVICES: u32 = 8;
/// Upstream student OOM preference.
pub const STUDENT_OOM_SCORE_ADJ: i32 = 1000;
/// Upstream `_SELF_TEST_TIMEOUT_SECONDS`.
pub const SELF_TEST_TIMEOUT_S: f64 = 1.0;
/// Upstream kill deadline.
pub const KILL_DEADLINE_S: f64 = 30.0;
/// Upstream pass interval.
pub const KILL_PASS_INTERVAL_S: f64 = 0.05;
/// Upstream in-cohort `kill(-1)` rounds (gVisor races fork chains).
pub const COHORT_KILL_ROUNDS: usize = 100;
/// Upstream max /proc sweep walks.
pub const MAX_SWEEP_WALKS: usize = 10;
/// The mount namespace flag (upstream `CLONE_NEWNS`).
pub const CLONE_NEWNS: i64 = 0x0002_0000;
/// The IPC namespace flag (upstream `CLONE_NEWIPC`).
pub const CLONE_NEWIPC: i64 = 0x0800_0000;

// ---------------------------------------------------------------------------
// resource limits
// ---------------------------------------------------------------------------

/// A byte/count limit on a path set (upstream `FileLimit`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileLimit {
    /// Paths the limit applies to.
    pub paths: Vec<String>,
    /// Byte cap (None = unbounded).
    pub bytes: Option<u64>,
    /// File-count cap.
    pub count: Option<u64>,
}

/// The student's resource limits (upstream `ResourceLimits`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceLimits {
    /// Memory bytes.
    pub memory_bytes: Option<u64>,
    /// Process count.
    pub process_count: Option<u32>,
    /// File caps.
    pub file: Option<FileLimit>,
}

impl ResourceLimits {
    /// The default limits for a sandbox memory budget
    /// (upstream `apply_default_limits`): memory = budget − harness
    /// reserve, processes = 2048, files = 0.8 of free disk capped at
    /// 1,000,000 entries.
    pub fn defaults(sandbox_memory_bytes: u64, free_disk_bytes: u64) -> Self {
        Self {
            memory_bytes: Some(sandbox_memory_bytes.saturating_sub(HARNESS_RESERVE_BYTES)),
            process_count: Some(STUDENT_PROCESS_LIMIT),
            file: Some(FileLimit {
                paths: vec![
                    "/workdir".into(),
                    "/tmp".into(),
                    "/var/tmp".into(),
                    "/dev/shm".into(),
                ],
                bytes: Some((free_disk_bytes as f64 * FREE_DISK_FRACTION) as u64),
                count: Some(STUDENT_FILE_COUNT_LIMIT),
            }),
        }
    }
}

/// Upstream `sandbox_memory_bytes` precedence: explicit env > inherited
/// cgroup limit > physical RAM.
pub fn sandbox_memory_bytes(
    env_bytes: Option<u64>,
    inherited_cgroup_limit: Option<u64>,
    physical_bytes: u64,
) -> Option<u64> {
    if let Some(b) = env_bytes.filter(|b| *b > HARNESS_RESERVE_BYTES) {
        return Some(b);
    }
    inherited_cgroup_limit.or(Some(physical_bytes))
}

// ---------------------------------------------------------------------------
// the firewall (iptables owner rules, built in the exact upstream order)
// ---------------------------------------------------------------------------

/// One iptables rule (as the shell command fragment upstream builds).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirewallRule {
    /// The full `-A OUTPUT ...` rule text.
    pub text: String,
}

/// The student-network policy (upstream `KAROTTE_STUDENT_NETWORK`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkPolicy {
    /// Default: localhost + own addresses + allowlist only.
    Strict,
    /// Also metadata + private ranges (upstream `internal`).
    Internal,
}

impl NetworkPolicy {
    /// The upstream private/metadata ranges allowed under `internal`
    /// (v4: 169.254.169.254, 169.254/16, 10/8, 172.16/12, 192.168/16).
    pub fn internal_ranges_v4() -> Vec<&'static str> {
        vec![
            "169.254.0.0/16",
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
        ]
    }
}

/// The firewall plan for one student uid (upstream
/// `restrict_to_internal_network`).
#[derive(Debug, Clone)]
pub struct FirewallPlan {
    /// Rules in application order.
    pub rules: Vec<FirewallRule>,
    /// The final target (REJECT upstream; DROP on gVisor).
    pub final_target: &'static str,
}

/// Build the firewall rules for `uid` in the **exact upstream order**:
///
/// 1. DROP each blocked port (tcp),
/// 2. ACCEPT loopback,
/// 3. ACCEPT the sandbox's own interface addresses,
/// 4. internal ranges (if internal),
/// 5. ACCEPT each allowed ip,
/// 6. the final REJECT (or DROP under gVisor, which uses
///    iptables-legacy).
pub fn firewall_plan(
    sandbox: Sandbox,
    uid: u32,
    blocked_ports: &[u16],
    own_addresses: &[Ipv4Addr],
    policy: NetworkPolicy,
    allowed_ips: &[Ipv4Addr],
) -> FirewallPlan {
    let mut rules = Vec::new();
    let owner = format!("-m owner --uid-owner {uid}");
    // 1. Blocked ports first (before the localhost ACCEPT — upstream order).
    for p in blocked_ports {
        rules.push(FirewallRule {
            text: format!("-A OUTPUT {owner} -p tcp --dport {p} -j DROP"),
        });
    }
    // 2. Loopback.
    rules.push(FirewallRule {
        text: format!("-A OUTPUT {owner} -d 127.0.0.0/8 -j ACCEPT"),
    });
    // 3. The sandbox's own addresses.
    for a in own_addresses {
        rules.push(FirewallRule {
            text: format!("-A OUTPUT {owner} -d {a} -j ACCEPT"),
        });
    }
    // 4. Internal ranges.
    if policy == NetworkPolicy::Internal {
        for range in NetworkPolicy::internal_ranges_v4() {
            rules.push(FirewallRule {
                text: format!("-A OUTPUT {owner} -d {range} -j ACCEPT"),
            });
        }
    }
    // 5. Explicit allowlist.
    for a in allowed_ips {
        rules.push(FirewallRule {
            text: format!("-A OUTPUT {owner} -d {a} -j ACCEPT"),
        });
    }
    // 6. The final reject/deny.
    let final_target = if sandbox == Sandbox::Gvisor {
        "DROP"
    } else {
        "REJECT"
    };
    rules.push(FirewallRule {
        text: format!("-A OUTPUT {owner} -j {final_target}"),
    });
    FirewallPlan {
        rules,
        final_target,
    }
}

/// The upstream canaries: 1.1.1.1:80 always; 169.254.169.254:80 and the
/// default gateway's :80/:53 when not internal
/// (upstream `firewall_canaries` — a connection that *succeeds* proves
/// the firewall is broken).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Canary {
    /// Target host.
    pub host: Ipv4Addr,
    /// Target port.
    pub port: u16,
}

/// The default gateway from a `/proc/net/route`-style table: rows of
/// `iface destination gateway flags ...` with gateway hex
/// little-endian and flags bit 0x2 (`RTF_GATEWAY`).
pub fn default_gateway(rows: &[(String, u32, u32, u32)]) -> Option<Ipv4Addr> {
    for (_iface, _dest, gateway, flags) in rows {
        if flags & 0x2 != 0 {
            // little-endian hex to dotted quad
            let g = *gateway;
            return Some(Ipv4Addr::new(
                (g & 0xff) as u8,
                ((g >> 8) & 0xff) as u8,
                ((g >> 16) & 0xff) as u8,
                ((g >> 24) & 0xff) as u8,
            ));
        }
    }
    None
}

/// The canary targets for a policy and gateway (upstream order).
pub fn firewall_canaries(policy: NetworkPolicy, gateway: Option<Ipv4Addr>) -> Vec<Canary> {
    let mut out = vec![Canary {
        host: Ipv4Addr::new(1, 1, 1, 1),
        port: 80,
    }];
    if policy == NetworkPolicy::Strict {
        out.push(Canary {
            host: Ipv4Addr::new(169, 254, 169, 254),
            port: 80,
        });
        if let Some(g) = gateway {
            out.push(Canary { host: g, port: 80 });
            out.push(Canary { host: g, port: 53 });
        }
    }
    out
}

/// Whether the firewall is effective given canary reachability
/// (upstream `reachable_as` + rule application): the firewall took iff
/// rules applied and no canary is reachable afterwards.
pub fn firewall_effective(rules_took: bool, canaries_reached_after: &[bool]) -> Contract {
    if !rules_took {
        return Contract::Unsupported;
    }
    if canaries_reached_after.iter().any(|r| *r) {
        // A reachable canary means the firewall is broken — a failure,
        // surfaced as Reaped/Prevented neither; upstream raises.
        Contract::Unsupported
    } else {
        Contract::Prevented
    }
}

/// The firewall execution backend.
pub trait FirewallBackend: Send + Sync {
    /// Apply one rule; Ok(()) = written. On gVisor a "successful" write
    /// is still untrusted (upstream warns and returns False).
    fn apply_rule(&self, rule: &FirewallRule) -> Result<()>;

    /// Parse `iptables -S OUTPUT` and delete rules owned by `uid`
    /// (upstream `rules_owned_by` + lift loop).
    fn lift_rules(&self, uid: u32) -> Result<usize>;
}

/// A recording backend for tests.
#[derive(Default)]
pub struct RecordingFirewall {
    /// Every applied rule text, in order.
    pub applied: std::sync::Mutex<Vec<String>>,
    /// Rules reported by a simulated `iptables -S`.
    pub listing: std::sync::Mutex<Vec<String>>,
}

impl FirewallBackend for RecordingFirewall {
    fn apply_rule(&self, rule: &FirewallRule) -> Result<()> {
        self.applied.lock().unwrap().push(rule.text.clone());
        self.listing.lock().unwrap().push(rule.text.clone());
        Ok(())
    }

    fn lift_rules(&self, uid: u32) -> Result<usize> {
        let needle = format!("--uid-owner {uid}");
        let mut listing = self.listing.lock().unwrap();
        let before = listing.len();
        listing.retain(|r| !r.contains(&needle));
        Ok(before - listing.len())
    }
}

/// Apply a plan through a backend; returns whether every rule took
/// (upstream aborts on the first failure unless tolerating).
pub fn apply_plan(backend: &dyn FirewallBackend, plan: &FirewallPlan) -> Result<bool> {
    for rule in &plan.rules {
        backend.apply_rule(rule)?;
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// process cohort reaping (process_utils.py)
// ---------------------------------------------------------------------------

/// One pass of the upstream reaping loop (per pass, in order):
///
/// 1. cgroup kill,
/// 2. foreign pidns init kill,
/// 3. in-cohort `kill(-1)` (a demoted helper running 100 rounds),
/// 4. `/proc` sweep.
#[derive(Debug, Clone, Default)]
pub struct KillPass {
    /// The cgroup subtree kill ran.
    pub cgroup_kill: bool,
    /// Foreign PID-namespace init kills ran (killing a pidns init drops
    /// everything inside).
    pub pidns_kill: bool,
    /// The in-cohort `kill(-1)` helper ran this pass. A clean `/proc`
    /// sweep only proves the cohort is gone when this ran (the
    /// load-bearing boolean of the whole design).
    pub cohort_kill_ran: bool,
    /// Processes seen in the final sweep.
    pub remaining: u32,
}

/// The cohort backend.
pub trait CohortBackend: Send + Sync {
    /// Run one pass (cgroup kill + pidns kill + cohort helper + sweep).
    fn pass(&self) -> KillPass;
    /// Whether the uid owns any processes right now.
    fn cohort_alive(&self) -> bool;
}

/// The full reap loop (upstream `kill_processes`): up to
/// `KILL_DEADLINE_S` with `KILL_PASS_INTERVAL_S` between passes; a clean
/// sweep counts only when the in-cohort kill ran that pass; at the
/// deadline the cohort is *unreapable* — a student misbehavior (score 0),
/// because a live student process could still race the grader.
pub fn kill_processes(backend: &dyn CohortBackend, now: &mut dyn FnMut() -> f64) -> Result<()> {
    let start = now();
    let deadline = start + KILL_DEADLINE_S;
    loop {
        let pass = backend.pass();
        if pass.remaining == 0 {
            if pass.cohort_kill_ran {
                return Ok(());
            }
            // A clean sweep without the in-cohort kill proves nothing.
            return Err(Error::Confinement(
                "clean /proc sweep without the in-cohort kill — broken grader".into(),
            ));
        }
        if now() >= deadline {
            return Err(Error::Misbehavior(
                StudentMisbehaviorError::UnreapableCohort(format!(
                    "{KILL_DEADLINE_S} second kill deadline exceeded"
                )),
            ));
        }
    }
}

/// The demoted-subprocess wrapper argv (upstream
/// `wrap_to_disable_networking`): `unshare --user --net
/// [--map-current-user] -- argv`; gVisor omits the mapping (its procfs
/// rejects uid_map writes).
pub fn wrap_to_disable_networking(sandbox: Sandbox, argv: &[String]) -> Vec<String> {
    let mut out = vec!["unshare".to_string(), "--user".into(), "--net".into()];
    if sandbox.maps_userns() {
        out.push("--map-current-user".into());
    }
    out.push("--".into());
    out.extend(argv.iter().cloned());
    out
}

/// The PID-namespace wrapper argv (upstream `student_session_command`
/// with a PID namespace): `--pid --fork --mount-proc
/// --kill-child=SIGKILL --setuid <uid> --setgid <uid>` — *without*
/// `--user`: root already holds CAP_SYS_ADMIN, and a userns would leave
/// the student as overflow-uid nobody.
pub fn pid_namespace_argv(uid: u32, argv: &[String]) -> Vec<String> {
    let mut out = vec![
        "unshare".to_string(),
        "--pid".into(),
        "--fork".into(),
        "--mount-proc".into(),
        "--kill-child=SIGKILL".into(),
        format!("--setuid {uid}"),
        format!("--setgid {uid}"),
    ];
    out.push("--".into());
    out.extend(argv.iter().cloned());
    out
}

/// The demotion preexec sequence (upstream `make_demote_fn`'s `_demote`,
/// in order — the order is the security property): re-enter cwd → fchown
/// stdio fds → isolate mounts (CLONE_NEWNS + MS_REC|MS_PRIVATE first) →
/// IPC namespace → oom_score_adj 1000 → setgroups([]) → setgid → setuid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DemoteStep {
    /// `chdir` back to the cwd, verifying `(st_dev, st_ino)`.
    ReenterCwd,
    /// `fchown` the stdio pipe fds to the student (root-owned pipe inodes
    /// would block reopening `/dev/stdin`).
    ChownStdioFds,
    /// Mount namespace + private propagation + tmpfs over ephemeral dirs
    /// + ro bind-remounts + ro-tmpfs covers.
    IsolateMounts,
    /// IPC namespace (SysV objects start empty, die with the namespace).
    UnshareIpc,
    /// `/proc/self/oom_score_adj = 1000`.
    OomScoreAdj,
    /// `setgroups([])`.
    SetGroups,
    /// `setgid(uid)`.
    SetGid,
    /// `setuid(uid)`.
    SetUid,
}

/// The full upstream demotion sequence, in order.
pub fn demotion_sequence() -> Vec<DemoteStep> {
    vec![
        DemoteStep::ReenterCwd,
        DemoteStep::ChownStdioFds,
        DemoteStep::IsolateMounts,
        DemoteStep::UnshareIpc,
        DemoteStep::OomScoreAdj,
        DemoteStep::SetGroups,
        DemoteStep::SetGid,
        DemoteStep::SetUid,
    ]
}

/// The student identity env (upstream `student_identity_env`): HOME (the
/// workdir fallback), USER, LOGNAME for the demoted uid.
pub fn student_identity_env(uid: u32, workdir: &str, username: &str) -> Vec<(String, String)> {
    vec![
        ("HOME".to_string(), workdir.to_string()),
        ("USER".to_string(), username.to_string()),
        ("LOGNAME".to_string(), username.to_string()),
        ("KAROTTE_DEMOTE_ID".to_string(), uid.to_string()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_strings_match_upstream() {
        assert_eq!(Contract::Prevented.as_str(), "prevented");
        assert_eq!(Contract::Reaped.as_str(), "detected_and_reaped");
        assert_eq!(Contract::Unsupported.as_str(), "not_supported");
    }

    #[test]
    fn sandbox_parsing_and_legacy_alias() {
        assert_eq!(Sandbox::from_env_value("firecracker"), Sandbox::Vm);
        assert_eq!(Sandbox::from_env_value("vm"), Sandbox::Vm);
        assert_eq!(Sandbox::from_env_value("gvisor"), Sandbox::Gvisor);
        assert_eq!(Sandbox::from_env_value("runc"), Sandbox::Runc);
        assert!(!Sandbox::Gvisor.uses_cgroups());
        assert!(Sandbox::Vm.uses_cgroups());
        assert!(!Sandbox::Gvisor.maps_userns());
    }

    #[test]
    fn default_limits_reserve_harness_memory() {
        let l = ResourceLimits::defaults(5 << 30, 100 << 30);
        assert_eq!(l.memory_bytes, Some((5 << 30) - (1 << 30)));
        assert_eq!(l.process_count, Some(2048));
        let file = l.file.unwrap();
        assert_eq!(file.count, Some(1_000_000));
        assert_eq!(file.bytes, Some((100 << 30) * 4 / 5));
    }

    #[test]
    fn sandbox_memory_precedence() {
        // explicit env only counts above the harness reserve.
        assert_eq!(
            sandbox_memory_bytes(Some(1 << 20), None, 16 << 30),
            Some(16 << 30)
        );
        assert_eq!(
            sandbox_memory_bytes(Some(2 << 30), Some(4 << 30), 16 << 30),
            Some(2 << 30)
        );
        assert_eq!(
            sandbox_memory_bytes(None, Some(4 << 30), 16 << 30),
            Some(4 << 30)
        );
        assert_eq!(sandbox_memory_bytes(None, None, 16 << 30), Some(16 << 30));
    }

    #[test]
    fn firewall_rules_follow_upstream_order() {
        let plan = firewall_plan(
            Sandbox::Runc,
            1000,
            &[8001, 8080],
            &[Ipv4Addr::new(172, 17, 0, 2)],
            NetworkPolicy::Strict,
            &[Ipv4Addr::new(10, 1, 2, 3)],
        );
        let texts: Vec<&str> = plan.rules.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(texts.len(), 6);
        // blocked ports before loopback
        assert!(texts[0].contains("--dport 8001 -j DROP"));
        assert!(texts[1].contains("--dport 8080 -j DROP"));
        // loopback
        assert!(texts[2].contains("-d 127.0.0.0/8 -j ACCEPT"));
        // own address
        assert!(texts[3].contains("-d 172.17.0.2 -j ACCEPT"));
        // allowlist
        assert!(texts[4].contains("-d 10.1.2.3 -j ACCEPT"));
        // final reject, owner-matched
        assert_eq!(texts[5], "-A OUTPUT -m owner --uid-owner 1000 -j REJECT");
        assert_eq!(plan.final_target, "REJECT");
        // every rule carries the owner match — the firewall is per-uid.
        for t in &texts {
            assert!(t.contains("-m owner --uid-owner 1000"), "{t}");
        }
    }

    #[test]
    fn gvisor_uses_drop_and_internal_adds_ranges() {
        let plan = firewall_plan(
            Sandbox::Gvisor,
            1000,
            &[],
            &[],
            NetworkPolicy::Internal,
            &[],
        );
        assert_eq!(plan.final_target, "DROP");
        let joined: String = plan
            .rules
            .iter()
            .map(|r| r.text.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("169.254.0.0/16"));
        assert!(joined.contains("10.0.0.0/8"));
        assert!(joined.contains("172.16.0.0/12"));
        assert!(joined.contains("192.168.0.0/16"));
    }

    #[test]
    fn gateway_parsing_little_endian() {
        // 192.168.1.1 as little-endian hex = 0x0101A8C0
        let rows = vec![("eth0".to_string(), 0, 0, 0), ("lo".to_string(), 0, 0, 0)];
        // 0x0101A8C0 → 192.168.1.1
        let rows2 = vec![("eth0".to_string(), 0, 0x0101A8C0, 0x2)];
        assert_eq!(default_gateway(&rows), None);
        assert_eq!(default_gateway(&rows2), Some(Ipv4Addr::new(192, 168, 1, 1)));
        assert_eq!(
            default_gateway(&[("w".into(), 0, 0xAC100001, 0x2)]),
            Some(Ipv4Addr::new(1, 0, 16, 172))
        );
    }

    #[test]
    fn canaries_strict_vs_internal() {
        let g = Some(Ipv4Addr::new(192, 168, 0, 1));
        let strict = firewall_canaries(NetworkPolicy::Strict, g);
        assert_eq!(strict.len(), 4); // 1.1.1.1 + metadata + gw:80 + gw:53
        assert_eq!(strict[0].host, Ipv4Addr::new(1, 1, 1, 1));
        assert_eq!(strict[1].host, Ipv4Addr::new(169, 254, 169, 254));
        let internal = firewall_canaries(NetworkPolicy::Internal, g);
        assert_eq!(internal.len(), 1);
    }

    #[test]
    fn firewall_effectiveness_contract() {
        // rules took + nothing reachable → Prevented
        assert_eq!(
            firewall_effective(true, &[false, false]),
            Contract::Prevented
        );
        // rules took + canary reachable → broken (Unsupported upstream)
        assert_eq!(
            firewall_effective(true, &[false, true]),
            Contract::Unsupported
        );
        // rules didn't take → Unsupported
        assert_eq!(firewall_effective(false, &[]), Contract::Unsupported);
    }

    #[test]
    fn recording_backend_applies_and_lifts() {
        let plan = firewall_plan(
            Sandbox::Runc,
            1000,
            &[8001],
            &[],
            NetworkPolicy::Strict,
            &[],
        );
        let backend = RecordingFirewall::default();
        assert!(apply_plan(&backend, &plan).unwrap());
        assert_eq!(backend.applied.lock().unwrap().len(), plan.rules.len());
        let lifted = backend.lift_rules(1000).unwrap();
        assert_eq!(lifted, plan.rules.len());
        assert!(backend.listing.lock().unwrap().is_empty());
    }

    #[test]
    fn kill_processes_requires_cohort_kill_for_clean_sweep() {
        struct CleanWithoutHelper;
        impl CohortBackend for CleanWithoutHelper {
            fn pass(&self) -> KillPass {
                KillPass {
                    cgroup_kill: true,
                    pidns_kill: true,
                    cohort_kill_ran: false,
                    remaining: 0,
                }
            }
            fn cohort_alive(&self) -> bool {
                false
            }
        }
        let mut t = 0.0;
        let mut clock = move || {
            let now = t;
            t += 0.05;
            now
        };
        let err = kill_processes(&CleanWithoutHelper, &mut clock).unwrap_err();
        assert!(err.to_string().contains("in-cohort kill"));
    }

    #[test]
    fn kill_processes_reaps_then_times_out_as_misbehavior() {
        struct Sometimes;
        impl CohortBackend for Sometimes {
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
        let mut t = 0.0;
        let mut clock = move || {
            let now = t;
            t += 0.05;
            now
        };
        assert!(kill_processes(&Sometimes, &mut clock).is_ok());

        struct Stuck;
        impl CohortBackend for Stuck {
            fn pass(&self) -> KillPass {
                KillPass {
                    cgroup_kill: true,
                    pidns_kill: true,
                    cohort_kill_ran: true,
                    remaining: 7,
                }
            }
            fn cohort_alive(&self) -> bool {
                true
            }
        }
        let mut t2 = 0.0;
        let mut clock2 = move || {
            let now = t2;
            t2 += 5.0; // fast-forward past the 30s deadline
            now
        };
        let err = kill_processes(&Stuck, &mut clock2).unwrap_err();
        match err {
            Error::Misbehavior(StudentMisbehaviorError::UnreapableCohort(_)) => {}
            other => panic!("expected UnreapableCohort, got {other:?}"),
        }
    }

    #[test]
    fn network_namespace_wrapper_gvisor_shape() {
        let argv = vec!["bash".to_string(), "-c".to_string(), "true".to_string()];
        let wrapped = wrap_to_disable_networking(Sandbox::Runc, &argv);
        assert_eq!(
            wrapped,
            vec![
                "unshare",
                "--user",
                "--net",
                "--map-current-user",
                "--",
                "bash",
                "-c",
                "true"
            ]
        );
        let gvisor = wrap_to_disable_networking(Sandbox::Gvisor, &argv);
        assert!(!gvisor.contains(&"--map-current-user".to_string()));
    }

    #[test]
    fn pid_namespace_wrapper_shape() {
        let argv = vec!["python".to_string()];
        let wrapped = pid_namespace_argv(1000, &argv);
        assert_eq!(
            wrapped,
            vec![
                "unshare",
                "--pid",
                "--fork",
                "--mount-proc",
                "--kill-child=SIGKILL",
                "--setuid 1000",
                "--setgid 1000",
                "--",
                "python"
            ]
        );
        assert!(!wrapped.contains(&"--user".to_string()));
    }

    #[test]
    fn demotion_sequence_order_is_the_security_property() {
        let seq = demotion_sequence();
        assert_eq!(seq.first(), Some(&DemoteStep::ReenterCwd));
        // uid drop LAST, after everything that needs root.
        assert_eq!(seq.last(), Some(&DemoteStep::SetUid));
        let oom_pos = seq
            .iter()
            .position(|s| *s == DemoteStep::OomScoreAdj)
            .unwrap();
        let groups_pos = seq
            .iter()
            .position(|s| *s == DemoteStep::SetGroups)
            .unwrap();
        let gid_pos = seq.iter().position(|s| *s == DemoteStep::SetGid).unwrap();
        let uid_pos = seq.iter().position(|s| *s == DemoteStep::SetUid).unwrap();
        assert!(oom_pos < groups_pos && groups_pos < gid_pos && gid_pos < uid_pos);
    }

    #[test]
    fn student_identity_env_shape() {
        let env = student_identity_env(1000, "/workdir", "student");
        assert_eq!(env[0], ("HOME".to_string(), "/workdir".to_string()));
        assert_eq!(env[1], ("USER".to_string(), "student".to_string()));
        assert_eq!(
            env[3],
            ("KAROTTE_DEMOTE_ID".to_string(), "1000".to_string())
        );
    }
}
