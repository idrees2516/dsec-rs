//! Driver configuration + host capability detection.

use std::path::{Path, PathBuf};
use std::time::Duration;

/// Where the `firecracker` binary is looked up (in order):
/// 1. `DSEC_FIRECRACKER_PATH` env var
/// 2. `DSEC_FIRECRACKER_BIN` env var (alias)
/// 3. `firecracker` on PATH (`which`)
pub const FIRECRACKER_PATH_ENV: &str = "DSEC_FIRECRACKER_PATH";

/// Kernel image used for every microVM boot (env: `DSEC_FC_KERNEL`).
pub const KERNEL_PATH_ENV: &str = "DSEC_FC_KERNEL";

/// Configuration for [`FirecrackerDriver`](crate::FirecrackerDriver).
#[derive(Debug, Clone)]
pub struct FirecrackerConfig {
    /// Firecracker binary (absolute or PATH-resolvable).
    pub binary: PathBuf,
    /// Kernel image handed to `PUT /boot-source`.
    pub kernel_image: PathBuf,
    /// Kernel command line. Defaults boot the read-only rootfs drive.
    pub kernel_args: String,
    /// Directory for per-VM API sockets.
    pub api_sock_dir: PathBuf,
    /// Directory where EROFS images are materialized as raw drive
    /// files (one per image, shared read-only across VMs — the
    /// paper's shared, deduplicated image cache semantics).
    pub rootfs_dir: PathBuf,
    /// Directory for per-VM writable scratch drives (working state).
    pub scratch_dir: PathBuf,
    /// Directory for snapshot / memory files.
    pub snapshot_dir: PathBuf,
    /// How long to wait for the API socket to appear after launch.
    pub boot_timeout: Duration,
    /// Per-request timeout against the VMM API.
    pub request_timeout: Duration,
    /// Test escape hatch: skip the `/dev/kvm` + binary checks so the
    /// full request sequencing can be driven against a fake VMM.
    /// Real deployments MUST leave this false.
    pub allow_missing_host_deps: bool,
}

impl Default for FirecrackerConfig {
    fn default() -> Self {
        FirecrackerConfig {
            binary: resolve_binary().unwrap_or_else(|| PathBuf::from("firecracker")),
            kernel_image: std::env::var(KERNEL_PATH_ENV)
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/var/lib/dsec/vmlinux")),
            kernel_args: "console=ttyS0 reboot=t panic=1 pci=off \
                          i8042.noaux i8042.nomux i8042.nopan i8042.dumbkbd \
                          quiet root=/dev/vda ro init=/sbin/init"
                .to_string(),
            api_sock_dir: std::env::temp_dir().join("dsec-fc/sock"),
            rootfs_dir: std::env::temp_dir().join("dsec-fc/rootfs"),
            scratch_dir: std::env::temp_dir().join("dsec-fc/scratch"),
            snapshot_dir: std::env::temp_dir().join("dsec-fc/snap"),
            boot_timeout: Duration::from_secs(15),
            request_timeout: Duration::from_secs(10),
            allow_missing_host_deps: false,
        }
    }
}

/// Resolves the firecracker binary path (env override, then `which`).
pub fn resolve_binary() -> Option<PathBuf> {
    for var in [FIRECRACKER_PATH_ENV, "DSEC_FIRECRACKER_BIN"] {
        if let Ok(p) = std::env::var(var) {
            let path = PathBuf::from(p);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    which("firecracker")
}

/// `which`-style PATH lookup without external crates.
fn which(bin: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(bin);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Host capability report — what `FirecrackerDriver::boot` needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capability {
    /// The firecracker binary was found.
    pub binary_found: bool,
    /// `/dev/kvm` exists and is writable (Firecracker requires KVM).
    pub kvm_available: bool,
    /// The kernel image exists.
    pub kernel_found: bool,
}

impl Capability {
    /// Everything needed for real boots.
    pub fn ready(&self) -> bool {
        self.binary_found && self.kvm_available && self.kernel_found
    }

    /// Human-readable blockers, in boot-relevance order.
    pub fn blockers(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.binary_found {
            out.push(format!(
                "firecracker binary not found (set {FIRECRACKER_PATH_ENV})"
            ));
        }
        if !self.kvm_available {
            out.push("/dev/kvm missing or not writable (Firecracker needs KVM)".into());
        }
        if !self.kernel_found {
            out.push(format!("kernel image not found (set {KERNEL_PATH_ENV})"));
        }
        out
    }
}

/// Probes the host for everything a real boot needs.
pub fn detect() -> Capability {
    let kvm = Path::new("/dev/kvm").exists();
    let kvm_writable = kvm
        && std::fs::metadata("/dev/kvm")
            .map(|m| !m.permissions().readonly())
            .unwrap_or(false);
    let kernel = std::env::var(KERNEL_PATH_ENV)
        .map(|k| Path::new(&k).is_file())
        .unwrap_or_else(|_| Path::new("/var/lib/dsec/vmlinux").is_file());
    Capability {
        binary_found: resolve_binary().is_some(),
        kvm_available: kvm_writable,
        kernel_found: kernel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_reports_blockers_on_bare_host() {
        // This CI box has neither firecracker nor /dev/kvm — the report
        // must be honest and list blockers (unless someone installs
        // them, in which case everything is simply ready).
        let cap = detect();
        if !cap.ready() {
            assert!(!cap.blockers().is_empty());
        }
    }

    #[test]
    fn capability_ready_when_all_present() {
        let cap = Capability {
            binary_found: true,
            kvm_available: true,
            kernel_found: true,
        };
        assert!(cap.ready());
        assert!(cap.blockers().is_empty());
    }
}
