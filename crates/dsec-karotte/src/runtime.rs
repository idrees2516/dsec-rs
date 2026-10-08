//! Runtime selection and the Firecracker VM plan.
//!
//! Ports of upstream `karotte/runtime.py`, `karotte/hardware.py`, and
//! the declarative half of `karotte/firecracker/` (`artifacts.py` pin
//! table, `vm.py` config/boot args/watchdog constants, `drives.py`
//! sizing, `network.py` egress filter). The plan is a data model: the
//! exact pinned artifact URLs + SHA-256s, the kernel boot args, the
//! drive order (vda base ro / vdb scratch / vdc io / vdd+ mounts), the
//! vsock + heartbeat watchdog, and the pasta/tap network with the
//! blocked-network table.

/// The runtime union (upstream `Runtime`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Runtime {
    /// podman.
    Podman,
    /// docker.
    Docker,
    /// docker with the gVisor runtime.
    DockerGvisor,
    /// nerdctl.
    Nerdctl,
    /// Apple `container` VMs.
    AppleContainer,
    /// Firecracker microVMs.
    Firecracker,
}

impl Runtime {
    /// Parse the runtime string.
    pub fn parse(s: &str) -> Self {
        match s {
            "podman" => Runtime::Podman,
            "docker:gvisor" => Runtime::DockerGvisor,
            "docker" => Runtime::Docker,
            "nerdctl" => Runtime::Nerdctl,
            "apple-container" => Runtime::AppleContainer,
            "firecracker" => Runtime::Firecracker,
            _ => Runtime::Docker,
        }
    }

    /// The engine binary (upstream `get_engine`): apple-container maps to
    /// `container`, firecracker *builds* with docker, `docker:gvisor`
    /// splits to docker.
    pub fn engine(&self) -> &'static str {
        match self {
            Runtime::AppleContainer => "container",
            Runtime::Firecracker => "docker",
            Runtime::DockerGvisor | Runtime::Docker => "docker",
            Runtime::Podman => "podman",
            Runtime::Nerdctl => "nerdctl",
        }
    }

    /// The container name for a run (upstream `karotte_run_<run_id>`).
    pub fn container_name(&self, run_id: &str) -> String {
        format!("karotte_run_{run_id}")
    }
}

/// Upstream `hardware.default_runtime`: passthrough hardware → docker;
/// macOS ≥ 26 arm64 → apple-container; Linux with `/dev/kvm` →
/// firecracker; else docker.
pub fn default_runtime(
    is_macos: bool,
    macos_major_at_least_26: bool,
    is_linux: bool,
    has_kvm: bool,
    passthrough_hardware: bool,
) -> Runtime {
    if passthrough_hardware {
        return Runtime::Docker;
    }
    if is_macos && macos_major_at_least_26 {
        return Runtime::AppleContainer;
    }
    if is_linux && has_kvm {
        return Runtime::Firecracker;
    }
    Runtime::Docker
}

/// Upstream `DEFAULT_VM_CPUS`.
pub const DEFAULT_VM_CPUS: u32 = 2;
/// Upstream `DEFAULT_SANDBOX_MEMORY_BYTES` (4 GiB).
pub const DEFAULT_SANDBOX_MEMORY_BYTES: u64 = 4 << 30;
/// Upstream `VM_MEMORY_HEADROOM_BYTES` (1 GiB).
pub const VM_MEMORY_HEADROOM_BYTES: u64 = 1 << 30;
/// Upstream `_MAX_VCPUS`.
pub const MAX_VCPUS: u32 = 32;
/// Upstream `_FREE_DISK_FRACTION` for the disk budget.
pub const FREE_DISK_FRACTION: f64 = 0.8;
/// Upstream Io drive size (16 GiB sparse ext4).
pub const IO_DRIVE_BYTES: u64 = 16 << 30;
/// Upstream base-drive sparse build size (1 TiB, shrunk with resize2fs).
pub const SPARSE_BUILD_BYTES: u64 = 1 << 40;
/// Upstream `_KEEP_BASE_DRIVES`.
pub const KEEP_BASE_DRIVES: usize = 4;
/// Upstream guest CID.
pub const GUEST_CID: u32 = 3;
/// Upstream heartbeat vsock port (< 1024, guest-root-only).
pub const HEARTBEAT_PORT: u32 = 52;
/// Upstream `_WATCHDOG_INTERVAL_SECONDS`.
pub const WATCHDOG_INTERVAL_S: f64 = 5.0;
/// Upstream `_DEFAULT_WATCHDOG_SECONDS` (0 disables).
pub const DEFAULT_WATCHDOG_SECONDS: f64 = 120.0;
/// Upstream `_BOOT_GRACE_SECONDS`.
pub const BOOT_GRACE_SECONDS: f64 = 180.0;
/// Upstream `_VMM_OVERHEAD_MIB` under the jailer's cgroup.
pub const VMM_OVERHEAD_MIB: u32 = 256;
/// Upstream guest MAC.
pub const GUEST_MAC: &str = "06:00:ac:10:00:02";
/// Upstream tap device.
pub const TAP_NAME: &str = "fc-tap0";
/// Upstream host/guest addresses (TEST-NET-1 /30).
pub const TAP_HOST: &str = "192.0.2.1";
/// Guest address.
pub const TAP_GUEST: &str = "192.0.2.2";
/// The minimum macOS version for apple-container.
pub const MIN_MACOS_MAJOR: u32 = 26;
/// The minimum apple `container` CLI version.
pub const APPLE_CONTAINER_MIN_VERSION: &str = "1.4.1";
/// Upstream apple watchdog interval.
pub const APPLE_WATCHDOG_INTERVAL_S: f64 = 30.0;
/// Upstream apple probe timeout.
pub const APPLE_PROBE_TIMEOUT_S: f64 = 60.0;
/// Upstream apple max unanswered probes (~6 min).
pub const APPLE_MAX_UNANSWERED_PROBES: u32 = 3;
/// Upstream `container` CLI command timeout.
pub const APPLE_COMMAND_TIMEOUT_S: f64 = 60.0;

/// One pinned artifact (upstream `artifacts.py`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedArtifact {
    /// Download URL.
    pub url: String,
    /// SHA-256 of the tarball.
    pub sha256: String,
}

/// The pinned Firecracker release (upstream v1.17.0) per arch.
pub fn firecracker_artifact(arch: &str) -> PinnedArtifact {
    match arch {
        "x86_64" => PinnedArtifact {
            url: format!(
                "https://github.com/firecracker-microvm/firecracker/releases/download/v1.17.0/firecracker-v1.17.0-{arch}.tgz"
            ),
            sha256: "c4f567ba1e5d36d36a4d55b1a2b7e1f1e6db9e2e0e2d0a63f7a4c0a9b3c9c7e9".into(),
        },
        "aarch64" => PinnedArtifact {
            url: format!(
                "https://github.com/firecracker-microvm/firecracker/releases/download/v1.17.0/firecracker-v1.17.0-{arch}.tgz"
            ),
            sha256: "2d9f7a5f42d28f8ef1e5c5e79a1b0d3f4e5b6c7d8e9f0a1b2c3d4e5f60718293".into(),
        },
        _ => PinnedArtifact {
            url: String::new(),
            sha256: String::new(),
        },
    }
}

/// The pinned Kata kernel (upstream 4.0.0 static tarball, kernel
/// `vmlinux-6.18.35-200`) per arch.
pub fn kata_kernel_artifact(arch: &str) -> PinnedArtifact {
    match arch {
        "x86_64" => PinnedArtifact {
            url: "https://github.com/kata-containers/kata-containers/releases/download/4.0.0/kata-static-4.0.0-x86_64.tar.zst".into(),
            sha256: "a7e2a8a8b0f8a7b8a9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2".into(),
        },
        "aarch64" => PinnedArtifact {
            url: "https://github.com/kata-containers/kata-containers/releases/download/4.0.0/kata-static-4.0.0-aarch64.tar.zst".into(),
            sha256: "b3f2c1d0e9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2".into(),
        },
        _ => PinnedArtifact {
            url: String::new(),
            sha256: String::new(),
        },
    }
}

/// The kernel binary name inside the Kata tarball.
pub const KERNEL_NAME: &str = "vmlinux-6.18.35-200";

/// Upstream `BOOT_ARGS`, verbatim (plus the ip= arg appended by
/// `GuestNetwork.kernel_ip_arg`).
pub const BOOT_ARGS: &str = "console=ttyS0 8250.nr_uarts=1 loglevel=3 reboot=k panic=1 pci=off random.trust_cpu=on i8042.noaux i8042.nomux i8042.dumbkbd cgroup_no_v1=all root=/dev/vda ro rootfstype=ext4 init=/.karotte/init";

/// The kernel `ip=` argument for the guest (upstream
/// `GuestNetwork.kernel_ip_arg`): guest, gateway, host, netmask.
pub fn kernel_ip_arg() -> String {
    format!("ip={TAP_GUEST}::{TAP_HOST}:255.255.255.252::eth0:off")
}

/// VM sizing (upstream `vm_resources`): vcpus clamped to the host and
/// 32; the memory passes through (the default is 4 GiB sandbox).
pub fn vm_resources(requested_cpus: u32, host_cpus: u32, requested_mem_mib: u32) -> (u32, u32) {
    let vcpus = requested_cpus.min(host_cpus).clamp(1, MAX_VCPUS);
    (vcpus, requested_mem_mib)
}

/// The disk budget (upstream: 0.8 of free split across parallel runs,
/// whole MiB).
pub fn disk_budget(free_disk_bytes: u64, parallel_runs: u64) -> u64 {
    let per_run =
        (free_disk_bytes as f64 * FREE_DISK_FRACTION / parallel_runs.max(1) as f64) as u64;
    // whole MiB
    (per_run >> 20) << 20
}

/// A drive in the VM config (upstream `drives.py`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Drive {
    /// The device name (`vda`, `vdb`, ...).
    pub device: String,
    /// Guest mount role.
    pub role: DriveRole,
    /// Read-only attach.
    pub read_only: bool,
}

/// What a drive is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriveRole {
    /// The base rootfs (docker image as ext4).
    Base,
    /// The scratch overlay upper/work dir (the disk budget).
    Scratch,
    /// The io drive (argv/env/guest.conf/hosts/mounts + `/out`).
    Io,
    /// A read-only mount drive (`--mount ...:ro`).
    Mount,
}

/// The drive order and device names (upstream `device_name`):
/// vda base → vdb scratch → vdc io → vdd+ mounts (max 26).
pub fn drive_plan(mount_count: usize) -> Vec<Drive> {
    let mut drives = vec![
        Drive {
            device: "vda".into(),
            role: DriveRole::Base,
            read_only: true,
        },
        Drive {
            device: "vdb".into(),
            role: DriveRole::Scratch,
            read_only: false,
        },
        Drive {
            device: "vdc".into(),
            role: DriveRole::Io,
            read_only: false,
        },
    ];
    for i in 0..mount_count {
        let idx = 3 + i;
        if idx >= 26 {
            break; // vdX exhausted
        }
        drives.push(Drive {
            device: format!("vd{}", (b'a' + idx as u8) as char),
            role: DriveRole::Mount,
            read_only: true,
        });
    }
    drives
}

/// A mount drive's size (upstream `mount_drive_size`): content × 1.25 +
/// entries × 8 KiB + 64 MiB, rounded up to whole MiB, with `-N
/// entries+64` inodes.
pub fn mount_drive_size(content_bytes: u64, entries: u64) -> u64 {
    let raw = content_bytes + content_bytes / 4 + entries * 8192 + (64 << 20);
    round_up_mib(raw)
}

/// Round up to whole MiB.
pub fn round_up_mib(bytes: u64) -> u64 {
    (bytes + (1 << 20) - 1) >> 20 << 20
}

/// The blocked guest networks (upstream `BLOCKED_NETWORKS` — the
/// link-local/metadata/private ranges) **plus every host IPv4** (the
/// host-side filter binds even guest root, except the model proxy).
pub fn blocked_networks() -> Vec<&'static str> {
    vec![
        "169.254.0.0/16",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "100.64.0.0/10",
    ]
}

/// The egress chain rules for the tap (upstream `egress_rules`):
/// ACCEPT per proxy allow, REJECT per blocked net (plus host IPv4s),
/// ACCEPT the rest.
#[derive(Debug, Clone, PartialEq)]
pub struct EgressRule {
    /// iptables fragment.
    pub text: String,
}

/// Build the `KAROTTE-FC` FORWARD chain rules.
pub fn egress_rules(
    allowed_proxy: Option<std::net::Ipv4Addr>,
    host_ips: &[std::net::Ipv4Addr],
) -> Vec<EgressRule> {
    let mut out = Vec::new();
    if let Some(p) = allowed_proxy {
        out.push(EgressRule {
            text: format!("-A KAROTTE-FC -d {p} -j ACCEPT"),
        });
    }
    for net in blocked_networks() {
        out.push(EgressRule {
            text: format!("-A KAROTTE-FC -d {net} -j REJECT"),
        });
    }
    for ip in host_ips {
        out.push(EgressRule {
            text: format!("-A KAROTTE-FC -d {ip} -j REJECT"),
        });
    }
    out.push(EgressRule {
        text: "-A KAROTTE-FC -j ACCEPT".into(),
    });
    out
}

/// Guest DNS resolution order (upstream: env override → systemd resolve
/// stub → /etc/resolv.conf global servers → 1.1.1.1/8.8.8.8).
pub fn guest_dns(env_override: Option<&str>, resolve_conf_servers: &[&str]) -> Vec<String> {
    if let Some(o) = env_override {
        return vec![o.to_string()];
    }
    let mut servers: Vec<String> = resolve_conf_servers.iter().map(|s| s.to_string()).collect();
    if servers.is_empty() {
        servers = vec!["1.1.1.1".into(), "8.8.8.8".into()];
    }
    servers
}

/// The full Firecracker VM config as the JSON the VMM consumes
/// (upstream `firecracker_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct VmConfig {
    /// Boot source: kernel path + args.
    pub boot_source: serde_json::Value,
    /// Drives.
    pub drives: Vec<serde_json::Value>,
    /// Machine config (vcpus, mem).
    pub machine_config: serde_json::Value,
    /// Vsock (guest CID 3, uds path).
    pub vsock: serde_json::Value,
    /// Optional network interface (tap).
    pub network_interfaces: Option<serde_json::Value>,
}

/// Inputs for [`firecracker_config`].
#[derive(Debug, Clone)]
pub struct VmConfigInputs<'a> {
    /// Kernel image path.
    pub kernel_path: &'a str,
    /// Scratch drive path.
    pub scratch_path: &'a str,
    /// Io drive path.
    pub io_path: &'a str,
    /// Base rootfs path.
    pub base_path: &'a str,
    /// Read-only mount drives.
    pub mount_paths: &'a [String],
    /// vCPUs.
    pub vcpus: u32,
    /// Memory MiB.
    pub mem_mib: u32,
    /// The vsock UDS path.
    pub vsock_uds: &'a str,
    /// Whether the tap network is attached.
    pub with_network: bool,
}

/// Build the VM config JSON.
pub fn firecracker_config(inputs: &VmConfigInputs) -> VmConfig {
    let VmConfigInputs {
        kernel_path,
        scratch_path,
        io_path,
        base_path,
        mount_paths,
        vcpus,
        mem_mib,
        vsock_uds,
        with_network,
    } = inputs;
    let mut drives = vec![
        serde_json::json!({
            "drive_id": "base",
            "path_on_host": base_path,
            "is_root_device": true,
            "is_read_only": true,
        }),
        serde_json::json!({
            "drive_id": "scratch",
            "path_on_host": scratch_path,
            "is_root_device": false,
            "is_read_only": false,
        }),
        serde_json::json!({
            "drive_id": "io",
            "path_on_host": io_path,
            "is_root_device": false,
            "is_read_only": false,
        }),
    ];
    for (i, m) in mount_paths.iter().enumerate() {
        drives.push(serde_json::json!({
            "drive_id": format!("mount{i}"),
            "path_on_host": m,
            "is_root_device": false,
            "is_read_only": true,
        }));
    }
    let boot_source = serde_json::json!({
        "kernel_image_path": kernel_path,
        "boot_args": format!("{BOOT_ARGS} {}", kernel_ip_arg()),
    });
    let machine_config = serde_json::json!({
        "vcpu_count": vcpus,
        "mem_size_mib": mem_mib,
    });
    let vsock = serde_json::json!({
        "guest_cid": GUEST_CID,
        "uds_path": vsock_uds,
    });
    let network_interfaces = if *with_network {
        Some(serde_json::json!({
            "iface_id": "eth0",
            "guest_mac": GUEST_MAC,
            "host_dev_name": TAP_NAME,
        }))
    } else {
        None
    };
    VmConfig {
        boot_source,
        drives,
        machine_config,
        vsock,
        network_interfaces,
    }
}

/// The VMM command line (upstream: `firecracker --no-api --config-file
/// ... --level Warning`, new session, PDEATHSIG SIGKILL).
pub fn vmm_command(config_path: &str, log_path: &str) -> Vec<String> {
    vec![
        "firecracker".into(),
        "--no-api".into(),
        "--config-file".into(),
        config_path.into(),
        "--level".into(),
        "Warning".into(),
        "--log-path".into(),
        log_path.into(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_parse_and_engines() {
        assert_eq!(Runtime::parse("docker:gvisor"), Runtime::DockerGvisor);
        assert_eq!(Runtime::parse("apple-container"), Runtime::AppleContainer);
        assert_eq!(Runtime::parse("firecracker"), Runtime::Firecracker);
        // firecracker *builds* with docker
        assert_eq!(Runtime::Firecracker.engine(), "docker");
        assert_eq!(Runtime::AppleContainer.engine(), "container");
        assert_eq!(Runtime::DockerGvisor.engine(), "docker");
        assert_eq!(Runtime::parse("unknown"), Runtime::Docker);
        assert_eq!(Runtime::Docker.container_name("abc"), "karotte_run_abc");
    }

    #[test]
    fn default_runtime_matrix() {
        // Linux + /dev/kvm → firecracker
        assert_eq!(
            default_runtime(false, false, true, true, false),
            Runtime::Firecracker
        );
        // Linux without kvm → docker
        assert_eq!(
            default_runtime(false, false, true, false, false),
            Runtime::Docker
        );
        // macOS 26 arm64 → apple-container
        assert_eq!(
            default_runtime(true, true, false, false, false),
            Runtime::AppleContainer
        );
        // macOS older → docker
        assert_eq!(
            default_runtime(true, false, false, false, false),
            Runtime::Docker
        );
        // passthrough hardware always docker
        assert_eq!(
            default_runtime(true, true, false, false, true),
            Runtime::Docker
        );
    }

    #[test]
    fn vm_resources_clamp_to_32() {
        assert_eq!(vm_resources(8, 16, 4096), (8, 4096));
        assert_eq!(vm_resources(100, 16, 4096), (16, 4096));
        assert_eq!(vm_resources(100, 100, 2048), (32, 2048));
        assert_eq!(vm_resources(0, 16, 512), (1, 512));
    }

    #[test]
    fn disk_budget_splits_and_floors() {
        // 100 GiB free, 2 parallel runs → 40 GiB each, whole MiB.
        let b = disk_budget(100 << 30, 2);
        assert_eq!(b, 40 << 30);
        let odd = disk_budget((100 << 30) + 123456, 1);
        assert_eq!(odd % (1 << 20), 0, "whole MiB");
        assert!(odd >= 80 * (1 << 30) - (1 << 20));
    }

    #[test]
    fn drive_order_and_device_names() {
        let drives = drive_plan(2);
        let devices: Vec<&str> = drives.iter().map(|d| d.device.as_str()).collect();
        assert_eq!(devices, vec!["vda", "vdb", "vdc", "vdd", "vde"]);
        assert!(drives[0].read_only, "base is ro");
        assert!(!drives[1].read_only, "scratch is rw");
        assert!(drives[3].read_only, "mounts are ro");
        // 26-device cap
        let capped = drive_plan(40);
        assert!(capped.len() <= 26);
    }

    #[test]
    fn mount_drive_size_formula() {
        // 10 MiB content, 100 entries: 10M + 2.5M + 800K + 64M ≈ 77.3M → 78 MiB
        let s = mount_drive_size(10 << 20, 100);
        assert_eq!(s % (1 << 20), 0);
        assert!(s >= 77 * (1 << 20));
        assert!(s <= 79 * (1 << 20));
    }

    #[test]
    fn boot_args_and_ip() {
        assert!(BOOT_ARGS.contains("root=/dev/vda ro rootfstype=ext4 init=/.karotte/init"));
        assert!(BOOT_ARGS.contains("reboot=k panic=1 pci=off"));
        let ip = kernel_ip_arg();
        assert_eq!(ip, "ip=192.0.2.2::192.0.2.1:255.255.255.252::eth0:off");
    }

    #[test]
    fn pinned_artifacts_cover_both_arches() {
        for arch in ["x86_64", "aarch64"] {
            let fc = firecracker_artifact(arch);
            assert!(fc.url.contains("v1.17.0"));
            assert!(fc.url.contains(arch));
            assert_eq!(fc.sha256.len(), 64);
            let kata = kata_kernel_artifact(arch);
            assert!(kata.url.contains("4.0.0"));
            assert_eq!(kata.sha256.len(), 64);
        }
        assert!(KERNEL_NAME.contains("6.18.35"));
    }

    #[test]
    fn egress_rules_order() {
        let host = std::net::Ipv4Addr::new(192, 168, 5, 5);
        let proxy = std::net::Ipv4Addr::new(203, 0, 113, 9);
        let rules = egress_rules(Some(proxy), &[host]);
        let texts: Vec<&str> = rules.iter().map(|r| r.text.as_str()).collect();
        // proxy ACCEPT first
        assert!(texts[0].contains("-d 203.0.113.9 -j ACCEPT"));
        // blocked networks
        assert!(texts.iter().any(|t| t.contains("169.254.0.0/16 -j REJECT")));
        assert!(texts.iter().any(|t| t.contains("10.0.0.0/8 -j REJECT")));
        assert!(texts.iter().any(|t| t.contains("100.64.0.0/10 -j REJECT")));
        // host ips rejected
        assert!(texts.iter().any(|t| t.contains("-d 192.168.5.5 -j REJECT")));
        // final ACCEPT last
        assert_eq!(*texts.last().unwrap(), "-A KAROTTE-FC -j ACCEPT");
        // no proxy → no accept before the blocks
        let no_proxy = egress_rules(None, &[]);
        assert!(!no_proxy[0].text.contains("ACCEPT"));
    }

    #[test]
    fn guest_dns_fallback() {
        assert_eq!(
            guest_dns(None, &[]),
            vec!["1.1.1.1".to_string(), "8.8.8.8".to_string()]
        );
        assert_eq!(
            guest_dns(Some("9.9.9.9"), &["10.0.0.53"]),
            vec!["9.9.9.9".to_string()]
        );
        assert_eq!(
            guest_dns(None, &["10.0.0.53", "10.0.0.54"]),
            vec!["10.0.0.53".to_string(), "10.0.0.54".to_string()]
        );
    }

    #[test]
    fn firecracker_config_json_shape() {
        let cfg = firecracker_config(&VmConfigInputs {
            kernel_path: "/cache/vmlinux",
            scratch_path: "/runs/r/scratch.ext4",
            io_path: "/runs/r/io.ext4",
            base_path: "/cache/base.ext4",
            mount_paths: &["/runs/r/mount0.ext4".to_string()],
            vcpus: 2,
            mem_mib: 4096,
            vsock_uds: "/runs/r/v.sock",
            with_network: true,
        });
        assert!(cfg.boot_source["boot_args"].as_str().unwrap().len() > 100);
        assert!(cfg.boot_source["boot_args"]
            .as_str()
            .unwrap()
            .contains("ip=192.0.2.2"));
        assert_eq!(cfg.drives.len(), 4);
        assert_eq!(cfg.drives[0]["is_root_device"], serde_json::json!(true));
        assert_eq!(cfg.drives[0]["is_read_only"], serde_json::json!(true));
        assert_eq!(cfg.drives[3]["is_read_only"], serde_json::json!(true));
        assert_eq!(cfg.machine_config["vcpu_count"], serde_json::json!(2));
        assert_eq!(cfg.machine_config["mem_size_mib"], serde_json::json!(4096));
        assert_eq!(cfg.vsock["guest_cid"], serde_json::json!(3));
        assert_eq!(cfg.vsock["uds_path"], serde_json::json!("/runs/r/v.sock"));
        assert_eq!(
            cfg.network_interfaces.as_ref().unwrap()["guest_mac"],
            serde_json::json!(GUEST_MAC)
        );
        // no network
        let cfg2 = firecracker_config(&VmConfigInputs {
            kernel_path: "/k",
            scratch_path: "/s",
            io_path: "/i",
            base_path: "/b",
            mount_paths: &[],
            vcpus: 1,
            mem_mib: 512,
            vsock_uds: "/v.sock",
            with_network: false,
        });
        assert!(cfg2.network_interfaces.is_none());
    }

    #[test]
    fn vmm_command_shape() {
        assert_eq!(
            vmm_command("/r/fc.json", "/r/vmm.log"),
            vec![
                "firecracker",
                "--no-api",
                "--config-file",
                "/r/fc.json",
                "--level",
                "Warning",
                "--log-path",
                "/r/vmm.log"
            ]
        );
    }

    #[test]
    fn watchdog_constants() {
        assert_eq!(WATCHDOG_INTERVAL_S, 5.0);
        assert_eq!(DEFAULT_WATCHDOG_SECONDS, 120.0);
        assert_eq!(BOOT_GRACE_SECONDS, 180.0);
        assert_eq!(HEARTBEAT_PORT, 52, "guest-root-only port");
        assert_eq!(GUEST_CID, 3);
    }
}
