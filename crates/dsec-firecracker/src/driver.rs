//! The real Firecracker driver behind dsec-runtime's `MicrovmDriver`
//! trait.
//!
//! Boot sequence per sandbox (the paper's MicroVM backend): launch the
//! `firecracker` process with a dedicated API socket, configure the
//! machine shape from the sandbox spec (`cpu_millicores` → vCPUs,
//! `mem_mib` → guest memory), attach the shared read-only EROFS
//! rootfs drive plus a per-VM writable scratch drive, and start the
//! instance. Pause/resume use the real VMM state machine, snapshots
//! map to `PUT /snapshots/create` (diff mode = the pack_diff
//! analogue), and destroy asks the guest to power off
//! (`SendCtrlAltDel`; the default boot args pair it with
//! `reboot=t panic=1`) before force-killing.
//!
//! The process-launch step is abstracted ([`VmmLauncher`]) so tests
//! drive the complete request sequencing against an in-process fake
//! VMM ([`crate::fake::FakeVmm`]) over a REAL unix socket.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use dsec_runtime::microvm::{MicrovmDriver, MicrovmHandle, SnapshotPaths};
use dsec_runtime::{BoxFut, Error as RtError, Result as RtResult, SandboxSpec};
use dsec_storage::erofs::ImageRegistry;

use crate::client::FcClient;
use crate::config::{detect, Capability, FirecrackerConfig};
use crate::rootfs::RootfsCache;

/// Launches a VMM serving the Firecracker API at `sock`.
pub trait VmmLauncher: Send + Sync {
    /// Returns the VMM process id (0 when no real process backs it).
    fn launch(&self, sock: &Path) -> std::io::Result<u32>;
}

/// Real launcher: spawns the firecracker binary.
#[derive(Debug, Clone)]
pub struct ProcessLauncher {
    pub binary: PathBuf,
}

impl VmmLauncher for ProcessLauncher {
    fn launch(&self, sock: &Path) -> std::io::Result<u32> {
        let child = Command::new(&self.binary)
            .arg("--api-sock")
            .arg(sock)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;
        Ok(child.id())
    }
}

/// Test launcher: binds a [`crate::fake::FakeVmm`] to the socket
/// instead of spawning a process. Every launched fake is retained and
/// queryable via [`fakes`](Self::fakes) for request-log assertions.
/// Requires a running tokio runtime context (boot always runs inside
/// one).
#[derive(Clone, Default)]
pub struct FakeLauncher {
    fakes: std::sync::Arc<std::sync::Mutex<Vec<std::sync::Arc<crate::fake::FakeVmm>>>>,
}

impl FakeLauncher {
    /// Every fake VMM launched so far, in launch order.
    pub fn fakes(&self) -> Vec<std::sync::Arc<crate::fake::FakeVmm>> {
        self.fakes.lock().expect("fake launcher poisoned").clone()
    }
}

impl VmmLauncher for FakeLauncher {
    fn launch(&self, sock: &Path) -> std::io::Result<u32> {
        let fake = crate::fake::FakeVmm::spawn(sock)?;
        self.fakes
            .lock()
            .expect("fake launcher poisoned")
            .push(fake);
        Ok(0)
    }
}

/// The Firecracker microVM driver.
pub struct FirecrackerDriver {
    config: FirecrackerConfig,
    launcher: Arc<dyn VmmLauncher>,
    registry: Arc<ImageRegistry>,
    rootfs_cache: Arc<RootfsCache>,
}

impl FirecrackerDriver {
    /// Real-process driver: the firecracker binary, KVM and kernel
    /// image must be resolvable (see [`crate::config::detect`]).
    pub fn new(config: FirecrackerConfig, registry: Arc<ImageRegistry>) -> Self {
        let launcher: Arc<dyn VmmLauncher> = Arc::new(ProcessLauncher {
            binary: config.binary.clone(),
        });
        FirecrackerDriver::with_launcher(config, registry, launcher)
    }

    /// Driver with an injected launcher (tests use [`FakeLauncher`]).
    pub fn with_launcher(
        config: FirecrackerConfig,
        registry: Arc<ImageRegistry>,
        launcher: Arc<dyn VmmLauncher>,
    ) -> Self {
        let rootfs_dir = config.rootfs_dir.clone();
        FirecrackerDriver {
            config,
            launcher,
            registry,
            rootfs_cache: Arc::new(RootfsCache::new(rootfs_dir)),
        }
    }

    pub fn config(&self) -> &FirecrackerConfig {
        &self.config
    }

    /// Host capability snapshot (binary, KVM, kernel).
    pub fn capability(&self) -> Capability {
        detect()
    }

    fn check_host_deps(&self) -> RtResult<()> {
        if self.config.allow_missing_host_deps {
            return Ok(());
        }
        let cap = detect();
        if !cap.ready() {
            return Err(RtError::Other(format!(
                "firecracker backend unavailable: {}",
                cap.blockers().join("; ")
            )));
        }
        Ok(())
    }

    fn vm_id(sid: u64) -> String {
        format!("sb-{sid:06x}")
    }
}

impl MicrovmDriver for FirecrackerDriver {
    fn name(&self) -> &'static str {
        "firecracker"
    }

    fn boot(&self, sid: u64, spec: &SandboxSpec) -> BoxFut<RtResult<MicrovmHandle>> {
        // Clone everything the 'static future needs (no borrows of self
        // or spec).
        let config = self.config.clone();
        let launcher = self.launcher.clone();
        let registry = self.registry.clone();
        let rootfs_cache = self.rootfs_cache.clone();
        let host_deps = self.check_host_deps();
        let image_id = spec.image_id.clone();
        let vcpus = ((spec.resources.cpu_millicores + 999) / 1000).max(1) as u32;
        let mem = spec.resources.mem_mib.max(64) as u32;
        Box::pin(async move {
            host_deps?;
            let vm_id = Self::vm_id(sid);
            let sock = config.api_sock_dir.join(format!("{vm_id}.sock"));
            let _ = std::fs::remove_file(&sock);
            let pid = launcher.launch(&sock)?;
            wait_sock(&sock, config.boot_timeout).await?;

            // Shared, materialized-once read-only rootfs (image cache
            // dedup semantics), plus a per-VM writable scratch drive.
            let image = registry
                .get(&image_id)
                .ok_or_else(|| RtError::ImageNotLocal(image_id.clone()))?
                .image()
                .clone();
            let rootfs = rootfs_cache.get_or_materialize(&image)?;
            let scratch = config.scratch_dir.join(format!("{vm_id}.scratch.img"));
            crate::rootfs::create_scratch(&scratch, 64 * 1024 * 1024)?;

            let mut client = FcClient::new(&sock, config.request_timeout);
            // 1. Machine shape from the sandbox spec.
            let body = format!(r#"{{"vcpu_count":{vcpus},"mem_size_mib":{mem},"smt":false}}"#);
            let r = client.put("/machine-config", &body).await?;
            if !r.is_success() {
                return Err(fault("machine-config", r));
            }
            // 2. Kernel.
            let boot = serde_json::json!({
                "kernel_image_path": config.kernel_image.display().to_string(),
                "boot_args": config.kernel_args,
            });
            let r = client.put("/boot-source", &boot.to_string()).await?;
            if !r.is_success() {
                return Err(fault("boot-source", r));
            }
            // 3. Drives: shared read-only rootfs + per-VM scratch.
            let root_body = serde_json::json!({
                "drive_id": "rootfs",
                "path_on_host": rootfs.display().to_string(),
                "is_root_device": true,
                "is_read_only": true,
            });
            let r = client.put("/drives/rootfs", &root_body.to_string()).await?;
            if !r.is_success() {
                return Err(fault("drives/rootfs", r));
            }
            let scratch_body = serde_json::json!({
                "drive_id": "scratch",
                "path_on_host": scratch.display().to_string(),
                "is_root_device": false,
                "is_read_only": false,
            });
            let r = client
                .put("/drives/scratch", &scratch_body.to_string())
                .await?;
            if !r.is_success() {
                return Err(fault("drives/scratch", r));
            }
            // 4. Start.
            let r = client
                .put("/actions", r#"{"action_type":"InstanceStart"}"#)
                .await?;
            if !r.is_success() {
                return Err(fault("InstanceStart", r));
            }
            Ok(MicrovmHandle {
                vm_id,
                api_sock: sock,
                pid,
                started: true,
            })
        })
    }

    fn pause(&self, vm: &MicrovmHandle) -> BoxFut<RtResult<()>> {
        let sock = vm.api_sock.clone();
        let timeout = self.config.request_timeout;
        Box::pin(async move {
            let mut client = FcClient::new(&sock, timeout);
            let r = client.patch("/vm", r#"{"state":"Paused"}"#).await?;
            ok_or_fault("vm Paused", r)
        })
    }

    fn resume(&self, vm: &MicrovmHandle) -> BoxFut<RtResult<()>> {
        let sock = vm.api_sock.clone();
        let timeout = self.config.request_timeout;
        Box::pin(async move {
            let mut client = FcClient::new(&sock, timeout);
            let r = client.patch("/vm", r#"{"state":"Resumed"}"#).await?;
            ok_or_fault("vm Resumed", r)
        })
    }

    fn snapshot(&self, vm: &MicrovmHandle, diff: bool) -> BoxFut<RtResult<SnapshotPaths>> {
        let sock = vm.api_sock.clone();
        let vm_id = vm.vm_id.clone();
        let dir = self.config.snapshot_dir.clone();
        let timeout = self.config.request_timeout;
        Box::pin(async move {
            let snapshot_path = dir.join(format!("{vm_id}.snap"));
            let mem_file_path = dir.join(format!("{vm_id}.mem"));
            let mut client = FcClient::new(&sock, timeout);
            let body = serde_json::json!({
                "snapshot_path": snapshot_path.display().to_string(),
                "mem_file_path": mem_file_path.display().to_string(),
                "diff": diff,
            });
            let r = client.put("/snapshots/create", &body.to_string()).await?;
            if !r.is_success() {
                return Err(fault("snapshots/create", r));
            }
            Ok(SnapshotPaths {
                snapshot_path,
                mem_file_path,
                diff_paths: Vec::new(),
            })
        })
    }

    fn restore(
        &self,
        snap: &SnapshotPaths,
        _spec: &SandboxSpec,
    ) -> BoxFut<RtResult<MicrovmHandle>> {
        // Clone out everything the 'static future needs.
        let snapshot_path = snap.snapshot_path.clone();
        let mem_file_path = snap.mem_file_path.clone();
        let timeout = self.config.request_timeout;
        let api_dir = self.config.api_sock_dir.clone();
        let launcher = self.launcher.clone();
        let host_deps = self.check_host_deps();
        Box::pin(async move {
            host_deps?;
            let stem = snapshot_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("rep")
                .to_string();
            let vm_id = format!("rep-{stem}");
            let sock = api_dir.join(format!("{vm_id}.sock"));
            let _ = std::fs::remove_file(&sock);
            let pid = launcher.launch(&sock)?;
            wait_sock(&sock, std::time::Duration::from_secs(15)).await?;
            let mut client = FcClient::new(&sock, timeout);
            let body = serde_json::json!({
                "snapshot_path": snapshot_path.display().to_string(),
                "mem_file_path": mem_file_path.display().to_string(),
                "diff_drive_ids": ["scratch"],
            });
            let r = client.put("/snapshots/load", &body.to_string()).await?;
            if !r.is_success() {
                return Err(fault("snapshots/load", r));
            }
            Ok(MicrovmHandle {
                vm_id,
                api_sock: sock,
                pid,
                started: true,
            })
        })
    }

    fn destroy(&self, vm: &MicrovmHandle) -> BoxFut<RtResult<()>> {
        let sock = vm.api_sock.clone();
        let pid = vm.pid;
        let timeout = self.config.request_timeout;
        Box::pin(async move {
            let mut client = FcClient::new(&sock, timeout);
            // Graceful exit first (boot args pair SendCtrlAltDel with
            // reboot=t/panic=1 → power off); 4xx on an already-exited
            // VMM is fine.
            let _ = client
                .put("/actions", r#"{"action_type":"SendCtrlAltDel"}"#)
                .await;
            if pid > 0 {
                for _ in 0..200 {
                    if !Path::new("/proc").join(pid.to_string()).exists() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                let _ = Command::new("kill")
                    .arg(pid.to_string())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status();
            }
            let _ = std::fs::remove_file(&sock);
            Ok(())
        })
    }
}

fn fault(step: &str, r: crate::client::FcResponse) -> RtError {
    RtError::Other(format!(
        "firecracker {step} failed ({}): {}",
        r.status,
        r.fault().unwrap_or_default()
    ))
}

fn ok_or_fault(step: &str, r: crate::client::FcResponse) -> RtResult<()> {
    if r.is_success() {
        Ok(())
    } else {
        Err(fault(step, r))
    }
}

/// Waits until the API socket exists and accepts connections.
async fn wait_sock(sock: &Path, timeout: Duration) -> RtResult<()> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if sock.exists() && tokio::net::UnixStream::connect(sock).await.is_ok() {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(RtError::Other(format!(
                "vmm api socket {} did not appear within {timeout:?}",
                sock.display()
            )));
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
