//! Node-side microVM driver contract.
//!
//! The paper's MicroVM backend is a Firecracker-class VMM: the default
//! `dsec-runtime` configuration simulates it deterministically through
//! the node's [`NodeLatencyProfile`] (the numbers the paper measures:
//! ~900 ms cold boot, ~4 s pause/resume checkpoint cycles). This module
//! defines the driver surface a **real** VMM implementation plugs into —
//! the `dsec-firecracker` crate provides one that drives an actual
//! Firecracker process over its Unix-socket API.
//!
//! Mapping to the paper's lifecycle operations:
//!
//! | DSec operation | Firecracker API |
//! |----------------|-----------------|
//! | create (MicroVM) | `PUT /machine-config`, `/boot-source`, `/drives`, `/actions InstanceStart` |
//! | pause | `PATCH /vm {"state":"Paused"}` |
//! | resume | `PATCH /vm {"state":"Resumed"}` |
//! | pack_diff (snapshot) | `PUT /snapshots/create` (diff mode) |
//! | replica / fast resume | `PUT /snapshots/load` |
//! | destroy | `SendCtrlAltDel` / process teardown |
//!
//! [`EdgeNode`] consults the driver whenever an instance carries a
//! [`MicrovmHandle`]; with no driver installed every path behaves
//! exactly as the latency-profile simulation (regression-tested).

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use crate::backend::SandboxSpec;
use crate::Result;

/// Boxed future used by the driver surface (object-safe async trait).
pub type BoxFut<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;

/// A booted microVM managed by a [`MicrovmDriver`].
#[derive(Debug, Clone)]
pub struct MicrovmHandle {
    /// Driver-local VM identifier.
    pub vm_id: String,
    /// Path of the VMM's management (API) socket.
    pub api_sock: PathBuf,
    /// PID of the VMM process (0 when driver-managed without a process).
    pub pid: u32,
    /// True once the guest reported started.
    pub started: bool,
}

/// Locations of a created snapshot (the `pack_diff` analogue).
#[derive(Debug, Clone, Default)]
pub struct SnapshotPaths {
    pub snapshot_path: PathBuf,
    pub mem_file_path: PathBuf,
    /// Diff-drive files produced in diff mode (base + working-state delta).
    pub diff_paths: Vec<PathBuf>,
}

/// Lifecycle driver for real microVMs.
///
/// Implementations must be usable from async contexts without blocking
/// the runtime; long operations (boot, snapshot) should await their VMM
/// honestly so latency accounting stays truthful.
pub trait MicrovmDriver: Send + Sync {
    /// Driver name (diagnostics; e.g. `"firecracker"`).
    fn name(&self) -> &'static str;

    /// Boots a microVM for `sid`; the returned handle is stored on the
    /// sandbox instance and consulted by pause/resume/destroy.
    fn boot(&self, sid: u64, spec: &SandboxSpec) -> BoxFut<Result<MicrovmHandle>>;

    /// Pauses the VM (real VMM pause; the paper's checkpoint window).
    fn pause(&self, vm: &MicrovmHandle) -> BoxFut<Result<()>>;

    /// Resumes the VM.
    fn resume(&self, vm: &MicrovmHandle) -> BoxFut<Result<()>>;

    /// Snapshots the VM. `diff` selects incremental mode — the
    /// pack_diff analogue: base + dirty delta instead of a full copy.
    fn snapshot(&self, vm: &MicrovmHandle, diff: bool) -> BoxFut<Result<SnapshotPaths>>;

    /// Restores a VM from a snapshot (replica / fast-resume path).
    fn restore(&self, snap: &SnapshotPaths, spec: &SandboxSpec) -> BoxFut<Result<MicrovmHandle>>;

    /// Tears the VM down and releases its resources.
    fn destroy(&self, vm: &MicrovmHandle) -> BoxFut<Result<()>>;
}

/// Simulated driver used when no real VMM is configured: sleeps the
/// node's latency profile so the observable timing matches the paper.
///
/// This exists so deployments can install a uniform driver object even
/// in simulation mode (the default `EdgeNode` path needs no driver at
/// all — the profile sleeps live in the backend factory).
pub struct SimulatedMicrovmDriver {
    /// Cold-boot latency applied by `boot`.
    pub boot_latency: std::time::Duration,
    pub pause_latency: std::time::Duration,
    pub resume_latency: std::time::Duration,
}

impl SimulatedMicrovmDriver {
    /// Latencies drawn from the node profile (one draw per operation at
    /// construction; matches the profile's statistical shape).
    pub fn from_profile(p: &dsec_storage::latency::NodeLatencyProfile) -> Self {
        SimulatedMicrovmDriver {
            boot_latency: p.microvm_create.sample(),
            pause_latency: p.pause.sample(),
            resume_latency: p.resume.sample(),
        }
    }
}

impl MicrovmDriver for SimulatedMicrovmDriver {
    fn name(&self) -> &'static str {
        "simulated"
    }

    fn boot(&self, sid: u64, _spec: &SandboxSpec) -> BoxFut<Result<MicrovmHandle>> {
        let d = self.boot_latency;
        Box::pin(async move {
            if !d.is_zero() {
                tokio::time::sleep(d).await;
            }
            Ok(MicrovmHandle {
                vm_id: format!("sb-{sid:06x}"),
                api_sock: PathBuf::new(),
                pid: 0,
                started: true,
            })
        })
    }

    fn pause(&self, _vm: &MicrovmHandle) -> BoxFut<Result<()>> {
        let d = self.pause_latency;
        Box::pin(async move {
            if !d.is_zero() {
                tokio::time::sleep(d).await;
            }
            Ok(())
        })
    }

    fn resume(&self, _vm: &MicrovmHandle) -> BoxFut<Result<()>> {
        let d = self.resume_latency;
        Box::pin(async move {
            if !d.is_zero() {
                tokio::time::sleep(d).await;
            }
            Ok(())
        })
    }

    fn snapshot(&self, _vm: &MicrovmHandle, _diff: bool) -> BoxFut<Result<SnapshotPaths>> {
        Box::pin(async { Ok(SnapshotPaths::default()) })
    }

    fn restore(&self, _snap: &SnapshotPaths, _spec: &SandboxSpec) -> BoxFut<Result<MicrovmHandle>> {
        Box::pin(async {
            Ok(MicrovmHandle {
                vm_id: String::new(),
                api_sock: PathBuf::new(),
                pid: 0,
                started: true,
            })
        })
    }

    fn destroy(&self, _vm: &MicrovmHandle) -> BoxFut<Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsec_storage::latency::NodeLatencyProfile;
    use std::time::Duration;

    #[tokio::test]
    async fn simulated_driver_applies_profile_latencies() {
        let driver = SimulatedMicrovmDriver {
            boot_latency: Duration::from_millis(20),
            pause_latency: Duration::from_millis(5),
            resume_latency: Duration::from_millis(3),
        };
        let spec = SandboxSpec::default();
        let t0 = std::time::Instant::now();
        let vm = driver.boot(7, &spec).await.unwrap();
        assert!(t0.elapsed() >= Duration::from_millis(18));
        assert_eq!(vm.vm_id, "sb-000007");
        assert!(vm.started);
        let t1 = std::time::Instant::now();
        driver.pause(&vm).await.unwrap();
        assert!(t1.elapsed() >= Duration::from_millis(4));
    }

    #[tokio::test]
    async fn simulated_driver_from_paper_profile() {
        let p = NodeLatencyProfile::paper(3);
        let d = SimulatedMicrovmDriver::from_profile(&p);
        // The paper's microVM cold-boot mean is ~900 ms; allow the
        // profile's own definition to lead (check it is substantial).
        assert!(d.boot_latency >= Duration::from_millis(500));
    }
}
