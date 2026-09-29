//! Sandbox backends: FnCall (pre-created containers), Container, MicroVM
//! and FullVM.
//!
//! All four share the same guest model (layered filesystem + Chronus
//! sessions); what differs is creation latency and provisioning cost, which
//! is exactly the axis the paper explores. FnCall hands out pre-created
//! prepared guests from a warm pool (~5 ms path), Container builds a fresh
//! guest (~80 ms), MicroVM pays the Firecracker-class boot (~900 ms), and
//! FullVM pays a full QEMU boot (~8 s).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::chronus::Chronus;
use crate::resource::{ResourceRequest, SandboxGovernor};
use crate::state::{SandboxState, StateCell};
use dsec_storage::latency::NodeLatencyProfile;

/// Which backend materializes a sandbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    Fncall,
    Container,
    Microvm,
    Fullvm,
}

impl BackendKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendKind::Fncall => "fncall",
            BackendKind::Container => "container",
            BackendKind::Microvm => "microvm",
            BackendKind::Fullvm => "fullvm",
        }
    }
}

/// Runtime-side sandbox specification (what the Edge receives).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SandboxSpec {
    pub image_id: String,
    pub backend: BackendKind,
    pub resources: ResourceRequest,
    pub project: String,
    pub env: HashMap<String, String>,
    pub labels: HashMap<String, String>,
    /// SCHED_IDLE class (agent sandboxes default to idle).
    pub cpu_idle: bool,
}

impl Default for SandboxSpec {
    fn default() -> Self {
        SandboxSpec {
            image_id: "dsec/agent-base".to_string(),
            backend: BackendKind::Fncall,
            resources: ResourceRequest::default(),
            project: "root".to_string(),
            env: HashMap::new(),
            labels: HashMap::new(),
            cpu_idle: true,
        }
    }
}

/// A running sandbox: guest state + lifecycle cell + governor.
#[derive(Debug)]
pub struct BackendInstance {
    pub sid: u64,
    pub kind: BackendKind,
    pub state: StateCell,
    pub created_epoch_ms: u64,
    pub governor: SandboxGovernor,
    pub chronus: Chronus,
    /// Overlay device (for pack_diff / replication).
    pub overlay: Arc<dsec_storage::overlay::OverlayDev>,
    pub hostname: String,
}

impl BackendInstance {
    pub fn state(&self) -> SandboxState {
        self.state.get()
    }

    pub fn mem_mib(&self) -> i64 {
        self.governor.request.mem_mib
    }

    /// Working-state snapshot (paper: pack_diff). Carries dirty blocks
    /// plus the layered-FS metadata so replicas see identical files.
    pub fn snapshot_diff(&self, epoch_ms: u64) -> dsec_storage::packdiff::DiffPack {
        self.chronus.fs().snapshot_diff(epoch_ms)
    }

    /// Applies a working-state pack (replica path).
    pub fn apply_diff(&self, pack: &dsec_storage::packdiff::DiffPack) -> crate::Result<()> {
        Ok(self.chronus.fs().apply_diff(pack)?)
    }

    pub fn dirty_bytes(&self) -> u64 {
        self.overlay.dirty_bytes()
    }
}

/// One warm guest in the FnCall pool.
struct Prepared {
    overlay: Arc<dsec_storage::overlay::OverlayDev>,
    fs: Arc<dsec_storage::imagefs::LayeredImage>,
}

/// Creates backend instances; owns the FnCall warm pool.
pub struct BackendFactory {
    registry: Arc<dsec_storage::erofs::ImageRegistry>,
    latencies: NodeLatencyProfile,
    fncall_target: usize,
    pools: Mutex<HashMap<String, Vec<Prepared>>>,
    pool_hits: AtomicU64,
    pool_misses: AtomicU64,
}

impl BackendFactory {
    pub fn new(
        registry: Arc<dsec_storage::erofs::ImageRegistry>,
        latencies: NodeLatencyProfile,
        fncall_target: usize,
    ) -> Self {
        BackendFactory {
            registry,
            latencies,
            fncall_target,
            pools: Mutex::new(HashMap::new()),
            pool_hits: AtomicU64::new(0),
            pool_misses: AtomicU64::new(0),
        }
    }

    pub fn registry(&self) -> &Arc<dsec_storage::erofs::ImageRegistry> {
        &self.registry
    }

    pub fn latencies(&self) -> &NodeLatencyProfile {
        &self.latencies
    }

    /// Fills the FnCall pool for an image to target depth.
    pub fn prewarm(&self, image_id: &str, count: usize) {
        let Some(loader) = self.registry.get(image_id) else {
            return;
        };
        let mut pools = self.pools.lock().expect("pool poisoned");
        let pool = pools.entry(image_id.to_string()).or_default();
        for _ in 0..count {
            if pool.len() >= self.fncall_target {
                break;
            }
            let overlay = Arc::new(dsec_storage::overlay::OverlayDev::new(loader.clone()));
            let fs = Arc::new(dsec_storage::imagefs::LayeredImage::new(overlay.clone()));
            pool.push(Prepared { overlay, fs });
        }
    }

    pub fn pool_depth(&self, image_id: &str) -> usize {
        self.pools
            .lock()
            .expect("pool poisoned")
            .get(image_id)
            .map(|p| p.len())
            .unwrap_or(0)
    }

    pub fn pool_stats(&self) -> (u64, u64) {
        (
            self.pool_hits.load(Ordering::Relaxed),
            self.pool_misses.load(Ordering::Relaxed),
        )
    }

    async fn sleep(model: &dsec_storage::latency::LatencyModel) {
        let d = model.sample();
        if !d.is_zero() {
            tokio::time::sleep(d).await;
        }
    }

    /// Materializes a sandbox. The latency injected follows the backend
    /// kind; FnCall pops from the warm pool when possible.
    pub async fn create(
        &self,
        sid: u64,
        spec: &SandboxSpec,
    ) -> crate::Result<Arc<BackendInstance>> {
        let loader = self
            .registry
            .get(&spec.image_id)
            .ok_or_else(|| crate::Error::ImageNotLocal(spec.image_id.clone()))?;

        let prepared: Option<Prepared> = if spec.backend == BackendKind::Fncall {
            let mut pools = self.pools.lock().expect("pool poisoned");
            let hit = pools.get_mut(&spec.image_id).and_then(|p| p.pop());
            if hit.is_some() {
                self.pool_hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.pool_misses.fetch_add(1, Ordering::Relaxed);
            }
            hit
        } else {
            None
        };

        match spec.backend {
            BackendKind::Fncall => Self::sleep(&self.latencies.fncall_create).await,
            BackendKind::Container => Self::sleep(&self.latencies.container_create).await,
            BackendKind::Microvm => Self::sleep(&self.latencies.microvm_create).await,
            BackendKind::Fullvm => Self::sleep(&self.latencies.fullvm_create).await,
        }

        // Replenish the FnCall pool in the background (paper: pre-created
        // container pool refilled asynchronously).
        if spec.backend == BackendKind::Fncall {
            let image_id = spec.image_id.clone();
            let this = self as &BackendFactory;
            let target = self.fncall_target;
            let depth = self.pool_depth(&image_id);
            if depth < target {
                this.prewarm(&image_id, target - depth);
            }
        }

        let (overlay, fs) = match prepared {
            Some(p) => (p.overlay, p.fs),
            None => {
                let overlay = Arc::new(dsec_storage::overlay::OverlayDev::new(loader.clone()));
                let fs = Arc::new(dsec_storage::imagefs::LayeredImage::new(overlay.clone()));
                (overlay, fs)
            }
        };

        let hostname = format!("sb-{:06x}", sid);
        let mut env = spec.env.clone();
        env.insert("DSEC_SANDBOX_ID".to_string(), sid.to_string());
        env.insert("DSEC_NODE_HOSTNAME".to_string(), hostname.clone());
        let chronus = Chronus::new(fs, hostname.clone());
        chronus.sessions.seed_default(env);

        let governor = SandboxGovernor {
            request: spec.resources,
            cpu_class: if spec.cpu_idle {
                crate::resource::CpuClass::Idle
            } else {
                crate::resource::CpuClass::Normal
            },
            policy_profile: format!("dsec-{}", spec.project),
        };

        let epoch_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Ok(Arc::new(BackendInstance {
            sid,
            kind: spec.backend,
            state: StateCell::creating(),
            created_epoch_ms: epoch_ms,
            governor,
            chronus,
            overlay,
            hostname,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsec_storage::cache::LruBlockCache;
    use dsec_storage::erofs::{ErofsImageBuilder, ImageRegistry};
    use dsec_storage::latency::{LatencyModel, NodeLatencyProfile};
    use std::time::Duration;

    fn factory(latency: NodeLatencyProfile, warm: usize) -> BackendFactory {
        let image = Arc::new(ErofsImageBuilder::agent_base().build());
        let mut reg = ImageRegistry::default();
        reg.register(
            image,
            Arc::new(LruBlockCache::new(64)),
            LatencyModel::fixed(Duration::ZERO),
        );
        let reg = Arc::new(reg);
        let f = BackendFactory::new(reg, latency, warm);
        f.prewarm("dsec/agent-base", warm);
        f
    }

    #[tokio::test]
    async fn creates_all_backends() {
        let f = factory(NodeLatencyProfile::zero(), 0);
        for kind in [
            BackendKind::Fncall,
            BackendKind::Container,
            BackendKind::Microvm,
            BackendKind::Fullvm,
        ] {
            let spec = SandboxSpec {
                backend: kind,
                ..Default::default()
            };
            let inst = f.create(kind_slot(), &spec).await.unwrap();
            assert_eq!(inst.kind, kind);
            assert_eq!(inst.state(), SandboxState::Creating);
        }
    }

    fn kind_slot() -> u64 {
        1
    }

    #[tokio::test]
    async fn fncall_pool_hit_and_miss() {
        let f = factory(NodeLatencyProfile::zero(), 2);
        assert_eq!(f.pool_depth("dsec/agent-base"), 2);
        // First two come from the pool.
        let a = f.create(1, &SandboxSpec::default()).await.unwrap();
        let b = f.create(2, &SandboxSpec::default()).await.unwrap();
        assert_eq!(a.sid, 1);
        assert_eq!(b.sid, 2);
        let (hits, _) = f.pool_stats();
        assert_eq!(hits, 2);
        // Pool was replenished to target after each pop.
        assert_eq!(f.pool_depth("dsec/agent-base"), 2);
    }

    #[tokio::test]
    async fn missing_image_is_rejected() {
        let f = factory(NodeLatencyProfile::zero(), 0);
        let spec = SandboxSpec {
            image_id: "dsec/ghost".into(),
            ..Default::default()
        };
        assert!(matches!(
            f.create(1, &spec).await,
            Err(crate::Error::ImageNotLocal(_))
        ));
    }

    #[tokio::test]
    async fn creation_latency_follows_profile() {
        let f = factory(NodeLatencyProfile::paper(7), 0);
        let spec = SandboxSpec {
            backend: BackendKind::Microvm,
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let _ = f.create(1, &spec).await.unwrap();
        assert!(
            t0.elapsed() >= Duration::from_millis(800),
            "microvm path too fast"
        );
    }

    #[tokio::test]
    async fn instance_supports_packdiff() {
        let f = factory(NodeLatencyProfile::zero(), 0);
        let inst = f.create(9, &SandboxSpec::default()).await.unwrap();
        // Write through the guest fs to dirty a block.
        inst.chronus
            .fs()
            .write_file("/tmp/warm.bin", b"working state")
            .await
            .unwrap();
        assert!(inst.dirty_bytes() >= 4096);
        let pack = inst.snapshot_diff(1);
        assert_eq!(pack.blocks.len(), 1);
        // Fresh replica receives the same working state.
        let replica = f.create(10, &SandboxSpec::default()).await.unwrap();
        replica.apply_diff(&pack).unwrap();
        let data = replica
            .chronus
            .fs()
            .read_file("/tmp/warm.bin")
            .await
            .unwrap();
        assert_eq!(data, b"working state");
    }
}
