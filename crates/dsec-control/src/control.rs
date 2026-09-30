//! The control plane service core: ties IAM, registry, placement and the
//! provisioner together, mirroring the paper's stateless apiserver logic
//! (state lives in the registry; the apiserver is a thin REST shim).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use serde::Serialize;

use crate::error::{BoxFuture, Error, Result};
use crate::iam::{Action, Iam, Role};
use crate::metrics::MetricsRegistry;
use crate::model::{NodeInfo, Quota, SandboxRecord, SandboxSpec, SandboxState};
use crate::placement::{PlacementDecision, PlacementEngine, ScoreWeights};
use crate::ratelimit::{RateLimitConfig, RateLimiter};
use crate::registry::Registry;
use crate::watcher::Watcher;

/// What the runtime reports back after provisioning.
#[derive(Debug, Clone, Serialize)]
pub struct ProvisionedSandbox {
    pub sid: u64,
    pub state: SandboxState,
}

/// The contract the node-local runtime implements for the control plane
/// (in the real system this crosses the network; in tests it is an
/// in-process adapter over `dsec-runtime`'s EdgeNode).
pub trait Provisioner: Send + Sync {
    fn provision(
        &self,
        node_id: &str,
        spec: &SandboxSpec,
        sid: u64,
    ) -> BoxFuture<'static, Result<ProvisionedSandbox>>;
    fn pause(&self, sid: u64, node_id: &str) -> BoxFuture<'static, Result<()>>;
    fn resume(&self, sid: u64, node_id: &str) -> BoxFuture<'static, Result<()>>;
    fn destroy(&self, sid: u64, node_id: &str) -> BoxFuture<'static, Result<()>>;
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterSummary {
    pub revision: u64,
    pub nodes: usize,
    pub nodes_unhealthy: usize,
    pub sandboxes_total: usize,
    pub sandboxes_active: usize,
    pub sandboxes_paused: usize,
    pub cpu_millicores_available: i64,
    pub mem_mib_available: i64,
}

pub struct ControlPlane {
    pub registry: Registry,
    pub iam: Iam,
    pub placement: PlacementEngine,
    pub metrics: MetricsRegistry,
    rate_limiter: Mutex<RateLimiter>,
    provisioner: RwLock<Option<Arc<dyn Provisioner>>>,
    watcher: RwLock<Option<Arc<Watcher>>>,
    next_sid: AtomicU64,
}

fn epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl ControlPlane {
    pub fn new(seed: u64, k: usize) -> Self {
        let plane = ControlPlane {
            registry: Registry::new(),
            iam: Iam::new(),
            placement: PlacementEngine::new(k, seed, ScoreWeights::default()),
            metrics: MetricsRegistry::new(),
            rate_limiter: Mutex::new(RateLimiter::new(RateLimitConfig::default())),
            provisioner: RwLock::new(None),
            watcher: RwLock::new(None),
            next_sid: AtomicU64::new(1),
        };
        plane.metrics.set_gauge("dsec_cluster_revision", 0.0);
        plane
    }

    pub fn set_provisioner(&self, p: Arc<dyn Provisioner>) {
        *self.provisioner.write().expect("plane poisoned") = Some(p);
    }

    /// Overrides the API rate limit (benchmarks disable throttling).
    pub fn set_rate_limit(&self, config: RateLimitConfig) {
        *self.rate_limiter.lock().expect("rate limiter poisoned") = RateLimiter::new(config);
    }

    pub fn provisioner(&self) -> Option<Arc<dyn Provisioner>> {
        self.provisioner.read().expect("plane poisoned").clone()
    }

    pub fn set_watcher(&self, w: Arc<Watcher>) {
        *self.watcher.write().expect("plane poisoned") = Some(w);
    }

    pub fn watcher(&self) -> Option<Arc<Watcher>> {
        self.watcher.read().expect("plane poisoned").clone()
    }

    // -- projects & tokens --------------------------------------------------

    pub fn create_project(
        &self,
        parent: &str,
        segment: &str,
        quota: Quota,
    ) -> Result<crate::iam::Project> {
        let p = self
            .iam
            .create_project(parent, segment, quota, epoch_ms())?;
        self.metrics.incr("dsec_projects_created_total");
        Ok(p)
    }

    pub fn create_token(&self, project: &str, role: Role) -> Result<crate::iam::TokenInfo> {
        self.iam.create_token(project, role, epoch_ms())
    }

    // -- nodes ----------------------------------------------------------------

    /// Registers (or refreshes) a node snapshot.
    pub fn register_node(&self, info: NodeInfo) {
        self.metrics.incr("dsec_nodes_registered_total");
        self.registry.upsert_node(info);
    }

    /// Node heartbeat (from the apiserver).
    pub async fn heartbeat(&self, node_id: &str) -> Result<()> {
        if self.registry.node(node_id).is_none() {
            return Err(Error::NodeNotFound(node_id.to_string()));
        }
        if let Some(w) = self.watcher() {
            w.heartbeat(node_id).await;
        }
        self.metrics.incr("dsec_node_heartbeats_total");
        Ok(())
    }

    // -- sandboxes --------------------------------------------------------------

    /// Full creation path: authorize -> rate limit -> quota -> place ->
    /// provision -> record.
    pub async fn create_sandbox(&self, token: &str, spec: SandboxSpec) -> Result<SandboxRecord> {
        let t0 = std::time::Instant::now();
        self.iam
            .authorize(token, Action::CreateSandbox, &spec.project)?;
        if let Err(retry_ms) = self
            .rate_limiter
            .lock()
            .expect("rate limiter poisoned")
            .check(token)
        {
            self.metrics.incr("dsec_api_rate_limited_total");
            return Err(Error::RateLimited(retry_ms));
        }
        if !self.iam.project_exists(&spec.project) {
            return Err(Error::ProjectNotFound(spec.project.clone()));
        }
        self.iam
            .check_quota(&spec.project, &spec.resources, &self.registry)?;

        let nodes = self.registry.nodes();
        let decision: PlacementDecision = self.placement.place(&spec, &nodes)?;
        self.metrics.incr("dsec_placement_decisions_total");
        if decision.burst {
            self.metrics.incr("dsec_placement_burst_total");
        }

        let sid = self.next_sid.fetch_add(1, Ordering::Relaxed);
        // Provision (or inert mode when no provisioner is attached).
        let state = if let Some(p) = self.provisioner() {
            match p.provision(&decision.node_id, &spec, sid).await {
                Ok(provisioned) => provisioned.state,
                Err(e) => {
                    self.metrics.incr("dsec_sandboxes_failed_total");
                    return Err(e);
                }
            }
        } else {
            SandboxState::Ready
        };

        let now = epoch_ms();
        let record = SandboxRecord {
            sid,
            spec: spec.clone(),
            node_id: decision.node_id.clone(),
            state,
            created_epoch_ms: now,
            updated_epoch_ms: now,
        };
        self.registry.insert_sandbox(record.clone());
        // Project the placement onto the node snapshot (in-place, no
        // clone-modify-upsert round trip).
        self.registry.adjust_node_resources(
            &decision.node_id,
            -spec.resources.cpu_millicores,
            -spec.resources.mem_mib,
            -1,
        );
        self.metrics.incr("dsec_sandboxes_created_total");
        let elapsed = t0.elapsed().as_millis() as u64;
        self.metrics
            .set_gauge("dsec_create_last_ms", elapsed as f64);
        Ok(record)
    }

    pub async fn pause_sandbox(&self, token: &str, sid: u64) -> Result<SandboxRecord> {
        let record = self
            .registry
            .sandbox(sid)
            .ok_or(Error::SandboxNotFound(sid))?;
        self.iam
            .authorize(token, Action::PauseResume, &record.spec.project)?;
        if record.state != SandboxState::Ready {
            return Err(Error::InvalidArgument(format!(
                "cannot pause sandbox in state {}",
                record.state.as_str()
            )));
        }
        if let Some(p) = self.provisioner() {
            p.pause(sid, &record.node_id).await?;
        }
        let updated = self
            .registry
            .update_sandbox_state(sid, SandboxState::Paused)
            .ok_or(Error::SandboxNotFound(sid))?;
        self.metrics.incr("dsec_sandboxes_paused_total");
        Ok(updated)
    }

    pub async fn resume_sandbox(&self, token: &str, sid: u64) -> Result<SandboxRecord> {
        let record = self
            .registry
            .sandbox(sid)
            .ok_or(Error::SandboxNotFound(sid))?;
        self.iam
            .authorize(token, Action::PauseResume, &record.spec.project)?;
        if record.state != SandboxState::Paused {
            return Err(Error::InvalidArgument(format!(
                "cannot resume sandbox in state {}",
                record.state.as_str()
            )));
        }
        if let Some(p) = self.provisioner() {
            p.resume(sid, &record.node_id).await?;
        }
        let updated = self
            .registry
            .update_sandbox_state(sid, SandboxState::Ready)
            .ok_or(Error::SandboxNotFound(sid))?;
        self.metrics.incr("dsec_sandboxes_resumed_total");
        Ok(updated)
    }

    pub async fn destroy_sandbox(&self, token: &str, sid: u64) -> Result<SandboxRecord> {
        let record = self
            .registry
            .sandbox(sid)
            .ok_or(Error::SandboxNotFound(sid))?;
        self.iam
            .authorize(token, Action::DestroySandbox, &record.spec.project)?;
        self.registry
            .update_sandbox_state(sid, SandboxState::Destroying);
        if let Some(p) = self.provisioner() {
            let _ = p.destroy(sid, &record.node_id).await;
        }
        let removed = self
            .registry
            .remove_sandbox(sid)
            .ok_or(Error::SandboxNotFound(sid))?;
        // Return resources to the node snapshot (in-place).
        self.registry.adjust_node_resources(
            &removed.node_id,
            removed.spec.resources.cpu_millicores,
            removed.spec.resources.mem_mib,
            1,
        );
        self.metrics.incr("dsec_sandboxes_destroyed_total");
        Ok(removed)
    }

    // -- cluster ops ---------------------------------------------------------

    /// Evicts every sandbox on a node (watcher path). Returns count.
    pub async fn evict_node(&self, node_id: &str) -> Result<usize> {
        let victims = self.registry.remove_node(node_id);
        if let Some(p) = self.provisioner() {
            for v in &victims {
                let _ = p.destroy(v.sid, node_id).await;
            }
        }
        Ok(victims.len())
    }

    /// Preempts `count` sandboxes in a project subtree (paused first,
    /// lowest priority first). Returns the preempted records.
    pub async fn preempt(
        &self,
        token: &str,
        project: &str,
        count: usize,
    ) -> Result<Vec<SandboxRecord>> {
        self.iam.authorize(token, Action::Admin, project)?;
        let records: Vec<SandboxRecord> = self
            .registry
            .sandboxes()
            .into_iter()
            .filter(|r| {
                r.spec.project == project
                    || r.spec
                        .project
                        .strip_prefix(project)
                        .is_some_and(|rest| rest.starts_with('/'))
            })
            .collect();
        let victims = crate::watcher::preemption_victims(&records, count);
        let mut out = Vec::new();
        for v in victims {
            if let Ok(r) = self.destroy_sandbox(token, v.sid).await {
                out.push(r);
            }
        }
        self.metrics
            .incr_by("dsec_sandboxes_preempted_total", out.len() as u64);
        Ok(out)
    }

    pub fn cluster_summary(&self) -> ClusterSummary {
        let nodes = self.registry.nodes();
        let sandboxes = self.registry.sandboxes();
        let active = sandboxes
            .iter()
            .filter(|r| r.state == SandboxState::Ready)
            .count();
        let paused = sandboxes
            .iter()
            .filter(|r| r.state == SandboxState::Paused)
            .count();
        ClusterSummary {
            revision: self.registry.revision(),
            nodes: nodes.len(),
            nodes_unhealthy: nodes
                .iter()
                .filter(|n| n.status != crate::model::NodeStatus::Healthy)
                .count(),
            sandboxes_total: sandboxes.len(),
            sandboxes_active: active,
            sandboxes_paused: paused,
            cpu_millicores_available: nodes.iter().map(|n| n.cpu_available).sum(),
            mem_mib_available: nodes.iter().map(|n| n.mem_available).sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Locality, NodeStatus, Resources};

    /// Inert provisioner that tracks calls.
    struct MockProvisioner {
        calls: Arc<std::sync::atomic::AtomicU64>,
    }

    impl Provisioner for MockProvisioner {
        fn provision(
            &self,
            _node: &str,
            _spec: &SandboxSpec,
            sid: u64,
        ) -> BoxFuture<'static, Result<ProvisionedSandbox>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move {
                Ok(ProvisionedSandbox {
                    sid,
                    state: SandboxState::Ready,
                })
            })
        }
        fn pause(&self, _sid: u64, _node: &str) -> BoxFuture<'static, Result<()>> {
            Box::pin(async { Ok(()) })
        }
        fn resume(&self, _sid: u64, _node: &str) -> BoxFuture<'static, Result<()>> {
            Box::pin(async { Ok(()) })
        }
        fn destroy(&self, _sid: u64, _node: &str) -> BoxFuture<'static, Result<()>> {
            Box::pin(async { Ok(()) })
        }
    }

    fn node(id: &str, cpu: i64, mem: i64) -> NodeInfo {
        NodeInfo {
            node_id: id.to_string(),
            cpu_millicores: cpu,
            mem_mib: mem,
            max_sandboxes: 100,
            cpu_available: cpu,
            mem_available: mem,
            slots_available: 100,
            status: NodeStatus::Healthy,
            locality: Locality::Local,
            images: vec!["dsec/agent-base".to_string()],
            labels: Default::default(),
            admitted_projects: vec!["root".to_string()],
            cost_multiplier: 1.0,
        }
    }

    fn plane_with_nodes() -> Arc<ControlPlane> {
        let plane = Arc::new(ControlPlane::new(42, 4));
        plane.register_node(node("n1", 4000, 4096));
        plane.register_node(node("n2", 4000, 4096));
        plane
    }

    #[tokio::test]
    async fn full_creation_flow() {
        let plane = plane_with_nodes();
        let calls = Arc::new(std::sync::atomic::AtomicU64::new(0));
        plane.set_provisioner(Arc::new(MockProvisioner {
            calls: calls.clone(),
        }));
        let token = plane.create_token("root", Role::Admin).unwrap().token;
        let spec = SandboxSpec::default();
        let rec = plane.create_sandbox(&token, spec.clone()).await.unwrap();
        assert_eq!(rec.state, SandboxState::Ready);
        assert!(rec.node_id == "n1" || rec.node_id == "n2");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        // Node availability was decremented.
        let n = plane.registry.node(&rec.node_id).unwrap();
        assert_eq!(n.cpu_available, 4000 - spec.resources.cpu_millicores);
        // Pause -> resume -> destroy lifecycle.
        plane.pause_sandbox(&token, rec.sid).await.unwrap();
        assert_eq!(
            plane.registry.sandbox(rec.sid).unwrap().state,
            SandboxState::Paused
        );
        plane.resume_sandbox(&token, rec.sid).await.unwrap();
        plane.destroy_sandbox(&token, rec.sid).await.unwrap();
        assert!(plane.registry.sandbox(rec.sid).is_none());
        // Resources returned.
        let n2 = plane.registry.node(&rec.node_id).unwrap();
        assert_eq!(
            n2.cpu_available,
            n.cpu_available + spec.resources.cpu_millicores
        );
    }

    #[tokio::test]
    async fn quota_rejected_before_provision() {
        let plane = plane_with_nodes();
        let calls = Arc::new(std::sync::atomic::AtomicU64::new(0));
        plane.set_provisioner(Arc::new(MockProvisioner {
            calls: calls.clone(),
        }));
        plane
            .create_project("root", "tiny", Quota::limited(1000, 1024, 1))
            .unwrap();
        let token = plane.create_token("root/tiny", Role::Writer).unwrap().token;
        let spec = SandboxSpec {
            project: "root/tiny".into(),
            ..Default::default()
        };
        // 1st fits, 2nd exceeds sandbox quota.
        plane.create_sandbox(&token, spec.clone()).await.unwrap();
        assert!(matches!(
            plane.create_sandbox(&token, spec).await,
            Err(Error::QuotaExceeded { .. })
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 1); // never provisioned the rejected one
    }

    #[tokio::test]
    async fn unauthorized_token_rejected() {
        let plane = plane_with_nodes();
        let reader = plane.create_token("root", Role::Reader).unwrap().token;
        assert!(matches!(
            plane.create_sandbox(&reader, SandboxSpec::default()).await,
            Err(Error::Unauthorized(_))
        ));
    }

    #[tokio::test]
    async fn eviction_removes_records() {
        let plane = plane_with_nodes();
        let token = plane.create_token("root", Role::Writer).unwrap().token;
        for i in 0..3 {
            let _ = plane
                .create_sandbox(&token, SandboxSpec::default())
                .await
                .unwrap();
            let _ = i;
        }
        let count = plane.evict_node("n1").await.unwrap() + plane.evict_node("n2").await.unwrap();
        assert_eq!(count, 3);
        assert_eq!(plane.registry.sandboxes().len(), 0);
        assert!(plane.registry.node("n1").is_none());
    }

    #[tokio::test]
    async fn preemption_order() {
        let plane = plane_with_nodes();
        let token = plane.create_token("root", Role::Admin).unwrap().token;
        // One paused, one low priority.
        let a = plane
            .create_sandbox(
                &token,
                SandboxSpec {
                    priority: 5,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let low = plane
            .create_sandbox(
                &token,
                SandboxSpec {
                    priority: 1,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let _ = plane
            .create_sandbox(
                &token,
                SandboxSpec {
                    priority: 9,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        plane.pause_sandbox(&token, a.sid).await.unwrap();
        let _ = plane.pause_sandbox(&token, low.sid).await.unwrap();
        let victims = plane.preempt(&token, "root", 2).await.unwrap();
        assert_eq!(victims.len(), 2);
        // Paused + low priority first.
        assert!(victims.iter().any(|v| v.sid == low.sid));
        assert!(victims.iter().any(|v| v.sid == a.sid));
    }

    #[tokio::test]
    async fn burst_when_local_full() {
        let plane = Arc::new(ControlPlane::new(42, 4));
        // One tiny local node + one cloud node.
        plane.register_node(node("local", 500, 512));
        plane.register_node(NodeInfo {
            node_id: "cloud".into(),
            cpu_millicores: 8000,
            mem_mib: 8192,
            max_sandboxes: 100,
            cpu_available: 8000,
            mem_available: 8192,
            slots_available: 100,
            status: NodeStatus::Healthy,
            locality: Locality::Cloud,
            images: vec!["dsec/agent-base".into()],
            labels: Default::default(),
            admitted_projects: vec!["root".into()],
            cost_multiplier: 3.0,
        });
        let token = plane.create_token("root", Role::Writer).unwrap().token;
        // First fill the local node.
        let local_spec = SandboxSpec {
            resources: Resources {
                cpu_millicores: 500,
                mem_mib: 256,
            },
            allow_burst: true,
            ..Default::default()
        };
        let a = plane
            .create_sandbox(&token, local_spec.clone())
            .await
            .unwrap();
        assert_eq!(a.node_id, "local");
        // Second must burst to cloud.
        let b = plane.create_sandbox(&token, local_spec).await.unwrap();
        assert_eq!(b.node_id, "cloud");
        // Without burst flag: no candidate.
        let noc = SandboxSpec {
            resources: Resources {
                cpu_millicores: 500,
                mem_mib: 256,
            },
            allow_burst: false,
            ..Default::default()
        };
        assert!(matches!(
            plane.create_sandbox(&token, noc).await,
            Err(Error::NoCandidate(_))
        ));
    }
}
