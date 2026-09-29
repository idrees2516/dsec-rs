//! Full-stack integration: wires the control plane to node runtimes.
//!
//! This is the assembly a real deployment would do (and what tests,
//! benchmarks and examples reuse): `EdgeNode`s are wrapped into a
//! [`MultiNodeProvisioner`] for the control plane, nodes register their
//! capacity snapshots, the apiserver is served, and clients receive a
//! [`DsecClient`] with a data-plane transport.
//!
//! Two transports are available, mirroring the paper's data plane:
//! - [`ChannelTransport`] — in-process (deterministic, high-throughput)
//! - [`UdsTransport`] — real Unix domain sockets at `root/<node>.sock`

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use dsec_control::model::{
    BackendKind as ControlBackend, NodeInfo, NodeStatus, Resources, SandboxSpec as ControlSpec,
    SandboxState as ControlState,
};
use dsec_control::{
    ControlPlane, Error as ControlError, ProvisionedSandbox, Provisioner, Result as ControlResult,
};
use dsec_runtime::backend::BackendKind as RuntimeBackend;
use dsec_runtime::state::SandboxState as RuntimeState;
use dsec_runtime::EdgeNode;
use dsec_storage::latency::NodeLatencyProfile;

use crate::transport::{ChannelTransport, Transport, UdsTransport};
use crate::{DsecClient, Endpoint};

/// Maps a control-plane backend kind onto the runtime's.
pub fn to_runtime_backend(b: ControlBackend) -> RuntimeBackend {
    match b {
        ControlBackend::Fncall => RuntimeBackend::Fncall,
        ControlBackend::Container => RuntimeBackend::Container,
        ControlBackend::Microvm => RuntimeBackend::Microvm,
        ControlBackend::Fullvm => RuntimeBackend::Fullvm,
    }
}

/// Maps a runtime lifecycle state onto the control-plane's.
pub fn to_control_state(s: RuntimeState) -> ControlState {
    match s {
        RuntimeState::Creating => ControlState::Creating,
        RuntimeState::Ready => ControlState::Ready,
        RuntimeState::Paused => ControlState::Paused,
        RuntimeState::Destroying => ControlState::Destroying,
        RuntimeState::Destroyed => ControlState::Destroyed,
        RuntimeState::Failed => ControlState::Failed,
    }
}

/// Converts a control-plane sandbox spec into a runtime spec.
pub fn to_runtime_spec(spec: &ControlSpec) -> dsec_runtime::backend::SandboxSpec {
    dsec_runtime::backend::SandboxSpec {
        image_id: spec.image.clone(),
        backend: to_runtime_backend(spec.backend),
        resources: dsec_runtime::resource::ResourceRequest {
            cpu_millicores: spec.resources.cpu_millicores,
            mem_mib: spec.resources.mem_mib,
        },
        project: spec.project.clone(),
        env: HashMap::new(),
        labels: spec.labels.clone(),
        cpu_idle: true,
    }
}

/// Builds the node snapshot the control plane places against.
pub fn node_info_for(node: &Arc<EdgeNode>, locality_cloud: bool, cost: f64) -> NodeInfo {
    let summary = node.summary();
    NodeInfo {
        node_id: node.node_id.clone(),
        cpu_millicores: summary.cpu_millicores_total,
        mem_mib: summary.mem_mib_total,
        max_sandboxes: node.capacity().max_sandboxes,
        cpu_available: summary.cpu_millicores_total - summary.cpu_millicores_used,
        mem_available: summary.mem_mib_total - summary.mem_mib_used,
        slots_available: node.capacity().sandbox_slots_free(),
        status: NodeStatus::Healthy,
        locality: if locality_cloud {
            dsec_control::model::Locality::Cloud
        } else {
            dsec_control::model::Locality::Local
        },
        images: summary.images,
        labels: HashMap::new(),
        admitted_projects: node.projects_admitted(),
        cost_multiplier: cost,
    }
}

/// Provisioner over one or more Edge nodes (dispatch by node id).
pub struct MultiNodeProvisioner {
    nodes: HashMap<String, Arc<EdgeNode>>,
}

impl MultiNodeProvisioner {
    pub fn new(nodes: Vec<Arc<EdgeNode>>) -> Self {
        MultiNodeProvisioner {
            nodes: nodes.into_iter().map(|n| (n.node_id.clone(), n)).collect(),
        }
    }

    fn node(&self, node_id: &str) -> ControlResult<Arc<EdgeNode>> {
        self.nodes
            .get(node_id)
            .cloned()
            .ok_or_else(|| ControlError::NodeNotFound(node_id.to_string()))
    }
}

fn rt_err(node_id: &str, e: dsec_runtime::Error) -> ControlError {
    ControlError::ProvisionFailed {
        node: node_id.to_string(),
        message: e.to_string(),
    }
}

impl Provisioner for MultiNodeProvisioner {
    fn provision(
        &self,
        node_id: &str,
        spec: &ControlSpec,
        sid: u64,
    ) -> dsec_control::BoxFuture<'static, ControlResult<ProvisionedSandbox>> {
        let node = match self.node(node_id) {
            Ok(n) => n,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let rt_spec = to_runtime_spec(spec);
        let node_id = node_id.to_string();
        Box::pin(async move {
            let entry = node
                .create_with_sid(sid, rt_spec)
                .await
                .map_err(|e| rt_err(&node_id, e))?;
            Ok(ProvisionedSandbox {
                sid: entry.sid,
                state: to_control_state(entry.instance.state()),
            })
        })
    }

    fn pause(
        &self,
        sid: u64,
        node_id: &str,
    ) -> dsec_control::BoxFuture<'static, ControlResult<()>> {
        let node = match self.node(node_id) {
            Ok(n) => n,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let node_id = node_id.to_string();
        Box::pin(async move {
            node.pause(sid)
                .await
                .map(|_| ())
                .map_err(|e| rt_err(&node_id, e))
        })
    }

    fn resume(
        &self,
        sid: u64,
        node_id: &str,
    ) -> dsec_control::BoxFuture<'static, ControlResult<()>> {
        let node = match self.node(node_id) {
            Ok(n) => n,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let node_id = node_id.to_string();
        Box::pin(async move {
            node.resume(sid)
                .await
                .map(|_| ())
                .map_err(|e| rt_err(&node_id, e))
        })
    }

    fn destroy(
        &self,
        sid: u64,
        node_id: &str,
    ) -> dsec_control::BoxFuture<'static, ControlResult<()>> {
        let node = match self.node(node_id) {
            Ok(n) => n,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let node_id = node_id.to_string();
        Box::pin(async move {
            node.destroy(sid)
                .await
                .map(|_| ())
                .map_err(|e| rt_err(&node_id, e))
        })
    }
}

/// How the data plane is reached.
#[derive(Clone)]
pub enum DataPlane {
    /// In-process channels.
    Channel,
    /// UDS sockets under `root`.
    Uds { root: std::path::PathBuf },
}

impl DataPlane {
    fn transport(&self, nodes: &[Arc<EdgeNode>]) -> Arc<dyn Transport> {
        match self {
            DataPlane::Channel => Arc::new(ChannelTransport::with_nodes(nodes.to_vec())),
            DataPlane::Uds { root } => Arc::new(UdsTransport::new(root.clone())),
        }
    }
}

/// A locally assembled full stack.
pub struct LocalCluster {
    pub plane: Arc<ControlPlane>,
    pub nodes: Vec<Arc<EdgeNode>>,
    pub api_addr: SocketAddr,
    pub admin_token: String,
    transport: Arc<dyn Transport>,
    _uds_handles: Vec<tokio::task::JoinHandle<()>>,
}

/// Configuration for [`spawn_local_cluster`].
pub struct ClusterConfig {
    pub node_count: usize,
    pub cpu_millicores: i64,
    pub mem_mib: i64,
    pub max_sandboxes: i64,
    pub latencies: NodeLatencyProfile,
    pub seed: u64,
    pub data_plane: DataPlane,
    /// Warm FnCall pool depth per node.
    pub prewarm: usize,
    /// Extra project admitted on every node.
    pub project: Option<String>,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        ClusterConfig {
            node_count: 1,
            cpu_millicores: 64_000,
            mem_mib: 64_000,
            max_sandboxes: 10_000,
            latencies: NodeLatencyProfile::zero(),
            seed: 42,
            data_plane: DataPlane::Channel,
            prewarm: 8,
            project: None,
        }
    }
}

/// Assembles nodes + control plane + apiserver + transports.
pub async fn spawn_local_cluster(cfg: ClusterConfig) -> LocalCluster {
    // Image + node runtimes.
    let image = Arc::new(dsec_storage::erofs::ErofsImageBuilder::agent_base().build());
    let mut registry = dsec_storage::erofs::ImageRegistry::default();
    let cache = Arc::new(dsec_storage::cache::LruBlockCache::new(4096));
    registry.register(image, cache, cfg.latencies.block_fetch.clone());

    let mut nodes = Vec::new();
    for i in 0..cfg.node_count {
        let node = EdgeNode::new(
            format!("node-{}", i + 1),
            Arc::new(registry.clone()),
            cfg.latencies.clone(),
            cfg.cpu_millicores,
            cfg.mem_mib,
            cfg.max_sandboxes,
            cfg.seed + i as u64,
        );
        node.factory.prewarm("dsec/agent-base", cfg.prewarm);
        if let Some(project) = &cfg.project {
            node.admit_project(project);
        }
        nodes.push(Arc::new(node));
    }

    // Control plane + provisioner. Rate limiting is disabled for local
    // clusters (benchmarks and tests drive unthrottled load; production
    // deployments keep the apiserver default).
    let plane = Arc::new(ControlPlane::new(cfg.seed, 8));
    plane.set_rate_limit(dsec_control::ratelimit::RateLimitConfig {
        capacity: f64::INFINITY,
        refill_per_sec: f64::INFINITY,
    });
    plane.set_provisioner(Arc::new(MultiNodeProvisioner::new(nodes.clone())));
    for node in &nodes {
        plane.register_node(node_info_for(node, false, 1.0));
    }

    // UDS data plane: serve one socket per node.
    let mut uds_handles = Vec::new();
    let data_plane = match &cfg.data_plane {
        DataPlane::Uds { root } => {
            let _ = std::fs::create_dir_all(root);
            for node in &nodes {
                let path = root.join(format!("{}.sock", node.node_id));
                let _ = std::fs::remove_file(&path);
                if let Ok(h) =
                    dsec_runtime::aether::serve_uds(node.clone(), path.to_string_lossy().as_ref())
                        .await
                {
                    uds_handles.push(h);
                }
            }
            DataPlane::Uds { root: root.clone() }
        }
        other => other.clone(),
    };
    let transport = data_plane.transport(&nodes);

    // Admin token + apiserver.
    let admin_token = plane
        .create_token("root", dsec_control::iam::Role::Admin)
        .unwrap()
        .token;
    let api_addr = dsec_control::apiserver::serve(plane.clone(), "127.0.0.1:0")
        .await
        .unwrap();

    LocalCluster {
        plane,
        nodes,
        api_addr,
        admin_token,
        transport,
        _uds_handles: uds_handles,
    }
}

impl LocalCluster {
    /// A client using the cluster's data-plane transport.
    pub fn client(&self) -> DsecClient {
        DsecClient::new(
            self.admin_token.clone(),
            Endpoint::localhost(self.api_addr.port()),
            self.transport.clone(),
        )
    }

    /// Node info snapshot for a node (tests refresh placement view).
    pub fn refresh_node(&self, idx: usize) {
        let node = &self.nodes[idx];
        self.plane.register_node(node_info_for(node, false, 1.0));
    }
}

/// Default resources for tests.
pub fn tiny_resources() -> Resources {
    Resources {
        cpu_millicores: 500,
        mem_mib: 256,
    }
}
