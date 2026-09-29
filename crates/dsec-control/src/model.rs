//! Control-plane data model: sandbox specs, node snapshots, records.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Backend selector (wire enum; mirrors the runtime's BackendKind).
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

/// Resources requested for a sandbox.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Resources {
    pub cpu_millicores: i64,
    pub mem_mib: i64,
}

impl Default for Resources {
    fn default() -> Self {
        Resources {
            cpu_millicores: 500,
            mem_mib: 256,
        }
    }
}

/// Sandbox creation request (control-plane side).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SandboxSpec {
    pub image: String,
    pub backend: BackendKind,
    pub resources: Resources,
    pub project: String,
    #[serde(default)]
    pub labels: HashMap<String, String>,
    /// Scheduling priority; higher = more important (preemption evicts
    /// low priority first, paused before running).
    #[serde(default)]
    pub priority: i32,
    /// Allow overflow to cloud nodes when no local node fits.
    #[serde(default)]
    pub allow_burst: bool,
}

impl Default for SandboxSpec {
    fn default() -> Self {
        SandboxSpec {
            image: "dsec/agent-base".to_string(),
            backend: BackendKind::Fncall,
            resources: Resources::default(),
            project: "root".to_string(),
            labels: HashMap::new(),
            priority: 0,
            allow_burst: false,
        }
    }
}

/// Node health as tracked by the Watcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl NodeStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeStatus::Healthy => "healthy",
            NodeStatus::Degraded => "degraded",
            NodeStatus::Unhealthy => "unhealthy",
        }
    }
}

/// Where a node lives (cloud bursting).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Locality {
    Local,
    Cloud,
}

/// Node snapshot as seen by the control plane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub cpu_millicores: i64,
    pub mem_mib: i64,
    pub max_sandboxes: i64,
    pub cpu_available: i64,
    pub mem_available: i64,
    pub slots_available: i64,
    pub status: NodeStatus,
    pub locality: Locality,
    pub images: Vec<String>,
    pub labels: HashMap<String, String>,
    /// Projects allowed to schedule here (paper: nested project pools).
    pub admitted_projects: Vec<String>,
    /// Cost multiplier (cloud > 1).
    pub cost_multiplier: f64,
}

impl NodeInfo {
    /// Hard-filter check.
    pub fn fits(&self, spec: &SandboxSpec) -> bool {
        self.status == NodeStatus::Healthy
            && self.cpu_available >= spec.resources.cpu_millicores
            && self.mem_available >= spec.resources.mem_mib
            && self.slots_available > 0
            && self.images.contains(&spec.image)
            && self
                .admitted_projects
                .iter()
                .any(|p| spec.project == *p || spec.project.starts_with(&format!("{}/", p)))
    }

    /// Best-effort packing score components, higher = better.
    pub fn score_parts(&self, spec: &SandboxSpec) -> (f64, f64, f64) {
        // Packing: prefer to fill the node that fits tightest (reduces
        // fragmentation across the fleet, like bin-packing best-fit).
        let cpu_pack = spec.resources.cpu_millicores as f64 / (self.cpu_available.max(1)) as f64;
        let mem_pack = spec.resources.mem_mib as f64 / (self.mem_available.max(1)) as f64;
        let fit = (cpu_pack + mem_pack) / 2.0;
        // Leftover fragmentation after placement (lower leftover -> higher score).
        let leftover_cpu = (self.cpu_available - spec.resources.cpu_millicores) as f64
            / self.cpu_millicores.max(1) as f64;
        let leftover_mem =
            (self.mem_available - spec.resources.mem_mib) as f64 / self.mem_mib.max(1) as f64;
        let frag = 1.0 - (leftover_cpu + leftover_mem) / 2.0;
        // Image locality: local nodes with the image cached win.
        let locality = if self.locality == Locality::Cloud {
            0.0
        } else {
            1.0
        };
        let _ = spec;
        (fit, frag, locality)
    }
}

/// Sandbox record kept by the registry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SandboxRecord {
    pub sid: u64,
    pub spec: SandboxSpec,
    pub node_id: String,
    pub state: SandboxState,
    pub created_epoch_ms: u64,
    pub updated_epoch_ms: u64,
}

/// Lifecycle state mirrored from the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxState {
    Creating,
    Ready,
    Paused,
    Destroying,
    Destroyed,
    Failed,
}

impl SandboxState {
    pub fn as_str(&self) -> &'static str {
        match self {
            SandboxState::Creating => "creating",
            SandboxState::Ready => "ready",
            SandboxState::Paused => "paused",
            SandboxState::Destroying => "destroying",
            SandboxState::Destroyed => "destroyed",
            SandboxState::Failed => "failed",
        }
    }
}

/// Quota for a project (per subtree, rolled up the hierarchy).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Quota {
    /// -1 means unlimited.
    pub max_cpu_millicores: i64,
    pub max_mem_mib: i64,
    pub max_sandboxes: i64,
}

impl Default for Quota {
    fn default() -> Self {
        Quota {
            max_cpu_millicores: -1,
            max_mem_mib: -1,
            max_sandboxes: -1,
        }
    }
}

impl Quota {
    pub fn limited(max_cpu: i64, max_mem: i64, max_sandboxes: i64) -> Self {
        Quota {
            max_cpu_millicores: max_cpu,
            max_mem_mib: max_mem,
            max_sandboxes,
        }
    }
}
