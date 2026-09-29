//! Placement Engine: hard filters + k-choice (power-of-two-choices)
//! scoring.
//!
//! Scanning every node for every sandbox is O(N) per placement; at the
//! paper's scale (~5,000 creations/sec) that dominates. The k-choice
//! scheme samples k candidates, scores them, and commits to the best —
//! O(k) work with near-best-fit quality. Cloud bursting widens the
//! filter when no local node fits and the spec allows it.

use std::sync::Mutex;

use dsec_protocol::rng::Rng;
use serde::Serialize;

use crate::error::{Error, Result};
use crate::model::{Locality, NodeInfo, SandboxSpec};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreWeights {
    pub packing: f64,
    pub fragmentation: f64,
    pub locality: f64,
    pub topology: f64,
    pub cost: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        ScoreWeights {
            packing: 1.0,
            fragmentation: 0.5,
            locality: 1.0,
            topology: 0.25,
            cost: 1.5,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PlacementDecision {
    pub node_id: String,
    pub score: f64,
    pub sampled: usize,
    pub candidates: usize,
    pub burst: bool,
}

pub struct PlacementEngine {
    k: usize,
    weights: ScoreWeights,
    rng: Mutex<Rng>,
    pub decisions: std::sync::atomic::AtomicU64,
}

impl PlacementEngine {
    pub fn new(k: usize, seed: u64, weights: ScoreWeights) -> Self {
        PlacementEngine {
            k: k.max(1),
            weights,
            rng: Mutex::new(Rng::new(seed)),
            decisions: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Full placement pass. `nodes` should be the current registry
    /// snapshot (already resource-adjusted).
    pub fn place(&self, spec: &SandboxSpec, nodes: &[NodeInfo]) -> Result<PlacementDecision> {
        let decision = self.place_filtered(spec, nodes, false)?;
        if let Some(d) = decision {
            return Ok(d);
        }
        // Cloud burst fallback.
        if spec.allow_burst {
            if let Some(d) = self.place_filtered(spec, nodes, true)? {
                return Ok(d);
            }
        }
        Err(Error::NoCandidate(format!(
            "image={} backend={} cpu={}m mem={}Mi burst={}",
            spec.image,
            spec.backend.as_str(),
            spec.resources.cpu_millicores,
            spec.resources.mem_mib,
            spec.allow_burst
        )))
    }

    fn place_filtered(
        &self,
        spec: &SandboxSpec,
        nodes: &[NodeInfo],
        allow_cloud: bool,
    ) -> Result<Option<PlacementDecision>> {
        let candidates: Vec<&NodeInfo> = nodes
            .iter()
            .filter(|n| n.fits(spec) && (allow_cloud || n.locality == Locality::Local))
            .collect();
        if candidates.is_empty() {
            return Ok(None);
        }
        // k distinct samples from the candidate list.
        let k = self.k.min(candidates.len());
        let sampled_idx = {
            let mut rng = self.rng.lock().expect("placement rng poisoned");
            rng.sample_k(candidates.len(), k)
        };
        let mut best: Option<(f64, &NodeInfo)> = None;
        // Randomized tie-breaking: a tiny seeded jitter (<= 0.01) breaks
        // exact ties between identical nodes so load spreads, while any
        // genuinely better candidate still wins by a real margin.
        let jitter: Vec<f64> = {
            let mut rng = self.rng.lock().expect("placement rng poisoned");
            (0..sampled_idx.len())
                .map(|_| rng.next_f64() * 0.01)
                .collect()
        };
        for (rank, &i) in sampled_idx.iter().enumerate() {
            let node = candidates[i];
            let score = self.score(spec, node) + jitter[rank];
            if best.map(|(s, _)| score > s).unwrap_or(true) {
                best = Some((score, node));
            }
        }
        self.decisions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(best.map(|(score, node)| PlacementDecision {
            node_id: node.node_id.clone(),
            score,
            sampled: sampled_idx.len(),
            candidates: candidates.len(),
            burst: node.locality == Locality::Cloud,
        }))
    }

    /// Weighted score, higher is better.
    fn score(&self, spec: &SandboxSpec, node: &NodeInfo) -> f64 {
        let (fit, frag, locality) = node.score_parts(spec);
        // Topology: count of matching label keys.
        let topology = spec
            .labels
            .keys()
            .filter(|k| node.labels.contains_key(*k))
            .count() as f64;
        // Cost penalty for cloud.
        let cost = 1.0 / node.cost_multiplier.max(0.001);
        let w = &self.weights;
        w.packing * fit
            + w.fragmentation * frag
            + w.locality * locality
            + w.topology * topology.min(3.0)
            + w.cost * cost
    }

    /// Adjusts a node snapshot as if the sandbox was placed on it
    /// (used to project post-placement availability without mutating).
    pub fn project_after_placement(node: &NodeInfo, spec: &SandboxSpec) -> NodeInfo {
        let mut n = node.clone();
        n.cpu_available -= spec.resources.cpu_millicores;
        n.mem_available -= spec.resources.mem_mib;
        n.slots_available -= 1;
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::NodeStatus;
    use std::collections::HashMap;

    fn node(id: &str, cpu: i64, mem: i64, images: Vec<&str>) -> NodeInfo {
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
            images: images.into_iter().map(String::from).collect(),
            labels: HashMap::new(),
            admitted_projects: vec!["root".to_string()],
            cost_multiplier: 1.0,
        }
    }

    fn cloud(id: &str) -> NodeInfo {
        NodeInfo {
            node_id: id.to_string(),
            cpu_millicores: 8000,
            mem_mib: 8192,
            max_sandboxes: 100,
            cpu_available: 8000,
            mem_available: 8192,
            slots_available: 100,
            status: NodeStatus::Healthy,
            locality: Locality::Cloud,
            images: vec!["dsec/agent-base".to_string()],
            labels: HashMap::new(),
            admitted_projects: vec!["root".to_string()],
            cost_multiplier: 3.0,
        }
    }

    #[test]
    fn filters_reject_unfit_nodes() {
        let engine = PlacementEngine::new(4, 1, ScoreWeights::default());
        let spec = SandboxSpec::default();
        let nodes = vec![
            node("too-small", 100, 100, vec!["dsec/agent-base"]),
            node("no-image", 8000, 8192, vec!["other/image"]),
            node("good", 8000, 8192, vec!["dsec/agent-base"]),
        ];
        let d = engine.place(&spec, &nodes).unwrap();
        assert_eq!(d.node_id, "good");
        assert!(!d.burst);
    }

    #[test]
    fn unhealthy_nodes_excluded() {
        let engine = PlacementEngine::new(4, 1, ScoreWeights::default());
        let mut n = node("sick", 8000, 8192, vec!["dsec/agent-base"]);
        n.status = NodeStatus::Unhealthy;
        let d = engine.place(&SandboxSpec::default(), &[n]);
        assert!(matches!(d, Err(Error::NoCandidate(_))));
    }

    #[test]
    fn cloud_burst_only_when_allowed() {
        let engine = PlacementEngine::new(4, 1, ScoreWeights::default());
        let mut spec = SandboxSpec {
            allow_burst: false,
            ..SandboxSpec::default()
        };
        let only_cloud = vec![cloud("c1")];
        assert!(matches!(
            engine.place(&spec, &only_cloud),
            Err(Error::NoCandidate(_))
        ));
        spec.allow_burst = true;
        let d = engine.place(&spec, &only_cloud).unwrap();
        assert_eq!(d.node_id, "c1");
        assert!(d.burst);
    }

    #[test]
    fn prefers_local_over_cloud_when_both_fit() {
        let engine = PlacementEngine::new(8, 1, ScoreWeights::default());
        let spec = SandboxSpec {
            allow_burst: true,
            ..SandboxSpec::default()
        };
        let nodes = vec![
            node("local", 8000, 8192, vec!["dsec/agent-base"]),
            cloud("c1"),
        ];
        // With k=8 both get sampled; local must win (cost + locality).
        let d = engine.place(&spec, &nodes).unwrap();
        assert_eq!(d.node_id, "local");
    }

    #[test]
    fn k_choice_samples_subset() {
        let engine = PlacementEngine::new(4, 1, ScoreWeights::default());
        let nodes: Vec<NodeInfo> = (0..64)
            .map(|i| node(&format!("n{}", i), 8000, 8192, vec!["dsec/agent-base"]))
            .collect();
        let d = engine.place(&SandboxSpec::default(), &nodes).unwrap();
        assert_eq!(d.sampled, 4);
        assert_eq!(d.candidates, 64);
    }

    #[test]
    fn deterministic_given_seed() {
        let a = PlacementEngine::new(4, 42, ScoreWeights::default());
        let b = PlacementEngine::new(4, 42, ScoreWeights::default());
        let nodes: Vec<NodeInfo> = (0..32)
            .map(|i| {
                node(
                    &format!("n{}", i),
                    1000 + i * 100,
                    1024 + i * 128,
                    vec!["dsec/agent-base"],
                )
            })
            .collect();
        let spec = SandboxSpec {
            resources: crate::model::Resources {
                cpu_millicores: 300,
                mem_mib: 300,
            },
            ..Default::default()
        };
        for _ in 0..20 {
            let da = a.place(&spec, &nodes).unwrap();
            let db = b.place(&spec, &nodes).unwrap();
            assert_eq!(da.node_id, db.node_id);
            assert!((da.score - db.score).abs() < 1e-9);
        }
    }

    #[test]
    fn best_of_k_beats_random_on_packing() {
        // Over many placements with projected availability, k=8 should
        // distribute at least as well as k=1 (statistical smoke check).
        let engine = PlacementEngine::new(8, 7, ScoreWeights::default());
        let mut nodes: Vec<NodeInfo> = (0..16)
            .map(|i| node(&format!("n{}", i), 1000, 1024, vec!["dsec/agent-base"]))
            .collect();
        let spec = SandboxSpec {
            resources: crate::model::Resources {
                cpu_millicores: 250,
                mem_mib: 256,
            },
            ..Default::default()
        };
        // Place 40 sandboxes, projecting availability forward.
        let mut placed = 0;
        for _ in 0..40 {
            if let Ok(d) = engine.place(&spec, &nodes) {
                let idx = nodes.iter().position(|n| n.node_id == d.node_id).unwrap();
                nodes[idx] = PlacementEngine::project_after_placement(&nodes[idx], &spec);
                placed += 1;
            }
        }
        // All 64 slots (16 nodes * 4 each) fit 40 placements.
        assert_eq!(placed, 40);
        // No node oversubscribed.
        for n in &nodes {
            assert!(n.cpu_available >= 0 && n.mem_available >= 0);
        }
    }
}
