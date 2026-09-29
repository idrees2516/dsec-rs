//! Watcher: node health via heartbeats, eviction of sandboxes on lost
//! nodes, and preemption under quota pressure.
//!
//! The paper's Watcher reconciles observed cluster state with the
//! control plane's records; here nodes heartbeat the apiserver and the
//! watcher task enforces a TTL, marking silent nodes unhealthy and
//! evicting their sandboxes through the provisioner (best-effort).

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex as AsyncMutex;

use crate::control::ControlPlane;
use crate::model::{NodeStatus, SandboxState};

#[derive(Debug, Clone)]
pub struct WatcherConfig {
    /// Nodes silent longer than this are marked unhealthy.
    pub ttl: Duration,
    /// Poll interval for the health task.
    pub check_interval: Duration,
    /// Unhealthy node grace before eviction.
    pub eviction_grace: Duration,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        WatcherConfig {
            ttl: Duration::from_secs(30),
            check_interval: Duration::from_secs(1),
            eviction_grace: Duration::from_secs(10),
        }
    }
}

pub struct Watcher {
    config: WatcherConfig,
    last_seen: AsyncMutex<std::collections::HashMap<String, Instant>>,
    marked_at: AsyncMutex<std::collections::HashMap<String, Instant>>,
    plane: Arc<ControlPlane>,
    evictions: std::sync::atomic::AtomicU64,
}

impl Watcher {
    pub fn new(plane: Arc<ControlPlane>, config: WatcherConfig) -> Arc<Self> {
        Arc::new(Watcher {
            config,
            last_seen: AsyncMutex::new(std::collections::HashMap::new()),
            marked_at: AsyncMutex::new(std::collections::HashMap::new()),
            plane,
            evictions: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Records a heartbeat (called by the apiserver).
    pub async fn heartbeat(&self, node_id: &str) {
        let mut seen = self.last_seen.lock().await;
        let first = !seen.contains_key(node_id);
        seen.insert(node_id.to_string(), Instant::now());
        drop(seen);
        if first {
            // Node recovering: clear its unhealthy mark.
            self.marked_at.lock().await.remove(node_id);
            self.plane
                .registry
                .set_node_status(node_id, NodeStatus::Healthy);
        } else {
            // A heartbeat from a previously unhealthy node revives it.
            let was_unhealthy = self
                .plane
                .registry
                .node(node_id)
                .map(|n| n.status != NodeStatus::Healthy)
                .unwrap_or(false);
            if was_unhealthy {
                self.marked_at.lock().await.remove(node_id);
                self.plane
                    .registry
                    .set_node_status(node_id, NodeStatus::Healthy);
            }
        }
    }

    /// One health sweep: mark expired nodes, evict after grace.
    pub async fn sweep_once(&self) {
        let now = Instant::now();
        let expired: Vec<String> = {
            let seen = self.last_seen.lock().await;
            self.plane
                .registry
                .nodes()
                .into_iter()
                .filter(|n| {
                    seen.get(&n.node_id)
                        .map(|t| now.duration_since(*t) > self.config.ttl)
                        .unwrap_or(true) // never seen: counts as expired
                })
                .map(|n| n.node_id)
                .collect()
        };
        for node_id in expired {
            if self
                .plane
                .registry
                .set_node_status(&node_id, NodeStatus::Unhealthy)
            {
                self.marked_at
                    .lock()
                    .await
                    .entry(node_id.clone())
                    .or_insert_with(Instant::now);
                self.plane
                    .metrics
                    .incr("dsec_watcher_nodes_unhealthy_total");
            }
        }
        // Eviction pass.
        let now = Instant::now();
        let to_evict: Vec<String> = {
            let marked = self.marked_at.lock().await;
            marked
                .iter()
                .filter(|(_, t)| now.duration_since(**t) > self.config.eviction_grace)
                .map(|(id, _)| id.clone())
                .collect()
        };
        for node_id in to_evict {
            let count = self.plane.evict_node(&node_id).await.unwrap_or(0);
            if count > 0 {
                self.evictions
                    .fetch_add(count as u64, std::sync::atomic::Ordering::Relaxed);
                self.plane
                    .metrics
                    .incr_by("dsec_watcher_sandboxes_evicted_total", count as u64);
            }
            self.marked_at.lock().await.remove(&node_id);
        }
    }

    /// Spawns the periodic sweep task.
    pub fn spawn(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let watcher = self.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(watcher.config.check_interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                ticker.tick().await;
                watcher.sweep_once().await;
            }
        })
    }

    pub fn evictions(&self) -> u64 {
        self.evictions.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Preemption policy: under resource pressure, victims are chosen
/// paused-first, then lowest priority, then oldest.
pub fn preemption_victims(
    records: &[crate::model::SandboxRecord],
    count: usize,
) -> Vec<crate::model::SandboxRecord> {
    let mut sorted = records.to_vec();
    sorted.sort_by(|a, b| {
        let pa = a.state == SandboxState::Paused;
        let pb = b.state == SandboxState::Paused;
        pb.cmp(&pa) // paused (true) sorts first
            .then(a.spec.priority.cmp(&b.spec.priority)) // lowest priority first
            .then(a.created_epoch_ms.cmp(&b.created_epoch_ms)) // oldest first
    });
    sorted.truncate(count);
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{SandboxRecord, SandboxSpec};

    fn record(sid: u64, state: SandboxState, priority: i32, created: u64) -> SandboxRecord {
        SandboxRecord {
            sid,
            spec: SandboxSpec {
                priority,
                ..Default::default()
            },
            node_id: "n1".into(),
            state,
            created_epoch_ms: created,
            updated_epoch_ms: created,
        }
    }

    #[test]
    fn preemption_prefers_paused_then_low_priority() {
        let records = vec![
            record(1, SandboxState::Ready, 5, 100),
            record(2, SandboxState::Paused, 5, 200),
            record(3, SandboxState::Ready, 1, 300),
            record(4, SandboxState::Paused, 1, 400),
        ];
        let victims = preemption_victims(&records, 2);
        assert_eq!(victims[0].sid, 4); // paused + lowest priority
        assert_eq!(victims[1].sid, 2); // paused
        assert_eq!(preemption_victims(&records, 1)[0].sid, 4);
    }
}
