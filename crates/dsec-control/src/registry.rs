//! Versioned cluster state (the paper's etcd-backed store, in-process).
//!
//! Every mutation bumps a monotonic revision and publishes an event;
//! subscribers (watcher, apiserver long-poll, tests) observe the log.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use serde::Serialize;
use tokio::sync::broadcast;

use crate::model::{NodeInfo, SandboxRecord, SandboxState};

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum EventKind {
    NodeRegistered,
    NodeUpdated,
    NodeLost,
    SandboxCreated,
    SandboxUpdated,
    SandboxDestroyed,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Event {
    pub revision: u64,
    pub kind: EventKind,
    pub key: String,
}

#[derive(Debug)]
pub struct Registry {
    nodes: RwLock<HashMap<String, NodeInfo>>,
    sandboxes: RwLock<HashMap<u64, SandboxRecord>>,
    revision: AtomicU64,
    events: broadcast::Sender<Event>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1024);
        Registry {
            nodes: RwLock::new(HashMap::new()),
            sandboxes: RwLock::new(HashMap::new()),
            revision: AtomicU64::new(0),
            events: tx,
        }
    }

    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Relaxed)
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.events.subscribe()
    }

    fn publish(&self, kind: EventKind, key: String) {
        let rev = self.revision.fetch_add(1, Ordering::Relaxed) + 1;
        let _ = self.events.send(Event {
            revision: rev,
            kind,
            key,
        });
    }

    // -- nodes ------------------------------------------------------------

    pub fn upsert_node(&self, info: NodeInfo) {
        let key = info.node_id.clone();
        let existed = self
            .nodes
            .write()
            .expect("registry poisoned")
            .insert(key.clone(), info)
            .is_some();
        self.publish(
            if existed {
                EventKind::NodeUpdated
            } else {
                EventKind::NodeRegistered
            },
            key,
        );
    }

    pub fn node(&self, id: &str) -> Option<NodeInfo> {
        self.nodes
            .read()
            .expect("registry poisoned")
            .get(id)
            .cloned()
    }

    pub fn nodes(&self) -> Vec<NodeInfo> {
        let mut v: Vec<NodeInfo> = self
            .nodes
            .read()
            .expect("registry poisoned")
            .values()
            .cloned()
            .collect();
        v.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        v
    }

    pub fn set_node_status(&self, id: &str, status: crate::model::NodeStatus) -> bool {
        let mut nodes = self.nodes.write().expect("registry poisoned");
        if let Some(n) = nodes.get_mut(id) {
            n.status = status;
            drop(nodes);
            self.publish(EventKind::NodeUpdated, id.to_string());
            true
        } else {
            false
        }
    }

    /// Removes a node, purges its sandbox records and returns them.
    pub fn remove_node(&self, id: &str) -> Vec<SandboxRecord> {
        self.nodes.write().expect("registry poisoned").remove(id);
        self.publish(EventKind::NodeLost, id.to_string());
        let mut map = self.sandboxes.write().expect("registry poisoned");
        let victims: Vec<SandboxRecord> =
            map.values().filter(|r| r.node_id == id).cloned().collect();
        for v in &victims {
            map.remove(&v.sid);
        }
        victims
    }

    // -- sandboxes ---------------------------------------------------------

    pub fn insert_sandbox(&self, record: SandboxRecord) {
        let key = record.sid.to_string();
        self.sandboxes
            .write()
            .expect("registry poisoned")
            .insert(record.sid, record);
        self.publish(EventKind::SandboxCreated, key);
    }

    pub fn sandbox(&self, sid: u64) -> Option<SandboxRecord> {
        self.sandboxes
            .read()
            .expect("registry poisoned")
            .get(&sid)
            .cloned()
    }

    pub fn update_sandbox_state(&self, sid: u64, state: SandboxState) -> Option<SandboxRecord> {
        let mut map = self.sandboxes.write().expect("registry poisoned");
        if let Some(r) = map.get_mut(&sid) {
            r.state = state;
            let out = r.clone();
            drop(map);
            self.publish(EventKind::SandboxUpdated, sid.to_string());
            Some(out)
        } else {
            None
        }
    }

    pub fn remove_sandbox(&self, sid: u64) -> Option<SandboxRecord> {
        let out = self
            .sandboxes
            .write()
            .expect("registry poisoned")
            .remove(&sid);
        if out.is_some() {
            self.publish(EventKind::SandboxDestroyed, sid.to_string());
        }
        out
    }

    pub fn sandboxes(&self) -> Vec<SandboxRecord> {
        let mut v: Vec<SandboxRecord> = self
            .sandboxes
            .read()
            .expect("registry poisoned")
            .values()
            .cloned()
            .collect();
        v.sort_by_key(|r| r.sid);
        v
    }

    /// Aggregated resource usage per project subtree.
    pub fn project_usage(&self, project: &str) -> Usage {
        let mut u = Usage::default();
        for r in self.sandboxes.read().expect("registry poisoned").values() {
            if project_is_in(r.spec.project.as_str(), project) {
                u.sandboxes += 1;
                if r.state != SandboxState::Paused {
                    u.cpu_millicores += r.spec.resources.cpu_millicores;
                    u.mem_mib += r.spec.resources.mem_mib;
                }
            }
        }
        u
    }
}

/// A project (or subtree) usage snapshot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Usage {
    pub sandboxes: i64,
    pub cpu_millicores: i64,
    pub mem_mib: i64,
}

/// Is `project` inside subtree rooted at `root` (inclusive)?
fn project_is_in(project: &str, root: &str) -> bool {
    project == root || project.starts_with(&format!("{}/", root))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Locality, NodeStatus, SandboxSpec};

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
            labels: HashMap::new(),
            admitted_projects: vec!["root".to_string()],
            cost_multiplier: 1.0,
        }
    }

    fn record(sid: u64, project: &str) -> SandboxRecord {
        SandboxRecord {
            sid,
            spec: SandboxSpec {
                project: project.into(),
                ..Default::default()
            },
            node_id: "n1".into(),
            state: SandboxState::Ready,
            created_epoch_ms: 0,
            updated_epoch_ms: 0,
        }
    }

    #[test]
    fn upsert_and_query() {
        let r = Registry::new();
        r.upsert_node(node("n1", 1000, 1000));
        r.upsert_node(node("n2", 2000, 2000));
        assert_eq!(r.nodes().len(), 2);
        assert_eq!(r.node("n1").unwrap().cpu_millicores, 1000);
        // Update path.
        r.upsert_node(node("n1", 4000, 4000));
        assert_eq!(r.node("n1").unwrap().cpu_millicores, 4000);
        assert_eq!(r.nodes().len(), 2);
    }

    #[test]
    fn sandbox_lifecycle_events() {
        let r = Registry::new();
        let mut rx = r.subscribe();
        r.insert_sandbox(record(1, "root"));
        r.update_sandbox_state(1, SandboxState::Paused).unwrap();
        assert_eq!(r.sandbox(1).unwrap().state, SandboxState::Paused);
        r.remove_sandbox(1);
        assert!(r.sandbox(1).is_none());
        // Three events observed, revisions monotonic.
        let mut revs = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            revs.push(ev.revision);
        }
        assert_eq!(revs.len(), 3);
        assert!(revs.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn project_usage_subtree() {
        let r = Registry::new();
        r.insert_sandbox(record(1, "root"));
        r.insert_sandbox(record(2, "root/team-a"));
        r.insert_sandbox(record(3, "root/team-a/sub"));
        r.insert_sandbox(record(4, "other"));
        // Paused sandboxes do not count cpu/mem.
        r.update_sandbox_state(3, SandboxState::Paused).unwrap();
        assert_eq!(r.project_usage("root").sandboxes, 3);
        assert_eq!(r.project_usage("root/team-a").sandboxes, 2);
        assert_eq!(r.project_usage("other").sandboxes, 1);
        assert_eq!(r.project_usage("nope").sandboxes, 0);
        let root_usage = r.project_usage("root");
        assert_eq!(root_usage.cpu_millicores, 1000); // 2 active of 3
    }

    #[test]
    fn remove_node_returns_sandboxes() {
        let r = Registry::new();
        r.upsert_node(node("n1", 1000, 1000));
        r.insert_sandbox(record(1, "root"));
        r.insert_sandbox(record(2, "root"));
        let evicted = r.remove_node("n1");
        assert_eq!(evicted.len(), 2);
        assert!(r.node("n1").is_none());
    }
}
