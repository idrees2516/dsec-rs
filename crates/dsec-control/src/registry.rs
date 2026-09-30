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
    /// Incremental per-project usage index (mirrors the record map
    /// exactly, so `project_usage` is O(depth) instead of a full scan).
    /// Keyed by every project on the record's ancestor chain.
    usage: RwLock<HashMap<String, Usage>>,
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
            usage: RwLock::new(HashMap::new()),
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
        drop(map);
        for v in &victims {
            self.usage_add(v, -1);
        }
        victims
    }

    // -- sandboxes ---------------------------------------------------------

    /// Applies a record's usage delta to every project on its ancestor
    /// chain (a sandbox in `root/team-a` counts against `root`,
    /// `root/team-a` and itself). Paused records count the sandbox but
    /// not cpu/mem, exactly like the derived scan it replaces.
    fn usage_add(&self, rec: &SandboxRecord, sign: i64) {
        let mut usage = self.usage.write().expect("registry poisoned");
        let mut cur = rec.spec.project.as_str();
        let (d_sbx, mut d_cpu, mut d_mem) = (sign, 0i64, 0i64);
        if rec.state != SandboxState::Paused {
            d_cpu = sign * rec.spec.resources.cpu_millicores;
            d_mem = sign * rec.spec.resources.mem_mib;
        }
        loop {
            // Hit path mutates in place; only a first-seen project key
            // allocates.
            let entry = match usage.get_mut(cur) {
                Some(u) => u,
                None => usage.entry(cur.to_string()).or_default(),
            };
            entry.sandboxes += d_sbx;
            entry.cpu_millicores += d_cpu;
            entry.mem_mib += d_mem;
            match cur.rfind('/') {
                Some(i) => cur = &cur[..i],
                None => break,
            }
        }
    }

    pub fn insert_sandbox(&self, record: SandboxRecord) {
        let key = record.sid.to_string();
        self.usage_add(&record, 1);
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
            let old_state = r.state;
            r.state = state;
            let out = r.clone();
            let paused_changed =
                (old_state == SandboxState::Paused) != (state == SandboxState::Paused);
            drop(map);
            if paused_changed {
                // Paused records don't count cpu/mem: swap the old-state
                // contribution for the new-state one (sandbox count is
                // unchanged either way).
                let mut before = out.clone();
                before.state = old_state;
                self.usage_add(&before, -1);
                self.usage_add(&out, 1);
            }
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
        if let Some(r) = &out {
            self.usage_add(r, -1);
            self.publish(EventKind::SandboxDestroyed, sid.to_string());
        }
        out
    }

    pub fn sandbox_count(&self) -> usize {
        self.sandboxes.read().expect("registry poisoned").len()
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

    /// Aggregated resource usage per project subtree. Served from the
    /// incremental index (O(1) after the record mutations that maintain
    /// it); identical to the full-scan derivation, which the tests
    /// cross-check.
    pub fn project_usage(&self, project: &str) -> Usage {
        self.usage
            .read()
            .expect("registry poisoned")
            .get(project)
            .copied()
            .unwrap_or_default()
    }

    /// In-place resource projection onto a node snapshot (create/destroy
    /// paths). Equivalent to clone-modify-upsert but without copying the
    /// node's strings/vectors; the published event stream is identical.
    pub fn adjust_node_resources(
        &self,
        node_id: &str,
        cpu_delta: i64,
        mem_delta: i64,
        slots_delta: i64,
    ) -> bool {
        let existed;
        {
            let mut nodes = self.nodes.write().expect("registry poisoned");
            match nodes.get_mut(node_id) {
                Some(n) => {
                    n.cpu_available += cpu_delta;
                    n.mem_available += mem_delta;
                    n.slots_available += slots_delta;
                    existed = true;
                }
                None => existed = false,
            }
        }
        if existed {
            self.publish(EventKind::NodeUpdated, node_id.to_string());
        }
        existed
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
#[cfg_attr(not(test), allow(dead_code))]
fn project_is_in(project: &str, root: &str) -> bool {
    project == root
        || project
            .strip_prefix(root)
            .is_some_and(|rest| rest.starts_with('/'))
}

// The full-scan reference used to validate the incremental index in tests.
#[cfg(test)]
fn project_usage_scan(registry: &Registry, project: &str) -> Usage {
    let mut u = Usage::default();
    let map = registry.sandboxes.read().expect("registry poisoned");
    for r in map.values() {
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
    fn usage_index_matches_full_scan() {
        // The incremental index must equal the derived full scan across a
        // mixed sequence of inserts, pause toggles, removes and node loss.
        let r = Registry::new();
        r.upsert_node(node("n1", 1000, 1000));
        for i in 0..20 {
            let project = match i % 4 {
                0 => "root",
                1 => "root/team-a",
                2 => "root/team-a/sub",
                _ => "solo",
            };
            r.insert_sandbox(record(i, project));
        }
        for i in 0..20 {
            if i % 3 == 0 {
                r.update_sandbox_state(i, SandboxState::Paused).unwrap();
            }
        }
        for i in 0..20 {
            if i % 5 == 0 {
                r.update_sandbox_state(i, SandboxState::Ready).unwrap();
            }
        }
        for i in 0..20 {
            if i % 7 == 0 {
                r.remove_sandbox(i);
            }
        }
        for project in ["root", "root/team-a", "root/team-a/sub", "solo", "nope"] {
            assert_eq!(
                r.project_usage(project),
                project_usage_scan(&r, project),
                "index mismatch for {project}"
            );
        }
        // Node loss purges records and the index together.
        r.remove_node("n1");
        for project in ["root", "root/team-a", "solo"] {
            assert_eq!(r.project_usage(project), project_usage_scan(&r, project));
        }
    }

    #[test]
    fn sandbox_count_is_o1() {
        let r = Registry::new();
        assert_eq!(r.sandbox_count(), 0);
        r.insert_sandbox(record(1, "root"));
        r.insert_sandbox(record(2, "root"));
        assert_eq!(r.sandbox_count(), 2);
        r.remove_sandbox(1);
        assert_eq!(r.sandbox_count(), 1);
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
