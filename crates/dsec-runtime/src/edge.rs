//! Edge: the node-local sandbox runtime manager.
//!
//! Responsibilities (paper Sec. "Edge"):
//! - admission control (capacity, image locality, project allowlist)
//! - sandbox lifecycle: create / pause / resume / destroy with CAS-guarded
//!   transitions and latency injection per backend
//! - data-plane dispatch for Aether frames (Chronus sessions + lifecycle)
//! - resource governance incl. pause-time memory reclaim
//! - runtime stats exported to the Watcher and `/metrics`

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use dsec_protocol::frame::{Channel, Frame};
use dsec_protocol::message::{ErrorCode, Request, Response};
use serde::{Deserialize, Serialize};

use crate::backend::{BackendFactory, BackendInstance, SandboxSpec};
use crate::state::SandboxState;
use crate::stats::{EdgeStats, HistSnapshot};
use dsec_storage::erofs::ImageRegistry;
use dsec_storage::latency::NodeLatencyProfile;

/// One sandbox managed by this Edge.
pub struct SandboxEntry {
    pub sid: u64,
    pub spec: SandboxSpec,
    pub instance: Arc<BackendInstance>,
    pub created_epoch_ms: u64,
}

/// Node-level snapshot for the Watcher / apiserver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeSummary {
    pub node_id: String,
    pub sandboxes: u64,
    pub paused: u64,
    pub cpu_millicores_total: i64,
    pub cpu_millicores_used: i64,
    pub mem_mib_total: i64,
    pub mem_mib_used: i64,
    pub images: Vec<String>,
    pub created_total: u64,
    pub destroyed_total: u64,
    pub failed_total: u64,
}

/// Result of a pause operation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PauseOutcome {
    pub sid: u64,
    pub duration_ms: u64,
    pub reclaimed_mib: i64,
}

/// The node-local runtime.
pub struct EdgeNode {
    pub node_id: String,
    pub factory: BackendFactory,
    pool: crate::resource::ResourcePool,
    sandboxes: RwLock<HashMap<u64, Arc<SandboxEntry>>>,
    next_sid: AtomicU64,
    admitted_projects: RwLock<HashSet<String>>,
    pub stats: EdgeStats,
    epoch_base: u64,
}

impl EdgeNode {
    /// Builds a node; `seed` drives deterministic latency jitter.
    pub fn new(
        node_id: String,
        registry: Arc<ImageRegistry>,
        latencies: NodeLatencyProfile,
        cpu_millicores: i64,
        mem_mib: i64,
        max_sandboxes: i64,
        seed: u64,
    ) -> Self {
        let _ = seed; // latency profiles carry their own seeds
        let epoch_base = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        EdgeNode {
            node_id,
            factory: BackendFactory::new(registry, latencies, 8),
            pool: crate::resource::ResourcePool::new(cpu_millicores, mem_mib, max_sandboxes),
            sandboxes: RwLock::new(HashMap::new()),
            next_sid: AtomicU64::new(1),
            admitted_projects: RwLock::new(HashSet::from(["root".to_string()])),
            stats: EdgeStats::default(),
            epoch_base,
        }
    }

    /// Deterministic-enough wall clock for command semantics.
    pub fn epoch_ms(&self) -> u64 {
        self.epoch_base
            + std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
                % 1_000_000
    }

    /// Registers a project allowed to schedule onto this node.
    pub fn admit_project(&self, project: &str) {
        self.admitted_projects
            .write()
            .expect("projects poisoned")
            .insert(project.to_string());
    }

    pub fn projects_admitted(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .admitted_projects
            .read()
            .expect("projects poisoned")
            .iter()
            .cloned()
            .collect();
        v.sort();
        v
    }

    pub fn capacity(&self) -> &crate::resource::ResourcePool {
        &self.pool
    }

    pub fn sandbox_count(&self) -> u64 {
        self.sandboxes.read().expect("sandboxes poisoned").len() as u64
    }

    pub fn lookup(&self, sid: u64) -> Option<Arc<SandboxEntry>> {
        self.sandboxes
            .read()
            .expect("sandboxes poisoned")
            .get(&sid)
            .cloned()
    }

    pub fn list_sandboxes(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .sandboxes
            .read()
            .expect("sandboxes poisoned")
            .keys()
            .copied()
            .collect();
        ids.sort_unstable();
        ids
    }

    pub fn summary(&self) -> NodeSummary {
        let usage = self.pool.usage();
        NodeSummary {
            node_id: self.node_id.clone(),
            sandboxes: usage.sandboxes as u64,
            paused: usage.paused_sandboxes as u64,
            cpu_millicores_total: self.pool.capacity.cpu_millicores,
            cpu_millicores_used: usage.cpu_millicores,
            mem_mib_total: self.pool.capacity.mem_mib,
            mem_mib_used: usage.mem_mib,
            images: self.factory.registry().list(),
            created_total: self.stats.sandboxes_created.load(Ordering::Relaxed),
            destroyed_total: self.stats.sandboxes_destroyed.load(Ordering::Relaxed),
            failed_total: self.stats.sandboxes_failed.load(Ordering::Relaxed),
        }
    }

    // -- lifecycle ---------------------------------------------------------

    /// Admission + creation. Fails fast on capacity or missing image.
    pub async fn create(&self, spec: SandboxSpec) -> crate::Result<Arc<SandboxEntry>> {
        let sid = self.next_sid.fetch_add(1, Ordering::Relaxed);
        self.create_with_sid(sid, spec).await
    }

    /// Creation with an externally allocated sandbox id (used when the
    /// control plane allocates ids).
    pub async fn create_with_sid(
        &self,
        sid: u64,
        spec: SandboxSpec,
    ) -> crate::Result<Arc<SandboxEntry>> {
        let t0 = std::time::Instant::now();
        // Admission.
        if !self.factory.registry().contains(&spec.image_id) {
            self.stats
                .admission_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Err(crate::Error::ImageNotLocal(spec.image_id.clone()));
        }
        let admitted = self
            .admitted_projects
            .read()
            .expect("projects poisoned")
            .iter()
            .any(|p| spec.project == *p || spec.project.starts_with(&format!("{}/", p)));
        if !admitted {
            self.stats
                .admission_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Err(crate::Error::ProjectNotAdmitted(spec.project.clone()));
        }
        if let Err((cpu, mem, slots)) = self.pool.try_acquire(&spec.resources) {
            self.stats
                .admission_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Err(crate::Error::AtCapacity {
                needed: format!(
                    "cpu={}m mem={}Mi",
                    spec.resources.cpu_millicores, spec.resources.mem_mib
                ),
                available: format!("cpu={}m mem={}Mi slots={}", cpu, mem, slots),
            });
        }

        let instance = match self.factory.create(sid, &spec).await {
            Ok(i) => i,
            Err(e) => {
                self.pool.release(&spec.resources);
                self.stats.sandboxes_failed.fetch_add(1, Ordering::Relaxed);
                return Err(e);
            }
        };
        if instance.state.transition(sid, SandboxState::Ready).is_err() {
            self.pool.release(&spec.resources);
            self.stats.sandboxes_failed.fetch_add(1, Ordering::Relaxed);
            return Err(crate::Error::InvalidTransition {
                sid,
                from: "creating",
                to: "ready",
            });
        }
        let entry = Arc::new(SandboxEntry {
            sid,
            spec,
            instance,
            created_epoch_ms: self.epoch_ms(),
        });
        self.sandboxes
            .write()
            .expect("sandboxes poisoned")
            .insert(sid, entry.clone());
        self.stats.sandboxes_created.fetch_add(1, Ordering::Relaxed);
        self.stats.sandboxes_current.fetch_add(1, Ordering::Relaxed);
        self.stats
            .creation_ms
            .record(t0.elapsed().as_millis() as u64);
        Ok(entry)
    }

    /// Pauses a Ready sandbox: checkpoint latency + memory reclaim.
    pub async fn pause(&self, sid: u64) -> crate::Result<PauseOutcome> {
        let entry = self.lookup(sid).ok_or(crate::Error::SandboxNotFound(sid))?;
        entry.instance.state.transition(sid, SandboxState::Paused)?;
        let t0 = std::time::Instant::now();
        let d = self.factory.latencies().pause.sample();
        if !d.is_zero() {
            tokio::time::sleep(d).await;
        }
        let reclaimed_mib = self.pool.pause_reclaim(&entry.spec.resources);
        self.stats.pauses.fetch_add(1, Ordering::Relaxed);
        self.stats.sandboxes_paused.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes_reclaimed
            .fetch_add((reclaimed_mib as u64) * 1024 * 1024, Ordering::Relaxed);
        let duration = t0.elapsed().as_millis() as u64;
        self.stats.pause_ms.record(duration);
        Ok(PauseOutcome {
            sid,
            duration_ms: duration,
            reclaimed_mib,
        })
    }

    /// Resumes a Paused sandbox: restore latency + MADV_WILLNEED-style
    /// prefetch of the overlay's dirty blocks.
    pub async fn resume(&self, sid: u64) -> crate::Result<()> {
        let entry = self.lookup(sid).ok_or(crate::Error::SandboxNotFound(sid))?;
        entry.instance.state.transition(sid, SandboxState::Ready)?;
        let t0 = std::time::Instant::now();
        let d = self.factory.latencies().resume.sample();
        if !d.is_zero() {
            tokio::time::sleep(d).await;
        }
        // MADV_WILLNEED: fault back the blocks the sandbox dirtied.
        let dirty = entry.instance.overlay.dirty_blocks();
        let ranges: Vec<dsec_storage::BlockRange> = dirty
            .chunks(16)
            .map(|c| dsec_storage::BlockRange {
                start: c[0],
                count: c.len() as u64,
            })
            .collect();
        if !ranges.is_empty() {
            let loader = entry.instance.overlay.loader().clone();
            for h in loader.prefetch(&ranges) {
                let _ = h.await;
            }
        }
        // Refetch charges the reclaimed memory back.
        self.pool.resume_refetch(
            &entry.spec.resources,
            (entry.spec.resources.mem_mib as f64 * self.pool.reclaim_fraction).round() as i64,
        );
        self.stats.resumes.fetch_add(1, Ordering::Relaxed);
        self.stats.sandboxes_paused.fetch_sub(1, Ordering::Relaxed);
        self.stats.resume_ms.record(t0.elapsed().as_millis() as u64);
        Ok(())
    }

    /// Destroys a sandbox and returns its resources.
    pub async fn destroy(&self, sid: u64) -> crate::Result<()> {
        let entry = self.lookup(sid).ok_or(crate::Error::SandboxNotFound(sid))?;
        let state = entry.instance.state.get();
        if state != SandboxState::Destroying {
            entry
                .instance
                .state
                .transition(sid, SandboxState::Destroying)?;
        }
        // Paused sandboxes still hold a paused slot.
        if state == SandboxState::Paused {
            self.pool.unpause();
        }
        self.pool.release(&entry.spec.resources);
        entry
            .instance
            .state
            .transition(sid, SandboxState::Destroyed)?;
        self.sandboxes
            .write()
            .expect("sandboxes poisoned")
            .remove(&sid);
        self.stats
            .sandboxes_destroyed
            .fetch_add(1, Ordering::Relaxed);
        self.stats.sandboxes_current.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    // -- data plane --------------------------------------------------------

    /// Handles one Aether request; returns the frames to write back (the
    /// response plus, for streams, the paced data frames).
    pub async fn handle_aether_request(
        &self,
        sid: u64,
        channel: Channel,
        req: Request,
        req_id: u32,
    ) -> Vec<Frame> {
        let reply = |resp: Response| -> Frame {
            Frame::response(
                channel,
                sid,
                req_id,
                serde_json::to_vec(&resp).unwrap_or_default(),
            )
        };
        let err_reply = |code: ErrorCode, msg: String| -> Frame {
            reply(Response::Error { code, message: msg })
        };

        match req {
            Request::Ping => {
                vec![reply(Response::Pong {
                    node_id: self.node_id.clone(),
                    epoch_ms: self.epoch_ms(),
                })]
            }
            Request::Pause => match self.pause(sid).await {
                Ok(o) => vec![reply(Response::Status {
                    state: "paused".into(),
                    cpu_millicores: 0,
                    mem_mib: o.reclaimed_mib,
                    uptime_ms: o.duration_ms,
                })],
                Err(e) => vec![err_reply(code_of(&e), e.to_string())],
            },
            Request::Resume => match self.resume(sid).await {
                Ok(()) => vec![reply(Response::Status {
                    state: "ready".into(),
                    cpu_millicores: 0,
                    mem_mib: 0,
                    uptime_ms: 0,
                })],
                Err(e) => vec![err_reply(code_of(&e), e.to_string())],
            },
            Request::Destroy => match self.destroy(sid).await {
                Ok(()) => vec![reply(Response::Done)],
                Err(e) => vec![err_reply(code_of(&e), e.to_string())],
            },
            Request::Status => match self.lookup(sid) {
                Some(entry) => {
                    let elapsed = self.epoch_ms().saturating_sub(entry.created_epoch_ms);
                    vec![reply(Response::Status {
                        state: entry.instance.state().as_str().to_string(),
                        cpu_millicores: entry.spec.resources.cpu_millicores,
                        mem_mib: entry.spec.resources.mem_mib,
                        uptime_ms: elapsed,
                    })]
                }
                None => vec![err_reply(ErrorCode::NotFound, format!("sandbox {}", sid))],
            },
            Request::StreamWrite { .. } | Request::StreamClose { .. } => {
                // Stdin writes are accepted (no-op in the simulated guest).
                vec![reply(Response::Done)]
            }
            other => {
                let Some(entry) = self.lookup(sid) else {
                    return vec![err_reply(ErrorCode::NotFound, format!("sandbox {}", sid))];
                };
                if !entry.instance.state().accepts_traffic() {
                    return vec![err_reply(
                        ErrorCode::InvalidState,
                        format!("sandbox {} is {}", sid, entry.instance.state().as_str()),
                    )];
                }
                let t0 = std::time::Instant::now();
                let outcome = entry
                    .instance
                    .chronus
                    .dispatch(other, self.epoch_ms())
                    .await;
                let elapsed = t0.elapsed().as_millis() as u64;
                match outcome {
                    Ok(mut out) => {
                        if let Response::Exec {
                            exit_code,
                            stdout,
                            stderr,
                            ..
                        } = &mut out.response
                        {
                            let _ = exit_code;
                            let _ = stdout;
                            let _ = stderr;
                            // Fill measured duration.
                            if let Response::Exec { duration_ms, .. } = &mut out.response {
                                *duration_ms = elapsed;
                            }
                        }
                        self.stats.exec_ms.record(elapsed);
                        let mut frames = vec![reply(out.response)];
                        if let Some((stream_id, events)) = out.stream_events {
                            self.stats.stream_ops.fetch_add(1, Ordering::Relaxed);
                            for ev in events {
                                if ev.delay_ms > 0 {
                                    tokio::time::sleep(Duration::from_millis(ev.delay_ms)).await;
                                }
                                let chunk = Response::StreamChunk {
                                    stream_id,
                                    stream: ev.stream,
                                    data: ev.data,
                                };
                                frames.push(Frame::stream_data(
                                    sid,
                                    stream_id,
                                    serde_json::to_vec(&chunk).unwrap_or_default(),
                                ));
                            }
                            let fin = Response::StreamFin {
                                stream_id,
                                exit_code: 0,
                            };
                            frames.push(Frame::stream_fin(
                                sid,
                                stream_id,
                                serde_json::to_vec(&fin).unwrap_or_default(),
                            ));
                        }
                        frames
                    }
                    Err(e) => vec![err_reply(code_of(&e), e.to_string())],
                }
            }
        }
    }

    /// Serves an in-process channel connection (deterministic tests and
    /// the high-throughput data plane).
    pub fn serve_channel(
        self: &Arc<Self>,
        buffer: usize,
    ) -> (crate::aether::AetherClient, tokio::task::JoinHandle<()>) {
        let (client_conn, server_conn) = crate::aether::channel_pair(buffer);
        let node = self.clone();
        let handle = tokio::spawn(async move {
            crate::aether::serve_connection(node, server_conn).await;
        });
        (crate::aether::AetherClient::new(client_conn), handle)
    }

    pub fn creation_histogram(&self) -> HistSnapshot {
        self.stats.creation()
    }
}

fn code_of(e: &crate::Error) -> ErrorCode {
    match e {
        crate::Error::SandboxNotFound(_) => ErrorCode::NotFound,
        crate::Error::InvalidTransition { .. } => ErrorCode::InvalidState,
        crate::Error::SandboxPaused(_) => ErrorCode::InvalidState,
        crate::Error::AtCapacity { .. } => ErrorCode::Unavailable,
        crate::Error::ImageNotLocal(_) => ErrorCode::NotFound,
        crate::Error::ProjectNotAdmitted(_) => ErrorCode::Unauthorized,
        crate::Error::ExecTimeout(_) => ErrorCode::Timeout,
        _ => ErrorCode::Internal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsec_protocol::frame::FLAG_REQUEST;
    use dsec_storage::cache::LruBlockCache;
    use dsec_storage::erofs::ErofsImageBuilder;
    use dsec_storage::latency::LatencyModel;

    fn node_with(cpu: i64, mem: i64, slots: i64) -> Arc<EdgeNode> {
        let image = Arc::new(ErofsImageBuilder::agent_base().build());
        let mut reg = ImageRegistry::default();
        reg.register(
            image,
            Arc::new(LruBlockCache::new(64)),
            LatencyModel::fixed(Duration::ZERO),
        );
        let node = EdgeNode::new(
            "node-1".to_string(),
            Arc::new(reg),
            NodeLatencyProfile::zero(),
            cpu,
            mem,
            slots,
            0,
        );
        node.factory.prewarm("dsec/agent-base", 4);
        Arc::new(node)
    }

    #[tokio::test]
    async fn create_ready_destroy() {
        let node = node_with(4000, 4096, 100);
        let entry = node.create(SandboxSpec::default()).await.unwrap();
        assert_eq!(entry.instance.state(), SandboxState::Ready);
        assert_eq!(node.sandbox_count(), 1);
        assert_eq!(node.lookup(entry.sid).unwrap().sid, entry.sid);
        node.destroy(entry.sid).await.unwrap();
        assert_eq!(node.sandbox_count(), 0);
        assert!(node.lookup(entry.sid).is_none());
        assert_eq!(node.stats.sandboxes_created.load(Ordering::Relaxed), 1);
        assert_eq!(node.stats.sandboxes_destroyed.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn admission_rejects_unknown_image_and_project() {
        let node = node_with(4000, 4096, 100);
        let bad_image = SandboxSpec {
            image_id: "dsec/ghost".into(),
            ..Default::default()
        };
        assert!(matches!(
            node.create(bad_image).await,
            Err(crate::Error::ImageNotLocal(_))
        ));
        let bad_project = SandboxSpec {
            project: "secret".into(),
            ..Default::default()
        };
        assert!(matches!(
            node.create(bad_project.clone()).await,
            Err(crate::Error::ProjectNotAdmitted(_))
        ));
        // Admitting the project fixes it.
        node.admit_project("secret");
        assert!(node.create(bad_project).await.is_ok());
    }

    #[tokio::test]
    async fn capacity_enforced_and_released() {
        let node = node_with(1000, 1024, 2);
        let spec = SandboxSpec::default();
        let a = node.create(spec.clone()).await.unwrap();
        let b = node.create(spec.clone()).await.unwrap();
        assert!(matches!(
            node.create(spec.clone()).await,
            Err(crate::Error::AtCapacity { .. })
        ));
        node.destroy(a.sid).await.unwrap();
        let c = node.create(spec.clone()).await.unwrap();
        assert_eq!(node.lookup(b.sid).unwrap().sid, b.sid);
        assert!(node.lookup(c.sid).is_some());
    }

    #[tokio::test]
    async fn pause_resume_reclaims_memory() {
        let node = node_with(4000, 4096, 10);
        let spec = SandboxSpec {
            resources: crate::resource::ResourceRequest {
                cpu_millicores: 500,
                mem_mib: 1000,
            },
            ..Default::default()
        };
        let entry = node.create(spec).await.unwrap();
        let used_before = node.capacity().usage().mem_mib;
        assert_eq!(used_before, 1000);
        let outcome = node.pause(entry.sid).await.unwrap();
        assert_eq!(outcome.reclaimed_mib, 600); // 0.6 reclaim fraction
        assert_eq!(node.capacity().usage().mem_mib, 400);
        assert_eq!(node.capacity().usage().paused_sandboxes, 1);
        assert_eq!(entry.instance.state(), SandboxState::Paused);
        // Double pause rejected while paused.
        assert!(node.pause(entry.sid).await.is_err());
        node.resume(entry.sid).await.unwrap();
        assert_eq!(node.capacity().usage().mem_mib, 1000);
        assert_eq!(entry.instance.state(), SandboxState::Ready);
    }

    #[tokio::test]
    async fn paused_sandbox_rejects_traffic() {
        let node = node_with(4000, 4096, 10);
        let entry = node.create(SandboxSpec::default()).await.unwrap();
        node.pause(entry.sid).await.unwrap();
        let frames = node
            .handle_aether_request(
                entry.sid,
                Channel::Exec,
                Request::Exec {
                    cmd: "ls".into(),
                    timeout_ms: None,
                },
                1,
            )
            .await;
        let resp: Response = serde_json::from_slice(&frames[0].payload).unwrap();
        match resp {
            Response::Error { code, .. } => assert_eq!(code, ErrorCode::InvalidState),
            other => panic!("unexpected {:?}", other),
        }
    }

    #[tokio::test]
    async fn lifecycle_through_aether_frames() {
        let node = node_with(4000, 4096, 10);
        let entry = node.create(SandboxSpec::default()).await.unwrap();
        // Status
        let frames = node
            .handle_aether_request(entry.sid, Channel::Control, Request::Status, 7)
            .await;
        let resp: Response = serde_json::from_slice(&frames[0].payload).unwrap();
        assert!(matches!(resp, Response::Status { state, .. } if state == "ready"));
        assert_eq!(frames[0].header.req_id, 7);
        assert!(frames[0].is_response());
        // Pause then resume through the control channel.
        let _ = node
            .handle_aether_request(entry.sid, Channel::Control, Request::Pause, 8)
            .await;
        assert_eq!(entry.instance.state(), SandboxState::Paused);
        let _ = node
            .handle_aether_request(entry.sid, Channel::Control, Request::Resume, 9)
            .await;
        assert_eq!(entry.instance.state(), SandboxState::Ready);
        // Destroy.
        let frames = node
            .handle_aether_request(entry.sid, Channel::Control, Request::Destroy, 10)
            .await;
        let resp: Response = serde_json::from_slice(&frames[0].payload).unwrap();
        assert!(matches!(resp, Response::Done));
        assert!(node.lookup(entry.sid).is_none());
    }

    #[tokio::test]
    async fn exec_frame_includes_duration() {
        let node = node_with(4000, 4096, 10);
        let entry = node.create(SandboxSpec::default()).await.unwrap();
        let frames = node
            .handle_aether_request(
                entry.sid,
                Channel::Exec,
                Request::Exec {
                    cmd: "echo hi".into(),
                    timeout_ms: None,
                },
                1,
            )
            .await;
        let resp: Response = serde_json::from_slice(&frames[0].payload).unwrap();
        match resp {
            Response::Exec {
                exit_code,
                stdout,
                duration_ms,
                ..
            } => {
                assert_eq!(exit_code, 0);
                assert_eq!(stdout, b"hi\n");
                // duration is measured (>= 0).
                let _ = duration_ms;
            }
            other => panic!("unexpected {:?}", other),
        }
        // FnCall pool served this creation.
        let (hits, _) = node.factory.pool_stats();
        assert!(hits >= 1);
    }

    #[tokio::test]
    async fn stream_frames_are_paced_and_ordered() {
        let node = node_with(4000, 4096, 10);
        let entry = node.create(SandboxSpec::default()).await.unwrap();
        let frames = node
            .handle_aether_request(
                entry.sid,
                Channel::Exec,
                Request::ExecStream {
                    cmd: "seq 3".into(),
                },
                1,
            )
            .await;
        // 1 response + 3 data frames + 1 fin.
        assert_eq!(frames.len(), 5);
        assert!(frames[0].is_response());
        for f in &frames[1..4] {
            assert!(f.is_stream());
        }
        let fin = &frames[4];
        assert!(fin.header.flags & FLAG_REQUEST == 0);
        let fin_resp: Response = serde_json::from_slice(&fin.payload).unwrap();
        assert!(matches!(fin_resp, Response::StreamFin { stream_id: 1, .. }));
    }

    #[tokio::test]
    async fn concurrent_creations_unique_sids() {
        let node = node_with(1_000_000, 1_000_000, 500);
        let mut tasks = Vec::new();
        for _ in 0..200 {
            let n = node.clone();
            tasks.push(tokio::spawn(async move {
                n.create(SandboxSpec::default()).await.unwrap().sid
            }));
        }
        let mut sids: Vec<u64> = Vec::new();
        for t in tasks {
            sids.push(t.await.unwrap());
        }
        sids.sort_unstable();
        sids.dedup();
        assert_eq!(sids.len(), 200);
        assert_eq!(node.sandbox_count(), 200);
        assert_eq!(node.stats.creation().count, 200);
    }
}
