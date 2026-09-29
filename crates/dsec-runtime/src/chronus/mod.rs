//! Chronus: the in-sandbox session abstraction.
//!
//! A session is a persistent shell context (cwd + env) multiplexed over
//! the sandbox's Aether channel. The paper's Chronus wraps shell
//! execution, filesystem access, proxied HTTP and streaming I/O behind
//! one interface; the dispatcher below maps protocol requests onto those
//! services.

pub mod exec;
pub mod http;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;

use dsec_protocol::message::{Request, Response};
use dsec_storage::imagefs::LayeredImage;

pub use exec::{ExecEvent, ExecOutput};

/// Default per-command timeout when the caller does not specify one.
pub const DEFAULT_EXEC_TIMEOUT_MS: u64 = 30_000;

/// One Chronus shell session.
#[derive(Debug, Clone)]
pub struct Session {
    pub id: u32,
    pub cwd: String,
    pub env: HashMap<String, String>,
    pub opened_ms: u64,
    pub commands_run: u64,
    pub last_exit_code: i32,
}

impl Session {
    pub fn new(id: u32, cwd: String, env: HashMap<String, String>, opened_ms: u64) -> Self {
        let mut env = env;
        env.entry("USER".to_string())
            .or_insert_with(|| "agent".to_string());
        env.entry("HOME".to_string())
            .or_insert_with(|| "/home/agent".to_string());
        env.entry("SHELL".to_string())
            .or_insert_with(|| "/bin/sh".to_string());
        Session {
            id,
            cwd,
            env,
            opened_ms,
            commands_run: 0,
            last_exit_code: 0,
        }
    }
}

/// Session table for one sandbox.
#[derive(Debug, Default)]
pub struct SessionManager {
    sessions: Mutex<HashMap<u32, Session>>,
    next_id: AtomicU32,
    total_opened: AtomicU64,
}

impl SessionManager {
    pub fn new() -> Self {
        SessionManager::default()
    }

    /// Opens a session; returns its id.
    pub fn open(&self, cwd: &str, env: HashMap<String, String>, epoch_ms: u64) -> u32 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let session = Session::new(id, cwd.to_string(), env, epoch_ms);
        self.sessions
            .lock()
            .expect("sessions poisoned")
            .insert(id, session);
        self.total_opened.fetch_add(1, Ordering::Relaxed);
        id
    }

    /// Ensures a session exists (implicit session 0 for plain exec calls).
    pub fn ensure_default(&self, epoch_ms: u64) -> u32 {
        let mut sessions = self.sessions.lock().expect("sessions poisoned");
        if let Some(s) = sessions.get(&0) {
            return s.id;
        }
        let session = Session::new(0, "/".to_string(), HashMap::new(), epoch_ms);
        sessions.insert(0, session);
        0
    }

    /// Seeds session 0 with the sandbox-level environment (used at
    /// backend creation so bare `exec` calls see the sandbox env).
    pub fn seed_default(&self, env: HashMap<String, String>) {
        let session = Session::new(0, "/".to_string(), env, 0);
        self.sessions
            .lock()
            .expect("sessions poisoned")
            .insert(0, session);
    }

    pub fn get(&self, id: u32) -> Option<Session> {
        self.sessions
            .lock()
            .expect("sessions poisoned")
            .get(&id)
            .cloned()
    }

    /// Mutates a session after a command ran.
    pub fn after_exec(&self, id: u32, new_cwd: &str, exit_code: i32) {
        if let Some(s) = self
            .sessions
            .lock()
            .expect("sessions poisoned")
            .get_mut(&id)
        {
            s.cwd = new_cwd.to_string();
            s.commands_run += 1;
            s.last_exit_code = exit_code;
        }
    }

    pub fn close(&self, id: u32) -> bool {
        self.sessions
            .lock()
            .expect("sessions poisoned")
            .remove(&id)
            .is_some()
    }

    pub fn count(&self) -> usize {
        self.sessions.lock().expect("sessions poisoned").len()
    }

    pub fn total_opened(&self) -> u64 {
        self.total_opened.load(Ordering::Relaxed)
    }
}

/// Dispatches a protocol request against a sandbox's guest state.
///
/// Stream-open requests return the full event list plus a stream id; the
/// Edge pushes them as paced frames. All other requests map 1:1 onto
/// responses.
pub struct DispatchOutcome {
    pub response: Response,
    /// For `ExecStream`: events to push as stream frames.
    pub stream_events: Option<(u32, Vec<ExecEvent>)>,
}

impl DispatchOutcome {
    fn plain(response: Response) -> Self {
        DispatchOutcome {
            response,
            stream_events: None,
        }
    }
}

/// Chronus service bound to one sandbox.
#[derive(Debug)]
pub struct Chronus {
    pub sessions: SessionManager,
    fs: std::sync::Arc<LayeredImage>,
    hostname: String,
    /// Stream ids are per-sandbox, allocated by the Edge.
    next_stream: AtomicU32,
    http: http::EgressRouter,
    exec_count: AtomicU64,
}

impl Chronus {
    pub fn new(fs: std::sync::Arc<LayeredImage>, hostname: String) -> Self {
        Chronus {
            sessions: SessionManager::new(),
            fs,
            hostname,
            next_stream: AtomicU32::new(1),
            http: http::EgressRouter::default(),
            exec_count: AtomicU64::new(0),
        }
    }

    pub fn fs(&self) -> &std::sync::Arc<LayeredImage> {
        &self.fs
    }

    pub fn hostname(&self) -> &str {
        &self.hostname
    }

    pub fn execs(&self) -> u64 {
        self.exec_count.load(Ordering::Relaxed)
    }

    /// Handles one request. `epoch_ms` is injectable for determinism.
    pub async fn dispatch(&self, req: Request, epoch_ms: u64) -> crate::Result<DispatchOutcome> {
        use Request::*;
        match req {
            Ping => Ok(DispatchOutcome::plain(Response::Pong {
                node_id: self.hostname.clone(),
                epoch_ms,
            })),
            SessionOpen { cwd, env } => {
                let id = self.sessions.open(&cwd, env, epoch_ms);
                Ok(DispatchOutcome::plain(Response::SessionOpened {
                    session_id: id,
                }))
            }
            SessionClose => Ok(DispatchOutcome::plain(Response::SessionClosed)),
            Exec { cmd, timeout_ms } => {
                let sid = self.sessions.ensure_default(epoch_ms);
                let session = self.sessions.get(sid).expect("default session");
                let timeout = timeout_ms.unwrap_or(DEFAULT_EXEC_TIMEOUT_MS);
                let out = match tokio::time::timeout(
                    std::time::Duration::from_millis(timeout),
                    exec::run(
                        &cmd,
                        &self.fs,
                        &session.cwd,
                        &session.env,
                        &self.hostname,
                        epoch_ms,
                    ),
                )
                .await
                {
                    Ok(out) => out,
                    Err(_) => {
                        return Ok(DispatchOutcome::plain(Response::Error {
                            code: dsec_protocol::message::ErrorCode::Timeout,
                            message: format!("exec timed out after {} ms", timeout),
                        }))
                    }
                };
                self.sessions.after_exec(sid, &out.cwd_after, out.exit_code);
                self.exec_count.fetch_add(1, Ordering::Relaxed);
                Ok(DispatchOutcome::plain(Response::Exec {
                    exit_code: out.exit_code,
                    stdout: out.stdout(),
                    stderr: out.stderr(),
                    duration_ms: 0,
                }))
            }
            ExecStream { cmd } => {
                let sid = self.sessions.ensure_default(epoch_ms);
                let session = self.sessions.get(sid).expect("default session");
                let out = exec::run(
                    &cmd,
                    &self.fs,
                    &session.cwd,
                    &session.env,
                    &self.hostname,
                    epoch_ms,
                )
                .await;
                self.sessions.after_exec(sid, &out.cwd_after, out.exit_code);
                self.exec_count.fetch_add(1, Ordering::Relaxed);
                let stream_id = self.next_stream.fetch_add(1, Ordering::Relaxed);
                Ok(DispatchOutcome {
                    response: Response::StreamOpened { stream_id },
                    stream_events: Some((stream_id, out.events)),
                })
            }
            FsRead { path } => match self.fs.read_file(&path).await {
                Ok(data) => Ok(DispatchOutcome::plain(Response::FileData { data })),
                Err(e) => Ok(DispatchOutcome::plain(Response::Error {
                    code: dsec_protocol::message::ErrorCode::NotFound,
                    message: e.to_string(),
                })),
            },
            FsWrite { path, data, append } => {
                let r = if append {
                    self.fs.append_file(&path, &data).await
                } else {
                    self.fs.write_file(&path, &data).await
                };
                match r {
                    Ok(_) => Ok(DispatchOutcome::plain(Response::Done)),
                    Err(e) => Ok(DispatchOutcome::plain(Response::Error {
                        code: dsec_protocol::message::ErrorCode::Internal,
                        message: e.to_string(),
                    })),
                }
            }
            FsList { path } => match self.fs.list_dir(&path) {
                Ok(mut names) => {
                    names.sort();
                    Ok(DispatchOutcome::plain(Response::Entries { names }))
                }
                Err(e) => Ok(DispatchOutcome::plain(Response::Error {
                    code: dsec_protocol::message::ErrorCode::NotFound,
                    message: e.to_string(),
                })),
            },
            FsStat { path } => match self.fs.stat(&path) {
                Ok(m) => Ok(DispatchOutcome::plain(Response::Stat {
                    size: m.size,
                    is_dir: m.is_dir,
                })),
                Err(e) => Ok(DispatchOutcome::plain(Response::Error {
                    code: dsec_protocol::message::ErrorCode::NotFound,
                    message: e.to_string(),
                })),
            },
            FsMkdir { path } => match self.fs.mkdir_p(&path) {
                Ok(()) => Ok(DispatchOutcome::plain(Response::Done)),
                Err(e) => Ok(DispatchOutcome::plain(Response::Error {
                    code: dsec_protocol::message::ErrorCode::Internal,
                    message: e.to_string(),
                })),
            },
            FsRm { path } => match self.fs.rm(&path, false) {
                Ok(()) => Ok(DispatchOutcome::plain(Response::Done)),
                Err(e) => Ok(DispatchOutcome::plain(Response::Error {
                    code: dsec_protocol::message::ErrorCode::Internal,
                    message: e.to_string(),
                })),
            },
            HttpGet { url, headers } => {
                let resp = self.http.get(&url, &headers);
                Ok(DispatchOutcome::plain(Response::Http {
                    status: resp.status,
                    body: resp.body,
                }))
            }
            // Lifecycle requests are handled by the Edge, not Chronus.
            StreamWrite { .. } | StreamClose { .. } | Pause | Resume | Destroy | Status => Err(
                crate::Error::Other("lifecycle/stream request routed to Chronus".into()),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chronus() -> Chronus {
        let image =
            std::sync::Arc::new(dsec_storage::erofs::ErofsImageBuilder::agent_base().build());
        let loader = std::sync::Arc::new(dsec_storage::erofs::OnDemandLoader::new(
            image,
            std::sync::Arc::new(dsec_storage::cache::LruBlockCache::new(64)),
            dsec_storage::latency::LatencyModel::fixed(std::time::Duration::ZERO),
        ));
        let overlay = std::sync::Arc::new(dsec_storage::overlay::OverlayDev::new(loader));
        let fs = std::sync::Arc::new(LayeredImage::new(overlay));
        Chronus::new(fs, "sb-42".to_string())
    }

    #[tokio::test]
    async fn exec_through_session() {
        let c = chronus();
        let out = c
            .dispatch(
                Request::Exec {
                    cmd: "cat /etc/hostname".into(),
                    timeout_ms: None,
                },
                1,
            )
            .await
            .unwrap();
        match out.response {
            Response::Exec {
                exit_code, stdout, ..
            } => {
                assert_eq!(exit_code, 0);
                assert_eq!(stdout, b"sandbox-agent");
            }
            other => panic!("unexpected {:?}", other),
        }
        assert_eq!(c.execs(), 1);
    }

    #[tokio::test]
    async fn session_cwd_persists() {
        let c = chronus();
        let open = c
            .dispatch(
                Request::SessionOpen {
                    cwd: "/".into(),
                    env: HashMap::new(),
                },
                1,
            )
            .await
            .unwrap();
        let sid = match open.response {
            Response::SessionOpened { session_id } => session_id,
            other => panic!("unexpected {:?}", other),
        };
        let _ = c
            .dispatch(
                Request::Exec {
                    cmd: "cd /etc".into(),
                    timeout_ms: None,
                },
                2,
            )
            .await
            .unwrap();
        // Session dispatch uses ensure_default -> session 0; explicit session exec:
        let sess = c.sessions.get(sid).unwrap();
        assert_eq!(sess.cwd, "/"); // explicit session untouched by default-session exec
    }

    #[tokio::test]
    async fn exec_timeout_produces_error() {
        let c = chronus();
        let out = c
            .dispatch(
                Request::Exec {
                    cmd: "sleep 5000".into(),
                    timeout_ms: Some(30),
                },
                1,
            )
            .await
            .unwrap();
        match out.response {
            Response::Error { code, .. } => {
                assert_eq!(code, dsec_protocol::message::ErrorCode::Timeout);
            }
            other => panic!("unexpected {:?}", other),
        }
    }

    #[tokio::test]
    async fn fs_ops_via_protocol() {
        let c = chronus();
        let w = c
            .dispatch(
                Request::FsWrite {
                    path: "/tmp/x".into(),
                    data: b"hi".to_vec(),
                    append: false,
                },
                1,
            )
            .await
            .unwrap();
        assert!(matches!(w.response, Response::Done));
        let r = c
            .dispatch(
                Request::FsRead {
                    path: "/tmp/x".into(),
                },
                2,
            )
            .await
            .unwrap();
        match r.response {
            Response::FileData { data } => assert_eq!(data, b"hi"),
            other => panic!("unexpected {:?}", other),
        }
        let l = c
            .dispatch(
                Request::FsList {
                    path: "/etc".into(),
                },
                3,
            )
            .await
            .unwrap();
        match l.response {
            Response::Entries { names } => {
                assert_eq!(names, vec!["hostname", "os-release", "passwd"])
            }
            other => panic!("unexpected {:?}", other),
        }
        let st = c
            .dispatch(
                Request::FsStat {
                    path: "/etc".into(),
                },
                4,
            )
            .await
            .unwrap();
        match st.response {
            Response::Stat { is_dir, .. } => assert!(is_dir),
            other => panic!("unexpected {:?}", other),
        }
    }

    #[tokio::test]
    async fn exec_stream_returns_events() {
        let c = chronus();
        let out = c
            .dispatch(
                Request::ExecStream {
                    cmd: "seq 3".into(),
                },
                1,
            )
            .await
            .unwrap();
        let (stream_id, events) = out.stream_events.expect("stream events");
        assert_eq!(stream_id, 1);
        assert_eq!(events.len(), 3);
        assert!(matches!(out.response, Response::StreamOpened { .. }));
    }

    #[tokio::test]
    async fn http_get_fake_route() {
        let c = chronus();
        let out = c
            .dispatch(
                Request::HttpGet {
                    url: "https://api.internal/health".into(),
                    headers: vec![],
                },
                1,
            )
            .await
            .unwrap();
        match out.response {
            Response::Http { status, body } => {
                assert_eq!(status, 200);
                assert_eq!(body, b"{\"status\":\"ok\"}");
            }
            other => panic!("unexpected {:?}", other),
        }
        let missing = c
            .dispatch(
                Request::HttpGet {
                    url: "https://api.internal/nope".into(),
                    headers: vec![],
                },
                2,
            )
            .await
            .unwrap();
        match missing.response {
            Response::Http { status, .. } => assert_eq!(status, 404),
            other => panic!("unexpected {:?}", other),
        }
    }
}
