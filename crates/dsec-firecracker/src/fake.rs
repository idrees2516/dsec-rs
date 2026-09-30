//! In-process fake of the Firecracker VMM API for testing the driver
//! over a REAL unix socket.
//!
//! The fake implements the request/state subset the driver uses
//! (machine-config, boot-source, drives, actions, vm state,
//! snapshots), enforces ordering with proper 4xx responses, records
//! every request for assertions, and touches the filesystem for
//! snapshot paths so restore flows behave like the real VMM.
//!
//! It is `pub` so downstream users can test their own integration
//! wiring without a hypervisor.

use std::path::Path;
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};

/// Recorded request: `"<METHOD> <path>"`.
pub type RequestLog = Arc<Mutex<Vec<String>>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum VmState {
    #[default]
    NotStarted,
    Running,
    Paused,
    Exited,
}

#[derive(Default)]
struct Inner {
    machine_config: Option<String>,
    boot_source: Option<String>,
    drives: Vec<String>,
    state: VmState,
    snapshotted: Option<bool>, // Some(diff_flag)
    restored: bool,
    exited_cleanly: bool,
}

/// A running fake VMM bound to one API socket.
pub struct FakeVmm {
    pub requests: RequestLog,
    inner: Arc<Mutex<Inner>>,
}

impl FakeVmm {
    /// Spawns the fake server on `sock` (best-effort removes a stale
    /// socket file first). Returns the fake for assertions; cloning the
    /// returned `Arc` shares the same log/state.
    pub fn spawn(sock: &Path) -> std::io::Result<std::sync::Arc<Self>> {
        let _ = std::fs::remove_file(sock);
        if let Some(dir) = sock.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let listener = UnixListener::bind(sock)?;
        let requests: RequestLog = Arc::new(Mutex::new(Vec::new()));
        let inner = Arc::new(Mutex::new(Inner::default()));
        {
            let requests = requests.clone();
            let inner = inner.clone();
            tokio::spawn(async move {
                loop {
                    let Ok((stream, _)) = listener.accept().await else {
                        break;
                    };
                    let requests = requests.clone();
                    let inner = inner.clone();
                    tokio::spawn(async move {
                        let _ = serve_stream(stream, requests, inner).await;
                    });
                }
            });
        }
        Ok(std::sync::Arc::new(FakeVmm { requests, inner }))
    }

    /// Current guest state as a string ("not_started" / "running" /
    /// "paused" / "exited").
    pub fn state(&self) -> &'static str {
        match self.inner.lock().expect("fake vmm poisoned").state {
            VmState::NotStarted => "not_started",
            VmState::Running => "running",
            VmState::Paused => "paused",
            VmState::Exited => "exited",
        }
    }

    /// Whether a snapshot was taken and whether it was diff-mode.
    pub fn snapshot_diff(&self) -> Option<bool> {
        self.inner.lock().expect("fake vmm poisoned").snapshotted
    }

    pub fn restored(&self) -> bool {
        self.inner.lock().expect("fake vmm poisoned").restored
    }

    pub fn exited_cleanly(&self) -> bool {
        self.inner.lock().expect("fake vmm poisoned").exited_cleanly
    }
}

async fn serve_stream(
    mut stream: UnixStream,
    requests: RequestLog,
    inner: Arc<Mutex<Inner>>,
) -> std::io::Result<()> {
    // Tiny persistent-buffer HTTP parsing: requests are small.
    let mut buf: Vec<u8> = Vec::with_capacity(1024);
    let mut chunk = [0u8; 2048];
    loop {
        // Read until a full request (headers + content-length body).
        let (method, path, body, consumed) = loop {
            if let Some(parsed) = try_parse(&buf) {
                break parsed;
            }
            let n = stream.read(&mut chunk).await?;
            if n == 0 {
                return Ok(()); // client closed
            }
            buf.extend_from_slice(&chunk[..n]);
        };
        buf.drain(..consumed);
        requests
            .lock()
            .expect("fake vmm poisoned")
            .push(format!("{method} {path}"));
        let (status, out_body) = handle(&method, &path, &body, &inner);
        let reason = match status {
            200 => "OK",
            204 => "No Content",
            400 => "Bad Request",
            404 => "Not Found",
            409 => "Conflict",
            _ => "Error",
        };
        let resp = if out_body.is_empty() {
            format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n"
            )
        } else {
            format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{}",
                out_body.len(),
                out_body
            )
        };
        stream.write_all(resp.as_bytes()).await?;
        stream.flush().await?;
    }
}

/// Parses one request from `buf`; returns (method, path, body, bytes consumed).
fn try_parse(buf: &[u8]) -> Option<(String, String, String, usize)> {
    let head_end = buf.windows(4).position(|w| w == b"\r\n\r\n")?;
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.lines();
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let content_length: usize = lines
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            if k.eq_ignore_ascii_case("content-length") {
                v.trim().parse().ok()
            } else {
                None
            }
        })
        .unwrap_or(0);
    let total = head_end + 4 + content_length;
    if buf.len() < total {
        return None;
    }
    let body = String::from_utf8_lossy(&buf[head_end + 4..total]).to_string();
    Some((method, path, body, total))
}

fn handle(method: &str, path: &str, body: &str, inner: &Arc<Mutex<Inner>>) -> (u16, String) {
    let mut vm = inner.lock().expect("fake vmm poisoned");
    let started = matches!(vm.state, VmState::Running | VmState::Paused);
    match (method, path) {
        ("PUT", "/machine-config") => {
            if started {
                return err400("VM already started");
            }
            vm.machine_config = Some(body.to_string());
            (204, String::new())
        }
        ("PATCH", "/machine-config") => {
            if matches!(vm.state, VmState::Exited) {
                return err400("VM exited");
            }
            (204, String::new())
        }
        ("PUT", "/boot-source") => {
            if started {
                return err400("VM already started");
            }
            if !body.contains("kernel_image_path") {
                return err400("kernel_image_path required");
            }
            vm.boot_source = Some(body.to_string());
            (204, String::new())
        }
        ("PUT", p) if p.starts_with("/drives/") => {
            if started {
                return err400("VM already started");
            }
            vm.drives.push(p.to_string());
            (204, String::new())
        }
        ("GET", "/vm") => {
            let state = match vm.state {
                VmState::NotStarted => "Not started",
                VmState::Running => "Running",
                VmState::Paused => "Paused",
                VmState::Exited => "Exited",
            };
            (200, format!("{{\"state\":\"{state}\"}}"))
        }
        ("PATCH", "/vm") => {
            let want_paused = body.contains("Paused");
            let want_resumed = body.contains("Resumed");
            match (vm.state, want_paused, want_resumed) {
                (VmState::Running, true, false) => {
                    vm.state = VmState::Paused;
                    (204, String::new())
                }
                (VmState::Paused, false, true) => {
                    vm.state = VmState::Running;
                    (204, String::new())
                }
                (s, _, _) => err400(&format!("invalid transition from {:?} with body {body}", s)),
            }
        }
        ("PUT", "/actions") => {
            if body.contains("InstanceStart") {
                if vm.machine_config.is_none() || vm.boot_source.is_none() {
                    return err400("machine-config and boot-source required before start");
                }
                if !vm.drives.iter().any(|d| d.contains("rootfs")) {
                    return err400("root drive required before start");
                }
                vm.state = VmState::Running;
                (204, String::new())
            } else if body.contains("SendCtrlAltDel") {
                vm.exited_cleanly = true;
                vm.state = VmState::Exited;
                (204, String::new())
            } else {
                err400("unknown action")
            }
        }
        ("PUT", "/snapshots/create") => {
            if !matches!(vm.state, VmState::Paused) {
                return err400("snapshot requires a paused VM");
            }
            let diff = body.contains("\"diff\":true") || body.contains("\"diff\": true");
            vm.snapshotted = Some(diff);
            // Touch the requested files like the real VMM would.
            let _ = touch_paths(body);
            (204, String::new())
        }
        ("PUT", "/snapshots/load") => {
            vm.restored = true;
            vm.state = VmState::Running;
            (204, String::new())
        }
        (_, "/mmds") => (204, String::new()),
        _ => (404, "{\"fault_message\":\"unknown endpoint\"}".into()),
    }
}

fn err400(msg: &str) -> (u16, String) {
    (
        400,
        format!("{{\"fault_message\":\"{}\"}}", msg.replace('"', "'")),
    )
}

/// Creates the snapshot_path / mem_file_path files named in a
/// snapshots/create body (so restore flows see real files).
fn touch_paths(body: &str) -> std::io::Result<()> {
    let v: serde_json::Value = serde_json::from_str(body)?;
    for key in ["snapshot_path", "mem_file_path"] {
        if let Some(p) = v.get(key).and_then(|p| p.as_str()) {
            if let Some(dir) = std::path::Path::new(p).parent() {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(p, b"fake-snapshot")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fake_vmm_enforces_boot_order() {
        let dir = std::env::temp_dir().join(format!("fake-fc-{}", std::process::id()));
        let sock = dir.join("fc.sock");
        let fake = FakeVmm::spawn(&sock).unwrap();
        let mut client = crate::client::FcClient::new(&sock, std::time::Duration::from_secs(2));
        // Start before configuration must fail.
        let r = client
            .put("/actions", r#"{"action_type":"InstanceStart"}"#)
            .await
            .unwrap();
        assert_eq!(r.status, 400);
        // Full sequence succeeds.
        client
            .put("/machine-config", r#"{"vcpu_count":1,"mem_size_mib":256}"#)
            .await
            .unwrap();
        client
            .put(
                "/boot-source",
                r#"{"kernel_image_path":"/k","boot_args":"x"}"#,
            )
            .await
            .unwrap();
        client
            .put(
                "/drives/rootfs",
                r#"{"path_on_host":"/r","is_root_device":true,"is_read_only":true}"#,
            )
            .await
            .unwrap();
        let r = client
            .put("/actions", r#"{"action_type":"InstanceStart"}"#)
            .await
            .unwrap();
        assert!(r.is_success());
        assert_eq!(fake.state(), "running");
        // Pause → snapshot → resume.
        client.patch("/vm", r#"{"state":"Paused"}"#).await.unwrap();
        let snap = dir.join("snap.bin");
        let mem = dir.join("mem.bin");
        let body = format!(
            r#"{{"snapshot_path":"{}","mem_file_path":"{}","diff":false}}"#,
            snap.display(),
            mem.display()
        );
        let r = client.put("/snapshots/create", &body).await.unwrap();
        assert!(r.is_success());
        assert!(snap.is_file());
        client.patch("/vm", r#"{"state":"Resumed"}"#).await.unwrap();
        assert_eq!(fake.state(), "running");
        // Unknown endpoint 404.
        let r = client.get("/nope").await.unwrap();
        assert_eq!(r.status, 404);
        assert!(r.fault().is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
