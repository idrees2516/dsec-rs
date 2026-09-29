//! The `Sandbox` handle: the libdsec sandbox object.
//!
//! Management ops (pause/resume/destroy) go through the REST plane;
//! data ops (exec, files, HTTP, streams) go through the node's Aether
//! client, correlating by sandbox id.

use std::sync::Arc;

use dsec_control::model::SandboxRecord;
use dsec_protocol::frame::Channel;
use dsec_protocol::message::{Request, Response};
use dsec_runtime::AetherClient;

use crate::error::{Error, Result};
use crate::http::HttpClient;

/// Blocking exec result.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecResult {
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub duration_ms: u64,
}

/// File metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FileStat {
    pub size: u64,
    pub is_dir: bool,
}

pub struct Sandbox {
    record: SandboxRecord,
    aether: Arc<AetherClient>,
    rest: Arc<HttpClient>,
}

impl std::fmt::Debug for Sandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sandbox")
            .field("sid", &self.record.sid)
            .field("node_id", &self.record.node_id)
            .field("state", &self.record.state)
            .finish()
    }
}

impl Sandbox {
    pub(crate) fn new(
        record: SandboxRecord,
        aether: Arc<AetherClient>,
        rest: Arc<HttpClient>,
    ) -> Self {
        Sandbox {
            record,
            aether,
            rest,
        }
    }

    pub fn sid(&self) -> u64 {
        self.record.sid
    }

    pub fn node_id(&self) -> &str {
        &self.record.node_id
    }

    pub fn record(&self) -> &SandboxRecord {
        &self.record
    }

    async fn call(&self, channel: Channel, req: Request) -> Result<Response> {
        Ok(self.aether.call(self.record.sid, channel, req).await?)
    }

    // -- exec ----------------------------------------------------------------

    /// Runs a command to completion (default 30 s timeout).
    pub async fn execute(&self, cmd: &str) -> Result<ExecResult> {
        self.execute_with_timeout(cmd, None).await
    }

    pub async fn execute_with_timeout(
        &self,
        cmd: &str,
        timeout_ms: Option<u64>,
    ) -> Result<ExecResult> {
        match self
            .call(
                Channel::Exec,
                Request::Exec {
                    cmd: cmd.to_string(),
                    timeout_ms,
                },
            )
            .await?
        {
            Response::Exec {
                exit_code,
                stdout,
                stderr,
                duration_ms,
            } => Ok(ExecResult {
                exit_code,
                stdout,
                stderr,
                duration_ms,
            }),
            Response::Error { code, message } => {
                Err(Error::Protocol(dsec_protocol::Error::Protocol {
                    code: code as u32,
                    message,
                }))
            }
            other => Err(Error::SandboxUnusable(
                self.record.sid,
                format!("unexpected exec response {:?}", other),
            )),
        }
    }

    /// Opens a streamed exec; chunks arrive as paced stream events.
    pub async fn execute_stream(&self, cmd: &str) -> Result<StreamOutput> {
        let handle = self.aether.open_stream(self.record.sid, cmd).await?;
        Ok(StreamOutput { handle })
    }

    /// Opens a persistent Chronus session (cwd + env survive across execs).
    pub async fn open_session(
        &self,
        cwd: &str,
        env: std::collections::HashMap<String, String>,
    ) -> Result<u32> {
        match self
            .call(
                Channel::Exec,
                Request::SessionOpen {
                    cwd: cwd.to_string(),
                    env,
                },
            )
            .await?
        {
            Response::SessionOpened { session_id } => Ok(session_id),
            other => Err(Error::SandboxUnusable(
                self.record.sid,
                format!("unexpected session response {:?}", other),
            )),
        }
    }

    // -- filesystem ------------------------------------------------------------

    pub async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        match self
            .call(
                Channel::Fs,
                Request::FsRead {
                    path: path.to_string(),
                },
            )
            .await?
        {
            Response::FileData { data } => Ok(data),
            Response::Error { code, message } => {
                Err(Error::Protocol(dsec_protocol::Error::Protocol {
                    code: code as u32,
                    message,
                }))
            }
            other => Err(Error::SandboxUnusable(
                self.record.sid,
                format!("unexpected fs response {:?}", other),
            )),
        }
    }

    pub async fn write_file(&self, path: &str, data: &[u8]) -> Result<()> {
        match self
            .call(
                Channel::Fs,
                Request::FsWrite {
                    path: path.to_string(),
                    data: data.to_vec(),
                    append: false,
                },
            )
            .await?
        {
            Response::Done => Ok(()),
            Response::Error { code, message } => {
                Err(Error::Protocol(dsec_protocol::Error::Protocol {
                    code: code as u32,
                    message,
                }))
            }
            other => Err(Error::SandboxUnusable(
                self.record.sid,
                format!("unexpected fs response {:?}", other),
            )),
        }
    }

    pub async fn append_file(&self, path: &str, data: &[u8]) -> Result<()> {
        match self
            .call(
                Channel::Fs,
                Request::FsWrite {
                    path: path.to_string(),
                    data: data.to_vec(),
                    append: true,
                },
            )
            .await?
        {
            Response::Done => Ok(()),
            Response::Error { code, message } => {
                Err(Error::Protocol(dsec_protocol::Error::Protocol {
                    code: code as u32,
                    message,
                }))
            }
            other => Err(Error::SandboxUnusable(
                self.record.sid,
                format!("unexpected fs response {:?}", other),
            )),
        }
    }

    pub async fn list_dir(&self, path: &str) -> Result<Vec<String>> {
        match self
            .call(
                Channel::Fs,
                Request::FsList {
                    path: path.to_string(),
                },
            )
            .await?
        {
            Response::Entries { names } => Ok(names),
            other => Err(Error::SandboxUnusable(
                self.record.sid,
                format!("unexpected fs response {:?}", other),
            )),
        }
    }

    pub async fn stat(&self, path: &str) -> Result<FileStat> {
        match self
            .call(
                Channel::Fs,
                Request::FsStat {
                    path: path.to_string(),
                },
            )
            .await?
        {
            Response::Stat { size, is_dir } => Ok(FileStat { size, is_dir }),
            other => Err(Error::SandboxUnusable(
                self.record.sid,
                format!("unexpected fs response {:?}", other),
            )),
        }
    }

    pub async fn mkdir_p(&self, path: &str) -> Result<()> {
        match self
            .call(
                Channel::Fs,
                Request::FsMkdir {
                    path: path.to_string(),
                },
            )
            .await?
        {
            Response::Done => Ok(()),
            other => Err(Error::SandboxUnusable(
                self.record.sid,
                format!("unexpected fs response {:?}", other),
            )),
        }
    }

    pub async fn rm(&self, path: &str) -> Result<()> {
        match self
            .call(
                Channel::Fs,
                Request::FsRm {
                    path: path.to_string(),
                },
            )
            .await?
        {
            Response::Done => Ok(()),
            other => Err(Error::SandboxUnusable(
                self.record.sid,
                format!("unexpected fs response {:?}", other),
            )),
        }
    }

    // -- proxied HTTP ------------------------------------------------------------

    pub async fn http_get(&self, url: &str) -> Result<(u16, Vec<u8>)> {
        match self
            .call(
                Channel::Http,
                Request::HttpGet {
                    url: url.to_string(),
                    headers: Vec::new(),
                },
            )
            .await?
        {
            Response::Http { status, body } => Ok((status, body)),
            other => Err(Error::SandboxUnusable(
                self.record.sid,
                format!("unexpected http response {:?}", other),
            )),
        }
    }

    // -- lifecycle (management plane) ---------------------------------------------

    pub async fn pause(&self) -> Result<()> {
        let _: SandboxRecord = self
            .rest
            .post(
                &format!("/v1/sandboxes/{}/pause", self.record.sid),
                &serde_json::json!({}),
            )
            .await?;
        Ok(())
    }

    pub async fn resume(&self) -> Result<()> {
        let _: SandboxRecord = self
            .rest
            .post(
                &format!("/v1/sandboxes/{}/resume", self.record.sid),
                &serde_json::json!({}),
            )
            .await?;
        Ok(())
    }

    /// Destroys the sandbox via the control plane.
    pub async fn destroy(self) -> Result<SandboxRecord> {
        self.rest
            .delete(&format!("/v1/sandboxes/{}", self.record.sid))
            .await
    }

    /// Current runtime state (data-plane probe).
    pub async fn state(&self) -> Result<String> {
        crate::transport::ping(&self.aether, self.record.sid).await
    }

    /// Resets transient state (pool release path): clears /tmp.
    pub async fn reset(&self) -> Result<()> {
        if let Ok(children) = self.list_dir("/tmp").await {
            for child in children {
                let path = format!("/tmp/{}", child);
                // Best-effort; some entries may be dirs.
                if self.rm(&path).await.is_err() {
                    // Recursive clear through the shell as fallback.
                    let _ = self.execute(&format!("rm -r {}", path)).await;
                }
            }
        }
        Ok(())
    }
}

/// Streamed exec output.
pub struct StreamOutput {
    handle: dsec_runtime::aether::StreamHandle,
}

impl StreamOutput {
    /// Waits for the next stream event (chunk or fin).
    pub async fn next(&mut self) -> Option<dsec_runtime::aether::StreamEvent> {
        self.handle.next().await
    }

    /// Collects the whole stream (convenience for tests).
    pub async fn collect(mut self) -> Result<ExecResult> {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut exit_code = 0;
        while let Some(ev) = self.handle.next().await {
            match ev {
                dsec_runtime::aether::StreamEvent::Chunk { data, stream, .. } => match stream {
                    dsec_protocol::message::Stdio::Stdout => stdout.extend_from_slice(&data),
                    dsec_protocol::message::Stdio::Stderr => stderr.extend_from_slice(&data),
                },
                dsec_runtime::aether::StreamEvent::Fin {
                    exit_code: code, ..
                } => {
                    exit_code = code;
                    break;
                }
            }
        }
        Ok(ExecResult {
            exit_code,
            stdout,
            stderr,
            duration_ms: 0,
        })
    }
}
