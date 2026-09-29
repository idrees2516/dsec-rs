//! # dsec-sdk
//!
//! The Rust port of the paper's `libdsec` client library.
//!
//! API mapping (libdsec Python -> dsec-sdk Rust):
//!
//! | libdsec (paper)                       | dsec-sdk                                |
//! |---------------------------------------|-----------------------------------------|
//! | `dsec.DsecClient(token, endpoint)`    | [`DsecClient::new`]                     |
//! | `client.sandboxes.create(spec)`       | [`DsecClient::create_sandbox`]          |
//! | `sandbox.execute(cmd)`                | [`Sandbox::execute`]                    |
//! | `sandbox.execute_stream(cmd)`         | [`Sandbox::execute_stream`]             |
//! | `sandbox.filesystem.read_file(path)`  | [`Sandbox::read_file`]                  |
//! | `sandbox.filesystem.write_file(...)`  | [`Sandbox::write_file`]                 |
//! | `sandbox.http.get(url)`               | [`Sandbox::http_get`]                   |
//! | `sandbox.pause()` / `resume()`        | [`Sandbox::pause`] / [`Sandbox::resume`]|
//! | `sandbox.pool(...)`                   | [`SandboxPool`]                         |
//!
//! Two planes, exactly as in the paper:
//! - **management plane**: REST calls to the apiserver (`http::HttpClient`)
//! - **data plane**: Aether frames to the node runtime — in-process
//!   channel transport (deterministic simulation) or UDS (`transport`).
//!
//! Retries use exponential backoff; every call carries a request id and
//! honors a configurable timeout.

pub mod error;
pub mod http;
pub mod integration;
pub mod pool;
pub mod sandbox;
pub mod transport;

pub use error::{Error, Result};
pub use pool::{PoolConfig, SandboxPool};
pub use sandbox::{ExecResult, FileStat, Sandbox, StreamOutput};
pub use transport::Transport;

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use dsec_control::model::{SandboxRecord, SandboxSpec};

/// Where the apiserver listens.
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub host: String,
    pub port: u16,
}

impl Endpoint {
    pub fn http(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    pub fn localhost(port: u16) -> Self {
        Endpoint {
            host: "127.0.0.1".to_string(),
            port,
        }
    }
}

/// Client request-id sequence.
pub(crate) fn next_request_id() -> u32 {
    static SEQ: AtomicU32 = AtomicU32::new(1);
    SEQ.fetch_add(1, Ordering::Relaxed)
}

/// The libdsec client.
pub struct DsecClient {
    token: String,
    pub(crate) rest: http::HttpClient,
    /// Data-plane transport per node.
    transport: Arc<dyn Transport>,
}

impl DsecClient {
    /// `transport` supplies Aether clients per node (channel or UDS).
    pub fn new(token: String, endpoint: Endpoint, transport: Arc<dyn Transport>) -> Self {
        DsecClient {
            rest: http::HttpClient::new(endpoint, token.clone()),
            transport,
            token,
        }
    }

    pub fn token(&self) -> &str {
        &self.token
    }

    pub fn with_retry(self, max_attempts: u32, base_backoff: std::time::Duration) -> Self {
        DsecClient {
            token: self.token,
            rest: self.rest.with_retry(max_attempts, base_backoff),
            transport: self.transport,
        }
    }

    /// Creates a sandbox through the control plane and attaches a data
    /// plane channel to its node.
    pub async fn create_sandbox(&self, spec: SandboxSpec) -> Result<Sandbox> {
        let record: SandboxRecord = self.rest.post("/v1/sandboxes", &spec).await?;
        self.attach(record).await
    }

    pub async fn list_sandboxes(&self, project: Option<&str>) -> Result<Vec<SandboxRecord>> {
        let path = match project {
            Some(p) => format!("/v1/sandboxes?project={}", p),
            None => "/v1/sandboxes".to_string(),
        };
        self.rest.get(&path).await
    }

    pub async fn get_sandbox(&self, sid: u64) -> Result<SandboxRecord> {
        self.rest.get(&format!("/v1/sandboxes/{}", sid)).await
    }

    pub async fn cluster(&self) -> Result<dsec_control::apiserver::ClusterResp> {
        self.rest.get("/v1/cluster").await
    }

    pub async fn destroy_sandbox(&self, sid: u64) -> Result<SandboxRecord> {
        self.rest.delete(&format!("/v1/sandboxes/{}", sid)).await
    }

    /// Wraps an existing record into a data-plane-attached handle.
    pub async fn attach(&self, record: SandboxRecord) -> Result<Sandbox> {
        let aether = self.transport.client_for(&record.node_id).await?;
        Ok(Sandbox::new(record, aether, Arc::new(self.rest.clone())))
    }
}
