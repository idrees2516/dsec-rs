//! Data-plane transport: how the SDK reaches node runtimes.
//!
//! The paper's Aether terminates UDS (containers) or vsock (microVMs) at
//! the node; the SDK multiplexes every sandbox of one node over a single
//! connection. Here the transport trait abstracts where that connection
//! lives: in-process channels (deterministic simulation / tests) or real
//! UDS sockets. Connections are cached per node.

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use dsec_protocol::frame::Channel;
use dsec_protocol::message::{Request, Response};
use dsec_runtime::{AetherClient, EdgeNode};

use crate::error::{Error, Result};

pub type BoxFut<T> = Pin<Box<dyn Future<Output = T> + Send>>;

type ClientCache = Arc<Mutex<HashMap<String, Arc<AetherClient>>>>;

/// Supplies Aether clients per node (one multiplexed connection each).
pub trait Transport: Send + Sync {
    fn client_for(&self, node_id: &str) -> BoxFut<Result<Arc<AetherClient>>>;
}

/// In-process channel transport: talks straight to `EdgeNode`s by id.
pub struct ChannelTransport {
    nodes: HashMap<String, Arc<EdgeNode>>,
    clients: ClientCache,
}

impl ChannelTransport {
    /// Single-node convenience.
    pub fn new(node: Arc<EdgeNode>) -> Self {
        ChannelTransport::with_nodes(vec![node])
    }

    /// Multi-node routing (each node gets its own multiplexed connection).
    pub fn with_nodes(nodes: Vec<Arc<EdgeNode>>) -> Self {
        ChannelTransport {
            nodes: nodes.into_iter().map(|n| (n.node_id.clone(), n)).collect(),
            clients: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Transport for ChannelTransport {
    fn client_for(&self, node_id: &str) -> BoxFut<Result<Arc<AetherClient>>> {
        if let Some(c) = self
            .clients
            .lock()
            .expect("transport poisoned")
            .get(node_id)
        {
            let c = c.clone();
            return Box::pin(async move { Ok(c) });
        }
        let node = self.nodes.get(node_id).cloned();
        let cache = self.clients.clone();
        let node_id = node_id.to_string();
        Box::pin(async move {
            let node = node.ok_or_else(|| {
                Error::SandboxUnusable(0, format!("channel transport has no node {}", node_id))
            })?;
            let (client, _serve) = node.serve_channel(256);
            let client = Arc::new(client);
            cache
                .lock()
                .expect("transport poisoned")
                .insert(node_id, client.clone());
            Ok(client)
        })
    }
}

/// Real UDS transport (production data plane).
pub struct UdsTransport {
    root: PathBuf,
    clients: ClientCache,
}

impl UdsTransport {
    /// `root/<node_id>.sock` is the node's Aether endpoint.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        UdsTransport {
            root: root.into(),
            clients: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn socket_path(&self, node_id: &str) -> PathBuf {
        self.root.join(format!("{}.sock", node_id))
    }
}

impl Transport for UdsTransport {
    fn client_for(&self, node_id: &str) -> BoxFut<Result<Arc<AetherClient>>> {
        if let Some(c) = self
            .clients
            .lock()
            .expect("transport poisoned")
            .get(node_id)
        {
            let c = c.clone();
            return Box::pin(async move { Ok(c) });
        }
        let path = self.socket_path(node_id);
        let cache = self.clients.clone();
        let node_id = node_id.to_string();
        Box::pin(async move {
            let path_str = path.to_string_lossy().to_string();
            let client = AetherClient::connect_uds(&path_str).await?;
            let client = Arc::new(client);
            cache
                .lock()
                .expect("transport poisoned")
                .insert(node_id, client.clone());
            Ok(client)
        })
    }
}

/// Liveness probe through a transport (used by pool health checks).
pub async fn ping(client: &AetherClient, sid: u64) -> Result<String> {
    match client.call(sid, Channel::Control, Request::Status).await? {
        Response::Status { state, .. } => Ok(state),
        _ => Err(Error::SandboxUnusable(
            sid,
            "unexpected status response".into(),
        )),
    }
}
