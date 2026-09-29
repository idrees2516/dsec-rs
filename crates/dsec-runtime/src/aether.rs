//! Aether: the per-sandbox proxy.
//!
//! One Aether connection per (client, node) pair multiplexes every
//! sandbox's traffic: frames carry (sandbox id, channel, request id) and
//! are demultiplexed server-side to the right Chronus session. The
//! transport is swappable — in-process channels for deterministic
//! simulation and high-throughput tests, real Unix domain sockets as the
//! production data plane (the paper switches the same layer between UDS
//! and vsock for microVMs).
//!
//! Wire behavior: request frames are answered with response frames
//! carrying the same request id; server-initiated stream frames carry
//! `FLAG_STREAM_DATA` and the stream id in the request-id field.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dsec_protocol::codec;
use dsec_protocol::frame::{Channel, Frame};
use dsec_protocol::message::{Request, Response, Stdio};
use dsec_protocol::{Error as ProtoError, Result as ProtoResult};

use crate::edge::EdgeNode;

/// A duplex connection, either in-process or a real UDS socket.
pub enum AetherConn {
    Channel {
        tx: tokio::sync::mpsc::Sender<Frame>,
        rx: tokio::sync::mpsc::Receiver<Frame>,
    },
    Unix(Box<tokio::net::UnixStream>),
}

/// Crossed mpsc pair: (client side, server side).
pub fn channel_pair(buffer: usize) -> (AetherConn, AetherConn) {
    let (c2s_tx, c2s_rx) = tokio::sync::mpsc::channel(buffer);
    let (s2c_tx, s2c_rx) = tokio::sync::mpsc::channel(buffer);
    (
        AetherConn::Channel {
            tx: s2c_tx,
            rx: c2s_rx,
        }, // client reads s2c, writes c2s
        AetherConn::Channel {
            tx: c2s_tx,
            rx: s2c_rx,
        },
    )
}

/// Cloneable writer half.
#[derive(Clone)]
pub enum AetherWriter {
    Channel(tokio::sync::mpsc::Sender<Frame>),
    Unix(Arc<tokio::sync::Mutex<tokio::net::unix::OwnedWriteHalf>>),
}

impl AetherWriter {
    pub async fn send(&self, frame: Frame) -> ProtoResult<()> {
        match self {
            AetherWriter::Channel(tx) => tx
                .send(frame)
                .await
                .map_err(|_| ProtoError::Unavailable("channel closed".into())),
            AetherWriter::Unix(w) => {
                let mut guard = w.lock().await;
                codec::write_frame(&mut *guard, &frame).await
            }
        }
    }
}

/// Reader half.
pub enum AetherReader {
    Channel(tokio::sync::mpsc::Receiver<Frame>),
    Unix(tokio::net::unix::OwnedReadHalf),
}

impl AetherReader {
    pub async fn recv(&mut self) -> ProtoResult<Option<Frame>> {
        match self {
            AetherReader::Channel(rx) => rx
                .recv()
                .await
                .map(Some)
                .ok_or_else(|| ProtoError::Unavailable("channel closed".into())),
            AetherReader::Unix(r) => codec::read_frame(r).await,
        }
    }
}

/// Splits a connection into (reader, writer).
pub fn split_conn(conn: AetherConn) -> (AetherReader, AetherWriter) {
    match conn {
        AetherConn::Channel { tx, rx } => {
            (AetherReader::Channel(rx), AetherWriter::Channel(tx.clone()))
        }
        AetherConn::Unix(stream) => {
            let (read, write) = stream.into_split();
            (
                AetherReader::Unix(read),
                AetherWriter::Unix(Arc::new(tokio::sync::Mutex::new(write))),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Server side (Edge)
// ---------------------------------------------------------------------------

/// Serves one connection: demultiplexes request frames to sandboxes and
/// pushes responses / stream chunks back.
pub async fn serve_connection(node: Arc<EdgeNode>, conn: AetherConn) {
    let (mut reader, writer) = split_conn(conn);
    loop {
        let frame = match reader.recv().await {
            Ok(Some(f)) => f,
            _ => break,
        };
        if !frame.is_request() {
            continue;
        }
        let Ok(req) = serde_json::from_slice::<Request>(&frame.payload) else {
            let err = Response::Error {
                code: dsec_protocol::message::ErrorCode::Internal,
                message: "malformed request payload".into(),
            };
            let reply = Frame::response(
                Channel::from_u16(frame.header.channel).unwrap_or(Channel::Control),
                frame.header.sid,
                frame.header.req_id,
                serde_json::to_vec(&err).unwrap_or_default(),
            );
            let _ = writer.send(reply).await;
            continue;
        };
        let node = node.clone();
        let writer = writer.clone();
        let sid = frame.header.sid;
        let req_id = frame.header.req_id;
        let channel = Channel::from_u16(frame.header.channel).unwrap_or(Channel::Control);
        // Pipelined: each request runs as its own task.
        tokio::spawn(async move {
            let frames = node.handle_aether_request(sid, channel, req, req_id).await;
            for f in frames {
                let _ = writer.send(f).await;
            }
        });
    }
}

/// Listens on a UDS path and serves every accepted connection.
pub async fn serve_uds(
    node: Arc<EdgeNode>,
    path: &str,
) -> ProtoResult<tokio::task::JoinHandle<()>> {
    let listener = tokio::net::UnixListener::bind(path).map_err(ProtoError::Io)?;
    let path = path.to_string();
    let handle = tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let node = node.clone();
            tokio::spawn(async move {
                serve_connection(node, AetherConn::Unix(Box::new(stream))).await;
            });
        }
        let _ = std::fs::remove_file(&path);
    });
    Ok(handle)
}

// ---------------------------------------------------------------------------
// Client side (SDK)
// ---------------------------------------------------------------------------

/// A stream event surfaced to the SDK.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    Chunk {
        stream_id: u32,
        stream: Stdio,
        data: Vec<u8>,
    },
    Fin {
        stream_id: u32,
        exit_code: i32,
    },
}

/// Multiplexing client for one node connection.
pub struct AetherClient {
    writer: AetherWriter,
    pending: Arc<Mutex<HashMap<u32, tokio::sync::oneshot::Sender<Response>>>>,
    /// Receivers staged by the reader loop when a stream is opened.
    /// Frames are processed in order, so the receiver is always staged
    /// before the first chunk frame arrives (removes the open/chunk race).
    ready: Arc<Mutex<HashMap<u32, tokio::sync::mpsc::Receiver<StreamEvent>>>>,
    next_req_id: Arc<AtomicU32>,
    call_timeout: Duration,
}

impl AetherClient {
    /// Wraps a client-side connection, spawning the demux reader task.
    pub fn new(conn: AetherConn) -> Self {
        let (reader, writer) = split_conn(conn);
        let pending: Arc<Mutex<HashMap<u32, tokio::sync::oneshot::Sender<Response>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let streams: Arc<Mutex<HashMap<u32, tokio::sync::mpsc::Sender<StreamEvent>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let ready: Arc<Mutex<HashMap<u32, tokio::sync::mpsc::Receiver<StreamEvent>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let reader_pending = pending.clone();
        let reader_streams = streams;
        let reader_ready = ready.clone();
        tokio::spawn(async move {
            reader_loop(reader, reader_pending, reader_streams, reader_ready).await;
        });
        AetherClient {
            writer,
            pending,
            ready,
            next_req_id: Arc::new(AtomicU32::new(1)),
            call_timeout: Duration::from_secs(30),
        }
    }

    /// Connects over a real UDS socket.
    pub async fn connect_uds(path: &str) -> ProtoResult<Self> {
        let stream = tokio::net::UnixStream::connect(path)
            .await
            .map_err(ProtoError::Io)?;
        Ok(Self::new(AetherConn::Unix(Box::new(stream))))
    }

    pub fn set_call_timeout(&mut self, d: Duration) {
        self.call_timeout = d;
    }

    fn alloc_req_id(&self) -> u32 {
        self.next_req_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Request-id allocator shared with stream handles.
    pub fn req_id_counter(&self) -> Arc<AtomicU32> {
        self.next_req_id.clone()
    }

    /// Request/response call, correlating by request id.
    pub async fn call(&self, sid: u64, channel: Channel, req: Request) -> ProtoResult<Response> {
        let req_id = self.alloc_req_id();
        let payload = serde_json::to_vec(&req)?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.pending
            .lock()
            .expect("pending poisoned")
            .insert(req_id, tx);
        let frame = Frame::request(channel, sid, req_id, payload);
        if let Err(e) = self.writer.send(frame).await {
            self.pending
                .lock()
                .expect("pending poisoned")
                .remove(&req_id);
            return Err(e);
        }
        match tokio::time::timeout(self.call_timeout, rx).await {
            Err(_) => {
                self.pending
                    .lock()
                    .expect("pending poisoned")
                    .remove(&req_id);
                Err(ProtoError::Timeout(format!(
                    "call timed out after {:?}",
                    self.call_timeout
                )))
            }
            Ok(Err(_)) => Err(ProtoError::Unavailable("connection closed".into())),
            Ok(Ok(resp)) => match resp {
                Response::Error { code, message } => Err(ProtoError::Protocol {
                    code: code as u32,
                    message,
                }),
                other => Ok(other),
            },
        }
    }

    /// Opens a streamed exec; events arrive via the returned handle.
    /// The reader loop stages the receiver before any chunk can arrive.
    pub async fn open_stream(&self, sid: u64, cmd: &str) -> ProtoResult<StreamHandle> {
        let resp = self
            .call(
                sid,
                Channel::Exec,
                Request::ExecStream {
                    cmd: cmd.to_string(),
                },
            )
            .await?;
        match resp {
            Response::StreamOpened { stream_id } => {
                let rx = self
                    .ready
                    .lock()
                    .expect("ready poisoned")
                    .remove(&stream_id)
                    .ok_or_else(|| ProtoError::Internal("stream receiver missing".into()))?;
                Ok(StreamHandle {
                    client_writer: self.writer.clone(),
                    next_req_id: self.next_req_id.clone(),
                    stream_id,
                    rx,
                })
            }
            other => Err(ProtoError::Protocol {
                code: 9,
                message: format!("expected StreamOpened, got {:?}", other),
            }),
        }
    }

    /// Number of in-flight unanswered requests (diagnostics).
    pub fn in_flight(&self) -> usize {
        self.pending.lock().expect("pending poisoned").len()
    }
}

async fn reader_loop(
    mut reader: AetherReader,
    pending: Arc<Mutex<HashMap<u32, tokio::sync::oneshot::Sender<Response>>>>,
    streams: Arc<Mutex<HashMap<u32, tokio::sync::mpsc::Sender<StreamEvent>>>>,
    ready: Arc<Mutex<HashMap<u32, tokio::sync::mpsc::Receiver<StreamEvent>>>>,
) {
    loop {
        let frame = match reader.recv().await {
            Ok(Some(f)) => f,
            _ => break,
        };
        if frame.is_stream() {
            if let Ok(resp) = serde_json::from_slice::<Response>(&frame.payload) {
                match resp {
                    Response::StreamChunk {
                        stream_id,
                        data,
                        stream,
                    } => {
                        if let Some(tx) = streams.lock().expect("streams poisoned").get(&stream_id)
                        {
                            let _ = tx.try_send(StreamEvent::Chunk {
                                stream_id,
                                stream,
                                data,
                            });
                        }
                    }
                    Response::StreamFin {
                        stream_id,
                        exit_code,
                    } => {
                        let mut map = streams.lock().expect("streams poisoned");
                        if let Some(tx) = map.remove(&stream_id) {
                            let _ = tx.try_send(StreamEvent::Fin {
                                stream_id,
                                exit_code,
                            });
                        }
                    }
                    _ => {}
                }
            }
        } else if frame.is_response() {
            if let Ok(resp) = serde_json::from_slice::<Response>(&frame.payload) {
                // Stage the stream receiver BEFORE completing the call, so
                // chunk frames (which follow) are never dropped.
                if let Response::StreamOpened { stream_id } = &resp {
                    let (tx, rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);
                    streams
                        .lock()
                        .expect("streams poisoned")
                        .insert(*stream_id, tx);
                    ready.lock().expect("ready poisoned").insert(*stream_id, rx);
                }
                if let Some(tx) = pending
                    .lock()
                    .expect("pending poisoned")
                    .remove(&frame.header.req_id)
                {
                    let _ = tx.send(resp);
                }
            }
        }
    }
    // Connection closed: fail every pending call.
    let mut map = pending.lock().expect("pending poisoned");
    map.clear();
}

/// Handle to one live output stream.
pub struct StreamHandle {
    client_writer: AetherWriter,
    next_req_id: Arc<AtomicU32>,
    stream_id: u32,
    rx: tokio::sync::mpsc::Receiver<StreamEvent>,
}

impl StreamHandle {
    pub async fn next(&mut self) -> Option<StreamEvent> {
        self.rx.recv().await
    }

    /// Writes to the stream's stdin.
    pub async fn write_stdin(&self, data: Vec<u8>) -> ProtoResult<()> {
        let req_id = self.next_req_id.fetch_add(1, Ordering::Relaxed);
        let req = Request::StreamWrite {
            stream_id: self.stream_id,
            data,
        };
        let frame = Frame::request(Channel::Stream, 0, req_id, serde_json::to_vec(&req)?);
        self.client_writer.send(frame).await
    }

    /// Closes the stream (EOF from the client side).
    pub async fn close(&self) -> ProtoResult<()> {
        let req_id = self.next_req_id.fetch_add(1, Ordering::Relaxed);
        let req = Request::StreamClose {
            stream_id: self.stream_id,
        };
        let frame = Frame::request(Channel::Stream, 0, req_id, serde_json::to_vec(&req)?);
        self.client_writer.send(frame).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::SandboxSpec;
    use dsec_storage::cache::LruBlockCache;
    use dsec_storage::erofs::{ErofsImageBuilder, ImageRegistry};
    use dsec_storage::latency::{LatencyModel, NodeLatencyProfile};
    use std::time::Duration;

    async fn test_node() -> Arc<EdgeNode> {
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
            4000,
            4096,
            100,
            0,
        );
        node.factory.prewarm("dsec/agent-base", 4);
        Arc::new(node)
    }

    #[tokio::test]
    async fn channel_roundtrip_exec() {
        let node = test_node().await;
        let entry = node.create(SandboxSpec::default()).await.unwrap();
        let (client_conn, server_conn) = channel_pair(64);
        tokio::spawn(serve_connection(node.clone(), server_conn));
        let client = AetherClient::new(client_conn);
        let resp = client
            .call(
                entry.sid,
                Channel::Exec,
                Request::Exec {
                    cmd: "cat /etc/hostname".into(),
                    timeout_ms: None,
                },
            )
            .await
            .unwrap();
        match resp {
            Response::Exec {
                exit_code, stdout, ..
            } => {
                assert_eq!(exit_code, 0);
                assert_eq!(stdout, b"sandbox-agent");
            }
            other => panic!("unexpected {:?}", other),
        }
    }

    #[tokio::test]
    async fn multiplexes_two_sandboxes() {
        let node = test_node().await;
        let a = node.create(SandboxSpec::default()).await.unwrap();
        let b = node.create(SandboxSpec::default()).await.unwrap();
        // Distinct working states.
        a.instance
            .chronus
            .fs()
            .write_file("/tmp/who", a.instance.hostname.as_bytes())
            .await
            .unwrap();
        b.instance
            .chronus
            .fs()
            .write_file("/tmp/who", b.instance.hostname.as_bytes())
            .await
            .unwrap();
        let (client_conn, server_conn) = channel_pair(256);
        tokio::spawn(serve_connection(node.clone(), server_conn));
        let client = Arc::new(AetherClient::new(client_conn));
        // Concurrent interleaved requests for both sandboxes.
        let mut tasks = Vec::new();
        for i in 0..20u32 {
            let sid = if i % 2 == 0 { a.sid } else { b.sid };
            let c = client.clone();
            tasks.push(tokio::spawn(async move {
                c.call(
                    sid,
                    Channel::Exec,
                    Request::Exec {
                        cmd: "cat /tmp/who".into(),
                        timeout_ms: None,
                    },
                )
                .await
                .unwrap()
            }));
        }
        for (i, t) in tasks.into_iter().enumerate() {
            match t.await.unwrap() {
                Response::Exec { stdout, .. } => {
                    let sid = if i % 2 == 0 { a.sid } else { b.sid };
                    let expect = if i % 2 == 0 { &a.instance } else { &b.instance };
                    assert_eq!(stdout, expect.hostname.as_bytes(), "sid {}", sid);
                }
                other => panic!("unexpected {:?}", other),
            }
        }
    }

    #[tokio::test]
    async fn stream_events_flow() {
        let node = test_node().await;
        let entry = node.create(SandboxSpec::default()).await.unwrap();
        let (client_conn, server_conn) = channel_pair(64);
        tokio::spawn(serve_connection(node.clone(), server_conn));
        let client = AetherClient::new(client_conn);
        let mut handle = client.open_stream(entry.sid, "seq 4").await.unwrap();
        let mut chunks = Vec::new();
        while let Some(ev) = handle.next().await {
            match ev {
                StreamEvent::Chunk { data, .. } => chunks.extend_from_slice(&data),
                StreamEvent::Fin { exit_code, .. } => {
                    assert_eq!(exit_code, 0);
                    break;
                }
            }
        }
        assert_eq!(chunks, b"1\n2\n3\n4\n");
    }

    #[tokio::test]
    async fn uds_transport_end_to_end() {
        let node = test_node().await;
        let entry = node.create(SandboxSpec::default()).await.unwrap();
        let dir = std::env::temp_dir().join(format!("dsec-uds-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("aether.sock");
        let _ = std::fs::remove_file(&path);
        let _listener = serve_uds(node.clone(), path.to_str().unwrap())
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        let client = AetherClient::connect_uds(path.to_str().unwrap())
            .await
            .unwrap();
        let resp = client
            .call(
                entry.sid,
                Channel::Fs,
                Request::FsRead {
                    path: "/etc/os-release".into(),
                },
            )
            .await
            .unwrap();
        match resp {
            Response::FileData { data } => {
                assert!(String::from_utf8(data).unwrap().contains("DSec Linux"))
            }
            other => panic!("unexpected {:?}", other),
        }
        // A second sandbox on the same socket.
        let entry2 = node.create(SandboxSpec::default()).await.unwrap();
        let resp2 = client
            .call(
                entry2.sid,
                Channel::Exec,
                Request::Exec {
                    cmd: "hostname".into(),
                    timeout_ms: None,
                },
            )
            .await
            .unwrap();
        assert!(matches!(resp2, Response::Exec { exit_code: 0, .. }));
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn error_response_maps_to_err() {
        let node = test_node().await;
        let (client_conn, server_conn) = channel_pair(64);
        tokio::spawn(serve_connection(node.clone(), server_conn));
        let client = AetherClient::new(client_conn);
        let err = client
            .call(
                999_999,
                Channel::Exec,
                Request::Exec {
                    cmd: "ls".into(),
                    timeout_ms: None,
                },
            )
            .await
            .unwrap_err();
        assert_eq!(
            err.code(),
            dsec_protocol::message::ErrorCode::NotFound as u32
        );
    }
}
