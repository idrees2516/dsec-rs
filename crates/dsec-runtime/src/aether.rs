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
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
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

    /// Sends many frames as one transmission.
    ///
    /// For the UDS transport this is the batching fast path: every frame
    /// is encoded into ONE buffer and issued with a single `write_all` +
    /// flush — N frames cost one syscall instead of N (the stream reader
    /// has always parsed frame-by-frame, so coalescing is wire-faithful).
    /// For the in-process channel transport it degenerates to N cheap
    /// buffered sends.
    pub async fn send_batch(&self, frames: Vec<Frame>) -> ProtoResult<()> {
        match self {
            AetherWriter::Channel(tx) => {
                for f in frames {
                    tx.send(f)
                        .await
                        .map_err(|_| ProtoError::Unavailable("channel closed".into()))?;
                }
                Ok(())
            }
            AetherWriter::Unix(w) => {
                let mut guard = w.lock().await;
                let cap: usize = frames
                    .iter()
                    .map(|f| dsec_protocol::frame::HEADER_LEN + f.payload.len())
                    .sum();
                let mut buf: Vec<u8> = Vec::with_capacity(cap);
                for f in &frames {
                    buf.extend_from_slice(&f.encode());
                }
                use tokio::io::AsyncWriteExt;
                guard.write_all(&buf).await?;
                guard.flush().await?;
                Ok(())
            }
        }
    }
}

/// Reader half.
///
/// [`recv_many`] is the batching fast path: it drains every frame that
/// is already deliverable in ONE wake, so a pipelined batch of N frames
/// costs one park/unpark cycle instead of N. For the UDS transport the
/// read side keeps a persistent buffer, so one `read` syscall can
/// yield many frames (the write side coalesces symmetrically in
/// [`AetherWriter::send_batch`]).
pub enum AetherReader {
    Channel(tokio::sync::mpsc::Receiver<Frame>),
    Unix {
        stream: tokio::net::unix::OwnedReadHalf,
        buf: Vec<u8>,
    },
}

impl AetherReader {
    pub async fn recv(&mut self) -> ProtoResult<Option<Frame>> {
        let mut one = Vec::with_capacity(1);
        let n = self.recv_many(&mut one, 1).await?;
        Ok(if n == 0 { None } else { one.pop() })
    }

    /// Receives up to `limit` frames, appending them to `out`; returns
    /// the number appended (`0` only when the peer closed and nothing is
    /// buffered). Frames keep wire order.
    pub async fn recv_many(&mut self, out: &mut Vec<Frame>, limit: usize) -> ProtoResult<usize> {
        match self {
            AetherReader::Channel(rx) => {
                Ok(rx.recv_many(out, limit).await) // 0 => closed and empty
            }
            AetherReader::Unix { stream, buf } => loop {
                // Parse every complete frame already in the buffer.
                let mut parsed = 0usize;
                while out.len() < limit && buf.len() >= dsec_protocol::frame::HEADER_LEN {
                    const HL: usize = dsec_protocol::frame::HEADER_LEN;
                    let payload_len =
                        u32::from_be_bytes([buf[17], buf[18], buf[19], buf[20]]) as usize;
                    if payload_len > dsec_protocol::MAX_PAYLOAD {
                        return Err(ProtoError::Protocol {
                            code: 9,
                            message: format!(
                                "payload {} exceeds limit {}",
                                payload_len,
                                dsec_protocol::MAX_PAYLOAD
                            ),
                        });
                    }
                    if buf.len() < HL + payload_len {
                        break; // incomplete frame: read more
                    }
                    let header: [u8; HL] = buf[..HL].try_into().unwrap();
                    let payload = buf[HL..HL + payload_len].to_vec();
                    let frame = Frame::decode_parts(&header, &payload)?;
                    buf.drain(..HL + payload_len);
                    out.push(frame);
                    parsed += 1;
                }
                if parsed > 0 {
                    return Ok(parsed);
                }
                // Buffer empty and no complete frame: pull more bytes.
                use tokio::io::AsyncReadExt;
                let mut chunk = [0u8; 8192];
                let n = stream.read(&mut chunk).await?;
                if n == 0 {
                    return Ok(0); // clean EOF (nothing buffered)
                }
                buf.extend_from_slice(&chunk[..n]);
            },
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
                AetherReader::Unix {
                    stream: read,
                    buf: Vec::with_capacity(8192),
                },
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
///
/// The reader drains every ready frame in one wake (`recv_many`), so a
/// pipelined batch of N requests costs one park/unpark cycle on the
/// receive side; each request still runs as its own task (pipelined
/// execution), preserving per-request concurrency semantics.
pub async fn serve_connection(node: Arc<EdgeNode>, conn: AetherConn) {
    let (mut reader, writer) = split_conn(conn);
    let mut frames: Vec<Frame> = Vec::with_capacity(64);
    loop {
        let n = match reader.recv_many(&mut frames, 64).await {
            Ok(n) => n,
            Err(_) => break,
        };
        if n == 0 {
            break;
        }
        for frame in frames.drain(..n) {
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

/// Aggregate call statistics (Relaxed-order counters; diagnostics for
/// locating whether a workload is data-plane or CPU bound).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CallStats {
    /// Completed `call` invocations.
    pub calls: u64,
    /// Sum of wall time spent inside `call` (nanoseconds).
    pub call_ns: u64,
    /// Calls that hit the deadline.
    pub timeouts: u64,
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
    counters: CallCounters,
}

#[derive(Default)]
struct CallCounters {
    calls: AtomicU64,
    call_ns: AtomicU64,
    timeouts: AtomicU64,
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
            counters: CallCounters::default(),
        }
    }

    /// Snapshot of aggregate call statistics.
    pub fn call_stats(&self) -> CallStats {
        CallStats {
            calls: self.counters.calls.load(Ordering::Relaxed),
            call_ns: self.counters.call_ns.load(Ordering::Relaxed),
            timeouts: self.counters.timeouts.load(Ordering::Relaxed),
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
    ///
    /// Timed: the aggregate counters power the I/O-boundness diagnostics
    /// (`call_stats`) at two relaxed atomic stores per call.
    pub async fn call(&self, sid: u64, channel: Channel, req: Request) -> ProtoResult<Response> {
        let t0 = std::time::Instant::now();
        let out = self.call_inner(sid, channel, req).await;
        self.counters.calls.fetch_add(1, Ordering::Relaxed);
        self.counters
            .call_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if matches!(&out, Err(ProtoError::Timeout(_))) {
            self.counters.timeouts.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    async fn call_inner(&self, sid: u64, channel: Channel, req: Request) -> ProtoResult<Response> {
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
            Ok(Ok(resp)) => map_response(resp),
        }
    }

    /// Pipelined request/response batch — the throughput fast path.
    ///
    /// All requests are serialized and their reply slots registered under
    /// ONE lock pass, every frame is sent as ONE transmission
    /// ([`AetherWriter::send_batch`]), and the replies are awaited under
    /// ONE deadline. A batch therefore costs one wakeup chain and one
    /// timer instead of N — the wire behavior (frames, correlation,
    /// per-request demux) is identical to N individual `call`s, which is
    /// exactly what Aether's multiplexing was designed for.
    ///
    /// Result order matches the input order. If the whole batch cannot be
    /// sent, every slot fails with the transport error; if the batch
    /// deadline passes, the remaining slots fail with `Timeout` and their
    /// pending entries are dropped (already-answered ones were removed by
    /// the reader loop; double-removal is a no-op).
    ///
    /// Counters: `call_stats` records `calls += len` and the batch's wall
    /// time once (per-request latency for a batch is therefore reported
    /// amortized).
    pub async fn call_batch(
        &self,
        calls: Vec<(u64, Channel, Request)>,
    ) -> Vec<ProtoResult<Response>> {
        let t0 = std::time::Instant::now();
        let n = calls.len();
        let out = self.call_batch_inner(calls).await;
        self.counters.calls.fetch_add(n as u64, Ordering::Relaxed);
        self.counters
            .call_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if out.iter().any(|r| matches!(r, Err(ProtoError::Timeout(_)))) {
            self.counters.timeouts.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    async fn call_batch_inner(
        &self,
        calls: Vec<(u64, Channel, Request)>,
    ) -> Vec<ProtoResult<Response>> {
        enum Slot {
            Ready {
                rx: tokio::sync::oneshot::Receiver<Response>,
            },
            Failed(ProtoError),
        }
        let mut slots: Vec<Slot> = Vec::with_capacity(calls.len());
        let mut frames: Vec<Frame> = Vec::with_capacity(calls.len());
        let mut req_ids: Vec<u32> = Vec::with_capacity(calls.len());
        {
            let mut pending = self.pending.lock().expect("pending poisoned");
            for (sid, channel, req) in calls {
                let req_id = self.alloc_req_id();
                match serde_json::to_vec(&req) {
                    Ok(payload) => {
                        let (tx, rx) = tokio::sync::oneshot::channel();
                        pending.insert(req_id, tx);
                        req_ids.push(req_id);
                        frames.push(Frame::request(channel, sid, req_id, payload));
                        slots.push(Slot::Ready { rx });
                    }
                    Err(e) => slots.push(Slot::Failed(e.into())),
                }
            }
        }
        if let Err(e) = self.writer.send_batch(frames).await {
            let mut pending = self.pending.lock().expect("pending poisoned");
            for id in req_ids {
                pending.remove(&id);
            }
            // The transport error applies to the whole batch; per-slot it
            // is reported as an `Unavailable` transport failure (the
            // original error is not `Clone`).
            let slot_msg = format!("batch send failed: {e}");
            return slots
                .into_iter()
                .map(|s| match s {
                    Slot::Ready { .. } => Err(ProtoError::Unavailable(slot_msg.clone())),
                    Slot::Failed(e) => Err(e),
                })
                .collect();
        }
        // One deadline for the whole in-flight batch.
        let deadline = tokio::time::Instant::now() + self.call_timeout;
        let mut timed_out = false;
        let mut out: Vec<ProtoResult<Response>> = Vec::with_capacity(slots.len());
        for slot in slots {
            match slot {
                Slot::Failed(e) => out.push(Err(e)),
                Slot::Ready { rx } => {
                    if timed_out {
                        out.push(Err(ProtoError::Timeout(format!(
                            "batch deadline ({:?}) exceeded",
                            self.call_timeout
                        ))));
                        continue;
                    }
                    match tokio::time::timeout_at(deadline, rx).await {
                        Err(_) => timed_out = true,
                        Ok(Err(_)) => {
                            out.push(Err(ProtoError::Unavailable("connection closed".into())));
                            continue;
                        }
                        Ok(Ok(resp)) => {
                            out.push(map_response(resp));
                            continue;
                        }
                    }
                    out.push(Err(ProtoError::Timeout(format!(
                        "batch deadline ({:?}) exceeded",
                        self.call_timeout
                    ))));
                }
            }
        }
        if timed_out {
            // Clear leftovers so a late reply cannot hit a dead slot.
            let mut pending = self.pending.lock().expect("pending poisoned");
            for id in req_ids {
                pending.remove(&id);
            }
        }
        out
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

/// Uniform response mapping: protocol-level errors become `Err`, every
/// other response is `Ok` (shared by `call` and `call_batch`).
fn map_response(resp: Response) -> ProtoResult<Response> {
    match resp {
        Response::Error { code, message } => Err(ProtoError::Protocol {
            code: code as u32,
            message,
        }),
        other => Ok(other),
    }
}

async fn reader_loop(
    mut reader: AetherReader,
    pending: Arc<Mutex<HashMap<u32, tokio::sync::oneshot::Sender<Response>>>>,
    streams: Arc<Mutex<HashMap<u32, tokio::sync::mpsc::Sender<StreamEvent>>>>,
    ready: Arc<Mutex<HashMap<u32, tokio::sync::mpsc::Receiver<StreamEvent>>>>,
) {
    let mut frames: Vec<Frame> = Vec::with_capacity(64);
    loop {
        let n = match reader.recv_many(&mut frames, 64).await {
            Ok(n) => n,
            Err(_) => break,
        };
        if n == 0 {
            break;
        }
        // Completed calls accumulated across the drained frames; the
        // pending map is taken once for the whole batch.
        let mut completed: Vec<(u32, Response)> = Vec::with_capacity(n);
        for frame in frames.drain(..n) {
            if frame.is_stream() {
                if let Ok(resp) = serde_json::from_slice::<Response>(&frame.payload) {
                    match resp {
                        Response::StreamChunk {
                            stream_id,
                            data,
                            stream,
                        } => {
                            if let Some(tx) =
                                streams.lock().expect("streams poisoned").get(&stream_id)
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
                    completed.push((frame.header.req_id, resp));
                }
            }
        }
        if !completed.is_empty() {
            let mut map = pending.lock().expect("pending poisoned");
            for (req_id, resp) in completed {
                if let Some(tx) = map.remove(&req_id) {
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

    #[tokio::test]
    async fn call_batch_matches_individual_calls() {
        let node = test_node().await;
        let a = node.create(SandboxSpec::default()).await.unwrap();
        let b = node.create(SandboxSpec::default()).await.unwrap();
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
        let client = AetherClient::new(client_conn);

        let mk = |sid| {
            (
                sid,
                Channel::Exec,
                Request::Exec {
                    cmd: "cat /tmp/who".into(),
                    timeout_ms: None,
                },
            )
        };
        let calls: Vec<(u64, Channel, Request)> = (0..24)
            .map(|i| mk(if i % 2 == 0 { a.sid } else { b.sid }))
            .collect();
        let batch: Vec<_> = client.call_batch(calls).await;
        assert_eq!(batch.len(), 24);
        for (i, r) in batch.into_iter().enumerate() {
            let resp = r.unwrap();
            match resp {
                Response::Exec { stdout, .. } => {
                    let expect = if i % 2 == 0 { &a.instance } else { &b.instance };
                    assert_eq!(stdout, expect.hostname.as_bytes(), "slot {}", i);
                }
                other => panic!("unexpected {:?}", other),
            }
        }
        // Batch stats: calls counted, nothing pending left behind.
        let stats = client.call_stats();
        assert_eq!(stats.calls, 24);
        assert_eq!(stats.timeouts, 0);
        assert_eq!(client.in_flight(), 0);
    }

    #[tokio::test]
    async fn call_batch_propagates_per_slot_errors() {
        let node = test_node().await;
        let good = node.create(SandboxSpec::default()).await.unwrap();
        let (client_conn, server_conn) = channel_pair(64);
        tokio::spawn(serve_connection(node.clone(), server_conn));
        let client = AetherClient::new(client_conn);
        let calls = vec![
            (
                good.sid,
                Channel::Exec,
                Request::Exec {
                    cmd: "hostname".into(),
                    timeout_ms: None,
                },
            ),
            (
                999_999,
                Channel::Exec,
                Request::Exec {
                    cmd: "ls".into(),
                    timeout_ms: None,
                },
            ),
        ];
        let out = client.call_batch(calls).await;
        assert!(out[0].is_ok());
        assert_eq!(
            out[1].as_ref().unwrap_err().code(),
            dsec_protocol::message::ErrorCode::NotFound as u32
        );
        assert_eq!(client.in_flight(), 0);
    }

    #[tokio::test]
    async fn call_batch_over_uds_coalesces_to_one_write() {
        // Wire-level: a batch over the UDS transport must be decodable
        // frame-by-frame by the stock reader (coalescing is transparent).
        let node = test_node().await;
        let entry = node.create(SandboxSpec::default()).await.unwrap();
        let dir = std::env::temp_dir().join(format!("dsec-uds-batch-{}", std::process::id()));
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
        let calls: Vec<(u64, Channel, Request)> = (0..16)
            .map(|i| {
                (
                    entry.sid,
                    Channel::Fs,
                    Request::FsRead {
                        // Even slots read an existing file, odd ones a
                        // missing path — per-slot success/error alignment.
                        path: if i % 2 == 0 {
                            "/etc/hostname".into()
                        } else {
                            "/etc/hostname-missing".into()
                        },
                    },
                )
            })
            .collect();
        let out = client.call_batch(calls).await;
        assert_eq!(out.len(), 16);
        // Half exist, half do not — errors must align per slot.
        for (i, r) in out.into_iter().enumerate() {
            if i % 2 == 0 {
                assert!(matches!(r.unwrap(), Response::FileData { .. }), "slot {i}");
            } else {
                assert!(r.is_err(), "slot {i}");
            }
        }
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[tokio::test]
    async fn call_batch_deadline_times_out_and_clears_pending() {
        let (client_conn, server_conn) = channel_pair(8);
        // Server that swallows requests without replying.
        tokio::spawn(async move {
            let (mut reader, _writer) = split_conn(server_conn);
            while let Ok(Some(_f)) = reader.recv().await {}
        });
        let mut client = AetherClient::new(client_conn);
        client.set_call_timeout(Duration::from_millis(30));
        let calls: Vec<(u64, Channel, Request)> = (0..4)
            .map(|_| (1, Channel::Control, Request::Ping))
            .collect();
        let t0 = std::time::Instant::now();
        let out = client.call_batch(calls).await;
        assert!(t0.elapsed() >= Duration::from_millis(25));
        assert!(out.iter().all(|r| matches!(r, Err(ProtoError::Timeout(_)))));
        assert_eq!(client.in_flight(), 0);
        assert_eq!(client.call_stats().timeouts, 1);
    }
}
