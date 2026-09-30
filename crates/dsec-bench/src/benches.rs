//! Benchmark implementations.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dsec_control::model::SandboxSpec as ControlSpec;
use dsec_protocol::frame::{Channel, Frame};
use dsec_protocol::message::Request;
use dsec_protocol::rng::Rng;
use dsec_rl::fastenv::FastCounterEnv;
use dsec_rl::spaces::Action;
use dsec_rl::{Driver, EnvPool, ReplayBuffer};
use dsec_sdk::integration::{spawn_local_cluster, ClusterConfig};

use crate::report::{percentiles, BenchResult, Percentiles};

fn now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as f64)
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// 1. Sandbox creation rate
// ---------------------------------------------------------------------------

/// Creates `count` sandboxes with bounded concurrency; returns
/// (creations_per_sec, per-creation latency percentiles).
pub async fn creation_rate(
    count: usize,
    concurrency: usize,
    paper_latency: bool,
) -> (f64, Percentiles) {
    let latencies = if paper_latency {
        dsec_storage::latency::NodeLatencyProfile::paper(7)
    } else {
        dsec_storage::latency::NodeLatencyProfile::zero()
    };
    let cluster = spawn_local_cluster(ClusterConfig {
        latencies,
        prewarm: 64,
        ..Default::default()
    })
    .await;
    let token = cluster
        .plane
        .create_token("root", dsec_control::iam::Role::Writer)
        .unwrap()
        .token;
    let plane = cluster.plane.clone();
    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));

    let t0 = Instant::now();
    let mut tasks = Vec::with_capacity(count);
    for _ in 0..count {
        let permit = sem.clone().acquire_owned().await.unwrap();
        let plane = plane.clone();
        let token = token.clone();
        tasks.push(tokio::spawn(async move {
            let start = now_ms();
            let spec = ControlSpec::default();
            let _ = plane.create_sandbox(&token, spec).await;
            let end = now_ms();
            drop(permit);
            end - start
        }));
    }
    // Per-task latency returned via the join handle — no shared Mutex on
    // the creation hot path.
    let lat: Vec<f64> = futures_join_all(tasks).await;
    let elapsed = t0.elapsed().as_secs_f64();
    let created = cluster.plane.registry.sandbox_count() as f64;
    (created / elapsed.max(1e-9), percentiles(&lat))
}

/// Await a batch of tasks, keeping successful values and skipping
/// panicked tasks (mirrors the old `let _ = t.await` tolerance).
async fn futures_join_all<T: Send + 'static>(tasks: Vec<tokio::task::JoinHandle<T>>) -> Vec<T> {
    let mut out = Vec::with_capacity(tasks.len());
    for t in tasks {
        if let Ok(v) = t.await {
            out.push(v);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 2. Burst: full lifecycle at scale
// ---------------------------------------------------------------------------

pub struct BurstStats {
    pub total: usize,
    pub seconds: f64,
    pub peak_concurrent: u64,
    pub execs: u64,
    pub errors: u64,
}

/// Create `count` sandboxes (create + exec + destroy), reporting totals.
pub async fn burst_lifecycle(count: usize, concurrency: usize) -> BurstStats {
    let cluster = spawn_local_cluster(ClusterConfig {
        cpu_millicores: 10_000_000,
        mem_mib: 10_000_000,
        max_sandboxes: count as i64 + 100,
        prewarm: 128,
        ..Default::default()
    })
    .await;
    let token = cluster
        .plane
        .create_token("root", dsec_control::iam::Role::Writer)
        .unwrap()
        .token;
    let plane = Arc::new(cluster.plane.clone());
    let nodes = cluster.nodes.clone();
    let errors = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let execs = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));

    let t0 = Instant::now();
    let peak = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut tasks = Vec::new();
    for _ in 0..count {
        let permit = sem.clone().acquire_owned().await.unwrap();
        let plane = plane.clone();
        let token = token.clone();
        let errors = errors.clone();
        let execs = execs.clone();
        let peak = peak.clone();
        let nodes = nodes.clone();
        tasks.push(tokio::spawn(async move {
            let spec = ControlSpec::default();
            match plane.create_sandbox(&token, spec).await {
                Ok(rec) => {
                    let cur = plane.registry.sandbox_count() as u64;
                    peak.fetch_max(cur, std::sync::atomic::Ordering::Relaxed);
                    // One real exec through the node's data plane.
                    let node = &nodes[0];
                    let frames = node
                        .handle_aether_request(
                            rec.sid,
                            Channel::Exec,
                            Request::Exec {
                                cmd: "echo burst".into(),
                                timeout_ms: None,
                            },
                            1,
                        )
                        .await;
                    if frames.len() == 1 {
                        execs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    let _ = plane.destroy_sandbox(&token, rec.sid).await;
                }
                Err(_) => {
                    errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            drop(permit);
        }));
    }
    for t in tasks {
        let _ = t.await;
    }
    let seconds = t0.elapsed().as_secs_f64();
    BurstStats {
        total: count,
        seconds,
        peak_concurrent: peak.load(std::sync::atomic::Ordering::Relaxed),
        execs: execs.load(std::sync::atomic::Ordering::Relaxed),
        errors: errors.load(std::sync::atomic::Ordering::Relaxed),
    }
}

// ---------------------------------------------------------------------------
// 3. Pause / resume latency (paper profile)
// ---------------------------------------------------------------------------

pub async fn pause_resume(cycles: usize) -> (Percentiles, Percentiles) {
    let cluster = spawn_local_cluster(ClusterConfig {
        latencies: dsec_storage::latency::NodeLatencyProfile::paper(11),
        ..Default::default()
    })
    .await;
    let token = cluster
        .plane
        .create_token("root", dsec_control::iam::Role::Writer)
        .unwrap()
        .token;
    let plane = cluster.plane.clone();
    let spec = ControlSpec::default();
    let rec = plane.create_sandbox(&token, spec).await.unwrap();
    let mut pause_lat = Vec::new();
    let mut resume_lat = Vec::new();
    for _ in 0..cycles {
        let t0 = Instant::now();
        plane.pause_sandbox(&token, rec.sid).await.unwrap();
        pause_lat.push(t0.elapsed().as_secs_f64() * 1000.0);
        let t1 = Instant::now();
        plane.resume_sandbox(&token, rec.sid).await.unwrap();
        resume_lat.push(t1.elapsed().as_secs_f64() * 1000.0);
    }
    let _ = plane.destroy_sandbox(&token, rec.sid).await;
    (percentiles(&pause_lat), percentiles(&resume_lat))
}

// ---------------------------------------------------------------------------
// 4. RL throughput
// ---------------------------------------------------------------------------

/// Pure-compute env stepping (pufferlib's speed floor analogue).
pub fn rl_fast_steps(num_envs: usize, workers: usize, steps: u64) -> f64 {
    let envs: Vec<Box<dyn dsec_rl::Env>> = (0..num_envs)
        .map(|i| Box::new(FastCounterEnv::new(i as u64 + 1, 64)) as Box<dyn dsec_rl::Env>)
        .collect();
    let mut pool = EnvPool::new(envs).with_workers(workers);
    pool.reset_all();
    let t0 = Instant::now();
    for _ in 0..steps {
        let actions: Vec<Action> = (0..num_envs)
            .map(|i| Action::Discrete((i % 8) as i64))
            .collect();
        let _ = pool.step_parallel(&actions).unwrap();
    }
    let elapsed = t0.elapsed().as_secs_f64();
    (steps as f64 * num_envs as f64) / elapsed.max(1e-9)
}

/// Buffer enqueue throughput.
pub fn rl_buffer_enqueue(num_envs: usize, steps: u64, obs_size: usize, hidden: usize) -> f64 {
    // Ring capacity mirrors real rollout horizons (memory-bounded).
    let mut buf = ReplayBuffer::new(
        8192,
        num_envs,
        dsec_rl::ObsSpace::Flat { size: obs_size },
        dsec_rl::ActionSpace::discrete(8),
        hidden,
    );
    let obs = vec![0.5f32; num_envs * obs_size];
    let actions: Vec<Action> = (0..num_envs).map(|_| Action::Discrete(3)).collect();
    let rewards = vec![0.0f32; num_envs];
    let dones = vec![false; num_envs];
    let lps = vec![0.0f32; num_envs];
    let vals = vec![0.0f32; num_envs];
    let h = vec![0.1f32; num_envs * hidden];
    let c = vec![0.0f32; num_envs * hidden];
    let t0 = Instant::now();
    for _ in 0..steps {
        buf.enqueue(
            &obs, &actions, &rewards, &dones, &lps, &vals, &h, &c, &h, &c,
        )
        .unwrap();
    }
    let elapsed = t0.elapsed().as_secs_f64();
    (steps as f64 * num_envs as f64) / elapsed.max(1e-9)
}

/// GAE over a full buffer.
pub fn rl_gae(num_envs: usize, steps: usize) -> f64 {
    let mut buf = ReplayBuffer::new(
        steps.min(8192),
        num_envs,
        dsec_rl::ObsSpace::Flat { size: 16 },
        dsec_rl::ActionSpace::discrete(8),
        0,
    );
    let obs = vec![0.5f32; num_envs * 16];
    let actions: Vec<Action> = (0..num_envs).map(|_| Action::Discrete(3)).collect();
    let rewards = vec![0.0f32; num_envs];
    let dones = vec![false; num_envs];
    let lps = vec![0.0f32; num_envs];
    let vals = vec![0.0f32; num_envs];
    for _ in 0..steps {
        buf.enqueue(
            &obs,
            &actions,
            &rewards,
            &dones,
            &lps,
            &vals,
            &[],
            &[],
            &[],
            &[],
        )
        .unwrap();
    }
    let t0 = Instant::now();
    buf.compute_gae(0.99, 0.95);
    let elapsed = t0.elapsed().as_secs_f64();
    (steps as f64 * num_envs as f64) / elapsed.max(1e-9)
}

/// Full driver rollout on fast envs (steps/sec incl. hidden bookkeeping).
pub fn rl_driver(num_envs: usize, workers: usize, steps: u64, hidden: usize) -> f64 {
    let envs: Vec<Box<dyn dsec_rl::Env>> = (0..num_envs)
        .map(|i| Box::new(FastCounterEnv::new(i as u64 + 1, 128)) as Box<dyn dsec_rl::Env>)
        .collect();
    let pool = EnvPool::new(envs).with_workers(workers);
    let buffer = ReplayBuffer::new(
        8192,
        num_envs,
        dsec_rl::ObsSpace::Flat { size: 8 },
        dsec_rl::ActionSpace::discrete(8),
        hidden,
    );
    let policy = Arc::new(|obs: &[f32], _h: &mut [f32], _c: &mut [f32]| {
        // Hash policy: deterministic pseudo-argmax over the obs.
        obs.iter()
            .step_by(8)
            .map(|o| Action::Discrete(((*o * 97.0) as i64).rem_euclid(8)))
            .collect::<Vec<Action>>()
    });
    let mut driver = Driver::new(pool, buffer, policy, Default::default(), 7);
    let stats = driver.rollout(steps).unwrap();
    driver.finish_phase();
    stats.steps_per_sec
}

/// Sandbox-backed env stepping (real Chronus data plane per step).
pub fn rl_sandbox_steps(num_envs: usize, steps: u64) -> f64 {
    use dsec_rl::sandbox_env::SandboxEnvBuilder;
    let builder = SandboxEnvBuilder::new(13);
    let envs = builder.build_envs(num_envs, 13).unwrap();
    let mut pool = EnvPool::new(envs).with_workers(4);
    pool.reset_all();
    let t0 = Instant::now();
    for _ in 0..steps {
        let actions: Vec<Action> = (0..num_envs)
            .map(|i| Action::Discrete((i % 6) as i64))
            .collect();
        let _ = pool.step_parallel(&actions).unwrap();
    }
    let elapsed = t0.elapsed().as_secs_f64();
    (steps as f64 * num_envs as f64) / elapsed.max(1e-9)
}

// ---------------------------------------------------------------------------
// 5. pack_diff
// ---------------------------------------------------------------------------

pub struct PackdiffStats {
    pub full_bytes: u64,
    pub pack_bytes: u64,
    pub ratio: f64,
    pub packs_per_sec: f64,
    pub applies_per_sec: f64,
}

pub fn packdiff(files: usize, mutate_frac: f64, iterations: usize) -> PackdiffStats {
    use dsec_storage::erofs::{ErofsImageBuilder, OnDemandLoader};
    use dsec_storage::imagefs::LayeredImage;
    use dsec_storage::overlay::OverlayDev;

    let mut builder = ErofsImageBuilder::new("bench/big");
    for i in 0..files {
        builder = builder.add_file(
            format!("/data/f{:04}.bin", i),
            vec![(i % 251) as u8; 512],
            0o644,
        );
    }
    let image = Arc::new(builder.build());
    let cache = Arc::new(dsec_storage::cache::LruBlockCache::new(4096));
    let loader = Arc::new(OnDemandLoader::new(
        image,
        cache,
        dsec_storage::latency::LatencyModel::fixed(Duration::ZERO),
    ));
    let overlay = Arc::new(OverlayDev::new(loader.clone()));
    let fs = Arc::new(LayeredImage::new(overlay.clone()));
    // Mutate a fraction of files at the block level (sync path; same-size
    // writes keep the file table stable).
    let mutate = (files as f64 * mutate_frac) as usize;
    {
        let meta = loader.image().meta().clone();
        for i in 0..mutate {
            let path = format!("/data/f{:04}.bin", i);
            if let Some(e) = meta.entries.iter().find(|e| e.path == path) {
                let mut block = [0u8; dsec_storage::BLOCK_SIZE];
                let payload = b"mutated payload here";
                block[..payload.len()].copy_from_slice(payload);
                overlay.write_block(e.first_block, block);
            }
        }
    }
    let full = overlay.base_image().full_size();
    let pack = fs.snapshot_diff(now_ms() as u64);
    let pack_bytes = pack.blocks.len() as u64 * dsec_storage::BLOCK_SIZE as u64;
    let ratio = pack_bytes as f64 / full.max(1) as f64;

    // Throughput: repeated snapshot+apply on fresh replicas.
    let t0 = Instant::now();
    let mut count = 0usize;
    for _ in 0..iterations {
        let _ = fs.snapshot_diff(now_ms() as u64);
        count += 1;
    }
    // After the first snapshot the dirty set is empty (nothing new), so
    // throughput on empty packs is the management-path cost. Report that.
    let packs_per_sec = count as f64 / t0.elapsed().as_secs_f64().max(1e-9);

    let t1 = Instant::now();
    for _ in 0..iterations {
        // Replica construction seeds the table from the pack's shared
        // Arc<str> keys (refcount inserts) instead of re-interning the
        // base image's strings and immediately replacing them.
        let replica_overlay = Arc::new(OverlayDev::new(loader.clone()));
        let replica_fs = Arc::new(LayeredImage::replica_from(replica_overlay, &pack).unwrap());
        let _ = replica_fs.overlay().apply_diff(&pack);
    }
    let applies_per_sec = iterations as f64 / t1.elapsed().as_secs_f64().max(1e-9);

    PackdiffStats {
        full_bytes: full,
        pack_bytes,
        ratio,
        packs_per_sec,
        applies_per_sec,
    }
}

// ---------------------------------------------------------------------------
// 6. Placement throughput + k-choice quality
// ---------------------------------------------------------------------------

pub struct PlacementStats {
    pub decisions_per_sec: f64,
    pub k1_avg_score: f64,
    pub k8_avg_score: f64,
    pub exhaustive_avg_score: f64,
    pub nodes: usize,
    pub k: usize,
}

pub fn placement_bench(nodes: usize, decisions: usize, seed: u64) -> PlacementStats {
    use dsec_control::model::{Locality, NodeInfo, NodeStatus};
    use dsec_control::placement::{PlacementEngine, ScoreWeights};
    use std::collections::HashMap;
    let mut node_list: Vec<NodeInfo> = (0..nodes)
        .map(|i| NodeInfo {
            node_id: format!("n{}", i),
            cpu_millicores: 8000,
            mem_mib: 8192,
            max_sandboxes: 200,
            cpu_available: 1000 + (i as i64 % 800),
            mem_available: 1024 + (i as i64 % 7000),
            slots_available: 100,
            status: NodeStatus::Healthy,
            locality: Locality::Local,
            images: vec!["dsec/agent-base".into()],
            labels: HashMap::new(),
            admitted_projects: vec!["root".into()],
            cost_multiplier: 1.0,
        })
        .collect();
    node_list.sort_by(|a, b| a.node_id.cmp(&b.node_id));

    let spec = ControlSpec::default();
    let engine = PlacementEngine::new(8, seed, ScoreWeights::default());
    let t0 = Instant::now();
    for _ in 0..decisions {
        let _ = engine.place(&spec, &node_list).unwrap();
    }
    let dps = decisions as f64 / t0.elapsed().as_secs_f64().max(1e-9);

    // Quality: average chosen-node score for k=1 vs k=8 vs exhaustive.
    let w = ScoreWeights::default();
    let score_of = |n: &NodeInfo| -> f64 {
        let (fit, frag, loc) = n.score_parts(&spec);
        w.packing * fit + w.fragmentation * frag + w.locality * loc
    };
    let mut rng = Rng::new(seed);
    let mut k1 = 0.0;
    let mut k8 = 0.0;
    let mut exhaustive = 0.0;
    let trials = 500;
    for _ in 0..trials {
        // k=1: random node.
        let i = rng.below(nodes);
        k1 += score_of(&node_list[i]);
        // k=8: best of 8 sampled.
        let sample = rng.sample_k(nodes, 8);
        k8 += sample
            .iter()
            .map(|&i| score_of(&node_list[i]))
            .fold(f64::MIN, f64::max);
        // exhaustive.
        exhaustive += node_list.iter().map(score_of).fold(f64::MIN, f64::max);
    }
    PlacementStats {
        decisions_per_sec: dps,
        k1_avg_score: k1 / trials as f64,
        k8_avg_score: k8 / trials as f64,
        exhaustive_avg_score: exhaustive / trials as f64,
        nodes,
        k: 8,
    }
}

// ---------------------------------------------------------------------------
// 7. HTTP RPS (apiserver)
// ---------------------------------------------------------------------------

pub async fn http_rps(requests: usize, concurrency: usize) -> f64 {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let cluster = spawn_local_cluster(ClusterConfig::default()).await;
    let addr = cluster.api_addr;
    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let t0 = Instant::now();
    let mut tasks = Vec::new();
    for _ in 0..requests {
        let permit = sem.clone().acquire_owned().await.unwrap();
        tasks.push(tokio::spawn(async move {
            let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
            let req = format!(
                "GET /healthz HTTP/1.1\r\nhost: {}\r\nconnection: close\r\n\r\n",
                addr
            );
            stream.write_all(req.as_bytes()).await.unwrap();
            let mut buf = Vec::new();
            let _ = stream.read_to_end(&mut buf).await;
            drop(permit);
        }));
    }
    for t in tasks {
        let _ = t.await;
    }
    requests as f64 / t0.elapsed().as_secs_f64().max(1e-9)
}

/// Keep-alive variant: each client pipelines many requests over ONE TCP
/// connection (the transport a real DsecClient pool uses). Responses are
/// framed by Content-Length, matching HTTP/1.1 persistent connections.
pub async fn http_rps_keepalive(clients: usize, per_client: usize) -> f64 {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let cluster = spawn_local_cluster(ClusterConfig::default()).await;
    let addr = cluster.api_addr;
    let total = clients * per_client;
    let t0 = Instant::now();
    let mut tasks = Vec::new();
    for _ in 0..clients {
        tasks.push(tokio::spawn(async move {
            let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
            stream.set_nodelay(true).ok();
            let req = format!(
                "GET /healthz HTTP/1.1\r\nhost: {}\r\nconnection: keep-alive\r\n\r\n",
                addr
            );
            // Buffered client-side framing: read chunks, carve out
            // complete responses (header + Content-Length body) from a
            // pending buffer. No per-byte reads.
            let mut pending: Vec<u8> = Vec::with_capacity(4096);
            let mut chunk = [0u8; 4096];
            'client: for _ in 0..per_client {
                stream.write_all(req.as_bytes()).await.unwrap();
                loop {
                    if let Some(consumed) = parse_http_response(&pending) {
                        pending.drain(..consumed);
                        break; // next request
                    }
                    let n = stream.read(&mut chunk).await.unwrap();
                    if n == 0 {
                        break 'client;
                    }
                    pending.extend_from_slice(&chunk[..n]);
                }
            }
        }));
    }
    for t in tasks {
        let _ = t.await;
    }
    let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
    total as f64 / elapsed
}

/// If `buf` holds one complete HTTP/1.1 response (headers + body),
/// returns the total byte length to consume.
fn parse_http_response(buf: &[u8]) -> Option<usize> {
    let hdr_end = buf.windows(4).position(|w| w == b"\r\n\r\n")? + 4;
    let head = std::str::from_utf8(&buf[..hdr_end]).ok()?;
    let body_len: usize = head
        .lines()
        .find_map(|l| {
            l.split_once(':')
                .filter(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                .and_then(|(_, v)| v.trim().parse().ok())
        })
        .unwrap_or(0);
    let total = hdr_end + body_len;
    (buf.len() >= total).then_some(total)
}

// ---------------------------------------------------------------------------
// 8. Protocol codec throughput
// ---------------------------------------------------------------------------

pub fn codec_throughput(frames: usize) -> (f64, f64) {
    let payload: Vec<u8> = br#"{"op":"exec","cmd":"echo benchmark","timeout_ms":1000}"#.to_vec();
    let frame = Frame::request(Channel::Exec, 42, 7, payload);
    let t0 = Instant::now();
    let mut encoded = 0usize;
    let mut wires = Vec::with_capacity(frames);
    for _ in 0..frames {
        let w = frame.encode();
        encoded += w.len();
        wires.push(w);
    }
    let encode_mbs = encoded as f64 / 1e6 / t0.elapsed().as_secs_f64().max(1e-9);

    let t1 = Instant::now();
    let mut decoded = 0usize;
    for w in &wires {
        let f = Frame::decode(w).unwrap();
        decoded += f.payload.len();
    }
    let decode_mbs = decoded as f64 / 1e6 / t1.elapsed().as_secs_f64().max(1e-9);
    (encode_mbs, decode_mbs)
}

// ---------------------------------------------------------------------------
// Result assembly
// ---------------------------------------------------------------------------

/// Async benchmarks (need the tokio runtime): creation, burst, pause, HTTP.
pub async fn run_async_set(quick: bool) -> Vec<BenchResult> {
    let mut out = Vec::new();
    let scale = if quick { 0.1 } else { 1.0 };

    // 1. Creation rate — pure software path (zero injected latency).
    let create_count = if quick { 500 } else { 10_000 };
    let (cps, lat) = creation_rate(create_count, 64, false).await;
    out.push(BenchResult {
        name: "sandbox_create_software_path".into(),
        value: cps,
        unit: "creates/sec".into(),
        reference: Some("DSec paper: ~5,000/s cluster-wide (incl. cold backends)".into()),
        notes: "single simulated node, FnCall warm pool, zero injected latency".into(),
        p50_ms: Some(lat.p50_ms),
        p99_ms: Some(lat.p99_ms),
        samples: Some(lat.samples),
    });

    // 1b. Creation rate with paper latency profile (fncall ~5ms path).
    let (cps2, lat2) = creation_rate(create_count, 256, true).await;
    out.push(BenchResult {
        name: "sandbox_create_paper_latency".into(),
        value: cps2,
        unit: "creates/sec".into(),
        reference: Some("DSec paper: ~5,000/s with ~1s avg cold creation".into()),
        notes: "FnCall ~5ms injected latency, 256 concurrent creations".into(),
        p50_ms: Some(lat2.p50_ms),
        p99_ms: Some(lat2.p99_ms),
        samples: Some(lat2.samples),
    });

    // 2. Burst: paper-scale 100k lifecycle in full mode.
    let burst_count = if quick { 2000 } else { 100_000 };
    let burst = burst_lifecycle(burst_count, 256).await;
    out.push(BenchResult {
        name: "burst_lifecycle".into(),
        value: burst.total as f64 / burst.seconds.max(1e-9),
        unit: "sandboxes/sec (create+exec+destroy)".into(),
        reference: Some("DSec paper: 100k+ concurrent, ~5k/s create rate".into()),
        notes: format!(
            "{} sandboxes, peak concurrent {}, {} execs, {} errors",
            burst.total, burst.peak_concurrent, burst.execs, burst.errors
        ),
        p50_ms: None,
        p99_ms: None,
        samples: None,
    });

    // 3. Pause / resume (paper latency profile).
    let (pause, resume) = pause_resume(if quick { 5 } else { 20 }).await;
    out.push(BenchResult {
        name: "pause_latency".into(),
        value: pause.p50_ms,
        unit: "ms (p50)".into(),
        reference: Some("DSec paper: ~4s average pause (kernel checkpoint+reclaim)".into()),
        notes: "userspace sim: state machine + reclaim accounting only".into(),
        p50_ms: Some(pause.p50_ms),
        p99_ms: Some(pause.p99_ms),
        samples: Some(pause.samples),
    });
    out.push(BenchResult {
        name: "resume_latency".into(),
        value: resume.p50_ms,
        unit: "ms (p50)".into(),
        reference: Some("DSec paper: supports preemption, resume ~1.5s class".into()),
        notes: "userspace sim".into(),
        p50_ms: Some(resume.p50_ms),
        p99_ms: Some(resume.p99_ms),
        samples: Some(resume.samples),
    });

    // 7. HTTP.
    let rps = http_rps((20_000.0 * scale) as usize, 16).await;
    out.push(BenchResult {
        name: "apiserver_rps".into(),
        value: rps,
        unit: "requests/sec".into(),
        reference: Some("stateless apiserver must sustain create-rate traffic".into()),
        notes: "GET /healthz, 16 concurrent keep-alive-free clients".into(),
        p50_ms: None,
        p99_ms: None,
        samples: None,
    });

    // 7b. HTTP with keep-alive pipelining (the DsecClient transport shape).
    let ka = http_rps_keepalive(16, (1_250.0 * scale) as usize).await;
    out.push(BenchResult {
        name: "apiserver_rps_keepalive".into(),
        value: ka,
        unit: "requests/sec".into(),
        reference: Some("stateless apiserver must sustain create-rate traffic".into()),
        notes: "GET /healthz, 16 keep-alive connections, pipelined".into(),
        p50_ms: None,
        p99_ms: None,
        samples: None,
    });
    out
}

/// Sync benchmarks (own thread, may build private runtimes): RL, pack_diff,
/// placement, codec.
pub fn run_sync_set(quick: bool) -> Vec<BenchResult> {
    let mut out = Vec::new();
    let scale = if quick { 0.1 } else { 1.0 };

    // 4. RL throughput.
    let fast = rl_fast_steps(64, 4, (200_000.0 * scale) as u64);
    out.push(BenchResult {
        name: "rl_env_steps_fast".into(),
        value: fast,
        unit: "steps/sec".into(),
        reference: Some(
            "PufferLib claims ~1M+ steps/s at scale (Cython, large obs batches)".into(),
        ),
        notes: "64 envs x 4 threads, pure-compute env (no I/O)".into(),
        p50_ms: None,
        p99_ms: None,
        samples: None,
    });

    let enqueue = rl_buffer_enqueue(16, (200_000.0 * scale) as u64, 128, 64);
    out.push(BenchResult {
        name: "rl_buffer_enqueue".into(),
        value: enqueue,
        unit: "transitions/sec".into(),
        reference: Some("PufferReplayBuffer C core: ~10M+ transitions/s (Cython memcpy)".into()),
        notes: "obs 128 f32, hidden 64 f32 x2 (actor+critic)".into(),
        p50_ms: None,
        p99_ms: None,
        samples: None,
    });

    let gae = rl_gae(64, 8192);
    out.push(BenchResult {
        name: "rl_gae".into(),
        value: gae,
        unit: "transitions/sec".into(),
        reference: Some("PufferLib computes GAE in C during train()".into()),
        notes: "full-window GAE over 64 envs".into(),
        p50_ms: None,
        p99_ms: None,
        samples: None,
    });

    let driver = rl_driver(64, 4, (50_000.0 * scale) as u64, 64);
    out.push(BenchResult {
        name: "rl_driver_e2e".into(),
        value: driver,
        unit: "steps/sec".into(),
        reference: None,
        notes: "driver rollout incl. LSTM forward + hidden reset bookkeeping".into(),
        p50_ms: None,
        p99_ms: None,
        samples: None,
    });

    let sandbox_steps = rl_sandbox_steps(8, (2_000.0 * scale) as u64);
    out.push(BenchResult {
        name: "rl_env_steps_sandbox".into(),
        value: sandbox_steps,
        unit: "steps/sec".into(),
        reference: Some("DSec: RL agent loop over real sandbox sessions".into()),
        notes: "8 sandbox envs, real Chronus exec per step (zero-latency channel transport)".into(),
        p50_ms: None,
        p99_ms: None,
        samples: None,
    });

    // 5. pack_diff.
    let pd = packdiff(512, 0.05, 200);
    out.push(BenchResult {
        name: "packdiff_size_ratio".into(),
        value: pd.ratio,
        unit: "pack bytes / full image bytes".into(),
        reference: Some("DSec paper: pack_diff replicates working state w/o full copy".into()),
        notes: format!(
            "full {} KiB, pack {} KiB (5% of files mutated)",
            pd.full_bytes / 1024,
            pd.pack_bytes / 1024
        ),
        p50_ms: None,
        p99_ms: None,
        samples: None,
    });
    out.push(BenchResult {
        name: "packdiff_apply".into(),
        value: pd.applies_per_sec,
        unit: "applies/sec".into(),
        reference: None,
        notes: "fresh replica overlay + block import + file table import".into(),
        p50_ms: None,
        p99_ms: None,
        samples: None,
    });
    out.push(BenchResult {
        name: "packdiff_snapshot".into(),
        value: pd.packs_per_sec,
        unit: "packs/sec".into(),
        reference: None,
        notes: "dirty-set capture + serialization (post-first-snapshot management path)".into(),
        p50_ms: None,
        p99_ms: None,
        samples: None,
    });

    // 6. Placement.
    let pl = placement_bench(1024, (20_000.0 * scale) as usize, 42);
    out.push(BenchResult {
        name: "placement_decisions".into(),
        value: pl.decisions_per_sec,
        unit: "decisions/sec".into(),
        reference: Some("~5k creations/s x candidate scan must not bottleneck".into()),
        notes: format!("{} nodes, k={} sampled", pl.nodes, pl.k),
        p50_ms: None,
        p99_ms: None,
        samples: None,
    });
    let k8_quality = 100.0 * (pl.k8_avg_score - pl.k1_avg_score)
        / (pl.exhaustive_avg_score - pl.k1_avg_score).max(1e-9);
    out.push(BenchResult {
        name: "placement_kchoice_quality".into(),
        value: k8_quality,
        unit: "% of exhaustive-improvement achieved".into(),
        reference: Some("power-of-two-choices theory: k=8 approximates best-fit well".into()),
        notes: format!(
            "avg score k=1 {:.4}, k=8 {:.4}, exhaustive {:.4}",
            pl.k1_avg_score, pl.k8_avg_score, pl.exhaustive_avg_score
        ),
        p50_ms: None,
        p99_ms: None,
        samples: None,
    });

    // 8. Codec.
    let (enc, dec) = codec_throughput(200_000);
    out.push(BenchResult {
        name: "protocol_encode".into(),
        value: enc,
        unit: "MB/s".into(),
        reference: None,
        notes: "frame + CRC32 over 55-byte JSON payloads".into(),
        p50_ms: None,
        p99_ms: None,
        samples: None,
    });
    out.push(BenchResult {
        name: "protocol_decode".into(),
        value: dec,
        unit: "MB/s".into(),
        reference: None,
        notes: "CRC-verified decode".into(),
        p50_ms: None,
        p99_ms: None,
        samples: None,
    });

    out
}
