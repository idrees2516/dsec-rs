//! Burst simulation at paper scale: 100,000 sandbox lifecycles
//! (create -> exec -> destroy) through the full control plane, with
//! live progress and final stats.
//!
//! Run: `cargo run --release --example burst_simulation -p dsec-sdk -- [count]`

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use dsec_control::model::SandboxSpec;
use dsec_sdk::integration::{spawn_local_cluster, ClusterConfig};

#[tokio::main]
async fn main() {
    let count: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let concurrency = 256;

    println!(
        "== dsec-rs burst simulation: {} lifecycles, concurrency {} ==",
        count, concurrency
    );
    let cluster = spawn_local_cluster(ClusterConfig {
        cpu_millicores: 100_000_000,
        mem_mib: 100_000_000,
        max_sandboxes: count as i64 + 1000,
        prewarm: 256,
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

    let ok = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));
    let peak = Arc::new(AtomicU64::new(0));
    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));

    let t0 = Instant::now();
    let mut tasks = Vec::with_capacity(count.min(10_000));
    let mut progress = 0usize;
    while progress < count {
        let batch = 1000.min(count - progress);
        for _ in 0..batch {
            let permit = sem.clone().acquire_owned().await.unwrap();
            let plane = plane.clone();
            let token = token.clone();
            let ok = ok.clone();
            let errors = errors.clone();
            let peak = peak.clone();
            let nodes = nodes.clone();
            tasks.push(tokio::spawn(async move {
                let spec = SandboxSpec::default();
                match plane.create_sandbox(&token, spec).await {
                    Ok(rec) => {
                        ok.fetch_add(1, Ordering::Relaxed);
                        let cur = plane.registry.sandboxes().len() as u64;
                        peak.fetch_max(cur, Ordering::Relaxed);
                        // One real exec through the node data plane.
                        let node = &nodes[0];
                        let _ = node
                            .handle_aether_request(
                                rec.sid,
                                dsec_protocol::frame::Channel::Exec,
                                dsec_protocol::message::Request::Exec {
                                    cmd: "echo burst".into(),
                                    timeout_ms: None,
                                },
                                1,
                            )
                            .await;
                        let _ = plane.destroy_sandbox(&token, rec.sid).await;
                    }
                    Err(_) => {
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
                drop(permit);
            }));
        }
        for t in tasks.drain(..) {
            let _ = t.await;
        }
        progress += batch;
        let done = ok.load(Ordering::Relaxed) + errors.load(Ordering::Relaxed);
        print!(
            "\r{}/{} lifecycles ({} ok, {} errors)",
            done,
            count,
            ok.load(Ordering::Relaxed),
            errors.load(Ordering::Relaxed)
        );
    }
    println!();
    let secs = t0.elapsed().as_secs_f64();
    println!("------------------------------------------------------------");
    println!("total      : {} lifecycles in {:.2}s", count, secs);
    println!(
        "throughput : {:.0} sandboxes/sec (create+exec+destroy)",
        count as f64 / secs
    );
    println!(
        "peak live  : {} concurrent sandboxes",
        peak.load(Ordering::Relaxed)
    );
    println!("errors     : {}", errors.load(Ordering::Relaxed));
    println!(
        "node stats : created={} destroyed={}",
        cluster.nodes[0]
            .stats
            .sandboxes_created
            .load(Ordering::Relaxed),
        cluster.nodes[0]
            .stats
            .sandboxes_destroyed
            .load(Ordering::Relaxed)
    );
    println!(
        "creation p50/p99: {:.3}ms / {:.3}ms",
        cluster.nodes[0].stats.creation().p50_ms as f64,
        cluster.nodes[0].stats.creation().p99_ms as f64
    );
}
