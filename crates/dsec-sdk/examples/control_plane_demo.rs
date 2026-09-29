//! Full-stack demo: IAM project + quota + REST creation + sandbox
//! operations + pack_diff replication + pause/resume + streaming + pool.
//!
//! Run: `cargo run --release --example control_plane_demo -p dsec-sdk`

use std::sync::Arc;
use std::time::Duration;

use dsec_control::model::{Quota, Resources, SandboxSpec};
use dsec_sdk::integration::{spawn_local_cluster, ClusterConfig};
use dsec_sdk::{DsecClient, Endpoint};

#[tokio::main]
async fn main() {
    println!("== dsec-rs control plane demo ==");

    // 1. Assemble a 3-node cluster (zero injected latency for snappiness).
    let cluster = spawn_local_cluster(ClusterConfig {
        node_count: 3,
        ..Default::default()
    })
    .await;
    println!(
        "cluster: {} nodes, apiserver at {}",
        cluster.nodes.len(),
        cluster.api_addr
    );

    // 2. IAM: nested project with a quota chain.
    cluster
        .plane
        .create_project("root", "lab", Quota::limited(4000, 4096, 8))
        .unwrap();
    cluster
        .plane
        .create_project("root/lab", "rl-team", Quota::default())
        .unwrap();
    let lab_token = cluster
        .plane
        .create_token("root/lab/rl-team", dsec_control::iam::Role::Writer)
        .unwrap()
        .token;

    // 3. Client with the lab token (channel data plane).
    let client = DsecClient::new(
        lab_token.clone(),
        Endpoint::localhost(cluster.api_addr.port()),
        Arc::new(dsec_sdk::transport::ChannelTransport::with_nodes(
            cluster.nodes.clone(),
        )),
    );

    // 4. Create sandboxes through REST; operate via the data plane.
    let spec = SandboxSpec {
        project: "root/lab/rl-team".into(),
        resources: Resources {
            cpu_millicores: 500,
            mem_mib: 256,
        },
        ..Default::default()
    };
    let mut sandboxes = Vec::new();
    for _ in 0..3 {
        let sandbox = client.create_sandbox(spec.clone()).await.unwrap();
        let hostname =
            String::from_utf8_lossy(&sandbox.execute("cat /etc/hostname").await.unwrap().stdout)
                .trim()
                .to_string();
        println!(
            "sandbox {} on {}: {}",
            sandbox.sid(),
            sandbox.node_id(),
            hostname
        );
        sandboxes.push(sandbox);
    }

    // 5. Filesystem + HTTP through Chronus.
    let sandbox = &sandboxes[0];
    sandbox
        .write_file("/tmp/work.txt", b"agent working state")
        .await
        .unwrap();
    println!(
        "wrote /tmp/work.txt ({} bytes)",
        sandbox.stat("/tmp/work.txt").await.unwrap().size
    );
    let (status, body) = sandbox
        .http_get("https://api.internal/version")
        .await
        .unwrap();
    println!(
        "proxied http: {} {:?}",
        status,
        String::from_utf8_lossy(&body)
    );

    // 6. pack_diff: replicate sandbox 1's working state onto sandbox 2.
    // (Cross-node replication: the base image is shared, so only the
    // dirty blocks + metadata move — exactly the paper's fast path.)
    let node_of = |node_id: &str| {
        cluster
            .nodes
            .iter()
            .find(|n| n.node_id == node_id)
            .cloned()
            .unwrap()
    };
    let src_node = node_of(sandboxes[0].node_id());
    let dst_node = node_of(sandboxes[1].node_id());
    let src_entry = src_node.lookup(sandboxes[0].sid()).unwrap();
    let dst_entry = dst_node.lookup(sandboxes[1].sid()).unwrap();
    let pack = src_entry.instance.snapshot_diff(1);
    dst_entry.instance.apply_diff(&pack).unwrap();
    let data = sandboxes[1].read_file("/tmp/work.txt").await.unwrap();
    println!(
        "pack_diff: {} blocks moved; replica reads {:?}",
        pack.blocks.len(),
        String::from_utf8_lossy(&data)
    );

    // 7. Pause / resume via REST; watch the record state.
    sandboxes[0].pause().await.unwrap();
    println!(
        "paused: {:?}",
        client.get_sandbox(sandboxes[0].sid()).await.unwrap().state
    );
    sandboxes[0].resume().await.unwrap();
    println!(
        "resumed: {:?}",
        client.get_sandbox(sandboxes[0].sid()).await.unwrap().state
    );

    // 8. Streaming exec.
    let stream = sandboxes[0].execute_stream("seq 5").await.unwrap();
    let result = stream.collect().await.unwrap();
    println!("stream: {:?}", String::from_utf8_lossy(&result.stdout));

    // 9. Quota enforcement: fill to the 8-sandbox / 4000m cap.
    let mut created = 3;
    for _ in 3..8 {
        if client.create_sandbox(spec.clone()).await.is_ok() {
            created += 1;
        }
    }
    println!("sandboxes under quota: {}", created);
    match client.create_sandbox(spec.clone()).await {
        Err(dsec_sdk::Error::Http { status, message }) => {
            println!("quota hit: HTTP {} {}", status, message);
        }
        other => panic!("expected quota rejection, got {:?}", other.err()),
    }

    // 10. Sandbox pool with warm reuse (fresh client, same transport).
    let client2 = DsecClient::new(
        lab_token,
        Endpoint::localhost(cluster.api_addr.port()),
        Arc::new(dsec_sdk::transport::ChannelTransport::with_nodes(
            cluster.nodes.clone(),
        )),
    );
    let pool = dsec_sdk::SandboxPool::new(
        Arc::new(client2),
        dsec_sdk::PoolConfig {
            min_idle: 2,
            max_size: 8,
            idle_ttl: Duration::from_secs(60),
            spec: spec.clone(),
        },
    );
    // The quota is already full, so the pool stays empty — demonstrate
    // acquire failure instead (the honest demo at this quota).
    match pool.acquire().await {
        Err(dsec_sdk::Error::Http { status, message }) => {
            println!("pool acquire under full quota: HTTP {} {}", status, message);
        }
        Ok(_) => println!("pool acquired (quota had room)"),
        Err(other) => println!("pool acquire error: {:?}", other),
    }

    println!("== demo complete ==");
}
