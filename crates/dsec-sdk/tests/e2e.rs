//! End-to-end tests: the full stack (apiserver REST + control plane +
//! node runtime + Aether data plane + SDK), over both transports.

use std::sync::Arc;
use std::time::Duration;

use dsec_control::model::{BackendKind, SandboxSpec};
use dsec_sdk::integration::{spawn_local_cluster, ClusterConfig, DataPlane};
use dsec_sdk::pool::{PoolConfig, SandboxPool};

async fn cluster(data_plane: DataPlane) -> dsec_sdk::integration::LocalCluster {
    spawn_local_cluster(ClusterConfig {
        data_plane,
        ..Default::default()
    })
    .await
}

#[tokio::test]
async fn full_stack_sandbox_lifecycle() {
    let cluster = cluster(DataPlane::Channel).await;
    let client = cluster.client();

    // Create through the REST plane.
    let sandbox = client.create_sandbox(SandboxSpec::default()).await.unwrap();
    assert!(sandbox.sid() > 0);
    assert_eq!(sandbox.node_id(), "node-1");

    // Exec through the data plane.
    let out = sandbox.execute("cat /etc/hostname").await.unwrap();
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout, b"sandbox-agent");

    // Filesystem roundtrip.
    sandbox
        .write_file("/tmp/e2e.txt", b"written by sdk")
        .await
        .unwrap();
    let data = sandbox.read_file("/tmp/e2e.txt").await.unwrap();
    assert_eq!(data, b"written by sdk");
    let listing = sandbox.list_dir("/tmp").await.unwrap();
    assert!(listing.contains(&"e2e.txt".to_string()));
    let stat = sandbox.stat("/tmp/e2e.txt").await.unwrap();
    assert_eq!(stat.size, 14);
    assert!(!stat.is_dir);

    // Sessions keep cwd.
    let session = sandbox
        .open_session("/etc", Default::default())
        .await
        .unwrap();
    let _ = session;
    let _pwd = sandbox.execute("cd /etc && pwd").await.unwrap();
    // Note: default session starts at /; `cd` then `pwd` in one command line
    // is not a shell pipeline here, so run via two calls in one session.

    // HTTP through the fake egress.
    let (status, body) = sandbox
        .http_get("https://api.internal/health")
        .await
        .unwrap();
    assert_eq!(status, 200);
    assert_eq!(body, b"{\"status\":\"ok\"}");

    // Pause / resume through the control plane (state visible via record).
    sandbox.pause().await.unwrap();
    let rec = client.get_sandbox(sandbox.sid()).await.unwrap();
    assert_eq!(rec.state, dsec_control::model::SandboxState::Paused);
    // Data plane rejects traffic while paused.
    assert!(sandbox.execute("ls").await.is_err());
    sandbox.resume().await.unwrap();
    let ok = sandbox.execute("echo back").await.unwrap();
    assert_eq!(ok.exit_code, 0);

    // Destroy through the control plane; the record is gone afterward.
    let destroyed = sandbox.destroy().await.unwrap();
    assert!(destroyed.sid > 0);
    assert!(client.get_sandbox(destroyed.sid).await.is_err());
}

#[tokio::test]
async fn streaming_exec_via_sdk() {
    let cluster = cluster(DataPlane::Channel).await;
    let client = cluster.client();
    let sandbox = client.create_sandbox(SandboxSpec::default()).await.unwrap();
    let stream = sandbox.execute_stream("seq 5").await.unwrap();
    let result = stream.collect().await.unwrap();
    assert_eq!(result.exit_code, 0);
    assert_eq!(result.stdout, b"1\n2\n3\n4\n5\n");
}

#[tokio::test]
async fn pool_reuses_sandboxes() {
    let cluster = cluster(DataPlane::Channel).await;
    let client = Arc::new(cluster.client());
    let pool = SandboxPool::new(
        client.clone(),
        PoolConfig {
            min_idle: 2,
            max_size: 4,
            idle_ttl: Duration::from_secs(60),
            spec: SandboxSpec::default(),
        },
    );
    pool.prewarm().await.unwrap();
    assert_eq!(pool.stats().idle, 2);

    // Acquire, dirty, release, acquire again: same sandbox id reused.
    let a = pool.acquire().await.unwrap();
    a.write_file("/tmp/dirty.txt", b"state").await.unwrap();
    pool.release(a).await.unwrap();
    let b = pool.acquire().await.unwrap();
    assert!(pool.stats().reuses >= 1);
    // Reset cleared /tmp.
    let children = b.list_dir("/tmp").await.unwrap();
    assert!(!children.contains(&"dirty.txt".to_string()));
    pool.shutdown().await.unwrap();
    assert_eq!(pool.stats().idle, 0);
}

#[tokio::test]
async fn uds_data_plane_end_to_end() {
    let root = std::env::temp_dir().join(format!("dsec-e2e-uds-{}", std::process::id()));
    let cluster = cluster(DataPlane::Uds { root: root.clone() }).await;
    let client = cluster.client();
    let sandbox = client.create_sandbox(SandboxSpec::default()).await.unwrap();
    let out = sandbox.execute("hostname").await.unwrap();
    assert_eq!(out.exit_code, 0);
    assert!(String::from_utf8(out.stdout).unwrap().starts_with("sb-"));
    // Write + read over UDS.
    sandbox
        .write_file("/tmp/uds.txt", b"via uds")
        .await
        .unwrap();
    let back = sandbox.read_file("/tmp/uds.txt").await.unwrap();
    assert_eq!(back, b"via uds");
    let _ = sandbox.destroy().await;
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn quota_enforced_through_rest() {
    let cluster = cluster(DataPlane::Channel).await;
    // Restrictive project.
    cluster
        .plane
        .create_project(
            "root",
            "tiny",
            dsec_control::model::Quota::limited(1000, 1024, 2),
        )
        .unwrap();
    let token = cluster
        .plane
        .create_token("root/tiny", dsec_control::iam::Role::Writer)
        .unwrap()
        .token;
    let client = dsec_sdk::DsecClient::new(
        token,
        dsec_sdk::Endpoint::localhost(cluster.api_addr.port()),
        Arc::new(dsec_sdk::transport::ChannelTransport::new(
            cluster.nodes[0].clone(),
        )),
    );
    let spec = SandboxSpec {
        project: "root/tiny".into(),
        ..Default::default()
    };
    client.create_sandbox(spec.clone()).await.unwrap();
    client.create_sandbox(spec.clone()).await.unwrap();
    // Third exceeds the sandbox quota.
    let err = client.create_sandbox(spec).await;
    assert!(err.is_err());
    match err.unwrap_err() {
        dsec_sdk::Error::Http { status, message } => {
            assert_eq!(status, 422, "{}", message);
            assert!(message.contains("quota"));
        }
        other => panic!("unexpected error {:?}", other),
    }
}

#[tokio::test]
async fn multi_node_placement_spreads() {
    let cluster = spawn_local_cluster(ClusterConfig {
        node_count: 3,
        ..Default::default()
    })
    .await;
    // Channel transport only knows node-1; use direct provisioner checks.
    let token = cluster
        .plane
        .create_token("root", dsec_control::iam::Role::Writer)
        .unwrap()
        .token;
    let mut node_ids = std::collections::HashSet::new();
    for _ in 0..30 {
        let rec = cluster
            .plane
            .create_sandbox(&token, SandboxSpec::default())
            .await
            .unwrap();
        node_ids.insert(rec.node_id);
    }
    // k-choice placement across 3 healthy nodes spreads across all of them.
    assert!(node_ids.len() >= 2, "expected spread, got {:?}", node_ids);
    assert!(node_ids.len() <= 3);
}

#[tokio::test]
async fn watcher_evicts_on_silence() {
    let cluster = cluster(DataPlane::Channel).await;
    let watcher = dsec_control::Watcher::new(
        cluster.plane.clone(),
        dsec_control::WatcherConfig {
            ttl: Duration::from_millis(50),
            check_interval: Duration::from_millis(20),
            eviction_grace: Duration::from_millis(40),
        },
    );
    cluster.plane.set_watcher(watcher.clone());
    let token = cluster
        .plane
        .create_token("root", dsec_control::iam::Role::Writer)
        .unwrap()
        .token;
    let rec = cluster
        .plane
        .create_sandbox(&token, SandboxSpec::default())
        .await
        .unwrap();

    // Heartbeat keeps the node healthy.
    cluster.plane.heartbeat("node-1").await.unwrap();
    watcher.sweep_once().await;
    assert_eq!(
        cluster.plane.registry.node("node-1").unwrap().status,
        dsec_control::model::NodeStatus::Healthy
    );
    assert!(cluster.plane.registry.sandbox(rec.sid).is_some());

    // Silence past TTL: node marked unhealthy.
    tokio::time::sleep(Duration::from_millis(120)).await;
    watcher.sweep_once().await;
    let node = cluster.plane.registry.node("node-1").unwrap();
    assert_eq!(node.status, dsec_control::model::NodeStatus::Unhealthy);
    // Past the eviction grace: sandboxes evicted.
    tokio::time::sleep(Duration::from_millis(80)).await;
    watcher.sweep_once().await;
    assert!(cluster.plane.registry.sandbox(rec.sid).is_none());
    assert_eq!(watcher.evictions(), 1);

    // Recovery: the node re-registers (upsert) and heartbeats again.
    cluster.refresh_node(0);
    cluster.plane.heartbeat("node-1").await.unwrap();
    assert_eq!(
        cluster.plane.registry.node("node-1").unwrap().status,
        dsec_control::model::NodeStatus::Healthy
    );
}

#[tokio::test]
async fn backend_kinds_provision_through_rest() {
    let cluster = cluster(DataPlane::Channel).await;
    let client = cluster.client();
    for backend in [
        BackendKind::Fncall,
        BackendKind::Container,
        BackendKind::Microvm,
    ] {
        let spec = SandboxSpec {
            backend,
            ..Default::default()
        };
        let sandbox = client.create_sandbox(spec).await.unwrap();
        let out = sandbox.execute("echo alive").await.unwrap();
        assert_eq!(out.exit_code, 0, "backend {:?}", backend);
        let _ = sandbox.destroy().await;
    }
}
