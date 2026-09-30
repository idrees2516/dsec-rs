//! End-to-end driver wiring tests: the full Firecracker request
//! sequence and the EdgeNode integration, exercised over REAL unix
//! sockets against the fake VMM (no KVM needed).
//!
//! A real-firecracker end-to-end test runs only when the host provides
//! the binary + /dev/kvm + kernel (opt-in via `DSEC_FIRECRACKER_E2E=1`).

use std::sync::Arc;
use std::time::Duration;

use dsec_firecracker::{FakeLauncher, FirecrackerConfig, FirecrackerDriver};
use dsec_runtime::backend::{BackendKind, SandboxSpec};
use dsec_runtime::microvm::MicrovmDriver;
use dsec_runtime::{BackendFactory, EdgeNode};
use dsec_storage::cache::LruBlockCache;
use dsec_storage::erofs::{ErofsImageBuilder, ImageRegistry};
use dsec_storage::latency::{LatencyModel, NodeLatencyProfile};

fn test_registry() -> Arc<ImageRegistry> {
    let image = Arc::new(ErofsImageBuilder::agent_base().build());
    let mut reg = ImageRegistry::default();
    reg.register(
        image,
        Arc::new(LruBlockCache::new(64)),
        LatencyModel::fixed(Duration::ZERO),
    );
    Arc::new(reg)
}

fn test_config(tag: &str) -> FirecrackerConfig {
    let base = std::env::temp_dir().join(format!("dsec-fc-e2e-{tag}-{}", std::process::id()));
    FirecrackerConfig {
        allow_missing_host_deps: true,
        api_sock_dir: base.join("sock"),
        rootfs_dir: base.join("rootfs"),
        scratch_dir: base.join("scratch"),
        snapshot_dir: base.join("snap"),
        boot_timeout: Duration::from_secs(2),
        request_timeout: Duration::from_secs(2),
        ..Default::default()
    }
}

#[tokio::test]
async fn full_lifecycle_over_fake_vmm() {
    let registry = test_registry();
    let launcher = FakeLauncher::default();
    let driver = FirecrackerDriver::with_launcher(
        test_config("lifecycle"),
        registry,
        Arc::new(launcher.clone()),
    );

    let spec = SandboxSpec {
        backend: BackendKind::Microvm,
        ..Default::default()
    };
    let vm = driver.boot(1, &spec).await.unwrap();
    assert_eq!(vm.vm_id, "sb-000001");
    assert!(vm.started);
    assert!(vm.api_sock.exists());

    // The full Firecracker request sequence, in order.
    let fake = &launcher.fakes()[0];
    let requests = fake.requests.lock().unwrap().clone();
    let paths: Vec<&str> = requests.iter().map(String::as_str).collect();
    assert_eq!(
        paths,
        vec![
            "PUT /machine-config",
            "PUT /boot-source",
            "PUT /drives/rootfs",
            "PUT /drives/scratch",
            "PUT /actions",
        ]
    );
    assert_eq!(fake.state(), "running");

    // Pause → snapshot (diff) → resume.
    driver.pause(&vm).await.unwrap();
    assert_eq!(fake.state(), "paused");
    let snap = driver.snapshot(&vm, true).await.unwrap();
    assert!(snap.snapshot_path.is_file());
    assert_eq!(fake.snapshot_diff(), Some(true));
    driver.resume(&vm).await.unwrap();
    assert_eq!(fake.state(), "running");

    // Restore into a replica (fast-resume path).
    let replica = driver.restore(&snap, &spec).await.unwrap();
    assert!(replica.vm_id.starts_with("rep-"));
    assert!(launcher.fakes().len() >= 2);
    assert!(launcher.fakes().last().unwrap().restored());

    // Destroy: graceful power-off request + socket cleanup.
    driver.destroy(&vm).await.unwrap();
    assert_eq!(fake.state(), "exited");
    assert!(fake.exited_cleanly());
    assert!(!vm.api_sock.exists());
}

#[tokio::test]
async fn shared_rootfs_materialized_once_across_vms() {
    let registry = test_registry();
    let launcher = FakeLauncher::default();
    let config = test_config("shared");
    let rootfs_dir = config.rootfs_dir.clone();
    let driver = FirecrackerDriver::with_launcher(config, registry, Arc::new(launcher.clone()));
    let spec = SandboxSpec {
        backend: BackendKind::Microvm,
        ..Default::default()
    };
    let a = driver.boot(10, &spec).await.unwrap();
    let b = driver.boot(11, &spec).await.unwrap();
    // Exactly one rootfs drive file for the shared image, referenced
    // read-only by both VMs.
    let files: Vec<_> = std::fs::read_dir(&rootfs_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(files.len(), 1, "expected a single shared rootfs image");
    let request_logs: Vec<Vec<String>> = launcher
        .fakes()
        .iter()
        .map(|f| f.requests.lock().unwrap().clone())
        .collect();
    let drive_requests: Vec<&String> = request_logs
        .iter()
        .flatten()
        .filter(|r| r.starts_with("PUT /drives/rootfs"))
        .collect();
    assert_eq!(drive_requests.len(), 2);
    assert_eq!(drive_requests[0], drive_requests[1]);
    let _ = driver.destroy(&a).await;
    let _ = driver.destroy(&b).await;
}

#[tokio::test]
async fn missing_image_fails_boot() {
    let registry = Arc::new(ImageRegistry::default()); // empty
    let launcher = FakeLauncher::default();
    let driver = FirecrackerDriver::with_launcher(
        test_config("missing"),
        registry,
        Arc::new(launcher.clone()),
    );
    let spec = SandboxSpec {
        backend: BackendKind::Microvm,
        image_id: "dsec/ghost".into(),
        ..Default::default()
    };
    let err = driver.boot(2, &spec).await.unwrap_err();
    assert!(err.to_string().contains("ghost"), "err: {err}");
}

#[tokio::test]
async fn host_deps_enforced_for_real_launchers() {
    // Real ProcessLauncher path (allow_missing_host_deps = false): on a
    // box without firecracker/KVM, boot must fail with the honest
    // blocker list, never spawn a mystery process.
    let registry = test_registry();
    let mut cfg = test_config("hostdeps");
    cfg.allow_missing_host_deps = false; // enforce real capability checks
    let driver = FirecrackerDriver::new(cfg, registry);
    let spec = SandboxSpec {
        backend: BackendKind::Microvm,
        ..Default::default()
    };
    match driver.boot(3, &spec).await {
        Ok(_) => {
            // A host with everything installed: fine.
            assert!(driver.capability().ready());
        }
        Err(e) => {
            let msg = e.to_string();
            assert!(msg.contains("unavailable"), "err: {msg}");
        }
    }
}

/// EdgeNode integration: a MicroVM sandbox created through an EdgeNode
/// with the driver installed routes its whole lifecycle through the
/// Firecracker API instead of the latency-profile simulation, and the
/// other backends keep the simulated path.
#[tokio::test]
async fn edge_node_routes_lifecycle_through_driver() {
    let registry = test_registry();
    let launcher = FakeLauncher::default();
    let driver = Arc::new(FirecrackerDriver::with_launcher(
        test_config("edge"),
        registry.clone(),
        Arc::new(launcher.clone()),
    ));
    let factory =
        BackendFactory::new(registry, NodeLatencyProfile::zero(), 0).with_microvm_driver(driver);
    let node = EdgeNode::with_factory("fc-node".into(), factory, 100_000, 100_000, 100);

    let spec = SandboxSpec {
        backend: BackendKind::Microvm,
        ..Default::default()
    };
    let entry = node.create(spec).await.unwrap();
    assert_eq!(entry.sid, 1);
    // The instance carries a real VM handle.
    let inst = entry.instance.clone();
    assert!(inst.vm.is_some(), "expected driver-backed microVM");
    assert_eq!(inst.vm.as_ref().unwrap().vm_id, "sb-000001");

    // Lifecycle through the VMM API (no simulated sleeps involved).
    node.pause(entry.sid).await.unwrap();
    assert_eq!(launcher.fakes()[0].state(), "paused");
    node.resume(entry.sid).await.unwrap();
    assert_eq!(launcher.fakes()[0].state(), "running");
    node.destroy(entry.sid).await.unwrap();
    assert_eq!(launcher.fakes()[0].state(), "exited");

    // Non-microVM backends stay on the simulated path (no VM handle).
    let fncall = SandboxSpec::default(); // Fncall
    let plain = node.create(fncall).await.unwrap();
    assert!(plain.instance.vm.is_none());
    let _ = node.destroy(plain.sid).await;
}

/// Optional: true end-to-end against a real firecracker binary.
/// Run on a KVM host with DSEC_FIRECRACKER_PATH + DSEC_FC_KERNEL set:
/// `DSEC_FIRECRACKER_E2E=1 cargo test -p dsec-firecracker --test integration`
#[tokio::test]
async fn real_firecracker_e2e_opt_in() {
    if std::env::var("DSEC_FIRECRACKER_E2E").ok().as_deref() != Some("1") {
        eprintln!("skipping real e2e (set DSEC_FIRECRACKER_E2E=1 to enable)");
        return;
    }
    let cap = dsec_firecracker::detect();
    assert!(cap.ready(), "host not capable: {:?}", cap.blockers());
    let registry = test_registry();
    let driver = FirecrackerDriver::new(FirecrackerConfig::default(), registry);
    let spec = SandboxSpec {
        backend: BackendKind::Microvm,
        ..Default::default()
    };
    let vm = driver.boot(9, &spec).await.expect("real boot");
    driver.pause(&vm).await.expect("pause");
    let snap = driver.snapshot(&vm, true).await.expect("snapshot");
    driver.resume(&vm).await.expect("resume");
    driver.destroy(&vm).await.expect("destroy");
    assert!(snap.snapshot_path.is_file());
}
