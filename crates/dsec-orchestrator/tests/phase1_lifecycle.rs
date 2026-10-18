//! Phase 1 integration: single agent, real environment, no training.
//!
//! Proves, over a **real HTTP model endpoint** (the in-process mock
//! server on a real TCP socket) and **both sandbox backends**:
//!
//! 1. the full task lifecycle — create → journaled multi-turn rollout
//!    (model call → tool call → observation → final answer) → verify →
//!    reward — with every boundary journaled;
//! 2. pause/resume: a paused task transparently resumes and still
//!    completes with the correct reward;
//! 3. failure handling: a dead model endpoint fails the task cleanly
//!    (journaled `task_failed`), never corrupting the journal;
//! 4. the dsec-sdk backend runs a file task through the real cluster
//!    (control plane + edge node + Aether transport).

#![cfg(feature = "server")]

use std::sync::Arc;

use serde_json::json;

use dsec_agentenv::envgen::{
    CheckSpec, ColumnSpec, DomainSpec, EnvSpec, RowSpec, RubricSpec, SystemSpec, TableSpec, ToolOp,
    ToolSpec,
};
use dsec_orchestrator::mockserver;
use dsec_orchestrator::model::OpenAiCompatClient;
use dsec_orchestrator::taskenv::{AgentEnvBackend, FileTask};
use dsec_orchestrator::worker::{Crashpoint, TaskOutcome, WorkerConfig, WorkerContainer};
use dsec_orchestrator::Journal;

const SECRET: &str = "KEY-7F3A-92Z";

fn test_spec() -> EnvSpec {
    EnvSpec {
        domain: DomainSpec::General,
        instance_id: "p1-test-env".into(),
        instruction: "Find the value of record R-1 and report it.".into(),
        systems: vec![SystemSpec {
            name: "notes".into(),
            tables: vec![TableSpec {
                name: "records".into(),
                primary_key: "rec_id".into(),
                columns: vec![
                    ColumnSpec {
                        name: "rec_id".into(),
                        sql_type: "TEXT".into(),
                    },
                    ColumnSpec {
                        name: "value".into(),
                        sql_type: "TEXT".into(),
                    },
                ],
            }],
            rows: vec![RowSpec {
                table: "records".into(),
                values: json!({"rec_id": "R-1", "value": SECRET})
                    .as_object()
                    .unwrap()
                    .clone(),
            }],
        }],
        tools: vec![ToolSpec {
            server: "notes".into(),
            name: "get_record".into(),
            description: "Fetch one record by id.".into(),
            op: ToolOp::Get {
                table: "records".into(),
                id_param: "rec_id".into(),
            },
        }],
        workspace: vec![],
        rubric: vec![RubricSpec {
            id: "answer".into(),
            weight: 1.0,
            method: "rule".into(),
            question: String::new(),
            pass_anchor: SECRET.into(),
            check: Some(CheckSpec::AllText {
                needles: vec![SECRET.into()],
            }),
            gate: false,
        }],
        protect_sources: false,
    }
}

async fn mock_model(
    tool: &str,
    params: serde_json::Value,
    final_text: &str,
) -> (String, Arc<OpenAiCompatClient>) {
    let responder = Arc::new(mockserver::TaskSolverResponder::new(
        tool, params, final_text,
    ));
    let (addr, _handle) =
        mockserver::serve(std::net::SocketAddr::from(([127, 0, 0, 1], 0)), responder).await;
    let url = format!("http://{}", addr);
    let client = Arc::new(OpenAiCompatClient::new(&url, "mock").unwrap());
    (url, client)
}

fn journal_path(name: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("dsec-p1-{}-{name}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

#[tokio::test]
async fn full_lifecycle_over_real_http() {
    let (_url, model) = mock_model(
        "notes.get_record",
        json!({"rec_id": "R-1"}),
        &format!("The record value is {SECRET}."),
    )
    .await;

    let bundle = test_spec().compile().unwrap().boot().unwrap();
    let env = AgentEnvBackend::new(bundle);
    let env_exec_baseline = 0;

    let path = journal_path("lifecycle");
    let journal = Journal::create(&path).unwrap();
    let mut worker = WorkerContainer::new(
        "task-p1".into(),
        Box::new(env),
        model,
        journal,
        WorkerConfig::default(),
    )
    .unwrap();

    let outcome = worker.run_task().await.unwrap();
    match &outcome {
        TaskOutcome::Completed { reward, rollout } => {
            assert!(reward.score().unwrap() > 0.99, "reward {:?}", reward);
            assert!(rollout.completed);
            assert!(rollout.final_message.contains(SECRET));
            assert_eq!(rollout.tool_turns, 1);
            // transcript: system, user, assistant(call), user(obs), assistant(final)
            assert_eq!(rollout.messages.len(), 5);
        }
        TaskOutcome::Failed { error } => panic!("task failed: {error}"),
    }

    // the env executed exactly one tool call (the ledger prevented any
    // re-execution even before recovery is involved)
    let env = worker.into_env().unwrap();
    assert_eq!(env.real_exec_count(), env_exec_baseline + 1);

    // journal: full lifecycle with intact chain
    let report = dsec_orchestrator::journal::replay(&path).unwrap();
    assert!(report.tampered_at.is_none());
    let kinds: Vec<&str> = report.events.iter().map(|e| e.kind.as_str()).collect();
    for expected in [
        "task_created",
        "turn_started",
        "model_call_started",
        "model_call_completed",
        "tool_call_started",
        "tool_call_completed",
        "verify_started",
        "task_completed",
    ] {
        assert!(
            kinds.contains(&expected),
            "journal missing {expected}: {kinds:?}"
        );
    }
    // the reward was recorded, unmasked
    let completed = report
        .events
        .iter()
        .find(|e| e.kind == "task_completed")
        .unwrap();
    assert!(completed.payload["score"].as_f64().unwrap() > 0.99);
    assert_eq!(completed.payload["masked"], false);
}

#[tokio::test]
async fn paused_task_resumes_and_completes() {
    let (_url, model) = mock_model(
        "notes.get_record",
        json!({"rec_id": "R-1"}),
        &format!("The record value is {SECRET}."),
    )
    .await;

    let bundle = test_spec().compile().unwrap().boot().unwrap();
    let env = AgentEnvBackend::new(bundle);
    let path = journal_path("pause");
    let journal = Journal::create(&path).unwrap();
    let mut worker = WorkerContainer::new(
        "task-pause".into(),
        Box::new(env),
        model,
        journal,
        WorkerConfig::default(),
    )
    .unwrap();

    // pause BEFORE the rollout: run_task must transparently resume
    worker.pause().await.unwrap();
    assert!(worker.env_mut().unwrap().is_paused());

    let outcome = worker.run_task().await.unwrap();
    match &outcome {
        TaskOutcome::Completed { reward, .. } => {
            assert!(reward.score().unwrap() > 0.99);
        }
        TaskOutcome::Failed { error } => panic!("paused/resumed task failed: {error}"),
    }
    // journal captured both the pause and the resume
    let report = dsec_orchestrator::journal::replay(&path).unwrap();
    let kinds: Vec<&str> = report.events.iter().map(|e| e.kind.as_str()).collect();
    assert!(kinds.contains(&"task_paused"));
    assert!(kinds.contains(&"task_resumed"));
}

#[tokio::test]
async fn dead_model_endpoint_fails_task_cleanly() {
    // a client pointed at a closed port, single attempt, tight backoff
    let model = Arc::new(
        OpenAiCompatClient::new("http://127.0.0.1:1", "mock")
            .unwrap()
            .with_retry(
                std::time::Duration::from_millis(10),
                1,
                std::time::Duration::from_millis(1),
            ),
    );

    let bundle = test_spec().compile().unwrap().boot().unwrap();
    let env = AgentEnvBackend::new(bundle);
    let path = journal_path("deadend");
    let journal = Journal::create(&path).unwrap();
    let mut worker = WorkerContainer::new(
        "task-dead".into(),
        Box::new(env),
        model,
        journal,
        WorkerConfig {
            max_model_failures: 2,
            ..Default::default()
        },
    )
    .unwrap();

    let outcome = worker.run_task().await.unwrap();
    match &outcome {
        TaskOutcome::Failed { error } => {
            assert!(error.contains("model"), "error: {error}");
        }
        TaskOutcome::Completed { .. } => panic!("dead endpoint must fail the task"),
    }
    // journal records the failure and the chain stays intact
    let report = dsec_orchestrator::journal::replay(&path).unwrap();
    assert!(report.tampered_at.is_none());
    let failed = report.events.iter().find(|e| e.kind == "task_failed");
    assert!(failed.is_some(), "task_failed must be journaled");
}

#[tokio::test]
async fn crash_during_pause_is_recoverable_state() {
    // crash BETWEEN the env pausing and the journal recording it: the
    // env is paused, the journal still says running. Recovery must
    // reconcile (resume) and complete.
    let (_url, model) = mock_model(
        "notes.get_record",
        json!({"rec_id": "R-1"}),
        &format!("The record value is {SECRET}."),
    )
    .await;

    let bundle = test_spec().compile().unwrap().boot().unwrap();
    let env = AgentEnvBackend::new(bundle);
    let path = journal_path("crashpause");
    let journal = Journal::create(&path).unwrap();
    let mut worker = WorkerContainer::new(
        "task-crashpause".into(),
        Box::new(env),
        model.clone(),
        journal,
        WorkerConfig::default(),
    )
    .unwrap()
    .with_crashpoints(&[Crashpoint::DuringSandboxPause]);

    // the crash fires mid-pause
    let err = worker.pause().await;
    assert!(matches!(
        err,
        Err(dsec_orchestrator::Error::CrashInjected { .. })
    ));
    // env paused, journal has no task_paused
    assert!(worker.env_mut().unwrap().is_paused());
    let report = dsec_orchestrator::journal::replay(&path).unwrap();
    assert!(!report.events.iter().any(|e| e.kind == "task_paused"));

    // the sandbox outlives the worker; recovery re-attaches
    let env = worker.into_env().unwrap();
    let recovered = WorkerContainer::recover(
        "task-crashpause".into(),
        env,
        model,
        path,
        WorkerConfig::default(),
        Default::default(),
    )
    .await
    .unwrap();
    let mut recovered = recovered;
    assert!(
        !recovered.env_mut().unwrap().is_paused(),
        "recovery resumed the env"
    );
    let outcome = recovered.run_task().await.unwrap();
    assert!(matches!(outcome, TaskOutcome::Completed { .. }));
}

#[tokio::test]
async fn dsec_sandbox_backend_file_task_lifecycle() {
    use dsec_control::model::{Resources, SandboxSpec};
    use dsec_orchestrator::taskenv::DsecSandboxBackend;
    use dsec_sdk::integration::{spawn_local_cluster, ClusterConfig, DataPlane};

    let (_url, model) = mock_model(
        "write_file",
        json!({"path": "/tmp/out.txt", "content": "SECRET-42"}),
        "DONE",
    )
    .await;

    // a REAL local cluster with the UDS data plane (real sockets)
    let tmp = tempfile::tempdir().unwrap();
    let cluster = spawn_local_cluster(ClusterConfig {
        node_count: 1,
        data_plane: DataPlane::Uds {
            root: tmp.path().join("uds"),
        },
        ..Default::default()
    })
    .await;

    let task = FileTask::write_file_task("p1-file", "/tmp/out.txt", "SECRET-42");
    let spec = SandboxSpec {
        project: "root".into(),
        resources: Resources {
            cpu_millicores: 500,
            mem_mib: 128,
        },
        ..Default::default()
    };
    let env = DsecSandboxBackend::provision(task, &cluster, spec)
        .await
        .unwrap();

    let _path = journal_path("dsecfile");
    let journal = Journal::create(&_path).unwrap();
    let mut worker = WorkerContainer::new(
        "task-dsec-file".into(),
        Box::new(env),
        model,
        journal,
        WorkerConfig::default(),
    )
    .unwrap();

    let outcome = worker.run_task().await.unwrap();
    match &outcome {
        TaskOutcome::Completed { reward, .. } => {
            assert!(
                reward.score().unwrap() > 0.99,
                "file check failed: {reward:?}"
            );
        }
        TaskOutcome::Failed { error } => panic!("dsec file task failed: {error}"),
    }

    // the sandbox executed the write exactly once
    let mut env = worker.into_env().unwrap();
    assert_eq!(env.real_exec_count(), 1);

    // real pause/resume on the node (state preserved), then destroy
    env.pause().await.unwrap();
    assert!(env.is_paused());
    env.resume().await.unwrap();
    env.destroy().await.unwrap();
}
