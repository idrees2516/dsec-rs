//! Phase 1 demo: a single agent runs one task through a real model
//! endpoint and sandbox backend, with the full lifecycle journaled.
//!
//! Two backends, both driven over real HTTP by the mock model server:
//! 1. `--agentenv` — a `dsec-agentenv` pod (EnvSpec DSL, MCP tools,
//!    native rubric verifier);
//! 2. `--dsec` — a `dsec-sdk` sandbox on a real local cluster (UDS
//!    data plane).
//!
//! For a REAL model endpoint, set `DSEC_MODEL_URL`, `DSEC_MODEL_KEY`,
//! `DSEC_MODEL_NAME` and pass `--real`.
//!
//! Run: `cargo run --release --example phase1_task --features server -p dsec-orchestrator [-- --dsec|--real]`

use std::sync::Arc;

use serde_json::json;

use dsec_agentenv::envgen::{
    CheckSpec, ColumnSpec, DomainSpec, EnvSpec, RowSpec, RubricSpec, SystemSpec, TableSpec, ToolOp,
    ToolSpec,
};
use dsec_orchestrator::mockserver;
use dsec_orchestrator::model::{ModelClient, OpenAiCompatClient};
use dsec_orchestrator::taskenv::{AgentEnvBackend, DsecSandboxBackend, FileTask, TaskEnvironment};
use dsec_orchestrator::worker::{TaskOutcome, WorkerConfig, WorkerContainer};
use dsec_orchestrator::Journal;

const SECRET: &str = "KEY-7F3A-92Z";

fn test_spec() -> EnvSpec {
    EnvSpec {
        domain: DomainSpec::General,
        instance_id: "demo-kw".into(),
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

fn journal_path(name: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("dsec-demo-p1-{}-{name}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_dsec = args.iter().any(|a| a == "--dsec");
    let use_real = args.iter().any(|a| a == "--real");

    // ---- the model endpoint ----
    let model: Arc<dyn ModelClient> = if use_real {
        println!("== real endpoint mode (DSEC_MODEL_URL) ==");
        Arc::new(OpenAiCompatClient::from_env().expect("set DSEC_MODEL_URL"))
    } else {
        let final_text = if use_dsec {
            "DONE".to_string()
        } else {
            format!("The record value is {SECRET}.")
        };
        let responder = Arc::new(mockserver::TaskSolverResponder::new(
            if use_dsec {
                "write_file"
            } else {
                "notes.get_record"
            },
            if use_dsec {
                json!({"path": "/tmp/out.txt", "content": "SECRET-42"})
            } else {
                json!({"rec_id": "R-1"})
            },
            final_text,
        ));
        let (addr, _h) =
            mockserver::serve(std::net::SocketAddr::from(([127, 0, 0, 1], 0)), responder).await;
        println!("== mock model server at {addr} (real HTTP) ==");
        Arc::new(OpenAiCompatClient::new(&format!("http://{addr}"), "mock").expect("client"))
    };

    // ---- the sandbox backend + the worker container ----
    let journal_path = journal_path(if use_dsec { "dsec" } else { "agentenv" });
    let journal = Journal::create(&journal_path).unwrap();

    let outcome = if use_dsec {
        run_dsec_backend(model, journal).await
    } else {
        run_agentenv_backend(model, journal).await
    }
    .expect("task run");

    match outcome {
        TaskOutcome::Completed { reward, rollout } => {
            println!();
            println!("task completed: score={:?}", reward.score());
            println!("  tool turns : {}", rollout.tool_turns);
            println!("  tokens     : {}", rollout.tokens);
            println!("  final      : {}", rollout.final_message);
        }
        TaskOutcome::Failed { error } => {
            println!("task FAILED: {error}");
        }
    }
    println!("journal: {}", journal_path.display());
    println!("== phase 1 demo complete ==");
}

async fn run_agentenv_backend(
    model: Arc<dyn ModelClient>,
    journal: Journal,
) -> dsec_orchestrator::Result<TaskOutcome> {
    let bundle = test_spec().compile()?.boot()?;
    let env = AgentEnvBackend::new(bundle);
    let mut worker = WorkerContainer::new(
        "demo-agentenv".into(),
        Box::new(env),
        model,
        journal,
        WorkerConfig::default(),
    )?;
    worker.run_task().await
}

async fn run_dsec_backend(
    model: Arc<dyn ModelClient>,
    journal: Journal,
) -> dsec_orchestrator::Result<TaskOutcome> {
    use dsec_control::model::{Resources, SandboxSpec};
    use dsec_sdk::integration::{spawn_local_cluster, ClusterConfig, DataPlane};

    let tmp = tempfile::tempdir()?;
    let cluster = spawn_local_cluster(ClusterConfig {
        node_count: 1,
        data_plane: DataPlane::Uds {
            root: tmp.path().join("uds"),
        },
        ..Default::default()
    })
    .await;
    let task = FileTask::write_file_task("demo-file", "/tmp/out.txt", "SECRET-42");
    let spec = SandboxSpec {
        project: "root".into(),
        resources: Resources {
            cpu_millicores: 500,
            mem_mib: 128,
        },
        ..Default::default()
    };
    let env = DsecSandboxBackend::provision(task, &cluster, spec).await?;
    let mut env: Box<dyn TaskEnvironment> = Box::new(env);
    let r = env.pause().await; // demo pause/resume on the real node
    let _ = r;
    env.resume().await?;
    let mut worker = WorkerContainer::new(
        "demo-dsec-file".into(),
        env,
        model,
        journal,
        WorkerConfig::default(),
    )?;
    worker.run_task().await
}
