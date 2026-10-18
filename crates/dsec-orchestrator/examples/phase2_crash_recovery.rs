//! Phase 2 demo: crash → recovery, proving no duplicate side effects.
//!
//! Runs a task with the crashpoint `AfterToolExecBeforePersist` armed:
//! the tool executes, the worker "dies" before journaling the outcome.
//! The environment survives (the sandbox outlives the worker), a new
//! worker recovers from the journal, and the re-issued tool call is
//! answered from the sandbox's idempotency ledger — the side effect
//! happens exactly once.
//!
//! Run: `cargo run --release --example phase2_crash_recovery --features server -p dsec-orchestrator`

use std::collections::BTreeSet;
use std::sync::Arc;

use serde_json::json;

use dsec_agentenv::envgen::{
    CheckSpec, ColumnSpec, DomainSpec, EnvSpec, RowSpec, RubricSpec, SystemSpec, TableSpec, ToolOp,
    ToolSpec,
};
use dsec_orchestrator::model::{ChatResponse, ScriptedModelClient, ToolCallOut};
use dsec_orchestrator::taskenv::AgentEnvBackend;
use dsec_orchestrator::worker::{Crashpoint, TaskOutcome, WorkerConfig, WorkerContainer};
use dsec_orchestrator::{Error, Journal};

const SECRET: &str = "KEY-7F3A-92Z";

fn spec() -> EnvSpec {
    EnvSpec {
        domain: DomainSpec::General,
        instance_id: "p2-demo".into(),
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

fn model() -> ScriptedModelClient {
    ScriptedModelClient::new(vec![
        ChatResponse {
            text: String::new(),
            tool_calls: vec![ToolCallOut {
                id: "call_1".into(),
                name: "notes.get_record".into(),
                arguments: r#"{"rec_id":"R-1"}"#.into(),
            }],
            finish_reason: "tool_calls".into(),
            usage_tokens: 8,
            served_model: "scripted".into(),
        },
        ChatResponse {
            text: format!("The record value is {SECRET}."),
            tool_calls: vec![],
            finish_reason: "stop".into(),
            usage_tokens: 12,
            served_model: "scripted".into(),
        },
    ])
}

#[tokio::main]
async fn main() {
    let journal_path =
        std::env::temp_dir().join(format!("dsec-demo-p2-{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&journal_path);

    // one model client shared by both workers — its script is
    // exhausted after the crash, so recovery continues to the final
    // turn instead of issuing a NEW tool call
    let model = Arc::new(model());

    // worker 1: crashes right after the tool executed, before the
    // outcome was journaled
    let journal = Journal::create(&journal_path).unwrap();
    let mut worker = WorkerContainer::new(
        "demo-p2".into(),
        Box::new(AgentEnvBackend::new(
            spec().compile().unwrap().boot().unwrap(),
        )),
        model.clone(),
        journal,
        WorkerConfig::default(),
    )
    .unwrap()
    .with_crashpoints(&[Crashpoint::AfterToolExecBeforePersist]);

    println!("== phase 2 demo: crash after tool execution ==");
    match worker.run_task().await {
        Err(Error::CrashInjected { point }) => {
            println!("worker crashed at {point:?} (side effect happened, outcome not journaled)");
        }
        other => panic!("expected crash, got {other:?}"),
    }
    let env = worker.into_env().unwrap();
    let exec_after_crash = env.real_exec_count();
    println!("side effects before crash : {exec_after_crash}");

    // worker 2: recovers from the journal + the SURVIVING sandbox
    let recovered = WorkerContainer::recover(
        "demo-p2".into(),
        env,
        model,
        journal_path.clone(),
        WorkerConfig::default(),
        BTreeSet::new(),
    )
    .await
    .unwrap();
    let mut recovered = recovered;
    let outcome = recovered.run_task().await.unwrap();

    let env = recovered.into_env().unwrap();
    println!(
        "side effects after recover: {} (must equal the above!)",
        env.real_exec_count()
    );
    match outcome {
        TaskOutcome::Completed { reward, rollout } => {
            println!("recovered task completed: score={:?}", reward.score());
            println!("  transcript length: {}", rollout.messages.len());
        }
        TaskOutcome::Failed { error } => println!("recovered task failed: {error}"),
    }
    let report = dsec_orchestrator::journal::replay(&journal_path).unwrap();
    println!(
        "journal: {} events, chain intact={}, recovered-marker={}",
        report.events.len(),
        report.tampered_at.is_none(),
        report.events.iter().any(|e| e.kind == "recovered")
    );
    println!("journal file: {}", journal_path.display());
    println!("== phase 2 demo complete ==");
}
