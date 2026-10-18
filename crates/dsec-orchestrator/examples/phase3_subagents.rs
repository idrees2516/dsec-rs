//! Phase 3 demo: branch-safe subagents.
//!
//! Forks a subagent on an isolated workspace; the subagent updates a
//! row; its patch is validated (three-way merge) and applied to the
//! parent. Then a second subagent + a parent-side edit to the same row
//! demonstrate conflict detection.
//!
//! Run: `cargo run --release --example phase3_subagents --features server -p dsec-orchestrator`

use std::sync::Arc;

use serde_json::json;

use dsec_agentenv::agentloop::ToolCallReq;
use dsec_agentenv::envgen::{
    CheckSpec, ColumnSpec, DomainSpec, EnvSpec, RowSpec, RubricSpec, SystemSpec, TableSpec, ToolOp,
    ToolSpec,
};
use dsec_orchestrator::model::{ChatResponse, ScriptedModelClient, ToolCallOut};
use dsec_orchestrator::subagent::{MergeOutcome, SubagentManager, SubagentSpec};
use dsec_orchestrator::taskenv::{AgentEnvBackend, TaskEnvironment};
use dsec_orchestrator::worker::WorkerConfig;

fn spec() -> EnvSpec {
    EnvSpec {
        domain: DomainSpec::General,
        instance_id: "p3-demo".into(),
        instruction: "Manage record R-1.".into(),
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
                values: json!({"rec_id": "R-1", "value": "initial"})
                    .as_object()
                    .unwrap()
                    .clone(),
            }],
        }],
        tools: vec![ToolSpec {
            server: "notes".into(),
            name: "update_record".into(),
            description: "Update one record by id.".into(),
            op: ToolOp::Update {
                table: "records".into(),
                id_param: "rec_id".into(),
                field_params: vec!["value".into()],
            },
        }],
        workspace: vec![],
        rubric: vec![RubricSpec {
            id: "answer".into(),
            weight: 1.0,
            method: "rule".into(),
            question: String::new(),
            pass_anchor: String::new(),
            check: Some(CheckSpec::AllText {
                needles: vec!["done".into()],
            }),
            gate: false,
        }],
        protect_sources: false,
    }
}

fn updater(value: &str) -> ScriptedModelClient {
    ScriptedModelClient::new(vec![
        ChatResponse {
            text: String::new(),
            tool_calls: vec![ToolCallOut {
                id: "call_1".into(),
                name: "notes.update_record".into(),
                arguments: json!({"rec_id": "R-1", "value": value}).to_string(),
            }],
            finish_reason: "tool_calls".into(),
            usage_tokens: 8,
            served_model: "scripted".into(),
        },
        ChatResponse {
            text: "done".into(),
            tool_calls: vec![],
            finish_reason: "stop".into(),
            usage_tokens: 6,
            served_model: "scripted".into(),
        },
    ])
}

async fn row(env: &mut dyn TaskEnvironment) -> String {
    let state = env.capture_state().await.unwrap();
    state
        .rows
        .get(&(
            "notes".to_string(),
            "records".to_string(),
            "R-1".to_string(),
        ))
        .map(|r| r["value"].as_str().unwrap_or("?").to_string())
        .unwrap_or_else(|| "<missing>".into())
}

#[tokio::main]
async fn main() {
    println!("== phase 3 demo: branch-safe subagents ==");
    let dir = std::env::temp_dir().join(format!("dsec-demo-p3-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut parent: Box<dyn TaskEnvironment> = Box::new(AgentEnvBackend::new(
        spec().compile().unwrap().boot().unwrap(),
    ));
    println!("parent row before: {}", row(parent.as_mut()).await);

    // ---- a clean branch ----
    let mut manager = SubagentManager::new(&dir, "parent", Arc::new(updater("subagent-write")), 4);
    manager
        .spawn(
            parent.as_mut(),
            SubagentSpec::new("branch-1", "Update R-1 to 'subagent-write'.", 5),
            WorkerConfig::default(),
        )
        .await
        .unwrap();
    // isolation: parent unchanged while the branch runs
    println!(
        "parent row while branch runs (isolated): {}",
        row(parent.as_mut()).await
    );
    let outcome = manager.join("branch-1", parent.as_mut()).await.unwrap();
    println!("merge outcome 1: {outcome:?}");
    println!("parent row after merge: {}", row(parent.as_mut()).await);

    // ---- a conflicting branch: parent edits the same row post-fork ----
    let mut manager2 = SubagentManager::new(&dir, "parent", Arc::new(updater("loser-write")), 4);
    manager2
        .spawn(
            parent.as_mut(),
            SubagentSpec::new("branch-2", "Update R-1 again.", 5),
            WorkerConfig::default(),
        )
        .await
        .unwrap();
    // the parent concurrently modifies the same row
    parent
        .call_tool(
            &ToolCallReq {
                id: "parent-edit".into(),
                server: "notes".into(),
                tool: "update_record".into(),
                params: json!({"rec_id": "R-1", "value": "parent-write"}),
            },
            "parent:tool:0:parent-edit",
        )
        .await
        .unwrap();
    let outcome2 = manager2.join("branch-2", parent.as_mut()).await.unwrap();
    println!("merge outcome 2 (conflict expected): {outcome2:?}");
    assert!(matches!(outcome2, MergeOutcome::Conflict { .. }));
    println!(
        "parent row after conflict (unchanged): {}",
        row(parent.as_mut()).await
    );
    println!("branch journals: {}", dir.display());
    println!("== phase 3 demo complete ==");
}
