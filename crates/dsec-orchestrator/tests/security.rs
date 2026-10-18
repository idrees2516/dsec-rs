//! Phase 7 (orchestrator half): attacker-style tests.
//!
//! The attacker model: an adversary who can rewrite journal files,
//! forge completion events, replay tool-call keys, and try to redirect
//! the model client's egress. Every test asserts the attack is
//! *detected* or *neutralized*, not merely discouraged.

use std::sync::Arc;

use serde_json::json;

use dsec_agentenv::agentloop::ToolCallReq;
use dsec_orchestrator::model::{ModelClient, NetPolicy, OpenAiCompatClient, ScriptedModelClient};
use dsec_orchestrator::taskenv::{AgentEnvBackend, TaskEnvironment};
use dsec_orchestrator::worker::{Crashpoint, WorkerConfig, WorkerContainer};
use dsec_orchestrator::Journal;

fn minimal_env() -> AgentEnvBackend {
    use dsec_agentenv::envgen::*;
    let spec = EnvSpec {
        domain: DomainSpec::General,
        instance_id: "sec-test".into(),
        instruction: "noop".into(),
        systems: vec![],
        tools: vec![],
        workspace: vec![],
        rubric: vec![],
        protect_sources: false,
    };
    AgentEnvBackend::new(spec.compile().unwrap().boot().unwrap())
}

fn journal_path(name: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("dsec-sec-{}-{name}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

#[tokio::test]
async fn attacker_forging_task_completed_is_detected() {
    // run a task that crashes before its reward is journaled, then try
    // to append a forged "task_completed" with score 1.0
    let model = Arc::new(ScriptedModelClient::new(vec![]));
    let path = journal_path("forge");
    let journal = Journal::create(&path).unwrap();
    let mut worker = WorkerContainer::new(
        "sec-forge".into(),
        Box::new(minimal_env()),
        model.clone(),
        journal,
        WorkerConfig::default(),
    )
    .unwrap()
    .with_crashpoints(&[Crashpoint::AfterModelCallBeforePersist]);
    let _ = worker.run_task().await; // crash fires
    let env = worker.into_env().unwrap();

    // attacker appends a forged event (valid JSON, wrong hash — the
    // chain hash cannot be computed without replaying the whole prefix
    // honestly, and the seq is off)
    let forged = json!({
        "seq": 99,
        "task_id": "sec-forge",
        "ts_ms": 0,
        "kind": "task_completed",
        "payload": {"score": 1.0, "masked": false},
        "prev_hash": "deadbeef",
        "hash": "deadbeef",
    });
    let mut raw = std::fs::read_to_string(&path).unwrap();
    raw.push_str(&serde_json::to_string(&forged).unwrap());
    raw.push('\n');
    std::fs::write(&path, raw).unwrap();

    let result = WorkerContainer::recover(
        "sec-forge".into(),
        env,
        model,
        path,
        WorkerConfig::default(),
        Default::default(),
    )
    .await;
    match result {
        Err(dsec_orchestrator::Error::JournalTampered { .. }) => {}
        other => panic!(
            "forged completion must be detected, got ok={}",
            other.is_ok()
        ),
    }
}

#[tokio::test]
async fn tool_call_replay_attack_executes_once() {
    // an attacker replays the same tool-call idempotency key: the env
    // ledger must answer from cache — the side effect happens once
    let mut env: Box<dyn TaskEnvironment> = Box::new(minimal_env_with_tool());

    let call = ToolCallReq {
        id: "c1".into(),
        server: "notes".into(),
        tool: "update_counter".into(),
        params: json!({"id": "main", "n": 7}),
    };
    let first = env.call_tool(&call, "victim:tool:0:c1").await.unwrap();
    let replayed = env.call_tool(&call, "victim:tool:0:c1").await.unwrap();
    assert_eq!(first.observation, replayed.observation, "ledger replay");
    assert_eq!(env.real_exec_count(), 1, "replayed key must NOT re-execute");

    // a different key with the same params DOES execute (fresh call)
    let _ = env.call_tool(&call, "victim:tool:1:c1").await.unwrap();
    assert_eq!(env.real_exec_count(), 2);
}

fn minimal_env_with_tool() -> AgentEnvBackend {
    use dsec_agentenv::envgen::*;
    let spec = EnvSpec {
        domain: DomainSpec::General,
        instance_id: "sec-tool-env".into(),
        instruction: "noop".into(),
        systems: vec![SystemSpec {
            name: "notes".into(),
            tables: vec![TableSpec {
                name: "counters".into(),
                primary_key: "id".into(),
                columns: vec![
                    ColumnSpec {
                        name: "id".into(),
                        sql_type: "TEXT".into(),
                    },
                    ColumnSpec {
                        name: "n".into(),
                        sql_type: "INTEGER".into(),
                    },
                ],
            }],
            rows: vec![RowSpec {
                table: "counters".into(),
                values: json!({"id": "main", "n": 0}).as_object().unwrap().clone(),
            }],
        }],
        tools: vec![ToolSpec {
            server: "notes".into(),
            name: "update_counter".into(),
            description: "Update a counter.".into(),
            op: ToolOp::Update {
                table: "counters".into(),
                id_param: "id".into(),
                field_params: vec!["n".into()],
            },
        }],
        workspace: vec![],
        rubric: vec![],
        protect_sources: false,
    };
    AgentEnvBackend::new(spec.compile().unwrap().boot().unwrap())
}

#[tokio::test]
async fn egress_redirection_is_denied_before_any_socket() {
    // the model client is pinned to its endpoint: a tampered policy
    // (or config) pointing elsewhere is denied before connect
    let client = OpenAiCompatClient::new("http://127.0.0.1:9", "m").unwrap();
    // sanity: its own endpoint passes the policy check
    client.net_check_self().unwrap();

    // attacker swaps the policy to allow only another host
    let hijacked = client.clone().with_net_policy(NetPolicy::Allowlist {
        hosts: vec!["evil.example".into()],
        ports: vec![443],
    });
    let req = dsec_orchestrator::model::ChatRequest {
        model: "m".into(),
        messages: vec![],
        tools: vec![],
        temperature: 0.0,
        max_tokens: 1,
        idempotency_key: "k".into(),
    };
    match hijacked.complete(req).await {
        Err(dsec_orchestrator::Error::NetPolicyDenied { host, port, reason }) => {
            assert_eq!(port, 9);
            assert!(host.contains("127.0.0.1"));
            assert!(reason.contains("allowlist"));
        }
        other => panic!("egress must be denied, got {other:?}"),
    }
}

#[tokio::test]
async fn truncation_attack_on_middle_event_detected() {
    // deleting a middle event (e.g. the tool_call_completed that
    // recorded a side effect) breaks the chain links
    let model = Arc::new(ScriptedModelClient::new(vec![]));
    let path = journal_path("trunc");
    let journal = Journal::create(&path).unwrap();
    let worker = WorkerContainer::new(
        "sec-trunc".into(),
        Box::new(minimal_env()),
        model,
        journal,
        WorkerConfig::default(),
    )
    .unwrap();
    drop(worker);
    let mut j = dsec_orchestrator::journal::Journal::open_append(&path).unwrap();
    for i in 0..5 {
        j.append("sec-trunc", "filler", json!({"i": i})).unwrap();
    }
    drop(j);

    // attacker removes the middle line
    let lines: Vec<String> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|l| l.to_string())
        .collect();
    let kept: Vec<String> = lines
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 2)
        .map(|(_, l)| l.to_string())
        .collect();
    std::fs::write(&path, format!("{}\n", kept.join("\n"))).unwrap();

    let report = dsec_orchestrator::journal::replay(&path).unwrap();
    assert!(report.tampered_at.is_some(), "middle truncation detected");
}
