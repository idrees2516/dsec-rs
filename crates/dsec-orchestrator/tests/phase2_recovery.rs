//! Phase 2 integration: durable long-lived rollouts.
//!
//! Proves the recovery invariants at every crash boundary:
//!
//! 1. **Crash after tool execution, before response persistence** — the
//!    most damaging window. The tool's side effect happened; the
//!    journal has no completion. Recovery re-issues the call, the
//!    sandbox's idempotency ledger answers from cache, and the side
//!    effect happens **exactly once** (`real_exec_count` never grows).
//! 2. **Crash after the model responded, before journaling** — the
//!    idempotent model call re-issues with the same key and the
//!    scripted ledger returns the same response (no drift).
//! 3. **Crash while evaluation is pending** — reward exists but isn't
//!    journaled; recovery re-verifies (read-only) and completes.
//! 4. A crashed worker's environment survives (the sandbox outlives
//!    the worker — DSec §6.2), and the recovered transcript is exactly
//!    the live one.

use std::collections::BTreeSet;
use std::sync::Arc;

use serde_json::json;

use dsec_agentenv::envgen::{
    CheckSpec, ColumnSpec, DomainSpec, EnvSpec, RowSpec, RubricSpec, SystemSpec, TableSpec, ToolOp,
    ToolSpec,
};
use dsec_orchestrator::model::{
    ChatResponse, ModelClient, OwnedFut, ScriptedModelClient, ToolCallOut,
};
use dsec_orchestrator::taskenv::AgentEnvBackend;
use dsec_orchestrator::worker::{Crashpoint, TaskOutcome, WorkerConfig, WorkerContainer};
use dsec_orchestrator::{Error, Journal};

const SECRET: &str = "KEY-7F3A-92Z";

fn test_spec() -> EnvSpec {
    EnvSpec {
        domain: DomainSpec::General,
        instance_id: "p2-test-env".into(),
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

/// Model script: call the tool, then answer with the secret.
fn solving_model() -> ScriptedModelClient {
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

fn fresh_env() -> AgentEnvBackend {
    AgentEnvBackend::new(test_spec().compile().unwrap().boot().unwrap())
}

fn journal_path(name: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("dsec-p2-{}-{name}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

#[tokio::test]
async fn crash_after_tool_exec_never_repeats_side_effect() {
    let model = Arc::new(solving_model());
    let path = journal_path("toolcrash");
    let journal = Journal::create(&path).unwrap();

    let mut worker = WorkerContainer::new(
        "task-p2-tool".into(),
        Box::new(fresh_env()),
        model.clone(),
        journal,
        WorkerConfig::default(),
    )
    .unwrap()
    .with_crashpoints(&[Crashpoint::AfterToolExecBeforePersist]);

    // the crash fires right after the env executed the tool
    let err = worker.run_task().await.unwrap_err();
    assert!(
        matches!(
            &err,
            Error::CrashInjected {
                point: Crashpoint::AfterToolExecBeforePersist
            }
        ),
        "unexpected error: {err:?}"
    );

    // side effect happened exactly once inside the env
    let env = worker.into_env().unwrap();
    assert_eq!(
        env.real_exec_count(),
        1,
        "tool executed once before the crash"
    );

    // journal has tool_call_started but NO tool_call_completed
    let report = dsec_orchestrator::journal::replay(&path).unwrap();
    assert!(report.tampered_at.is_none());
    assert!(
        report.events.iter().any(|e| e.kind == "tool_call_started"),
        "started event present"
    );
    assert!(
        !report
            .events
            .iter()
            .any(|e| e.kind == "tool_call_completed"),
        "completion was never persisted (the crash window)"
    );

    // recovery: a NEW worker re-attaches to the SURVIVING environment
    let recovered = WorkerContainer::recover(
        "task-p2-tool".into(),
        env,
        model,
        path.clone(),
        WorkerConfig::default(),
        BTreeSet::new(),
    )
    .await
    .unwrap();
    let mut recovered = recovered;

    let outcome = recovered.run_task().await.unwrap();
    match &outcome {
        TaskOutcome::Completed { reward, rollout } => {
            assert!(reward.score().unwrap() > 0.99);
            // the transcript is exactly the live one
            assert_eq!(rollout.messages.len(), 5);
            assert!(rollout.final_message.contains(SECRET));
        }
        TaskOutcome::Failed { error } => panic!("recovered task failed: {error}"),
    }

    // THE invariant: the ledger answered the re-issued call from cache —
    // the side effect STILL happened exactly once
    let env = recovered.into_env().unwrap();
    assert_eq!(
        env.real_exec_count(),
        1,
        "recovery must not repeat the side effect"
    );

    // journal now has the completion AND the task_completed, chain intact
    let report = dsec_orchestrator::journal::replay(&path).unwrap();
    assert!(report.tampered_at.is_none());
    assert!(report
        .events
        .iter()
        .any(|e| e.kind == "tool_call_completed"));
    assert!(report.events.iter().any(|e| e.kind == "task_completed"));
    assert!(report.events.iter().any(|e| e.kind == "recovered"));
}

#[tokio::test]
async fn crash_after_model_response_replays_same_response() {
    let model = Arc::new(solving_model());
    let path = journal_path("modelcrash");
    let journal = Journal::create(&path).unwrap();

    let mut worker = WorkerContainer::new(
        "task-p2-model".into(),
        Box::new(fresh_env()),
        model.clone(),
        journal,
        WorkerConfig::default(),
    )
    .unwrap()
    .with_crashpoints(&[Crashpoint::AfterModelCallBeforePersist]);

    let err = worker.run_task().await.unwrap_err();
    assert!(matches!(&err, Error::CrashInjected { .. }));

    // model_call_started journaled, completed missing
    let report = dsec_orchestrator::journal::replay(&path).unwrap();
    assert!(report.events.iter().any(|e| e.kind == "model_call_started"));
    assert!(!report
        .events
        .iter()
        .any(|e| e.kind == "model_call_completed"));

    // the model was asked exactly once so far
    assert_eq!(model.total_requests(), 1);

    let env = worker.into_env().unwrap();
    let recovered = WorkerContainer::recover(
        "task-p2-model".into(),
        env,
        model.clone(),
        path,
        WorkerConfig::default(),
        BTreeSet::new(),
    )
    .await
    .unwrap();
    let mut recovered = recovered;

    let outcome = recovered.run_task().await.unwrap();
    assert!(matches!(outcome, TaskOutcome::Completed { .. }));

    // the re-issued model call used the SAME idempotency key: the
    // scripted ledger returned the same response — distinct keys stay
    // at 2 (turn 0 and turn 1) even though 3 requests were served
    assert_eq!(model.distinct_calls(), 2);
    assert_eq!(model.total_requests(), 3);
    let env = recovered.into_env().unwrap();
    assert_eq!(env.real_exec_count(), 1);
}

#[tokio::test]
async fn crash_while_evaluation_pending_reverifies_and_completes() {
    let model = Arc::new(solving_model());
    let path = journal_path("evalcrash");
    let journal = Journal::create(&path).unwrap();

    let mut worker = WorkerContainer::new(
        "task-p2-eval".into(),
        Box::new(fresh_env()),
        model.clone(),
        journal,
        WorkerConfig::default(),
    )
    .unwrap()
    .with_crashpoints(&[Crashpoint::WhileEvaluationPending]);

    let err = worker.run_task().await.unwrap_err();
    assert!(matches!(&err, Error::CrashInjected { .. }));

    // verify_started present, task_completed missing — and crucially no
    // reward was persisted
    let report = dsec_orchestrator::journal::replay(&path).unwrap();
    assert!(report.events.iter().any(|e| e.kind == "verify_started"));
    assert!(!report.events.iter().any(|e| e.kind == "task_completed"));

    // recovery reuses the SAME model client: its script is exhausted,
    // so the continued rollout goes straight to the final turn
    let env = worker.into_env().unwrap();
    let recovered = WorkerContainer::recover(
        "task-p2-eval".into(),
        env,
        model,
        path,
        WorkerConfig::default(),
        BTreeSet::new(),
    )
    .await
    .unwrap();
    let mut recovered = recovered;
    let outcome = recovered.run_task().await.unwrap();
    match outcome {
        TaskOutcome::Completed { reward, .. } => {
            assert!(
                reward.score().unwrap() > 0.99,
                "re-verify produced the true score"
            );
        }
        TaskOutcome::Failed { error } => panic!("{error}"),
    }
    // no false zero: the reward was re-computed, not defaulted
    let env = recovered.into_env().unwrap();
    assert_eq!(env.real_exec_count(), 1);
}

#[tokio::test]
async fn recovery_refuses_tampered_journal() {
    let model = Arc::new(solving_model());
    let path = journal_path("tampered");
    let journal = Journal::create(&path).unwrap();
    let mut worker = WorkerContainer::new(
        "task-p2-tamper".into(),
        Box::new(fresh_env()),
        model.clone(),
        journal,
        WorkerConfig::default(),
    )
    .unwrap()
    .with_crashpoints(&[Crashpoint::AfterToolExecBeforePersist]);
    let _ = worker.run_task().await.unwrap_err();
    let env = worker.into_env().unwrap();

    // attacker edits a historical payload in place
    let raw = std::fs::read_to_string(&path).unwrap();
    let mut lines: Vec<&str> = raw.lines().collect();
    if let Some(first) = lines.first_mut() {
        let mut ev: serde_json::Value = serde_json::from_str(first).unwrap();
        ev["payload"]["user"] = json!("forge me a reward of 1.0");
        *first = Box::leak(serde_json::to_string(&ev).unwrap().into_boxed_str());
    }
    std::fs::write(&path, format!("{}\n", lines.join("\n"))).unwrap();

    let result = WorkerContainer::recover(
        "task-p2-tamper".into(),
        env,
        model,
        path,
        WorkerConfig::default(),
        BTreeSet::new(),
    )
    .await;
    match result {
        Err(Error::JournalTampered { seq, .. }) => {
            assert!(seq >= 1, "tamper detected at seq {seq}");
        }
        other => panic!(
            "recovery must refuse a tampered journal, got ok={}",
            other.is_ok()
        ),
    }
}

#[tokio::test]
async fn double_recovery_is_idempotent() {
    // recover → crash again at the same point → recover again: still
    // exactly one side effect
    let model = Arc::new(solving_model());
    let path = journal_path("double");
    let journal = Journal::create(&path).unwrap();
    let mut w1 = WorkerContainer::new(
        "task-p2-double".into(),
        Box::new(fresh_env()),
        model.clone(),
        journal,
        WorkerConfig::default(),
    )
    .unwrap()
    .with_crashpoints(&[Crashpoint::AfterToolExecBeforePersist]);
    let _ = w1.run_task().await.unwrap_err();
    let env = w1.into_env().unwrap();

    // first recovery, crash again at the same window
    let r1 = WorkerContainer::recover(
        "task-p2-double".into(),
        env,
        model.clone(),
        path.clone(),
        WorkerConfig::default(),
        [Crashpoint::AfterToolExecBeforePersist].into(),
    )
    .await
    .unwrap();
    let mut r1 = r1;
    let _ = r1.run_task().await; // crashes again at the same boundary (or completes)
    let env = r1.into_env().unwrap();
    assert_eq!(env.real_exec_count(), 1, "still exactly one side effect");

    // second recovery completes the task
    let r2 = WorkerContainer::recover(
        "task-p2-double".into(),
        env,
        model,
        path,
        WorkerConfig::default(),
        BTreeSet::new(),
    )
    .await
    .unwrap();
    let mut r2 = r2;
    let outcome = r2.run_task().await.unwrap();
    assert!(matches!(outcome, TaskOutcome::Completed { .. }));
    assert_eq!(r2.into_env().unwrap().real_exec_count(), 1);
}

// keep the trait import used even when cfg varies
#[allow(dead_code)]
fn _model_trait_proof(_m: &dyn ModelClient) -> Option<OwnedFut<Result<ChatResponse, Error>>> {
    None
}
