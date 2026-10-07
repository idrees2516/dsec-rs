//! End-to-end integration: the Live RL pipeline over generated
//! environment variants.
//!
//! Covers the full story this crate exists for:
//!
//! 1. **envgen** — one knowledge-work spec + seeded variants become a
//!    task family;
//! 2. **task** — the family emits verl-format rows;
//! 3. **live** — a `LiveTrainer` step runs asynchronous group rollouts
//!    (scripted policies standing in for the model), computes rewards
//!    through the full masking contract, applies GRPO + GAR;
//! 4. **repo2rl** — a mined repository task round-trips through the same
//!    row schema with its test-driven verifier;
//! 5. **anti-hacking** — GRS synthesizes a rubric from contrasting
//!    rollouts; screening drops a zero-tool full score.

#![cfg(feature = "server")] // the trainer path needs nothing, but the
                            // final server smoke does; keep the whole suite behind one feature run.

use dsec_agentenv::agentloop::ScriptedPolicy;
use dsec_agentenv::envgen::{templates, EnvSpec, SpecProvider, Variants};
use dsec_agentenv::live::{grs, self_correction_pairs, LiveConfig, LiveTrainer};
use dsec_agentenv::repo2rl::{CommitRecord, FileDiff, MiningConfig, RepoMiner};
use dsec_agentenv::state::PostState;
use dsec_agentenv::task::TaskRow;
use dsec_agentenv::verifier::{AnchorJudge, RubricEngine, VerifyToggles};
use std::collections::BTreeMap;
use std::sync::Arc;

fn good_answer(spec: &EnvSpec) -> String {
    // derive a passing answer from the spec's rubric: judge anchors + rule
    // needles, all mentioned
    let mut text = String::from("Report: ");
    for r in &spec.rubric {
        if let Some(dsec_agentenv::envgen::CheckSpec::AllText { needles }) = &r.check {
            for n in needles {
                text.push_str(n);
                text.push(' ');
            }
        }
        if let Some(dsec_agentenv::envgen::CheckSpec::AnyText { needles, .. }) = &r.check {
            for n in needles {
                text.push_str(n);
                text.push(' ');
            }
        }
        if !r.pass_anchor.is_empty() {
            // quoted spans of the anchor are the required tokens
            for tok in dsec_agentenv::verifier::extract_anchor_tokens(&r.pass_anchor) {
                text.push_str(&tok);
                text.push(' ');
            }
        }
    }
    text
}

fn factory_for(spec: &EnvSpec) -> dsec_agentenv::live::PolicyFactory {
    // 60% of members solve the task with one tool call; the rest answer
    // wrongly (and one member never finishes — exhausted script).
    let answer = good_answer(spec);
    let list_tool = spec
        .tools
        .first()
        .map(|t| (format!("{}.{}", t.server, t.name), serde_json::json!({})))
        .unwrap_or_else(|| (String::new(), serde_json::json!({})));
    let wrong = "cannot determine the answer from the available records".to_string();
    Arc::new(move |seed| match seed % 10 {
        0 => Box::new(ScriptedPolicy::new(vec![])), // exhausted script
        1 | 3 => Box::new(ScriptedPolicy::new(vec![(wrong.clone(), vec![])])),
        // a lucky full-score guess with zero tool calls — the
        // adversarial screen flags exactly this pattern
        2 => Box::new(ScriptedPolicy::new(vec![(answer.clone(), vec![])])),
        _ => Box::new(ScriptedPolicy::new(vec![
            ("checking the systems".to_string(), vec![list_tool.clone()]),
            (answer.clone(), vec![]),
        ])),
    })
}

#[tokio::test]
async fn live_pipeline_over_generated_variants() {
    // 1. one base spec -> 3 variants
    let base = templates::knowledge_work();
    let specs = Variants::generate(&base, 3, 7);
    assert_eq!(specs.len(), 3);

    // 2. emit rows
    let rows: Vec<TaskRow> = specs
        .iter()
        .enumerate()
        .map(|(i, s)| s.to_task_row(i as u64))
        .collect();
    assert!(rows
        .iter()
        .all(|r| r.data_source == "mimoagent/terminal_bench"));

    // 3. trainer step: 3 prompts x 8 members, async under concurrency 4
    let provider = Arc::new(SpecProvider::new(specs.clone()));
    let cfg = LiveConfig {
        group_size: 8,
        concurrency: 4,
        ..Default::default()
    };
    let factory = factory_for(&specs[0]);
    let trainer = LiveTrainer::new(cfg, provider, factory, Arc::new(AnchorJudge::new()));
    let (records, stats) = trainer.step(&rows).await.unwrap();

    assert_eq!(stats.prompts, 3);
    assert_eq!(stats.rollouts, 24);
    // the seeded rollouts never mask (the testbed is healthy)
    assert_eq!(stats.masked, 0);
    // some members fail (wrong answers) and some are screened (the
    // zero-tool lucky guesses) — a legit partial group
    assert!(
        stats.screened > 0,
        "the lucky-guess members must be screened"
    );
    assert!(stats.trainable > 0 && stats.trainable < 24);
    assert!(stats.mean_reward > 0.0 && stats.mean_reward < 1.0);
    // GRPO advantages assigned to every trainable record
    assert!(records
        .iter()
        .filter(|r| r.trainable())
        .all(|r| r.advantage.is_some()));

    // 4. GRS: contrasting rollouts within a group -> synthesized rubric
    let items = grs(&records, 8, &BTreeMap::new(), 4);
    // the knowledge-work groups contain pass/fail contrast, so at least
    // one synthesized item exists per contrasting group
    assert!(!items.is_empty());
    for it in &items {
        assert!(it.method == "rule");
    }

    // 5. self-correction pairs: failures paired with the best pass
    let pairs = self_correction_pairs(&records, 8);
    assert!(!pairs.is_empty());
    assert!(pairs.iter().all(|(wrong, right)| wrong != right));
}

#[tokio::test]
async fn masking_contract_drops_broken_testbed_from_training() {
    let specs = Variants::generate(&templates::knowledge_work(), 2, 11);
    let rows: Vec<TaskRow> = specs
        .iter()
        .enumerate()
        .map(|(i, s)| s.to_task_row(i as u64))
        .collect();
    let provider = Arc::new(SpecProvider::new(specs.clone()));
    let cfg = LiveConfig {
        group_size: 4,
        concurrency: 2,
        ..Default::default()
    };
    // the judge is DOWN: every reward phase breaks
    let factory = factory_for(&specs[0]);
    let trainer = LiveTrainer::new(
        cfg,
        provider,
        factory,
        Arc::new(dsec_agentenv::verifier::UnavailableJudge),
    );
    let (records, stats) = trainer.step(&rows).await.unwrap();
    assert_eq!(stats.masked, 8);
    assert_eq!(stats.trainable, 0);
    assert!(records.iter().all(|r| r.reward.masked()));
    assert!(records.iter().all(|r| r.advantage.is_none()));
}

#[test]
fn repo2rl_pipeline_emits_trainable_rows() {
    let commits = vec![CommitRecord {
        sha: "feed1234".into(),
        author: "alice".into(),
        message: "fix: decode loop drops final chunk".into(),
        is_merge: false,
        files: vec![
            FileDiff {
                path: "src/codec/decoder.rs".into(),
                added: vec!["while let Some(chunk) = stream.next_chunk().await".into()],
                removed: vec![],
                is_test: false,
            },
            FileDiff {
                path: "src/codec/tests/decode_test.rs".into(),
                added: vec![],
                removed: vec![],
                is_test: true,
            },
        ],
    }];
    let mut patches = BTreeMap::new();
    patches.insert(
        "feed1234".to_string(),
        vec!["test_decoder_flushes_final_chunk".to_string()],
    );
    let miner = RepoMiner::new(MiningConfig::default());
    let tasks = miner.mine("codec", &commits, &patches);
    assert_eq!(tasks.len(), 1);
    let task = &tasks[0];
    assert!(task.problem_statement.contains("final chunk"));
    // row schema
    let row = &task.row;
    assert_eq!(row.data_source, "repo:codec");
    assert_eq!(row.instance().unwrap().cwd, "/testbed");
    // test-driven verifier: per-case items + the golden-patch gate
    let rubric = task.verifier();
    assert!(rubric.items.len() >= 2);
    // the engine grades a passing answer
    let post = PostState::default();
    let untouched = BTreeMap::from([("codec".to_string(), true)]);
    let ctx = dsec_agentenv::verifier::EvalCtx {
        agent_output: "fixed: while let Some(chunk) = stream.next_chunk().await; \
                       test_decoder_flushes_final_chunk passes",
        post_state: &post,
        source_conserved: true,
        state_untouched: &untouched,
    };
    let out = RubricEngine::new(&rubric, &AnchorJudge::new(), VerifyToggles::default()).run(&ctx);
    assert_eq!(out.score, Some(1.0));
}

#[tokio::test]
async fn agentenv_server_smoke() {
    use dsec_agentenv::server::EnvServer;
    use dsec_agentenv::verifier::RewardConfig;

    let srv = Arc::new(EnvServer::new(
        Arc::new(AnchorJudge::new()),
        RewardConfig::default(),
    ));
    let router = srv.router();
    use tower::ServiceExt;

    // create an env from the terminal template
    let spec = templates::terminal();
    let body = serde_json::json!({"spec": spec}).to_string();
    let resp = router
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/envs")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let id = v["id"].as_u64().unwrap();

    // call the fs tool
    let call = serde_json::json!({
        "server": "fs", "tool": "read_file", "params": {"name": "key.txt"}
    });
    let resp = router
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri(format!("/envs/{id}/tools/call"))
                .header("content-type", "application/json")
                .body(axum::body::Body::from(call.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v["result"].to_string().contains("KEY-7F3A-92Z"));

    // verify
    let verify = serde_json::json!({"final_message": "the key is KEY-7F3A-92Z"});
    let resp = router
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri(format!("/envs/{id}/verify"))
                .header("content-type", "application/json")
                .body(axum::body::Body::from(verify.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["reward"].as_f64(), Some(1.0));
}
