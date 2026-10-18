//! Phase 5 end-to-end: a real trainer connected to a real model server
//! with explicit policy versions and bounded staleness.
//!
//! The loop here is the contract the phase demands:
//! 1. rollouts are generated against the **published** policy over real
//!    HTTP (the orchestrator's OpenAI client → dsec-train's policy
//!    server);
//! 2. every rollout sample is stamped with the policy version that
//!    served it (`served_version` in the response);
//! 3. the trainer steps: staleness-filtered samples → real gradients →
//!    barrier publication of `policy-v{major+1}`;
//! 4. the very next request is served by the NEW version;
//! 5. stale-stamped samples are rejected, never trained.

#![cfg(feature = "server")]

use dsec_orchestrator::model::{ChatRequest, ModelClient};
use dsec_rl::reward::RewardRecord;
use dsec_rl::tito::TranscriptTurn;
use dsec_train::policy::PolicyVersion;
use dsec_train::trainer::{RolloutSample, Trainer, TrainerConfig};

fn sample(exec: u64, group: u64, reward: f32, version: u64) -> RolloutSample {
    RolloutSample {
        execution: dsec_rl::loss::ExecutionInput {
            execution_id: exec,
            group_id: group,
            turns: vec![
                TranscriptTurn::System("sys words".into()),
                TranscriptTurn::User("task words".into()),
                TranscriptTurn::Assistant {
                    text: if reward > 0.5 {
                        "good answer words".into()
                    } else {
                        "bad answer words".into()
                    },
                    tool_calls: vec![],
                },
            ],
            reward: RewardRecord::valid(exec, group, reward),
        },
        policy_version: version,
    }
}

#[tokio::test]
async fn full_training_loop_over_real_http() {
    let mut trainer = Trainer::new(TrainerConfig {
        lr: 0.2,
        hidden: 24,
        ..Default::default()
    });
    assert_eq!(trainer.published().major, 0);

    // the model server serves the PUBLISHED policy over real TCP
    let (addr, _handle) = dsec_train::server::serve(
        trainer.store(),
        std::net::SocketAddr::from(([127, 0, 0, 1], 0)),
        0.2,
        8,
    )
    .await;
    let client =
        dsec_orchestrator::model::OpenAiCompatClient::new(&format!("http://{addr}"), "policy")
            .unwrap();

    // 1. a rollout against v0
    let req = ChatRequest {
        model: "policy".into(),
        messages: vec![dsec_orchestrator::model::ChatMessage::new(
            "user",
            "sys words task words",
        )],
        tools: vec![],
        temperature: 0.2,
        max_tokens: 8,
        idempotency_key: "rollout-1".into(),
    };
    let r1 = client.complete(req.clone()).await.unwrap();
    let served_v1: u64 = r1
        .served_model
        .trim_start_matches("policy-v")
        .parse()
        .expect("versioned model name");
    assert_eq!(served_v1, 0, "first rollout served by v0");
    assert!(!r1.text.is_empty(), "the tiny LM generated a completion");

    // 2. the trainer steps on stamped samples → publishes v1
    let result = trainer
        .step(&[
            sample(1, 1, 1.0, served_v1),
            sample(2, 1, 0.0, served_v1),
            sample(3, 1, 0.5, served_v1),
            sample(4, 1, 0.25, served_v1),
        ])
        .unwrap();
    assert!(result.rejected_stale.is_empty());
    let published: PolicyVersion = result.published.expect("step published");
    assert_eq!(published.major, 1);

    // 3. the very next request is served by the NEW version
    let r2 = client
        .complete(ChatRequest {
            idempotency_key: "rollout-2".into(),
            ..req
        })
        .await
        .unwrap();
    let served_v2: u64 = r2
        .served_model
        .trim_start_matches("policy-v")
        .parse()
        .unwrap();
    assert_eq!(served_v2, 1, "publication is immediately visible");

    // 4. stale samples are rejected: v0-stamped samples while v2+ is
    //    published (bound 1)
    let _ = trainer
        .step(&[sample(5, 2, 1.0, 1), sample(6, 2, 0.0, 1)])
        .unwrap();
    assert_eq!(trainer.published().major, 2);
    let stale = trainer
        .step(&[sample(7, 3, 1.0, 0), sample(8, 3, 0.0, 0)])
        .unwrap();
    assert_eq!(stale.rejected_stale.len(), 2, "v0 samples are stale at v2");
    assert!(stale.published.is_none(), "nothing trained from stale data");
}

#[tokio::test]
async fn publication_crash_leaves_old_version_served() {
    let mut trainer = Trainer::new(TrainerConfig::default());
    let (addr, _handle) = dsec_train::server::serve(
        trainer.store(),
        std::net::SocketAddr::from(([127, 0, 0, 1], 0)),
        0.2,
        8,
    )
    .await;
    let client =
        dsec_orchestrator::model::OpenAiCompatClient::new(&format!("http://{addr}"), "policy")
            .unwrap();

    // a training step whose publication crashes mid-flip
    let _ = trainer
        .step_with_publication_crash(&[sample(1, 1, 1.0, 0), sample(2, 1, 0.0, 0)])
        .unwrap();
    assert!(
        trainer.store().barrier_closed(),
        "barrier closed by the crash"
    );

    // the OLD version is still served — no half-published policy
    let resp = client
        .complete(ChatRequest {
            model: "policy".into(),
            messages: vec![dsec_orchestrator::model::ChatMessage::new("user", "hello")],
            tools: vec![],
            temperature: 0.2,
            max_tokens: 4,
            idempotency_key: "post-crash".into(),
        })
        .await
        .unwrap();
    let served: u64 = resp
        .served_model
        .trim_start_matches("policy-v")
        .parse()
        .unwrap();
    assert_eq!(served, 0, "rollouts never see a half-published policy");

    // recovery discards the staged flip and re-opens publication
    let v = trainer.recover_publication(false).unwrap();
    assert_eq!(v.major, 0);
    assert!(!trainer.store().barrier_closed());
}
