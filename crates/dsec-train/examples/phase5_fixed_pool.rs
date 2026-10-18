//! Phase 5 demo: the fixed-pool training loop over real HTTP.
//!
//! A trainer + policy model server + the orchestrator's OpenAI client:
//! rollouts are stamped with the served policy version, the trainer
//! steps (real gradients, barrier publication), and the next request is
//! served by the new version.
//!
//! Run: `cargo run --release --example phase5_fixed_pool --features server -p dsec-train`

use std::sync::Arc;

use dsec_orchestrator::model::{ChatMessage, ChatRequest, ModelClient, OpenAiCompatClient};
use dsec_rl::reward::RewardRecord;
use dsec_rl::tito::TranscriptTurn;
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

#[tokio::main]
async fn main() {
    println!("== phase 5 demo: fixed-pool online training ==");
    let mut trainer = Trainer::new(TrainerConfig {
        lr: 0.2,
        hidden: 24,
        staleness_bound: 1,
        pool_rollout_gpus: 8,
        pool_train_gpus: 8,
        ..Default::default()
    });

    // the model server serves the PUBLISHED policy over real TCP
    let (addr, _handle) = dsec_train::server::serve(
        trainer.store(),
        std::net::SocketAddr::from(([127, 0, 0, 1], 0)),
        0.2,
        8,
    )
    .await;
    println!("policy server at {addr} (real HTTP)");
    let client = Arc::new(OpenAiCompatClient::new(&format!("http://{addr}"), "policy").unwrap());

    // three training steps: rollout (stamped) → step → verify the new
    // version is immediately served
    for round in 0..3 {
        // 1. a rollout against the currently published policy
        let resp = client
            .complete(ChatRequest {
                model: "policy".into(),
                messages: vec![ChatMessage::new(
                    "user",
                    "sys words task words good answer words",
                )],
                tools: vec![],
                temperature: 0.2,
                max_tokens: 6,
                idempotency_key: format!("round-{round}"),
            })
            .await
            .unwrap();
        let served: u64 = resp
            .served_model
            .trim_start_matches("policy-v")
            .parse()
            .unwrap();
        println!(
            "round {round}: rollout served by v{served}, completion = {:?}",
            resp.text.chars().take(40).collect::<String>()
        );

        // 2. train on a stamped group (the toy RL signal: reward 'good')
        let group: Vec<RolloutSample> = (0..4)
            .map(|i| {
                sample(
                    1000 * round + i,
                    round,
                    if i % 2 == 0 { 1.0 } else { 0.0 },
                    served,
                )
            })
            .collect();
        let result = trainer.step(&group).unwrap();
        let v = result.published.unwrap();
        println!(
            "  step: terms={} loss={:.4} → published v{} (staleness bound {})",
            result.terms,
            result.loss,
            v.major,
            trainer.published().major
        );

        // 3. the next request sees the new version
        let resp2 = client
            .complete(ChatRequest {
                model: "policy".into(),
                messages: vec![ChatMessage::new("user", "sys words task words")],
                tools: vec![],
                temperature: 0.2,
                max_tokens: 4,
                idempotency_key: format!("round-{round}-verify"),
            })
            .await
            .unwrap();
        let served2: u64 = resp2
            .served_model
            .trim_start_matches("policy-v")
            .parse()
            .unwrap();
        assert_eq!(served2, v.major, "publication must be immediately visible");
    }

    // 4. staleness demo: version-0 samples are now stale
    let stale = trainer
        .step(&[sample(9999, 99, 1.0, 0), sample(10000, 99, 0.0, 0)])
        .unwrap();
    println!(
        "stale check: {} samples from v0 rejected (bound = 1, published = v{})",
        stale.rejected_stale.len(),
        trainer.published().major
    );
    assert_eq!(stale.rejected_stale.len(), 2);
    println!("== phase 5 demo complete ==");
}
