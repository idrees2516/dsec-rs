//! The environment zoo: compile every domain template, run scripted
//! rollouts, print rewards.
//!
//! ```bash
//! cargo run -p dsec-agentenv --example envgen_zoo
//! ```

use dsec_agentenv::agentloop::{AgentLoop, LoopConfig, ScriptedPolicy};
use dsec_agentenv::envgen::{templates, Variants};
use dsec_agentenv::verifier::{AnchorJudge, RewardConfig, VerifierHarness};

fn main() {
    println!("dsec-agentenv environment zoo\n");

    let mut specs = vec![
        ("knowledge-work", templates::knowledge_work()),
        ("terminal (computer use)", templates::terminal()),
        ("webdev", templates::webdev()),
    ];
    // plus two seeded variants of the knowledge-work family
    for v in Variants::generate(&templates::knowledge_work(), 3, 42)
        .into_iter()
        .skip(1)
    {
        specs.push(("knowledge-work (variant)", v));
    }

    let judge = AnchorJudge::new();
    for (name, spec) in &specs {
        let bundle = match spec.compile().and_then(|b| b.boot()) {
            Ok(b) => b,
            Err(e) => {
                println!("[{name}] compile failed: {e}");
                continue;
            }
        };
        let mut pod = bundle.pod;
        let row = spec.to_task_row(0);
        println!(
            "[{name}] {} | systems: {} | tools: {} | rubric items: {} | ports {:?}",
            row.extra_info.instance_id,
            pod.state.len(),
            bundle.tools.len(),
            bundle.rubric.items.len(),
            pod.listening
        );

        // a scripted rollout: probe the first tool, then answer from the
        // rubric's ground truth (rule needles + judge anchors)
        let mut answer = String::from("Report: ");
        for r in &spec.rubric {
            if let Some(dsec_agentenv::envgen::CheckSpec::AllText { needles }) = &r.check {
                for n in needles {
                    answer.push_str(n);
                    answer.push(' ');
                }
            }
            if let Some(dsec_agentenv::envgen::CheckSpec::AnyText { needles, .. }) = &r.check {
                for n in needles {
                    answer.push_str(n);
                    answer.push(' ');
                }
            }
            for tok in dsec_agentenv::verifier::extract_anchor_tokens(&r.pass_anchor) {
                answer.push_str(&tok);
                answer.push(' ');
            }
        }
        let first_tool = spec
            .tools
            .first()
            .map(|t| (format!("{}.{}", t.server, t.name), serde_json::json!({})))
            .unwrap_or_else(|| (String::new(), serde_json::json!({})));
        let mut policy = ScriptedPolicy::new(vec![
            ("probing the environment".to_string(), vec![first_tool]),
            (answer, vec![]),
        ]);
        let mut lp = AgentLoop::new(LoopConfig::default());
        let rollout = lp.run(&mut pod, &spec.instruction, &bundle.tools, &mut policy);

        let harness = VerifierHarness::new(&bundle.rubric, &judge, RewardConfig::default());
        let outcome =
            harness.calculate_reward(&mut pod, &bundle.manifest, &rollout.to_agent_result());
        println!(
            "  rollout: {} tool turns, {} tokens, completed={} -> reward {:?} (masked: {})",
            rollout.tool_turns,
            rollout.tokens,
            rollout.completed,
            outcome.score(),
            outcome.masked()
        );
    }
    println!("\nthe zoo compiles, boots, rolls out, and grades — deterministically.");
}
