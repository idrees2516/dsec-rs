//! The full Karotte pipeline, end to end, deterministically: confinement
//! plan → firewall → MCP server → agent loop → judges → transcript →
//! stdout render → backend calls.
//!
//! ```bash
//! cargo run -p dsec-karotte --example full_run
//! ```

use dsec_karotte::cgroups::{detect_layout, parse_mounts, StudentCgroup};
use dsec_karotte::confinement::{
    firewall_canaries, firewall_effective, firewall_plan, kill_processes, Contract, NetworkPolicy,
    ResourceLimits, Sandbox,
};
use dsec_karotte::judges::{RegexJudge, RubricCriterion, RubricJudge, ScriptedCompletions};
use dsec_karotte::mcp::{McpServer, RpcRequest, ToolDescriptor};
use dsec_karotte::message_loop::{MapDispatcher, MessageLoop, ScriptedSource};
use dsec_karotte::runner::Runner;
use dsec_karotte::schemas::{EvaluationRunConfig, Event, Message, Role};
use dsec_karotte::streaming::{backend_calls, render_event_stdout, Broadcaster, MAX_DISPLAY_CHARS};
use dsec_karotte::task::{create_task, StepConfig};
use dsec_karotte::text::truncate_middle;
use std::net::Ipv4Addr;
use std::sync::Arc;

fn main() {
    println!("dsec-karotte — the Karotte pipeline, layer by layer\n");

    // 1. Confinement: the resource plan for a 5 GiB sandbox.
    let limits = ResourceLimits::defaults(5 << 30, 100 << 30);
    println!(
        "confinement: memory {} MiB (5 GiB VM minus the 1 GiB harness reserve), {} processes, disk {} GiB (80% of free)",
        limits.memory_bytes.unwrap() >> 20,
        limits.process_count.unwrap(),
        limits.file.as_ref().unwrap().bytes.unwrap() >> 30
    );

    // 2. The cgroup the student lands in (mount table parsed v2-style).
    let mounts =
        parse_mounts("cgroup2 /sys/fs/cgroup cgroup2 rw,nosuid 0 0\n/dev/vda1 / ext4 rw 0 0\n");
    let layout = detect_layout(&mounts, "/session.slice", "cpuset cpu io memory pids");
    let group = StudentCgroup::v2("/sys/fs/cgroup/karotte_uid_1000");
    println!("cgroups: layout {:?}, group {}", layout, group.path);
    for (file, value) in group.set_memory_limit_writes(limits.memory_bytes) {
        println!("  write {value:>12} -> {file}");
    }
    println!(
        "  write    1 -> {} (an OOM kill reaps the whole group)",
        group.delegation_writes()[1].0
    );

    // 3. The firewall: owner-matched rules in the upstream order.
    let plan = firewall_plan(
        Sandbox::Vm,
        1000,
        &[8001, 8080],
        &[Ipv4Addr::new(192, 0, 2, 1)],
        NetworkPolicy::Strict,
        &[Ipv4Addr::new(203, 0, 113, 9)],
    );
    println!(
        "\nfirewall ({} rules, final {}):",
        plan.rules.len(),
        plan.final_target
    );
    for rule in &plan.rules {
        println!("  iptables {}", rule.text);
    }
    let canaries = firewall_canaries(NetworkPolicy::Strict, Some(Ipv4Addr::new(192, 168, 0, 1)));
    let reached = firewall_effective(true, &[false, false, false, false]);
    println!(
        "canaries: {} targets, contract = {:?} ({})",
        canaries.len(),
        reached,
        reached.as_str()
    );
    assert_eq!(reached, Contract::Prevented);

    // 4. The cohort reap: a backend whose passes clean up.
    struct Cleaning;
    impl dsec_karotte::confinement::CohortBackend for Cleaning {
        fn pass(&self) -> dsec_karotte::confinement::KillPass {
            dsec_karotte::confinement::KillPass {
                cgroup_kill: true,
                pidns_kill: true,
                cohort_kill_ran: true,
                remaining: 0,
            }
        }
        fn cohort_alive(&self) -> bool {
            false
        }
    }
    let mut t = 0.0;
    let mut clock = move || {
        let now = t;
        t += 0.05;
        now
    };
    kill_processes(&Cleaning, &mut clock).expect("the cohort reaps clean");
    println!("cohort: cgroup.kill + pidns + in-cohort kill(-1) -> clean sweep");

    // 5. The MCP server: register the bash tool, then call it.
    let catalog = vec![(
        ToolDescriptor {
            name: "bash".into(),
            description: "Run a bash command as the student.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }),
        },
        Arc::new(|args: &serde_json::Value| {
            let cmd = args.get("command").and_then(|c| c.as_str()).unwrap_or("");
            Ok(dsec_karotte::schemas::CallToolResult::text(format!(
                "ran: {cmd}"
            )))
        }) as dsec_karotte::mcp::ToolHandler,
    )];
    let srv = McpServer::new(catalog);
    let resp = srv.handle(&RpcRequest {
        id: serde_json::json!(1),
        method: "initialize".into(),
        params: serde_json::json!({}),
    });
    let server_info = &resp.result.unwrap()["serverInfo"];
    println!(
        "\nmcp: {} v{}",
        server_info["name"].as_str().unwrap(),
        server_info["version"].as_str().unwrap()
    );
    srv.handle(&RpcRequest {
        id: serde_json::json!(2),
        method: "tools/call".into(),
        params: serde_json::json!({"name": "register_tools", "arguments": {"tools": ["bash"]}}),
    });
    let resp = srv.handle(&RpcRequest {
        id: serde_json::json!(3),
        method: "tools/call".into(),
        params: serde_json::json!({"name": "bash", "arguments": {"command": "cat key.txt"}}),
    });
    println!(
        "mcp: bash -> {}",
        resp.result.unwrap()["content"][0]["text"].as_str().unwrap()
    );

    // 6. The run: a two-step task, a scripted model, regex + rubric judges.
    let rubric = vec![
        RubricCriterion::new("the answer contains the activation key", 1.0),
        RubricCriterion::new("the work is shown", 0.5),
    ];
    let judge_client: Arc<dyn dsec_karotte::judges::CompletionClient> =
        Arc::new(ScriptedCompletions::new(vec![
            "YES\nthe key is quoted".into(),
            "NO\nno work shown".into(),
        ]));
    let step1_judge: Arc<dyn dsec_karotte::judges::Judge> =
        Arc::new(RegexJudge::new(vec!["KEY-7F3A-92Z".into()]));
    let step2_judge: Arc<dyn dsec_karotte::judges::Judge> = Arc::new(
        RubricJudge::new(rubric, judge_client, 0.0).with_contexts(vec![Box::new(
            dsec_karotte::judges::TranscriptContext::messages(None),
        )]),
    );
    let task = create_task(
        "activation-key",
        vec!["bash".into(), "view_lines_in_file".into()],
        vec![
            StepConfig::new("Find the activation key and report it.", step1_judge),
            StepConfig::new("Explain how you found it.", step2_judge),
        ],
        Some("You are a careful assistant working inside a sandbox.".into()),
    );

    let agent = Arc::new(MessageLoop::new(
        Arc::new(ScriptedSource::new(vec![
            Message {
                content: None,
                role: Role::Assistant,
                tool_calls: Some(vec![dsec_karotte::schemas::ToolCall::function(
                    "call-1",
                    "bash",
                    r#"{"command": "cat key.txt"}"#,
                )]),
                reasoning_content: None,
                tool_call_id: None,
            },
            Message::text(Role::Assistant, "the key is KEY-7F3A-92Z"),
            Message::text(Role::Assistant, "I ran cat key.txt to find it."),
        ])),
        {
            let mut dispatcher = MapDispatcher::new();
            dispatcher.register("bash", |args| {
                let cmd = args.get("command").and_then(|c| c.as_str()).unwrap_or("");
                Ok(dsec_karotte::schemas::CallToolResult::text(format!(
                    "ran: {cmd}"
                )))
            });
            Arc::new(dispatcher)
        },
    ));

    let mut config = EvaluationRunConfig::new("run-42", "activation-key", "fake/model");
    config.transcript_file = Some("/out/transcript.json".into());
    config.turn_limit = Some(10);
    config.on_step_time_limit = dsec_karotte::schemas::LimitAction::Score;
    let writes: std::sync::Mutex<Vec<(String, String)>> = std::sync::Mutex::new(Vec::new());
    let runner = Runner {
        config,
        task,
        agent,
        durable_write: Box::new(|p, b| {
            writes.lock().unwrap().push((p.to_string(), b.to_string()));
            Ok(())
        }),
    };
    let outcome = runner.run();

    // 7. The fan-out: stdout render, websocket replay, backend calls.
    println!("\nstdout stream:");
    for event in &outcome.transcript.events {
        let line = render_event_stdout(event);
        if !line.is_empty() {
            println!("{}", truncate_middle(&line, MAX_DISPLAY_CHARS / 10));
        }
    }
    println!(
        "{}",
        dsec_karotte::streaming::render_final(
            &outcome.status,
            outcome.steps_passed,
            outcome.n_steps,
            outcome.score
        )
    );

    let mut broadcaster = Broadcaster::new();
    for event in outcome.transcript.events.clone() {
        broadcaster.append(event);
    }
    let client = broadcaster.register().unwrap();
    let sent = broadcaster.broadcast_tick();
    println!(
        "\nwebsocket: client {client} received {} events (full replay, in order)",
        sent[&client].len()
    );

    let calls = backend_calls(&outcome.transcript.events);
    let appends = calls
        .iter()
        .filter(|c| {
            matches!(
                c,
                dsec_karotte::streaming::BackendCall::AppendTranscript { .. }
            )
        })
        .count();
    println!(
        "backend: {} calls ({appends} positional appends, chunk-free)",
        calls.len()
    );

    let (path, body) = writes.lock().unwrap()[0].clone();
    println!(
        "\ntranscript: written durably to {path} ({} bytes, {} events)",
        body.len(),
        outcome.transcript.events.len()
    );
    assert_eq!(outcome.status, dsec_karotte::schemas::RunStatus::Passed);
    assert_eq!(outcome.steps_passed, 2);
    // Step 1: regex 1.0. Step 2: rubric — one YES criterion of 1.5 total
    // weight → score 1.0, strictly above the 0.0 threshold.
    assert_eq!(outcome.score, Some(1.0));
    let last = outcome.transcript.events.last().unwrap();
    assert!(matches!(last, Event::TaskCompleted { .. }));
    println!("\nthe whole pipeline — confinement, cgroups, firewall, mcp, agent loop, judges, transcript — runs deterministically.");
}
