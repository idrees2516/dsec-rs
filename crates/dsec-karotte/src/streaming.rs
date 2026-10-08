//! Event fan-out: the stdout renderer, the websocket broadcaster model,
//! and the backend streamer.
//!
//! Ports of upstream `karotte/transcript_streaming/`. Every event from
//! the runner fans out to three sinks: stdout (a human-readable render),
//! the websocket (a cursor-based broadcaster with replay), and the
//! training backend (positional RunState updates). Chunk events are
//! excluded from the transcript and the backend stream.

use crate::schemas::{Event, RunState};
use std::collections::BTreeMap;

/// Upstream `_MAX_DISPLAY_CHARS`.
pub const MAX_DISPLAY_CHARS: usize = 100_000;
/// Upstream `SEND_EVENT_LOOP_INTERVAL_S`.
pub const SEND_EVENT_LOOP_INTERVAL_S: f64 = 0.1;
/// Upstream `NO_CLIENT_TIMEOUT_S`.
pub const NO_CLIENT_TIMEOUT_S: f64 = 10.0;
/// Upstream external-message wait-log interval.
pub const BACKEND_WAIT_LOG_INTERVAL_S: f64 = 30.0;

// ---------------------------------------------------------------------------
// stdout renderer (stream_transcript_to_stdout.py)
// ---------------------------------------------------------------------------

/// Render one event for stdout (upstream Rich console lines).
pub fn render_event_stdout(event: &Event) -> String {
    let truncated = |t: &str| crate::text::truncate_middle(t, MAX_DISPLAY_CHARS);
    match event {
        Event::TaskStarted {
            run_id,
            task_id,
            model,
            n_steps,
            ..
        } => format!(
            "━{}━\nTask: {task_id} (run {run_id}) | model {} | {n_steps} steps",
            "─".repeat(76),
            model.as_deref().unwrap_or("?")
        ),
        Event::TaskPreHookCompleted { metadata } => {
            if metadata.is_empty() {
                String::new()
            } else {
                let pairs: Vec<String> =
                    metadata.iter().map(|(k, v)| format!("{k}: {v}")).collect();
                format!("pre-hook: {}", pairs.join(", "))
            }
        }
        Event::StepStarted { step } => format!("Starting step {}", step + 1),
        Event::StepCompleted { step } => format!("Step {} completed", step + 1),
        Event::MessageChunk { delta } => match &delta.content {
            Some(c) => format!("🗣️ Student: {}", truncated(c)),
            None => String::new(),
        },
        Event::MessageChunkReset {} => " (retry)".to_string(),
        Event::MessageAdded { message, .. } => match message.role {
            crate::schemas::Role::User => {
                format!("👤 User: {}", truncated(&msg_text(message)))
            }
            crate::schemas::Role::Assistant => {
                format!("🗣️ Student: {}", truncated(&msg_text(message)))
            }
            _ => String::new(), // system/tool are not rendered
        },
        Event::ToolCallStarted { tool_call } => {
            // bash is special-cased upstream: the command lines are
            // printed directly.
            let name = tool_call.function.name.clone().unwrap_or_default();
            if name == "bash" {
                if let Ok(args) = tool_call.function.parse_arguments() {
                    if let Some(cmd) = args.get("command").and_then(|c| c.as_str()) {
                        return format!("🔧 Calling tool: bash\n$ {}", truncated(cmd));
                    }
                }
            }
            format!("🔧 Calling tool: {name} {}", tool_call.function.arguments)
        }
        Event::ToolCallCompleted {
            tool_call_id,
            result,
        } => {
            let mut parts = vec![format!("✅ Tool call completed ({tool_call_id}):")];
            if let Some(sc) = &result.structured_content {
                if let Some(obj) = sc.as_object() {
                    for (k, v) in obj {
                        // bash stdout and the resource metrics are
                        // special-cased/skipped upstream.
                        if k == "stdout" && v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false)
                        {
                            continue;
                        }
                        if k == "_karotte_resource_metrics" {
                            parts.push("**Container Resources**".into());
                            continue;
                        }
                        parts.push(format!("  {k}: {}", render_value(v)));
                    }
                }
            }
            parts.join("\n")
        }
        Event::AnswersSubmitted { answers } => {
            let pairs: Vec<String> = answers.iter().map(|(k, v)| format!("{k}: {v}")).collect();
            format!("✅ Answers submitted: {}", pairs.join(", "))
        }
        Event::Scoring {
            scoring,
            resource_metrics,
        } => {
            let mut lines = vec!["✅ Scoring completed:".to_string()];
            lines.push(format!("  Score: {}", scoring.score));
            if scoring.continue_task {
                lines.push("  Continue task? -> Yes".into());
            } else {
                lines.push("  Continue task? -> No".into());
            }
            for (k, v) in &scoring.metadata {
                lines.push(format!("  {k}: {v}"));
            }
            if let Some(rm) = resource_metrics {
                lines.push(format!(
                    "  resources: cpu {:.1}% peak, mem {:.1}MB peak",
                    rm.peak_cpu_percent, rm.peak_memory_mb
                ));
            }
            lines.join("\n")
        }
        Event::Error {
            exception_type,
            message,
            ..
        } => format!("💥 Error occurred: {exception_type}: {message}"),
        Event::TaskCompleted { status } => format!("🏁 Result: {status:?}"),
        Event::TokenUsage {
            input_tokens,
            output_tokens,
            ..
        } => format!("tokens: {input_tokens} in / {output_tokens} out"),
    }
}

/// Render a JSON value without the string quotes (upstream prints the
/// raw value).
fn render_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn msg_text(m: &crate::schemas::Message) -> String {
    m.content
        .as_ref()
        .and_then(|c| c.as_text())
        .unwrap_or_default()
}

/// The final summary line (upstream `🏁 Result: ...`):
/// `{status}, {steps_passed}/{n_steps} steps passed, final score {score|N/A}`.
pub fn render_final(
    status: &crate::schemas::RunStatus,
    steps_passed: usize,
    n_steps: usize,
    score: Option<f64>,
) -> String {
    let score = match score {
        Some(s) => format!("{s}"),
        None => "N/A".to_string(),
    };
    format!(
        "🏁 Result: {:?}, {}/{} steps passed, final score {score}",
        status, steps_passed, n_steps
    )
}

// ---------------------------------------------------------------------------
// websocket broadcaster (stream_transcript_to_websocket.py)
// ---------------------------------------------------------------------------

/// One connected client: a cursor into the event log
/// (upstream per-client cursors).
#[derive(Debug, Clone)]
pub struct WsClient {
    /// Client id.
    pub id: u64,
    /// Next event index to send.
    pub cursor: usize,
    /// Whether the client is still connected.
    pub connected: bool,
}

/// The broadcaster (upstream `WebsocketBroadcaster`): keeps the full
/// event list; each `broadcast_tick` sends every pending event to each
/// lagging client in order; `wait_for_broadcasting_to_complete` requires
/// all clients caught up and at least one client present, within
/// `NO_CLIENT_TIMEOUT_S`.
#[derive(Debug, Default)]
pub struct Broadcaster {
    events: Vec<Event>,
    clients: BTreeMap<u64, WsClient>,
    next_client_id: u64,
    shutting_down: bool,
}

impl Broadcaster {
    /// An empty broadcaster.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a client at cursor 0 (immediate close when shutting
    /// down, upstream).
    pub fn register(&mut self) -> Option<u64> {
        if self.shutting_down {
            return None;
        }
        self.next_client_id += 1;
        let id = self.next_client_id;
        self.clients.insert(
            id,
            WsClient {
                id,
                cursor: 0,
                connected: true,
            },
        );
        Some(id)
    }

    /// Append one event (runner side).
    pub fn append(&mut self, event: Event) {
        self.events.push(event);
    }

    /// One broadcast loop iteration: returns the messages sent per
    /// client (client id → serialized events), dropping clients that
    /// failed.
    pub fn broadcast_tick(&mut self) -> BTreeMap<u64, Vec<String>> {
        let mut out = BTreeMap::new();
        for client in self.clients.values_mut() {
            if !client.connected {
                continue;
            }
            let mut sent = Vec::new();
            while client.cursor < self.events.len() {
                let ev = &self.events[client.cursor];
                sent.push(serde_json::to_string(ev).unwrap_or_default());
                client.cursor += 1;
            }
            if !sent.is_empty() {
                out.insert(client.id, sent);
            }
        }
        out
    }

    /// Simulate a client disconnect.
    pub fn drop_client(&mut self, id: u64) {
        if let Some(c) = self.clients.get_mut(&id) {
            c.connected = false;
        }
        self.clients.remove(&id);
    }

    /// Whether every connected client is caught up.
    pub fn all_caught_up(&self) -> bool {
        self.clients.values().all(|c| c.cursor >= self.events.len())
    }

    /// `wait_for_broadcasting_to_complete(timeout_s)` semantics: all
    /// clients caught up **and** at least one client exists.
    pub fn broadcasting_complete(&self) -> bool {
        !self.clients.is_empty() && self.all_caught_up()
    }

    /// Number of events buffered.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether nothing is buffered.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

// ---------------------------------------------------------------------------
// backend streamer (stream_transcript_to_backend.py)
// ---------------------------------------------------------------------------

/// The backend request model: which API call each event triggers
/// (upstream appends every non-chunk event, and additionally updates
/// run state on task lifecycle + usage events).
#[derive(Debug, Clone, PartialEq)]
pub enum BackendCall {
    /// `create_transcript` for the run id (first event only).
    CreateTranscript {
        /// Run id.
        run_id: String,
    },
    /// `append_transcript(run_id, event, seq)` — positional writes,
    /// idempotent for replays (the seq resets to 0 at stream start).
    AppendTranscript {
        /// Run id.
        run_id: String,
        /// The event JSON.
        event: String,
        /// Position in the stream.
        seq: usize,
    },
    /// `update_run_state` (task lifecycle + usage events).
    UpdateRunState {
        /// Run id.
        run_id: String,
        /// The current run state fields.
        state: RunState,
    },
}

/// Translate an event stream into backend calls (upstream
/// `stream_transcript_to_backend`): chunk events are skipped; the first
/// event must be `task_started` (builds the RunState, resets seq);
/// `TaskStarted|TaskCompleted|TokenUsage` additionally trigger
/// `update_run_state`.
pub fn backend_calls(events: &[Event]) -> Vec<BackendCall> {
    let mut out = Vec::new();
    let mut state: Option<RunState> = None;
    let mut seq = 0usize;
    let mut run_id = String::new();
    for ev in events {
        if ev.is_chunk() {
            continue;
        }
        if state.is_none() {
            // The first event is asserted to be task_started upstream.
            if let Event::TaskStarted {
                run_id: rid,
                task_id,
                n_steps,
                ..
            } = ev
            {
                run_id = rid.clone();
                state = Some(RunState::from_task_started(rid, task_id, *n_steps));
                out.push(BackendCall::CreateTranscript {
                    run_id: rid.clone(),
                });
            } else {
                continue;
            }
        }
        if let Some(st) = state.as_mut() {
            st.apply(ev);
        }
        out.push(BackendCall::AppendTranscript {
            run_id: run_id.clone(),
            event: serde_json::to_string(ev).unwrap_or_default(),
            seq,
        });
        seq += 1;
        let triggers_update = matches!(
            ev,
            Event::TaskStarted { .. } | Event::TaskCompleted { .. } | Event::TokenUsage { .. }
        );
        if triggers_update {
            out.push(BackendCall::UpdateRunState {
                run_id: run_id.clone(),
                state: state.clone().unwrap_or_default(),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemas::{CallToolResult, Message, RunStatus, Scoring, ToolCall};

    fn ev_user(text: &str) -> Event {
        Event::MessageAdded {
            message: Message::text(crate::schemas::Role::User, text),
            finish_reason: None,
            raw: None,
        }
    }

    #[test]
    fn stdout_renders_task_header_and_steps() {
        let e = Event::TaskStarted {
            run_id: "r1".into(),
            task_id: "t1".into(),
            model: Some("claude-opus-5-5".into()),
            n_steps: 2,
            reasoning_effort: None,
        };
        let s = render_event_stdout(&e);
        assert!(s.starts_with('━'));
        assert!(s.contains("Task: t1 (run r1)"));
        assert!(s.contains("claude-opus-5-5"));
        assert!(s.contains("2 steps"));

        assert_eq!(
            render_event_stdout(&Event::StepStarted { step: 0 }),
            "Starting step 1"
        );
        assert_eq!(
            render_event_stdout(&Event::StepCompleted { step: 1 }),
            "Step 2 completed"
        );
    }

    #[test]
    fn stdout_renders_roles_with_glyphs() {
        assert!(render_event_stdout(&ev_user("do the thing")).contains("👤 User: do the thing"));
        let assistant = Event::MessageAdded {
            message: Message::text(crate::schemas::Role::Assistant, "the answer"),
            finish_reason: None,
            raw: None,
        };
        assert!(render_event_stdout(&assistant).contains("🗣️ Student: the answer"));
        let system = Event::MessageAdded {
            message: Message::text(crate::schemas::Role::System, "sysprompt"),
            finish_reason: None,
            raw: None,
        };
        assert_eq!(render_event_stdout(&system), "");
    }

    #[test]
    fn stdout_special_cases_bash_commands() {
        let call = ToolCall::function(
            "c1",
            "bash",
            serde_json::json!({"command": "ls -la /workdir"}).to_string(),
        );
        let rendered = render_event_stdout(&Event::ToolCallStarted { tool_call: call });
        assert!(rendered.contains("🔧 Calling tool: bash"));
        assert!(rendered.contains("$ ls -la /workdir"));

        let other = ToolCall::function("c2", "view_lines_in_file", r#"{"file_path":"/f"}"#);
        let rendered2 = render_event_stdout(&Event::ToolCallStarted { tool_call: other });
        assert!(rendered2.contains("🔧 Calling tool: view_lines_in_file"));
    }

    #[test]
    fn stdout_renders_scoring_with_continue() {
        let e = Event::Scoring {
            scoring: Scoring::pass(0.75),
            resource_metrics: None,
        };
        let s = render_event_stdout(&e);
        assert!(s.contains("✅ Scoring completed:"));
        assert!(s.contains("Score: 0.75"));
        assert!(s.contains("Continue task? -> Yes"));
    }

    #[test]
    fn final_summary_line() {
        let s = render_final(&RunStatus::Passed, 2, 3, Some(0.5));
        assert_eq!(s, "🏁 Result: Passed, 2/3 steps passed, final score 0.5");
        let s2 = render_final(&RunStatus::Error, 0, 2, None);
        assert!(s2.contains("N/A"));
    }

    #[test]
    fn broadcaster_replays_to_late_clients() {
        let mut b = Broadcaster::new();
        b.append(ev_user("one"));
        b.append(ev_user("two"));
        // A client registers after events exist — the cursor starts at 0,
        // so it receives the full replay in order.
        let c1 = b.register().unwrap();
        let sent = b.broadcast_tick();
        assert_eq!(sent[&c1].len(), 2);
        assert!(sent[&c1][0].contains("one"));
        // Caught up; a second tick sends nothing.
        assert!(b.broadcast_tick().is_empty());
        assert!(b.broadcasting_complete());

        // A second client joins later and gets the replay too.
        let c2 = b.register().unwrap();
        b.append(ev_user("three"));
        let sent2 = b.broadcast_tick();
        assert_eq!(sent2[&c2].len(), 3, "full replay in order");
        assert!(b.all_caught_up());
    }

    #[test]
    fn broadcaster_requires_at_least_one_client() {
        let mut b = Broadcaster::new();
        b.append(ev_user("x"));
        // No clients: broadcasting never "completes".
        assert!(!b.broadcasting_complete());
        b.register().unwrap();
        b.broadcast_tick();
        assert!(b.broadcasting_complete());
    }

    #[test]
    fn broadcaster_registration_refused_when_shutting_down() {
        let mut b = Broadcaster::new();
        b.shutting_down = true;
        assert!(b.register().is_none());
    }

    #[test]
    fn backend_calls_skip_chunks_and_are_positional() {
        let events = vec![
            Event::TaskStarted {
                run_id: "r1".into(),
                task_id: "t1".into(),
                model: None,
                n_steps: 1,
                reasoning_effort: None,
            },
            Event::MessageChunk {
                delta: crate::schemas::Delta {
                    content: Some("stream".into()),
                    ..Default::default()
                },
            },
            Event::MessageChunkReset {},
            ev_user("instructions"),
            Event::TokenUsage {
                input_tokens: 5,
                output_tokens: 2,
                cache_read_tokens: None,
                cache_write_tokens: None,
                estimated: false,
                service_tier: None,
            },
            Event::TaskCompleted {
                status: RunStatus::Passed,
            },
        ];
        let calls = backend_calls(&events);
        let kinds: Vec<&str> = calls
            .iter()
            .map(|c| match c {
                BackendCall::CreateTranscript { .. } => "create",
                BackendCall::AppendTranscript { .. } => "append",
                BackendCall::UpdateRunState { .. } => "update",
            })
            .collect();
        // chunk events skipped; create + 4 appends + updates on
        // task_started, token_usage, task_completed.
        assert_eq!(
            kinds,
            vec!["create", "append", "update", "append", "append", "update", "append", "update"]
        );
        // Seq is positional and contiguous.
        let seqs: Vec<usize> = calls
            .iter()
            .filter_map(|c| match c {
                BackendCall::AppendTranscript { seq, .. } => Some(*seq),
                _ => None,
            })
            .collect();
        assert_eq!(seqs, vec![0, 1, 2, 3]);
        // The final run state carries the totals.
        if let Some(BackendCall::UpdateRunState { state, .. }) = calls.last() {
            assert_eq!(state.total_input_tokens, Some(5));
            assert_eq!(state.status, RunStatus::Passed);
        } else {
            panic!("expected trailing update");
        }
    }

    #[test]
    fn long_outputs_are_middle_truncated() {
        let long = "x".repeat(MAX_DISPLAY_CHARS + 1000);
        let e = Event::MessageAdded {
            message: Message::text(crate::schemas::Role::User, long),
            finish_reason: None,
            raw: None,
        };
        let s = render_event_stdout(&e);
        assert!(s.contains("truncated"));
        assert!(s.chars().count() < MAX_DISPLAY_CHARS + 10_000);
    }

    #[test]
    fn tool_completion_renders_structured_content() {
        let e = Event::ToolCallCompleted {
            tool_call_id: "c1".into(),
            result: CallToolResult {
                content: vec![],
                structured_content: Some(serde_json::json!({
                    "stdout": "file-a file-b",
                    "exit_code": 0
                })),
                is_error: false,
            },
        };
        let s = render_event_stdout(&e);
        assert!(s.contains("✅ Tool call completed (c1):"));
        assert!(s.contains("stdout: file-a file-b"));
    }
}
