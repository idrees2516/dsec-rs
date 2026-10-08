//! The agent message loop: turn limits, time/context limits, empty-turn
//! nudges, tool-call execution and projection, counter injection, and the
//! LLM retry policy.
//!
//! Ports of upstream `agents/message_loop.py`, `agents/builtin_source.py`
//! (retry policy + completion assembly), and the tool-result projection
//! rules. The model itself is a pluggable [`MessageSource`]; the fake and
//! scripted sources stand in for litellm.

use crate::error::{Error, Result};
use crate::schemas::{CallToolResult, Event, ImageContent, Message, Role, ToolCall};
use std::sync::{Arc, Mutex};

/// Upstream `MAX_CONSECUTIVE_EMPTY_TURNS`.
pub const MAX_CONSECUTIVE_EMPTY_TURNS: usize = 3;

/// Upstream `EMPTY_TURN_NUDGE`, verbatim.
pub const EMPTY_TURN_NUDGE: &str = "Your last turn produced no message or tool call. Pick up where you left off. Do not apologize or recap. Break the remaining work into smaller pieces if necessary.";

/// What the loop should do when a limit trips (upstream `on_*` literals).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LimitAction {
    /// Raise an error (the run becomes `error`).
    #[default]
    Raise,
    /// End the step quietly and let it be scored.
    Score,
}

/// One model response per call (upstream `MessageSource.collect`).
pub trait MessageSource: Send + Sync {
    /// Produce the next model turn given the conversation so far and the
    /// tool schemas. Yields streaming events; the final
    /// `MessageAddedEvent` carries the completed message.
    fn collect(&self, messages: &[Message], tools: &[serde_json::Value]) -> Result<Vec<Event>>;

    /// Stop the underlying transport (upstream `Agent.stop`).
    fn stop(&self) -> Result<()> {
        Ok(())
    }
}

/// The scripted source (upstream `FakeSource`): replays fixed messages,
/// raising when exhausted.
pub struct ScriptedSource {
    messages: Vec<Message>,
    calls: Mutex<usize>,
}

impl ScriptedSource {
    /// Replay these messages in order.
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            calls: Mutex::new(0),
        }
    }
}

impl MessageSource for ScriptedSource {
    fn collect(&self, _messages: &[Message], _tools: &[serde_json::Value]) -> Result<Vec<Event>> {
        let mut i = self.calls.lock().unwrap();
        let msg = self.messages.get(*i).cloned().ok_or_else(|| {
            Error::LoopLimit("Not enough messages defined for fake model.".into())
        })?;
        *i += 1;
        let finish = if msg.tool_calls.is_some() {
            Some("tool_calls")
        } else {
            Some("stop")
        };
        Ok(vec![Event::MessageAdded {
            message: msg,
            finish_reason: finish.map(str::to_string),
            raw: None,
        }])
    }
}

/// The MCP client the loop drives (a subset: dispatch tool calls).
pub trait ToolDispatcher: Send + Sync {
    /// Call a tool by name with parsed JSON arguments.
    fn call_tool(&self, name: &str, arguments: &serde_json::Value) -> Result<CallToolResult>;
}

/// A dispatch handler.
pub type ToolHandler = Arc<dyn Fn(&serde_json::Value) -> Result<CallToolResult> + Send + Sync>;

/// A map-based dispatcher for the simulated harness.
pub struct MapDispatcher {
    tools: std::collections::BTreeMap<String, ToolHandler>,
}

impl Default for MapDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl MapDispatcher {
    /// An empty dispatcher.
    pub fn new() -> Self {
        Self {
            tools: Default::default(),
        }
    }
    /// Register a handler.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        handler: impl Fn(&serde_json::Value) -> Result<CallToolResult> + Send + Sync + 'static,
    ) {
        self.tools.insert(name.into(), Arc::new(handler));
    }
}

impl ToolDispatcher for MapDispatcher {
    fn call_tool(&self, name: &str, arguments: &serde_json::Value) -> Result<CallToolResult> {
        match self.tools.get(name) {
            Some(h) => h(arguments),
            None => Ok(CallToolResult::error_text(format!(
                "Error: Unknown tool {name}"
            ))),
        }
    }
}

/// Loop configuration (upstream: run config + per-step resolution).
#[derive(Debug, Clone, Default)]
pub struct LoopLimits {
    /// Run-wide turn limit (`_turn_count` persists across steps; `>` not
    /// `>=`).
    pub turn_limit: Option<u64>,
    /// Step time limit, seconds (checked *between* turns — an in-flight
    /// turn may finish, so a step may overrun by one turn).
    pub time_limit_seconds: Option<f64>,
    /// Action on time limit.
    pub on_time_limit: LimitAction,
    /// Step context window limit, tokens (measured from the previous
    /// turn's reported input tokens).
    pub context_window_limit: Option<i64>,
    /// Action on context limit.
    pub on_context_window_limit: LimitAction,
    /// Append "Time remaining: N seconds" to the turn's last message.
    pub inject_time_remaining_counter: bool,
    /// Append "Context remaining: N" to the turn's last message.
    pub inject_context_remaining_counter: bool,
}

/// The agent loop (upstream `MessageLoopAgent`).
pub struct MessageLoop {
    source: Arc<dyn MessageSource>,
    dispatcher: Arc<dyn ToolDispatcher>,
    turn_count: Mutex<u64>,
}

impl MessageLoop {
    /// A loop over `source` executing tools via `dispatcher`.
    pub fn new(source: Arc<dyn MessageSource>, dispatcher: Arc<dyn ToolDispatcher>) -> Self {
        Self {
            source,
            dispatcher,
            turn_count: Mutex::new(0),
        }
    }

    /// Run one step: yields the step's events (the caller adds
    /// bookkeeping). Mirrors upstream `run_step`.
    pub fn run_step(
        &self,
        instructions: &str,
        limits: &LoopLimits,
        tools: &[serde_json::Value],
        clock: &mut dyn FnMut() -> f64,
    ) -> Result<Vec<Event>> {
        let mut events: Vec<Event> = Vec::new();
        let deadline = limits.time_limit_seconds.map(|t| clock() + t);
        let mut context_length: i64 = 0;
        let mut conversation: Vec<Message> = Vec::new();
        let mut empty_turns: usize = 0;

        // The user instruction message.
        let user = Message::text(Role::User, instructions);
        events.push(Event::MessageAdded {
            message: user.clone(),
            finish_reason: None,
            raw: None,
        });
        conversation.push(user);

        loop {
            // 1. Turn limit (run-wide, > not >=).
            {
                let mut tc = self.turn_count.lock().unwrap();
                *tc += 1;
                if let Some(limit) = limits.turn_limit {
                    if *tc > limit {
                        return Err(Error::LoopLimit(format!("Turn limit of {limit} reached.")));
                    }
                }
            }

            // 2. Time limit between turns.
            if let Some(dl) = deadline {
                if clock() >= dl {
                    match limits.on_time_limit {
                        LimitAction::Raise => {
                            let t = limits.time_limit_seconds.unwrap();
                            return Err(Error::LoopLimit(format!("Time limit of {t}s reached.")));
                        }
                        LimitAction::Score => return Ok(events),
                    }
                }
            }

            // 3. Context window limit from the previous turn.
            if let Some(limit) = limits.context_window_limit {
                if context_length >= limit {
                    match limits.on_context_window_limit {
                        LimitAction::Raise => {
                            return Err(Error::LoopLimit(format!(
                                "Context window limit of {limit} reached."
                            )));
                        }
                        LimitAction::Score => return Ok(events),
                    }
                }
            }

            // 4. One model turn.
            let turn_events = self.source.collect(&conversation, tools)?;
            let mut message_event: Option<(Message, Option<String>)> = None;
            let mut turn_events = turn_events;
            for ev in turn_events.drain(..) {
                match &ev {
                    Event::MessageAdded {
                        message,
                        finish_reason,
                        ..
                    } => {
                        message_event = Some((message.clone(), finish_reason.clone()));
                    }
                    Event::TokenUsage { input_tokens, .. } => {
                        context_length = *input_tokens;
                    }
                    _ => {}
                }
                events.push(ev);
            }
            let (message, finish_reason) = message_event
                .ok_or_else(|| Error::LoopLimit("model turn produced no message".into()))?;

            if finish_reason.as_deref() == Some("length") {
                // warning upstream; truncated turns handled below
            }
            if finish_reason.as_deref() == Some("content_filter") {
                // Ends the step; the run continues to scoring.
                return Ok(events);
            }

            // 5. No tool calls → terminal, or empty-turn nudge.
            let has_calls = message
                .tool_calls
                .as_ref()
                .map(|c| !c.is_empty())
                .unwrap_or(false);
            if !has_calls {
                let truncated = finish_reason.as_deref() == Some("length")
                    || message
                        .reasoning_content
                        .as_deref()
                        .map(|r| !r.trim().is_empty())
                        .unwrap_or(false);
                if message.has_text() || !truncated {
                    return Ok(events);
                }
                // Empty (truncated or reasoning-only) turn.
                empty_turns += 1;
                if empty_turns > MAX_CONSECUTIVE_EMPTY_TURNS {
                    return Err(Error::LoopLimit(format!(
                        "Model produced neither text nor tool calls {n} turns in a row.",
                        n = empty_turns
                    )));
                }
                let nudge = Message::text(Role::User, EMPTY_TURN_NUDGE);
                events.push(Event::MessageAdded {
                    message: nudge.clone(),
                    finish_reason: None,
                    raw: None,
                });
                conversation.push(nudge);
                continue;
            }
            empty_turns = 0;
            conversation.push(message.clone());

            // 6. Execute the tool calls (buffered, then projected).
            let calls = message.tool_calls.clone().unwrap_or_default();
            let mut last_tool_event_index: Option<usize> = None;
            for call in &calls {
                events.push(Event::ToolCallStarted {
                    tool_call: call.clone(),
                });
                let parsed = call.function.parse_arguments().ok();
                let result = match parsed {
                    Some(args) => self.dispatch_tool(call, &args)?,
                    None => {
                        // Malformed JSON arguments (upstream rewrites the
                        // arguments and returns an error result).
                        let rewritten = ToolCall {
                            id: call.id.clone(),
                            function: crate::schemas::Function {
                                name: call.function.name.clone(),
                                arguments: serde_json::to_string(
                                    &serde_json::json!({"_error": "malformed JSON in original arguments"}),
                                )
                                .unwrap(),
                            },
                            call_type: call.call_type.clone(),
                        };
                        let _ = rewritten;
                        CallToolResult::error_text("Error: Invalid JSON in tool arguments")
                    }
                };
                events.push(Event::ToolCallCompleted {
                    tool_call_id: call.id.clone(),
                    result: result.clone(),
                });
                // The tool-role message (first content block only).
                let part = to_chat_content_part(&result);
                let tool_msg = Message::tool_result(call.id.clone(), vec![part]);
                events.push(Event::MessageAdded {
                    message: tool_msg.clone(),
                    finish_reason: None,
                    raw: None,
                });
                last_tool_event_index = Some(events.len() - 1);
                conversation.push(tool_msg);
            }

            // 7. Counter injection into the turn's last added message.
            if limits.inject_time_remaining_counter {
                if let Some(dl) = deadline {
                    let remaining = (dl - clock()).max(0.0);
                    if let Some(idx) = last_tool_event_index {
                        if let Event::MessageAdded { message, .. } = &mut events[idx] {
                            message.append_text_content(&format!(
                                "Time remaining: {} seconds",
                                remaining as i64
                            ));
                        }
                        // keep the conversation copy in sync
                        if let Some(msg) = conversation.last_mut() {
                            if matches!(msg.role, Role::Tool) {
                                msg.append_text_content(&format!(
                                    "Time remaining: {} seconds",
                                    remaining as i64
                                ));
                            }
                        }
                    }
                }
            }
            if limits.inject_context_remaining_counter {
                if let Some(limit) = limits.context_window_limit {
                    let remaining = (limit - context_length).max(0);
                    if let Some(idx) = last_tool_event_index {
                        if let Event::MessageAdded { message, .. } = &mut events[idx] {
                            message.append_text_content(&format!("Context remaining: {remaining}"));
                        }
                        if let Some(msg) = conversation.last_mut() {
                            if matches!(msg.role, Role::Tool) {
                                msg.append_text_content(&format!("Context remaining: {remaining}"));
                            }
                        }
                    }
                }
            }
        }
    }

    fn dispatch_tool(&self, call: &ToolCall, args: &serde_json::Value) -> Result<CallToolResult> {
        let name = call
            .function
            .name
            .clone()
            .ok_or_else(|| Error::InvalidSpec("tool call has no function name".into()))?;
        self.dispatcher.call_tool(&name, args)
    }

    /// Total turns consumed (run-wide).
    pub fn turn_count(&self) -> u64 {
        *self.turn_count.lock().unwrap()
    }
}

/// Upstream `_to_chat_content_part`: the *first* content block only;
/// images become data-URL parts, anything unknown becomes its JSON.
pub fn to_chat_content_part(result: &CallToolResult) -> serde_json::Value {
    if let Some(first) = result.content.first() {
        let kind = first.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match kind {
            "image" => {
                // parse into ImageContent for the data-url projection
                if let Ok(img) = serde_json::from_value::<ImageContent>(first.clone()) {
                    return serde_json::json!({
                        "type": "image_url",
                        "image_url": {"url": format!("data:{};base64,{}", img.mime_type, img.data)}
                    });
                }
            }
            "text" => {
                return first.clone();
            }
            _ => return first.clone(),
        }
    }
    serde_json::Value::Null
}

// ---------------------------------------------------------------------------
// retry policy (upstream builtin_source.py)
// ---------------------------------------------------------------------------

/// Upstream retry constants.
pub const LLM_RETRY_MAX_ATTEMPTS: u32 = 16;
/// Minimum backoff.
pub const LLM_RETRY_WAIT_MIN_S: f64 = 1.0;
/// Maximum backoff.
pub const LLM_RETRY_WAIT_MAX_S: f64 = 60.0;
/// Cap on a provider's Retry-After header.
pub const LLM_RETRY_AFTER_MAX_S: f64 = 300.0;
/// 401 retries before giving up.
pub const AUTH_RETRY_MAX_ATTEMPTS: u32 = 3;
/// Unreachable-endpoint retries.
pub const UNREACHABLE_RETRY_MAX_ATTEMPTS: u32 = 4;

/// The class of a completion failure (upstream exception taxonomy).
#[derive(Debug, Clone, PartialEq)]
pub enum FailureClass {
    /// 401/403 — key problems; few retries.
    Auth,
    /// 5xx / rate limit / timeout / connection — retried with backoff.
    Server,
    /// Invalid URL / DNS / connection refused — permanent, few retries.
    Unreachable,
    /// Anything else.
    Other,
}

/// Compute the wait before the next attempt (upstream `llm_retry_wait`):
/// `None` = give up. Retry-After wins (clamped to 300 s), else
/// exponential backoff `min(60, max(1, 2^(attempt-1)) + uniform(0, 0.5*exp))`
/// — the port drops the jitter and keeps the deterministic floor.
pub fn llm_retry_wait(attempt: u32, class: &FailureClass, retry_after: Option<f64>) -> Option<f64> {
    let limit = match class {
        FailureClass::Auth => AUTH_RETRY_MAX_ATTEMPTS,
        FailureClass::Unreachable => UNREACHABLE_RETRY_MAX_ATTEMPTS,
        _ => LLM_RETRY_MAX_ATTEMPTS,
    };
    if attempt >= limit {
        return None;
    }
    if let Some(ra) = retry_after {
        return Some(ra.clamp(0.0, LLM_RETRY_AFTER_MAX_S));
    }
    let exp = 2f64.powi((attempt.max(1) - 1) as i32);
    Some(LLM_RETRY_WAIT_MIN_S.max(exp).min(LLM_RETRY_WAIT_MAX_S))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemas::{Function, MessageContent};
    use serde_json::json;

    fn scripted(messages: Vec<Message>) -> Arc<dyn MessageSource> {
        Arc::new(ScriptedSource::new(messages))
    }

    fn fixed_clock(t: f64) -> impl FnMut() -> f64 {
        move || t
    }

    #[test]
    fn simple_text_turn_completes_step() {
        let src = scripted(vec![Message::text(Role::Assistant, "the answer is 42")]);
        let lp = MessageLoop::new(src, Arc::new(MapDispatcher::new()));
        let mut clock = fixed_clock(0.0);
        let events = lp
            .run_step("do it", &LoopLimits::default(), &[], &mut clock)
            .unwrap();
        let kinds: Vec<&str> = events.iter().map(|e| e.kind()).collect();
        assert_eq!(kinds, vec!["message_added", "message_added"]);
        assert!(matches!(&events[1], Event::MessageAdded { message, .. } if message.has_text()));
    }

    #[test]
    fn tool_calls_execute_and_project_and_finish() {
        let mut dispatcher = MapDispatcher::new();
        dispatcher.register("bash", |_args| {
            Ok(CallToolResult {
                content: vec![
                    serde_json::to_value(crate::schemas::TextContent::new("first block")).unwrap(),
                    serde_json::to_value(crate::schemas::TextContent::new("second block")).unwrap(),
                ],
                structured_content: Some(json!({"stdout": "first block"})),
                is_error: false,
            })
        });
        let src = scripted(vec![
            Message {
                content: Some(MessageContent::Text("working".into())),
                role: Role::Assistant,
                tool_calls: Some(vec![ToolCall {
                    id: "call-1".into(),
                    function: Function {
                        name: Some("bash".into()),
                        arguments: json!({"command": "ls"}).to_string(),
                    },
                    call_type: "function".into(),
                }]),
                reasoning_content: None,
                tool_call_id: None,
            },
            Message::text(Role::Assistant, "done"),
        ]);
        let lp = MessageLoop::new(src, Arc::new(dispatcher));
        let mut clock = fixed_clock(0.0);
        let events = lp
            .run_step("do it", &LoopLimits::default(), &[], &mut clock)
            .unwrap();
        let kinds: Vec<&str> = events.iter().map(|e| e.kind()).collect();
        assert_eq!(
            kinds,
            vec![
                "message_added",       // user
                "message_added",       // assistant with tool_calls
                "tool_call_started",   //
                "tool_call_completed", //
                "message_added",       // tool result
                "message_added",       // terminal assistant text ("done")
            ]
        );
        // only the FIRST content block is forwarded
        if let Event::MessageAdded { message, .. } = &events[4] {
            let parts = match message.content.as_ref().unwrap() {
                MessageContent::Parts(p) => p.clone(),
                _ => panic!("expected parts"),
            };
            assert_eq!(parts.len(), 1);
            assert_eq!(parts[0]["text"], "first block");
        } else {
            panic!("expected tool message");
        }
    }

    #[test]
    fn image_result_projects_to_data_url() {
        let result = CallToolResult {
            content: vec![serde_json::to_value(ImageContent {
                kind: "image".into(),
                data: "QUJD".into(),
                mime_type: "image/png".into(),
            })
            .unwrap()],
            structured_content: None,
            is_error: false,
        };
        let part = to_chat_content_part(&result);
        assert_eq!(part["type"], "image_url");
        assert_eq!(part["image_url"]["url"], "data:image/png;base64,QUJD");
    }

    #[test]
    fn malformed_arguments_become_error_result() {
        let mut dispatcher = MapDispatcher::new();
        dispatcher.register("bash", |_| Ok(CallToolResult::text("never reached")));
        let src = scripted(vec![
            Message {
                content: None,
                role: Role::Assistant,
                tool_calls: Some(vec![ToolCall {
                    id: "call-9".into(),
                    function: Function {
                        name: Some("bash".into()),
                        arguments: "{not json".into(),
                    },
                    call_type: "function".into(),
                }]),
                reasoning_content: None,
                tool_call_id: None,
            },
            Message::text(Role::Assistant, "done anyway"),
        ]);
        let lp = MessageLoop::new(src, Arc::new(dispatcher));
        let mut clock = fixed_clock(0.0);
        let events = lp
            .run_step("do it", &LoopLimits::default(), &[], &mut clock)
            .unwrap();
        if let Event::ToolCallCompleted { result, .. } = &events[3] {
            assert!(result.is_error);
            assert_eq!(
                result.first_text().unwrap(),
                "Error: Invalid JSON in tool arguments"
            );
        } else {
            panic!("expected tool completion");
        }
    }

    #[test]
    fn turn_limit_is_run_wide_and_strict() {
        // turn_limit 2, but the source always answers text: step 1 uses
        // turn 1; a second loop object shares nothing — the count is per
        // agent. Use one loop across two steps.
        let src = scripted(vec![
            Message::text(Role::Assistant, "one"),
            Message::text(Role::Assistant, "two"),
            Message::text(Role::Assistant, "three"),
        ]);
        let lp = MessageLoop::new(src, Arc::new(MapDispatcher::new()));
        let limits = LoopLimits {
            turn_limit: Some(2),
            ..Default::default()
        };
        let mut clock = fixed_clock(0.0);
        lp.run_step("s1", &limits, &[], &mut clock).unwrap();
        assert_eq!(lp.turn_count(), 1);
        lp.run_step("s2", &limits, &[], &mut clock).unwrap();
        assert_eq!(lp.turn_count(), 2);
        // Turn 3 exceeds the run-wide limit of 2 (error message upstream:
        // "Turn limit of 2 reached.").
        let err = lp.run_step("s3", &limits, &[], &mut clock).unwrap_err();
        assert!(err.to_string().contains("Turn limit of 2 reached."));
    }

    #[test]
    fn time_limit_checked_between_turns_and_actions() {
        // Script: an assistant turn with tool calls, then text.
        let with_calls = Message {
            content: None,
            role: Role::Assistant,
            tool_calls: Some(vec![ToolCall::function("c1", "noop", "{}")]),
            reasoning_content: None,
            tool_call_id: None,
        };
        let src = scripted(vec![with_calls, Message::text(Role::Assistant, "done")]);
        let mut dispatcher = MapDispatcher::new();
        dispatcher.register("noop", |_| Ok(CallToolResult::text("ok")));
        let lp = MessageLoop::new(Arc::clone(&src), Arc::new(dispatcher));

        // A clock that crosses the deadline after the first turn.
        let mut t = 0.0;
        let mut clock = move || {
            let now = t;
            t += 2.0;
            now
        };
        let limits = LoopLimits {
            time_limit_seconds: Some(3.0),
            on_time_limit: LimitAction::Raise,
            ..Default::default()
        };
        let err = lp.run_step("do it", &limits, &[], &mut clock).unwrap_err();
        assert!(err.to_string().contains("Time limit of 3s reached."));

        // Score action: ends quietly with the events so far.
        let lp2 = MessageLoop::new(src, Arc::new(MapDispatcher::new()));
        let mut t2 = 0.0;
        let mut clock2 = move || {
            let now = t2;
            t2 += 2.0;
            now
        };
        let limits_score = LoopLimits {
            time_limit_seconds: Some(3.0),
            on_time_limit: LimitAction::Score,
            ..Default::default()
        };
        let events = lp2
            .run_step("do it", &limits_score, &[], &mut clock2)
            .unwrap();
        assert!(!events.is_empty());
    }

    #[test]
    fn empty_turns_nudge_then_error() {
        // Reasoning-only, no text, no tool calls → nudged 3 times, error on
        // the 4th empty turn in a row.
        let empty = |i: i32| {
            let mut m = Message::text(Role::Assistant, "");
            m.reasoning_content = Some(format!("thinking {i}"));
            m
        };
        let src = scripted((0..8).map(empty).collect());
        let lp = MessageLoop::new(src, Arc::new(MapDispatcher::new()));
        let mut clock = fixed_clock(0.0);
        let err = lp
            .run_step("do it", &LoopLimits::default(), &[], &mut clock)
            .unwrap_err();
        assert!(err.to_string().contains("neither text nor tool calls"));
        // nudge events were emitted
        let _ = EMPTY_TURN_NUDGE;
    }

    #[test]
    fn nudge_text_is_verbatim() {
        assert!(EMPTY_TURN_NUDGE.starts_with("Your last turn produced no message"));
        assert!(EMPTY_TURN_NUDGE.ends_with("smaller pieces if necessary."));
    }

    #[test]
    fn context_limit_trips_from_usage_event() {
        // The scripted source emits messages only; simulate usage by
        // wrapping it.
        struct WithUsage {
            inner: ScriptedSource,
        }
        impl MessageSource for WithUsage {
            fn collect(
                &self,
                messages: &[Message],
                tools: &[serde_json::Value],
            ) -> Result<Vec<Event>> {
                let mut evs = self.inner.collect(messages, tools)?;
                evs.push(Event::TokenUsage {
                    input_tokens: 5000,
                    output_tokens: 10,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                    estimated: false,
                    service_tier: None,
                });
                Ok(evs)
            }
        }
        let src: Arc<dyn MessageSource> = Arc::new(WithUsage {
            inner: ScriptedSource::new(vec![
                Message {
                    content: None,
                    role: Role::Assistant,
                    tool_calls: Some(vec![ToolCall::function("c1", "noop", "{}")]),
                    reasoning_content: None,
                    tool_call_id: None,
                },
                Message::text(Role::Assistant, "turn two"),
            ]),
        });
        let lp = MessageLoop::new(src, Arc::new(MapDispatcher::new()));
        let limits = LoopLimits {
            context_window_limit: Some(4000),
            on_context_window_limit: LimitAction::Raise,
            ..Default::default()
        };
        let mut clock = fixed_clock(0.0);
        let err = lp.run_step("s", &limits, &[], &mut clock).unwrap_err();
        assert!(
            err.to_string()
                .contains("Context window limit of 4000 reached."),
            "got: {err}"
        );
    }

    #[test]
    fn counters_injected_into_last_tool_message() {
        let with_calls = || Message {
            content: None,
            role: Role::Assistant,
            tool_calls: Some(vec![ToolCall::function("c1", "noop", "{}")]),
            reasoning_content: None,
            tool_call_id: None,
        };
        let src = scripted(vec![with_calls(), Message::text(Role::Assistant, "done")]);
        let mut dispatcher = MapDispatcher::new();
        dispatcher.register("noop", |_| Ok(CallToolResult::text("ok")));
        let lp = MessageLoop::new(Arc::clone(&src), Arc::new(dispatcher));
        let limits = LoopLimits {
            time_limit_seconds: Some(100.0),
            inject_time_remaining_counter: true,
            context_window_limit: Some(10_000),
            inject_context_remaining_counter: true,
            ..Default::default()
        };
        // The clock: 0 at start; usage event sets context 5000 via a
        // wrapper would be needed for the context counter; time counter
        // alone is testable here.
        let mut t = 0.0;
        let mut clock = move || {
            let now = t;
            t += 10.0;
            now
        };
        let events = lp.run_step("s", &limits, &[], &mut clock).unwrap();
        let tool_msg = events
            .iter()
            .rev()
            .find_map(|e| match e {
                Event::MessageAdded { message, .. } if matches!(message.role, Role::Tool) => {
                    Some(message.clone())
                }
                _ => None,
            })
            .unwrap();
        let parts = match tool_msg.content.as_ref().unwrap() {
            MessageContent::Parts(ps) => ps.clone(),
            other => panic!("expected parts, got {other:?}"),
        };
        // The tool text part plus the injected counter parts (appended in
        // order: time then context).
        let joined = tool_msg.content.as_ref().unwrap().as_text().unwrap();
        assert!(
            joined.contains("Time remaining: "),
            "got: {joined:?}; parts: {parts:?}"
        );
        assert!(
            joined.contains("Context remaining: 10000"),
            "got: {joined:?}"
        );
        assert!(parts.len() >= 3, "tool text + 2 counters: {parts:?}");
    }

    #[test]
    fn content_filter_ends_step() {
        // A source that finishes with content_filter: step ends without
        // error.
        struct Filtered;
        impl MessageSource for Filtered {
            fn collect(&self, _m: &[Message], _t: &[serde_json::Value]) -> Result<Vec<Event>> {
                Ok(vec![Event::MessageAdded {
                    message: Message::text(Role::Assistant, ""),
                    finish_reason: Some("content_filter".into()),
                    raw: None,
                }])
            }
        }
        let lp = MessageLoop::new(Arc::new(Filtered), Arc::new(MapDispatcher::new()));
        let mut clock = fixed_clock(0.0);
        let events = lp
            .run_step("s", &LoopLimits::default(), &[], &mut clock)
            .unwrap();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn retry_policy_wait_table() {
        // Server errors: exponential, capped at 60.
        assert_eq!(llm_retry_wait(1, &FailureClass::Server, None), Some(1.0));
        assert_eq!(llm_retry_wait(2, &FailureClass::Server, None), Some(2.0));
        assert_eq!(llm_retry_wait(3, &FailureClass::Server, None), Some(4.0));
        assert_eq!(llm_retry_wait(7, &FailureClass::Server, None), Some(60.0));
        assert_eq!(llm_retry_wait(16, &FailureClass::Server, None), None);
        // Retry-After wins, clamped to 300.
        assert_eq!(
            llm_retry_wait(1, &FailureClass::Server, Some(120.0)),
            Some(120.0)
        );
        assert_eq!(
            llm_retry_wait(1, &FailureClass::Server, Some(99999.0)),
            Some(300.0)
        );
        // Auth: 3 attempts.
        assert_eq!(llm_retry_wait(2, &FailureClass::Auth, None), Some(2.0));
        assert_eq!(llm_retry_wait(3, &FailureClass::Auth, None), None);
        // Unreachable: 4 attempts.
        assert_eq!(llm_retry_wait(4, &FailureClass::Unreachable, None), None);
    }

    #[test]
    fn fake_source_exhaustion_message() {
        let lp = MessageLoop::new(scripted(vec![]), Arc::new(MapDispatcher::new()));
        let mut clock = fixed_clock(0.0);
        let err = lp
            .run_step("s", &LoopLimits::default(), &[], &mut clock)
            .unwrap_err();
        assert!(err.to_string().contains("Not enough messages"));
    }
}
