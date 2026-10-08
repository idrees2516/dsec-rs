//! Wire schemas: chat messages, scorings, the full event union, the
//! transcript, run state, and the run configuration.
//!
//! A faithful port of upstream `karotte/schemas/` (`chat.py`, `scoring.py`,
//! `transcript.py`, `run_state.py`, `evaluation_run_config.py`,
//! `http_mcp_server_config.py`, `websocket_config.py`, `data_mount.py`).
//! Field names, the `type` discriminators, and the MCP wire renames
//! (`structuredContent`, `isError`) match the upstream JSON byte-for-byte
//! so transcripts interoperate.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// chat.py
// ---------------------------------------------------------------------------

/// Chat role (upstream `Role`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// The model.
    Assistant,
    /// The user / task instructions.
    User,
    /// The system prompt.
    System,
    /// A tool result message.
    Tool,
    /// Legacy function-call result role.
    Function,
}

/// A function call inside a message (upstream `Function`): the tool name
/// plus its JSON-encoded arguments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Function {
    /// Tool name.
    #[serde(default)]
    pub name: Option<String>,
    /// JSON-encoded arguments string.
    #[serde(default)]
    pub arguments: String,
}

impl Function {
    /// Parse the arguments JSON (upstream `json.loads(arguments or "{}")`).
    pub fn parse_arguments(&self) -> Result<serde_json::Value, serde_json::Error> {
        if self.arguments.trim().is_empty() {
            return Ok(serde_json::Value::Object(Default::default()));
        }
        serde_json::from_str(&self.arguments)
    }
}

/// A complete tool call on a message (upstream
/// `ChatCompletionMessageToolCall`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned call id.
    pub id: String,
    /// The called function.
    pub function: Function,
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub call_type: String,
}

impl ToolCall {
    /// A well-formed function tool call.
    pub fn function(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            function: Function {
                name: Some(name.into()),
                arguments: arguments.into(),
            },
            call_type: "function".into(),
        }
    }
}

/// Message content: plain text or a list of content parts
/// (upstream `Message.content`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text.
    Text(String),
    /// Content parts: `{"type":"text","text":...}` or
    /// `{"type":"image_url","image_url":{"url":"data:...;base64,..."}}`.
    Parts(Vec<serde_json::Value>),
}

impl MessageContent {
    /// Plain-text view: text, or the concatenation of text parts.
    pub fn as_text(&self) -> Option<String> {
        match self {
            MessageContent::Text(s) => Some(s.clone()),
            MessageContent::Parts(parts) => {
                let mut out = String::new();
                for p in parts {
                    if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                        if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                            out.push_str(t);
                        }
                    }
                }
                Some(out)
            }
        }
    }
}

/// A chat message (upstream `Message`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Content (text or parts).
    #[serde(default)]
    pub content: Option<MessageContent>,
    /// Role.
    #[serde(default = "default_role")]
    pub role: Role,
    /// Tool calls issued by the assistant.
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Reasoning content (reasoning models).
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// The tool_call_id this message answers (role=tool).
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

fn default_role() -> Role {
    Role::Assistant
}

impl Message {
    /// A system/user/assistant text message.
    pub fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            content: Some(MessageContent::Text(content.into())),
            role,
            tool_calls: None,
            reasoning_content: None,
            tool_call_id: None,
        }
    }

    /// A tool-result message answering `tool_call_id`.
    pub fn tool_result(tool_call_id: impl Into<String>, parts: Vec<serde_json::Value>) -> Self {
        Self {
            content: if parts.is_empty() {
                None
            } else {
                Some(MessageContent::Parts(parts))
            },
            role: Role::Tool,
            tool_calls: None,
            reasoning_content: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    /// Whether the message carries any non-whitespace text.
    pub fn has_text(&self) -> bool {
        self.content
            .as_ref()
            .and_then(|c| c.as_text())
            .map(|t| !t.trim().is_empty())
            .unwrap_or(false)
    }

    /// Append `\n\n{text}` (upstream `_append_text_content`): `None`
    /// becomes text, text is extended, parts get a trailing text part.
    pub fn append_text_content(&mut self, text: &str) {
        self.content = Some(match self.content.take() {
            None => MessageContent::Text(text.to_string()),
            Some(MessageContent::Text(s)) => {
                if s.is_empty() {
                    MessageContent::Text(text.to_string())
                } else {
                    MessageContent::Text(format!("{s}\n\n{text}"))
                }
            }
            Some(MessageContent::Parts(mut parts)) => {
                parts.push(serde_json::json!({"type": "text", "text": text}));
                MessageContent::Parts(parts)
            }
        });
    }
}

/// A streaming delta (upstream `Delta`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Delta {
    /// Text content fragment.
    #[serde(default)]
    pub content: Option<String>,
    /// Role on the first chunk.
    #[serde(default)]
    pub role: Option<Role>,
    /// Partial tool calls.
    #[serde(default)]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// Reasoning fragment.
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

// ---------------------------------------------------------------------------
// scoring.py
// ---------------------------------------------------------------------------

/// A scoring outcome (upstream frozen dataclass `Scoring`):
/// `score` + `metadata` + `continue_task`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scoring {
    /// The score (weighted rubric sum, regex 0/1, ...).
    pub score: f64,
    /// Free-form judge metadata (bounded by [`crate::text::sanitize_metadata`]).
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    /// Whether the run proceeds to the next step. A `false` ends the task
    /// with status `failed`.
    #[serde(default = "default_true")]
    pub continue_task: bool,
}

fn default_true() -> bool {
    true
}

impl Scoring {
    /// A passing scoring.
    pub fn pass(score: f64) -> Self {
        Self {
            score,
            metadata: BTreeMap::new(),
            continue_task: true,
        }
    }

    /// The misbehavior scoring: score 0, `continue_task=false`, a
    /// `misbehavior` metadata entry (upstream `misbehavior_scoring`).
    pub fn misbehavior(err: &crate::error::StudentMisbehaviorError) -> Self {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "misbehavior".to_string(),
            serde_json::Value::String(err.to_metadata()),
        );
        Self {
            score: 0.0,
            metadata,
            continue_task: false,
        }
    }
}

/// Run status (upstream `RunStatus`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    /// Not started.
    #[default]
    Pending,
    /// Running.
    Running,
    /// All steps continued.
    Passed,
    /// A step's scoring said stop.
    Failed,
    /// An exception escaped.
    Error,
}

// ---------------------------------------------------------------------------
// transcript.py — the event union
// ---------------------------------------------------------------------------

/// A text content block of an MCP tool result
/// (upstream `mcp.types.TextContent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextContent {
    /// Discriminator, always `"text"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The text.
    pub text: String,
}

impl TextContent {
    /// A new text block.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            kind: "text".into(),
            text: text.into(),
        }
    }
}

/// An image content block (upstream `mcp.types.ImageContent`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageContent {
    /// Discriminator, always `"image"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Image data, base64 (no data-url prefix).
    pub data: String,
    /// MIME type.
    #[serde(rename = "mimeType")]
    pub mime_type: String,
}

/// MCP `CallToolResult` (upstream wire names preserved: `content`,
/// `structuredContent`, `isError`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallToolResult {
    /// Content blocks (text/image).
    #[serde(default)]
    pub content: Vec<serde_json::Value>,
    /// Structured (JSON) result.
    #[serde(
        rename = "structuredContent",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub structured_content: Option<serde_json::Value>,
    /// Whether the tool call errored.
    #[serde(
        rename = "isError",
        default,
        skip_serializing_if = "std::ops::Not::not"
    )]
    pub is_error: bool,
}

impl CallToolResult {
    /// A text-only result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![serde_json::to_value(TextContent::new(text)).unwrap()],
            structured_content: None,
            is_error: false,
        }
    }

    /// An error result (upstream `_build_call_tool_result(..., is_error=True)`):
    /// text block + `structuredContent` mirror + `isError`.
    pub fn error_text(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            content: vec![serde_json::to_value(TextContent::new(text.clone())).unwrap()],
            structured_content: Some(serde_json::json!({"result": text})),
            is_error: true,
        }
    }

    /// The first text block's text, if any.
    pub fn first_text(&self) -> Option<String> {
        self.content
            .first()
            .and_then(|b| b.get("text"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
    }
}

/// Resource sample (upstream `ResourceSample`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceSample {
    /// Milliseconds since epoch.
    pub timestamp_ms: i64,
    /// CPU percent, 0–100 (normalized across cgroup CPUs).
    pub cpu_percent: f64,
    /// Memory MB.
    pub memory_mb: f64,
}

/// Aggregated resource metrics (upstream `ResourceMetrics`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// All samples.
    #[serde(default)]
    pub samples: Vec<ResourceSample>,
    /// Peak CPU %.
    #[serde(default)]
    pub peak_cpu_percent: f64,
    /// Average CPU %.
    #[serde(default)]
    pub avg_cpu_percent: f64,
    /// Peak memory MB.
    #[serde(default)]
    pub peak_memory_mb: f64,
    /// Average memory MB.
    #[serde(default)]
    pub avg_memory_mb: f64,
}

impl ResourceMetrics {
    /// Aggregate peaks/averages over `samples` (upstream `_aggregate`).
    pub fn aggregate(samples: &[ResourceSample]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let n = samples.len() as f64;
        let (peak_cpu, avg_cpu, peak_mem, avg_mem) =
            samples
                .iter()
                .fold((0.0f64, 0.0f64, 0.0f64, 0.0f64), |(pc, ac, pm, am), s| {
                    (
                        pc.max(s.cpu_percent),
                        ac + s.cpu_percent / n,
                        pm.max(s.memory_mb),
                        am + s.memory_mb / n,
                    )
                });
        Self {
            samples: samples.to_vec(),
            peak_cpu_percent: peak_cpu,
            avg_cpu_percent: avg_cpu,
            peak_memory_mb: peak_mem,
            avg_memory_mb: avg_mem,
        }
    }
}

/// The event union. Tagged by `type`, snake_case, field names and
/// orderings identical to upstream `transcript.py`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// `task_started`
    #[serde(rename_all = "snake_case")]
    TaskStarted {
        /// Run id.
        run_id: String,
        /// Task id.
        task_id: String,
        /// Model id, if any.
        #[serde(default)]
        model: Option<String>,
        /// Step count (or -1 when unmeasurable).
        n_steps: i64,
        /// The applied reasoning effort, if any.
        #[serde(default)]
        reasoning_effort: Option<String>,
    },
    /// `task_pre_hook_completed`
    #[serde(rename_all = "snake_case")]
    TaskPreHookCompleted {
        /// Metadata returned by the task's pre-hook.
        #[serde(default)]
        metadata: BTreeMap<String, serde_json::Value>,
    },
    /// `step_started`
    #[serde(rename_all = "snake_case")]
    StepStarted {
        /// 0-indexed step.
        step: usize,
    },
    /// `step_completed`
    #[serde(rename_all = "snake_case")]
    StepCompleted {
        /// 0-indexed step.
        step: usize,
    },
    /// `message_chunk` (streaming; never persisted in the transcript).
    #[serde(rename_all = "snake_case")]
    MessageChunk {
        /// The delta.
        delta: Delta,
    },
    /// `message_chunk_reset` (streaming retry marker).
    #[serde(rename_all = "snake_case")]
    MessageChunkReset {},
    /// `message_added`
    #[serde(rename_all = "snake_case")]
    MessageAdded {
        /// The completed message.
        message: Message,
        /// Finish reason (stop / length / tool_calls / content_filter ...).
        #[serde(default)]
        finish_reason: Option<String>,
        /// Agent-native passthrough payload.
        #[serde(default)]
        raw: Option<serde_json::Value>,
    },
    /// `tool_call_started`
    #[serde(rename_all = "snake_case")]
    ToolCallStarted {
        /// The tool call.
        tool_call: ToolCall,
    },
    /// `tool_call_completed`
    #[serde(rename_all = "snake_case")]
    ToolCallCompleted {
        /// The answered tool-call id.
        tool_call_id: String,
        /// The MCP result.
        result: CallToolResult,
    },
    /// `answers_submitted`
    #[serde(rename_all = "snake_case")]
    AnswersSubmitted {
        /// Answer key/values.
        answers: BTreeMap<String, String>,
    },
    /// `scoring`
    #[serde(rename_all = "snake_case")]
    Scoring {
        /// The scoring.
        scoring: Scoring,
        /// Resource metrics sampled during scoring (when profiling).
        #[serde(default)]
        resource_metrics: Option<ResourceMetrics>,
    },
    /// `error`
    #[serde(rename_all = "snake_case")]
    Error {
        /// Exception type name.
        exception_type: String,
        /// Message.
        message: String,
        /// Traceback, if available.
        #[serde(default)]
        traceback: Option<String>,
    },
    /// `task_completed`
    #[serde(rename_all = "snake_case")]
    TaskCompleted {
        /// Final status.
        status: RunStatus,
    },
    /// `token_usage`
    #[serde(rename_all = "snake_case")]
    TokenUsage {
        /// Prompt tokens.
        input_tokens: i64,
        /// Completion tokens.
        output_tokens: i64,
        /// Cache-read tokens, when reported.
        #[serde(default)]
        cache_read_tokens: Option<i64>,
        /// Cache-write tokens, when reported.
        #[serde(default)]
        cache_write_tokens: Option<i64>,
        /// Estimated (not provider-reported) usage.
        #[serde(default)]
        estimated: bool,
        /// Provider service tier.
        #[serde(default)]
        service_tier: Option<String>,
    },
}

impl Event {
    /// The upstream `type` discriminator string.
    pub fn kind(&self) -> &'static str {
        match self {
            Event::TaskStarted { .. } => "task_started",
            Event::TaskPreHookCompleted { .. } => "task_pre_hook_completed",
            Event::StepStarted { .. } => "step_started",
            Event::StepCompleted { .. } => "step_completed",
            Event::MessageChunk { .. } => "message_chunk",
            Event::MessageChunkReset { .. } => "message_chunk_reset",
            Event::MessageAdded { .. } => "message_added",
            Event::ToolCallStarted { .. } => "tool_call_started",
            Event::ToolCallCompleted { .. } => "tool_call_completed",
            Event::AnswersSubmitted { .. } => "answers_submitted",
            Event::Scoring { .. } => "scoring",
            Event::Error { .. } => "error",
            Event::TaskCompleted { .. } => "task_completed",
            Event::TokenUsage { .. } => "token_usage",
        }
    }

    /// Whether this is a streaming chunk event (excluded from the
    /// persisted transcript and the backend stream).
    pub fn is_chunk(&self) -> bool {
        matches!(
            self,
            Event::MessageChunk { .. } | Event::MessageChunkReset { .. }
        )
    }

    /// Extract scoring fields (`(score, continue_task)`) if this is a
    /// `Scoring` event.
    pub fn as_scoring(&self) -> Option<(&Scoring, Option<&ResourceMetrics>)> {
        match self {
            Event::Scoring {
                scoring,
                resource_metrics,
            } => Some((scoring, resource_metrics.as_ref())),
            _ => None,
        }
    }
}

/// The transcript: run id + append-only events (upstream `Transcript`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    /// Run id.
    #[serde(default)]
    pub run_id: String,
    /// Events, in order. Chunk events are excluded upstream; this type
    /// keeps whatever is appended — the runner filters.
    #[serde(default)]
    pub events: Vec<Event>,
}

impl Transcript {
    /// All messages added so far (upstream `Transcript.messages`).
    pub fn messages(&self) -> Vec<&Message> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::MessageAdded { message, .. } => Some(message),
                _ => None,
            })
            .collect()
    }

    /// Merged answers from all `answers_submitted` events (later wins).
    pub fn answers(&self) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        for e in &self.events {
            if let Event::AnswersSubmitted { answers } = e {
                for (k, v) in answers {
                    out.insert(k.clone(), v.clone());
                }
            }
        }
        out
    }

    /// Tool calls paired with their results, in order
    /// (upstream `TranscriptContext(tool=...)` pairing).
    pub fn tool_exchange(&self) -> Vec<(ToolCall, Option<&CallToolResult>)> {
        let mut out: Vec<(ToolCall, Option<&CallToolResult>)> = Vec::new();
        for e in &self.events {
            match e {
                Event::ToolCallStarted { tool_call } => {
                    out.push((tool_call.clone(), None));
                }
                Event::ToolCallCompleted {
                    tool_call_id,
                    result,
                } => {
                    if let Some(slot) = out
                        .iter_mut()
                        .rev()
                        .find(|(tc, res)| tc.id == *tool_call_id && res.is_none())
                    {
                        slot.1 = Some(result);
                    }
                }
                _ => {}
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// run_state.py
// ---------------------------------------------------------------------------

/// Live run state, updated event-by-event (upstream `RunState`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunState {
    /// Run id.
    #[serde(default)]
    pub run_id: String,
    /// Task id.
    #[serde(default)]
    pub task_id: String,
    /// Status.
    #[serde(default)]
    pub status: RunStatus,
    /// Start time (ISO-8601 or free-form).
    #[serde(default)]
    pub start_time: Option<String>,
    /// Latest score.
    #[serde(default)]
    pub score: Option<f64>,
    /// Total step count.
    #[serde(default)]
    pub n_steps: i64,
    /// Current step (0-indexed).
    #[serde(default)]
    pub current_step: i64,
    /// Task/run metadata.
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    /// Summed prompt tokens.
    #[serde(default)]
    pub total_input_tokens: Option<i64>,
    /// Summed completion tokens.
    #[serde(default)]
    pub total_output_tokens: Option<i64>,
    /// Summed cache-read tokens.
    #[serde(default)]
    pub total_cache_read_tokens: Option<i64>,
    /// Summed cache-write tokens.
    #[serde(default)]
    pub total_cache_write_tokens: Option<i64>,
}

impl RunState {
    /// Whether the run is over (upstream `is_terminal`).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            RunStatus::Passed | RunStatus::Failed | RunStatus::Error
        )
    }

    /// Seed from a `task_started` event (upstream `__init__(event)`).
    pub fn from_task_started(run_id: &str, task_id: &str, n_steps: i64) -> Self {
        Self {
            run_id: run_id.to_string(),
            task_id: task_id.to_string(),
            status: RunStatus::Running,
            start_time: None,
            score: None,
            n_steps,
            current_step: 0,
            metadata: BTreeMap::new(),
            total_input_tokens: None,
            total_output_tokens: None,
            total_cache_read_tokens: None,
            total_cache_write_tokens: None,
        }
    }

    /// Apply an event (upstream `RunState.apply`).
    pub fn apply(&mut self, event: &Event) {
        match event {
            Event::TaskStarted {
                run_id,
                task_id,
                n_steps,
                ..
            } => {
                self.run_id = run_id.clone();
                self.task_id = task_id.clone();
                self.status = RunStatus::Running;
                self.n_steps = *n_steps;
                self.current_step = 0;
            }
            Event::TaskPreHookCompleted { metadata } => {
                for (k, v) in metadata {
                    self.metadata.insert(k.clone(), v.clone());
                }
            }
            Event::AnswersSubmitted { answers } => {
                for (k, v) in answers {
                    self.metadata
                        .insert(k.clone(), serde_json::Value::String(v.clone()));
                }
            }
            Event::StepStarted { step } => self.current_step = *step as i64,
            Event::Scoring { scoring, .. } => self.score = Some(scoring.score),
            Event::TaskCompleted { status } => self.status = *status,
            Event::TokenUsage {
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_write_tokens,
                ..
            } => {
                *self.total_input_tokens.get_or_insert(0) += input_tokens;
                *self.total_output_tokens.get_or_insert(0) += output_tokens;
                if let Some(cr) = cache_read_tokens {
                    *self.total_cache_read_tokens.get_or_insert(0) += cr;
                }
                if let Some(cw) = cache_write_tokens {
                    *self.total_cache_write_tokens.get_or_insert(0) += cw;
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// http_mcp_server_config.py / websocket_config.py / data_mount.py
// ---------------------------------------------------------------------------

/// HTTP MCP server config (upstream `HttpMcpServerConfig`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HttpMcpServerConfig {
    /// Bind host.
    #[serde(default = "default_mcp_host")]
    pub host: String,
    /// Bind port (upstream default 8080).
    #[serde(default = "default_mcp_port")]
    pub port: u16,
    /// Wrap tool calls and scoring in the resource sampler.
    #[serde(default)]
    pub profile_tool_calls: bool,
}

fn default_mcp_host() -> String {
    "0.0.0.0".into()
}
fn default_mcp_port() -> u16 {
    8080
}

impl Default for HttpMcpServerConfig {
    fn default() -> Self {
        Self {
            host: default_mcp_host(),
            port: default_mcp_port(),
            profile_tool_calls: false,
        }
    }
}

impl HttpMcpServerConfig {
    /// The URL the *client* connects to (upstream `client_url`): wildcard
    /// hosts map to loopback, path `/mcp`.
    pub fn client_url(&self) -> String {
        let host = match self.host.as_str() {
            "0.0.0.0" | "::" | "" => "127.0.0.1",
            h => h,
        };
        format!("http://{host}:{}/mcp", self.port)
    }
}

/// WebSocket config (upstream `WebSocketConfig`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebSocketConfig {
    /// Bind host.
    #[serde(default = "default_ws_host")]
    pub host: String,
    /// Bind port (upstream default 8001).
    #[serde(default = "default_ws_port")]
    pub port: u16,
}

fn default_ws_host() -> String {
    "0.0.0.0".into()
}
fn default_ws_port() -> u16 {
    8001
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            host: default_ws_host(),
            port: default_ws_port(),
        }
    }
}

/// A pinned read-only data mount (upstream `DataMount`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataMount {
    /// Dataset name.
    pub name: String,
    /// Pinned version.
    pub version: String,
    /// Where it appears in the container.
    pub container_path: String,
    /// Mount mode — only `read` exists upstream.
    #[serde(default = "default_mount_type")]
    pub mount_type: String,
}

fn default_mount_type() -> String {
    "read".into()
}

// ---------------------------------------------------------------------------
// evaluation_run_config.py
// ---------------------------------------------------------------------------

/// Behavior when a step limit trips (upstream literals).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LimitAction {
    /// Raise: the run errors out.
    Error,
    /// End the step and score it.
    Score,
}

/// Reasoning effort selector (upstream `ReasoningEffort` levels are
/// provider-specific; `min`/`max` resolve to the model's bounds).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ReasoningEffort {
    /// `min` — the model's lowest level.
    Min,
    /// `max` — the model's highest level.
    Max,
    /// A provider-native level name.
    Level(String),
}

/// The evaluation run configuration (upstream `EvaluationRunConfig`,
/// trimmed to the fields the Rust harness drives).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationRunConfig {
    /// Run id.
    pub run_id: String,
    /// Task id.
    pub task_id: String,
    /// Explicit agent name (None → auto).
    #[serde(default)]
    pub agent: Option<String>,
    /// Model id.
    pub model: String,
    /// Model API key.
    #[serde(default)]
    pub model_api_key: Option<String>,
    /// Override for the rubric-judge model.
    #[serde(default)]
    pub rubric_judge_model: Option<String>,
    /// Override for the rubric-judge key.
    #[serde(default)]
    pub rubric_judge_api_key: Option<String>,
    /// Whether hints are enabled.
    #[serde(default = "default_true")]
    pub use_hints: bool,
    /// Requested reasoning effort.
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Run-wide turn limit.
    #[serde(default)]
    pub turn_limit: Option<u64>,
    /// Per-step time limit (single or per-step list), seconds.
    #[serde(default)]
    pub step_time_limit_seconds: Option<serde_json::Value>,
    /// Time-limit action.
    #[serde(default = "default_error_action")]
    pub on_step_time_limit: LimitAction,
    /// Inject "Time remaining: N seconds" counters.
    #[serde(default = "default_true")]
    pub inject_time_remaining_counter: bool,
    /// Per-step context window limit (single or list), tokens.
    #[serde(default)]
    pub step_context_window_limit: Option<serde_json::Value>,
    /// Context-limit action.
    #[serde(default = "default_error_action")]
    pub on_step_context_window_limit: LimitAction,
    /// Inject "Context remaining: N" counters.
    #[serde(default = "default_true")]
    pub inject_context_remaining_counter: bool,
    /// MCP server config.
    #[serde(default)]
    pub mcp_server_config: HttpMcpServerConfig,
    /// WebSocket config.
    #[serde(default)]
    pub websocket_config: WebSocketConfig,
    /// Where the transcript is written (durable).
    #[serde(default)]
    pub transcript_file: Option<String>,
    /// Replace the model with scripted messages.
    #[serde(default)]
    pub use_fake_model: bool,
    /// Extra env/task config.
    #[serde(default)]
    pub extra_config: Option<serde_json::Value>,
    /// Training-backend URI (external agent).
    #[serde(default)]
    pub backend_uri: Option<String>,
    /// Whether artifacts are saved.
    #[serde(default = "default_true")]
    pub save_artifacts: bool,
    /// Replace step instructions entirely (str or per-step list).
    #[serde(default)]
    pub task_instructions_override: Option<serde_json::Value>,
    /// Append to step instructions (str or per-step list).
    #[serde(default)]
    pub extra_task_instructions: Option<serde_json::Value>,
    /// Extra artifact paths (str or list).
    #[serde(default)]
    pub extra_artifact_paths: Option<serde_json::Value>,
}

fn default_error_action() -> LimitAction {
    LimitAction::Error
}

impl EvaluationRunConfig {
    /// A minimal config for `run_id`/`task_id`/`model`.
    pub fn new(
        run_id: impl Into<String>,
        task_id: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            task_id: task_id.into(),
            agent: None,
            model: model.into(),
            model_api_key: None,
            rubric_judge_model: None,
            rubric_judge_api_key: None,
            use_hints: true,
            reasoning_effort: None,
            turn_limit: None,
            step_time_limit_seconds: None,
            on_step_time_limit: LimitAction::Error,
            inject_time_remaining_counter: true,
            step_context_window_limit: None,
            on_step_context_window_limit: LimitAction::Error,
            inject_context_remaining_counter: true,
            mcp_server_config: HttpMcpServerConfig::default(),
            websocket_config: WebSocketConfig::default(),
            transcript_file: None,
            use_fake_model: false,
            extra_config: None,
            backend_uri: None,
            save_artifacts: true,
            task_instructions_override: None,
            extra_task_instructions: None,
            extra_artifact_paths: None,
        }
    }

    /// The resolved agent (upstream `resolved_agent`): explicit wins;
    /// training checkpoints (`pt/...`, `__karotte_special__/training`)
    /// route to the external backend; otherwise `builtin`.
    pub fn resolved_agent(&self) -> String {
        if let Some(a) = &self.agent {
            return a.clone();
        }
        if self.model.starts_with("pt/") || self.model == "__karotte_special__/training" {
            "external".into()
        } else {
            "builtin".into()
        }
    }

    /// Resolve the per-step time limit (single value or indexed list;
    /// out-of-range → `None` = unlimited).
    pub fn resolve_step_time_limit(&self, index: usize) -> Option<f64> {
        resolve_value_or_list(&self.step_time_limit_seconds, index).and_then(|v| v.as_f64())
    }

    /// Resolve the per-step context-window limit.
    pub fn resolve_step_context_window_limit(&self, index: usize) -> Option<i64> {
        resolve_value_or_list(&self.step_context_window_limit, index).and_then(|v| v.as_i64())
    }

    /// Resolve step instructions (upstream `resolve_step_instructions`):
    /// an override replaces entirely; extras append with `\n\n`.
    pub fn resolve_step_instructions(&self, original: &str, index: usize) -> String {
        if let Some(ov) = resolve_value_or_list(&self.task_instructions_override, index)
            .and_then(|v| v.as_str().map(|s| s.to_string()))
        {
            return ov;
        }
        if let Some(extra) = resolve_value_or_list(&self.extra_task_instructions, index)
            .and_then(|v| v.as_str().map(|s| s.to_string()))
        {
            if !extra.is_empty() {
                return format!("{original}\n\n{extra}");
            }
        }
        original.to_string()
    }

    /// Extra artifact paths (upstream `extra_artifact_paths`).
    pub fn extra_artifact_paths(&self) -> Vec<String> {
        match &self.extra_artifact_paths {
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            _ => Vec::new(),
        }
    }
}

fn resolve_value_or_list(v: &Option<serde_json::Value>, index: usize) -> Option<serde_json::Value> {
    match v {
        None => None,
        Some(serde_json::Value::Null) => None,
        Some(single @ serde_json::Value::String(_))
        | Some(single @ serde_json::Value::Number(_)) => Some(single.clone()),
        Some(serde_json::Value::Array(items)) => items.get(index).cloned(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_append_text_content_all_shapes() {
        let mut m = Message::text(Role::Assistant, "base");
        m.append_text_content("added");
        assert_eq!(
            m.content.as_ref().unwrap().as_text().unwrap(),
            "base\n\nadded"
        );

        let mut none = Message {
            content: None,
            role: Role::Assistant,
            tool_calls: None,
            reasoning_content: None,
            tool_call_id: None,
        };
        none.append_text_content("first");
        assert_eq!(none.content.as_ref().unwrap().as_text().unwrap(), "first");

        let mut parts = Message::tool_result(
            "id",
            vec![serde_json::json!({"type": "text", "text": "part"})],
        );
        parts.append_text_content("tail");
        match parts.content.as_ref().unwrap() {
            MessageContent::Parts(ps) => {
                assert_eq!(ps.len(), 2);
                assert_eq!(ps[1]["text"], "tail");
            }
            _ => panic!("expected parts"),
        }
    }

    #[test]
    fn event_roundtrip_preserves_wire_names() {
        let ev = Event::ToolCallCompleted {
            tool_call_id: "call_1".into(),
            result: CallToolResult::error_text("Error: Invalid JSON in tool arguments"),
        };
        let s = serde_json::to_string(&ev).unwrap();
        assert!(s.contains(r#""type":"tool_call_completed""#));
        assert!(s.contains("structuredContent"));
        assert!(s.contains("isError"));
        let back: Event = serde_json::from_str(&s).unwrap();
        assert_eq!(back, ev);
    }

    #[test]
    fn call_tool_result_wire_shape() {
        let r = CallToolResult::text("hello");
        let s = serde_json::to_string(&r).unwrap();
        // serde_json Value objects iterate in BTree order ("text" < "type");
        // the parts carry both keys, order-insensitively.
        assert!(s.contains(r#""type":"text""#));
        assert!(s.contains(r#""text":"hello""#));
        assert!(!s.contains("isError"), "default omitted");
        let e = CallToolResult::error_text("bad");
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains(r#""isError":true"#));
        assert!(s.contains(r#""structuredContent":{"result":"bad"}"#));
    }

    #[test]
    fn transcript_answers_merge_later_wins() {
        let mut t = Transcript {
            run_id: "r".into(),
            events: vec![
                Event::AnswersSubmitted {
                    answers: BTreeMap::from([("a".into(), "1".into())]),
                },
                Event::AnswersSubmitted {
                    answers: BTreeMap::from([("a".into(), "2".into()), ("b".into(), "9".into())]),
                },
            ],
        };
        assert_eq!(t.answers().get("a").map(String::as_str), Some("2"));
        t.events.clear();
        assert!(t.answers().is_empty());
    }

    #[test]
    fn transcript_tool_exchange_pairs_by_id() {
        let t = Transcript {
            run_id: "r".into(),
            events: vec![
                Event::ToolCallStarted {
                    tool_call: ToolCall::function("id1", "bash", r#"{"command":"ls"}"#),
                },
                Event::ToolCallCompleted {
                    tool_call_id: "id1".into(),
                    result: CallToolResult::text("file"),
                },
            ],
        };
        let ex = t.tool_exchange();
        assert_eq!(ex.len(), 1);
        assert!(ex[0].1.is_some());
        assert_eq!(ex[0].1.unwrap().first_text().unwrap(), "file");
    }

    #[test]
    fn run_state_apply_sums_tokens_and_tracks() {
        let mut rs = RunState::from_task_started("r", "t", 2);
        assert_eq!(rs.status, RunStatus::Running);
        rs.apply(&Event::StepStarted { step: 1 });
        rs.apply(&Event::Scoring {
            scoring: Scoring::pass(0.5),
            resource_metrics: None,
        });
        rs.apply(&Event::TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_tokens: Some(2),
            cache_write_tokens: None,
            estimated: false,
            service_tier: None,
        });
        rs.apply(&Event::TaskCompleted {
            status: RunStatus::Passed,
        });
        assert_eq!(rs.current_step, 1);
        assert_eq!(rs.score, Some(0.5));
        assert_eq!(rs.total_input_tokens, Some(10));
        assert_eq!(rs.total_cache_read_tokens, Some(2));
        assert_eq!(rs.total_cache_write_tokens, None);
        assert!(rs.is_terminal());
    }

    #[test]
    fn mcp_client_url_maps_wildcard_to_loopback() {
        let c = HttpMcpServerConfig::default();
        assert_eq!(c.client_url(), "http://127.0.0.1:8080/mcp");
        let c2 = HttpMcpServerConfig {
            host: "10.0.0.5".into(),
            port: 9000,
            profile_tool_calls: false,
        };
        assert_eq!(c2.client_url(), "http://10.0.0.5:9000/mcp");
    }

    #[test]
    fn run_config_resolvers() {
        let mut c = EvaluationRunConfig::new("r", "t", "claude-opus-5-5");
        assert_eq!(c.resolved_agent(), "builtin");
        c.model = "pt/checkpoint-42".into();
        assert_eq!(c.resolved_agent(), "external");

        c.step_time_limit_seconds = Some(serde_json::json!(30.0));
        assert_eq!(c.resolve_step_time_limit(99), Some(30.0));
        c.step_time_limit_seconds = Some(serde_json::json!([10.0, 20.0]));
        assert_eq!(c.resolve_step_time_limit(0), Some(10.0));
        assert_eq!(c.resolve_step_time_limit(1), Some(20.0));
        assert_eq!(c.resolve_step_time_limit(2), None);

        c.extra_task_instructions = Some(serde_json::json!("extra bits"));
        assert_eq!(
            c.resolve_step_instructions("do it", 0),
            "do it\n\nextra bits"
        );
        c.task_instructions_override = Some(serde_json::json!(["one", "two"]));
        assert_eq!(c.resolve_step_instructions("do it", 1), "two");
    }

    #[test]
    fn misbehavior_scoring_shape() {
        let s = Scoring::misbehavior(&crate::error::StudentMisbehaviorError::Symlink {
            path: "/workdir/answer".into(),
        });
        assert_eq!(s.score, 0.0);
        assert!(!s.continue_task);
        assert!(s.metadata.contains_key("misbehavior"));
        let m = s.metadata["misbehavior"].as_str().unwrap();
        assert!(m.contains("symlink"));
        assert!(m.contains("/workdir/answer"));
    }

    #[test]
    fn resource_metrics_aggregate() {
        let samples = vec![
            ResourceSample {
                timestamp_ms: 0,
                cpu_percent: 10.0,
                memory_mb: 100.0,
            },
            ResourceSample {
                timestamp_ms: 200,
                cpu_percent: 50.0,
                memory_mb: 300.0,
            },
        ];
        let m = ResourceMetrics::aggregate(&samples);
        assert_eq!(m.peak_cpu_percent, 50.0);
        assert_eq!(m.avg_cpu_percent, 30.0);
        assert_eq!(m.peak_memory_mb, 300.0);
        assert_eq!(m.avg_memory_mb, 200.0);
    }
}
