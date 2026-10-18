//! An OpenAI-compatible mock model server for tests, examples, and
//! drills — served by axum over a real TCP socket, so the
//! [`crate::model::OpenAiCompatClient`] exercises its full HTTP path
//! (connect, headers, idempotency key, parse, retries).
//!
//! Two responders ship:
//! - [`ScriptedResponder`] — pops a fixed response queue (deterministic
//!   drills);
//! - [`TaskSolverResponder`] — a *rule policy* that looks at the
//!   transcript: if its tool call hasn't been answered yet, issue it;
//!   once the observation is present, emit the final answer. This
//!   drives realistic multi-turn rollouts without a real LLM.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use crate::model::{ChatRequest, ChatResponse, ToolCallOut};

/// Awaiting alias for axum types (feature-gated).
use axum::extract::State;
use axum::routing::post;
use axum::Router;

/// How the mock decides what to answer.
pub trait Responder: Send + Sync {
    /// Produces the response for one request.
    fn respond(&self, req: &ChatRequest) -> ChatResponse;
}

/// Pops scripted responses in order; exhausted → empty final answer.
pub struct ScriptedResponder {
    queue: Mutex<VecDeque<ChatResponse>>,
}

impl ScriptedResponder {
    /// New responder over a response sequence.
    pub fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            queue: Mutex::new(responses.into()),
        }
    }
}

impl Responder for ScriptedResponder {
    fn respond(&self, _req: &ChatRequest) -> ChatResponse {
        self.queue
            .lock()
            .map(|mut q| q.pop_front().unwrap_or_default())
            .unwrap_or_default()
    }
}

/// A deterministic rule policy: call `tool` with `params`, then answer
/// `final_text` once the tool observation arrived.
pub struct TaskSolverResponder {
    call: ToolCallOut,
    final_text: String,
}

impl TaskSolverResponder {
    /// New solver: one tool call then a final answer.
    pub fn new(tool: &str, params: Value, final_text: impl Into<String>) -> Self {
        Self {
            call: ToolCallOut {
                id: "call_1".into(),
                name: tool.into(),
                arguments: params.to_string(),
            },
            final_text: final_text.into(),
        }
    }
}

impl Responder for TaskSolverResponder {
    fn respond(&self, req: &ChatRequest) -> ChatResponse {
        // has the observation for our call id arrived?
        let answered = req
            .messages
            .iter()
            .any(|m| m.role == "tool" && m.tool_call_id.as_deref() == Some(&self.call.id));
        if answered {
            ChatResponse {
                text: self.final_text.clone(),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage_tokens: 12,
                served_model: "mock-task-solver".into(),
            }
        } else {
            ChatResponse {
                text: String::new(),
                tool_calls: vec![self.call.clone()],
                finish_reason: "tool_calls".into(),
                usage_tokens: 8,
                served_model: "mock-task-solver".into(),
            }
        }
    }
}

#[derive(Clone)]
struct ServerState {
    responder: Arc<dyn Responder>,
}

/// Runs the mock server on `addr` (0 for an ephemeral port); returns
/// the bound address and the join handle.
pub async fn serve(
    addr: std::net::SocketAddr,
    responder: Arc<dyn Responder>,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let state = ServerState { responder };
    let app = Router::new()
        .route("/v1/chat/completions", post(completions))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind mock model server");
    let bound = listener.local_addr().expect("local addr");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("mock model server");
    });
    (bound, handle)
}

async fn completions(State(state): State<ServerState>, body: String) -> String {
    let req = parse_request(&body);
    let resp = match req {
        Some(req) => state.responder.respond(&req),
        None => ChatResponse::default(),
    };
    // OpenAI-compatible envelope
    json!({
        "model": resp.served_model,
        "choices": [{
            "message": {
                "content": if resp.text.is_empty() { Value::Null } else { json!(resp.text) },
                "tool_calls": resp.tool_calls.iter().map(|tc| json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                })).collect::<Vec<_>>(),
            },
            "finish_reason": resp.finish_reason,
        }],
        "usage": {"total_tokens": resp.usage_tokens},
    })
    .to_string()
}

fn parse_request(body: &str) -> Option<ChatRequest> {
    let v: Value = serde_json::from_str(body).ok()?;
    let messages = v["messages"]
        .as_array()?
        .iter()
        .filter_map(|m| {
            let role = m["role"].as_str()?.to_string();
            let content = m["content"].as_str().unwrap_or_default().to_string();
            let tool_calls: Vec<ToolCallOut> = m["tool_calls"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|tc| ToolCallOut {
                            id: tc["id"].as_str().unwrap_or_default().to_string(),
                            name: tc["function"]["name"]
                                .as_str()
                                .unwrap_or_default()
                                .to_string(),
                            arguments: tc["function"]["arguments"]
                                .as_str()
                                .unwrap_or_default()
                                .to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            let tool_call_id = m["tool_call_id"].as_str().map(|s| s.to_string());
            Some(crate::model::ChatMessage {
                role,
                content,
                tool_calls,
                tool_call_id,
            })
        })
        .collect();
    Some(ChatRequest {
        model: v["model"].as_str().unwrap_or_default().to_string(),
        messages,
        tools: v["tools"].as_array().cloned().unwrap_or_default(),
        temperature: 0.2,
        max_tokens: 1024,
        idempotency_key: String::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ChatMessage, ModelClient, OpenAiCompatClient};

    #[tokio::test]
    async fn full_http_roundtrip_task_solver() {
        let responder = Arc::new(TaskSolverResponder::new(
            "crm.get_customer",
            json!({"id": "C1"}),
            "The customer is active.",
        ));
        let (addr, _h) = serve(std::net::SocketAddr::from(([127, 0, 0, 1], 0)), responder).await;
        let client = OpenAiCompatClient::new(&format!("http://{}", addr), "mock").unwrap();

        // first turn: the model wants its tool call
        let req1 = ChatRequest {
            model: "mock".into(),
            messages: vec![
                ChatMessage::new("system", "sys"),
                ChatMessage::new("user", "task"),
            ],
            tools: vec![],
            temperature: 0.2,
            max_tokens: 64,
            idempotency_key: "k1".into(),
        };
        let r1 = client.complete(req1).await.unwrap();
        assert_eq!(r1.tool_calls.len(), 1);
        assert_eq!(r1.tool_calls[0].name, "crm.get_customer");

        // second turn: observation present → final answer
        let req2 = ChatRequest {
            model: "mock".into(),
            messages: vec![
                ChatMessage::new("system", "sys"),
                ChatMessage::new("user", "task"),
                ChatMessage::assistant("", r1.tool_calls.clone()),
                ChatMessage::tool_observation("call_1", "{\"status\":\"active\"}"),
            ],
            tools: vec![],
            temperature: 0.2,
            max_tokens: 64,
            idempotency_key: "k2".into(),
        };
        let r2 = client.complete(req2).await.unwrap();
        assert!(r2.tool_calls.is_empty());
        assert_eq!(r2.text, "The customer is active.");
    }
}
