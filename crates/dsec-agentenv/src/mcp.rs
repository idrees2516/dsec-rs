//! MCP (Model Context Protocol) — the tool-serving protocol of the
//! environment sidecar.
//!
//! The knowledge-work environments expose each simulated business system
//! as an MCP server over **streamable-http** (`POST /mcp` with
//! JSON-RPC 2.0 envelopes, one response per request). The harness
//! discovers the tool surface with `tools/list` and forwards model tool
//! calls with `tools/call`, tunnelling through the main container for
//! black-box agents (the stdio bridge) or straight to 127.0.0.1 for
//! out-of-pod white-box agents.
//!
//! This module ports the protocol surface:
//!
//! * [`RpcRequest`] / [`RpcResponse`] / [`RpcError`] — JSON-RPC 2.0
//!   envelopes with the MCP method set (`initialize`,
//!   `tools/list`, `tools/call`);
//! * [`ToolInfo`] — the MCP tool descriptor (`name`, `description`,
//!   `inputSchema`) and its conversion to the OpenAI function-tool shape
//!   the model APIs consume (`{"type":"function","function":{...}}`);
//! * [`ToolCallOutcome`] — the `tools/call` result union
//!   (`content`/`isError`), mapping transport failures to the harness
//!   `transport_error` category the agent loop distinguishes from tool
//!   exceptions;
//! * [`McpClient`] — the client half: an in-process transport over a
//!   [`SimPod`] handle (the same JSON-RPC
//!   semantics as the HTTP transport, minus the socket — the wire bytes
//!   are identical, which keeps replayable golden tests).

use crate::error::{Error, Result};
use crate::topology::SimPod;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// MCP protocol version advertised by the client.
pub const PROTOCOL_VERSION: &str = "2025-03-26";

/// JSON-RPC 2.0 request envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RpcRequest {
    /// `"tools/list"` / `"tools/call"` / `"initialize"`.
    pub method: String,
    /// Request id (monotone per client).
    pub id: u64,
    /// JSON-RPC version — always `"2.0"`.
    #[serde(default = "jsonrpc")]
    pub jsonrpc: String,
    /// Method params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

fn jsonrpc() -> String {
    "2.0".to_string()
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RpcError {
    /// Error code (e.g. `-32601` method not found).
    pub code: i64,
    /// Error message.
    pub message: String,
}

/// JSON-RPC 2.0 response envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RpcResponse {
    /// Matches the request id.
    pub id: u64,
    /// Result on success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error on failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

impl RpcResponse {
    /// Builds a success response.
    pub fn ok(id: u64, result: Value) -> Self {
        Self {
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Builds an error response.
    pub fn err(id: u64, code: i64, message: impl Into<String>) -> Self {
        Self {
            id,
            result: None,
            error: Some(RpcError {
                code,
                message: message.into(),
            }),
        }
    }

    /// Unwraps into the result or an [`Error::RpcError`].
    pub fn into_result(self) -> Result<Value> {
        if let Some(e) = self.error {
            return Err(Error::RpcError {
                code: e.code,
                message: e.message,
            });
        }
        Ok(self.result.unwrap_or(Value::Null))
    }
}

/// The MCP tool descriptor (`tools/list` entry).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolInfo {
    /// Tool name.
    pub name: String,
    /// Human description.
    #[serde(default)]
    pub description: String,
    /// JSON Schema of the params object.
    #[serde(default = "empty_schema")]
    pub input_schema: Value,
}

fn empty_schema() -> Value {
    json!({"type": "object", "properties": {}})
}

impl ToolInfo {
    /// Converts to the OpenAI function-tool schema the chat APIs and the
    /// verl tool parser consume.
    pub fn to_function_tool(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            }
        })
    }
}

/// The `tools/call` outcome: content blocks plus the `isError` flag.
///
/// A tool that *ran and failed* (bad params, business rule violation)
/// sets `is_error` — the agent loop surfaces that as a user-turn
/// observation. A *transport* failure (server down, pod died) never
/// reaches this type; it surfaces as [`Error`] and maps to the
/// `transport_error` infra category instead.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallOutcome {
    /// Text content blocks.
    pub content: Vec<ContentBlock>,
    /// Whether the tool ran and reported failure.
    #[serde(rename = "isError", default)]
    pub is_error: bool,
}

/// One content block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Text content.
    #[serde(rename = "text")]
    Text {
        /// The text.
        text: String,
    },
}

impl ToolCallOutcome {
    /// Successful single-text outcome.
    pub fn text(payload: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text {
                text: payload.into(),
            }],
            is_error: false,
        }
    }

    /// Tool-level failure (the agent sees it, training continues).
    pub fn tool_error(payload: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text {
                text: payload.into(),
            }],
            is_error: true,
        }
    }

    /// Flattens the text blocks.
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .map(|b| match b {
                ContentBlock::Text { text } => text.clone(),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// The MCP client — JSON-RPC semantics over an in-process pod handle.
///
/// Wire compatibility: every request/response produced here is exactly
/// what the streamable-http transport would carry (same envelopes, same
/// methods, same result shapes). The in-process transport exists because
/// the pod lives in the same process in the deterministic test bed; a
/// socket transport would plug in behind the same [`McpTransport`]
/// trait.
pub struct McpClient {
    server: String,
    next_id: u64,
    transport: Box<dyn McpTransport>,
}

impl McpClient {
    /// Creates a client over a transport for one server.
    pub fn new(server: impl Into<String>, transport: Box<dyn McpTransport>) -> Self {
        Self {
            server: server.into(),
            next_id: 0,
            transport,
        }
    }

    /// Creates the in-process client over a pod.
    ///
    /// The pod must not be shared across threads concurrently with this
    /// client's calls (single-rollout ownership, as upstream).
    pub fn over_pod(pod: &mut SimPod) -> impl Iterator<Item = (String, McpClient)> + '_ {
        let names: Vec<String> = pod.servers.keys().cloned().collect();
        names.into_iter().map(move |name| {
            let client = McpClient {
                server: name.clone(),
                next_id: 0,
                transport: Box::new(PodTransport {
                    pod: pod as *mut SimPod,
                }),
            };
            (name, client)
        })
    }

    /// `initialize` handshake.
    pub fn initialize(&mut self) -> Result<Value> {
        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "dsec-agentenv", "version": env!("CARGO_PKG_VERSION")},
        });
        self.call("initialize", Some(params))
    }

    /// `tools/list` — the discovery surface.
    pub fn list_tools(&mut self) -> Result<Vec<ToolInfo>> {
        let v = self.call("tools/list", Some(json!({})))?;
        let tools = v
            .get("tools")
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();
        Ok(serde_json::from_value(Value::Array(tools))?)
    }

    /// `tools/call`.
    pub fn call_tool(&mut self, tool: &str, params: Value) -> Result<ToolCallOutcome> {
        let v = self.call(
            "tools/call",
            Some(json!({"name": tool, "arguments": params})),
        )?;
        Ok(serde_json::from_value(v)?)
    }

    /// Server name.
    pub fn server_name(&self) -> &str {
        &self.server
    }

    fn call(&mut self, method: &str, params: Option<Value>) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let req = RpcRequest {
            method: method.to_string(),
            id,
            jsonrpc: jsonrpc(),
            params,
        };
        let resp = self.transport.exchange(&self.server, &req)?;
        resp.into_result()
    }
}

/// Transport abstraction behind [`McpClient`].
pub trait McpTransport {
    /// One JSON-RPC exchange: request in, response out.
    fn exchange(&mut self, server: &str, request: &RpcRequest) -> Result<RpcResponse>;
}

struct PodTransport {
    pod: *mut SimPod,
}

impl McpTransport for PodTransport {
    fn exchange(&mut self, server: &str, request: &RpcRequest) -> Result<RpcResponse> {
        // SAFETY: the client is constructed from a &mut SimPod borrow and
        // lives strictly within that borrow's lifetime (see over_pod), so
        // the pointer is valid and unaliased for the client's lifetime.
        let pod = unsafe { &mut *self.pod };
        match request.method.as_str() {
            "initialize" => Ok(RpcResponse::ok(
                request.id,
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": {"name": server, "version": env!("CARGO_PKG_VERSION")},
                }),
            )),
            "tools/list" => {
                let tools: Vec<Value> = pod
                    .servers
                    .get(server)
                    .map(|ts| {
                        ts.iter()
                            .map(|t| {
                                json!({
                                    "name": t.name,
                                    "description": t.description,
                                    "inputSchema": t.params_schema,
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if tools.is_empty() && !pod.servers.contains_key(server) {
                    return Ok(RpcResponse::err(
                        request.id,
                        -32601,
                        format!("unknown server {server}"),
                    ));
                }
                Ok(RpcResponse::ok(request.id, json!({"tools": tools})))
            }
            "tools/call" => {
                let name = request
                    .params
                    .as_ref()
                    .and_then(|p| p.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let args = request
                    .params
                    .as_ref()
                    .and_then(|p| p.get("arguments"))
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                match pod.call_tool(server, name, args) {
                    Ok(v) => {
                        let is_err = v.get("error").is_some();
                        let text = if is_err {
                            v.get("error").cloned().unwrap_or_default().to_string()
                        } else {
                            v.to_string()
                        };
                        Ok(RpcResponse::ok(
                            request.id,
                            json!({"content": [{"type": "text", "text": text}], "isError": is_err}),
                        ))
                    }
                    Err(Error::ToolFailed { tool, message }) => Ok(RpcResponse::ok(
                        request.id,
                        json!({"content": [{"type": "text", "text": message}], "isError": true, "_tool": tool}),
                    )),
                    Err(e) => Ok(RpcResponse::err(request.id, -32603, e.to_string())),
                }
            }
            m => Ok(RpcResponse::err(
                request.id,
                -32601,
                format!("method not found: {m}"),
            )),
        }
    }
}

/// Discovers every tool of every server on a pod and converts them to
/// the OpenAI function-tool schema list — the `discover_mcp_tools`
/// step of the rollout setup.
pub fn discover_pod_tools(pod: &SimPod) -> Vec<(String, ToolInfo)> {
    pod.tool_catalog()
        .into_iter()
        .map(|(server, name, schema, desc)| {
            (
                server,
                ToolInfo {
                    name,
                    description: desc,
                    input_schema: schema,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Schema, StateDb, Table};
    use crate::topology::ToolDef;
    use serde_json::json;

    fn demo_pod() -> SimPod {
        let schema = Schema::new().table(Table::text("customers", "id", &["id", "name", "status"]));
        let mut db = StateDb::new(schema);
        db.seed_rows(
            "customers",
            [json!({"id": "C1", "name": "Acme", "status": "active"})
                .as_object()
                .unwrap()
                .clone()],
        );
        SimPod::builder("demo")
            .system("crm", db)
            .server(
                "crm",
                vec![ToolDef {
                    name: "get_customer".into(),
                    params_schema: json!({"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}),
                    description: "Fetch one customer".into(),
                    exec: Box::new(|ctx| {
                        let id = ctx.param_str("id").unwrap_or_default();
                        match ctx.db("crm").and_then(|db| db.get("customers", id)) {
                            Some(row) => json!(row),
                            None => json!({"error": format!("no customer {id}")}),
                        }
                    }),
                }],
            )
            .build()
    }

    #[test]
    fn rpc_envelopes_roundtrip() {
        let req = RpcRequest {
            method: "tools/list".into(),
            id: 7,
            jsonrpc: jsonrpc(),
            params: Some(json!({})),
        };
        let s = serde_json::to_string(&req).unwrap();
        assert!(s.contains("\"jsonrpc\":\"2.0\""));
        let back: RpcRequest = serde_json::from_str(&s).unwrap();
        assert_eq!(back, req);
    }

    #[test]
    fn tool_info_converts_to_openai_function_schema() {
        let t = ToolInfo {
            name: "get_customer".into(),
            description: "Fetch one".into(),
            input_schema: json!({"type": "object", "properties": {"id": {"type": "string"}}}),
        };
        let f = t.to_function_tool();
        assert_eq!(f["type"], "function");
        assert_eq!(f["function"]["name"], "get_customer");
        assert_eq!(f["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn client_handshake_and_discovery() {
        let mut pod = demo_pod();
        pod.execute(
            "python3 /installed-agent/sidecar_entrypoint.py --start-and-detach",
            crate::manifest::SIDECAR,
            30,
        );
        for (name, mut client) in McpClient::over_pod(&mut pod) {
            assert_eq!(name, "crm");
            let init = client.initialize().unwrap();
            assert_eq!(init["serverInfo"]["name"], "crm");
            let tools = client.list_tools().unwrap();
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].name, "get_customer");
            let out = client
                .call_tool("get_customer", json!({"id": "C1"}))
                .unwrap();
            assert!(!out.is_error);
            assert!(out.text_content().contains("Acme"));
            // tool-level failure: is_error, message visible
            let miss = client
                .call_tool("get_customer", json!({"id": "X9"}))
                .unwrap();
            assert!(miss.is_error);
            assert!(miss.text_content().contains("no customer"));
        }
    }

    #[test]
    fn unknown_server_is_method_not_found() {
        let mut pod = demo_pod();
        let mut client = McpClient::new("ghost", Box::new(PodTransport { pod: &mut pod }));
        let err = client.list_tools().unwrap_err();
        match err {
            Error::RpcError { code, .. } => assert_eq!(code, -32601),
            other => panic!("expected rpc error, got {other:?}"),
        }
    }

    #[test]
    fn outcome_serialization_matches_mcp_shape() {
        let o = ToolCallOutcome::text("ok");
        let s = serde_json::to_string(&o).unwrap();
        assert!(s.contains("\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]"));
        let back: ToolCallOutcome = serde_json::from_str(&s).unwrap();
        assert_eq!(back, o);
        let e = ToolCallOutcome::tool_error("bad");
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"isError\":true"));
    }
}
