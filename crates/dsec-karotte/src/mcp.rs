//! The MCP tool server: the JSON-RPC 2.0 subset Karotte uses
//! (`initialize`, `tools/list`, `tools/call`, plus the `register_tools`
//! meta-tool), tool discovery validation, and resource sampling.
//!
//! Ports of upstream `karotte/mcp_servers/` (`mcp_server.py`,
//! `discover_tools.py`, `resource_sampler.py`,
//! `resource_profiling_middleware.py`). The wire protocol is modeled as
//! request/response values so it is testable without a transport; an
//! HTTP transport can be layered on top.

use crate::error::{Error, Result};
use crate::schemas::{CallToolResult, ResourceMetrics, ResourceSample};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// The MCP protocol version Karotte speaks.
pub const MCP_PROTOCOL_VERSION: &str = "2025-03-26";
/// Upstream server name.
pub const SERVER_NAME: &str = "karotte MCP Server";
/// Upstream `ResourceSampler.sample_interval_s`.
pub const SAMPLE_INTERVAL_S: f64 = 0.2;
/// Upstream `ResourceSampler.max_samples` (12 h at 0.2 s).
pub const MAX_SAMPLES: usize = 216_000;
/// The metadata key the profiling middleware injects
/// (upstream `_karotte_resource_metrics`).
pub const RESOURCE_METRICS_KEY: &str = "_karotte_resource_metrics";

// ---------------------------------------------------------------------------
// JSON-RPC envelopes
// ---------------------------------------------------------------------------

/// A JSON-RPC request.
#[derive(Debug, Clone, PartialEq)]
pub struct RpcRequest {
    /// Request id (echoed in the response).
    pub id: Value,
    /// Method name.
    pub method: String,
    /// Method params.
    pub params: Value,
}

/// A JSON-RPC response.
#[derive(Debug, Clone, PartialEq)]
pub struct RpcResponse {
    /// Echoed id (or null for notifications).
    pub id: Value,
    /// The result, when Ok.
    pub result: Option<Value>,
    /// The error, when Err.
    pub error: Option<RpcError>,
}

/// A JSON-RPC error object.
#[derive(Debug, Clone, PartialEq)]
pub struct RpcError {
    /// Error code.
    pub code: i64,
    /// Message.
    pub message: String,
    /// Extra data.
    pub data: Option<Value>,
}

/// Standard JSON-RPC / MCP error codes used here.
pub const PARSE_ERROR: i64 = -32700;
/// Invalid request shape.
pub const INVALID_REQUEST: i64 = -32600;
/// Unknown method.
pub const METHOD_NOT_FOUND: i64 = -32601;
/// Bad params.
pub const INVALID_PARAMS: i64 = -32602;
/// Tool execution error.
pub const INTERNAL_ERROR: i64 = -32603;

impl RpcResponse {
    /// An OK response.
    pub fn ok(id: Value, result: Value) -> Self {
        Self {
            id,
            result: Some(result),
            error: None,
        }
    }

    /// An error response.
    pub fn err(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            id,
            result: None,
            error: Some(RpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }

    /// Serialize to the wire JSON.
    pub fn to_json(&self) -> Value {
        let mut v = json!({"jsonrpc": "2.0", "id": self.id});
        let obj = v.as_object_mut().unwrap();
        if let Some(r) = &self.result {
            obj.insert("result".into(), r.clone());
        }
        if let Some(e) = &self.error {
            let mut eo = json!({"code": e.code, "message": e.message});
            if let Some(d) = &e.data {
                eo.as_object_mut().unwrap().insert("data".into(), d.clone());
            }
            obj.insert("error".into(), eo);
        }
        v
    }
}

impl RpcRequest {
    /// Parse a JSON-RPC request from wire JSON.
    pub fn from_json(v: &Value) -> Option<Self> {
        let id = v.get("id").cloned().unwrap_or(Value::Null);
        let method = v.get("method")?.as_str()?.to_string();
        let params = v.get("params").cloned().unwrap_or(json!({}));
        Some(Self { id, method, params })
    }
}

// ---------------------------------------------------------------------------
// Tool registry + discovery validation (discover_tools.py)
// ---------------------------------------------------------------------------

/// A registered tool's descriptor.
#[derive(Debug, Clone)]
pub struct ToolDescriptor {
    /// Tool name (module name upstream).
    pub name: String,
    /// The description (docstring upstream).
    pub description: String,
    /// JSON schema of the parameters.
    pub input_schema: Value,
}

/// One executable tool (upstream: the module's class `__call__`).
pub type ToolHandler = Arc<dyn Fn(&Value) -> Result<CallToolResult> + Send + Sync>;

/// Validation errors for tool registration (upstream `InvalidToolError`
/// causes). A tool must be named, documented, fully typed, and return a
/// `ToolResult` — the Rust port checks the descriptor's shape.
pub fn validate_tool_descriptor(d: &ToolDescriptor) -> Result<()> {
    if d.name.is_empty() {
        return Err(Error::InvalidSpec("Tool name must not be empty".into()));
    }
    if d.description.trim().is_empty() {
        return Err(Error::InvalidSpec(format!(
            "Tool {:?} not found or has no docstring (the docstring is the tool description)",
            d.name
        )));
    }
    let schema_type = d
        .input_schema
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("");
    if schema_type != "object" {
        return Err(Error::InvalidSpec(format!(
            "Tool {:?} parameters must be typed (schema type=object)",
            d.name
        )));
    }
    Ok(())
}

/// The tool registry (upstream `McpServer` state).
pub struct ToolRegistry {
    tools: Mutex<BTreeMap<String, (ToolDescriptor, ToolHandler)>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self {
            tools: Mutex::new(BTreeMap::new()),
        }
    }

    /// Register a tool after validation.
    pub fn register(&self, descriptor: ToolDescriptor, handler: ToolHandler) -> Result<()> {
        validate_tool_descriptor(&descriptor)?;
        self.tools
            .lock()
            .unwrap()
            .insert(descriptor.name.clone(), (descriptor, handler));
        Ok(())
    }

    /// Whether a tool is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.lock().unwrap().contains_key(name)
    }

    /// The registered names, sorted.
    pub fn names(&self) -> Vec<String> {
        self.tools.lock().unwrap().keys().cloned().collect()
    }

    /// `tools/list` result: descriptors as MCP tool infos.
    pub fn list_tools(&self) -> Vec<Value> {
        self.tools
            .lock()
            .unwrap()
            .values()
            .map(|(d, _)| {
                json!({
                    "name": d.name,
                    "description": d.description,
                    "inputSchema": d.input_schema,
                })
            })
            .collect()
    }

    /// `tools/call`: dispatch with the upstream error contract.
    pub fn call_tool(&self, name: &str, arguments: &Value) -> Result<CallToolResult> {
        let guard = self.tools.lock().unwrap();
        let Some((_, handler)) = guard.get(name) else {
            // Upstream surfaces unknown tools as an RPC error.
            return Err(Error::NotFound(format!(
                "Tool {name:?} not found. Known tools: {:?}",
                guard.keys().collect::<Vec<_>>()
            )));
        };
        let handler = handler.clone();
        drop(guard);
        handler(arguments)
    }
}

// ---------------------------------------------------------------------------
// The MCP server: register_tools meta-tool + method dispatch
// ---------------------------------------------------------------------------

/// The server (upstream `McpServer` + `HttpMcpServer`).
pub struct McpServer {
    registry: Arc<ToolRegistry>,
    /// The catalog of *installable* tools (upstream: importable modules).
    catalog: BTreeMap<String, (ToolDescriptor, ToolHandler)>,
    register_tools_installed: Mutex<bool>,
}

impl McpServer {
    /// A server whose `register_tools` meta-tool installs from `catalog`.
    pub fn new(catalog: Vec<(ToolDescriptor, ToolHandler)>) -> Self {
        let catalog: BTreeMap<String, (ToolDescriptor, ToolHandler)> = catalog
            .into_iter()
            .map(|(d, h)| (d.name.clone(), (d, h)))
            .collect();
        Self {
            registry: Arc::new(ToolRegistry::new()),
            catalog,
            register_tools_installed: Mutex::new(false),
        }
    }

    /// The shared registry (for direct harness use).
    pub fn registry(&self) -> Arc<ToolRegistry> {
        self.registry.clone()
    }

    /// Test helper: registry contains check.
    #[cfg(test)]
    fn contains_check(&self, name: &str) -> bool {
        self.registry.contains(name)
    }

    /// Whether the register_tools meta-tool is still exposed (it removes
    /// itself after a successful call, upstream).
    pub fn register_tools_available(&self) -> bool {
        !*self.register_tools_installed.lock().unwrap()
    }

    /// Handle one JSON-RPC request (the whole protocol surface).
    pub fn handle(&self, req: &RpcRequest) -> RpcResponse {
        match req.method.as_str() {
            "initialize" => RpcResponse::ok(
                req.id.clone(),
                json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": {"name": SERVER_NAME, "version": env!("CARGO_PKG_VERSION")}
                }),
            ),
            "tools/list" => {
                // Before register_tools runs, only the meta-tool is listed.
                let mut tools = self.registry.list_tools();
                if self.register_tools_available() {
                    tools.push(json!({
                        "name": "register_tools",
                        "description": "Register the tools this task exposes (removes itself).",
                        "inputSchema": {"type": "object", "properties": {
                            "tools": {"type": "array", "items": {"type": "string"}},
                            "name_overrides": {"type": "object"}
                        }, "required": ["tools"]}
                    }));
                }
                RpcResponse::ok(req.id.clone(), json!({"tools": tools}))
            }
            "tools/call" => {
                let name = req
                    .params
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("");
                let args = req.params.get("arguments").cloned().unwrap_or(json!({}));
                if name == "register_tools" {
                    return self.register_tools(req, &args);
                }
                match self.registry.call_tool(name, &args) {
                    Ok(result) => {
                        let content = result.content.clone();
                        let mut out = json!({
                            "content": content,
                            "isError": result.is_error,
                        });
                        if let Some(sc) = &result.structured_content {
                            out.as_object_mut()
                                .unwrap()
                                .insert("structuredContent".into(), sc.clone());
                        }
                        RpcResponse::ok(req.id.clone(), out)
                    }
                    Err(e) => RpcResponse::err(req.id.clone(), METHOD_NOT_FOUND, e.to_string()),
                }
            }
            other => RpcResponse::err(
                req.id.clone(),
                METHOD_NOT_FOUND,
                format!("Unknown method {other:?}"),
            ),
        }
    }

    /// The `register_tools` meta-tool: installs the requested tools, then
    /// removes itself (upstream `McpServer.register_tools`).
    fn register_tools(&self, req: &RpcRequest, args: &Value) -> RpcResponse {
        let names: Vec<String> = args
            .get("tools")
            .and_then(|t| t.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let overrides: BTreeMap<String, String> = args
            .get("name_overrides")
            .and_then(|o| o.as_object())
            .map(|o| {
                o.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();

        for name in &names {
            match self.catalog.get(name) {
                Some((d, h)) => {
                    let d = ToolDescriptor {
                        name: overrides
                            .get(name)
                            .cloned()
                            .unwrap_or_else(|| d.name.clone()),
                        description: d.description.clone(),
                        input_schema: d.input_schema.clone(),
                    };
                    if let Err(e) = self.registry.register(d, h.clone()) {
                        return RpcResponse::err(
                            req.id.clone(),
                            INVALID_PARAMS,
                            format!("register_tools: {e}"),
                        );
                    }
                }
                None => {
                    return RpcResponse::err(
                        req.id.clone(),
                        INVALID_PARAMS,
                        format!("Tool {name:?} not found"),
                    );
                }
            }
        }
        // Upstream: remove the meta-tool after registration.
        *self.register_tools_installed.lock().unwrap() = true;
        RpcResponse::ok(req.id.clone(), Value::Null)
    }
}

// ---------------------------------------------------------------------------
// resource_sampler.py
// ---------------------------------------------------------------------------

/// A cgroup/psutil sample source for the resource sampler.
pub trait SampleSource: Send + Sync {
    /// Read the current CPU usage (µs) and memory (bytes) under the
    /// cgroup v2 model (upstream reads `cpu.stat` `usage_usec` and
    /// `memory.current`), falling back to system-wide counters.
    fn sample(&self) -> Result<(u64, u64)>;

    /// The CPU capacity (for % normalization).
    fn num_cpus(&self) -> u64;
}

/// The resource sampler (upstream `ResourceSampler`): polls a
/// [`SampleSource`] at `SAMPLE_INTERVAL_S`, aggregates into
/// [`ResourceMetrics`].
pub struct ResourceSampler {
    source: Arc<dyn SampleSource>,
    samples: Mutex<Vec<ResourceSample>>,
    last: Mutex<Option<(std::time::Instant, u64, u64)>>,
    max_samples: usize,
}

/// The CPU-percent formula (upstream: `(Δusec/1e6) / Δwall * 100 /
/// num_cpus`, clamped to 0–100).
pub fn cpu_percent(delta_usec: u64, wall_s: f64, num_cpus: u64) -> f64 {
    if wall_s <= 0.0 {
        return 0.0;
    }
    let delta = delta_usec as f64 / 1e6;
    let cpus = num_cpus.max(1) as f64;
    (delta / wall_s * 100.0 / cpus).clamp(0.0, 100.0)
}

impl ResourceSampler {
    /// A sampler over `source`.
    pub fn new(source: Arc<dyn SampleSource>) -> Self {
        Self {
            source,
            samples: Mutex::new(Vec::new()),
            last: Mutex::new(None),
            max_samples: MAX_SAMPLES,
        }
    }

    /// Take one sample now (upstream's loop calls this every interval).
    pub fn tick(&self) -> Result<()> {
        let (cpu_usec, mem_bytes) = self.source.sample()?;
        let now = std::time::Instant::now();
        let mut last = self.last.lock().unwrap();
        let (cpu_percent_v, delta_ms) = match *last {
            None => (0.0, 0),
            Some((t, prev_cpu, _)) => {
                let wall = now.duration_since(t).as_secs_f64();
                let delta = cpu_usec.saturating_sub(prev_cpu);
                (
                    cpu_percent(delta, wall, self.source.num_cpus()),
                    (wall * 1000.0) as i64,
                )
            }
        };
        let cpu_percent = cpu_percent_v;
        *last = Some((now, cpu_usec, mem_bytes));
        drop(last);
        let mut samples = self.samples.lock().unwrap();
        if samples.len() < self.max_samples {
            samples.push(ResourceSample {
                timestamp_ms: delta_ms.max(0),
                cpu_percent,
                memory_mb: mem_bytes as f64 / (1024.0 * 1024.0),
            });
        }
        Ok(())
    }

    /// Aggregate and clear (upstream `_aggregate` on stop).
    pub fn metrics(&self) -> ResourceMetrics {
        let samples = self.samples.lock().unwrap().clone();
        ResourceMetrics::aggregate(&samples)
    }

    /// Number of samples taken.
    pub fn len(&self) -> usize {
        self.samples.lock().unwrap().len()
    }

    /// Whether no samples were taken.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The resource-profiling middleware behavior (upstream
/// `ResourceProfilingMiddleware.on_call_tool`): wrap the call, then
/// inject `_karotte_resource_metrics` into the structured content.
pub fn profile_tool_call(
    sampler: &ResourceSampler,
    call: impl FnOnce() -> Result<CallToolResult>,
) -> Result<CallToolResult> {
    sampler.tick()?;
    let result = call();
    sampler.tick()?;
    let mut result = result?;
    let metrics = sampler.metrics();
    let sc = result
        .structured_content
        .get_or_insert_with(|| Value::Object(Default::default()));
    if let Some(obj) = sc.as_object_mut() {
        obj.insert(
            RESOURCE_METRICS_KEY.to_string(),
            serde_json::to_value(&metrics).unwrap_or(Value::Null),
        );
    }
    Ok(result)
}

/// A static sample source for tests.
pub struct StaticSource {
    /// (cpu_usec, mem_bytes) readings, consumed in order (last repeats).
    pub readings: Mutex<Vec<(u64, u64)>>,
    /// CPU count.
    pub cpus: u64,
}

impl SampleSource for StaticSource {
    fn sample(&self) -> Result<(u64, u64)> {
        let mut r = self.readings.lock().unwrap();
        if r.len() > 1 {
            Ok(r.remove(0))
        } else {
            Ok(r.first().copied().unwrap_or((0, 0)))
        }
    }
    fn num_cpus(&self) -> u64 {
        self.cpus
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemas::TextContent;

    fn bash_tool_descriptor() -> ToolDescriptor {
        ToolDescriptor {
            name: "bash".into(),
            description: "Run a bash command as the student.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_s": {"type": "number"}
                },
                "required": ["command"]
            }),
        }
    }

    fn catalog() -> Vec<(ToolDescriptor, ToolHandler)> {
        vec![(
            bash_tool_descriptor(),
            Arc::new(|args: &Value| {
                let cmd = args.get("command").and_then(|c| c.as_str()).unwrap_or("");
                Ok(CallToolResult {
                    content: vec![
                        serde_json::to_value(TextContent::new(format!("ran: {cmd}"))).unwrap(),
                    ],
                    structured_content: Some(json!({"stdout": format!("ran: {cmd}")})),
                    is_error: false,
                })
            }),
        )]
    }

    #[test]
    fn initialize_handshake() {
        let srv = McpServer::new(catalog());
        let resp = srv.handle(&RpcRequest {
            id: json!(1),
            method: "initialize".into(),
            params: json!({}),
        });
        let r = resp.to_json();
        assert_eq!(r["result"]["protocolVersion"], MCP_PROTOCOL_VERSION);
        assert_eq!(r["result"]["serverInfo"]["name"], SERVER_NAME);
    }

    #[test]
    fn tools_list_before_and_after_registration() {
        let srv = McpServer::new(catalog());
        let resp = srv.handle(&RpcRequest {
            id: json!(1),
            method: "tools/list".into(),
            params: json!({}),
        });
        let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
        // Only the meta-tool is listed before registration.
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "register_tools");

        // register the bash tool
        let resp = srv.handle(&RpcRequest {
            id: json!(2),
            method: "tools/call".into(),
            params: json!({"name": "register_tools", "arguments": {"tools": ["bash"]}}),
        });
        assert!(resp.error.is_none());
        assert!(srv.contains_check("bash"));
        // meta-tool removed itself
        assert!(!srv.register_tools_available());

        let resp = srv.handle(&RpcRequest {
            id: json!(3),
            method: "tools/list".into(),
            params: json!({}),
        });
        let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "bash");
        assert!(tools[0]["inputSchema"]["properties"]["command"].is_object());
    }

    #[test]
    fn tools_call_dispatches_and_shapes_result() {
        let srv = McpServer::new(catalog());
        srv.handle(&RpcRequest {
            id: json!(1),
            method: "tools/call".into(),
            params: json!({"name": "register_tools", "arguments": {"tools": ["bash"]}}),
        });
        let resp = srv.handle(&RpcRequest {
            id: json!(2),
            method: "tools/call".into(),
            params: json!({"name": "bash", "arguments": {"command": "ls"}}),
        });
        let r = resp.to_json();
        assert_eq!(r["result"]["isError"], json!(false));
        assert!(r["result"]["structuredContent"]["stdout"]
            .as_str()
            .unwrap()
            .contains("ran: ls"));
        assert_eq!(r["result"]["content"][0]["type"], "text");
    }

    #[test]
    fn unknown_tool_is_method_not_found() {
        let srv = McpServer::new(catalog());
        let resp = srv.handle(&RpcRequest {
            id: json!(1),
            method: "tools/call".into(),
            params: json!({"name": "nonexistent", "arguments": {}}),
        });
        let e = resp.error.unwrap();
        assert_eq!(e.code, METHOD_NOT_FOUND);
    }

    #[test]
    fn unknown_method_rejected() {
        let srv = McpServer::new(catalog());
        let resp = srv.handle(&RpcRequest {
            id: json!(9),
            method: "resources/list".into(),
            params: json!({}),
        });
        assert_eq!(resp.error.unwrap().code, METHOD_NOT_FOUND);
    }

    #[test]
    fn register_unknown_tool_fails_without_installing() {
        let srv = McpServer::new(catalog());
        let resp = srv.handle(&RpcRequest {
            id: json!(1),
            method: "tools/call".into(),
            params: json!({"name": "register_tools", "arguments": {"tools": ["nope"]}}),
        });
        assert!(resp.error.is_some());
        assert!(srv.register_tools_available(), "meta-tool not consumed");
    }

    #[test]
    fn descriptor_validation_upstream_messages() {
        let no_doc = ToolDescriptor {
            name: "x".into(),
            description: "  ".into(),
            input_schema: json!({"type": "object"}),
        };
        let err = validate_tool_descriptor(&no_doc).unwrap_err();
        assert!(err.to_string().contains("no docstring"));
        let untyped = ToolDescriptor {
            name: "x".into(),
            description: "desc".into(),
            input_schema: json!({}),
        };
        let err2 = validate_tool_descriptor(&untyped).unwrap_err();
        assert!(err2.to_string().contains("must be typed"));
    }

    #[test]
    fn rpc_roundtrip_shapes() {
        let req = RpcRequest::from_json(&json!({
            "jsonrpc": "2.0", "id": 7, "method": "tools/list", "params": {}
        }))
        .unwrap();
        assert_eq!(req.id, json!(7));
        assert_eq!(req.method, "tools/list");

        let resp = RpcResponse::err(json!(7), INVALID_PARAMS, "bad");
        let wire = resp.to_json();
        assert_eq!(wire["error"]["code"], json!(INVALID_PARAMS));
        assert_eq!(wire["jsonrpc"], "2.0");
    }

    #[test]
    fn cpu_percent_formula_normalizes_across_cpus() {
        // 1e6 µs (1 CPU-second) in 1s wall on 2 CPUs → 50%.
        assert_eq!(cpu_percent(1_000_000, 1.0, 2), 50.0);
        assert_eq!(cpu_percent(1_000_000, 1.0, 1), 100.0);
        assert_eq!(cpu_percent(2_000_000, 1.0, 2), 100.0, "clamped");
        assert_eq!(cpu_percent(500_000, 2.0, 1), 25.0);
        assert_eq!(cpu_percent(0, 0.0, 2), 0.0, "no wall time yet");
    }

    #[test]
    fn resource_sampler_records_samples_and_memory() {
        let src = Arc::new(StaticSource {
            readings: Mutex::new(vec![(0, 0), (1_000_000, 10 * 1024 * 1024)]),
            cpus: 2,
        });
        let sampler = ResourceSampler::new(src);
        sampler.tick().unwrap();
        sampler.tick().unwrap();
        let m = sampler.metrics();
        assert_eq!(m.samples.len(), 2);
        assert_eq!(m.samples[1].memory_mb, 10.0);
        assert_eq!(m.peak_memory_mb, 10.0);
        assert_eq!(m.avg_memory_mb, 5.0);
        assert!(m.samples[1].cpu_percent >= 0.0);
    }

    #[test]
    fn profiling_middleware_injects_metrics() {
        let src = Arc::new(StaticSource {
            readings: Mutex::new(vec![(0, 0), (5, 5)]),
            cpus: 1,
        });
        let sampler = ResourceSampler::new(src);
        let result = profile_tool_call(&sampler, || Ok(CallToolResult::text("did work"))).unwrap();
        let sc = result.structured_content.unwrap();
        assert!(sc.get(RESOURCE_METRICS_KEY).is_some());
        assert!(!sc[RESOURCE_METRICS_KEY]["samples"]
            .as_array()
            .unwrap()
            .is_empty());
    }
}
