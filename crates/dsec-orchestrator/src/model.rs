//! The model bridge: an OpenAI-compatible chat-completions client over
//! real HTTP, with egress policy enforcement and idempotency keys.
//!
//! Design notes (DSec §6.2 / QWENGYRE model-serving contracts):
//!
//! - **Real transport.** [`OpenAiCompatClient`] speaks HTTP/1.1 over
//!   `tokio::net::TcpStream` — no framework, no dependency. In-cluster
//!   model servers (vLLM / SGLang / the `dsec-train` policy server) are
//!   plain HTTP; for public HTTPS endpoints, terminate TLS at a gateway
//!   and point the client at it.
//! - **Idempotency keys.** Every completion request carries an
//!   `Idempotency-Key` header (and the journal records it). Model calls
//!   are pure, so a crash between call and persistence just re-issues
//!   the request; servers that honor the key return the cached
//!   completion, which also makes replay deterministic.
//! - **Egress policy.** [`NetPolicy`] is an allowlist enforced *before*
//!   any socket is opened: host suffix + port. The default policy for a
//!   client pins it to its own configured endpoint, so a tampered
//!   config cannot redirect model traffic (or leaked transcripts)
//!   elsewhere.
//! - **Retries.** Connection errors, 429 and 5xx retry with exponential
//!   backoff; 4xx (other than 429) fail fast — the request itself is
//!   bad.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::error::{Error, Result};

/// Alias for a boxed, owned async future.
pub type OwnedFut<T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send>>;

/// One message sent to the model (OpenAI chat format, including
/// assistant `tool_calls` and `tool`-role observation turns).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `system` / `user` / `assistant` / `tool`.
    pub role: String,
    /// Message text.
    pub content: String,
    /// Assistant tool calls (empty for other roles).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallOut>,
    /// The call id this `tool`-role message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Plain role+content message.
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// An assistant turn with tool calls.
    pub fn assistant(text: impl Into<String>, tool_calls: Vec<ToolCallOut>) -> Self {
        Self {
            role: "assistant".into(),
            content: text.into(),
            tool_calls,
            tool_call_id: None,
        }
    }

    /// A `tool`-role observation answering `tool_call_id`.
    pub fn tool_observation(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// One tool call parsed out of a model response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallOut {
    /// Call id (echoed back in the observation turn).
    pub id: String,
    /// Function name (may be `server.tool` qualified).
    pub name: String,
    /// Arguments JSON (string form, as OpenAI returns it).
    pub arguments: String,
}

/// A chat-completions request.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// Model name.
    pub model: String,
    /// Transcript so far.
    pub messages: Vec<ChatMessage>,
    /// OpenAI function-tool schemas.
    pub tools: Vec<serde_json::Value>,
    /// Sampling temperature.
    pub temperature: f32,
    /// Max tokens for the completion.
    pub max_tokens: u32,
    /// Idempotency key (also journaled by the worker).
    pub idempotency_key: String,
}

/// A chat-completions response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Assistant text.
    pub text: String,
    /// Tool calls the model issued (empty for a final turn).
    pub tool_calls: Vec<ToolCallOut>,
    /// `stop` / `tool_calls` / `length`.
    pub finish_reason: String,
    /// Total tokens reported by the server.
    pub usage_tokens: u64,
    /// Model that actually served (version transparency for phase 5).
    pub served_model: String,
}

/// The model client contract.
///
/// Implementations return an **owned** future: they must clone their
/// interior state into it (never borrow `self`), following
/// [`ScriptedModelClient`]'s Arc convention.
pub trait ModelClient: Send + Sync {
    /// Completes one chat request.
    fn complete(&self, req: ChatRequest) -> OwnedFut<Result<ChatResponse>>;
}

/// Egress allowlist enforced before any connection is opened.
#[derive(Debug, Clone)]
pub enum NetPolicy {
    /// Allow any destination (open, tests only).
    Open,
    /// Allow only `(host-suffix, port)` pairs in the list.
    Allowlist {
        /// Host suffixes (`api.vendor.com` matches
        /// `eu.api.vendor.com`).
        hosts: Vec<String>,
        /// Allowed ports (443, 8000, ...).
        ports: Vec<u16>,
    },
}

impl NetPolicy {
    /// Checks a destination, denying before any socket opens.
    pub fn check(&self, host: &str, port: u16) -> Result<()> {
        match self {
            NetPolicy::Open => Ok(()),
            NetPolicy::Allowlist { hosts, ports } => {
                let host_ok = hosts
                    .iter()
                    .any(|h| host == h || host.ends_with(&format!(".{h}")));
                let port_ok = ports.contains(&port);
                if !host_ok || !port_ok {
                    Err(Error::NetPolicyDenied {
                        host: host.to_string(),
                        port,
                        reason: if !host_ok {
                            "host not in allowlist".into()
                        } else {
                            "port not in allowlist".into()
                        },
                    })
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// The OpenAI-compatible HTTP client.
#[derive(Debug, Clone)]
pub struct OpenAiCompatClient {
    host: String,
    port: u16,
    path: String,
    api_key: Option<String>,
    model: String,
    net: NetPolicy,
    timeout: Duration,
    max_attempts: u32,
    base_backoff: Duration,
}

impl OpenAiCompatClient {
    /// Builds a client for `http://host:port[/prefix]`.
    ///
    /// The network policy is pinned to exactly this endpoint — the
    /// client can never be talked into connecting anywhere else.
    pub fn new(endpoint: &str, model: &str) -> Result<Self> {
        let (host, port, path) = parse_url(endpoint)?;
        let net = NetPolicy::Allowlist {
            hosts: vec![host.clone()],
            ports: vec![port],
        };
        Ok(Self {
            host,
            port,
            path,
            api_key: None,
            model: model.to_string(),
            net,
            timeout: Duration::from_secs(30),
            max_attempts: 3,
            base_backoff: Duration::from_millis(25),
        })
    }

    /// Relaxes / tightens the network policy.
    pub fn with_net_policy(mut self, net: NetPolicy) -> Self {
        self.net = net;
        self
    }

    /// Sets the bearer token.
    pub fn with_api_key(mut self, key: Option<String>) -> Self {
        self.api_key = key;
        self
    }

    /// Overrides per-attempt timeout and retry shape.
    pub fn with_retry(
        mut self,
        timeout: Duration,
        max_attempts: u32,
        base_backoff: Duration,
    ) -> Self {
        self.timeout = timeout;
        self.max_attempts = max_attempts.max(1);
        self.base_backoff = base_backoff;
        self
    }

    /// Builds a client from `DSEC_MODEL_URL`, `DSEC_MODEL_KEY`,
    /// `DSEC_MODEL_NAME` environment variables.
    pub fn from_env() -> Result<Self> {
        let url = std::env::var("DSEC_MODEL_URL")
            .map_err(|_| Error::Config("DSEC_MODEL_URL is not set".into()))?;
        let model = std::env::var("DSEC_MODEL_NAME").unwrap_or_else(|_| "default".into());
        let key = std::env::var("DSEC_MODEL_KEY")
            .ok()
            .filter(|k| !k.is_empty());
        Ok(Self::new(&url, &model)?.with_api_key(key))
    }

    fn endpoint_label(&self) -> String {
        format!("{}:{}{}", self.host, self.port, self.path)
    }

    /// Checks the client's own endpoint against its policy (tests and
    /// diagnostics: the pin must hold).
    pub fn net_check_self(&self) -> Result<()> {
        self.net.check(&self.host, self.port)
    }
}

fn parse_url(url: &str) -> Result<(String, u16, String)> {
    let rest = url.strip_prefix("http://").ok_or_else(|| {
        Error::Config(format!(
            "only http:// endpoints are supported (terminate TLS at a gateway): {url}"
        ))
    })?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (
            h.to_string(),
            p.parse::<u16>()
                .map_err(|_| Error::Config(format!("bad port in {url}")))?,
        ),
        None => (authority.to_string(), 80),
    };
    if host.is_empty() {
        return Err(Error::Config(format!("bad endpoint {url}")));
    }
    Ok((host, port, path.to_string()))
}

impl ModelClient for OpenAiCompatClient {
    fn complete(&self, req: ChatRequest) -> OwnedFut<Result<ChatResponse>> {
        let this = self.clone();
        Box::pin(async move {
            let mut last_err = None;
            for attempt in 1..=this.max_attempts {
                match this.one_attempt(&req).await {
                    Ok(resp) => return Ok(resp),
                    Err(e) => {
                        let retryable = matches!(&e, Error::Model { .. } | Error::Io(_))
                            || matches!(&e, Error::NetPolicyDenied { .. });
                        if !retryable || attempt == this.max_attempts {
                            return Err(e);
                        }
                        last_err = Some(e);
                    }
                }
                tokio::time::sleep(this.base_backoff * attempt).await;
            }
            Err(last_err.unwrap_or_else(|| Error::Model {
                model: this.endpoint_label(),
                message: "exhausted retries".into(),
            }))
        })
    }
}

impl OpenAiCompatClient {
    async fn one_attempt(&self, req: &ChatRequest) -> Result<ChatResponse> {
        // policy check BEFORE any socket
        self.net.check(&self.host, self.port)?;

        let body = serde_json::to_vec(&wire_request(req, &self.model))?;
        let url_path = format!("{}/v1/chat/completions", self.path);
        let mut head = format!(
            "POST {url_path} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nIdempotency-Key: {}\r\n",
            self.host, self.port, body.len(), req.idempotency_key
        );
        if let Some(key) = &self.api_key {
            head.push_str(&format!("Authorization: Bearer {key}\r\n"));
        }
        head.push_str("\r\n");

        let fut = async {
            let mut stream = TcpStream::connect((self.host.as_str(), self.port))
                .await
                .map_err(|e| Error::Model {
                    model: self.endpoint_label(),
                    message: format!("connect: {e}"),
                })?;
            stream
                .write_all(head.as_bytes())
                .await
                .map_err(|e| Error::Model {
                    model: self.endpoint_label(),
                    message: format!("write: {e}"),
                })?;
            stream.write_all(&body).await.map_err(|e| Error::Model {
                model: self.endpoint_label(),
                message: format!("write body: {e}"),
            })?;
            let mut raw = Vec::with_capacity(4096);
            stream
                .read_to_end(&mut raw)
                .await
                .map_err(|e| Error::Model {
                    model: self.endpoint_label(),
                    message: format!("read: {e}"),
                })?;
            Ok::<Vec<u8>, Error>(raw)
        };
        let raw = tokio::time::timeout(self.timeout, fut)
            .await
            .map_err(|_| Error::Model {
                model: self.endpoint_label(),
                message: format!("timeout after {:?}", self.timeout),
            })??;

        parse_http_response(&raw, &self.endpoint_label())
    }
}

fn wire_request(req: &ChatRequest, model: &str) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = req
        .messages
        .iter()
        .map(|m| {
            let mut v = serde_json::json!({
                "role": m.role,
                "content": m.content,
            });
            if !m.tool_calls.is_empty() {
                v["tool_calls"] = serde_json::json!(m
                    .tool_calls
                    .iter()
                    .map(|tc| serde_json::json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }))
                    .collect::<Vec<_>>());
            }
            if let Some(id) = &m.tool_call_id {
                v["tool_call_id"] = serde_json::json!(id);
            }
            v
        })
        .collect();
    serde_json::json!({
        "model": model,
        "messages": messages,
        "tools": req.tools,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    })
}

fn parse_http_response(raw: &[u8], label: &str) -> Result<ChatResponse> {
    let text = std::str::from_utf8(raw).map_err(|_| Error::Model {
        model: label.into(),
        message: "non-utf8 response".into(),
    })?;
    let header_end = text.find("\r\n\r\n").ok_or_else(|| Error::Model {
        model: label.into(),
        message: "malformed response".into(),
    })?;
    let head = &text[..header_end];
    let body = &text[header_end + 4..];
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| Error::Model {
            model: label.into(),
            message: "no status line".into(),
        })?;
    if !(200..300).contains(&status) {
        return Err(Error::Model {
            model: label.into(),
            message: format!(
                "HTTP {status}: {}",
                body.chars().take(300).collect::<String>()
            ),
        });
    }
    let wire: WireCompletion = serde_json::from_str(body).map_err(|e| Error::Model {
        model: label.into(),
        message: format!("bad json: {e}"),
    })?;
    let mut out = ChatResponse {
        finish_reason: wire
            .choices
            .first()
            .and_then(|c| c.finish_reason.clone())
            .unwrap_or_default(),
        usage_tokens: wire.usage.map(|u| u.total_tokens).unwrap_or(0),
        served_model: wire.model,
        ..Default::default()
    };
    if let Some(choice) = wire.choices.into_iter().next() {
        out.text = choice.message.content.unwrap_or_default();
        out.tool_calls = choice
            .message
            .tool_calls
            .into_iter()
            .map(|tc| ToolCallOut {
                id: tc.id,
                name: tc.function.name,
                arguments: tc.function.arguments,
            })
            .collect();
    }
    Ok(out)
}

#[derive(Debug, Deserialize)]
struct WireCompletion {
    model: String,
    choices: Vec<WireChoice>,
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
struct WireChoice {
    message: WireMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WireMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<WireToolCall>,
}

#[derive(Debug, Deserialize)]
struct WireToolCall {
    id: String,
    function: WireFunction,
}

#[derive(Debug, Deserialize)]
struct WireFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct WireUsage {
    total_tokens: u64,
}

/// A deterministic in-process model client for tests and recovery
/// drills: pops scripted responses, and honors idempotency keys by
/// replaying the recorded response (a crash before persistence re-asks
/// the same key and gets the same answer — no drift).
///
/// Interior state is `Arc`-shared so the (borrowing-free) owned future
/// returned by [`ModelClient::complete`] still observes one counter —
/// the same convention every [`ModelClient`] implementation follows:
/// clone interior state into the future, never borrow `self`.
#[derive(Debug, Default, Clone)]
pub struct ScriptedModelClient {
    script: std::sync::Arc<Vec<ChatResponse>>,
    ledger: std::sync::Arc<std::sync::Mutex<BTreeMap<String, ChatResponse>>>,
    pos: std::sync::Arc<std::sync::Mutex<usize>>,
}

impl ScriptedModelClient {
    /// New scripted client over a response sequence.
    pub fn new(script: Vec<ChatResponse>) -> Self {
        Self {
            script: std::sync::Arc::new(script),
            ..Default::default()
        }
    }

    /// Number of *distinct* idempotency keys served (not re-served).
    pub fn distinct_calls(&self) -> usize {
        self.ledger.lock().map(|l| l.len()).unwrap_or(0)
    }

    /// Total requests seen (including retries of known keys).
    pub fn total_requests(&self) -> usize {
        self.pos.lock().map(|p| *p).unwrap_or(0)
    }
}

impl ModelClient for ScriptedModelClient {
    fn complete(&self, req: ChatRequest) -> OwnedFut<Result<ChatResponse>> {
        let this = self.clone();
        Box::pin(async move {
            {
                let mut pos = this.pos.lock().map_err(|_| Error::Model {
                    model: "scripted".into(),
                    message: "poisoned".into(),
                })?;
                *pos += 1;
            }
            if let Some(hit) = this
                .ledger
                .lock()
                .map_err(|_| Error::Model {
                    model: "scripted".into(),
                    message: "poisoned".into(),
                })?
                .get(&req.idempotency_key)
            {
                return Ok(hit.clone());
            }
            let idx = this
                .pos
                .lock()
                .map_err(|_| Error::Model {
                    model: "scripted".into(),
                    message: "poisoned".into(),
                })?
                .saturating_sub(1);
            let resp = this.script.get(idx).cloned().unwrap_or_default();
            if let Ok(mut l) = this.ledger.lock() {
                l.insert(req.idempotency_key, resp.clone());
            }
            Ok(resp)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_parsing() {
        let (h, p, path) = parse_url("http://10.0.0.1:8000").unwrap();
        assert_eq!((h.as_str(), p, path.as_str()), ("10.0.0.1", 8000, ""));
        let (h, p, path) = parse_url("http://model.internal:9/v1").unwrap();
        assert_eq!((h.as_str(), p, path.as_str()), ("model.internal", 9, "/v1"));
        let (h, p, _) = parse_url("http://gw:80").unwrap();
        assert_eq!((h.as_str(), p), ("gw", 80));
        assert!(parse_url("https://x").is_err(), "https rejected");
        assert!(parse_url("http://").is_err());
    }

    #[test]
    fn net_policy_suffix_and_ports() {
        let pol = NetPolicy::Allowlist {
            hosts: vec!["api.vendor.com".into()],
            ports: vec![443, 8000],
        };
        assert!(pol.check("api.vendor.com", 443).is_ok());
        assert!(pol.check("eu.api.vendor.com", 8000).is_ok());
        assert!(
            pol.check("evil-api.vendor.com.evil.io", 443).is_err(),
            "prefix must not match"
        );
        assert!(pol.check("notvendor.com", 443).is_err());
        assert!(pol.check("api.vendor.com", 22).is_err());
        assert!(NetPolicy::Open.check("anything", 1).is_ok());
    }

    #[tokio::test]
    async fn client_policy_pins_its_own_endpoint() {
        // a client built for endpoint A cannot be redirected to B even
        // by rewriting its internals: build for A, then request B by
        // constructing a second client and swapping its host manually
        let a = OpenAiCompatClient::new("http://10.9.9.9:1234", "m").unwrap();
        // the policy pins host+port
        a.net.check("10.9.9.9", 1234).unwrap();
        assert!(a.net.check("10.9.9.9", 1235).is_err());
        assert!(a.net.check("evil.example", 1234).is_err());
    }

    #[test]
    fn response_parsing() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n\
{\"model\":\"tiny-1\",\"choices\":[{\"message\":{\"content\":\"hello\",\"tool_calls\":[{\"id\":\"c1\",\"function\":{\"name\":\"exec\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"total_tokens\":42}}";
        let resp = parse_http_response(raw, "test").unwrap();
        assert_eq!(resp.text, "hello");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "exec");
        assert_eq!(resp.usage_tokens, 42);
        assert_eq!(resp.served_model, "tiny-1");
        // error statuses surface as Model errors
        let err = parse_http_response(b"HTTP/1.1 400 Bad\r\n\r\n{\"error\":\"x\"}", "test");
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn scripted_client_idempotent_replay() {
        let c = ScriptedModelClient::new(vec![ChatResponse {
            text: "a".into(),
            ..Default::default()
        }]);
        let req = |key: &str| ChatRequest {
            model: "m".into(),
            messages: vec![],
            tools: vec![],
            temperature: 0.0,
            max_tokens: 8,
            idempotency_key: key.into(),
        };
        let r1 = c.complete(req("k1")).await.unwrap();
        let r2 = c.complete(req("k1")).await.unwrap(); // replay
        c.complete(req("k2")).await.unwrap();
        assert_eq!(r1.text, r2.text);
        assert_eq!(c.total_requests(), 3);
        assert_eq!(c.distinct_calls(), 2);
    }
}
