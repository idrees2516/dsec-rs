//! The AgentEnv HTTP server — the Scale AgentEnv framework's env-server,
//! in Rust.
//!
//! The framework's core contract: environments live behind a server and
//! are driven over a **standardized REST API**, so trainers and agents
//! never link against environment code. One server hosts many
//! environments; each environment is created from a task spec, stepped
//! (observation / tool call / reset), and destroyed. This module ports
//! that surface over the deterministic environment stack:
//!
//! ```text
//! POST   /envs                create env from an EnvSpec (or compile+boot)
//! GET    /envs                list env ids + phases
//! GET    /envs/:id            env info (phase, tools, listening ports)
//! GET    /envs/:id/obs        current observation text
//! POST   /envs/:id/tools/call  dispatch one tool call
//! POST   /envs/:id/reset      re-seed the env (state reset, fresh workspace)
//! POST   /envs/:id/verify     run the verifier, return the reward (masked
//!                             outcomes carry the category — never a false 0)
//! DELETE /envs/:id            destroy
//! GET    /health              liveness
//! GET    /metrics             server counters
//! ```
//!
//! Concurrency: one `Mutex`-guarded env registry (environments are
//! process-local, single-rollout ownership — same as upstream, where an
//! env actor owns its pod). The server is feature-gated behind `server`
//! (axum).

use crate::envgen::EnvSpec;
use crate::error::{Error, Result};
use crate::live::EnvBundle;
use crate::verifier::{AgentRolloutResult, Judge, RewardConfig, RewardOutcome, VerifierHarness};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Server-side error → HTTP response.
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

impl From<Error> for ApiError {
    fn from(e: Error) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: e.to_string(),
        }
    }
}

type ApiResult<T> = std::result::Result<T, ApiError>;

/// One hosted environment.
struct HostedEnv {
    id: u64,
    spec: EnvSpec,
    bundle: EnvBundle,
    /// Current observation text (last tool result / instruction).
    observation: String,
}

/// The env server state.
pub struct EnvServer {
    envs: Mutex<BTreeMap<u64, HostedEnv>>,
    next_id: AtomicU64,
    judge: Arc<dyn Judge>,
    reward_config: RewardConfig,
    counters: Mutex<BTreeMap<&'static str, u64>>,
}

impl EnvServer {
    /// New empty server with a judge and reward config.
    pub fn new(judge: Arc<dyn Judge>, reward_config: RewardConfig) -> Self {
        let mut counters = BTreeMap::new();
        counters.insert("envs_created", 0);
        counters.insert("envs_destroyed", 0);
        counters.insert("tool_calls", 0);
        counters.insert("verifications", 0);
        counters.insert("masked_rewards", 0);
        Self {
            envs: Mutex::new(BTreeMap::new()),
            next_id: AtomicU64::new(1),
            judge,
            reward_config,
            counters: Mutex::new(counters),
        }
    }

    fn bump(&self, key: &'static str) {
        if let Ok(mut c) = self.counters.lock() {
            *c.entry(key).or_insert(0) += 1;
        }
    }

    /// Builds the axum router.
    pub fn router(self: Arc<Self>) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/metrics", get(metrics))
            .route("/envs", get(list_envs).post(create_env))
            .route("/envs/{id}", get(env_info).delete(destroy_env))
            .route("/envs/{id}/obs", get(get_obs))
            .route("/envs/{id}/tools/call", post(call_tool))
            .route("/envs/{id}/reset", post(reset_env))
            .route("/envs/{id}/verify", post(verify_env))
            .with_state(self)
    }

    /// Starts serving on `addr` (blocks until shutdown).
    pub async fn serve(self, addr: std::net::SocketAddr) -> Result<()> {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(Error::Io)?;
        let arc = Arc::new(self);
        axum::serve(listener, arc.router())
            .await
            .map_err(|e| Error::Io(std::io::Error::other(e)))?;
        Ok(())
    }
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok", "server": "dsec-agentenv"}))
}

async fn metrics(State(server): State<Arc<EnvServer>>) -> impl IntoResponse {
    let counters = server
        .counters
        .lock()
        .map(|c| c.clone())
        .unwrap_or_default();
    Json(Value::Object(
        counters
            .into_iter()
            .map(|(k, v)| (k.to_string(), json!(v)))
            .collect(),
    ))
}

#[derive(Deserialize, Serialize)]
struct CreateEnvReq {
    /// The full EnvSpec.
    spec: EnvSpec,
}

#[derive(Serialize)]
struct EnvIdResp {
    id: u64,
    instance_id: String,
}

async fn create_env(
    State(server): State<Arc<EnvServer>>,
    Json(req): Json<CreateEnvReq>,
) -> ApiResult<Json<EnvIdResp>> {
    let bundle = req.spec.compile()?.boot()?;
    let id = server.next_id.fetch_add(1, Ordering::SeqCst);
    let observation = format!("{}\n\n{}", req.spec.instruction, env_tool_hint(&bundle));
    let env = HostedEnv {
        id,
        spec: req.spec,
        bundle,
        observation,
    };
    let instance_id = env.spec.instance_id.clone();
    server.envs.lock().map_err(|_| poisoned())?.insert(id, env);
    server.bump("envs_created");
    Ok(Json(EnvIdResp { id, instance_id }))
}

fn poisoned() -> ApiError {
    ApiError {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: "env registry poisoned".into(),
    }
}

fn env_tool_hint(bundle: &EnvBundle) -> String {
    let names: Vec<String> = bundle
        .tools
        .iter()
        .map(|(s, t)| format!("{s}.{}", t.name))
        .collect();
    format!("Available tools: {}", names.join(", "))
}

async fn list_envs(State(server): State<Arc<EnvServer>>) -> ApiResult<Json<Value>> {
    let envs = server.envs.lock().map_err(|_| poisoned())?;
    let rows: Vec<Value> = envs
        .values()
        .map(|e| {
            json!({
                "id": e.id,
                "instance_id": e.spec.instance_id,
                "phase": e.bundle.pod.phase,
                "ports": e.bundle.pod.listening,
            })
        })
        .collect();
    Ok(Json(json!({ "envs": rows })))
}

async fn env_info(
    State(server): State<Arc<EnvServer>>,
    Path(id): Path<u64>,
) -> ApiResult<Json<Value>> {
    let envs = server.envs.lock().map_err(|_| poisoned())?;
    let e = envs.get(&id).ok_or(not_found(id))?;
    Ok(Json(json!({
        "id": e.id,
        "instance_id": e.spec.instance_id,
        "phase": e.bundle.pod.phase,
        "ports": e.bundle.pod.listening,
        "tools": e.bundle.tools.iter().map(|(s, t)| json!({
            "server": s, "name": t.name, "description": t.description,
        })).collect::<Vec<_>>(),
        "systems": e.bundle.pod.state.keys().collect::<Vec<_>>(),
    })))
}

fn not_found(id: u64) -> ApiError {
    ApiError {
        status: StatusCode::NOT_FOUND,
        message: format!("env {id} not found"),
    }
}

async fn get_obs(
    State(server): State<Arc<EnvServer>>,
    Path(id): Path<u64>,
) -> ApiResult<Json<Value>> {
    let envs = server.envs.lock().map_err(|_| poisoned())?;
    let e = envs.get(&id).ok_or(not_found(id))?;
    Ok(Json(json!({ "id": id, "observation": e.observation })))
}

#[derive(Deserialize)]
struct ToolCallReq {
    /// MCP server name.
    server: String,
    /// Tool name.
    tool: String,
    /// Arguments object.
    #[serde(default = "empty_obj")]
    params: Value,
}

fn empty_obj() -> Value {
    json!({})
}

async fn call_tool(
    State(server): State<Arc<EnvServer>>,
    Path(id): Path<u64>,
    Json(req): Json<ToolCallReq>,
) -> ApiResult<Json<Value>> {
    let mut envs = server.envs.lock().map_err(|_| poisoned())?;
    let e = envs.get_mut(&id).ok_or(not_found(id))?;
    let outcome = e
        .bundle
        .pod
        .call_tool(&req.server, &req.tool, req.params.clone())?;
    e.observation = outcome.to_string();
    server.bump("tool_calls");
    Ok(Json(json!({ "id": id, "result": outcome })))
}

async fn reset_env(
    State(server): State<Arc<EnvServer>>,
    Path(id): Path<u64>,
) -> ApiResult<Json<Value>> {
    let mut envs = server.envs.lock().map_err(|_| poisoned())?;
    let e = envs.get_mut(&id).ok_or(not_found(id))?;
    // rebuild from the spec: fresh DBs, fresh workspace, fresh sessions
    e.bundle = e.spec.compile()?.boot()?;
    e.observation = format!("{}\n\n{}", e.spec.instruction, env_tool_hint(&e.bundle));
    Ok(Json(
        json!({ "id": id, "reset": true, "phase": e.bundle.pod.phase }),
    ))
}

#[derive(Deserialize, Default)]
struct VerifyReq {
    /// Final agent message to grade (persisted as answer.md when present).
    #[serde(default)]
    final_message: Option<String>,
}

async fn verify_env(
    State(server): State<Arc<EnvServer>>,
    Path(id): Path<u64>,
    body: Option<Json<VerifyReq>>,
) -> ApiResult<Json<Value>> {
    let mut envs = server.envs.lock().map_err(|_| poisoned())?;
    let e = envs.get_mut(&id).ok_or(not_found(id))?;
    let rollout = AgentRolloutResult {
        final_message: body.and_then(|Json(b)| b.final_message).unwrap_or_default(),
        completed: true,
        tool_turns: 0,
    };
    let harness = VerifierHarness::new(
        &e.bundle.rubric,
        server.judge.as_ref(),
        server.reward_config.clone(),
    );
    let outcome = harness.calculate_reward(&mut e.bundle.pod, &e.bundle.manifest, &rollout);
    server.bump("verifications");
    let payload = match &outcome {
        RewardOutcome::Valid {
            score,
            raw,
            results,
        } => {
            json!({
                "id": id,
                "reward": score,
                "raw": raw,
                "masked": false,
                "results": results,
            })
        }
        RewardOutcome::TestbedCorrupted { kind, detail } => {
            server.bump("masked_rewards");
            json!({
                "id": id,
                "reward": null,
                "masked": true,
                "category": kind.as_category(),
                "reward_error": kind.as_reward_error(),
                "detail": detail,
            })
        }
    };
    Ok(Json(payload))
}

async fn destroy_env(
    State(server): State<Arc<EnvServer>>,
    Path(id): Path<u64>,
) -> ApiResult<Json<Value>> {
    let mut envs = server.envs.lock().map_err(|_| poisoned())?;
    let removed = envs.remove(&id).ok_or(not_found(id))?;
    server.bump("envs_destroyed");
    Ok(Json(
        json!({ "id": id, "destroyed": true, "instance_id": removed.spec.instance_id }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verifier::AnchorJudge;
    use serde_json::json;

    fn server() -> Arc<EnvServer> {
        Arc::new(EnvServer::new(
            Arc::new(AnchorJudge::new()),
            RewardConfig::default(),
        ))
    }

    fn spec() -> EnvSpec {
        crate::envgen::templates::terminal()
    }

    fn judged_spec() -> EnvSpec {
        crate::envgen::templates::knowledge_work()
    }

    async fn create(server: &Arc<EnvServer>) -> u64 {
        create_with(server, spec()).await
    }

    async fn create_with(server: &Arc<EnvServer>, spec: EnvSpec) -> u64 {
        let router = server.clone().router();
        let req = CreateEnvReq { spec };
        let resp = router
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/envs")
                    .header("content-type", "application/json")
                    .body(serde_json::to_string(&req).unwrap())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        v["id"].as_u64().unwrap()
    }

    #[tokio::test]
    async fn env_lifecycle_over_rest() {
        use axum::body::Body;
        let srv = server();
        let router = srv.clone().router();
        let id = create(&srv).await;

        // obs contains the instruction and tool hint
        let resp = router
            .clone()
            .oneshot(get_req(&format!("/envs/{id}/obs")))
            .await
            .unwrap();
        let v: Value = body_json(resp).await;
        assert!(v["observation"]
            .as_str()
            .unwrap()
            .contains("activation key"));

        // tool call: read key.txt
        let call = json!({"server": "fs", "tool": "read_file", "params": {"name": "key.txt"}});
        let resp = router
            .clone()
            .oneshot(post_json(&format!("/envs/{id}/tools/call"), &call))
            .await
            .unwrap();
        let v: Value = body_json(resp).await;
        assert!(v["result"].to_string().contains("KEY-7F3A-92Z"));

        // verify with the found key as the final message
        let verify = json!({"final_message": "the key is KEY-7F3A-92Z"});
        let resp = router
            .clone()
            .oneshot(post_json(&format!("/envs/{id}/verify"), &verify))
            .await
            .unwrap();
        let v: Value = body_json(resp).await;
        assert_eq!(v["reward"].as_f64(), Some(1.0));
        assert_eq!(v["masked"].as_bool(), Some(false));

        // reset + destroy
        let resp = router
            .clone()
            .oneshot(post_json(&format!("/envs/{id}/reset"), &json!({})))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let resp = router
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri(format!("/envs/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        // metrics reflect the lifecycle
        let resp = router.oneshot(get_req("/metrics")).await.unwrap();
        let v: Value = body_json(resp).await;
        assert_eq!(v["envs_created"].as_u64(), Some(1));
        assert_eq!(v["envs_destroyed"].as_u64(), Some(1));
        assert_eq!(v["tool_calls"].as_u64(), Some(1));
    }

    #[tokio::test]
    async fn masked_verification_reports_category_not_zero() {
        // a rubric WITH a judged item: the unavailable judge breaks the
        // reward phase -> masked, category reported, reward null
        let srv = Arc::new(EnvServer::new(
            Arc::new(crate::verifier::UnavailableJudge),
            RewardConfig::default(),
        ));
        let router = srv.clone().router();
        let id = create_with(&srv, judged_spec()).await;
        let resp = router
            .oneshot(post_json(&format!("/envs/{id}/verify"), &json!({})))
            .await
            .unwrap();
        let v: Value = body_json(resp).await;
        assert!(v["reward"].is_null());
        assert_eq!(v["masked"].as_bool(), Some(true));
        assert_eq!(v["category"].as_str(), Some("reward/testbed_corrupted"));
    }

    #[tokio::test]
    async fn unknown_env_is_404() {
        let srv = server();
        let router = srv.router();
        let resp = router.oneshot(get_req("/envs/999/obs")).await.unwrap();
        assert_eq!(resp.status(), 404);
    }

    // ---- helpers ----
    fn get_req(uri: &str) -> axum::http::Request<axum::body::Body> {
        axum::http::Request::builder()
            .method("GET")
            .uri(uri)
            .body(axum::body::Body::empty())
            .unwrap()
    }

    fn post_json(uri: &str, body: &Value) -> axum::http::Request<axum::body::Body> {
        axum::http::Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    async fn body_json(resp: axum::response::Response) -> Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    use tower::ServiceExt;
}
