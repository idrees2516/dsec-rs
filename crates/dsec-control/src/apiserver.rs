//! Stateless REST apiserver (axum).
//!
//! Routes mirror the paper's apiserver surface; state is injected as
//! `Arc<ControlPlane>` (the paper keeps it in etcd — the in-process
//! registry is the simulation of that store). Every mutating route
//! requires a bearer token and passes through the rate limiter inside
//! the control plane.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::control::ControlPlane;
use crate::error::Error;
use crate::iam::Role;
use crate::model::{Quota, SandboxRecord, SandboxSpec};

pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl From<Error> for ApiError {
    fn from(e: Error) -> Self {
        let status = match &e {
            Error::ProjectExists(_) | Error::InvalidArgument(_) => StatusCode::CONFLICT,
            Error::ProjectNotFound(_) | Error::SandboxNotFound(_) | Error::NodeNotFound(_) => {
                StatusCode::NOT_FOUND
            }
            Error::ParentNotFound(_) => StatusCode::BAD_REQUEST,
            Error::QuotaExceeded { .. } => StatusCode::UNPROCESSABLE_ENTITY,
            Error::NoCandidate(_) => StatusCode::SERVICE_UNAVAILABLE,
            Error::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            Error::RateLimited(_) => StatusCode::TOO_MANY_REQUESTS,
            Error::ProvisionFailed { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Error::NodeUnhealthy { .. } => StatusCode::SERVICE_UNAVAILABLE,
            Error::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        ApiError {
            status,
            message: e.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

type ApiResult<T> = Result<T, ApiError>;

fn bearer_token(headers: &HeaderMap) -> Result<String, ApiError> {
    if let Some(v) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        if let Some(token) = v.strip_prefix("Bearer ") {
            return Ok(token.to_string());
        }
    }
    if let Some(v) = headers.get("x-dsec-token").and_then(|v| v.to_str().ok()) {
        return Ok(v.to_string());
    }
    Err(ApiError {
        status: StatusCode::UNAUTHORIZED,
        message: "missing bearer token".into(),
    })
}

// -- request / response DTOs -----------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CreateProjectReq {
    pub parent: String,
    pub segment: String,
    #[serde(default)]
    pub quota: Option<Quota>,
}

#[derive(Debug, Serialize)]
pub struct CreateProjectResp {
    pub name: String,
    pub parent: Option<String>,
    pub quota: Quota,
}

#[derive(Debug, Deserialize)]
pub struct CreateTokenReq {
    pub project: String,
    pub role: Role,
}

#[derive(Debug, Deserialize, Default)]
pub struct ListQuery {
    pub project: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClusterResp {
    pub revision: u64,
    pub nodes: usize,
    pub nodes_unhealthy: usize,
    pub sandboxes_total: usize,
    pub sandboxes_active: usize,
    pub sandboxes_paused: usize,
    pub cpu_millicores_available: i64,
    pub mem_mib_available: i64,
}

// -- handlers ----------------------------------------------------------------

async fn healthz() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

async fn metrics_text(State(plane): State<Arc<ControlPlane>>) -> impl IntoResponse {
    let mut text = plane.metrics.expose();
    let summary = plane.cluster_summary();
    text.push_str(&format!(
        "dsec_sandboxes_active {}\n",
        summary.sandboxes_active
    ));
    text.push_str(&format!("dsec_nodes {}\n", summary.nodes));
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        text,
    )
}

async fn create_project(
    State(plane): State<Arc<ControlPlane>>,
    headers: HeaderMap,
    Json(req): Json<CreateProjectReq>,
) -> ApiResult<Json<CreateProjectResp>> {
    let token = bearer_token(&headers)?;
    plane
        .iam
        .authorize(&token, crate::iam::Action::CreateProject, &req.parent)?;
    let quota = req.quota.unwrap_or_default();
    let p = plane.create_project(&req.parent, &req.segment, quota)?;
    Ok(Json(CreateProjectResp {
        name: p.name,
        parent: p.parent,
        quota: p.quota,
    }))
}

async fn get_project(
    State(plane): State<Arc<ControlPlane>>,
    Path(name): Path<String>,
) -> ApiResult<Json<crate::iam::Project>> {
    Ok(Json(plane.iam.project(&name)?))
}

async fn create_token(
    State(plane): State<Arc<ControlPlane>>,
    headers: HeaderMap,
    Json(req): Json<CreateTokenReq>,
) -> ApiResult<Json<crate::iam::TokenInfo>> {
    let token = bearer_token(&headers)?;
    // Only admins mint tokens.
    plane
        .iam
        .authorize(&token, crate::iam::Action::Admin, &req.project)?;
    Ok(Json(plane.create_token(&req.project, req.role)?))
}

async fn create_sandbox(
    State(plane): State<Arc<ControlPlane>>,
    headers: HeaderMap,
    Json(spec): Json<SandboxSpec>,
) -> ApiResult<Json<SandboxRecord>> {
    let token = bearer_token(&headers)?;
    Ok(Json(plane.create_sandbox(&token, spec).await?))
}

async fn list_sandboxes(
    State(plane): State<Arc<ControlPlane>>,
    Query(q): Query<ListQuery>,
) -> ApiResult<Json<Vec<SandboxRecord>>> {
    let mut records = plane.registry.sandboxes();
    if let Some(project) = q.project {
        records.retain(|r| {
            r.spec.project == project
                || r.spec.project.starts_with(&format!("{}/", project))
                || project == "root"
        });
    }
    Ok(Json(records))
}

async fn get_sandbox(
    State(plane): State<Arc<ControlPlane>>,
    Path(sid): Path<u64>,
) -> ApiResult<Json<SandboxRecord>> {
    Ok(Json(
        plane
            .registry
            .sandbox(sid)
            .ok_or(Error::SandboxNotFound(sid))?,
    ))
}

async fn delete_sandbox(
    State(plane): State<Arc<ControlPlane>>,
    headers: HeaderMap,
    Path(sid): Path<u64>,
) -> ApiResult<Json<SandboxRecord>> {
    let token = bearer_token(&headers)?;
    Ok(Json(plane.destroy_sandbox(&token, sid).await?))
}

async fn pause_sandbox(
    State(plane): State<Arc<ControlPlane>>,
    headers: HeaderMap,
    Path(sid): Path<u64>,
) -> ApiResult<Json<SandboxRecord>> {
    let token = bearer_token(&headers)?;
    Ok(Json(plane.pause_sandbox(&token, sid).await?))
}

async fn resume_sandbox(
    State(plane): State<Arc<ControlPlane>>,
    headers: HeaderMap,
    Path(sid): Path<u64>,
) -> ApiResult<Json<SandboxRecord>> {
    let token = bearer_token(&headers)?;
    Ok(Json(plane.resume_sandbox(&token, sid).await?))
}

async fn list_nodes(
    State(plane): State<Arc<ControlPlane>>,
) -> ApiResult<Json<Vec<crate::model::NodeInfo>>> {
    Ok(Json(plane.registry.nodes()))
}

async fn heartbeat(
    State(plane): State<Arc<ControlPlane>>,
    headers: HeaderMap,
    Path(node_id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let token = bearer_token(&headers)?;
    plane
        .iam
        .authorize(&token, crate::iam::Action::Heartbeat, "root")?;
    plane.heartbeat(&node_id).await?;
    Ok(Json(json!({ "ok": true })))
}

async fn cluster(State(plane): State<Arc<ControlPlane>>) -> ApiResult<Json<ClusterResp>> {
    let s = plane.cluster_summary();
    Ok(Json(ClusterResp {
        revision: s.revision,
        nodes: s.nodes,
        nodes_unhealthy: s.nodes_unhealthy,
        sandboxes_total: s.sandboxes_total,
        sandboxes_active: s.sandboxes_active,
        sandboxes_paused: s.sandboxes_paused,
        cpu_millicores_available: s.cpu_millicores_available,
        mem_mib_available: s.mem_mib_available,
    }))
}

// -- router ------------------------------------------------------------------

pub fn router(plane: Arc<ControlPlane>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics_text))
        .route("/v1/projects", post(create_project))
        .route("/v1/projects/{name}", get(get_project))
        .route("/v1/tokens", post(create_token))
        .route("/v1/sandboxes", post(create_sandbox).get(list_sandboxes))
        .route(
            "/v1/sandboxes/{id}",
            get(get_sandbox).delete(delete_sandbox),
        )
        .route("/v1/sandboxes/{id}/pause", post(pause_sandbox))
        .route("/v1/sandboxes/{id}/resume", post(resume_sandbox))
        .route("/v1/nodes", get(list_nodes))
        .route("/v1/nodes/{id}/heartbeat", post(heartbeat))
        .route("/v1/cluster", get(cluster))
        .with_state(plane)
}

/// Binds and serves in the background; returns the bound address.
pub async fn serve(plane: Arc<ControlPlane>, bind: &str) -> std::io::Result<SocketAddr> {
    let listener = tokio::net::TcpListener::bind(bind).await?;
    let addr = listener.local_addr()?;
    let app = router(plane);
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Ok(addr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Locality, NodeStatus};

    #[tokio::test]
    async fn healthz_over_http() {
        let plane = Arc::new(ControlPlane::new(1, 2));
        let addr = serve(plane, "127.0.0.1:0").await.unwrap();
        let resp = reqwest_free_get(addr, "/healthz").await;
        assert!(resp.0 == 200, "status {}", resp.0);
        assert!(resp.1.contains("ok"));
    }

    #[tokio::test]
    async fn create_sandbox_requires_auth() {
        let plane = Arc::new(ControlPlane::new(1, 2));
        let addr = serve(plane, "127.0.0.1:0").await.unwrap();
        let body = serde_json::to_string(&SandboxSpec::default()).unwrap();
        let resp = reqwest_free_post(addr, "/v1/sandboxes", &body, None).await;
        assert_eq!(resp.0, 401, "{}", resp.1);
    }

    #[tokio::test]
    async fn sandbox_crud_over_http() {
        let plane = Arc::new(ControlPlane::new(1, 2));
        plane.register_node(crate::model::NodeInfo {
            node_id: "n1".into(),
            cpu_millicores: 4000,
            mem_mib: 4096,
            max_sandboxes: 10,
            cpu_available: 4000,
            mem_available: 4096,
            slots_available: 10,
            status: NodeStatus::Healthy,
            locality: Locality::Local,
            images: vec!["dsec/agent-base".into()],
            labels: Default::default(),
            admitted_projects: vec!["root".into()],
            cost_multiplier: 1.0,
        });
        let token = plane.create_token("root", Role::Admin).unwrap().token;
        let addr = serve(plane.clone(), "127.0.0.1:0").await.unwrap();

        // Create.
        let body = serde_json::to_string(&SandboxSpec::default()).unwrap();
        let (status, text) = reqwest_free_post(addr, "/v1/sandboxes", &body, Some(&token)).await;
        assert_eq!(status, 200, "{}", text);
        let rec: SandboxRecord = serde_json::from_str(&text).unwrap();
        assert_eq!(rec.node_id, "n1");

        // Read.
        let (s, get_text) = reqwest_free_get(addr, &format!("/v1/sandboxes/{}", rec.sid)).await;
        assert_eq!(s, 200);
        let rec2: SandboxRecord = serde_json::from_str(&get_text).unwrap();
        assert_eq!(rec2.sid, rec.sid);

        // Pause + resume + delete.
        let (s, _) = reqwest_free_post(
            addr,
            &format!("/v1/sandboxes/{}/pause", rec.sid),
            "{}",
            Some(&token),
        )
        .await;
        assert_eq!(s, 200);
        let (s, _) = reqwest_free_post(
            addr,
            &format!("/v1/sandboxes/{}/resume", rec.sid),
            "{}",
            Some(&token),
        )
        .await;
        assert_eq!(s, 200);
        // Delete uses the raw HTTP DELETE.
        let (s, _) = raw_request(
            addr,
            "DELETE",
            &format!("/v1/sandboxes/{}", rec.sid),
            "",
            Some(&token),
        )
        .await;
        assert_eq!(s, 200);
        // Gone.
        let (s, _) = reqwest_free_get(addr, &format!("/v1/sandboxes/{}", rec.sid)).await;
        assert_eq!(s, 404);
    }

    #[tokio::test]
    async fn metrics_exposed() {
        let plane = Arc::new(ControlPlane::new(1, 2));
        plane.metrics.incr("dsec_test_total");
        let addr = serve(plane, "127.0.0.1:0").await.unwrap();
        let (s, text) = reqwest_free_get(addr, "/metrics").await;
        assert_eq!(s, 200);
        assert!(text.contains("dsec_test_total 1"));
        assert!(text.contains("dsec_sandboxes_active"));
    }

    // -- minimal HTTP/1.1 client over tokio (no external deps) ------------

    async fn raw_request(
        addr: SocketAddr,
        method: &str,
        path: &str,
        body: &str,
        token: Option<&str>,
    ) -> (u16, String) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut req = format!(
            "{} {} HTTP/1.1\r\nhost: {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n",
            method,
            path,
            addr,
            body.len()
        );
        if let Some(t) = token {
            req.push_str(&format!("authorization: Bearer {}\r\n", t));
        }
        req.push_str("\r\n");
        req.push_str(body);
        stream.write_all(req.as_bytes()).await.unwrap();
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await.unwrap();
        let text = String::from_utf8_lossy(&buf).to_string();
        let status: u16 = text
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let body = text
            .split_once("\r\n\r\n")
            .map(|(_, b)| b.to_string())
            .unwrap_or_default();
        (status, body)
    }

    async fn reqwest_free_get(addr: SocketAddr, path: &str) -> (u16, String) {
        raw_request(addr, "GET", path, "", None).await
    }

    async fn reqwest_free_post(
        addr: SocketAddr,
        path: &str,
        body: &str,
        token: Option<&str>,
    ) -> (u16, String) {
        raw_request(addr, "POST", path, body, token).await
    }
}
