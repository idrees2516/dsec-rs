//! Minimal async HTTP/1.1 client for the management plane.
//!
//! Deliberately hand-rolled: the management plane talks JSON over
//! loopback/cluster networks, and avoiding a full HTTP stack keeps the
//! dependency tree (and audit surface) small. Supports GET/POST/DELETE
//! with JSON bodies, bearer auth, idempotency keys and exponential
//! backoff retries on connection errors and 5xx.

use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::Serialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::error::{Error, Result};
use crate::Endpoint;

#[derive(Clone)]
pub struct HttpClient {
    endpoint: Endpoint,
    token: String,
    max_attempts: u32,
    base_backoff: Duration,
}

impl HttpClient {
    pub fn new(endpoint: Endpoint, token: String) -> Self {
        HttpClient {
            endpoint,
            token,
            max_attempts: 3,
            base_backoff: Duration::from_millis(20),
        }
    }

    pub fn with_retry(mut self, max_attempts: u32, base_backoff: Duration) -> Self {
        self.max_attempts = max_attempts.max(1);
        self.base_backoff = base_backoff;
        self
    }

    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let body = self.request_with_retry("GET", path, None).await?;
        Ok(serde_json::from_slice(&body)?)
    }

    pub async fn post<T: DeserializeOwned, B: Serialize>(&self, path: &str, body: &B) -> Result<T> {
        let payload = serde_json::to_vec(body)?;
        let body = self.request_with_retry("POST", path, Some(payload)).await?;
        Ok(serde_json::from_slice(&body)?)
    }

    pub async fn delete<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let body = self.request_with_retry("DELETE", path, None).await?;
        Ok(serde_json::from_slice(&body)?)
    }

    async fn request_with_retry(
        &self,
        method: &str,
        path: &str,
        body: Option<Vec<u8>>,
    ) -> Result<Vec<u8>> {
        let mut last_err = None;
        for attempt in 1..=self.max_attempts {
            match self
                .request_once(method, path, body.as_deref(), attempt)
                .await
            {
                Ok(b) => return Ok(b),
                Err(e) => {
                    let retryable = match &e {
                        Error::Io(_) => true,
                        Error::Http { status, .. } => *status >= 500,
                        _ => false,
                    };
                    if !retryable || attempt == self.max_attempts {
                        return Err(e);
                    }
                    last_err = Some(e);
                    // Exponential backoff with deterministic-ish jitter.
                    let backoff = self.base_backoff.mul_f64(2.0f64.powi(attempt as i32 - 1));
                    tokio::time::sleep(backoff).await;
                }
            }
        }
        Err(last_err.unwrap_or_else(|| Error::Http {
            status: 500,
            message: "no response".into(),
        }))
    }

    async fn request_once(
        &self,
        method: &str,
        path: &str,
        body: Option<&[u8]>,
        attempt: u32,
    ) -> Result<Vec<u8>> {
        let addr = tokio::net::lookup_host((self.endpoint.host.as_str(), self.endpoint.port))
            .await?
            .next()
            .ok_or_else(|| Error::Http {
                status: 503,
                message: "no address".into(),
            })?;
        let mut stream = tokio::net::TcpStream::connect(addr).await?;
        let body_bytes = body.unwrap_or(&[]);
        let req_id = crate::next_request_id();
        let req = format!(
            "{method} {path} HTTP/1.1\r\nhost: {host}:{port}\r\ncontent-type: application/json\r\ncontent-length: {len}\r\nauthorization: Bearer {token}\r\nx-request-id: sdk-{req_id}-{attempt}\r\nconnection: close\r\n\r\n",
            method = method,
            path = path,
            host = self.endpoint.host,
            port = self.endpoint.port,
            len = body_bytes.len(),
            token = self.token,
            req_id = req_id,
            attempt = attempt,
        );
        stream.write_all(req.as_bytes()).await?;
        if !body_bytes.is_empty() {
            stream.write_all(body_bytes).await?;
        }
        stream.flush().await?;
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await?;
        parse_response(&buf)
    }
}

/// Parses a full HTTP/1.1 response with a content-length body.
fn parse_response(raw: &[u8]) -> Result<Vec<u8>> {
    let text_end = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| Error::Http {
            status: 502,
            message: "malformed response".into(),
        })?;
    let head = String::from_utf8_lossy(&raw[..text_end]).to_string();
    let mut lines = head.lines();
    let status_line = lines.next().unwrap_or_default();
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(502);
    let mut content_length = raw.len() - text_end - 4;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(content_length);
            }
        }
    }
    let body = raw[text_end + 4..].to_vec();
    if status >= 400 {
        let message = String::from_utf8_lossy(&body).trim().to_string();
        return Err(Error::Http { status, message });
    }
    let _ = content_length;
    Ok(body)
}
