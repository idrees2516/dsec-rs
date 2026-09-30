//! Minimal HTTP/1.1 client for the Firecracker VMM API over its Unix
//! socket.
//!
//! Firecracker exposes a REST-ish API (PUT/PATCH/GET on fixed paths)
//! with JSON bodies and `Content-Length` framed responses. One
//! keep-alive connection is reused for every request on a socket —
//! exactly how the `firecracker` binary expects to be driven.

use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

/// One HTTP response from the VMM.
#[derive(Debug, Clone)]
pub struct FcResponse {
    pub status: u16,
    pub body: String,
}

impl FcResponse {
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Firecracker errors carry `{"fault_message": "..."}`.
    pub fn fault(&self) -> Option<String> {
        serde_json::from_str::<serde_json::Value>(&self.body)
            .ok()
            .and_then(|v| {
                v.get("fault_message")
                    .and_then(|f| f.as_str())
                    .map(String::from)
            })
    }
}

/// Keep-alive HTTP/1.1 client over one UDS connection.
pub struct FcClient {
    sock: PathBuf,
    stream: Option<UnixStream>,
    request_timeout: Duration,
}

impl FcClient {
    pub fn new(sock: impl Into<PathBuf>, request_timeout: Duration) -> Self {
        FcClient {
            sock: sock.into(),
            stream: None,
            request_timeout,
        }
    }

    /// The API socket path this client talks to.
    pub fn sock(&self) -> &Path {
        &self.sock
    }

    async fn ensure_connected(&mut self) -> io::Result<()> {
        if self.stream.is_none() {
            self.stream = Some(UnixStream::connect(&self.sock).await?);
        }
        Ok(())
    }

    /// Issues one request; reconnects once if the keep-alive connection
    /// was closed by the peer.
    pub async fn request(
        &mut self,
        method: &str,
        path: &str,
        body: Option<&str>,
    ) -> io::Result<FcResponse> {
        let mut last_err = None;
        for attempt in 0..2 {
            if attempt == 1 {
                self.stream = None; // force reconnect
            }
            if let Err(e) = self.ensure_connected().await {
                last_err = Some(e);
                continue;
            }
            match self.request_once(method, path, body).await {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    self.stream = None;
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| io::Error::other("request failed")))
    }

    async fn request_once(
        &mut self,
        method: &str,
        path: &str,
        body: Option<&str>,
    ) -> io::Result<FcResponse> {
        let body = body.unwrap_or("");
        let stream = self.stream.as_mut().expect("connected");
        let req = format!(
            "{method} {path} HTTP/1.1\r\nHost: firecracker\r\nUser-Agent: dsec-firecracker\r\nAccept: */*\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{}",
            body.len(),
            body
        );
        let fut = async {
            stream.write_all(req.as_bytes()).await?;
            stream.flush().await?;
            read_response(stream).await
        };
        match tokio::time::timeout(self.request_timeout, fut).await {
            Ok(r) => r,
            Err(_) => Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("vmm request {method} {path} timed out"),
            )),
        }
    }

    pub async fn put(&mut self, path: &str, body: &str) -> io::Result<FcResponse> {
        self.request("PUT", path, Some(body)).await
    }

    pub async fn patch(&mut self, path: &str, body: &str) -> io::Result<FcResponse> {
        self.request("PATCH", path, Some(body)).await
    }

    pub async fn get(&mut self, path: &str) -> io::Result<FcResponse> {
        self.request("GET", path, None).await
    }
}

/// Reads one HTTP/1.1 response (headers + Content-Length body).
async fn read_response(stream: &mut UnixStream) -> io::Result<FcResponse> {
    let mut buf: Vec<u8> = Vec::with_capacity(512);
    let mut chunk = [0u8; 1024];
    // Headers.
    let header_end = loop {
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos;
        }
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "vmm closed"));
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let headers = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let status = headers
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse::<u16>().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad status line"))?;
    let content_length: usize = headers
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            if k.eq_ignore_ascii_case("content-length") {
                v.trim().parse().ok()
            } else {
                None
            }
        })
        .unwrap_or(0);
    let mut body: Vec<u8> = buf[header_end + 4..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "body cut short",
            ));
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);
    Ok(FcResponse {
        status,
        body: String::from_utf8_lossy(&body).to_string(),
    })
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_subslice() {
        assert_eq!(find_subslice(b"abc\r\n\r\ndef", b"\r\n\r\n"), Some(3));
        assert_eq!(find_subslice(b"abcdef", b"\r\n\r\n"), None);
    }
}
