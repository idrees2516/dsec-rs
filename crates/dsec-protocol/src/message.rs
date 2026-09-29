//! Request/response message types carried in frame payloads (JSON).
//!
//! These mirror the Chronus session API from the paper: shell execution,
//! filesystem access, proxied HTTP and streaming I/O, plus the control
//! channel used for lifecycle operations (pause/resume/destroy).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Stable error codes shared across client, apiserver and runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum ErrorCode {
    Ok = 0,
    NotFound = 1,
    InvalidState = 2,
    QuotaExceeded = 3,
    Unauthorized = 4,
    RateLimited = 5,
    Timeout = 6,
    InvalidArgument = 7,
    Unavailable = 8,
    Internal = 9,
}

/// Control-plane / session requests sent by the SDK through Aether.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    /// Liveness probe for the node/sandbox path.
    Ping,
    /// Open a Chronus shell session.
    SessionOpen {
        cwd: String,
        env: HashMap<String, String>,
    },
    /// Close a session.
    SessionClose,
    /// Execute a command to completion.
    Exec {
        cmd: String,
        timeout_ms: Option<u64>,
    },
    /// Execute a command with streamed stdout/stderr chunks.
    ExecStream {
        cmd: String,
    },
    /// Write to the stdin of a live stream.
    StreamWrite {
        stream_id: u32,
        data: Vec<u8>,
    },
    /// Close / EOF a live stream.
    StreamClose {
        stream_id: u32,
    },
    // --- filesystem channel ---
    FsRead {
        path: String,
    },
    FsWrite {
        path: String,
        data: Vec<u8>,
        append: bool,
    },
    FsList {
        path: String,
    },
    FsStat {
        path: String,
    },
    FsMkdir {
        path: String,
    },
    FsRm {
        path: String,
    },
    // --- proxied HTTP channel (egress through the node) ---
    HttpGet {
        url: String,
        headers: Vec<(String, String)>,
    },
    // --- lifecycle control channel ---
    Pause,
    Resume,
    Destroy,
    /// Sandbox status snapshot.
    Status,
}

/// Runtime responses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Response {
    Pong {
        node_id: String,
        epoch_ms: u64,
    },
    SessionOpened {
        session_id: u32,
    },
    SessionClosed,
    Exec {
        exit_code: i32,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        duration_ms: u64,
    },
    StreamOpened {
        stream_id: u32,
    },
    StreamChunk {
        stream_id: u32,
        data: Vec<u8>,
        stream: Stdio,
    },
    StreamFin {
        stream_id: u32,
        exit_code: i32,
    },
    FileData {
        data: Vec<u8>,
    },
    Stat {
        size: u64,
        is_dir: bool,
    },
    Entries {
        names: Vec<String>,
    },
    Done,
    Http {
        status: u16,
        body: Vec<u8>,
    },
    Status {
        state: String,
        cpu_millicores: i64,
        mem_mib: i64,
        uptime_ms: u64,
    },
    Error {
        code: ErrorCode,
        message: String,
    },
}

/// Which stdio stream a chunk belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stdio {
    Stdout,
    Stderr,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serde_roundtrip() {
        let reqs = vec![
            Request::Ping,
            Request::SessionOpen {
                cwd: "/".into(),
                env: HashMap::new(),
            },
            Request::Exec {
                cmd: "cat /etc/hostname".into(),
                timeout_ms: Some(5000),
            },
            Request::ExecStream {
                cmd: "seq 3".into(),
            },
            Request::StreamWrite {
                stream_id: 2,
                data: b"hi\n".to_vec(),
            },
            Request::StreamClose { stream_id: 2 },
            Request::FsRead {
                path: "/etc/hostname".into(),
            },
            Request::FsWrite {
                path: "/tmp/x".into(),
                data: vec![1, 2, 3],
                append: false,
            },
            Request::FsList { path: "/".into() },
            Request::FsStat {
                path: "/etc".into(),
            },
            Request::FsMkdir {
                path: "/data".into(),
            },
            Request::FsRm {
                path: "/data".into(),
            },
            Request::HttpGet {
                url: "https://example.com".into(),
                headers: vec![],
            },
            Request::Pause,
            Request::Resume,
            Request::Destroy,
            Request::Status,
        ];
        for r in reqs {
            let s = serde_json::to_string(&r).unwrap();
            let back: Request = serde_json::from_str(&s).unwrap();
            assert_eq!(back, r);
        }
    }

    #[test]
    fn response_serde_roundtrip() {
        let resps = vec![
            Response::Pong {
                node_id: "n1".into(),
                epoch_ms: 1,
            },
            Response::SessionOpened { session_id: 4 },
            Response::SessionClosed,
            Response::Exec {
                exit_code: 0,
                stdout: b"ok".to_vec(),
                stderr: vec![],
                duration_ms: 12,
            },
            Response::StreamOpened { stream_id: 9 },
            Response::StreamChunk {
                stream_id: 9,
                data: b"chunk".to_vec(),
                stream: Stdio::Stderr,
            },
            Response::StreamFin {
                stream_id: 9,
                exit_code: 0,
            },
            Response::FileData { data: vec![9] },
            Response::Stat {
                size: 5,
                is_dir: false,
            },
            Response::Entries {
                names: vec!["a".into()],
            },
            Response::Done,
            Response::Http {
                status: 200,
                body: b"{}".to_vec(),
            },
            Response::Status {
                state: "ready".into(),
                cpu_millicores: 500,
                mem_mib: 256,
                uptime_ms: 10,
            },
            Response::Error {
                code: ErrorCode::NotFound,
                message: "no sandbox".into(),
            },
        ];
        for r in resps {
            let s = serde_json::to_string(&r).unwrap();
            let back: Response = serde_json::from_str(&s).unwrap();
            assert_eq!(back, r);
        }
    }

    #[test]
    fn op_tag_shape() {
        let s = serde_json::to_string(&Request::Ping).unwrap();
        assert_eq!(s, r#"{"op":"ping"}"#);
    }
}
