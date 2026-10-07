//! The Live RL pipeline behind the AgentEnv REST server.
//!
//! Boots the server on a local port, then drives one environment over
//! HTTP exactly like a remote trainer would: create, observe, call a
//! tool, verify, destroy.
//!
//! ```bash
//! cargo run -p dsec-agentenv --features server --example live_pipeline
//! ```

use dsec_agentenv::envgen::templates;
use dsec_agentenv::server::EnvServer;
use dsec_agentenv::verifier::{AnchorJudge, RewardConfig};
use serde_json::{json, Value};
use std::sync::Arc;

#[tokio::main]
async fn main() -> dsec_agentenv::Result<()> {
    // 1. boot the env server
    let srv = Arc::new(EnvServer::new(
        Arc::new(AnchorJudge::new()),
        RewardConfig::default(),
    ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let app = srv.clone().router();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    let base = format!("http://{addr}");
    println!("agentenv server listening on {base}");

    // 2. create an env from the terminal template
    let spec = templates::terminal();
    let resp: Value = reqwest_json(&base, "POST", "/envs", Some(json!({"spec": spec}))).await?;
    let id = resp["id"].as_u64().unwrap();
    println!(
        "created env {id} ({})",
        resp["instance_id"].as_str().unwrap()
    );

    // 3. observe
    let resp: Value = reqwest_json(&base, "GET", &format!("/envs/{id}/obs"), None).await?;
    println!(
        "observation: {}",
        resp["observation"]
            .as_str()
            .unwrap()
            .lines()
            .next()
            .unwrap()
    );

    // 4. tool call (plain reqwest-free JSON over tokio)
    let resp: Value = reqwest_json(
        &base,
        "POST",
        &format!("/envs/{id}/tools/call"),
        Some(json!({"server": "fs", "tool": "read_file", "params": {"name": "key.txt"}})),
    )
    .await?;
    println!(
        "tool result: {}",
        resp["result"]
            .to_string()
            .trim_matches('"')
            .chars()
            .take(60)
            .collect::<String>()
    );

    // 5. verify (the masking contract: masked envs report a category,
    //    never a false zero)
    let resp: Value = reqwest_json(
        &base,
        "POST",
        &format!("/envs/{id}/verify"),
        Some(json!({"final_message": "the activation key is KEY-7F3A-92Z"})),
    )
    .await?;
    if resp["masked"].as_bool().unwrap_or(false) {
        println!(
            "MASKED: {} ({})",
            resp["category"].as_str().unwrap_or("?"),
            resp["reward_error"].as_str().unwrap_or("?")
        );
    } else {
        println!("reward: {}", resp["reward"].as_f64().unwrap());
    }

    // 6. metrics + destroy
    let resp: Value = reqwest_json(&base, "GET", "/metrics", None).await?;
    println!("metrics: {resp}");
    let resp: Value = reqwest_json(&base, "DELETE", &format!("/envs/{id}"), None).await?;
    println!("destroyed: {}", resp["destroyed"].as_bool().unwrap());
    Ok(())
}

/// Minimal JSON HTTP client over tokio (no reqwest dependency).
async fn reqwest_json(
    base: &str,
    method: &str,
    path: &str,
    body: Option<Value>,
) -> dsec_agentenv::Result<Value> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut stream = tokio::net::TcpStream::connect(base.trim_start_matches("http://")).await?;
    let payload = body.map(|b| b.to_string()).unwrap_or_default();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nhost: {base}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
        payload.len()
    );
    stream.write_all(req.as_bytes()).await?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await?;
    let text = String::from_utf8_lossy(&buf);
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    Ok(serde_json::from_str(&body)?)
}
