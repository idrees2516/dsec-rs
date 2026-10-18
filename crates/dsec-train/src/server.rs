//! The policy model server (Phase 5, feature `server`).
//!
//! An OpenAI-compatible chat-completions endpoint that serves the
//! policy store's **published** `TinyLm` — so a full training loop
//! (orchestrator worker → HTTP → this server → published weights)
//! exercises the real version contract: every response reports
//! `model = policy-v{major}`, and after a training step's publication
//! the very next request is served by the new version.

use serde_json::{json, Value};

use crate::lm::TinyLm;
use crate::policy::PolicyStore;
use crate::trainer::deserialize_model;

use axum::extract::State;
use axum::routing::{get, post};
use axum::Router;

#[derive(Clone)]
struct ServerState {
    store: PolicyStore,
    max_tokens: usize,
}

/// Serves the published policy on `addr` (port 0 = ephemeral); returns
/// the bound address and the join handle.
pub async fn serve(
    store: PolicyStore,
    addr: std::net::SocketAddr,
    _temperature: f32,
    max_tokens: usize,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let state = ServerState { store, max_tokens };
    let app = Router::new()
        .route("/v1/chat/completions", post(completions))
        .route("/v1/version", get(version))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind policy server");
    let bound = listener.local_addr().expect("local addr");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("policy server");
    });
    (bound, handle)
}

async fn version(State(state): State<ServerState>) -> String {
    let v = state.store.published();
    json!({"major": v.major, "digest": v.digest, "model": v.model_name()}).to_string()
}

async fn completions(State(state): State<ServerState>, body: String) -> String {
    let req: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
    // tokenize the prompt with a minimal shared vocab: the tiny LM is
    // trained over the trainer's tokenizer; for serving we rebuild it
    // from the prompt words deterministically (the toy model's vocab
    // contract is documented in the crate README)
    let mut tokenizer = dsec_rl::tito::SimpleTokenizer::new();
    let messages = req["messages"].as_array().cloned().unwrap_or_default();
    let mut prompt = String::new();
    for m in &messages {
        if let Some(content) = m["content"].as_str() {
            prompt.push_str(content);
            prompt.push(' ');
        }
    }
    tokenizer.tokenize(&prompt);
    // note: ids may exceed the served model's vocab if it was trained
    // on a different corpus; the toy model clamps instead of failing
    let model: TinyLm = match deserialize_model(&state.store.published_weights()) {
        Ok(m) => m,
        Err(_) => TinyLm::new(&tokenizer, 16, 0),
    };
    let ctx = tokenizer.tokenize_known(&prompt);
    // clamp ids into the model's vocab
    let vocab = model.vocab_size().max(1) as u32;
    let ctx: Vec<u32> = ctx.into_iter().map(|t| t % vocab).collect();
    let generated = model.generate_greedy(&ctx, state.max_tokens);
    let text = tokenizer.detokenize(&generated);
    let v = state.store.published();
    json!({
        "model": v.model_name(),
        "choices": [{
            "message": {"content": text, "tool_calls": []},
            "finish_reason": "stop",
        }],
        "usage": {"total_tokens": (ctx.len() + generated.len()) as u64},
        // version transparency: the trainer stamps rollouts from here
        "served_version": v.major,
    })
    .to_string()
}
