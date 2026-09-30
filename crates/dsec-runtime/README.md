# dsec-runtime

[![CI](https://github.com/idrees2516/dsec-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/idrees2516/dsec-rs/actions)
[![crates.io](https://img.shields.io/crates/v/dsec-runtime.svg)](https://crates.io/crates/dsec-runtime)

The node-local runtime of the DSec reimplementation: the **EdgeNode**
data plane. One node serves thousands of sandboxes across the four
paper backends (FnCall warm pool, Container, MicroVM, FullVM) over the
Aether multiplexed transport, with Chronus as the in-sandbox session
layer.

## What's inside

- **EdgeNode** — admission (lock-free optimistic reserve/verify/
  rollback), lifecycle state machine (creating → ready → paused →
  destroying), pause-time memory reclaim and resume-time
  MADV_WILLNEED-style prefetch, metrics histograms.
- **Aether** — per-connection multiplexing client/server. Pipelined
  `call_batch` sends N requests as one coalesced transmission awaited
  under a single deadline; both reader loops drain whole batches per
  wake (`recv_many`), so a tick costs one wakeup chain instead of N.
  Transports: in-process channels (deterministic simulation) and real
  UDS (production data plane).
- **Backends** — `BackendFactory` + `BackendInstance` for all four
  paper backends, with the simulated latency profiles the paper
  measures (FnCall ~5 ms, MicroVM ~900 ms, ...).
- **MicrovmDriver** — the trait real VMM implementations plug into;
  `dsec-firecracker` provides one. Install with
  `BackendFactory::with_microvm_driver(...)` and MicroVM create/pause/
  resume/destroy route through the real VMM API.
- **Chronus** — sessions (cwd/env), the shell-like command interpreter
  (including `&&` compound lines, redirection), streamed exec,
  proxied HTTP.

## Example

```rust,no_run
use std::sync::Arc;
use dsec_runtime::{EdgeNode, AetherClient};
use dsec_protocol::frame::Channel;
use dsec_protocol::message::{Request, Response};

# fn main() {} // async plumbing omitted; see examples/ in the repo
# #[allow(dead_code)]
async fn demo(node: Arc<EdgeNode>, client: &AetherClient) {
    // Pipelined batch: one transmission, one deadline, N replies.
    let calls = vec![
        (1, Channel::Exec, Request::Exec { cmd: "ls /task".into(), timeout_ms: None }),
        (2, Channel::Exec, Request::Exec { cmd: "hostname".into(), timeout_ms: None }),
    ];
    let replies: Vec<Result<Response, _>> = client.call_batch(calls).await;
    let _ = replies;
    let _ = node;
}
```

Used by `dsec-control` (nodes behind the control plane), `dsec-sdk`,
`dsec-rl` (sandbox envs) and `dsec-firecracker`.
