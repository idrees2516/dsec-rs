# dsec-sdk

[![CI](https://github.com/idrees2516/dsec-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/idrees2516/dsec-rs/actions)
[![crates.io](https://img.shields.io/crates/v/dsec-sdk.svg)](https://crates.io/crates/dsec-sdk)

The Rust port of the paper's **libdsec** SDK: how agents and training
loops consume sandboxes.

## What's inside

- **DsecClient** — REST client for the apiserver (create/destroy/
  pause/resume/snapshot), token auth.
- **Sandbox** — a live handle: `exec` (batched or single), file
  read/write, streaming exec, pause/resume — all through the node's
  Aether connection.
- **SandboxPool** — create-on-demand + warm pool of sandboxes for
  bursty agent workloads.
- **LocalCluster** — an in-process cluster (control plane + edge node
  + Aether) for tests and examples; the same code paths run against a
  real apiserver in production.
- **Transport** — the connection layer (UDS by default), with
  pipelined batch calls on the hot path.

## Example

```rust,no_run
use dsec_sdk::LocalCluster;
# #[tokio::main]
# async fn main() -> anyhow::Result<()> { // dsec-sdk's own error type in real code
let cluster = LocalCluster::start(dsec_sdk::ClusterConfig::default()).await?;
let sb = cluster.create_agent_sandbox("proj-1").await?;
let out = sb.exec("cat /etc/hostname").await?;
# Ok(())
# }
```

See `crates/dsec-sdk/examples` in the repository for a complete
round trip, and `dsec-rl` for RL training on top of these pools.
