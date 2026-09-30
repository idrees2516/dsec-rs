# dsec-protocol

[![CI](https://github.com/idrees2516/dsec-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/idrees2516/dsec-rs/actions)
[![crates.io](https://img.shields.io/crates/v/dsec-protocol.svg)](https://crates.io/crates/dsec-protocol)

Wire protocol for **Aether**, the per-sandbox proxy in the DSec
(DeepSeek Elastic Compute) architecture — the Rust reimplementation of
the paper's frame multiplexing layer.

## What's inside

- **Frames** — 25-byte header (magic, version, flags, sandbox id,
  channel, request id, CRC-32) + payload; stream-data frames carry
  out-of-band stream ids.
- **Codec** — async `read_frame`/`write_frame` plus
  copy-free `decode_parts`; CRC-32 is hardware-accelerated
  (`crc32fast`, PCLMULQDQ on x86-64).
- **Messages** — the request/response surface (exec, exec-stream,
  filesystem, control: create/pause/resume/destroy, HTTP proxy) shared
  by the edge and every SDK.
- **RNG** — splitmix64, the deterministic PRNG every layer shares so
  simulations replay exactly.

## Example

```rust
use dsec_protocol::codec;
use dsec_protocol::frame::{Channel, Frame};

let frame = Frame::request(Channel::Exec, 42, 7, br#"{"cmd":"ls"}"#.to_vec());
let wire = frame.encode();
// Aether verifies the CRC on every read.
```

Batches coalesce: multiple frames in one write are parsed
frame-by-frame by `read_frame` — the pipelined fast path used by the
SDK.

Crate graph: `dsec-protocol` is the leaf; it is used by
`dsec-storage`, `dsec-runtime`, `dsec-control`, `dsec-sdk`, `dsec-rl`
and `dsec-firecracker`.
