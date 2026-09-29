# Security Policy

## Supported versions

| version | supported |
|---------|-----------|
| 0.1.x   | yes       |

## Reporting a vulnerability

**Please do not open public issues for security vulnerabilities.**

Use GitHub's **private vulnerability reporting** on this repository
(Security → Report a vulnerability), or email the maintainers directly.
We aim to acknowledge reports within 48 hours and will coordinate a fix +
disclosure timeline with you.

## Scope notes

dsec-rs is a **deterministic userspace simulation / research reimplementation**
of the DSec paper. It does not itself create real containers, microVMs, or
kernel sandboxing, and it is **not an isolation boundary**:

- The Chronus command interpreter and filesystem model are simulations —
  they must never be pointed at a real host filesystem or given
  untrusted command input with real side effects.
- The fake egress HTTP router does not enforce real network policy;
  eBPF/AppArmor enforcement in the paper is modeled, not implemented.
- The UDS transports are suitable for local, trusted-process topologies.

If you are wiring dsec-rs components to real backends (Firecracker,
containerd, real mounts), treat isolation as your own responsibility and
apply the paper's mitigations (per-sandbox policy profiles, egress control,
idle scheduling) at the real layer.

## Hardening expectations

- All fuzzing-relevant parsing (Aether codec) is CRC32-verified and tested
  against malformed frames; treat external wire input as untrusted.
- Dependencies are kept minimal (tokio, axum, serde, thiserror) and updated
  via Dependabot; run `cargo audit` if you handle secrets or untrusted
  input.
