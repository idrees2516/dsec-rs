# dsec-karotte

**A Rust port of [Karotte](https://github.com/preferencemodel/karotte)** — the
open-source framework for building *robust* RL environments: agents trained
inside VMs they can attack, graded by judges they cannot reach.

Karotte's design premise is that the RL student is a *trained adversary*:
whatever raises its score gets reinforced — including reading the answer off
disk, killing the grader, or planting symlinks for it to follow. The
framework's answer is layered confinement plus a scoring model that keeps
**student misbehavior** (score 0), **task failure** (score 0), and
**infrastructure failure** (mask the run) strictly separate.

This crate ports the framework's core as deterministic userspace logic: rule
builders, protocol models, and algorithms that produce and consume the *same
artifacts* as upstream (iptables commands, cgroup file writes, event JSON,
judge trees) — testable without root or VMs. Everything runs as a simulation:
no namespaces, no root, no network.

```
cargo run -p dsec-karotte --example full_run
```

## The one contract that matters

Every security decision returns a `Contract`:

| value            | meaning                                        |
|------------------|------------------------------------------------|
| `Prevented`      | the kernel refused it at the boundary           |
| `Reaped`         | a watchdog detected it and reaped it            |
| `Unsupported`    | no mechanism on this host — said out loud       |

The third state exists because **a write that "succeeded" is not evidence of
enforcement**: gVisor accepts cgroup writes and iptables rules and enforces
neither. Karotte detects this and refuses to pretend.

And the scoring contract (upstream `StudentMisbehaviorError`): an attack on
the harness is *scored zero and labeled*, never an infra error, never a
silent pass:

- symlink planted at the submission path → `misbehavior` metadata, score 0
- FIFO planted to hang the grader → `misbehavior`, score 0
- unkillable process cohort before grading → `misbehavior`, score 0
- grader crashed on its own → `ErrorEvent`, run masked

## The stack, layer by layer

| module           | role |
|------------------|------|
| `schemas`        | chat messages, scorings, the 14-event union, transcript, run state, run config — wire-compatible JSON (`type` discriminators, `structuredContent`/`isError` MCP renames) |
| `confinement`    | the `Contract` tri-state, resource limits (1 GiB harness reserve, 2048 procs, 10⁶ files, 80% disk), the iptables owner-match firewall builder in the exact upstream order, canary self-tests, the cohort-reap state machine (cgroup kill → pidns kill → in-cohort `kill(-1)` ×100 → `/proc` sweep), demotion sequence |
| `cgroups`        | `/proc/mounts` parsing (v1+v2), the student-group model: `memory.max` **paired with `memory.swap.max=0`**, `pids.max`, `memory.oom.group=1`, atomic `cgroup.kill` |
| `memory_watch`   | the watchdog weigher: RSS → PSS on the precise pass, tmpfs `st_blocks` scratch, SysV segments charged to the *creator*; reap keeps watching (SIGKILL survives tmpfs) |
| `reclaim`        | the fd-safe sweep: dir-fd-relative ops only, `O_NOFOLLOW`, virtual-fstype sparing, rmdir-on-unwind, 600 s deadline → misbehavior |
| `submission`     | defensive custody copy: no symlinks, no special files, 256 MiB/file · 1 GiB total · 10⁴ entries · depth 32, sparse holes, provokable-errno mapping |
| `judges`         | regex, rubric (LLM-backed, YES/NO parsing, weighted, strict `>`), executable (output-file rewrite, `>=`), `And`/`Or` short-circuit composites with the upstream Unicode tree (✔/❗/⊘, ├─/└─) |
| `model_spec`     | the catalog tables: effort ladders per family, 128 k output tiers, gVisor... grok temperature 0.7, provider request shaping, deepseek/xai tool-call repairs |
| `message_loop`   | run-wide turn limit (`>`), between-turn time/context checks, 3-empty-turn nudge, buffered tool calls with first-block projection, "Time remaining: N seconds" / "Context remaining: N" injection, the 16-attempt retry ladder with Retry-After clamping |
| `mcp`            | the JSON-RPC subset (initialize / tools/list / tools/call / the self-removing `register_tools` meta-tool), discovery validation, the 0.2 s resource sampler with `_karotte_resource_metrics` injection |
| `tools`          | `bash` (marker fd 231, `<<exit>>` nonce epilogue, 1 MiB hard cap + 8192 tail, syntax-check exit 2), `view_lines_in_file` (sed ranges, 384 KiB JSON-safe cap), `replace_in_file` (10 MiB cap, hand-rolled unified diff with `\ No newline at end of file`), `view_image_file` (5 MB, dims ≤ 2000) |
| `runner`         | the exact event order (golden upstream): `task_started → pre-hook → system → [step_started → …agent… → scoring → step_completed] → task_completed`, misbehavior folded into score 0, durable transcript write no matter what |
| `streaming`      | the three fan-outs: stdout renderer (👤/🗣️/🔧/✅/🏁), cursor-replay websocket broadcaster, positional backend calls (chunk-free) |
| `runtime`        | runtime selection matrix, the Firecracker plan: pinned artifacts, `console=ttyS0 … init=/.karotte/init` boot args, vda/vdb/vdc/vdd drive order, vsock CID 3 + heartbeat port 52, 120 s watchdog, TEST-NET-1 tap + egress filter with host-IP rejection |

## Upstream fidelity notes

Constants, thresholds, orderings, and message strings are ported verbatim
(1 GiB harness reserve, 2048 process limit, 600 s reclaim deadline, 30 s
kill deadline, 100 `kill(-1)` rounds, the empty-turn nudge text, the
truncation notes, the nudge/marker/counters, the judge tree glyphs). The
Linux-syscall layer (namespaces, cgroupfs, iptables) is modeled behind
small traits (`FirewallBackend`, `CohortBackend`, `ProcView`, `FsView`,
`CustodyFs`, `ExecBackend`) — the port builds the exact commands and file
contents, and the in-memory backends verify the algorithms.

Two deliberate divergences, documented in code: Python floats format as
`1.0` where Rust prints `1` (the judge tree formatter keeps Python's
shape), and `serde_json` object keys iterate in sorted order (upstream MCP
parts are key-order-insensitive).

## Tests

178 unit tests, deterministic, covering: the masking/misbehavior contract
both ways, the firewall rule order, canaries, cohort-reap semantics (the
load-bearing "clean sweep counts only when the in-cohort kill ran"),
cgroup write pairing, the PSS/tmpfs/SysV weigher, fd-safe sweep and
custody refusals, judge composition and short-circuit trees, rubric
thresholds (`>` vs `>=`), the message-loop limit semantics, MCP protocol
shapes, tool caps and diff rendering, event ordering, fan-out replay.

```bash
cargo test -p dsec-karotte
cargo clippy -p dsec-karotte --all-targets -- -D warnings
```
