# dsec-profiling

Sampling-profiler harness for the sandbox-env stepping path — the tool
behind the §10 evidence in [`docs/performance.md`](../../docs/performance.md).

`dsec-bench` tells you *how fast*; `dsec-profiling` tells you *where the
time goes*. It runs the exact bench workload (the deterministic
`(i % 6)` action schedule over a `BatchedSandboxEnvPool`) under a
SIGPROF-based sampler — unprivileged, no `perf` needed — and reports:

- a **flamegraph SVG** (`sandbox-env-<mode>.svg`),
- a **leaf-symbol profile** (self time per function),
- the **wall-time share inside the Aether data plane**
  (`call`/`call_batch`) — the I/O-boundness measurement that justified
  the pipelined batch data plane of v0.2.0.

## Usage

```bash
cargo run -p dsec-profiling --release -- [serial|batch] [envs] [seconds] [out.svg]
```

Defaults: `batch`, 8 envs, 4 s, `sandbox-env-batch.svg`. Build with
`--release`: the profile is meaningless in debug builds.

## Layout

One file, [`src/main.rs`](src/main.rs): argument parsing, the
`pprof`-driven sampler guard, the workload loop, and the three report
renderers.

## Status

Internal tooling — `publish = false` (see
[`docs/publishing.md`](../../docs/publishing.md)). No tests of its own;
it exercises `dsec-rl`'s public API, which is covered by `dsec-rl`'s
suite.
