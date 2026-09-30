# dsec-bench

[![CI](https://github.com/idrees2516/dsec-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/idrees2516/dsec-rs/actions)
[![crates.io](https://img.shields.io/crates/v/dsec-bench.svg)](https://crates.io/crates/dsec-bench)

Benchmark suite for the dsec-rs workspace — the numbers that anchor
the reimplementation to the paper's reported figures and drive the
optimization work.

## Install & run

```sh
cargo install dsec-bench
dsec-bench            # full suite; JSON + markdown to ./dsec-rs-bench/
dsec-bench --quick    # 10% scale smoke pass (CI uses this)
dsec-bench creation --count 5000 --concurrency 256
dsec-bench rl --envs 64 --steps 200000
dsec-bench burst --count 20000
```

## Suites

creation rate + latency percentiles, burst lifecycle (create/exec/
destroy at concurrency), pause/resume, keep-alive apiserver RPS,
placement decisions/s, pack_diff size ratio + throughput, RL
throughput (fast envs, buffer enqueue, GAE, driver e2e), and the two
sandbox-env paths (per-env reference vs pipelined batch pool).

Every result carries a reference line (the paper's number or the
PufferLib reference) so regressions against the base case are visible
at a glance.

Internal profiling harnesses (pprof flamegraphs) live in the
repository as the unpublished `dsec-profiling` crate.
