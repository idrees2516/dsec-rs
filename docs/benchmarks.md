# dsec-rs benchmark results

Machine: 2-core CI-class VM. Numbers below are from the v0.3.0 optimized
build; speedup claims in the README and `docs/performance.md` come from
interleaved A/B runs against baseline binaries, not single-shot runs
(this class of shared machine shows +/-20% run-to-run variance).

| benchmark | result | reference | p50 (ms) | p99 (ms) | notes |
|---|---|---|---|---|---|
| sandbox_create_software_path | 8336.47 creates/sec | DSec paper: ~5,000/s cluster-wide (incl. cold backends) | 0.000 | 0.000 | single simulated node, FnCall warm pool, zero injected latency |
| sandbox_create_paper_latency | 7961.11 creates/sec | DSec paper: ~5,000/s with ~1s avg cold creation | 0.000 | 6.000 | FnCall ~5ms injected latency, 256 concurrent creations |
| burst_lifecycle | 55263.72 sandboxes/sec (create+exec+destroy) | DSec paper: 100k+ concurrent, ~5k/s create rate | - | - | 100000 sandboxes, peak concurrent 2, 100000 execs, 0 errors |
| pause_latency | 4633.88 ms (p50) | DSec paper: ~4s average pause (kernel checkpoint+reclaim) | 4633.875 | 4994.253 | userspace sim: state machine + reclaim accounting only |
| resume_latency | 1738.94 ms (p50) | DSec paper: supports preemption, resume ~1.5s class | 1738.937 | 1874.100 | userspace sim |
| apiserver_rps | 22354.76 requests/sec | stateless apiserver must sustain create-rate traffic | - | - | GET /healthz, 16 concurrent keep-alive-free clients |
| apiserver_rps_keepalive | 112634.12 requests/sec | stateless apiserver must sustain create-rate traffic | - | - | GET /healthz, 16 keep-alive connections, pipelined |
| rl_env_steps_fast | 3003821.02 steps/sec | PufferLib claims ~1M+ steps/s at scale (Cython, large obs batches) | - | - | 64 envs x 4 threads, pure-compute env (no I/O) |
| rl_buffer_enqueue | 10352042.06 transitions/sec | PufferReplayBuffer C core: ~10M+ transitions/s (Cython memcpy) | - | - | obs 128 f32, hidden 64 f32 x2 (actor+critic) |
| rl_gae | 128675072.91 transitions/sec | PufferLib computes GAE in C during train() | - | - | full-window GAE over 64 envs |
| rl_driver_e2e | 176542.31 steps/sec | - | - | - | driver rollout incl. LSTM forward + hidden reset bookkeeping |
| rl_env_steps_sandbox_batched | 35060.65 steps/sec | DSec: RL agent loop over real sandbox sessions | - | - | 8 sandbox envs, pipelined: one coalesced Aether transmission per vectorized step |
| rl_env_steps_sandbox | 30766.25 steps/sec | reference per-env path (one round trip per step) | - | - | 8 sandbox envs, real Chronus exec per step (zero-latency channel transport) |
| packdiff_size_ratio | 0.05 pack bytes / full image bytes | DSec paper: pack_diff replicates working state w/o full copy | - | - | full 2048 KiB, pack 100 KiB (5% of files mutated) |
| packdiff_apply | 34814.50 applies/sec | - | - | - | fresh replica overlay + block import + file table import |
| packdiff_snapshot | 8642295.39 packs/sec | - | - | - | dirty-set capture + serialization (post-first-snapshot management path) |
| placement_decisions | 132677.95 decisions/sec | ~5k creations/s x candidate scan must not bottleneck | - | - | 1024 nodes, k=8 sampled |
| placement_kchoice_quality | 67.65 % of exhaustive-improvement achieved | power-of-two-choices theory: k=8 approximates best-fit well | - | - | avg score k=1 1.7138, k=8 1.7964, exhaustive 1.8359 |
| protocol_encode | 987.33 MB/s | - | - | - | frame + CRC32 over 55-byte JSON payloads |
| protocol_decode | 798.32 MB/s | - | - | - | CRC-verified decode |
