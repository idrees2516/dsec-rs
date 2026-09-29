# dsec-rs benchmark results

- date: unix:1790712342
- rustc: unknown
- cores: 2

| benchmark | result | reference | p50 (ms) | p99 (ms) | notes |
|---|---|---|---|---|---|
| sandbox_create_software_path | 3416.90 creates/sec | DSec paper: ~5,000/s cluster-wide (incl. cold backends) | 0.000 | 0.000 | single simulated node, FnCall warm pool, zero injected latency |
| sandbox_create_paper_latency | 5806.14 creates/sec | DSec paper: ~5,000/s with ~1s avg cold creation | 0.000 | 6.000 | FnCall ~5ms injected latency, 256 concurrent creations |
| burst_lifecycle | 54731.10 sandboxes/sec (create+exec+destroy) | DSec paper: 100k+ concurrent, ~5k/s create rate | - | - | 100000 sandboxes, peak concurrent 2, 100000 execs, 0 errors |
| pause_latency | 4633.87 ms (p50) | DSec paper: ~4s average pause (kernel checkpoint+reclaim) | 4633.875 | 4994.182 | userspace sim: state machine + reclaim accounting only |
| resume_latency | 1738.34 ms (p50) | DSec paper: supports preemption, resume ~1.5s class | 1738.340 | 1873.109 | userspace sim |
| apiserver_rps | 20869.17 requests/sec | stateless apiserver must sustain create-rate traffic | - | - | GET /healthz, 16 concurrent keep-alive-free clients |
| rl_env_steps_fast | 551449.23 steps/sec | PufferLib claims ~1M+ steps/s at scale (Cython, large obs batches) | - | - | 64 envs x 4 threads, pure-compute env (no I/O) |
| rl_buffer_enqueue | 12241805.64 transitions/sec | PufferReplayBuffer C core: ~10M+ transitions/s (Cython memcpy) | - | - | obs 128 f32, hidden 64 f32 x2 (actor+critic) |
| rl_gae | 22601929.14 transitions/sec | PufferLib computes GAE in C during train() | - | - | full-window GAE over 64 envs |
| rl_driver_e2e | 132488.16 steps/sec | - | - | - | driver rollout incl. LSTM forward + hidden reset bookkeeping |
| rl_env_steps_sandbox | 14075.81 steps/sec | DSec: RL agent loop over real sandbox sessions | - | - | 8 sandbox envs, real Chronus exec per step (zero-latency channel transport) |
| packdiff_size_ratio | 0.05 pack bytes / full image bytes | DSec paper: pack_diff replicates working state w/o full copy | - | - | full 2048 KiB, pack 100 KiB (5% of files mutated) |
| packdiff_apply | 8620.18 applies/sec | - | - | - | fresh replica overlay + block import + file table import |
| packdiff_snapshot | 47817.58 packs/sec | - | - | - | dirty-set capture + serialization (post-first-snapshot management path) |
| placement_decisions | 124676.80 decisions/sec | ~5k creations/s x candidate scan must not bottleneck | - | - | 1024 nodes, k=8 sampled |
| placement_kchoice_quality | 67.65 % of exhaustive-improvement achieved | power-of-two-choices theory: k=8 approximates best-fit well | - | - | avg score k=1 1.7138, k=8 1.7964, exhaustive 1.8359 |
| protocol_encode | 446.36 MB/s | - | - | - | frame + CRC32 over 55-byte JSON payloads |
| protocol_decode | 311.47 MB/s | - | - | - | CRC-verified decode |
