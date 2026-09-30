//! Sampling-profiler harness for the sandbox-env stepping path.
//!
//! Runs `dsec-bench`'s sandbox-env workload under a SIGPROF-based
//! sampler (works unprivileged — no `perf` needed) and reports a
//! flamegraph SVG, a leaf-symbol profile, and the wall-time share spent
//! inside the Aether data plane (`call`/`call_batch`) — the
//! I/O-boundness measurement.
//!
//! Usage: `dsec-profiling [serial|batch] [envs] [seconds] [out.svg]`
//! Build with symbols:  `cargo run -p dsec-profiling --release`

use std::sync::Arc;
use std::time::Instant;

use dsec_rl::sandbox_env::{BatchedSandboxEnvPool, SandboxEnvBuilder};
use dsec_rl::spaces::Action;

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "batch".into());
    let num_envs: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let seconds: f64 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4.0);
    let out = std::env::args()
        .nth(4)
        .unwrap_or_else(|| format!("sandbox-env-{mode}.svg"));

    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(500)
        .build()
        .expect("start profiler");

    let builder = SandboxEnvBuilder::new(13);
    // Deterministic action schedule identical to the bench (fixed (i % 6)).
    let actions: Vec<Action> = (0..num_envs)
        .map(|i| Action::Discrete((i % 6) as i64))
        .collect();

    let (ticks, total_reward, dones, elapsed) = if mode == "exec_cost" {
        // Server-side micro-bench: `handle_aether_request` with NO
        // transport — isolates JSON + interpreter + FS cost.
        use dsec_protocol::frame::Channel;
        use dsec_protocol::message::Request;
        use dsec_runtime::backend::SandboxSpec;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let image = Arc::new(dsec_storage::erofs::ErofsImageBuilder::agent_base().build());
        let mut registry = dsec_storage::erofs::ImageRegistry::default();
        registry.register(
            image,
            Arc::new(dsec_storage::cache::LruBlockCache::new(1024)),
            dsec_storage::latency::LatencyModel::fixed(std::time::Duration::ZERO),
        );
        let node = dsec_runtime::EdgeNode::new(
            "bench-node".into(),
            Arc::new(registry),
            dsec_storage::latency::NodeLatencyProfile::zero(),
            1_000_000,
            1_000_000,
            100_000,
            1,
        );
        let sid = rt
            .block_on(async { node.create(SandboxSpec::default()).await })
            .unwrap()
            .sid;
        let n = (seconds * 200_000.0) as u64;
        let t0 = Instant::now();
        let mut checksum = 0usize;
        let mut i = 0u64;
        while i < n {
            let frames = rt.block_on(node.handle_aether_request(
                sid,
                Channel::Exec,
                Request::Exec {
                    cmd: if i % 2 == 0 {
                        "cat /etc/hostname".into()
                    } else {
                        "echo staging && echo ready".into()
                    },
                    timeout_ms: None,
                },
                i as u32,
            ));
            checksum += frames.len();
            i += 1;
        }
        let e = t0.elapsed().as_secs_f64();
        println!("exec/s (no transport): {}", n as f64 / e);
        println!("per-exec:              {:.2} us", e * 1e6 / n as f64);
        println!("checksum:              {checksum}");
        return;
    } else if mode == "serial" {
        use dsec_rl::envpool::EnvPool;
        let envs = builder.build_envs(num_envs, 13).expect("build envs");
        let mut pool = EnvPool::new(envs).with_workers(4);
        pool.reset_all();
        let t0 = Instant::now();
        let mut ticks = 0u64;
        let mut total_reward = 0.0f32;
        let mut dones = 0u64;
        while t0.elapsed().as_secs_f64() < seconds {
            let results = pool.step_parallel(&actions).expect("step");
            total_reward += results.iter().map(|r| r.reward).sum::<f32>();
            dones += results.iter().filter(|r| r.done).count() as u64;
            ticks += 1;
        }
        (ticks, total_reward, dones, t0.elapsed().as_secs_f64())
    } else {
        let mut pool: BatchedSandboxEnvPool = builder.build_batch_pool(num_envs, 13).expect("pool");
        pool.reset_all();
        let t0 = Instant::now();
        let mut ticks = 0u64;
        let mut total_reward = 0.0f32;
        let mut dones = 0u64;
        while t0.elapsed().as_secs_f64() < seconds {
            let results = pool.step_all(&actions).expect("step");
            total_reward += results.iter().map(|r| r.reward).sum::<f32>();
            dones += results.iter().filter(|r| r.done).count() as u64;
            ticks += 1;
        }
        (ticks, total_reward, dones, t0.elapsed().as_secs_f64())
    };
    let stats = builder.client().call_stats();

    let steps_per_sec = (ticks as f64 * num_envs as f64) / elapsed;
    let call_ms = stats.call_ns as f64 / 1e6;
    let call_share = stats.call_ns as f64 / (elapsed * num_envs as f64 * 1e9) * 100.0;
    println!("mode:               {mode}");
    println!("envs:               {num_envs}");
    println!("steps/s:            {steps_per_sec:.0}");
    println!("ticks:              {ticks}");
    println!("wall time:          {elapsed:.2} s");
    println!(
        "data-plane calls:   {} (timeouts {})",
        stats.calls, stats.timeouts
    );
    println!(
        "avg call latency:   {:.1} us",
        if stats.calls > 0 {
            call_ms * 1000.0 / stats.calls as f64
        } else {
            0.0
        }
    );
    println!("call-time share:    {call_share:.1}% of worker wall time");
    println!("(checksum) reward:  {total_reward:.2}  dones: {dones}");

    match guard.report().build() {
        Ok(report) => {
            let file = std::fs::File::create(&out).expect("create flamegraph");
            report.flamegraph(file).expect("write flamegraph");
            println!("flamegraph:         {out}");
            // Leaf-symbol aggregation over the sample map.
            let sym_name = |s: &pprof::Symbol| -> String {
                s.name
                    .as_ref()
                    .and_then(|b| String::from_utf8(b.clone()).ok())
                    .unwrap_or_else(|| "?".into())
            };
            let mut leaves: std::collections::HashMap<String, isize> = Default::default();
            let mut total: isize = 0;
            let mut stacks: Vec<(isize, String)> = Vec::new();
            for (frames, count) in report.data.iter() {
                total += *count;
                let leaf = frames
                    .frames
                    .last()
                    .and_then(|syms| syms.last())
                    .map(&sym_name)
                    .unwrap_or_else(|| "?".into());
                *leaves.entry(leaf).or_default() += *count;
                let stack = frames
                    .frames
                    .iter()
                    .rev()
                    .map(|syms| syms.last().map(&sym_name).unwrap_or_else(|| "?".into()))
                    .collect::<Vec<_>>()
                    .join(" <- ");
                stacks.push((*count, stack));
            }
            let mut leaves: Vec<(isize, String)> =
                leaves.into_iter().map(|(k, v)| (v, k)).collect();
            leaves.sort_by_key(|&(c, _)| std::cmp::Reverse(c));
            stacks.sort_by_key(|&(c, _)| std::cmp::Reverse(c));
            let mut text = String::new();
            for (c, leaf) in &leaves {
                text.push_str(&format!(
                    "{:>7} {:>6.2}%  {}\n",
                    c,
                    *c as f64 / total as f64 * 100.0,
                    leaf
                ));
            }
            text.push_str("\ntop complete stacks:\n");
            for (c, s) in stacks.iter().take(12) {
                text.push_str(&format!("{c:>7}  {s}\n"));
            }
            let _ = std::fs::write(format!("{out}.txt"), &text);
            println!("samples:            {total}");
            println!("\ntop leaf symbols (CPU-side):");
            for (c, leaf) in leaves.iter().take(20) {
                let pct = *c as f64 / total as f64 * 100.0;
                println!("  {c:>6} {pct:>5.1}%  {leaf}");
            }
            println!("\ntop stacks:");
            for (c, s) in stacks.iter().take(8) {
                println!("  {c:>6}  {s}");
            }
        }
        Err(e) => eprintln!("no samples collected: {e:?}"),
    }
}
