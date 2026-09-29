//! dsec-bench: benchmark suite for the dsec-rs workspace.
//!
//! Usage:
//!   dsec-bench [--quick] [--out DIR]
//!   dsec-bench creation [--count N] [--concurrency C] [--paper-latency]
//!   dsec-bench rl [--envs N] [--steps N]
//!   dsec-bench burst [--count N]

mod benches;
mod report;

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let quick = args.iter().any(|a| a == "--quick");
    let out_dir = flag_value(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/z/my-project/download/dsec-rs-bench"));

    let mode = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "all".into());

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");

    let results = rt.block_on(async {
        match mode.as_str() {
            "creation" => {
                let count: usize = flag_value(&args, "--count")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(5000);
                let conc: usize = flag_value(&args, "--concurrency")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(256);
                let paper = args.iter().any(|a| a == "--paper-latency");
                let (cps, lat) = benches::creation_rate(count, conc, paper).await;
                vec![report::BenchResult {
                    name: format!(
                        "sandbox_create_{}",
                        if paper {
                            "paper_latency"
                        } else {
                            "software_path"
                        }
                    ),
                    value: cps,
                    unit: "creates/sec".into(),
                    reference: Some("DSec paper: ~5,000/s".into()),
                    notes: format!("count={} concurrency={}", count, conc),
                    p50_ms: Some(lat.p50_ms),
                    p99_ms: Some(lat.p99_ms),
                    samples: Some(lat.samples),
                }]
            }
            "burst" => {
                let count: usize = flag_value(&args, "--count")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(20_000);
                let b = benches::burst_lifecycle(count, 128).await;
                vec![report::BenchResult {
                    name: "burst_lifecycle".into(),
                    value: b.total as f64 / b.seconds.max(1e-9),
                    unit: "sandboxes/sec".into(),
                    reference: Some("DSec paper: 100k+ concurrent sandboxes".into()),
                    notes: format!(
                        "total={} peak={} execs={} errors={}",
                        b.total, b.peak_concurrent, b.execs, b.errors
                    ),
                    p50_ms: None,
                    p99_ms: None,
                    samples: None,
                }]
            }
            "rl" => {
                let envs: usize = flag_value(&args, "--envs")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(64);
                let steps: u64 = flag_value(&args, "--steps")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(200_000);
                let fast = benches::rl_fast_steps(envs, 4, steps);
                vec![report::BenchResult {
                    name: "rl_env_steps_fast".into(),
                    value: fast,
                    unit: "steps/sec".into(),
                    reference: Some("PufferLib: ~1M+ steps/s".into()),
                    notes: format!("{} envs x 4 threads", envs),
                    p50_ms: None,
                    p99_ms: None,
                    samples: None,
                }]
            }
            _ => {
                let mut merged = Vec::new();
                // Sync benchmarks run on a dedicated thread (they build
                // private runtimes for sandbox envs).
                let sync_handle = std::thread::spawn(move || benches::run_sync_set(quick));
                let async_results = benches::run_async_set(quick).await;
                let sync_results = sync_handle.join().unwrap_or_default();
                merged.extend(async_results);
                merged.extend(sync_results);
                merged
            }
        }
    });

    let report = report::BenchReport::new(results);
    report.print();

    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!("warning: cannot create {}: {}", out_dir.display(), e);
        return;
    }
    let json_path = out_dir.join("bench_results.json");
    let md_path = out_dir.join("bench_results.md");
    match report.write_json(&json_path) {
        Ok(()) => println!("json  -> {}", json_path.display()),
        Err(e) => eprintln!("json write failed: {}", e),
    }
    match report.write_markdown(&md_path) {
        Ok(()) => println!("table -> {}", md_path.display()),
        Err(e) => eprintln!("markdown write failed: {}", e),
    }
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
