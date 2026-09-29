//! Benchmark result types and report writing.

use std::path::Path;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub name: String,
    pub value: f64,
    pub unit: String,
    /// Paper / SOTA reference line, when applicable.
    pub reference: Option<String>,
    pub notes: String,
    pub p50_ms: Option<f64>,
    pub p99_ms: Option<f64>,
    pub samples: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Percentiles {
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub samples: u64,
}

/// Computes percentiles from a latencies vector (milliseconds).
pub fn percentiles(latencies_ms: &[f64]) -> Percentiles {
    if latencies_ms.is_empty() {
        return Percentiles::default();
    }
    let mut v = latencies_ms.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pick = |q: f64| -> f64 {
        let idx = ((v.len() as f64 - 1.0) * q).round() as usize;
        v[idx.min(v.len() - 1)]
    };
    Percentiles {
        p50_ms: pick(0.50),
        p99_ms: pick(0.99),
        samples: v.len() as u64,
    }
}

#[derive(Debug, Serialize)]
pub struct BenchReport {
    pub timestamp: String,
    pub rustc: String,
    pub cores: usize,
    pub results: Vec<BenchResult>,
}

impl BenchReport {
    pub fn new(results: Vec<BenchResult>) -> Self {
        BenchReport {
            timestamp: chrono_free_now(),
            rustc: rustc_version(),
            cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            results,
        }
    }

    pub fn write_json(&self, path: &Path) -> std::io::Result<()> {
        let s = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, s)
    }

    pub fn write_markdown(&self, path: &Path) -> std::io::Result<()> {
        let mut md = String::new();
        md.push_str("# dsec-rs benchmark results\n\n");
        md.push_str(&format!(
            "- date: {}\n- rustc: {}\n- cores: {}\n\n",
            self.timestamp, self.rustc, self.cores
        ));
        md.push_str("| benchmark | result | reference | p50 (ms) | p99 (ms) | notes |\n");
        md.push_str("|---|---|---|---|---|---|\n");
        for r in &self.results {
            md.push_str(&format!(
                "| {} | {:.2} {} | {} | {} | {} | {} |\n",
                r.name,
                r.value,
                r.unit,
                r.reference.as_deref().unwrap_or("-"),
                r.p50_ms.map(|v| format!("{:.3}", v)).unwrap_or("-".into()),
                r.p99_ms.map(|v| format!("{:.3}", v)).unwrap_or("-".into()),
                r.notes,
            ));
        }
        std::fs::write(path, md)
    }

    /// Console table.
    pub fn print(&self) {
        println!("{:=<80}", "");
        println!(
            "dsec-rs benchmarks (rustc {}, {} cores)",
            self.rustc, self.cores
        );
        println!("{:=<80}", "");
        for r in &self.results {
            let p = match (r.p50_ms, r.p99_ms) {
                (Some(a), Some(b)) => format!("p50={:.3}ms p99={:.3}ms", a, b),
                _ => String::new(),
            };
            println!("{:<38} {:>12.2} {:<10} {}", r.name, r.value, r.unit, p);
            if let Some(reference) = &r.reference {
                println!("{:<38} {:>12} {}", "", "ref:", reference);
            }
            if !r.notes.is_empty() {
                println!("{:<38} {}", "", r.notes);
            }
        }
        println!("{:=<80}", "");
    }
}

fn chrono_free_now() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map(|s| {
            // Fixed-format epoch seconds; a full date needs a tz lib.
            format!("unix:{}", s)
        })
        .unwrap_or_default()
}

fn rustc_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}
