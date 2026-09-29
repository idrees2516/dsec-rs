//! Metrics primitives: bucketed histogram with percentile queries and a
//! per-node stats registry. Mirrors what the paper's Edge exports to the
//! Watcher / apiserver `/metrics` endpoint.

use std::sync::atomic::{AtomicU64, Ordering};

/// Log2-ish bucketed latency histogram (1 ms granularity up to 16 s).
#[derive(Debug)]
pub struct Histogram {
    /// Buckets: 0..=1 ms, 2..3 ms, 4..7 ms, ... (bucket i covers
    /// [2^i, 2^(i+1)) ms, bucket 0 covers [0, 2) ms).
    buckets: Vec<AtomicU64>,
    count: AtomicU64,
    sum: AtomicU64,
    min: AtomicU64,
    max: AtomicU64,
}

const MAX_BUCKET: usize = 15; // up to 2^15 ms = ~32 s

impl Histogram {
    pub fn new() -> Self {
        Histogram {
            buckets: (0..=MAX_BUCKET).map(|_| AtomicU64::new(0)).collect(),
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            min: AtomicU64::new(u64::MAX),
            max: AtomicU64::new(0),
        }
    }

    fn bucket_of(ms: u64) -> usize {
        let mut b = 0usize;
        let mut v = ms;
        while v > 1 && b < MAX_BUCKET {
            v >>= 1;
            b += 1;
        }
        b
    }

    pub fn record(&self, ms: u64) {
        let b = Self::bucket_of(ms);
        self.buckets[b].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(ms, Ordering::Relaxed);
        self.min.fetch_min(ms, Ordering::Relaxed);
        self.max.fetch_max(ms, Ordering::Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    pub fn mean_ms(&self) -> f64 {
        let c = self.count();
        if c == 0 {
            0.0
        } else {
            self.sum.load(Ordering::Relaxed) as f64 / c as f64
        }
    }

    pub fn min_ms(&self) -> u64 {
        let m = self.min.load(Ordering::Relaxed);
        if m == u64::MAX {
            0
        } else {
            m
        }
    }

    pub fn max_ms(&self) -> u64 {
        self.max.load(Ordering::Relaxed)
    }

    /// Percentile in milliseconds (linear interpolation inside bucket).
    pub fn percentile(&self, p: f64) -> u64 {
        let total = self.count();
        if total == 0 {
            return 0;
        }
        let target = ((p.clamp(0.0, 1.0)) * total as f64).ceil() as u64;
        let mut acc = 0u64;
        for b in 0..=MAX_BUCKET {
            acc += self.buckets[b].load(Ordering::Relaxed);
            if acc >= target {
                // Linear interpolation between bucket bounds.
                let lo = if b == 0 { 0 } else { 1u64 << b };
                let hi = 1u64 << (b + 1);
                let in_bucket = self.buckets[b].load(Ordering::Relaxed);
                let before = acc - in_bucket;
                let frac = if in_bucket > 0 {
                    (target.saturating_sub(before)) as f64 / in_bucket as f64
                } else {
                    1.0
                };
                let val = lo as f64 + frac * (hi - lo) as f64;
                return val as u64;
            }
        }
        self.max_ms()
    }

    pub fn snapshot(&self) -> HistSnapshot {
        HistSnapshot {
            count: self.count(),
            mean_ms: self.mean_ms(),
            min_ms: self.min_ms(),
            p50_ms: self.percentile(0.50),
            p90_ms: self.percentile(0.90),
            p99_ms: self.percentile(0.99),
            max_ms: self.max_ms(),
        }
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HistSnapshot {
    pub count: u64,
    pub mean_ms: f64,
    pub min_ms: u64,
    pub p50_ms: u64,
    pub p90_ms: u64,
    pub p99_ms: u64,
    pub max_ms: u64,
}

/// EdgeNode runtime counters.
#[derive(Debug, Default)]
pub struct EdgeStats {
    pub sandboxes_created: AtomicU64,
    pub sandboxes_destroyed: AtomicU64,
    pub sandboxes_failed: AtomicU64,
    pub sandboxes_current: AtomicU64,
    pub sandboxes_paused: AtomicU64,
    pub pauses: AtomicU64,
    pub resumes: AtomicU64,
    pub execs: AtomicU64,
    pub exec_errors: AtomicU64,
    pub fs_ops: AtomicU64,
    pub http_ops: AtomicU64,
    pub stream_ops: AtomicU64,
    pub bytes_reclaimed: AtomicU64,
    pub admission_rejections: AtomicU64,
    pub creation_ms: Histogram,
    pub pause_ms: Histogram,
    pub resume_ms: Histogram,
    pub exec_ms: Histogram,
}

impl EdgeStats {
    pub fn creation(&self) -> HistSnapshot {
        self.creation_ms.snapshot()
    }
    pub fn pause(&self) -> HistSnapshot {
        self.pause_ms.snapshot()
    }
    pub fn resume(&self) -> HistSnapshot {
        self.resume_ms.snapshot()
    }
    pub fn exec(&self) -> HistSnapshot {
        self.exec_ms.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_percentiles() {
        let h = Histogram::new();
        for i in 1..=100u64 {
            h.record(i);
        }
        let s = h.snapshot();
        assert_eq!(s.count, 100);
        assert_eq!(s.min_ms, 1);
        assert_eq!(s.max_ms, 100);
        // p50 near 50, p99 near 99 (bucket interpolation keeps it close).
        assert!((s.p50_ms as i64 - 50).abs() < 12, "p50={}", s.p50_ms);
        assert!(s.p99_ms >= 90, "p99={}", s.p99_ms);
        assert!(s.mean_ms > 40.0 && s.mean_ms < 60.0);
    }

    #[test]
    fn empty_histogram() {
        let h = Histogram::new();
        let s = h.snapshot();
        assert_eq!(s.count, 0);
        assert_eq!(s.p50_ms, 0);
        assert_eq!(s.max_ms, 0);
    }

    #[test]
    fn bucket_boundaries() {
        assert_eq!(Histogram::bucket_of(0), 0);
        assert_eq!(Histogram::bucket_of(1), 0);
        assert_eq!(Histogram::bucket_of(2), 1);
        assert_eq!(Histogram::bucket_of(3), 1);
        assert_eq!(Histogram::bucket_of(4), 2);
        assert_eq!(Histogram::bucket_of(4000), 11);
        assert_eq!(Histogram::bucket_of(4096), 12);
    }
}
