//! Prometheus-style metrics registry (text exposition).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

#[derive(Debug, Default)]
pub struct MetricsRegistry {
    counters: Mutex<BTreeMap<String, Arc<AtomicU64>>>,
    gauges: RwLock<BTreeMap<String, f64>>,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fetches or creates a counter.
    pub fn counter(&self, name: &str) -> Arc<AtomicU64> {
        let mut map = self.counters.lock().expect("metrics poisoned");
        map.entry(name.to_string())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone()
    }

    pub fn incr(&self, name: &str) {
        self.counter(name).fetch_add(1, Ordering::Relaxed);
    }

    pub fn incr_by(&self, name: &str, by: u64) {
        self.counter(name).fetch_add(by, Ordering::Relaxed);
    }

    pub fn set_gauge(&self, name: &str, value: f64) {
        self.gauges
            .write()
            .expect("metrics poisoned")
            .insert(name.to_string(), value);
    }

    /// Prometheus text format exposition.
    pub fn expose(&self) -> String {
        let mut out = String::new();
        let counters = self.counters.lock().expect("metrics poisoned");
        for (name, c) in counters.iter() {
            let v = c.load(Ordering::Relaxed);
            out.push_str(&format!("{} {}\n", name, v));
        }
        drop(counters);
        let gauges = self.gauges.read().expect("metrics poisoned");
        for (name, v) in gauges.iter() {
            out.push_str(&format!("{} {}\n", name, v));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_and_exposition() {
        let m = MetricsRegistry::new();
        m.incr("dsec_a_total");
        m.incr("dsec_a_total");
        m.incr_by("dsec_b_total", 10);
        m.set_gauge("dsec_active", 3.5);
        let text = m.expose();
        assert!(text.contains("dsec_a_total 2"));
        assert!(text.contains("dsec_b_total 10"));
        assert!(text.contains("dsec_active 3.5"));
    }

    #[test]
    fn shared_counters_from_threads() {
        let m = std::sync::Arc::new(MetricsRegistry::new());
        let c = m.counter("shared_total");
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let c = c.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        c.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(m.expose().contains("shared_total 400"));
    }
}
