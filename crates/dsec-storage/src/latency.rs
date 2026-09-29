//! Injectable latency model for deterministic simulation.
//!
//! The real system's latencies come from 3FS reads, disk and network; here
//! they are modeled as a fixed component plus seeded jitter, so tests can
//! run with zero latency (pure logic) and benchmarks can replay the
//! paper's timing profile (e.g. ~1 s microVM creation, ~4 s pause).

use std::sync::Mutex;
use std::time::Duration;

use dsec_protocol::rng::Rng;

/// Named latency profiles used across the workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// No injected latency: pure logic, fastest possible.
    Zero,
    /// Latencies scaled from the DSec paper's reported numbers.
    Paper,
    /// Stress profile: 4x the paper profile, to shake out queuing.
    Stress,
}

#[derive(Debug)]
pub struct LatencyModel {
    fixed: Duration,
    jitter: Duration,
    rng: Option<Mutex<Rng>>,
}

impl Clone for LatencyModel {
    fn clone(&self) -> Self {
        // Re-seed from a draw so clones stay deterministic yet independent.
        let seed = self
            .rng
            .as_ref()
            .and_then(|m| m.lock().ok().map(|mut r| r.next_u64()))
            .unwrap_or(0);
        LatencyModel {
            fixed: self.fixed,
            jitter: self.jitter,
            rng: if self.rng.is_some() {
                Some(Mutex::new(Rng::new(seed)))
            } else {
                None
            },
        }
    }
}

impl LatencyModel {
    /// Constant latency without jitter (zero-latency tests use `ZERO`).
    pub fn fixed(d: Duration) -> Self {
        LatencyModel {
            fixed: d,
            jitter: Duration::ZERO,
            rng: None,
        }
    }

    /// Fixed + uniform jitter in `[fixed, fixed + jitter]`, seeded.
    pub fn with_jitter(fixed: Duration, jitter: Duration, seed: u64) -> Self {
        LatencyModel {
            fixed,
            jitter,
            rng: Some(Mutex::new(Rng::new(seed))),
        }
    }

    pub fn profile(p: Profile, seed: u64) -> Self {
        let (fixed, jitter) = match p {
            Profile::Zero => (Duration::ZERO, Duration::ZERO),
            Profile::Paper => (Duration::from_millis(1), Duration::from_millis(2)),
            Profile::Stress => (Duration::from_millis(4), Duration::from_millis(8)),
        };
        if jitter.is_zero() {
            LatencyModel::fixed(fixed)
        } else {
            LatencyModel::with_jitter(fixed, jitter, seed)
        }
    }

    /// Draws one latency sample (jitter is uniform in `[0, jitter]`).
    pub fn sample(&self) -> Duration {
        if let Some(rng) = &self.rng {
            let j = self.jitter.as_nanos() as f64;
            let t = rng.lock().map(|mut r| r.next_f64()).unwrap_or(0.0) * j;
            self.fixed + Duration::from_nanos(t as u64)
        } else {
            self.fixed
        }
    }

    pub fn is_zero(&self) -> bool {
        self.fixed.is_zero() && self.jitter.is_zero()
    }
}

/// Per-operation latency schedule mirroring the paper's reported numbers
/// (creation ~1 s end-to-end average, pause ~4 s average, etc.). Backends
/// and storage use the fields relevant to them.
#[derive(Debug, Clone)]
pub struct NodeLatencyProfile {
    /// Cold fetch of one block from the backing store (3FS read path).
    pub block_fetch: LatencyModel,
    /// FnCall warm-path sandbox creation.
    pub fncall_create: LatencyModel,
    /// Container backend creation.
    pub container_create: LatencyModel,
    /// MicroVM (Firecracker-class) creation.
    pub microvm_create: LatencyModel,
    /// Full VM (QEMU-class) creation.
    pub fullvm_create: LatencyModel,
    /// Pause (checkpoint + memory reclaim).
    pub pause: LatencyModel,
    /// Resume (restore + MADV_WILLNEED prefetch).
    pub resume: LatencyModel,
}

impl NodeLatencyProfile {
    /// Zero-latency: deterministic unit tests.
    pub fn zero() -> Self {
        NodeLatencyProfile {
            block_fetch: LatencyModel::fixed(Duration::ZERO),
            fncall_create: LatencyModel::fixed(Duration::ZERO),
            container_create: LatencyModel::fixed(Duration::ZERO),
            microvm_create: LatencyModel::fixed(Duration::ZERO),
            fullvm_create: LatencyModel::fixed(Duration::ZERO),
            pause: LatencyModel::fixed(Duration::ZERO),
            resume: LatencyModel::fixed(Duration::ZERO),
        }
    }

    /// Paper-realistic profile, seeded for reproducibility.
    ///
    /// Anchor numbers (paper Fig. creation latency / pause latency):
    /// - FnCall warm path: ~5 ms (pre-created container handout)
    /// - Container: ~80 ms
    /// - MicroVM: ~900 ms (Firecracker-class boot + attach)
    /// - FullVM: ~8 s
    /// - Pause: ~4 s average (snapshot + reclaim)
    /// - Resume: ~1.5 s
    /// - Block fetch: ~1.5 ms (network storage read of 4 KiB)
    pub fn paper(seed: u64) -> Self {
        let j = |ms: u64| {
            LatencyModel::with_jitter(
                Duration::from_millis(ms),
                Duration::from_millis(ms / 4 + 1),
                seed,
            )
        };
        NodeLatencyProfile {
            block_fetch: j(1),
            fncall_create: j(5),
            container_create: j(80),
            microvm_create: j(900),
            fullvm_create: j(8000),
            pause: j(4000),
            resume: j(1500),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_profile_is_zero() {
        let l = LatencyModel::profile(Profile::Zero, 1);
        assert!(l.is_zero());
        assert_eq!(l.sample(), Duration::ZERO);
    }

    #[test]
    fn jitter_stays_in_bounds() {
        let l = LatencyModel::with_jitter(Duration::from_millis(5), Duration::from_millis(2), 3);
        for _ in 0..500 {
            let s = l.sample();
            assert!(s >= Duration::from_millis(5));
            assert!(s <= Duration::from_millis(7));
        }
    }

    #[test]
    fn deterministic_samples() {
        let a = LatencyModel::with_jitter(Duration::from_millis(1), Duration::from_millis(10), 42);
        let b = LatencyModel::with_jitter(Duration::from_millis(1), Duration::from_millis(10), 42);
        for _ in 0..100 {
            assert_eq!(a.sample(), b.sample());
        }
    }
}
