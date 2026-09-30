//! Shared deterministic PRNG (splitmix64), used by every simulation
//! component so that benchmarks and tests are bit-for-bit reproducible.
//!
//! `splitmix64` is a 64-bit generator with excellent statistical
//! properties and a tiny state; we derive `u32`/`f64`/range draws from it
//! in a fixed, platform-independent way.

#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// New generator; seed 0 is remapped to avoid the fixed point.
    pub fn new(seed: u64) -> Self {
        Rng {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `[0, n)`; `n` must be non-zero.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform duration in `[min, max]`.
    pub fn duration_range(
        &mut self,
        min: std::time::Duration,
        max: std::time::Duration,
    ) -> std::time::Duration {
        let min_us = min.as_micros() as f64;
        let max_us = max.as_micros() as f64;
        let t = min_us + self.next_f64() * (max_us - min_us);
        std::time::Duration::from_micros(t.max(0.0) as u64)
    }

    /// Fisher-Yates sample of `k` distinct indices from `0..n`
    /// (partial shuffle, O(n) time, O(n) space when `k < n`).
    pub fn sample_k(&mut self, n: usize, k: usize) -> Vec<usize> {
        if k >= n {
            return (0..n).collect();
        }
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + self.below(n - i);
            idx.swap(i, j);
        }
        idx.truncate(k);
        idx
    }

    /// Floyd's algorithm: `k` distinct uniform indices from `0..n` in
    /// O(k^2) time and O(k) space (for `k << n` this avoids the O(n)
    /// index materialization of [`Self::sample_k`]). Returned sorted
    /// ascending so callers can select positions during a single scan.
    ///
    /// Loop invariant: at step `j`, `j` itself can never already be
    /// selected (earlier draws are bounded by earlier `j`s), so the
    /// "already present" branch inserting `j` is always safe and each
    /// iteration adds exactly one element.
    pub fn sample_k_floyd(&mut self, n: usize, k: usize) -> Vec<usize> {
        if k >= n {
            return (0..n).collect();
        }
        let mut selected: Vec<usize> = Vec::with_capacity(k);
        for j in (n - k)..n {
            let t = self.below(j + 1);
            if selected.contains(&t) {
                selected.push(j);
            } else {
                selected.push(t);
            }
        }
        selected.sort_unstable();
        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_across_instances() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn distinct_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let da: u64 = (0..8).map(|_| a.next_u64()).fold(0u64, |x, y| x ^ y);
        let db: u64 = (0..8).map(|_| b.next_u64()).fold(0u64, |x, y| x ^ y);
        assert_ne!(da, db);
    }

    #[test]
    fn sample_k_distinct_and_in_range() {
        let mut rng = Rng::new(7);
        let s = rng.sample_k(100, 10);
        assert_eq!(s.len(), 10);
        let mut sorted = s.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 10);
        assert!(sorted.iter().all(|&i| i < 100));
        // k >= n returns identity
        assert_eq!(rng.sample_k(3, 5), vec![0, 1, 2]);
    }

    #[test]
    fn floyd_sample_distinct_sorted_uniform() {
        let mut rng = Rng::new(31);
        for (n, k) in [(1000, 8), (64, 8), (10, 3), (5, 5), (3, 10)] {
            let s = rng.sample_k_floyd(n, k);
            let expected = k.min(n);
            assert_eq!(s.len(), expected);
            assert!(s.windows(2).all(|w| w[0] < w[1]), "sorted, k={} n={}", k, n);
            assert!(s.iter().all(|&i| i < n));
        }
        // Statistical sanity: over many draws every position is reachable.
        let mut seen = vec![0usize; 40];
        for _ in 0..4000 {
            for i in rng.sample_k_floyd(40, 4) {
                seen[i] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c > 0),
            "uniform reachability: {:?}",
            seen
        );
    }

    #[test]
    fn f64_in_unit_range() {
        let mut rng = Rng::new(9);
        for _ in 0..10_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }
}
