//! Per-token token-bucket rate limiter (apiserver protection).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct RateLimitConfig {
    pub capacity: f64,
    pub refill_per_sec: f64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        RateLimitConfig {
            capacity: 200.0,
            refill_per_sec: 100.0,
        }
    }
}

#[derive(Debug)]
struct Bucket {
    tokens: f64,
    last: Instant,
}

#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    buckets: Mutex<HashMap<String, Bucket>>,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        RateLimiter {
            config,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// Checks one unit against the bucket; on denial returns the
    /// suggested retry-after in milliseconds.
    pub fn check(&self, key: &str) -> Result<(), u64> {
        let mut buckets = self.buckets.lock().expect("ratelimit poisoned");
        let now = Instant::now();
        let bucket = buckets.entry(key.to_string()).or_insert_with(|| Bucket {
            tokens: self.config.capacity,
            last: now,
        });
        let elapsed = now.duration_since(bucket.last).as_secs_f64();
        bucket.last = now;
        bucket.tokens =
            (bucket.tokens + elapsed * self.config.refill_per_sec).min(self.config.capacity);
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            Ok(())
        } else {
            let need = 1.0 - bucket.tokens;
            let retry_ms = (need / self.config.refill_per_sec * 1000.0).ceil() as u64;
            Err(retry_ms)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn allows_burst_then_limits() {
        let rl = RateLimiter::new(RateLimitConfig {
            capacity: 5.0,
            refill_per_sec: 1.0,
        });
        for _ in 0..5 {
            assert!(rl.check("t").is_ok());
        }
        let retry = rl.check("t").unwrap_err();
        assert!(retry > 0 && retry <= 1000);
    }

    #[test]
    fn refills_over_time() {
        let rl = RateLimiter::new(RateLimitConfig {
            capacity: 2.0,
            refill_per_sec: 1000.0,
        });
        assert!(rl.check("k").is_ok());
        assert!(rl.check("k").is_ok());
        assert!(rl.check("k").is_err());
        std::thread::sleep(Duration::from_millis(5));
        assert!(rl.check("k").is_ok());
    }

    #[test]
    fn keys_are_independent() {
        let rl = RateLimiter::new(RateLimitConfig {
            capacity: 1.0,
            refill_per_sec: 0.0,
        });
        assert!(rl.check("a").is_ok());
        assert!(rl.check("b").is_ok());
        assert!(rl.check("a").is_err());
    }
}
