//! Sandbox pool: keep a set of warm sandboxes ready for bursty agent
//! loops (the paper's pool semantics — acquire, use, reset, release).
//!
//! Acquire pops an idle sandbox (validating it is still ready), creating
//! a new one when the pool is empty and under `max_size`. Release resets
//! transient state (/tmp) and parks the sandbox for reuse.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dsec_control::model::SandboxSpec;

use crate::error::{Error, Result};
use crate::sandbox::Sandbox;
use crate::DsecClient;

#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Minimum idle sandboxes kept warm.
    pub min_idle: usize,
    /// Hard cap on outstanding sandboxes created by this pool.
    pub max_size: usize,
    /// Idle sandboxes older than this are reaped.
    pub idle_ttl: Duration,
    /// Spec used for warm sandboxes.
    pub spec: SandboxSpec,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            min_idle: 2,
            max_size: 16,
            idle_ttl: Duration::from_secs(300),
            spec: SandboxSpec::default(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct PoolStats {
    pub idle: usize,
    pub outstanding: u64,
    pub total_created: u64,
    pub acquires: u64,
    pub reuses: u64,
    pub reaped: u64,
}

struct IdleEntry {
    sandbox: Sandbox,
    since: Instant,
}

pub struct SandboxPool {
    client: Arc<DsecClient>,
    config: PoolConfig,
    idle: Mutex<Vec<IdleEntry>>,
    outstanding: AtomicU64,
    total_created: AtomicU64,
    acquires: AtomicU64,
    reuses: AtomicU64,
    reaped: AtomicU64,
}

impl SandboxPool {
    pub fn new(client: Arc<DsecClient>, config: PoolConfig) -> Arc<Self> {
        Arc::new(SandboxPool {
            client,
            config,
            idle: Mutex::new(Vec::new()),
            outstanding: AtomicU64::new(0),
            total_created: AtomicU64::new(0),
            acquires: AtomicU64::new(0),
            reuses: AtomicU64::new(0),
            reaped: AtomicU64::new(0),
        })
    }

    /// Creates sandboxes up to `min_idle`.
    pub async fn prewarm(&self) -> Result<usize> {
        let need = {
            let idle = self.idle.lock().expect("pool poisoned");
            self.config.min_idle.saturating_sub(idle.len())
        };
        for _ in 0..need {
            let sandbox = self.client.create_sandbox(self.config.spec.clone()).await?;
            self.total_created.fetch_add(1, Ordering::Relaxed);
            self.outstanding.fetch_add(1, Ordering::Relaxed);
            self.idle.lock().expect("pool poisoned").push(IdleEntry {
                sandbox,
                since: Instant::now(),
            });
        }
        Ok(need)
    }

    /// Takes a sandbox from the pool (or creates one).
    pub async fn acquire(&self) -> Result<Sandbox> {
        self.acquires.fetch_add(1, Ordering::Relaxed);
        // Reap expired idle entries first.
        self.reap_expired().await;
        loop {
            let popped = self
                .idle
                .lock()
                .expect("pool poisoned")
                .pop()
                .map(|e| e.sandbox);
            if let Some(sandbox) = popped {
                // Validate still usable.
                match sandbox.state().await {
                    Ok(state) if state == "ready" => {
                        self.reuses.fetch_add(1, Ordering::Relaxed);
                        return Ok(sandbox);
                    }
                    _ => {
                        // Stale sandbox: destroy and keep looking.
                        let _ = sandbox.destroy().await;
                        self.outstanding.fetch_sub(1, Ordering::Relaxed);
                        continue;
                    }
                }
            }
            // Empty pool: create unless capped.
            let outstanding = self.outstanding.load(Ordering::Relaxed);
            if outstanding >= self.config.max_size as u64 {
                return Err(Error::PoolExhausted(format!(
                    "{}/{}",
                    outstanding, self.config.max_size
                )));
            }
            let sandbox = self.client.create_sandbox(self.config.spec.clone()).await?;
            self.total_created.fetch_add(1, Ordering::Relaxed);
            self.outstanding.fetch_add(1, Ordering::Relaxed);
            return Ok(sandbox);
        }
    }

    /// Returns a sandbox to the pool after resetting it.
    pub async fn release(&self, sandbox: Sandbox) -> Result<()> {
        sandbox.reset().await?;
        self.idle.lock().expect("pool poisoned").push(IdleEntry {
            sandbox,
            since: Instant::now(),
        });
        Ok(())
    }

    /// Destroys a sandbox outright (leaking it out of the pool).
    pub async fn discard(&self, sandbox: Sandbox) -> Result<()> {
        sandbox.destroy().await?;
        self.outstanding.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    async fn reap_expired(&self) {
        let now = Instant::now();
        let expired: Vec<Sandbox> = {
            let mut idle = self.idle.lock().expect("pool poisoned");
            let mut keep: Vec<IdleEntry> = Vec::with_capacity(idle.len());
            let mut expired = Vec::new();
            for e in idle.drain(..) {
                if now.duration_since(e.since) > self.config.idle_ttl {
                    expired.push(e.sandbox);
                } else {
                    keep.push(e);
                }
            }
            *idle = keep;
            expired
        };
        for s in expired {
            let _ = s.destroy().await;
            self.outstanding.fetch_sub(1, Ordering::Relaxed);
            self.reaped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Destroys every idle sandbox.
    pub async fn shutdown(&self) -> Result<usize> {
        let drained: Vec<Sandbox> = self
            .idle
            .lock()
            .expect("pool poisoned")
            .drain(..)
            .map(|e| e.sandbox)
            .collect();
        let count = drained.len();
        for s in drained {
            let _ = s.destroy().await;
            self.outstanding.fetch_sub(1, Ordering::Relaxed);
        }
        Ok(count)
    }

    pub fn stats(&self) -> PoolStats {
        PoolStats {
            idle: self.idle.lock().expect("pool poisoned").len(),
            outstanding: self.outstanding.load(Ordering::Relaxed),
            total_created: self.total_created.load(Ordering::Relaxed),
            acquires: self.acquires.load(Ordering::Relaxed),
            reuses: self.reuses.load(Ordering::Relaxed),
            reaped: self.reaped.load(Ordering::Relaxed),
        }
    }
}
