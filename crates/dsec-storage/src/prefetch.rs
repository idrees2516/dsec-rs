//! Asynchronous prefetch pipeline and zero-copy shared regions.
//!
//! The paper's storage plane overlaps image loading with sandbox boot
//! ("Data asynchronous prefetch"): Edge issues prefetch hints for blocks
//! it expects the sandbox to touch (metadata, /etc, entrypoint binaries)
//! while creation proceeds. [`AsyncDataPipeline`] bounds in-flight fetches
//! with a semaphore, exactly like a real staged pipeline.
//!
//! [`SharedRegion`] is the userspace stand-in for the torch shared-memory
//! handoff: envpool writes observation slots into one pinned-style
//! allocation and the training side borrows `&[f32]` views of the same
//! buffer — no intermediate copy.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use dsec_protocol::rng::Rng;

use crate::erofs::OnDemandLoader;
use crate::BlockRange;

/// Bounded-concurrency prefetch engine.
#[derive(Debug)]
pub struct AsyncDataPipeline {
    loader: Arc<OnDemandLoader>,
    inflight_limit: usize,
    bytes_prefetched: AtomicU64,
    blocks_prefetched: AtomicU64,
    batches: AtomicU64,
}

impl AsyncDataPipeline {
    pub fn new(loader: Arc<OnDemandLoader>, inflight_limit: usize, seed: u64) -> Self {
        let _ = seed; // reserved for future seeded probe patterns
        AsyncDataPipeline {
            loader,
            inflight_limit,
            bytes_prefetched: AtomicU64::new(0),
            blocks_prefetched: AtomicU64::new(0),
            batches: AtomicU64::new(0),
        }
    }

    /// Prefetches a batch of ranges with bounded parallelism; returns the
    /// number of blocks actually fetched (cache misses).
    pub async fn prefetch_ranges(&self, ranges: &[BlockRange]) -> u64 {
        self.batches.fetch_add(1, Ordering::Relaxed);
        let blocks: Vec<u64> = ranges
            .iter()
            .flat_map(|r| r.start..r.start + r.count)
            .filter(|&b| b < self.loader.image().block_count())
            .collect();
        let sem = Arc::new(tokio::sync::Semaphore::new(self.inflight_limit));
        let mut handles = Vec::with_capacity(blocks.len());
        for b in blocks {
            // Acquire before spawn; the permit lives inside the task so
            // at most `inflight_limit` fetches run concurrently.
            let permit = match sem.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => break,
            };
            let loader = self.loader.clone();
            handles.push(tokio::spawn(async move {
                let _permit = permit;
                let before = loader.stats().fetches;
                let _ = loader.read_block(b).await;
                loader.stats().fetches > before
            }));
        }
        let mut fetched = 0u64;
        for h in handles {
            if h.await.unwrap_or(false) {
                fetched += 1;
            }
        }
        self.blocks_prefetched.fetch_add(fetched, Ordering::Relaxed);
        self.bytes_prefetched
            .fetch_add(fetched * crate::BLOCK_SIZE as u64, Ordering::Relaxed);
        fetched
    }

    /// Prefetches whole files by path (used at sandbox creation).
    pub async fn prefetch_files(&self, paths: &[&str]) -> u64 {
        let image = self.loader.image();
        let mut ranges = Vec::new();
        for p in paths {
            if let Some(e) = image.meta().entries.iter().find(|e| e.path == *p) {
                ranges.push(BlockRange {
                    start: e.first_block,
                    count: e.block_count,
                });
            }
        }
        self.prefetch_ranges(&ranges).await
    }

    /// Issues a random probe batch (used by benchmarks to measure cache
    /// behavior under uniform traffic).
    pub async fn prefetch_random(&self, n_blocks: u64, seed: u64) -> u64 {
        let mut rng = Rng::new(seed);
        let total = self.loader.image().block_count();
        if total == 0 {
            return 0;
        }
        let ranges: Vec<BlockRange> = (0..n_blocks)
            .map(|_| {
                let start = rng.below(total as usize) as u64;
                BlockRange { start, count: 1 }
            })
            .collect();
        self.prefetch_ranges(&ranges).await
    }

    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            batches: self.batches.load(Ordering::Relaxed),
            blocks_prefetched: self.blocks_prefetched.load(Ordering::Relaxed),
            bytes_prefetched: self.bytes_prefetched.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PipelineStats {
    pub batches: u64,
    pub blocks_prefetched: u64,
    pub bytes_prefetched: u64,
}

static SHARED_REGION_IDS: AtomicU64 = AtomicU64::new(1);

/// Zero-copy observation region shared between envpool writers and the
/// training reader (torch shared-memory analogue).
#[derive(Debug)]
pub struct SharedRegion {
    id: u64,
    slot_len: usize,
    slots: usize,
    buf: Arc<RwLock<Box<[f32]>>>,
}

impl SharedRegion {
    /// `slots * slot_len` f32 elements, zero-initialized.
    pub fn new(slots: usize, slot_len: usize) -> Self {
        SharedRegion {
            id: SHARED_REGION_IDS.fetch_add(1, Ordering::Relaxed),
            slot_len,
            slots,
            buf: Arc::new(RwLock::new(
                vec![0.0f32; slots * slot_len].into_boxed_slice(),
            )),
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn slot_len(&self) -> usize {
        self.slot_len
    }

    pub fn total_len(&self) -> usize {
        self.slots * self.slot_len
    }

    /// Writes one observation slot.
    pub fn write_slot(&self, slot: usize, obs: &[f32]) {
        assert!(slot < self.slots, "slot out of range");
        assert_eq!(obs.len(), self.slot_len, "observation length mismatch");
        let mut buf = self.buf.write().expect("region poisoned");
        let off = slot * self.slot_len;
        buf[off..off + obs.len()].copy_from_slice(obs);
    }

    /// Bulk-writes the whole region (length must match exactly).
    pub fn write_all(&self, data: &[f32]) {
        let mut buf = self.buf.write().expect("region poisoned");
        assert_eq!(buf.len(), data.len(), "region length mismatch");
        buf.copy_from_slice(data);
    }

    /// Zero-copy borrow of the entire region (single heap allocation).
    pub fn read_view(&self) -> RegionView<'_> {
        RegionView {
            guard: self.buf.read().expect("region poisoned"),
        }
    }

    /// Two regions alias the same buffer iff they share the allocation.
    pub fn same_allocation(&self, other: &SharedRegion) -> bool {
        Arc::ptr_eq(&self.buf, &other.buf)
    }

    /// Cheap handle for a writer task.
    pub fn handle(&self) -> SharedRegionHandle {
        SharedRegionHandle {
            slot_len: self.slot_len,
            slots: self.slots,
            buf: self.buf.clone(),
        }
    }
}

/// Borrowed zero-copy view of a [`SharedRegion`].
pub struct RegionView<'a> {
    guard: std::sync::RwLockReadGuard<'a, Box<[f32]>>,
}

impl RegionView<'_> {
    /// The whole region as one contiguous slice (zero-copy).
    pub fn as_slice(&self) -> &[f32] {
        &self.guard
    }

    /// One slot view of `slot_len` elements.
    pub fn slot_at(&self, i: usize, slot_len: usize) -> &[f32] {
        let off = i * slot_len;
        &self.guard[off..off + slot_len]
    }
}

/// Cloneable handle for writer tasks.
#[derive(Debug)]
pub struct SharedRegionHandle {
    slot_len: usize,
    slots: usize,
    buf: Arc<RwLock<Box<[f32]>>>,
}

impl SharedRegionHandle {
    pub fn write_slot(&self, slot: usize, obs: &[f32]) {
        assert!(slot < self.slots);
        assert_eq!(obs.len(), self.slot_len);
        let mut buf = self.buf.write().expect("region poisoned");
        let off = slot * self.slot_len;
        buf[off..off + obs.len()].copy_from_slice(obs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::LruBlockCache;
    use crate::erofs::{ErofsImageBuilder, OnDemandLoader};
    use crate::latency::LatencyModel;
    use std::time::Duration;

    fn loader() -> Arc<OnDemandLoader> {
        let image = Arc::new(ErofsImageBuilder::agent_base().build());
        Arc::new(OnDemandLoader::new(
            image,
            Arc::new(LruBlockCache::new(128)),
            LatencyModel::fixed(Duration::ZERO),
        ))
    }

    #[tokio::test]
    async fn prefetch_bounded_and_effective() {
        let l = loader();
        let pipe = AsyncDataPipeline::new(l.clone(), 4, 7);
        pipe.prefetch_files(&["/etc/hostname", "/etc/os-release", "/etc/passwd"])
            .await;
        let s = pipe.stats();
        assert!(s.blocks_prefetched >= 3);
        // All target blocks now cached.
        let hits_before = l.stats().hits;
        let _ = l.read_block(0).await.unwrap();
        assert_eq!(l.stats().hits, hits_before + 1);
    }

    #[tokio::test]
    async fn random_probe_respects_bounds() {
        let l = loader();
        let pipe = AsyncDataPipeline::new(l, 2, 9);
        pipe.prefetch_random(8, 11).await;
        let s = pipe.stats();
        // Random probes may repeat; only distinct misses fetch.
        assert!(s.blocks_prefetched >= 1 && s.blocks_prefetched <= 8);
    }

    #[test]
    fn shared_region_zero_copy_semantics() {
        let region = SharedRegion::new(4, 8);
        let handle = region.handle();
        handle.write_slot(2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let view = region.read_view();
        let s = view.as_slice();
        assert_eq!(s.len(), 32);
        assert_eq!(&s[16..24], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        // Slot views of the same allocation.
        assert_eq!(
            view.slot_at(2, 8),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn writers_from_threads_share_allocation() {
        let region = SharedRegion::new(8, 4);
        let region = Arc::new(region);
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let h = region.handle();
                std::thread::spawn(move || {
                    h.write_slot(i, &[i as f32; 4]);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let view = region.read_view();
        for i in 0..8 {
            assert_eq!(view.slot_at(i, 4), &[i as f32; 4]);
        }
    }
}
