//! Node-level LRU block cache.
//!
//! Every block cached here is an `Arc<Block>`; sandboxes on the same node
//! share the same physical content through refcounts. This reproduces the
//! paper's page-cache sharing between sibling sandboxes (virtio-pmem DAX
//! path) in userspace: dedup is structural, not heuristic.

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::{Block, BlockId};

#[derive(Debug, Default)]
struct Inner {
    map: HashMap<BlockId, (std::sync::Arc<Block>, u64)>,
    order: BTreeSet<(u64, BlockId)>,
    tick: u64,
    evictions: u64,
}

#[derive(Debug)]
pub struct LruBlockCache {
    capacity_blocks: usize,
    inner: Mutex<Inner>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl LruBlockCache {
    pub fn new(capacity_blocks: usize) -> Self {
        LruBlockCache {
            capacity_blocks,
            inner: Mutex::new(Inner::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Cache lookup; refreshes recency on hit.
    pub fn get(&self, id: BlockId) -> Option<std::sync::Arc<Block>> {
        let mut inner = self.inner.lock().ok()?;
        let Some((block, tick)) = inner.map.get(&id).cloned() else {
            drop(inner);
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        self.hits.fetch_add(1, Ordering::Relaxed);
        inner.order.remove(&(tick, id));
        inner.tick += 1;
        let new_tick = inner.tick;
        inner.order.insert((new_tick, id));
        // Refresh the existing entry in place instead of re-inserting
        // (drops one Arc clone and one hash lookup from the hit path).
        if let Some(e) = inner.map.get_mut(&id) {
            e.1 = new_tick;
        }
        Some(block)
    }

    /// Inserts a block, evicting the least-recently-used entry if full.
    /// Returns the number of bytes resident after the insert.
    pub fn put(&self, id: BlockId, block: std::sync::Arc<Block>) -> usize {
        let mut inner = self.inner.lock().expect("cache poisoned");
        if let Some((_, old_tick)) = inner.map.get(&id).cloned() {
            // Already present: refresh recency, keep one copy.
            inner.order.remove(&(old_tick, id));
            inner.tick += 1;
            let t = inner.tick;
            inner.order.insert((t, id));
            inner.map.insert(id, (block, t));
        } else {
            if inner.map.len() >= self.capacity_blocks {
                if let Some(&(tick, victim)) = inner.order.iter().next() {
                    inner.order.remove(&(tick, victim));
                    inner.map.remove(&victim);
                    inner.evictions += 1;
                }
            }
            inner.tick += 1;
            let t = inner.tick;
            inner.order.insert((t, id));
            inner.map.insert(id, (block, t));
        }
        inner.map.len() * crate::BLOCK_SIZE
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|i| i.map.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn stats(&self) -> CacheStats {
        let (len, evictions) = self
            .inner
            .lock()
            .map(|i| (i.map.len(), i.evictions))
            .unwrap_or((0, 0));
        CacheStats {
            resident_blocks: len,
            resident_bytes: len * crate::BLOCK_SIZE,
            evictions,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CacheStats {
    pub resident_blocks: usize,
    pub resident_bytes: usize,
    pub evictions: u64,
    pub hits: u64,
    pub misses: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(fill: u8) -> std::sync::Arc<Block> {
        let mut b = [0u8; crate::BLOCK_SIZE];
        b.fill(fill);
        std::sync::Arc::new(b)
    }

    #[test]
    fn hit_miss_accounting() {
        let c = LruBlockCache::new(4);
        assert!(c.get(1).is_none());
        c.put(1, block(1));
        assert!(c.get(1).is_some());
        let s = c.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert_eq!(s.resident_blocks, 1);
        assert!((s.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn evicts_lru() {
        let c = LruBlockCache::new(2);
        c.put(1, block(1));
        c.put(2, block(2));
        c.get(1); // refresh 1
        c.put(3, block(3)); // evicts 2
        assert!(c.get(2).is_none());
        assert!(c.get(1).is_some());
        assert!(c.get(3).is_some());
        assert_eq!(c.stats().evictions, 1);
    }

    #[test]
    fn shared_arc_dedups_memory() {
        let c = LruBlockCache::new(64);
        let b = block(7);
        c.put(9, b.clone());
        let got = c.get(9).unwrap();
        assert!(std::sync::Arc::ptr_eq(&got, &b));
        // Second put of same Arc must not duplicate.
        c.put(9, b.clone());
        assert_eq!(c.stats().resident_blocks, 1);
    }

    #[test]
    fn concurrent_access() {
        let c = std::sync::Arc::new(LruBlockCache::new(128));
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let c = c.clone();
                std::thread::spawn(move || {
                    for i in 0..1000u64 {
                        if let Some(b) = c.get(i % 50) {
                            assert_eq!(b[0], (i % 50) as u8);
                        } else {
                            let mut blk = [0u8; crate::BLOCK_SIZE];
                            blk.fill((i % 50) as u8);
                            c.put(i % 50, std::sync::Arc::new(blk));
                        }
                    }
                    t
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let s = c.stats();
        assert_eq!(s.resident_blocks, 50);
    }
}
