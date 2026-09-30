//! Per-sandbox copy-on-write overlay (OverlayBD-style layered device).
//!
//! Reads consult the private CoW map first, then fall through to the
//! shared on-demand loader. Writes land in the private map and mark the
//! block dirty — the dirty set is exactly what [`crate::packdiff`]
//! snapshots for working-state replication. Sparse writes beyond the base
//! image grow the device implicitly.

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::erofs::OnDemandLoader;
use crate::{Block, BlockId, Result};

#[derive(Debug, Default)]
struct OverlayInner {
    cow: HashMap<BlockId, Arc<Block>>,
    dirty: BTreeSet<BlockId>,
}

/// Copy-on-write layered block device over a shared base image.
#[derive(Debug)]
pub struct OverlayDev {
    loader: Arc<OnDemandLoader>,
    inner: Mutex<OverlayInner>,
    /// Highest written block + 1 (device may grow past the base).
    virtual_blocks: AtomicU64,
    written_bytes: AtomicU64,
}

impl OverlayDev {
    pub fn new(loader: Arc<OnDemandLoader>) -> Self {
        let base_blocks = loader.image().block_count();
        OverlayDev {
            loader,
            inner: Mutex::new(OverlayInner::default()),
            virtual_blocks: AtomicU64::new(base_blocks),
            written_bytes: AtomicU64::new(0),
        }
    }

    pub fn loader(&self) -> &Arc<OnDemandLoader> {
        &self.loader
    }

    pub fn base_image(&self) -> &Arc<crate::erofs::ErofsImage> {
        self.loader.image()
    }

    /// Current device size in blocks (base + grown tail).
    pub fn block_count(&self) -> u64 {
        let base = self.loader.image().block_count();
        let virt = self.virtual_blocks.load(Ordering::Relaxed);
        base.max(virt)
    }

    /// Reads one block through the CoW path.
    pub async fn read_block(&self, id: BlockId) -> Result<Arc<Block>> {
        if let Some(b) = self.inner.lock().expect("overlay poisoned").cow.get(&id) {
            return Ok(b.clone());
        }
        self.loader.read_block(id).await
    }

    /// Writes one block; the write is private to this overlay and marks
    /// the block dirty for the next `pack_diff`.
    pub fn write_block(&self, id: BlockId, data: Block) {
        let mut inner = self.inner.lock().expect("overlay poisoned");
        inner.cow.insert(id, Arc::new(data));
        inner.dirty.insert(id);
        drop(inner);
        self.grow_to(id + 1);
        self.written_bytes
            .fetch_add(crate::BLOCK_SIZE as u64, Ordering::Relaxed);
    }

    /// Byte-range read (block-aligned internally; sparse tail is zeros).
    pub async fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        let bs = crate::BLOCK_SIZE as u64;
        let mut done = 0usize;
        while done < out.len() {
            let block = (offset + done as u64) / bs;
            let in_off = ((offset + done as u64) % bs) as usize;
            let n = (bs as usize - in_off).min(out.len() - done);
            if block >= self.block_count() {
                out[done..done + n].fill(0);
                done += n;
                continue;
            }
            if let Some(b) = self
                .inner
                .lock()
                .expect("overlay poisoned")
                .cow
                .get(&block)
                .cloned()
            {
                out[done..done + n].copy_from_slice(&b[in_off..in_off + n]);
                done += n;
                continue;
            }
            let b = self.loader.read_block(block).await?;
            out[done..done + n].copy_from_slice(&b[in_off..in_off + n]);
            done += n;
        }
        Ok(())
    }

    /// Byte-range write (partial blocks are read-modify-write).
    pub async fn write_range(&self, offset: u64, data: &[u8]) -> Result<()> {
        let bs = crate::BLOCK_SIZE as u64;
        let mut done = 0usize;
        while done < data.len() {
            let block = (offset + done as u64) / bs;
            let in_off = ((offset + done as u64) % bs) as usize;
            let n = (bs as usize - in_off).min(data.len() - done);
            let mut buf = [0u8; crate::BLOCK_SIZE];
            if (in_off != 0 || n != crate::BLOCK_SIZE) && block < self.block_count() {
                let existing = self.read_block(block).await?;
                buf.copy_from_slice(&existing[..]);
            }
            buf[in_off..in_off + n].copy_from_slice(&data[done..done + n]);
            self.write_block(block, buf);
            done += n;
        }
        Ok(())
    }

    fn grow_to(&self, blocks: u64) {
        let mut cur = self.virtual_blocks.load(Ordering::Relaxed);
        while blocks > cur {
            match self.virtual_blocks.compare_exchange(
                cur,
                blocks,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Atomically consumes the dirty set and returns the dirty blocks'
    /// (id, content) pairs straight out of the CoW map — one lock pass,
    /// no full map clone (the pack_diff snapshot hot path).
    pub(crate) fn take_dirty_blocks(&self) -> Vec<(BlockId, Vec<u8>)> {
        let mut inner = self.inner.lock().expect("overlay poisoned");
        let dirty = std::mem::take(&mut inner.dirty);
        let mut out = Vec::with_capacity(dirty.len());
        for id in &dirty {
            if let Some(b) = inner.cow.get(id) {
                out.push((*id, b.to_vec()));
            }
        }
        out
    }

    /// Blocks dirtied since the last snapshot.
    pub fn dirty_blocks(&self) -> Vec<BlockId> {
        self.inner
            .lock()
            .expect("overlay poisoned")
            .dirty
            .iter()
            .copied()
            .collect()
    }

    pub fn dirty_bytes(&self) -> u64 {
        self.inner.lock().expect("overlay poisoned").dirty.len() as u64 * crate::BLOCK_SIZE as u64
    }

    /// Bytes written through this overlay (CoW volume).
    pub fn written_bytes(&self) -> u64 {
        self.written_bytes.load(Ordering::Relaxed)
    }

    /// Mark blocks clean after an external apply.
    pub(crate) fn clear_dirty(&self, blocks: &[BlockId]) {
        let mut inner = self.inner.lock().expect("overlay poisoned");
        for b in blocks {
            inner.dirty.remove(b);
        }
    }

    /// Inserts a block from an applied diff (no dirty marking).
    pub(crate) fn import_block(&self, id: BlockId, data: Block) {
        let mut inner = self.inner.lock().expect("overlay poisoned");
        inner.cow.insert(id, Arc::new(data));
        self.grow_to(id + 1);
    }

    /// Snapshot of private CoW state (used to fork sandboxes).
    pub fn fork_state(&self) -> HashMap<BlockId, Arc<Block>> {
        self.inner.lock().expect("overlay poisoned").cow.clone()
    }

    /// Restores private CoW state onto a fresh overlay (mirror of fork).
    pub fn restore_state(&self, state: &HashMap<BlockId, Arc<Block>>) {
        let mut inner = self.inner.lock().expect("overlay poisoned");
        for (id, b) in state {
            inner.cow.insert(*id, b.clone());
            self.grow_to(id + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::LruBlockCache;
    use crate::erofs::{ErofsImageBuilder, OnDemandLoader};
    use crate::latency::LatencyModel;
    use std::time::Duration;

    fn make_overlay() -> OverlayDev {
        let image = Arc::new(ErofsImageBuilder::agent_base().build());
        let loader = Arc::new(OnDemandLoader::new(
            image,
            Arc::new(LruBlockCache::new(64)),
            LatencyModel::fixed(Duration::ZERO),
        ));
        OverlayDev::new(loader)
    }

    #[tokio::test]
    async fn read_passes_through_to_base() {
        let ov = make_overlay();
        let image = ov.base_image();
        let e = image
            .meta()
            .entries
            .iter()
            .find(|e| e.path == "/etc/os-release")
            .unwrap();
        let mut buf = vec![0u8; e.size as usize];
        ov.read_range(e.first_block * crate::BLOCK_SIZE as u64, &mut buf)
            .await
            .unwrap();
        assert!(String::from_utf8(buf)
            .unwrap()
            .starts_with("NAME=\"DSec Linux\""));
    }

    #[tokio::test]
    async fn write_is_private_and_visible() {
        let ov = make_overlay();
        ov.write_range(0, b"OVERWRITTEN!!!").await.unwrap();
        let mut buf = vec![0u8; 14];
        ov.read_range(0, &mut buf).await.unwrap();
        assert_eq!(&buf, b"OVERWRITTEN!!!");
        // Base image untouched (shared Arc).
        let base = ov.base_image();
        assert_ne!(&base.block(0).unwrap()[..13], &buf[..]);
        assert_eq!(ov.dirty_bytes(), crate::BLOCK_SIZE as u64);
    }

    #[tokio::test]
    async fn partial_write_read_modify_write() {
        let ov = make_overlay();
        let orig = {
            let mut b = vec![0u8; 16];
            ov.read_range(0, &mut b).await.unwrap();
            b
        };
        ov.write_range(4, b"XY").await.unwrap();
        let mut after = vec![0u8; 16];
        ov.read_range(0, &mut after).await.unwrap();
        assert_eq!(&after[..4], &orig[..4]);
        assert_eq!(&after[4..6], b"XY");
        assert_eq!(&after[6..], &orig[6..]);
    }

    #[tokio::test]
    async fn grows_beyond_base() {
        let ov = make_overlay();
        let base_blocks = ov.base_image().block_count();
        let tail_off = (base_blocks + 3) * crate::BLOCK_SIZE as u64;
        ov.write_range(tail_off, b"tail").await.unwrap();
        let mut buf = vec![0u8; 4];
        ov.read_range(tail_off, &mut buf).await.unwrap();
        assert_eq!(&buf, b"tail");
        assert_eq!(ov.block_count(), base_blocks + 4);
    }

    #[tokio::test]
    async fn fork_state_replicates() {
        let a = make_overlay();
        a.write_range(0, b"cloned state").await.unwrap();
        let state = a.fork_state();
        let b = make_overlay();
        b.restore_state(&state);
        let mut buf = vec![0u8; 12];
        b.read_range(0, &mut buf).await.unwrap();
        assert_eq!(&buf, b"cloned state");
    }
}
