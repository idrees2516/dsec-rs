//! `pack_diff`: incremental working-state replication.
//!
//! When the agent loop needs to fan out an army of replicas of a running
//! sandbox (or migrate one), shipping the full image is wasteful — the
//! base layer is already present on the destination. A `DiffPack` carries
//! only the dirty blocks captured since the last snapshot, which is the
//! paper's `pack_diff` fast path.

use serde::{Deserialize, Serialize};

use crate::overlay::OverlayDev;
use crate::{Block, BlockId, Result};

/// Serialized incremental snapshot of an overlay's working state.
///
/// `Block` (`[u8; 4096]`) has no serde impl for large arrays, so packs
/// store owned `Vec<u8>` payloads (they are transfer units anyway).
///
/// A pack may also carry the layered filesystem's metadata (the file table
/// and the next free block), because in the real system FS metadata lives in
/// blocks and therefore rides the diff automatically. Block-only packs
/// (metadata omitted) remain valid — used when only raw device state is
/// replicated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffPack {
    pub base_image_id: String,
    pub base_digest: String,
    pub created_epoch_ms: u64,
    /// (block id, content) pairs, sorted by block id.
    pub blocks: Vec<(BlockId, Vec<u8>)>,
    /// Layered-FS metadata shipped with the diff.
    #[serde(default)]
    pub files: Vec<FileWire>,
    /// Next free block of the layered allocator (prevents extent reuse).
    #[serde(default)]
    pub next_free: BlockId,
}

/// Wire form of one file-table entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileWire {
    pub path: String,
    pub meta: crate::imagefs::FileMeta,
}

impl PartialEq for DiffPack {
    fn eq(&self, other: &Self) -> bool {
        self.base_image_id == other.base_image_id
            && self.base_digest == other.base_digest
            && self.created_epoch_ms == other.created_epoch_ms
            && self.blocks == other.blocks
            && self.files == other.files
            && self.next_free == other.next_free
    }
}

/// Cost accounting for a pack vs a full image transfer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiffStats {
    pub changed_blocks: u64,
    pub total_blocks: u64,
    pub pack_bytes: u64,
    pub full_image_bytes: u64,
}

impl DiffStats {
    /// Fraction of the image that had to move.
    pub fn changed_ratio(&self) -> f64 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        self.changed_blocks as f64 / self.total_blocks as f64
    }
    /// Bytes saved relative to a full copy.
    pub fn bytes_saved(&self) -> u64 {
        self.full_image_bytes.saturating_sub(self.pack_bytes)
    }
}

impl OverlayDev {
    /// Captures and clears the dirty set into a `DiffPack`.
    pub fn snapshot_diff(&self, epoch_ms: u64) -> DiffPack {
        let dirty = self.take_dirty();
        let cow = self.fork_state();
        let mut blocks: Vec<(BlockId, Vec<u8>)> = dirty
            .iter()
            .filter_map(|id| cow.get(id).map(|b| (*id, b.to_vec())))
            .collect();
        blocks.sort_by_key(|(id, _)| *id);
        DiffPack {
            base_image_id: self.base_image().image_id().to_string(),
            base_digest: self.base_image().digest().to_string(),
            created_epoch_ms: epoch_ms,
            blocks,
            files: Vec::new(),
            next_free: 0,
        }
    }

    /// Applies a pack onto this overlay (replica side). The applied blocks
    /// are marked clean so a subsequent diff chain stays minimal.
    pub fn apply_diff(&self, pack: &DiffPack) -> Result<()> {
        let image = self.base_image();
        if pack.base_image_id != image.image_id() || pack.base_digest != image.digest() {
            return Err(crate::Error::DigestMismatch(
                pack.base_image_id.clone(),
                image.digest().to_string(),
                pack.base_digest.clone(),
            ));
        }
        let ids: Vec<BlockId> = pack.blocks.iter().map(|(id, _)| *id).collect();
        for (id, data) in &pack.blocks {
            let block: Block = data
                .as_slice()
                .try_into()
                .map_err(|_| crate::Error::Other("bad block length in diff pack".into()))?;
            self.import_block(*id, block);
        }
        self.clear_dirty(&ids);
        Ok(())
    }

    /// Size/cost comparison of the last-capturable diff vs a full copy.
    pub fn diff_stats(&self) -> DiffStats {
        let dirty = self.dirty_blocks().len() as u64;
        let total = self.base_image().block_count();
        DiffStats {
            changed_blocks: dirty,
            total_blocks: total,
            pack_bytes: dirty * crate::BLOCK_SIZE as u64,
            full_image_bytes: total * crate::BLOCK_SIZE as u64,
        }
    }
}

/// Packs + applies between two overlays over the same base image.
pub fn replicate(src: &OverlayDev, dst: &OverlayDev, epoch_ms: u64) -> Result<DiffStats> {
    let pack = src.snapshot_diff(epoch_ms);
    let stats = DiffStats {
        changed_blocks: pack.blocks.len() as u64,
        total_blocks: dst.base_image().block_count(),
        pack_bytes: pack.blocks.len() as u64 * crate::BLOCK_SIZE as u64,
        full_image_bytes: dst.base_image().full_size(),
    };
    dst.apply_diff(&pack)?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::LruBlockCache;
    use crate::erofs::{ErofsImageBuilder, OnDemandLoader};
    use crate::latency::LatencyModel;
    use crate::overlay::OverlayDev;
    use std::sync::Arc;
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
    async fn replicate_small_diff() {
        let src = make_overlay();
        let dst = make_overlay();
        // Mutate a small region.
        src.write_range(0, b"working-state delta").await.unwrap();
        let stats = replicate(&src, &dst, 1234).unwrap();
        assert_eq!(stats.changed_blocks, 1);
        assert!(stats.changed_ratio() < 0.3);
        assert!(stats.bytes_saved() > 0);
        let mut buf = vec![0u8; 19];
        dst.read_range(0, &mut buf).await.unwrap();
        assert_eq!(&buf, b"working-state delta");
        // Dirty set consumed on source, clear on destination.
        assert_eq!(src.dirty_blocks().len(), 0);
        assert_eq!(dst.dirty_blocks().len(), 0);
    }

    #[tokio::test]
    async fn diff_chain_is_incremental() {
        let src = make_overlay();
        let dst = make_overlay();
        src.write_range(0, b"step1").await.unwrap();
        let p1 = src.snapshot_diff(1);
        assert_eq!(p1.blocks.len(), 1);
        // No new writes -> empty diff.
        let p2 = src.snapshot_diff(2);
        assert_eq!(p2.blocks.len(), 0);
        // New write to a different block -> one-block diff.
        src.write_range(crate::BLOCK_SIZE as u64, b"step2")
            .await
            .unwrap();
        let p3 = src.snapshot_diff(3);
        assert_eq!(p3.blocks.len(), 1);
        assert_eq!(p3.blocks[0].0, 1);
        dst.apply_diff(&p1).unwrap();
        dst.apply_diff(&p3).unwrap();
        let mut b = vec![0u8; 5];
        dst.read_range(0, &mut b).await.unwrap();
        assert_eq!(&b, b"step1");
        dst.read_range(crate::BLOCK_SIZE as u64, &mut b)
            .await
            .unwrap();
        assert_eq!(&b, b"step2");
    }

    #[tokio::test]
    async fn digest_mismatch_rejected() {
        let src = make_overlay();
        let other_image = Arc::new(
            ErofsImageBuilder::new("dsec/other-base")
                .add_file("/x", vec![1, 2, 3], 0o644)
                .build(),
        );
        let loader = Arc::new(OnDemandLoader::new(
            other_image,
            Arc::new(LruBlockCache::new(8)),
            LatencyModel::fixed(Duration::ZERO),
        ));
        let dst = OverlayDev::new(loader);
        let pack = src.snapshot_diff(1);
        assert!(dst.apply_diff(&pack).is_err());
    }

    #[test]
    fn pack_serde_roundtrip() {
        let mut block = [0u8; crate::BLOCK_SIZE];
        block[..4].copy_from_slice(b"pack");
        let pack = DiffPack {
            base_image_id: "dsec/agent-base".into(),
            base_digest: "dsec1-deadbeef".into(),
            created_epoch_ms: 42,
            blocks: vec![(7, block.to_vec())],
            files: vec![],
            next_free: 0,
        };
        let s = serde_json::to_vec(&pack).unwrap();
        let back: DiffPack = serde_json::from_slice(&s).unwrap();
        assert_eq!(back, pack);
    }
}
