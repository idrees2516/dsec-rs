//! EROFS-style read-only base images and the on-demand loader.
//!
//! An [`ErofsImage`] is an immutable set of 4 KiB blocks plus a metadata
//! table mapping file paths to block extents. Images are `Arc`-shared
//! across every sandbox on a node: creation never copies the image, it
//! only starts faulting blocks in through the [`OnDemandLoader`], which is
//! the userspace rendition of the paper's EROFS rootfs with asynchronous
//! on-demand reads from the backing store (3FS in production, a latency
//! model here).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::cache::LruBlockCache;
use crate::latency::LatencyModel;
use crate::{Block, BlockId, BlockRange, Result};

/// One file (or directory marker) in an image's metadata table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: String,
    /// First block of the extent (blocks are contiguous within a file).
    pub first_block: BlockId,
    pub block_count: u64,
    /// Logical file size in bytes (trailing block padding excluded).
    pub size: u64,
    pub mode: u32,
    pub is_dir: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageMeta {
    pub image_id: String,
    pub digest: String,
    pub size_bytes: u64,
    pub block_count: u64,
    pub entries: Vec<FileEntry>,
}

/// Immutable, shareable base image.
#[derive(Debug)]
pub struct ErofsImage {
    meta: ImageMeta,
    blocks: Arc<Vec<Block>>,
}

impl ErofsImage {
    pub fn meta(&self) -> &ImageMeta {
        &self.meta
    }

    pub fn image_id(&self) -> &str {
        &self.meta.image_id
    }

    pub fn digest(&self) -> &str {
        &self.meta.digest
    }

    pub fn block_count(&self) -> u64 {
        self.meta.block_count
    }

    /// Number of physical bytes if the image were copied in full.
    pub fn full_size(&self) -> u64 {
        self.meta.block_count * crate::BLOCK_SIZE as u64
    }

    /// Direct block access (no cache); used by the loader on cache misses.
    pub fn block(&self, id: BlockId) -> Option<&Block> {
        self.blocks.get(id as usize)
    }

    pub fn blocks(&self) -> Arc<Vec<Block>> {
        self.blocks.clone()
    }
}

/// Deterministic content digest: chained CRC32 over blocks + metadata.
fn compute_digest(image_id: &str, blocks: &[Block], entries: &[FileEntry]) -> String {
    let mut crc = dsec_protocol::frame::CRC32::compute(image_id.as_bytes());
    for b in blocks {
        crc = dsec_protocol::frame::CRC32::compute(&crc.to_be_bytes())
            ^ dsec_protocol::frame::CRC32::compute(b);
    }
    for e in entries {
        crc ^= dsec_protocol::frame::CRC32::compute(e.path.as_bytes());
        crc = crc.rotate_left(1);
    }
    format!("dsec1-{:08x}", crc)
}

/// Builds images from a flat path -> content table.
#[derive(Debug, Default)]
pub struct ErofsImageBuilder {
    image_id: String,
    files: Vec<(String, Vec<u8>, u32)>,
    dirs: Vec<String>,
}

impl ErofsImageBuilder {
    pub fn new(image_id: impl Into<String>) -> Self {
        ErofsImageBuilder {
            image_id: image_id.into(),
            files: Vec::new(),
            dirs: Vec::new(),
        }
    }

    pub fn add_file(
        mut self,
        path: impl Into<String>,
        content: impl Into<Vec<u8>>,
        mode: u32,
    ) -> Self {
        self.files.push((path.into(), content.into(), mode));
        self
    }

    pub fn add_dir(mut self, path: impl Into<String>) -> Self {
        self.dirs.push(path.into());
        self
    }

    /// Standard tiny agent image used across tests and examples.
    pub fn agent_base() -> Self {
        let hostname = format!("{}-agent", "sandbox");
        Self::new("dsec/agent-base")
            .add_dir("/etc")
            .add_dir("/home/agent")
            .add_dir("/tmp")
            .add_dir("/var/log")
            .add_file("/etc/hostname", hostname, 0o644)
            .add_file(
                "/etc/os-release",
                "NAME=\"DSec Linux\"\nVERSION=\"1.0\"\n"
                    .to_string()
                    .into_bytes(),
                0o644,
            )
            .add_file(
                "/etc/passwd",
                "root:x:0:0:root:/root:/bin/sh\nagent:x:1000:1000:agent:/home/agent:/bin/sh\n"
                    .to_string()
                    .into_bytes(),
                0o644,
            )
            .add_file("/bin/sh", vec![0x7f, b'E', b'L', b'F'], 0o755)
            .add_file(
                "/home/agent/README.md",
                "# Agent sandbox\nRun tasks here.\n"
                    .to_string()
                    .into_bytes(),
                0o644,
            )
            .add_file(
                "/var/log/syslog",
                "boot ok\n".to_string().into_bytes(),
                0o600,
            )
    }

    pub fn build(self) -> ErofsImage {
        let mut entries: Vec<FileEntry> = Vec::new();
        let mut blocks: Vec<Block> = Vec::new();
        let mut next_block: BlockId = 0;

        for dir in self.dirs {
            entries.push(FileEntry {
                path: dir,
                first_block: 0,
                block_count: 0,
                size: 0,
                mode: 0o755,
                is_dir: true,
            });
        }
        for (path, content, mode) in self.files {
            let n_blocks = content.len().div_ceil(crate::BLOCK_SIZE).max(1) as u64;
            let first = next_block;
            for i in 0..n_blocks as usize {
                let mut b = [0u8; crate::BLOCK_SIZE];
                let off = i * crate::BLOCK_SIZE;
                let end = (off + crate::BLOCK_SIZE).min(content.len());
                b[..end - off].copy_from_slice(&content[off..end]);
                blocks.push(b);
            }
            next_block += n_blocks;
            entries.push(FileEntry {
                path,
                first_block: first,
                block_count: n_blocks,
                size: content.len() as u64,
                mode,
                is_dir: false,
            });
        }
        entries.sort_by(|a, b| a.path.cmp(&b.path));
        let digest = compute_digest(&self.image_id, &blocks, &entries);
        let size_bytes = entries.iter().filter(|e| !e.is_dir).map(|e| e.size).sum();
        ErofsImage {
            meta: ImageMeta {
                image_id: self.image_id,
                digest,
                size_bytes,
                block_count: blocks.len() as u64,
                entries,
            },
            blocks: Arc::new(blocks),
        }
    }
}

/// Stats for the on-demand path.
#[derive(Debug, Default, Clone, Copy)]
pub struct LoaderStats {
    pub hits: u64,
    pub misses: u64,
    pub fetches: u64,
    pub bytes_fetched: u64,
    pub prefetch_hits: u64,
}

impl LoaderStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
    /// Effective I/O amplification: bytes moved vs bytes logically read.
    pub fn io_amplification(&self, bytes_read: u64) -> f64 {
        if bytes_read == 0 {
            0.0
        } else {
            self.bytes_fetched as f64 / bytes_read as f64
        }
    }
}

/// Fault-driven block loader with a shared LRU cache.
#[derive(Debug)]
pub struct OnDemandLoader {
    image: Arc<ErofsImage>,
    cache: Arc<LruBlockCache>,
    latency: LatencyModel,
    stats_hits: AtomicU64,
    stats_misses: AtomicU64,
    stats_fetches: AtomicU64,
    stats_bytes: AtomicU64,
}

impl OnDemandLoader {
    pub fn new(image: Arc<ErofsImage>, cache: Arc<LruBlockCache>, latency: LatencyModel) -> Self {
        OnDemandLoader {
            image,
            cache,
            latency,
            stats_hits: AtomicU64::new(0),
            stats_misses: AtomicU64::new(0),
            stats_fetches: AtomicU64::new(0),
            stats_bytes: AtomicU64::new(0),
        }
    }

    pub fn image(&self) -> &Arc<ErofsImage> {
        &self.image
    }

    pub fn cache(&self) -> &Arc<LruBlockCache> {
        &self.cache
    }

    /// Reads one block, faulting it from the backing store on miss.
    pub async fn read_block(&self, id: BlockId) -> Result<Arc<Block>> {
        if let Some(b) = self.cache.get(id) {
            self.stats_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(b);
        }
        self.stats_misses.fetch_add(1, Ordering::Relaxed);
        let src = self
            .image
            .block(id)
            .cloned()
            .ok_or_else(|| crate::Error::BlockOutOfRange(id, self.image.block_count()))?;
        let d = self.latency.sample();
        if !d.is_zero() {
            tokio::time::sleep(d).await;
        }
        self.stats_fetches.fetch_add(1, Ordering::Relaxed);
        self.stats_bytes
            .fetch_add(crate::BLOCK_SIZE as u64, Ordering::Relaxed);
        let arc = Arc::new(src);
        self.cache.put(id, arc.clone());
        Ok(arc)
    }

    /// Reads a byte range into `out` (offset within the block device).
    pub async fn read_range(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        let bs = crate::BLOCK_SIZE as u64;
        let mut done = 0usize;
        while done < out.len() {
            let block = (offset + done as u64) / bs;
            let in_off = ((offset + done as u64) % bs) as usize;
            let n = (bs as usize - in_off).min(out.len() - done);
            if block >= self.image.block_count() {
                // Sparse tail reads as zeros.
                out[done..done + n].fill(0);
                done += n;
                continue;
            }
            let b = self.read_block(block).await?;
            out[done..done + n].copy_from_slice(&b[in_off..in_off + n]);
            done += n;
        }
        Ok(())
    }

    /// Warms the cache for the given ranges without waiting.
    /// Returns spawned task handles (the async prefetch pipeline).
    pub fn prefetch(self: &Arc<Self>, ranges: &[BlockRange]) -> Vec<tokio::task::JoinHandle<()>> {
        ranges
            .iter()
            .flat_map(|r| r.start..r.start + r.count)
            .filter(|&b| b < self.image.block_count())
            .map(|b| {
                let loader = self.clone();
                tokio::spawn(async move {
                    let _ = loader.read_block(b).await;
                })
            })
            .collect()
    }

    pub fn stats(&self) -> LoaderStats {
        LoaderStats {
            hits: self.stats_hits.load(Ordering::Relaxed),
            misses: self.stats_misses.load(Ordering::Relaxed),
            fetches: self.stats_fetches.load(Ordering::Relaxed),
            bytes_fetched: self.stats_bytes.load(Ordering::Relaxed),
            prefetch_hits: 0,
        }
    }
}

/// A registry of images available on a node (image locality for placement).
#[derive(Debug, Default, Clone)]
pub struct ImageRegistry {
    images: HashMap<String, (Arc<ErofsImage>, Arc<OnDemandLoader>)>,
}

impl ImageRegistry {
    pub fn register(
        &mut self,
        image: Arc<ErofsImage>,
        cache: Arc<LruBlockCache>,
        latency: LatencyModel,
    ) {
        let loader = Arc::new(OnDemandLoader::new(image.clone(), cache, latency));
        self.images
            .insert(image.image_id().to_string(), (image, loader));
    }

    pub fn get(&self, image_id: &str) -> Option<Arc<OnDemandLoader>> {
        self.images.get(image_id).map(|(_, l)| l.clone())
    }

    pub fn contains(&self, image_id: &str) -> bool {
        self.images.contains_key(image_id)
    }

    pub fn list(&self) -> Vec<String> {
        self.images.keys().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn zero_loader() -> (Arc<ErofsImage>, Arc<OnDemandLoader>) {
        let image = Arc::new(ErofsImageBuilder::agent_base().build());
        let cache = Arc::new(LruBlockCache::new(64));
        let loader = Arc::new(OnDemandLoader::new(
            image.clone(),
            cache,
            LatencyModel::fixed(Duration::ZERO),
        ));
        (image, loader)
    }

    #[tokio::test]
    async fn faults_and_caches() {
        let (image, loader) = zero_loader();
        let e = image
            .meta()
            .entries
            .iter()
            .find(|e| e.path == "/etc/hostname")
            .unwrap();
        let mut buf = vec![0u8; e.size as usize];
        loader
            .read_range(e.first_block * crate::BLOCK_SIZE as u64, &mut buf)
            .await
            .unwrap();
        assert_eq!(String::from_utf8(buf).unwrap(), "sandbox-agent");
        let s = loader.stats();
        assert_eq!(s.misses, 1); // one block
                                 // Second read hits cache, no new fetch.
        let mut buf2 = vec![0u8; e.size as usize];
        loader
            .read_range(e.first_block * crate::BLOCK_SIZE as u64, &mut buf2)
            .await
            .unwrap();
        let s = loader.stats();
        assert_eq!(s.fetches, 1);
        assert_eq!(s.hits, 1);
    }

    #[tokio::test]
    async fn cache_shared_across_loaders() {
        let (image, loader) = zero_loader();
        let cache = loader.cache().clone();
        let other = OnDemandLoader::new(image.clone(), cache, LatencyModel::fixed(Duration::ZERO));
        let _ = loader.read_block(0).await.unwrap();
        let _ = other.read_block(0).await.unwrap();
        let s = other.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 0); // shared node cache
    }

    #[tokio::test]
    async fn prefetch_warms_cache() {
        let (image, loader) = zero_loader();
        let handles = loader.prefetch(&[BlockRange {
            start: 0,
            count: image.block_count(),
        }]);
        for h in handles {
            h.await.unwrap();
        }
        let s = loader.stats();
        assert_eq!(s.fetches, image.block_count());
        assert!(s.bytes_fetched > 0);
    }

    #[test]
    fn digest_stable_and_content_addressed() {
        let a = ErofsImageBuilder::agent_base().build();
        let b = ErofsImageBuilder::agent_base().build();
        assert_eq!(a.digest(), b.digest());
        let c = ErofsImageBuilder::new("dsec/agent-base")
            .add_file("/etc/hostname", "different".to_string().into_bytes(), 0o644)
            .build();
        assert_ne!(a.digest(), c.digest());
    }

    #[test]
    fn registry_lookup() {
        let image = Arc::new(ErofsImageBuilder::agent_base().build());
        let mut reg = ImageRegistry::default();
        reg.register(
            image,
            Arc::new(LruBlockCache::new(8)),
            LatencyModel::fixed(Duration::ZERO),
        );
        assert!(reg.contains("dsec/agent-base"));
        assert!(!reg.contains("missing"));
        assert_eq!(reg.len(), 1);
    }
}
