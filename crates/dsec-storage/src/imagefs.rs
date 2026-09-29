//! File-level view layered on the CoW overlay: the guest filesystem.
//!
//! The base image's metadata table seeds the file map; writes go through
//! the overlay's block device (read-modify-write for partial blocks, fresh
//! extents when a file grows). Chronus sessions operate on this view.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::overlay::OverlayDev;
use crate::{Block, BlockId, Result};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FileMeta {
    pub first_block: BlockId,
    pub block_count: u64,
    pub size: u64,
    pub mode: u32,
    pub is_dir: bool,
}

/// Layered guest filesystem: base entries + private writes.
#[derive(Debug)]
pub struct LayeredImage {
    overlay: Arc<OverlayDev>,
    files: Mutex<BTreeMap<String, FileMeta>>,
    next_free: AtomicU64,
    version: AtomicU64,
}

impl LayeredImage {
    /// Builds the file table from the base image metadata.
    pub fn new(overlay: Arc<OverlayDev>) -> Self {
        let mut files = BTreeMap::new();
        for e in &overlay.base_image().meta().entries {
            files.insert(
                e.path.clone(),
                FileMeta {
                    first_block: e.first_block,
                    block_count: e.block_count,
                    size: e.size,
                    mode: e.mode,
                    is_dir: e.is_dir,
                },
            );
        }
        let base_blocks = overlay.base_image().block_count();
        LayeredImage {
            overlay,
            files: Mutex::new(files),
            next_free: AtomicU64::new(base_blocks),
            version: AtomicU64::new(0),
        }
    }

    pub fn overlay(&self) -> &Arc<OverlayDev> {
        &self.overlay
    }

    /// Number of mutations applied (for cache-invalidation tests).
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    fn alloc_extent(&self, len: usize) -> BlockId {
        let n_blocks = len.div_ceil(crate::BLOCK_SIZE).max(1) as u64;

        self.next_free.fetch_add(n_blocks, Ordering::Relaxed)
    }

    pub fn exists(&self, path: &str) -> bool {
        let Ok(p) = normalize(path) else { return false };
        self.files.lock().expect("fs poisoned").contains_key(&p)
    }

    pub fn stat(&self, path: &str) -> Result<FileMeta> {
        let p = normalize(path)?;
        self.files
            .lock()
            .expect("fs poisoned")
            .get(&p)
            .copied()
            .ok_or(crate::Error::PathNotFound(p))
    }

    /// Reads a file's full logical content.
    pub async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        let p = normalize(path)?;
        let meta = self
            .files
            .lock()
            .expect("fs poisoned")
            .get(&p)
            .copied()
            .ok_or_else(|| crate::Error::PathNotFound(p.clone()))?;
        if meta.is_dir {
            return Err(crate::Error::NotAFile(p));
        }
        let mut out = Vec::with_capacity(meta.size as usize);
        for i in 0..meta.block_count {
            let block = self.overlay.read_block(meta.first_block + i).await?;
            out.extend_from_slice(&block[..]);
        }
        out.truncate(meta.size as usize);
        Ok(out)
    }

    /// Writes a file (create or overwrite). Returns bytes written.
    pub async fn write_file(&self, path: &str, data: &[u8]) -> Result<u64> {
        let p = normalize(path)?;
        {
            let files = self.files.lock().expect("fs poisoned");
            if let Some(meta) = files.get(&p) {
                if meta.is_dir {
                    return Err(crate::Error::NotAFile(p));
                }
            }
        }
        self.write_extent(&p, data).await?;
        self.version.fetch_add(1, Ordering::Relaxed);
        Ok(data.len() as u64)
    }

    /// Appends to a file (creating it if absent).
    pub async fn append_file(&self, path: &str, data: &[u8]) -> Result<u64> {
        let p = normalize(path)?;
        // Extract the decision under a short-lived guard; never hold the
        // file-table lock across the await below (re-entrant deadlock).
        let meta = self.files.lock().expect("fs poisoned").get(&p).cloned();
        let existing = match meta {
            Some(m) if !m.is_dir => self.read_file(&p).await?,
            Some(_) => return Err(crate::Error::NotAFile(p)),
            None => Vec::new(),
        };
        let mut combined = existing;
        combined.extend_from_slice(data);
        self.write_extent(&p, &combined).await?;
        self.version.fetch_add(1, Ordering::Relaxed);
        Ok(combined.len() as u64)
    }

    async fn write_extent(&self, p: &str, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            let first = self.alloc_extent(1);
            self.files.lock().expect("fs poisoned").insert(
                p.to_string(),
                FileMeta {
                    first_block: first,
                    block_count: 0,
                    size: 0,
                    mode: 0o644,
                    is_dir: false,
                },
            );
            return Ok(());
        }
        let first = self.alloc_extent(data.len());
        let n_blocks = data.len().div_ceil(crate::BLOCK_SIZE) as u64;
        for (i, chunk) in data.chunks(crate::BLOCK_SIZE).enumerate() {
            let mut block: Block = [0u8; crate::BLOCK_SIZE];
            block[..chunk.len()].copy_from_slice(chunk);
            self.overlay.write_block(first + i as u64, block);
        }
        self.files.lock().expect("fs poisoned").insert(
            p.to_string(),
            FileMeta {
                first_block: first,
                block_count: n_blocks,
                size: data.len() as u64,
                mode: 0o644,
                is_dir: false,
            },
        );
        Ok(())
    }

    /// Lists direct children of a directory, sorted.
    pub fn list_dir(&self, path: &str) -> Result<Vec<String>> {
        let p = normalize(path)?;
        let files = self.files.lock().expect("fs poisoned");
        match files.get(&p) {
            Some(m) if m.is_dir => {}
            Some(_) => return Err(crate::Error::NotADirectory(p)),
            None if p != "/" => return Err(crate::Error::PathNotFound(p)),
            _ => {}
        }
        let prefix = if p == "/" {
            String::new()
        } else {
            format!("{}/", p.trim_end_matches('/'))
        };
        let mut out = Vec::new();
        for path in files.keys() {
            if let Some(rest) = path.strip_prefix(&prefix) {
                if !rest.is_empty() && !rest.contains('/') {
                    out.push(rest.to_string());
                }
            }
        }
        Ok(out)
    }

    /// Creates a directory and any missing parents (mkdir -p semantics).
    pub fn mkdir_p(&self, path: &str) -> Result<()> {
        let p = normalize(path)?;
        if p == "/" {
            return Ok(());
        }
        let mut files = self.files.lock().expect("fs poisoned");
        let mut cur = String::new();
        for seg in p.trim_matches('/').split('/') {
            cur.push('/');
            cur.push_str(seg);
            files.entry(cur.clone()).or_insert_with(|| FileMeta {
                first_block: 0,
                block_count: 0,
                size: 0,
                mode: 0o755,
                is_dir: true,
            });
        }
        self.version.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Removes a file, or a directory when `recursive`.
    pub fn rm(&self, path: &str, recursive: bool) -> Result<()> {
        let p = normalize(path)?;
        if p == "/" {
            return Err(crate::Error::InvalidArgument("refusing to remove /".into()));
        }
        let mut files = self.files.lock().expect("fs poisoned");
        let meta = files
            .get(&p)
            .copied()
            .ok_or_else(|| crate::Error::PathNotFound(p.clone()))?;
        if meta.is_dir {
            let prefix = format!("{}/", p);
            let has_children = files.keys().any(|k| k.starts_with(&prefix));
            if has_children && !recursive {
                return Err(crate::Error::InvalidArgument(format!(
                    "{} not empty (use recursive)",
                    p
                )));
            }
            let victims: Vec<String> = files
                .keys()
                .filter(|k| k.starts_with(&prefix) || k.as_str() == p.as_str())
                .cloned()
                .collect();
            for v in victims {
                files.remove(&v);
            }
        } else {
            files.remove(&p);
        }
        self.version.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    // -- pack_diff integration (working-state replication) ---------------

    /// Snapshots the full working state: dirty blocks + file table + the
    /// block allocator cursor. A replica that applies this pack observes
    /// the same filesystem.
    pub fn snapshot_diff(&self, epoch_ms: u64) -> crate::packdiff::DiffPack {
        let mut pack = self.overlay.snapshot_diff(epoch_ms);
        let files = self.files.lock().expect("fs poisoned");
        pack.files = files
            .iter()
            .map(|(path, meta)| crate::packdiff::FileWire {
                path: path.clone(),
                meta: *meta,
            })
            .collect();
        drop(files);
        pack.next_free = self.next_free.load(Ordering::Relaxed);
        pack
    }

    /// Applies a working-state pack (blocks + metadata).
    pub fn apply_diff(&self, pack: &crate::packdiff::DiffPack) -> Result<()> {
        self.overlay.apply_diff(pack)?;
        if !pack.files.is_empty() || pack.next_free > 0 {
            let mut files = self.files.lock().expect("fs poisoned");
            files.clear();
            for fw in &pack.files {
                files.insert(fw.path.clone(), fw.meta);
            }
            drop(files);
            self.next_free.store(
                pack.next_free.max(self.next_free.load(Ordering::Relaxed)),
                Ordering::Relaxed,
            );
        }
        self.version.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// Normalizes a path: absolute, no `..`, no trailing slash, collapsed `//`.
pub fn normalize(path: &str) -> Result<String> {
    if !path.starts_with('/') {
        return Err(crate::Error::InvalidPath(path.to_string()));
    }
    let mut stack: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                stack
                    .pop()
                    .ok_or_else(|| crate::Error::InvalidPath(path.to_string()))?;
            }
            s => stack.push(s),
        }
    }
    if stack.is_empty() {
        Ok("/".to_string())
    } else {
        Ok(format!("/{}", stack.join("/")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::erofs::{ErofsImageBuilder, OnDemandLoader};
    use crate::latency::LatencyModel;
    use crate::overlay::OverlayDev;
    use std::time::Duration;

    fn make_fs() -> (Arc<LayeredImage>, Arc<OverlayDev>) {
        let image = Arc::new(ErofsImageBuilder::agent_base().build());
        let loader = Arc::new(OnDemandLoader::new(
            image,
            Arc::new(crate::cache::LruBlockCache::new(64)),
            LatencyModel::fixed(Duration::ZERO),
        ));
        let overlay = Arc::new(OverlayDev::new(loader));
        (Arc::new(LayeredImage::new(overlay.clone())), overlay)
    }

    #[tokio::test]
    async fn reads_base_files() {
        let (fs, _) = make_fs();
        let hostname = fs.read_file("/etc/hostname").await.unwrap();
        assert_eq!(String::from_utf8(hostname).unwrap(), "sandbox-agent");
        let passwd = fs.read_file("/etc/passwd").await.unwrap();
        assert!(String::from_utf8(passwd).unwrap().contains("agent:x:1000"));
    }

    #[tokio::test]
    async fn write_read_new_file() {
        let (fs, _) = make_fs();
        fs.write_file("/tmp/notes.txt", b"hello world")
            .await
            .unwrap();
        let back = fs.read_file("/tmp/notes.txt").await.unwrap();
        assert_eq!(back, b"hello world");
        assert!(fs.exists("/tmp/notes.txt"));
    }

    #[tokio::test]
    async fn overwrite_and_append() {
        let (fs, _) = make_fs();
        fs.write_file("/tmp/a", b"0123456789").await.unwrap();
        fs.write_file("/tmp/a", b"short").await.unwrap();
        assert_eq!(fs.read_file("/tmp/a").await.unwrap(), b"short");
        fs.append_file("/tmp/a", b"-suffix").await.unwrap();
        assert_eq!(fs.read_file("/tmp/a").await.unwrap(), b"short-suffix");
    }

    #[tokio::test]
    async fn large_file_spans_blocks() {
        let (fs, _) = make_fs();
        let data: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        fs.write_file("/tmp/big.bin", &data).await.unwrap();
        let back = fs.read_file("/tmp/big.bin").await.unwrap();
        assert_eq!(back.len(), data.len());
        assert_eq!(back, data);
    }

    #[tokio::test]
    async fn list_mkdir_rm() {
        let (fs, _) = make_fs();
        fs.mkdir_p("/data/nested/deep").unwrap();
        fs.write_file("/data/nested/deep/f.txt", b"x")
            .await
            .unwrap();
        assert_eq!(fs.list_dir("/data").unwrap(), vec!["nested".to_string()]);
        assert_eq!(
            fs.list_dir("/data/nested/deep").unwrap(),
            vec!["f.txt".to_string()]
        );
        // rm file, then dirs
        fs.rm("/data/nested/deep/f.txt", false).unwrap();
        assert!(fs.rm("/data", false).is_err()); // non-empty
        fs.rm("/data", true).unwrap();
        assert!(!fs.exists("/data"));
    }

    #[tokio::test]
    async fn mutations_mark_overlay_dirty_for_packdiff() {
        let (fs, overlay) = make_fs();
        fs.write_file("/tmp/state.bin", b"dirty bytes")
            .await
            .unwrap();
        assert!(overlay.dirty_bytes() >= crate::BLOCK_SIZE as u64);
    }

    #[tokio::test]
    async fn shared_view_across_handles() {
        let (fs, _) = make_fs();
        fs.write_file("/tmp/shared.txt", b"v1").await.unwrap();
        let clone = fs.clone();
        let reader = tokio::spawn(async move { clone.read_file("/tmp/shared.txt").await.unwrap() });
        assert_eq!(reader.await.unwrap(), b"v1");
    }

    #[test]
    fn path_normalization() {
        assert_eq!(normalize("/").unwrap(), "/");
        assert_eq!(normalize("/etc//passwd/").unwrap(), "/etc/passwd");
        assert_eq!(normalize("/a/./b").unwrap(), "/a/b");
        assert_eq!(normalize("/a/b/../c").unwrap(), "/a/c");
        assert!(normalize("relative").is_err());
        assert!(normalize("/a/../..").is_err());
    }
}
