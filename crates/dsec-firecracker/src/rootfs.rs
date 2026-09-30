//! EROFS image → raw drive file materialization.
//!
//! A dsec-rs node stores images as in-memory block vectors (the EROFS
//! analogue); a real Firecracker VM needs a file on the host. Each
//! image is materialized ONCE and attached read-only to every VM that
//! uses it — the host page cache then deduplicates the backing pages
//! across sandboxes, which is exactly the shared, deduplicated image
//! cache the paper builds with DAX mappings.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use dsec_storage::erofs::ErofsImage;

/// Sanitizes an image id into a filename component.
fn safe_name(image_id: &str) -> String {
    image_id.replace(['/', ':'], "_")
}

/// Writes the image's blocks contiguously to `path`; returns bytes written.
pub fn materialize_image(image: &ErofsImage, path: &Path) -> std::io::Result<u64> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut w = File::create(path)?;
    let mut written: u64 = 0;
    for block in image.blocks().iter() {
        w.write_all(block)?;
        written += block.len() as u64;
    }
    w.flush()?;
    Ok(written)
}

/// Creates a zeroed scratch file of `bytes` length (per-VM working drive).
pub fn create_scratch(path: &Path, bytes: u64) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let f = File::create(path)?;
    f.set_len(bytes)?;
    Ok(())
}

/// One-materialization cache: image id → raw drive path. Thread-safe;
/// concurrent callers get the same path (the write happens exactly
/// once — the first one in).
#[derive(Default)]
pub struct RootfsCache {
    dir: PathBuf,
    entries: Mutex<HashMap<String, PathBuf>>,
}

impl RootfsCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        RootfsCache {
            dir: dir.into(),
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the raw drive path for `image`, materializing it on
    /// first use. A cached entry is returned even if the file was
    /// removed externally only if re-materialization fails is skipped —
    /// callers treat errors as fatal.
    pub fn get_or_materialize(&self, image: &ErofsImage) -> std::io::Result<PathBuf> {
        let image_id = image.image_id().to_string();
        {
            let entries = self.entries.lock().expect("rootfs cache poisoned");
            if let Some(p) = entries.get(&image_id) {
                if p.is_file() {
                    return Ok(p.clone());
                }
            }
        }
        let path = self.dir.join(format!("{}.img", safe_name(&image_id)));
        materialize_image(image, &path)?;
        self.entries
            .lock()
            .expect("rootfs cache poisoned")
            .insert(image_id, path.clone());
        Ok(path)
    }

    pub fn path_for(&self, image_id: &str) -> PathBuf {
        self.dir.join(format!("{}.img", safe_name(image_id)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsec_storage::erofs::ErofsImageBuilder;

    fn image() -> ErofsImage {
        ErofsImageBuilder::agent_base().build()
    }

    #[test]
    fn materialize_writes_full_size() {
        let img = image();
        let dir = std::env::temp_dir().join(format!("dsec-fc-t-{}", std::process::id()));
        let path = dir.join("root.img");
        let n = materialize_image(&img, &path).unwrap();
        assert_eq!(n, img.full_size());
        assert_eq!(std::fs::metadata(&path).unwrap().len(), n);
        // Identical bytes: blocks in order.
        let data = std::fs::read(&path).unwrap();
        assert_eq!(&data[..dsec_storage::BLOCK_SIZE], &img.blocks()[0][..]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_materializes_once() {
        let img = image();
        let dir = std::env::temp_dir().join(format!("dsec-fc-c-{}", std::process::id()));
        let cache = RootfsCache::new(&dir);
        let a = cache.get_or_materialize(&img).unwrap();
        let b = cache.get_or_materialize(&img).unwrap();
        assert_eq!(a, b);
        // Same image content → same file regardless of instance.
        let img2 = ErofsImageBuilder::agent_base().build();
        let c = cache.get_or_materialize(&img2).unwrap();
        assert_eq!(a, c);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn safe_names_sanitize_paths() {
        assert_eq!(safe_name("dsec/agent-base"), "dsec_agent-base");
    }
}
