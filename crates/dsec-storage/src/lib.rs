//! # dsec-storage
//!
//! Userspace re-creation of the DSec storage stack:
//!
//! - [`erofs`] — read-only base images (`ErofsImage`) shared across all
//!   sandboxes on a node (the paper's EROFS-on-3FS rootfs), plus an
//!   [`erofs::OnDemandLoader`] that faults blocks in lazily instead of
//!   copying the whole image at sandbox creation.
//! - [`cache`] — node-level LRU block cache. Because cached blocks are
//!   `Arc`-shared, identical blocks are stored exactly once no matter how
//!   many sandboxes read them — the userspace analogue of the paper's
//!   page-cache sharing with virtio-pmem DAX.
//! - [`overlay`] — per-sandbox copy-on-write overlay (OverlayBD-style
//!   layered block device in userspace).
//! - [`imagefs`] — file-level view layered on the overlay: the guest
//!   filesystem Chronus sessions operate on.
//! - [`packdiff`] — incremental dirty-block snapshots (`DiffPack`) used to
//!   replicate sandbox working state without transferring the full image,
//!   the paper's `pack_diff` fast path for agent-loop containers.
//! - [`prefetch`] — staged asynchronous prefetch pipeline (the paper's
//!   data-asynchronous-prefetch image loading) and zero-copy
//!   `SharedRegion`s standing in for torch shared-memory handoff.
//! - [`latency`] — injectable latency model so benchmarks can replay the
//!   paper's timing profile deterministically.

pub mod cache;
pub mod erofs;
pub mod error;
pub mod imagefs;
pub mod latency;
pub mod overlay;
pub mod packdiff;
pub mod prefetch;

/// Logical block size (matches the 4 KiB EROFS/overlay block granularity).
pub const BLOCK_SIZE: usize = 4096;

/// A single block's raw content.
pub type Block = [u8; BLOCK_SIZE];

/// Block identifier (linear block index within an image).
pub type BlockId = u64;

/// Contiguous block range, used by prefetch hints and diff packs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockRange {
    pub start: BlockId,
    pub count: u64,
}

pub use error::{Error, Result};
