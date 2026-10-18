//! Policy versions, the policy store, and the publication barrier.
//!
//! The Phase 5/6 contract the whole crate hangs on:
//!
//! - every set of trainable weights is a **policy version**
//!   `(major, digest)` — the digest is the SHA-256 of the serialized
//!   parameters, so two identical models are the same version;
//! - exactly one version is **published** at any time; rollouts stamp
//!   the version they served (the model server reports it), and the
//!   trainer rejects samples stamped with versions older than
//!   `published - staleness_bound`;
//! - publication goes through a **barrier**: a training step stages
//!   the new weights, the barrier closes, the published pointer flips,
//!   the barrier opens. A crash mid-publication (the Phase 6
//!   crashpoint) leaves the barrier closed with the OLD version still
//!   published — rollouts never see a half-published policy.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A policy version.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PolicyVersion {
    /// Monotonic major (training step) number.
    pub major: u64,
    /// SHA-256 of the serialized parameters.
    pub digest: String,
}

impl PolicyVersion {
    /// The model name the policy server reports for this version.
    pub fn model_name(&self) -> String {
        format!("policy-v{}", self.major)
    }
}

/// The publication barrier.
#[derive(Debug, Default)]
pub struct PublicationBarrier {
    closed: AtomicBool,
}

impl PublicationBarrier {
    /// Whether publication is in progress (new weights staged but not
    /// yet visible).
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    fn open(&self) {
        self.closed.store(false, Ordering::SeqCst);
    }
}

/// A versioned weight checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// The version this checkpoint stages.
    pub version: PolicyVersion,
    /// Serialized model parameters.
    pub weights: Vec<u8>,
}

/// The shared policy store: version registry + published pointer +
/// publication barrier. Clones share state (Arc inside), so the model
/// server and the trainer observe the same publication timeline.
#[derive(Clone)]
pub struct PolicyStore {
    inner: Arc<Mutex<StoreInner>>,
    barrier: Arc<PublicationBarrier>,
}

#[derive(Debug)]
struct StoreInner {
    versions: BTreeMap<u64, PolicyVersion>,
    weights: BTreeMap<u64, Vec<u8>>,
    published: PolicyVersion,
    /// A staged-but-unpublished checkpoint (publication interrupted).
    staged: Option<Checkpoint>,
}

impl PolicyStore {
    /// A store seeded with the initial (version 0) weights.
    pub fn new(version0: PolicyVersion, weights: Vec<u8>) -> Self {
        let mut versions = BTreeMap::new();
        versions.insert(0, version0.clone());
        let mut weight_map: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
        weight_map.insert(0, weights);
        Self {
            inner: Arc::new(Mutex::new(StoreInner {
                versions,
                weights: weight_map,
                published: version0,
                staged: None,
            })),
            barrier: Arc::new(PublicationBarrier::default()),
        }
    }

    /// The currently published version (what rollouts must serve).
    pub fn published(&self) -> PolicyVersion {
        self.inner.lock().expect("policy store").published.clone()
    }

    /// The weights of the published version.
    pub fn published_weights(&self) -> Vec<u8> {
        let guard = self.inner.lock().expect("policy store");
        guard
            .weights
            .get(&guard.published.major)
            .cloned()
            .unwrap_or_default()
    }

    /// Whether the publication barrier is closed (staging in flight).
    pub fn barrier_closed(&self) -> bool {
        self.barrier.is_closed()
    }

    /// A staged-but-unpublished checkpoint, if a publication was
    /// interrupted.
    pub fn staged(&self) -> Option<Checkpoint> {
        self.inner.lock().expect("policy store").staged.clone()
    }

    /// Registers a completed training step's weights under
    /// `version.major` (not yet visible to rollouts).
    pub fn stage(&self, version: PolicyVersion, weights: Vec<u8>) -> Result<()> {
        let mut guard = self.inner.lock().expect("policy store");
        if self.barrier.is_closed() {
            return Err(Error::Policy(
                "publication already in progress (barrier closed)".into(),
            ));
        }
        guard.versions.insert(version.major, version.clone());
        guard.weights.insert(version.major, weights);
        guard.staged = Some(Checkpoint {
            version,
            weights: Vec::new(),
        });
        // the actual weights live in `weights`; the checkpoint marks
        // the staging intent
        Ok(())
    }

    /// Publishes the staged version atomically: barrier closes →
    /// pointer flips → barrier opens.
    ///
    /// `crash_before_flip` is the Phase 6 crashpoint: staging happened,
    /// the pointer never moved, the barrier stays CLOSED. Recovery
    /// ([`PolicyStore::recover_publication`]) then discards or
    /// completes.
    pub fn publish(&self, crash_before_flip: bool) -> Result<PolicyVersion> {
        let mut guard = self.inner.lock().expect("policy store");
        let staged = guard
            .staged
            .clone()
            .ok_or_else(|| Error::Policy("publish called without a staged checkpoint".into()))?;
        self.barrier.close();
        if crash_before_flip {
            // the crash fires: barrier left closed, old version still
            // published (weights exist on disk but are not served)
            return Err(Error::Policy(
                "CRASH: publication interrupted before flip".into(),
            ));
        }
        guard.published = staged.version;
        guard.staged = None;
        self.barrier.open();
        Ok(guard.published.clone())
    }

    /// Recovery for an interrupted publication. `complete` decides
    /// whether to finish the flip (the staged weights are valid) or
    /// discard the staged checkpoint (retrain). Either way the barrier
    /// re-opens — a stuck barrier would freeze all publication.
    pub fn recover_publication(&self, complete: bool) -> Result<PolicyVersion> {
        let mut guard = self.inner.lock().expect("policy store");
        let staged = match guard.staged.clone() {
            Some(s) => s,
            None => {
                self.barrier.open();
                return Ok(guard.published.clone());
            }
        };
        if complete {
            guard.published = staged.version;
        }
        guard.staged = None;
        self.barrier.open();
        Ok(guard.published.clone())
    }

    /// All known versions (for tests/diagnostics).
    pub fn versions(&self) -> Vec<PolicyVersion> {
        self.inner
            .lock()
            .expect("policy store")
            .versions
            .values()
            .cloned()
            .collect()
    }
}
