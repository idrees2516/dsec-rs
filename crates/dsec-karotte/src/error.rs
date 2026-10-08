//! Error taxonomy for the Karotte environment harness.
//!
//! The single distinction that shapes the whole crate, inherited from
//! upstream Karotte: a **student misbehavior** (the RL agent under training
//! attacked the harness — planted a symlink, filled the disk, escaped its
//! process cohort) must be *scored as zero*, while an **infrastructure
//! failure** (the grader broke, the disk filled on its own) must *abort or
//! mask the run*. Confusing the two either trains the policy on a false
//! zero or lets an attack slide — see [`StudentMisbehaviorError`] and the
//! misbehavior scoring path in [`crate::task`].

use thiserror::Error;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Library-level errors.
#[derive(Debug, Error)]
pub enum Error {
    /// A spec/config failed validation.
    #[error("invalid config: {0}")]
    InvalidSpec(String),

    /// A requested task/tool/model id does not exist.
    #[error("not found: {0}")]
    NotFound(String),

    /// The operation is unsupported in this environment (the
    /// [`Contract::Unsupported`](crate::confinement::Contract) case).
    #[error("unsupported on this host: {0}")]
    Unsupported(String),

    /// A confinement mechanism refused or failed.
    #[error("confinement failure: {0}")]
    Confinement(String),

    /// A subprocess exited non-zero.
    #[error("process failed (rc={rc}): {output}")]
    ProcessFailed {
        /// Return code.
        rc: i32,
        /// Captured output (truncated by the caller when huge).
        output: String,
    },

    /// A subprocess exceeded its wall-clock budget.
    #[error("process timed out after {timeout_sec}s: {command}")]
    ProcessTimeout {
        /// Timeout budget, seconds.
        timeout_sec: u64,
        /// Command that timed out.
        command: String,
    },

    /// The agent loop hit one of its limits (turn / time / context /
    /// empty-turn) and the policy says "error".
    #[error("agent loop limit: {0}")]
    LoopLimit(String),

    /// A judge or hook raised [`StudentMisbehaviorError`] — scored 0.
    #[error("student misbehavior: {0}")]
    Misbehavior(#[from] StudentMisbehaviorError),

    /// JSON (de)serialization error.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    /// I/O error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// The error a scoring-time hook or judge raises when the *student* broke
/// the rules. Upstream: `karotte.student_misbehavior.StudentMisbehaviorError`.
///
/// Raising this anywhere inside `Step::score` replaces the scoring with
/// `score = 0.0, continue_task = false` and a `misbehavior` metadata entry —
/// the run *completes* (status `failed`), it does not become an infra error.
#[derive(Debug, Error)]
pub enum StudentMisbehaviorError {
    /// The student planted a symlink where a regular file/dir was required
    /// (submission custody, reclaim, staged copy-back).
    #[error("symlink at {path}")]
    Symlink {
        /// The offending path.
        path: String,
    },

    /// The student planted a special file (fifo/socket/device) where a
    /// regular file was required — classic grader-DoS shape.
    #[error("special file (mode {mode:o}) at {path}")]
    SpecialFile {
        /// The st_mode of the offending entry.
        mode: u32,
        /// The offending path.
        path: String,
    },

    /// A submission or write exceeded a hard size/entry/depth cap.
    #[error("limit exceeded: {what}")]
    LimitExceeded {
        /// Human description of which cap.
        what: String,
    },

    /// A destination error the student can provoke (ENAMETOOLONG, EEXIST,
    /// ENOSPC, EDQUOT, EFBIG) during custody copy.
    #[error("dest error {errno}: {message}")]
    DestError {
        /// The errno name.
        errno: &'static str,
        /// The strerror-style message.
        message: String,
    },

    /// The reclaim sweep could not free everything the student owned, or
    /// hit its deadline — disk state is untrustworthy.
    #[error("reclaim failed: {0}")]
    Reclaim(String),

    /// The student's process cohort could not be reaped before grading;
    /// a live student process could still race the grader.
    #[error("unreapable process cohort: {0}")]
    UnreapableCohort(String),

    /// Free-form misbehavior text (used by ports of specific checks).
    #[error("{0}")]
    Other(String),
}

impl StudentMisbehaviorError {
    /// Convert any `Display` misbehavior into the escaped metadata string
    /// recorded on the scoring event (upstream runs it through
    /// `escape_surrogates` because the message may quote student-chosen,
    /// non-UTF-8-safe filenames).
    pub fn to_metadata(&self) -> String {
        crate::text::escape_surrogates(&self.to_string())
    }
}
