//! Sandbox lifecycle state machine (Edge-managed).
//!
//! ```text
//!            create()              pause()
//!  Creating ────────► Ready ◄──────────────► Paused
//!     │                 │  └──────resume()────┘
//!     │ fail            │ destroy()
//!     ▼                 ▼
//!   Failed ────────► Destroying ────────► Destroyed
//! ```
//!
//! Transitions are guarded by compare-and-swap on an atomic, so concurrent
//! lifecycle calls (e.g. pause racing destroy) resolve deterministically:
//! exactly one caller wins each transition.

use std::sync::atomic::{AtomicU8, Ordering};

use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SandboxState {
    Creating = 0,
    Ready = 1,
    Paused = 2,
    Destroying = 3,
    Destroyed = 4,
    Failed = 5,
}

impl SandboxState {
    pub fn as_str(self) -> &'static str {
        match self {
            SandboxState::Creating => "creating",
            SandboxState::Ready => "ready",
            SandboxState::Paused => "paused",
            SandboxState::Destroying => "destroying",
            SandboxState::Destroyed => "destroyed",
            SandboxState::Failed => "failed",
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(SandboxState::Creating),
            1 => Some(SandboxState::Ready),
            2 => Some(SandboxState::Paused),
            3 => Some(SandboxState::Destroying),
            4 => Some(SandboxState::Destroyed),
            5 => Some(SandboxState::Failed),
            _ => None,
        }
    }

    /// Whether data-plane requests (exec/fs/http) are accepted.
    pub fn accepts_traffic(self) -> bool {
        matches!(self, SandboxState::Ready)
    }
}

/// CAS-guarded state cell.
#[derive(Debug, Default)]
pub struct StateCell {
    inner: AtomicU8,
}

impl StateCell {
    /// Starts in `Creating`.
    pub fn creating() -> Self {
        StateCell {
            inner: AtomicU8::new(SandboxState::Creating as u8),
        }
    }

    pub fn get(&self) -> SandboxState {
        SandboxState::from_u8(self.inner.load(Ordering::Acquire)).unwrap_or(SandboxState::Failed)
    }

    /// Attempts a transition; returns the previous state on failure.
    pub fn transition(&self, sid: u64, to: SandboxState) -> crate::Result<()> {
        let allowed: &[(SandboxState, SandboxState)] = &[
            (SandboxState::Creating, SandboxState::Ready),
            (SandboxState::Creating, SandboxState::Failed),
            (SandboxState::Ready, SandboxState::Paused),
            (SandboxState::Ready, SandboxState::Destroying),
            (SandboxState::Paused, SandboxState::Ready),
            (SandboxState::Paused, SandboxState::Destroying),
            (SandboxState::Destroying, SandboxState::Destroyed),
        ];
        let mut cur = self.inner.load(Ordering::Acquire);
        loop {
            let cur_state = SandboxState::from_u8(cur).unwrap_or(SandboxState::Failed);
            if !allowed.contains(&(cur_state, to)) {
                return Err(Error::InvalidTransition {
                    sid,
                    from: cur_state.as_str(),
                    to: to.as_str(),
                });
            }
            match self
                .inner
                .compare_exchange(cur, to as u8, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return Ok(()),
                Err(actual) => cur = actual,
            }
        }
    }

    /// Unconditional move used by teardown paths (`Destroyed`/`Failed`).
    pub fn force(&self, to: SandboxState) {
        self.inner.store(to as u8, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_lifecycle() {
        let c = StateCell::creating();
        assert_eq!(c.get(), SandboxState::Creating);
        c.transition(1, SandboxState::Ready).unwrap();
        c.transition(1, SandboxState::Paused).unwrap();
        c.transition(1, SandboxState::Ready).unwrap();
        c.transition(1, SandboxState::Destroying).unwrap();
        c.transition(1, SandboxState::Destroyed).unwrap();
    }

    #[test]
    fn invalid_transitions_rejected() {
        let c = StateCell::creating();
        assert!(c.transition(1, SandboxState::Paused).is_err());
        c.transition(1, SandboxState::Ready).unwrap();
        assert!(c.transition(1, SandboxState::Ready).is_err()); // self-loop
        assert!(c.transition(1, SandboxState::Failed).is_err());
    }

    #[test]
    fn concurrent_pause_destroy_race_resolves() {
        let c = std::sync::Arc::new(StateCell::creating());
        c.transition(0, SandboxState::Ready).unwrap();
        let a = c.clone();
        let b = c.clone();
        let h1 = std::thread::spawn(move || a.transition(0, SandboxState::Paused));
        let h2 = std::thread::spawn(move || b.transition(0, SandboxState::Destroying));
        let r1 = h1.join().unwrap();
        let r2 = h2.join().unwrap();
        // At least one wins (both may: pause -> destroy is a legal chain).
        assert!(!(r1.is_err() && r2.is_err()));
        let st = c.get();
        assert!(st == SandboxState::Paused || st == SandboxState::Destroying);
    }

    #[test]
    fn accepts_traffic_only_when_ready() {
        let c = StateCell::creating();
        assert!(!c.get().accepts_traffic());
        c.transition(0, SandboxState::Ready).unwrap();
        assert!(c.get().accepts_traffic());
        c.transition(0, SandboxState::Paused).unwrap();
        assert!(!c.get().accepts_traffic());
    }
}
