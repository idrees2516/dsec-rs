//! Observation and action spaces (pufferlib's flat/mat distinction).
//!
//! PufferLib distinguishes *flat* observations (one vector per env) from
//! *mat* observations (a matrix per env, e.g. token or feature
//! sequences). Storage is always contiguous f32; the space record tells
//! the buffer and training code how to interpret it.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObsSpace {
    /// `obs_size` floats per env.
    Flat { size: usize },
    /// `rows x cols` floats per env (row-major).
    Mat { rows: usize, cols: usize },
}

impl ObsSpace {
    pub fn size(&self) -> usize {
        match self {
            ObsSpace::Flat { size } => *size,
            ObsSpace::Mat { rows, cols } => rows * cols,
        }
    }

    pub fn mat_shape(&self) -> Option<(usize, usize)> {
        match self {
            ObsSpace::Mat { rows, cols } => Some((*rows, *cols)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActionSpace {
    Discrete { n: usize },
    Box { low: Vec<f32>, high: Vec<f32> },
}

impl ActionSpace {
    pub fn discrete(n: usize) -> Self {
        ActionSpace::Discrete { n }
    }

    pub fn size(&self) -> usize {
        match self {
            ActionSpace::Discrete { .. } => 1,
            ActionSpace::Box { low, .. } => low.len(),
        }
    }

    pub fn is_discrete(&self) -> bool {
        matches!(self, ActionSpace::Discrete { .. })
    }
}

/// One action from the policy.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Discrete(i64),
    Box(Vec<f32>),
}

impl Action {
    pub fn as_index(&self) -> i64 {
        match self {
            Action::Discrete(i) => *i,
            Action::Box(_) => -1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(ObsSpace::Flat { size: 32 }.size(), 32);
        assert_eq!(ObsSpace::Mat { rows: 4, cols: 8 }.size(), 32);
        assert_eq!(ObsSpace::Mat { rows: 4, cols: 8 }.mat_shape(), Some((4, 8)));
        assert_eq!(ObsSpace::Flat { size: 32 }.mat_shape(), None);
        assert_eq!(ActionSpace::discrete(6).size(), 1);
        assert_eq!(
            ActionSpace::Box {
                low: vec![0.0; 3],
                high: vec![1.0; 3]
            }
            .size(),
            3
        );
    }
}
