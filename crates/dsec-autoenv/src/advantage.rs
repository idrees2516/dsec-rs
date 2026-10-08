//! Role-separated group advantages, DAPO filtering, and batch filling
//! (Section 3.3).
//!
//! "For role q ∈ {P, S}, let τ_{q,1}, ..., τ_{q,G_q} be a group of
//! trajectories with rewards r_{q,i} = r_q(τ_{q,i}). The advantage of
//! τ_{q,i} is
//!
//! ```text
//! Â_{q,i} = r_{q,i} − (1/G_q) Σ_j r_{q,j}.        (Eq. 3)
//! ```
//!
//! "Following previous self-play work, we compute advantage baselines
//! separately for the two roles. Proposer rewards are centered within
//! proposer groups, while solver rewards are centered across attempts on
//! the same environment. Following DAPO, we drop groups whose rewards do
//! not vary. The solver trains at every step. Every k solver steps, the
//! proposer generates new environments and trains on the resulting proposer
//! trajectories." (Appendix G.1: "Rewards are mean-centered without division
//! by the group standard deviation.")

use crate::trajectory::{Role, TrajectoryGroup};

/// Eq. 3: group-relative, mean-centered advantages — *without* division by
/// the group standard deviation.
pub fn group_advantage(rewards: &[f64]) -> Vec<f64> {
    if rewards.is_empty() {
        return Vec::new();
    }
    let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
    rewards.iter().map(|r| r - mean).collect()
}

/// DAPO filter: drop groups whose rewards do not vary.
///
/// Returns `(kept, dropped)`. A group with zero reward variance carries no
/// group-relative learning signal ("`[0.5, 0.5]` ... has zero GRPO
/// signal").
pub fn dapo_filter(groups: Vec<TrajectoryGroup>) -> (Vec<TrajectoryGroup>, Vec<TrajectoryGroup>) {
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for g in groups {
        if g.rewards_vary() {
            kept.push(g);
        } else {
            dropped.push(g);
        }
    }
    (kept, dropped)
}

/// Compute advantages for one group; the DPPO wiring attaches them to
/// token steps downstream. Returns per-trajectory advantages aligned with
/// the group's rewards.
pub fn group_advantages(group: &TrajectoryGroup) -> Vec<f64> {
    group_advantage(&group.rewards)
}

/// Fill a training batch by oversampling.
///
/// Appendix G.1: "We oversample and discard groups with no reward
/// variation, collecting additional groups until the batch is filled."
/// `sampler` is invoked repeatedly; each invocation yields newly collected
/// groups (already DAPO-filtered by the caller or here). Filling stops when
/// `need` groups are retained or `max_rounds` sampler invocations have been
/// attempted (a safety valve so adversarial samplers cannot loop forever).
///
/// Returns the retained groups and how many were collected-then-discarded
/// (the denominator of the useful-group rate, Figure 7: "groups retained
/// for training as a percentage of all groups collected").
pub fn fill_batch(
    need: usize,
    max_rounds: usize,
    mut sampler: impl FnMut() -> Vec<TrajectoryGroup>,
) -> (Vec<TrajectoryGroup>, usize) {
    let mut retained: Vec<TrajectoryGroup> = Vec::new();
    let mut discarded = 0usize;
    let mut rounds = 0usize;
    while retained.len() < need && rounds < max_rounds {
        rounds += 1;
        let batch = sampler();
        if batch.is_empty() {
            break;
        }
        let (kept, dropped) = dapo_filter(batch);
        discarded += dropped.len();
        retained.extend(kept);
    }
    if retained.len() > need {
        let overflow = retained.split_off(need);
        discarded += overflow.len();
    }
    (retained, discarded)
}

/// Rate of groups retained for training out of all groups collected
/// (Figure 7, left).
pub fn useful_group_rate(retained: usize, collected: usize) -> f64 {
    if collected == 0 {
        0.0
    } else {
        retained as f64 / collected as f64
    }
}

/// The role separation: solver groups are centered across attempts on the
/// same environment; proposer groups across trajectories sharing one
/// generation assignment. Both use the same mean-centering; the *keys* of
/// the groups differ, which is the entire separation.
#[derive(Debug, Clone, PartialEq)]
pub struct RoleBatch {
    /// Which role this batch trains.
    pub role: Role,
    /// The retained groups.
    pub groups: Vec<TrajectoryGroup>,
    /// Advantages per group, aligned with each group's rewards.
    pub advantages: Vec<Vec<f64>>,
}

/// Build the advantage set for a role's retained groups.
pub fn role_batch(role: Role, groups: Vec<TrajectoryGroup>) -> RoleBatch {
    let advantages = groups.iter().map(group_advantages).collect();
    RoleBatch {
        role,
        groups,
        advantages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{Outcome, Trajectory};

    fn solver_group(key: &str, rewards: &[f64]) -> TrajectoryGroup {
        let trajs: Vec<Trajectory> = rewards
            .iter()
            .map(|r| Trajectory {
                role: Role::Solver,
                prompt: format!("solve {key}"),
                turns: vec![],
                outcome: Some(Outcome {
                    reward: *r,
                    verifier_stdout: String::new(),
                    passed_tests: 0,
                    total_tests: 4,
                    failed_checks: vec![],
                }),
            })
            .collect();
        TrajectoryGroup::new(Role::Solver, key, trajs)
    }

    #[test]
    fn eq3_mean_centered_no_std_division() {
        let adv = group_advantage(&[0.0, 0.0, 1.0]);
        // Mean = 1/3; advantages = -1/3, -1/3, 2/3 (NOT divided by std).
        assert!((adv[0] + 1.0 / 3.0).abs() < 1e-9);
        assert!((adv[2] - 2.0 / 3.0).abs() < 1e-9);
        assert!((adv.iter().sum::<f64>()).abs() < 1e-9);
    }

    #[test]
    fn dapo_drops_flat_groups() {
        let groups = vec![
            solver_group("env_1", &[0.0, 1.0]),
            solver_group("env_2", &[0.5, 0.5, 0.5]),
            solver_group("env_3", &[1.0, 1.0]),
        ];
        let (kept, dropped) = dapo_filter(groups);
        assert_eq!(kept.len(), 1);
        assert_eq!(dropped.len(), 2);
        assert_eq!(kept[0].group_key, "env_1");
    }

    #[test]
    fn fill_batch_oversamples_until_full() {
        // First sampler round yields 1 varying + 2 flat groups; second
        // round yields 2 varying groups.
        let mut round = 0;
        let (retained, discarded) = fill_batch(3, 10, || {
            round += 1;
            match round {
                1 => vec![
                    solver_group("a", &[0.0, 1.0]),
                    solver_group("b", &[0.5, 0.5]),
                    solver_group("c", &[1.0, 1.0]),
                ],
                2 => vec![
                    solver_group("d", &[0.25, 0.75]),
                    solver_group("e", &[0.0, 0.5]),
                ],
                _ => vec![],
            }
        });
        assert_eq!(retained.len(), 3);
        assert_eq!(discarded, 2);
        assert_eq!(round, 2);
    }

    #[test]
    fn fill_batch_overflow_counts_as_discarded() {
        let (retained, discarded) = fill_batch(1, 5, || {
            vec![
                solver_group("a", &[0.0, 1.0]),
                solver_group("b", &[0.2, 0.8]),
            ]
        });
        assert_eq!(retained.len(), 1);
        assert_eq!(discarded, 1);
    }

    #[test]
    fn useful_group_rate_math() {
        assert!((useful_group_rate(16, 64) - 0.25).abs() < 1e-9);
        assert_eq!(useful_group_rate(0, 0), 0.0);
    }

    #[test]
    fn role_batch_builds_aligned_advantages() {
        let groups = vec![solver_group("env_1", &[0.0, 1.0])];
        let rb = role_batch(Role::Solver, groups);
        assert_eq!(rb.role, Role::Solver);
        assert_eq!(rb.advantages[0].len(), 2);
        assert!((rb.advantages[0][0] + 0.5).abs() < 1e-9);
        assert!((rb.advantages[0][1] - 0.5).abs() < 1e-9);
    }
}
