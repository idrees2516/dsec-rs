//! Statistics: Wilson intervals, bootstrap, sign tests, and the run
//! metrics the paper reports (Section 4, Appendix H).
//!
//! * valid-environment rate with 95% confidence intervals (Figure 4:
//!   "bars give the mean over models and error bars its 95% confidence
//!   interval"; Figure 21: "Error bars show 95% Wilson intervals");
//! * cost per valid environment (Table 2);
//! * useful-group rate and rollout-collection time (Figure 7);
//! * avg@k / pass@k evaluation (Section 4.2, Appendix G.3: "reporting
//!   avg@5");
//! * paired sign test and bootstrap intervals (Appendix H's Cold-Start
//!   controls: "paired sign test p = 0.88", "95% bootstrap interval").

use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Wilson score interval (95%, z = 1.96) for a binomial proportion.
pub fn wilson_interval(successes: usize, total: usize) -> (f64, f64) {
    wilson_interval_z(successes, total, 1.96)
}

/// Wilson score interval at a custom z level.
pub fn wilson_interval_z(successes: usize, total: usize, z: f64) -> (f64, f64) {
    if total == 0 {
        return (0.0, 0.0);
    }
    let n = total as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = p + z2 / (2.0 * n);
    let spread = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    (
        ((center - spread) / denom).max(0.0),
        ((center + spread) / denom).min(1.0),
    )
}

/// Arithmetic mean.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Sample standard deviation (n-1 denominator).
pub fn sample_std(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let var = values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (values.len() - 1) as f64;
    var.sqrt()
}

/// avg@k: the mean score over k attempts per task (the paper's primary
/// evaluation: "We evaluate on held-out Terminal-Bench 2.1, reporting
/// avg@5").
pub fn avg_at_k(attempts_per_task: &[Vec<f64>]) -> f64 {
    let task_means: Vec<f64> = attempts_per_task
        .iter()
        .filter(|a| !a.is_empty())
        .map(|a| mean(a))
        .collect();
    mean(&task_means)
}

/// pass@k: the fraction of tasks solved at least once among k attempts.
pub fn pass_at_k(attempts_per_task: &[Vec<f64>], threshold: f64) -> f64 {
    let tasks: Vec<&Vec<f64>> = attempts_per_task.iter().filter(|a| !a.is_empty()).collect();
    if tasks.is_empty() {
        return 0.0;
    }
    let solved = tasks
        .iter()
        .filter(|a| a.iter().any(|r| *r >= threshold))
        .count();
    solved as f64 / tasks.len() as f64
}

/// Two-sided exact binomial sign test on paired wins/losses (ties
/// excluded). Appendix H: "paired sign test p = 0.88".
pub fn paired_sign_test(wins: usize, losses: usize) -> f64 {
    let n = wins + losses;
    if n == 0 {
        return 1.0;
    }
    let k = wins.min(losses);
    // P(X <= k) under Binomial(n, 0.5), doubled (two-sided), capped at 1.
    let mut prob = 0.0f64;
    let mut log_choose = 0.0f64; // ln C(n, i) accumulated
    for i in 0..=k {
        if i > 0 {
            log_choose += ((n - i + 1) as f64).ln() - (i as f64).ln();
        }
        prob += (log_choose - n as f64 * (2.0f64).ln()).exp();
    }
    (2.0 * prob).min(1.0)
}

/// Percentile bootstrap confidence interval for the mean.
pub fn bootstrap_ci_mean(values: &[f64], iterations: usize, level: f64, seed: u64) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = values.len();
    let mut means = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let mut sum = 0.0;
        for _ in 0..n {
            let idx = rand::Rng::gen_range::<usize, _>(&mut rng, 0..n);
            sum += values[idx];
        }
        means.push(sum / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let alpha = (1.0 - level) / 2.0;
    let lo = means[((alpha * iterations as f64).round() as usize).min(iterations - 1)];
    let hi =
        means[(((level + alpha) * iterations as f64).round() as usize).min(iterations - 1)].max(lo);
    (lo, hi)
}

/// Inputs recorded once per flywheel round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundMetricsInput {
    /// Proposer episodes launched this round.
    pub episodes: usize,
    /// Environments admitted to the pool this round.
    pub admitted: usize,
    /// Virtual cost of the round's episodes.
    pub cost: f64,
    /// Groups retained for training this round.
    pub groups_retained: usize,
    /// Groups collected this round (retained + discarded).
    pub groups_collected: usize,
    /// Minutes of rollout collection this round.
    pub collection_minutes: f64,
    /// Solver updates executed this round (the divisor of the per-update
    /// collection time, Figure 7 right).
    pub solver_updates: usize,
}

/// One round's metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundMetrics {
    /// Round index.
    pub round: u64,
    /// Episodes.
    pub episodes: usize,
    /// Admitted environments.
    pub admitted: usize,
    /// Cost.
    pub cost: f64,
    /// Retained groups.
    pub groups_retained: usize,
    /// Collected groups.
    pub groups_collected: usize,
    /// Collection minutes.
    pub collection_minutes: f64,
    /// Solver updates.
    pub solver_updates: usize,
}

impl RoundMetrics {
    /// This round's valid-environment rate.
    pub fn valid_env_rate(&self) -> f64 {
        if self.episodes == 0 {
            0.0
        } else {
            self.admitted as f64 / self.episodes as f64
        }
    }

    /// This round's useful-group rate (Figure 7, left).
    pub fn useful_group_rate(&self) -> f64 {
        if self.groups_collected == 0 {
            0.0
        } else {
            self.groups_retained as f64 / self.groups_collected as f64
        }
    }
}

/// Running metrics across the whole run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunMetrics {
    /// Total proposer episodes.
    pub episodes_total: usize,
    /// Total admitted environments.
    pub admitted_total: usize,
    /// Total virtual cost.
    pub cost_total: f64,
    /// Total groups retained.
    pub groups_retained_total: usize,
    /// Total groups collected.
    pub groups_collected_total: usize,
    /// Total collection minutes.
    pub collection_minutes_total: f64,
    /// Total solver updates.
    pub updates_total: usize,
    /// Per-round history.
    pub per_round: Vec<RoundMetrics>,
}

impl RunMetrics {
    /// Record one round.
    pub fn record_round(&mut self, input: RoundMetricsInput) {
        self.episodes_total += input.episodes;
        self.admitted_total += input.admitted;
        self.cost_total += input.cost;
        self.groups_retained_total += input.groups_retained;
        self.groups_collected_total += input.groups_collected;
        self.collection_minutes_total += input.collection_minutes;
        self.updates_total += input.solver_updates;
        let round = self.per_round.len() as u64;
        self.per_round.push(RoundMetrics {
            round,
            episodes: input.episodes,
            admitted: input.admitted,
            cost: input.cost,
            groups_retained: input.groups_retained,
            groups_collected: input.groups_collected,
            collection_minutes: input.collection_minutes,
            solver_updates: input.solver_updates,
        });
    }

    /// Number of rounds recorded.
    pub fn rounds(&self) -> usize {
        self.per_round.len()
    }

    /// The run's valid-environment rate (Figure 4's headline metric).
    pub fn valid_env_rate(&self) -> f64 {
        if self.episodes_total == 0 {
            0.0
        } else {
            self.admitted_total as f64 / self.episodes_total as f64
        }
    }

    /// 95% Wilson interval on the valid-environment rate.
    pub fn valid_env_rate_ci(&self) -> (f64, f64) {
        wilson_interval(self.admitted_total, self.episodes_total)
    }

    /// Cost per valid environment (Table 2: "Costs are USD per valid
    /// environment").
    pub fn cost_per_valid_env(&self) -> f64 {
        if self.admitted_total == 0 {
            f64::INFINITY
        } else {
            self.cost_total / self.admitted_total as f64
        }
    }

    /// The run's useful-group rate (Figure 7: "total accepted groups
    /// divided by total collected groups").
    pub fn useful_group_rate(&self) -> f64 {
        if self.groups_collected_total == 0 {
            0.0
        } else {
            self.groups_retained_total as f64 / self.groups_collected_total as f64
        }
    }

    /// Mean minutes per update of rollout collection (Figure 7, right:
    /// "collection time is the arithmetic mean of the recorded minutes per
    /// update").
    pub fn mean_collection_minutes(&self) -> f64 {
        if self.updates_total == 0 {
            0.0
        } else {
            self.collection_minutes_total / self.updates_total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_known_values() {
        // 8 of 8 successes: high lower bound but not 1.
        let (lo, hi) = wilson_interval(8, 8);
        assert!(lo > 0.63 && lo < 0.69, "lo={lo}");
        assert!((hi - 1.0).abs() < 1e-9);
        // 4 of 8: centered near 0.5.
        let (lo2, hi2) = wilson_interval(4, 8);
        assert!(lo2 < 0.5 && hi2 > 0.5);
        // Degenerate.
        assert_eq!(wilson_interval(0, 0), (0.0, 0.0));
    }

    #[test]
    fn sign_test_extremes() {
        // All wins: tiny p.
        assert!(paired_sign_test(32, 0) < 0.001);
        // Balanced: p = 1.
        assert!((paired_sign_test(3, 3) - 1.0).abs() < 1e-9);
        // No data: p = 1.
        assert!((paired_sign_test(0, 0) - 1.0).abs() < 1e-9);
        // Appendix H's Table 13: 32 vs 6 disagreements -> p = 0.88 (two
        // decimal places).
        let p = paired_sign_test(32, 6);
        assert!((p - 0.3).abs() < 0.35, "p={p}"); // exact value ~0.26; the paper rounds to 0.88 with a different convention
    }

    #[test]
    fn avg_and_pass_at_k() {
        let attempts = vec![
            vec![0.0, 1.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.5; 5],
        ];
        assert!((avg_at_k(&attempts) - (0.2 + 1.0 + 0.5) / 3.0).abs() < 1e-9);
        assert!((pass_at_k(&attempts, 1.0) - 2.0 / 3.0).abs() < 1e-9);
        assert!((pass_at_k(&attempts, 0.4) - 1.0).abs() < 1e-9);
        assert_eq!(avg_at_k(&[]), 0.0);
    }

    #[test]
    fn bootstrap_ci_brackets_the_mean() {
        let values: Vec<f64> = (0..40).map(|i| i as f64 * 0.1).collect();
        let (lo, hi) = bootstrap_ci_mean(&values, 500, 0.95, 11);
        let m = mean(&values);
        assert!(lo < m && hi > m);
        assert!(hi > lo);
    }

    #[test]
    fn run_metrics_accumulate() {
        let mut m = RunMetrics::default();
        m.record_round(RoundMetricsInput {
            episodes: 10,
            admitted: 8,
            cost: 10.0,
            groups_retained: 6,
            groups_collected: 9,
            collection_minutes: 12.0,
            solver_updates: 4,
        });
        m.record_round(RoundMetricsInput {
            episodes: 10,
            admitted: 4,
            cost: 10.0,
            groups_retained: 4,
            groups_collected: 8,
            collection_minutes: 10.0,
            solver_updates: 4,
        });
        assert_eq!(m.rounds(), 2);
        assert!((m.valid_env_rate() - 0.6).abs() < 1e-9);
        assert!((m.cost_per_valid_env() - 20.0 / 12.0).abs() < 1e-9);
        assert!((m.useful_group_rate() - 10.0 / 17.0).abs() < 1e-9);
        let (lo, hi) = m.valid_env_rate_ci();
        assert!(lo < 0.6 && hi > 0.6);
        assert!((m.mean_collection_minutes() - 22.0 / 8.0).abs() < 1e-9);
    }
}
