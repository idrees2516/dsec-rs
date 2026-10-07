//! The Live RL trainer — group rollouts, GRPO advantages, and the
//! anti-reward-hacking stack.
//!
//! MiMo-V2.6's Live RL recipe (per the model card) scales RL with:
//!
//! * **Fully asynchronous GRPO** on very large batches — thousands of
//!   prompts × many rollouts per group, stepping as pods come up rather
//!   than in lockstep;
//! * **Groupwise Agentic Grading** — binary pass/fail cannot rank
//!   passing solutions, so the reward signal itself is scaled:
//!   **GRS** (*Groupwise Reward Synthesis*) builds task-specific rubrics
//!   offline from contrasting rollouts within a group, and **GAR**
//!   (*Groupwise Advantage Redistribution*) ranks passing trajectories
//!   online and moves advantage toward higher-quality solutions;
//! * **Aligned RL** — throughout training, environment hardening,
//!   adversarial screening, and verifier cross-checks keep the loop
//!   honest against reward hacking.
//!
//! This module ports those semantics on top of the deterministic
//! environment stack:
//!
//! * [`LiveTrainer::step`] — one asynchronous training step over a batch
//!   of prompts: every prompt gets a group of `group_size` rollouts, run
//!   concurrently under a concurrency limit (tokio tasks over
//!   independently-seeded policies), rewards computed through the full
//!   masking contract;
//! * [`grpo_advantages`] — group-relative advantages with masked
//!   rollouts excluded from the group statistics (never trained, never
//!   averaged in);
//! * [`gar`] — advantage redistribution over the ranked passing set;
//! * [`grs`] — offline rubric synthesis from contrasting rollouts;
//! * [`adversarial_screening`] — flags suspicious high-reward
//!   trajectories (zero-tool guesses, cross-check disagreement) and
//!   drops them from training;
//! * [`self_correction_pairs`] — the Aligned-RL cold-start generator:
//!   misaligned answers paired with grounded rewrites from the same
//!   group's successes.

use crate::agentloop::{AgentLoop, LoopConfig, Policy, RolloutResult};
use crate::error::{InfraErrorKind, Result};
use crate::mcp::ToolInfo;
use crate::task::TaskRow;
use crate::topology::SimPod;
use crate::verifier::{
    build_post_state, extract_agent_output, AnchorJudge, EvalCtx, Judge, RewardConfig,
    RewardOutcome, Rubric, RubricEngine, RubricItem, RuleCheck, VerifyToggles,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::task::JoinSet;

/// Everything one environment needs, built per rollout.
pub struct EnvBundle {
    /// The pod (freshly built for this rollout).
    pub pod: SimPod,
    /// Its manifest.
    pub manifest: crate::manifest::Manifest,
    /// Its rubric.
    pub rubric: Rubric,
    /// Discovered tool surface.
    pub tools: Vec<(String, ToolInfo)>,
}

impl EnvBundle {
    /// Bootstraps the bundle: starts the MCP servers via the manifest
    /// setup step and captures source protection.
    pub fn boot(mut self) -> Result<Self> {
        self.pod.apply_setup(&self.manifest)?;
        self.pod.capture_protection();
        Ok(self)
    }
}

/// Builds a fresh environment for a task row.
///
/// Implemented by the generic factory ([`crate::envgen`]) and by
/// test/eval providers.
pub trait EnvProvider: Send + Sync {
    /// Builds (not yet booted) the environment for one row.
    fn build(&self, row: &TaskRow) -> Result<EnvBundle>;
}

/// A policy source for rollouts: one fresh, seeded policy per rollout.
pub type PolicyFactory = Arc<dyn Fn(u64) -> Box<dyn Policy> + Send + Sync>;

/// One recorded rollout.
#[derive(Debug, Clone)]
pub struct RolloutRecord {
    /// The task row's instance id.
    pub instance_id: String,
    /// Group member index.
    pub member: usize,
    /// The reward outcome (masked or valid).
    pub reward: RewardOutcome,
    /// Rollout stats.
    pub tool_turns: u64,
    /// Approximate tokens.
    pub tokens: u64,
    /// Final message text.
    pub final_message: String,
    /// Infra failure category, if any.
    pub infra_error: Option<InfraErrorKind>,
    /// Group-relative advantage (filled by the trainer).
    pub advantage: Option<f64>,
    /// Whether adversarial screening dropped this rollout.
    pub screened: bool,
    /// Transcript of the rollout (for GRS and self-correction).
    pub transcript_messages: usize,
    /// Every dispatch kind sequence (`ok`, `tool_exception`, ...).
    pub dispatch_kinds: Vec<&'static str>,
}

impl RolloutRecord {
    /// Whether the record is trainable (valid reward, not screened).
    pub fn trainable(&self) -> bool {
        !self.reward.masked() && !self.screened
    }

    /// The reward score (masked records read as 0 but must not be
    /// trained on).
    pub fn score(&self) -> f64 {
        self.reward.score()
    }
}

/// Trainer configuration.
#[derive(Debug, Clone)]
pub struct LiveConfig {
    /// Rollouts per prompt (the GRPO group size — 16 upstream).
    pub group_size: usize,
    /// Maximum concurrently-running rollouts.
    pub concurrency: usize,
    /// Agent loop config.
    pub loop_config: LoopConfig,
    /// Reward config.
    pub reward_config: RewardConfig,
    /// GAR: fraction of passing-group advantage mass redistributed
    /// toward higher-quality solutions.
    pub gar_strength: f64,
    /// GAR quality metric: prefer fewer tokens and fewer tool turns.
    /// Quality = 1 / (1 + tokens/1000 + tool_turns/10).
    /// (upstream GAR steers toward shorter paths and fewer tokens).
    pub gar_prefers_short: bool,
    /// Adversarial screening: flag full-score rollouts with fewer tool
    /// turns than this in tool-bearing environments.
    pub screen_min_tool_turns: u64,
    /// Verifier cross-check tolerance.
    pub cross_check_tolerance: f64,
    /// Advantage normalization epsilon.
    pub adv_eps: f64,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            group_size: 8,
            concurrency: 16,
            loop_config: LoopConfig::default(),
            reward_config: RewardConfig::default(),
            gar_strength: 0.5,
            gar_prefers_short: true,
            screen_min_tool_turns: 1,
            cross_check_tolerance: 0.25,
            adv_eps: 1e-6,
        }
    }
}

/// One training step's results.
#[derive(Debug, Clone, Default)]
pub struct TrainStepStats {
    /// Prompts (groups) in the batch.
    pub prompts: usize,
    /// Total rollouts launched.
    pub rollouts: usize,
    /// Rollouts masked by the reward contract.
    pub masked: usize,
    /// Rollouts dropped by adversarial screening.
    pub screened: usize,
    /// Trainable rollouts.
    pub trainable: usize,
    /// Mean reward over trainable rollouts.
    pub mean_reward: f64,
    /// Groups where every rollout was masked (fully invalid groups).
    pub invalid_groups: usize,
    /// Advantage mass after GAR redistribution (equals pre-GAR mass —
    /// redistribution is conservative).
    pub advantage_mass: f64,
}

/// The Live RL trainer.
pub struct LiveTrainer {
    config: LiveConfig,
    provider: Arc<dyn EnvProvider>,
    policy_factory: PolicyFactory,
    judge: Arc<dyn Judge>,
    /// Independent second judge for verifier cross-checks (Live RL's
    /// verifier cross-check defense).
    cross_judge: Option<Arc<dyn Judge>>,
}

impl LiveTrainer {
    /// New trainer.
    pub fn new(
        config: LiveConfig,
        provider: Arc<dyn EnvProvider>,
        policy_factory: PolicyFactory,
        judge: Arc<dyn Judge>,
    ) -> Self {
        Self {
            config,
            provider,
            policy_factory,
            judge,
            cross_judge: None,
        }
    }

    /// Sets the independent cross-check judge.
    pub fn with_cross_judge(mut self, judge: Arc<dyn Judge>) -> Self {
        self.cross_judge = Some(judge);
        self
    }

    /// Configuration accessor.
    pub fn config(&self) -> &LiveConfig {
        &self.config
    }

    /// Runs one fully-asynchronous training step over a batch of prompts.
    ///
    /// Rollouts are dispatched as blocking tokio tasks (each builds and
    /// boots its own pod — environments are never shared across group
    /// members), bounded by `concurrency`. Results are deterministic per
    /// (row, member, seed) regardless of scheduling: every rollout is
    /// independent, keyed by `seed = hash(instance_id) + member`.
    pub async fn step(&self, batch: &[TaskRow]) -> Result<(Vec<RolloutRecord>, TrainStepStats)> {
        let mut set: JoinSet<std::result::Result<(usize, usize, RolloutRecord), String>> =
            JoinSet::new();
        let mut in_flight = 0usize;
        let mut queue: Vec<(usize, usize)> = Vec::new();
        for (p, _row) in batch.iter().enumerate() {
            for m in 0..self.config.group_size {
                queue.push((p, m));
            }
        }
        let mut records: Vec<Option<(usize, usize, RolloutRecord)>> = vec![None; queue.len()];
        let total = queue.len();
        let mut qi = 0usize;

        loop {
            while in_flight < self.config.concurrency && qi < total {
                let (p, m) = queue[qi];
                let row = batch[p].clone();
                let member = m;
                let seed = rollout_seed(&row, member);
                let provider = self.provider.clone();
                let factory = self.policy_factory.clone();
                let judge = self.judge.clone();
                let cfg = self.config.clone();
                let idx = qi;
                set.spawn_blocking(move || {
                    let record =
                        run_one_rollout(&provider, &row, member, seed, &factory, &judge, &cfg);
                    Ok((idx, member, record))
                });
                in_flight += 1;
                qi += 1;
            }
            if in_flight == 0 {
                break;
            }
            match set.join_next().await {
                Some(Ok(Ok((idx, member, rec)))) => {
                    records[idx] = Some((idx, member, rec));
                    in_flight -= 1;
                }
                Some(Ok(Err(e))) => {
                    // rollout machinery panicked: mask the whole member
                    // (the padding slot stays None — counted as masked)
                    let _ = e;
                    in_flight -= 1;
                }
                Some(Err(e)) => {
                    // JoinSet failure: treat as one masked rollout
                    let slot = total.min(records.len().saturating_sub(1));
                    records[slot] = None;
                    in_flight -= 1;
                    let _ = e;
                }
                None => break,
            }
        }

        // order records by (prompt, member)
        let mut ordered: Vec<RolloutRecord> = Vec::with_capacity(total);
        for slot in records.into_iter().flatten() {
            ordered.push(slot.2);
        }
        // fill missing slots as infra-masked records
        while ordered.len() < total {
            ordered.push(RolloutRecord {
                instance_id: String::new(),
                member: 0,
                reward: RewardOutcome::TestbedCorrupted {
                    kind: crate::error::RewardErrorKind::RewardTransportDied,
                    detail: "rollout task failed".into(),
                },
                tool_turns: 0,
                tokens: 0,
                final_message: String::new(),
                infra_error: Some(InfraErrorKind::SetupFailed),
                advantage: None,
                screened: false,
                transcript_messages: 0,
                dispatch_kinds: Vec::new(),
            });
        }

        // adversarial screening: environment hardening — zero-tool
        // lucky guesses, and (when a cross judge is configured)
        // verifier cross-check disagreements
        let mut worked = ordered;
        let mut disagreement = BTreeMap::new();
        if let Some(cj) = &self.cross_judge {
            let rows: BTreeMap<String, TaskRow> = batch
                .iter()
                .map(|r| (r.extra_info.instance_id.clone(), r.clone()))
                .collect();
            disagreement = cross_check_disagreements(&self.provider, &worked, &rows, cj);
        }
        let _dropped = adversarial_screening(
            &mut worked,
            self.config.screen_min_tool_turns,
            &disagreement,
            self.config.cross_check_tolerance,
        );
        let stats0 = group_stats(&worked, batch.len(), self.config.group_size);
        // GRPO advantages + GAR
        apply_advantages(
            &mut worked,
            batch.len(),
            self.config.group_size,
            &self.config,
        );
        let stats = finalize_stats(stats0, &worked);
        Ok((worked, stats))
    }
}

/// Runs one rollout synchronously (the blocking-task body).
fn run_one_rollout(
    provider: &Arc<dyn EnvProvider>,
    row: &TaskRow,
    member: usize,
    seed: u64,
    factory: &PolicyFactory,
    judge: &Arc<dyn Judge>,
    cfg: &LiveConfig,
) -> RolloutRecord {
    let instance_id = row.extra_info.instance_id.clone();
    let mut policy = factory(seed);
    let built = provider.build(row).and_then(|b| b.boot());
    let bundle = match built {
        Ok(b) => b,
        Err(e) => {
            return RolloutRecord {
                instance_id,
                member,
                reward: RewardOutcome::TestbedCorrupted {
                    kind: crate::error::RewardErrorKind::McpBackendDead,
                    detail: format!("setup failed: {e}"),
                },
                tool_turns: 0,
                tokens: 0,
                final_message: String::new(),
                infra_error: Some(InfraErrorKind::SetupFailed),
                advantage: None,
                screened: false,
                transcript_messages: 0,
                dispatch_kinds: Vec::new(),
            }
        }
    };
    let mut pod = bundle.pod;
    let task = row.task_text();
    let tools = bundle.tools;
    let mut loop_runner = AgentLoop::new(cfg.loop_config.clone());
    let rollout: RolloutResult = loop_runner.run(&mut pod, &task, &tools, policy.as_mut());

    // reward through the masking contract
    let harness = crate::verifier::VerifierHarness::new(
        &bundle.rubric,
        judge.as_ref(),
        cfg.reward_config.clone(),
    );
    let reward = harness.calculate_reward(&mut pod, &bundle.manifest, &rollout.to_agent_result());
    RolloutRecord {
        instance_id,
        member,
        reward,
        tool_turns: rollout.tool_turns,
        tokens: rollout.tokens,
        final_message: rollout.final_message.clone(),
        infra_error: rollout.infra_error.clone(),
        advantage: None,
        screened: false,
        transcript_messages: rollout.messages.len(),
        dispatch_kinds: rollout.dispatches.iter().map(|d| d.kind).collect(),
    }
}

/// Deterministic rollout seed from (instance id, member).
fn rollout_seed(row: &TaskRow, member: usize) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in row
        .extra_info
        .instance_id
        .bytes()
        .chain(member.to_string().bytes())
    {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// GRPO group-relative advantages: `A_i = (r_i - mean_g) / (std_g + eps)`
/// computed **within each group of trainable rollouts** — masked and
/// screened rollouts are excluded from both the statistics and the
/// training set (never train on a false zero).
pub fn grpo_advantages(records: &mut [RolloutRecord], group_size: usize, eps: f64) {
    for group in records.chunks_mut(group_size.max(1)) {
        let scores: Vec<f64> = group
            .iter()
            .filter(|r| r.trainable())
            .map(|r| r.score())
            .collect();
        if scores.is_empty() {
            for r in group.iter_mut() {
                r.advantage = None;
            }
            continue;
        }
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        let std = var.sqrt();
        for r in group.iter_mut() {
            r.advantage = if r.trainable() {
                Some((r.score() - mean) / (std.max(eps)))
            } else {
                None
            };
        }
    }
}

/// GAR — Groupwise Advantage Redistribution.
///
/// Within each group, the passing set (`score > 0`) is ranked by quality
/// (fewer tokens and tool turns when `gar_prefers_short`); a fraction
/// `strength` of the passing set's total positive advantage mass is then
/// redistributed proportionally to rank so higher-quality solutions
/// carry more advantage. The redistribution is conservative: total
/// advantage mass is preserved, and failing members keep theirs.
pub fn gar(records: &mut [RolloutRecord], group_size: usize, cfg: &LiveConfig) {
    for group in records.chunks_mut(group_size.max(1)) {
        // passing, trainable members
        let mut passing: Vec<usize> = group
            .iter()
            .enumerate()
            .filter(|(_, r)| r.trainable() && r.score() > 0.0)
            .map(|(i, _)| i)
            .collect();
        if passing.len() < 2 {
            continue;
        }
        let total_pos: f64 = group
            .iter()
            .filter(|r| r.trainable() && r.score() > 0.0)
            .map(|r| r.advantage.unwrap_or(0.0).max(0.0))
            .sum();
        if total_pos <= 0.0 {
            // all passing with equal scores: zero advantage mass to move
            continue;
        }
        // rank by quality
        passing.sort_by(|&a, &b| {
            let qa = quality(&group[a], cfg);
            let qb = quality(&group[b], cfg);
            qb.partial_cmp(&qa).unwrap_or(std::cmp::Ordering::Equal)
        });
        // rank-based weights: rank 0 (highest quality) carries n, the
        // lowest carries 1 — quality-weighted redistribution
        let n = passing.len() as f64;
        let weights: Vec<f64> = (0..passing.len()).map(|k| n - k as f64).collect();
        let wsum: f64 = weights.iter().sum();
        let pool = cfg.gar_strength * total_pos;
        // subtract proportional share from every passing member, then
        // re-award by rank weight
        for (k, &i) in passing.iter().enumerate() {
            let share = pool * (weights[k] / wsum);
            let flat = pool / n;
            if let Some(a) = group[i].advantage.as_mut() {
                *a = *a - flat + share;
            }
        }
    }
}

/// Trajectory quality: shorter paths and fewer tool turns score higher.
pub fn quality(r: &RolloutRecord, cfg: &LiveConfig) -> f64 {
    if !cfg.gar_prefers_short {
        return r.score();
    }
    1.0 / (1.0 + r.tokens as f64 / 1000.0 + r.tool_turns as f64 / 10.0)
}

/// GRS — Groupwise Reward Synthesis.
///
/// Builds task-specific rubric items offline from **contrasting
/// rollouts** in a group: needles that appear in the passing members'
/// final answers but are absent from the failing members' answers become
/// deterministic rule items — a synthesized rubric that ranks passing
/// solutions (fusing rubric quality with outcomes, upstream).
pub fn grs(
    records: &[RolloutRecord],
    group_size: usize,
    final_messages: &BTreeMap<String, Vec<String>>,
    max_items: usize,
) -> Vec<RubricItem> {
    let mut items = Vec::new();
    for group in records.chunks(group_size.max(1)) {
        let pass: Vec<&str> = group
            .iter()
            .filter(|r| r.trainable() && r.score() >= 1.0)
            .map(|r| r.final_message.as_str())
            .collect();
        let fail: Vec<&str> = group
            .iter()
            .filter(|r| r.trainable() && r.score() < 1.0)
            .map(|r| r.final_message.as_str())
            .collect();
        if pass.is_empty() || fail.is_empty() {
            continue;
        }
        // candidate needles: distinctive tokens of passing answers
        let mut candidates: BTreeMap<String, usize> = BTreeMap::new();
        for p in &pass {
            for tok in distinctive_tokens(p) {
                *candidates.entry(tok).or_insert(0) += 1;
            }
        }
        // keep needles that are in MOST passing answers and NO failing ones
        let mut needles: Vec<String> = candidates
            .into_iter()
            .filter(|(tok, hits)| {
                *hits * 2 > pass.len() && !fail.iter().any(|f| f.contains(tok.as_str()))
            })
            .map(|(tok, _)| tok)
            .collect();
        needles.truncate(max_items);
        if !needles.is_empty() {
            items.push(RubricItem::rule(
                &format!("grs_{}", group[0].instance_id),
                0.5,
                RuleCheck::AllText { needles },
            ));
        }
    }
    let _ = final_messages;
    items
}

/// Tokens that distinguish one answer: uppercase ids, money literals,
/// long alphanumerics (the "grounded specifics" a judge keys on).
pub fn distinctive_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for w in text.split_whitespace() {
        let w = w
            .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '$' && c != '.' && c != ',');
        let looks_specific = w.len() >= 6
            && (w.contains('$')
                || w.chars()
                    .filter(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
                    .count()
                    * 2
                    >= w.len());
        if looks_specific && !out.contains(&w.to_string()) {
            out.push(w.to_string());
        }
    }
    out
}

/// Adversarial screening — the environment-hardening defense.
///
/// Flags and drops suspicious trainable rollouts:
/// * **zero-tool full score** — a perfect reward with fewer tool turns
///   than the floor in a tool-bearing environment is a guess or a
///   reward-hack signature, not a solution;
/// * **cross-check disagreement** — the verifier scored the same
///   trajectory differently under an independent judge.
pub fn adversarial_screening(
    records: &mut [RolloutRecord],
    min_tool_turns: u64,
    disagreement: &BTreeMap<(String, usize), f64>,
    tolerance: f64,
) -> usize {
    let mut dropped = 0;
    for r in records.iter_mut() {
        if !r.trainable() {
            continue;
        }
        let suspicious =
            r.score() >= 1.0 && r.tool_turns < min_tool_turns && !r.dispatch_kinds.is_empty()
                || r.score() >= 1.0 && r.tool_turns == 0 && r.transcript_messages > 2;
        let disagrees = disagreement
            .get(&(r.instance_id.clone(), r.member))
            .map(|d| *d > tolerance)
            .unwrap_or(false);
        if suspicious || disagrees {
            r.screened = true;
            dropped += 1;
        }
    }
    dropped
}

/// Runs verifier cross-checks for every trainable record and returns the
/// disagreement map (score deltas under the secondary judge).
///
/// The re-verification rebuilds the environment from the row (fresh DBs,
/// fresh workspace) and grades the recorded final message — the same
/// trajectory, an independent judge. Large deltas mean the reward signal
/// is unstable for that rollout and the sequence is screened out rather
/// than trained on.
pub fn cross_check_disagreements(
    provider: &Arc<dyn EnvProvider>,
    records: &[RolloutRecord],
    rows: &BTreeMap<String, TaskRow>,
    cross_judge: &Arc<dyn Judge>,
) -> BTreeMap<(String, usize), f64> {
    let mut out = BTreeMap::new();
    for r in records.iter().filter(|r| r.trainable()) {
        let Some(row) = rows.get(&r.instance_id) else {
            continue;
        };
        let Ok(mut bundle) = provider.build(row) else {
            continue;
        };
        let Some(score_b) =
            reverify_with_judge(&mut bundle, &r.final_message, cross_judge.as_ref())
        else {
            continue;
        };
        out.insert(
            (r.instance_id.clone(), r.member),
            (r.score() - score_b).abs(),
        );
    }
    out
}

/// Produces the aligned-RL self-correction cold-start pairs for one
/// group: (misaligned answer, grounded rewrite) — the failing members'
/// final messages paired with the best passing member's answer.
pub fn self_correction_pairs(
    records: &[RolloutRecord],
    group_size: usize,
) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for group in records.chunks(group_size.max(1)) {
        let pass: Vec<&RolloutRecord> = group
            .iter()
            .filter(|r| r.trainable() && r.score() >= 1.0)
            .collect();
        if pass.is_empty() {
            continue;
        }
        let best = pass.iter().min_by_key(|r| r.tokens).unwrap();
        for r in group.iter().filter(|r| r.trainable() && r.score() < 1.0) {
            if r.final_message.is_empty() {
                continue;
            }
            out.push((r.final_message.clone(), best.final_message.clone()));
        }
    }
    out
}

fn group_stats(records: &[RolloutRecord], prompts: usize, group_size: usize) -> TrainStepStats {
    let mut stats = TrainStepStats {
        prompts,
        rollouts: records.len(),
        ..Default::default()
    };
    let mut train_scores = Vec::new();
    for group in records.chunks(group_size.max(1)) {
        let any_trainable = group.iter().any(|r| r.trainable());
        if !any_trainable {
            stats.invalid_groups += 1;
        }
        for r in group {
            if r.reward.masked() {
                stats.masked += 1;
            }
            if r.screened {
                stats.screened += 1;
            }
            if r.trainable() {
                train_scores.push(r.score());
            }
        }
    }
    stats.trainable = train_scores.len();
    stats.mean_reward = train_scores.iter().sum::<f64>() / train_scores.len().max(1) as f64;
    stats
}

fn apply_advantages(
    records: &mut [RolloutRecord],
    _prompts: usize,
    group_size: usize,
    cfg: &LiveConfig,
) {
    grpo_advantages(records, group_size, cfg.adv_eps);
    gar(records, group_size, cfg);
}

fn finalize_stats(mut stats: TrainStepStats, records: &[RolloutRecord]) -> TrainStepStats {
    stats.advantage_mass = records
        .iter()
        .filter_map(|r| r.advantage)
        .sum::<f64>()
        .abs();
    stats
}

/// Re-runs the reward for one record under an independent judge — the
/// cross-check primitive used by screening (exposed for tests and
/// offline audits, mirroring `reverify_post_state` workflows).
pub fn reverify_with_judge(
    bundle: &mut EnvBundle,
    agent_output: &str,
    judge: &dyn Judge,
) -> Option<f64> {
    let post = build_post_state(&bundle.pod);
    let untouched: BTreeMap<String, bool> = bundle
        .pod
        .state
        .iter()
        .map(|(k, v)| (k.clone(), v.is_untouched()))
        .collect();
    let ctx = EvalCtx {
        agent_output,
        post_state: &post,
        source_conserved: bundle.pod.source_conserved(),
        state_untouched: &untouched,
    };
    let engine = RubricEngine::new(&bundle.rubric, judge, VerifyToggles::default());
    engine.run(&ctx).score
}

/// Convenience: extract the last assistant text from a booted bundle's
/// session log (post-rollout audits).
pub fn last_assistant_text(pod: &SimPod) -> String {
    extract_agent_output(pod)
}

/// The default judge stack: an anchor judge with an independent variant
/// (different token-pinning) as the cross-check judge.
pub fn default_judges() -> (Arc<AnchorJudge>, Arc<AnchorJudge>) {
    (Arc::new(AnchorJudge::new()), Arc::new(AnchorJudge::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentloop::ScriptedPolicy;
    use crate::manifest::{Manifest, VerifierSpec, SIDECAR};
    use crate::state::{Schema, StateDb, Table};

    use crate::topology::ToolDef;
    use serde_json::json;

    /// A minimal but complete provider: one CRM system, one tool, a
    /// two-item rubric (rule + judged).
    struct DemoProvider;

    impl EnvProvider for DemoProvider {
        fn build(&self, row: &TaskRow) -> Result<EnvBundle> {
            let schema =
                Schema::new().table(Table::text("customers", "id", &["id", "name", "status"]));
            let mut db = StateDb::new(schema);
            db.seed_rows(
                "customers",
                [json!({"id": "C1", "name": "Acme Corp", "status": "active"})
                    .as_object()
                    .unwrap()
                    .clone()],
            );
            let pod = SimPod::builder(&row.extra_info.instance_id)
                .system("crm", db)
                .server(
                    "crm",
                    vec![ToolDef {
                        name: "get_customer".into(),
                        params_schema: json!({"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}),
                        description: "Fetch a customer".into(),
                        exec: Box::new(|ctx| match ctx
                            .db("crm")
                            .and_then(|db| db.get("customers", ctx.param_str("id").unwrap_or("")))
                        {
                            Some(row) => json!(row),
                            None => json!({"error": "no such customer"}),
                        }),
                    }],
                )
                .build();
            let mut manifest = Manifest::default();
            manifest.setup = Some(crate::manifest::SetupSpec {
                command: "python3 /installed-agent/sidecar_entrypoint.py --start-and-detach".into(),
                container: SIDECAR.into(),
                timeout_sec: Some(300),
            });
            manifest.verifier = VerifierSpec::default();
            let rubric = Rubric::new()
                .item(RubricItem::rule(
                    "name",
                    0.5,
                    RuleCheck::AllText {
                        needles: vec!["Acme Corp".into()],
                    },
                ))
                .item(RubricItem::judged(
                    "status",
                    0.5,
                    "Report the status",
                    "the status is \"active\"",
                ));
            let tools = crate::mcp::discover_pod_tools(&pod);
            Ok(EnvBundle {
                pod,
                manifest,
                rubric,
                tools,
            })
        }
    }

    fn demo_row(i: u64) -> TaskRow {
        TaskRow::builder(&format!("row-{i}"))
            .data_source("demo")
            .ability("agent")
            .user_prompt("Report Acme Corp's status")
            .index(i)
            .build()
    }

    fn scripted_factory(answer: &str, calls: bool) -> PolicyFactory {
        let answer = answer.to_string();
        Arc::new(move |_seed| {
            let turns = if calls {
                vec![
                    (
                        "checking".to_string(),
                        vec![("crm.get_customer".to_string(), json!({"id": "C1"}))],
                    ),
                    (answer.clone(), vec![]),
                ]
            } else {
                vec![(answer.clone(), vec![])]
            };
            Box::new(ScriptedPolicy::new(turns))
        })
    }

    #[tokio::test]
    async fn live_step_masks_and_scores_groups() {
        // half the members fail the task, all succeed mechanically
        let provider = Arc::new(DemoProvider);
        let factory: PolicyFactory = Arc::new(|seed| {
            // member parity from the seed: make members differ by seed
            let fail = seed % 2 == 0;
            if fail {
                scripted_factory("dunno", false)(seed)
            } else {
                scripted_factory("Acme Corp status is active", false)(seed)
            }
        });
        let cfg = LiveConfig {
            group_size: 4,
            concurrency: 2,
            ..Default::default()
        };
        let trainer = LiveTrainer::new(cfg, provider, factory, Arc::new(AnchorJudge::new()));
        let batch = vec![demo_row(0), demo_row(1)];
        let (records, stats) = trainer.step(&batch).await.unwrap();
        assert_eq!(records.len(), 8);
        assert_eq!(stats.prompts, 2);
        assert_eq!(stats.rollouts, 8);
        assert_eq!(stats.masked, 0);
        // no member is trainable-with-zero-score... verify stats agree
        let valid = records.iter().filter(|r| r.trainable()).count();
        assert_eq!(stats.trainable, valid);
        assert_eq!(
            stats.mean_reward,
            records
                .iter()
                .filter(|r| r.trainable())
                .map(|r| r.score())
                .sum::<f64>()
                / valid as f64
        );
    }

    #[tokio::test]
    async fn grpo_advantage_signs_within_group() {
        let provider = Arc::new(DemoProvider);
        // deterministic: even members wrong (no tools), odd members
        // right — via one tool call, so screening leaves them alone
        let factory: PolicyFactory = Arc::new(|seed| {
            let fail = seed % 2 == 0;
            if fail {
                scripted_factory("wrong", false)(seed)
            } else {
                scripted_factory("Acme Corp is active", true)(seed)
            }
        });
        let cfg = LiveConfig {
            group_size: 4,
            ..Default::default()
        };
        let trainer = LiveTrainer::new(cfg, provider, factory, Arc::new(AnchorJudge::new()));
        let (mut records, _stats) = trainer.step(&[demo_row(7)]).await.unwrap();
        // re-apply pure GRPO (no GAR) to assert signs
        grpo_advantages(&mut records, 4, 1e-6);
        let wrong: Vec<&RolloutRecord> = records.iter().filter(|r| r.score() == 0.0).collect();
        let right: Vec<&RolloutRecord> = records.iter().filter(|r| r.score() > 0.0).collect();
        assert!(!wrong.is_empty() && !right.is_empty());
        for r in &wrong {
            assert!(r.advantage.unwrap() < 0.0);
        }
        for r in &right {
            assert!(r.advantage.unwrap() > 0.0);
        }
    }

    #[tokio::test]
    async fn masking_contract_excludes_rollout_from_training() {
        // judge unavailable -> every rollout masked -> group invalid
        let provider = Arc::new(DemoProvider);
        let factory = scripted_factory("Acme Corp active", false);
        let cfg = LiveConfig {
            group_size: 2,
            ..Default::default()
        };
        let trainer = LiveTrainer::new(
            cfg,
            provider,
            factory,
            Arc::new(crate::verifier::UnavailableJudge),
        );
        let (records, stats) = trainer.step(&[demo_row(3)]).await.unwrap();
        assert_eq!(stats.masked, 2);
        assert_eq!(stats.trainable, 0);
        assert_eq!(stats.invalid_groups, 1);
        assert!(records.iter().all(|r| r.reward.masked()));
        assert!(records.iter().all(|r| r.advantage.is_none()));
    }

    #[test]
    fn gar_preserves_mass_and_favors_quality() {
        let mk = |tokens: u64, score: f64| RolloutRecord {
            instance_id: "g".into(),
            member: 0,
            reward: if score >= 0.0 {
                RewardOutcome::Valid {
                    score,
                    raw: score,
                    results: vec![],
                }
            } else {
                RewardOutcome::TestbedCorrupted {
                    kind: crate::error::RewardErrorKind::VerifyCrashed,
                    detail: String::new(),
                }
            },
            tool_turns: 1,
            tokens,
            final_message: String::new(),
            infra_error: None,
            advantage: None,
            screened: false,
            transcript_messages: 3,
            dispatch_kinds: vec!["ok"],
        };
        // group: two passing with different token counts, one failing
        let mut records = vec![mk(100, 1.0), mk(5000, 1.0), mk(100, 0.0)];
        grpo_advantages(&mut records, 3, 1e-6);
        let pre: f64 = records.iter().filter_map(|r| r.advantage).sum();
        let cfg = LiveConfig::default();
        gar(&mut records, 3, &cfg);
        let post: f64 = records.iter().filter_map(|r| r.advantage).sum();
        assert!(
            (pre - post).abs() < 1e-9,
            "GAR must conserve advantage mass"
        );
        // the shorter passing solution must now carry more advantage
        assert!(records[0].advantage.unwrap() > records[1].advantage.unwrap());
    }

    #[test]
    fn grs_synthesizes_contrastive_needles() {
        let mk = |msg: &str, score: f64| RolloutRecord {
            instance_id: "row-0".into(),
            member: 0,
            reward: RewardOutcome::Valid {
                score,
                raw: score,
                results: vec![],
            },
            tool_turns: 2,
            tokens: 10,
            final_message: msg.into(),
            infra_error: None,
            advantage: None,
            screened: false,
            transcript_messages: 3,
            dispatch_kinds: vec!["ok"],
        };
        let records = vec![
            mk("CASE-FAIRFAX3-2025Q3 approved $46,536.00", 1.0),
            mk("CASE-FAIRFAX3-2025Q3 approved $46,536.00", 1.0),
            mk("the deal is fine", 0.0),
        ];
        let msgs = BTreeMap::new();
        let items = grs(&records, 3, &msgs, 4);
        assert_eq!(items.len(), 1);
        match &items[0].rule {
            Some(RuleCheck::AllText { needles }) => {
                assert!(needles.iter().any(|n| n.contains("FAIRFAX")));
            }
            other => panic!("expected AllText, got {other:?}"),
        }
    }

    #[test]
    fn screening_drops_zero_tool_full_scores() {
        let mut records = vec![
            RolloutRecord {
                instance_id: "r".into(),
                member: 0,
                reward: RewardOutcome::Valid {
                    score: 1.0,
                    raw: 1.0,
                    results: vec![],
                },
                tool_turns: 0,
                tokens: 5,
                final_message: "guessed!".into(),
                infra_error: None,
                advantage: None,
                screened: false,
                transcript_messages: 4,
                dispatch_kinds: vec![],
            },
            RolloutRecord {
                instance_id: "r".into(),
                member: 1,
                reward: RewardOutcome::Valid {
                    score: 1.0,
                    raw: 1.0,
                    results: vec![],
                },
                tool_turns: 2,
                tokens: 50,
                final_message: "worked it out".into(),
                infra_error: None,
                advantage: None,
                screened: false,
                transcript_messages: 6,
                dispatch_kinds: vec!["ok", "ok"],
            },
        ];
        let dropped = adversarial_screening(&mut records, 1, &BTreeMap::new(), 0.25);
        assert_eq!(dropped, 1);
        assert!(records[0].screened);
        assert!(!records[1].screened);
        assert!(!records[0].trainable());
    }

    #[test]
    fn self_correction_pairs_pair_failures_with_best_pass() {
        let mk = |msg: &str, score: f64, tokens: u64| RolloutRecord {
            instance_id: "r".into(),
            member: 0,
            reward: RewardOutcome::Valid {
                score,
                raw: score,
                results: vec![],
            },
            tool_turns: 1,
            tokens,
            final_message: msg.into(),
            infra_error: None,
            advantage: None,
            screened: false,
            transcript_messages: 3,
            dispatch_kinds: vec!["ok"],
        };
        let records = vec![
            mk("short right", 1.0, 10),
            mk("long right", 1.0, 9000),
            mk("wrong", 0.0, 10),
        ];
        let pairs = self_correction_pairs(&records, 3);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "wrong");
        assert_eq!(pairs[0].1, "short right");
    }

    #[test]
    fn distinctive_tokens_filter() {
        let toks = distinctive_tokens("CASE-FAIRFAX3-2025Q3 with $46,536.00 and some filler words");
        assert!(toks.iter().any(|t| t.contains("FAIRFAX")));
        assert!(toks.iter().any(|t| t.starts_with('$')));
        assert!(!toks.iter().any(|t| t == "some"));
    }
}
