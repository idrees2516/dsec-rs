//! The verifier pipeline and the reward contract.
//!
//! Upstream, reward flows through a strict two-stage contract:
//!
//! 1. The **task verifier** (`run_verify.py` + `verify.py` +
//!    `verifier_meta.json`) runs *inside the sidecar* after the rollout.
//!    It extracts the agent's final answer from the session logs, builds
//!    a `post_state` snapshot of every simulated system's database, and
//!    grades a rubric of weighted items (deterministic rules + LLM-judge
//!    questions). It writes `/logs/verifier/reward.json` containing
//!    **either** `{"reward": 0.42}` **or** `{"reward_error": "..."}` —
//!    never a zero on testbed failure.
//! 2. The **harness** (`_do_calculate_reward`) reads that file back. Any
//!    missing/invalid/out-of-range reward, dead MCP port, missing judge
//!    credential, or crashed verifier maps to the
//!    [`RewardErrorKind`] taxonomy and the rollout is **masked** — the
//!    trainer drops the sequence instead of training on a false zero.
//!
//! Both stages are ported here:
//!
//! * [`RubricItem`] / [`Rubric`] — the `verifier_meta.json` schema
//!   (`id`, `tier`, `method` `rule|llm`, `weight`, `gate`, `question`,
//!   `pass_anchor`) plus the deterministic check language;
//! * [`Judge`] — the LLM-judge half, with deterministic anchor-matching
//!   and injectable failure modes;
//! * [`RubricEngine`] — the `run_verify.py` equivalent: rubric filtering
//!   by `VERIFY_DETERMINISTIC` / `VERIFY_AGENT_JUDGE`, weighted scoring,
//!   the `src_protect` source-conservation gate, `reward.json` /
//!   `reward_detail.json` emission;
//! * [`VerifierHarness::calculate_reward`] — the `_do_calculate_reward`
//!   equivalent implementing the full masking contract end to end.

use crate::error::RewardErrorKind;
use crate::manifest::MAIN;
use crate::manifest::{Manifest, VerifierSpec, SIDECAR};
use crate::state::PostState;
use crate::topology::SimPod;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

/// A deterministic rubric check.
///
/// The released environments express these as Python predicates over
/// `(agent_output, post_state)`; the port models the recurring shapes
/// plus a closure escape hatch.
#[derive(Clone)]
pub enum RuleCheck {
    /// The agent output must contain every needle (case-insensitive).
    AllText {
        /// Required substrings.
        needles: Vec<String>,
    },
    /// The agent output must mention at least `min_hits` of the needles.
    AnyText {
        /// Candidate substrings.
        needles: Vec<String>,
        /// Minimum hits.
        min_hits: usize,
    },
    /// `post_state[system][table][pk][field] == value`.
    StateEq {
        /// System name.
        system: String,
        /// Table name.
        table: String,
        /// Primary-key value.
        pk: String,
        /// Column.
        field: String,
        /// Expected value.
        value: Value,
    },
    /// A system's DB must be untouched by tool mutations.
    StateUntouched {
        /// System name.
        system: String,
    },
    /// The auto source-conservation gate (protected workspace files).
    SourceConserved,
    /// Custom predicate over the evaluation context.
    Custom(std::sync::Arc<dyn Fn(&EvalCtx) -> bool + Send + Sync>),
}

impl std::fmt::Debug for RuleCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllText { needles } => {
                f.debug_struct("AllText").field("needles", needles).finish()
            }
            Self::AnyText { needles, min_hits } => f
                .debug_struct("AnyText")
                .field("needles", needles)
                .field("min_hits", min_hits)
                .finish(),
            Self::StateEq {
                system,
                table,
                pk,
                field,
                value,
            } => f
                .debug_struct("StateEq")
                .field("system", system)
                .field("table", table)
                .field("pk", pk)
                .field("field", field)
                .field("value", value)
                .finish(),
            Self::StateUntouched { system } => f
                .debug_struct("StateUntouched")
                .field("system", system)
                .finish(),
            Self::SourceConserved => f.write_str("SourceConserved"),
            Self::Custom(_) => f.write_str("Custom"),
        }
    }
}

/// Evaluation context handed to rules.
pub struct EvalCtx<'a> {
    /// The agent's final answer text.
    pub agent_output: &'a str,
    /// DB snapshot after the rollout.
    pub post_state: &'a PostState,
    /// Whether the pod's workspace passed the source-conservation gate.
    pub source_conserved: bool,
    /// Per-system untouched flags (no tool-path mutations) — backs the
    /// [`RuleCheck::StateUntouched`] rule.
    pub state_untouched: &'a BTreeMap<String, bool>,
}

/// One rubric item — `verifier_meta.json` schema.
#[derive(Debug, Clone)]
pub struct RubricItem {
    /// Item id (joined back to results by the annotator).
    pub id: String,
    /// `critical` / `important` (display; scoring is weight-driven).
    pub tier: String,
    /// `rule` (deterministic) or `llm` (judge).
    pub method: String,
    /// Weight in the weighted score.
    pub weight: f64,
    /// Gate items force score 0 when failed.
    pub gate: bool,
    /// The files the item grades (`answer.md`).
    pub files: Vec<String>,
    /// Judge question (llm items).
    pub question: String,
    /// Ground-truth anchor text (llm items).
    pub pass_anchor: String,
    /// Deterministic rule (rule items).
    pub rule: Option<RuleCheck>,
}

impl RubricItem {
    /// A rule item with default shape.
    pub fn rule(id: &str, weight: f64, check: RuleCheck) -> Self {
        Self {
            id: id.into(),
            tier: "critical".into(),
            method: "rule".into(),
            weight,
            gate: false,
            files: vec!["answer.md".into()],
            question: String::new(),
            pass_anchor: String::new(),
            rule: Some(check),
        }
    }

    /// An llm-judge item.
    pub fn judged(id: &str, weight: f64, question: &str, pass_anchor: &str) -> Self {
        Self {
            id: id.into(),
            tier: "critical".into(),
            method: "llm".into(),
            weight,
            gate: false,
            files: vec!["answer.md".into()],
            question: question.into(),
            pass_anchor: pass_anchor.into(),
            rule: None,
        }
    }

    /// Marks the item as a score gate.
    pub fn gate(mut self) -> Self {
        self.gate = true;
        self
    }
}

/// The whole rubric.
#[derive(Debug, Clone, Default)]
pub struct Rubric {
    /// Items in declaration order.
    pub items: Vec<RubricItem>,
}

impl Rubric {
    /// Empty rubric.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an item.
    pub fn item(mut self, it: RubricItem) -> Self {
        self.items.push(it);
        self
    }
}

/// The LLM-judge verdict.
#[derive(Debug, Clone, PartialEq)]
pub enum JudgeVerdict {
    /// The item passed.
    Pass,
    /// The item failed.
    Fail,
    /// The judge endpoint was unavailable — a testbed error, not a task
    /// failure (drives the masking contract).
    Unavailable,
}

/// The judge half of verification.
pub trait Judge: Send + Sync {
    /// Judges one llm-rubric item against the agent output.
    fn judge(&self, item: &RubricItem, agent_output: &str) -> JudgeVerdict;
}

/// Deterministic anchor judge: passes when the agent output contains
/// every required token (case-insensitive).
///
/// This is the stand-in for the hosted LLM judge: the same contract, a
/// reproducible decision function — so training runs and tests are
/// deterministic and replayable.
#[derive(Debug, Clone, Default)]
pub struct AnchorJudge {
    /// `item id -> required tokens`; items absent from the map fall back
    /// to tokens extracted from `pass_anchor` (quoted spans and money /
    /// number literals).
    pub required: BTreeMap<String, Vec<String>>,
}

impl AnchorJudge {
    /// Empty judge (falls back to anchor extraction).
    pub fn new() -> Self {
        Self::default()
    }

    /// Pins required tokens for one item.
    pub fn require(mut self, id: &str, tokens: &[&str]) -> Self {
        self.required.insert(
            id.to_string(),
            tokens.iter().map(|t| t.to_string()).collect(),
        );
        self
    }

    fn tokens_for(&self, item: &RubricItem) -> Vec<String> {
        if let Some(t) = self.required.get(&item.id) {
            return t.clone();
        }
        extract_anchor_tokens(&item.pass_anchor)
    }
}

impl Judge for AnchorJudge {
    fn judge(&self, item: &RubricItem, agent_output: &str) -> JudgeVerdict {
        let tokens = self.tokens_for(item);
        if tokens.is_empty() {
            // nothing judgeable: fall back to a substring similarity gate
            let anchor = item.pass_anchor.to_lowercase();
            let out = agent_output.to_lowercase();
            let hits = anchor
                .split_whitespace()
                .filter(|w| w.len() > 4 && out.contains(*w))
                .count();
            let total = anchor.split_whitespace().filter(|w| w.len() > 4).count();
            return if total > 0 && hits * 2 >= total {
                JudgeVerdict::Pass
            } else {
                JudgeVerdict::Fail
            };
        }
        let out = agent_output.to_lowercase();
        let all = tokens.iter().all(|t| out.contains(&t.to_lowercase()));
        if all {
            JudgeVerdict::Pass
        } else {
            JudgeVerdict::Fail
        }
    }
}

/// A judge whose endpoint is down — exercises the masking contract.
#[derive(Debug, Clone, Default)]
pub struct UnavailableJudge;

impl Judge for UnavailableJudge {
    fn judge(&self, _item: &RubricItem, _agent_output: &str) -> JudgeVerdict {
        JudgeVerdict::Unavailable
    }
}

/// Extracts the checkable tokens from a pass-anchor: quoted spans and
/// money/number literals (`$7,183,750.00`, `CASE-FAIRFAX3-2025Q3`, ...).
pub fn extract_anchor_tokens(anchor: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    // quoted spans
    let bytes: Vec<char> = anchor.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == '"' || bytes[i] == '`' {
            let quote = bytes[i];
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && bytes[j] != quote {
                j += 1;
            }
            if j > start {
                let span: String = bytes[start..j].iter().collect();
                if span.len() >= 3 {
                    tokens.push(span);
                }
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    // money and numbers
    for word in anchor.split(|c: char| c.is_whitespace()) {
        let w = word
            .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '$' && c != ',' && c != '.');
        if w.starts_with('$') && w.len() > 2 {
            tokens.push(w.to_string());
        }
    }
    tokens.dedup();
    tokens
}

/// Per-item result in the detail payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RubricResult {
    /// Item id.
    pub id: String,
    /// Whether the item passed.
    pub passed: bool,
    /// Method tag joined back from `verifier_meta.json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// Weight joined back from `verifier_meta.json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    /// Gate flag joined back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate: Option<bool>,
    /// Human message.
    pub message: String,
}

/// The engine output (`run_verify.py` result dict shape).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineOutput {
    /// Weighted score in `[0, 1]` (present on success only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Upstream-style error key (present on testbed failure only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward_error: Option<String>,
    /// Strict pass: every item passed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_pass: Option<bool>,
    /// Deterministic-rubric subscore.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deterministic_score: Option<f64>,
    /// Judge-rubric subscore.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_score: Option<f64>,
    /// All judge items passed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_all_passed: Option<bool>,
    /// Per-item results.
    #[serde(default)]
    pub results: Vec<RubricResult>,
}

impl EngineOutput {
    /// Emulates the `run_verify.py` error path: no parseable `reward` key.
    pub fn error(kind: &str) -> Self {
        Self {
            score: None,
            reward_error: Some(kind.to_string()),
            strict_pass: None,
            deterministic_score: None,
            judge_score: None,
            judge_all_passed: None,
            results: Vec::new(),
        }
    }
}

/// Verification toggles — the `VERIFY_*` env surface.
#[derive(Debug, Clone)]
pub struct VerifyToggles {
    /// Include deterministic rubrics (env `VERIFY_DETERMINISTIC`, default on).
    pub deterministic: bool,
    /// Include llm-judge rubrics (env `VERIFY_AGENT_JUDGE`).
    pub agent_judge: bool,
}

impl Default for VerifyToggles {
    fn default() -> Self {
        Self {
            deterministic: true,
            agent_judge: true,
        }
    }
}

/// The `run_verify.py` equivalent: filters rubrics, evaluates, scores,
/// and emits the reward files into the pod's `/logs/verifier`.
pub struct RubricEngine<'a> {
    rubric: &'a Rubric,
    judge: &'a dyn Judge,
    toggles: VerifyToggles,
}

impl<'a> RubricEngine<'a> {
    /// New engine over a rubric, judge and toggles.
    pub fn new(rubric: &'a Rubric, judge: &'a dyn Judge, toggles: VerifyToggles) -> Self {
        Self {
            rubric,
            judge,
            toggles,
        }
    }

    /// Filters rubric items by the toggles (the `filter_rubrics` step).
    pub fn filter(&self) -> Vec<&RubricItem> {
        self.rubric
            .items
            .iter()
            .filter(|it| match it.method.as_str() {
                "llm" | "agent_judge" => self.toggles.agent_judge,
                _ => self.toggles.deterministic,
            })
            .collect()
    }

    /// Runs the filtered rubric. Never writes files — pure evaluation
    /// (the harness writes the contract files).
    pub fn run(&self, ctx: &EvalCtx) -> EngineOutput {
        let items = self.filter();
        let mut results = Vec::new();
        let mut judge_unavailable = false;
        let (mut det_w, mut det_s, mut jud_w, mut jud_s) = (0.0, 0.0, 0.0, 0.0);
        let mut gate_failed = false;
        let mut judge_all = true;

        for it in items {
            let (passed, message) = match it.method.as_str() {
                "llm" | "agent_judge" => match self.judge.judge(it, ctx.agent_output) {
                    JudgeVerdict::Pass => (true, format!("judge passed: {}", it.id)),
                    JudgeVerdict::Fail => (false, format!("judge failed: {}", it.id)),
                    JudgeVerdict::Unavailable => {
                        judge_unavailable = true;
                        (false, format!("judge unavailable: {}", it.id))
                    }
                },
                _ => match &it.rule {
                    Some(rule) => {
                        let passed = evaluate_rule(rule, ctx);
                        (
                            passed,
                            format!("rule {}: {}", it.id, if passed { "pass" } else { "fail" }),
                        )
                    }
                    None => (false, format!("rule item {} has no rule", it.id)),
                },
            };
            if it.method == "llm" || it.method == "agent_judge" {
                jud_w += it.weight;
                if passed {
                    jud_s += it.weight;
                } else {
                    judge_all = false;
                }
            } else {
                det_w += it.weight;
                if passed {
                    det_s += it.weight;
                }
            }
            if it.gate && !passed {
                gate_failed = true;
            }
            results.push(RubricResult {
                id: it.id.clone(),
                passed,
                method: Some(it.method.clone()),
                weight: Some(it.weight),
                gate: Some(it.gate),
                message,
            });
        }

        // src_protect: the auto source-conservation gate — appended to
        // every rubric, id joins to no meta item (method "gate:auto").
        if !results.iter().any(|r| r.id == "src_protect") {
            let conserved = ctx.source_conserved;
            if !conserved {
                gate_failed = true;
            }
            results.push(RubricResult {
                id: "src_protect".into(),
                passed: conserved,
                method: Some("gate:auto".into()),
                weight: Some(0.0),
                gate: Some(true),
                message: if conserved {
                    "workspace sources conserved".into()
                } else {
                    "protected workspace files were modified".into()
                },
            });
        }

        if judge_unavailable {
            return EngineOutput::error("judge_unavailable");
        }

        let total_w = det_w + jud_w;
        let mut score = if total_w > 0.0 {
            (det_s + jud_s) / total_w
        } else {
            0.0
        };
        if gate_failed {
            score = 0.0;
        }
        let det_score = if det_w > 0.0 { det_s / det_w } else { 0.0 };
        let jud_score = if jud_w > 0.0 { jud_s / jud_w } else { 0.0 };
        let strict = results.iter().all(|r| r.passed);
        EngineOutput {
            score: Some(score),
            reward_error: None,
            strict_pass: Some(strict),
            deterministic_score: Some(det_score),
            judge_score: Some(jud_score),
            judge_all_passed: Some(judge_all),
            results,
        }
    }
}

fn evaluate_rule(rule: &RuleCheck, ctx: &EvalCtx) -> bool {
    match rule {
        RuleCheck::AllText { needles } => {
            let out = ctx.agent_output.to_lowercase();
            needles.iter().all(|n| out.contains(&n.to_lowercase()))
        }
        RuleCheck::AnyText { needles, min_hits } => {
            let out = ctx.agent_output.to_lowercase();
            let hits = needles
                .iter()
                .filter(|n| out.contains(&n.to_lowercase()))
                .count();
            hits >= *min_hits
        }
        RuleCheck::StateEq {
            system,
            table,
            pk,
            field,
            value,
        } => ctx
            .post_state
            .systems
            .get(system)
            .and_then(|t| t.get(table))
            .and_then(|rows| {
                rows.iter().find(|r| {
                    r.values().any(|v| v.as_str() == Some(pk.as_str()))
                        || r.iter()
                            .any(|(k, v)| *k == "id" && v.as_str() == Some(pk.as_str()))
                        || r.iter().any(|(k, _)| *k == *pk)
                })
            })
            .map(|row| {
                row.get(field)
                    .map(|v| v == value)
                    .unwrap_or(value.is_null())
            })
            .unwrap_or(false),
        RuleCheck::StateUntouched { system } => {
            ctx.state_untouched.get(system).copied().unwrap_or(false)
        }
        RuleCheck::SourceConserved => ctx.source_conserved,
        RuleCheck::Custom(f) => f(ctx),
    }
}

/// The agent-side result fed into reward extraction.
#[derive(Debug, Clone, Default)]
pub struct AgentRolloutResult {
    /// Final assistant message (answer.md candidate).
    pub final_message: String,
    /// Whether the rollout completed normally (vs. step limit).
    pub completed: bool,
    /// Number of tool turns used.
    pub tool_turns: u64,
}

/// Extracts the last assistant message from the pod's session logs —
/// the `extract_agent_output` step (falls back to workspace `answer.md`).
pub fn extract_agent_output(pod: &SimPod) -> String {
    // injected sessions first (verifier-side view), then main-side view
    for (container, base) in [
        (SIDECAR, "/tmp/agent_output/sessions"),
        (MAIN, "/tmp/mimo-claude-logs/sessions"),
    ] {
        let Some((vid, rel, _)) = pod.containers.get(container).and_then(|c| c.resolve(base))
        else {
            continue;
        };
        let Some(vol) = pod.volumes.get(&vid) else {
            continue;
        };
        let mut files: Vec<&String> = vol
            .files
            .keys()
            .filter(|k| k.starts_with(&rel) && k.ends_with(".jsonl"))
            .collect();
        files.sort();
        let mut last = String::new();
        for f in files {
            if let Some(bytes) = vol.read(f) {
                for line in String::from_utf8_lossy(bytes).lines() {
                    if let Ok(ev) = serde_json::from_str::<Value>(line) {
                        if ev.get("type").and_then(|t| t.as_str()) == Some("assistant") {
                            if let Some(parts) = ev
                                .get("message")
                                .and_then(|m| m.get("content"))
                                .and_then(|c| c.as_array())
                            {
                                let text: Vec<&str> = parts
                                    .iter()
                                    .filter_map(|b| {
                                        b.get("text")
                                            .and_then(|t| t.as_str())
                                            .filter(|t| !t.trim().is_empty())
                                    })
                                    .collect();
                                if !text.is_empty() {
                                    last = text.join("\n");
                                }
                            }
                        }
                    }
                }
            }
        }
        if !last.is_empty() {
            return last;
        }
    }
    // fallback: answer.md written by the agent
    pod.read_file(MAIN, "/work/workspace/answer.md")
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default()
}

/// Builds the post-state snapshot of every system DB — the
/// `build_post_state` step.
pub fn build_post_state(pod: &SimPod) -> PostState {
    PostState::snapshot(pod.state.iter().map(|(k, v)| (k.as_str(), v)))
}

/// Reward harness configuration.
#[derive(Debug, Clone)]
pub struct RewardConfig {
    /// Whether the judge credential is present (`GA_JUDGE_KEY`).
    pub judge_available: bool,
    /// `reward_binary_fullscore`: binarize the reward at 1.0.
    pub binary_fullscore: bool,
    /// Verifier timeout override, seconds.
    pub verifier_timeout: Option<u64>,
    /// VERIFY toggles.
    pub toggles: VerifyToggles,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            judge_available: true,
            binary_fullscore: false,
            verifier_timeout: None,
            toggles: VerifyToggles::default(),
        }
    }
}

/// The reward outcome — the two-arm contract.
#[derive(Debug, Clone, PartialEq)]
pub enum RewardOutcome {
    /// A legitimate reward in `[0, 1]`.
    Valid {
        /// The score (post-binarization).
        score: f64,
        /// The raw score before binarization.
        raw: f64,
        /// Per-item results.
        results: Vec<RubricResult>,
    },
    /// The testbed broke — the rollout must be masked, never trained as
    /// a zero.
    TestbedCorrupted {
        /// Fine-grained category.
        kind: RewardErrorKind,
        /// Human detail.
        detail: String,
    },
}

impl RewardOutcome {
    /// The score if valid (0.0 when masked — trainers must consult the
    /// variant, not just this helper).
    pub fn score(&self) -> f64 {
        match self {
            Self::Valid { score, .. } => *score,
            Self::TestbedCorrupted { .. } => 0.0,
        }
    }

    /// Whether the rollout should be masked out of training.
    pub fn masked(&self) -> bool {
        matches!(self, Self::TestbedCorrupted { .. })
    }
}

/// The harness-side reward pipeline (`_do_calculate_reward`).
pub struct VerifierHarness<'a> {
    rubric: &'a Rubric,
    judge: &'a dyn Judge,
    config: RewardConfig,
    /// Local task dir for manifest-relative upload sources (optional).
    task_dir: std::path::PathBuf,
}

impl<'a> VerifierHarness<'a> {
    /// New harness.
    pub fn new(rubric: &'a Rubric, judge: &'a dyn Judge, config: RewardConfig) -> Self {
        Self {
            rubric,
            judge,
            config,
            task_dir: std::path::PathBuf::new(),
        }
    }

    /// Pins the local task dir for manifest-relative verifier uploads.
    pub fn with_task_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.task_dir = dir.into();
        self
    }

    /// Runs the full contract against a pod whose rollout has finished.
    ///
    /// Stage order matches upstream exactly: judge-credential check,
    /// MCP liveness probes, verifier-entry presence, late uploads,
    /// session injection, answer.md persistence, reward-file reset,
    /// verifier execution, reward.json parse + range check,
    /// binarization, detail annotation.
    pub fn calculate_reward(
        &self,
        pod: &mut SimPod,
        manifest: &Manifest,
        rollout: &AgentRolloutResult,
    ) -> RewardOutcome {
        // 1. judge credential
        if !self.config.judge_available && self.judge_needed() {
            return RewardOutcome::TestbedCorrupted {
                kind: RewardErrorKind::JudgeKeyMissing,
                detail: "GA_JUDGE_KEY missing from controller env".into(),
            };
        }
        // 2. MCP liveness probes (from the verifier container)
        let probe_container = manifest.verifier.container.clone();
        for wp in &manifest.wait_ports {
            if !pod.port_live(*wp) {
                return RewardOutcome::TestbedCorrupted {
                    kind: RewardErrorKind::McpBackendDead,
                    detail: format!("dead_port={}", wp),
                };
            }
        }
        let _ = probe_container;
        // 3. verifier entry presence. Four material sources count:
        //    declared uploads, an in-pod /work copy, the staging volume
        //    (the factory's vstage), or a non-empty native rubric —
        //    this port's native engine IS the verifier when no script
        //    bundle was uploaded.
        let v: &VerifierSpec = &manifest.verifier;
        if let Some(entry) = verifier_entry(&v.command) {
            let base = entry.rsplit('/').next().unwrap_or(&entry).to_string();
            let staged = format!("/vstage/{base}");
            let uploaded = v.uploads.iter().any(|u| u.target == entry);
            let in_pod = pod.read_file(SIDECAR, &entry).is_some()
                || pod.read_file(SIDECAR, &staged).is_some();
            let has_native = !self.rubric.items.is_empty();
            if !uploaded && !in_pod && !has_native {
                return RewardOutcome::TestbedCorrupted {
                    kind: RewardErrorKind::MissingRunVerify,
                    detail: format!("missing_script={}", entry),
                };
            }
        }
        // 4. late uploads (ground truth becomes visible only now) —
        //    local task-dir files first, then the pod's staging volume
        for u in &v.uploads {
            let local = crate::manifest::resolve_source(&self.task_dir, &u.source);
            if local.exists() {
                let _ = pod.copy_to(&local, &u.target, &v.container);
            } else if let Some(bytes) = pod.read_file(SIDECAR, &format!("/vstage/{}", u.source)) {
                let _ = pod.write_file(&v.container, &u.target, &bytes);
            }
        }
        // 5. session injection: main sessions -> verifier container
        inject_agent_sessions(pod);
        // 6. persist answer.md (an agent-written file wins)
        if pod
            .read_file(MAIN, "/work/workspace/answer.md")
            .map(|b| b.is_empty())
            .unwrap_or(true)
            && !rollout.final_message.is_empty()
        {
            let _ = pod.write_file(
                MAIN,
                "/work/workspace/answer.md",
                rollout.final_message.as_bytes(),
            );
        }
        // 7. reset reward files
        let logs = "verifier";
        if let Some(vol) = pod.volumes.get_mut(crate::topology::vols::LOGS) {
            vol.write(&format!("{logs}/reward.json"), Vec::new());
            vol.write(&format!("{logs}/reward_detail.json"), Vec::new());
        }
        // 8. execute the verifier (native rubric engine) — writes the
        //    reward files inside the pod
        let agent_output = extract_agent_output(pod);
        let post_state = build_post_state(pod);
        let untouched: BTreeMap<String, bool> = pod
            .state
            .iter()
            .map(|(k, v)| (k.clone(), v.is_untouched()))
            .collect();
        let ctx = EvalCtx {
            agent_output: &agent_output,
            post_state: &post_state,
            source_conserved: pod.source_conserved(),
            state_untouched: &untouched,
        };
        let engine = RubricEngine::new(self.rubric, self.judge, self.config.toggles.clone());
        let output = engine.run(&ctx);
        // masking contract: reward.json carries EITHER {"reward"} OR
        // {"reward_error"} — never a false zero
        if let Some(vol) = pod.volumes.get_mut(crate::topology::vols::LOGS) {
            let payload = match output.score {
                Some(s) => json!({"reward": s}).to_string(),
                None => json!({"reward_error": output.reward_error.clone().unwrap_or_default()})
                    .to_string(),
            };
            vol.write(&format!("{logs}/reward.json"), payload.into_bytes());
            let detail = serde_json::to_string_pretty(&output).unwrap_or_default();
            vol.write(&format!("{logs}/reward_detail.json"), detail.into_bytes());
        }
        // 9. read back + parse (the harness only trusts the file)
        let raw = pod
            .read_file(SIDECAR, &v.reward_file)
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .unwrap_or_default();
        let parsed: std::result::Result<Map<String, Value>, _> = serde_json::from_str(raw.trim());
        let score = match parsed {
            Ok(map) => match map.get("reward").and_then(|r| r.as_f64()) {
                Some(s) => s,
                None => {
                    return RewardOutcome::TestbedCorrupted {
                        kind: RewardErrorKind::MissingOrInvalidRewardJson,
                        detail: format!(
                            "reward_json_raw={}",
                            raw.chars().take(200).collect::<String>()
                        ),
                    }
                }
            },
            Err(_) => {
                return RewardOutcome::TestbedCorrupted {
                    kind: RewardErrorKind::MissingOrInvalidRewardJson,
                    detail: format!(
                        "reward_json_raw={}",
                        raw.chars().take(200).collect::<String>()
                    ),
                }
            }
        };
        // 10. range check
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            return RewardOutcome::TestbedCorrupted {
                kind: RewardErrorKind::RewardOutOfRange,
                detail: format!("reward={score}"),
            };
        }
        // 11. binarization + detail annotation
        let effective = if self.config.binary_fullscore {
            if score >= 1.0 - 1e-6 {
                1.0
            } else {
                0.0
            }
        } else {
            score
        };
        RewardOutcome::Valid {
            score: effective,
            raw: score,
            results: output.results,
        }
    }

    fn judge_needed(&self) -> bool {
        self.rubric
            .items
            .iter()
            .any(|it| it.method == "llm" || it.method == "agent_judge")
    }
}

/// Extracts the entry-script token from a verifier command (`python3
/// /work/run_verify.py` → `/work/run_verify.py`).
fn verifier_entry(command: &str) -> Option<String> {
    crate::topology::shell_split(command)
        .into_iter()
        .find(|t| t.starts_with("/work/") && (t.ends_with(".py") || t.ends_with(".sh")))
}

/// Copies the main-side session logs to the verifier container — the
/// two-hop `inject_agent_sessions` (no shared volume between the session
/// dir and the verifier container, deliberately).
fn inject_agent_sessions(pod: &mut SimPod) {
    let Some(bytes) = pod.read_file(MAIN, "/tmp/mimo-claude-logs/sessions/session-0.jsonl") else {
        return;
    };
    let _ = pod.write_file(
        SIDECAR,
        "/tmp/agent_output/sessions/session-0.jsonl",
        &bytes,
    );
}

/// The verifier cross-check — anti-reward-hacking defense.
///
/// Runs the rubric twice: once with the primary judge and once with an
/// independent second judge. A large disagreement flags verifier
/// instability (the reward signal cannot be trusted for this rollout —
/// upstream keeps the loop honest with exactly this class of
/// cross-check during Live RL).
pub fn cross_check(
    rubric: &Rubric,
    judge_a: &dyn Judge,
    judge_b: &dyn Judge,
    ctx: &EvalCtx,
) -> CrossCheck {
    let out_a = RubricEngine::new(rubric, judge_a, VerifyToggles::default()).run(ctx);
    let out_b = RubricEngine::new(rubric, judge_b, VerifyToggles::default()).run(ctx);
    let sa = out_a.score.unwrap_or(0.0);
    let sb = out_b.score.unwrap_or(0.0);
    CrossCheck {
        score_a: sa,
        score_b: sb,
        disagreement: (sa - sb).abs(),
        both_valid: out_a.score.is_some() && out_b.score.is_some(),
    }
}

/// Result of [`cross_check`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossCheck {
    /// Primary judge score.
    pub score_a: f64,
    /// Secondary judge score.
    pub score_b: f64,
    /// Absolute disagreement.
    pub disagreement: f64,
    /// Whether both runs produced valid scores.
    pub both_valid: bool,
}

impl CrossCheck {
    /// Whether the disagreement is within tolerance.
    pub fn agree(&self, tolerance: f64) -> bool {
        self.both_valid && self.disagreement <= tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Schema, StateDb, Table};
    use serde_json::json;

    fn pod_with_workspace(answer: &str) -> SimPod {
        let schema = Schema::new().table(Table::text("deals", "id", &["id", "stage", "amount"]));
        let mut db = StateDb::new(schema);
        db.seed_rows(
            "deals",
            [
                json!({"id": "D1", "stage": "recommended", "amount": "$7,183,750.00"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ],
        );
        let mut pod = SimPod::builder("t")
            .system("crm", db)
            .workspace_file("brief.md", b"brief".to_vec())
            .build();
        pod.capture_protection();
        pod.append_session(
            &json!({"type": "user", "message": {"content": [{"type": "text", "text": "task"}]}}),
        );
        pod.append_session(&json!({"type": "assistant", "message": {"content": [{"type": "text", "text": answer}]}}));
        pod
    }

    fn manifest_with_verifier() -> Manifest {
        Manifest {
            verifier: VerifierSpec {
                uploads: vec![
                    crate::manifest::Upload {
                        source: "verify.py".into(),
                        target: "/work/verify.py".into(),
                        container: SIDECAR.into(),
                    },
                    crate::manifest::Upload {
                        source: "run_verify.py".into(),
                        target: "/work/run_verify.py".into(),
                        container: SIDECAR.into(),
                    },
                ],
                command: "python3 /work/run_verify.py".into(),
                container: SIDECAR.into(),
                reward_file: "/logs/verifier/reward.json".into(),
                reward_detail_file: "/logs/verifier/reward_detail.json".into(),
                timeout_sec: None,
                env_passthrough: None,
            },
            ..Manifest::default()
        }
    }

    #[test]
    fn anchor_token_extraction() {
        let anchor = "the county component = $9,141.00; the case is \"CASE-FAIRFAX3-2025Q3\" and the summary is $46,536.00";
        let tokens = extract_anchor_tokens(anchor);
        assert!(tokens.contains(&"CASE-FAIRFAX3-2025Q3".to_string()));
        assert!(tokens.contains(&"$9,141.00".to_string()));
        assert!(tokens.contains(&"$46,536.00".to_string()));
    }

    #[test]
    fn engine_scores_weighted_and_gates() {
        let rubric = Rubric::new()
            .item(RubricItem::rule(
                "amount",
                0.4,
                RuleCheck::AllText {
                    needles: vec!["$7,183,750.00".into()],
                },
            ))
            .item(RubricItem::judged(
                "structure",
                0.6,
                "Identify the structure",
                "the record is \"CASE-FAIRFAX3-2025Q3\"",
            ));
        let pod = pod_with_workspace(
            "The record CASE-FAIRFAX3-2025Q3 is recommended with amount $7,183,750.00.",
        );
        let post = build_post_state(&pod);
        let untouched = BTreeMap::new();
        let ctx = EvalCtx {
            agent_output:
                "The record CASE-FAIRFAX3-2025Q3 is recommended with amount $7,183,750.00.",
            post_state: &post,
            source_conserved: true,
            state_untouched: &untouched,
        };
        let judge = AnchorJudge::new();
        let out = RubricEngine::new(&rubric, &judge, VerifyToggles::default()).run(&ctx);
        assert_eq!(out.score, Some(1.0));
        assert_eq!(out.strict_pass, Some(true));
        // src_protect auto-appended
        assert!(out
            .results
            .iter()
            .any(|r| r.id == "src_protect" && r.passed));
        // a failing answer halves the weighted score
        let ctx2 = EvalCtx {
            agent_output: "wrong",
            post_state: &post,
            source_conserved: true,
            state_untouched: &untouched,
        };
        let out2 = RubricEngine::new(&rubric, &judge, VerifyToggles::default()).run(&ctx2);
        assert_eq!(out2.score, Some(0.0));
    }

    #[test]
    fn judge_unavailable_masks_via_reward_error() {
        let rubric = Rubric::new().item(RubricItem::judged("q", 1.0, "q?", "anchor"));
        let post = PostState::default();
        let untouched = BTreeMap::new();
        let ctx = EvalCtx {
            agent_output: "ans",
            post_state: &post,
            source_conserved: true,
            state_untouched: &untouched,
        };
        let out = RubricEngine::new(&rubric, &UnavailableJudge, VerifyToggles::default()).run(&ctx);
        assert!(out.score.is_none());
        assert_eq!(out.reward_error.as_deref(), Some("judge_unavailable"));
    }

    #[test]
    fn full_contract_happy_path() {
        let rubric = Rubric::new()
            .item(RubricItem::rule(
                "amt",
                0.5,
                RuleCheck::AllText {
                    needles: vec!["$7,183,750.00".into()],
                },
            ))
            .item(RubricItem::judged(
                "struct",
                0.5,
                "q",
                "record \"CASE-FAIRFAX3-2025Q3\"",
            ));
        let mut pod = pod_with_workspace("Record CASE-FAIRFAX3-2025Q3 amount $7,183,750.00.");
        let manifest = manifest_with_verifier();
        let judge = AnchorJudge::new();
        let h = VerifierHarness::new(&rubric, &judge, RewardConfig::default());
        let rollout = AgentRolloutResult {
            final_message: "Record CASE-FAIRFAX3-2025Q3 amount $7,183,750.00.".into(),
            completed: true,
            tool_turns: 3,
        };
        let outcome = h.calculate_reward(&mut pod, &manifest, &rollout);
        assert_eq!(
            outcome,
            RewardOutcome::Valid {
                score: 1.0,
                raw: 1.0,
                results: outcome_results(&outcome)
            }
        );
        // reward.json in the pod carries the reward
        let raw = String::from_utf8(
            pod.read_file(SIDECAR, "/logs/verifier/reward.json")
                .unwrap(),
        )
        .unwrap();
        assert!(raw.contains("\"reward\":1.0"));
        // answer.md was persisted (agent did not write one)
        assert!(pod.read_file(MAIN, "/work/workspace/answer.md").is_some());
    }

    fn outcome_results(o: &RewardOutcome) -> Vec<RubricResult> {
        match o {
            RewardOutcome::Valid { results, .. } => results.clone(),
            _ => panic!("expected valid"),
        }
    }

    #[test]
    fn masking_contract_paths() {
        let rubric = Rubric::new().item(RubricItem::judged("q", 1.0, "q", "\"X\""));
        let judge = AnchorJudge::new();
        let rollout = AgentRolloutResult::default();
        // (a) judge credential missing
        let mut pod = pod_with_workspace("X");
        let m = manifest_with_verifier();
        let cfg = RewardConfig {
            judge_available: false,
            ..Default::default()
        };
        let out =
            VerifierHarness::new(&rubric, &judge, cfg).calculate_reward(&mut pod, &m, &rollout);
        assert!(matches!(
            out,
            RewardOutcome::TestbedCorrupted {
                kind: RewardErrorKind::JudgeKeyMissing,
                ..
            }
        ));
        // (b) dead MCP port
        let mut pod = pod_with_workspace("X");
        let mut m2 = manifest_with_verifier();
        m2.wait_ports = vec![39101];
        let out = VerifierHarness::new(&rubric, &judge, RewardConfig::default())
            .calculate_reward(&mut pod, &m2, &rollout);
        assert!(matches!(
            out,
            RewardOutcome::TestbedCorrupted {
                kind: RewardErrorKind::McpBackendDead,
                ..
            }
        ));
        // (c) missing verifier entry — an EMPTY native rubric (no
        // materials at all) with no uploads and no staged script
        let mut pod = pod_with_workspace("X");
        let mut m3 = manifest_with_verifier();
        m3.verifier.command = "python3 /work/run_verify.py".into();
        m3.verifier.uploads.clear();
        let empty_rubric = Rubric::new();
        let out = VerifierHarness::new(&empty_rubric, &judge, RewardConfig::default())
            .calculate_reward(&mut pod, &m3, &rollout);
        assert!(matches!(
            out,
            RewardOutcome::TestbedCorrupted {
                kind: RewardErrorKind::MissingRunVerify,
                ..
            }
        ));
        // (d) judge unavailable -> reward_error json -> masked
        let mut pod = pod_with_workspace("X");
        let m4 = manifest_with_verifier();
        let out = VerifierHarness::new(&rubric, &UnavailableJudge, RewardConfig::default())
            .calculate_reward(&mut pod, &m4, &rollout);
        assert!(matches!(
            out,
            RewardOutcome::TestbedCorrupted {
                kind: RewardErrorKind::MissingOrInvalidRewardJson,
                ..
            }
        ));
    }

    #[test]
    fn binarization_on_fullscore() {
        let rubric = Rubric::new().item(RubricItem::rule(
            "a",
            1.0,
            RuleCheck::AllText {
                needles: vec!["A".into()],
            },
        ));
        let mut pod = pod_with_workspace("A");
        let m = manifest_with_verifier();
        let judge = AnchorJudge::new();
        let cfg = RewardConfig {
            binary_fullscore: true,
            ..Default::default()
        };
        let out = VerifierHarness::new(&rubric, &judge, cfg.clone()).calculate_reward(
            &mut pod,
            &m,
            &AgentRolloutResult {
                final_message: "A".into(),
                completed: true,
                tool_turns: 0,
            },
        );
        assert_eq!(out.score(), 1.0);
        // partial -> 0
        let rubric2 = Rubric::new()
            .item(RubricItem::rule(
                "a",
                0.5,
                RuleCheck::AllText {
                    needles: vec!["A".into()],
                },
            ))
            .item(RubricItem::rule(
                "b",
                0.5,
                RuleCheck::AllText {
                    needles: vec!["B".into()],
                },
            ));
        let mut pod2 = pod_with_workspace("A");
        let out2 = VerifierHarness::new(&rubric2, &judge, cfg).calculate_reward(
            &mut pod2,
            &m,
            &AgentRolloutResult {
                final_message: "A".into(),
                completed: true,
                tool_turns: 0,
            },
        );
        assert_eq!(out2.score(), 0.0);
    }

    #[test]
    fn cross_check_detects_disagreement() {
        let rubric = Rubric::new().item(RubricItem::judged("q", 1.0, "q", "\"X\""));
        let post = PostState::default();
        let untouched = BTreeMap::new();
        let ctx = EvalCtx {
            agent_output: "X",
            post_state: &post,
            source_conserved: true,
            state_untouched: &untouched,
        };
        let cc = cross_check(&rubric, &AnchorJudge::new(), &UnavailableJudge, &ctx);
        assert!(!cc.both_valid);
        assert!(!cc.agree(0.1));
        let cc2 = cross_check(&rubric, &AnchorJudge::new(), &AnchorJudge::new(), &ctx);
        assert!(cc2.agree(0.0));
    }
}
