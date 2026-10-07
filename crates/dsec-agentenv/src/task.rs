//! Task rows — the verl/HuggingFace dataset format used by
//! `XiaomiMiMo/MiMo-V2.6-RL-oss`.
//!
//! Every domain in the released dataset (code / cyber / general / music /
//! webdev) is one flat table whose rows serialize to exactly this shape:
//!
//! ```json
//! {
//!   "data_source": "opensource-code",
//!   "ability": "swe",
//!   "agent_name": "mimo_swe_agent",
//!   "prompt": [{"role": "user", "content": "<problem statement>"}],
//!   "reward_model": {"style": "rule", "ground_truth": ""},
//!   "extra_info": {
//!     "dataset_type": "opensource-code",
//!     "index": 1,
//!     "instance_id": "format-code-task-001457",
//!     "instance_json": "{\"cwd\": \"/testbed\", \"docker_image\": \"...\", ...}"
//!   }
//! }
//! ```
//!
//! Two representation rules are load-bearing upstream and are preserved
//! here verbatim:
//!
//! * `prompt` is a list of chat messages; SWE-style rows historically
//!   omit the `role` key on the single user turn, so `role` deserializes
//!   optionally and defaults to `"user"`.
//! * the inner task instance lives as a **JSON string** under
//!   `extra_info.instance_json` (kept string-typed to avoid Arrow
//!   struct-union corruption, per the upstream `MimoAgentSWEDataset`
//!   adapter) and is expanded only at rollout time.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

/// One chat message in a task prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `system` / `user` / `assistant`. Absent on legacy SWE rows; defaults
    /// to `user` on deserialize.
    #[serde(default = "default_role", skip_serializing_if = "is_default_role")]
    pub role: String,
    /// Message text.
    pub content: String,
}

fn default_role() -> String {
    "user".to_string()
}

fn is_default_role(role: &str) -> bool {
    role == "user"
}

/// Reward descriptor: style `rule` (verifier-computed) with an optional
/// ground-truth payload. The released dataset pins `style: "rule"` and an
/// empty ground truth for every domain — grading lives in the environment
/// sidecar, not in the row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewardModel {
    /// Reward style; `rule` throughout the released dataset.
    pub style: String,
    /// Ground truth payload (usually empty — the verifier holds the key).
    #[serde(default)]
    pub ground_truth: String,
}

impl Default for RewardModel {
    fn default() -> Self {
        Self {
            style: "rule".into(),
            ground_truth: String::new(),
        }
    }
}

/// The inner task instance, expanded from `extra_info.instance_json`.
///
/// Field-by-field this is the union observed across the five released
/// domains: SWE-style rows carry `cwd` + `docker_image` +
/// `problem_statement`; terminal-bench rows add `agent_timeout_sec` /
/// `allow_internet`; knowledge-work rows carry `env_task_dir`; webdev adds
/// `category`; all fields are optional because no single domain sets them
/// all.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TaskInstance {
    /// Unique instance id (e.g. `format-code-task-001457`, `arvo_35858`).
    #[serde(default)]
    pub instance_id: String,
    /// Dataset family tag (mirrors `extra_info.dataset_type`).
    #[serde(default)]
    pub dataset_type: String,
    /// Docker image reference for the task container.
    #[serde(default)]
    pub docker_image: String,
    /// Working directory inside the container.
    #[serde(default)]
    pub cwd: String,
    /// The problem statement shown to the agent.
    #[serde(default)]
    pub problem_statement: String,
    /// Wall-clock budget for the agent, seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_timeout_sec: Option<f64>,
    /// Whether the rollout may reach the internet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_internet: Option<bool>,
    /// Path to the prepared task directory (knowledge-work envs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_task_dir: Option<String>,
    /// Extra domain-specific fields preserved verbatim.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl TaskInstance {
    /// Parses the `instance_json` string payload.
    pub fn parse(raw: &str) -> Result<Self> {
        Ok(serde_json::from_str(raw)?)
    }

    /// Serializes back into the string payload form.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string(self).expect("task instance serialization is infallible")
    }
}

/// `extra_info` — everything the harness needs beyond the prompt.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExtraInfo {
    /// Monotone index within the dataset shard.
    #[serde(default)]
    pub index: u64,
    /// Instance id (dup of the inner one for flat greppability).
    #[serde(default)]
    pub instance_id: String,
    /// Dataset family tag.
    #[serde(default)]
    pub dataset_type: String,
    /// The inner instance as a JSON string.
    #[serde(default)]
    pub instance_json: String,
    /// Provenance / constraint fields (music: bpm, meter, tag; webdev:
    /// category; ...). Preserved verbatim.
    #[serde(flatten)]
    pub fields: serde_json::Map<String, Value>,
}

/// One row of an agentic RL dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskRow {
    /// Dataset source tag (`opensource-code`, `arvo`, `music`,
    /// `blackbox/webdev`, `mimoagent/terminal_bench`, ...).
    #[serde(default)]
    pub data_source: String,
    /// Ability tag (`swe`, `agent`, `music_generation`, `webdev`, ...).
    #[serde(default)]
    pub ability: String,
    /// Agent harness name (`mimo_swe_agent`, ...).
    #[serde(default)]
    pub agent_name: String,
    /// Prompt messages (usually one user turn).
    pub prompt: Vec<ChatMessage>,
    /// Reward descriptor.
    #[serde(default)]
    pub reward_model: RewardModel,
    /// Harness metadata.
    #[serde(default)]
    pub extra_info: ExtraInfo,
}

impl TaskRow {
    /// Flattens the prompt into the plain task text (messages joined with
    /// blank lines). This is what the rollout shows the agent as the task.
    pub fn task_text(&self) -> String {
        self.prompt
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Expands and parses `extra_info.instance_json`.
    pub fn instance(&self) -> Result<TaskInstance> {
        if self.extra_info.instance_json.is_empty() {
            return Ok(TaskInstance::default());
        }
        TaskInstance::parse(&self.extra_info.instance_json)
    }

    /// The effective agent wall-clock budget, seconds (default 900,
    /// matching `DEFAULT_VERIFIER_TIMEOUT` upstream when unset).
    pub fn agent_timeout(&self) -> f64 {
        self.instance()
            .ok()
            .and_then(|i| i.agent_timeout_sec)
            .unwrap_or(900.0)
    }

    /// Serializes to a single JSONL line.
    pub fn to_jsonl(&self) -> String {
        serde_json::to_string(self).expect("row serialization is infallible")
    }

    /// Builds a minimal row — the builder entry point used by
    /// [`crate::envgen`] and [`crate::repo2rl`].
    pub fn builder(instance_id: &str) -> TaskRowBuilder {
        TaskRowBuilder::new(instance_id)
    }
}

/// Builder for [`TaskRow`] — mirrors how the upstream `build_parquet.py`
/// scripts assemble rows (`to_verl_row`).
pub struct TaskRowBuilder {
    instance_id: String,
    data_source: String,
    ability: String,
    agent_name: String,
    prompt: Vec<ChatMessage>,
    reward_model: RewardModel,
    instance: TaskInstance,
    index: u64,
    fields: serde_json::Map<String, Value>,
}

impl TaskRowBuilder {
    /// New builder for an instance id.
    pub fn new(instance_id: &str) -> Self {
        Self {
            instance_id: instance_id.to_string(),
            data_source: String::new(),
            ability: String::new(),
            agent_name: "mimo_swe_agent".into(),
            prompt: Vec::new(),
            reward_model: RewardModel::default(),
            instance: TaskInstance::default(),
            index: 0,
            fields: serde_json::Map::new(),
        }
    }

    /// Sets the `data_source` metric-grouping tag.
    pub fn data_source(mut self, v: &str) -> Self {
        self.data_source = v.to_string();
        self.instance.dataset_type = v.to_string();
        self
    }

    /// Sets the `ability` tag.
    pub fn ability(mut self, v: &str) -> Self {
        self.ability = v.to_string();
        self
    }

    /// Sets the agent harness name.
    pub fn agent_name(mut self, v: &str) -> Self {
        self.agent_name = v.to_string();
        self
    }

    /// Appends a user-turn message with the task text.
    pub fn user_prompt(mut self, text: &str) -> Self {
        self.prompt.push(ChatMessage {
            role: "user".into(),
            content: text.into(),
        });
        self.instance.problem_statement = text.to_string();
        self
    }

    /// Sets the working directory inside the container.
    pub fn cwd(mut self, v: &str) -> Self {
        self.instance.cwd = v.to_string();
        self
    }

    /// Sets the container image reference.
    pub fn docker_image(mut self, v: &str) -> Self {
        self.instance.docker_image = v.to_string();
        self
    }

    /// Sets the prepared task directory (knowledge-work envs).
    pub fn env_task_dir(mut self, v: &str) -> Self {
        self.instance.env_task_dir = Some(v.to_string());
        self
    }

    /// Sets the agent wall-clock budget, seconds.
    pub fn agent_timeout_sec(mut self, v: f64) -> Self {
        self.instance.agent_timeout_sec = Some(v);
        self
    }

    /// Sets the flat row index.
    pub fn index(mut self, v: u64) -> Self {
        self.index = v;
        self
    }

    /// Adds a provenance field to `extra_info` (music-style constraints,
    /// webdev category, ...).
    pub fn extra_field(mut self, key: &str, value: Value) -> Self {
        self.fields.insert(key.to_string(), value);
        self
    }

    /// Adds a field to the inner instance JSON.
    pub fn instance_field(mut self, key: &str, value: Value) -> Self {
        self.instance.extra.insert(key.to_string(), value);
        self
    }

    /// Builds the row.
    pub fn build(self) -> TaskRow {
        let mut instance = self.instance;
        instance.instance_id = self.instance_id.clone();
        if instance.cwd.is_empty() {
            instance.cwd = "/work/workspace".into();
        }
        let instance_json = instance.to_json_string();
        // NOTE: index / instance_id / dataset_type live as explicit
        // ExtraInfo fields — never duplicated into the flattened map
        // (serde rejects duplicate keys on the way back in).
        let fields = self.fields;
        TaskRow {
            data_source: self.data_source,
            ability: self.ability,
            agent_name: self.agent_name,
            prompt: self.prompt,
            reward_model: self.reward_model,
            extra_info: ExtraInfo {
                index: self.index,
                instance_id: self.instance_id,
                dataset_type: instance.dataset_type,
                instance_json,
                fields,
            },
        }
    }
}

/// A whole dataset shard in memory.
#[derive(Debug, Clone, Default)]
pub struct TaskDataset {
    /// Rows in file order.
    pub rows: Vec<TaskRow>,
}

impl TaskDataset {
    /// Empty dataset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads a JSONL shard (one [`TaskRow`] per line; blank lines ignored).
    pub fn load_jsonl(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_jsonl_str(&text)
    }

    /// Parses JSONL text.
    pub fn from_jsonl_str(text: &str) -> Result<Self> {
        let mut rows = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            rows.push(serde_json::from_str(line)?);
        }
        Ok(Self { rows })
    }

    /// Serializes to JSONL text.
    pub fn to_jsonl(&self) -> String {
        let mut s = String::new();
        for r in &self.rows {
            s.push_str(&r.to_jsonl());
            s.push('\n');
        }
        s
    }

    /// Appends a row.
    pub fn push(&mut self, row: TaskRow) {
        self.rows.push(row);
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// Domain presets for the five released configs, pinned to the values the
/// upstream `build_parquet` scripts and dataset cards use. New tasks built
/// with [`crate::envgen`] default to these tags when the domain matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// Software engineering on real repositories — executable-test verifier.
    Code,
    /// Vulnerability reproduction (ARVO) — rule-check verifier.
    Cyber,
    /// Knowledge work over simulated business systems — rubric judging.
    General,
    /// Symbolic music composition (ABC) — rule checks.
    Music,
    /// Web development — visual grading.
    Webdev,
}

impl Domain {
    /// `(data_source, ability)` tags for the domain.
    pub fn tags(&self) -> (&'static str, &'static str) {
        match self {
            Self::Code => ("opensource-code", "swe"),
            Self::Cyber => ("arvo", "swe"),
            Self::General => ("mimoagent/terminal_bench", "agent"),
            Self::Music => ("music", "music_generation"),
            Self::Webdev => ("blackbox/webdev", "webdev"),
        }
    }

    /// Human-readable verifier family per the dataset card.
    pub fn verifier_family(&self) -> &'static str {
        match self {
            Self::Code => "executable tests",
            Self::Cyber => "rule checks",
            Self::General => "rubric-based judging",
            Self::Music => "rule checks",
            Self::Webdev => "visual grading",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const CODE_ROW: &str = r#"{
      "data_source": "opensource-code",
      "ability": "swe",
      "agent_name": "mimo_swe_agent",
      "prompt": [{"content": "[FEAT] Make `secrets` optional."}],
      "reward_model": {"ground_truth": "", "style": "rule"},
      "extra_info": {
        "dataset_type": "opensource-code",
        "index": 1,
        "instance_id": "format-code-task-001457",
        "instance_json": "{\"cwd\": \"/testbed\", \"dataset_type\": \"opensource-code\", \"docker_image\": \"format-code-task-001457:latest\", \"instance_id\": \"format-code-task-001457\", \"problem_statement\": \"[FEAT] Make `secrets` optional.\"}"
      }
    }"#;

    #[test]
    fn parses_legacy_prompt_without_role() {
        let row: TaskRow = serde_json::from_str(CODE_ROW).unwrap();
        assert_eq!(row.prompt.len(), 1);
        assert_eq!(row.prompt[0].role, "user");
        // Serialization keeps the payload lean: `role` is not re-emitted
        // for the default user turn, matching the released rows.
        let out = row.to_jsonl();
        assert!(out.contains("\"content\":\"[FEAT]"));
        assert!(!out.contains("\"role\":\"user\""));
    }

    #[test]
    fn instance_json_roundtrip() {
        let row: TaskRow = serde_json::from_str(CODE_ROW).unwrap();
        let inst = row.instance().unwrap();
        assert_eq!(inst.cwd, "/testbed");
        assert_eq!(inst.docker_image, "format-code-task-001457:latest");
        assert_eq!(inst.instance_id, "format-code-task-001457");
        let back = inst.to_json_string();
        assert_eq!(TaskInstance::parse(&back).unwrap(), inst);
    }

    #[test]
    fn agent_timeout_defaults_and_overrides() {
        let row: TaskRow = serde_json::from_str(CODE_ROW).unwrap();
        assert_eq!(row.agent_timeout(), 900.0);
        let row = TaskRow::builder("t1")
            .agent_timeout_sec(30.0)
            .user_prompt("hi")
            .build();
        assert_eq!(row.agent_timeout(), 30.0);
    }

    #[test]
    fn builder_produces_schema_complete_rows() {
        let row = TaskRow::builder("arvo_35858")
            .data_source("arvo")
            .ability("swe")
            .cwd("/testbed")
            .docker_image("arvo-rl:v1-arvo-35858")
            .user_prompt("AddressSanitizer: heap-buffer-overflow in extract_name")
            .index(0)
            .extra_field("lang", json!("c"))
            .build();
        assert_eq!(row.data_source, "arvo");
        assert_eq!(row.extra_info.instance_id, "arvo_35858");
        let inst = row.instance().unwrap();
        assert_eq!(inst.docker_image, "arvo-rl:v1-arvo-35858");
        assert_eq!(row.extra_info.fields.get("lang"), Some(&json!("c")));
        // The row must survive a JSON roundtrip without losing fields.
        let rt: TaskRow = serde_json::from_str(&row.to_jsonl()).unwrap();
        assert_eq!(rt, row);
    }

    #[test]
    fn jsonl_dataset_roundtrip() {
        let a = TaskRow::builder("a").user_prompt("A").index(0).build();
        let b = TaskRow::builder("b").user_prompt("B").index(1).build();
        let mut ds = TaskDataset::new();
        ds.push(a);
        ds.push(b);
        let text = ds.to_jsonl();
        let back = TaskDataset::from_jsonl_str(&text).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back.rows[1].extra_info.instance_id, "b");
    }

    #[test]
    fn domain_tags_match_released_dataset() {
        assert_eq!(Domain::Code.tags(), ("opensource-code", "swe"));
        assert_eq!(Domain::Cyber.tags(), ("arvo", "swe"));
        assert_eq!(
            Domain::General.tags(),
            ("mimoagent/terminal_bench", "agent")
        );
        assert_eq!(Domain::Music.tags(), ("music", "music_generation"));
        assert_eq!(Domain::Webdev.tags(), ("blackbox/webdev", "webdev"));
        assert_eq!(Domain::Code.verifier_family(), "executable tests");
        assert_eq!(Domain::General.verifier_family(), "rubric-based judging");
        assert_eq!(Domain::Webdev.verifier_family(), "visual grading");
    }
}
