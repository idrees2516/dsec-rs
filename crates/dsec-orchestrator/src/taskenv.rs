//! Task environments: the sandbox-facing half of the worker container.
//!
//! DSec §6.2 splits a rollout into an **agent sandbox** (the scaffold +
//! tools) and a **worker container** (the scaffold-agnostic control
//! layer). This module is the worker's view of the agent sandbox: the
//! [`TaskEnvironment`] trait, with two real backends:
//!
//! - [`AgentEnvBackend`] — a `dsec-agentenv` pod (EnvSpec-compiled,
//!   MCP tools, native rubric verifier). The pod's `/tmp` volume holds
//!   the **idempotency ledger**, which survives worker crashes exactly
//!   like the real sandbox survives them.
//! - [`DsecSandboxBackend`] — a `dsec-sdk` sandbox on a local cluster
//!   (control plane + edge node + Aether transport + Chronus). Tools
//!   are `exec` / `read_file` / `write_file` / `list_dir`; the ledger
//!   lives in the sandbox filesystem; pause/resume are the real
//!   node-level operations; forks are real `pack_diff` replications
//!   across sandboxes.
//!
//! Both backends implement **idempotent tool calls**: `call_tool(key)`
//! first consults the durable ledger; a key that already executed
//! returns the recorded observation *without re-executing the side
//! effect*. This is the mechanism that makes crash recovery safe (the
//! paper: "completed operations reused recorded results rather than
//! being re-executed, avoiding duplicate side effects from
//! non-idempotent commands") — with the crucial addition that the
//! ledger also covers the crash window *between* execution and journal
//! persistence.
//!
//! Ownership follows the paper's lifecycle: the **sandbox outlives the
//! worker**. A crashed worker hands its environment back
//! ([`crate::worker::WorkerContainer::into_env`]); recovery re-attaches
//! a new worker to the surviving environment.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use dsec_agentenv::agentloop::ToolCallReq;
use dsec_agentenv::live::EnvBundle;
use dsec_agentenv::verifier::{AgentRolloutResult, RewardConfig, RewardOutcome, VerifierHarness};
use dsec_rl::reward::{RewardRecord, RewardValidity};

use crate::error::{Error, Result};
use crate::model::OwnedFut;

/// Borrowing async future type for trait methods.
pub type EnvFut<'a, T> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send + 'a>>;

/// Where the idempotency ledger lives inside every sandbox (both
/// backends resolve it through their own filesystem).
pub const LEDGER_PATH: &str = "/tmp/.dsec-idem-ledger.jsonl";

/// A tool outcome (mirrors the agentenv dispatch contract).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutcome {
    /// Observation text folded back into the transcript.
    pub observation: String,
    /// `ok` / `tool_exception` / `transport_error`.
    pub kind: String,
}

/// Key of a state row for patches.
pub type RowKey = (String, String, String);

/// The branch-point state: files + rows captured when a subagent forked
/// (or when a merge rollback point was taken).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BranchBase {
    /// `path -> bytes` of the agent workspace.
    pub files: BTreeMap<String, Vec<u8>>,
    /// `(system, table, pk) -> row`.
    pub rows: BTreeMap<RowKey, Value>,
}

/// One patch operation returned by a subagent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatchOp {
    /// A file created or modified in the branch.
    FileWrite {
        /// Workspace path.
        path: String,
        /// FNV-1a hash at the branch point (`None` when the file did
        /// not exist) — the three-way-merge base.
        base_hash: Option<u64>,
        /// New content.
        content: Vec<u8>,
    },
    /// A file deleted in the branch.
    FileDelete {
        /// Workspace path.
        path: String,
        /// Hash at the branch point.
        base_hash: u64,
    },
    /// A state row upserted in the branch.
    RowUpsert {
        /// System name.
        system: String,
        /// Table name.
        table: String,
        /// Primary key.
        pk: String,
        /// Row at the branch point (JSON `null` when absent).
        base: Value,
        /// New row.
        new: Value,
    },
}

/// A subagent's workspace patch: what changed in the branch relative to
/// the branch point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspacePatch {
    /// The subagent task id.
    pub subagent: String,
    /// Branch-point state (embedded so the validator can three-way
    /// merge without extra bookkeeping).
    pub base: BranchBase,
    /// Operations in branch order.
    pub ops: Vec<PatchOp>,
}

/// FNV-1a 64 (stable, same as the pod's internal hashing).
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// The sandbox contract the worker container drives.
pub trait TaskEnvironment: Send {
    /// Backend label (`agentenv` / `dsec-sandbox`).
    fn kind(&self) -> &'static str;
    /// The task instruction (user turn).
    fn instruction(&self) -> String;
    /// Re-targets the instruction (subagent forks inherit the parent's
    /// task marker; this replaces it with the branch's sub-task).
    fn set_instruction(&mut self, instruction: &str);
    /// OpenAI function-tool schemas offered to the model.
    fn tool_schemas(&self) -> Vec<Value>;
    /// Executes (or replays from the ledger) one tool call under an
    /// idempotency key. Ledger hits do **not** re-execute.
    fn call_tool(&mut self, call: &ToolCallReq, idem_key: &str) -> EnvFut<'_, ToolOutcome>;
    /// Number of *real* executions performed (ledger hits excluded) —
    /// the duplicate-side-effect detector.
    fn real_exec_count(&self) -> u64;
    /// Verifies the finished rollout.
    fn verify(&mut self, rollout: &AgentRolloutResult) -> EnvFut<'_, RewardRecord>;
    /// Pauses the sandbox (resource reclaim; state preserved).
    fn pause(&mut self) -> EnvFut<'_, ()>;
    /// Resumes the sandbox.
    fn resume(&mut self) -> EnvFut<'_, ()>;
    /// Whether the sandbox is currently paused.
    fn is_paused(&self) -> bool;
    /// Forks an isolated branch workspace sharing the read-only tool
    /// surface (the in-memory `pack_diff`).
    fn fork(&mut self, label: &str) -> EnvFut<'_, Box<dyn TaskEnvironment>>;
    /// Captures the current state (rollback point / merge input).
    fn capture_state(&mut self) -> EnvFut<'_, BranchBase>;
    /// Diffs the current state against `base` into a patch.
    fn patch_since(&mut self, subagent: &str, base: &BranchBase) -> EnvFut<'_, WorkspacePatch>;
    /// Applies patch ops (assumes validation already happened).
    fn apply_patch(&mut self, patch: &WorkspacePatch) -> EnvFut<'_, ()>;
    /// Restores a previously captured state (rollback).
    fn restore_state(&mut self, base: &BranchBase) -> EnvFut<'_, ()>;
    /// Whether workspace sources are still conserved (post-merge
    /// protection check; backends without protection return `true`).
    fn sources_conserved(&self) -> bool;
    /// Destroys the environment.
    fn destroy(&mut self) -> EnvFut<'_, ()>;
}

// ---------------------------------------------------------------------------
// AgentEnv backend
// ---------------------------------------------------------------------------

/// The `dsec-agentenv` pod backend.
pub struct AgentEnvBackend {
    /// The booted environment bundle.
    pub(crate) bundle: EnvBundle,
    judge: Box<dyn dsec_agentenv::verifier::Judge>,
    paused: bool,
    exec_count: u64,
}

impl AgentEnvBackend {
    /// Wraps a booted bundle.
    pub fn new(bundle: EnvBundle) -> Self {
        Self::with_judge(
            bundle,
            Box::new(dsec_agentenv::verifier::AnchorJudge::new()),
        )
    }

    /// Wraps a booted bundle with a custom judge.
    pub fn with_judge(bundle: EnvBundle, judge: Box<dyn dsec_agentenv::verifier::Judge>) -> Self {
        Self {
            bundle,
            judge,
            paused: false,
            exec_count: 0,
        }
    }

    fn ledger_read(&self) -> BTreeMap<String, (String, String)> {
        read_ledger_from(&self.bundle.pod)
    }

    fn ledger_append(
        &mut self,
        key: &str,
        call: &ToolCallReq,
        outcome: &ToolOutcome,
    ) -> Result<()> {
        let entry = serde_json::json!({
            "key": key,
            "tool": call.qualified(),
            "params": call.params,
            "observation": outcome.observation,
            "kind": outcome.kind,
        });
        let mut text = String::new();
        if let Some(existing) = self
            .bundle
            .pod
            .read_file(dsec_agentenv::manifest::MAIN, LEDGER_PATH)
        {
            text.push_str(&String::from_utf8_lossy(&existing));
            if !text.is_empty() && !text.ends_with('\n') {
                text.push('\n');
            }
        }
        text.push_str(&serde_json::to_string(&entry)?);
        text.push('\n');
        self.bundle
            .pod
            .write_file(dsec_agentenv::manifest::MAIN, LEDGER_PATH, text.as_bytes())?;
        Ok(())
    }

    fn workspace_files(&self) -> BTreeMap<String, Vec<u8>> {
        let mut out = BTreeMap::new();
        if let Some(vol) = self
            .bundle
            .pod
            .volumes
            .get(dsec_agentenv::topology::vols::WORKSPACE)
        {
            for (k, e) in &vol.files {
                if !e.is_dir {
                    out.insert(k.trim_start_matches('/').to_string(), e.content.clone());
                }
            }
        }
        out
    }

    fn state_rows(&self) -> BTreeMap<RowKey, Value> {
        let mut out = BTreeMap::new();
        for (sys, db) in &self.bundle.pod.state {
            for table in db.table_names() {
                let Some(pk_col) = db.primary_key(&table) else {
                    continue;
                };
                for row in db.select_all(&table) {
                    let pk = row
                        .get(&pk_col)
                        .and_then(|v| v.as_str().map(|s| s.to_string()))
                        .unwrap_or_default();
                    out.insert((sys.clone(), table.clone(), pk), serde_json::json!(row));
                }
            }
        }
        out
    }

    fn write_workspace_file(&mut self, path: &str, content: &[u8]) -> Result<()> {
        let full = format!("/work/workspace/{}", path.trim_start_matches('/'));
        self.bundle
            .pod
            .write_file(dsec_agentenv::manifest::MAIN, &full, content)
            .map_err(Error::AgentEnv)
    }

    fn remove_workspace_file(&mut self, path: &str) {
        let rel = path.trim_start_matches('/').to_string();
        if let Some(vol) = self
            .bundle
            .pod
            .volumes
            .get_mut(dsec_agentenv::topology::vols::WORKSPACE)
        {
            vol.rm(&rel);
        }
    }

    fn set_row(&mut self, system: &str, table: &str, pk: &str, row: &Value) -> Result<()> {
        let Some(db) = self.bundle.pod.state.get_mut(system) else {
            return Err(Error::Env(format!("no system {system}")));
        };
        let obj = row
            .as_object()
            .ok_or_else(|| Error::Env("row is not an object".into()))?
            .clone();
        if db.get(table, pk).is_some() {
            // strip the primary key: update() rejects pk patches
            let pk_col = db.primary_key(table).unwrap_or_default();
            let mut patch = obj.clone();
            patch.remove(&pk_col);
            db.update(table, pk, patch).map_err(Error::Env)?;
        } else {
            db.insert(table, obj).map_err(Error::Env)?;
        }
        Ok(())
    }
}

fn read_ledger_from(pod: &dsec_agentenv::topology::SimPod) -> BTreeMap<String, (String, String)> {
    let mut out = BTreeMap::new();
    if let Some(bytes) = pod.read_file(dsec_agentenv::manifest::MAIN, LEDGER_PATH) {
        if let Ok(text) = String::from_utf8(bytes) {
            for line in text.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<Value>(line) {
                    let key = v["key"].as_str().unwrap_or_default().to_string();
                    let obs = v["observation"].as_str().unwrap_or_default().to_string();
                    let kind = v["kind"].as_str().unwrap_or("ok").to_string();
                    out.insert(key, (obs, kind));
                }
            }
        }
    }
    out
}

impl TaskEnvironment for AgentEnvBackend {
    fn kind(&self) -> &'static str {
        "agentenv"
    }

    fn instruction(&self) -> String {
        self.bundle
            .pod
            .read_file(dsec_agentenv::manifest::MAIN, "/tmp/.dsec-task.txt")
            .and_then(|b| String::from_utf8(b).ok())
            .unwrap_or_default()
    }

    fn set_instruction(&mut self, instruction: &str) {
        let _ = self.bundle.pod.write_file(
            dsec_agentenv::manifest::MAIN,
            "/tmp/.dsec-task.txt",
            instruction.as_bytes(),
        );
    }

    fn tool_schemas(&self) -> Vec<Value> {
        self.bundle
            .tools
            .iter()
            .map(|(_, t)| t.to_function_tool())
            .collect()
    }

    fn call_tool(&mut self, call: &ToolCallReq, idem_key: &str) -> EnvFut<'_, ToolOutcome> {
        let call = call.clone();
        let idem_key = idem_key.to_string();
        Box::pin(async move {
            // transparent resume (paper §6.3: requests to a paused
            // sandbox resume it)
            self.paused = false;
            // ledger hit? replay the recorded observation — never
            // re-execute
            if let Some((obs, kind)) = self.ledger_read().get(&idem_key) {
                return Ok(ToolOutcome {
                    observation: obs.clone(),
                    kind: kind.clone(),
                });
            }
            let outcome =
                match self
                    .bundle
                    .pod
                    .call_tool(&call.server, &call.tool, call.params.clone())
                {
                    Ok(v) => {
                        let is_err = v.get("error").is_some();
                        ToolOutcome {
                            observation: dsec_agentenv::agentloop::truncate_observation(
                                &v.to_string(),
                                4000,
                            ),
                            kind: if is_err {
                                "tool_exception".into()
                            } else {
                                "ok".into()
                            },
                        }
                    }
                    Err(e) => ToolOutcome {
                        observation: dsec_agentenv::agentloop::truncate_observation(
                            &e.to_string(),
                            4000,
                        ),
                        kind: "transport_error".into(),
                    },
                };
            self.exec_count += 1;
            self.ledger_append(&idem_key, &call, &outcome)?;
            Ok(outcome)
        })
    }

    fn real_exec_count(&self) -> u64 {
        self.exec_count
    }

    fn verify(&mut self, rollout: &AgentRolloutResult) -> EnvFut<'_, RewardRecord> {
        let rollout = rollout.clone();
        Box::pin(async move {
            let harness = VerifierHarness::new(
                &self.bundle.rubric,
                self.judge.as_ref(),
                RewardConfig::default(),
            );
            let outcome =
                harness.calculate_reward(&mut self.bundle.pod, &self.bundle.manifest, &rollout);
            Ok(reward_record_from(outcome))
        })
    }

    fn pause(&mut self) -> EnvFut<'_, ()> {
        Box::pin(async move {
            // in-process pod: pause is logical; the dsec backend does
            // the real resource reclaim
            self.paused = true;
            Ok(())
        })
    }

    fn resume(&mut self) -> EnvFut<'_, ()> {
        Box::pin(async move {
            self.paused = false;
            Ok(())
        })
    }

    fn is_paused(&self) -> bool {
        self.paused
    }

    fn fork(&mut self, label: &str) -> EnvFut<'_, Box<dyn TaskEnvironment>> {
        let label = label.to_string();
        Box::pin(async move {
            let pod = self.bundle.pod.fork(&label);
            let bundle = EnvBundle {
                pod,
                manifest: self.bundle.manifest.clone(),
                rubric: self.bundle.rubric.clone(),
                tools: self.bundle.tools.clone(),
            };
            let child = AgentEnvBackend::new(bundle);
            Ok(Box::new(child) as Box<dyn TaskEnvironment>)
        })
    }

    fn capture_state(&mut self) -> EnvFut<'_, BranchBase> {
        Box::pin(async move {
            Ok(BranchBase {
                files: self.workspace_files(),
                rows: self.state_rows(),
            })
        })
    }

    fn patch_since(&mut self, subagent: &str, base: &BranchBase) -> EnvFut<'_, WorkspacePatch> {
        let subagent = subagent.to_string();
        let base = base.clone();
        Box::pin(async move {
            let current_files = self.workspace_files();
            let current_rows = self.state_rows();
            let mut ops = Vec::new();
            for (path, content) in &current_files {
                let base_hash = base.files.get(path).map(|b| fnv1a(b));
                if base_hash != Some(fnv1a(content)) {
                    ops.push(PatchOp::FileWrite {
                        path: path.clone(),
                        base_hash,
                        content: content.clone(),
                    });
                }
            }
            for (path, bytes) in &base.files {
                if !current_files.contains_key(path) {
                    ops.push(PatchOp::FileDelete {
                        path: path.clone(),
                        base_hash: fnv1a(bytes),
                    });
                }
            }
            for (key, row) in &current_rows {
                let base_row = base.rows.get(key).cloned().unwrap_or(Value::Null);
                if &base_row != row {
                    ops.push(PatchOp::RowUpsert {
                        system: key.0.clone(),
                        table: key.1.clone(),
                        pk: key.2.clone(),
                        base: base_row,
                        new: row.clone(),
                    });
                }
            }
            Ok(WorkspacePatch {
                subagent,
                base,
                ops,
            })
        })
    }

    fn apply_patch(&mut self, patch: &WorkspacePatch) -> EnvFut<'_, ()> {
        let patch = patch.clone();
        Box::pin(async move {
            for op in &patch.ops {
                match op {
                    PatchOp::FileWrite { path, content, .. } => {
                        self.write_workspace_file(path, content)?;
                    }
                    PatchOp::FileDelete { path, .. } => {
                        self.remove_workspace_file(path);
                    }
                    PatchOp::RowUpsert {
                        system,
                        table,
                        pk,
                        new,
                        ..
                    } => {
                        self.set_row(system, table, pk, new)?;
                    }
                }
            }
            Ok(())
        })
    }

    fn restore_state(&mut self, base: &BranchBase) -> EnvFut<'_, ()> {
        let base = base.clone();
        Box::pin(async move {
            let current = self.workspace_files();
            for path in current.keys() {
                if !base.files.contains_key(path) {
                    self.remove_workspace_file(path);
                }
            }
            for (path, bytes) in &base.files {
                self.write_workspace_file(path, bytes)?;
            }
            let current_rows = self.state_rows();
            for key in current_rows.keys() {
                if !base.rows.contains_key(key) {
                    if let Some(db) = self.bundle.pod.state.get_mut(&key.0) {
                        let _ = db.delete(&key.1, &key.2);
                    }
                }
            }
            for (key, row) in &base.rows {
                self.set_row(&key.0, &key.1, &key.2, row)?;
            }
            Ok(())
        })
    }

    fn sources_conserved(&self) -> bool {
        self.bundle.pod.source_conserved()
    }

    fn destroy(&mut self) -> EnvFut<'_, ()> {
        Box::pin(async move { Ok(()) })
    }
}

/// Maps the agentenv two-arm reward onto the dsec-rl validity tri-state.
pub fn reward_record_from(outcome: RewardOutcome) -> RewardRecord {
    match outcome {
        RewardOutcome::Valid { score, .. } => RewardRecord {
            execution_id: 0,
            group_id: 0,
            raw: score as f32,
            validity: if score > 0.0 {
                RewardValidity::Valid
            } else {
                RewardValidity::Invalid
            },
        },
        RewardOutcome::TestbedCorrupted { .. } => RewardRecord {
            execution_id: 0,
            group_id: 0,
            raw: f32::NAN,
            validity: RewardValidity::EvaluatorFailure,
        },
    }
}

// ---------------------------------------------------------------------------
// File tasks + dsec-sdk sandbox backend
// ---------------------------------------------------------------------------

/// One file-content check for a file task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCheck {
    /// Sandbox path to read.
    pub path: String,
    /// Needle that must appear in the file.
    pub needle: String,
}

/// A file-manipulation task for the dsec sandbox backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileTask {
    /// Task name.
    pub name: String,
    /// The instruction shown to the model.
    pub instruction: String,
    /// Initial files `(path, bytes)` staged into the sandbox.
    pub files: Vec<(String, Vec<u8>)>,
    /// Post-conditions checked at verify time.
    pub checks: Vec<FileCheck>,
}

impl FileTask {
    /// A minimal task: write `content` to `path`.
    pub fn write_file_task(name: &str, path: &str, content: &str) -> Self {
        Self {
            name: name.to_string(),
            instruction: format!(
                "Write exactly the following content to {path} using the write_file tool, \
                 then reply DONE.\n\n{content}"
            ),
            files: vec![],
            checks: vec![FileCheck {
                path: path.to_string(),
                needle: content.to_string(),
            }],
        }
    }
}

/// The dsec-sdk sandbox backend: real control plane, real Aether
/// transport, real node-level pause/resume, real cross-sandbox
/// pack_diff forks.
pub struct DsecSandboxBackend {
    task: FileTask,
    sandbox: Option<dsec_sdk::Sandbox>,
    client: Arc<dsec_sdk::DsecClient>,
    nodes: Vec<Arc<dsec_runtime::EdgeNode>>,
    spec: dsec_control::model::SandboxSpec,
    paused: bool,
    exec_count: u64,
    known_paths: std::collections::BTreeSet<String>,
}

impl DsecSandboxBackend {
    /// Creates the backend over an already-provisioned sandbox (staging
    /// the task's initial files into it).
    pub async fn provision(
        task: FileTask,
        cluster: &dsec_sdk::integration::LocalCluster,
        spec: dsec_control::model::SandboxSpec,
    ) -> Result<Self> {
        let client = Arc::new(cluster.client());
        let sandbox = client.create_sandbox(spec.clone()).await?;
        for (path, bytes) in &task.files {
            sandbox.write_file(path, bytes).await?;
        }
        let known_paths = task.files.iter().map(|(p, _)| p.clone()).collect();
        Ok(Self {
            task,
            sandbox: Some(sandbox),
            client,
            nodes: cluster.nodes.clone(),
            spec,
            paused: false,
            exec_count: 0,
            known_paths,
        })
    }

    /// The live sandbox handle (for direct inspection).
    pub fn sandbox(&self) -> Result<&dsec_sdk::Sandbox> {
        self.sandbox
            .as_ref()
            .ok_or_else(|| Error::Env("sandbox destroyed".into()))
    }

    fn sbx(&self) -> Result<&dsec_sdk::Sandbox> {
        self.sandbox
            .as_ref()
            .ok_or_else(|| Error::Env("sandbox destroyed".into()))
    }

    async fn ledger_read(&self) -> BTreeMap<String, (String, String)> {
        let mut out = BTreeMap::new();
        let Ok(sbx) = self.sbx() else { return out };
        if let Ok(bytes) = sbx.read_file(LEDGER_PATH).await {
            if let Ok(text) = String::from_utf8(bytes) {
                for line in text.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    if let Ok(v) = serde_json::from_str::<Value>(line) {
                        let key = v["key"].as_str().unwrap_or_default().to_string();
                        let obs = v["observation"].as_str().unwrap_or_default().to_string();
                        let kind = v["kind"].as_str().unwrap_or("ok").to_string();
                        out.insert(key, (obs, kind));
                    }
                }
            }
        }
        out
    }

    async fn current_files(&self) -> BTreeMap<String, Vec<u8>> {
        let mut out = BTreeMap::new();
        let Ok(sbx) = self.sbx() else { return out };
        for path in &self.known_paths {
            if let Ok(bytes) = sbx.read_file(path).await {
                out.insert(path.clone(), bytes);
            }
        }
        out
    }

    fn tools_json() -> Vec<Value> {
        vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "exec",
                    "description": "Run a shell command in the sandbox",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from the sandbox",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file in the sandbox",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List a directory in the sandbox",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                }
            }),
        ]
    }
}

impl TaskEnvironment for DsecSandboxBackend {
    fn kind(&self) -> &'static str {
        "dsec-sandbox"
    }

    fn instruction(&self) -> String {
        self.task.instruction.clone()
    }

    fn set_instruction(&mut self, instruction: &str) {
        self.task.instruction = instruction.to_string();
    }

    fn tool_schemas(&self) -> Vec<Value> {
        Self::tools_json()
    }

    fn call_tool(&mut self, call: &ToolCallReq, idem_key: &str) -> EnvFut<'_, ToolOutcome> {
        let call = call.clone();
        let idem_key = idem_key.to_string();
        Box::pin(async move {
            if let Some((obs, kind)) = self.ledger_read().await.get(&idem_key) {
                return Ok(ToolOutcome {
                    observation: obs.clone(),
                    kind: kind.clone(),
                });
            }
            let mut touched: Option<String> = None;
            let outcome = match call.tool.as_str() {
                "exec" => {
                    let cmd = call
                        .params
                        .get("cmd")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    let res = self.sbx()?.execute_with_timeout(cmd, Some(30_000)).await?;
                    ToolOutcome {
                        observation: dsec_agentenv::agentloop::truncate_observation(
                            &format!(
                                "exit={}\nstdout:\n{}\nstderr:\n{}",
                                res.exit_code,
                                String::from_utf8_lossy(&res.stdout),
                                String::from_utf8_lossy(&res.stderr)
                            ),
                            4000,
                        ),
                        kind: if res.exit_code == 0 {
                            "ok".into()
                        } else {
                            "tool_exception".into()
                        },
                    }
                }
                "read_file" => {
                    let path = call
                        .params
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    let bytes = self.sbx()?.read_file(path).await?;
                    touched = Some(path.to_string());
                    ToolOutcome {
                        observation: dsec_agentenv::agentloop::truncate_observation(
                            &String::from_utf8_lossy(&bytes),
                            4000,
                        ),
                        kind: "ok".into(),
                    }
                }
                "write_file" => {
                    let path = call
                        .params
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    let content = call
                        .params
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    self.sbx()?.write_file(path, content.as_bytes()).await?;
                    touched = Some(path.to_string());
                    ToolOutcome {
                        observation: format!("wrote {} bytes to {path}", content.len()),
                        kind: "ok".into(),
                    }
                }
                "list_dir" => {
                    let path = call
                        .params
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    let entries = self.sbx()?.list_dir(path).await?;
                    ToolOutcome {
                        observation: entries.join("\n"),
                        kind: "ok".into(),
                    }
                }
                other => ToolOutcome {
                    observation: format!("unknown tool {other}"),
                    kind: "tool_exception".into(),
                },
            };
            if let Some(p) = touched {
                self.known_paths.insert(p);
            }
            self.exec_count += 1;
            // persist to the durable ledger inside the sandbox
            let entry = serde_json::json!({
                "key": idem_key,
                "tool": call.qualified(),
                "params": call.params,
                "observation": outcome.observation,
                "kind": outcome.kind,
            });
            if let Ok(sbx) = self.sbx() {
                let _ = sbx
                    .append_file(
                        LEDGER_PATH,
                        format!("{}\n", serde_json::to_string(&entry)?).as_bytes(),
                    )
                    .await;
            }
            Ok(outcome)
        })
    }

    fn real_exec_count(&self) -> u64 {
        self.exec_count
    }

    fn verify(&mut self, _rollout: &AgentRolloutResult) -> EnvFut<'_, RewardRecord> {
        Box::pin(async move {
            let mut passed = 0usize;
            let mut total = 0usize;
            let mut infra_failure = false;
            for check in &self.task.checks {
                total += 1;
                match self.sbx()?.read_file(&check.path).await {
                    Ok(bytes) => {
                        if String::from_utf8_lossy(&bytes).contains(&check.needle) {
                            passed += 1;
                        }
                    }
                    Err(_) => {
                        // a missing check file is a genuine fail; a
                        // transport-level failure is infra — the sandbox
                        // distinguishes them, we conservatively mark
                        // infra only when the sandbox itself is gone
                        if self.sandbox.is_none() {
                            infra_failure = true;
                        }
                    }
                }
            }
            let record = if infra_failure {
                RewardRecord::evaluator_failure(0, 0)
            } else if total == 0 {
                RewardRecord::valid(0, 0, 1.0)
            } else {
                RewardRecord::valid(0, 0, passed as f32 / total as f32)
            };
            Ok(record)
        })
    }

    fn pause(&mut self) -> EnvFut<'_, ()> {
        Box::pin(async move {
            let sbx = self.sbx()?;
            sbx.pause().await?;
            self.paused = true;
            Ok(())
        })
    }

    fn resume(&mut self) -> EnvFut<'_, ()> {
        Box::pin(async move {
            if self.paused {
                let sbx = self.sbx()?;
                sbx.resume().await?;
                self.paused = false;
            }
            Ok(())
        })
    }

    fn is_paused(&self) -> bool {
        self.paused
    }

    fn fork(&mut self, _label: &str) -> EnvFut<'_, Box<dyn TaskEnvironment>> {
        let _label = _label.to_string();
        Box::pin(async move {
            // provision a fresh sandbox, then pack_diff the parent's
            // writable state onto it (the paper's cross-node fast path)
            let (parent_sid, parent_node_id) = {
                let parent = self.sbx()?;
                (parent.sid(), parent.node_id().to_string())
            };
            let child_sandbox = self.client.create_sandbox(self.spec.clone()).await?;
            let src_node = self
                .nodes
                .iter()
                .find(|n| n.node_id == parent_node_id)
                .cloned()
                .ok_or_else(|| Error::Env("parent node not found".into()))?;
            let dst_node = self
                .nodes
                .iter()
                .find(|n| n.node_id == child_sandbox.node_id())
                .cloned()
                .ok_or_else(|| Error::Env("child node not found".into()))?;
            let src_entry = src_node
                .lookup(parent_sid)
                .ok_or_else(|| Error::Env("parent entry not found".into()))?;
            let dst_entry = dst_node
                .lookup(child_sandbox.sid())
                .ok_or_else(|| Error::Env("child entry not found".into()))?;
            let pack = src_entry.instance.snapshot_diff(1);
            dst_entry
                .instance
                .apply_diff(&pack)
                .map_err(|e| Error::Env(format!("apply_diff: {e}")))?;
            // the parent's ledger and files ride along in the diff
            let child = DsecSandboxBackend {
                task: self.task.clone(),
                sandbox: Some(child_sandbox),
                client: self.client.clone(),
                nodes: self.nodes.clone(),
                spec: self.spec.clone(),
                paused: false,
                exec_count: 0,
                known_paths: self.known_paths.clone(),
            };
            Ok(Box::new(child) as Box<dyn TaskEnvironment>)
        })
    }

    fn capture_state(&mut self) -> EnvFut<'_, BranchBase> {
        Box::pin(async move {
            Ok(BranchBase {
                files: self.current_files().await,
                rows: BTreeMap::new(),
            })
        })
    }

    fn patch_since(&mut self, subagent: &str, base: &BranchBase) -> EnvFut<'_, WorkspacePatch> {
        let subagent = subagent.to_string();
        let base = base.clone();
        Box::pin(async move {
            let current = self.current_files().await;
            let mut ops = Vec::new();
            for (path, content) in &current {
                let base_hash = base.files.get(path).map(|b| fnv1a(b));
                if base_hash.is_none() || base_hash != Some(fnv1a(content)) {
                    ops.push(PatchOp::FileWrite {
                        path: path.clone(),
                        base_hash,
                        content: content.clone(),
                    });
                }
            }
            for (path, bytes) in &base.files {
                if !current.contains_key(path) {
                    ops.push(PatchOp::FileDelete {
                        path: path.clone(),
                        base_hash: fnv1a(bytes),
                    });
                }
            }
            Ok(WorkspacePatch {
                subagent,
                base,
                ops,
            })
        })
    }

    fn apply_patch(&mut self, patch: &WorkspacePatch) -> EnvFut<'_, ()> {
        let patch = patch.clone();
        Box::pin(async move {
            for op in &patch.ops {
                if let PatchOp::FileWrite { path, .. } = op {
                    self.known_paths.insert(path.clone());
                }
            }
            for op in &patch.ops {
                match op {
                    PatchOp::FileWrite { path, content, .. } => {
                        self.sbx()?.write_file(path, content).await?;
                    }
                    PatchOp::FileDelete { path, .. } => {
                        self.sbx()?.rm(path).await?;
                    }
                    PatchOp::RowUpsert { .. } => {
                        return Err(Error::Env(
                            "row patches do not apply to file-task sandboxes".into(),
                        ));
                    }
                }
            }
            Ok(())
        })
    }

    fn restore_state(&mut self, base: &BranchBase) -> EnvFut<'_, ()> {
        let base = base.clone();
        Box::pin(async move {
            let current = self.current_files().await;
            for path in current.keys() {
                if !base.files.contains_key(path) {
                    self.sbx()?.rm(path).await?;
                }
            }
            for (path, bytes) in &base.files {
                self.sbx()?.write_file(path, bytes).await?;
            }
            Ok(())
        })
    }

    fn sources_conserved(&self) -> bool {
        true // file tasks have no protected sources
    }

    fn destroy(&mut self) -> EnvFut<'_, ()> {
        Box::pin(async move {
            if let Some(sbx) = self.sandbox.take() {
                sbx.destroy().await?;
            }
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

/// A task specification a factory can build.
#[derive(Debug, Clone)]
pub enum TaskSpec {
    /// An agentenv pod task (EnvSpec DSL).
    AgentEnv {
        /// The environment spec.
        spec: dsec_agentenv::envgen::EnvSpec,
        /// Task instruction override (defaults to the spec's).
        instruction: Option<String>,
    },
}

/// Builds task environments.
pub trait EnvFactory: Send + Sync {
    /// Builds a fresh environment for a task.
    fn build(&self, task: &TaskSpec) -> OwnedFut<Result<Box<dyn TaskEnvironment>>>;
}

/// Builds agentenv pods from specs.
#[derive(Default)]
pub struct AgentEnvFactory;

impl EnvFactory for AgentEnvFactory {
    fn build(&self, task: &TaskSpec) -> OwnedFut<Result<Box<dyn TaskEnvironment>>> {
        let task = task.clone();
        Box::pin(async move {
            match task {
                TaskSpec::AgentEnv { spec, instruction } => {
                    let bundle = spec.compile()?.boot()?;
                    let mut env = AgentEnvBackend::new(bundle);
                    let instr = instruction.unwrap_or_else(|| spec.instruction.clone());
                    env.bundle.pod.write_file(
                        dsec_agentenv::manifest::MAIN,
                        "/tmp/.dsec-task.txt",
                        instr.as_bytes(),
                    )?;
                    Ok(Box::new(env) as Box<dyn TaskEnvironment>)
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_stable() {
        assert_eq!(fnv1a(b"abc"), fnv1a(b"abc"));
        assert_ne!(fnv1a(b"abc"), fnv1a(b"abd"));
    }
}
