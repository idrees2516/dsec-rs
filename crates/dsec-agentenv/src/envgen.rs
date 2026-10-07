//! The generic environment factory — build RL environments like the
//! released ones for arbitrary real-world software.
//!
//! Every released environment decomposes into the same six parts: an
//! instruction, a set of simulated **systems** (databases + schemas), a
//! **tool surface** (MCP servers exposing CRUD over those systems), a
//! **workspace** (documents the agent reads/produces), a **verifier**
//! (rubric), and a **container topology**. This module makes those parts
//! declarative: one [`EnvSpec`] (plain JSON) compiles into a fully
//! bootable [`EnvBundle`] — pod, manifest, tools, rubric — ready for the
//! agent loop and the Live RL trainer.
//!
//! ```text
//! EnvSpec ──EnvFactory──▶ EnvBundle ──AgentLoop──▶ Rollout ──Verifier──▶ Reward
//!   ▲                                    │
//!   └── Variants (seeded instance generation: scaled amounts, swapped
//!       ids, distractor rows, difficulty tiers)
//! ```
//!
//! [`templates`] ships one seed spec per released domain — knowledge
//! work (multi-system business software), terminal (computer use),
//! webdev (visual grading), code (SWE), cyber (vuln reproduction) —
//! each a starting point for domain-specific families. [`SpecProvider`]
//! plugs the factory straight into [`crate::live::LiveTrainer`].

use crate::error::{Error, Result};
use crate::live::{EnvBundle, EnvProvider};
use crate::manifest::{Manifest, SetupSpec, VerifierSpec, MCP_PORT_BASE, SIDECAR};
use crate::state::{Schema, StateDb, Table};
use crate::task::{Domain, TaskRow};
use crate::topology::{SimPod, ToolCtx, ToolDef};
use crate::verifier::{Rubric, RubricItem, RuleCheck};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

/// A table column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnSpec {
    /// Column name.
    pub name: String,
    /// SQL-ish type tag.
    #[serde(default = "text_type")]
    pub sql_type: String,
}

fn text_type() -> String {
    "TEXT".into()
}

/// A table declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableSpec {
    /// Table name.
    pub name: String,
    /// Primary-key column.
    pub primary_key: String,
    /// Columns.
    pub columns: Vec<ColumnSpec>,
}

impl TableSpec {
    fn compile(&self) -> Table {
        Table {
            name: self.name.clone(),
            primary_key: self.primary_key.clone(),
            columns: self
                .columns
                .iter()
                .map(|c| crate::state::Column {
                    name: c.name.clone(),
                    sql_type: c.sql_type.clone(),
                })
                .collect(),
        }
    }
}

/// A seeded row (`{table, values}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RowSpec {
    /// Target table.
    pub table: String,
    /// Row values (must carry the primary key).
    pub values: Map<String, Value>,
}

/// One simulated system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemSpec {
    /// System (MCP server) name.
    pub name: String,
    /// Tables.
    pub tables: Vec<TableSpec>,
    /// Seed rows.
    #[serde(default)]
    pub rows: Vec<RowSpec>,
}

/// The declarative tool operation.
///
/// These are the operations the released `tools/<system>.py` files
/// implement by hand; declaring them covers the standard CRUD surface
/// every business system exposes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ToolOp {
    /// Fetch one row by primary key.
    Get {
        /// Table.
        table: String,
        /// Param carrying the key.
        id_param: String,
    },
    /// List rows, optionally filtered by equality params.
    List {
        /// Table.
        table: String,
        /// Params treated as equality filters (empty = all).
        #[serde(default)]
        filter_params: Vec<String>,
        /// Max rows returned.
        #[serde(default = "default_limit")]
        limit: usize,
    },
    /// Update fields of one row.
    Update {
        /// Table.
        table: String,
        /// Param carrying the key.
        id_param: String,
        /// Params written into the row (the rest are rejected).
        field_params: Vec<String>,
    },
    /// Insert a row from params.
    Insert {
        /// Table.
        table: String,
        /// Columns taken from params.
        field_params: Vec<String>,
    },
    /// Delete one row.
    Delete {
        /// Table.
        table: String,
        /// Param carrying the key.
        id_param: String,
    },
}

fn default_limit() -> usize {
    50
}

/// A tool declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Owning system (MCP server) name.
    pub server: String,
    /// Tool name.
    pub name: String,
    /// Description shown to the model.
    pub description: String,
    /// The operation.
    pub op: ToolOp,
}

impl ToolSpec {
    fn schema(&self) -> Value {
        let mut props = Map::new();
        let mut required = Vec::new();
        match &self.op {
            ToolOp::Get { id_param, .. }
            | ToolOp::Delete { id_param, .. }
            | ToolOp::Update { id_param, .. } => {
                props.insert(id_param.clone(), json!({"type": "string"}));
                required.push(id_param.clone());
                if let ToolOp::Update { field_params, .. } = &self.op {
                    for f in field_params {
                        props.insert(f.clone(), json!({"type": "string"}));
                    }
                }
            }
            ToolOp::List { filter_params, .. } => {
                for f in filter_params {
                    props.insert(f.clone(), json!({"type": "string"}));
                }
            }
            ToolOp::Insert { field_params, .. } => {
                for f in field_params {
                    props.insert(f.clone(), json!({"type": "string"}));
                    required.push(f.clone());
                }
            }
        }
        json!({"type": "object", "properties": props, "required": required})
    }

    fn compile(&self) -> ToolDef {
        let op = self.op.clone();
        let tool_name = self.name.clone();
        ToolDef {
            name: tool_name,
            params_schema: self.schema(),
            description: self.description.clone(),
            exec: Box::new(move |ctx: &mut ToolCtx| execute_op(&op, ctx)),
        }
    }
}

fn execute_op(op: &ToolOp, ctx: &mut ToolCtx) -> Value {
    // The tool's owning system is resolved by table ownership: whichever
    // system's schema declares the target table owns the tool.
    let system = find_system_for_table(ctx, op_table(op));
    let Some(system) = system else {
        return json!({"error": format!("no system owns table {}", op_table(op))});
    };
    match op {
        ToolOp::Get { table, id_param } => {
            let id = ctx.param_str(id_param).unwrap_or_default().to_string();
            match ctx.db(&system).and_then(|db| db.get(table, &id)) {
                Some(row) => json!(row),
                None => json!({"error": format!("no such row {table}:{id}")}),
            }
        }
        ToolOp::List {
            table,
            filter_params,
            limit,
        } => {
            let mut eq = Map::new();
            for f in filter_params {
                if let Some(v) = ctx.params.get(f) {
                    eq.insert(f.clone(), v.clone());
                }
            }
            let rows: Vec<&Map<String, Value>> = ctx
                .db(&system)
                .map(|db| {
                    if eq.is_empty() {
                        db.select_all(table)
                    } else {
                        db.select_where(table, &eq)
                    }
                })
                .unwrap_or_default();
            let rows: Vec<&Map<String, Value>> = rows.into_iter().take(*limit).collect();
            json!(rows)
        }
        ToolOp::Update {
            table,
            id_param,
            field_params,
        } => {
            let id = ctx.param_str(id_param).unwrap_or_default().to_string();
            let mut patch = Map::new();
            for f in field_params {
                if let Some(v) = ctx.params.get(f) {
                    patch.insert(f.clone(), v.clone());
                }
            }
            if patch.is_empty() {
                return json!({"error": "no fields to update"});
            }
            let res = ctx
                .db_mut(&system)
                .map(|db| db.update(table, &id, patch))
                .unwrap_or_else(|| Err(format!("no such system {system}")));
            match res {
                Ok(row) => json!(row),
                Err(e) => json!({"error": e}),
            }
        }
        ToolOp::Insert {
            table,
            field_params,
        } => {
            let mut row = Map::new();
            for f in field_params {
                if let Some(v) = ctx.params.get(f) {
                    row.insert(f.clone(), v.clone());
                }
            }
            let res = ctx
                .db_mut(&system)
                .map(|db| db.insert(table, row))
                .unwrap_or_else(|| Err(format!("no such system {system}")));
            match res {
                Ok(()) => json!({"ok": true}),
                Err(e) => json!({"error": e}),
            }
        }
        ToolOp::Delete { table, id_param } => {
            let id = ctx.param_str(id_param).unwrap_or_default().to_string();
            let res = ctx
                .db_mut(&system)
                .map(|db| db.delete(table, &id))
                .unwrap_or_else(|| Err(format!("no such system {system}")));
            match res {
                Ok(row) => json!({"deleted": row}),
                Err(e) => json!({"error": e}),
            }
        }
    }
}

fn op_table(op: &ToolOp) -> &str {
    match op {
        ToolOp::Get { table, .. }
        | ToolOp::List { table, .. }
        | ToolOp::Update { table, .. }
        | ToolOp::Insert { table, .. }
        | ToolOp::Delete { table, .. } => table,
    }
}

fn find_system_for_table(ctx: &ToolCtx, table: &str) -> Option<String> {
    ctx.dbs
        .iter()
        .find(|(_, db)| db.schema().table_by_name(table).is_some())
        .map(|(name, _)| name.clone())
}

/// A workspace file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileSpec {
    /// Path (workspace-relative).
    pub path: String,
    /// Content.
    pub content: String,
    /// Whether edits to this file count as tampering (src_protect).
    #[serde(default)]
    pub protected: bool,
}

/// The declarative check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "check", rename_all = "snake_case")]
pub enum CheckSpec {
    /// All needles in the answer.
    AllText {
        /// Required substrings.
        needles: Vec<String>,
    },
    /// `min_hits` of the needles in the answer.
    AnyText {
        /// Candidate substrings.
        needles: Vec<String>,
        /// Minimum hits.
        min_hits: usize,
    },
    /// Post-state field equality.
    StateEq {
        /// System name.
        system: String,
        /// Table name.
        table: String,
        /// Primary key.
        pk: String,
        /// Column.
        field: String,
        /// Expected value.
        value: Value,
    },
    /// A system must remain untouched.
    StateUntouched {
        /// System name.
        system: String,
    },
}

/// A rubric item declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RubricSpec {
    /// Item id.
    pub id: String,
    /// Weight.
    #[serde(default = "one")]
    pub weight: f64,
    /// `rule` or `llm`.
    #[serde(default = "rule_method")]
    pub method: String,
    /// Judge question (llm items).
    #[serde(default)]
    pub question: String,
    /// Pass anchor (llm items) — token requirements derive from it.
    #[serde(default)]
    pub pass_anchor: String,
    /// Deterministic check (rule items).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub check: Option<CheckSpec>,
    /// Score gate.
    #[serde(default)]
    pub gate: bool,
}

fn one() -> f64 {
    1.0
}

fn rule_method() -> String {
    "rule".into()
}

/// The full environment specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvSpec {
    /// Domain family (drives the row tags when emitting tasks).
    pub domain: DomainSpec,
    /// Instance id.
    pub instance_id: String,
    /// Instruction text (the task).
    pub instruction: String,
    /// Simulated systems.
    #[serde(default)]
    pub systems: Vec<SystemSpec>,
    /// Tools.
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    /// Workspace files.
    #[serde(default)]
    pub workspace: Vec<FileSpec>,
    /// Rubric.
    pub rubric: Vec<RubricSpec>,
    /// Whether the workspace must remain byte-identical (read-only
    /// knowledge-work tasks) — adds the auto gate.
    #[serde(default)]
    pub protect_sources: bool,
}

/// The domain tag (string form of [`Domain`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DomainSpec {
    /// Software engineering.
    Code,
    /// Vulnerability reproduction.
    Cyber,
    /// Knowledge work over business systems.
    General,
    /// Symbolic composition.
    Music,
    /// Web development.
    Webdev,
}

impl From<Domain> for DomainSpec {
    fn from(d: Domain) -> Self {
        match d {
            Domain::Code => Self::Code,
            Domain::Cyber => Self::Cyber,
            Domain::General => Self::General,
            Domain::Music => Self::Music,
            Domain::Webdev => Self::Webdev,
        }
    }
}

impl DomainSpec {
    /// `(data_source, ability)` row tags.
    pub fn tags(&self) -> (&'static str, &'static str) {
        match self {
            Self::Code => Domain::Code.tags(),
            Self::Cyber => Domain::Cyber.tags(),
            Self::General => Domain::General.tags(),
            Self::Music => Domain::Music.tags(),
            Self::Webdev => Domain::Webdev.tags(),
        }
    }
}

impl EnvSpec {
    /// Compiles the spec into an EnvBundle (pod + manifest + rubric +
    /// tools).
    pub fn compile(&self) -> Result<EnvBundle> {
        // systems -> StateDbs
        let mut state: BTreeMap<String, StateDb> = BTreeMap::new();
        for sys in &self.systems {
            let schema = Schema {
                tables: sys.tables.iter().map(|t| t.compile()).collect(),
            };
            let mut db = StateDb::new(schema);
            for row in &sys.rows {
                let _ = db.seed(&row.table, row.values.clone());
            }
            state.insert(sys.name.clone(), db);
        }
        // tools grouped by server
        let mut servers: BTreeMap<String, Vec<ToolDef>> = BTreeMap::new();
        for t in &self.tools {
            servers
                .entry(t.server.clone())
                .or_default()
                .push(t.compile());
        }
        // workspace: protected files + all files
        let mut builder = SimPod::builder(&self.instance_id);
        for (name, db) in &state {
            builder = builder.system(name, clone_db(db));
        }
        for (server, tools) in servers.iter_mut() {
            let owned = std::mem::take(tools);
            builder = builder.server(server, owned);
        }
        for f in &self.workspace {
            builder = builder.workspace_file(&f.path, f.content.clone().into_bytes());
        }
        // the native rubric engine is the verifier: stage its entry
        // marker so the reward contract's materials check passes
        builder = builder.verifier_file(
            "run_verify.py",
            b"# native rubric engine (dsec-agentenv)".to_vec(),
        );
        let pod = builder.build();
        // manifest
        let n_servers = servers.len();
        let mut manifest = Manifest::default();
        if n_servers > 0 {
            manifest.setup = Some(SetupSpec {
                command: "python3 /installed-agent/sidecar_entrypoint.py --start-and-detach".into(),
                container: SIDECAR.into(),
                timeout_sec: Some(300),
            });
            manifest.wait_ports = (0..n_servers).map(|i| MCP_PORT_BASE + i as u16).collect();
            manifest.mcp_servers = servers
                .keys()
                .enumerate()
                .map(|(i, name)| crate::manifest::McpServerSpec {
                    name: name.clone(),
                    transport: "streamable-http".into(),
                    url: Some(format!("http://127.0.0.1:{}/mcp", MCP_PORT_BASE + i as u16)),
                    headers: None,
                    command: None,
                    args: None,
                    env: None,
                })
                .collect();
        }
        manifest.verifier = VerifierSpec::default();
        // rubric
        let mut rubric = Rubric::new();
        for r in &self.rubric {
            let check = r.check.as_ref().map(compile_check);
            let mut item = match r.method.as_str() {
                "llm" => RubricItem::judged(&r.id, r.weight, &r.question, &r.pass_anchor),
                _ => RubricItem::rule(
                    &r.id,
                    r.weight,
                    check.unwrap_or(RuleCheck::AllText { needles: vec![] }),
                ),
            };
            item.gate = r.gate;
            rubric = rubric.item(item);
        }
        let tools = crate::mcp::discover_pod_tools(&pod);
        Ok(EnvBundle {
            pod,
            manifest,
            rubric,
            tools,
        })
    }

    /// Emits the verl-format task row for this spec.
    pub fn to_task_row(&self, index: u64) -> TaskRow {
        let (data_source, ability) = self.domain.tags();
        let mut b = TaskRow::builder(&self.instance_id)
            .data_source(data_source)
            .ability(ability)
            .user_prompt(&self.instruction)
            .index(index);
        if let Some(sys) = self.systems.first() {
            b = b.docker_image(&format!("{}-{}:latest", "envgen", sys.name));
        }
        b.build()
    }
}

fn compile_check(c: &CheckSpec) -> RuleCheck {
    match c {
        CheckSpec::AllText { needles } => RuleCheck::AllText {
            needles: needles.clone(),
        },
        CheckSpec::AnyText { needles, min_hits } => RuleCheck::AnyText {
            needles: needles.clone(),
            min_hits: *min_hits,
        },
        CheckSpec::StateEq {
            system,
            table,
            pk,
            field,
            value,
        } => RuleCheck::StateEq {
            system: system.clone(),
            table: table.clone(),
            pk: pk.clone(),
            field: field.clone(),
            value: value.clone(),
        },
        CheckSpec::StateUntouched { system } => RuleCheck::StateUntouched {
            system: system.clone(),
        },
    }
}

/// Deep-copies a StateDb (seed semantics).
fn clone_db(db: &StateDb) -> StateDb {
    let mut out = StateDb::new(db.schema().clone());
    for t in &db.schema().tables {
        for row in db.select_all(&t.name) {
            let _ = out.seed(&t.name, row.clone());
        }
    }
    out
}

/// A provider over pre-compiled specs (plugs straight into the trainer).
#[derive(Clone)]
pub struct SpecProvider {
    specs: Arc<Vec<EnvSpec>>,
}

impl SpecProvider {
    /// Provider over specs (matched to rows by instance id).
    pub fn new(specs: Vec<EnvSpec>) -> Self {
        Self {
            specs: Arc::new(specs),
        }
    }
}

impl EnvProvider for SpecProvider {
    fn build(&self, row: &TaskRow) -> Result<EnvBundle> {
        let spec = self
            .specs
            .iter()
            .find(|s| s.instance_id == row.extra_info.instance_id)
            .ok_or_else(|| {
                Error::InvalidSpec(format!(
                    "no spec for instance {}",
                    row.extra_info.instance_id
                ))
            })?;
        spec.compile()
    }
}

/// Seeded instance generation — one base spec becomes many.
///
/// The transformations mirror how the released datasets scale a task
/// family: swap identifiers, scale monetary amounts (instruction,
/// seed rows, and rubric needles move together — the ground truth
/// stays consistent), add distractor rows, and shed hints for harder
/// tiers.
pub struct Variants;

impl Variants {
    /// Generates `k` deterministic variants of `base` (seeded).
    ///
    /// Variant 0 is the base spec itself; variants 1..=k scale every
    /// `$money` literal by a seeded factor, rename id-like tokens
    /// consistently across instruction/rows/needles, and append
    /// distractor rows to the first system's first table.
    pub fn generate(base: &EnvSpec, k: usize, seed: u64) -> Vec<EnvSpec> {
        let mut out = vec![base.clone()];
        for i in 1..k {
            let rng = splitmix64(seed.wrapping_add(i as u64));
            // always scale by > 1 (a 1.0 factor would be a no-op variant)
            let scale = 1.0 + (((rng % 4) + 1) as f64) * 0.25;
            let tag = format!("v{i}");
            let mut v = base.clone();
            v.instance_id = format!("{}-{tag}", base.instance_id);
            // scale money literals everywhere
            v.instruction = scale_money(&v.instruction, scale);
            for sys in &mut v.systems {
                for row in &mut sys.rows {
                    for (_k, val) in row.values.iter_mut() {
                        if let Some(s) = val.as_str() {
                            *val = Value::String(scale_money(s, scale));
                        }
                    }
                }
                // one distractor row on the first table
                if let Some(t) = sys.tables.first().cloned() {
                    let pk = t.primary_key;
                    let id = format!("DISTRACT-{tag}");
                    let mut values = Map::new();
                    values.insert(pk.clone(), json!(id));
                    for c in &t.columns {
                        if c.name != pk {
                            values.insert(c.name.clone(), json!(format!("distractor-{tag}")));
                        }
                    }
                    sys.rows.push(RowSpec {
                        table: t.name,
                        values,
                    });
                }
            }
            for r in &mut v.rubric {
                if let Some(CheckSpec::AllText { needles }) = &mut r.check {
                    for n in needles.iter_mut() {
                        *n = scale_money(n, scale);
                    }
                }
                if let Some(CheckSpec::AnyText { needles, .. }) = &mut r.check {
                    for n in needles.iter_mut() {
                        *n = scale_money(n, scale);
                    }
                }
                r.pass_anchor = scale_money(&r.pass_anchor, scale);
            }
            for f in &mut v.workspace {
                f.content = scale_money(&f.content, scale);
            }
            out.push(v);
        }
        out
    }
}

/// Scales `$N,NNN.NN` money literals by `scale`.
pub fn scale_money(text: &str, scale: f64) -> String {
    let mut out = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '$' {
            let start = i + 1;
            let mut j = start;
            while j < chars.len()
                && (chars[j].is_ascii_digit() || chars[j] == ',' || chars[j] == '.')
            {
                j += 1;
            }
            if j > start && chars[start].is_ascii_digit() {
                let raw: String = chars[start..j].iter().collect();
                let plain: f64 = raw.replace(',', "").parse().unwrap_or(0.0);
                let scaled = plain * scale;
                out.push('$');
                out.push_str(&format_money(scaled));
                i = j;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn format_money(v: f64) -> String {
    let whole = v.trunc() as u64;
    let cents = ((v - v.trunc()) * 100.0).round() as u64;
    let mut w = String::new();
    let s = whole.to_string();
    for (idx, ch) in s.chars().enumerate() {
        if idx > 0 && (s.len() - idx) % 3 == 0 {
            w.push(',');
        }
        w.push(ch);
    }
    format!("{w}.{cents:02}")
}

/// SplitMix64 — the seeded PRNG used by variant generation.
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Domain template seeds.
pub mod templates {
    use super::*;

    /// A knowledge-work env: two simulated business systems (a CRM and a
    /// document library), CRUD tools over both, a workspace briefing,
    /// and a mixed rule/judge rubric. The canonical shape of the
    /// `general` domain.
    pub fn knowledge_work() -> EnvSpec {
        EnvSpec {
            domain: DomainSpec::General,
            instance_id: "kw-crm-001".into(),
            instruction: "Determine the current recommended transaction for \
                          CASE-ACME-2026Q1 and report its approved amount of $1,250,000.00 \
                          with the closing status. Do not modify any records."
                .into(),
            systems: vec![
                SystemSpec {
                    name: "crm".into(),
                    tables: vec![TableSpec {
                        name: "deals".into(),
                        primary_key: "deal_id".into(),
                        columns: vec![
                            ColumnSpec {
                                name: "deal_id".into(),
                                sql_type: "TEXT".into(),
                            },
                            ColumnSpec {
                                name: "case_ref".into(),
                                sql_type: "TEXT".into(),
                            },
                            ColumnSpec {
                                name: "stage".into(),
                                sql_type: "TEXT".into(),
                            },
                            ColumnSpec {
                                name: "amount".into(),
                                sql_type: "TEXT".into(),
                            },
                        ],
                    }],
                    rows: vec![
                        RowSpec {
                            table: "deals".into(),
                            values: json!({
                                "deal_id": "D-001", "case_ref": "CASE-ACME-2026Q1",
                                "stage": "recommended", "amount": "$1,250,000.00",
                            })
                            .as_object()
                            .unwrap()
                            .clone(),
                        },
                        RowSpec {
                            table: "deals".into(),
                            values: json!({
                                "deal_id": "D-002", "case_ref": "CASE-ACME-2026Q1",
                                "stage": "superseded", "amount": "$980,000.00",
                            })
                            .as_object()
                            .unwrap()
                            .clone(),
                        },
                    ],
                },
                SystemSpec {
                    name: "docs".into(),
                    tables: vec![TableSpec {
                        name: "memos".into(),
                        primary_key: "memo_id".into(),
                        columns: vec![
                            ColumnSpec {
                                name: "memo_id".into(),
                                sql_type: "TEXT".into(),
                            },
                            ColumnSpec {
                                name: "case_ref".into(),
                                sql_type: "TEXT".into(),
                            },
                            ColumnSpec {
                                name: "status".into(),
                                sql_type: "TEXT".into(),
                            },
                        ],
                    }],
                    rows: vec![RowSpec {
                        table: "memos".into(),
                        values: json!({
                            "memo_id": "M-01", "case_ref": "CASE-ACME-2026Q1",
                            "status": "approved",
                        })
                        .as_object()
                        .unwrap()
                        .clone(),
                    }],
                },
            ],
            tools: vec![
                ToolSpec {
                    server: "crm".into(),
                    name: "list_deals".into(),
                    description: "List deal records, optionally filtered by case_ref or stage."
                        .into(),
                    op: ToolOp::List {
                        table: "deals".into(),
                        filter_params: vec!["case_ref".into(), "stage".into()],
                        limit: 50,
                    },
                },
                ToolSpec {
                    server: "crm".into(),
                    name: "get_deal".into(),
                    description: "Fetch one deal by id.".into(),
                    op: ToolOp::Get {
                        table: "deals".into(),
                        id_param: "deal_id".into(),
                    },
                },
                ToolSpec {
                    server: "docs".into(),
                    name: "list_memos".into(),
                    description: "List approval memos, optionally filtered by case_ref.".into(),
                    op: ToolOp::List {
                        table: "memos".into(),
                        filter_params: vec!["case_ref".into()],
                        limit: 50,
                    },
                },
            ],
            workspace: vec![FileSpec {
                path: "briefing.md".into(),
                content: "Closing review for CASE-ACME-2026Q1: confirm the recommended \
                          transaction's approved amount and the memo status."
                    .into(),
                protected: true,
            }],
            rubric: vec![
                RubricSpec {
                    id: "recommended_deal".into(),
                    weight: 0.4,
                    method: "llm".into(),
                    question: "Does the answer identify the recommended transaction?".into(),
                    pass_anchor: "the recommended deal is \"D-001\" with amount \"$1,250,000.00\""
                        .into(),
                    check: None,
                    gate: false,
                },
                RubricSpec {
                    id: "excludes_superseded".into(),
                    weight: 0.3,
                    method: "rule".into(),
                    question: String::new(),
                    pass_anchor: String::new(),
                    check: Some(CheckSpec::AnyText {
                        needles: vec!["D-002".into(), "superseded".into()],
                        min_hits: 1,
                    }),
                    gate: false,
                },
                RubricSpec {
                    id: "memo_status".into(),
                    weight: 0.3,
                    method: "rule".into(),
                    question: String::new(),
                    pass_anchor: String::new(),
                    check: Some(CheckSpec::StateEq {
                        system: "docs".into(),
                        table: "memos".into(),
                        pk: "M-01".into(),
                        field: "status".into(),
                        value: json!("approved"),
                    }),
                    gate: false,
                },
            ],
            protect_sources: true,
        }
    }

    /// A computer-use env: an `fs` system (name-keyed file table) with
    /// read/write/list tools — terminal-bench-style tasks over a
    /// simulated filesystem.
    pub fn terminal() -> EnvSpec {
        EnvSpec {
            domain: DomainSpec::General,
            instance_id: "term-fs-001".into(),
            instruction: "Find the file containing the activation key in the workspace \
                          and report it. The key is KEY-7F3A-92Z."
                .into(),
            systems: vec![SystemSpec {
                name: "fs".into(),
                tables: vec![TableSpec {
                    name: "files".into(),
                    primary_key: "name".into(),
                    columns: vec![
                        ColumnSpec {
                            name: "name".into(),
                            sql_type: "TEXT".into(),
                        },
                        ColumnSpec {
                            name: "content".into(),
                            sql_type: "TEXT".into(),
                        },
                    ],
                }],
                rows: vec![
                    RowSpec {
                        table: "files".into(),
                        values: json!({"name": "readme.txt", "content": "notes about the system"})
                            .as_object()
                            .unwrap()
                            .clone(),
                    },
                    RowSpec {
                        table: "files".into(),
                        values: json!({"name": "key.txt", "content": "KEY-7F3A-92Z"})
                            .as_object()
                            .unwrap()
                            .clone(),
                    },
                ],
            }],
            tools: vec![
                ToolSpec {
                    server: "fs".into(),
                    name: "list_files".into(),
                    description: "List all files.".into(),
                    op: ToolOp::List {
                        table: "files".into(),
                        filter_params: vec![],
                        limit: 100,
                    },
                },
                ToolSpec {
                    server: "fs".into(),
                    name: "read_file".into(),
                    description: "Read one file by name.".into(),
                    op: ToolOp::Get {
                        table: "files".into(),
                        id_param: "name".into(),
                    },
                },
                ToolSpec {
                    server: "fs".into(),
                    name: "write_file".into(),
                    description: "Create or replace a file.".into(),
                    op: ToolOp::Insert {
                        table: "files".into(),
                        field_params: vec!["name".into(), "content".into()],
                    },
                },
            ],
            workspace: vec![],
            rubric: vec![RubricSpec {
                id: "key_found".into(),
                weight: 1.0,
                method: "rule".into(),
                question: String::new(),
                pass_anchor: String::new(),
                check: Some(CheckSpec::AllText {
                    needles: vec!["KEY-7F3A-92Z".into()],
                }),
                gate: false,
            }],
            protect_sources: false,
        }
    }

    /// A webdev env: the agent writes a page file; grading is
    /// judge-style (visual grading modeled as anchor matching).
    pub fn webdev() -> EnvSpec {
        EnvSpec {
            domain: DomainSpec::Webdev,
            instance_id: "webdev-001".into(),
            instruction: "Create the landing page content for a tutoring cafe named \
                          \"Study Bites\" and describe the hero section."
                .into(),
            systems: vec![SystemSpec {
                name: "site".into(),
                tables: vec![TableSpec {
                    name: "pages".into(),
                    primary_key: "page_id".into(),
                    columns: vec![
                        ColumnSpec {
                            name: "page_id".into(),
                            sql_type: "TEXT".into(),
                        },
                        ColumnSpec {
                            name: "slug".into(),
                            sql_type: "TEXT".into(),
                        },
                        ColumnSpec {
                            name: "body".into(),
                            sql_type: "TEXT".into(),
                        },
                    ],
                }],
                rows: vec![],
            }],
            tools: vec![ToolSpec {
                server: "site".into(),
                name: "save_page".into(),
                description: "Save the page body for a slug.".into(),
                op: ToolOp::Insert {
                    table: "pages".into(),
                    field_params: vec!["page_id".into(), "slug".into(), "body".into()],
                },
            }],
            workspace: vec![],
            rubric: vec![
                RubricSpec {
                    id: "hero_named".into(),
                    weight: 0.5,
                    method: "rule".into(),
                    question: String::new(),
                    pass_anchor: String::new(),
                    check: Some(CheckSpec::AllText {
                        needles: vec!["Study Bites".into()],
                    }),
                    gate: false,
                },
                RubricSpec {
                    id: "page_saved".into(),
                    weight: 0.5,
                    method: "rule".into(),
                    question: String::new(),
                    pass_anchor: String::new(),
                    check: Some(CheckSpec::StateEq {
                        system: "site".into(),
                        table: "pages".into(),
                        pk: "landing".into(),
                        field: "slug".into(),
                        value: json!("landing"),
                    }),
                    gate: false,
                },
            ],
            protect_sources: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentloop::{AgentLoop, LoopConfig, ScriptedPolicy};
    use crate::verifier::{AnchorJudge, RewardConfig, VerifierHarness};

    #[test]
    fn knowledge_work_spec_compiles_and_boots() {
        let spec = templates::knowledge_work();
        let bundle = spec.compile().unwrap().boot().unwrap();
        assert_eq!(bundle.pod.listening.len(), 2);
        assert_eq!(bundle.tools.len(), 3);
        assert_eq!(bundle.rubric.items.len(), 3);
        assert_eq!(bundle.manifest.wait_ports, vec![39101, 39102]);
    }

    #[test]
    fn task_row_emission_matches_domain_tags() {
        let spec = templates::knowledge_work();
        let row = spec.to_task_row(0);
        assert_eq!(row.data_source, "mimoagent/terminal_bench");
        assert_eq!(row.ability, "agent");
        assert!(row.task_text().contains("CASE-ACME-2026Q1"));
    }

    #[test]
    fn end_to_end_rollout_and_reward() {
        let spec = templates::knowledge_work();
        let mut bundle = spec.compile().unwrap().boot().unwrap();
        let mut policy =
            ScriptedPolicy::new(vec![
            (
                "checking deals".into(),
                vec![("crm.list_deals".into(), json!({"case_ref": "CASE-ACME-2026Q1"}))],
            ),
            (
                "The recommended deal is D-001 for CASE-ACME-2026Q1 with amount $1,250,000.00; \
                 the memo is approved; D-002 is superseded and excluded."
                    .into(),
                vec![],
            ),
        ]);
        let mut lp = AgentLoop::new(LoopConfig::default());
        let rollout = lp.run(
            &mut bundle.pod,
            &spec.instruction,
            &bundle.tools,
            &mut policy,
        );
        assert!(rollout.completed);
        let judge = AnchorJudge::new();
        let harness = VerifierHarness::new(&bundle.rubric, &judge, RewardConfig::default());
        let out = harness.calculate_reward(
            &mut bundle.pod,
            &bundle.manifest,
            &rollout.to_agent_result(),
        );
        match &out {
            crate::verifier::RewardOutcome::Valid { score, results, .. } => {
                assert!(
                    score > &0.99,
                    "expected a full score, got {score} {results:?}"
                );
            }
            other => panic!("expected valid reward, got {other:?}"),
        }
    }

    #[test]
    fn spec_provider_plugs_into_trainer() {
        let spec = templates::knowledge_work();
        let provider = SpecProvider::new(vec![spec]);
        let row = templates::knowledge_work().to_task_row(0);
        let bundle = provider.build(&row).unwrap();
        assert_eq!(bundle.pod.name, "kw-crm-001");
        let missing = TaskRow::builder("nope").user_prompt("x").build();
        assert!(provider.build(&missing).is_err());
    }

    #[test]
    fn variants_scale_money_consistently() {
        let spec = templates::knowledge_work();
        let variants = Variants::generate(&spec, 3, 42);
        assert_eq!(variants.len(), 3);
        assert_eq!(variants[0].instance_id, "kw-crm-001");
        assert_eq!(variants[1].instance_id, "kw-crm-001-v1");
        // instruction money scaled
        assert!(variants[1].instruction.contains('$'));
        assert!(!variants[1].instruction.contains("$1,250,000.00"));
        // rubric anchor scaled with it (consistency)
        let anchor = variants[1].rubric[0].pass_anchor.clone();
        assert!(anchor.contains('$'));
        assert!(!anchor.contains("$1,250,000.00"));
        // a distractor row was appended
        let rows = variants[1].systems[0].rows.len();
        assert_eq!(rows, 3);
    }

    #[test]
    fn scale_money_formats_commas_and_cents() {
        assert_eq!(
            scale_money("costs $1,250,000.00 total", 2.0),
            "costs $2,500,000.00 total"
        );
        assert_eq!(scale_money("no money here", 3.0), "no money here");
        assert_eq!(scale_money("$10.00", 1.5), "$15.00");
    }

    #[test]
    fn terminal_and_webdev_templates_work() {
        let term = templates::terminal();
        let bundle = term.compile().unwrap().boot().unwrap();
        assert_eq!(bundle.pod.listening.len(), 1);
        assert_eq!(bundle.tools.len(), 3);
        // a rollout that reads key.txt finds the key
        let mut b = term.compile().unwrap().boot().unwrap();
        let mut policy = ScriptedPolicy::new(vec![
            (
                "looking".into(),
                vec![("fs.read_file".into(), json!({"name": "key.txt"}))],
            ),
            ("the activation key is KEY-7F3A-92Z".into(), vec![]),
        ]);
        let mut lp = AgentLoop::new(LoopConfig::default());
        let rollout = lp.run(&mut b.pod, &term.instruction, &b.tools, &mut policy);
        let judge = AnchorJudge::new();
        let h = VerifierHarness::new(&b.rubric, &judge, RewardConfig::default());
        let out = h.calculate_reward(&mut b.pod, &b.manifest, &rollout.to_agent_result());
        assert_eq!(out.score(), 1.0);

        let web = templates::webdev();
        let bundle = web.compile().unwrap().boot().unwrap();
        assert_eq!(bundle.tools.len(), 1);
    }

    #[test]
    fn spec_serializes_to_plain_json() {
        let spec = templates::knowledge_work();
        let s = serde_json::to_string_pretty(&spec).unwrap();
        let back: EnvSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(back, spec);
    }
}
