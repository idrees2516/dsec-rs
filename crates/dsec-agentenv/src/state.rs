//! Simulated business-system state — the `system/<mcp>/state.db` layer.
//!
//! Upstream, every MCP server in the knowledge-work environments is backed
//! by a SQLite database under `/work/system/<mcp>/state.db` (+ a
//! `schema.sql`). The verifier snapshots every table of every DB into a
//! `post_state` map and grades the agent on the *effect of its tool calls*
//! — not on what it claims. The sidecar owns the DBs physically; the main
//! container never mounts them, so the only way to mutate state is through
//! the MCP tools.
//!
//! This module ports that data model as a deterministic in-process
//! relational store:
//!
//! * [`Schema`] — table definitions (`name`, columns, primary key).
//! * [`StateDb`] — rows keyed by primary key, CRUD with row-level
//!   accounting (inserts / updates / deletes counters per table — used by
//!   the verifier's `src_protect`-style gates and anti-hack checks).
//! * [`PostState`] — the verifier-facing snapshot: every table of every
//!   named DB, identical in shape to the upstream `build_post_state` map.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// A column definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Column {
    /// Column name.
    pub name: String,
    /// SQL-ish type tag (`TEXT`, `REAL`, `INTEGER`, `BOOL`, ...). Carried
    /// for fidelity; the store itself is dynamically typed like SQLite.
    pub sql_type: String,
}

/// A table definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Table {
    /// Table name.
    pub name: String,
    /// Column definitions in order.
    pub columns: Vec<Column>,
    /// Primary-key column (single-column keys only, as in the released
    /// schemas).
    pub primary_key: String,
}

impl Table {
    /// Builds a TEXT-column table helper.
    pub fn text(name: &str, pk: &str, cols: &[&str]) -> Self {
        Self {
            name: name.to_string(),
            columns: cols
                .iter()
                .map(|c| Column {
                    name: c.to_string(),
                    sql_type: "TEXT".into(),
                })
                .collect(),
            primary_key: pk.to_string(),
        }
    }
}

/// Schema for one simulated system (one `state.db`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Schema {
    /// Tables in declaration order.
    pub tables: Vec<Table>,
}

impl Schema {
    /// Empty schema.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a table.
    pub fn table(mut self, t: Table) -> Self {
        self.tables.push(t);
        self
    }

    /// Looks a table up by name.
    pub fn table_by_name(&self, name: &str) -> Option<&Table> {
        self.tables.iter().find(|t| t.name == name)
    }
}

/// Per-table mutation accounting — the side input to anti-hack rubrics.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TableDelta {
    /// Rows inserted since construction.
    pub inserts: u64,
    /// Rows updated since construction.
    pub updates: u64,
    /// Rows deleted since construction.
    pub deletes: u64,
}

/// One simulated relational database.
///
/// Rows are `Map<String, Value>` keyed by the primary-key column. A
/// missing column deserializes as `Value::Null`, mirroring SQLite `NULL`.
#[derive(Debug, Clone)]
pub struct StateDb {
    schema: Schema,
    tables: BTreeMap<String, BTreeMap<String, Map<String, Value>>>,
    deltas: BTreeMap<String, TableDelta>,
}

/// Extracts a row's primary-key string: JSON strings use their raw
/// text (never the quoted JSON form); other scalars fall back to the
/// JSON rendering.
fn key_of(row: &Map<String, Value>, pk: &str) -> Option<String> {
    row.get(pk).map(|v| match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    })
}

impl StateDb {
    /// Creates an empty DB with a schema.
    pub fn new(schema: Schema) -> Self {
        let mut tables = BTreeMap::new();
        let mut deltas = BTreeMap::new();
        for t in &schema.tables {
            tables.insert(t.name.clone(), BTreeMap::new());
            deltas.insert(t.name.clone(), TableDelta::default());
        }
        Self {
            schema,
            tables,
            deltas,
        }
    }

    /// The schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Seeds one row (during environment construction). Duplicate keys
    /// overwrite; seed writes are NOT counted in the delta accounting.
    pub fn seed(&mut self, table: &str, row: Map<String, Value>) -> Option<()> {
        let t = self.schema.table_by_name(table)?;
        let pk = t.primary_key.clone();
        let key = key_of(&row, &pk)?;
        self.tables.get_mut(table)?.insert(key, row);
        Some(())
    }

    /// Seeds many rows at once (returns the number applied).
    pub fn seed_rows(
        &mut self,
        table: &str,
        rows: impl IntoIterator<Item = Map<String, Value>>,
    ) -> usize {
        rows.into_iter()
            .filter(|r| self.seed(table, r.clone()).is_some())
            .count()
    }

    /// Inserts a row (tool-path mutation; counted in deltas).
    pub fn insert(
        &mut self,
        table: &str,
        row: Map<String, Value>,
    ) -> std::result::Result<(), String> {
        let t = self
            .schema
            .table_by_name(table)
            .ok_or_else(|| format!("no such table: {table}"))?
            .clone();
        let key = key_of(&row, &t.primary_key)
            .ok_or_else(|| format!("row missing primary key {}", t.primary_key))?;
        let tbl = self
            .tables
            .get_mut(table)
            .expect("schema tables are pre-created");
        if tbl.contains_key(&key) {
            return Err(format!(
                "UNIQUE constraint failed: {table}.{}",
                t.primary_key
            ));
        }
        tbl.insert(key, row);
        if let Some(d) = self.deltas.get_mut(table) {
            d.inserts += 1;
        }
        Ok(())
    }

    /// Updates a row by primary key (merge semantics; counted).
    pub fn update(
        &mut self,
        table: &str,
        key: &str,
        patch: Map<String, Value>,
    ) -> std::result::Result<Map<String, Value>, String> {
        let t = self
            .schema
            .table_by_name(table)
            .ok_or_else(|| format!("no such table: {table}"))?;
        if patch.contains_key(&t.primary_key) {
            return Err("primary key is immutable".into());
        }
        let tbl = self
            .tables
            .get_mut(table)
            .expect("schema tables are pre-created");
        let row = tbl
            .get_mut(key)
            .ok_or_else(|| format!("no such row: {table}:{key}"))?;
        for (k, v) in patch {
            row.insert(k, v);
        }
        let snapshot = row.clone();
        if let Some(d) = self.deltas.get_mut(table) {
            d.updates += 1;
        }
        Ok(snapshot)
    }

    /// Deletes a row by primary key (counted).
    pub fn delete(
        &mut self,
        table: &str,
        key: &str,
    ) -> std::result::Result<Map<String, Value>, String> {
        if self.schema.table_by_name(table).is_none() {
            return Err(format!("no such table: {table}"));
        }
        let tbl = self
            .tables
            .get_mut(table)
            .expect("schema tables are pre-created");
        let row = tbl
            .remove(key)
            .ok_or_else(|| format!("no such row: {table}:{key}"))?;
        if let Some(d) = self.deltas.get_mut(table) {
            d.deletes += 1;
        }
        Ok(row)
    }

    /// Reads a row by primary key.
    pub fn get(&self, table: &str, key: &str) -> Option<&Map<String, Value>> {
        self.tables.get(table)?.get(key)
    }

    /// Reads a whole table (primary-key order — BTreeMap).
    pub fn select_all(&self, table: &str) -> Vec<&Map<String, Value>> {
        self.tables
            .get(table)
            .map(|t| t.values().collect())
            .unwrap_or_default()
    }

    /// Selects rows matching all equality predicates (`col = value`).
    pub fn select_where(&self, table: &str, eq: &Map<String, Value>) -> Vec<&Map<String, Value>> {
        self.select_all(table)
            .into_iter()
            .filter(|row| {
                eq.iter()
                    .all(|(k, v)| row.get(k).map(|rv| rv == v).unwrap_or(false))
            })
            .collect()
    }

    /// The delta accounting since construction.
    pub fn deltas(&self) -> &BTreeMap<String, TableDelta> {
        &self.deltas
    }

    /// True when no tool-path mutation has been applied to any table.
    /// Backs the "do not change any files or database records" style
    /// rubric gates: a knowledge-work task can demand a read-only rollout.
    pub fn is_untouched(&self) -> bool {
        self.deltas
            .values()
            .all(|d| d.inserts == 0 && d.updates == 0 && d.deletes == 0)
    }
}

/// The verifier-facing snapshot of every DB of every system — the
/// `build_post_state` map, `system/<mcp>` name → table → rows.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PostState {
    /// `system name -> table name -> rows`.
    pub systems: BTreeMap<String, BTreeMap<String, Vec<Map<String, Value>>>>,
}

impl PostState {
    /// Snapshots a set of named DBs.
    pub fn snapshot<'a>(dbs: impl IntoIterator<Item = (&'a str, &'a StateDb)>) -> Self {
        let mut systems = BTreeMap::new();
        for (name, db) in dbs {
            let mut tables = BTreeMap::new();
            for t in &db.schema.tables {
                let rows: Vec<Map<String, Value>> =
                    db.select_all(&t.name).into_iter().cloned().collect();
                tables.insert(t.name.clone(), rows);
            }
            systems.insert(name.to_string(), tables);
        }
        Self { systems }
    }

    /// Rows of one table of one system.
    pub fn rows(&self, system: &str, table: &str) -> &[Map<String, Value>] {
        self.systems
            .get(system)
            .and_then(|t| t.get(table))
            .map(|r| r.as_slice())
            .unwrap_or_default()
    }

    /// First row where `col == value` in a system/table (the common
    /// ground-truth lookup in rubric predicates).
    pub fn find(
        &self,
        system: &str,
        table: &str,
        col: &str,
        value: &str,
    ) -> Option<&Map<String, Value>> {
        self.rows(system, table)
            .iter()
            .find(|r| r.get(col).and_then(|v| v.as_str()) == Some(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn crm_db() -> StateDb {
        let schema = Schema::new()
            .table(Table::text(
                "customers",
                "id",
                &["id", "name", "status", "tier"],
            ))
            .table(Table::text(
                "deals",
                "deal_id",
                &["deal_id", "customer", "stage", "amount"],
            ));
        let mut db = StateDb::new(schema);
        db.seed_rows(
            "customers",
            [
                json!({"id": "C1", "name": "Acme", "status": "active", "tier": "gold"}),
                json!({"id": "C2", "name": "Globex", "status": "churned", "tier": "silver"}),
            ]
            .into_iter()
            .map(|v| v.as_object().cloned().unwrap()),
        );
        db
    }

    #[test]
    fn crud_roundtrip_with_unique_and_immutability() {
        let mut db = crm_db();
        // insert + duplicate rejected
        db.insert(
            "customers",
            json!({"id": "C3", "name": "Initech", "status": "active", "tier": "bronze"})
                .as_object()
                .unwrap()
                .clone(),
        )
        .unwrap();
        let dup = db.insert(
            "customers",
            json!({"id": "C3", "name": "dupe"})
                .as_object()
                .unwrap()
                .clone(),
        );
        assert!(dup.is_err());
        // update + pk immutability
        let updated = db
            .update(
                "customers",
                "C1",
                json!({"tier": "platinum"}).as_object().unwrap().clone(),
            )
            .unwrap();
        assert_eq!(updated.get("tier"), Some(&json!("platinum")));
        assert_eq!(updated.get("name"), Some(&json!("Acme")));
        assert!(db
            .update(
                "customers",
                "C1",
                json!({"id": "X"}).as_object().unwrap().clone()
            )
            .is_err());
        // delete + missing
        db.delete("customers", "C2").unwrap();
        assert!(db.get("customers", "C2").is_none());
        assert!(db.delete("customers", "C2").is_err());
        // deltas
        let d = db.deltas().get("customers").unwrap();
        assert_eq!((d.inserts, d.updates, d.deletes), (1, 1, 1));
    }

    #[test]
    fn seed_is_untouched_then_tool_mutations_flag() {
        let db = crm_db();
        assert!(db.is_untouched());
        let mut dirty = crm_db();
        dirty.seed_rows(
            "deals",
            [
                json!({"deal_id": "D1", "customer": "C1", "stage": "open", "amount": "$1.00"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ],
        );
        assert!(dirty.is_untouched(), "seed writes are not tool mutations");
        dirty
            .update(
                "deals",
                "D1",
                json!({"stage": "won"}).as_object().unwrap().clone(),
            )
            .unwrap();
        assert!(!dirty.is_untouched());
    }

    #[test]
    fn select_where_equality() {
        let db = crm_db();
        let eq = json!({"status": "active"}).as_object().unwrap().clone();
        let rows = db.select_where("customers", &eq);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("name"), Some(&json!("Acme")));
    }

    #[test]
    fn post_state_snapshot_and_lookup() {
        let db = crm_db();
        let ps = PostState::snapshot([("crm", &db)]);
        assert_eq!(ps.rows("crm", "customers").len(), 2);
        let acme = ps.find("crm", "customers", "id", "C1").unwrap();
        assert_eq!(acme.get("name"), Some(&json!("Acme")));
        // absent system/table degrade to empty
        assert!(ps.rows("nope", "customers").is_empty());
        assert!(ps.rows("crm", "nope").is_empty());
    }

    #[test]
    fn schema_and_rows_serialize() {
        let db = crm_db();
        let ps = PostState::snapshot([("crm", &db)]);
        let s = serde_json::to_string(&ps).unwrap();
        let back: PostState = serde_json::from_str(&s).unwrap();
        assert_eq!(back, ps);
    }
}
