//! The external orchestrator journal — DSec §6.2's command log, hardened.
//!
//! Every state transition of every task is appended as one JSONL line,
//! **fsync'd before the caller proceeds**, and chained with SHA-256:
//! each event's hash covers its content plus the previous event's
//! hash. That gives three properties the recovery path leans on:
//!
//! 1. **Durability** — a crash at any point leaves a prefix of fully
//!    persisted events (a torn final line is detected and dropped).
//! 2. **Tamper evidence** — rewriting any historical event, reordering
//!    lines, or truncating the middle breaks the chain and replay
//!    refuses to continue past it (attacker tests cover this).
//! 3. **Idempotent recovery** — replay rebuilds each task's transcript
//!    and its completed tool calls, so a restarted worker reuses
//!    recorded results instead of re-executing side effects.
//!
//! Event kinds (see [`EventKind`]) are the task lifecycle's full
//! vocabulary; payloads are JSON objects.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use dsec_agentenv::agentloop::{Message, ToolCallReq};
use dsec_protocol::sha256_hex;

use crate::error::{Error, Result};

/// Canonical event kinds used by the worker and subagent managers.
pub mod event_kind {
    /// Task created: `{system, user, tools}`.
    pub const TASK_CREATED: &str = "task_created";
    /// A model turn started: `{turn, key}`.
    pub const TURN_STARTED: &str = "turn_started";
    /// Model request issued: `{key, digest}`.
    pub const MODEL_CALL_STARTED: &str = "model_call_started";
    /// Model response persisted: `{key, text, tool_calls}`.
    pub const MODEL_CALL_COMPLETED: &str = "model_call_completed";
    /// Tool execution started (idempotency key recorded): `{call_id, key, tool, params}`.
    pub const TOOL_CALL_STARTED: &str = "tool_call_started";
    /// Tool outcome persisted: `{call_id, key, observation, kind}`.
    pub const TOOL_CALL_COMPLETED: &str = "tool_call_completed";
    /// Observation appended to the transcript: `{call_id}`.
    pub const OBSERVATION_PERSISTED: &str = "observation_persisted";
    /// Environment paused.
    pub const TASK_PAUSED: &str = "task_paused";
    /// Environment resumed.
    pub const TASK_RESUMED: &str = "task_resumed";
    /// Environment snapshot taken: `{snapshot_id, label}`.
    pub const SNAPSHOT_CREATED: &str = "snapshot_created";
    /// Verification started.
    pub const VERIFY_STARTED: &str = "verify_started";
    /// Verification done: `{score, masked, trainable}`.
    pub const TASK_COMPLETED: &str = "task_completed";
    /// Task failed: `{error}`.
    pub const TASK_FAILED: &str = "task_failed";
    /// Subagent spawned: `{subagent_id, parent, base_snapshot}`.
    pub const SUBAGENT_SPAWNED: &str = "subagent_spawned";
    /// Subagent patch returned: `{subagent_id, ops}`.
    pub const PATCH_RETURNED: &str = "patch_returned";
    /// Patch merged into parent: `{subagent_id, outcome}`.
    pub const MERGE_APPLIED: &str = "merge_applied";
    /// Branch abandoned without merging: `{subagent_id}`.
    pub const BRANCH_ABANDONED: &str = "branch_abandoned";
    /// Worker recovered from a crash: `{from_seq}`.
    pub const RECOVERED: &str = "recovered";
}

/// One journal event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEvent {
    /// 1-based monotonic sequence number.
    pub seq: u64,
    /// Task namespace this event belongs to.
    pub task_id: String,
    /// Wall-clock milliseconds since epoch.
    pub ts_ms: u64,
    /// One of the [`event_kind`] strings (or a custom kind).
    pub kind: String,
    /// Event payload.
    pub payload: Value,
    /// Hash of the previous event (`""` for the first).
    pub prev_hash: String,
    /// SHA-256 chain hash of this event (covers all fields above).
    pub hash: String,
}

impl JournalEvent {
    /// The event body that the chain hash covers.
    fn chain_body(&self) -> String {
        // field order fixed for canonical hashing
        format!(
            "{}|{}|{}|{}|{}|{}",
            self.seq, self.task_id, self.ts_ms, self.kind, self.payload, self.prev_hash
        )
    }

    /// Verifies this single event's hash.
    pub fn verify_hash(&self) -> bool {
        sha256_hex(self.chain_body().as_bytes()) == self.hash
    }

    /// Computes the chain hash for a candidate body.
    fn compute_hash(
        seq: u64,
        task_id: &str,
        ts_ms: u64,
        kind: &str,
        payload: &Value,
        prev_hash: &str,
    ) -> String {
        sha256_hex(format!("{seq}|{task_id}|{ts_ms}|{kind}|{payload}|{prev_hash}").as_bytes())
    }
}

/// The append-only journal handle.
pub struct Journal {
    path: PathBuf,
    file: std::fs::File,
    seq: u64,
    last_hash: String,
    task_seq: BTreeMap<String, u64>,
}

impl Journal {
    /// Creates a fresh journal (truncating any existing file) and
    /// fsyncs the directory entry.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(&path)?;
        sync_dir(&path);
        Ok(Self {
            path,
            file,
            seq: 0,
            last_hash: String::new(),
            task_seq: BTreeMap::new(),
        })
    }

    /// Opens an existing journal for append after verifying its chain.
    ///
    /// A torn final line (crash artifact) is truncated before appending;
    /// a tampered journal is refused outright.
    pub fn open_append(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let report = replay(&path)?;
        if let Some(seq) = report.tampered_at {
            return Err(Error::JournalTampered {
                seq,
                detail: "refusing to append to a tampered journal".into(),
            });
        }
        let file = std::fs::OpenOptions::new().append(true).open(&path)?;
        if report.truncated_tail {
            // drop the torn partial line: truncate to the end of the
            // last complete line
            let raw = std::fs::read_to_string(&path)?;
            let mut offset = 0usize;
            for line in raw.split_inclusive('\n') {
                if line.ends_with('\n') {
                    offset += line.len();
                } else {
                    break;
                }
            }
            file.set_len(offset as u64)?;
        }
        let seq = report.events.last().map(|e| e.seq).unwrap_or(0);
        let last_hash = report
            .events
            .last()
            .map(|e| e.hash.clone())
            .unwrap_or_default();
        let mut task_seq = BTreeMap::new();
        for e in &report.events {
            task_seq.insert(e.task_id.clone(), e.seq);
        }
        Ok(Self {
            path,
            file,
            seq,
            last_hash,
            task_seq,
        })
    }

    /// Appends one event, fsyncs, and returns it.
    pub fn append(&mut self, task_id: &str, kind: &str, payload: Value) -> Result<JournalEvent> {
        let per_task = self.task_seq.get(task_id).copied().unwrap_or(0) + 1;
        self.task_seq.insert(task_id.to_string(), per_task);
        let event = JournalEvent {
            seq: self.seq + 1,
            task_id: task_id.to_string(),
            ts_ms: now_ms(),
            kind: kind.to_string(),
            payload,
            prev_hash: self.last_hash.clone(),
            hash: String::new(),
        };
        let hash = JournalEvent::compute_hash(
            event.seq,
            &event.task_id,
            event.ts_ms,
            &event.kind,
            &event.payload,
            &event.prev_hash,
        );
        let event = JournalEvent { hash, ..event };
        let mut line = serde_json::to_string(&event)?;
        line.push('\n');
        self.file.write_all(line.as_bytes())?;
        self.file.sync_data()?; // the durability point
        self.seq = event.seq;
        self.last_hash = event.hash.clone();
        Ok(event)
    }

    /// Current sequence position.
    pub fn position(&self) -> u64 {
        self.seq
    }

    /// Journal path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn sync_dir(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
}

/// Replay outcome.
#[derive(Debug)]
pub struct ReplayReport {
    /// Verified events, in order.
    pub events: Vec<JournalEvent>,
    /// The final line was torn (partial JSON) — the valid prefix is
    /// kept; this is the expected artifact of a crash mid-append.
    pub truncated_tail: bool,
    /// First sequence number whose hash chain failed, if any. Replay
    /// stops before it and refuses to trust the rest.
    pub tampered_at: Option<u64>,
}

impl ReplayReport {
    /// Events for one task, in order.
    pub fn events_for(&self, task_id: &str) -> Vec<&JournalEvent> {
        self.events
            .iter()
            .filter(|e| e.task_id == task_id)
            .collect()
    }
    /// All task ids seen.
    pub fn task_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.events.iter().map(|e| e.task_id.clone()).collect();
        ids.dedup();
        ids
    }
}

/// Replays and verifies a journal file.
///
/// Verification, in order: (1) each line parses as JSON; a torn final
/// line is reported as [`ReplayReport::truncated_tail`]; (2) each
/// event's own hash matches its content; (3) `prev_hash` links match
/// and `seq` increments by one. The first failure sets
/// [`ReplayReport::tampered_at`] and truncates the trusted prefix.
pub fn replay(path: impl AsRef<Path>) -> Result<ReplayReport> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path)
        .map_err(|e| Error::Journal(format!("read {}: {e}", path.display())))?;
    let mut events = Vec::new();
    let mut truncated_tail = false;
    let mut tampered_at = None;

    let lines: Vec<&str> = raw.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let event: JournalEvent = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(_) => {
                if i == lines.len() - 1 {
                    truncated_tail = true; // torn tail from a crash mid-append
                    break;
                }
                tampered_at = Some(events.len() as u64 + 1);
                break;
            }
        };
        // hash + chain checks
        let expected_seq = events.len() as u64 + 1;
        let prev_ok = event.prev_hash
            == events
                .last()
                .map(|e: &JournalEvent| e.hash.clone())
                .unwrap_or_default();
        if !event.verify_hash() || !prev_ok || event.seq != expected_seq {
            tampered_at = Some(expected_seq);
            break;
        }
        events.push(event);
    }
    Ok(ReplayReport {
        events,
        truncated_tail,
        tampered_at,
    })
}

/// The rebuilt in-memory state of one task after replay.
#[derive(Debug, Clone, Default)]
pub struct TaskReplayState {
    /// Task id.
    pub task_id: String,
    /// System prompt (from `task_created`).
    pub system: String,
    /// User task text.
    pub user: String,
    /// The transcript rebuilt from completed model calls and persisted
    /// observations.
    pub messages: Vec<Message>,
    /// Completed tool calls by idempotency key: `(request, observation,
    /// kind)` — replayed results, never re-executed.
    pub completed_tools: BTreeMap<String, (ToolCallReq, String, String)>,
    /// Whether the task was completed (and its outcome).
    pub completed: bool,
    /// Recorded score, if completed.
    pub score: Option<Value>,
    /// Whether the task failed.
    pub failed: Option<String>,
    /// Tool turns used.
    pub tool_turns: u64,
    /// Model-call turn counter.
    pub turn: u64,
    /// Whether the env was left paused.
    pub paused: bool,
    /// A tool call that was started but never completed (crash window):
    /// its idempotency key, to be re-issued against the env ledger.
    pub pending_tool: Option<ToolCallReq>,
    /// A model call that was started but never completed.
    pub pending_model_key: Option<String>,
}

impl TaskReplayState {
    /// Rebuilds one task's state from its (verified) events.
    pub fn build(task_id: &str, events: &[&JournalEvent]) -> Self {
        let mut st = TaskReplayState {
            task_id: task_id.to_string(),
            ..Default::default()
        };
        for e in events {
            match e.kind.as_str() {
                event_kind::TASK_CREATED => {
                    st.system = e.payload["system"].as_str().unwrap_or_default().to_string();
                    st.user = e.payload["user"].as_str().unwrap_or_default().to_string();
                    st.messages = vec![
                        Message::System(st.system.clone()),
                        Message::User(st.user.clone()),
                    ];
                }
                event_kind::MODEL_CALL_COMPLETED => {
                    let text = e.payload["text"].as_str().unwrap_or_default().to_string();
                    let calls: Vec<ToolCallReq> = e.payload["tool_calls"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .map(|c| ToolCallReq {
                                    id: c["id"].as_str().unwrap_or_default().to_string(),
                                    server: c["server"].as_str().unwrap_or_default().to_string(),
                                    tool: c["tool"].as_str().unwrap_or_default().to_string(),
                                    params: c["params"].clone(),
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    st.messages.push(Message::Assistant {
                        text: text.clone(),
                        calls: calls.clone(),
                    });
                    st.turn += 1;
                }
                event_kind::TOOL_CALL_STARTED => {
                    st.pending_tool = Some(ToolCallReq {
                        id: e.payload["call_id"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string(),
                        server: e.payload["server"].as_str().unwrap_or_default().to_string(),
                        tool: e.payload["tool"].as_str().unwrap_or_default().to_string(),
                        params: e.payload["params"].clone(),
                    });
                }
                event_kind::TOOL_CALL_COMPLETED => {
                    let key = e.payload["key"].as_str().unwrap_or_default().to_string();
                    if let Some(req) = st.pending_tool.take() {
                        st.completed_tools.insert(
                            key,
                            (
                                req,
                                e.payload["observation"]
                                    .as_str()
                                    .unwrap_or_default()
                                    .to_string(),
                                e.payload["kind"].as_str().unwrap_or("ok").to_string(),
                            ),
                        );
                        st.tool_turns += 1;
                    }
                }
                event_kind::OBSERVATION_PERSISTED => {
                    // the observation user-turn is already part of the
                    // transcript when MODEL_CALL_COMPLETED/OBSERVATION
                    // pairs are intact; observations are appended by the
                    // replayer below (see finalize)
                }
                event_kind::TASK_PAUSED => st.paused = true,
                event_kind::TASK_RESUMED => st.paused = false,
                event_kind::TASK_COMPLETED => {
                    st.completed = true;
                    st.score = Some(e.payload.clone());
                }
                event_kind::TASK_FAILED => {
                    st.failed = Some(e.payload["error"].as_str().unwrap_or_default().to_string());
                }
                _ => {}
            }
        }
        st.finalize_observations(events);
        st
    }

    /// Observation turns must interleave correctly with assistant
    /// turns: the journal records observations separately, so the
    /// replayer inserts each completed tool's observation turn right
    /// after the assistant turn that requested it.
    fn finalize_observations(&mut self, events: &[&JournalEvent]) {
        let mut obs: BTreeMap<String, String> = BTreeMap::new();
        for e in events {
            if e.kind == event_kind::TOOL_CALL_COMPLETED {
                let call_id = e.payload["call_id"].as_str().unwrap_or_default();
                let obs_text = e.payload["observation"].as_str().unwrap_or_default();
                obs.insert(
                    format!("{}|{}", e.seq, call_id),
                    format!("[tool {call_id} tool]\n{obs_text}"),
                );
            }
        }
        if obs.is_empty() {
            return;
        }
        // rebuild: walk events in order, push assistant turns from
        // MODEL_CALL_COMPLETED and observation turns from
        // TOOL_CALL_COMPLETED in sequence
        let mut rebuilt = vec![
            Message::System(self.system.clone()),
            Message::User(self.user.clone()),
        ];
        let mut obs_iter: Vec<(String, String)> = obs.into_iter().collect();
        obs_iter.sort();
        let mut obs_pos = 0;
        for e in events {
            match e.kind.as_str() {
                event_kind::MODEL_CALL_COMPLETED => {
                    let text = e.payload["text"].as_str().unwrap_or_default().to_string();
                    let calls: Vec<ToolCallReq> = e.payload["tool_calls"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .map(|c| ToolCallReq {
                                    id: c["id"].as_str().unwrap_or_default().to_string(),
                                    server: c["server"].as_str().unwrap_or_default().to_string(),
                                    tool: c["tool"].as_str().unwrap_or_default().to_string(),
                                    params: c["params"].clone(),
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    rebuilt.push(Message::Assistant { text, calls });
                }
                event_kind::TOOL_CALL_COMPLETED => {
                    if let Some((_, text)) = obs_iter.get(obs_pos) {
                        rebuilt.push(Message::User(text.clone()));
                        obs_pos += 1;
                    }
                }
                _ => {}
            }
        }
        self.messages = rebuilt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir();
        let p = dir.join(format!(
            "dsec-journal-test-{}-{name}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn append_replay_roundtrip() {
        let path = tmp("roundtrip");
        let mut j = Journal::create(&path).unwrap();
        j.append(
            "t1",
            event_kind::TASK_CREATED,
            json!({"system": "s", "user": "u"}),
        )
        .unwrap();
        j.append(
            "t1",
            event_kind::MODEL_CALL_COMPLETED,
            json!({"text": "hi", "tool_calls": []}),
        )
        .unwrap();
        j.append(
            "t1",
            event_kind::TOOL_CALL_STARTED,
            json!({"call_id": "c1", "key": "k1", "tool": "exec", "params": {}}),
        )
        .unwrap();
        j.append(
            "t1",
            event_kind::TOOL_CALL_COMPLETED,
            json!({"call_id": "c1", "key": "k1", "observation": "done", "kind": "ok"}),
        )
        .unwrap();
        let report = replay(&path).unwrap();
        assert_eq!(report.events.len(), 4);
        assert!(report.tampered_at.is_none());
        assert!(!report.truncated_tail);
        let st = TaskReplayState::build("t1", &report.events_for("t1"));
        // transcript: system, user, assistant, user(observation)
        assert_eq!(st.messages.len(), 4);
        assert_eq!(st.completed_tools.len(), 1);
        assert_eq!(
            st.messages[3],
            Message::User("[tool c1 tool]\ndone".to_string())
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn crash_torn_tail_is_recoverable() {
        let path = tmp("torn");
        let mut j = Journal::create(&path).unwrap();
        j.append("t1", "a", json!({"x": 1})).unwrap();
        j.append("t1", "b", json!({"x": 2})).unwrap();
        // simulate a crash mid-append: torn final line
        let mut raw = std::fs::read_to_string(&path).unwrap();
        raw.push_str("{\"seq\":3,\"task_id\":\"t1\",\"ts\":1"); // torn
        std::fs::write(&path, raw).unwrap();
        let report = replay(&path).unwrap();
        assert!(report.truncated_tail);
        assert_eq!(report.events.len(), 2);
        assert!(report.tampered_at.is_none());
        // append continues from the valid prefix
        let mut j2 = Journal::open_append(&path).unwrap();
        j2.append("t1", "c", json!({"x": 3})).unwrap();
        let report2 = replay(&path).unwrap();
        assert_eq!(report2.events.len(), 3);
        assert!(report2.tampered_at.is_none());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn tampering_middle_line_is_detected() {
        let path = tmp("tamper");
        let mut j = Journal::create(&path).unwrap();
        j.append("t1", "a", json!({"x": 1})).unwrap();
        j.append("t1", "b", json!({"x": 2})).unwrap();
        j.append("t1", "c", json!({"x": 3})).unwrap();
        // attacker rewrites the middle payload
        let lines: Vec<String> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|l| l.to_string())
            .collect();
        let mut ev: JournalEvent = serde_json::from_str(&lines[1]).unwrap();
        ev.payload = json!({"x": 999});
        std::fs::write(
            &path,
            format!(
                "{}\n{}\n{}\n",
                lines[0],
                serde_json::to_string(&ev).unwrap(),
                lines[2]
            ),
        )
        .unwrap();
        let report = replay(&path).unwrap();
        assert_eq!(report.tampered_at, Some(2));
        assert_eq!(
            report.events.len(),
            1,
            "trust stops before the tampered event"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn reordering_breaks_the_chain() {
        let path = tmp("reorder");
        let mut j = Journal::create(&path).unwrap();
        j.append("t1", "a", json!({"x": 1})).unwrap();
        j.append("t1", "b", json!({"x": 2})).unwrap();
        j.append("t1", "c", json!({"x": 3})).unwrap();
        let lines: Vec<String> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|l| l.to_string())
            .collect();
        // swap lines 1 and 2
        std::fs::write(&path, format!("{}\n{}\n{}\n", lines[0], lines[2], lines[1])).unwrap();
        let report = replay(&path).unwrap();
        assert!(report.tampered_at.is_some());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn truncating_middle_is_detected() {
        let path = tmp("truncmid");
        let mut j = Journal::create(&path).unwrap();
        for i in 0..5 {
            j.append("t1", "e", json!({"i": i})).unwrap();
        }
        let lines: Vec<String> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|l| l.to_string())
            .collect();
        // drop line 2 (a middle event) — chain links mismatch
        let kept: Vec<String> = lines
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 1)
            .map(|(_, l)| l.to_string())
            .collect();
        std::fs::write(&path, format!("{}\n", kept.join("\n"))).unwrap();
        let report = replay(&path).unwrap();
        assert!(report.tampered_at.is_some());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn replayed_tool_results_are_reused_not_reexecuted() {
        // the contract the worker's recovery relies on: a completed
        // tool call carries its observation in the journal
        let path = tmp("reuse");
        let mut j = Journal::create(&path).unwrap();
        j.append(
            "t1",
            event_kind::TASK_CREATED,
            json!({"system": "s", "user": "u"}),
        )
        .unwrap();
        j.append("t1", event_kind::TOOL_CALL_STARTED, json!({"call_id": "c1", "key": "k1", "server": "crm", "tool": "get", "params": {"id": "C1"}})).unwrap();
        j.append("t1", event_kind::TOOL_CALL_COMPLETED, json!({"call_id": "c1", "key": "k1", "observation": "{\"status\":\"vip\"}", "kind": "ok"})).unwrap();
        let report = replay(&path).unwrap();
        let st = TaskReplayState::build("t1", &report.events_for("t1"));
        let (req, obs, kind) = st.completed_tools.get("k1").unwrap();
        assert_eq!(req.tool, "get");
        assert!(obs.contains("vip"));
        assert_eq!(kind, "ok");
        // no pending tool: the pair completed
        assert!(st.pending_tool.is_none());
        std::fs::remove_file(&path).ok();
    }
}
