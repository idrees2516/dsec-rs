//! The worker container — DSec §6.2's scaffold-agnostic control layer.
//!
//! Owns one task's rollout end to end:
//!
//! 1. **Durable loop.** Every boundary is journaled *before and after*:
//!    `turn_started → model_call_started → model_call_completed →
//!    tool_call_started → tool_call_completed → observation persisted →
//!    … → verify_started → task_completed`. Each append is fsync'd.
//! 2. **Idempotent tool calls.** The tool idempotency key is
//!    `task:tool:turn:call_id`; the environment's ledger dedups
//!    execution, so recovery never repeats side effects — including in
//!    the crash window *after* execution but *before* journal
//!    persistence.
//! 3. **Pause / resume.** `pause()` reclaims the sandbox (the dsec
//!    backend does it for real) while the journal records the state; a
//!    resumed task continues mid-rollout with the full transcript.
//! 4. **Failure handling.** Model endpoints that stay down fail the
//!    task cleanly (`task_failed`); tool *exceptions* are agent-visible
//!    and training continues; tool *transport* errors are infra
//!    failures that abort the rollout (recoverable).
//! 5. **Crash injection.** [`Crashpoint`]s fire at the exact boundaries
//!    where silent bugs live, so tests can prove recovery invariants.
//! 6. **Recovery.** [`WorkerContainer::recover`] replays the journal,
//!    rebuilds the transcript, finishes interrupted dispatches via the
//!    env ledger, and continues the task.

use std::collections::BTreeSet;
use std::sync::Arc;

use serde_json::{json, Value};

use dsec_agentenv::agentloop::{approx_tokens, truncate_observation, Message, ToolCallReq};
use dsec_rl::reward::{RewardRecord, RewardValidity};
use dsec_rl::tito::TranscriptTurn;

use crate::error::{Error, Result};
use crate::journal::{event_kind, replay, Journal, TaskReplayState};
use crate::model::{ChatMessage, ChatRequest, ModelClient, ToolCallOut};
use crate::taskenv::TaskEnvironment;

/// The crash-injection boundaries (see the phased build plan): the
/// places where the most damaging bugs — silent duplicate side
/// effects, false-zero rewards, lost transcripts — would hide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Crashpoint {
    /// After the env executed the tool, before the outcome is
    /// journaled — the duplicate-side-effect window.
    AfterToolExecBeforePersist,
    /// After the model responded, before the response is journaled.
    AfterModelCallBeforePersist,
    /// After the sandbox pause took effect, before `task_paused` is
    /// journaled.
    DuringSandboxPause,
    /// After the evaluator returned, before the reward is journaled.
    WhileEvaluationPending,
}

/// Worker configuration.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Maximum tool turns before the forced final answer.
    pub step_limit: u64,
    /// Observation truncation cap, bytes.
    pub obs_truncate: usize,
    /// Whole-trajectory token budget (approx).
    pub response_budget: Option<u64>,
    /// Consecutive model failures tolerated before `task_failed`.
    pub max_model_failures: u32,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            step_limit: 30,
            obs_truncate: 4000,
            response_budget: None,
            max_model_failures: 3,
        }
    }
}

/// The worker's lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    /// Running the loop.
    Running,
    /// Paused (env reclaimed, state preserved).
    Paused,
    /// Task finished with a recorded reward.
    Completed,
    /// Task failed (model or infra).
    Failed,
}

/// The rollout summary a completed task produces.
#[derive(Debug, Clone)]
pub struct WorkerRollout {
    /// Full transcript (agentenv message shape).
    pub messages: Vec<Message>,
    /// Final assistant text.
    pub final_message: String,
    /// Whether the policy produced a final turn.
    pub completed: bool,
    /// Tool turns used.
    pub tool_turns: u64,
    /// Approximate tokens consumed.
    pub tokens: u64,
    /// Tool dispatch observations, in order (`call_id, observation,
    /// kind`).
    pub dispatches: Vec<(String, String, String)>,
}

impl WorkerRollout {
    /// The rollout as TITO transcript turns (feeds the Phase 4 RL data
    /// path).
    pub fn to_transcript_turns(&self) -> Vec<TranscriptTurn> {
        let mut out = Vec::new();
        for m in &self.messages {
            match m {
                Message::System(s) => out.push(TranscriptTurn::System(s.clone())),
                Message::User(s) => {
                    // observation turns are tagged by the loop format
                    if let Some((call_id, content)) = parse_observation_turn(s) {
                        out.push(TranscriptTurn::ToolObservation { call_id, content });
                        continue;
                    }
                    out.push(TranscriptTurn::User(s.clone()));
                }
                Message::Assistant { text, calls } => out.push(TranscriptTurn::Assistant {
                    text: text.clone(),
                    tool_calls: calls
                        .iter()
                        .map(|c| (c.id.clone(), c.qualified(), c.params.to_string()))
                        .collect(),
                }),
            }
        }
        out
    }
}

/// The final task outcome.
#[derive(Debug, Clone)]
pub enum TaskOutcome {
    /// The task finished and was verified.
    Completed {
        /// Reward with validity.
        reward: RewardRecord,
        /// Rollout summary.
        rollout: WorkerRollout,
    },
    /// The task failed.
    Failed {
        /// Why.
        error: String,
    },
}

/// The worker container.
pub struct WorkerContainer {
    task_id: String,
    env: Option<Box<dyn TaskEnvironment>>,
    model: Arc<dyn ModelClient>,
    journal: Journal,
    config: WorkerConfig,
    messages: Vec<Message>,
    tool_schemas: Vec<Value>,
    completed_tool_keys: BTreeSet<String>,
    pending_dispatch: Vec<ToolCallReq>,
    turn: u64,
    tool_turns: u64,
    tokens: u64,
    state: WorkerState,
    outcome: Option<TaskOutcome>,
    crashpoints: BTreeSet<Crashpoint>,
    model_failures: u32,
}

impl WorkerContainer {
    /// Creates a worker for a fresh task and journals `task_created`.
    pub fn new(
        task_id: String,
        env: Box<dyn TaskEnvironment>,
        model: Arc<dyn ModelClient>,
        mut journal: Journal,
        config: WorkerConfig,
    ) -> Result<Self> {
        let tool_schemas = env.tool_schemas();
        let instruction = env.instruction();
        let system = format!(
            "You are an agent working in a sandboxed environment.\n\n\
             Task:\n{instruction}\n\n\
             Interact with the environment through the tools. When you have \
             the final answer, reply with it as plain text and no tool calls. \
             The environment records your session; the grader reads your last \
             assistant message."
        );
        journal.append(
            &task_id,
            event_kind::TASK_CREATED,
            json!({"system": system, "user": instruction, "backend": env.kind()}),
        )?;
        let messages = vec![Message::System(system), Message::User(instruction)];
        Ok(Self {
            task_id,
            env: Some(env),
            model,
            journal,
            config,
            messages,
            tool_schemas,
            completed_tool_keys: BTreeSet::new(),
            pending_dispatch: Vec::new(),
            turn: 0,
            tool_turns: 0,
            tokens: 0,
            state: WorkerState::Running,
            outcome: None,
            crashpoints: BTreeSet::new(),
            model_failures: 0,
        })
    }

    /// Arms crashpoints (tests / drills).
    pub fn with_crashpoints(mut self, points: &[Crashpoint]) -> Self {
        self.crashpoints = points.iter().copied().collect();
        self
    }

    /// Current lifecycle state.
    pub fn state(&self) -> WorkerState {
        self.state
    }

    /// The recorded outcome, if finished.
    pub fn outcome(&self) -> Option<&TaskOutcome> {
        self.outcome.as_ref()
    }

    /// Transcript so far.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Hands the surviving environment back (the sandbox outlives the
    /// worker — the recovery re-attach path).
    pub fn into_env(mut self) -> Option<Box<dyn TaskEnvironment>> {
        self.env.take()
    }

    /// The env, borrowed (subagent spawning etc.).
    pub fn env_mut(&mut self) -> Option<&mut Box<dyn TaskEnvironment>> {
        self.env.as_mut()
    }

    fn fire(&self, point: Crashpoint) -> Result<()> {
        if self.crashpoints.contains(&point) {
            return Err(Error::CrashInjected { point });
        }
        Ok(())
    }

    fn tool_key(&self, turn: u64, call_id: &str) -> String {
        format!("{}:tool:{}:{}", self.task_id, turn, call_id)
    }

    /// Pauses: sandbox reclaim first, then the journal record. The
    /// crash window between them is exactly
    /// [`Crashpoint::DuringSandboxPause`].
    pub async fn pause(&mut self) -> Result<()> {
        let env = self.env.as_mut().ok_or_else(|| Error::Task {
            task: self.task_id.clone(),
            message: "env gone".into(),
        })?;
        env.pause().await?;
        self.fire(Crashpoint::DuringSandboxPause)?;
        self.journal
            .append(&self.task_id, event_kind::TASK_PAUSED, json!({}))?;
        self.state = WorkerState::Paused;
        Ok(())
    }

    /// Resumes a paused task.
    pub async fn resume(&mut self) -> Result<()> {
        let env = self.env.as_mut().ok_or_else(|| Error::Task {
            task: self.task_id.clone(),
            message: "env gone".into(),
        })?;
        if env.is_paused() {
            env.resume().await?;
        }
        self.journal
            .append(&self.task_id, event_kind::TASK_RESUMED, json!({}))?;
        self.state = WorkerState::Running;
        Ok(())
    }

    /// Runs the task to completion (or failure). Paused tasks resume
    /// transparently first.
    pub async fn run_task(&mut self) -> Result<TaskOutcome> {
        if self.state == WorkerState::Completed || self.state == WorkerState::Failed {
            return Ok(self.outcome.clone().unwrap_or(TaskOutcome::Failed {
                error: "no outcome".into(),
            }));
        }
        if self.state == WorkerState::Paused {
            self.resume().await?;
        }

        // finish any dispatches interrupted by the crash (recovery path)
        if !self.pending_dispatch.is_empty() {
            let calls = std::mem::take(&mut self.pending_dispatch);
            self.dispatch_calls(calls).await?;
        }

        let mut steps_left = self.config.step_limit.saturating_sub(self.tool_turns);
        loop {
            let budget_exhausted = self
                .config
                .response_budget
                .map(|b| self.tokens >= b)
                .unwrap_or(false)
                || steps_left == 0;

            // ---- model call ----
            self.journal.append(
                &self.task_id,
                event_kind::TURN_STARTED,
                json!({"turn": self.turn, "steps_left": steps_left}),
            )?;
            let key = format!("{}:model:{}", self.task_id, self.turn);
            self.journal.append(
                &self.task_id,
                event_kind::MODEL_CALL_STARTED,
                json!({"key": key, "messages": self.messages.len()}),
            )?;
            let request = ChatRequest {
                model: String::new(), // client fills its own default
                messages: self.chat_messages(),
                tools: self.tool_schemas.clone(),
                temperature: 0.2,
                max_tokens: 1024,
                idempotency_key: key.clone(),
            };
            let resp = match self.model.complete(request).await {
                Ok(r) => {
                    self.model_failures = 0;
                    r
                }
                Err(e) => {
                    self.model_failures += 1;
                    if self.model_failures >= self.config.max_model_failures {
                        let msg = format!("model endpoint failed: {e}");
                        self.fail(&msg).await?;
                        return Ok(TaskOutcome::Failed { error: msg });
                    }
                    // transient: retry the same turn (the idempotency
                    // key keeps any honoring server consistent)
                    continue;
                }
            };
            self.fire(Crashpoint::AfterModelCallBeforePersist)?;
            let calls: Vec<ToolCallReq> = resp
                .tool_calls
                .iter()
                .map(|c| ToolCallReq {
                    id: c.id.clone(),
                    server: c
                        .name
                        .split_once('.')
                        .map(|(s, _)| s.to_string())
                        .unwrap_or_default(),
                    tool: c
                        .name
                        .rsplit_once('.')
                        .map(|(_, t)| t.to_string())
                        .unwrap_or_else(|| c.name.clone()),
                    params: serde_json::from_str(&c.arguments).unwrap_or(Value::Null),
                })
                .collect();
            self.journal.append(
                &self.task_id,
                event_kind::MODEL_CALL_COMPLETED,
                json!({
                    "key": key,
                    "text": resp.text,
                    "served": resp.served_model,
                    "usage_tokens": resp.usage_tokens,
                    // parsed form (server/tool split) so replay rebuilds
                    // identical ToolCallReq values
                    "tool_calls": calls.iter().map(|c| json!({
                        "id": c.id, "server": c.server, "tool": c.tool, "params": c.params,
                    })).collect::<Vec<_>>(),
                }),
            )?;
            self.tokens += approx_tokens(&resp.text);
            self.messages.push(Message::Assistant {
                text: resp.text.clone(),
                calls: calls.clone(),
            });
            self.turn += 1;

            if calls.is_empty() || budget_exhausted {
                if calls.is_empty() && !resp.text.is_empty() {
                    // final answer
                    break;
                }
                if budget_exhausted && !calls.is_empty() {
                    // forced wrap-up turn already emitted: record it as
                    // step-limited completion
                    break;
                }
                break;
            }

            // ---- tool dispatch ----
            self.dispatch_calls(calls).await?;
            self.tool_turns += 1;
            steps_left = steps_left.saturating_sub(1);
            if self.env.as_ref().map(|e| e.real_exec_count()) == Some(0) && false {
                // (unreachable guard kept for symmetry; real-exec
                // counting is asserted by tests)
            }
        }

        // ---- verification ----
        self.journal
            .append(&self.task_id, event_kind::VERIFY_STARTED, json!({}))?;
        let rollout = self.rollout_summary();
        let agent_result = dsec_agentenv::verifier::AgentRolloutResult {
            final_message: rollout.final_message.clone(),
            completed: rollout.completed,
            tool_turns: rollout.tool_turns,
        };
        let env = self.env.as_mut().ok_or_else(|| Error::Task {
            task: self.task_id.clone(),
            message: "env gone".into(),
        })?;
        let mut reward = env.verify(&agent_result).await?;
        self.fire(Crashpoint::WhileEvaluationPending)?;
        // stamp identity (the worker owns execution/group ids)
        reward.execution_id = execution_id_of(&self.task_id);
        let reward_json = json!({
            "score": reward.score().map(|s| s as f64),
            "masked": matches!(reward.validity, RewardValidity::EvaluatorFailure),
            "validity": format!("{:?}", reward.validity),
        });
        self.journal
            .append(&self.task_id, event_kind::TASK_COMPLETED, reward_json)?;
        self.state = WorkerState::Completed;
        let outcome = TaskOutcome::Completed {
            reward,
            rollout: self.rollout_summary(),
        };
        self.outcome = Some(outcome.clone());
        Ok(outcome)
    }

    async fn dispatch_calls(&mut self, calls: Vec<ToolCallReq>) -> Result<()> {
        let turn = self.turn.saturating_sub(1);
        for call in calls {
            let key = self.tool_key(turn, &call.id);
            // already replayed from the journal? skip (idempotent
            // continuation)
            if self.completed_tool_keys.contains(&key) {
                continue;
            }
            self.journal.append(
                &self.task_id,
                event_kind::TOOL_CALL_STARTED,
                json!({
                    "call_id": call.id,
                    "key": key,
                    "server": call.server,
                    "tool": call.tool,
                    "params": call.params,
                }),
            )?;
            let env = self.env.as_mut().ok_or_else(|| Error::Task {
                task: self.task_id.clone(),
                message: "env gone".into(),
            })?;
            let outcome = env.call_tool(&call, &key).await?;
            // THE critical window: the side effect happened; if we die
            // here the journal has no completion — recovery re-issues
            // the call, the env ledger answers from cache, and no
            // second side effect occurs.
            self.fire(Crashpoint::AfterToolExecBeforePersist)?;
            self.journal.append(
                &self.task_id,
                event_kind::TOOL_CALL_COMPLETED,
                json!({
                    "call_id": call.id,
                    "key": key,
                    "observation": outcome.observation,
                    "kind": outcome.kind,
                }),
            )?;
            self.completed_tool_keys.insert(key);
            self.tokens += approx_tokens(&outcome.observation);
            let obs_text = truncate_observation(&outcome.observation, self.config.obs_truncate);
            self.messages.push(Message::User(format!(
                "[tool {} {}]\n{}",
                call.id,
                call.qualified(),
                obs_text
            )));
            if outcome.kind == "transport_error" {
                let msg = format!("tool transport error: {}", outcome.observation);
                self.fail(&msg).await?;
                return Err(Error::Task {
                    task: self.task_id.clone(),
                    message: msg,
                });
            }
        }
        Ok(())
    }

    async fn fail(&mut self, error: &str) -> Result<()> {
        self.journal.append(
            &self.task_id,
            event_kind::TASK_FAILED,
            json!({"error": error}),
        )?;
        self.state = WorkerState::Failed;
        self.outcome = Some(TaskOutcome::Failed {
            error: error.to_string(),
        });
        Ok(())
    }

    fn chat_messages(&self) -> Vec<ChatMessage> {
        let mut out = Vec::new();
        for m in &self.messages {
            match m {
                Message::System(s) => out.push(ChatMessage::new("system", s.clone())),
                Message::User(s) => {
                    // observation turns → tool role
                    if let Some(obs) = parse_observation_turn(s) {
                        out.push(ChatMessage::tool_observation(obs.0, obs.1));
                        continue;
                    }
                    out.push(ChatMessage::new("user", s.clone()));
                }
                Message::Assistant { text, calls } => {
                    let tool_calls = calls
                        .iter()
                        .map(|c| ToolCallOut {
                            id: c.id.clone(),
                            name: c.qualified(),
                            arguments: c.params.to_string(),
                        })
                        .collect();
                    out.push(ChatMessage::assistant(text.clone(), tool_calls));
                }
            }
        }
        out
    }

    fn rollout_summary(&self) -> WorkerRollout {
        let final_message = self
            .messages
            .iter()
            .rev()
            .find_map(|m| match m {
                Message::Assistant { text, .. } if !text.is_empty() => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let completed = matches!(self.state, WorkerState::Completed)
            || self
                .messages
                .iter()
                .rev()
                .any(|m| matches!(m, Message::Assistant { text, calls } if !text.is_empty() && calls.is_empty()));
        let dispatches = self
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::User(s) if s.starts_with("[tool ") => {
                    let id = parse_observation_turn(s)
                        .map(|(id, _)| id)
                        .unwrap_or_default();
                    Some((id, s.clone(), "ok".to_string()))
                }
                _ => None,
            })
            .collect();
        WorkerRollout {
            messages: self.messages.clone(),
            final_message,
            completed,
            tool_turns: self.tool_turns,
            tokens: self.tokens,
            dispatches,
        }
    }

    /// Recovers a worker from the journal + the surviving environment.
    ///
    /// Rebuilds the transcript; re-attaches to the env (reconciling
    /// pause state — the env is the source of truth for its own
    /// state); finishes dispatches that were in flight when the crash
    /// hit (idempotently, via the env ledger); then continues with
    /// [`WorkerContainer::run_task`].
    pub async fn recover(
        task_id: String,
        env: Box<dyn TaskEnvironment>,
        model: Arc<dyn ModelClient>,
        journal_path: std::path::PathBuf,
        config: WorkerConfig,
        crashpoints: BTreeSet<Crashpoint>,
    ) -> Result<Self> {
        let report = replay(&journal_path)?;
        if let Some(seq) = report.tampered_at {
            return Err(Error::JournalTampered {
                seq,
                detail: "hash chain broken — refusing to recover from a tampered journal".into(),
            });
        }
        let st = TaskReplayState::build(&task_id, &report.events_for(&task_id));
        let mut journal = Journal::open_append(&journal_path)?;
        journal.append(
            &task_id,
            event_kind::RECOVERED,
            json!({"from_seq": report.events.len()}),
        )?;

        // reconcile pause state with the surviving env
        let mut env = env;
        if st.paused && !env.is_paused() {
            // journal says paused, env says running: nothing to do
        } else if !st.paused && env.is_paused() {
            // journal says running (crash during pause), env is paused:
            // resume
            env.resume().await?;
        }

        // dispatches in flight at crash time: calls of the last
        // assistant-with-calls message whose idempotency key has no
        // journaled completion. The key's turn index is the MODEL-CALL
        // index of that message, counted by walking the transcript.
        let tool_schemas = env.tool_schemas();
        let completed_keys: BTreeSet<String> = st.completed_tools.keys().cloned().collect();
        let mut assistant_index: u64 = 0;
        let mut last_call_turn: Option<u64> = None;
        let mut last_calls: Vec<ToolCallReq> = Vec::new();
        for m in &st.messages {
            if let Message::Assistant { calls, .. } = m {
                if !calls.is_empty() {
                    last_call_turn = Some(assistant_index);
                    last_calls = calls.clone();
                }
                assistant_index += 1;
            }
        }
        let pending: Vec<ToolCallReq> = match last_call_turn {
            Some(t) => last_calls
                .into_iter()
                .filter(|c| {
                    let key = format!("{}:tool:{}:{}", task_id, t, c.id);
                    !completed_keys.contains(&key)
                })
                .collect(),
            None => Vec::new(),
        };

        let state = if st.completed {
            WorkerState::Completed
        } else if st.failed.is_some() {
            WorkerState::Failed
        } else {
            WorkerState::Running
        };

        Ok(Self {
            task_id,
            env: Some(env),
            model,
            journal,
            config,
            messages: st.messages,
            tool_schemas,
            completed_tool_keys: completed_keys,
            pending_dispatch: pending,
            turn: st.turn,
            tool_turns: st.tool_turns,
            tokens: 0,
            state,
            outcome: None,
            crashpoints,
            model_failures: 0,
        })
    }
}

/// Deterministic execution id from a task id (fnv1a64).
pub fn execution_id_of(task_id: &str) -> u64 {
    crate::taskenv::fnv1a(task_id.as_bytes())
}

/// Parses an observation turn `[tool {call_id} {tool}]\n{content}` into
/// `(call_id, content)`.
fn parse_observation_turn(s: &str) -> Option<(String, String)> {
    let rest = s.strip_prefix('[')?;
    let (head, tail) = rest.split_once(']')?;
    let mut parts = head.splitn(3, ' ');
    if parts.next()? != "tool" {
        return None;
    }
    let call_id = parts.next()?.to_string();
    let content = tail.trim_start_matches(['\n', ' ']).to_string();
    Some((call_id, content))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_turns_mapping() {
        let rollout = WorkerRollout {
            messages: vec![
                Message::System("sys".into()),
                Message::User("task".into()),
                Message::Assistant {
                    text: "call it".into(),
                    calls: vec![ToolCallReq {
                        id: "c1".into(),
                        server: "crm".into(),
                        tool: "get".into(),
                        params: json!({"id": "C1"}),
                    }],
                },
                Message::User("[tool c1 crm.get]\n{\"ok\":true}".into()),
            ],
            final_message: String::new(),
            completed: false,
            tool_turns: 1,
            tokens: 10,
            dispatches: vec![],
        };
        let turns = rollout.to_transcript_turns();
        assert_eq!(turns.len(), 4);
        assert!(matches!(turns[3], TranscriptTurn::ToolObservation { .. }));
        assert!(matches!(turns[2], TranscriptTurn::Assistant { .. }));
    }

    #[test]
    fn chat_message_mapping() {
        let msgs = vec![
            Message::System("s".into()),
            Message::User("t".into()),
            Message::Assistant {
                text: "a".into(),
                calls: vec![ToolCallReq {
                    id: "c1".into(),
                    server: String::new(),
                    tool: "exec".into(),
                    params: json!({}),
                }],
            },
            Message::User("[tool c1 exec]\nout".into()),
        ];
        let w = WorkerContainer {
            task_id: "t".into(),
            env: None,
            model: Arc::new(crate::model::ScriptedModelClient::new(vec![])),
            journal: Journal::create(std::env::temp_dir().join("unused-j")).unwrap(),
            config: WorkerConfig::default(),
            messages: msgs,
            tool_schemas: vec![],
            completed_tool_keys: BTreeSet::new(),
            pending_dispatch: vec![],
            turn: 1,
            tool_turns: 1,
            tokens: 0,
            state: WorkerState::Running,
            outcome: None,
            crashpoints: BTreeSet::new(),
            model_failures: 0,
        };
        let chat = w.chat_messages();
        assert_eq!(chat.len(), 4);
        assert_eq!(chat[2].tool_calls.len(), 1);
        assert_eq!(chat[3].role, "tool");
        assert_eq!(chat[3].tool_call_id.as_deref(), Some("c1"));
        // call-id extraction: "[tool c1 exec]" -> "c1"
        assert_eq!(chat[3].content, "out");
    }
}
