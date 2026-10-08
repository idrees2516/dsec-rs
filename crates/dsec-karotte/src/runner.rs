//! The evaluation runner: the exact upstream event flow.
//!
//! Ports of upstream `karotte/evaluation_runner.py`. Event ordering is
//! the contract (it is golden-tested upstream):
//! `task_started → task_pre_hook_completed → [message_added(system)] →
//! [per step: step_started → …agent events… → scoring →
//! step_completed] → task_completed(status)`, with scoring order
//! `extra artifacts → pre_scoring_hook → judge → ScoringEvent →
//! post_hook` and misbehavior folded into score 0. On any error:
//! `error → task_completed("error")`. The transcript is written durably
//! at the end no matter what.

use crate::error::{Error, Result};
use crate::message_loop::{LimitAction, LoopLimits, MessageLoop};
use crate::schemas::{EvaluationRunConfig, Event, Message, Role, RunStatus, Scoring, Transcript};
use crate::task::Task;
use std::sync::Arc;

/// The durable-write hook: `(path, body)`.
pub type DurableWrite<'a> = Box<dyn Fn(&str, &str) -> Result<()> + 'a>;

/// The runner's dependencies.
pub struct Runner<'a> {
    /// The run configuration.
    pub config: EvaluationRunConfig,
    /// The task to run.
    pub task: Arc<dyn Task>,
    /// The agent loop.
    pub agent: Arc<MessageLoop>,
    /// Durable-write hook for the transcript file (tests capture here).
    pub durable_write: DurableWrite<'a>,
}

/// The run outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct RunOutcome {
    /// Final status.
    pub status: RunStatus,
    /// Final score (last step's scoring).
    pub score: Option<f64>,
    /// Steps whose `continue_task` was true.
    pub steps_passed: usize,
    /// Total steps.
    pub n_steps: usize,
    /// The complete transcript (persisted events, chunk events excluded).
    pub transcript: Transcript,
}

impl<'a> Runner<'a> {
    /// Run the task end to end (upstream `run()`), driving the exact
    /// event order.
    pub fn run(&self) -> RunOutcome {
        let mut transcript = Transcript {
            run_id: self.config.run_id.clone(),
            events: Vec::new(),
        };
        let status: RunStatus;
        let mut last_score: Option<f64> = None;
        let mut steps_passed = 0usize;

        let steps = self.task.steps();
        let n_steps = steps.len() as i64;

        // task_started
        transcript.events.push(Event::TaskStarted {
            run_id: self.config.run_id.clone(),
            task_id: self.task.id(),
            model: Some(self.config.model.clone()),
            n_steps,
            reasoning_effort: None,
        });

        // task_pre_hook_completed
        let metadata = self.task.pre_hook().unwrap_or_default();
        transcript
            .events
            .push(Event::TaskPreHookCompleted { metadata });

        // System prompt (skipped for CLI agents upstream).
        if let Some(sys) = self.task.system_prompt() {
            transcript.events.push(Event::MessageAdded {
                message: Message::text(Role::System, sys),
                finish_reason: None,
                raw: None,
            });
        }

        // Step loop.
        for (i, step) in steps.iter().enumerate() {
            transcript.events.push(Event::StepStarted { step: i });

            // The step instructions (resolved: overrides / extras).
            let instructions = self
                .config
                .resolve_step_instructions(&step.instructions(), i);
            let limits = LoopLimits {
                turn_limit: self.config.turn_limit,
                time_limit_seconds: self.config.resolve_step_time_limit(i),
                on_time_limit: match self.config.on_step_time_limit {
                    crate::schemas::LimitAction::Error => LimitAction::Raise,
                    crate::schemas::LimitAction::Score => LimitAction::Score,
                },
                context_window_limit: self.config.resolve_step_context_window_limit(i),
                on_context_window_limit: match self.config.on_step_context_window_limit {
                    crate::schemas::LimitAction::Error => LimitAction::Raise,
                    crate::schemas::LimitAction::Score => LimitAction::Score,
                },
                inject_time_remaining_counter: self.config.inject_time_remaining_counter,
                inject_context_remaining_counter: self.config.inject_context_remaining_counter,
            };

            // Agent step: on loop-limit errors with action Raise, the run
            // errors out (upstream propagates the exception).
            match self.agent.run_step(
                &instructions,
                &limits,
                &[],
                &mut std::time::Instant::now().clone_open(),
            ) {
                Ok(mut events) => {
                    for ev in events.drain(..) {
                        if !ev.is_chunk() {
                            transcript.events.push(ev);
                        }
                    }
                }
                Err(e) => {
                    transcript.events.push(error_event(&e));
                    status = RunStatus::Error;
                    transcript.events.push(Event::TaskCompleted { status });
                    return self.finish(transcript, status, last_score, steps_passed, n_steps);
                }
            }

            // Scoring: pre_scoring_hook → judge (misbehavior → 0).
            let scoring = step.score(&transcript);
            last_score = Some(scoring.score);
            transcript.events.push(Event::Scoring {
                scoring: sanitize_scoring(scoring.clone()),
                resource_metrics: None,
            });
            // post_hook after the ScoringEvent, before StepCompleted.
            let _ = step.post_hook();
            transcript.events.push(Event::StepCompleted { step: i });

            if scoring.continue_task {
                steps_passed += 1;
            } else {
                status = RunStatus::Failed;
                transcript.events.push(Event::TaskCompleted { status });
                return self.finish(transcript, status, last_score, steps_passed, n_steps);
            }
        }

        status = RunStatus::Passed;
        transcript.events.push(Event::TaskCompleted { status });
        self.finish(transcript, status, last_score, steps_passed, n_steps)
    }

    fn finish(
        &self,
        transcript: Transcript,
        status: RunStatus,
        score: Option<f64>,
        steps_passed: usize,
        n_steps: i64,
    ) -> RunOutcome {
        // The transcript is written durably no matter what (upstream
        // `_maybe_save_transcript_to_file` in the outer finally).
        if let Some(path) = &self.config.transcript_file {
            let body = serde_json::to_string_pretty(&transcript).unwrap_or_default();
            let _ = (self.durable_write)(path, &body);
        }
        RunOutcome {
            status,
            score,
            steps_passed,
            n_steps: n_steps.max(0) as usize,
            transcript,
        }
    }
}

/// Sanitize a scoring's metadata before it lands on the event
/// (upstream `sanitize_metadata` on ScoringEvents).
fn sanitize_scoring(mut s: Scoring) -> Scoring {
    let obj: serde_json::Value =
        serde_json::to_value(&s.metadata).unwrap_or(serde_json::Value::Null);
    let cleaned = crate::text::sanitize_metadata(&obj);
    if let Ok(cleaned) = serde_json::from_value(cleaned) {
        s.metadata = cleaned;
    }
    s
}

/// Build the upstream ErrorEvent from an error (exception type, escaped
/// message, traceback).
pub fn error_event(e: &Error) -> Event {
    Event::Error {
        exception_type: exception_type_name(e),
        message: crate::text::escape_surrogates(&e.to_string()),
        traceback: None,
    }
}

fn exception_type_name(e: &Error) -> String {
    match e {
        Error::LoopLimit(_) => "StepTimeLimitReachedError".to_string(),
        Error::Misbehavior(_) => "StudentMisbehaviorError".to_string(),
        Error::Confinement(_) => "RuntimeError".to_string(),
        Error::NotFound(_) => "ValueError".to_string(),
        Error::InvalidSpec(_) => "ValueError".to_string(),
        Error::Unsupported(_) => "NotSupportedError".to_string(),
        Error::ProcessFailed { .. } | Error::ProcessTimeout { .. } => {
            "CalledProcessError".to_string()
        }
        Error::Json(_) => "JSONDecodeError".to_string(),
        Error::Io(_) => "OSError".to_string(),
    }
}

/// Extension so run_step's clock parameter can default.
trait CloneInstant {
    fn clone_open(&self) -> Box<dyn FnMut() -> f64 + 'static>;
}

impl CloneInstant for std::time::Instant {
    fn clone_open(&self) -> Box<dyn FnMut() -> f64 + 'static> {
        let start = *self;
        Box::new(move || start.elapsed().as_secs_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judges::AlwaysPassJudge;
    use crate::message_loop::{MapDispatcher, ScriptedSource};
    use crate::schemas::Message;
    use crate::task::{create_task, StepConfig};
    use std::sync::Mutex;

    fn agent_for(messages: Vec<Message>) -> Arc<MessageLoop> {
        Arc::new(MessageLoop::new(
            Arc::new(ScriptedSource::new(messages)),
            Arc::new(MapDispatcher::new()),
        ))
    }

    fn runner<'r>(
        config: EvaluationRunConfig,
        task: Arc<dyn Task>,
        agent: Arc<MessageLoop>,
        writes: &'r Mutex<Vec<(String, String)>>,
    ) -> Runner<'r> {
        // The hook captures the shared sink by reference (lifetime 'r).
        let w: DurableWrite<'r> = Box::new(move |p, b| {
            writes.lock().unwrap().push((p.to_string(), b.to_string()));
            Ok(())
        });
        Runner {
            config,
            task,
            agent,
            durable_write: w,
        }
    }

    #[test]
    fn golden_event_order_pass() {
        let steps = vec![
            StepConfig::new("step one", Arc::new(AlwaysPassJudge)),
            StepConfig::new("step two", Arc::new(AlwaysPassJudge)),
        ];
        let task = create_task(
            "golden-task",
            vec!["bash".into()],
            steps,
            Some("be a good agent".into()),
        );
        let agent = agent_for(vec![
            Message::text(Role::Assistant, "working on one"),
            Message::text(Role::Assistant, "working on two"),
        ]);
        let mut config = EvaluationRunConfig::new("run-1", "golden-task", "fake/model");
        config.transcript_file = Some("/out/transcript.json".into());
        let writes: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
        let outcome = runner(config, task, agent, &writes).run();

        assert_eq!(outcome.status, RunStatus::Passed);
        assert_eq!(outcome.steps_passed, 2);
        assert_eq!(outcome.score, Some(1.0));

        let kinds: Vec<&str> = outcome.transcript.events.iter().map(|e| e.kind()).collect();
        assert_eq!(
            kinds,
            vec![
                "task_started",
                "task_pre_hook_completed",
                "message_added", // system
                "step_started",
                "message_added", // user instructions
                "message_added", // assistant
                "scoring",
                "step_completed",
                "step_started",
                "message_added",
                "message_added",
                "scoring",
                "step_completed",
                "task_completed",
            ]
        );

        // The transcript was written durably once.
        assert_eq!(writes.lock().unwrap().len(), 1);
        let (path, body) = writes.lock().unwrap()[0].clone();
        assert_eq!(path, "/out/transcript.json");
        assert!(body.contains("\"task_started\""));
        assert!(body.contains("\"task_completed\""));
    }

    #[test]
    fn failed_step_stops_the_run() {
        // A regex judge that never matches → score 0, continue_task false.
        let fail: Arc<dyn crate::judges::Judge> = Arc::new(crate::judges::RegexJudge::new(vec![
            "impossible-pattern-xyz".into(),
        ]));
        let steps = vec![
            StepConfig::new("step one", fail),
            StepConfig::new("step two", Arc::new(AlwaysPassJudge)),
        ];
        let task = create_task("failing-task", vec![], steps, None);
        let agent = agent_for(vec![Message::text(Role::Assistant, "trying")]);
        let config = EvaluationRunConfig::new("run-2", "failing-task", "fake/model");
        let writes: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
        let outcome = runner(config, task, agent, &writes).run();

        assert_eq!(outcome.status, RunStatus::Failed);
        assert_eq!(outcome.steps_passed, 0);
        assert_eq!(outcome.score, Some(0.0));
        // The second step never started.
        assert!(!outcome
            .transcript
            .events
            .iter()
            .any(|e| matches!(e, Event::StepStarted { step: 1 })));
    }

    #[test]
    fn misbehavior_scores_zero_but_completes() {
        use crate::judges::MisbehaviorJudge;
        let steps = vec![StepConfig::new("cheat step", Arc::new(MisbehaviorJudge))];
        let task = create_task("cheat-task", vec![], steps, None);
        let agent = agent_for(vec![Message::text(Role::Assistant, "I will cheat")]);
        let config = EvaluationRunConfig::new("run-3", "cheat-task", "fake/model");
        let writes: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
        let outcome = runner(config, task, agent, &writes).run();

        // Misbehavior → failed (score 0), NOT error.
        assert_eq!(outcome.status, RunStatus::Failed);
        assert_eq!(outcome.score, Some(0.0));
        let scoring = outcome
            .transcript
            .events
            .iter()
            .find_map(|e| e.as_scoring().map(|(s, _)| s.clone()))
            .unwrap();
        assert!(scoring.metadata.contains_key("misbehavior"));
        assert_eq!(scoring.score, 0.0);
    }

    #[test]
    fn loop_limit_error_produces_error_event_and_error_status() {
        // The scripted source runs out of messages mid-step.
        let steps = vec![StepConfig::new("step one", Arc::new(AlwaysPassJudge))];
        let task = create_task("err-task", vec![], steps, None);
        let agent = agent_for(vec![]); // exhausted immediately
        let mut config = EvaluationRunConfig::new("run-4", "err-task", "fake/model");
        config.transcript_file = Some("/out/t.json".into());
        let writes: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
        let outcome = runner(config, task, agent, &writes).run();

        assert_eq!(outcome.status, RunStatus::Error);
        let last = outcome.transcript.events.last().unwrap();
        assert!(matches!(
            last,
            Event::TaskCompleted {
                status: RunStatus::Error
            }
        ));
        let has_error = outcome
            .transcript
            .events
            .iter()
            .any(|e| matches!(e, Event::Error { exception_type, .. } if exception_type == "StepTimeLimitReachedError" || exception_type.contains("Limit")));
        assert!(has_error);
        // The transcript is still written on error.
        assert_eq!(writes.lock().unwrap().len(), 1);
    }

    #[test]
    fn instructions_resolved_with_overrides_and_extras() {
        let steps = vec![StepConfig::new("original", Arc::new(AlwaysPassJudge))];
        let task = create_task("instr-task", vec![], steps, None);
        let agent = agent_for(vec![Message::text(Role::Assistant, "ok")]);
        let mut config = EvaluationRunConfig::new("r", "instr-task", "m");
        config.extra_task_instructions = Some(serde_json::json!("extra guidance"));
        let writes: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
        let outcome = runner(config, task, agent, &writes).run();
        // The user message carries original + extras.
        let user_msg = outcome
            .transcript
            .events
            .iter()
            .find_map(|e| match e {
                Event::MessageAdded { message, .. } if matches!(message.role, Role::User) => {
                    message.content.as_ref().and_then(|c| c.as_text())
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(user_msg, "original\n\nextra guidance");
    }

    #[test]
    fn no_system_prompt_no_system_message() {
        let steps = vec![StepConfig::new("s", Arc::new(AlwaysPassJudge))];
        let task = create_task("nosys", vec![], steps, None);
        let agent = agent_for(vec![Message::text(Role::Assistant, "hi")]);
        let config = EvaluationRunConfig::new("r", "nosys", "m");
        let writes: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
        let outcome = runner(config, task, agent, &writes).run();
        assert!(!outcome
            .transcript
            .events
            .iter()
            .any(|e| matches!(e, Event::MessageAdded { message, .. } if matches!(message.role, Role::System))));
    }

    #[test]
    fn scoring_metadata_is_sanitized() {
        let huge = "y".repeat(2 * crate::text::MAX_METADATA_VALUE_CHARS);
        // A judge that returns huge metadata.
        struct HugeMeta {
            value: String,
        }
        impl crate::judges::Judge for HugeMeta {
            fn evaluate(&self, _t: &Transcript) -> Result<Scoring> {
                let mut md = std::collections::BTreeMap::new();
                md.insert(
                    "k".to_string(),
                    serde_json::Value::String(self.value.clone()),
                );
                Ok(Scoring {
                    score: 1.0,
                    metadata: md,
                    continue_task: true,
                })
            }
        }
        let steps = vec![StepConfig::new(
            "s",
            Arc::new(HugeMeta {
                value: huge.clone(),
            }),
        )];
        let task = create_task("meta-task", vec![], steps, None);
        let agent = agent_for(vec![Message::text(Role::Assistant, "hi")]);
        let config = EvaluationRunConfig::new("r", "meta-task", "m");
        let writes: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
        let outcome = runner(config, task, agent, &writes).run();
        let scoring = outcome
            .transcript
            .events
            .iter()
            .find_map(|e| e.as_scoring().map(|(s, _)| s.clone()))
            .unwrap();
        let v = scoring.metadata["k"].as_str().unwrap();
        assert!(v.chars().count() < huge.chars().count(), "metadata bounded");
    }
}
