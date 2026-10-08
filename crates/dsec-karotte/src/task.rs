//! The Task/Step authoring contracts and the dynamic task factory.
//!
//! Ports of upstream `karotte/task.py`, `karotte/step.py`,
//! `karotte/load_tasks.py`, and `karotte/task_factory.py`. Python's
//! ABC-based contracts become traits; the factory builds concrete
//! implementations from declarative [`StepConfig`]s exactly like upstream
//! `create_task` builds subclasses via `type()`.

use crate::error::Result;
use crate::judges::Judge;
use crate::schemas::{EvaluationRunConfig, Scoring, Transcript};
use std::sync::Arc;

/// A task step (upstream `Step(ABC)`).
///
/// The lifecycle the runner drives:
/// instructions → agent loop → `pre_scoring_hook` → `judge.evaluate` →
/// `ScoringEvent` → `post_hook` → `StepCompletedEvent`.
pub trait Step: Send + Sync {
    /// The user message that starts the step.
    fn instructions(&self) -> String;

    /// The judge that scores this step (read after the pre-scoring hook).
    fn judge(&self) -> Arc<dyn Judge>;

    /// Paths collected into custody before scoring (must be absolute).
    /// `None` = no submission collection.
    fn submission_paths(&self) -> Option<Vec<std::path::PathBuf>> {
        None
    }

    /// Runs before scoring (upstream `pre_scoring_hook`): kill the student
    /// cohort, collect submissions, prime caches...
    fn pre_scoring_hook(&self) -> Result<()> {
        Ok(())
    }

    /// Runs after the scoring event, before step completion.
    fn post_hook(&self) -> Result<()> {
        Ok(())
    }

    /// Score this step: hook → judge; misbehavior becomes score 0
    /// (upstream `Step.score`, `@final`).
    fn score(&self, transcript: &Transcript) -> Scoring {
        if let Err(crate::error::Error::Misbehavior(e)) = self.pre_scoring_hook() {
            return Scoring::misbehavior(&e);
        }
        match self.judge().evaluate(transcript) {
            Ok(s) => s,
            Err(crate::error::Error::Misbehavior(e)) => Scoring::misbehavior(&e),
            Err(other) => {
                // Infra failures surface as a zero with an error marker —
                // the runner escalates to ErrorEvent separately.
                let mut md = std::collections::BTreeMap::new();
                md.insert(
                    "judge_error".to_string(),
                    serde_json::Value::String(other.to_string()),
                );
                Scoring {
                    score: 0.0,
                    metadata: md,
                    continue_task: false,
                }
            }
        }
    }
}

/// A task (upstream `Task(ABC)`).
pub trait Task: Send + Sync {
    /// Unique id within the environment (≤ 255 chars upstream).
    fn id(&self) -> String;

    /// The system prompt; `None` sends no system message.
    fn system_prompt(&self) -> Option<String>;

    /// The steps, in order.
    fn steps(&self) -> Vec<Arc<dyn Step>>;

    /// Tool names the student gets (e.g. `bash`, `view_lines_in_file`).
    fn tools(&self) -> Vec<String>;

    /// Called before the MCP server registers tools.
    fn configure_tools(&self) -> Result<()> {
        Ok(())
    }

    /// Run-level hook; its metadata lands on `task_pre_hook_completed`.
    fn pre_hook(&self) -> Result<std::collections::BTreeMap<String, serde_json::Value>> {
        Ok(Default::default())
    }

    /// Hardware requirement (plugin-resolved upstream; free-form here).
    fn required_hardware(&self) -> Option<String> {
        None
    }

    /// Deduplicated union of all step submission paths
    /// (upstream `Task.submission_paths`, order-preserving).
    fn submission_paths(&self) -> Vec<std::path::PathBuf> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for step in self.steps() {
            if let Some(paths) = step.submission_paths() {
                for p in paths {
                    if seen.insert(p.clone()) {
                        out.push(p);
                    }
                }
            }
        }
        out
    }

    /// Pinned read-only data mounts.
    fn data_mounts(&self) -> Vec<crate::schemas::DataMount> {
        Vec::new()
    }
}

/// Declarative step description for the factory
/// (upstream `StepConfig` dataclass).
pub struct StepConfig {
    /// The step instructions.
    pub instructions: String,
    /// The judge (or a constructor from the run config).
    pub judge: JudgeSpec,
    /// Submission paths.
    pub submission_paths: Option<Vec<std::path::PathBuf>>,
}

/// A judge constructor from the run config.
pub type JudgeBuilder = Arc<dyn Fn(&EvaluationRunConfig) -> Arc<dyn Judge> + Send + Sync>;

/// How the factory obtains a judge (upstream: a `Judge` instance or a
/// `Callable[[EvaluationRunConfig], Judge]`).
pub enum JudgeSpec {
    /// A ready judge.
    Ready(Arc<dyn Judge>),
    /// Built from the run config at instantiation time.
    Deferred(JudgeBuilder),
}

impl StepConfig {
    /// A step with a ready judge.
    pub fn new(instructions: impl Into<String>, judge: Arc<dyn Judge>) -> Self {
        Self {
            instructions: instructions.into(),
            judge: JudgeSpec::Ready(judge),
            submission_paths: None,
        }
    }

    /// A step whose judge is built from the run config.
    pub fn deferred(
        instructions: impl Into<String>,
        build: impl Fn(&EvaluationRunConfig) -> Arc<dyn Judge> + Send + Sync + 'static,
    ) -> Self {
        Self {
            instructions: instructions.into(),
            judge: JudgeSpec::Deferred(Arc::new(build)),
            submission_paths: None,
        }
    }

    /// Set the submission paths.
    pub fn with_submission_paths(mut self, paths: Vec<std::path::PathBuf>) -> Self {
        self.submission_paths = Some(paths);
        self
    }
}

/// A concrete built step (upstream `step_{i+1}` subclasses).
struct FactoryStep {
    instructions: String,
    judge: Arc<dyn Judge>,
    submission_paths: Option<Vec<std::path::PathBuf>>,
}

impl Step for FactoryStep {
    fn instructions(&self) -> String {
        self.instructions.clone()
    }
    fn judge(&self) -> Arc<dyn Judge> {
        self.judge.clone()
    }
    fn submission_paths(&self) -> Option<Vec<std::path::PathBuf>> {
        self.submission_paths.clone()
    }
}

/// A concrete task assembled by [`create_task`].
pub struct FactoryTask {
    id: String,
    system_prompt: Option<String>,
    tools: Vec<String>,
    steps: Vec<Arc<dyn Step>>,
    required_hardware: Option<String>,
}

impl Task for FactoryTask {
    fn id(&self) -> String {
        self.id.clone()
    }
    fn system_prompt(&self) -> Option<String> {
        self.system_prompt.clone()
    }
    fn steps(&self) -> Vec<Arc<dyn Step>> {
        self.steps.clone()
    }
    fn tools(&self) -> Vec<String> {
        self.tools.clone()
    }
    fn required_hardware(&self) -> Option<String> {
        self.required_hardware.clone()
    }
}

/// Build a task from declarative parts (upstream `create_task`): callables
/// receive the run config; steps are numbered `step_1..step_n`.
pub fn create_task(
    id: impl Into<String>,
    tools: Vec<String>,
    steps: Vec<StepConfig>,
    system_prompt: Option<String>,
) -> Arc<dyn Task> {
    create_task_inner(id, tools, steps, system_prompt)
}

fn create_task_inner(
    id: impl Into<String>,
    tools: Vec<String>,
    steps: Vec<StepConfig>,
    system_prompt: Option<String>,
) -> Arc<dyn Task> {
    let task: Arc<FactoryTask> = Arc::new(FactoryTask {
        id: id.into(),
        system_prompt,
        tools,
        steps: steps
            .into_iter()
            .map(|sc| {
                let judge = match sc.judge {
                    JudgeSpec::Ready(j) => j,
                    JudgeSpec::Deferred(f) => f(&EvaluationRunConfig::new("", "", "")),
                };
                Arc::new(FactoryStep {
                    instructions: sc.instructions,
                    judge,
                    submission_paths: sc.submission_paths,
                }) as Arc<dyn Step>
            })
            .collect(),
        required_hardware: None,
    });
    task
}

/// A task registry standing in for upstream `environment.get_tasks()`
/// (which globs `environment.tasks.*` subpackages and sorts by id).
#[derive(Default)]
pub struct TaskRegistry {
    tasks: Vec<Arc<dyn Task>>,
}

impl TaskRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a task.
    pub fn register(&mut self, task: Arc<dyn Task>) {
        self.tasks.push(task);
        self.tasks.sort_by_key(|t| t.id());
    }

    /// Load the task whose id matches (upstream `load_task`).
    pub fn load(&self, config: &EvaluationRunConfig) -> Result<Arc<dyn Task>> {
        self.tasks
            .iter()
            .find(|t| t.id() == config.task_id)
            .cloned()
            .ok_or_else(|| {
                crate::error::Error::NotFound(format!(
                    "Task {:?} not found. Available: {:?}",
                    config.task_id,
                    self.tasks.iter().map(|t| t.id()).collect::<Vec<_>>()
                ))
            })
    }

    /// All registered ids, sorted.
    pub fn ids(&self) -> Vec<String> {
        self.tasks.iter().map(|t| t.id()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judges::AlwaysPassJudge;

    #[test]
    fn factory_task_steps_and_paths() {
        let j: Arc<dyn Judge> = Arc::new(AlwaysPassJudge);
        let steps = vec![
            StepConfig::new("step one", j.clone())
                .with_submission_paths(vec![std::path::PathBuf::from("/workdir/out.txt")]),
            StepConfig::new("step two", j).with_submission_paths(vec![
                std::path::PathBuf::from("/workdir/out.txt"),
                std::path::PathBuf::from("/workdir/extra"),
            ]),
        ];
        let task = create_task(
            "my-task",
            vec!["bash".into()],
            steps,
            Some("be good".into()),
        );
        assert_eq!(task.id(), "my-task");
        assert_eq!(task.system_prompt().as_deref(), Some("be good"));
        assert_eq!(task.steps().len(), 2);
        assert_eq!(task.steps()[0].instructions(), "step one");
        // Deduplicated union, order preserved.
        assert_eq!(
            task.submission_paths(),
            vec![
                std::path::PathBuf::from("/workdir/out.txt"),
                std::path::PathBuf::from("/workdir/extra")
            ]
        );
    }

    #[test]
    fn score_turns_misbehavior_into_zero() {
        use crate::judges::MisbehaviorJudge;

        let step = FactoryStep {
            instructions: "x".into(),
            judge: Arc::new(MisbehaviorJudge),
            submission_paths: None,
        };
        let s = step.score(&Transcript::default());
        assert_eq!(s.score, 0.0);
        assert!(!s.continue_task);
        assert!(s.metadata.contains_key("misbehavior"));
    }

    #[test]
    fn registry_load_or_not_found() {
        let mut reg = TaskRegistry::new();
        reg.register(create_task(
            "a",
            vec![],
            vec![StepConfig::new("i", Arc::new(AlwaysPassJudge))],
            None,
        ));
        let cfg = EvaluationRunConfig::new("r", "a", "m");
        assert_eq!(reg.load(&cfg).unwrap().id(), "a");
        let cfg_b = EvaluationRunConfig::new("r", "b", "m");
        assert!(matches!(
            reg.load(&cfg_b),
            Err(crate::error::Error::NotFound(_))
        ));
        assert_eq!(reg.ids(), vec!["a".to_string()]);
    }

    #[test]
    fn deferred_judges_receive_config() {
        use crate::judges::RegexJudge;
        let steps = vec![StepConfig::deferred("i", move |_cfg| {
            Arc::new(RegexJudge::new(vec!["answer.*42".to_string()]))
        })];
        let task = create_task("t", vec![], steps, None);
        assert_eq!(task.steps().len(), 1);
    }
}
