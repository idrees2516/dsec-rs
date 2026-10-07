//! The multi-turn agent loop — the rollout engine.
//!
//! Upstream, mimoagent's `DefaultAgent` owns this loop and verl's
//! `AgentLoop` bridges it into the training engine: the message list is
//! append-only, assistant turns land right after each model query, tool
//! calls dispatch to the env actor next to the pod, and exceptions map
//! to user-turn observations. The contracts this port preserves:
//!
//! * **append-only messages** — `run(task)` primes `[system, instance]`
//!   then loops `step()`; the rollout's token buffer is exactly the
//!   message sequence (single-threaded, no subagent trees);
//! * **parallel tool dispatch** with per-call results folded back in
//!   order;
//! * **observation truncation** — tool outputs beyond the cap are
//!   head-truncated with a marker;
//! * **step limit** — `step_limit` tool turns; the loop then forces a
//!   final no-tools turn ("provide your final answer");
//! * **response budget** — a token ceiling on the whole trajectory
//!   (the `TokenTrace` / `ResponseBudgetExhausted` semantics);
//! * **error taxonomy** — a tool that ran and failed is a
//!   `tool_exception` (agent-visible, training continues); a transport
//!   break is a `transport_error` (infra failure — see
//!   [`InfraErrorKind`]);
//! * **session logs** — every assistant turn is appended to the pod's
//!   session store, which the verifier later reads to extract the final
//!   answer.

use crate::error::InfraErrorKind;
use crate::mcp::ToolInfo;
use crate::topology::SimPod;
use serde_json::{json, Value};

/// One message in the rollout transcript.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// System / instruction turn.
    System(String),
    /// User (task, tool observations) turn.
    User(String),
    /// Assistant turn: final text and/or tool calls.
    Assistant {
        /// Text the model produced (empty when it only called tools).
        text: String,
        /// Tool calls issued with this turn.
        calls: Vec<ToolCallReq>,
    },
}

/// A tool call requested by the policy.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallReq {
    /// Call id (echoed in the observation turn).
    pub id: String,
    /// Server name (empty when the call used the bare tool name).
    pub server: String,
    /// Tool name.
    pub tool: String,
    /// Arguments object.
    pub params: Value,
}

impl ToolCallReq {
    /// The namespaced name (`server.tool`).
    pub fn qualified(&self) -> String {
        if self.server.is_empty() {
            self.tool.clone()
        } else {
            format!("{}.{}", self.server, self.tool)
        }
    }
}

/// What the policy produced for one turn.
#[derive(Debug, Clone, Default)]
pub struct PolicyTurn {
    /// Text reply (final answer when `tool_calls` is empty).
    pub text: String,
    /// Tool calls to dispatch.
    pub tool_calls: Vec<ToolCallReq>,
}

/// Per-turn context handed to the policy: the transcript plus the
/// discovered tool surface (OpenAI function-tool schemas).
pub struct TurnCtx<'a> {
    /// Transcript so far (append-only).
    pub messages: &'a [Message],
    /// `(server, function_tool_schema)` rows.
    pub tools: &'a [(String, Value)],
    /// The remaining tool turns before the step limit forces a final
    /// answer.
    pub steps_left: u64,
    /// Whether the response budget is nearly exhausted (the policy should
    /// wrap up).
    pub budget_exhausted: bool,
}

/// The policy — the model under training (or a scripted stand-in).
///
/// The RL bridge implements this with the trainer's rollout model:
/// `turn()` re-enters the training engine token-in/token-out, exactly
/// like verl's `_VerlRolloutModel.query()`.
pub trait Policy: Send {
    /// Produces one turn given the context.
    fn turn(&mut self, ctx: &TurnCtx) -> PolicyTurn;
}

/// A scripted policy — deterministic replay of a fixed turn sequence.
///
/// The final turn (empty `tool_calls`) ends the rollout; a script that
/// runs out is treated as step-limited.
pub struct ScriptedPolicy {
    script: Vec<PolicyTurn>,
    pos: usize,
}

impl ScriptedPolicy {
    /// New scripted policy over turn specs:
    /// `("text", [("tool", params), ...])` tuples.
    pub fn new(turns: Vec<(String, Vec<(String, Value)>)>) -> Self {
        let script = turns
            .into_iter()
            .map(|(text, calls)| {
                let tool_calls = calls
                    .into_iter()
                    .enumerate()
                    .map(|(i, (tool, params))| {
                        let (server, tool) = tool.split_once('.').unwrap_or(("", tool.as_ref()));
                        ToolCallReq {
                            id: format!("call_{i}"),
                            server: server.to_string(),
                            tool: tool.to_string(),
                            params,
                        }
                    })
                    .collect();
                PolicyTurn { text, tool_calls }
            })
            .collect();
        Self { script, pos: 0 }
    }

    /// Progress through the script (for assertions).
    pub fn position(&self) -> usize {
        self.pos
    }
}

impl Policy for ScriptedPolicy {
    fn turn(&mut self, _ctx: &TurnCtx) -> PolicyTurn {
        let t = self.script.get(self.pos).cloned().unwrap_or_default();
        self.pos += 1;
        t
    }
}

/// Loop configuration.
#[derive(Debug, Clone)]
pub struct LoopConfig {
    /// Maximum tool turns before the forced final answer.
    pub step_limit: u64,
    /// Observation truncation cap, bytes (head-truncated with marker).
    pub obs_truncate: usize,
    /// Whole-trajectory token budget (chars/4 approximation).
    pub response_budget: Option<u64>,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            step_limit: 30,
            obs_truncate: 4000,
            response_budget: None,
        }
    }
}

/// One dispatched tool call's outcome in the transcript.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDispatch {
    /// The request as issued.
    pub req: ToolCallReq,
    /// Observation text folded back into the transcript.
    pub observation: String,
    /// `ok` / `tool_exception` / `transport_error`.
    pub kind: &'static str,
}

/// The rollout result.
#[derive(Debug, Clone, Default)]
pub struct RolloutResult {
    /// Full append-only transcript.
    pub messages: Vec<Message>,
    /// Final assistant text (empty when the policy never produced one).
    pub final_message: String,
    /// Whether the rollout ended with a policy final turn (vs. budget or
    /// step limit exhaustion).
    pub completed: bool,
    /// Tool turns used.
    pub tool_turns: u64,
    /// Every dispatch, in order.
    pub dispatches: Vec<ToolDispatch>,
    /// Approximate tokens consumed (chars/4).
    pub tokens: u64,
    /// Infra failure category, when the rollout aborted abnormally.
    pub infra_error: Option<InfraErrorKind>,
}

impl RolloutResult {
    /// Agent-side result for the verifier harness.
    pub fn to_agent_result(&self) -> crate::verifier::AgentRolloutResult {
        crate::verifier::AgentRolloutResult {
            final_message: self.final_message.clone(),
            completed: self.completed,
            tool_turns: self.tool_turns,
        }
    }
}

/// The agent loop.
pub struct AgentLoop {
    config: LoopConfig,
}

impl AgentLoop {
    /// New loop with a config.
    pub fn new(config: LoopConfig) -> Self {
        Self { config }
    }

    /// The default system prompt (the harness contract shown to the
    /// policy: tools, workspace, final-answer discipline).
    pub fn system_prompt(task: &str, tools: &[(String, ToolInfo)]) -> String {
        let tool_lines: Vec<String> = tools
            .iter()
            .map(|(s, t)| format!("- {}.{}: {}", s, t.name, t.description))
            .collect();
        format!(
            "You are an agent working in a sandboxed environment.\n\n\
             Task:\n{task}\n\n\
             Tools (call by qualified name):\n{}\n\n\
             Interact with the environment through the tools. When you have \
             the final answer, reply with it as plain text and no tool calls. \
             The environment records your session; the grader reads your last \
             assistant message.",
            tool_lines.join("\n")
        )
    }

    /// Runs one rollout to completion.
    pub fn run(
        &mut self,
        pod: &mut SimPod,
        task: &str,
        tools: &[(String, ToolInfo)],
        policy: &mut dyn Policy,
    ) -> RolloutResult {
        let fn_tools: Vec<(String, Value)> = tools
            .iter()
            .map(|(s, t)| (s.clone(), t.to_function_tool()))
            .collect();
        let mut messages = vec![
            Message::System(Self::system_prompt(task, tools)),
            Message::User(task.to_string()),
        ];
        let mut result = RolloutResult::default();
        let mut steps_left = self.config.step_limit;

        loop {
            let budget_exhausted = self
                .config
                .response_budget
                .map(|b| result.tokens >= b)
                .unwrap_or(false)
                || steps_left == 0;
            let ctx = TurnCtx {
                messages: &messages,
                tools: &fn_tools,
                steps_left,
                budget_exhausted,
            };
            let turn = policy.turn(&ctx);
            result.tokens += approx_tokens(&turn.text);

            // session log: every assistant turn is recorded (the verifier
            // extracts the LAST one)
            pod.append_session(&json!({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": turn.text}]},
                "tool_calls": turn.tool_calls.iter().map(|c| json!({
                    "id": c.id, "server": c.server, "tool": c.tool, "arguments": c.params,
                })).collect::<Vec<_>>(),
            }));

            if turn.tool_calls.is_empty() || budget_exhausted {
                messages.push(Message::Assistant {
                    text: turn.text.clone(),
                    calls: vec![],
                });
                if !turn.text.is_empty() {
                    result.final_message = turn.text;
                    result.completed = true;
                }
                if budget_exhausted && !turn.tool_calls.is_empty() {
                    result.infra_error = Some(InfraErrorKind::SeqTimeout);
                }
                break;
            }

            // dispatch (in order; each outcome becomes one user turn)
            let calls = turn.tool_calls.clone();
            messages.push(Message::Assistant {
                text: turn.text.clone(),
                calls: calls.clone(),
            });
            result.tokens += calls
                .iter()
                .map(|c| approx_tokens(&c.params.to_string()))
                .sum::<u64>();
            let mut any_transport = false;
            for req in calls {
                let dispatch = self.dispatch(pod, tools, &req);
                if dispatch.kind == "transport_error" {
                    any_transport = true;
                }
                result.tokens += approx_tokens(&dispatch.observation);
                messages.push(Message::User(format!(
                    "[tool {} {}]\n{}",
                    req.id,
                    req.qualified(),
                    dispatch.observation
                )));
                result.dispatches.push(dispatch);
            }
            result.tool_turns += 1;
            steps_left = steps_left.saturating_sub(1);

            if any_transport {
                result.infra_error = Some(InfraErrorKind::PodConnTimeout);
                break;
            }
        }

        result.messages = messages;
        result
    }

    fn dispatch(
        &self,
        pod: &mut SimPod,
        tools: &[(String, ToolInfo)],
        req: &ToolCallReq,
    ) -> ToolDispatch {
        // resolve server by qualified or bare name
        let (server, tool) = if !req.server.is_empty() {
            (req.server.clone(), req.tool.clone())
        } else {
            match tools.iter().find(|(_, t)| t.name == req.tool) {
                Some((s, t)) => (s.clone(), t.name.clone()),
                None => {
                    return ToolDispatch {
                        req: req.clone(),
                        observation: format!("tool {} is not available", req.tool),
                        kind: "tool_exception",
                    }
                }
            }
        };
        // server down? transport error (infra — not the agent's fault)
        let port = pod
            .servers
            .keys()
            .position(|k| *k == server)
            .map(|i| crate::manifest::MCP_PORT_BASE + i as u16);
        if let Some(p) = port {
            if !pod.port_live(p) {
                return ToolDispatch {
                    req: req.clone(),
                    observation: format!("mcp backend {} unreachable (port {})", server, p),
                    kind: "transport_error",
                };
            }
        } else {
            return ToolDispatch {
                req: req.clone(),
                observation: format!("unknown mcp server {server}"),
                kind: "transport_error",
            };
        }
        match pod.call_tool(&server, &tool, req.params.clone()) {
            Ok(v) => {
                let is_err = v.get("error").is_some();
                let text = truncate_observation(&v.to_string(), self.config.obs_truncate);
                ToolDispatch {
                    req: req.clone(),
                    observation: text,
                    kind: if is_err { "tool_exception" } else { "ok" },
                }
            }
            Err(e) => ToolDispatch {
                req: req.clone(),
                observation: truncate_observation(&e.to_string(), self.config.obs_truncate),
                kind: "transport_error",
            },
        }
    }
}

/// Observation truncation: head-truncate with a marker (upstream caps
/// tool outputs so one chatty tool cannot eat the context).
pub fn truncate_observation(text: &str, cap: usize) -> String {
    if text.len() <= cap {
        return text.to_string();
    }
    // cut on a char boundary
    let mut end = cap;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n...[truncated {} of {} bytes]",
        &text[..end],
        text.len() - end,
        text.len()
    )
}

/// Chars/4 token approximation (deterministic budget accounting).
pub fn approx_tokens(text: &str) -> u64 {
    (text.len() as u64).div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Schema, StateDb, Table};
    use crate::topology::ToolDef;
    use serde_json::json;

    fn tools_pod() -> SimPod {
        let schema = Schema::new().table(Table::text("customers", "id", &["id", "name", "status"]));
        let mut db = StateDb::new(schema);
        db.seed_rows(
            "customers",
            [json!({"id": "C1", "name": "Acme", "status": "active"})
                .as_object()
                .unwrap()
                .clone()],
        );
        let mut pod = SimPod::builder("p")
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
        // bring the server up
        pod.execute(
            "python3 /installed-agent/sidecar_entrypoint.py --start-and-detach",
            crate::manifest::SIDECAR,
            30,
        );
        pod
    }

    fn tool_surface(pod: &SimPod) -> Vec<(String, ToolInfo)> {
        crate::mcp::discover_pod_tools(pod)
    }

    #[test]
    fn happy_two_tool_rollout() {
        let mut pod = tools_pod();
        let tools = tool_surface(&pod);
        let mut policy = ScriptedPolicy::new(vec![
            (
                "let me check".into(),
                vec![("crm.get_customer".into(), json!({"id": "C1"}))],
            ),
            ("found".into(), vec![]),
        ]);
        let mut lp = AgentLoop::new(LoopConfig::default());
        let res = lp.run(&mut pod, "Find Acme's status", &tools, &mut policy);
        assert!(res.completed);
        assert_eq!(res.final_message, "found");
        assert_eq!(res.tool_turns, 1);
        assert_eq!(res.dispatches.len(), 1);
        assert_eq!(res.dispatches[0].kind, "ok");
        assert!(res.dispatches[0].observation.contains("Acme"));
        // transcript: system, user, assistant(tool), user(obs), assistant(final)
        assert_eq!(res.messages.len(), 5);
        // session log recorded two assistant turns
        let raw = String::from_utf8(
            pod.read_file(
                crate::manifest::MAIN,
                "/tmp/mimo-claude-logs/sessions/session-0.jsonl",
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(raw.lines().count(), 2);
    }

    #[test]
    fn tool_exception_is_user_visible_not_infra() {
        let mut pod = tools_pod();
        let tools = tool_surface(&pod);
        let mut policy = ScriptedPolicy::new(vec![
            (
                "trying a bad id".into(),
                vec![("crm.get_customer".into(), json!({"id": "X"}))],
            ),
            ("no such customer, done".into(), vec![]),
        ]);
        let mut lp = AgentLoop::new(LoopConfig::default());
        let res = lp.run(&mut pod, "t", &tools, &mut policy);
        assert!(res.completed);
        assert_eq!(res.dispatches[0].kind, "tool_exception");
        assert!(res.dispatches[0].observation.contains("no such customer"));
        assert!(res.infra_error.is_none());
    }

    #[test]
    fn transport_error_aborts_rollout() {
        let mut pod = tools_pod();
        pod.listening.clear(); // kill the MCP backend mid-flight
        let tools = tool_surface(&pod);
        let mut policy = ScriptedPolicy::new(vec![(
            "checking".into(),
            vec![("crm.get_customer".into(), json!({"id": "C1"}))],
        )]);
        let mut lp = AgentLoop::new(LoopConfig::default());
        let res = lp.run(&mut pod, "t", &tools, &mut policy);
        assert!(!res.completed);
        assert_eq!(res.dispatches[0].kind, "transport_error");
        assert_eq!(res.infra_error, Some(InfraErrorKind::PodConnTimeout));
    }

    #[test]
    fn step_limit_forces_final_turn() {
        let mut pod = tools_pod();
        let tools = tool_surface(&pod);
        // script that never stops calling tools
        let endless: Vec<(String, Vec<(String, Value)>)> = (0..50)
            .map(|i| {
                (
                    format!("turn {i}"),
                    vec![("crm.get_customer".into(), json!({"id": "C1"}))],
                )
            })
            .collect();
        let mut policy = ScriptedPolicy::new(endless);
        let mut lp = AgentLoop::new(LoopConfig {
            step_limit: 3,
            ..Default::default()
        });
        let res = lp.run(&mut pod, "t", &tools, &mut policy);
        assert_eq!(res.tool_turns, 3);
        assert!(res.infra_error.is_some());
    }

    #[test]
    fn response_budget_exhaustion_forces_wrap_up() {
        let mut pod = tools_pod();
        let tools = tool_surface(&pod);
        let mut policy = ScriptedPolicy::new(vec![
            (
                "a fairly long reply that consumes many tokens".into(),
                vec![("crm.get_customer".into(), json!({"id": "C1"}))],
            ),
            ("final answer now".into(), vec![]),
        ]);
        let cfg = LoopConfig {
            response_budget: Some(1),
            ..Default::default()
        };
        let mut lp = AgentLoop::new(cfg);
        let res = lp.run(&mut pod, "t", &tools, &mut policy);
        // turn 1 legitimately ran (budget checked before the turn);
        // turn 2 was forced into a no-tools final answer
        assert_eq!(res.tool_turns, 1);
        assert!(res.completed);
        assert_eq!(res.final_message, "final answer now");
    }

    #[test]
    fn observation_truncation_marker() {
        let long = "x".repeat(100);
        let t = truncate_observation(&long, 10);
        assert!(t.starts_with("xxxxxxxxxx"));
        assert!(t.contains("[truncated 90 of 100 bytes]"));
        // char-boundary safety
        let uni = "é".repeat(40);
        let t2 = truncate_observation(&uni, 5);
        assert!(t2.starts_with("éé"));
    }

    #[test]
    fn system_prompt_lists_qualified_tools() {
        let s = AgentLoop::system_prompt(
            "do it",
            &[(
                "crm".to_string(),
                ToolInfo {
                    name: "get_customer".into(),
                    description: "Fetch a customer".into(),
                    input_schema: json!({}),
                },
            )],
        );
        assert!(s.contains("- crm.get_customer: Fetch a customer"));
        assert!(s.contains("final answer"));
    }
}
