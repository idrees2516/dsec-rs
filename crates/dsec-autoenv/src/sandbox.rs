//! The proposer sandbox: resources, egress rules, web-tool gating, child
//! sandboxes, and the Figure-19 image/runtime configuration.
//!
//! "The proposer works in a sandbox, an isolated container with a
//! *workspace*, child containers for building and testing environments, and
//! restricted web access." (Section 3.1). Table 10 pins the resources:
//! proposer 16 CPUs / 32 GiB / 2-hour episode budget / no turn limit; solver
//! 4 CPUs / 8 GiB. "Each proposer runs in a coding agent: Claude Code for
//! Claude Opus 5, Codex for the GPT models, and the Pi coding agent for all
//! other models. All solvers run in the Pi coding agent."
//!
//! Egress: "allow = [search, packages, services]" while "egress rules block
//! Terminal-Bench and other benchmark pages." Web tools are only usable
//! when the runtime preamble lists them (the auto_env_scaling skill's
//! gating rule).

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// The coding harness an agent runs in (Table 10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodingAgent {
    /// Claude Code (used by Claude models).
    ClaudeCode,
    /// Codex (used by the GPT models).
    Codex,
    /// The Pi coding agent (all other models; all solvers).
    Pi,
}

impl CodingAgent {
    /// Name as rendered in logs.
    pub fn name(&self) -> &'static str {
        match self {
            CodingAgent::ClaudeCode => "claude-code",
            CodingAgent::Codex => "codex",
            CodingAgent::Pi => "pi",
        }
    }
}

/// Sandbox resources (Table 10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SandboxResources {
    /// CPU cores.
    pub cpus: u32,
    /// Memory in GiB.
    pub memory_gib: u32,
    /// Wall-clock budget per episode, in (virtual) seconds; `None` = no
    /// budget. Proposer: 2 hours. Solver: no explicit budget.
    pub time_budget_sec: Option<f64>,
    /// Turn limit; `None` = none (the proposer has no turn limit).
    pub turn_limit: Option<u32>,
    /// The coding harness.
    pub agent: CodingAgent,
}

impl SandboxResources {
    /// Table 10's proposer sandbox: 16 CPUs, 32 GiB, 2h, no turn limit.
    pub fn proposer(agent: CodingAgent) -> Self {
        SandboxResources {
            cpus: 16,
            memory_gib: 32,
            time_budget_sec: Some(2.0 * 3600.0),
            turn_limit: None,
            agent,
        }
    }

    /// Table 10's solver sandbox: 4 CPUs, 8 GiB, always the Pi agent.
    pub fn solver() -> Self {
        SandboxResources {
            cpus: 4,
            memory_gib: 8,
            time_budget_sec: None,
            turn_limit: None,
            agent: CodingAgent::Pi,
        }
    }
}

/// The allowed egress classes (Figure 19: `allow = ["search", "packages",
/// "services"]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EgressClass {
    /// Web search endpoints.
    Search,
    /// Package registries (PyPI, apt, Docker Hub).
    Packages,
    /// The flywheel's own service endpoints (validation, calibration,
    /// decontamination).
    Services,
}

impl EgressClass {
    /// The TOML tag of this class.
    pub fn name(&self) -> &'static str {
        match self {
            EgressClass::Search => "search",
            EgressClass::Packages => "packages",
            EgressClass::Services => "services",
        }
    }

    /// Hosts representing this class.
    pub fn hosts(&self) -> &'static [&'static str] {
        match self {
            EgressClass::Search => &["search.example.com", "www.google.com", "www.bing.com"],
            EgressClass::Packages => &[
                "pypi.org",
                "files.pythonhosted.org",
                "registry-1.docker.io",
                "deb.debian.org",
            ],
            EgressClass::Services => &[
                "validation.internal",
                "calibration.internal",
                "decontamination.internal",
            ],
        }
    }
}

/// Held-out benchmark sources blocked for the proposer ("egress rules block
/// Terminal-Bench and other benchmark pages").
pub const BLOCKED_BENCHMARK_HOSTS: &[&str] = &[
    "terminal-bench.ai",
    "www.terminal-bench.com",
    "github.com/laude-institute/terminal-bench",
    "raw.githubusercontent.com/laude-institute/terminal-bench",
    "huggingface.co/datasets/terminal-bench",
];

/// The egress policy of a sandbox.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EgressPolicy {
    /// Allowed egress classes.
    pub allow: Vec<EgressClass>,
    /// Explicitly blocked hosts (benchmark sources).
    pub blocked_hosts: Vec<String>,
}

impl Default for EgressPolicy {
    fn default() -> Self {
        EgressPolicy {
            allow: vec![
                EgressClass::Search,
                EgressClass::Packages,
                EgressClass::Services,
            ],
            blocked_hosts: BLOCKED_BENCHMARK_HOSTS
                .iter()
                .map(|h| h.to_string())
                .collect(),
        }
    }
}

impl EgressPolicy {
    /// Check a host against the policy. Benchmark pages are always denied;
    /// everything else must fall into an allowed class.
    pub fn check(&self, host: &str) -> Result<()> {
        if self.blocked_hosts.iter().any(|h| host == h.as_str())
            || BLOCKED_BENCHMARK_HOSTS.contains(&host)
        {
            return Err(Error::EgressDenied {
                host: host.to_string(),
                rule: "held-out benchmark source blocked".into(),
            });
        }
        let allowed = self
            .allow
            .iter()
            .flat_map(|c| c.hosts().iter())
            .any(|h| *h == host);
        if allowed {
            Ok(())
        } else {
            Err(Error::EgressDenied {
                host: host.to_string(),
                rule: "not in an allowed egress class".into(),
            })
        }
    }
}

/// A running sandbox for one agent episode.
#[derive(Debug, Clone)]
pub struct Sandbox {
    /// Resources granted to this sandbox.
    pub resources: SandboxResources,
    /// Egress policy.
    pub egress: EgressPolicy,
    /// Tools the runtime preamble actually lists (web gating). Empty means
    /// no web access.
    pub available_tools: Vec<String>,
    /// Virtual wall-clock consumed so far.
    pub elapsed_sec: f64,
    /// Child sandboxes spawned for build/test work.
    pub children: Vec<ChildSandbox>,
}

impl Sandbox {
    /// Create a sandbox with the default egress policy.
    pub fn new(resources: SandboxResources, available_tools: Vec<String>) -> Self {
        Sandbox {
            resources,
            egress: EgressPolicy::default(),
            available_tools,
            elapsed_sec: 0.0,
            children: Vec::new(),
        }
    }

    /// The web-tool gate: `web_search` / `web_fetch` / `web_search.py` are
    /// only usable "when the runtime preamble explicitly lists a web tool
    /// (`web_search`/`web_fetch` or `web_search.py`)".
    pub fn check_tool(&self, tool: &str) -> Result<()> {
        let is_web_tool = tool == "web_search" || tool == "web_fetch" || tool == "web_search.py";
        if !is_web_tool {
            return Ok(());
        }
        // Per-tool gating: the preamble must list this tool (the
        // `web_search.py` script satisfies `web_search`).
        let listed = self
            .available_tools
            .iter()
            .any(|t| t == tool || (tool == "web_search" && t == "web_search.py"));
        if listed {
            Ok(())
        } else {
            Err(Error::ToolUnavailable {
                tool: tool.to_string(),
                detail: "no web tool in the runtime preamble; do not attempt searches".into(),
            })
        }
    }

    /// Check an egress attempt.
    pub fn check_egress(&self, host: &str) -> Result<()> {
        self.egress.check(host)
    }

    /// Consume virtual wall-clock time; errors when the episode budget is
    /// exhausted (the proposer's 2 hours — Figure 23 shows GPT-5.6-sol
    /// "hit the wall clock").
    pub fn tick(&mut self, seconds: f64) -> Result<()> {
        self.elapsed_sec += seconds;
        match self.resources.time_budget_sec {
            Some(budget) if self.elapsed_sec > budget => Err(Error::Invariant(format!(
                "wall-clock budget exhausted: {:.0}s > {:.0}s",
                self.elapsed_sec, budget
            ))),
            _ => Ok(()),
        }
    }

    /// Whether the turn limit (if any) is still open after `turns` turns.
    pub fn within_turn_limit(&self, turns: u32) -> bool {
        match self.resources.turn_limit {
            Some(limit) => turns < limit,
            None => true,
        }
    }

    /// Spawn a child sandbox for building or testing a task environment
    /// ("It can launch child sandboxes to build and test tasks").
    pub fn spawn_child(&mut self, purpose: ChildPurpose) -> ChildSandbox {
        let child = ChildSandbox {
            id: format!("child_{}", self.children.len() + 1),
            purpose,
            cpus: 4,
            memory_gib: 8,
        };
        self.children.push(child.clone());
        child
    }
}

/// What a child sandbox is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChildPurpose {
    /// Building a task environment image.
    Build,
    /// Running a reference solution and tests.
    Test,
    /// Running the current solver for difficulty estimation.
    SolverEstimate,
}

/// A child sandbox: a throwaway container for build/test work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChildSandbox {
    /// Child id.
    pub id: String,
    /// Why it was spawned.
    pub purpose: ChildPurpose,
    /// CPU cores.
    pub cpus: u32,
    /// Memory in GiB.
    pub memory_gib: u32,
}

/// `sandbox.toml` — the runtime configuration supplied at launch (Figure
/// 19): service endpoints, retained memory, and network rules.
///
/// "At launch, configuration supplies service endpoints and retained
/// memory (Figure 19). Credentials are supplied separately; network
/// restrictions must be enforced by the sandbox infrastructure."
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// `[agent]`: the coding harness name.
    pub harness: String,
    /// `[agent]`: workspace root.
    pub workspace: String,
    /// `[services]`: validation endpoint.
    pub validation_endpoint: String,
    /// `[services]`: calibration endpoint.
    pub calibration_endpoint: String,
    /// `[services]`: decontamination endpoint.
    pub decontamination_endpoint: String,
    /// `[network]`: allowed egress classes.
    pub allow: Vec<EgressClass>,
    /// `[network]`: blocked sources.
    pub block: Vec<String>,
    /// `[memory]`: seed memory path.
    pub seed: String,
    /// `[memory]`: previous-round restore path.
    pub restore: String,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        SandboxConfig {
            harness: "pi".into(),
            workspace: "/workspace".into(),
            validation_endpoint: "http://validation.internal".into(),
            calibration_endpoint: "http://calibration.internal".into(),
            decontamination_endpoint: "http://decontamination.internal".into(),
            allow: vec![
                EgressClass::Search,
                EgressClass::Packages,
                EgressClass::Services,
            ],
            block: vec!["held-out benchmark sources".into()],
            seed: "/opt/seed_memory".into(),
            restore: "${PREVIOUS_ROUND}".into(),
        }
    }
}

impl SandboxConfig {
    /// Render `sandbox.toml`.
    pub fn render_toml(&self) -> String {
        let _allow: Vec<String> = self.allow.iter().map(|c| c.name().to_string()).collect();
        format!(
            "[agent]\nharness = \"{}\"\nworkspace = \"{}\"\n\n[services]\nvalidation = \"{}\"\ncalibration = \"{}\"\ndecontamination = \"{}\"\n\n[network]\nallow = [{}]\nblock = [{}]\n\n[memory]\nseed = \"{}\"\nrestore = \"{}\"\n",
            self.harness,
            self.workspace,
            self.validation_endpoint,
            self.calibration_endpoint,
            self.decontamination_endpoint,
            self.allow.iter().map(|a| format!("\"{}\"", a.name())).collect::<Vec<_>>().join(", "),
            self.block.iter().map(|b| format!("\"{b}\"")).collect::<Vec<_>>().join(", "),
            self.seed,
            self.restore,
        )
    }
}

/// The proposer sandbox image (Figure 19's Dockerfile, simplified as the
/// paper presents it).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SandboxImage {
    /// Base image.
    pub base: String,
    /// Harness packages installed.
    pub installs: Vec<String>,
}

impl Default for SandboxImage {
    fn default() -> Self {
        SandboxImage {
            base: "python:3.12-slim".into(),
            installs: vec![
                "runtime".into(),
                "harbor".into(),
                "sandbox_client".into(),
                "coding_harness".into(),
            ],
        }
    }
}

impl SandboxImage {
    /// Render the Dockerfile.
    pub fn render_dockerfile(&self) -> String {
        let mut df = format!("FROM {}\n", self.base);
        for group in self.installs.chunks(2) {
            df.push_str(&format!("INSTALL {}\n", group.join(", ")));
        }
        df.push_str("# Pi / Claude Code / Codex\n");
        for (src, dst) in [
            ("skills/", "/opt/skills/"),
            ("tools/", "/opt/tools/"),
            ("clients/", "/opt/clients/"),
            ("templates/", "/opt/templates/"),
            ("seed_memory/", "/opt/seed_memory/"),
        ] {
            df.push_str(&format!("COPY {src} {dst}\n"));
        }
        df.push_str(
            "COPY sandbox.toml /opt/config/\n\nWORKDIR /workspace\nENTRYPOINT [\"/opt/init.sh\"]\n",
        );
        df
    }
}

/// The `SANDBOX_CONTRACT.md` content: the rules of the proposer's sandbox.
pub fn sandbox_contract_md() -> String {
    let mut s = String::from("# Sandbox contract\n\n");
    s.push_str("This sandbox hosts one proposer episode of the AutoEnvScaling flywheel.\n\n");
    s.push_str("## Resources (Table 10)\n\n");
    s.push_str("- 16 CPUs, 32 GiB memory.\n- Wall-clock budget: 2 hours per episode. Episodes that exceed it are cut (the wall clock, not a turn limit).\n- No turn limit: work until the environment is finished.\n- A coding harness (Claude Code / Codex / Pi) drives the terminal.\n\n");
    s.push_str("## Workspace rules\n\n");
    s.push_str("- Editable: `tools/`, `cookbook/`, `memory/` (your own notes and lessons), `output/`, `logs/`.\n- Host-managed (read-only): `memory/past_environments/`, `memory/past_trajectories/` — refreshed from the training pool each round.\n- Fixed: `skills/` (authoring instructions), `assignment/` (the current assignment), and this contract. Do not modify them during generation.\n- The final admission checks (validation, calibration, decontamination) run OUTSIDE this sandbox on the host. You cannot edit them; only satisfy them.\n\n");
    s.push_str("## Network\n\n");
    s.push_str("- Allowed classes: search, packages, services.\n- Blocked: Terminal-Bench and other held-out benchmark pages, at the infrastructure level.\n- Web tools (`web_search` / `web_fetch`) work only when the runtime preamble lists them. If no such preamble is present, you have no web access; do not attempt searches.\n\n");
    s.push_str("## Child sandboxes\n\n");
    s.push_str("- You may spawn child containers to build images and run reference solutions and tests.\n- Child sandboxes are throwaway: they never persist between episodes.\n\n");
    s.push_str("## Submission\n\n");
    s.push_str("- Ship the finished canonical task (instruction.md, task.toml, environment/, solution/, tests/) into `output/tasks/<env_id>/`.\n- Submit only when the oracle passes AND the no-op fails.\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table10_resources() {
        let p = SandboxResources::proposer(CodingAgent::Pi);
        assert_eq!(p.cpus, 16);
        assert_eq!(p.memory_gib, 32);
        assert_eq!(p.time_budget_sec, Some(7200.0));
        assert_eq!(p.turn_limit, None);

        let s = SandboxResources::solver();
        assert_eq!(s.cpus, 4);
        assert_eq!(s.memory_gib, 8);
        assert_eq!(s.agent, CodingAgent::Pi);
    }

    #[test]
    fn egress_blocks_benchmarks() {
        let policy = EgressPolicy::default();
        assert!(policy.check("terminal-bench.ai").is_err());
        assert!(policy
            .check("github.com/laude-institute/terminal-bench")
            .is_err());
        // Allowed classes pass.
        assert!(policy.check("pypi.org").is_ok());
        assert!(policy.check("www.google.com").is_ok());
        // Unknown host denied.
        assert!(policy.check("evil.example.org").is_err());
    }

    #[test]
    fn web_tool_gating() {
        let sb = Sandbox::new(SandboxResources::solver(), vec!["bash".into()]);
        assert!(sb.check_tool("bash").is_ok());
        assert!(sb.check_tool("web_search").is_err());
        assert!(sb.check_tool("web_fetch").is_err());

        let sb_web = Sandbox::new(
            SandboxResources::solver(),
            vec!["bash".into(), "web_search".into()],
        );
        assert!(sb_web.check_tool("web_search").is_ok());
        assert!(sb_web.check_tool("web_fetch").is_err()); // only web_search is listed
    }

    #[test]
    fn wall_clock_budget() {
        let mut sb = Sandbox::new(SandboxResources::proposer(CodingAgent::Pi), vec![]);
        assert!(sb.tick(3600.0).is_ok());
        assert!(sb.tick(3599.0).is_ok());
        let err = sb.tick(2.0).unwrap_err();
        assert!(err.to_string().contains("budget"));
    }

    #[test]
    fn child_sandboxes() {
        let mut sb = Sandbox::new(SandboxResources::proposer(CodingAgent::Codex), vec![]);
        let c1 = sb.spawn_child(ChildPurpose::Build);
        let c2 = sb.spawn_child(ChildPurpose::Test);
        assert_eq!(c1.id, "child_1");
        assert_eq!(c2.id, "child_2");
        assert_eq!(sb.children.len(), 2);
        assert_eq!(c1.cpus, 4);
    }

    #[test]
    fn sandbox_toml_renders() {
        let cfg = SandboxConfig::default();
        let toml = cfg.render_toml();
        assert!(toml.contains("[agent]"));
        assert!(toml.contains("harness = \"pi\""));
        assert!(toml.contains("[services]"));
        assert!(toml.contains("validation = \"http://validation.internal\""));
        assert!(toml.contains("[network]"));
        assert!(toml.contains("\"search\""));
        assert!(toml.contains("[memory]"));
        assert!(toml.contains("restore = \"${PREVIOUS_ROUND}\""));
    }

    #[test]
    fn sandbox_dockerfile_renders() {
        let df = SandboxImage::default().render_dockerfile();
        assert!(df.starts_with("FROM python:3.12-slim"));
        assert!(df.contains("COPY skills/ /opt/skills/"));
        assert!(df.contains("COPY tools/ /opt/tools/"));
        assert!(df.contains("COPY seed_memory/ /opt/seed_memory/"));
        assert!(df.contains("WORKDIR /workspace"));
        assert!(df.contains("ENTRYPOINT [\"/opt/init.sh\"]"));
    }

    #[test]
    fn contract_mentions_fixed_regions() {
        let contract = sandbox_contract_md();
        assert!(contract.contains("# Sandbox contract"));
        assert!(contract.contains("Fixed"));
        assert!(contract.contains("OUTSIDE this sandbox"));
        assert!(contract.contains("2 hours"));
        assert!(contract.contains("no web access"));
    }
}
