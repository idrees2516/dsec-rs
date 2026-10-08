//! The Harbor task model: what a proposer episode must ship.
//!
//! Every proposer episode produces one environment as a complete Harbor task
//! (Section 3.1): an instruction, an execution environment, a reference
//! solution, and tests. "The solver sees only the instruction and works
//! inside the environment's container, while the reference solution and tests
//! stay hidden."
//!
//! This module covers the full authoring surface the paper engages with
//! (Appendix E.1): the scaffold produced by `harbor task init`, `task.toml`
//! with the task / metadata / agent / verifier / environment sections, the
//! three-layer network policy (baselines, phase overrides, run-time merges),
//! shared vs. separate verifier environments, the multi-step `steps/` layout
//! with per-step `min_reward` gating and the `mean` / `final` reward
//! roll-up strategies, and the common authoring pitfalls encoded as checks.

use crate::error::{Error, Result};
use crate::world::{SolutionScript, TestSuite};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A skill/domain tag used to steer curriculum difficulty (an "additional
/// skill" when hardening, Section 3.1).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SkillTag(pub String);

impl SkillTag {
    /// Construct a tag from any string.
    pub fn new(name: impl Into<String>) -> Self {
        SkillTag(name.into())
    }

    /// The tag name.
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SkillTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Network access mode for a container phase.
///
/// Modes (Appendix E.1 Step 6): `public`, `no-network`, or `allowlist` with
/// `allowed_hosts = ["pypi.org"]` — exact hostnames, IPv4/IPv6 address
/// literals or CIDR ranges, or leading wildcard hostnames; *not* URLs, ports,
/// or paths.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkMode {
    /// Unrestricted network access (the default when `[environment]` omits
    /// `network_mode`).
    Public,
    /// Fully offline. The skill mandates `network_mode = "no-network"` at
    /// solve time ("Self-contained & deterministic: no network at solve
    /// time").
    NoNetwork,
    /// Allow only the listed hosts.
    Allowlist {
        /// Exact hostnames, IP literals, CIDRs, or leading wildcards.
        allowed_hosts: Vec<String>,
    },
}

impl NetworkMode {
    /// The TOML representation used in `task.toml`.
    pub fn toml_name(&self) -> &str {
        match self {
            NetworkMode::Public => "public",
            NetworkMode::NoNetwork => "no-network",
            NetworkMode::Allowlist { .. } => "allowlist",
        }
    }

    /// Whether a host is reachable under this mode.
    pub fn allows_host(&self, host: &str) -> bool {
        match self {
            NetworkMode::Public => true,
            NetworkMode::NoNetwork => false,
            NetworkMode::Allowlist { allowed_hosts } => {
                allowed_hosts.iter().any(|h| host_matches(h, host))
            }
        }
    }

    /// Validate that every entry in an allowlist is a legal host spec: exact
    /// hostname, IPv4/IPv6 literal, CIDR range, or leading wildcard. URLs,
    /// ports, and paths are rejected (Appendix E.1).
    pub fn validate_allowed_hosts(&self) -> Result<()> {
        let NetworkMode::Allowlist { allowed_hosts } = self else {
            return Ok(());
        };
        for h in allowed_hosts {
            if !is_valid_allowed_host(h) {
                return Err(Error::invalid_task(format!(
                    "allowed_hosts entry {h:?} is not an exact hostname, IP literal, CIDR, or leading wildcard (no URLs, ports, or paths)"
                )));
            }
        }
        Ok(())
    }
}

/// Whether `spec` (an allowlist entry) covers `host`.
fn host_matches(spec: &str, host: &str) -> bool {
    if let Some(suffix) = spec.strip_prefix("*.") {
        return host.strip_suffix(&format!(".{suffix}")).is_some() || host == suffix;
    }
    if let Some(cidr_prefix) = spec.rsplit_once('/') {
        // CIDR: compare the address bits.
        if let (Some(base), Ok(bits)) = (parse_ip_v4(cidr_prefix.0), cidr_prefix.1.parse::<u32>()) {
            if let Some(h) = parse_ip_v4(host) {
                let mask = if bits == 0 {
                    0
                } else {
                    u32::MAX << (32 - bits.min(32))
                };
                return (base & mask) == (h & mask);
            }
        }
        return false;
    }
    spec == host
}

/// Parse a dotted-quad IPv4 address into its bits.
fn parse_ip_v4(s: &str) -> Option<u32> {
    let parts: Vec<u32> = s
        .split('.')
        .map(|p| p.parse::<u32>().ok())
        .collect::<Option<Vec<_>>>()?;
    if parts.len() != 4 || parts.iter().any(|p| *p > 255) {
        return None;
    }
    Some((parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3])
}

/// Validate a single allowlist host entry: exact hostname, IPv4/IPv6
/// literal, CIDR range, or leading wildcard. URLs, ports, and paths are
/// rejected.
fn is_valid_allowed_host(h: &str) -> bool {
    if h.is_empty() || h.len() > 253 {
        return false;
    }
    if h.contains("://") {
        return false; // URL form
    }
    let body = h.strip_prefix("*.").unwrap_or(h);
    if body.is_empty() {
        return false;
    }
    if let Some((base, bits)) = body.split_once('/') {
        // CIDR: base must be an IP literal, bits numeric and <= 128.
        if bits.contains('/') {
            return false;
        }
        let Ok(bits) = bits.parse::<u32>() else {
            return false;
        };
        return bits <= 128 && (parse_ip_v4(base).is_some() || is_ipv6_literal(base));
    }
    if body.contains('/') {
        return false; // path
    }
    if parse_ip_v4(body).is_some() {
        return true; // IPv4 literal
    }
    if body.contains(':') {
        return is_ipv6_literal(body); // IPv6 literal; ports get rejected here
    }
    // Hostname: labels of [a-z0-9-].
    body.split('.').all(|label| {
        !label.is_empty()
            && label
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    })
}

/// Loose IPv6 literal check: hex digits, colons, at most one `::`, and at
/// least one colon. Rejects `host:port` forms (non-hex port).
fn is_ipv6_literal(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    s.chars()
        .all(|c| c.is_ascii_hexdigit() || c == ':' || c == '.')
        && s.matches("::").count() <= 1
        && s.contains(':')
}

/// The verifier's container arrangement (Appendix E.1 Step 4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VerifierMode {
    /// Default: the verifier runs in the agent container; the network
    /// baseline is `[environment]`.
    Shared,
    /// `tests/` is the verifier image build context and the image must
    /// provide `/tests/test.sh`. Harbor copies `/logs/artifacts` and
    /// configured artifacts into the verifier environment, *not* the agent's
    /// whole workspace.
    Separate {
        /// Baseline network mode of the verifier environment.
        network_mode: NetworkMode,
        /// Verifier image (used as the build context base).
        docker_image: Option<String>,
    },
}

impl VerifierMode {
    /// TOML fragment rendered under `[verifier]`.
    pub fn toml_fragment(&self) -> String {
        match self {
            VerifierMode::Shared => String::new(),
            VerifierMode::Separate {
                network_mode,
                docker_image,
            } => {
                let mut s = String::from(
                    "[verifier]\nenvironment_mode = \"separate\"\n\n[verifier.environment]\n",
                );
                s.push_str(&format!(
                    "network_mode = \"{}\"\n",
                    network_mode.toml_name()
                ));
                if let Some(img) = docker_image {
                    s.push_str(&format!("docker_image = \"{img}\"\n"));
                }
                s
            }
        }
    }
}

/// Task difficulty for `[metadata]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Difficulty {
    /// Easy.
    Easy,
    /// Medium.
    Medium,
    /// Hard.
    Hard,
}

impl Difficulty {
    /// Rendered value in `task.toml`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Difficulty::Easy => "easy",
            Difficulty::Medium => "medium",
            Difficulty::Hard => "hard",
        }
    }
}

/// The `[environment]` container definition: either a Dockerfile to build or
/// a pre-built image plus runtime files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskEnvironment {
    /// Base image (e.g. `ubuntu:24.04`, `python:3.12-slim`).
    pub base_image: String,
    /// Packages the task requires — installed by the Dockerfile, *not* the
    /// solution (pitfall: "Installing the solution into the Dockerfile →
    /// agent already gets the answer").
    pub packages: Vec<String>,
    /// Files planted into the container workdir at environment start (task
    /// state: data files, configs, manifests).
    pub files: BTreeMap<String, String>,
    /// Working directory inside the container.
    pub workdir: String,
    /// Baseline network mode applied at environment start.
    pub network_mode: NetworkMode,
    /// CPU cores.
    pub cpus: u32,
    /// RAM in MB.
    pub memory_mb: u32,
    /// Disk in MB.
    pub storage_mb: u32,
    /// Optional GPU count.
    pub gpus: Option<u32>,
    /// Optional GPU type constraint.
    pub gpu_types: Option<Vec<String>>,
    /// Extra allowed hosts merged in at run time by
    /// `--allow-environment-host`.
    pub extra_allowed_hosts: Vec<String>,
}

impl Default for TaskEnvironment {
    fn default() -> Self {
        TaskEnvironment {
            base_image: "ubuntu:24.04".into(),
            packages: Vec::new(),
            files: BTreeMap::new(),
            workdir: "/app".into(),
            network_mode: NetworkMode::Public,
            cpus: 1,
            memory_mb: 2048,
            storage_mb: 10240,
            gpus: None,
            gpu_types: None,
            extra_allowed_hosts: Vec::new(),
        }
    }
}

impl TaskEnvironment {
    /// Render a Dockerfile for this environment.
    pub fn render_dockerfile(&self) -> String {
        let mut df = format!("FROM {}\nWORKDIR {}\n", self.base_image, self.workdir);
        if !self.packages.is_empty() {
            df.push_str(&format!(
                "RUN apt-get update && apt-get install -y {} && rm -rf /var/lib/apt/lists/*\n",
                self.packages.join(" ")
            ));
        }
        for (path, content) in &self.files {
            df.push_str(&format!("COPY data/{path} {path}\n"));
            let _ = content;
        }
        df
    }

    /// Effective baseline mode after merging run-time `--allow-environment-host`
    /// flags into `extra_allowed_hosts` (Appendix E.1: merged into
    /// `[environment] extra_allowed_hosts` → `[environment]` baseline).
    ///
    /// On a `public` baseline, run-time host flags "emit a warning and are
    /// ignored".
    pub fn baseline_with_runtime_hosts(
        &self,
        extra_hosts: &[String],
    ) -> (NetworkMode, Vec<String>) {
        let mut warnings = Vec::new();
        match &self.network_mode {
            NetworkMode::Public => {
                if !extra_hosts.is_empty() {
                    warnings.push(
                        "run-time host flags ignored on a public baseline (warning)".to_string(),
                    );
                }
                (NetworkMode::Public, warnings)
            }
            NetworkMode::NoNetwork => {
                if !extra_hosts.is_empty() {
                    warnings.push(
                        "run-time host flags ignored on a no-network baseline (warning)"
                            .to_string(),
                    );
                }
                (NetworkMode::NoNetwork, warnings)
            }
            NetworkMode::Allowlist { allowed_hosts } => {
                let mut merged = allowed_hosts.clone();
                merged.extend(extra_hosts.iter().cloned());
                let mut all = self.extra_allowed_hosts.clone();
                all.extend(extra_hosts.iter().cloned());
                merged.extend(all);
                merged.sort();
                merged.dedup();
                (
                    NetworkMode::Allowlist {
                        allowed_hosts: merged,
                    },
                    warnings,
                )
            }
        }
    }
}

/// A phase override for the agent or verifier phase. Overrides are only
/// applied when set **and** different from the phase baseline; "Matching the
/// baseline is a no-op" (Appendix E.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PhaseOverride {
    /// Network mode override for the phase.
    pub network_mode: Option<NetworkMode>,
    /// Extra allowed hosts merged into this phase's allowlist by run-time
    /// flags (`--allow-agent-host` → agent phase).
    pub extra_allowed_hosts: Vec<String>,
}

impl PhaseOverride {
    /// Whether this override, if set, actually differs from the baseline.
    /// If it matches, applying it is a no-op.
    pub fn differs_from(&self, baseline: &NetworkMode) -> bool {
        match &self.network_mode {
            None => false,
            Some(mode) => mode != baseline,
        }
    }

    /// Resolve the effective mode for the phase: the override if it differs,
    /// otherwise the baseline.
    pub fn resolve(&self, baseline: &NetworkMode) -> NetworkMode {
        if self.differs_from(baseline) {
            self.network_mode.clone().unwrap()
        } else {
            baseline.clone()
        }
    }
}

/// How per-step rewards roll up into the trial-level verifier result
/// (multi-step tasks, Appendix E.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RewardStrategy {
    /// Per-key mean across steps that produced a result (the default). Good
    /// for continuous progress rewards.
    #[default]
    Mean,
    /// The last step's verifier result, verbatim. Caveat: "if `min_reward`
    /// triggers an early abort, `"final"` uses the *aborted* step's result,
    /// not the intended final step."
    Final,
}

impl RewardStrategy {
    /// Rendered value in `task.toml`.
    pub fn as_str(&self) -> &'static str {
        match self {
            RewardStrategy::Mean => "mean",
            RewardStrategy::Final => "final",
        }
    }
}

/// Per-step pass gate: a scalar threshold on the step's overall reward, or a
/// dict gating on specific keys of a multi-dimensional reward.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MinReward {
    /// Abort the trial if the step's overall reward is below this value.
    Scalar(f64),
    /// Abort the trial if any keyed reward is below its threshold
    /// (`min_reward = { correctness = 0.8, style = 0.5 }`).
    Keys(BTreeMap<String, f64>),
}

impl MinReward {
    /// Whether a step result with overall `reward` and keyed `rewards`
    /// passes this gate. Failing aborts the step and the trial.
    pub fn passes(&self, reward: f64, keyed: &BTreeMap<String, f64>) -> bool {
        match self {
            MinReward::Scalar(t) => reward >= *t,
            MinReward::Keys(gates) => gates
                .iter()
                .all(|(k, t)| keyed.get(k).map(|v| *v >= *t).unwrap_or(false)),
        }
    }
}

/// One step of a multi-step task. Steps share one container; files persist
/// across steps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepSpec {
    /// Step name; must match the directory under `steps/`.
    pub name: String,
    /// The prompt for this step.
    pub instruction: String,
    /// Files uploaded to WORKDIR before the agent runs.
    pub workdir: BTreeMap<String, String>,
    /// The reserved `workdir/setup.sh`: runs after the workdir upload and
    /// before the agent, as the step's agent user, with cwd = WORKDIR.
    /// Non-zero exit aborts the step and the trial. Should `rm -- "$0"` on
    /// its last line if the agent must not see it.
    pub setup_sh: Option<SolutionScript>,
    /// Per-step verifier.
    pub tests: TestSuite,
    /// Per-step Oracle solution.
    pub solution: Option<SolutionScript>,
    /// Pass gate for the step.
    pub min_reward: MinReward,
    /// Per-step agent timeout override.
    pub agent_timeout_sec: Option<f64>,
    /// Per-step verifier timeout override.
    pub verifier_timeout_sec: Option<f64>,
    /// Per-step network overrides.
    pub agent_network: PhaseOverride,
    /// Artifacts collected into `steps/{name}/artifacts/` after this step's
    /// verification.
    pub artifacts: Vec<String>,
}

impl StepSpec {
    /// Check the reserved-filename conventions for `setup.sh`.
    pub fn check_setup_conventions(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if let Some(setup) = &self.setup_sh {
            if !setup.executable {
                issues.push(format!(
                    "steps/{}/workdir/setup.sh is not executable (non-zero exit handling requires +x)",
                    self.name
                ));
            }
            let self_removing = setup
                .ops
                .iter()
                .any(|op| matches!(op, crate::world::ShellOp::RmSelf));
            if !self_removing {
                issues.push(format!(
                    "steps/{}/workdir/setup.sh does not `rm -- \"$0\"` on its last line; the agent will see it",
                    self.name
                ));
            }
        }
        issues
    }
}

/// The `task.toml` configuration sections the paper's tasks use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskToml {
    /// `schema_version` ("1.4" for multi-step tasks).
    pub schema_version: String,
    /// `[task]` section.
    pub task: TaskMeta,
    /// `[metadata]` section.
    pub metadata: TaskMetadata,
    /// `[agent]` phase override + timeout.
    pub agent: AgentConfig,
    /// `[verifier]` section.
    pub verifier: VerifierConfig,
    /// `multi_step_reward_strategy` (multi-step tasks only).
    pub reward_strategy: RewardStrategy,
}

/// The `[task]` section: name, version, description, keywords.
///
/// Keywords are "always populated — used for search / filtering": 3–8
/// lowercase tokens covering the domain, the verifier style, and notable
/// hardware. An empty keyword list leaves the task "invisible to registry
/// search".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskMeta {
    /// Full task name `"<org>/<task-name>"`.
    pub name: String,
    /// Task version.
    pub version: String,
    /// One-line description.
    pub description: String,
    /// 3–8 lowercase search/filter tokens.
    pub keywords: Vec<String>,
}

/// The `[metadata]` section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskMetadata {
    /// Difficulty bucket.
    pub difficulty: Difficulty,
    /// Category (programming, machine-learning, gpu, ...).
    pub category: String,
    /// Free-form tags.
    pub tags: Vec<String>,
}

/// The `[agent]` section: timeout plus optional phase override.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentConfig {
    /// How long the agent has (seconds).
    pub timeout_sec: f64,
    /// Phase override for the agent phase.
    pub network: PhaseOverride,
    /// Non-root user to run the agent as.
    pub user: Option<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            timeout_sec: 120.0,
            network: PhaseOverride::default(),
            user: None,
        }
    }
}

/// The `[verifier]` section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifierConfig {
    /// How long tests have (seconds).
    pub timeout_sec: f64,
    /// Shared vs. separate verifier environment.
    pub environment_mode: VerifierMode,
    /// Environment variables for the verifier (e.g. API keys for Reward Kit
    /// judges).
    pub env: BTreeMap<String, String>,
    /// Phase override for the verifier phase.
    pub network: PhaseOverride,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        VerifierConfig {
            timeout_sec: 600.0,
            environment_mode: VerifierMode::Shared,
            env: BTreeMap::new(),
            network: PhaseOverride::default(),
        }
    }
}

/// A complete Harbor task: instruction + environment + solution + tests,
/// optionally multi-step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarborTask {
    /// Environment id assigned by the flywheel (`env_007`) — distinct from
    /// the Harbor name.
    pub env_id: String,
    /// The `[task]` metadata.
    pub meta: TaskMeta,
    /// The instruction the solver sees (`instruction.md`).
    pub instruction: Instruction,
    /// The container definition (`environment/Dockerfile`).
    pub environment: TaskEnvironment,
    /// The reference solution (`solution/solve.sh`) — hidden from the solver.
    pub solution: SolutionScript,
    /// The verifier (`tests/`).
    pub tests: TestSuite,
    /// The populated `README.md` (never a stub).
    pub readme: String,
    /// Multi-step steps; empty for single-step tasks.
    pub steps: Vec<StepSpec>,
    /// Full `task.toml` model.
    pub toml: TaskToml,
    /// Skills this task exercises (drives "additional skill" hardening and
    /// solver skill-matching in the simulated world).
    pub skills: Vec<SkillTag>,
}

impl HarborTask {
    /// The canonical file layout of a single-step task.
    pub const CANONICAL_FILES: &'static [&'static str] = &[
        "instruction.md",
        "task.toml",
        "environment/Dockerfile",
        "solution/solve.sh",
        "tests/test.sh",
    ];

    /// Whether this is a multi-step task.
    pub fn is_multi_step(&self) -> bool {
        !self.steps.is_empty()
    }

    /// Total number of individual test checks across the task (the `N` of
    /// Eq. 1: the solver's reward is the fraction of these that pass).
    pub fn total_checks(&self) -> usize {
        if self.is_multi_step() {
            self.steps.iter().map(|s| s.tests.checks.len()).sum()
        } else {
            self.tests.checks.len()
        }
    }

    /// List the per-check names, prefixed by step for multi-step tasks.
    pub fn check_names(&self) -> Vec<String> {
        if self.is_multi_step() {
            self.steps
                .iter()
                .flat_map(|s| {
                    s.tests
                        .checks
                        .iter()
                        .map(move |c| format!("{}/{}", s.name, c.name()))
                })
                .collect()
        } else {
            self.tests
                .checks
                .iter()
                .map(|c| c.name().to_string())
                .collect()
        }
    }

    /// Structural lint: the common authoring pitfalls of Appendix E.1,
    /// encoded as machine checks. Returns one issue string per pitfall hit.
    ///
    /// 1. reward file forgotten (suite with no checks produces no reward);
    /// 2. relative paths in `test.sh`;
    /// 3. the solution pre-installed in the Dockerfile (environment files
    ///    overlapping the solution's outputs);
    /// 4. the test script leaking into `instruction.md`;
    /// 5. `solution/solve.sh` not executable (Oracle agent fails);
    /// 6. empty `keywords` (invisible to registry search);
    /// 7. `README.md` left as a stub.
    pub fn lint(&self) -> Vec<String> {
        let mut issues = Vec::new();

        // 1. Reward file forgotten.
        let suites: Vec<&TestSuite> = if self.is_multi_step() {
            self.steps.iter().map(|s| &s.tests).collect()
        } else {
            vec![&self.tests]
        };
        if suites.iter().any(|s| s.checks.is_empty()) {
            issues.push(
                "verifier has no checks: the reward file will never be written (silent 0)".into(),
            );
        }

        // 2. Relative paths in test.sh.
        for suite in &suites {
            for line in suite.test_sh.lines() {
                let t = line.trim();
                if t.starts_with("pytest ") && !t.contains("/tests/") {
                    issues.push(format!(
                        "test.sh uses a relative path: {t:?} (Harbor runs it from a different cwd)"
                    ));
                }
            }
        }

        // 3. Solution installed into the Dockerfile: environment pre-plants a
        //    file the solution is supposed to produce.
        for op in &self.solution.ops {
            if let crate::world::ShellOp::WriteFile { path, .. } = op {
                if self.environment.files.contains_key(path) {
                    issues.push(format!(
                        "environment pre-plants {path}, which the reference solution also writes (agent already gets the answer)"
                    ));
                }
            }
        }

        // 4. Test script leaks into the instruction (whole-word match, so
        //    "summary" does not false-positive on "summary.json").
        for suite in &suites {
            for check in &suite.checks {
                if contains_word(&self.instruction.body, check.name()) {
                    issues.push(format!(
                        "instruction.md leaks the test name {:?} (gaming becomes trivial)",
                        check.name()
                    ));
                }
            }
        }

        // 5. solve.sh executable.
        if !self.solution.executable {
            issues.push(
                "solution/solve.sh is not executable (chmod +x) — the Oracle agent will fail"
                    .into(),
            );
        }

        // 6. Keywords populated.
        if self.meta.keywords.is_empty() {
            issues.push("keywords = [] leaves the task invisible to registry search".into());
        } else if self.meta.keywords.len() < 3 || self.meta.keywords.len() > 8 {
            issues.push(format!(
                "keywords should be 3-8 lowercase tokens, got {}",
                self.meta.keywords.len()
            ));
        }

        // 7. README populated.
        if is_readme_stub(&self.readme) {
            issues.push("README.md is a stub — no one can understand the task at a glance".into());
        }

        // Multi-step conventions.
        for step in &self.steps {
            issues.extend(step.check_setup_conventions());
        }

        issues
    }

    /// Validate the network-policy layering of this task against a provider's
    /// capability set (Appendix E.1): a phase override that differs from its
    /// baseline requires `dynamic_network_policy` (E2B supports it; plain
    /// Docker does not).
    pub fn check_network_policy(&self, supports_dynamic: bool) -> Vec<String> {
        let mut warnings = Vec::new();
        let baseline = self.effective_verifier_baseline();
        let agent_differs = self
            .toml
            .agent
            .network
            .differs_from(&self.effective_agent_baseline());
        if agent_differs && !supports_dynamic {
            warnings.push(format!(
                "agent phase override ({}) differs from the [environment] baseline; provider without dynamic_network_policy rejects this task — use a separate verifier env or match the baseline",
                self.toml.agent.network.network_mode.as_ref().map(|m| m.toml_name()).unwrap_or("unset")
            ));
        }
        if self.toml.verifier.network.differs_from(&baseline) && !supports_dynamic {
            warnings.push("verifier phase override differs from its baseline; requires dynamic_network_policy".into());
        }
        warnings
    }

    /// Effective agent-phase baseline: `[environment]` (the shared baseline).
    pub fn effective_agent_baseline(&self) -> NetworkMode {
        self.environment.network_mode.clone()
    }

    /// Effective verifier baseline: `[verifier.environment]` for a separate
    /// verifier, else a copy of `[environment]` (Appendix E.1).
    pub fn effective_verifier_baseline(&self) -> NetworkMode {
        match &self.toml.verifier.environment_mode {
            VerifierMode::Shared => self.environment.network_mode.clone(),
            VerifierMode::Separate { network_mode, .. } => network_mode.clone(),
        }
    }

    /// Render `task.toml`.
    pub fn render_task_toml(&self) -> String {
        let mut s = String::new();
        if self.is_multi_step() {
            s.push_str(&format!(
                "schema_version = \"{}\"\n\n",
                self.toml.schema_version
            ));
            s.push_str(&format!(
                "multi_step_reward_strategy = \"{}\"\n\n",
                self.toml.reward_strategy.as_str()
            ));
        }
        s.push_str(&format!(
            "[task]\nname = \"{}\"\nversion = \"{}\"\n",
            self.meta.name, self.meta.version
        ));
        s.push_str(&format!("description = \"{}\"\n", self.meta.description));
        s.push_str(&format!(
            "keywords = {}\n\n",
            toml_string_list(&self.meta.keywords)
        ));
        s.push_str("[metadata]\n");
        s.push_str(&format!(
            "difficulty = \"{}\"\n",
            self.toml.metadata.difficulty.as_str()
        ));
        s.push_str(&format!("category = \"{}\"\n", self.toml.metadata.category));
        s.push_str(&format!(
            "tags = {}\n\n",
            toml_string_list(&self.toml.metadata.tags)
        ));
        s.push_str("[agent]\n");
        s.push_str(&format!("timeout_sec = {}\n", self.toml.agent.timeout_sec));
        if let Some(user) = &self.toml.agent.user {
            s.push_str(&format!("user = \"{user}\"\n"));
        }
        if let Some(mode) = &self.toml.agent.network.network_mode {
            s.push_str(&format!("network_mode = \"{}\"\n", mode.toml_name()));
        }
        s.push('\n');
        s.push_str("[verifier]\n");
        s.push_str(&format!(
            "timeout_sec = {}\n",
            self.toml.verifier.timeout_sec
        ));
        for (k, v) in &self.toml.verifier.env {
            s.push_str(&format!("[verifier.env]\n{k} = \"{v}\"\n"));
        }
        s.push('\n');
        s.push_str("[environment]\n");
        s.push_str(&format!(
            "network_mode = \"{}\"\n",
            self.environment.network_mode.toml_name()
        ));
        s.push_str(&format!("cpus = {}\n", self.environment.cpus));
        s.push_str(&format!("memory_mb = {}\n", self.environment.memory_mb));
        s.push_str(&format!("storage_mb = {}\n", self.environment.storage_mb));
        if let Some(g) = self.environment.gpus {
            s.push_str(&format!("gpus = {g}\n"));
        }
        s.push_str(&format!(
            "\n{}",
            self.toml.verifier.environment_mode.toml_fragment()
        ));
        if self.is_multi_step() {
            s.push('\n');
            for step in &self.steps {
                s.push_str(&format!("[[steps]]\nname = \"{}\"\n", step.name));
                match &step.min_reward {
                    MinReward::Scalar(t) => s.push_str(&format!("min_reward = {t}\n")),
                    MinReward::Keys(gates) => {
                        let inner: Vec<String> =
                            gates.iter().map(|(k, v)| format!("{k} = {v}")).collect();
                        s.push_str(&format!("min_reward = {{ {} }}\n", inner.join(", ")));
                    }
                }
                if let Some(t) = step.agent_timeout_sec {
                    s.push_str(&format!("[steps.agent]\ntimeout_sec = {t}\n"));
                }
                if let Some(t) = step.verifier_timeout_sec {
                    s.push_str(&format!("[steps.verifier]\ntimeout_sec = {t}\n"));
                }
                s.push('\n');
            }
        }
        s
    }
}

/// The instruction (`instruction.md`): the only part of the task the solver
/// sees.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instruction {
    /// The markdown body of the instruction.
    pub body: String,
}

impl Instruction {
    /// Build an instruction from a body.
    pub fn new(body: impl Into<String>) -> Self {
        Instruction { body: body.into() }
    }
}

/// Whole-token containment: `needle` appears in `haystack` delimited by
/// non-word characters (alphanumeric and `_` count as word characters).
fn contains_word(haystack: &str, needle: &str) -> bool {
    haystack
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .any(|w| w == needle)
}

/// Whether a README is an unpopulated stub left behind by
/// `harbor task init`.
fn is_readme_stub(readme: &str) -> bool {
    let trimmed = readme.trim();
    trimmed.is_empty() || trimmed.len() < 80 || trimmed.starts_with("# TODO")
}

/// Render a TOML string list.
fn toml_string_list(items: &[String]) -> String {
    let quoted: Vec<String> = items.iter().map(|k| format!("\"{k}\"")).collect();
    format!("[{}]", quoted.join(", "))
}

/// Flags accepted by `harbor task init` (Appendix E.1 Step 1).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScaffoldFlags {
    /// `--description "..."`.
    pub description: Option<String>,
    /// `--author "Jane Doe <jane@example.com>"` (repeatable).
    pub authors: Vec<String>,
    /// `--no-pytest` — skip the pytest test template (use for Reward Kit or
    /// custom verifiers).
    pub no_pytest: bool,
    /// `--no-solution` — skip the `solution/` directory.
    pub no_solution: bool,
}

/// The scaffold produced by `harbor task init "<org>/<task-name>"`:
/// instruction.md, task.toml, environment/Dockerfile, solution/solve.sh,
/// tests/test.sh (Appendix E.1).
#[derive(Debug, Clone, PartialEq)]
pub struct Scaffold {
    /// Files of the scaffold, keyed by relative path.
    pub files: BTreeMap<String, String>,
}

/// Scaffold a new task, the way `harbor task init` does. The proposer is
/// required to start here and edit the scaffold rather than hand-rolling the
/// directory (auto_env_scaling skill: "Do NOT hand-roll the task directory —
/// run `harbor task init` and edit the scaffold").
pub fn scaffold_task(org: &str, name: &str, flags: &ScaffoldFlags) -> Scaffold {
    let mut files = BTreeMap::new();
    files.insert(
        "instruction.md".into(),
        "# TODO: write the instruction\n".into(),
    );
    let mut toml = format!("[task]\nname = \"{org}/{name}\"\nversion = \"1.0.0\"\n");
    if let Some(d) = &flags.description {
        toml.push_str(&format!("description = \"{d}\"\n"));
    }
    for a in &flags.authors {
        toml.push_str(&format!("author = \"{a}\"\n"));
    }
    files.insert("task.toml".into(), toml);
    files.insert(
        "environment/Dockerfile".into(),
        "FROM ubuntu:24.04\nWORKDIR /app\n# Install what the task requires — NOT the solution\n"
            .into(),
    );
    if !flags.no_solution {
        files.insert(
            "solution/solve.sh".into(),
            "#!/bin/bash\n# Reference solution\n".into(),
        );
    }
    if !flags.no_pytest {
        files.insert(
            "tests/test.sh".into(),
            "#!/bin/bash\nuvx --with pytest==8.4.1 pytest /tests/test_outputs.py\nif [ $? -eq 0 ]; then\n  echo 1 > /logs/verifier/reward.txt\nelse\n  echo 0 > /logs/verifier/reward.txt\nfi\n".into(),
        );
        files.insert(
            "tests/test_outputs.py".into(),
            "from pathlib import Path\n\ndef test_placeholder():\n    assert True\n".into(),
        );
    }
    files.insert("README.md".into(), "# TODO: describe the task\n".into());
    Scaffold { files }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{RewardFile, ShellOp, TestCheck, TestSuite};

    fn suite(n: usize) -> TestSuite {
        TestSuite {
            test_sh: "#!/bin/bash\nuvx --with pytest==8.4.1 pytest /tests/test_outputs.py\n"
                .to_string(),
            checks: (0..n)
                .map(|i| TestCheck::FileExists {
                    name: format!("check_{i}"),
                    path: format!("/app/out_{i}.txt"),
                })
                .collect(),
            reward_file: RewardFile::Txt,
        }
    }

    fn task() -> HarborTask {
        HarborTask {
            env_id: "env_001".into(),
            meta: TaskMeta {
                name: "org/task-a".into(),
                version: "1.0.0".into(),
                description: "demo".into(),
                keywords: vec!["python".into(), "csv".into(), "pytest".into()],
            },
            instruction: Instruction::new("# Do the thing\nProduce /app/out_0.txt with the answer.\n"),
            environment: TaskEnvironment::default(),
            solution: SolutionScript {
                ops: vec![ShellOp::WriteFile { path: "/app/out_0.txt".into(), content: "42".into() }],
                executable: true,
            },
            tests: suite(3),
            readme: "# Task A\n\nA reconciliation task over two exports. See instruction.md.\n\n## Running\n\nharbor run -p . -a oracle\n".into(),
            steps: Vec::new(),
            toml: TaskToml {
                schema_version: "1.0".into(),
                task: TaskMeta {
                    name: "org/task-a".into(),
                    version: "1.0.0".into(),
                    description: "demo".into(),
                    keywords: vec!["python".into(), "csv".into(), "pytest".into()],
                },
                metadata: TaskMetadata {
                    difficulty: Difficulty::Medium,
                    category: "programming".into(),
                    tags: vec![],
                },
                agent: AgentConfig::default(),
                verifier: VerifierConfig::default(),
                reward_strategy: RewardStrategy::Mean,
            },
            skills: vec![SkillTag::new("python")],
        }
    }

    #[test]
    fn lint_clean_task_has_no_issues() {
        assert_eq!(task().lint(), Vec::<String>::new());
    }

    #[test]
    fn lint_catches_all_seven_pitfalls() {
        let mut t = task();
        t.tests = suite(0); // 1: no checks -> no reward file
        t.tests.test_sh = "pytest test_outputs.py".into(); // 2: relative path
        t.environment
            .files
            .insert("/app/out_0.txt".into(), "42".into()); // 3: solution pre-installed
        t.instruction.body.push_str("\nThen pass check_0.\n"); // 4: leak — but suite(0) has no checks; add one
        t.tests.checks = vec![TestCheck::FileExists {
            name: "check_0".into(),
            path: "/app/out_0.txt".into(),
        }];
        t.solution.executable = false; // 5: not executable
        t.meta.keywords = Vec::new(); // 6: invisible
        t.readme = "# TODO: describe the task\n".into(); // 7: stub

        let issues = t.lint();
        assert!(issues.len() >= 6, "expected >= 6 issues, got {issues:?}");
    }

    #[test]
    fn allowed_host_validation() {
        assert!(is_valid_allowed_host("pypi.org"));
        assert!(is_valid_allowed_host("*.pythonhosted.org"));
        assert!(is_valid_allowed_host("10.0.0.0/8"));
        assert!(is_valid_allowed_host("2001:db8::/32"));
        assert!(!is_valid_allowed_host("https://pypi.org"));
        assert!(!is_valid_allowed_host("pypi.org:443"));
        assert!(!is_valid_allowed_host("pypi.org/path"));
    }

    #[test]
    fn runtime_host_flags_ignored_on_public() {
        let env = TaskEnvironment::default(); // public
        let (mode, warnings) = env.baseline_with_runtime_hosts(&["pypi.org".into()]);
        assert_eq!(mode, NetworkMode::Public);
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn runtime_host_flags_merge_into_allowlist() {
        let env = TaskEnvironment {
            network_mode: NetworkMode::Allowlist {
                allowed_hosts: vec!["pypi.org".into()],
            },
            ..Default::default()
        };
        let (mode, warnings) = env.baseline_with_runtime_hosts(&["files.pythonhosted.org".into()]);
        assert!(warnings.is_empty());
        match mode {
            NetworkMode::Allowlist { allowed_hosts } => {
                assert!(allowed_hosts.contains(&"pypi.org".to_string()));
                assert!(allowed_hosts.contains(&"files.pythonhosted.org".to_string()));
            }
            other => panic!("expected allowlist, got {other:?}"),
        }
    }

    #[test]
    fn phase_override_noop_when_matching() {
        let baseline = NetworkMode::NoNetwork;
        let over = PhaseOverride {
            network_mode: Some(NetworkMode::NoNetwork),
            extra_allowed_hosts: vec![],
        };
        assert!(!over.differs_from(&baseline));
        assert_eq!(over.resolve(&baseline), baseline);

        let over2 = PhaseOverride {
            network_mode: Some(NetworkMode::Public),
            extra_allowed_hosts: vec![],
        };
        assert!(over2.differs_from(&baseline));
        assert_eq!(over2.resolve(&baseline), NetworkMode::Public);
    }

    #[test]
    fn cidr_matching() {
        assert!(host_matches("10.0.0.0/8", "10.1.2.3"));
        assert!(!host_matches("10.0.0.0/8", "192.168.1.1"));
        assert!(host_matches("*.pythonhosted.org", "a.b.pythonhosted.org"));
        assert!(host_matches("pypi.org", "pypi.org"));
        assert!(!host_matches("pypi.org", "evil-pypi.org"));
    }

    #[test]
    fn min_reward_gate_forms() {
        let scalar = MinReward::Scalar(1.0);
        assert!(scalar.passes(1.0, &BTreeMap::new()));
        assert!(!scalar.passes(0.9, &BTreeMap::new()));

        let mut gates = BTreeMap::new();
        gates.insert("correctness".to_string(), 0.8);
        gates.insert("style".to_string(), 0.5);
        let dict = MinReward::Keys(gates);
        let mut ok = BTreeMap::new();
        ok.insert("correctness".to_string(), 0.9);
        ok.insert("style".to_string(), 0.6);
        assert!(dict.passes(0.75, &ok));
        let mut bad = ok.clone();
        bad.insert("style".to_string(), 0.4);
        assert!(!dict.passes(0.75, &bad));
    }

    #[test]
    fn scaffold_layout() {
        let sc = scaffold_task(
            "acme",
            "widget-sort",
            &ScaffoldFlags {
                description: Some("sort widgets".into()),
                ..Default::default()
            },
        );
        assert_eq!(sc.files.len(), 7);
        assert!(sc.files.contains_key("instruction.md"));
        assert!(sc.files.contains_key("environment/Dockerfile"));
        assert!(sc.files.contains_key("solution/solve.sh"));
        assert!(sc.files.contains_key("tests/test.sh"));
        assert!(sc.files["task.toml"].contains("acme/widget-sort"));
    }

    #[test]
    fn scaffold_flags() {
        let sc = scaffold_task(
            "acme",
            "rk-task",
            &ScaffoldFlags {
                no_pytest: true,
                no_solution: true,
                ..Default::default()
            },
        );
        assert!(!sc.files.contains_key("tests/test.sh"));
        assert!(!sc.files.contains_key("solution/solve.sh"));
        assert!(!sc.files.contains_key("tests/test_outputs.py"));
    }

    #[test]
    fn verifier_baseline_rules() {
        let mut t = task();
        // Shared verifier: baseline = [environment].
        assert_eq!(t.effective_verifier_baseline(), NetworkMode::Public);
        t.toml.verifier.environment_mode = VerifierMode::Separate {
            network_mode: NetworkMode::NoNetwork,
            docker_image: Some("ubuntu:24.04".into()),
        };
        assert_eq!(t.effective_verifier_baseline(), NetworkMode::NoNetwork);
    }

    #[test]
    fn dynamic_network_policy_requirement() {
        let mut t = task();
        t.toml.agent.network.network_mode = Some(NetworkMode::Public);
        // Baseline is public; a public override is a no-op — no warning.
        assert!(t.check_network_policy(false).is_empty());
        t.environment.network_mode = NetworkMode::NoNetwork;
        // Override differs from baseline: Docker (no dynamic policy) rejects.
        assert_eq!(t.check_network_policy(false).len(), 1);
        assert!(t.check_network_policy(true).is_empty());
    }

    #[test]
    fn task_toml_renders_multi_step() {
        let mut t = task();
        t.steps = vec![StepSpec {
            name: "scaffold".into(),
            instruction: "scaffold it".into(),
            workdir: BTreeMap::new(),
            setup_sh: Some(SolutionScript {
                ops: vec![ShellOp::RmSelf],
                executable: true,
            }),
            tests: suite(2),
            solution: None,
            min_reward: MinReward::Scalar(1.0),
            agent_timeout_sec: Some(60.0),
            verifier_timeout_sec: Some(30.0),
            agent_network: PhaseOverride::default(),
            artifacts: vec![],
        }];
        t.toml.schema_version = "1.4".into();
        let rendered = t.render_task_toml();
        assert!(rendered.contains("multi_step_reward_strategy = \"mean\""));
        assert!(rendered.contains("[[steps]]"));
        assert!(rendered.contains("min_reward = 1"));
        assert_eq!(t.total_checks(), 2);
    }
}
