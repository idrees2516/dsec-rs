//! Environment manifests — the `manifest.json` contract.
//!
//! A manifest describes everything the harness must materialize to run one
//! task instance, in the author-facing harbor-parity shape:
//!
//! ```json
//! {
//!   "cwd": "/work/workspace",
//!   "uploads": [
//!     {"source": "workspace", "target": "/work/workspace", "container": "main"}
//!   ],
//!   "setup": {
//!     "command": "python3 /installed-agent/sidecar_entrypoint.py --start-and-detach",
//!     "container": "sidecar", "timeout_sec": 300
//!   },
//!   "wait_ports": [39101],
//!   "mcp_servers": [
//!     {"name": "crm", "transport": "streamable-http", "url": "http://127.0.0.1:39101/mcp"}
//!   ],
//!   "verifier": {
//!     "uploads": [{"source": "verify.py", "target": "/work/verify.py"}],
//!     "command": "python3 /work/run_verify.py",
//!     "container": "sidecar",
//!     "reward_file": "/logs/verifier/reward.json",
//!     "reward_detail_file": "/logs/verifier/reward_detail.json",
//!     "timeout_sec": null,
//!     "env_passthrough": null
//!   }
//! }
//! ```
//!
//! Invariants enforced on load (they are hard errors upstream too):
//!
//! * `setup` and `verifier` must never target the `main` (agent) container
//!   — the agent must not be able to touch the control surface.
//! * every upload `source` must resolve to an existing path.
//! * `container` defaults to `main` for plain uploads, `sidecar` for
//!   setup/verifier.
//!
//! When a task directory has no manifest the harness synthesizes the
//! default one (see [`Manifest::synthesize_default`]) — the exact
//! pre-manifest behavior: workspace to `main`, `system/`+`tools/` to the
//! sidecar, MCP servers on ports `39101+i` in tool enumeration order, and
//! the late-uploaded verifier bundle.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// First MCP port; the i-th tool (in enumeration order) serves `39101+i`.
/// Must stay in lockstep with the sidecar entrypoint's `MCP_PORT_BASE`.
pub const MCP_PORT_BASE: u16 = 39101;

/// Default verifier timeout, seconds.
pub const DEFAULT_VERIFIER_TIMEOUT: u64 = 900;

/// MCP startup timeout, seconds.
pub const MCP_STARTUP_TIMEOUT: u64 = 300;

/// The sidecar container name.
pub const SIDECAR: &str = "sidecar";

/// The agent container name.
pub const MAIN: &str = "main";

/// Verifier materials uploaded to the sidecar **at reward time only** —
/// ground truth is invisible during the rollout.
pub const VERIFIER_FILES: &[&str] = &[
    "run_verify.py",
    "verify.py",
    "rubrics.json",
    "answer_key.json",
    "verifier_meta.json",
    "_helpers.py",
];

/// Well-known harness paths inside the pod.
pub mod paths {
    /// Read-write agent workspace in `main`.
    pub const WORKSPACE: &str = "/work/workspace";
    /// Pod work root.
    pub const WORK: &str = "/work";
    /// Image-provided harness scripts in the sidecar.
    pub const INSTALLED_AGENT: &str = "/installed-agent";
    /// Verifier log directory.
    pub const VERIFIER_LOGS: &str = "/logs/verifier";
    /// Agent session log directory (injected to the verifier container).
    pub const AGENT_OUTPUT: &str = "/tmp/agent_output";
    /// Setup scratch directory in `main`.
    pub const SETUP: &str = "/work/_setup";
}

/// One upload directive: copy `source` (task-dir-relative) to `target`
/// (absolute, in-container) in `container`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Upload {
    /// Source path (relative to the task dir; absolute allowed).
    pub source: String,
    /// Absolute destination path inside the container.
    pub target: String,
    /// Destination container (`main` or `sidecar`); defaults to `main`.
    #[serde(default = "default_container")]
    pub container: String,
}

fn default_container() -> String {
    MAIN.to_string()
}

/// The post-upload setup command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetupSpec {
    /// Command string (run under `/bin/sh -c` semantics).
    pub command: String,
    /// Container to run in — never `main`.
    pub container: String,
    /// Wall-clock budget, seconds.
    #[serde(default, skip_serializing_if = "is_none_u64")]
    pub timeout_sec: Option<u64>,
}

fn is_none_u64(v: &Option<u64>) -> bool {
    v.is_none()
}

/// An MCP server declaration.
///
/// Transports: `streamable-http` (the default, URL-based), `http`, `sse`,
/// and `stdio` (command-based). URL entries translate to the SDK-native
/// `{type, url, headers}` shape; stdio entries to `{type, command, args,
/// env}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpServerSpec {
    /// Server name — also the tools-directory file stem.
    pub name: String,
    /// Transport; defaults to `streamable-http`.
    #[serde(default = "default_transport")]
    pub transport: String,
    /// HTTP-family URL (e.g. `http://127.0.0.1:39101/mcp`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Extra HTTP headers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<serde_json::Map<String, serde_json::Value>>,
    /// stdio command.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// stdio args.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<String>>,
    /// stdio env.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<serde_json::Map<String, serde_json::Value>>,
}

fn default_transport() -> String {
    "streamable-http".to_string()
}

impl McpServerSpec {
    /// SDK-native entry: `{type: http|sse, url, headers}` or
    /// `{type: stdio, command, args, env}`.
    pub fn sdk_entry(&self) -> serde_json::Value {
        if self.transport == "stdio" {
            let mut v = serde_json::json!({
                "type": "stdio",
                "command": self.command.clone().unwrap_or_default(),
            });
            if let Some(args) = &self.args {
                v["args"] = serde_json::json!(args);
            }
            if let Some(env) = &self.env {
                v["env"] = serde_json::json!(env);
            }
            v
        } else {
            let ty = if self.transport == "sse" {
                "sse"
            } else {
                "http"
            };
            let mut v = serde_json::json!({
                "type": ty,
                "url": self.url.clone().unwrap_or_default(),
            });
            if let Some(h) = &self.headers {
                v["headers"] = serde_json::json!(h);
            }
            v
        }
    }
}

/// The verifier bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifierSpec {
    /// Reward-time uploads (ground truth — sidecar only).
    #[serde(default)]
    pub uploads: Vec<Upload>,
    /// Verifier entry command.
    #[serde(default = "default_verifier_command")]
    pub command: String,
    /// Container — never `main`.
    #[serde(default = "default_verifier_container")]
    pub container: String,
    /// Where the verifier writes `{"reward": 0.42}`.
    pub reward_file: String,
    /// Where the verifier writes the per-rubric breakdown.
    pub reward_detail_file: String,
    /// Verifier wall-clock budget, seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_sec: Option<u64>,
    /// Env vars forwarded from the controller into the verifier process.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_passthrough: Option<Vec<String>>,
}

fn default_verifier_command() -> String {
    format!("python3 {}/run_verify.py", paths::WORK)
}

fn default_verifier_container() -> String {
    SIDECAR.to_string()
}

impl Default for VerifierSpec {
    fn default() -> Self {
        Self {
            uploads: Vec::new(),
            command: default_verifier_command(),
            container: default_verifier_container(),
            reward_file: format!("{}/reward.json", paths::VERIFIER_LOGS),
            reward_detail_file: format!("{}/reward_detail.json", paths::VERIFIER_LOGS),
            timeout_sec: None,
            env_passthrough: None,
        }
    }
}

/// The full environment manifest.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Agent working directory in `main`.
    #[serde(default = "default_cwd")]
    pub cwd: String,
    /// Pre-rollout uploads.
    #[serde(default)]
    pub uploads: Vec<Upload>,
    /// Post-upload setup step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub setup: Option<SetupSpec>,
    /// Ports the harness waits on before declaring the env ready.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub wait_ports: Vec<u16>,
    /// MCP server declarations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp_servers: Vec<McpServerSpec>,
    /// Verifier bundle.
    #[serde(default)]
    pub verifier: VerifierSpec,
}

fn default_cwd() -> String {
    paths::WORKSPACE.to_string()
}

impl Manifest {
    /// Parses a manifest, normalizing defaults and validating invariants.
    pub fn parse(text: &str, task_dir: &Path) -> Result<Self> {
        let m: Manifest = serde_json::from_str(text).map_err(Error::Json)?;
        m.validate(task_dir)
    }

    /// Loads `manifest.json` from a task dir, or synthesizes the default
    /// one when the file is absent.
    pub fn load_or_default(task_dir: &Path) -> Result<Self> {
        let mf = task_dir.join("manifest.json");
        if mf.is_file() {
            Self::parse(&std::fs::read_to_string(mf)?, task_dir)
        } else {
            Self::synthesize_default(task_dir).validate(task_dir)
        }
    }

    /// Applies the upstream normalization: defaults, container invariants,
    /// source existence.
    pub fn validate(self, task_dir: &Path) -> Result<Self> {
        let mut m = self;
        if m.cwd.is_empty() {
            m.cwd = paths::WORKSPACE.to_string();
        }
        for u in &mut m.uploads {
            if u.container.is_empty() {
                u.container = MAIN.to_string();
            }
        }
        if let Some(s) = &mut m.setup {
            if s.container.is_empty() {
                s.container = SIDECAR.to_string();
            }
            if s.container == MAIN {
                return Err(Error::InvalidSpec(format!(
                    "setup must not run in 'main' (agent) container: {s:?}"
                )));
            }
        }
        for wp in &m.wait_ports {
            if *wp == 0 {
                return Err(Error::InvalidSpec("wait_ports contains port 0".into()));
            }
        }
        let v = &mut m.verifier;
        if v.container.is_empty() {
            v.container = SIDECAR.to_string();
        }
        if v.container == MAIN {
            return Err(Error::InvalidSpec(format!(
                "verifier must not run in 'main' (agent) container: {v:?}"
            )));
        }
        if v.command.is_empty() {
            v.command = format!("python3 {}/run_verify.py", paths::WORK);
        }
        if v.reward_file.is_empty() {
            v.reward_file = format!("{}/reward.json", paths::VERIFIER_LOGS);
        }
        if v.reward_detail_file.is_empty() {
            v.reward_detail_file = format!("{}/reward_detail.json", paths::VERIFIER_LOGS);
        }
        for u in m.uploads.iter().chain(v.uploads.iter()) {
            if !resolve_source(task_dir, &u.source).exists() {
                return Err(Error::MissingSource {
                    src: u.source.clone(),
                    resolved: resolve_source(task_dir, &u.source).display().to_string(),
                });
            }
        }
        Ok(m)
    }

    /// Synthesizes the default manifest for a task dir laid out like the
    /// released environments (workspace/ + system/ + tools/ + verifier
    /// files). Reproduces the pre-manifest upstream behavior exactly:
    ///
    /// * `workspace/` → `main:/work/workspace`
    /// * `system/` and `tools/` → `sidecar:/work/{system,tools}`
    /// * payload scripts → `sidecar:/installed-agent/`
    /// * MCP ports `39101+i` and server URLs by tool enumeration order
    /// * verifier bundle late-uploaded to the sidecar
    pub fn synthesize_default(task_dir: &Path) -> Self {
        let mut uploads: Vec<Upload> = Vec::new();
        let have_ws = task_dir.join("workspace").is_dir();
        let have_sys = task_dir.join("system").is_dir();
        let tools_dir = task_dir.join("tools");
        let have_tools = tools_dir.is_dir();
        let mcp_names = enumerate_mcp_tools(&tools_dir);

        if have_ws {
            uploads.push(Upload {
                source: "workspace".into(),
                target: paths::WORKSPACE.into(),
                container: MAIN.into(),
            });
        }
        if have_sys {
            uploads.push(Upload {
                source: "system".into(),
                target: format!("{}/system", paths::WORK),
                container: SIDECAR.into(),
            });
        }
        if have_tools {
            uploads.push(Upload {
                source: "tools".into(),
                target: format!("{}/tools", paths::WORK),
                container: SIDECAR.into(),
            });
        }

        let mut setup = None;
        let mut wait_ports = Vec::new();
        let mut mcp_servers = Vec::new();
        if !mcp_names.is_empty() {
            for s in ["sidecar_entrypoint.py", "mcp_http.py"] {
                if task_dir.join(s).is_file() {
                    uploads.push(Upload {
                        source: s.into(),
                        target: format!("{}/{s}", paths::INSTALLED_AGENT),
                        container: SIDECAR.into(),
                    });
                }
            }
            if task_dir.join("mcp_bridge.py").is_file() {
                uploads.push(Upload {
                    source: "mcp_bridge.py".into(),
                    target: format!("{}/mcp_bridge.py", paths::SETUP),
                    container: MAIN.into(),
                });
            }
            for (i, n) in mcp_names.iter().enumerate() {
                wait_ports.push(MCP_PORT_BASE + i as u16);
                mcp_servers.push(McpServerSpec {
                    name: n.clone(),
                    transport: "streamable-http".into(),
                    url: Some(format!("http://127.0.0.1:{}/mcp", MCP_PORT_BASE + i as u16)),
                    headers: None,
                    command: None,
                    args: None,
                    env: None,
                });
            }
            setup = Some(SetupSpec {
                command: format!(
                    "python3 {}/sidecar_entrypoint.py --start-and-detach",
                    paths::INSTALLED_AGENT
                ),
                container: SIDECAR.into(),
                timeout_sec: Some(MCP_STARTUP_TIMEOUT),
            });
        }

        let verifier = VerifierSpec {
            uploads: VERIFIER_FILES
                .iter()
                .filter(|f| task_dir.join(f).is_file())
                .map(|f| Upload {
                    source: (*f).to_string(),
                    target: format!("{}/{f}", paths::WORK),
                    container: SIDECAR.to_string(),
                })
                .collect(),
            ..VerifierSpec::default()
        };

        Manifest {
            cwd: paths::WORKSPACE.into(),
            uploads,
            setup,
            wait_ports,
            mcp_servers,
            verifier,
        }
    }

    /// Server map (`name -> sdk entry`) — what gets stashed on the env for
    /// tool discovery.
    pub fn server_map(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut m = serde_json::Map::new();
        for s in &self.mcp_servers {
            m.insert(s.name.clone(), s.sdk_entry());
        }
        m
    }
}

/// Resolves a manifest source (task-dir-relative or absolute).
pub fn resolve_source(task_dir: &Path, source: &str) -> PathBuf {
    let p = Path::new(source);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        task_dir.join(p)
    }
}

/// Tool enumeration: sorted `.py` file stems in `tools/`, excluding
/// `_`-prefixed files and `tools_test.py`.
///
/// The port assignment `39101+i` uses this exact order; the sort key and
/// exclusions must match the sidecar entrypoint verbatim or tools silently
/// map to the wrong ports.
pub fn enumerate_mcp_tools(tools_dir: &Path) -> Vec<String> {
    let mut stems: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(tools_dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_file()
                && p.extension().map(|e| e == "py").unwrap_or(false)
                && !p
                    .file_stem()
                    .is_some_and(|s| s.to_string_lossy().starts_with('_'))
                && p.file_name().map(|n| n != "tools_test.py").unwrap_or(false)
            {
                if let Some(stem) = p.file_stem() {
                    stems.push(stem.to_string_lossy().into_owned());
                }
            }
        }
    }
    stems.sort();
    stems
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn mk_task_dir_named(root: &Path, name: &str) -> PathBuf {
        let dir = root.join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("workspace")).unwrap();
        fs::create_dir_all(dir.join("system/crm")).unwrap();
        fs::create_dir_all(dir.join("tools")).unwrap();
        fs::write(dir.join("tools/alpha.py"), "").unwrap();
        fs::write(dir.join("tools/beta.py"), "").unwrap();
        fs::write(dir.join("tools/_priv.py"), "").unwrap();
        fs::write(dir.join("tools/tools_test.py"), "").unwrap();
        fs::write(dir.join("workspace/brief.md"), "brief").unwrap();
        fs::write(dir.join("system/crm/state.db"), "db").unwrap();
        dir
    }

    #[test]
    fn default_manifest_matches_upstream_topology() {
        let tmp = tempdir();
        let dir = mk_task_dir_named(tmp.path(), "t1");
        let m = Manifest::load_or_default(&dir).unwrap();
        // workspace → main, system/tools → sidecar
        assert!(m
            .uploads
            .iter()
            .any(|u| u.source == "workspace" && u.container == "main"));
        assert!(m
            .uploads
            .iter()
            .any(|u| u.source == "system" && u.container == "sidecar"));
        assert!(m
            .uploads
            .iter()
            .any(|u| u.source == "tools" && u.container == "sidecar"));
        // MCP ports in enumeration order (alpha, beta), 39101 + 39102
        assert_eq!(m.wait_ports, vec![39101, 39102]);
        assert_eq!(m.mcp_servers.len(), 2);
        assert_eq!(m.mcp_servers[0].name, "alpha");
        assert_eq!(
            m.mcp_servers[0].url.as_deref(),
            Some("http://127.0.0.1:39101/mcp")
        );
        // setup runs in the sidecar
        let setup = m.setup.expect("setup present");
        assert_eq!(setup.container, "sidecar");
        assert!(setup.command.contains("sidecar_entrypoint.py"));
        // verifier defaults
        assert_eq!(m.verifier.container, "sidecar");
        assert_eq!(m.verifier.reward_file, "/logs/verifier/reward.json");
    }

    #[test]
    fn tool_enumeration_excludes_private_and_test() {
        let tmp = tempdir();
        let dir = mk_task_dir_named(tmp.path(), "t2");
        assert_eq!(
            enumerate_mcp_tools(&dir.join("tools")),
            vec!["alpha", "beta"]
        );
    }

    #[test]
    fn rejects_verifier_or_setup_in_main() {
        let tmp = tempdir();
        let dir = mk_task_dir_named(tmp.path(), "t3");
        let base = Manifest::synthesize_default(&dir);
        let bad_setup = Manifest {
            setup: Some(SetupSpec {
                command: "x".into(),
                container: "main".into(),
                timeout_sec: None,
            }),
            ..base.clone()
        };
        assert!(bad_setup.validate(&dir).is_err());
        let bad_verifier = Manifest {
            verifier: VerifierSpec {
                container: "main".into(),
                ..base.verifier.clone()
            },
            ..base
        };
        assert!(bad_verifier.validate(&dir).is_err());
    }

    #[test]
    fn rejects_missing_upload_source() {
        let tmp = tempdir();
        let dir = mk_task_dir_named(tmp.path(), "t4");
        let base = Manifest::synthesize_default(&dir);
        let bad = Manifest {
            uploads: vec![Upload {
                source: "nope_dir".into(),
                target: "/x".into(),
                container: "main".into(),
            }],
            ..base
        };
        match bad.validate(&dir) {
            Err(Error::MissingSource { src, .. }) => assert_eq!(src, "nope_dir"),
            other => panic!("expected MissingSource, got {other:?}"),
        }
    }

    #[test]
    fn parses_the_released_manifest_shape() {
        let text = r#"{
          "cwd": "/work/workspace",
          "uploads": [
            {"source": "workspace", "target": "/work/workspace", "container": "main"},
            {"source": "system", "target": "/work/system", "container": "sidecar"},
            {"source": "tools", "target": "/work/tools", "container": "sidecar"},
            {"source": "sidecar_entrypoint.py", "target": "/installed-agent/sidecar_entrypoint.py", "container": "sidecar"},
            {"source": "mcp_bridge.py", "target": "/work/_setup/mcp_bridge.py", "container": "main"}
          ],
          "setup": {
            "command": "python3 /installed-agent/sidecar_entrypoint.py --start-and-detach",
            "container": "sidecar", "timeout_sec": 300
          },
          "wait_ports": [39101, 39102],
          "mcp_servers": [
            {"name": "dealcloud_disposition_workspace", "url": "http://127.0.0.1:39101/mcp"},
            {"name": "kpmg_tax_modeling_hub", "url": "http://127.0.0.1:39102/mcp"}
          ],
          "verifier": {
            "uploads": [{"source": "verify.py", "target": "/work/verify.py"}],
            "command": "python3 /work/run_verify.py",
            "reward_file": "/logs/verifier/reward.json",
            "reward_detail_file": "/logs/verifier/reward_detail.json",
            "timeout_sec": null,
            "env_passthrough": null
          }
        }"#;
        let tmp = tempdir();
        let dir = mk_task_dir_named(tmp.path(), "t5");
        fs::write(dir.join("verify.py"), "").unwrap();
        fs::write(dir.join("sidecar_entrypoint.py"), "").unwrap();
        fs::write(dir.join("mcp_bridge.py"), "").unwrap();
        let m = Manifest::parse(text, &dir).unwrap();
        assert_eq!(m.wait_ports, vec![39101, 39102]);
        assert_eq!(m.mcp_servers[1].name, "kpmg_tax_modeling_hub");
        assert_eq!(m.mcp_servers[0].transport, "streamable-http");
        let map = m.server_map();
        let entry = map.get("dealcloud_disposition_workspace").unwrap();
        assert_eq!(entry["type"], "http");
        assert_eq!(entry["url"], "http://127.0.0.1:39101/mcp");
        // serde roundtrip stability
        let rt: Manifest = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(rt, m);
    }

    #[test]
    fn stdio_transport_sdk_entry() {
        let s = McpServerSpec {
            name: "local".into(),
            transport: "stdio".into(),
            url: None,
            headers: None,
            command: Some("uvx".into()),
            args: Some(vec!["mcp-server-fs".into()]),
            env: None,
        };
        let e = s.sdk_entry();
        assert_eq!(e["type"], "stdio");
        assert_eq!(e["command"], "uvx");
        assert_eq!(e["args"][0], "mcp-server-fs");
    }

    /// Minimal tempdir helper (std-only, to avoid a dev-dependency).
    struct TempDir(PathBuf);
    impl TempDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn tempdir() -> TempDir {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let base = std::env::temp_dir();
        let uniq = format!(
            "dsec-agentenv-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        );
        let p = base.join(uniq);
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}
