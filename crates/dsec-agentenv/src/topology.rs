//! Pod topology — the two-container environment, simulated in-process.
//!
//! Upstream, one task instance runs as a pod with two containers sharing
//! a network namespace:
//!
//! * **`main`** — the agent's container. Sees `/work/workspace` (RW) only.
//! * **`sidecar`** — serves the environment's MCP tools over
//!   streamable-http (127.0.0.1:39101+i), owns the backing state
//!   (`/work/system/<mcp>/state.db`), and runs the verifier.
//!
//! The main container never mounts `/work/system` — DB isolation is
//! physical, so the agent cannot bypass the MCP tools with direct SQL.
//! The sidecar mounts `/work/workspace` read-only; the graded state is
//! never polluted from the control plane.
//!
//! [`SimPod`] reproduces that topology deterministically in userspace:
//!
//! * per-container **volumes** with mount access modes (RW / RO) and
//!   longest-prefix path resolution — `/work/system` simply does not
//!   exist in `main`'s namespace, exactly like the unmounted path in a
//!   real pod;
//! * a **command router** that maps the harness's known entry scripts to
//!   native behavior (sidecar entrypoint starts the MCP servers;
//!   `run_verify.py` runs the rubric engine) and simulates the small
//!   shell subset the agents actually use (`cat`, `ls`, `test`, `mkdir`,
//!   `rm`, `echo >`, `tail`, the `/proc/net/tcp` port probe);
//! * **listening ports** in the shared network namespace (the liveness
//!   probe reads them, a direct connect would be REJECTed under the
//!   isolation firewall — same observable behavior as upstream);
//! * **session logs** captured under `main:/tmp/mimo-claude-logs` and
//!   injected to the verifier container at reward time;
//! * **file hashes** over the workspace, backing the `src_protect`
//!   source-conservation gate.

use crate::error::{Error, Result};
use crate::manifest::{self, Manifest, MAIN, SIDECAR};
use crate::state::StateDb;
use std::collections::BTreeMap;
use std::path::Path;

/// Result of one simulated exec.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExecResult {
    /// Combined stdout+stderr.
    pub output: String,
    /// Process exit code.
    pub returncode: i32,
    /// Termination reason: `exit`, `timeout`, `denied`.
    pub reason: String,
}

impl ExecResult {
    fn ok(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            returncode: 0,
            reason: "exit".into(),
        }
    }

    fn err(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            returncode: 1,
            reason: "exit".into(),
        }
    }

    /// Whether the exec succeeded.
    pub fn succeeded(&self) -> bool {
        self.returncode == 0
    }
}

/// Mount access mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Read-write.
    Rw,
    /// Read-only.
    Ro,
}

/// One backing volume (a flat path → bytes tree).
#[derive(Debug, Default, Clone)]
pub struct Volume {
    /// `path -> entry`.
    pub files: BTreeMap<String, FileEntry>,
}

/// A file or directory entry.
#[derive(Debug, Clone, PartialEq)]
pub struct FileEntry {
    /// File content (empty for directories).
    pub content: Vec<u8>,
    /// Whether this is a directory node.
    pub is_dir: bool,
}

impl Volume {
    /// Writes a file (creating parent directory nodes).
    pub fn write(&mut self, path: &str, content: Vec<u8>) {
        self.ensure_parent(path);
        self.files.insert(
            path.trim_start_matches('/').to_string(),
            FileEntry {
                content,
                is_dir: false,
            },
        );
    }

    /// Copies a local directory tree in recursively (`copy_to` source).
    pub fn upload_dir(&mut self, local: &Path, sub: &str) -> usize {
        let mut n = 0;
        if let Ok(rd) = std::fs::read_dir(local) {
            for e in rd.flatten() {
                let rel = if sub.is_empty() {
                    e.file_name().to_string_lossy().into_owned()
                } else {
                    format!("{sub}/{}", e.file_name().to_string_lossy())
                };
                let p = e.path();
                if p.is_dir() {
                    self.mkdir(&rel);
                    n += self.upload_dir(&p, &rel);
                } else if let Ok(bytes) = std::fs::read(&p) {
                    self.write(&rel, bytes);
                    n += 1;
                }
            }
        }
        n
    }

    /// Creates a directory node.
    pub fn mkdir(&mut self, path: &str) {
        let clean = path.trim_matches('/');
        if clean.is_empty() {
            return;
        }
        self.files.entry(clean.to_string()).or_insert(FileEntry {
            content: Vec::new(),
            is_dir: true,
        });
        self.ensure_parent(clean);
    }

    fn ensure_parent(&mut self, path: &str) {
        let clean = path.trim_start_matches('/');
        let mut parts: Vec<&str> = clean.split('/').collect();
        parts.pop();
        let mut acc = String::new();
        for p in parts {
            if p.is_empty() {
                continue;
            }
            if !acc.is_empty() {
                acc.push('/');
            }
            acc.push_str(p);
            self.files.entry(acc.clone()).or_insert(FileEntry {
                content: Vec::new(),
                is_dir: true,
            });
        }
    }

    /// Reads a file.
    pub fn read(&self, path: &str) -> Option<&[u8]> {
        let e = self.files.get(path.trim_start_matches('/'))?;
        (!e.is_dir).then_some(&e.content[..])
    }

    /// Removes a file or directory (recursively; -f semantics — missing
    /// is fine).
    pub fn rm(&mut self, path: &str) {
        let clean = path.trim_start_matches('/');
        let keys: Vec<String> = self
            .files
            .keys()
            .filter(|k| *k == clean || k.starts_with(&format!("{clean}/")))
            .cloned()
            .collect();
        for k in keys {
            self.files.remove(&k);
        }
    }

    /// `ls` of one directory level.
    pub fn list(&self, path: &str) -> Vec<String> {
        let clean = path.trim_end_matches('/');
        let prefix = if clean.is_empty() {
            String::new()
        } else {
            format!("{}/", clean)
        };
        let mut names = Vec::new();
        for (k, e) in &self.files {
            if let Some(rest) = k.strip_prefix(&prefix) {
                if !rest.is_empty() && !rest.contains('/') {
                    names.push(if e.is_dir {
                        format!("{}/", rest)
                    } else {
                        rest.to_string()
                    });
                }
            }
        }
        names
    }
}

/// A container: named mounts over shared volumes.
#[derive(Debug, Clone)]
pub struct Container {
    /// Container name (`main` / `sidecar`).
    pub name: String,
    /// `mount point -> (volume id, access)`.
    pub mounts: BTreeMap<String, (String, Access)>,
}

impl Container {
    /// Resolves a container path to `(volume id, in-volume path, access)`.
    pub fn resolve(&self, path: &str) -> Option<(String, String, Access)> {
        let p = normalize(path);
        // longest mount-point prefix wins (mount points normalize the
        // same way as the query path); track the winning PREFIX LENGTH,
        // not the volume id length
        let mut best: Option<(usize, String, String, Access)> = None;
        for (mp, (vid, access)) in &self.mounts {
            let mpc = normalize(mp);
            if p == mpc || p.starts_with(&format!("{mpc}/")) {
                let n = mpc.len();
                if best.as_ref().is_none_or(|(bn, ..)| n > *bn) {
                    let rel = if p == mpc {
                        String::new()
                    } else {
                        p[mpc.len() + 1..].to_string()
                    };
                    best = Some((n, vid.clone(), rel, *access));
                }
            }
        }
        best.map(|(_, vid, rel, access)| (vid, rel, access))
    }
}

fn normalize(p: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for seg in p.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            s => out.push(s),
        }
    }
    out.join("/")
}

/// Volume ids used by the standard pod topology.
pub mod vols {
    /// Agent workspace (RW from main, RO from sidecar).
    pub const WORKSPACE: &str = "workspace";
    /// Simulated system DBs (RW, sidecar only).
    pub const SYSTEM: &str = "system";
    /// Tool definitions (sidecar).
    pub const TOOLS: &str = "tools";
    /// Image-provided harness scripts (sidecar).
    pub const INSTALLED: &str = "installed";
    /// Verifier logs (sidecar).
    pub const LOGS: &str = "logs";
    /// Setup scratch in main.
    pub const SETUP: &str = "setup";
    /// Agent session logs in main.
    pub const SESSIONS: &str = "sessions";
    /// Agent-output staging for the verifier.
    pub const AGENT_OUT: &str = "agent-out";
}

/// The MCP tool surface of one pod — registered by the environment
/// builder; executed against the pod's state DBs.
///
/// The tool is the only sanctioned mutation path into [`StateDb`],
/// mirroring how the real sidecar's `tools/<mcp>.py` is the only writer
/// of `state.db`.
pub struct ToolDef {
    /// Tool name (as exposed over MCP).
    pub name: String,
    /// JSON schema of the params object (OpenAI function-call shape).
    pub params_schema: serde_json::Value,
    /// Human description shown to the model.
    pub description: String,
    /// The executor: mutates state, returns the observation payload.
    pub exec: Box<dyn Fn(&mut ToolCtx) -> serde_json::Value + Send>,
}

impl std::fmt::Debug for ToolDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolDef")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("params_schema", &self.params_schema)
            .finish_non_exhaustive()
    }
}

/// Execution context handed to a tool.
pub struct ToolCtx<'a> {
    /// The pod's state DBs, `system name -> db`.
    pub dbs: &'a mut BTreeMap<String, StateDb>,
    /// The tool's params as parsed by the caller.
    pub params: serde_json::Value,
}

impl<'a> ToolCtx<'a> {
    /// Borrows one system's DB.
    pub fn db(&self, name: &str) -> Option<&StateDb> {
        self.dbs.get(name)
    }

    /// Mutably borrows one system's DB.
    pub fn db_mut(&mut self, name: &str) -> Option<&mut StateDb> {
        self.dbs.get_mut(name)
    }

    /// Typed param field.
    pub fn param_str(&self, key: &str) -> Option<&str> {
        self.params.get(key).and_then(|v| v.as_str())
    }
}

/// The simulated pod.
///
/// Construction goes through [`PodBuilder`] (used by
/// [`crate::envgen`]) which wires volumes, state DBs, tools, and the
/// verifier; the lifecycle methods below then implement the harness
/// contract: `upload -> setup -> (rollout) -> reward`.
pub struct SimPod {
    /// Pod name.
    pub name: String,
    /// Containers by name.
    pub containers: BTreeMap<String, Container>,
    /// Volumes by id.
    pub volumes: BTreeMap<String, Volume>,
    /// State DBs by system name (sidecar-owned).
    pub state: BTreeMap<String, StateDb>,
    /// MCP servers: `name -> {tools}`.
    pub servers: BTreeMap<String, Vec<ToolDef>>,
    /// Ports currently listening (shared netns).
    pub listening: Vec<u16>,
    /// Lifecycle state name.
    pub phase: &'static str,
    /// Workspace file hashes captured at ready-time (src_protect gate).
    protected_hashes: BTreeMap<String, u64>,
}

impl SimPod {
    /// New builder for a pod.
    pub fn builder(name: &str) -> PodBuilder {
        PodBuilder::new(name)
    }

    /// Copies a local file into a container path (`copy_to`).
    pub fn copy_to(&mut self, local: &Path, target: &str, container: &str) -> Result<()> {
        let (vid, rel, access) = self
            .containers
            .get(container)
            .and_then(|c| c.resolve(target))
            .ok_or_else(|| Error::ExecFailed {
                container: container.into(),
                rc: 1,
                output: format!("copy_to: no mount resolves {target:?}"),
            })?;
        if access == Access::Ro {
            return Err(Error::ExecFailed {
                container: container.into(),
                rc: 1,
                output: format!("copy_to: {target} is read-only"),
            });
        }
        let vol = self.volumes.get_mut(&vid).expect("mounted volumes exist");
        if local.is_dir() {
            vol.upload_dir(local, &rel);
        } else {
            let bytes = std::fs::read(local)?;
            vol.write(&rel, bytes);
        }
        Ok(())
    }

    /// Copies a container path out to the local filesystem (`copy_out`).
    pub fn copy_out(&self, container: &str, path: &str, local: &Path) -> Result<()> {
        let (vid, rel, _) = self
            .containers
            .get(container)
            .and_then(|c| c.resolve(path))
            .ok_or_else(|| Error::ExecFailed {
                container: container.into(),
                rc: 1,
                output: format!("copy_out: no mount resolves {path:?}"),
            })?;
        let vol = self.volumes.get(&vid).expect("mounted volumes exist");
        if let Some(bytes) = vol.read(&rel) {
            if let Some(parent) = local.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(local, bytes)?;
            Ok(())
        } else {
            Err(Error::ExecFailed {
                container: container.into(),
                rc: 1,
                output: format!("copy_out: no such file {path:?}"),
            })
        }
    }

    /// Reads a container path as bytes.
    pub fn read_file(&self, container: &str, path: &str) -> Option<Vec<u8>> {
        let (vid, rel, _) = self.containers.get(container)?.resolve(path)?;
        self.volumes.get(&vid)?.read(&rel).map(|b| b.to_vec())
    }

    /// Writes a container path (RW mounts only) — the `write`/`edit` tool
    /// path and `answer.md` persistence.
    pub fn write_file(&mut self, container: &str, path: &str, content: &[u8]) -> Result<()> {
        let (vid, rel, access) = self
            .containers
            .get(container)
            .and_then(|c| c.resolve(path))
            .ok_or_else(|| Error::ExecFailed {
                container: container.into(),
                rc: 1,
                output: format!("write: no mount resolves {path:?}"),
            })?;
        if access == Access::Ro {
            return Err(Error::ExecFailed {
                container: container.into(),
                rc: 1,
                output: format!("write: {path} is read-only"),
            });
        }
        let vol = self.volumes.get_mut(&vid).expect("mounted volumes exist");
        vol.write(&rel, content.to_vec());
        Ok(())
    }

    /// Appends a session event line to the agent session log (main).
    pub fn append_session(&mut self, line: &serde_json::Value) {
        let vol = self
            .volumes
            .get_mut(vols::SESSIONS)
            .expect("sessions volume exists");
        let path = "sessions/session-0.jsonl";
        let mut content = vol.read(path).map(|b| b.to_vec()).unwrap_or_default();
        content.extend_from_slice(serde_json::to_string(line).unwrap().as_bytes());
        content.push(b'\n');
        vol.write(path, content);
    }

    /// Executes a command in a container (simulated shell — see the
    /// module docs for the supported subset).
    pub fn execute(&mut self, command: &str, container: &str, timeout_sec: u64) -> ExecResult {
        // env prefix `K=V ...`
        let mut toks = shell_split(command);
        while toks
            .first()
            .map(|t| t.contains('=') && !t.starts_with('-'))
            .unwrap_or(false)
        {
            // a bare `FOO=bar` token: skip (env assignment)
            if toks.first().unwrap().split('=').count() == 2 {
                toks.remove(0);
            } else {
                break;
            }
        }
        // redirect handling: `cmd args > file`
        let mut redirect: Option<String> = None;
        if let Some(pos) = toks.iter().position(|t| t == ">") {
            if toks.len() > pos + 1 {
                redirect = Some(toks[pos + 1].clone());
                toks.truncate(pos);
            }
        }
        let Some(argv0) = toks.first().cloned() else {
            return ExecResult::ok("");
        };
        let argv0 = argv0.rsplit('/').next().unwrap_or(&argv0).to_string();
        let args: Vec<String> = toks[1..].to_vec();
        let res = self.run_command(&argv0, &args, container, timeout_sec, command);
        if let Some(target) = redirect {
            if res.returncode == 0 {
                if self
                    .write_file(container, &target, res.output.as_bytes())
                    .is_err()
                {
                    return ExecResult::err(format!("write {target}: read-only or unresolvable"));
                }
                return ExecResult::ok("");
            }
        }
        res
    }

    fn run_command(
        &mut self,
        argv0: &str,
        args: &[String],
        container: &str,
        timeout_sec: u64,
        raw: &str,
    ) -> ExecResult {
        if !self.containers.contains_key(container) {
            return ExecResult::err(format!("no such container: {container}"));
        }
        // python3 <script> routing — the harness's native entry points.
        if argv0 == "python3" || argv0 == "python" {
            if let Some(script) = args.first() {
                let base = script.rsplit('/').next().unwrap_or(script);
                return self.route_script(base, args, container, timeout_sec);
            }
            if args.first().map(|a| a == "-c").unwrap_or(false) {
                return self.port_probe(container);
            }
        }
        match argv0 {
            "cat" => {
                let mut out = String::new();
                for a in args {
                    match self.read_file(container, a) {
                        Some(b) => {
                            out.push_str(&String::from_utf8_lossy(&b));
                            out.push('\n');
                        }
                        None => {
                            return ExecResult::err(format!("cat: {a}: No such file or directory"))
                        }
                    }
                }
                ExecResult::ok(out)
            }
            "ls" => {
                let dir = args.iter().position(|a| !a.starts_with('-'));
                let target = dir.map(|i| args[i].as_str()).unwrap_or(".");
                match self.list_dir(container, target) {
                    Some(names) => ExecResult::ok(names.join("\n")),
                    None => ExecResult::err(format!("ls: cannot access '{target}'")),
                }
            }
            "test" => {
                let mut it = args.iter();
                while let Some(a) = it.next() {
                    match a.as_str() {
                        "-d" | "-e" => {
                            let p = it.next().cloned().unwrap_or_default();
                            if self.exists(container, &p, a == "-d") {
                                return ExecResult::ok("");
                            }
                            return ExecResult::err("");
                        }
                        "-s" => {
                            let p = it.next().cloned().unwrap_or_default();
                            let nonempty = self
                                .read_file(container, &p)
                                .map(|b| !b.is_empty())
                                .unwrap_or(false);
                            return if nonempty {
                                ExecResult::ok("")
                            } else {
                                ExecResult::err("")
                            };
                        }
                        _ => {}
                    }
                }
                ExecResult::err("test: unsupported expression")
            }
            "mkdir" => {
                let target = args
                    .iter()
                    .find(|a| !a.starts_with('-'))
                    .cloned()
                    .unwrap_or_default();
                match self.mkdir(container, &target) {
                    true => ExecResult::ok(""),
                    false => ExecResult::err(format!("mkdir: cannot create '{target}'")),
                }
            }
            "rm" => {
                for a in args.iter().filter(|a| !a.starts_with('-')) {
                    self.rm_path(container, a);
                }
                ExecResult::ok("")
            }
            "tail" => {
                // tail -c N <file>
                let mut n = 4000usize;
                let mut file = None;
                let mut i = 0;
                while i < args.len() {
                    match args[i].as_str() {
                        "-c" => {
                            n = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(n);
                            i += 2;
                        }
                        a if !a.starts_with('-') => {
                            file = Some(a.to_string());
                            i += 1;
                        }
                        _ => i += 1,
                    }
                }
                match file.and_then(|f| self.read_file(container, &f)) {
                    Some(b) => {
                        let s = String::from_utf8_lossy(&b);
                        let start = s.len().saturating_sub(n);
                        ExecResult::ok(&s[start..])
                    }
                    None => ExecResult::err("tail: no such file"),
                }
            }
            "true" => ExecResult::ok(""),
            "false" => ExecResult::err(""),
            "echo" => ExecResult::ok(args.join(" ")),
            "visudo" => ExecResult::ok(""),
            "id" => ExecResult::ok("uid=0(root) gid=0(root) groups=0(root)".to_string()),
            "sha256sum" | "md5sum" => {
                let mut out = Vec::new();
                for a in args {
                    match self.read_file(container, a) {
                        Some(b) => out.push(format!("{:016x}  {}", fnv1a(&b), a)),
                        None => return ExecResult::err(format!("{}: {a}: No such file", argv0)),
                    }
                }
                ExecResult::ok(out.join("\n"))
            }
            _ => ExecResult::err(format!("exec: {argv0}: command not simulated ({raw})")),
        }
    }

    /// Routes a `python3 <script>` invocation to the native handler.
    fn route_script(
        &mut self,
        base: &str,
        _args: &[String],
        container: &str,
        _timeout_sec: u64,
    ) -> ExecResult {
        match base {
            "sidecar_entrypoint.py" => {
                if container != SIDECAR {
                    return ExecResult::err("sidecar_entrypoint must run in the sidecar");
                }
                // --start-and-detach: bring every MCP server's port up.
                let ports: Vec<u16> = self
                    .servers
                    .keys()
                    .enumerate()
                    .map(|(i, _)| manifest::MCP_PORT_BASE + i as u16)
                    .collect();
                for p in &ports {
                    if !self.listening.contains(p) {
                        self.listening.push(*p);
                    }
                }
                self.phase = "ready";
                ExecResult::ok(format!(
                    "[sidecar] {} mcp server(s) started: ports {:?}",
                    self.servers.len(),
                    ports
                ))
            }
            "run_verify.py" => {
                if container != SIDECAR {
                    return ExecResult::err("run_verify must run in the sidecar");
                }
                // The native verifier runs through the registered harness
                // hook; the pod only mediates the contract here.
                let has = self
                    .volumes
                    .get(vols::INSTALLED)
                    .and_then(|v| v.read("run_verify.py"))
                    .is_some()
                    || self
                        .volumes
                        .get(vols::LOGS)
                        .and_then(|v| v.read("verify-ran"))
                        .is_some();
                if !has {
                    // fall back: verifier materials were late-uploaded to
                    // /work — model that path too
                    let uploaded = self.read_file(SIDECAR, "/work/run_verify.py").is_some();
                    if !uploaded {
                        return ExecResult::err("run_verify.py: No such file or directory");
                    }
                }
                // executed by the harness (verifier.rs); the pod just
                // records the invocation + timeout budget
                self.phase = "verifying";
                ExecResult::ok("[verify] runner invoked")
            }
            _ => ExecResult::err(format!("python3: can't open file '{base}'")),
        }
    }

    /// The `/proc/net/tcp` liveness probe (a root connect would be
    /// REJECTed under the isolation firewall; reading /proc works).
    fn port_probe(&mut self, container: &str) -> ExecResult {
        if self.containers.contains_key(container) {
            // probes succeed iff at least one expected port is listening
            let ok = self.listening.iter().max().is_some();
            if ok {
                ExecResult::ok("")
            } else {
                ExecResult::err("no listening ports")
            }
        } else {
            ExecResult::err("no such container")
        }
    }

    /// Whether a port is listening (shared netns — any container sees it).
    pub fn port_live(&self, port: u16) -> bool {
        self.listening.contains(&port)
    }

    /// Probes a specific port number via the exec surface (main path uses
    /// the generic python probe; this is the harness-side variant).
    pub fn probe_port(&self, port: u16) -> ExecResult {
        if self.port_live(port) {
            ExecResult::ok("")
        } else {
            ExecResult::err("port not listening")
        }
    }

    fn exists(&self, container: &str, path: &str, want_dir: bool) -> bool {
        let Some(c) = self.containers.get(container) else {
            return false;
        };
        let Some((vid, rel, _)) = c.resolve(path) else {
            return false;
        };
        let Some(vol) = self.volumes.get(&vid) else {
            return false;
        };
        match vol.files.get(&rel) {
            Some(e) => e.is_dir == want_dir,
            None => {
                // implicit dir: any child under rel/
                if want_dir {
                    vol.files.keys().any(|k| k.starts_with(&format!("{rel}/")))
                } else {
                    false
                }
            }
        }
    }

    fn list_dir(&self, container: &str, path: &str) -> Option<Vec<String>> {
        let c = self.containers.get(container)?;
        let (vid, rel, _) = c.resolve(path)?;
        let vol = self.volumes.get(&vid)?;
        if !rel.is_empty() {
            match vol.files.get(&rel) {
                Some(e) if !e.is_dir => return None,
                None if !vol.files.keys().any(|k| k.starts_with(&format!("{rel}/"))) => {
                    return None
                }
                _ => {}
            }
        }
        Some(vol.list(&rel))
    }

    fn mkdir(&mut self, container: &str, path: &str) -> bool {
        let Some(c) = self.containers.get(container) else {
            return false;
        };
        let Some((vid, rel, access)) = c.resolve(path) else {
            return false;
        };
        if access == Access::Ro {
            return false;
        }
        let Some(vol) = self.volumes.get_mut(&vid) else {
            return false;
        };
        vol.mkdir(&rel);
        true
    }

    fn rm_path(&mut self, container: &str, path: &str) {
        if let Some(c) = self.containers.get(container) {
            if let Some((vid, rel, access)) = c.resolve(path) {
                if access == Access::Ro {
                    return;
                }
                if let Some(vol) = self.volumes.get_mut(&vid) {
                    vol.rm(&rel);
                }
            }
        }
    }

    /// Calls an MCP tool (`server.tool`-namespaced) — the only sanctioned
    /// state mutation path.
    pub fn call_tool(
        &mut self,
        server: &str,
        tool: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let servers = self
            .servers
            .get_mut(server)
            .ok_or(Error::UnknownServer(server.to_string()))?;
        let idx = servers
            .iter()
            .position(|t| t.name == tool)
            .ok_or_else(|| Error::ToolFailed {
                tool: format!("{server}.{tool}"),
                message: "unknown tool".into(),
            })?;
        // Take the tool out, run it with mutable state, put it back — the
        // exec closure owns no pod borrows.
        let tool_def = servers.remove(idx);
        let mut ctx = ToolCtx {
            dbs: &mut self.state,
            params,
        };
        let out = (tool_def.exec)(&mut ctx);
        self.servers
            .get_mut(server)
            .expect("server row still present")
            .insert(idx, tool_def);
        Ok(out)
    }

    /// Lists tools across servers (agent-side discovery view).
    pub fn tool_catalog(&self) -> Vec<(String, String, serde_json::Value, String)> {
        let mut out = Vec::new();
        for (server, tools) in &self.servers {
            for t in tools {
                out.push((
                    server.clone(),
                    t.name.clone(),
                    t.params_schema.clone(),
                    t.description.clone(),
                ));
            }
        }
        out
    }

    /// Captures workspace file hashes (the `src_protect` baseline).
    pub fn capture_protection(&mut self) {
        let Some(vol) = self.volumes.get(vols::WORKSPACE) else {
            return;
        };
        self.protected_hashes = vol
            .files
            .iter()
            .filter(|(_, e)| !e.is_dir)
            .map(|(k, e)| (k.clone(), fnv1a(&e.content)))
            .collect();
    }

    /// Whether every protected workspace file is byte-identical to the
    /// baseline (source conservation). New files are allowed; edits and
    /// deletions of protected files are not.
    pub fn source_conserved(&self) -> bool {
        let Some(vol) = self.volumes.get(vols::WORKSPACE) else {
            return true;
        };
        for (k, h) in &self.protected_hashes {
            match vol.files.get(k) {
                Some(e) if !e.is_dir => {
                    if fnv1a(&e.content) != *h {
                        return false;
                    }
                }
                _ => return false, // deleted or turned into a dir
            }
        }
        true
    }
}

/// FNV-1a 64 (a stand-in for sha256 in the simulated shell — stable and
/// dependency-free; collisions are irrelevant for the gate's semantics).
pub fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Splits a command line on whitespace (single-token shell-lite; quoted
/// strings are respected).
pub fn shell_split(cmd: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for ch in cmd.chars() {
        match quote {
            Some(q) => {
                if ch == q {
                    quote = None;
                } else {
                    cur.push(ch);
                }
            }
            None => match ch {
                '\'' | '"' => quote = Some(ch),
                c if c.is_whitespace() => {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                }
                c => cur.push(c),
            },
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Builder for [`SimPod`] — assembles the standard topology plus the
/// environment's tools and state.
pub struct PodBuilder {
    name: String,
    state: BTreeMap<String, StateDb>,
    servers: BTreeMap<String, Vec<ToolDef>>,
    workspace_files: Vec<(String, Vec<u8>)>,
    verifier_files: Vec<(String, Vec<u8>)>,
}

impl PodBuilder {
    /// New builder with the standard two-container topology.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            state: BTreeMap::new(),
            servers: BTreeMap::new(),
            workspace_files: Vec::new(),
            verifier_files: Vec::new(),
        }
    }

    /// Registers a system DB.
    pub fn system(mut self, name: &str, db: StateDb) -> Self {
        self.state.insert(name.to_string(), db);
        self
    }

    /// Registers an MCP server with its tool surface.
    pub fn server(mut self, name: &str, tools: Vec<ToolDef>) -> Self {
        self.servers.insert(name.to_string(), tools);
        self
    }

    /// Adds a workspace file (uploaded to `main:/work/workspace`).
    pub fn workspace_file(mut self, path: &str, content: Vec<u8>) -> Self {
        self.workspace_files
            .push((path.trim_start_matches('/').to_string(), content));
        self
    }

    /// Adds a reward-time verifier file (late-uploaded to the sidecar).
    pub fn verifier_file(mut self, path: &str, content: Vec<u8>) -> Self {
        self.verifier_files
            .push((path.trim_start_matches('/').to_string(), content));
        self
    }

    /// Assembles the pod.
    pub fn build(self) -> SimPod {
        let mut volumes: BTreeMap<String, Volume> = BTreeMap::new();
        // workspace
        let mut ws = Volume::default();
        ws.mkdir("");
        for (p, c) in &self.workspace_files {
            ws.write(p, c.clone());
        }
        volumes.insert(vols::WORKSPACE.to_string(), ws);
        // system: system/<name>/state.db sentinel entries (the real DBs
        // live in self.state; volumes hold a marker so ls/cat see them)
        let mut sys = Volume::default();
        for name in self.state.keys() {
            sys.write(&format!("{name}/state.db"), b"sqlite".to_vec());
        }
        volumes.insert(vols::SYSTEM.to_string(), sys);
        volumes.insert(vols::TOOLS.to_string(), Volume::default());
        volumes.insert(vols::INSTALLED.to_string(), Volume::default());
        let mut logs = Volume::default();
        logs.mkdir("verifier");
        volumes.insert(vols::LOGS.to_string(), logs);
        volumes.insert(vols::SETUP.to_string(), Volume::default());
        volumes.insert(vols::SESSIONS.to_string(), Volume::default());
        volumes.insert(vols::AGENT_OUT.to_string(), Volume::default());
        // verifier bundle staging (late upload source)
        let mut vstage = Volume::default();
        for (p, c) in &self.verifier_files {
            vstage.write(p, c.clone());
        }
        volumes.insert("vstage".to_string(), vstage);

        let main = Container {
            name: MAIN.to_string(),
            mounts: {
                let mut m = BTreeMap::new();
                m.insert(
                    "/work/workspace".into(),
                    (vols::WORKSPACE.into(), Access::Rw),
                );
                m.insert("/work/_setup".into(), (vols::SETUP.into(), Access::Rw));
                m.insert("/tmp".into(), ("main-tmp".into(), Access::Rw));
                m.insert(
                    "/tmp/mimo-claude-logs".into(),
                    (vols::SESSIONS.into(), Access::Rw),
                );
                m
            },
        };
        // main-tmp volume
        volumes.insert("main-tmp".to_string(), Volume::default());
        let sidecar = Container {
            name: SIDECAR.to_string(),
            mounts: {
                let mut m = BTreeMap::new();
                m.insert("/work/system".into(), (vols::SYSTEM.into(), Access::Rw));
                m.insert("/work/tools".into(), (vols::TOOLS.into(), Access::Ro));
                m.insert(
                    "/installed-agent".into(),
                    (vols::INSTALLED.into(), Access::Rw),
                );
                m.insert("/logs".into(), (vols::LOGS.into(), Access::Rw));
                m.insert(
                    "/work/workspace".into(),
                    (vols::WORKSPACE.into(), Access::Ro),
                );
                m.insert(
                    "/tmp/agent_output".into(),
                    (vols::AGENT_OUT.into(), Access::Rw),
                );
                m.insert("/work".into(), ("sidecar-work".into(), Access::Rw));
                m.insert("/vstage".into(), ("vstage".into(), Access::Ro));
                m
            },
        };
        // sidecar /work shadow volume — /work/system & /work/workspace
        // mount-points (longer prefixes) take precedence over /work.
        volumes.insert("sidecar-work".to_string(), Volume::default());

        let mut pod = SimPod {
            name: self.name,
            containers: BTreeMap::from([(MAIN.to_string(), main), (SIDECAR.to_string(), sidecar)]),
            volumes,
            state: self.state,
            servers: self.servers,
            listening: Vec::new(),
            phase: "created",
            protected_hashes: BTreeMap::new(),
        };
        // /work dir node for main (ls /work shows workspace only)
        pod.capture_protection();
        pod
    }
}

impl SimPod {
    /// Applies one manifest upload batch (pre-rollout).
    pub fn apply_uploads(&mut self, task_dir: &Path, manifest: &Manifest) -> Result<()> {
        for u in &manifest.uploads {
            let src = manifest::resolve_source(task_dir, &u.source);
            let src = if src.exists() {
                src
            } else {
                // allow volume-staged sources (verifier files uploaded
                // from the builder's staging volume)
                let staged = self.read_file(SIDECAR, &format!("/vstage/{}", u.source));
                if let Some(bytes) = staged {
                    self.write_file(&u.container, &u.target, &bytes)?;
                    continue;
                }
                return Err(Error::MissingSource {
                    src: u.source.clone(),
                    resolved: src.display().to_string(),
                });
            };
            if src.is_dir() {
                // bulk upload: iterate children
                for e in std::fs::read_dir(&src)?.flatten() {
                    let target = format!(
                        "{}/{}",
                        u.target.trim_end_matches('/'),
                        e.file_name().to_string_lossy()
                    );
                    self.copy_to(&e.path(), &target, &u.container)?;
                }
            } else {
                self.copy_to(&src, &u.target, &u.container)?;
            }
        }
        Ok(())
    }

    /// Applies the manifest setup step and waits on the declared ports.
    pub fn apply_setup(&mut self, manifest: &Manifest) -> Result<ExecResult> {
        let Some(setup) = &manifest.setup else {
            self.phase = "ready";
            return Ok(ExecResult::ok(""));
        };
        let container = setup.container.clone();
        let cmd = setup.command.clone();
        let timeout = setup.timeout_sec.unwrap_or(manifest::MCP_STARTUP_TIMEOUT);
        let res = self.execute(&cmd, &container, timeout);
        if !res.succeeded() {
            self.phase = "setup-failed";
            return Err(Error::ExecFailed {
                container,
                rc: res.returncode,
                output: res.output,
            });
        }
        self.phase = "ready";
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_topology_isolation() {
        let pod = SimPod::builder("p1").build();
        // /work/system does not resolve in main
        assert!(pod.containers[MAIN]
            .resolve("/work/system/customers.db")
            .is_none());
        // ...and does in the sidecar (RW)
        let (vid, rel, access) = pod.containers[SIDECAR]
            .resolve("/work/system/crm/state.db")
            .unwrap();
        assert_eq!(vid, vols::SYSTEM);
        assert_eq!(rel, "crm/state.db");
        assert_eq!(access, Access::Rw);
        // workspace is RW from main, RO from sidecar
        let (_, _, a_main) = pod.containers[MAIN]
            .resolve("/work/workspace/x.md")
            .unwrap();
        let (_, _, a_side) = pod.containers[SIDECAR]
            .resolve("/work/workspace/x.md")
            .unwrap();
        assert_eq!(a_main, Access::Rw);
        assert_eq!(a_side, Access::Ro);
    }

    #[test]
    fn write_denied_from_readonly_sidecar_workspace() {
        let mut pod = SimPod::builder("p1")
            .workspace_file("brief.md", b"hi".to_vec())
            .build();
        // sidecar cannot write the workspace (read-only mount)
        assert!(pod
            .write_file(SIDECAR, "/work/workspace/answer.md", b"x")
            .is_err());
        // main can
        pod.write_file(MAIN, "/work/workspace/answer.md", b"the answer")
            .unwrap();
        assert_eq!(
            pod.read_file(MAIN, "/work/workspace/answer.md").unwrap(),
            b"the answer"
        );
        // and the sidecar can READ it (graded state)
        assert_eq!(
            pod.read_file(SIDECAR, "/work/workspace/answer.md").unwrap(),
            b"the answer"
        );
    }

    #[test]
    fn main_cannot_cat_sidecar_db() {
        let mut pod = SimPod::builder("p1")
            .system("crm", crate::state::StateDb::new(Default::default()))
            .build();
        pod.volumes
            .get_mut(vols::SYSTEM)
            .unwrap()
            .write("crm/state.db", b"secret".to_vec());
        let res = pod.execute("cat /work/system/crm/state.db", MAIN, 30);
        assert!(!res.succeeded());
        let res = pod.execute("cat /work/system/crm/state.db", SIDECAR, 30);
        assert!(res.succeeded());
        assert!(res.output.contains("secret"));
    }

    #[test]
    fn shell_subset_behaves_like_the_real_thing() {
        let mut pod = SimPod::builder("p1")
            .workspace_file("a.txt", b"hello world\n".to_vec())
            .workspace_file("dir/b.txt", b"inner".to_vec())
            .build();
        assert_eq!(
            pod.execute("cat /work/workspace/a.txt", MAIN, 5).output,
            "hello world\n\n"
        );
        assert!(pod
            .execute("ls /work/workspace", MAIN, 5)
            .output
            .contains("a.txt"));
        assert!(pod
            .execute("test -d /work/workspace/dir", MAIN, 5)
            .succeeded());
        assert!(!pod
            .execute("test -d /work/workspace/a.txt", MAIN, 5)
            .succeeded());
        assert!(pod
            .execute("test -s /work/workspace/a.txt", MAIN, 5)
            .succeeded());
        assert!(pod
            .execute("mkdir -p /work/workspace/new/dir", MAIN, 5)
            .succeeded());
        assert!(pod
            .execute("test -d /work/workspace/new/dir", MAIN, 5)
            .succeeded());
        // echo redirection
        pod.execute("echo done > /work/workspace/new/dir/out.txt", MAIN, 5);
        assert_eq!(
            pod.read_file(MAIN, "/work/workspace/new/dir/out.txt")
                .unwrap(),
            b"done"
        );
        // rm
        pod.execute("rm -f /work/workspace/a.txt", MAIN, 5);
        assert!(pod.read_file(MAIN, "/work/workspace/a.txt").is_none());
        // tail -c
        let r = pod.execute("tail -c 3 /work/workspace/dir/b.txt", MAIN, 5);
        assert_eq!(r.output, "ner");
    }

    #[test]
    fn env_prefix_and_redirect_together() {
        let mut pod = SimPod::builder("p").build();
        let r = pod.execute(
            "VERIFY_DETERMINISTIC=1 VERIFY_AGENT_JUDGE=1 echo payload > /tmp/out.txt",
            MAIN,
            5,
        );
        assert!(r.succeeded());
        assert_eq!(pod.read_file(MAIN, "/tmp/out.txt").unwrap(), b"payload");
    }

    #[test]
    fn setup_starts_mcp_ports_in_order() {
        let pod = SimPod::builder("p")
            .server("alpha", vec![])
            .server("beta", vec![])
            .build();
        let mut manifest = Manifest::default();
        manifest.setup = Some(crate::manifest::SetupSpec {
            command: "python3 /installed-agent/sidecar_entrypoint.py --start-and-detach".into(),
            container: SIDECAR.into(),
            timeout_sec: Some(300),
        });
        let mut pod = pod;
        pod.apply_setup(&manifest).unwrap();
        assert_eq!(pod.listening, vec![39101, 39102]);
        assert!(pod.port_live(39102));
        assert_eq!(pod.phase, "ready");
        // probe from main (shared netns)
        assert!(pod.probe_port(39101).succeeded());
        assert!(!pod.probe_port(39999).succeeded());
    }

    #[test]
    fn setup_rejected_in_main() {
        let mut pod = SimPod::builder("p").server("alpha", vec![]).build();
        let res = pod.execute(
            "python3 /installed-agent/sidecar_entrypoint.py --start-and-detach",
            MAIN,
            5,
        );
        assert!(!res.succeeded());
        assert_eq!(pod.listening.len(), 0);
    }

    #[test]
    fn source_conservation_gate() {
        let mut pod = SimPod::builder("p")
            .workspace_file("protected.md", b"v1".to_vec())
            .build();
        pod.capture_protection();
        assert!(pod.source_conserved());
        // a new file is fine
        pod.write_file(MAIN, "/work/workspace/answer.md", b"ans")
            .unwrap();
        assert!(pod.source_conserved());
        // an edit is not
        pod.write_file(MAIN, "/work/workspace/protected.md", b"tampered")
            .unwrap();
        assert!(!pod.source_conserved());
    }

    #[test]
    fn sessions_append_and_volume_read() {
        let mut pod = SimPod::builder("p").build();
        pod.append_session(&serde_json::json!({"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}));
        pod.append_session(&serde_json::json!({"type": "assistant", "message": {"content": [{"type": "text", "text": "final"}]}}));
        let raw = String::from_utf8(
            pod.read_file(MAIN, "/tmp/mimo-claude-logs/sessions/session-0.jsonl")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(raw.lines().count(), 2);
        assert!(raw.contains("final"));
    }

    #[test]
    fn shell_split_quotes() {
        let t = shell_split("cat 'a b.txt' \"c d.py\" e");
        assert_eq!(t, vec!["cat", "a b.txt", "c d.py", "e"]);
    }
}
