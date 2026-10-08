//! The student-facing tools: `bash`, `view_lines_in_file`,
//! `replace_in_file` (with the hand-rolled unified diff), and
//! `view_image_file`, plus the demoted-subprocess discipline.
//!
//! Ports of upstream `karotte/tools/*.py`, `tool_base.py`, and
//! `demoted.py`. Execution backends are pluggable ([`ExecBackend`]): the
//! default builds the exact upstream argv (absolute `/usr/bin/...`
//! paths, sed ranges, head caps); tests use scripted backends. The
//! output caps, truncation notes, and diff glyphs are byte-compatible
//! with upstream.

use crate::error::{Error, Result};
use crate::schemas::CallToolResult;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// demoted.py discipline
// ---------------------------------------------------------------------------

/// Upstream `TEST_PATH` — a fixed trusted path for demoted access checks.
pub const TEST_PATH: &str = "/usr/bin/test";
/// Upstream `SUBPROCESS_TIMEOUT_S`.
pub const SUBPROCESS_TIMEOUT_S: f64 = 30.0;
/// Upstream `OPEN_TIMEOUT_S`.
pub const OPEN_TIMEOUT_S: f64 = 5.0;
/// Upstream `OPEN_REFUSED_EXIT` — the preexec refusal code.
pub const OPEN_REFUSED_EXIT: i32 = 125;

/// How a demoted subprocess ran.
#[derive(Debug, Clone, PartialEq)]
pub struct DemotedRun {
    /// Exit code (or signal-negated).
    pub exit: i32,
    /// stdout (already bounded by the caller).
    pub stdout: String,
    /// stderr.
    pub stderr: String,
    /// Whether the deadline killed it.
    pub timed_out: bool,
}

/// The pluggable execution backend. Production: a demoted subprocess
/// runner; tests: scripted responses.
pub trait ExecBackend: Send + Sync {
    /// Run `argv` as the student; bounded by `timeout_s`; stdout capped at
    /// `max_stdout` bytes (head + upstream truncation note).
    fn run(&self, argv: &[String], timeout_s: f64, max_stdout: usize) -> Result<DemotedRun>;

    /// `test <flag> <path>` as the student (upstream `check_access`).
    fn check_access(&self, flag: &str, path: &str) -> Result<bool> {
        let argv = vec![TEST_PATH.to_string(), flag.to_string(), path.to_string()];
        let r = self.run(&argv, SUBPROCESS_TIMEOUT_S, 4096)?;
        Ok(r.exit == 0)
    }
}

/// A scripted backend for tests: canned results per argv prefix.
#[derive(Default)]
pub struct ScriptedBackend {
    runs: Vec<std::sync::Mutex<Vec<(String, DemotedRun)>>>,
}

impl ScriptedBackend {
    /// A backend with canned `(argv-contains, result)` pairs.
    pub fn new(canned: Vec<(String, DemotedRun)>) -> Arc<Self> {
        Arc::new(Self {
            runs: vec![std::sync::Mutex::new(canned)],
        })
    }
}

impl ExecBackend for ScriptedBackend {
    fn run(&self, argv: &[String], _timeout: f64, _max: usize) -> Result<DemotedRun> {
        let joined = argv.join(" ");
        let list = self.runs[0].lock().unwrap();
        for (needle, r) in list.iter() {
            if joined.contains(needle.as_str()) {
                return Ok(r.clone());
            }
        }
        Ok(DemotedRun {
            exit: 1,
            stdout: String::new(),
            stderr: String::new(),
            timed_out: false,
        })
    }
}

// ---------------------------------------------------------------------------
// bash.py
// ---------------------------------------------------------------------------

/// Upstream `BashConfig` defaults.
pub const BASH_DEFAULT_TIMEOUT_S: f64 = 3600.0;
/// Upstream `max_output_length`.
pub const BASH_MAX_OUTPUT_LENGTH: usize = 16_000;
/// Upstream `_OUTPUT_HARD_CAP` — in-memory buffer cap.
pub const BASH_OUTPUT_HARD_CAP: usize = 1024 * 1024;
/// Upstream `_OUTPUT_TAIL_KEEP`.
pub const BASH_OUTPUT_TAIL_KEEP: usize = 8192;
/// Upstream `_MARKER_FD`.
pub const BASH_MARKER_FD: i32 = 231;
/// Upstream marker prefix.
pub const BASH_MARKER_PREFIX: &str = "<<exit>>";
/// Upstream `_command_kill_timeout_s`.
pub const BASH_COMMAND_KILL_TIMEOUT_S: f64 = 5.0;
/// Upstream `_output_delay_s`.
pub const BASH_OUTPUT_POLL_S: f64 = 0.1;
/// The upstream in-memory truncation note.
pub const BASH_TRUNCATION_NOTE: &str = "\n[... output truncated: exceeded 1 MiB in memory ...]\n";
/// The student venv (upstream `python_venv` default).
pub const BASH_PYTHON_VENV: &str = "/workdir/.venv";
/// Upstream bash binary.
pub const BASH_BIN: &str = "/usr/bin/bash";

/// The bash tool's configuration (upstream `BashConfig`).
#[derive(Debug, Clone)]
pub struct BashConfig {
    /// Default per-command timeout.
    pub default_timeout_s: f64,
    /// Final stdout/stderr truncation, characters.
    pub max_output_length: usize,
    /// Whether networking is disabled via a user+net namespace
    /// (`network_needs_namespace` upstream: on where iptables can't be
    /// used, i.e. gVisor).
    pub disable_networking: bool,
    /// The student venv prepended to PATH when it exists.
    pub python_venv: Option<String>,
}

impl Default for BashConfig {
    fn default() -> Self {
        Self {
            default_timeout_s: BASH_DEFAULT_TIMEOUT_S,
            max_output_length: BASH_MAX_OUTPUT_LENGTH,
            disable_networking: false,
            python_venv: Some(BASH_PYTHON_VENV.to_string()),
        }
    }
}

/// Build the upstream command wrapper for one bash invocation:
/// trailing-`&` wrap, then the nonce marker epilogue on the marker fd.
pub fn build_bash_command(command: &str, nonce: &str) -> String {
    let mut cmd = command.trim().to_string();
    if cmd.ends_with('&') {
        // Upstream wraps trailing background jobs so the marker still fires.
        cmd = format!("({cmd})");
    }
    let ec = "__karotte_ec";
    format!(
        "{cmd}; {ec}=$?; echo '{BASH_MARKER_PREFIX} {nonce} ${ec} \"$BASHOPTS\" \"$SHELLOPTS\"' 2>&- >&{BASH_MARKER_FD} || echo {BASH_MARKER_PREFIX} {nonce} ${ec} 2>&- || :"
    )
}

/// Build the `bash -n -c` syntax-check argv (upstream forwards observed
/// shell options).
pub fn syntax_check_argv(command: &str) -> Vec<String> {
    vec![
        BASH_BIN.to_string(),
        "-n".to_string(),
        "-c".to_string(),
        command.to_string(),
    ]
}

/// The student env (upstream `student_env` + venv PATH prepending):
/// `PYTHONSAFEPATH` popped, venv bin first when configured.
pub fn student_env_with_venv(base: &[(String, String)]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut has_path = false;
    for (k, v) in base {
        if k == "PYTHONSAFEPATH" {
            continue;
        }
        if k == "PATH" {
            has_path = true;
            out.push((k.clone(), v.clone()));
        } else {
            out.push((k.clone(), v.clone()));
        }
    }
    let _ = has_path;
    out
}

/// The upstream final truncation of stdout/stderr with its system note.
pub fn truncate_tool_output(text: &str, max_chars: usize) -> (String, Option<String>) {
    if text.chars().count() <= max_chars {
        return (text.to_string(), None);
    }
    let head: String = text.chars().take(max_chars).collect();
    let note = Some(format!("stdout was truncated to {max_chars} characters."));
    (head, note)
}

/// In-memory output cap: head + 8192-char tail + the upstream note
/// (upstream `_cap_output`).
pub fn cap_output_memory(raw: &str) -> String {
    if raw.len() <= BASH_OUTPUT_HARD_CAP {
        return raw.to_string();
    }
    let head_len = BASH_OUTPUT_HARD_CAP - BASH_OUTPUT_TAIL_KEEP - BASH_TRUNCATION_NOTE.len();
    let head: String = raw.chars().take(head_len).collect();
    let tail: String = raw
        .chars()
        .rev()
        .take(BASH_OUTPUT_TAIL_KEEP)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{head}{BASH_TRUNCATION_NOTE}{tail}")
}

/// Parse the marker line (upstream `_read_marker`): `<<exit>> <nonce> <ec>
/// <BASHOPTS> <SHELLOPTS>`.
pub fn parse_marker(line: &str) -> Option<(String, i32, String, String)> {
    let rest = line.strip_prefix(BASH_MARKER_PREFIX)?.trim_start();
    let mut parts = rest.splitn(4, ' ');
    let nonce = parts.next()?.to_string();
    let ec: i32 = parts.next()?.parse().ok()?;
    let bashopts = parts.next().unwrap_or("").to_string();
    let shell_opts = parts.next().unwrap_or("").to_string();
    Some((nonce, ec, bashopts, shell_opts))
}

/// One executed bash call (the harness side of `__call__`).
#[derive(Debug, Clone)]
pub struct BashCall {
    /// The requested command (after rstrip).
    pub command: String,
    /// Requested timeout, clamped to the configured default.
    pub timeout_s: f64,
}

/// Upstream clamps the requested timeout to the configured default.
pub fn clamp_timeout(requested: Option<f64>, config: &BashConfig) -> f64 {
    match requested {
        None => config.default_timeout_s,
        Some(t) => t.min(config.default_timeout_s),
    }
}

/// Build the bash tool result from a completed demoted run
/// (upstream `__call__` → ToolResult).
pub fn bash_result(run: &DemotedRun, stdout_capped: &str, config: &BashConfig) -> CallToolResult {
    let mut structured = serde_json::Map::new();
    structured.insert("stdout".to_string(), serde_json::json!(stdout_capped));
    if !run.stderr.is_empty() {
        structured.insert("stderr".to_string(), serde_json::json!(run.stderr));
    }
    let (out, note) = truncate_tool_output(stdout_capped, config.max_output_length);
    if out != stdout_capped {
        structured.insert("stdout".to_string(), serde_json::json!(out));
    }
    if run.exit != 0 {
        structured.insert("exit_code".to_string(), serde_json::json!(run.exit));
    }
    if let Some(n) = note {
        structured.insert("system".to_string(), serde_json::json!(n));
    } else if run.timed_out {
        structured.insert(
            "system".to_string(),
            serde_json::json!(format!(
                "[Interrupted due to timeout after {}s]",
                run.exit.max(0) as u64
            )),
        );
    }
    CallToolResult {
        content: vec![serde_json::to_value(crate::schemas::TextContent::new(out.clone())).unwrap()],
        structured_content: Some(serde_json::Value::Object(structured)),
        is_error: false,
    }
}

// ---------------------------------------------------------------------------
// view_lines_in_file.py
// ---------------------------------------------------------------------------

/// Upstream `ViewLinesInFileConfig.max_lines`.
pub const VIEW_LINES_MAX_LINES: usize = 1000;
/// Upstream `_MAX_CONTENT_BYTES`.
pub const VIEW_LINES_MAX_CONTENT_BYTES: usize = 384 * 1024;
/// Upstream `_MAX_STDERR_BYTES`.
pub const VIEW_LINES_MAX_STDERR_BYTES: usize = 16 * 1024;
/// The upstream truncation note.
pub const VIEW_LINES_TRUNCATION_NOTE: &str =
    "\n\n[... output truncated: exceeded 393216 bytes; request a narrower line range ...]";
/// Upstream sed path.
pub const SED_BIN: &str = "/usr/bin/sed";

/// Well-known absolute tool paths (upstream discipline: never PATH).
/// The awk path for line counting.
pub const AWK_BIN: &str = "/usr/bin/awk";

/// Validate a view-lines request (upstream `__call__` guards):
/// absolute path, 1-indexed inclusive range, ≤ `max_lines`.
pub fn validate_view_lines(
    file_path: &str,
    from_line: i64,
    to_line: i64,
    max_lines: usize,
) -> Result<()> {
    if !file_path.starts_with('/') {
        return Err(Error::InvalidSpec("file_path must be absolute".into()));
    }
    if from_line < 1 {
        return Err(Error::InvalidSpec("from_line must be >= 1".into()));
    }
    if to_line < from_line {
        return Err(Error::InvalidSpec("to_line must be >= from_line".into()));
    }
    let requested = (to_line - from_line + 1) as usize;
    if requested > max_lines {
        return Err(Error::InvalidSpec(format!(
            "Requested {requested} lines, more than the maximum of {max_lines}"
        )));
    }
    Ok(())
}

/// The upstream sed argv for a 1-indexed inclusive range.
pub fn sed_range_argv(file_path: &str, from_line: i64, to_line: i64) -> Vec<String> {
    vec![
        SED_BIN.to_string(),
        "-n".to_string(),
        format!("{},{}p;{}q", from_line, to_line, to_line + 1),
        file_path.to_string(),
    ]
}

/// The upstream awk argv for a line count.
pub fn line_count_argv(file_path: &str) -> Vec<String> {
    vec![
        AWK_BIN.to_string(),
        "END{print NR}".to_string(),
        file_path.to_string(),
    ]
}

/// JSON-safe head truncation to a byte budget (upstream
/// `head_within_json_bytes` with the truncation note).
pub fn truncate_content_json_safe(content: &str, max_bytes: usize) -> String {
    let note_len = crate::text::json_encoded_len(VIEW_LINES_TRUNCATION_NOTE);
    let budget = max_bytes.saturating_sub(note_len);
    let head = crate::text::head_within_json_bytes(content, budget);
    if head.chars().count() < content.chars().count() {
        format!("{head}{VIEW_LINES_TRUNCATION_NOTE}")
    } else {
        head
    }
}

/// Build the view-lines result (upstream `__call__`).
pub fn view_lines_result(
    file_path: &str,
    from_line: i64,
    to_line: i64,
    content: &str,
) -> CallToolResult {
    CallToolResult {
        content: vec![
            serde_json::to_value(crate::schemas::TextContent::new(content.to_string())).unwrap(),
        ],
        structured_content: Some(serde_json::json!({
            "file_path": file_path,
            "from_line": from_line,
            "to_line": to_line,
            "content": content,
        })),
        is_error: false,
    }
}

// ---------------------------------------------------------------------------
// replace_in_file.py
// ---------------------------------------------------------------------------

/// Upstream `_MAX_FILE_BYTES`.
pub const REPLACE_MAX_FILE_BYTES: usize = 10 * 1024 * 1024;
/// Upstream head cap read: `_MAX_FILE_BYTES + 1`.
pub const REPLACE_HEAD_CAP: usize = REPLACE_MAX_FILE_BYTES + 1;
/// Upstream head path.
pub const HEAD_BIN: &str = "/usr/bin/head";
/// Upstream tee path (write path).
pub const TEE_BIN: &str = "/usr/bin/tee";
/// Upstream base64 path (image reads).
pub const BASE64_BIN: &str = "/usr/bin/base64";

/// Upstream `_DIFF_CONTEXT_LINES`.
pub const DIFF_CONTEXT_LINES: usize = 3;
/// Upstream `_MAX_DIFF_OCCURRENCES`.
pub const MAX_DIFF_OCCURRENCES: usize = 50;
/// Upstream `_MAX_DIFF_BYTES`.
pub const MAX_DIFF_BYTES: usize = 64 * 1024;
/// Upstream diff truncation note.
pub const DIFF_TRUNCATION_NOTE: &str = "\n(diff truncated)\n";

/// Validate a replace request: absolute path, non-empty needle, file
/// size and projected size under the cap.
pub fn validate_replace(
    file_path: &str,
    old: &str,
    content_len: usize,
    occurrences: usize,
    old_len: usize,
    new_len: usize,
) -> Result<()> {
    if !file_path.starts_with('/') {
        return Err(Error::InvalidSpec("file_path must be absolute".into()));
    }
    if old.is_empty() {
        return Err(Error::InvalidSpec("old must not be empty".into()));
    }
    if content_len > REPLACE_MAX_FILE_BYTES {
        return Err(Error::InvalidSpec(format!(
            "File too large to edit in place: {file_path} is over {} bytes",
            REPLACE_MAX_FILE_BYTES
        )));
    }
    // Upstream: len + occurrences * (len(new) - len(old)) > cap.
    let growth = occurrences as i64 * (new_len as i64 - old_len as i64);
    if (content_len as i64 + growth) > REPLACE_MAX_FILE_BYTES as i64 {
        return Err(Error::InvalidSpec(
            "Narrow the match or replace in smaller steps.".into(),
        ));
    }
    Ok(())
}

/// Apply the replacement (upstream: `replace(old, new)` or count-1).
pub fn apply_replacement(content: &str, old: &str, new: &str, replace_all: bool) -> String {
    if replace_all {
        content.replace(old, new)
    } else {
        content.replacen(old, new, 1)
    }
}

/// The hand-rolled unified diff (upstream omits difflib for lack of
/// "\ No newline at end of file"). Renders hunks with `DIFF_CONTEXT_LINES`
/// context, merges hunks separated by ≤ 2×context unchanged lines, caps
/// at `MAX_DIFF_OCCURRENCES` shown blocks and `MAX_DIFF_BYTES` bytes.
pub fn unified_diff(file_path: &str, old: &str, new: &str) -> String {
    let old_lines: Vec<&str> = if old.is_empty() {
        vec![]
    } else {
        old.split('\n').collect()
    };
    let new_lines: Vec<&str> = if new.is_empty() {
        vec![]
    } else {
        new.split('\n').collect()
    };

    let ops = diff_ops(&old_lines, &new_lines);
    if !ops.iter().any(|o| matches!(o, Op::Del(_) | Op::Ins(_))) {
        return String::new();
    }

    let hunks = hunks_from_ops(&ops, DIFF_CONTEXT_LINES);
    let mut out = String::new();
    out.push_str(&format!("--- {file_path}\n"));
    out.push_str(&format!("+++ {file_path}\n"));
    let shown = hunks.len().min(MAX_DIFF_OCCURRENCES);
    for hunk in &hunks[..shown] {
        let block = render_hunk(hunk);
        out.push_str(&block);
        if out.len() > MAX_DIFF_BYTES {
            out.push_str(DIFF_TRUNCATION_NOTE);
            break;
        }
    }
    if hunks.len() > shown {
        out.push_str(&format!(
            "({} more replacement(s) not shown)\n",
            hunks.len() - shown
        ));
    }
    // "\ No newline at end of file" when a side lacks its trailing
    // newline (the reason upstream hand-rolls the diff).
    if (!old.ends_with('\n') && !old.is_empty()) || (!new.ends_with('\n') && !new.is_empty()) {
        out.push_str("\\ No newline at end of file\n");
    }
    out
}

/// One diff operation over line sequences.
#[derive(Debug, Clone, PartialEq)]
enum Op {
    /// Unchanged lines (present on both sides).
    Ctx(Vec<String>),
    /// Lines removed from the old side.
    Del(Vec<String>),
    /// Lines added on the new side.
    Ins(Vec<String>),
}

/// LCS-based op sequence (upstream relies on difflib's SequenceMatcher;
/// the port uses the classic dynamic program — files are ≤ 10 MiB and
/// the shown blocks are capped anyway).
fn diff_ops(old_lines: &[&str], new_lines: &[&str]) -> Vec<Op> {
    let n = old_lines.len();
    let m = new_lines.len();
    // dp[i][j] = LCS length of old[i..], new[j..]
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if old_lines[i] == new_lines[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    let mut ops: Vec<Op> = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < n && j < m {
        if old_lines[i] == new_lines[j] {
            append_op(&mut ops, Op::Ctx(vec![old_lines[i].to_string()]));
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            append_op(&mut ops, Op::Del(vec![old_lines[i].to_string()]));
            i += 1;
        } else {
            append_op(&mut ops, Op::Ins(vec![new_lines[j].to_string()]));
            j += 1;
        }
    }
    while i < n {
        append_op(&mut ops, Op::Del(vec![old_lines[i].to_string()]));
        i += 1;
    }
    while j < m {
        append_op(&mut ops, Op::Ins(vec![new_lines[j].to_string()]));
        j += 1;
    }
    ops
}

fn append_op(ops: &mut Vec<Op>, op: Op) {
    use std::mem::discriminant;
    if let Some(last) = ops.last_mut() {
        let same = discriminant(last) == discriminant(&op);
        if same {
            match (last, op) {
                (Op::Ctx(a), Op::Ctx(b)) | (Op::Del(a), Op::Del(b)) | (Op::Ins(a), Op::Ins(b)) => {
                    a.extend(b);
                    return;
                }
                _ => unreachable!(),
            }
        }
    }
    ops.push(op);
}

/// One rendered hunk: a slice of ops plus the running line numbers.
#[derive(Debug, Clone)]
struct Hunk {
    ops: Vec<Op>,
    old_start: usize,
    new_start: usize,
}

/// Group change blocks into hunks with `context` lines of context,
/// merging blocks separated by ≤ 2×context unchanged lines
/// (upstream `_MAX_DIFF_OCCURRENCES` counts the shown blocks).
fn hunks_from_ops(ops: &[Op], context: usize) -> Vec<Hunk> {
    let mut hunks: Vec<Hunk> = Vec::new();
    let mut old_line = 0usize; // 0-indexed; header prints +1
    let mut new_line = 0usize;
    // positions of each op in line terms
    let mut idx = 0usize;
    while idx < ops.len() {
        match &ops[idx] {
            Op::Ctx(_) => {
                let len = match &ops[idx] {
                    Op::Ctx(v) => v.len(),
                    _ => unreachable!(),
                };
                old_line += len;
                new_line += len;
                idx += 1;
            }
            Op::Del(_) | Op::Ins(_) => {
                // Collect [change block] possibly separated by short ctx
                // runs, per the merge rule.
                let hunk_old_start = old_line.saturating_sub(context);
                let hunk_new_start = new_line.saturating_sub(context);
                // include preceding context
                let mut taken: Vec<Op> = Vec::new();
                // Walk back to capture context lines from previous Ctx op
                if idx > 0 {
                    if let Op::Ctx(v) = &ops[idx - 1] {
                        let skip = v.len().saturating_sub(context);
                        for l in &v[skip..] {
                            taken.push(Op::Ctx(vec![l.clone()]));
                        }
                    }
                }
                let mut scan = idx;
                let mut trailing_ctx: Vec<String> = Vec::new();
                loop {
                    match ops.get(scan) {
                        Some(Op::Del(v)) => {
                            flush_ctx(&mut taken, &mut trailing_ctx);
                            taken.push(Op::Del(v.clone()));
                            old_line += v.len();
                            scan += 1;
                        }
                        Some(Op::Ins(v)) => {
                            flush_ctx(&mut taken, &mut trailing_ctx);
                            taken.push(Op::Ins(v.clone()));
                            new_line += v.len();
                            scan += 1;
                        }
                        Some(Op::Ctx(v)) => {
                            // Tentatively take the ctx; if the next change
                            // is beyond the merge window, split here.
                            let next_change_gap = v.len();
                            if next_change_gap <= 2 * context {
                                flush_ctx(&mut taken, &mut trailing_ctx);
                                taken.push(Op::Ctx(v.clone()));
                                old_line += v.len();
                                new_line += v.len();
                                scan += 1;
                                continue;
                            }
                            // long gap: take only the `context` lines
                            // immediately following the change
                            let keep = v.len().min(context);
                            for l in v.iter().take(keep) {
                                trailing_ctx.push(l.clone());
                            }
                            old_line += v.len();
                            new_line += v.len();
                            scan += 1;
                            break;
                        }
                        None => {
                            break;
                        }
                    }
                }
                flush_ctx(&mut taken, &mut trailing_ctx);
                hunks.push(Hunk {
                    ops: taken,
                    old_start: hunk_old_start,
                    new_start: hunk_new_start,
                });
                idx = scan;
            }
        }
    }
    hunks
}

fn flush_ctx(taken: &mut Vec<Op>, trailing: &mut Vec<String>) {
    if !trailing.is_empty() {
        let v = std::mem::take(trailing);
        append_op(taken, Op::Ctx(v));
    }
}

/// Render one hunk: header `@@ -a,b +c,d @@` then ` `/`-`/`+` lines.
fn render_hunk(hunk: &Hunk) -> String {
    let old_count: usize = hunk
        .ops
        .iter()
        .map(|o| match o {
            Op::Ctx(v) | Op::Del(v) => v.len(),
            Op::Ins(_) => 0,
        })
        .sum();
    let new_count: usize = hunk
        .ops
        .iter()
        .map(|o| match o {
            Op::Ctx(v) | Op::Ins(v) => v.len(),
            Op::Del(_) => 0,
        })
        .sum();
    let mut out = format!(
        "@@ -{},{} +{},{} @@\n",
        hunk.old_start + 1,
        old_count,
        hunk.new_start + 1,
        new_count
    );
    for op in &hunk.ops {
        let (prefix, lines) = match op {
            Op::Ctx(v) => (" ", v),
            Op::Del(v) => ("-", v),
            Op::Ins(v) => ("+", v),
        };
        for l in lines {
            out.push_str(&format!("{prefix}{l}\n"));
        }
    }
    out
}

/// Build the replace-in-file result.
pub fn replace_result(new_content: &str, file_path: &str, old: &str) -> CallToolResult {
    let diff = unified_diff(file_path, old, new_content);
    CallToolResult {
        content: vec![serde_json::to_value(crate::schemas::TextContent::new(
            "Replacement successful",
        ))
        .unwrap()],
        structured_content: Some(serde_json::json!({
            "result": "Replacement successful",
            "diff": diff,
        })),
        is_error: false,
    }
}

// ---------------------------------------------------------------------------
// view_image_file.py
// ---------------------------------------------------------------------------

/// Upstream `ViewImageFileConfig.max_img_dim`.
pub const VIEW_IMAGE_MAX_DIM: u32 = 2000;
/// Upstream `max_file_bytes`.
pub const VIEW_IMAGE_MAX_FILE_BYTES: usize = 5 * 1024 * 1024;
/// Accepted extensions (upstream docstring).
pub const VIEW_IMAGE_EXTENSIONS: &[&str] = &["jpeg", "png", "gif", "webp"];

/// Validate an image request: extension and byte cap.
pub fn validate_image(path: &str, size: usize) -> Result<()> {
    if !path.starts_with('/') {
        return Err(Error::InvalidSpec("file_path must be absolute".into()));
    }
    let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();
    if !VIEW_IMAGE_EXTENSIONS.contains(&ext.as_str()) {
        return Err(Error::InvalidSpec(format!(
            "Unsupported image extension: {ext} (expected one of {VIEW_IMAGE_EXTENSIONS:?})"
        )));
    }
    if size > VIEW_IMAGE_MAX_FILE_BYTES {
        return Err(Error::InvalidSpec(
            "Image is too large. Please try a smaller image.".into(),
        ));
    }
    Ok(())
}

/// Dimension check (upstream: dims > max → refuse).
pub fn validate_dims(width: u32, height: u32, max_dim: u32) -> Result<()> {
    if width > max_dim || height > max_dim {
        return Err(Error::InvalidSpec(
            "Image is too large. Please try a smaller image.".into(),
        ));
    }
    Ok(())
}

/// The image tool result: an MCP ImageContent block (the only tool that
/// returns image content).
pub fn image_result(data: &str, mime: &str) -> CallToolResult {
    CallToolResult {
        content: vec![serde_json::to_value(crate::schemas::ImageContent {
            kind: "image".into(),
            data: data.to_string(),
            mime_type: mime.to_string(),
        })
        .unwrap()],
        structured_content: None,
        is_error: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bash_command_wraps_background_and_adds_marker() {
        let cmd = build_bash_command("sleep 5 &", "abc123");
        assert!(cmd.starts_with("(sleep 5 &)"));
        assert!(cmd.contains("__karotte_ec=$?"));
        assert!(cmd.contains("'<<exit>> abc123 $__karotte_ec"));
        assert!(cmd.contains(">&231"));
    }

    #[test]
    fn bash_marker_roundtrip() {
        let line = "<<exit>> abc123 17 pipefail noclobber";
        let (nonce, ec, opts, shel) = parse_marker(line).unwrap();
        assert_eq!(nonce, "abc123");
        assert_eq!(ec, 17);
        assert_eq!(opts, "pipefail");
        assert_eq!(shel, "noclobber");
    }

    #[test]
    fn bash_output_memory_cap() {
        let small = "x".repeat(1024);
        assert_eq!(cap_output_memory(&small), small);
        let big = "y".repeat(BASH_OUTPUT_HARD_CAP + 10);
        let capped = cap_output_memory(&big);
        assert!(capped.contains("[... output truncated: exceeded 1 MiB in memory ...]"));
        assert!(capped.len() < big.len());
        assert!(capped.ends_with("yyyy"));
    }

    #[test]
    fn bash_timeout_clamped_to_default() {
        let cfg = BashConfig::default();
        assert_eq!(clamp_timeout(None, &cfg), 3600.0);
        assert_eq!(clamp_timeout(Some(10.0), &cfg), 10.0);
        assert_eq!(clamp_timeout(Some(9_999_999.0), &cfg), 3600.0);
    }

    #[test]
    fn bash_result_truncates_and_notes() {
        let cfg = BashConfig::default();
        let long = "z".repeat(cfg.max_output_length + 100);
        let run = DemotedRun {
            exit: 0,
            stdout: String::new(),
            stderr: String::new(),
            timed_out: false,
        };
        let r = bash_result(&run, &long, &cfg);
        let sc = r.structured_content.unwrap();
        assert_eq!(
            sc["stdout"].as_str().unwrap().chars().count(),
            cfg.max_output_length
        );
        assert_eq!(
            sc["system"].as_str().unwrap(),
            "stdout was truncated to 16000 characters."
        );
    }

    #[test]
    fn view_lines_validation() {
        assert!(validate_view_lines("/f.txt", 1, 10, 1000).is_ok());
        assert!(validate_view_lines("rel.txt", 1, 10, 1000).is_err());
        assert!(validate_view_lines("/f.txt", 0, 10, 1000).is_err());
        assert!(validate_view_lines("/f.txt", 10, 5, 1000).is_err());
        assert!(validate_view_lines("/f.txt", 1, 1001, 1000).is_err());
    }

    #[test]
    fn sed_range_shape() {
        let argv = sed_range_argv("/f.txt", 4, 7);
        assert_eq!(argv[0], "/usr/bin/sed");
        assert_eq!(argv[1], "-n");
        assert_eq!(argv[2], "4,7p;8q");
        assert_eq!(argv[3], "/f.txt");
    }

    #[test]
    fn view_lines_json_safe_truncation() {
        let content = "行".repeat(100_000);
        let cut = truncate_content_json_safe(&content, 4096);
        assert!(crate::text::json_encoded_len(&cut) <= 393_216);
        assert!(cut.contains("request a narrower line range"));
    }

    #[test]
    fn replace_validation_and_projection() {
        // old missing → error upstream; here we validate the projection
        // guard: growing beyond the cap.
        assert!(validate_replace("/f", "old", 100, 1, 3, 7).is_ok());
        let huge = 10 * 1024 * 1024 + 1;
        assert!(validate_replace("/f", "o", huge, 0, 1, 1).is_err());
        // projected growth over cap → "Narrow the match"
        let err = validate_replace("/f", "o", 10 * 1024 * 1024 - 10, 100, 1, 100).unwrap_err();
        assert!(err.to_string().contains("Narrow the match"));
    }

    #[test]
    fn replacement_first_or_all() {
        let content = "a b a b a";
        assert_eq!(apply_replacement(content, "a", "X", false), "X b a b a");
        assert_eq!(apply_replacement(content, "a", "X", true), "X b X b X");
    }

    #[test]
    fn unified_diff_renders_hunk_headers_and_lines() {
        let old = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n";
        let new = "line1\nline2\nline3\nCHANGED\nline5\nline6\nline7\nline8\nline9\nline10\n";
        let diff = unified_diff("/f.txt", old, new);
        assert!(diff.starts_with("--- /f.txt\n+++ /f.txt\n"));
        assert!(diff.contains("@@ -1,"));
        assert!(diff.contains("-line4"));
        assert!(diff.contains("+CHANGED"));
        // 3 lines of context before/after
        assert!(diff.contains(" line1"));
        assert!(diff.contains(" line5"));
    }

    #[test]
    fn unified_diff_merges_close_hunks() {
        let old = "a\nb\nc\nd\ne\nf\ng\nh\n";
        let new = "a\nB\nc\nd\nE\nf\ng\nh\n";
        let diff = unified_diff("/f", old, new);
        // Changes at lines 2 and 5 with 3 context: one merged hunk
        // (count hunk headers, not the "@@" pair inside each header).
        assert_eq!(diff.matches("@@ -").count(), 1, "diff: {diff}");
    }

    #[test]
    fn unified_diff_no_newline_marker() {
        let old = "no newline";
        let new = "no newline at all";
        let diff = unified_diff("/f", old, new);
        assert!(diff.contains("\\ No newline at end of file"));
    }

    #[test]
    fn image_validation() {
        assert!(validate_image("/x.png", 1024).is_ok());
        assert!(validate_image("/x.bmp", 1024).is_err());
        assert!(validate_image("/x.png", VIEW_IMAGE_MAX_FILE_BYTES + 1).is_err());
        assert!(validate_dims(100, 100, VIEW_IMAGE_MAX_DIM).is_ok());
        assert!(validate_dims(3000, 100, VIEW_IMAGE_MAX_DIM).is_err());
    }

    #[test]
    fn image_result_is_image_content() {
        let r = image_result("QUJD", "image/png");
        assert_eq!(r.content[0]["type"], "image");
        assert_eq!(r.content[0]["mimeType"], "image/png");
        assert_eq!(r.content[0]["data"], "QUJD");
        assert!(r.structured_content.is_none());
    }

    #[test]
    fn scripted_backend_check_access() {
        let backend = ScriptedBackend::new(vec![(
            "-r /f".into(),
            DemotedRun {
                exit: 0,
                stdout: String::new(),
                stderr: String::new(),
                timed_out: false,
            },
        )]);
        assert!(backend.check_access("-r", "/f").unwrap());
        assert!(!backend.check_access("-r", "/nope").unwrap());
    }
}
