//! Chronus command interpreter.
//!
//! Sandboxes expose a shell-like session API; commands run against the
//! layered guest filesystem. Output is produced as an event vector so the
//! same execution path serves both blocking `exec` (events concatenated)
//! and streaming `exec_stream` (events pushed as they are "produced",
//! with per-event pacing for streaming semantics).

use std::collections::BTreeMap;

use dsec_protocol::message::Stdio;
use dsec_storage::imagefs::{normalize, LayeredImage};

/// One unit of command output.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecEvent {
    pub stream: Stdio,
    pub data: Vec<u8>,
    /// Suggested pacing before emitting this event (streaming mode only).
    pub delay_ms: u64,
}

/// Full command result.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecOutput {
    pub exit_code: i32,
    pub events: Vec<ExecEvent>,
    /// Working directory after the command (`cd`).
    pub cwd_after: String,
}

impl ExecOutput {
    pub fn stdout(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for e in &self.events {
            if e.stream == Stdio::Stdout {
                out.extend_from_slice(&e.data);
            }
        }
        out
    }

    pub fn stderr(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for e in &self.events {
            if e.stream == Stdio::Stderr {
                out.extend_from_slice(&e.data);
            }
        }
        out
    }
}

/// Redirection target parsed from `cmd ... > path` / `>> path`.
#[derive(Debug, Clone)]
struct Redirect {
    path: String,
    append: bool,
}

fn parse_redirect(tokens: &mut Vec<String>) -> Option<Redirect> {
    let mut i = 0;
    while i < tokens.len() {
        if (tokens[i] == ">" || tokens[i] == ">>") && i + 1 < tokens.len() {
            let path = tokens[i + 1].clone();
            let append = tokens[i] == ">>";
            tokens.drain(i..=i + 1);
            return Some(Redirect { path, append });
        }
        i += 1;
    }
    None
}

/// Resolves a (possibly relative) path against the session cwd.
pub fn resolve_path(cwd: &str, p: &str) -> String {
    let joined = if p.starts_with('/') {
        p.to_string()
    } else {
        format!("{}/{}", cwd.trim_end_matches('/'), p)
    };
    normalize(&joined).unwrap_or(joined)
}

/// Runs a command line inside a sandbox session.
pub async fn run(
    cmd: &str,
    fs: &LayeredImage,
    cwd: &str,
    env: &std::collections::HashMap<String, String>,
    hostname: &str,
    epoch_ms: u64,
) -> ExecOutput {
    let mut tokens: Vec<String> = cmd.split_whitespace().map(|s| s.to_string()).collect();
    let redirect = parse_redirect(&mut tokens);

    let mut out = if tokens.is_empty() {
        ExecOutput {
            exit_code: 0,
            events: Vec::new(),
            cwd_after: cwd.to_string(),
        }
    } else {
        run_builtin(&tokens, fs, cwd, env, hostname, epoch_ms).await
    };

    if let Some(red) = redirect {
        let stdout = out.stdout();
        let target = resolve_path(cwd, &red.path);
        if red.append {
            let _ = fs.append_file(&target, &stdout).await;
        } else {
            let _ = fs.write_file(&target, &stdout).await;
        }
        // Redirected output is not echoed.
        out.events.retain(|e| e.stream != Stdio::Stdout);
    }
    out
}

async fn run_builtin(
    tokens: &[String],
    fs: &LayeredImage,
    cwd: &str,
    env: &std::collections::HashMap<String, String>,
    hostname: &str,
    epoch_ms: u64,
) -> ExecOutput {
    let cmd = tokens[0].as_str();
    let args = &tokens[1..];
    let cwd_after = cwd.to_string();
    let base = |exit: i32, events: Vec<ExecEvent>| ExecOutput {
        exit_code: exit,
        events,
        cwd_after: cwd_after.clone(),
    };
    let say = |s: &str| {
        vec![ExecEvent {
            stream: Stdio::Stdout,
            data: format!("{}\n", s).into_bytes(),
            delay_ms: 0,
        }]
    };
    let err = |s: &str| {
        vec![ExecEvent {
            stream: Stdio::Stderr,
            data: format!("{}\n", s).into_bytes(),
            delay_ms: 0,
        }]
    };

    match cmd {
        "echo" => {
            let text = args.join(" ");
            base(0, say(&text))
        }
        "pwd" => base(0, say(cwd)),
        "cd" => {
            let target = args.first().map(|s| s.as_str()).unwrap_or("/");
            let resolved = resolve_path(cwd, target);
            match fs.stat(&resolved) {
                Ok(m) if m.is_dir => ExecOutput {
                    exit_code: 0,
                    events: Vec::new(),
                    cwd_after: resolved,
                },
                _ => base(1, err(&format!("cd: {}: No such file or directory", target))),
            }
        }
        "ls" => {
            let target = args.iter().find(|a| !a.starts_with('-')).cloned().unwrap_or_else(|| ".".to_string());
            let resolved = resolve_path(cwd, &target);
            match fs.list_dir(&resolved) {
                Ok(mut entries) => {
                    entries.sort();
                    base(0, say(&entries.join("\n")))
                }
                Err(e) => base(1, err(&format!("ls: {}: {}", target, e))),
            }
        }
        "cat" => {
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            let mut code = 0;
            if args.is_empty() {
                stderr.extend_from_slice(b"cat: missing operand\n");
                code = 1;
            }
            for f in args {
                let resolved = resolve_path(cwd, f);
                match fs.read_file(&resolved).await {
                    Ok(data) => stdout.extend_from_slice(&data),
                    Err(e) => {
                        stderr.extend_from_slice(format!("cat: {}: {}\n", f, e).as_bytes());
                        code = 1;
                    }
                }
            }
            base(code, vec![
                ExecEvent { stream: Stdio::Stdout, data: stdout, delay_ms: 0 },
                ExecEvent { stream: Stdio::Stderr, data: stderr, delay_ms: 0 },
            ])
        }
        "mkdir" => {
            let parents = args.iter().any(|a| a == "-p" || a == "--parents");
            let target = args.iter().find(|a| !a.starts_with('-'));
            match target {
                None => base(1, err("mkdir: missing operand")),
                Some(t) => {
                    let resolved = resolve_path(cwd, t);
                    if fs.exists(&resolved) && !parents {
                        base(1, err(&format!("mkdir: cannot create directory '{}': File exists", t)))
                    } else {
                        match fs.mkdir_p(&resolved) {
                            Ok(()) => base(0, vec![]),
                            Err(e) => base(1, err(&format!("mkdir: {}: {}", t, e))),
                        }
                    }
                }
            }
        }
        "touch" => match args.first() {
            None => base(1, err("touch: missing file operand")),
            Some(t) => {
                let resolved = resolve_path(cwd, t);
                if !fs.exists(&resolved) {
                    if let Err(e) = fs.write_file(&resolved, &[]).await {
                        return base(1, err(&format!("touch: {}: {}", t, e)));
                    }
                }
                base(0, vec![])
            }
        },
        "rm" => {
            let recursive = args.iter().any(|a| a == "-r" || a == "-rf" || a == "--recursive");
            let force = args.iter().any(|a| a == "-f" || a == "-rf");
            let target = args.iter().find(|a| !a.starts_with('-'));
            match target {
                None => base(1, err("rm: missing operand")),
                Some(t) => {
                    let resolved = resolve_path(cwd, t);
                    if !fs.exists(&resolved) {
                        if force {
                            base(0, vec![])
                        } else {
                            base(1, err(&format!("rm: cannot remove '{}': No such file or directory", t)))
                        }
                    } else {
                        match fs.rm(&resolved, recursive) {
                            Ok(()) => base(0, vec![]),
                            Err(e) => base(1, err(&format!("rm: {}: {}", t, e))),
                        }
                    }
                }
            }
        }
        "cp" => {
            if args.len() != 2 {
                return base(1, err("cp: usage: cp SRC DST"));
            }
            let src = resolve_path(cwd, &args[0]);
            let dst = resolve_path(cwd, &args[1]);
            match fs.read_file(&src).await {
                Ok(data) => match fs.write_file(&dst, &data).await {
                    Ok(_) => base(0, vec![]),
                    Err(e) => base(1, err(&format!("cp: {}: {}", args[1], e))),
                },
                Err(e) => base(1, err(&format!("cp: {}: {}", args[0], e))),
            }
        }
        "mv" => {
            if args.len() != 2 {
                return base(1, err("mv: usage: mv SRC DST"));
            }
            let src = resolve_path(cwd, &args[0]);
            let dst = resolve_path(cwd, &args[1]);
            match fs.read_file(&src).await {
                Ok(data) => {
                    let _ = fs.rm(&src, false);
                    match fs.write_file(&dst, &data).await {
                        Ok(_) => base(0, vec![]),
                        Err(e) => base(1, err(&format!("mv: {}: {}", args[1], e))),
                    }
                }
                Err(e) => base(1, err(&format!("mv: {}: {}", args[0], e))),
            }
        }
        "sleep" => {
            let ms: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(0);
            tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
            base(0, vec![])
        }
        "seq" => {
            // Streams: one event per line, paced.
            let n: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(0);
            let mut events = Vec::with_capacity(n as usize);
            for i in 1..=n {
                events.push(ExecEvent {
                    stream: Stdio::Stdout,
                    data: format!("{}\n", i).into_bytes(),
                    delay_ms: 1,
                });
            }
            base(0, events)
        }
        "env" | "printenv" => {
            let sorted: BTreeMap<&String, &String> = env.iter().collect();
            let text = sorted.iter().map(|(k, v)| format!("{}={}", k, v)).collect::<Vec<_>>().join("\n");
            if cmd == "printenv" && !args.is_empty() {
                match env.get(&args[0]) {
                    Some(v) => base(0, say(v)),
                    None => base(1, vec![]),
                }
            } else {
                base(0, say(&text))
            }
        }
        "true" => base(0, vec![]),
        "false" => base(1, vec![]),
        "exit" => {
            let code = args.first().and_then(|s| s.parse::<i32>().ok()).unwrap_or(0);
            base(code, vec![])
        }
        "date" => base(0, say(&format!("epoch:{}", epoch_ms))),
        "whoami" => base(0, say(env.get("USER").map(|s| s.as_str()).unwrap_or("agent"))),
        "hostname" => base(0, say(hostname)),
        "wc" => {
            let count_lines = args.iter().any(|a| a == "-l");
            let file = args.iter().find(|a| !a.starts_with('-'));
            match file {
                None => base(1, err("wc: missing operand")),
                Some(f) => match fs.read_file(&resolve_path(cwd, f)).await {
                    Ok(data) => {
                        let n = if count_lines {
                            data.iter().filter(|&&b| b == b'\n').count()
                        } else {
                            data.len()
                        };
                        base(0, say(&n.to_string()))
                    }
                    Err(e) => base(1, err(&format!("wc: {}: {}", f, e))),
                },
            }
        }
        "grep" => {
            if args.len() < 2 {
                return base(2, err("usage: grep PATTERN FILE..."));
            }
            let pattern = &args[0];
            let mut stdout = Vec::new();
            let mut code = 1;
            for f in &args[1..] {
                if let Ok(data) = fs.read_file(&resolve_path(cwd, f)).await {
                    for line in String::from_utf8_lossy(&data).lines() {
                        if line.contains(pattern.as_str()) {
                            stdout.extend_from_slice(format!("{}\n", line).as_bytes());
                            code = 0;
                        }
                    }
                }
            }
            base(code, vec![ExecEvent { stream: Stdio::Stdout, data: stdout, delay_ms: 0 }])
        }
        "head" => {
            // head [-n N] FILE
            let (n, file) = if args.len() >= 3 && args[0] == "-n" {
                (args[1].parse().unwrap_or(10), Some(args[2].clone()))
            } else if args.len() == 1 {
                (10, Some(args[0].clone()))
            } else {
                (10, None)
            };
            match file {
                None => base(1, err("head: missing operand")),
                Some(f) => match fs.read_file(&resolve_path(cwd, &f)).await {
                    Ok(data) => {
                        let text = String::from_utf8_lossy(&data);
                        let first: Vec<&str> = text.lines().take(n).collect();
                        base(0, say(&first.join("\n")))
                    }
                    Err(e) => base(1, err(&format!("head: {}: {}", f, e))),
                },
            }
        }
        "help" => base(0, say(
            "builtins: echo pwd cd ls cat mkdir touch rm cp mv sleep seq env printenv true false exit date whoami hostname wc grep head help",
        )),
        _ => base(127, err(&format!("{}: command not found", cmd))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsec_storage::cache::LruBlockCache;
    use dsec_storage::erofs::{ErofsImageBuilder, OnDemandLoader};
    use dsec_storage::latency::LatencyModel;
    use dsec_storage::overlay::OverlayDev;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    fn test_fs() -> Arc<LayeredImage> {
        let image = Arc::new(ErofsImageBuilder::agent_base().build());
        let loader = Arc::new(OnDemandLoader::new(
            image,
            Arc::new(LruBlockCache::new(64)),
            LatencyModel::fixed(Duration::ZERO),
        ));
        let overlay = Arc::new(OverlayDev::new(loader));
        Arc::new(LayeredImage::new(overlay))
    }

    fn env() -> HashMap<String, String> {
        let mut env = HashMap::new();
        env.insert("USER".to_string(), "agent".to_string());
        env.insert("HOME".to_string(), "/home/agent".to_string());
        env
    }

    async fn sh(fs: &LayeredImage, cmd: &str) -> ExecOutput {
        run(cmd, fs, "/", &env(), "sb-1", 1_700_000_000_000).await
    }

    #[tokio::test]
    async fn echo_and_exit_codes() {
        let fs = test_fs();
        let out = sh(&fs, "echo hello world").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout(), b"hello world\n");
        assert_eq!(sh(&fs, "false").await.exit_code, 1);
        assert_eq!(sh(&fs, "exit 3").await.exit_code, 3);
    }

    #[tokio::test]
    async fn cat_base_files() {
        let fs = test_fs();
        let out = sh(&fs, "cat /etc/hostname").await;
        assert_eq!(out.stdout(), b"sandbox-agent");
        let missing = sh(&fs, "cat /nope").await;
        assert_eq!(missing.exit_code, 1);
        assert!(missing.stderr().starts_with(b"cat: /nope:"));
    }

    #[tokio::test]
    async fn ls_and_cd() {
        let fs = test_fs();
        let out = sh(&fs, "ls /etc").await;
        let text = String::from_utf8(out.stdout()).unwrap();
        assert_eq!(text.trim(), "hostname\nos-release\npasswd");
        let cd = sh(&fs, "cd /etc").await;
        assert_eq!(cd.cwd_after, "/etc");
        assert_eq!(cd.exit_code, 0);
        let bad = sh(&fs, "cd /missing").await;
        assert_eq!(bad.exit_code, 1);
        let pwd = run("pwd", &fs, "/etc", &HashMap::new(), "sb", 1).await;
        assert_eq!(pwd.stdout(), b"/etc\n");
    }

    #[tokio::test]
    async fn redirection_writes_files() {
        let fs = test_fs();
        let out = sh(&fs, "echo captured > /tmp/out.txt").await;
        assert_eq!(out.exit_code, 0);
        assert!(out.stdout().is_empty()); // redirected
        let read = sh(&fs, "cat /tmp/out.txt").await;
        assert_eq!(read.stdout(), b"captured\n");
        let _ = sh(&fs, "echo more >> /tmp/out.txt").await;
        let read2 = sh(&fs, "cat /tmp/out.txt").await;
        assert_eq!(read2.stdout(), b"captured\nmore\n");
    }

    #[tokio::test]
    async fn file_toolchain() {
        let fs = test_fs();
        assert_eq!(sh(&fs, "mkdir -p /work/deep").await.exit_code, 0);
        assert_eq!(sh(&fs, "touch /work/deep/a.txt").await.exit_code, 0);
        assert!(sh(&fs, "cat /work/deep/a.txt").await.stdout().is_empty());
        let _ = sh(&fs, "echo one > /work/f1.txt").await;
        let _ = sh(&fs, "cp /work/f1.txt /work/f2.txt").await;
        assert_eq!(sh(&fs, "cat /work/f2.txt").await.stdout(), b"one\n");
        let _ = sh(&fs, "mv /work/f2.txt /work/f3.txt").await;
        assert!(!fs.exists("/work/f2.txt"));
        assert!(fs.exists("/work/f3.txt"));
        assert_eq!(sh(&fs, "wc -c /work/f1.txt").await.stdout(), b"4\n");
        assert_eq!(sh(&fs, "rm /work/f1.txt").await.exit_code, 0);
        assert_eq!(sh(&fs, "rm /work/f1.txt").await.exit_code, 1); // missing
        assert_eq!(sh(&fs, "rm -f /work/f1.txt").await.exit_code, 0); // forced ok
        assert_eq!(sh(&fs, "rm -r /work").await.exit_code, 0);
        assert!(!fs.exists("/work"));
    }

    #[tokio::test]
    async fn seq_streams_events() {
        let fs = test_fs();
        let out = sh(&fs, "seq 5").await;
        assert_eq!(out.events.len(), 5);
        assert!(out.events.iter().all(|e| e.delay_ms == 1));
        assert_eq!(out.stdout(), b"1\n2\n3\n4\n5\n");
    }

    #[tokio::test]
    async fn grep_and_head() {
        let fs = test_fs();
        let _ = sh(&fs, "echo alpha > /tmp/w.txt").await;
        let _ = sh(&fs, "echo beta >> /tmp/w.txt").await;
        let _ = sh(&fs, "echo gamma >> /tmp/w.txt").await;
        assert_eq!(sh(&fs, "grep amma /tmp/w.txt").await.stdout(), b"gamma\n");
        assert_eq!(sh(&fs, "grep zzz /tmp/w.txt").await.exit_code, 1);
        assert_eq!(
            sh(&fs, "head -n 2 /tmp/w.txt").await.stdout(),
            b"alpha\nbeta\n"
        );
    }

    #[tokio::test]
    async fn env_dumps_sorted() {
        let fs = test_fs();
        let out = sh(&fs, "env").await;
        let text = String::from_utf8(out.stdout()).unwrap();
        assert!(text.contains("USER=agent"));
        assert!(text.contains("HOME=/home/agent"));
        let pv = sh(&fs, "printenv USER").await;
        assert_eq!(pv.stdout(), b"agent\n");
    }

    #[tokio::test]
    async fn unknown_command_is_127() {
        let fs = test_fs();
        let out = sh(&fs, "frobnicate --now").await;
        assert_eq!(out.exit_code, 127);
        assert!(String::from_utf8(out.stderr())
            .unwrap()
            .contains("command not found"));
    }

    #[tokio::test]
    async fn sleep_measures_virtual_time() {
        let fs = test_fs();
        let t0 = std::time::Instant::now();
        let out = sh(&fs, "sleep 20").await;
        assert_eq!(out.exit_code, 0);
        assert!(t0.elapsed() >= std::time::Duration::from_millis(20));
    }

    #[tokio::test]
    async fn empty_command_is_noop() {
        let fs = test_fs();
        let out = sh(&fs, "   ").await;
        assert_eq!(out.exit_code, 0);
        assert!(out.events.is_empty());
    }
}
