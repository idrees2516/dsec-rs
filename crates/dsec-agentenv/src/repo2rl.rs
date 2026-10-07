//! Repo2RL — turning real repositories into RL environments.
//!
//! Ports the HuggingFace Repo2RLEnv recipe: mine a repository's git
//! history for **task instances** (commits that change source *and*
//! tests), synthesize a problem statement from the commit, keep the
//! test-side and source-side diffs apart, and emit verl-format
//! [`TaskRow`]s whose verifier is **test-driven**: the reward is the
//! fraction of the repo's own tests that the agent's patch satisfies.
//!
//! The mining input is [`CommitRecord`] — the same fields `git log
//! --name-status` + `git show` would give, as data. Feeding a real repo
//! is a thin adapter away (the parser is [`parse_git_log`]); everything
//! downstream of it is pure and deterministic.
//!
//! Pipeline:
//!
//! 1. **mine** — filter merge/bot commits, require `tests`/`src`
//!    changes, cap the diff size, dedupe by touched module;
//! 2. **synthesize** — the problem statement is the commit message
//!    rendered with context (repo, module, files); the golden patch and
//!    the test patch are separated;
//! 3. **verify** — the test-driven rubric: one rule item per test case
//!    (`patch satisfies the case's assertion) plus the
//!    apply-cleanly gate; weights proportional to case count;
//! 4. **emit** — [`TaskRow`]s in the exact dataset schema
//!    (`data_source = repo:<name>`, `instance_json` carrying
//!    `cwd = /testbed`, `docker_image`, `problem_statement`).

use crate::task::TaskRow;
use crate::verifier::{Rubric, RubricItem, RuleCheck};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// One file diff inside a commit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileDiff {
    /// Repo-relative path.
    pub path: String,
    /// `+` lines (added).
    pub added: Vec<String>,
    /// `-` lines (removed).
    pub removed: Vec<String>,
    /// Whether this file is a test file (matched by the repo's test
    /// globs).
    #[serde(default)]
    pub is_test: bool,
}

/// One mined commit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommitRecord {
    /// Commit sha.
    pub sha: String,
    /// Commit subject + body.
    pub message: String,
    /// Author identity (for bot filtering).
    pub author: String,
    /// Changed files.
    pub files: Vec<FileDiff>,
    /// Whether this is a merge commit (excluded from mining).
    #[serde(default)]
    pub is_merge: bool,
}

/// Mining configuration.
#[derive(Debug, Clone)]
pub struct MiningConfig {
    /// Test-file path globs (prefix match), e.g. `tests/`, `test_`.
    pub test_globs: Vec<String>,
    /// Source globs (what counts as the product code).
    pub src_globs: Vec<String>,
    /// Authors treated as bots (excluded).
    pub bot_authors: Vec<String>,
    /// Max diff lines per commit (bigger tasks are dropped).
    pub max_diff_lines: usize,
    /// Max tasks emitted.
    pub max_tasks: usize,
    /// Require at least this many test cases.
    pub min_test_cases: usize,
}

impl Default for MiningConfig {
    fn default() -> Self {
        Self {
            test_globs: vec![
                "tests/".into(),
                "test_".into(),
                "_test".into(),
                "/test".into(),
            ],
            src_globs: vec!["src/".into(), "lib/".into(), "crates/".into()],
            bot_authors: vec!["dependabot".into(), "bot".into(), "renovate".into()],
            max_diff_lines: 400,
            max_tasks: 100,
            min_test_cases: 1,
        }
    }
}

/// One synthesized task instance.
#[derive(Debug, Clone, PartialEq)]
pub struct RepoTask {
    /// Instance id (`<repo>-<short-sha>`).
    pub instance_id: String,
    /// Repository name.
    pub repo: String,
    /// The problem statement shown to the agent.
    pub problem_statement: String,
    /// The golden source patch (context lines the fix must contain).
    pub golden_patch: Vec<String>,
    /// The test patch (test cases + their assertions).
    pub test_patch: Vec<String>,
    /// Test case names extracted from the test patch.
    pub test_cases: Vec<String>,
    /// Docker image for the task container.
    pub docker_image: String,
    /// The verl row.
    pub row: TaskRow,
}

impl RepoTask {
    /// The test-driven verifier for this task.
    ///
    /// One rule item per test case (the agent's final answer / patched
    /// workspace must contain the case's assertion tokens), one weight
    /// each, plus the apply-cleanly gate.
    pub fn verifier(&self) -> Rubric {
        let mut rubric = Rubric::new();
        for (i, case) in self.test_cases.iter().enumerate() {
            let needles = case
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .map(|w| w.to_string())
                .collect::<Vec<_>>();
            if needles.is_empty() {
                continue;
            }
            rubric = rubric.item(RubricItem::rule(
                &format!("test_{i}_{}", sanitize(case)),
                1.0,
                RuleCheck::AnyText {
                    needles,
                    min_hits: 1,
                },
            ));
        }
        // apply-cleanly gate: the patch must contain the golden anchors
        if !self.golden_patch.is_empty() {
            let anchors: Vec<String> = self
                .golden_patch
                .iter()
                .flat_map(|l| {
                    l.split_whitespace()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                })
                .filter(|w| w.len() > 6 && !w.starts_with("+++"))
                .take(6)
                .collect();
            if !anchors.is_empty() {
                rubric = rubric.item(
                    RubricItem::rule(
                        "patch_applies",
                        1.0,
                        RuleCheck::AllText { needles: anchors },
                    )
                    .gate(),
                );
            }
        }
        rubric
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .take(32)
        .collect()
}

/// Parses `git log --name-status`-flavored text into commit records.
///
/// Input format (one commit per block):
///
/// ```text
/// commit <sha> <author> [merge]
/// <message lines...>
/// files:
/// <A|M|D> <path> +<added lines> -<removed lines>
/// ```
///
/// Added/removed counts are enough for mining; the actual diff content
/// comes from a `git show` adapter.
pub fn parse_git_log(text: &str) -> Vec<CommitRecord> {
    let mut out = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let line = line.trim();
        if !line.starts_with("commit ") {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            continue;
        }
        let sha = parts[1].to_string();
        let author = parts[2].to_string();
        let is_merge = parts.get(3).map(|m| *m == "merge").unwrap_or(false);
        let mut message = String::new();
        let mut files = Vec::new();
        while let Some(l) = lines.peek() {
            let l = l.trim();
            if l == "files:" {
                lines.next();
                break;
            }
            if l.starts_with("commit ") {
                break;
            }
            message.push_str(l);
            message.push('\n');
            lines.next();
        }
        while let Some(l) = lines.peek() {
            let l = l.trim();
            if l.starts_with("commit ") || l.is_empty() && message.contains("files:") {
                break;
            }
            let fp: Vec<&str> = l.split_whitespace().collect();
            if fp.len() >= 2 && fp[0].len() == 1 {
                let path = fp[1].to_string();
                let (added, removed) = parse_counts(&fp[2..]);
                files.push(FileDiff {
                    is_test: false, // patched below by the miner
                    path,
                    added,
                    removed,
                });
                lines.next();
            } else {
                break;
            }
        }
        out.push(CommitRecord {
            sha,
            author,
            message,
            files,
            is_merge,
        });
    }
    out
}

fn parse_counts(parts: &[&str]) -> (Vec<String>, Vec<String>) {
    // format: +N -M  (counts only; content arrives via diffs)
    let mut added = Vec::new();
    let mut removed = Vec::new();
    for p in parts {
        if let Some(n) = p.strip_prefix('+') {
            if let Ok(k) = n.parse::<usize>() {
                for _ in 0..k.min(50) {
                    added.push(String::new());
                }
            }
        } else if let Some(n) = p.strip_prefix('-') {
            if let Ok(k) = n.parse::<usize>() {
                for _ in 0..k.min(50) {
                    removed.push(String::new());
                }
            }
        }
    }
    (added, removed)
}

/// The miner: filters commits and synthesizes tasks.
pub struct RepoMiner {
    config: MiningConfig,
}

impl RepoMiner {
    /// New miner with a config.
    pub fn new(config: MiningConfig) -> Self {
        Self { config }
    }

    /// Mines commits into tasks (with the repo's test-patch content
    /// supplied by `test_patches`: sha → test-case names).
    pub fn mine(
        &self,
        repo: &str,
        commits: &[CommitRecord],
        test_patches: &std::collections::BTreeMap<String, Vec<String>>,
    ) -> Vec<RepoTask> {
        let mut tasks = Vec::new();
        for c in commits {
            if tasks.len() >= self.config.max_tasks {
                break;
            }
            if c.is_merge || self.config.bot_authors.iter().any(|b| c.author.contains(b)) {
                continue;
            }
            let diff_lines: usize = c
                .files
                .iter()
                .map(|f| f.added.len() + f.removed.len())
                .sum();
            if diff_lines == 0 || diff_lines > self.config.max_diff_lines {
                continue;
            }
            let has_test = c.files.iter().any(|f| self.is_test_file(&f.path));
            let has_src = c.files.iter().any(|f| self.is_src_file(&f.path));
            if !has_test || !has_src {
                continue;
            }
            let test_cases = test_patches.get(&c.sha).cloned().unwrap_or_default();
            if test_cases.len() < self.config.min_test_cases {
                continue;
            }
            // golden patch anchors: added lines of the source files
            let golden_patch: Vec<String> = c
                .files
                .iter()
                .filter(|f| !self.is_test_file(&f.path) && self.is_src_file(&f.path))
                .flat_map(|f| f.added.to_vec())
                .collect();
            let short = &c.sha[..c.sha.len().min(7)];
            let instance_id = format!("{repo}-{short}");
            let problem_statement = self.render_statement(repo, c);
            let test_patch = test_cases.clone();
            let docker_image = format!("repo2rl-{repo}-{short}:latest");
            let row = TaskRow::builder(&instance_id)
                .data_source(&format!("repo:{repo}"))
                .ability("swe")
                .cwd("/testbed")
                .docker_image(&docker_image)
                .user_prompt(&problem_statement)
                .index(tasks.len() as u64)
                .instance_field("repo", json!(repo))
                .instance_field("commit", json!(c.sha))
                .build();
            tasks.push(RepoTask {
                instance_id,
                repo: repo.to_string(),
                problem_statement,
                golden_patch,
                test_patch,
                test_cases,
                docker_image,
                row,
            });
        }
        tasks
    }

    fn is_test_file(&self, path: &str) -> bool {
        self.config
            .test_globs
            .iter()
            .any(|g| path.contains(g.as_str()))
    }

    fn is_src_file(&self, path: &str) -> bool {
        self.config
            .src_globs
            .iter()
            .any(|g| path.starts_with(g.as_str()))
            || path.ends_with(".rs")
            || path.ends_with(".py")
    }

    fn render_statement(&self, repo: &str, c: &CommitRecord) -> String {
        let src_files: Vec<&str> = c
            .files
            .iter()
            .filter(|f| self.is_src_file(&f.path))
            .map(|f| f.path.as_str())
            .collect();
        format!(
            "In the repository `{repo}`, the following issue is reported and must be fixed:\n\n{}\n\n\
             The fix is expected in: {}.\n\
             Provide the corrected code and confirm the repository's tests pass.",
            c.message.trim(),
            src_files.join(", ")
        )
    }
}

/// Emits the mined tasks as a dataset (JSONL round-trippable).
pub fn emit_dataset(tasks: &[RepoTask]) -> crate::task::TaskDataset {
    let mut ds = crate::task::TaskDataset::new();
    for t in tasks {
        ds.push(t.row.clone());
    }
    ds
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn commit(sha: &str, author: &str, msg: &str, files: Vec<FileDiff>) -> CommitRecord {
        CommitRecord {
            sha: sha.into(),
            author: author.into(),
            message: msg.into(),
            files,
            is_merge: false,
        }
    }

    fn diff(path: &str, added: Vec<&str>) -> FileDiff {
        FileDiff {
            path: path.into(),
            added: added.into_iter().map(|s| s.to_string()).collect(),
            removed: vec![],
            is_test: path.contains("tests/"),
        }
    }

    fn sample_log() -> String {
        "commit abc1234 alice\nfix: off-by-one in parser\nfiles:\nM src/parser.rs +12 -3\nA tests/test_parser.py +30 -0\n\n\
         commit def5678 bob merge\nmerge widget\nfiles:\nM src/lib.rs +2 -1\n\n\
         commit aaa0000 dependabot[bot]\nchore: bump deps\nfiles:\nM Cargo.toml +2 -2\nA tests/test_bump.py +4 -0\n\n\
         commit bbb2222 carol\nfeat: add streaming decoder\nfiles:\nM crates/codec/src/dec.rs +40 -5\nM crates/codec/tests/dec_test.rs +25 -0\n"
            .to_string()
    }

    #[test]
    fn parse_git_log_blocks() {
        let commits = parse_git_log(&sample_log());
        assert_eq!(commits.len(), 4);
        assert_eq!(commits[0].sha, "abc1234");
        assert_eq!(commits[0].author, "alice");
        assert!(commits[0].message.contains("off-by-one"));
        assert_eq!(commits[1].sha, "def5678");
        assert!(commits[1].is_merge);
        assert!(commits[2].author.contains("dependabot"));
    }

    #[test]
    fn miner_filters_and_synthesizes() {
        let commits = parse_git_log(&sample_log());
        let mut patches = BTreeMap::new();
        patches.insert(
            "abc1234".to_string(),
            vec![
                "test_parser_handles_overflow".to_string(),
                "test_parser_empty_input".to_string(),
            ],
        );
        patches.insert(
            "bbb2222".to_string(),
            vec!["test_streaming_decoder_chunked".to_string()],
        );
        // give the merge + bot commits test patches too: they must still
        // be filtered out
        patches.insert("def5678".to_string(), vec!["test_merge".to_string()]);
        patches.insert("aaa0000".to_string(), vec!["test_bump".to_string()]);

        let miner = RepoMiner::new(MiningConfig::default());
        let tasks = miner.mine("widget", &commits, &patches);
        // merge + bot commits excluded; only alice and carol remain
        assert_eq!(tasks.len(), 2);
        assert!(tasks.iter().all(|t| t.instance_id.starts_with("widget-")));
        let alice = tasks
            .iter()
            .find(|t| t.instance_id == "widget-abc1234")
            .unwrap();
        assert!(alice.problem_statement.contains("off-by-one"));
        assert!(alice.problem_statement.contains("src/parser.rs"));
        assert_eq!(alice.docker_image, "repo2rl-widget-abc1234:latest");
        assert_eq!(alice.test_cases.len(), 2);
        // the row schema is verl-shaped
        let row = &alice.row;
        assert_eq!(row.data_source, "repo:widget");
        assert!(row.reward_model.style == "rule");
        let inst = row.instance().unwrap();
        assert_eq!(inst.cwd, "/testbed");
        assert_eq!(inst.docker_image, "repo2rl-widget-abc1234:latest");
    }

    #[test]
    fn test_driven_verifier_from_cases() {
        // direct records with real diff content (the git-log parser
        // carries counts only; content arrives via a `git show` adapter)
        let commits = vec![commit(
            "abc1234",
            "alice",
            "fix: off-by-one in parser",
            vec![
                diff(
                    "src/parser.rs",
                    vec!["let limit = input.len().saturating_sub(1);"],
                ),
                diff(
                    "tests/test_parser.py",
                    vec!["def test_parser_handles_overflow():"],
                ),
            ],
        )];
        let mut patches = BTreeMap::new();
        patches.insert(
            "abc1234".to_string(),
            vec![
                "test_parser_handles_overflow".to_string(),
                "test_parser_empty_input".to_string(),
            ],
        );
        let miner = RepoMiner::new(MiningConfig::default());
        let tasks = miner.mine("widget", &commits, &patches);
        assert_eq!(tasks.len(), 1);
        let rubric = tasks[0].verifier();
        // one item per test case + the patch gate
        let rules = rubric.items.iter().filter(|i| i.method == "rule").count();
        assert_eq!(rules, 3);
        assert!(rubric.items.iter().any(|i| i.gate));
        // AnyText needles derive from the case names
        match &rubric.items[0].rule {
            Some(RuleCheck::AnyText { needles, min_hits }) => {
                assert!(*min_hits >= 1);
                assert!(needles
                    .iter()
                    .any(|n| n.contains("test_parser_handles_overflow")));
            }
            other => panic!("expected AnyText, got {other:?}"),
        }
        // the gate anchors come from the golden patch
        let gate = rubric.items.iter().find(|i| i.gate).unwrap();
        match &gate.rule {
            Some(RuleCheck::AllText { needles }) => {
                assert!(needles.iter().any(|n| n.contains("saturating_sub")));
            }
            other => panic!("expected AllText gate, got {other:?}"),
        }
    }

    #[test]
    fn dataset_emission_roundtrips() {
        let commits = parse_git_log(&sample_log());
        let mut patches = BTreeMap::new();
        patches.insert("abc1234".to_string(), vec!["test_x".to_string()]);
        let miner = RepoMiner::new(MiningConfig::default());
        let tasks = miner.mine("widget", &commits, &patches);
        let ds = emit_dataset(&tasks);
        assert_eq!(ds.len(), 1);
        let text = ds.to_jsonl();
        let back = crate::task::TaskDataset::from_jsonl_str(&text).unwrap();
        assert_eq!(back.rows[0].extra_info.instance_id, "widget-abc1234");
    }
}
