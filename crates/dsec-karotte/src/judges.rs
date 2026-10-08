//! The judge hierarchy: regex, rubric (LLM-backed via a pluggable
//! completion client), executable scoring scripts, always-pass, and the
//! short-circuiting `And`/`Or` composites with the upstream Unicode tree
//! rendering.
//!
//! Ports of upstream `karotte/judges/` (`judge.py`, `regex_judge.py`,
//! `rubric_judge.py`, `executable_judge.py`, `always_pass_judge.py`,
//! `_composite_judge.py`, `rubric_context.py`).

use crate::error::{Error, Result, StudentMisbehaviorError};
use crate::schemas::{Scoring, Transcript};
use std::collections::BTreeMap;
use std::sync::Arc;

/// A judge scores a transcript (upstream `Judge(ABC).evaluate`).
pub trait Judge: Send + Sync {
    /// Evaluate the transcript.
    fn evaluate(&self, transcript: &Transcript) -> Result<Scoring>;

    /// The display name used in composite trees (upstream class name).
    fn type_name(&self) -> &'static str {
        "Judge"
    }

    /// Compose with AND (short-circuits on failure). Mirrors upstream
    /// `__and__`: a composite on the *left* flattens; the right side does
    /// not.
    fn and(self, other: Arc<dyn Judge>) -> Arc<dyn Judge>
    where
        Self: Sized + 'static,
    {
        let mut judges = self.flatten_same_mode(CompositeMode::And);
        judges.push(other);
        Arc::new(CompositeJudge::new(judges, CompositeMode::And))
    }

    /// Compose with OR (short-circuits on success).
    fn or(self, other: Arc<dyn Judge>) -> Arc<dyn Judge>
    where
        Self: Sized + 'static,
    {
        let mut judges = self.flatten_same_mode(CompositeMode::Or);
        judges.push(other);
        Arc::new(CompositeJudge::new(judges, CompositeMode::Or))
    }

    /// Flattening hook: a composite of `mode` yields its children;
    /// anything else yields just itself.
    fn flatten_same_mode(self, _mode: CompositeMode) -> Vec<Arc<dyn Judge>>
    where
        Self: Sized + 'static,
    {
        vec![Arc::new(self)]
    }

    /// Boxed flattening (works on `Arc<dyn Judge>`): `Some(children)`
    /// when this is a composite of `mode`.
    fn boxed_flatten(&self, _mode: CompositeMode) -> Option<Vec<Arc<dyn Judge>>> {
        None
    }
}

/// A judge that always passes (upstream `AlwaysPassJudge`).
pub struct AlwaysPassJudge;

impl Judge for AlwaysPassJudge {
    fn evaluate(&self, _transcript: &Transcript) -> Result<Scoring> {
        Ok(Scoring::pass(1.0))
    }

    fn type_name(&self) -> &'static str {
        "AlwaysPassJudge"
    }
}

/// A judge that always raises misbehavior (for tests and as the
/// classification target of the misbehavior path).
pub struct MisbehaviorJudge;

impl Judge for MisbehaviorJudge {
    fn evaluate(&self, _transcript: &Transcript) -> Result<Scoring> {
        Err(Error::Misbehavior(StudentMisbehaviorError::Other(
            "planted misbehavior".into(),
        )))
    }

    fn type_name(&self) -> &'static str {
        "MisbehaviorJudge"
    }
}

/// Regex judge (upstream `RegexJudge`): every pattern must
/// `re.search`-match the concatenation of the model's text messages
/// (joined with `\n\n`). One non-match → score 0.
pub struct RegexJudge {
    patterns: Vec<String>,
}

impl RegexJudge {
    /// Build from raw pattern strings.
    pub fn new(patterns: Vec<String>) -> Self {
        Self { patterns }
    }

    /// The concatenated model text (non-empty contents, `\n\n`-joined).
    fn model_text(transcript: &Transcript) -> String {
        let texts: Vec<String> = transcript
            .messages()
            .iter()
            .filter_map(|m| {
                if matches!(m.role, crate::schemas::Role::Assistant) {
                    m.content.as_ref().and_then(|c| c.as_text())
                } else {
                    None
                }
            })
            .filter(|t| !t.is_empty())
            .collect();
        texts.join("\n\n")
    }
}

impl Judge for RegexJudge {
    fn evaluate(&self, transcript: &Transcript) -> Result<Scoring> {
        let text = Self::model_text(transcript);
        let mut score = 1.0;
        let mut metadata = BTreeMap::new();
        for p in &self.patterns {
            // rust_regex::Regex is pulled in via a tiny hand-rolled subset?
            // No — upstream patterns are Python `re` syntax. We use the
            // crate-level `regex-lite`-compatible helper below.
            if !search(p, &text) {
                score = 0.0;
                metadata.insert(
                    p.clone(),
                    serde_json::Value::String("Transcript contains no match.".into()),
                );
            }
        }
        Ok(Scoring {
            score,
            metadata,
            continue_task: score == 1.0,
        })
    }

    fn type_name(&self) -> &'static str {
        "RegexJudge"
    }
}

/// Minimal regex engine dispatch: upstream judges carry Python `re`
/// patterns. The port supports the common `re` subset through a tiny
/// backtracking matcher (literals, `.`, `*`, `+`, `?`, `[]`, `^`, `$`,
/// `|`, `()`, `\d \w \s \b`, escapes). This avoids a heavy dependency
/// while keeping the judge usable on plain patterns.
fn search(pattern: &str, text: &str) -> bool {
    let re = match MiniRegex::compile(pattern) {
        Ok(re) => re,
        Err(_) => return text.contains(pattern), // fall back to literal
    };
    let chars: Vec<char> = text.chars().collect();
    for start in 0..=chars.len() {
        if re.match_at(&chars, start).is_some() {
            return true;
        }
    }
    false
}

// A tiny backtracking regex (subset of Python `re`) — enough for judge
// patterns like "answer.*42", "^KEY-", "[A-Z]{3}\\d+", "(yes|no)".
mod mini_regex {
    #[derive(Debug, Clone)]
    enum Node {
        Char(char),
        Any,
        Class {
            neg: bool,
            items: Vec<ClassItem>,
        },
        Start,
        End,
        WordBoundary,
        Group(Vec<Vec<Node>>), // alternation of sequences
        Repeat {
            node: Box<Node>,
            min: usize,
            max: usize,
            greedy: bool,
        },
    }
    #[derive(Debug, Clone)]
    enum ClassItem {
        Single(char),
        Range(char, char),
        Digit,
        NotDigit,
        Word,
        NotWord,
        Space,
        NotSpace,
    }

    pub struct MiniRegex {
        alts: Vec<Vec<Node>>,
    }

    fn is_word(c: char) -> bool {
        c.is_alphanumeric() || c == '_'
    }

    fn class_matches(item: &ClassItem, c: char) -> bool {
        match item {
            ClassItem::Single(x) => *x == c,
            ClassItem::Range(a, b) => (*a..=*b).contains(&c),
            ClassItem::Digit => c.is_ascii_digit(),
            ClassItem::NotDigit => !c.is_ascii_digit(),
            ClassItem::Word => is_word(c),
            ClassItem::NotWord => !is_word(c),
            ClassItem::Space => c.is_whitespace(),
            ClassItem::NotSpace => !c.is_whitespace(),
        }
    }

    struct Parser<'a> {
        chars: Vec<char>,
        pos: usize,
        _src: &'a str,
    }

    impl<'a> Parser<'a> {
        fn peek(&self) -> Option<char> {
            self.chars.get(self.pos).copied()
        }
        fn next(&mut self) -> Option<char> {
            let c = self.peek();
            if c.is_some() {
                self.pos += 1;
            }
            c
        }
        fn parse_alts(&mut self) -> Result<Vec<Vec<Node>>, String> {
            let mut alts = vec![self.parse_seq()?];
            while self.peek() == Some('|') {
                self.next();
                alts.push(self.parse_seq()?);
            }
            Ok(alts)
        }
        fn parse_seq(&mut self) -> Result<Vec<Node>, String> {
            let mut seq = Vec::new();
            loop {
                match self.peek() {
                    None | Some('|') | Some(')') => break,
                    _ => {}
                }
                let atom = self.parse_atom()?;
                let node = self.parse_quantifier(atom)?;
                seq.push(node);
            }
            Ok(seq)
        }
        fn parse_quantifier(&mut self, atom: Node) -> Result<Node, String> {
            let (min, max) = match self.peek() {
                Some('*') => {
                    self.next();
                    (0, usize::MAX)
                }
                Some('+') => {
                    self.next();
                    (1, usize::MAX)
                }
                Some('?') => {
                    self.next();
                    (0, 1)
                }
                Some('{') => {
                    // {m}, {m,}, {m,n}
                    let save = self.pos;
                    self.next();
                    let mut m = String::new();
                    while self.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                        m.push(self.next().unwrap());
                    }
                    if m.is_empty() {
                        self.pos = save;
                        return Ok(atom);
                    }
                    let min: usize = m.parse().map_err(|_| "bad count".to_string())?;
                    let max = match self.peek() {
                        Some('}') => min,
                        Some(',') => {
                            self.next();
                            let mut n = String::new();
                            while self.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                                n.push(self.next().unwrap());
                            }
                            if n.is_empty() {
                                usize::MAX
                            } else {
                                n.parse().map_err(|_| "bad count".to_string())?
                            }
                        }
                        _ => return Err("unterminated {".into()),
                    };
                    if self.next() != Some('}') {
                        return Err("expected }".into());
                    }
                    (min, max)
                }
                _ => return Ok(atom),
            };
            let greedy = if self.peek() == Some('?') {
                self.next();
                false
            } else {
                true
            };
            Ok(Node::Repeat {
                node: Box::new(atom),
                min,
                max,
                greedy,
            })
        }
        fn parse_atom(&mut self) -> Result<Node, String> {
            match self.next() {
                None => Err("unexpected end".into()),
                Some('.') => Ok(Node::Any),
                Some('^') => Ok(Node::Start),
                Some('$') => Ok(Node::End),
                Some('(') => {
                    // optional (?:...) — non-capturing, same semantics here
                    if self.peek() == Some('?') {
                        let save = self.pos;
                        self.next();
                        if self.next() == Some(':') {
                            // ok non-capturing
                        } else {
                            self.pos = save;
                        }
                    }
                    let alts = self.parse_alts()?;
                    if self.next() != Some(')') {
                        return Err("expected )".into());
                    }
                    Ok(Node::Group(alts))
                }
                Some('[') => {
                    let mut neg = false;
                    if self.peek() == Some('^') {
                        self.next();
                        neg = true;
                    }
                    let mut items = Vec::new();
                    let mut first = true;
                    loop {
                        let c = self.next().ok_or("unterminated [")?;
                        if c == ']' && !first {
                            break;
                        }
                        first = false;
                        let lo = if c == '\\' {
                            match self.next().ok_or("dangling backslash")? {
                                'd' => {
                                    items.push(ClassItem::Digit);
                                    continue;
                                }
                                'D' => {
                                    items.push(ClassItem::NotDigit);
                                    continue;
                                }
                                'w' => {
                                    items.push(ClassItem::Word);
                                    continue;
                                }
                                'W' => {
                                    items.push(ClassItem::NotWord);
                                    continue;
                                }
                                's' => {
                                    items.push(ClassItem::Space);
                                    continue;
                                }
                                'S' => {
                                    items.push(ClassItem::NotSpace);
                                    continue;
                                }
                                'n' => '\n',
                                't' => '\t',
                                'r' => '\r',
                                other => other,
                            }
                        } else {
                            c
                        };
                        if self.peek() == Some('-')
                            && self.chars.get(self.pos + 1).copied() != Some(']')
                        {
                            self.next();
                            let hi = self.next().ok_or("unterminated range")?;
                            items.push(ClassItem::Range(lo, hi));
                        } else {
                            items.push(ClassItem::Single(lo));
                        }
                    }
                    Ok(Node::Class { neg, items })
                }
                Some('\\') => match self.next().ok_or("dangling backslash")? {
                    'd' => Ok(Node::Class {
                        neg: false,
                        items: vec![ClassItem::Digit],
                    }),
                    'D' => Ok(Node::Class {
                        neg: true,
                        items: vec![ClassItem::Digit],
                    }),
                    'w' => Ok(Node::Class {
                        neg: false,
                        items: vec![ClassItem::Word],
                    }),
                    'W' => Ok(Node::Class {
                        neg: true,
                        items: vec![ClassItem::Word],
                    }),
                    's' => Ok(Node::Class {
                        neg: false,
                        items: vec![ClassItem::Space],
                    }),
                    'S' => Ok(Node::Class {
                        neg: true,
                        items: vec![ClassItem::Space],
                    }),
                    'b' => Ok(Node::WordBoundary),
                    'n' => Ok(Node::Char('\n')),
                    't' => Ok(Node::Char('\t')),
                    'r' => Ok(Node::Char('\r')),
                    other => Ok(Node::Char(other)),
                },
                Some(c) => Ok(Node::Char(c)),
            }
        }
    }

    impl MiniRegex {
        pub fn compile(pattern: &str) -> Result<Self, String> {
            let mut p = Parser {
                chars: pattern.chars().collect(),
                pos: 0,
                _src: pattern,
            };
            let alts = p.parse_alts()?;
            if p.pos != p.chars.len() {
                return Err("trailing characters".into());
            }
            Ok(Self { alts })
        }

        /// Try to match starting exactly at `start`; returns the end
        /// position of the leftmost-longest/greedy match, if any.
        pub fn match_at(&self, text: &[char], start: usize) -> Option<usize> {
            for alt in &self.alts {
                if let Some(end) = match_seq(alt, text, start, text.len()) {
                    return Some(end);
                }
            }
            None
        }
    }

    fn match_seq(seq: &[Node], text: &[char], pos: usize, boundary: usize) -> Option<usize> {
        if seq.is_empty() {
            return Some(pos);
        }
        match &seq[0] {
            Node::Repeat {
                node,
                min,
                max,
                greedy,
            } => {
                // gather greedy extent
                let mut ends = vec![pos];
                let mut p = pos;
                while ends.len() <= *max {
                    match match_node(node, text, p, boundary) {
                        Some(next) if next > p || (node_is_zero_width(node) && next == p) => {
                            if next == p {
                                // zero-width repetition stops
                                break;
                            }
                            p = next;
                            ends.push(p);
                        }
                        _ => break,
                    }
                }
                let count_avail = ends.len() - 1; // number of successful reps
                if count_avail < *min {
                    return None;
                }
                let reps_order: Vec<usize> = if *greedy {
                    (*min..=count_avail).rev().collect()
                } else {
                    (*min..=count_avail).collect()
                };
                for r in reps_order {
                    let after = ends[r];
                    if let Some(end) = match_seq(&seq[1..], text, after, boundary) {
                        return Some(end);
                    }
                }
                None
            }
            first => {
                let next = match_node(first, text, pos, boundary)?;
                match_seq(&seq[1..], text, next, boundary)
            }
        }
    }

    fn node_is_zero_width(_n: &Node) -> bool {
        false
    }

    fn match_node(node: &Node, text: &[char], pos: usize, boundary: usize) -> Option<usize> {
        match node {
            Node::Char(c) => {
                if text.get(pos) == Some(c) {
                    Some(pos + 1)
                } else {
                    None
                }
            }
            Node::Any => {
                if pos < boundary {
                    Some(pos + 1)
                } else {
                    None
                }
            }
            Node::Class { neg, items } => {
                let c = *text.get(pos)?;
                let inside = items.iter().any(|i| class_matches(i, c));
                if inside != *neg {
                    Some(pos + 1)
                } else {
                    None
                }
            }
            Node::Start => {
                if pos == 0 {
                    Some(pos)
                } else {
                    None
                }
            }
            Node::End => {
                if pos == boundary {
                    Some(pos)
                } else {
                    None
                }
            }
            Node::WordBoundary => {
                let before = pos > 0 && text.get(pos - 1).map(|c| is_word(*c)).unwrap_or(false);
                let at = text.get(pos).map(|c| is_word(*c)).unwrap_or(false);
                if before != at {
                    Some(pos)
                } else {
                    None
                }
            }
            Node::Group(alts) => {
                for alt in alts {
                    if let Some(end) = match_seq(alt, text, pos, boundary) {
                        return Some(end);
                    }
                }
                None
            }
            Node::Repeat { .. } => {
                // bare repeat handled by seq; treat as single match
                match_seq(std::slice::from_ref(node), text, pos, boundary)
            }
        }
    }
}

use mini_regex::MiniRegex;

// ---------------------------------------------------------------------------
// rubric contexts
// ---------------------------------------------------------------------------

/// Renders the text a rubric judge evaluates (upstream `rubric_context.py`).
pub trait RubricContext: Send + Sync {
    /// Render the evaluation context for this transcript.
    fn render(&self, transcript: &Transcript) -> Result<String>;
}

/// The student's submitted answers (upstream `AnswersContext`): keyed, or
/// all key/values joined.
pub struct AnswersContext {
    key: Option<String>,
}

impl AnswersContext {
    /// All answers.
    pub fn all() -> Self {
        Self { key: None }
    }
    /// One answer key.
    pub fn keyed(key: impl Into<String>) -> Self {
        Self {
            key: Some(key.into()),
        }
    }
}

impl RubricContext for AnswersContext {
    fn render(&self, transcript: &Transcript) -> Result<String> {
        let answers = transcript.answers();
        Ok(match &self.key {
            Some(k) => answers.get(k).cloned().unwrap_or_default(),
            None => answers
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect::<Vec<_>>()
                .join("\n\n"),
        })
    }
}

/// A file-content loader (production: the fs layer; tests: closures).
pub type FileLoader = Arc<dyn Fn(&str) -> Result<String> + Send + Sync>;

/// File contents (upstream `FileContext`): `# File '<path>':\n<content>`.
pub struct FileContext {
    paths: Vec<String>,
    loader: FileLoader,
}

impl FileContext {
    /// Read via a pluggable loader (tests use closures; production uses
    /// the fs layer).
    pub fn with_loader(
        paths: Vec<String>,
        loader: impl Fn(&str) -> Result<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            paths,
            loader: Arc::new(loader),
        }
    }
}

impl RubricContext for FileContext {
    fn render(&self, _transcript: &Transcript) -> Result<String> {
        let parts: Vec<String> = self
            .paths
            .iter()
            .map(|p| match (self.loader)(p) {
                Ok(content) => format!("# File '{p}':\n{content}"),
                Err(e) => format!("# File '{p}': Error reading file: {e}"),
            })
            .collect();
        Ok(parts.join("\n\n"))
    }
}

/// The conversation or tool exchanges (upstream `TranscriptContext`).
pub struct TranscriptContext {
    tool: Option<String>,
    last: Option<usize>,
}

impl TranscriptContext {
    /// Message-mode context (last N messages when set).
    pub fn messages(last: Option<usize>) -> Self {
        Self { tool: None, last }
    }
    /// Tool-mode context: one tool name, or `*` for all tools.
    pub fn tool(tool: impl Into<String>, last: Option<usize>) -> Self {
        Self {
            tool: Some(tool.into()),
            last,
        }
    }
}

impl RubricContext for TranscriptContext {
    fn render(&self, transcript: &Transcript) -> Result<String> {
        match &self.tool {
            None => {
                let mut lines: Vec<String> = transcript
                    .messages()
                    .iter()
                    .map(|m| {
                        let content = match &m.content {
                            Some(c) => c
                                .as_text()
                                .unwrap_or_else(|| serde_json::to_string(c).unwrap_or_default()),
                            None => String::new(),
                        };
                        format!("[{}]: {content}", role_str(m.role))
                    })
                    .collect();
                if let Some(n) = self.last {
                    let skip = lines.len().saturating_sub(n);
                    lines.drain(0..skip);
                }
                Ok(lines.join("\n"))
            }
            Some(filter) => {
                let exchanges = transcript.tool_exchange();
                let selected: Vec<_> = exchanges
                    .iter()
                    .filter(|(tc, _)| filter == "*" || tc.function.name.as_deref() == Some(filter))
                    .collect();
                let mut lines: Vec<String> = selected
                    .iter()
                    .map(|(tc, result)| {
                        let name = tc.function.name.clone().unwrap_or_default();
                        let args = tc.function.arguments.clone();
                        let joined = result
                            .as_ref()
                            .and_then(|r| r.first_text())
                            .unwrap_or_default();
                        format!("Tool: {name}\nArguments: {args}\nResult:\n{joined}")
                    })
                    .collect();
                if let Some(n) = self.last {
                    let skip = lines.len().saturating_sub(n);
                    lines.drain(0..skip);
                }
                Ok(lines.join("\n\n"))
            }
        }
    }
}

fn role_str(r: crate::schemas::Role) -> &'static str {
    match r {
        crate::schemas::Role::Assistant => "assistant",
        crate::schemas::Role::User => "user",
        crate::schemas::Role::System => "system",
        crate::schemas::Role::Tool => "tool",
        crate::schemas::Role::Function => "function",
    }
}

// ---------------------------------------------------------------------------
// rubric judge (LLM-backed, via a pluggable completion client)
// ---------------------------------------------------------------------------

/// One rubric criterion (upstream `RubricCriterion` TypedDict).
#[derive(Debug, Clone)]
pub struct RubricCriterion {
    /// The criterion text.
    pub criterion: String,
    /// Weight (coerced to f64).
    pub weight: f64,
}

impl RubricCriterion {
    /// A weighted criterion.
    pub fn new(criterion: impl Into<String>, weight: f64) -> Self {
        Self {
            criterion: criterion.into(),
            weight,
        }
    }
}

/// The pluggable LLM completion client the rubric judge questions
/// (upstream calls `litellm.completion`). Implementations wrap real
/// providers; tests return scripted replies.
pub trait CompletionClient: Send + Sync {
    /// One completion: `prompt` → reply text.
    fn complete(&self, prompt: &str) -> Result<String>;
}

/// The scripted completion client (upstream `fake_model.py` analogue for
/// the judge path).
pub struct ScriptedCompletions {
    replies: Vec<String>,
    calls: std::sync::Mutex<Vec<String>>,
}

impl ScriptedCompletions {
    /// Replay these replies in order.
    pub fn new(replies: Vec<String>) -> Self {
        Self {
            replies,
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }
    /// The prompts it received.
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

impl CompletionClient for ScriptedCompletions {
    fn complete(&self, prompt: &str) -> Result<String> {
        self.calls.lock().unwrap().push(prompt.to_string());
        let i = self.calls.lock().unwrap().len() - 1;
        self.replies
            .get(i)
            .cloned()
            .ok_or_else(|| Error::LoopLimit("Not enough messages defined for fake model.".into()))
    }
}

/// The rubric judge (upstream `RubricJudge`): one LLM call per criterion,
/// YES/NO reply, weighted sum, `continue_task = total > threshold`
/// (strictly greater).
pub struct RubricJudge {
    rubric: Vec<RubricCriterion>,
    client: Arc<dyn CompletionClient>,
    contexts: Vec<Box<dyn RubricContext>>,
    continue_threshold: f64,
}

impl RubricJudge {
    /// Build with default contexts (`AnswersContext::all()`).
    pub fn new(
        rubric: Vec<RubricCriterion>,
        client: Arc<dyn CompletionClient>,
        continue_threshold: f64,
    ) -> Self {
        Self {
            rubric,
            client,
            contexts: vec![Box::new(AnswersContext::all())],
            continue_threshold,
        }
    }

    /// Replace the evaluation contexts.
    pub fn with_contexts(mut self, contexts: Vec<Box<dyn RubricContext>>) -> Self {
        self.contexts = contexts;
        self
    }

    /// The upstream criterion prompt (dedented).
    pub fn criterion_prompt(context: &str, criterion: &str) -> String {
        format!(
            "You are evaluating whether the following content meets a specific criterion.\n\n\
Content to evaluate: {context}\n\n\
Criterion: {criterion}\n\n\
Does the content meet this criterion? Respond with ONLY \"YES\" or \"NO\", followed by a brief explanation on a new line.\n\n\
Format: YES\n[brief explanation]\n\nor NO\n[brief explanation]"
        )
    }

    /// Parse a criterion reply (upstream `parse_reply`): first line
    /// upper-cased, met iff it starts with YES; rest is the reasoning.
    pub fn parse_reply(content: &str) -> (bool, String) {
        let (first, rest) = match content.split_once('\n') {
            Some((f, r)) => (f, r.to_string()),
            None => (content, String::new()),
        };
        let decision = first.trim().to_uppercase();
        let met = decision.starts_with("YES");
        let reasoning = if rest.trim().is_empty() {
            "No explanation provided".to_string()
        } else {
            rest.trim().to_string()
        };
        (met, reasoning)
    }

    /// Render the joined contexts (upstream `render_context`).
    fn render_context(&self, transcript: &Transcript) -> Result<String> {
        let parts: Vec<String> = self
            .contexts
            .iter()
            .map(|c| c.render(transcript))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();
        Ok(parts.join("\n\n"))
    }
}

impl Judge for RubricJudge {
    fn evaluate(&self, transcript: &Transcript) -> Result<Scoring> {
        let context_text = self.render_context(transcript)?;
        if context_text.is_empty() {
            return Ok(Scoring {
                score: 0.0,
                metadata: BTreeMap::from([(
                    "error".to_string(),
                    serde_json::Value::String("No context available for evaluation".into()),
                )]),
                continue_task: false,
            });
        }
        let mut total = 0.0;
        let mut metadata = BTreeMap::new();
        for (i, item) in self.rubric.iter().enumerate() {
            let prompt = Self::criterion_prompt(&context_text, &item.criterion);
            let reply = self.client.complete(&prompt)?;
            let (met, reasoning) = Self::parse_reply(&reply);
            let key = format!("criterion_{}_{:.30}", i, item.criterion);
            let prefix = if met { "✓" } else { "✗" };
            metadata.insert(
                key,
                serde_json::Value::String(format!("{prefix}: {reasoning}")),
            );
            if met {
                total += item.weight;
            }
        }
        metadata.insert(
            "context".to_string(),
            serde_json::Value::String(context_text),
        );
        metadata.insert(
            "final_score".to_string(),
            serde_json::Value::String(total.to_string()),
        );
        metadata.insert(
            "pass_threshold".to_string(),
            serde_json::Value::String(format!("> {}", self.continue_threshold)),
        );
        Ok(Scoring {
            score: total,
            metadata,
            continue_task: total > self.continue_threshold,
        })
    }

    fn type_name(&self) -> &'static str {
        "RubricJudge"
    }
}

// ---------------------------------------------------------------------------
// executable judge
// ---------------------------------------------------------------------------

/// The executable judge contract (upstream `ExecutableJudge`): the last
/// element of the command is an *output-file path* the judge rewrites to
/// a scratch location; the program must write
/// `{"score": <num>, "metadata": <obj>}` there. Continue iff
/// `score >= threshold` (**greater-or-equal**, unlike the rubric judge).
pub struct ExecutableJudge {
    /// The command (argv) to run; last element = output path.
    pub args: Vec<String>,
    /// Continue threshold (upstream default -1).
    pub continue_threshold: f64,
}

impl ExecutableJudge {
    /// Build with the upstream default threshold (-1).
    pub fn new(args: Vec<String>) -> Self {
        Self {
            args,
            continue_threshold: -1.0,
        }
    }

    /// Rewrite the trailing output path into `scratch_dir`
    /// (upstream rewrites to a TemporaryDirectory).
    fn rewrite_output_path(&self, scratch_dir: &std::path::Path) -> Result<Vec<String>> {
        let mut args = self.args.clone();
        let last = args
            .last()
            .ok_or_else(|| Error::InvalidSpec("executable judge needs an output path".into()))?;
        let base = std::path::Path::new(last)
            .file_name()
            .ok_or_else(|| Error::InvalidSpec("output path has no file name".into()))?
            .to_string_lossy()
            .to_string();
        let out = scratch_dir.join(base);
        let s = out.to_string_lossy().to_string();
        *args.last_mut().unwrap() = s;
        Ok(args)
    }

    /// The mapping of exit/parse failures to a zero scoring
    /// (upstream wraps everything into metadata).
    fn zero(metadata: BTreeMap<String, serde_json::Value>) -> Scoring {
        Scoring {
            score: 0.0,
            metadata,
            continue_task: false,
        }
    }
}

impl Judge for ExecutableJudge {
    fn evaluate(&self, _transcript: &Transcript) -> Result<Scoring> {
        use std::process::Command;

        let scratch = tempfile_dir()?;
        let argv = self.rewrite_output_path(&scratch)?;
        let output = Command::new(&argv[0]).args(&argv[1..]).output();

        let output = match output {
            Ok(o) => o,
            Err(e) => {
                let md = BTreeMap::from([(
                    "FileNotFoundError".to_string(),
                    serde_json::Value::String(e.to_string()),
                )]);
                return Ok(Self::zero(md));
            }
        };
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if !output.status.success() {
            let md = BTreeMap::from([
                (
                    "CalledProcessError".to_string(),
                    serde_json::Value::String(format!("rc={}", output.status.code().unwrap_or(-1))),
                ),
                ("stdout".to_string(), serde_json::Value::String(stdout)),
                ("stderr".to_string(), serde_json::Value::String(stderr)),
            ]);
            return Ok(Self::zero(md));
        }

        // Find the rewritten output file (last arg).
        let out_path = std::path::Path::new(argv.last().unwrap());
        let read = std::fs::read(out_path).map_err(Error::Io);
        let bytes = match read {
            Ok(b) => b,
            Err(e) => {
                let md = BTreeMap::from([(
                    "json_read_error".to_string(),
                    serde_json::Value::String(e.to_string()),
                )]);
                return Ok(Self::zero(md));
            }
        };
        let parsed: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                let md = BTreeMap::from([(
                    "json_read_error".to_string(),
                    serde_json::Value::String(e.to_string()),
                )]);
                return Ok(Self::zero(md));
            }
        };
        let score = parsed.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
        let mut metadata = BTreeMap::new();
        if let Some(m) = parsed.get("metadata").and_then(|m| m.as_object()) {
            for (k, v) in m {
                // Upstream: {k: str(v)} — strings pass through unquoted.
                let text = match v {
                    serde_json::Value::String(sv) => sv.clone(),
                    other => serde_json::to_string(other).unwrap_or_default(),
                };
                metadata.insert(k.clone(), serde_json::Value::String(text));
            }
        }
        metadata.insert("stdout".to_string(), serde_json::Value::String(stdout));
        metadata.insert("stderr".to_string(), serde_json::Value::String(stderr));
        Ok(Scoring {
            score,
            metadata,
            continue_task: score >= self.continue_threshold,
        })
    }

    fn type_name(&self) -> &'static str {
        "ExecutableJudge"
    }
}

fn tempfile_dir() -> Result<std::path::PathBuf> {
    // dev-dependency tempfile is only for tests; use a std fallback:
    let base = std::env::temp_dir();
    let name = format!("karotte-judge-{}-{}", std::process::id(), nanos());
    let dir = base.join(name);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// composites
// ---------------------------------------------------------------------------

/// The composite short-circuit mode (upstream `_ShortCircuitMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeMode {
    /// AND: stop at the first failing judge.
    And,
    /// OR: stop at the first passing judge.
    Or,
}

impl CompositeMode {
    /// The display name (upstream `_display_name`).
    pub fn display_name(&self) -> &'static str {
        match self {
            CompositeMode::And => "And",
            CompositeMode::Or => "Or",
        }
    }
}

/// One judge's outcome inside a composite.
struct JudgeOutcome {
    display_type: String,
    evaluated: bool,
    score: f64,
    metadata: Option<BTreeMap<String, serde_json::Value>>,
    continue_task: bool,
}

/// Free-function AND composition on boxed judges (the trait methods need
/// `Sized` receivers; this works on `Arc<dyn Judge>`). A composite on the
/// left flattens, mirroring upstream `__and__`.
pub fn and_judges(left: Arc<dyn Judge>, right: Arc<dyn Judge>) -> Arc<dyn Judge> {
    let mut judges = left.boxed_flatten(CompositeMode::And).unwrap_or_default();
    if judges.is_empty() {
        judges.push(left);
    }
    judges.push(right);
    Arc::new(CompositeJudge::new(judges, CompositeMode::And))
}

/// Free-function OR composition on boxed judges (left composite flattens).
pub fn or_judges(left: Arc<dyn Judge>, right: Arc<dyn Judge>) -> Arc<dyn Judge> {
    let mut judges = left.boxed_flatten(CompositeMode::Or).unwrap_or_default();
    if judges.is_empty() {
        judges.push(left);
    }
    judges.push(right);
    Arc::new(CompositeJudge::new(judges, CompositeMode::Or))
}

/// A composite judge of judges (upstream `AndJudge`/`OrJudge`, both `@final`).
pub struct CompositeJudge {
    judges: Vec<Arc<dyn Judge>>,
    mode: CompositeMode,
}

impl CompositeJudge {
    /// A composite of `judges` in `mode` (upstream requires ≥ 1).
    pub fn new(judges: Vec<Arc<dyn Judge>>, mode: CompositeMode) -> Self {
        Self { judges, mode }
    }
}

impl Judge for CompositeJudge {
    fn evaluate(&self, transcript: &Transcript) -> Result<Scoring> {
        if self.judges.is_empty() {
            return Err(Error::InvalidSpec(
                "Composite judge requires at least one judge".into(),
            ));
        }
        let mut outcomes: Vec<JudgeOutcome> = Vec::new();
        let mut last_scoring: Option<Scoring> = None;
        for j in &self.judges {
            let name = self.type_name_of(j).to_string();
            {
                let s = j.evaluate(transcript)?;
                let stop = match self.mode {
                    CompositeMode::And => !s.continue_task,
                    CompositeMode::Or => s.continue_task,
                };
                let outcome = JudgeOutcome {
                    display_type: name,
                    evaluated: true,
                    score: s.score,
                    metadata: Some(s.metadata.clone()),
                    continue_task: s.continue_task,
                };
                let short = stop && self.judges.len() > 1;
                outcomes.push(outcome);
                last_scoring = Some(s);
                if short {
                    // mark the rest as not evaluated
                    for rest in &self.judges[outcomes.len()..] {
                        outcomes.push(JudgeOutcome {
                            display_type: self.type_name_of(rest).to_string(),
                            evaluated: false,
                            score: 0.0,
                            metadata: None,
                            continue_task: false,
                        });
                    }
                    break;
                }
            }
        }
        let last = last_scoring.expect("at least one judge");
        let score = last.score;
        let continue_task = last.continue_task;
        let tuples: Vec<JudgeTreeRow> = outcomes
            .iter()
            .map(|o| {
                (
                    o.display_type.as_str(),
                    o.evaluated,
                    o.score,
                    o.metadata.as_ref(),
                    o.continue_task,
                )
            })
            .collect();
        let tree = format_judges_tree(self.mode.display_name(), &tuples, score, continue_task);
        let mut metadata = BTreeMap::new();
        metadata.insert("judges".to_string(), serde_json::Value::String(tree));
        Ok(Scoring {
            score,
            metadata,
            continue_task,
        })
    }

    fn flatten_same_mode(self, mode: CompositeMode) -> Vec<Arc<dyn Judge>> {
        if mode == self.mode {
            self.judges
        } else {
            vec![Arc::new(self)]
        }
    }

    fn boxed_flatten(&self, mode: CompositeMode) -> Option<Vec<Arc<dyn Judge>>> {
        if mode == self.mode {
            Some(self.judges.clone())
        } else {
            None
        }
    }

    fn type_name(&self) -> &'static str {
        self.mode.display_name()
    }
}

impl CompositeJudge {
    fn type_name_of(&self, j: &Arc<dyn Judge>) -> &str {
        j.type_name()
    }
}

/// A named judge wrapper so composite trees read like upstream
/// (`RegexJudge`, `RubricJudge`, ...).
pub struct NamedJudge {
    name: &'static str,
    inner: Arc<dyn Judge>,
}

impl NamedJudge {
    /// Wrap `inner` with a display name, boxed.
    pub fn wrap(name: &'static str, inner: Arc<dyn Judge>) -> Arc<dyn Judge> {
        Arc::new(Self { name, inner })
    }
}

impl Judge for NamedJudge {
    fn evaluate(&self, transcript: &Transcript) -> Result<Scoring> {
        self.inner.evaluate(transcript)
    }

    fn type_name(&self) -> &'static str {
        self.name
    }
}

// The upstream tree glyphs and layout, byte-for-byte.
const GLYPH_OK: &str = "✔";
const GLYPH_FAIL: &str = "❗";
const GLYPH_SKIPPED: &str = "⊘";
const BRANCH_MIDDLE: &str = "├─ ";
const BRANCH_LAST: &str = "└─ ";
const CHILD_PREFIX_CONT: &str = "│  ";
const CHILD_PREFIX_STOP: &str = "   ";

/// Python-style float rendering (f-strings print `1.0`, not `1`) so the
/// tree strings match upstream byte-for-byte.
fn py_float(f: f64) -> String {
    let s = format!("{f}");
    if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") {
        s
    } else {
        format!("{s}.0")
    }
}

/// One row of the composite tree: `(type, evaluated, score, metadata,
/// continue)`.
pub type JudgeTreeRow<'a> = (
    &'a str,
    bool,
    f64,
    Option<&'a BTreeMap<String, serde_json::Value>>,
    bool,
);

/// Render the composite metadata tree (upstream `_format_judges_tree`):
/// root line, a `│` line, then `├─`/`└─` lines per judge with icons,
/// scores, and spliced child composites.
pub fn format_judges_tree(
    root_type: &str,
    outcomes: &[JudgeTreeRow],
    score: f64,
    continue_task: bool,
) -> String {
    let root_icon = if continue_task { GLYPH_OK } else { GLYPH_FAIL };
    let mut lines = vec![
        format!("{root_icon} {root_type} (score: {})", py_float(score)),
        "│".to_string(),
    ];
    let n = outcomes.len();
    for (i, (dtype, evaluated, j_score, metadata, j_continue)) in outcomes.iter().enumerate() {
        let branch = if i + 1 == n {
            BRANCH_LAST
        } else {
            BRANCH_MIDDLE
        };
        let line = if !evaluated {
            format!("{branch}{GLYPH_SKIPPED} {dtype} (not evaluated)")
        } else {
            let icon = if *j_continue { GLYPH_OK } else { GLYPH_FAIL };
            format!("{branch}{icon} {dtype} (score: {})", py_float(*j_score))
        };
        lines.push(line);
        // Splice child metadata (a dict carrying a "judges" tree string).
        if *evaluated {
            if let Some(md) = metadata {
                if let Some(serde_json::Value::String(child)) = md.get("judges") {
                    let child_prefix = if i + 1 == n {
                        CHILD_PREFIX_STOP
                    } else {
                        CHILD_PREFIX_CONT
                    };
                    for (li, l) in child.lines().enumerate() {
                        if li < 2 {
                            continue; // skip the child's root + "│" lines
                        }
                        lines.push(format!("{child_prefix}{l}"));
                    }
                } else {
                    let child_prefix = if i + 1 == n {
                        CHILD_PREFIX_STOP
                    } else {
                        CHILD_PREFIX_CONT
                    };
                    for (k, v) in md.iter() {
                        lines.push(format!("{child_prefix}{k}: {}", format_value(v)));
                    }
                }
            }
        }
    }
    lines.join("\n")
}

/// Upstream `_format_value`: null/true/false/str/json.
fn format_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "null".into(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemas::{Event, Message};

    fn transcript_with_assistant(text: &str) -> Transcript {
        Transcript {
            run_id: "r".into(),
            events: vec![Event::MessageAdded {
                message: Message::text(crate::schemas::Role::Assistant, text),
                finish_reason: Some("stop".into()),
                raw: None,
            }],
        }
    }

    #[test]
    fn always_pass() {
        let s = AlwaysPassJudge.evaluate(&Transcript::default()).unwrap();
        assert_eq!(s.score, 1.0);
        assert!(s.continue_task);
    }

    #[test]
    fn regex_judge_matches_and_misses() {
        let j = RegexJudge::new(vec!["answer.*42".into(), "^KEY-".into()]);
        let t = transcript_with_assistant("KEY-7F3A: the answer is 42");
        let s = j.evaluate(&t).unwrap();
        assert_eq!(s.score, 1.0);
        assert!(s.continue_task);

        let t2 = transcript_with_assistant("the answer is 41");
        let s2 = j.evaluate(&t2).unwrap();
        assert_eq!(s2.score, 0.0);
        assert!(!s2.continue_task);
        assert_eq!(
            s2.metadata["^KEY-"].as_str().unwrap(),
            "Transcript contains no match."
        );
    }

    #[test]
    fn mini_regex_covers_common_shapes() {
        assert!(search("42", "xx42yy"));
        assert!(!search("99", "xx42yy"));
        assert!(search("^KEY-", "KEY-7F3A"));
        assert!(!search("^KEY-", "the KEY-"));
        assert!(search("[A-Z]{3}-\\d+", "code ABC-123 here"));
        assert!(!search("[A-Z]{3}-\\d+", "code abc-123 here"));
        assert!(search("(yes|no)", "maybe no maybe"));
        assert!(search("\\bword\\b", "a word here"));
        assert!(!search("\\bword\\b", "sword"));
        assert!(search("a+", "baaa"));
        assert!(search("colou?r", "color and colour"));
        assert!(search("a.c", "abc"));
        assert!(!search("a\\dc", "abc"));
        assert!(search("a\\dc", "a1c"));
        assert!(search("\\d+", "abc 123"));
        // literal fallback for unsupported syntax is handled by compile
        // failure → contains()
        assert!(search("(?P<x>a)", "(?P<x>a)"));
    }

    #[test]
    fn rubric_parse_reply_variants() {
        let (met, why) = RubricJudge::parse_reply("YES\nit is there");
        assert!(met);
        assert_eq!(why, "it is there");
        let (met2, why2) = RubricJudge::parse_reply("no");
        assert!(!met2);
        assert_eq!(why2, "No explanation provided");
        let (met3, _) = RubricJudge::parse_reply("  yes  \nreason");
        assert!(met3);
    }

    #[test]
    fn rubric_judge_weighted_scoring_and_threshold() {
        let replies = vec![
            "YES\nmet one".to_string(),
            "NO\nmissed two".to_string(),
            "YES\nmet three".to_string(),
        ];
        let client: Arc<dyn CompletionClient> = Arc::new(ScriptedCompletions::new(replies));
        let rubric = vec![
            RubricCriterion::new("criterion one", 1.0),
            RubricCriterion::new("criterion two", 2.0),
            RubricCriterion::new("criterion three", 1.5),
        ];
        let j = RubricJudge::new(rubric, client, 0.0);
        let mut t = Transcript::default();
        t.events.push(Event::AnswersSubmitted {
            answers: BTreeMap::from([("answer".to_string(), "42".to_string())]),
        });
        let s = j.evaluate(&t).unwrap();
        assert_eq!(s.score, 2.5);
        assert!(s.continue_task, "2.5 > 0.0 strictly");
        assert!(s.metadata.contains_key("criterion_0_criterion one"));
        assert_eq!(s.metadata["final_score"].as_str().unwrap(), "2.5");
        assert_eq!(s.metadata["pass_threshold"].as_str().unwrap(), "> 0");
    }

    #[test]
    fn rubric_judge_empty_context_is_zero() {
        let client: Arc<dyn CompletionClient> = Arc::new(ScriptedCompletions::new(vec![]));
        let j = RubricJudge::new(vec![RubricCriterion::new("x", 1.0)], client, 0.0);
        let s = j.evaluate(&Transcript::default()).unwrap();
        assert_eq!(s.score, 0.0);
        assert!(!s.continue_task);
        assert_eq!(
            s.metadata["error"].as_str().unwrap(),
            "No context available for evaluation"
        );
    }

    #[test]
    fn composite_and_short_circuits_on_failure() {
        // ok + fail + never-reached → AND stops at the middle judge; the
        // third shows up as "not evaluated".
        let fail_rubric: Arc<dyn Judge> =
            Arc::new(RegexJudge::new(vec!["nonexistent-pattern-xyz".into()]));
        let ok: Arc<dyn Judge> = Arc::new(RegexJudge::new(vec!["model".into()]));
        let never: Arc<dyn Judge> = Arc::new(RegexJudge::new(vec!["never".into()]));
        let and: Arc<dyn Judge> = Arc::new(CompositeJudge::new(
            vec![ok, fail_rubric, never],
            CompositeMode::And,
        ));
        let t = transcript_with_assistant("model output");
        let s = and.evaluate(&t).unwrap();
        assert!(!s.continue_task);
        let tree = s.metadata["judges"].as_str().unwrap();
        assert!(tree.starts_with('❗'));
        assert!(tree.contains("(not evaluated)"));
    }

    #[test]
    fn and_or_helpers_flatten_left_composites() {
        let a = RegexJudge::new(vec!["model".into()]);
        let b = RegexJudge::new(vec!["output".into()]);
        let c = RegexJudge::new(vec!["third".into()]);
        // (a AND b) AND c flattens the left composite → 3 children.
        let ab: Arc<dyn Judge> = a.and(Arc::new(b));
        let abc = and_judges(ab, Arc::new(c));
        let t = transcript_with_assistant("model output third");
        let s = abc.evaluate(&t).unwrap();
        assert!(s.continue_task);
        let tree = s.metadata["judges"].as_str().unwrap();
        assert_eq!(tree.matches("RegexJudge").count(), 3, "tree: {tree}");

        // OR composite on the left of an AND does not flatten.
        let x = AlwaysPassJudge;
        let y = RegexJudge::new(vec!["nope-xyz".into()]);
        let xy: Arc<dyn Judge> = x.or(Arc::new(y));
        let xy_and = and_judges(xy, Arc::new(RegexJudge::new(vec!["model".into()])));
        let s2 = xy_and.evaluate(&t).unwrap();
        let tree2 = s2.metadata["judges"].as_str().unwrap();
        // The OR composite is one judge line with spliced children.
        assert!(tree2.contains("Or (score:"));
    }

    #[test]
    fn composite_or_short_circuits_on_success() {
        let ok: Arc<dyn Judge> = Arc::new(RegexJudge::new(vec!["model".into()]));
        let other: Arc<dyn Judge> = Arc::new(RegexJudge::new(vec!["zzz".into()]));
        let or: Arc<dyn Judge> = Arc::new(CompositeJudge::new(vec![ok, other], CompositeMode::Or));
        let t = transcript_with_assistant("model output");
        let s = or.evaluate(&t).unwrap();
        assert!(s.continue_task);
        assert_eq!(s.score, 1.0);
    }

    #[test]
    fn composite_tree_glyphs_exact() {
        let outcomes = vec![
            ("RegexJudge", true, 1.0, None, true),
            ("RubricJudge", true, 0.5, None, false),
            ("ExecutableJudge", false, 0.0, None, false),
        ];
        let tree = format_judges_tree("And", &outcomes, 0.5, false);
        assert!(tree.starts_with("❗ And (score: 0.5)"));
        assert!(tree.contains("│"));
        assert!(tree.contains("├─ ✔ RegexJudge (score: 1.0)"));
        assert!(tree.contains("├─ ❗ RubricJudge (score: 0.5)"));
        assert!(tree.contains("└─ ⊘ ExecutableJudge (not evaluated)"));
    }

    #[test]
    fn answers_context_keyed_and_all() {
        let mut t = Transcript::default();
        t.events.push(Event::AnswersSubmitted {
            answers: BTreeMap::from([
                ("a".to_string(), "1".to_string()),
                ("b".to_string(), "2".to_string()),
            ]),
        });
        assert_eq!(AnswersContext::keyed("a").render(&t).unwrap(), "1");
        assert_eq!(AnswersContext::all().render(&t).unwrap(), "a: 1\n\nb: 2");
    }

    #[test]
    fn transcript_context_renders_messages_and_tools() {
        let mut t = Transcript::default();
        t.events.push(Event::MessageAdded {
            message: Message::text(crate::schemas::Role::User, "instructions"),
            finish_reason: None,
            raw: None,
        });
        t.events.push(Event::MessageAdded {
            message: Message::text(crate::schemas::Role::Assistant, "working"),
            finish_reason: None,
            raw: None,
        });
        t.events.push(Event::ToolCallStarted {
            tool_call: crate::schemas::ToolCall::function("id1", "bash", r#"{"command":"ls"}"#),
        });
        t.events.push(Event::ToolCallCompleted {
            tool_call_id: "id1".into(),
            result: crate::schemas::CallToolResult::text("file-a"),
        });
        let msgs = TranscriptContext::messages(Some(1)).render(&t).unwrap();
        assert_eq!(msgs, "[assistant]: working");
        let tools = TranscriptContext::tool("*", None).render(&t).unwrap();
        assert!(tools.contains("Tool: bash"));
        assert!(tools.contains("file-a"));
        let only_bash = TranscriptContext::tool("bash", None).render(&t).unwrap();
        assert!(!only_bash.is_empty());
    }

    #[test]
    fn executable_judge_runs_and_parses() {
        // A shell script that writes score JSON to the last argument.
        let script = if std::path::Path::new("/bin/sh").exists() {
            "/bin/sh"
        } else {
            return; // non-posix host: skip
        };
        // sh -c convention: the first arg after the script is $0; the
        // output path (rewritten by the judge to the scratch dir) is $1.
        let args = vec![
            script.to_string(),
            "-c".to_string(),
            "echo '{\"score\": 0.75, \"metadata\": {\"k\": \"v\"}}' > \"$1\"".to_string(),
            "scoring".to_string(),
            "/tmp/out.json".to_string(),
        ];
        let j = ExecutableJudge::new(args);
        let s = j.evaluate(&Transcript::default()).unwrap();
        assert_eq!(s.score, 0.75);
        assert_eq!(s.metadata["k"].as_str().unwrap(), "v");
        assert!(s.continue_task, "0.75 >= -1");
        assert!(s.metadata.contains_key("stdout"));
    }

    #[test]
    fn executable_judge_failure_maps_to_zero() {
        let j = ExecutableJudge::new(vec!["/nonexistent/binary".to_string(), "out".to_string()]);
        let s = j.evaluate(&Transcript::default()).unwrap();
        assert_eq!(s.score, 0.0);
        assert!(s.metadata.contains_key("FileNotFoundError"));
    }
}
