//! TITO — Tool-Interleaved Trajectory Optimization records.
//!
//! The RL data contract for agentic rollouts (the QWENGYRE data path):
//! every token in a rollout transcript is classified by **origin** so the
//! loss only ever trains on policy-generated targets:
//!
//! - [`TokenOrigin::ModelGenerated`] — tokens the policy itself produced
//!   (assistant text **and** tool-call argument JSON). These are the only
//!   trainable targets.
//! - [`TokenOrigin::ToolResult`] — tokens that came back from the
//!   environment (tool observations). Never a target: the policy did not
//!   generate them, so gradient credit through them is meaningless.
//! - [`TokenOrigin::HarnessInjected`] — tokens the harness wrote into the
//!   context (the user task text). Never a target.
//! - [`TokenOrigin::SystemTemplate`] — the system prompt / chat-template
//!   scaffolding. Never a target.
//!
//! The record keeps tokens, origins, and the model-call turn index in
//! lockstep, and exposes the target mask used by the loss.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Classification of every token in an agentic rollout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenOrigin {
    /// Policy-generated token (assistant text / tool-call arguments) —
    /// a valid training target.
    ModelGenerated,
    /// Environment-returned token (tool observation) — masked out.
    ToolResult,
    /// Harness-written token (the task prompt) — masked out.
    HarnessInjected,
    /// System-prompt / chat-template scaffolding — masked out.
    SystemTemplate,
}

impl TokenOrigin {
    /// Whether tokens of this origin are trainable targets.
    pub fn is_target(&self) -> bool {
        matches!(self, TokenOrigin::ModelGenerated)
    }
}

/// One turn of a rollout transcript, in a backend-neutral shape (the
/// orchestrator maps its richer transcript types onto this).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TranscriptTurn {
    /// System / instruction template turn.
    System(String),
    /// Harness-injected user turn (the task text).
    User(String),
    /// Assistant turn: text plus tool calls the model emitted. The
    /// arguments JSON is model-generated (trainable) exactly like the
    /// text.
    Assistant {
        /// Free-text reply.
        text: String,
        /// Tool calls: `(call id, tool name, arguments JSON)`.
        tool_calls: Vec<(String, String, String)>,
    },
    /// A tool observation folded back as a user turn.
    ToolObservation {
        /// Call id the observation answers.
        call_id: String,
        /// Observation content.
        content: String,
    },
}

/// End-of-sequence token id (reserved, never a vocab word).
pub const EOS_ID: u32 = 0;
/// End-of-sequence token text.
pub const EOS_TEXT: &str = "<eos>";

/// Deterministic word-level tokenizer with a growable shared vocabulary.
///
/// Word-level is deliberately coarse: the TITO/loss contracts under test
/// are about masking and normalization, not linguistic quality. Ids are
/// assigned in first-seen order, so a fixed transcript order gives a
/// fixed vocabulary — reproducible across runs and serializable for the
/// trainer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimpleTokenizer {
    vocab: BTreeMap<String, u32>,
}

impl SimpleTokenizer {
    /// Empty tokenizer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct tokens seen (excluding the reserved EOS id 0).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len() + 1 // + EOS
    }

    /// Tokenizes `text`, growing the vocabulary as needed.
    pub fn tokenize(&mut self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            if word.is_empty() {
                continue;
            }
            let next = self.vocab.len() as u32 + 1;
            let id = *self.vocab.entry(word.to_string()).or_insert(next);
            ids.push(id);
        }
        ids
    }

    /// Tokenizes without growing the vocabulary; unknown words map to
    /// nothing (skipped), so output length may be shorter than input
    /// words.
    pub fn tokenize_known(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .filter_map(|w| self.vocab.get(w).copied())
            .collect()
    }

    /// Inverse of [`Self::tokenize`] (lossy: whitespace-normalized).
    pub fn detokenize(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|id| {
                if *id == EOS_ID {
                    Some(EOS_TEXT)
                } else {
                    self.vocab
                        .iter()
                        .find(|(_, v)| **v == *id)
                        .map(|(k, _)| k.as_str())
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The EOS token sequence.
    pub fn eos(&self) -> Vec<u32> {
        vec![EOS_ID]
    }
}

/// One execution's tokenized transcript with per-token origins.
///
/// `tokens`, `origins`, and `turns` are parallel arrays: `turns[i]` is
/// the model-call index (0 for everything before the first model call)
/// token `i` belongs to. "Execution" is one harness run — one rollout —
/// possibly containing many model calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TitoRecord {
    /// Execution (rollout) id this record belongs to.
    pub execution_id: u64,
    /// Token ids, in transcript order.
    pub tokens: Vec<u32>,
    /// Origin of each token, parallel to `tokens`.
    pub origins: Vec<TokenOrigin>,
    /// Model-call index of each token (shared prefixes reuse indices
    /// across executions of the same task).
    pub turn_idx: Vec<u32>,
}

impl TitoRecord {
    /// Builds a TITO record from transcript turns, tokenizing through
    /// `tok` (which grows — share one tokenizer across a whole group so
    /// ids line up).
    pub fn build(execution_id: u64, turns: &[TranscriptTurn], tok: &mut SimpleTokenizer) -> Self {
        let mut tokens = Vec::new();
        let mut origins = Vec::new();
        let mut turn_idx = Vec::new();
        let mut model_call = 0u32;
        let push = |ids: Vec<u32>,
                    o: TokenOrigin,
                    t: u32,
                    tokens: &mut Vec<u32>,
                    origins: &mut Vec<TokenOrigin>,
                    turn_idx: &mut Vec<u32>| {
            for id in ids {
                tokens.push(id);
                origins.push(o);
                turn_idx.push(t);
            }
        };

        for turn in turns {
            match turn {
                TranscriptTurn::System(s) => {
                    push(
                        tok.tokenize(s),
                        TokenOrigin::SystemTemplate,
                        0,
                        &mut tokens,
                        &mut origins,
                        &mut turn_idx,
                    );
                }
                TranscriptTurn::User(s) => {
                    push(
                        tok.tokenize(s),
                        TokenOrigin::HarnessInjected,
                        0,
                        &mut tokens,
                        &mut origins,
                        &mut turn_idx,
                    );
                }
                TranscriptTurn::Assistant { text, tool_calls } => {
                    model_call += 1;
                    let t = model_call - 1;
                    push(
                        tok.tokenize(text),
                        TokenOrigin::ModelGenerated,
                        t,
                        &mut tokens,
                        &mut origins,
                        &mut turn_idx,
                    );
                    for (_id, _name, args) in tool_calls {
                        push(
                            tok.tokenize(args),
                            TokenOrigin::ModelGenerated,
                            t,
                            &mut tokens,
                            &mut origins,
                            &mut turn_idx,
                        );
                    }
                }
                TranscriptTurn::ToolObservation { content, .. } => {
                    push(
                        tok.tokenize(content),
                        TokenOrigin::ToolResult,
                        model_call.saturating_sub(1),
                        &mut tokens,
                        &mut origins,
                        &mut turn_idx,
                    );
                }
            }
        }
        Self {
            execution_id,
            tokens,
            origins,
            turn_idx,
        }
    }

    /// Target mask, parallel to `tokens`: `true` where the token is
    /// policy-generated and therefore a training target.
    pub fn target_mask(&self) -> Vec<bool> {
        self.origins.iter().map(|o| o.is_target()).collect()
    }

    /// Positions of trainable target tokens.
    pub fn target_positions(&self) -> Vec<usize> {
        self.origins
            .iter()
            .enumerate()
            .filter(|(_, o)| o.is_target())
            .map(|(i, _)| i)
            .collect()
    }

    /// Number of trainable target tokens.
    pub fn target_token_count(&self) -> usize {
        self.origins.iter().filter(|o| o.is_target()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript() -> Vec<TranscriptTurn> {
        vec![
            TranscriptTurn::System("You are an agent.".into()),
            TranscriptTurn::User("Find the status of customer C1.".into()),
            TranscriptTurn::Assistant {
                text: "Checking the customer.".into(),
                tool_calls: vec![(
                    "call_1".into(),
                    "crm.get_customer".into(),
                    r#"{"id":"C1"}"#.into(),
                )],
            },
            TranscriptTurn::ToolObservation {
                call_id: "call_1".into(),
                content: r#"{"id":"C1","status":"active"}"#.into(),
            },
            TranscriptTurn::Assistant {
                text: "The customer is active.".into(),
                tool_calls: vec![],
            },
        ]
    }

    #[test]
    fn origins_and_masks() {
        let mut tok = SimpleTokenizer::new();
        let rec = TitoRecord::build(1, &transcript(), &mut tok);

        // system + user are never targets
        assert_eq!(rec.origins[0], TokenOrigin::SystemTemplate);
        let first_target = rec.target_positions()[0];
        assert!(first_target >= 2, "targets start after system+user");

        // tool result tokens are masked out
        let obs_start = rec
            .origins
            .iter()
            .position(|o| *o == TokenOrigin::ToolResult)
            .expect("tool result present");
        assert!(!rec.target_mask()[obs_start]);

        // assistant text AND tool-call args are targets
        assert!(rec.target_token_count() > 0);
        let assistant_tokens: Vec<usize> = rec
            .origins
            .iter()
            .enumerate()
            .filter(|(_, o)| o.is_target())
            .map(|(i, _)| i)
            .collect();
        // two assistant turns → two model calls with targets
        let turns_with_targets: std::collections::BTreeSet<u32> =
            assistant_tokens.iter().map(|i| rec.turn_idx[*i]).collect();
        assert_eq!(turns_with_targets.len(), 2);
    }

    #[test]
    fn tokenizer_determinism_and_roundtrip() {
        let mut a = SimpleTokenizer::new();
        let mut b = SimpleTokenizer::new();
        let text = "the quick brown fox the lazy dog";
        assert_eq!(a.tokenize(text), b.tokenize(text));
        // same word → same id
        let ids = a.tokenize(text);
        assert_eq!(ids[0], ids[4], "'the' maps to one id");
        let back = a.detokenize(&ids);
        assert_eq!(back, text);
        // EOS_TEXT is just a vocab word like any other (6 distinct words
        // before it -> id 7); the reserved EOS *id* 0 is never assigned
        assert_eq!(a.tokenize(EOS_TEXT)[0], 7);
        assert_eq!(a.eos(), vec![EOS_ID]);
    }

    #[test]
    fn tokenize_known_skips_unknown() {
        let mut tok = SimpleTokenizer::new();
        tok.tokenize("known words");
        let ids = tok.tokenize_known("known stranger words");
        assert_eq!(ids.len(), 2, "unknown words are dropped");
        assert_eq!(tok.tokenize_known("totally unknown"), Vec::<u32>::new());
    }
}
