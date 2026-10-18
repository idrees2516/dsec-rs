//! Execution-level loss normalization with shared-target accounting
//! (QWENGYRE's loss contract, operationalized).
//!
//! The formula this module implements, per task group `g`:
//!
//! 1. Executions are inserted into a trajectory tree; target tokens
//!    sitting on **shared nodes** collapse to a single loss term
//!    ("shared-prefix targets contribute once"). The term's coefficient
//!    is the merged weight of every member execution, and its advantage
//!    is the mean of the member advantages (identical conditioning
//!    context by tree construction).
//! 2. Each **valid** execution `e` carries total weight `1/|E_g|`,
//!    spread over its own target tokens: each of its target tokens
//!    weighs `1/(|E_g| · T_e)` where `T_e` is the execution's target
//!    token count. "Averages token losses within each execution" — an
//!    execution with more tool turns does **not** get more total weight.
//! 3. Evaluator failures are excluded entirely: no score, no advantage,
//!    no terms, no weight — never converted to reward zero.
//!
//! The final scalar loss is `-Σ term.coefficient · term.advantage ·
//! logπ(term.target | term.context)`; terms carry everything the caller
//! needs to evaluate `logπ` and its gradient (the tiny trainer in
//! `dsec-train` consumes exactly this).

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::reward::{group_advantage, AdvantageConfig, RewardRecord};
use crate::tito::{SimpleTokenizer, TranscriptTurn};
use crate::trajectory::{NodeId, TrajectoryTree};

/// One execution fed to the loss.
#[derive(Debug, Clone)]
pub struct ExecutionInput {
    /// Execution id.
    pub execution_id: u64,
    /// Group id (advantage baseline scope).
    pub group_id: u64,
    /// The full transcript turns of the rollout.
    pub turns: Vec<TranscriptTurn>,
    /// The rollout's reward with validity.
    pub reward: RewardRecord,
}

/// One trainable term: a single (context, target-token) pair with its
/// merged coefficient and node advantage.
#[derive(Debug, Clone)]
pub struct LossTerm {
    /// A member execution (representative id, for tracing).
    pub execution_id: u64,
    /// Tree node the target token lives in.
    pub node_id: NodeId,
    /// The target token (policy-generated only, by TITO masks).
    pub target_token: u32,
    /// Conditioning context: all tokens preceding the target in the
    /// execution (identical across the node's members by construction).
    pub context_tokens: Vec<u32>,
    /// Merged weight: `Σ_{e ∈ members} 1/(|E_g| · T_e)`.
    pub coefficient: f32,
    /// Mean advantage of the node's member executions.
    pub advantage: f32,
}

/// Output of [`compute_loss_terms`].
#[derive(Debug, Clone, Default)]
pub struct LossOutput {
    /// Trainable terms (unique target tokens, shared ones merged).
    pub terms: Vec<LossTerm>,
    /// Deduplicated target-token count (what training "sees").
    pub unique_target_count: usize,
    /// Total coefficient mass per execution (each valid execution sums
    /// to exactly `1/|E_g|` — the execution-level normalization).
    pub per_execution_weight: BTreeMap<u64, f32>,
    /// Executions excluded from training (evaluator failures, no
    /// baseline groups).
    pub excluded: BTreeSet<u64>,
}

/// Computes the loss terms for a set of executions.
///
/// Executions are grouped by `group_id`; per group, advantages come
/// from [`group_advantage`]. Groups whose valid members carry no
/// advantage (fewer than two scorable members) contribute nothing and
/// are listed in `excluded`.
///
/// Token ids come from a fresh local tokenizer — see
/// [`compute_loss_terms_with`] to use a shared one (the trainer does,
/// so its model's vocabulary lines up with the term ids).
pub fn compute_loss_terms(
    executions: &[ExecutionInput],
    adv_config: &AdvantageConfig,
) -> LossOutput {
    let mut tok = SimpleTokenizer::new();
    compute_loss_terms_with(executions, adv_config, &mut tok)
}

/// [`compute_loss_terms`] over a caller-owned tokenizer: the shared
/// vocabulary makes the produced token ids meaningful to the caller's
/// model.
pub fn compute_loss_terms_with(
    executions: &[ExecutionInput],
    adv_config: &AdvantageConfig,
    tok: &mut SimpleTokenizer,
) -> LossOutput {
    // grow the shared vocabulary over every turn (first-seen ids)
    for exec in executions {
        for turn in &exec.turns {
            match turn {
                TranscriptTurn::System(s) | TranscriptTurn::User(s) => {
                    tok.tokenize(s);
                }
                TranscriptTurn::Assistant { text, tool_calls } => {
                    tok.tokenize(text);
                    for (_c, _n, a) in tool_calls {
                        tok.tokenize(a);
                    }
                }
                TranscriptTurn::ToolObservation { content, .. } => {
                    tok.tokenize(content);
                }
            }
        }
    }

    // 1. group advantages
    let rewards: Vec<RewardRecord> = executions.iter().map(|e| e.reward).collect();
    let advantages = group_advantage(&rewards, adv_config);

    // 2. group partitions
    let mut groups: BTreeMap<u64, Vec<&ExecutionInput>> = BTreeMap::new();
    for e in executions {
        groups.entry(e.group_id).or_default().push(e);
    }

    let mut out = LossOutput::default();

    for members in groups.values() {
        // valid members with a usable advantage drive training
        let trainable: Vec<&&ExecutionInput> = members
            .iter()
            .filter(|e| {
                e.reward.validity.trainable()
                    && advantages.get(&e.execution_id).is_some_and(|a| a.is_some())
            })
            .collect();
        if trainable.is_empty() {
            for e in members {
                out.excluded.insert(e.execution_id);
            }
            continue;
        }
        let valid_count = trainable.len() as f32;

        // 3. trajectory tree over ALL members (failed ones too, so
        //    shared-node membership is faithful); terms only come from
        //    trainable members' weights.
        let mut tree = TrajectoryTree::new();
        for e in members {
            tree.insert_execution(e.execution_id, &e.turns, tok);
        }

        // 4. per-execution target token counts (tokens on its path)
        let mut t_e: BTreeMap<u64, usize> = BTreeMap::new();
        for e in &trainable {
            let path = tree.path(e.execution_id).expect("inserted");
            let count: usize = path
                .iter()
                .map(|&n| {
                    tree.node(n)
                        .origins
                        .iter()
                        .filter(|o| o.is_target())
                        .count()
                })
                .sum();
            t_e.insert(e.execution_id, count.max(1));
            // weight budget: 1/|E_g| regardless of token count
            out.per_execution_weight
                .insert(e.execution_id, 1.0 / valid_count);
        }

        // 5. emit one term per unique (node, target position)
        for node_id in tree.target_nodes() {
            let node = tree.node(node_id);
            let positions = node.target_positions();
            if positions.is_empty() {
                continue;
            }
            // node members that are trainable with advantages
            let adv_members: Vec<(u64, f32)> = node
                .execs
                .iter()
                .filter_map(|eid| advantages.get(eid).and_then(|a| *a).map(|a| (*eid, a)))
                .collect();
            if adv_members.is_empty() {
                continue; // node only reachable through failed executions
            }
            let node_advantage: f32 =
                adv_members.iter().map(|(_, a)| a).sum::<f32>() / adv_members.len() as f32;

            // merged coefficient = Σ over trainable members on this node
            let coefficient: f32 = adv_members
                .iter()
                .map(|(eid, _)| {
                    let tokens = t_e.get(eid).copied().unwrap_or(1) as f32;
                    1.0 / (valid_count * tokens)
                })
                .sum();

            // conditioning context: tokens of the ancestors plus this
            // node's tokens before the target position
            let exec_rep = adv_members[0].0;
            let path = tree.path(exec_rep).expect("member path exists");
            let idx_in_path = path
                .iter()
                .position(|&n| n == node_id)
                .expect("node on member path");
            for &pos in &positions {
                let mut context = Vec::new();
                for &ancestor in path.iter().take(idx_in_path) {
                    context.extend_from_slice(&tree.node(ancestor).tokens);
                }
                context.extend_from_slice(&node.tokens[..pos]);
                out.terms.push(LossTerm {
                    execution_id: exec_rep,
                    node_id,
                    target_token: node.tokens[pos],
                    context_tokens: context,
                    coefficient,
                    advantage: node_advantage,
                });
            }
        }

        // excluded: members that were valid rewards but no baseline, or
        // evaluator failures
        for e in members {
            if !trainable.iter().any(|t| t.execution_id == e.execution_id) {
                out.excluded.insert(e.execution_id);
            }
        }
    }

    out.unique_target_count = out.terms.len();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reward::RewardValidity;

    fn exec(id: u64, group: u64, suffix: &str, reward: f32) -> ExecutionInput {
        ExecutionInput {
            execution_id: id,
            group_id: group,
            turns: vec![
                TranscriptTurn::System("sys words here".into()),
                TranscriptTurn::User("task words here".into()),
                TranscriptTurn::Assistant {
                    text: format!("answer words {suffix}"),
                    tool_calls: vec![],
                },
            ],
            reward: RewardRecord::valid(id, group, reward),
        }
    }

    #[test]
    fn shared_prefix_targets_contribute_once() {
        // 4 executions: three share the same 3-word target, one differs
        let execs = vec![
            exec(1, 1, "alpha", 1.0),
            exec(2, 1, "alpha", 0.0),
            exec(3, 1, "alpha", 0.5),
            exec(4, 1, "beta", 0.0),
        ];
        let out = compute_loss_terms(&execs, &AdvantageConfig::default());
        // unique targets: "answer words alpha" (3) + "answer words beta" (3)
        assert_eq!(out.unique_target_count, 6);
        // naive would be 3*3 + 3 = 12
        assert_eq!(out.terms.len(), 6);
        // each term appears exactly once per token (no duplicates by
        // (node, position))
        let mut seen = std::collections::BTreeSet::new();
        for t in &out.terms {
            assert!(seen.insert((t.node_id, t.target_token)));
        }
    }

    #[test]
    fn execution_level_normalization_equal_mass() {
        // one execution with 3 target tokens, one with 9 — each still
        // carries exactly 1/2 of the total weight
        let long = ExecutionInput {
            execution_id: 1,
            group_id: 1,
            turns: vec![
                TranscriptTurn::System("sys".into()),
                TranscriptTurn::User("task".into()),
                TranscriptTurn::Assistant {
                    text: "a b c d e f g h i".into(), // 9 target words
                    tool_calls: vec![],
                },
            ],
            reward: RewardRecord::valid(1, 1, 1.0),
        };
        let short = ExecutionInput {
            execution_id: 2,
            group_id: 1,
            turns: vec![
                TranscriptTurn::System("sys".into()),
                TranscriptTurn::User("task".into()),
                TranscriptTurn::Assistant {
                    text: "x y z".into(), // 3 target words
                    tool_calls: vec![],
                },
            ],
            reward: RewardRecord::valid(2, 1, 0.0),
        };
        let out = compute_loss_terms(&[long, short], &AdvantageConfig::default());
        let w1 = out.per_execution_weight[&1];
        let w2 = out.per_execution_weight[&2];
        assert!(
            (w1 - 0.5).abs() < 1e-6,
            "each execution weighs 1/2, got {w1}"
        );
        assert!((w2 - 0.5).abs() < 1e-6);
        // the 9-token execution's exclusive terms each weigh 1/(2*9):
        // averaging inside the execution — more tokens, not more mass
        let e1_coeff: Vec<f32> = out
            .terms
            .iter()
            .filter(|t| t.execution_id == 1)
            .map(|t| t.coefficient)
            .collect();
        assert!(!e1_coeff.is_empty());
        for c in e1_coeff {
            assert!((c - 1.0 / 18.0).abs() < 1e-6, "per-token weight {c}");
        }
        let e2_coeff: Vec<f32> = out
            .terms
            .iter()
            .filter(|t| t.execution_id == 2)
            .map(|t| t.coefficient)
            .collect();
        for c in e2_coeff {
            assert!((c - 1.0 / 6.0).abs() < 1e-6, "per-token weight {c}");
        }
        // total mass = 1.0 across both executions
        let total: f32 = out.terms.iter().map(|t| t.coefficient).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn evaluator_failure_produces_no_terms_and_no_weight() {
        let execs = vec![
            exec(1, 1, "aa", 1.0),
            exec(2, 1, "bb", 0.0),
            ExecutionInput {
                execution_id: 3,
                group_id: 1,
                turns: vec![
                    TranscriptTurn::System("sys words here".into()),
                    TranscriptTurn::User("task words here".into()),
                    TranscriptTurn::Assistant {
                        text: "answer words cc".into(),
                        tool_calls: vec![],
                    },
                ],
                reward: RewardRecord::evaluator_failure(3, 1),
            },
        ];
        let out = compute_loss_terms(&execs, &AdvantageConfig::default());
        assert!(out.excluded.contains(&3));
        assert!(!out.per_execution_weight.contains_key(&3));
        // no term references execution 3 as its member
        assert!(out.terms.iter().all(|t| t.execution_id != 3));
        // the failed execution's reward was NOT treated as 0: the two
        // valid members still see a 2-member baseline (advantages ±1)
        let advantages: Vec<f32> = out.terms.iter().map(|t| t.advantage).collect();
        assert!(advantages.iter().any(|a| *a > 0.9));
        assert!(advantages.iter().any(|a| *a < -0.9));
    }

    #[test]
    fn tool_and_harness_tokens_never_targets() {
        let e = ExecutionInput {
            execution_id: 1,
            group_id: 1,
            turns: vec![
                TranscriptTurn::System("sys template words".into()),
                TranscriptTurn::User("the task words".into()),
                TranscriptTurn::Assistant {
                    text: "calling now".into(),
                    tool_calls: vec![("c1".into(), "crm.get".into(), r#"{"id":"C1"}"#.into())],
                },
                TranscriptTurn::ToolObservation {
                    call_id: "c1".into(),
                    content: r#"{"status":"active"}"#.into(),
                },
                TranscriptTurn::Assistant {
                    text: "it is active".into(),
                    tool_calls: vec![],
                },
            ],
            reward: RewardRecord::valid(1, 1, 1.0),
        };
        let turns = e.turns.clone();
        let e2 = ExecutionInput {
            execution_id: 2,
            group_id: 1,
            turns: e.turns.clone(),
            reward: RewardRecord::valid(2, 1, 0.0),
        };
        let out = compute_loss_terms(&[e, e2], &AdvantageConfig::default());
        // contexts may contain tool/harness tokens (conditioning), but
        // targets must only be model-generated words
        let targets: std::collections::BTreeSet<u32> =
            out.terms.iter().map(|t| t.target_token).collect();
        assert!(!targets.is_empty());
        // count: assistant text "calling now" (2) + args words (1: {"id":"C1"}
        // is a single word token) + "it is active" (3) = 6 unique tokens
        assert_eq!(out.unique_target_count, 6);
        // reconstruct via tokenizer to assert none of the target tokens
        // detokenize into observation-only words
        let mut tok = SimpleTokenizer::new();
        for t in &turns {
            match t {
                TranscriptTurn::System(s) | TranscriptTurn::User(s) => {
                    tok.tokenize(s);
                }
                TranscriptTurn::Assistant { text, tool_calls } => {
                    tok.tokenize(text);
                    for (_, _, a) in tool_calls {
                        tok.tokenize(a);
                    }
                }
                TranscriptTurn::ToolObservation { content, .. } => {
                    tok.tokenize(content);
                }
            }
        }
        let obs_words: std::collections::BTreeSet<u32> = tok
            .tokenize_known(r#"{"status":"active"}"#)
            .into_iter()
            .collect();
        for t in &out.terms {
            assert!(
                !obs_words.contains(&t.target_token),
                "tool-result token became a target!"
            );
        }
    }

    #[test]
    fn singleton_group_is_excluded_not_zero_trained() {
        let execs = vec![exec(1, 5, "solo", 0.9)];
        let out = compute_loss_terms(&execs, &AdvantageConfig::default());
        assert!(out.terms.is_empty());
        assert!(out.excluded.contains(&1));
        assert_eq!(
            execs[0].reward.validity,
            RewardValidity::Valid,
            "excluded for lack of baseline, not because the reward was zeroed"
        );
    }
}
