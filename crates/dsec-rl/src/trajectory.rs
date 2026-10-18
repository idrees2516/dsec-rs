//! Trajectory trees with shared prefixes (QWENGYRE's trajectory
//! processor).
//!
//! Each execution's recorded model calls are organized into a tree: a
//! node is one turn at one position, keyed by `(parent node, turn
//! digest)`. Executions from the same task (a rollout group, or
//! branch-search samples) that share a prefix land on the *same* nodes —
//! "preserving each output's original conditioning context" (identical
//! node ⇒ identical context by construction) — and "deduplicates
//! redundant paths to bound training costs": a target token sitting in a
//! shared node is **counted once**, no matter how many executions pass
//! through it.
//!
//! The tree answers the questions the loss needs:
//! - which nodes hold trainable targets ([`TrajectoryTree::target_nodes`]);
//! - which executions pass through each node
//!   ([`TrajectoryNode::execs`]) — used to compute a mean advantage for
//!   shared targets and to weight terms per execution;
//! - the deduplicated target-token count
//!   ([`TrajectoryTree::shared_target_token_count`]).

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::tito::{SimpleTokenizer, TokenOrigin, TranscriptTurn};

/// Node id (index into the tree's arena).
pub type NodeId = usize;

/// Digest type for turn content (fnv1a64 — stable across runs).
pub type Digest = u64;

/// One turn at one position in the tree.
#[derive(Debug, Clone)]
pub struct TrajectoryNode {
    /// Parent node (`None` for the root).
    pub parent: Option<NodeId>,
    /// Digest of this turn's content (origin-tagged).
    pub digest: Digest,
    /// Depth = number of turns from the root.
    pub depth: usize,
    /// Executions whose path passes through this node.
    pub execs: BTreeSet<u64>,
    /// Token ids of this turn.
    pub tokens: Vec<u32>,
    /// Token origins of this turn (parallel to `tokens`).
    pub origins: Vec<TokenOrigin>,
    /// Model-call index within the execution.
    pub turn_idx: u32,
    /// Child node ids.
    pub children: Vec<NodeId>,
}

impl TrajectoryNode {
    /// Positions (within this node's tokens) that are trainable targets.
    pub fn target_positions(&self) -> Vec<usize> {
        self.origins
            .iter()
            .enumerate()
            .filter(|(_, o)| o.is_target())
            .map(|(i, _)| i)
            .collect()
    }

    /// Whether this node is shared by more than one execution.
    pub fn is_shared(&self) -> bool {
        self.execs.len() > 1
    }
}

/// A trajectory tree over executions of one task group.
#[derive(Debug, Clone, Default)]
pub struct TrajectoryTree {
    nodes: Vec<TrajectoryNode>,
    /// `(parent, digest) -> node id`, the dedup key.
    index: BTreeMap<(Option<NodeId>, Digest), NodeId>,
    /// `execution id -> path (node ids, root first)`.
    paths: BTreeMap<u64, Vec<NodeId>>,
}

impl TrajectoryTree {
    /// Empty tree.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts one execution's turns and returns its node path.
    ///
    /// Turns already present at the same position with the same content
    /// (same origin tags, same tokens) are reused — that is the shared
    /// prefix. The execution is added to every node's membership set.
    pub fn insert_execution(
        &mut self,
        exec_id: u64,
        turns: &[TranscriptTurn],
        tok: &SimpleTokenizer,
    ) -> Vec<NodeId> {
        let mut path: Vec<NodeId> = Vec::with_capacity(turns.len() + 1);
        let mut parent: Option<NodeId> = None;
        let mut model_call = 0u32;
        for turn in turns {
            let (origin, tokens, turn_idx) = match turn {
                TranscriptTurn::System(s) => {
                    (TokenOrigin::SystemTemplate, tok.tokenize_known(s), 0)
                }
                TranscriptTurn::User(s) => (TokenOrigin::HarnessInjected, tok.tokenize_known(s), 0),
                TranscriptTurn::Assistant { text, tool_calls } => {
                    let mut ids = tok.tokenize_known(text);
                    for (_cid, _name, args) in tool_calls {
                        ids.extend(tok.tokenize_known(args));
                    }
                    model_call += 1;
                    (TokenOrigin::ModelGenerated, ids, model_call - 1)
                }
                TranscriptTurn::ToolObservation { content, .. } => (
                    TokenOrigin::ToolResult,
                    tok.tokenize_known(content),
                    model_call.saturating_sub(1),
                ),
            };
            let digest = turn_digest(turn, &tokens);
            let key = (parent, digest);
            let origin_row = vec![origin; tokens.len()];
            let node_id = if let Some(&id) = self.index.get(&key) {
                id
            } else {
                let id = self.nodes.len();
                self.nodes.push(TrajectoryNode {
                    parent,
                    digest,
                    depth: parent.map(|p| self.nodes[p].depth + 1).unwrap_or(0),
                    execs: BTreeSet::new(),
                    tokens,
                    origins: origin_row,
                    turn_idx,
                    children: Vec::new(),
                });
                self.index.insert(key, id);
                if let Some(p) = parent {
                    self.nodes[p].children.push(id);
                }
                id
            };
            self.nodes[node_id].execs.insert(exec_id);
            path.push(node_id);
            parent = Some(node_id);
        }
        self.paths.insert(exec_id, path.clone());
        path
    }

    /// All nodes.
    pub fn nodes(&self) -> &[TrajectoryNode] {
        &self.nodes
    }

    /// Node by id.
    pub fn node(&self, id: NodeId) -> &TrajectoryNode {
        &self.nodes[id]
    }

    /// The recorded path of one execution.
    pub fn path(&self, exec_id: u64) -> Option<&[NodeId]> {
        self.paths.get(&exec_id).map(|v| v.as_slice())
    }

    /// Number of executions inserted.
    pub fn execution_count(&self) -> usize {
        self.paths.len()
    }

    /// Nodes containing at least one trainable target, each counted
    /// once regardless of how many executions share it.
    pub fn target_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.origins.iter().any(|o| o.is_target()))
            .map(|(i, _)| i)
            .collect()
    }

    /// Deduplicated trainable target-token count: shared-prefix targets
    /// contribute once.
    pub fn shared_target_token_count(&self) -> usize {
        self.nodes
            .iter()
            .map(|n| n.origins.iter().filter(|o| o.is_target()).count())
            .sum()
    }

    /// Naive (no dedup) target-token count — for tests to prove the
    /// dedup actually changes something.
    pub fn naive_target_token_count(&self) -> usize {
        self.paths
            .values()
            .map(|path| {
                path.iter()
                    .map(|&n| {
                        self.nodes[n]
                            .origins
                            .iter()
                            .filter(|o| o.is_target())
                            .count()
                    })
                    .sum::<usize>()
            })
            .sum()
    }
}

/// FNV-1a over the turn's structural tag + tokens (origin included so a
/// model turn and a tool turn with identical text never collide).
fn turn_digest(turn: &TranscriptTurn, tokens: &[u32]) -> Digest {
    let mut h: u64 = 0xcbf29ce484222325;
    let tag: u8 = match turn {
        TranscriptTurn::System(_) => 0,
        TranscriptTurn::User(_) => 1,
        TranscriptTurn::Assistant { .. } => 2,
        TranscriptTurn::ToolObservation { .. } => 3,
    };
    h ^= tag as u64;
    h = h.wrapping_mul(0x100000001b3);
    for t in tokens {
        h ^= *t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turns(extra: &str) -> Vec<TranscriptTurn> {
        vec![
            TranscriptTurn::System("sys prompt".into()),
            TranscriptTurn::User("do the task".into()),
            TranscriptTurn::Assistant {
                text: format!("working {extra}"),
                tool_calls: vec![],
            },
        ]
    }

    fn tree_of(specs: &[Vec<TranscriptTurn>]) -> (SimpleTokenizer, TrajectoryTree) {
        let mut tok = SimpleTokenizer::new();
        for t in specs {
            for turn in t {
                // grow the vocab over all turns
                match turn {
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
        }
        let mut tree = TrajectoryTree::new();
        for (i, t) in specs.iter().enumerate() {
            tree.insert_execution(i as u64, t, &tok);
        }
        (tok, tree)
    }

    #[test]
    fn shared_prefix_nodes_are_reused() {
        // three executions sharing system+user, differing in the final turn
        let specs = vec![turns("a"), turns("a"), turns("b")];
        let (_tok, tree) = tree_of(&specs);

        // 2 shared prefix nodes + 2 distinct assistant nodes = 4 nodes
        assert_eq!(tree.nodes().len(), 4);
        assert_eq!(tree.execution_count(), 3);

        // the shared prefix nodes have all 3 executions; the 'working a'
        // node has 2; 'working b' has 1
        let shared = tree.nodes().iter().filter(|n| n.is_shared()).count();
        assert_eq!(shared, 3, "sys + user + 'working a' are shared");
    }

    #[test]
    fn shared_prefix_targets_contribute_once() {
        let specs = vec![turns("a"), turns("a"), turns("a"), turns("b")];
        let (_tok, tree) = tree_of(&specs);
        let dedup = tree.shared_target_token_count();
        let naive = tree.naive_target_token_count();
        assert!(
            dedup < naive,
            "dedup {dedup} must be below naive {naive} when prefixes are shared"
        );
        // exactly: 'working a' (2 words) once + 'working b' (2 words) once
        assert_eq!(dedup, 4);
        // naive: 3 executions x 2 words + 1 x 2 words
        assert_eq!(naive, 3 * 2 + 2);
    }

    #[test]
    fn identical_executions_fully_collapse() {
        let specs = vec![turns("a"), turns("a")];
        let (_tok, tree) = tree_of(&specs);
        assert_eq!(tree.nodes().len(), 3);
        assert_eq!(tree.shared_target_token_count(), 2);
        assert_eq!(tree.naive_target_token_count(), 4);
    }

    #[test]
    fn divergent_executions_share_only_the_prompt() {
        let specs = vec![turns("a"), turns("b")];
        let (_tok, tree) = tree_of(&specs);
        // sys+user shared, assistants distinct
        assert_eq!(tree.nodes().len(), 4);
        for node in tree.nodes() {
            if node.depth < 2 {
                assert!(node.is_shared());
            } else {
                assert!(!node.is_shared());
            }
        }
    }

    #[test]
    fn tool_observations_never_become_target_nodes() {
        let mut tok = SimpleTokenizer::new();
        let t = vec![
            TranscriptTurn::System("s".into()),
            TranscriptTurn::User("u".into()),
            TranscriptTurn::Assistant {
                text: "call it".into(),
                tool_calls: vec![("c1".into(), "t".into(), "{}".into())],
            },
            TranscriptTurn::ToolObservation {
                call_id: "c1".into(),
                content: "tool says hello world".into(),
            },
        ];
        tok.tokenize("s");
        tok.tokenize("u");
        tok.tokenize("call it");
        tok.tokenize("{}");
        tok.tokenize("tool says hello world");
        let mut tree = TrajectoryTree::new();
        tree.insert_execution(0, &t, &tok);
        let targets = tree.target_nodes();
        assert_eq!(targets.len(), 1, "only the assistant node is a target");
        // and the tool node's tokens are all masked
        let tool_node = tree
            .nodes()
            .iter()
            .find(|n| n.origins.first() == Some(&TokenOrigin::ToolResult))
            .unwrap();
        assert!(tool_node.target_positions().is_empty());
    }
}
