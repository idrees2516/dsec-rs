//! The fixed-pool trainer (Phase 5).
//!
//! "Connect a real trainer and model server with explicit policy
//! versions and bounded staleness. Keep a static GPU split until the
//! data and policy-version contracts are reliable."
//!
//! The trainer consumes the Phase 4 loss contract
//! ([`dsec_rl::loss::compute_loss_terms`]) and updates the [`TinyLm`]
//! with real gradient steps. The **contracts**:
//!
//! - every rollout sample carries the policy version it was generated
//!   under (the model server reports `policy-v{major}`);
//! - samples older than `published.major - staleness_bound` are
//!   REJECTED for training (bounded staleness — no training on the
//!   distant past), reported per execution;
//! - each accepted step stages new weights and publishes them through
//!   the barrier; the returned version is what NEW rollouts must use;
//! - a pool shape (static GPU split) is configuration, deliberately
//!   fixed until the data + version contracts hold — the elastic
//!   scheduler (Phase 6) builds on top.

use serde::{Deserialize, Serialize};

use dsec_rl::loss::{compute_loss_terms_with, ExecutionInput, LossOutput};
use dsec_rl::reward::AdvantageConfig;
use dsec_rl::tito::SimpleTokenizer;

use crate::error::Result;
use crate::lm::TinyLm;
use crate::policy::{PolicyStore, PolicyVersion};

/// One gradient term as the trainer consumes it (mirrors
/// `dsec_rl::loss::LossTerm` without the tree internals).
#[derive(Debug, Clone)]
pub struct GradTerm {
    /// Conditioning context tokens.
    pub context_tokens: Vec<u32>,
    /// The target (policy-generated) token.
    pub target_token: u32,
    /// Merged coefficient (execution-level normalization).
    pub coefficient: f32,
    /// Node advantage (mean over member executions).
    pub advantage: f32,
}

/// A rollout sample with its policy-version stamp.
#[derive(Debug, Clone)]
pub struct RolloutSample {
    /// The execution (transcript + reward).
    pub execution: ExecutionInput,
    /// The policy version the rollout was generated under.
    pub policy_version: u64,
}

/// Trainer configuration.
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Learning rate.
    pub lr: f32,
    /// Maximum policy-version age accepted for training.
    pub staleness_bound: u64,
    /// Hidden size of the tiny LM.
    pub hidden: usize,
    /// RNG seed for initialization.
    pub seed: u64,
    /// Group advantage configuration.
    pub advantage: AdvantageConfig,
    /// Static GPU split (rollout : train) — fixed until the contracts
    /// hold; documented, not dynamically changed by this trainer.
    pub pool_rollout_gpus: usize,
    pub pool_train_gpus: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            lr: 0.1,
            staleness_bound: 1,
            hidden: 16,
            seed: 42,
            advantage: AdvantageConfig::default(),
            pool_rollout_gpus: 8,
            pool_train_gpus: 8,
        }
    }
}

/// One training step's outcome.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainStepResult {
    /// Executions accepted for training.
    pub accepted: Vec<u64>,
    /// Executions rejected as stale (with their version).
    pub rejected_stale: Vec<(u64, u64)>,
    /// Executions excluded by the loss path (evaluator failures, no
    /// baseline).
    pub excluded: Vec<u64>,
    /// Number of gradient terms (deduplicated target tokens).
    pub terms: usize,
    /// The pre-update loss.
    pub loss: f32,
    /// The newly published version (if the step trained and
    /// published).
    pub published: Option<PolicyVersion>,
}

/// The fixed-pool trainer.
pub struct Trainer {
    config: TrainerConfig,
    model: TinyLm,
    store: PolicyStore,
    tokenizer: SimpleTokenizer,
    step_count: u64,
}

impl Trainer {
    /// Creates a trainer: version 0 published immediately.
    pub fn new(config: TrainerConfig) -> Self {
        let mut tokenizer = SimpleTokenizer::new();
        // a working vocabulary so the tiny LM has something to model;
        // rollouts grow it before collection (see `ensure_vocab`).
        tokenizer.tokenize("the model works in a sandbox with tools");
        let model = TinyLm::new(&tokenizer, config.hidden, config.seed);
        let weights = serialize_model(&model);
        let v0 = PolicyVersion {
            major: 0,
            digest: model.digest(),
        };
        let store = PolicyStore::new(v0, weights);
        Self {
            config,
            model,
            store,
            tokenizer,
            step_count: 0,
        }
    }

    /// The published version (what rollouts must serve).
    pub fn published(&self) -> PolicyVersion {
        self.store.published()
    }

    /// The policy store (shared with the model server).
    pub fn store(&self) -> PolicyStore {
        self.store.clone()
    }

    /// The tokenizer (shared vocabulary).
    pub fn tokenizer(&self) -> &SimpleTokenizer {
        &self.tokenizer
    }

    /// Grows the vocabulary to cover every word in the executions'
    /// transcripts (rollout collection calls this before stepping so
    /// shared-prefix ids line up).
    pub fn ensure_vocab(&mut self, samples: &[RolloutSample]) {
        for s in samples {
            for turn in &s.execution.turns {
                let text = match turn {
                    dsec_rl::tito::TranscriptTurn::System(t) => t.clone(),
                    dsec_rl::tito::TranscriptTurn::User(t) => t.clone(),
                    dsec_rl::tito::TranscriptTurn::Assistant { text, tool_calls } => {
                        let mut all = text.clone();
                        for (_, _, args) in tool_calls {
                            all.push(' ');
                            all.push_str(args);
                        }
                        all
                    }
                    dsec_rl::tito::TranscriptTurn::ToolObservation { content, .. } => {
                        content.clone()
                    }
                };
                self.tokenizer.tokenize(&text);
            }
        }
        // resize the model if the vocabulary grew
        let vocab = self.tokenizer.vocab_size();
        if vocab > self.model.vocab_size() {
            self.model.resize_vocab(vocab, self.config.seed);
        }
    }

    /// The current model (tests / serving internals).
    pub fn model(&self) -> &TinyLm {
        &self.model
    }

    /// One training step over the collected samples.
    ///
    /// Rejections first (staleness), then the Phase 4 loss path
    /// (evaluator failures excluded there), then one gradient step and
    /// a barrier publication.
    pub fn step(&mut self, samples: &[RolloutSample]) -> Result<TrainStepResult> {
        let published = self.store.published();
        let mut result = TrainStepResult::default();

        // 1. bounded staleness: version contract enforcement
        let mut accepted_inputs: Vec<ExecutionInput> = Vec::new();
        for s in samples {
            let age = published.major.saturating_sub(s.policy_version);
            if age > self.config.staleness_bound {
                result
                    .rejected_stale
                    .push((s.execution.execution_id, s.policy_version));
            } else {
                accepted_inputs.push(s.execution.clone());
            }
        }
        result.accepted = accepted_inputs.iter().map(|e| e.execution_id).collect();
        if accepted_inputs.is_empty() {
            // nothing to train; no publication
            return Ok(result);
        }

        // 2. the Phase 4 loss path
        self.ensure_vocab(samples);
        let advantage = self.config.advantage.clone();
        let loss_out: LossOutput =
            compute_loss_terms_with(&accepted_inputs, &advantage, &mut self.tokenizer);
        result.excluded = loss_out.excluded.iter().copied().collect();
        result.terms = loss_out.unique_target_count;

        let terms: Vec<GradTerm> = loss_out
            .terms
            .iter()
            .map(|t| GradTerm {
                context_tokens: t.context_tokens.clone(),
                target_token: t.target_token,
                coefficient: t.coefficient,
                advantage: t.advantage,
            })
            .collect();

        // 3. real gradient step
        let loss = self.model.policy_gradient_step(&terms, self.config.lr)?;
        result.loss = loss;

        // 4. stage + publish through the barrier
        self.step_count += 1;
        let version = PolicyVersion {
            major: self.step_count,
            digest: self.model.digest(),
        };
        let weights = serialize_model(&self.model);
        self.store.stage(version.clone(), weights)?;
        let published = self.store.publish(false)?;
        result.published = Some(published);
        Ok(result)
    }

    /// Step with a crashpoint at publication (Phase 6 drills): the
    /// staged weights exist, the pointer never flipped, the barrier
    /// stays closed.
    pub fn step_with_publication_crash(
        &mut self,
        samples: &[RolloutSample],
    ) -> Result<TrainStepResult> {
        let result = self.step_no_publish(samples)?;
        // stage happened inside step_no_publish? no: it skips staging.
        // Stage here, then crash the publication.
        let version = PolicyVersion {
            major: self.step_count + 1,
            digest: self.model.digest(),
        };
        let weights = serialize_model(&self.model);
        self.store.stage(version, weights)?;
        let r = self.store.publish(true); // crash before flip
        if let Err(e) = r {
            assert!(e.to_string().contains("CRASH"));
        }
        Ok(result)
    }

    fn step_no_publish(&mut self, samples: &[RolloutSample]) -> Result<TrainStepResult> {
        // train without staging/publication (crash drill variant)
        let published = self.store.published();
        let mut result = TrainStepResult::default();
        let mut accepted_inputs: Vec<ExecutionInput> = Vec::new();
        for s in samples {
            let age = published.major.saturating_sub(s.policy_version);
            if age > self.config.staleness_bound {
                result
                    .rejected_stale
                    .push((s.execution.execution_id, s.policy_version));
            } else {
                accepted_inputs.push(s.execution.clone());
            }
        }
        result.accepted = accepted_inputs.iter().map(|e| e.execution_id).collect();
        if accepted_inputs.is_empty() {
            return Ok(result);
        }
        self.ensure_vocab(samples);
        let advantage = self.config.advantage.clone();
        let loss_out = compute_loss_terms_with(&accepted_inputs, &advantage, &mut self.tokenizer);
        result.excluded = loss_out.excluded.iter().copied().collect();
        result.terms = loss_out.unique_target_count;
        let terms: Vec<GradTerm> = loss_out
            .terms
            .iter()
            .map(|t| GradTerm {
                context_tokens: t.context_tokens.clone(),
                target_token: t.target_token,
                coefficient: t.coefficient,
                advantage: t.advantage,
            })
            .collect();
        result.loss = self.model.policy_gradient_step(&terms, self.config.lr)?;
        self.step_count += 1;
        Ok(result)
    }

    /// Recovers an interrupted publication: either completes the flip
    /// (the staged weights are valid) or discards it.
    pub fn recover_publication(&self, complete: bool) -> Result<PolicyVersion> {
        self.store.recover_publication(complete)
    }
}

/// Serializes a model for a checkpoint.
pub fn serialize_model(model: &TinyLm) -> Vec<u8> {
    serde_json::to_vec(model).expect("model serialization")
}

/// Deserializes a checkpoint.
pub fn deserialize_model(bytes: &[u8]) -> Result<TinyLm> {
    Ok(serde_json::from_slice(bytes)?)
}
