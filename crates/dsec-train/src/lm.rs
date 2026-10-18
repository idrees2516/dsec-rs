//! A tiny, genuinely-trainable language model with hand-written
//! forward and backward passes.
//!
//! This is the "real trainer" half of Phase 5: the model is small (a
//! mean-context linear LM over the shared tokenizer's vocabulary), but
//! every piece of the training contract is real — parameter tensors,
//! softmax log-probabilities, policy-gradient gradients derived by
//! hand, deterministic SGD updates, and SHA-256 weight digests for
//! policy versioning. Production would swap the model for vLLM/SGLang
//! serving plus a verl-style trainer; the **contracts** (versions,
//! staleness bounds, publication barriers) are what this crate tests.
//!
//! Architecture: `logit(v | ctx) = W[v] · h(ctx) + b[v]` where
//! `h(ctx)` is the mean of the context-token embeddings (a learned
//! start vector when the context is empty). Parameters:
//! `emb: V×H`, `w: V×H`, `b: V`, `start: H`.

use serde::{Deserialize, Serialize};

use dsec_protocol::rng::Rng;
use dsec_rl::tito::SimpleTokenizer;

use crate::error::{Error, Result};

/// A tiny LM over a fixed vocabulary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TinyLm {
    /// Token embeddings (`vocab_size × hidden`).
    pub emb: Vec<Vec<f32>>,
    /// Output weights (`vocab_size × hidden`).
    pub w: Vec<Vec<f32>>,
    /// Output biases (`vocab_size`).
    pub b: Vec<f32>,
    /// Start vector for empty contexts.
    pub start: Vec<f32>,
    /// RNG state for sampling (deterministic).
    pub rng_state: u64,
}

fn xavier(rows: usize, cols: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    let scale = (2.0f32 / (rows + cols) as f32).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| (rng.next_f64() as f32 * 2.0 - 1.0) * scale)
                .collect()
        })
        .collect()
}

impl TinyLm {
    /// Initializes a model over the tokenizer's current vocabulary.
    pub fn new(tokenizer: &SimpleTokenizer, hidden: usize, seed: u64) -> Self {
        let vocab = tokenizer.vocab_size();
        let mut rng = Rng::new(seed);
        Self {
            emb: xavier(vocab, hidden, &mut rng),
            w: xavier(vocab, hidden, &mut rng),
            b: vec![0.0; vocab],
            start: (0..hidden)
                .map(|_| (rng.next_f64() as f32 * 2.0 - 1.0) * 0.1)
                .collect(),
            rng_state: seed ^ 0x5DEECE66D,
        }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.w.len()
    }

    /// Hidden dimension.
    pub fn hidden(&self) -> usize {
        self.start.len()
    }

    /// The context vector `h(ctx)` (mean of embeddings / start).
    pub fn context_vector(&self, ctx: &[u32]) -> Vec<f32> {
        let h = self.hidden();
        let mut out = vec![0.0f32; h];
        if ctx.is_empty() {
            out.copy_from_slice(&self.start);
            return out;
        }
        for t in ctx {
            let e = &self.emb[*t as usize];
            for (o, v) in out.iter_mut().zip(e) {
                *o += v;
            }
        }
        let n = ctx.len() as f32;
        for o in out.iter_mut() {
            *o /= n;
        }
        out
    }

    /// Full logits over the vocabulary given a context vector.
    pub fn logits(&self, ctx: &[u32]) -> Vec<f32> {
        let h = self.context_vector(ctx);
        let mut out = Vec::with_capacity(self.vocab_size());
        for (v, wv) in self.w.iter().enumerate() {
            let mut dot = self.b[v];
            for (a, b) in h.iter().zip(wv) {
                dot += a * b;
            }
            out.push(dot);
        }
        out
    }

    /// Softmax log-probability of `target` given `ctx`.
    pub fn log_prob(&self, ctx: &[u32], target: u32) -> f32 {
        let logits = self.logits(ctx);
        softmax_logprob(&logits, target as usize)
    }

    /// Greedy argmax continuation of up to `max_tokens` tokens
    /// (deterministic).
    pub fn generate_greedy(&self, ctx: &[u32], max_tokens: usize) -> Vec<u32> {
        let mut ctx = ctx.to_vec();
        let mut out = Vec::new();
        for _ in 0..max_tokens {
            let logits = self.logits(&ctx);
            let best = argmax(&logits);
            out.push(best as u32);
            ctx.push(best as u32);
            if best as u32 == dsec_rl::tito::EOS_ID {
                break;
            }
        }
        out
    }

    /// Temperature continuation (deterministic seeded sampling).
    pub fn generate_sampled(
        &mut self,
        ctx: &[u32],
        max_tokens: usize,
        temperature: f32,
    ) -> Vec<u32> {
        let mut ctx = ctx.to_vec();
        let mut out = Vec::new();
        let temp = temperature.max(1e-3);
        for _ in 0..max_tokens {
            let logits = self.logits(&ctx);
            let probs = softmax(&logits, temp);
            let mut rng = Rng::new(self.rng_state);
            let pick = sample(&probs, rng.next_f64() as f32);
            self.rng_state = self.rng_state.wrapping_add(0x9E3779B97F4A7C15);
            out.push(pick as u32);
            ctx.push(pick as u32);
            if pick as u32 == dsec_rl::tito::EOS_ID {
                break;
            }
        }
        out
    }

    /// Policy-gradient update from loss terms (the Phase 4 contract):
    /// `L = -Σ term.coefficient · term.advantage · logπ(target|ctx)`.
    ///
    /// Returns the pre-update loss. Gradients are accumulated across
    /// terms (shared-prefix terms already merged upstream), then one
    /// SGD step applies.
    pub fn policy_gradient_step(
        &mut self,
        terms: &[crate::trainer::GradTerm],
        lr: f32,
    ) -> Result<f32> {
        let vocab = self.vocab_size();
        let hidden = self.hidden();
        let mut loss = 0.0f32;
        // gradient accumulators
        let mut g_emb = vec![vec![0.0f32; hidden]; vocab];
        let mut g_w = vec![vec![0.0f32; hidden]; vocab];
        let mut g_b = vec![0.0f32; vocab];
        let mut g_start = vec![0.0f32; hidden];

        for term in terms {
            let ctx = &term.context_tokens;
            let target = term.target_token as usize;
            if target >= vocab {
                return Err(Error::Config(format!(
                    "target token {target} out of vocab {vocab}"
                )));
            }
            let h = self.context_vector(ctx);
            let logits = self.logits(ctx);
            let logp = softmax_logprob(&logits, target);
            // d(loss)/d(logit_target) = -coeff*adv*(1 - p_target)
            let p_target = logits[target].exp() / logits.iter().map(|l| l.exp()).sum::<f32>();
            let dlogit = -term.coefficient * term.advantage * (1.0 - p_target);
            // into W[target] and b[target]
            for (k, hv) in h.iter().enumerate() {
                g_w[target][k] += dlogit * hv;
            }
            g_b[target] += dlogit;
            // back into h: dlogP/dh = dlogit * W[target]
            let mut dh = vec![0.0f32; hidden];
            for (k, wv) in self.w[target].iter().enumerate() {
                dh[k] = dlogit * wv;
            }
            // h is the mean of context embeddings (or start when empty)
            if ctx.is_empty() {
                for (g, d) in g_start.iter_mut().zip(dh) {
                    *g += d;
                }
            } else {
                let n = ctx.len() as f32;
                for t in ctx {
                    let ge = &mut g_emb[*t as usize];
                    for (g, d) in ge.iter_mut().zip(&dh) {
                        *g += d / n;
                    }
                }
            }
            loss -= term.coefficient * term.advantage * logp;
        }

        // SGD step
        for (v, (w, gw)) in self.w.iter_mut().zip(g_w).enumerate() {
            for (wv, gv) in w.iter_mut().zip(gw) {
                *wv -= lr * gv;
            }
            self.b[v] -= lr * g_b[v];
        }
        for (e, ge) in self.emb.iter_mut().zip(g_emb) {
            for (ev, gv) in e.iter_mut().zip(ge) {
                *ev -= lr * gv;
            }
        }
        for (s, g) in self.start.iter_mut().zip(g_start) {
            *s -= lr * g;
        }
        Ok(loss)
    }

    /// Grows the parameter tables when the vocabulary grows (new rows
    /// get seeded initialization).
    pub fn resize_vocab(&mut self, new_vocab: usize, seed: u64) {
        let hidden = self.hidden();
        let mut rng = Rng::new(seed ^ 0xA5A5A5A5);
        while self.emb.len() < new_vocab {
            self.emb.push(
                (0..hidden)
                    .map(|_| (rng.next_f64() as f32 * 2.0 - 1.0) * 0.05)
                    .collect(),
            );
        }
        while self.w.len() < new_vocab {
            self.w.push(
                (0..hidden)
                    .map(|_| (rng.next_f64() as f32 * 2.0 - 1.0) * 0.05)
                    .collect(),
            );
        }
        while self.b.len() < new_vocab {
            self.b.push(0.0);
        }
    }

    /// SHA-256 digest of the serialized parameters (policy versioning).
    pub fn digest(&self) -> String {
        let mut bytes = Vec::new();
        for row in self.emb.iter().chain(self.w.iter()) {
            for v in row {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in &self.b {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.start {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        dsec_protocol::sha256_hex(&bytes)
    }
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, x) in xs.iter().enumerate() {
        if *x > bv {
            bv = *x;
            best = i;
        }
    }
    best
}

fn softmax(logits: &[f32], temp: f32) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| ((l - max) / temp).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn softmax_logprob(logits: &[f32], target: usize) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|l| (l - max).exp()).sum();
    (logits[target] - max) - sum.ln()
}

fn sample(probs: &[f32], u: f32) -> usize {
    let mut acc = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if u <= acc {
            return i;
        }
    }
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> SimpleTokenizer {
        let mut t = SimpleTokenizer::new();
        t.tokenize("the quick brown fox jumps over the lazy dog");
        t
    }

    #[test]
    fn log_probs_are_normalized() {
        let t = tok();
        let lm = TinyLm::new(&t, 8, 42);
        let ctx = t.tokenize_known("the quick");
        // log p + log (1-p-ish): just sanity — finite, <= 0
        let lp = lm.log_prob(&ctx, 1);
        assert!(lp.is_finite());
        assert!(lp <= 1e-6, "log prob must be <= 0, got {lp}");
    }

    #[test]
    fn digest_changes_only_when_weights_change() {
        let t = tok();
        let lm = TinyLm::new(&t, 8, 7);
        let d0 = lm.digest();
        let same = lm.clone();
        assert_eq!(same.digest(), d0);
        let terms = [crate::trainer::GradTerm {
            context_tokens: vec![1, 2],
            target_token: 3,
            coefficient: 0.1,
            advantage: 1.0,
        }];
        let mut trained = lm;
        let loss0 = trained.policy_gradient_step(&terms, 0.5).unwrap();
        assert!(loss0.is_finite());
        assert_ne!(trained.digest(), d0, "training must change the weights");
    }

    #[test]
    fn policy_gradient_increases_rewarded_token_probability() {
        // THE real-learning test: with positive advantage on token 3,
        // repeated updates must raise p(3 | ctx)
        let t = tok();
        let mut lm = TinyLm::new(&t, 8, 1234);
        let ctx = t.tokenize_known("the quick");
        let target = 3u32;
        let p_before = lm.log_prob(&ctx, target).exp();
        let terms = [crate::trainer::GradTerm {
            context_tokens: ctx.clone(),
            target_token: target,
            coefficient: 1.0,
            advantage: 1.0,
        }];
        for _ in 0..30 {
            lm.policy_gradient_step(&terms, 0.5).unwrap();
        }
        let p_after = lm.log_prob(&ctx, target).exp();
        assert!(
            p_after > p_before,
            "rewarded token must gain probability: {p_before} -> {p_after}"
        );
    }

    #[test]
    fn negative_advantage_suppresses_token() {
        let t = tok();
        let mut lm = TinyLm::new(&t, 8, 99);
        let ctx = t.tokenize_known("the lazy");
        let target = 2u32;
        let p_before = lm.log_prob(&ctx, target).exp();
        let terms = [crate::trainer::GradTerm {
            context_tokens: ctx.clone(),
            target_token: target,
            coefficient: 1.0,
            advantage: -1.0,
        }];
        for _ in 0..30 {
            lm.policy_gradient_step(&terms, 0.5).unwrap();
        }
        let p_after = lm.log_prob(&ctx, target).exp();
        assert!(
            p_after < p_before,
            "punished token must lose probability: {p_before} -> {p_after}"
        );
    }

    #[test]
    fn generation_is_deterministic() {
        let t = tok();
        let mut lm = TinyLm::new(&t, 8, 5);
        let ctx = t.tokenize_known("the");
        let g1 = lm.generate_sampled(&ctx, 5, 0.8);
        let g2 = lm.generate_sampled(&ctx, 5, 0.8);
        // same rng state seed each call? rng_state mutates — reset it
        lm.rng_state = 5 ^ 0x5DEECE66D;
        let g3 = lm.generate_sampled(&ctx, 5, 0.8);
        assert_ne!(g1, g2, "rng advances between calls");
        assert_eq!(g1, g3, "same rng state gives the same sample");
        let greedy1 = lm.generate_greedy(&ctx, 3);
        let greedy2 = lm.generate_greedy(&ctx, 3);
        assert_eq!(greedy1, greedy2);
    }
}
