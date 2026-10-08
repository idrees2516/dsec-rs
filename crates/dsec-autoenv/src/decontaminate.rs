//! Train–test decontamination via n-gram overlap (Section 3.2, App. F.2).
//!
//! "Following Tmax, we measure train–test contamination as n-gram overlap
//! between the task descriptions of generated environments and those of our
//! evaluation benchmarks ... Using a sliding window of n=13 tokens with
//! stride 1, we flag a generated task if any of its windows matches at least
//! one 13-gram from a benchmark task. This check measures textual overlap;
//! it does not rule out paraphrased copies or other forms of contamination."

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration of the contamination check. Defaults follow the paper:
/// `n = 13`, stride `1`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecontaminationConfig {
    /// Window size in tokens (paper: 13).
    pub n: usize,
    /// Stride between windows (paper: 1).
    pub stride: usize,
}

impl Default for DecontaminationConfig {
    fn default() -> Self {
        DecontaminationConfig { n: 13, stride: 1 }
    }
}

/// One contamination hit: a shared n-gram window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContaminationMatch {
    /// The shared window, rendered as text.
    pub window: String,
    /// The held-out benchmark task it matched.
    pub benchmark_task: String,
}

/// Tokenizer: lowercase and split on non-alphanumeric characters. Word-level
/// tokens, the standard treatment for description-level n-grams.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

/// Render a window of tokens back to text.
fn window_text(tokens: &[String], start: usize, n: usize) -> String {
    tokens[start..(start + n).min(tokens.len())].join(" ")
}

/// The set of n-grams of one text, with the configured stride.
pub fn ngrams(text: &str, cfg: &DecontaminationConfig) -> HashSet<String> {
    let tokens = tokenize(text);
    if tokens.len() < cfg.n {
        return HashSet::new();
    }
    let mut grams = HashSet::new();
    let mut start = 0;
    while start + cfg.n <= tokens.len() {
        grams.insert(window_text(&tokens, start, cfg.n));
        start += cfg.stride.max(1);
    }
    grams
}

/// The contamination index over held-out benchmark task descriptions.
#[derive(Debug, Clone, Default)]
pub struct ContaminationIndex {
    /// Every held-out n-gram, mapped to the benchmark tasks containing it.
    grams: HashMap<String, Vec<String>>,
    /// Number of indexed tasks.
    pub indexed_tasks: usize,
}

impl ContaminationIndex {
    /// Build the index over `(task_id, instruction)` pairs.
    pub fn build(heldout: &[(String, String)]) -> Self {
        let cfg = DecontaminationConfig::default();
        let mut grams: HashMap<String, Vec<String>> = HashMap::new();
        for (id, text) in heldout {
            for gram in ngrams(text, &cfg) {
                grams.entry(gram).or_default().push(id.clone());
            }
        }
        ContaminationIndex {
            grams,
            indexed_tasks: heldout.len(),
        }
    }

    /// Check an instruction for overlap. Returns every matching window
    /// (usually checked as `is_empty()` for the flag).
    pub fn check(&self, instruction: &str) -> Vec<ContaminationMatch> {
        let cfg = DecontaminationConfig::default();
        let mut hits = Vec::new();
        for gram in ngrams(instruction, &cfg) {
            if let Some(tasks) = self.grams.get(&gram) {
                hits.push(ContaminationMatch {
                    window: gram.clone(),
                    benchmark_task: tasks.first().cloned().unwrap_or_default(),
                });
            }
        }
        hits.sort_by(|a, b| a.window.cmp(&b.window));
        hits
    }

    /// Whether the instruction is contaminated (shares at least one n-gram
    /// with a held-out task).
    pub fn is_contaminated(&self, instruction: &str) -> bool {
        !self.check(instruction).is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HELDOUT: &str = "Rebuild the PyTorch model from its saved weights and verify the outputs match the original within tolerance";

    fn index() -> ContaminationIndex {
        ContaminationIndex::build(&[("tb2.1/pytorch-restore".to_string(), HELDOUT.to_string())])
    }

    #[test]
    fn verbatim_overlap_is_flagged() {
        // Instruction that copies >= 13 consecutive tokens from the benchmark.
        let instruction = "First rebuild the pytorch model from its saved weights and verify the outputs match the original within tolerance, then report";
        assert!(index().is_contaminated(instruction));
    }

    #[test]
    fn clean_instruction_passes() {
        let instruction = "Reconcile the March billing export against the general ledger and write three report files to /app/out";
        assert!(!index().is_contaminated(instruction));
    }

    #[test]
    fn paraphrase_is_not_caught() {
        // The paper's caveat: the check measures textual overlap only.
        let paraphrase = "Restore the neural network checkpoint and confirm the reconstructed outputs agree with the originals to within the allowed error";
        assert!(!index().is_contaminated(paraphrase));
    }

    #[test]
    fn short_texts_have_no_grams() {
        assert!(ngrams("too short to matter", &DecontaminationConfig::default()).is_empty());
        let tokens = tokenize("Exactly thirteen word tokens appear in this particular test sequence right about here now yes");
        assert!(tokens.len() >= 13);
        assert_eq!(
            ngrams(&tokens.join(" "), &DecontaminationConfig::default()).len(),
            tokens.len() - 12
        );
    }

    #[test]
    fn tokenizer_normalizes() {
        assert_eq!(
            tokenize("The, QUICK-Brown fox!"),
            vec!["the", "quick", "brown", "fox"]
        );
    }

    #[test]
    fn stride_one_sliding_window() {
        let cfg = DecontaminationConfig { n: 2, stride: 1 };
        let grams = ngrams("a b c d", &cfg);
        assert_eq!(grams.len(), 3);
        assert!(grams.contains("a b"));
        assert!(grams.contains("c d"));
    }
}
