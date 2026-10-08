//! Clarification tasks: underspecified instructions, withheld details, the
//! `ask_human` tool, and Ask-F1 (Section 4.5, Appendix B).
//!
//! "We focus on tasks with incomplete instructions, where successful
//! execution requires obtaining missing information from the user."
//! HiL-Bench's three instruction settings:
//!
//! * **full_info** — the complete instruction (every withheld detail
//!   inlined);
//! * **ask_human** — the incomplete instruction plus the `ask_human` tool;
//! * **no tool** — the incomplete instruction without the tool.
//!
//! "Each [generated task] has an underspecified instruction, a registry of
//! withheld details that the ask_human tool can reveal, and tests that
//! depend on those details. The complete-information and no-tool variants
//! are derived from the same task without further model calls."
//!
//! Ask-F1 "measures whether the questions cover the information needed to
//! complete the task": blockers are retrieved by questions; precision
//! penalizes redundant or wasted questions, recall penalizes missed
//! blockers.

use crate::reward::{clarification_reward, ClarificationRewardConfig};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// The HiL-Bench blocker taxonomy (Table 6's five types).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockerType {
    /// Missing parameters (e.g. a numeric tolerance).
    MissingParameters,
    /// Business information (e.g. which statuses are in scope).
    BusinessInformation,
    /// Ambiguous requirements (e.g. precedence for duplicates).
    AmbiguousRequirements,
    /// A question the task must answer (e.g. report ordering).
    Question,
    /// Schema (e.g. how a computed field is defined).
    Schema,
}

impl BlockerType {
    /// The taxonomy label.
    pub fn label(&self) -> &'static str {
        match self {
            BlockerType::MissingParameters => "Missing parameters",
            BlockerType::BusinessInformation => "Business information",
            BlockerType::AmbiguousRequirements => "Ambiguous requirements",
            BlockerType::Question => "Question",
            BlockerType::Schema => "Schema",
        }
    }
}

/// One withheld detail: a gap in the instruction plus the answer
/// `ask_human` returns for it, plus the checks that depend on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Blocker {
    /// Blocker identifier (snake_case, e.g.
    /// `amount_agreement_tolerance`).
    pub id: String,
    /// The taxonomy type.
    pub kind: BlockerType,
    /// Keywords a question must contain to reveal this blocker.
    pub question_keywords: Vec<String>,
    /// The withheld answer.
    pub withheld_answer: String,
    /// The test checks that depend on this detail.
    pub affects_checks: Vec<String>,
}

/// The registry of withheld details backing the `ask_human` tool.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BlockerRegistry {
    /// All blockers.
    pub blockers: Vec<Blocker>,
}

impl BlockerRegistry {
    /// Build a registry.
    pub fn new(blockers: Vec<Blocker>) -> Self {
        BlockerRegistry { blockers }
    }

    /// The `ask_human` tool: reveal the blocker a question covers, or
    /// `None` when the question misses every withheld detail.
    pub fn ask(&self, question: &str) -> Option<&Blocker> {
        let q = question.to_lowercase();
        self.blockers.iter().find(|b| {
            b.question_keywords
                .iter()
                .all(|k| q.contains(&k.to_lowercase()))
        })
    }

    /// The blockers whose answers a list of questions revealed.
    pub fn revealed_by(&self, questions: &[String]) -> BTreeSet<String> {
        let mut revealed = BTreeSet::new();
        for q in questions {
            if let Some(b) = self.ask(q) {
                revealed.insert(b.id.clone());
            }
        }
        revealed
    }

    /// The set of blockers the task's tests depend on (all of them, by
    /// construction: "tests that depend on those details").
    pub fn needed(&self) -> BTreeSet<String> {
        self.blockers.iter().map(|b| b.id.clone()).collect()
    }
}

/// The three HiL instruction settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HilSetting {
    /// Complete information: the withheld answers are inlined into the
    /// instruction.
    FullInfo,
    /// Incomplete information with the `ask_human` tool.
    AskHuman,
    /// Incomplete information without the tool.
    NoTool,
}

/// The Ask-F1 decomposition over the needed blocker set.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AskF1 {
    /// Fraction of revealed (question, blocker) matches that were useful.
    pub precision: f64,
    /// Fraction of needed blockers revealed.
    pub recall: f64,
    /// Harmonic mean (0 when either is 0).
    pub f1: f64,
}

/// Compute Ask-F1: whether the questions cover the information needed.
///
/// * **recall** = |revealed ∩ needed| / |needed| — did the agent ask about
///   every withheld detail the tests depend on?
/// * **precision** = |revealed distinct| / |matching question-blocker
///   pairs| — redundant questions asking the same blocker, or questions
///   that reveal nothing, lower precision (questions that miss every
///   blocker count as wasted retrievals).
/// * **f1** = harmonic mean.
pub fn ask_f1(questions: &[String], registry: &BlockerRegistry) -> AskF1 {
    let needed = registry.needed();
    if needed.is_empty() {
        return AskF1 {
            precision: 1.0,
            recall: 1.0,
            f1: 1.0,
        };
    }
    let mut matches = 0usize;
    let mut revealed = BTreeSet::new();
    for q in questions {
        if let Some(b) = registry.ask(q) {
            matches += 1;
            revealed.insert(b.id.clone());
        } else {
            matches += 1; // a wasted question: retrieved nothing
        }
    }
    let recall = revealed.len() as f64 / needed.len() as f64;
    let precision = if matches == 0 {
        0.0
    } else {
        revealed.len() as f64 / matches as f64
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    AskF1 {
        precision,
        recall,
        f1,
    }
}

/// One HiL attempt: the questions asked (through `ask_human`) and whether
/// the task's tests passed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HilAttempt {
    /// The questions the agent asked, in order.
    pub questions: Vec<String>,
    /// Whether the agent had access to the `ask_human` tool.
    pub had_tool: bool,
    /// Whether the tests passed.
    pub task_success: bool,
}

impl HilAttempt {
    /// Whether the attempt asked at least one question (the ask-rate
    /// metric's numerator).
    pub fn asked(&self) -> bool {
        self.had_tool && !self.questions.is_empty()
    }

    /// Ask-F1 of the attempt (0 without the tool — nothing can be
    /// revealed).
    pub fn ask_f1(&self, registry: &BlockerRegistry) -> AskF1 {
        if !self.had_tool {
            return AskF1 {
                precision: 0.0,
                recall: 0.0,
                f1: 0.0,
            };
        }
        ask_f1(&self.questions, registry)
    }

    /// The shaped clarification reward (Section 4.5).
    pub fn shaped_reward(
        &self,
        registry: &BlockerRegistry,
        config: &ClarificationRewardConfig,
    ) -> f64 {
        let f1 = self.ask_f1(registry).f1;
        clarification_reward(self.task_success, f1, config)
    }
}

/// A HiL task: underspecified instruction, blocker registry, and the tests
/// that depend on the withheld details.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HilTask {
    /// The task name.
    pub name: String,
    /// The (incomplete) instruction.
    pub instruction: String,
    /// The withheld details.
    pub registry: BlockerRegistry,
    /// The test checks that depend on the blockers.
    pub check_names: Vec<String>,
}

impl HilTask {
    /// Derive the complete-information variant: every withheld answer
    /// inlined into the instruction ("without further model calls").
    pub fn derive_full_info(&self) -> String {
        let mut s = self.instruction.clone();
        s.push_str("\n\n## Conventions\n\n");
        for b in &self.registry.blockers {
            s.push_str(&format!(
                "- **{}** ({}) — {}\n",
                b.id,
                b.kind.label(),
                b.withheld_answer
            ));
        }
        s
    }

    /// Derive the no-tool variant: the incomplete instruction, tool
    /// removed.
    pub fn derive_no_tool(&self) -> String {
        self.instruction.clone()
    }
}

/// The example generated HiL-Bench task of Figure 9 and Table 6: the
/// March billing/ledger reconciliation with five withheld details, one per
/// blocker type.
pub fn example_billing_task() -> HilTask {
    let instruction = r#"# Reconcile the March billing and ledger exports

I'm closing out March on the accounts-receivable side and the two systems don't agree with each other. I've dropped both exports in `/app/exports`:

- `billing_export.csv` — pulled from the billing platform, one row per invoice: `invoice_id`, `customer_name`, `issue_date`, `status`, `net_amount`, `tax_amount`. The exporter is sloppy: it pads some fields with spaces, doesn't normalise the status casing, and quotes any customer name that contains a comma.
- `ledger_export.tsv` — pulled from the accounting ledger, tab-separated: `doc_ref`, `posted_on`, `amount`, `source_system`, `memo`. Amounts come out formatted for humans (thousands separators, negatives in parentheses) and the `doc_ref` casing and padding are inconsistent. Not every ledger row corresponds to an invoice.

Please write the reconciliation and leave three reports in `/app/out`:

**`matched.csv`** — header `invoice_id,billing_total,ledger_amount`, one row per invoice whose ledger posting agrees with the billing total, ordered by `invoice_id` ascending.

**`discrepancies.csv`** — header `invoice_id,billing_total,ledger_amount,difference`, one row per invoice that has a ledger posting whose amount disagrees with the billing total. `difference` is the ledger amount minus the billing total.

**`summary.json`** — a JSON object with these keys:
- `matched_count` — number of rows in `matched.csv`
- `discrepancy_count` — number of rows in `discrepancies.csv`
- `billing_only_count` — invoices with no ledger posting at all
- `ledger_only_count` — ledger rows that don't correspond to any invoice
- `total_variance` — the overall variance between the two systems for the month

A few conventions to follow:
- An invoice's billing total is `net_amount + tax_amount`.
- Match invoices to ledger rows on the invoice reference, ignoring case and surrounding whitespace.
- Round every money value to two decimals and write it with exactly two decimal places and no thousands separators. `total_variance` is a JSON number.
- Leave the files in `/app/exports` exactly as they are — I re-run this every month against a fresh pull, so the script has to work off untouched inputs.

The box is offline; Python 3 and its standard library are available and are all you need."#
        .to_string();

    let blockers = vec![
        Blocker {
            id: "amount_agreement_tolerance".into(),
            kind: BlockerType::MissingParameters,
            question_keywords: vec!["agree".to_string(), "tolerance".to_string()],
            withheld_answer: "Amounts agree when the difference between the ledger amount and the billing total, rounded to two decimals, is at most 0.02 in either direction, inclusive.".into(),
            affects_checks: vec!["test_matched_rows".into()],
        },
        Blocker {
            id: "invoice_status_scope".into(),
            kind: BlockerType::BusinessInformation,
            question_keywords: vec!["status".to_string(), "scope".to_string()],
            withheld_answer: "Only ISSUED and PAID invoices are in scope, compared case-insensitively. VOID and DRAFT invoices are dropped entirely, and ledger rows that point at them are not counted as ledger-only.".into(),
            affects_checks: vec!["test_matched_rows".into(), "test_summary_counts".into()],
        },
        Blocker {
            id: "duplicate_invoice_row_precedence".into(),
            kind: BlockerType::AmbiguousRequirements,
            question_keywords: vec!["duplicate".to_string(), "invoice".to_string()],
            withheld_answer: "For a repeated invoice_id, keep the last occurrence in file order and ignore earlier rows; do not sum them.".into(),
            affects_checks: vec!["test_matched_rows".into()],
        },
        Blocker {
            id: "discrepancy_report_ordering".into(),
            kind: BlockerType::Question,
            question_keywords: vec!["discrepancies".to_string(), "ordered".to_string()],
            withheld_answer: "Sort discrepancies.csv by the absolute difference, largest first, breaking ties by invoice_id ascending.".into(),
            affects_checks: vec!["test_discrepancies_order".into()],
        },
        Blocker {
            id: "total_variance_definition".into(),
            kind: BlockerType::Schema,
            question_keywords: vec!["total".to_string(), "variance".to_string()],
            withheld_answer: "total_variance is the sum of absolute per-row differences over the rows in discrepancies.csv only, rounded to two decimals.".into(),
            affects_checks: vec!["test_summary_counts".into()],
        },
    ];

    HilTask {
        name: "acme/reconcile-march-billing".into(),
        instruction,
        registry: BlockerRegistry::new(blockers),
        check_names: vec![
            "test_matched_rows".into(),
            "test_discrepancies_order".into(),
            "test_summary_counts".into(),
        ],
    }
}

/// Ask rates and Ask-F1 over a batch of attempts (Table 5's reporting).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HilBatchStats {
    /// Fraction of attempts that asked at least one question.
    pub ask_rate: f64,
    /// Mean task success.
    pub task_success_rate: f64,
    /// Mean Ask-F1.
    pub mean_ask_f1: f64,
}

/// Summarize a batch of HiL attempts.
pub fn summarize_attempts(attempts: &[HilAttempt], registry: &BlockerRegistry) -> HilBatchStats {
    if attempts.is_empty() {
        return HilBatchStats {
            ask_rate: 0.0,
            task_success_rate: 0.0,
            mean_ask_f1: 0.0,
        };
    }
    let n = attempts.len() as f64;
    HilBatchStats {
        ask_rate: attempts.iter().filter(|a| a.asked()).count() as f64 / n,
        task_success_rate: attempts.iter().filter(|a| a.task_success).count() as f64 / n,
        mean_ask_f1: attempts.iter().map(|a| a.ask_f1(registry).f1).sum::<f64>() / n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocker_types_match_table6() {
        let task = example_billing_task();
        let kinds: Vec<BlockerType> = task.registry.blockers.iter().map(|b| b.kind).collect();
        assert_eq!(
            kinds,
            vec![
                BlockerType::MissingParameters,
                BlockerType::BusinessInformation,
                BlockerType::AmbiguousRequirements,
                BlockerType::Question,
                BlockerType::Schema,
            ]
        );
    }

    #[test]
    fn ask_human_reveals_matching_blocker() {
        let task = example_billing_task();
        let hit = task
            .registry
            .ask("How much tolerance counts as agreement between the two amounts?")
            .unwrap();
        assert_eq!(hit.id, "amount_agreement_tolerance");
        assert!(hit.withheld_answer.contains("0.02"));
        let miss = task.registry.ask("What is the capital of France?");
        assert!(miss.is_none());
    }

    #[test]
    fn ask_f1_full_coverage() {
        let task = example_billing_task();
        let questions: Vec<String> = vec![
            "How much tolerance counts as agreement between the two amounts?".into(),
            "Which invoice statuses are in scope?".into(),
            "What about a duplicate invoice id?".into(),
            "How should the discrepancies report be ordered?".into(),
            "How is the total variance defined?".into(),
        ];
        let f1 = ask_f1(&questions, &task.registry);
        assert!((f1.recall - 1.0).abs() < 1e-9);
        assert!((f1.precision - 1.0).abs() < 1e-9);
        assert!((f1.f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ask_f1_penalizes_redundant_and_wasted_questions() {
        let task = example_billing_task();
        // 3 of 5 blockers covered; one duplicate question and one wasted.
        let questions: Vec<String> = vec![
            "How much tolerance counts as agreement between the two amounts?".into(),
            "Which invoice statuses are in scope?".into(),
            "How is the total variance defined?".into(),
            "How is the total variance defined?".into(), // redundant
            "What is the capital of France?".into(),     // wasted
        ];
        let f1 = ask_f1(&questions, &task.registry);
        assert!((f1.recall - 0.6).abs() < 1e-9);
        assert!((f1.precision - 3.0 / 5.0).abs() < 1e-9);
        let expected = 2.0 * f1.precision * f1.recall / (f1.precision + f1.recall);
        assert!((f1.f1 - expected).abs() < 1e-9);
    }

    #[test]
    fn derived_variants_need_no_model_calls() {
        let task = example_billing_task();
        let full = task.derive_full_info();
        assert!(full.contains("amount_agreement_tolerance"));
        assert!(full.contains("0.02"));
        assert!(full.contains("ISSUED and PAID"));
        assert_ne!(full, task.instruction);
        assert_eq!(task.derive_no_tool(), task.instruction);
    }

    #[test]
    fn no_tool_attempt_has_zero_ask_f1() {
        let task = example_billing_task();
        let a = HilAttempt {
            questions: vec![],
            had_tool: false,
            task_success: true,
        };
        assert_eq!(a.ask_f1(&task.registry).f1, 0.0);
        assert!(!a.asked());
    }

    #[test]
    fn shaped_reward_combines_success_and_f1() {
        let task = example_billing_task();
        let cfg = ClarificationRewardConfig::default();
        let a = HilAttempt {
            questions: vec![
                "How much tolerance counts as agreement between the two amounts?".into(),
            ],
            had_tool: true,
            task_success: true,
        };
        let f1 = a.ask_f1(&task.registry).f1; // recall 0.2, precision 1.0
        assert!((f1 - 2.0 * 1.0 * 0.2 / 1.2).abs() < 1e-9);
        assert!((a.shaped_reward(&task.registry, &cfg) - (1.0 + f1)).abs() < 1e-9);
    }

    #[test]
    fn batch_stats_shape() {
        let task = example_billing_task();
        let attempts = vec![
            HilAttempt {
                questions: vec!["Which invoice statuses are in scope?".into()],
                had_tool: true,
                task_success: true,
            },
            HilAttempt {
                questions: vec![],
                had_tool: true,
                task_success: false,
            },
        ];
        let stats = summarize_attempts(&attempts, &task.registry);
        assert!((stats.ask_rate - 0.5).abs() < 1e-9);
        assert!((stats.task_success_rate - 0.5).abs() < 1e-9);
        assert!(stats.mean_ask_f1 > 0.0);
    }
}
