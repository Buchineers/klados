//! Reduction rule trait and supporting types.

use std::time::Duration;
use crate::Instance;

/// What a rule does when it fires.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReductionAction {
    /// Parameter-preserving: collapse `remove` into `keep`.
    /// The `remove` label disappears; `keep` now represents both.
    Collapse { keep: u32, remove: u32 },
    /// Parameter-reducing: delete `victim` from the instance.
    /// The victim becomes a forced singleton in the expanded solution.
    Delete { victim: u32 },
}

/// Strategy for selecting among multiple candidates in parameter-reducing rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum VictimStrategy {
    /// Return the first candidate found (fast, deterministic by label order).
    #[default]
    First,
    /// Score all candidates by cascading effect, pick the one that creates
    /// the most new common cherries after deletion.
    MaxCascade,
    /// Pick the last candidate found (reversed label order — for comparison).
    Last,
}

/// Read-only context passed to every rule's `find` method.
pub struct RuleContext<'a> {
    pub instance: &'a Instance,
    pub protected_labels: &'a [u32],
    pub composite_rev: &'a [u32],
    pub victim_strategy: VictimStrategy,
}

impl<'a> RuleContext<'a> {
    /// Check whether a label in current reduced space maps to a protected original label.
    pub fn is_protected(&self, label: u32) -> bool {
        if self.protected_labels.is_empty() {
            return false;
        }
        let orig = self.composite_rev[label as usize];
        self.protected_labels.contains(&orig)
    }
}

/// A single reduction rule.
///
/// Each rule implements `find`, which scans the current instance for one
/// applicable pattern. The pipeline calls rules in priority order and
/// restarts from rule 0 after any rule fires.
pub trait ReductionRule: std::fmt::Debug {
    /// Short human-readable name (used as key in stats map).
    fn name(&self) -> &'static str;

    /// Try to find one application of this rule.
    /// Returns `None` if the rule does not apply to the current instance.
    fn find(&self, ctx: &RuleContext) -> Option<ReductionAction>;
}

/// A record of one rule application (for profiling/debugging).
#[derive(Clone, Debug)]
pub struct RuleEvent {
    pub rule_name: &'static str,
    pub action: ReductionAction,
    /// Labels in original space.
    pub original_labels: Vec<u32>,
    /// Time spent finding + applying this rule.
    pub duration: Duration,
    /// Instance size (leaves) after this rule fired.
    pub leaves_after: u32,
}
