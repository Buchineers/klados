//! Kernelization pipeline for MAF instances.
//!
//! Applies reduction rules iteratively until fixpoint. Rules are applied in
//! priority order; after any rule fires, the pipeline restarts from rule 0.
//! This matches the strategy prescribed by Kelk et al. and guarantees a
//! fixpoint (every rule application removes >= 1 leaf).
//!
//! ## Reduction rules (in priority order):
//! 1. **Cherry** (subtree) — collapse common cherries (parameter-preserving)
//! 2. **Chain** — compress common caterpillar chains (parameter-preserving)
//! 3. **3-2 chain** — delete leaves in interceptor configurations (parameter-reducing)

mod chain;
mod chain32;
mod cherry;
mod expansion;
mod helpers;
mod instance_ops;
pub mod rule;
mod short_chain;
mod stats;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use fixedbitset::FixedBitSet;
use crate::Instance;

pub use expansion::{expand_solution, build_rep_to_all};
pub use instance_ops::{reduce_instance, restrict_instance_simple, compose_reverse_maps};
pub use rule::{ReductionAction, ReductionRule, RuleContext, RuleEvent, VictimStrategy};
pub use stats::{KernelizeStats, build_surviving_taxa, print_stats, print_taxa_detail};

use chain::ChainRule;
use chain32::Chain32Rule;
use cherry::CherryRule;
use short_chain::{Rule3CrossedCherry, Rule4ExternalCherry};

/// Which reduction rules are enabled.
#[derive(Clone, Debug)]
pub struct KernelizeConfig {
    pub subtree: bool,
    pub chain: bool,
    pub chain32: bool,
    /// Enable 3-2 chain reduction for multi-tree instances (t >= 3).
    /// Proven safe for all t >= 2.
    pub chain32_multi: bool,
    /// Labels that must survive kernelization unchanged (used by cluster decomposition
    /// to protect ghost representative labels).
    pub protected_labels: Vec<u32>,
    /// Strategy for selecting among multiple parameter-reducing candidates.
    pub victim_strategy: VictimStrategy,
}

impl Default for KernelizeConfig {
    fn default() -> Self {
        Self {
            subtree: true,
            chain: true,
            chain32: true,
            chain32_multi: true,
            protected_labels: Vec::new(),
            victim_strategy: VictimStrategy::First,
        }
    }
}

/// Result of kernelization.
pub struct KernelizeResult {
    pub instance: Instance,
    pub stats: KernelizeStats,
    /// Maps reduced label -> original label.
    pub reverse_map: Vec<u32>,
    /// All collapses in original label space (for solution expansion).
    pub collapses_original: Vec<(u32, Vec<u32>)>,
    /// Parameter reduction from parameter-reducing rule deletions.
    pub param_reduction: usize,
    /// Ordered trace of all rule applications (for profiling).
    pub trace: Vec<RuleEvent>,
}

/// Build the ordered list of active reduction rules from config.
///
/// Priority order (cheapest first, restart from top after any hit):
/// 1. Cherry (subtree) — parameter-preserving, O(n) per scan
/// 2. Chain — parameter-preserving, O(n) per scan
/// 3. 3-2 chain — parameter-reducing, proven for all t >= 2
/// 4. Rule 3 (crossed cherry) — parameter-reducing, t = 2 only
/// 5. Rule 4 (external cherry) — parameter-reducing, t = 2 only
fn build_rules(config: &KernelizeConfig) -> Vec<Box<dyn ReductionRule>> {
    let mut rules: Vec<Box<dyn ReductionRule>> = Vec::new();
    if config.subtree {
        rules.push(Box::new(CherryRule));
    }
    if config.chain {
        rules.push(Box::new(ChainRule));
    }
    if config.chain32 || config.chain32_multi {
        rules.push(Box::new(Chain32Rule {
            allow_multi: config.chain32_multi,
        }));
    }
    // Rules 3 and 4 (Kelk & Linz 2020): crossed/external cherry on common 3-chains.
    // For ROOTED MAF, these are subsumed by the 3-2 interceptor rule.
    // Kept as code for reference but disabled — the 3-2 rule catches all their patterns.
    // rules.push(Box::new(Rule3CrossedCherry));
    // rules.push(Box::new(Rule4ExternalCherry));
    rules
}

/// Run the full kernelization pipeline.
pub fn kernelize(instance: &Instance, config: &KernelizeConfig) -> KernelizeResult {
    let original_leaves = instance.num_leaves;
    let rules = build_rules(config);

    let mut reduced = instance.clone();
    let mut composite_rev: Vec<u32> = (0..=instance.num_leaves).collect();
    let mut all_collapses_original: Vec<(u32, Vec<u32>)> = Vec::new();
    let mut deleted_labels_original: Vec<u32> = Vec::new();
    let mut rule_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut trace: Vec<RuleEvent> = Vec::new();

    // Priority-restart loop: try rules in order, restart from rule[0] on any hit.
    'outer: loop {
        let ctx = RuleContext {
            instance: &reduced,
            protected_labels: &config.protected_labels,
            composite_rev: &composite_rev,
            victim_strategy: config.victim_strategy,
        };

        for rule in &rules {
            let find_start = std::time::Instant::now();
            let found = rule.find(&ctx);
            if let Some(action) = found {
                *rule_counts.entry(rule.name()).or_default() += 1;

                let original_labels = match &action {
                    ReductionAction::Collapse { keep, remove } => {
                        vec![
                            composite_rev[*keep as usize],
                            composite_rev[*remove as usize],
                        ]
                    }
                    ReductionAction::Delete { victim } => {
                        vec![composite_rev[*victim as usize]]
                    }
                };

                match action {
                    ReductionAction::Collapse { keep: _, remove } => {
                        let orig_keep = original_labels[0];
                        let orig_remove = original_labels[1];
                        all_collapses_original.push((orig_keep, vec![orig_remove]));

                        let mut keep_set = FixedBitSet::with_capacity(
                            reduced.num_leaves as usize + 1,
                        );
                        for lbl in 1..=reduced.num_leaves {
                            keep_set.insert(lbl as usize);
                        }
                        keep_set.set(remove as usize, false);
                        let (r, rev) = restrict_instance_simple(&reduced, &keep_set);
                        composite_rev = compose_reverse_maps(&composite_rev, &rev, r.num_leaves);
                        reduced = r;
                    }
                    ReductionAction::Delete { victim } => {
                        deleted_labels_original.push(original_labels[0]);

                        let mut keep_set = FixedBitSet::with_capacity(
                            reduced.num_leaves as usize + 1,
                        );
                        for lbl in 1..=reduced.num_leaves {
                            keep_set.insert(lbl as usize);
                        }
                        keep_set.set(victim as usize, false);
                        let (r, rev) = restrict_instance_simple(&reduced, &keep_set);
                        composite_rev = compose_reverse_maps(&composite_rev, &rev, r.num_leaves);
                        reduced = r;
                    }
                }

                trace.push(RuleEvent {
                    rule_name: rule.name(),
                    action: match &trace.last() {
                        // Re-derive action with original labels for the trace
                        _ => if original_labels.len() == 2 {
                            ReductionAction::Collapse {
                                keep: original_labels[0],
                                remove: original_labels[1],
                            }
                        } else {
                            ReductionAction::Delete {
                                victim: original_labels[0],
                            }
                        }
                    },
                    original_labels,
                    duration: find_start.elapsed(),
                    leaves_after: reduced.num_leaves,
                });

                continue 'outer; // restart from highest-priority rule
            }
        }
        break; // fixpoint reached — no rule fired
    }

    let surviving_taxa = build_surviving_taxa(
        &composite_rev,
        &all_collapses_original,
        reduced.num_leaves,
        original_leaves,
    );

    let stats = KernelizeStats {
        original_leaves,
        reduced_leaves: reduced.num_leaves,
        rule_counts,
        deleted_labels: deleted_labels_original.clone(),
        surviving_taxa,
    };

    KernelizeResult {
        instance: reduced,
        stats,
        reverse_map: composite_rev,
        collapses_original: all_collapses_original,
        param_reduction: deleted_labels_original.len(),
        trace,
    }
}

/// Run kernelization with multiple victim selection strategies and return the best result.
///
/// For t >= 3 instances, different 3-2 victim choices can lead to different kernel sizes.
/// This runs all strategies and picks the smallest kernel. For t = 2, strategies produce
/// identical results so this just runs once.
pub fn kernelize_best(instance: &Instance, config: &KernelizeConfig) -> KernelizeResult {
    // For 2-tree instances, strategies are identical — skip the extra work
    if instance.num_trees() <= 2 {
        return kernelize(instance, config);
    }

    let strategies = [VictimStrategy::First, VictimStrategy::Last, VictimStrategy::MaxCascade];
    let mut best: Option<KernelizeResult> = None;

    for &strategy in &strategies {
        let mut cfg = config.clone();
        cfg.victim_strategy = strategy;
        let result = kernelize(instance, &cfg);

        let is_better = match &best {
            None => true,
            Some(prev) => result.stats.reduced_leaves < prev.stats.reduced_leaves,
        };

        if is_better {
            best = Some(result);
        }
    }

    best.unwrap()
}
