//! Tests for the kernelization pipeline and individual rules.

use crate::{Instance, Tree};
use super::*;
use super::rule::{RuleContext, VictimStrategy};
use super::cherry::CherryRule;
use super::chain32::Chain32Rule;

/// Build an Instance from Newick strings (without trailing semicolons).
/// Uses PACE format parsing internally.
fn instance_from_newick(newicks: &[&str]) -> Instance {
    use pace26io::binary_tree::IndexedBinTreeBuilder;
    use pace26io::pace::simplified::Instance as PaceInstance;
    use std::io::BufReader;

    let num_trees = newicks.len();

    // Count max leaf label to determine num_leaves
    let num_leaves: usize = newicks[0]
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().unwrap())
        .max()
        .unwrap_or(0);

    let tree_lines: Vec<String> = newicks.iter().map(|nw| format!("{};", nw)).collect();
    let input = format!("#p {} {}\n{}", num_trees, num_leaves, tree_lines.join("\n"));

    let mut builder = IndexedBinTreeBuilder::default();
    let reader = BufReader::new(input.as_bytes());
    let pace = PaceInstance::try_read(reader, &mut builder).unwrap();

    let n = pace.num_leaves as u32;
    let trees: Vec<Tree> = pace
        .trees
        .iter()
        .map(|t| Tree::from_cursor(t.top_down(), n))
        .collect();

    Instance::new(trees, n)
}

fn make_ctx<'a>(inst: &'a Instance, rev: &'a [u32]) -> RuleContext<'a> {
    RuleContext {
        instance: inst,
        protected_labels: &[],
        composite_rev: rev,
        victim_strategy: VictimStrategy::First,
    }
}

// ═══════════════════════════════════════════════════════════════
// Cherry rule tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn cherry_finds_common_cherry() {
    let inst = instance_from_newick(&["((1,2),(3,4))", "((1,2),(3,4))"]);
    let rev: Vec<u32> = (0..=inst.num_leaves).collect();
    let ctx = make_ctx(&inst, &rev);
    let rule = CherryRule;
    let action = rule.find(&ctx);
    assert!(matches!(action, Some(ReductionAction::Collapse { keep: 1, remove: 2 })));
}

#[test]
fn cherry_no_match_when_different() {
    let inst = instance_from_newick(&["((1,2),(3,4))", "((1,4),(2,3))"]);
    let rev: Vec<u32> = (0..=inst.num_leaves).collect();
    let ctx = make_ctx(&inst, &rev);
    let rule = CherryRule;
    assert!(rule.find(&ctx).is_none());
}

// ═══════════════════════════════════════════════════════════════
// 3-2 chain rule tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn chain32_finds_interceptor() {
    // tiny01: (((5,6),(3,4)),(1,2)) and (((((4,2),1),5),3),6)
    // The 3-2 rule should find a victim to delete.
    let inst = instance_from_newick(&[
        "(((5,6),(3,4)),(1,2))",
        "(((((4,2),1),5),3),6)",
    ]);
    let rev: Vec<u32> = (0..=inst.num_leaves).collect();
    let ctx = make_ctx(&inst, &rev);
    let rule = Chain32Rule { allow_multi: false, max_partners: usize::MAX };
    let action = rule.find(&ctx);
    assert!(matches!(action, Some(ReductionAction::Delete { .. })));
}

// ═══════════════════════════════════════════════════════════════
// Pipeline integration tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn tiny01_kernelizes_to_4_leaves() {
    let inst = instance_from_newick(&[
        "(((5,6),(3,4)),(1,2))",
        "(((((4,2),1),5),3),6)",
    ]);
    let config = KernelizeConfig::default();
    let result = kernelize(&inst, &config);
    assert_eq!(result.stats.original_leaves, 6);
    assert_eq!(result.stats.reduced_leaves, 4);
    assert_eq!(result.stats.subtree_removed(), 1);
    assert_eq!(result.stats.chain32_removed(), 1);
    assert_eq!(result.param_reduction, 1);
}

#[test]
fn tiny05_no_reduction() {
    let inst = instance_from_newick(&[
        "((1,2),(3,4))",
        "((1,4),(2,3))",
    ]);
    let config = KernelizeConfig::default();
    let result = kernelize(&inst, &config);
    assert_eq!(result.stats.original_leaves, 4);
    assert_eq!(result.stats.reduced_leaves, 4);
    assert_eq!(result.stats.total_removed(), 0);
    assert_eq!(result.param_reduction, 0);
}

#[test]
fn disabled_rules_are_skipped() {
    let inst = instance_from_newick(&[
        "(((5,6),(3,4)),(1,2))",
        "(((((4,2),1),5),3),6)",
    ]);
    let config = KernelizeConfig {
        subtree: false,
        chain: false,
        chain32: false,
        chain32_multi: false,
        ..KernelizeConfig::default()
    };
    let result = kernelize(&inst, &config);
    // With all rules disabled, no reduction should happen
    assert_eq!(result.stats.reduced_leaves, 6);
    assert_eq!(result.stats.total_removed(), 0);
}

#[test]
fn only_chain32_disabled() {
    let inst = instance_from_newick(&[
        "(((5,6),(3,4)),(1,2))",
        "(((((4,2),1),5),3),6)",
    ]);
    let config = KernelizeConfig {
        chain32: false,
        chain32_multi: false,
        ..KernelizeConfig::default()
    };
    let result = kernelize(&inst, &config);
    // Without 3-2, cherry can't fire (no common cherry in original),
    // so no reduction should happen
    assert_eq!(result.stats.reduced_leaves, 6);
    assert_eq!(result.param_reduction, 0);
}
