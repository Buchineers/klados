//! Tests for the kernelization pipeline and individual rules.

use super::chain32::Chain32Rule;
use super::cherry::CherryRule;
use super::rule::{RuleContext, VictimStrategy};
use super::*;
use crate::{Instance, Tree};

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
    assert!(matches!(
        action,
        Some(ReductionAction::Collapse { keep: 1, remove: 2 })
    ));
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
    let inst = instance_from_newick(&["(((5,6),(3,4)),(1,2))", "(((((4,2),1),5),3),6)"]);
    let rev: Vec<u32> = (0..=inst.num_leaves).collect();
    let ctx = make_ctx(&inst, &rev);
    let rule = Chain32Rule {
        allow_multi: false,
        max_partners: usize::MAX,
    };
    let action = rule.find(&ctx);
    assert!(matches!(action, Some(ReductionAction::Delete { .. })));
}

// ═══════════════════════════════════════════════════════════════
// Pipeline integration tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn tiny01_kernelizes_to_4_leaves() {
    let inst = instance_from_newick(&["(((5,6),(3,4)),(1,2))", "(((((4,2),1),5),3),6)"]);
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
    let inst = instance_from_newick(&["((1,2),(3,4))", "((1,4),(2,3))"]);
    let config = KernelizeConfig::default();
    let result = kernelize(&inst, &config);
    assert_eq!(result.stats.original_leaves, 4);
    assert_eq!(result.stats.reduced_leaves, 4);
    assert_eq!(result.stats.total_removed(), 0);
    assert_eq!(result.param_reduction, 0);
}

#[test]
fn disabled_rules_are_skipped() {
    let inst = instance_from_newick(&["(((5,6),(3,4)),(1,2))", "(((((4,2),1),5),3),6)"]);
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
    let inst = instance_from_newick(&["(((5,6),(3,4)),(1,2))", "(((((4,2),1),5),3),6)"]);
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

// ═══════════════════════════════════════════════════════════════
// Protected-label tests
// ═══════════════════════════════════════════════════════════════

fn make_ctx_protected<'a>(
    inst: &'a Instance,
    rev: &'a [u32],
    protected: &'a [u32],
) -> RuleContext<'a> {
    RuleContext {
        instance: inst,
        protected_labels: protected,
        composite_rev: rev,
        victim_strategy: VictimStrategy::First,
    }
}

#[test]
fn cherry_respects_protected_remove() {
    // (((1,2),(3,4)) both trees — {1,2} and {3,4} are cherries.
    // Protect 2: the rule should swap and collapse {2,1} instead of {1,2},
    // i.e. keep=1, remove=2 becomes keep=2, remove=1.
    let inst = instance_from_newick(&["((1,2),(3,4))", "((1,2),(3,4))"]);
    let rev: Vec<u32> = (0..=inst.num_leaves).collect();
    let protected = [2];
    let ctx = make_ctx_protected(&inst, &rev, &protected);
    let rule = CherryRule;
    let action = rule.find(&ctx);
    assert!(matches!(
        action,
        Some(ReductionAction::Collapse { keep: 2, remove: 1 })
    ));
}

#[test]
fn cherry_refuses_both_protected() {
    // Same cherries, but protect both {1, 2}. The swap won't help;
    // neither can be safely removed. Rule must return None.
    let inst = instance_from_newick(&["((1,2),(3,4))", "((1,2),(3,4))"]);
    let rev: Vec<u32> = (0..=inst.num_leaves).collect();
    let protected = [1, 2];
    let ctx = make_ctx_protected(&inst, &rev, &protected);
    let rule = CherryRule;
    assert!(rule.find(&ctx).is_none());
}

#[test]
fn chain32_refuses_protected_victim() {
    // tiny01 has a 3-2 chain that deletes a victim.
    // Protect the victim — the rule should skip.
    let inst = instance_from_newick(&["(((5,6),(3,4)),(1,2))", "(((((4,2),1),5),3),6)"]);
    let rev: Vec<u32> = (0..=inst.num_leaves).collect();

    // First, find which label is the victim without protection.
    let ctx = make_ctx(&inst, &rev);
    let rule = Chain32Rule {
        allow_multi: false,
        max_partners: usize::MAX,
    };
    let action = rule.find(&ctx);
    assert!(matches!(action, Some(ReductionAction::Delete { .. })));
    let victim = match action.unwrap() {
        ReductionAction::Delete { victim } => victim,
        _ => panic!("expected Delete"),
    };

    // Now protect that victim — rule should refuse.
    let protected = [victim];
    let ctx_prot = make_ctx_protected(&inst, &rev, &protected);
    assert!(rule.find(&ctx_prot).is_none());
}

#[test]
fn pipeline_preserves_protected_labels() {
    // tiny01 kernelizes from 6 to 4 leaves with Cherry + Chain32.
    // Protect one of the labels that would be collapsed/deleted.
    // The pipeline should leave it alone and produce a larger kernel.
    let inst = instance_from_newick(&["(((5,6),(3,4)),(1,2))", "(((((4,2),1),5),3),6)"]);
    let config = KernelizeConfig {
        protected_labels: vec![1, 2, 3, 4, 5, 6], // protect ALL leaves
        ..KernelizeConfig::default()
    };
    let result = kernelize(&inst, &config);
    // With all labels protected, no reduction should fire.
    assert_eq!(result.stats.reduced_leaves, 6);
    assert_eq!(result.stats.total_removed(), 0);

    // Now protect just the cherry-victim label and verify the cherry swaps
    // keep/remove (keeping the protected label) and still fires.
    // The default tiny01 run: cherry collapses {1,2} with keep=1, remove=2.
    // With label 2 protected, the rule swaps to keep=2, remove=1 → cherry still fires.
    // Then chain32 fires → final kernel is 4 leaves, but the protected label
    // survives as the representative.
    let config = KernelizeConfig {
        protected_labels: vec![2], // protect label 2 (default cherry remove)
        ..KernelizeConfig::default()
    };
    let result = kernelize(&inst, &config);
    // Cherry fires with roles swapped (keep=2, remove=1): 6 → 5 leaves.
    // Chain32 fires: 5 → 4 leaves. Protected label 2 survives.
    assert_eq!(result.stats.reduced_leaves, 4);
    assert_eq!(result.stats.subtree_removed(), 1); // cherry still fired
    assert_eq!(result.stats.chain32_removed(), 1); // chain32 still fired

    // Verify that the protected label 2 survives as a representative in the
    // reverse map (meaning it wasn't removed from the instance).
    let prot_label_present = result.reverse_map.contains(&2);
    assert!(
        prot_label_present,
        "protected label 2 should appear in the reverse map"
    );
}
