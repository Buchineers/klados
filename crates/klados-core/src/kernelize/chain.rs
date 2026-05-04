//! Chain reduction rule.

use crate::{NONE, Tree};
use super::helpers::*;
use super::rule::{ReductionAction, ReductionRule, RuleContext};

/// Chain reduction: compress common caterpillar chains.
///
/// For 2-tree instances, uses Kelk-style 4-chain walk (handles pendancy).
/// For multi-tree instances, uses sequence-based chain detection with
/// truncation length r = t + 1.
///
/// Parameter-preserving.
#[derive(Debug)]
pub struct ChainRule;

impl ReductionRule for ChainRule {
    fn name(&self) -> &'static str {
        "chain"
    }

    fn find(&self, ctx: &RuleContext) -> Option<ReductionAction> {
        let inst = ctx.instance;
        let found: Option<(u32, u32)> = if inst.num_trees() == 2 {
            find_4chain_candidate(&inst.trees[0], &inst.trees[1], inst.num_leaves)
        } else {
            let truncate_to = inst.trees.len() + 1;
            let collapses = find_common_chains(&inst.trees, inst.num_leaves, truncate_to);
            collapses.into_iter().next().and_then(|(rep, removed)| {
                removed.into_iter().next().map(|victim| (victim, rep))
            })
        };

        let (victim, chain_neighbor) = found?;

        // Refuse to absorb a protected label.
        if ctx.is_protected(victim) {
            return None;
        }

        // Chain reduction is a collapse: victim is absorbed by chain_neighbor.
        // We return Collapse with keep=chain_neighbor, remove=victim.
        Some(ReductionAction::Collapse {
            keep: chain_neighbor,
            remove: victim,
        })
    }
}

// ═══════════════════════════════════════════════════════════════
// 4-chain reduction (Kelk-style, 2-tree)
// ═══════════════════════════════════════════════════════════════

/// Find a single 4-chain reduction candidate (2-tree only).
fn find_4chain_candidate(t1: &Tree, t2: &Tree, num_leaves: u32) -> Option<(u32, u32)> {
    for x in 1..=num_leaves {
        let x_t1 = t1.node_by_label(x);
        let x_t2 = t2.node_by_label(x);
        if x_t1 == NONE || x_t2 == NONE {
            continue;
        }

        let p_t1 = t1.parent[x_t1 as usize];
        let p_t2 = t2.parent[x_t2 as usize];
        if p_t1 == NONE || p_t2 == NONE {
            continue;
        }
        if num_leaf_children(t1, p_t1) != 1 || num_leaf_children(t2, p_t2) != 1 {
            continue;
        }

        let at_t1 = non_leaf_child(t1, p_t1);
        let at_t2 = non_leaf_child(t2, p_t2);
        let (at_t1, at_t2) = match (at_t1, at_t2) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        if num_leaf_children(t1, at_t1) != 1 || num_leaf_children(t2, at_t2) != 1 {
            continue;
        }

        let c1 = leaf_child(t1, at_t1);
        let c2 = leaf_child(t2, at_t2);
        let (c1, c2) = match (c1, c2) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        let lbl1 = t1.label[c1 as usize];
        let lbl2 = t2.label[c2 as usize];
        if lbl1 != lbl2 {
            continue;
        }

        let at_t1 = non_leaf_child(t1, at_t1);
        let at_t2 = non_leaf_child(t2, at_t2);
        let (at_t1, at_t2) = match (at_t1, at_t2) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };

        let nlc1 = num_leaf_children(t1, at_t1);
        let nlc2 = num_leaf_children(t2, at_t2);
        if nlc1 == 0 && nlc2 == 0 {
            continue;
        }

        let (at_t1_next, at_t2_next, chain3_label);
        if nlc1 == 1 && nlc2 == 1 {
            let lc1 = leaf_child(t1, at_t1).unwrap();
            let lc2 = leaf_child(t2, at_t2).unwrap();
            if t1.label[lc1 as usize] != t2.label[lc2 as usize] {
                continue;
            }
            chain3_label = t1.label[lc1 as usize];
            at_t1_next = non_leaf_child(t1, at_t1);
            at_t2_next = non_leaf_child(t2, at_t2);
        } else if nlc1 == 1 && nlc2 == 2 {
            let lc1 = leaf_child(t1, at_t1).unwrap();
            let lbl = t1.label[lc1 as usize];
            if !has_leaf_child_with_label(t2, at_t2, lbl) {
                continue;
            }
            chain3_label = lbl;
            at_t1_next = non_leaf_child(t1, at_t1);
            at_t2_next = None;
        } else if nlc1 == 2 && nlc2 == 1 {
            let lc2 = leaf_child(t2, at_t2).unwrap();
            let lbl = t2.label[lc2 as usize];
            if !has_leaf_child_with_label(t1, at_t1, lbl) {
                continue;
            }
            chain3_label = lbl;
            at_t1_next = None;
            at_t2_next = non_leaf_child(t2, at_t2);
        } else {
            continue;
        }

        let _ = chain3_label;

        let check_t1 = at_t1_next.unwrap_or(at_t1);
        let check_t2 = at_t2_next.unwrap_or(at_t2);

        for y in 1..=num_leaves {
            if y == x || y == lbl1 || y == chain3_label {
                continue;
            }
            if has_leaf_child_with_label(t1, check_t1, y)
                && has_leaf_child_with_label(t2, check_t2, y)
            {
                return Some((y, x));
            }
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════
// Chain reduction (multi-tree)
// ═══════════════════════════════════════════════════════════════

/// Find common chains across all trees and return collapses.
fn find_common_chains(trees: &[Tree], num_leaves: u32, truncate_to: usize) -> Vec<(u32, Vec<u32>)> {
    let n = num_leaves as usize;
    if n < 5 || trees.len() < 2 {
        return Vec::new();
    }

    let all_chains: Vec<Vec<Vec<u32>>> = trees.iter().map(|t| extract_chains(t, n)).collect();

    let mut collapses = Vec::new();
    'outer: for chain in &all_chains[0] {
        let mut common_len = chain.len();

        for other_chains in &all_chains[1..] {
            let best_match = other_chains
                .iter()
                .filter_map(|oc| {
                    if oc.starts_with(chain.as_slice()) || chain.starts_with(oc.as_slice()) {
                        Some(chain.len().min(oc.len()))
                    } else {
                        None
                    }
                })
                .max();
            match best_match {
                Some(len) => common_len = common_len.min(len),
                None => continue 'outer,
            }
        }

        if common_len <= truncate_to {
            continue;
        }

        let representative = chain[0];
        let removed: Vec<u32> = chain[truncate_to..common_len].to_vec();
        if !removed.is_empty() {
            collapses.push((representative, removed));
        }
    }

    collapses
}

/// Extract all maximal chains from a tree.
fn extract_chains(tree: &Tree, n: usize) -> Vec<Vec<u32>> {
    let mut chains = Vec::new();
    let mut visited = vec![false; tree.num_nodes()];

    for start_node in tree.pre_order() {
        if tree.is_leaf(start_node) || visited[start_node as usize] {
            continue;
        }

        let mut chain = Vec::new();
        let mut cur = start_node;

        loop {
            if tree.is_leaf(cur) {
                break;
            }

            let (left, right) = match tree.children(cur) {
                Some(lr) => lr,
                None => break,
            };

            let (leaf_child, next_node) = if tree.is_leaf(left) && !tree.is_leaf(right) {
                (left, right)
            } else if tree.is_leaf(right) && !tree.is_leaf(left) {
                (right, left)
            } else {
                break;
            };

            let lbl = tree.label[leaf_child as usize];
            if lbl > 0 && (lbl as usize) <= n {
                chain.push(lbl);
                visited[leaf_child as usize] = true;
            }
            cur = next_node;
        }

        if !tree.is_leaf(cur) {
            if let Some((left, right)) = tree.children(cur) {
                if tree.is_leaf(left) && tree.is_leaf(right) {
                    let ll = tree.label[left as usize];
                    let rl = tree.label[right as usize];
                    if ll > 0 {
                        chain.push(ll);
                    }
                    if rl > 0 {
                        chain.push(rl);
                    }
                }
            }
        }

        if chain.len() >= 3 {
            chains.push(chain);
        }
    }

    chains
}
