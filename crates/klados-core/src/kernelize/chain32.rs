//! 3-2 chain reduction rule.
//!
//! Proven safe for all t >= 2.
//! See: "A generalized 3-2 chain reduction rule for rooted MAF on multiple trees"

use crate::{NONE, Tree};
use super::rule::{ReductionAction, ReductionRule, RuleContext, VictimStrategy};

/// 3-2 chain reduction: find a 3-2 interceptor configuration (p, q, r) and
/// delete the victim r.
///
/// For 2-tree instances: the original Kelk & Linz formulation.
/// For multi-tree instances (t >= 3): the generalized interceptor configuration.
///
/// Parameter-reducing: each application decreases the MAF by exactly 1.
#[derive(Debug)]
pub struct Chain32Rule {
    /// Whether to apply the rule on multi-tree instances (t >= 3).
    pub allow_multi: bool,
    /// Max distinct cherry partners to consider (2 = classic, usize::MAX = extended).
    pub max_partners: usize,
}

impl ReductionRule for Chain32Rule {
    fn name(&self) -> &'static str {
        "chain-3-2"
    }

    fn find(&self, ctx: &RuleContext) -> Option<ReductionAction> {
        let inst = ctx.instance;
        if inst.num_leaves < 3 {
            return None;
        }

        let max_p = self.max_partners;
        let victim = match ctx.victim_strategy {
            VictimStrategy::First => {
                if inst.num_trees() == 2 {
                    find_32_chain_candidate(&inst.trees[0], &inst.trees[1], inst.num_leaves)
                } else if self.allow_multi && inst.num_trees() >= 3 {
                    find_32_chain_candidate_multi(&inst.trees, inst.num_leaves, max_p)
                } else {
                    None
                }
            }
            VictimStrategy::Last => {
                let candidates = find_all_32_candidates(inst, self.allow_multi, max_p);
                candidates.into_iter().last()
            }
            VictimStrategy::MaxCascade => {
                let candidates = find_all_32_candidates(inst, self.allow_multi, max_p);
                if candidates.len() <= 1 {
                    candidates.into_iter().next()
                } else {
                    candidates
                        .into_iter()
                        .max_by_key(|&v| count_new_cherries_after_delete(inst, v))
                }
            }
        };

        // Refuse to delete a protected label.
        let v = victim?;
        if ctx.is_protected(v) {
            return None;
        }
        Some(ReductionAction::Delete { victim: v })
    }
}

/// Find ALL 3-2 chain candidates in the instance.
fn find_all_32_candidates(inst: &crate::Instance, allow_multi: bool, max_partners: usize) -> Vec<u32> {
    if inst.num_trees() == 2 {
        find_all_32_chain_candidates_2tree(&inst.trees[0], &inst.trees[1], inst.num_leaves)
    } else if allow_multi && inst.num_trees() >= 3 {
        find_all_32_chain_candidates_multi(&inst.trees, inst.num_leaves, max_partners)
    } else {
        Vec::new()
    }
}

/// Count how many new common cherries would appear if `victim` were deleted.
/// This is a cheap heuristic: we check for each pair of remaining leaves whether
/// they would become common cherries after the victim's removal.
fn count_new_cherries_after_delete(inst: &crate::Instance, victim: u32) -> usize {
    let trees = &inst.trees;
    let n = inst.num_leaves;
    let mut count = 0;

    // For each tree, find which leaf would become the new sibling of the victim's
    // sibling after deletion (i.e., the victim's parent gets suppressed).
    // Then check if any new cherry emerges across all trees.

    // Collect: for each tree, what pairs of leaves become siblings after deleting victim?
    // When victim is deleted: victim's parent p is suppressed, victim's sibling s
    // gets connected to victim's grandparent gp. If s is a leaf and gp's other child
    // was already a leaf, they form a new cherry.
    for a in 1..=n {
        if a == victim {
            continue;
        }
        // Check if {a, X} becomes a common cherry for some X after deleting victim
        // This happens if in every tree, after suppressing victim's parent:
        // - a's parent stays the same (a wasn't sibling of victim) → cherry must already exist
        // - OR a was sibling of victim → a now connects to grandparent, check if new sibling is same in all trees
        let mut new_sib_label: Option<u32> = None;
        let mut is_new_cherry = true;

        for tree in trees {
            let node_victim = tree.node_by_label(victim);
            let node_a = tree.node_by_label(a);
            if node_victim == NONE || node_a == NONE {
                is_new_cherry = false;
                break;
            }

            // Is a the sibling of victim in this tree?
            let sib_of_victim = tree.sibling(node_victim);
            if sib_of_victim == Some(node_a) {
                // After deletion, a gets connected to grandparent of victim.
                // a's new sibling is the other child of grandparent (the uncle).
                let parent_victim = tree.parent[node_victim as usize];
                if parent_victim == NONE {
                    is_new_cherry = false;
                    break;
                }
                let gp = tree.parent[parent_victim as usize];
                if gp == NONE {
                    is_new_cherry = false;
                    break;
                }
                // The uncle is the other child of gp (not parent_victim)
                let (gl, gr) = match tree.children(gp) {
                    Some(lr) => lr,
                    None => {
                        is_new_cherry = false;
                        break;
                    }
                };
                let uncle = if gl == parent_victim { gr } else { gl };
                if !tree.is_leaf(uncle) {
                    is_new_cherry = false;
                    break;
                }
                let uncle_label = tree.label[uncle as usize];
                match new_sib_label {
                    None => new_sib_label = Some(uncle_label),
                    Some(prev) if prev != uncle_label => {
                        is_new_cherry = false;
                        break;
                    }
                    _ => {}
                }
            }
            // If a is NOT sibling of victim, deleting victim doesn't change a's cherry status
            // so no new cherry involving a is created in this tree.
        }

        if is_new_cherry && new_sib_label.is_some() {
            count += 1;
        }
    }

    count
}

// ═══════════════════════════════════════════════════════════════
// 3-2 chain reduction (2-tree)
// ═══════════════════════════════════════════════════════════════

/// Find a single 3-2 chain reduction candidate in a 2-tree instance.
fn find_32_chain_candidate(t1: &Tree, t2: &Tree, num_leaves: u32) -> Option<u32> {
    for p in 1..=num_leaves {
        if let Some(victim) = check_32_at_pivot(t1, t2, p, num_leaves) {
            return Some(victim);
        }
    }
    None
}

/// Find ALL 3-2 chain candidates in a 2-tree instance.
fn find_all_32_chain_candidates_2tree(t1: &Tree, t2: &Tree, num_leaves: u32) -> Vec<u32> {
    let mut candidates = Vec::new();
    for p in 1..=num_leaves {
        if let Some(victim) = check_32_at_pivot(t1, t2, p, num_leaves) {
            if !candidates.contains(&victim) {
                candidates.push(victim);
            }
        }
    }
    candidates
}

/// Check 3-2 pattern at pivot p for 2-tree case.
fn check_32_at_pivot(t1: &Tree, t2: &Tree, p: u32, _num_leaves: u32) -> Option<u32> {
    let node_p_t1 = t1.node_by_label(p);
    let node_p_t2 = t2.node_by_label(p);
    if node_p_t1 == NONE || node_p_t2 == NONE {
        return None;
    }

    let sib_t1 = match t1.sibling(node_p_t1) {
        Some(s) if t1.is_leaf(s) => s,
        _ => return None,
    };
    let sib_t2 = match t2.sibling(node_p_t2) {
        Some(s) if t2.is_leaf(s) => s,
        _ => return None,
    };

    let q = t1.label[sib_t1 as usize];
    let r = t2.label[sib_t2 as usize];
    if q == r {
        return None;
    }

    // Forward: check parent(q in T2) == grandparent(r in T2)
    let node_q_t2 = t2.node_by_label(q);
    if node_q_t2 != NONE {
        let parent_q_t2 = t2.parent[node_q_t2 as usize];
        let parent_r_t2 = t2.parent[sib_t2 as usize];
        if parent_r_t2 != NONE {
            let gp_r_t2 = t2.parent[parent_r_t2 as usize];
            if parent_q_t2 != NONE && gp_r_t2 != NONE && parent_q_t2 == gp_r_t2 {
                return Some(r);
            }
        }
    }

    // Symmetric: check parent(r in T1) == grandparent(q in T1)
    let node_r_t1 = t1.node_by_label(r);
    if node_r_t1 != NONE {
        let parent_r_t1 = t1.parent[node_r_t1 as usize];
        let parent_q_t1 = t1.parent[sib_t1 as usize];
        if parent_q_t1 != NONE {
            let gp_q_t1 = t1.parent[parent_q_t1 as usize];
            if parent_r_t1 != NONE && gp_q_t1 != NONE && parent_r_t1 == gp_q_t1 {
                return Some(q);
            }
        }
    }

    None
}

// ═══════════════════════════════════════════════════════════════
// 3-2 chain reduction (multi-tree, proven for all t >= 2)
// ═══════════════════════════════════════════════════════════════

/// Find a single 3-2 chain reduction candidate across multiple trees.
fn find_32_chain_candidate_multi(trees: &[Tree], num_leaves: u32, max_partners: usize) -> Option<u32> {
    for p in 1..=num_leaves {
        if let Some(victim) = check_32_multi_at_pivot(trees, p, max_partners) {
            return Some(victim);
        }
    }
    None
}

/// Find ALL 3-2 chain candidates across multiple trees.
fn find_all_32_chain_candidates_multi(trees: &[Tree], num_leaves: u32, max_partners: usize) -> Vec<u32> {
    let mut candidates = Vec::new();
    for p in 1..=num_leaves {
        if let Some(victim) = check_32_multi_at_pivot(trees, p, max_partners) {
            if !candidates.contains(&victim) {
                candidates.push(victim);
            }
        }
    }
    candidates
}

/// Check 3-2 multi-tree pattern at pivot p.
fn check_32_multi_at_pivot(trees: &[Tree], p: u32, max_partners: usize) -> Option<u32> {
    let mut siblings: Vec<u32> = Vec::with_capacity(trees.len());
    for tree in trees {
        let node_p = tree.node_by_label(p);
        if node_p == NONE {
            return None;
        }
        match tree.sibling(node_p) {
            Some(s) if tree.is_leaf(s) => {
                siblings.push(tree.label[s as usize]);
            }
            _ => return None,
        }
    }

    let mut unique_sibs: Vec<u32> = siblings.clone();
    unique_sibs.sort_unstable();
    unique_sibs.dedup();

    if unique_sibs.len() < 2 || unique_sibs.len() > max_partners {
        return None;
    }

    // For each distinct partner as potential victim, check if the 3-2
    // interceptor condition holds. With 2 partners this is the classic rule.
    // With 3+ partners (only possible for t >= 3), we try each partner as victim:
    // trees where p's partner IS the victim must satisfy the interceptor condition
    // with some other partner as keeper. Trees where p's partner is NOT the victim
    // are automatically in "cherry state" (they don't have the victim as partner).
    for &victim in &unique_sibs {
        if try_32_delete_multi_partner(trees, &siblings, &unique_sibs, victim) {
            return Some(victim);
        }
    }
    None
}

/// Check if `victim` can be deleted with multiple possible keepers.
///
/// For each tree where p's partner is the victim, the interceptor condition must
/// hold with SOME keeper (any non-victim partner). For trees where p's partner
/// is not the victim, they're automatically in cherry state.
fn try_32_delete_multi_partner(
    trees: &[Tree],
    siblings: &[u32],
    _all_partners: &[u32],
    victim: u32,
) -> bool {
    let mut interceptor_exists = false;

    for (t_idx, tree) in trees.iter().enumerate() {
        if siblings[t_idx] != victim {
            // This tree has a non-victim partner → cherry state, OK
            continue;
        }

        // This tree has {p, victim} as cherry. Need interceptor condition
        // with some keeper: par(keeper in tree) == grandpar(victim in tree)
        let node_victim = tree.node_by_label(victim);
        if node_victim == NONE {
            return false;
        }
        let parent_victim = tree.parent[node_victim as usize];
        if parent_victim == NONE {
            return false;
        }
        let gp_victim = tree.parent[parent_victim as usize];
        if gp_victim == NONE {
            return false;
        }

        // Check if ANY non-victim partner satisfies the interceptor
        let mut found_keeper = false;
        for &other_partner in siblings.iter() {
            if other_partner == victim {
                continue;
            }
            let node_keeper = tree.node_by_label(other_partner);
            if node_keeper == NONE {
                continue;
            }
            let parent_keeper = tree.parent[node_keeper as usize];
            if parent_keeper != NONE && parent_keeper == gp_victim {
                found_keeper = true;
                break;
            }
        }

        if !found_keeper {
            return false;
        }
        interceptor_exists = true;
    }

    interceptor_exists
}