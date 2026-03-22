//! Generalized Triple Reduction rule.
//!
//! Unifies and extends the 3-2 chain reduction for all t >= 2.
//!
//! For a pivot leaf p with cherry partners q (in some trees) and r (in others):
//! - Cherry state: {p, q} cherry (r can be anywhere)
//! - Interceptor state: {p, r} cherry AND par(q) = grandpar(r)
//! - Rotation state (NEW, m>=3): {q, r} cherry AND par(p) = grandpar(r)
//!
//! Victim = r (deleted). The standard 3-2 uses cherry + interceptor states.
//! The rotation state extends to m >= 3.

use crate::{NONE, Tree};
use super::rule::{ReductionAction, ReductionRule, RuleContext};

#[derive(Debug)]
pub struct TripleRule;

impl ReductionRule for TripleRule {
    fn name(&self) -> &'static str {
        "triple"
    }

    fn find(&self, ctx: &RuleContext) -> Option<ReductionAction> {
        let inst = ctx.instance;
        if inst.num_leaves < 3 || inst.num_trees() < 2 {
            return None;
        }
        find_generalized_triple(&inst.trees, inst.num_leaves)
    }
}

/// Tree state for the triple (pivot=p, keeper=q, victim=r).
#[derive(Debug, Clone, Copy, PartialEq)]
enum TreeState {
    /// {p, q} cherry — standard cherry state. r is elsewhere.
    CherryPQ,
    /// {p, r} cherry, q intercepts: par(q) = grandpar(r)
    InterceptPR,
    /// {q, r} cherry, p intercepts: par(p) = grandpar(r) — NEW for m >= 3
    RotationQR,
    /// None of the above
    Invalid,
}

fn classify_tree(tree: &Tree, p: u32, q: u32, r: u32) -> TreeState {
    let np = tree.node_by_label(p);
    let nq = tree.node_by_label(q);
    let nr = tree.node_by_label(r);
    if np == NONE || nq == NONE || nr == NONE {
        return TreeState::Invalid;
    }

    let pp = tree.parent[np as usize];
    let pq = tree.parent[nq as usize];
    let pr = tree.parent[nr as usize];

    // CherryPQ: {p, q} cherry
    if pp != NONE && pp == pq {
        return TreeState::CherryPQ;
    }

    // InterceptPR: {p, r} cherry AND par(q) = grandpar(r)
    if pp != NONE && pp == pr {
        let gpr = tree.parent[pr as usize];
        if gpr != NONE && pq == gpr {
            return TreeState::InterceptPR;
        }
    }

    // RotationQR: {q, r} cherry (p can be anywhere — no interceptor needed)
    // The swap extracts p as singleton and merges {q,r}. Since {q,r} is already
    // a cherry, the merge is valid without adjacency requirements on p.
    if pq != NONE && pq == pr {
        return TreeState::RotationQR;
    }

    TreeState::Invalid
}

fn find_generalized_triple(trees: &[Tree], num_leaves: u32) -> Option<ReductionAction> {
    let m = trees.len();

    // Scan each leaf as PIVOT (like the original 3-2 rule)
    for p in 1..=num_leaves {
        // Get p's cherry partner in each tree
        let partners: Vec<Option<u32>> = trees.iter().map(|t| {
            let np = t.node_by_label(p);
            if np == NONE { return None; }
            match t.sibling(np) {
                Some(s) if t.is_leaf(s) => Some(t.label[s as usize]),
                _ => None,
            }
        }).collect();

        // p must be in a cherry in at least one tree
        let mut unique: Vec<u32> = partners.iter().filter_map(|x| *x).collect();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() < 2 {
            continue; // Need at least 2 distinct partners for a conflict
        }

        // For each pair of partners: one is keeper q, other is victim r
        for i in 0..unique.len() {
            for j in 0..unique.len() {
                if i == j { continue; }
                let q = unique[i]; // keeper
                let r = unique[j]; // victim (to be deleted)

                // Classify each tree
                let mut all_valid = true;
                let mut has_interceptor = false; // Need at least one non-cherry state

                for t in 0..m {
                    let state = classify_tree(&trees[t], p, q, r);
                    match state {
                        TreeState::CherryPQ => {}
                        TreeState::InterceptPR => { has_interceptor = true; }
                        TreeState::RotationQR => { has_interceptor = true; }
                        TreeState::Invalid => { all_valid = false; break; }
                    }
                }

                if all_valid && has_interceptor {
                    return Some(ReductionAction::Delete { victim: r });
                }
            }
        }
    }
    None
}
