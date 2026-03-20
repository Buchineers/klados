//! Cherry (subtree) reduction rule.

use crate::{NONE, Tree};
use super::rule::{ReductionAction, ReductionRule, RuleContext};

/// Subtree reduction: find a common cherry (two sibling leaves with the same
/// parent in all trees) and collapse them into one.
///
/// Parameter-preserving.
#[derive(Debug)]
pub struct CherryRule;

impl ReductionRule for CherryRule {
    fn name(&self) -> &'static str {
        "cherry"
    }

    fn find(&self, ctx: &RuleContext) -> Option<ReductionAction> {
        let (mut keep, mut remove) = find_common_cherry(
            &ctx.instance.trees,
            ctx.instance.num_leaves,
        )?;

        // Protect labels: if the remove_label is protected, swap it to be the representative
        if is_protected(ctx.protected_labels, ctx.composite_rev, remove)
            && !is_protected(ctx.protected_labels, ctx.composite_rev, keep)
        {
            std::mem::swap(&mut keep, &mut remove);
        }

        Some(ReductionAction::Collapse { keep, remove })
    }
}

/// Check whether a label in current reduced space maps to a protected original label.
fn is_protected(protected: &[u32], composite_rev: &[u32], label: u32) -> bool {
    if protected.is_empty() {
        return false;
    }
    let orig = composite_rev[label as usize];
    protected.contains(&orig)
}

/// Find a single common cherry (two sibling leaves with the same parent in all trees).
/// Returns Some((keep_label, remove_label)) if found, where keep < remove.
fn find_common_cherry(trees: &[Tree], num_leaves: u32) -> Option<(u32, u32)> {
    if trees.is_empty() || num_leaves < 2 {
        return None;
    }

    let ref_tree = &trees[0];
    for node in ref_tree.post_order() {
        if let Some((l, r)) = ref_tree.children(node) {
            if ref_tree.is_leaf(l) && ref_tree.is_leaf(r) {
                let ll = ref_tree.label[l as usize];
                let rl = ref_tree.label[r as usize];
                if ll == 0 || rl == 0 {
                    continue;
                }

                // Check if this cherry exists in all other trees
                let mut common = true;
                for other in &trees[1..] {
                    let nl = other.node_by_label(ll);
                    let nr = other.node_by_label(rl);
                    if nl == NONE || nr == NONE {
                        common = false;
                        break;
                    }
                    let pl = other.parent[nl as usize];
                    let pr = other.parent[nr as usize];
                    if pl == NONE || pl != pr {
                        common = false;
                        break;
                    }
                }

                if common {
                    let (keep, remove) = if ll < rl { (ll, rl) } else { (rl, ll) };
                    return Some((keep, remove));
                }
            }
        }
    }
    None
}
