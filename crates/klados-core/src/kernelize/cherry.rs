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

        // If the remove label is protected but keep is not, swap them so the
        // protected label survives as the representative.
        if ctx.is_protected(remove) && !ctx.is_protected(keep) {
            std::mem::swap(&mut keep, &mut remove);
        }

        // If both are protected, refuse to collapse — can't keep both.
        if ctx.is_protected(remove) {
            return None;
        }

        Some(ReductionAction::Collapse { keep, remove })
    }
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
