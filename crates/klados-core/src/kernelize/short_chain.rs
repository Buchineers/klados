//! Short chain reduction rules (Reductions 3 and 4 from Kelk & Linz 2020).
//!
//! These rules detect parameter-reducing patterns on common 3-chains that
//! the 3-2 interceptor rule does not cover.
//!
//! **Rule 3** (crossed cherry on 3-chain): Common pendant 3-chain where the
//! cherry is at opposite ends in different trees. Delete all 3 leaves.
//!
//! **Rule 4** (external cherry on 3-chain): Common 3-chain where one end
//! forms a cherry with an external leaf x in the other tree. Delete x.
//!
//! Currently proven for t = 2 only. Multi-tree generalization is an open problem.

use crate::{NONE, Tree};
use super::rule::{ReductionAction, ReductionRule, RuleContext};

// ═══════════════════════════════════════════════════════════════
// Rule 3: Crossed cherry on common 3-chain
// ═══════════════════════════════════════════════════════════════

/// Rule 3: Find a common 3-chain C = (l1, l2, l3) such that:
/// - {l1, l2} is a cherry in T and {l2, l3} is a cherry in T', OR
/// - {l2, l3} is a cherry in T and {l1, l2} is a cherry in T'
///
/// If found, delete all of C. Parameter reduces by 1.
///
/// Note: we delete the MIDDLE leaf (l2) since the other two are in cherries
/// that will be collapsed by the cherry rule on the next iteration.
/// Actually, from Lemma 1, deleting all of C at once reduces d by 1.
/// We implement this as deleting l2 (the shared cherry member) which is the
/// parameter-reducing step. The cherry rule will then collapse the remaining
/// leaves in subsequent passes.
#[derive(Debug)]
pub struct Rule3CrossedCherry;

impl ReductionRule for Rule3CrossedCherry {
    fn name(&self) -> &'static str {
        "rule3-crossed"
    }

    fn find(&self, ctx: &RuleContext) -> Option<ReductionAction> {
        let inst = ctx.instance;
        // Only proven for 2-tree instances
        if inst.num_trees() != 2 || inst.num_leaves < 3 {
            return None;
        }
        let t1 = &inst.trees[0];
        let t2 = &inst.trees[1];

        find_rule3_candidate(t1, t2, inst.num_leaves)
    }
}

/// Find a Rule 3 candidate.
///
/// Look for three leaves (a, b, c) that form a pendant 3-chain in both trees,
/// with the cherry at opposite ends:
/// - In T1: pendant chain with {a, b} as cherry (a,b siblings, c is one level up)
/// - In T2: pendant chain with {b, c} as cherry (b,c siblings, a is one level up)
/// (or the symmetric case)
///
/// We delete the entire chain. Since deleting 3 leaves at once is a special case,
/// we return Delete for one leaf and let the pipeline handle the rest.
/// Actually, the correct approach: delete all 3 at once and reduce param by 1.
/// We use Delete for the leaf that is NOT in a cherry in either tree — but all
/// three are in cherries. The original rule says "delete C from X".
///
/// For our pipeline, we delete one leaf (the middle one, l2) as parameter-reducing.
/// The remaining two leaves (l1, l3) will form a common cherry that the cherry
/// rule will collapse in the next pass. Net effect: 3 leaves removed, param -1.
fn find_rule3_candidate(t1: &Tree, t2: &Tree, num_leaves: u32) -> Option<ReductionAction> {
    // Strategy: for each cherry {a, b} in T1, check if there exists a c such that:
    // (a, b, c) is a pendant 3-chain in T1 (c is the chain head, {a,b} at bottom)
    // AND {b, c} is a cherry in T2
    // AND (a, b, c) forms a pendant 3-chain in T2 too (a is chain head, {b,c} at bottom)
    for node in t1.post_order() {
        if let Some((l, r)) = t1.children(node) {
            if !t1.is_leaf(l) || !t1.is_leaf(r) {
                continue;
            }
            let a = t1.label[l as usize];
            let b = t1.label[r as usize];
            if a == 0 || b == 0 {
                continue;
            }

            // {a, b} is a cherry in T1. Check both orientations:
            // Try: c is the "chain head" — the parent of the cherry's parent has c as leaf child
            let cherry_parent = node; // parent of {a, b} in T1
            let gp = t1.parent[cherry_parent as usize];
            if gp == NONE {
                continue;
            }
            // The other child of gp should be a single leaf c (pendant 3-chain)
            if let Some((gl, gr)) = t1.children(gp) {
                let uncle = if gl == cherry_parent { gr } else { gl };
                if !t1.is_leaf(uncle) {
                    continue;
                }
                let c = t1.label[uncle as usize];
                if c == 0 {
                    continue;
                }

                // Now we have pendant 3-chain in T1: (c, a, b) or (c, b, a)
                // with {a, b} cherry at bottom and c at top.
                // For Rule 3: check if {b, c} is cherry in T2 (or {a, c})

                // Check {b, c} cherry in T2
                let node_b_t2 = t2.node_by_label(b);
                let node_c_t2 = t2.node_by_label(c);
                if node_b_t2 != NONE && node_c_t2 != NONE {
                    let pb = t2.parent[node_b_t2 as usize];
                    let pc = t2.parent[node_c_t2 as usize];
                    if pb != NONE && pb == pc {
                        // {b, c} is cherry in T2!
                        // Check that (a, b, c) forms a pendant 3-chain in T2 too:
                        // a should be the uncle of {b, c}'s parent
                        let cherry_parent_t2 = pb;
                        let gp_t2 = t2.parent[cherry_parent_t2 as usize];
                        if gp_t2 != NONE {
                            if let Some((gl2, gr2)) = t2.children(gp_t2) {
                                let uncle_t2 = if gl2 == cherry_parent_t2 { gr2 } else { gl2 };
                                if t2.is_leaf(uncle_t2) && t2.label[uncle_t2 as usize] == a {
                                    // Perfect: pendant 3-chain in T2 with {b,c} cherry, a at top
                                    // Delete the middle leaf (b) — this is the parameter-reducing step.
                                    // Cherry rule will then collapse {a,c} in T1 side or similar.
                                    // Actually, deleting b breaks both cherries. Let's delete
                                    // the one leaf that is NOT the endpoint of either cherry in a
                                    // way that doesn't break the chain structure.
                                    //
                                    // Per the original rule: delete ALL of C, param -1.
                                    // Our pipeline handles one deletion at a time.
                                    // Delete b: {a,b} cherry in T1 breaks, {b,c} cherry in T2 breaks.
                                    //   After deleting b: a and c remain. In T1, a becomes sibling of
                                    //   c (gp's children become a and c). In T2, c becomes sibling
                                    //   of a (gp_t2's children become c and a). So {a,c} becomes a
                                    //   common cherry → cherry rule collapses them → 3 leaves total removed.
                                    return Some(ReductionAction::Delete { victim: b });
                                }
                            }
                        }
                    }
                }

                // Check {a, c} cherry in T2 (symmetric)
                let node_a_t2 = t2.node_by_label(a);
                let node_c_t2 = t2.node_by_label(c);
                if node_a_t2 != NONE && node_c_t2 != NONE {
                    let pa = t2.parent[node_a_t2 as usize];
                    let pc = t2.parent[node_c_t2 as usize];
                    if pa != NONE && pa == pc {
                        // {a, c} is cherry in T2!
                        let cherry_parent_t2 = pa;
                        let gp_t2 = t2.parent[cherry_parent_t2 as usize];
                        if gp_t2 != NONE {
                            if let Some((gl2, gr2)) = t2.children(gp_t2) {
                                let uncle_t2 = if gl2 == cherry_parent_t2 { gr2 } else { gl2 };
                                if t2.is_leaf(uncle_t2) && t2.label[uncle_t2 as usize] == b {
                                    // Pendant 3-chain in T2 with {a,c} cherry, b at top
                                    // Delete a: breaks {a,b} cherry in T1, breaks {a,c} cherry in T2
                                    // After: b and c remain, and should form common cherry
                                    return Some(ReductionAction::Delete { victim: a });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════
// Rule 4: External cherry on common 3-chain
// ═══════════════════════════════════════════════════════════════

/// Rule 4: Find a common 3-chain C = (l1, l2, l3) such that:
/// - {l2, l3} is a cherry in T
/// - {l3, x} is a cherry in T' with x not in C
///
/// If found, delete x. Parameter reduces by 1.
#[derive(Debug)]
pub struct Rule4ExternalCherry;

impl ReductionRule for Rule4ExternalCherry {
    fn name(&self) -> &'static str {
        "rule4-external"
    }

    fn find(&self, ctx: &RuleContext) -> Option<ReductionAction> {
        let inst = ctx.instance;
        // Only proven for 2-tree instances
        if inst.num_trees() != 2 || inst.num_leaves < 4 {
            return None;
        }
        let t1 = &inst.trees[0];
        let t2 = &inst.trees[1];

        // Try T1 has the cherry, T2 has the external cherry
        if let Some(action) = find_rule4_candidate(t1, t2, inst.num_leaves) {
            return Some(action);
        }
        // Try T2 has the cherry, T1 has the external cherry
        find_rule4_candidate(t2, t1, inst.num_leaves)
    }
}

/// Find a Rule 4 candidate.
///
/// In `t_cherry`: look for pendant 3-chain (l1, l2, l3) with {l2, l3} as cherry at bottom.
/// In `t_ext`: check that {l3, x} is a cherry for some x not in {l1, l2, l3}.
/// The chain must also exist in t_ext (common chain).
fn find_rule4_candidate(t_cherry: &Tree, t_ext: &Tree, num_leaves: u32) -> Option<ReductionAction> {
    // For each cherry {l2, l3} in t_cherry:
    for node in t_cherry.post_order() {
        if let Some((l, r)) = t_cherry.children(node) {
            if !t_cherry.is_leaf(l) || !t_cherry.is_leaf(r) {
                continue;
            }
            let l2 = t_cherry.label[l as usize];
            let l3 = t_cherry.label[r as usize];
            if l2 == 0 || l3 == 0 {
                continue;
            }

            // {l2, l3} is a cherry in t_cherry.
            // Check if there's an l1 making (l1, l2, l3) a pendant 3-chain
            let cherry_parent = node;
            let gp = t_cherry.parent[cherry_parent as usize];
            if gp == NONE {
                continue;
            }
            if let Some((gl, gr)) = t_cherry.children(gp) {
                let uncle = if gl == cherry_parent { gr } else { gl };
                if !t_cherry.is_leaf(uncle) {
                    continue;
                }
                let l1 = t_cherry.label[uncle as usize];
                if l1 == 0 {
                    continue;
                }

                // We have pendant 3-chain (l1, l2, l3) in t_cherry with {l2, l3} cherry.
                // Check: is (l1, l2, l3) a chain in t_ext? (not necessarily pendant)
                // For a chain: l1 is at top, l2 below, l3 below l2.
                // Check that they form a caterpillar path in t_ext.
                if !is_chain_in_tree(t_ext, l1, l2, l3) {
                    // Try the reverse: maybe the chain is oriented differently in t_ext
                    // A common chain means the same sequence appears, possibly reversed.
                    // Actually for rooted chains, order matters. Let's check both.
                    if !is_chain_in_tree(t_ext, l3, l2, l1) {
                        continue;
                    }
                }

                // Now check: {l3, x} is a cherry in t_ext for some x not in {l1, l2, l3}
                let node_l3_ext = t_ext.node_by_label(l3);
                if node_l3_ext == NONE {
                    continue;
                }
                let sib = match t_ext.sibling(node_l3_ext) {
                    Some(s) if t_ext.is_leaf(s) => s,
                    _ => continue,
                };
                let x = t_ext.label[sib as usize];
                if x == l1 || x == l2 || x == l3 || x == 0 {
                    continue;
                }

                // Also try: {l2, x} cherry in t_ext (symmetric in which end)
                // Actually Rule 4 specifically says {l3, x}, let's also check with
                // l2 and l3 swapped in the chain role.

                // Found! Delete x.
                return Some(ReductionAction::Delete { victim: x });
            }

            // Also check the other orientation: l3 is the chain head, {l2, l3} at bottom
            // means the cherry is NOT at the pendant end. Let's check if l2 or l3 forms
            // a cherry with an external leaf in t_ext.
            //
            // Try treating the cherry members in the other order: (l1 could be at bottom)
            // Actually we already checked both l2 and l3 labels from the cherry.
            // The key is: for each cherry {a,b} in t_cherry, check if it's at the bottom
            // of a pendant 3-chain. We do that above. So we just need to continue.
        }
    }
    None
}

/// Check if three leaves form a chain (caterpillar path) in a rooted tree.
/// Returns true if (l1, l2, l3) form a chain with l1 closest to root.
///
/// Chain structure: l1's parent has one leaf child (l1) and one internal child.
/// That internal child has one leaf child (l2) and one more child containing l3.
fn is_chain_in_tree(tree: &Tree, l1: u32, l2: u32, l3: u32) -> bool {
    let n1 = tree.node_by_label(l1);
    let n2 = tree.node_by_label(l2);
    let n3 = tree.node_by_label(l3);
    if n1 == NONE || n2 == NONE || n3 == NONE {
        return false;
    }

    let p1 = tree.parent[n1 as usize];
    if p1 == NONE {
        return false;
    }

    // p1 should have l1 as one child, and the other child is an internal node
    let (p1_left, p1_right) = match tree.children(p1) {
        Some(lr) => lr,
        None => return false,
    };
    let internal = if p1_left == n1 {
        p1_right
    } else if p1_right == n1 {
        p1_left
    } else {
        return false;
    };

    if tree.is_leaf(internal) {
        return false;
    }

    // internal should have l2 as one child
    let (int_left, int_right) = match tree.children(internal) {
        Some(lr) => lr,
        None => return false,
    };

    let has_l2 = (int_left == n2) || (int_right == n2);
    if !has_l2 {
        return false;
    }

    // The other child of internal should contain l3
    let other = if int_left == n2 { int_right } else { int_left };

    // l3 can be the other child directly (pendant chain) or deeper
    if other == n3 {
        return true; // pendant: l3 is direct child
    }
    if tree.is_leaf(other) {
        return false; // other is a leaf but not l3
    }

    // Check if l3 is a leaf child of other (one more level)
    if let Some((ol, or)) = tree.children(other) {
        if ol == n3 || or == n3 {
            return true;
        }
    }

    false
}
