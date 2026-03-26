//! Core branch-and-bound for 2-tree rSPR distance.
//!
//! Faithful port of rspr's rSPR_branch_and_bound_hlpr.
//! Phase 1: base algorithm only, no pruning optimizations.
//!
//! Flow:
//!   1. Process singletons (free: no k decrement)
//!   2. Find sibling pair in T1
//!   3. Case 2: pair matches in T2 → contract (free)
//!   4. Case 3: pair doesn't match → 3-way branch (k-1 each)

use klados_core::tree::{Label, NodeId, Tree, NONE};
use klados_core::{Instance, SolverStats};

use super::forest::{Forest, sync_twins};
use super::undo::{self, UndoMachine};
use crate::lower_bound::maf_bounds;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn solve(instance: &Instance, stats: &mut SolverStats) -> Option<Vec<Tree>> {
    debug_assert!(instance.num_trees() == 2);
    let n = instance.num_leaves;
    if n <= 1 {
        return Some(vec![instance.trees[0].clone()]);
    }

    let bounds = maf_bounds(&instance.trees, n);
    let lb = bounds.lower;
    let ub = bounds.upper.min(n as usize);
    stats.lower_bound = lb;
    stats.upper_bound = Some(ub);

    let lb_k = lb.saturating_sub(1);
    let ub_k = ub.saturating_sub(1);

    for k in lb_k..=ub_k {
        let mut f1 = Forest::from_tree(&instance.trees[0]);
        let mut f2 = Forest::from_tree(&instance.trees[1]);
        sync_twins(&mut f1, &mut f2);

        let mut forests = [f1, f2];
        let mut um = UndoMachine::new();

        let result = branch_and_bound(&mut forests, k as i32, &mut um, stats);

        if result >= 0 {
            return Some(extract_components(&forests));
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Branch-and-bound
// ---------------------------------------------------------------------------

fn branch_and_bound(
    forests: &mut [Forest; 2],
    mut k: i32,
    um: &mut UndoMachine,
    stats: &mut SolverStats,
) -> i32 {

    stats.nodes_explored += 1;

    loop {
        // --- Phase 1: Process singletons ---
        if !process_singletons(forests, &mut k, um) {
            return -1; // k went negative
        }

        // --- Phase 2: Find sibling pair in T1 ---
        match find_sibling_pair(forests) {
            PairResult::NoPairs => {
                return k;
            }
            PairResult::Case2 { t1_parent, t2_parent } => {
                do_case2_contract(forests, t1_parent, t2_parent, um);
                // Loop back to check for new singletons
                continue;
            }
            PairResult::Case3 { t1_a, t1_c, t2_a, t2_b, t2_c } => {
                if k <= 0 {
                    return -1;
                }
                return do_case3_branch(
                    forests, k, um, stats,
                    t1_a, t1_c, t2_a, t2_b, t2_c,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Singleton processing (Case 1)
// ---------------------------------------------------------------------------

/// Process all singletons. Returns false if k goes negative.
fn process_singletons(
    forests: &mut [Forest; 2],
    k: &mut i32,
    um: &mut UndoMachine,
) -> bool {
    loop {
        let singleton = find_singleton(forests);
        if singleton == NONE { return true; }

        // singleton is a T2 leaf that is a component root (singleton in T2)
        let t2_node = singleton;
        let t1_node = forests[1].twin[t2_node as usize];
        if t1_node == NONE { continue; }

        let t1_parent = forests[0].parent[t1_node as usize];
        if t1_parent == NONE {
            // T1_a is already a component root — skip
            continue;
        }

        // Check if parent is a sibling pair BEFORE cutting
        let was_sibling_pair = forests[0].is_sibling_pair(t1_parent);

        // Cut T1_a from its parent → T1_a becomes a new component
        undo::cut_parent(&mut forests[0], 0, t1_node, um);
        undo::add_component(&mut forests[0], 0, t1_node, um);

        // Contract the parent (may become degree-1 and get spliced)
        let contracted = undo::contract(&mut forests[0], 0, t1_parent, um);

        // If contracted node is a new sibling pair, it will be found naturally
        // in the next iteration of the main loop.
        let _ = (was_sibling_pair, contracted);
    }
}

/// Find a singleton: a T2 component that is a single leaf.
/// Skip singletons whose twin is already a component root in T1.
fn find_singleton(forests: &[Forest; 2]) -> NodeId {
    let f2 = &forests[1];
    let f1 = &forests[0];
    for &root in &f2.components {
        if f2.is_leaf(root) {
            let twin = f2.twin[root as usize];
            if twin != NONE && f1.parent[twin as usize] != NONE {
                // Twin has a parent in T1 → can cut it
                return root;
            }
        }
    }
    NONE
}

// ---------------------------------------------------------------------------
// Sibling pair detection
// ---------------------------------------------------------------------------

enum PairResult {
    NoPairs,
    Case2 { t1_parent: NodeId, t2_parent: NodeId },
    Case3 { t1_a: NodeId, t1_c: NodeId, t2_a: NodeId, t2_b: NodeId, t2_c: NodeId },
}

/// Find a sibling pair in T1 and classify it.
fn find_sibling_pair(forests: &[Forest; 2]) -> PairResult {
    let f1 = &forests[0];
    let f2 = &forests[1];

    // Walk T1 to find sibling pairs (internal node with 2 leaf children)
    for node in 0..f1.num_nodes as NodeId {
        if !f1.is_sibling_pair(node) { continue; }

        let t1_a = f1.left[node as usize];
        let t1_c = f1.right[node as usize];

        // Get T2 twins
        let t2_a = f1.twin[t1_a as usize];
        let t2_c = f1.twin[t1_c as usize];
        if t2_a == NONE || t2_c == NONE { continue; }

        // Case 2: T2_a and T2_c share a parent in T2
        let t2_a_parent = f2.parent[t2_a as usize];
        let t2_c_parent = f2.parent[t2_c as usize];
        if t2_a_parent != NONE && t2_a_parent == t2_c_parent {
            return PairResult::Case2 {
                t1_parent: node,
                t2_parent: t2_a_parent,
            };
        }

        // Case 3: different parents → need to branch
        // Orient: T2_a is the deeper one (further from root)
        let da = depth_to_root(f2, t2_a);
        let dc = depth_to_root(f2, t2_c);
        let (t1_a, t1_c, t2_a, t2_c) = if da >= dc {
            (t1_a, t1_c, t2_a, t2_c)
        } else {
            (t1_c, t1_a, t2_c, t2_a)
        };

        // T2_b = sibling of T2_a in T2
        let t2_b = f2.sibling(t2_a);
        if t2_b == NONE { continue; }

        return PairResult::Case3 { t1_a, t1_c, t2_a, t2_b, t2_c };
    }

    PairResult::NoPairs
}

/// Distance from node to its component root (via parent pointers).
fn depth_to_root(f: &Forest, mut node: NodeId) -> u16 {
    let mut d: u16 = 0;
    loop {
        let p = f.parent[node as usize];
        if p == NONE { return d; }
        d += 1;
        node = p;
    }
}

// ---------------------------------------------------------------------------
// Case 2: Contract matching sibling pair
// ---------------------------------------------------------------------------

fn do_case2_contract(
    forests: &mut [Forest; 2],
    t1_parent: NodeId,
    t2_parent: NodeId,
    um: &mut UndoMachine,
) {
    // Get children labels BEFORE detaching (right child's label is "kept")
    let t1_lc = forests[0].left[t1_parent as usize];
    let t1_rc = forests[0].right[t1_parent as usize];
    let t2_lc = forests[1].left[t2_parent as usize];
    let t2_rc = forests[1].right[t2_parent as usize];
    let kept_label = forests[0].label[t1_rc as usize];
    let removed_label = forests[0].label[t1_lc as usize];

    // Contract in T1: detach both children, parent becomes leaf
    undo::contract_sibling_pair(&mut forests[0], 0, t1_parent, um);
    // Contract in T2: detach both children, parent becomes leaf
    undo::contract_sibling_pair(&mut forests[1], 1, t2_parent, um);

    // Parent takes on "kept" label so it's visible to label-based operations
    undo::set_label(&mut forests[0], 0, t1_parent, kept_label, um);
    undo::set_label(&mut forests[1], 1, t2_parent, kept_label, um);
    // Update label_to_node: kept label → parent node
    undo::set_label_to_node(&mut forests[0], 0, kept_label, t1_parent, um);
    undo::set_label_to_node(&mut forests[1], 1, kept_label, t2_parent, um);
    // Removed label → NONE
    if removed_label != 0 {
        undo::set_label_to_node(&mut forests[0], 0, removed_label, NONE, um);
        undo::set_label_to_node(&mut forests[1], 1, removed_label, NONE, um);
        // Track: removed_label collapses into kept_label
        undo::set_collapsed(&mut forests[0], 0, removed_label, kept_label, um);
    }

    // Update twins: parents now represent the contracted pair
    undo::set_twin(&mut forests[0], 0, t1_parent, t2_parent, um);
    undo::set_twin(&mut forests[1], 1, t2_parent, t1_parent, um);
}

// ---------------------------------------------------------------------------
// Case 3: 3-way branching
// ---------------------------------------------------------------------------

fn do_case3_branch(
    forests: &mut [Forest; 2],
    k: i32,
    um: &mut UndoMachine,
    stats: &mut SolverStats,
    t1_a: NodeId,
    t1_c: NodeId,
    t2_a: NodeId,
    t2_b: NodeId,
    t2_c: NodeId,
) -> i32 {
    let mut best_k: i32 = -1;

    // --- Branch A: cut T2_a ---
    {
        let cp = um.checkpoint();
        let t2_a_parent = forests[1].parent[t2_a as usize];
        undo::cut_parent(&mut forests[1], 1, t2_a, um);
        undo::add_component(&mut forests[1], 1, t2_a, um);
        // Contract T2_a's old parent (now has 1 child)
        if t2_a_parent != NONE {
            let node = undo::contract(&mut forests[1], 1, t2_a_parent, um);
            let _ = node;
        }

        let result = branch_and_bound(forests, k - 1, um, stats);
        if result >= 0 { return result; }
        um.undo_to(cp, forests);
    }

    // --- Branch B: cut T2_b ---
    {
        let cp = um.checkpoint();
        let t2_b_parent = forests[1].parent[t2_b as usize];
        undo::cut_parent(&mut forests[1], 1, t2_b, um);
        undo::add_component(&mut forests[1], 1, t2_b, um);
        if t2_b_parent != NONE {
            let node = undo::contract(&mut forests[1], 1, t2_b_parent, um);
            let _ = node;
        }

        let result = branch_and_bound(forests, k - 1, um, stats);
        if result >= 0 { return result; }
        um.undo_to(cp, forests);
    }

    // --- Branch C: cut T2_c ---
    {
        let cp = um.checkpoint();
        let t2_c_parent = forests[1].parent[t2_c as usize];
        if t2_c_parent != NONE {
            undo::cut_parent(&mut forests[1], 1, t2_c, um);
            undo::add_component(&mut forests[1], 1, t2_c, um);
            let node = undo::contract(&mut forests[1], 1, t2_c_parent, um);
            let _ = node;
        }
        // If T2_c is already a root, cutting is free (k++ in rspr)
        // For now, still decrement k.

        let result = branch_and_bound(forests, k - 1, um, stats);
        if result >= 0 { return result; }
        um.undo_to(cp, forests);
    }

    best_k
}

// ---------------------------------------------------------------------------
// Solution extraction
// ---------------------------------------------------------------------------

/// Extract MAF components from the solved forest state.
fn extract_components(forests: &[Forest; 2]) -> Vec<Tree> {
    let f1 = &forests[0];
    let n = f1.num_leaves;

    // Resolve collapsed_into to final representatives (transitive closure)
    let mut collapsed: Vec<Label> = f1.collapsed_into[..=n as usize].to_vec();
    for _ in 0..n {
        let mut changed = false;
        for lbl in 1..=n {
            let target = collapsed[lbl as usize];
            if target != lbl && collapsed[target as usize] != target {
                collapsed[lbl as usize] = collapsed[target as usize];
                changed = true;
            }
        }
        if !changed { break; }
    }

    // Collect label sets per component
    let orig_tree = tree_from_original(f1);
    let mut result = Vec::new();
    for &root in &f1.components {
        let mut current_labels = Vec::new();
        collect_labels(f1, root, &mut current_labels);
        if current_labels.is_empty() { continue; }

        // Expand: find all original labels whose representative is in this component
        let mut leafset = fixedbitset::FixedBitSet::with_capacity(n as usize + 1);
        for &lbl in &current_labels {
            for orig in 1..=n {
                if collapsed[orig as usize] == lbl {
                    leafset.insert(orig as usize);
                }
            }
        }
        if leafset.count_ones(..) > 0 {
            let tree = Tree::component_from_leafset(&leafset, &orig_tree, n);
            result.push(tree);
        }
    }
    result
}

/// Collect current labels reachable from node.
fn collect_labels(f: &Forest, node: NodeId, out: &mut Vec<Label>) {
    let lbl = f.label[node as usize];
    if lbl != 0 {
        out.push(lbl);
        return;
    }
    let lc = f.left[node as usize];
    let rc = f.right[node as usize];
    if lc != NONE { collect_labels(f, lc, out); }
    if rc != NONE { collect_labels(f, rc, out); }
}


/// Reconstruct original Tree from Forest's immutable orig_* arrays.
fn tree_from_original(f: &Forest) -> Tree {
    let mut t = Tree::with_capacity(f.num_leaves);
    t.parent = f.orig_parent.clone();
    t.left = f.orig_left.clone();
    t.right = f.orig_right.clone();
    t.label = f.orig_label.clone();
    t.label_to_node = vec![NONE; f.num_leaves as usize + 1];
    for node in 0..f.num_nodes as NodeId {
        let lbl = f.orig_label[node as usize];
        if lbl != 0 && f.orig_left[node as usize] == NONE {
            t.label_to_node[lbl as usize] = node;
        }
    }
    t.num_leaves = f.num_leaves;
    t.root = f.root;
    t.depth = vec![0; f.num_nodes];
    t.subtree_size = vec![0; f.num_nodes];
    t
}
