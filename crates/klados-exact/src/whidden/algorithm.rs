//! Core branch-and-bound for 2-tree rSPR distance.
//!
//! Faithful port of rspr's rSPR_branch_and_bound_hlpr.
//!
//! Flow:
//!   1. Process singletons (free: no k decrement)
//!   2. Find sibling pair in T1
//!   3. Case 2: pair matches in T2 → contract (free)
//!   4. Case 3: optional BB prune check, then 3-way branch (k-1 each)

use klados_core::tree::{Label, NodeId, Tree, NONE};
use klados_core::{Instance, SolverStats};

use super::forest::{TwinForest, T1, T2};
use super::undo::{self, UndoMachine};
use crate::lower_bound::maf_bounds;

// ---------------------------------------------------------------------------
// Configuration — maps to rspr's optimization flags
// ---------------------------------------------------------------------------

/// Controls which rspr optimizations are active.
///
/// Flag names and semantics match rspr's globals.
/// `default()` enables the same set as rspr's `-allopt` (DEFAULT_OPTIMIZATIONS).
#[derive(Clone, Debug)]
pub struct BBConfig {
    // --- Approximation-based pruning ---
    /// BB: prune branches where 3-approx > 3k (rspr's `BB` flag).
    pub bb: bool,

    // --- Branching reductions (reduce 3-way to fewer branches) ---
    /// COB: "cut one B" — skip branch A when T2_ab and T2_c are siblings.
    pub cut_one_b: bool,
    /// RCOB: "reverse cut one B" — skip branch C via uncle check.
    pub reverse_cut_one_b: bool,
    /// RCOB3: variant of reverse_cut_one_b (rspr's REVERSE_CUT_ONE_B_3).
    pub reverse_cut_one_b_3: bool,
    /// C2B: "cut two B" — uncle-sibling leaf check.
    pub cut_two_b: bool,
    /// CAB: "cut all B" — when safe, only branch on B.
    pub cut_all_b: bool,
    /// SC: "separate components" — cut A and C when they're in separate components.
    pub cut_ac_separate_components: bool,

    // --- Edge protection ---
    /// EP: protect edges in T2 from being cut (reduces branching).
    pub edge_protection: bool,
    /// EP2B: edge protection variant for two-B case.
    pub edge_protection_two_b: bool,

    // --- Sibling pair ordering ---
    /// DPO: prefer deepest protected sibling pair.
    pub deepest_protected_order: bool,
    /// DO: prefer deepest sibling pair (tiebreaker).
    pub deepest_order: bool,
    /// Near-preorder traversal for sibling pair enumeration.
    pub near_preorder_sibling_pairs: bool,

    // --- Leaf reduction ---
    /// LR: leaf reduction rule (rspr's LEAF_REDUCTION).
    pub leaf_reduction: bool,
    /// LR2: second leaf reduction rule (rspr's LEAF_REDUCTION2).
    pub leaf_reduction2: bool,

    // --- Approximation optimizations (used inside approx_3) ---
    /// Approx COB: cut-one-B inside 3-approximation.
    pub approx_cut_one_b: bool,
    /// Approx C2B: cut-two-B inside 3-approximation.
    pub approx_cut_two_b: bool,
    /// Approx RCOB: reverse-cut-one-B inside 3-approximation.
    pub approx_reverse_cut_one_b: bool,

    // --- Prefer rho (cluster-related) ---
    /// PREFER_RHO: prefer the rho component for sibling pair search.
    pub prefer_rho: bool,
    /// Prefer non-branching sibling pairs (Case 2 over Case 3).
    pub prefer_nonbranching: bool,
}

impl Default for BBConfig {
    /// rspr's DEFAULT_OPTIMIZATIONS + DEFAULT_ALGORITHM.
    fn default() -> Self {
        Self {
            bb: true,
            cut_one_b: true,
            reverse_cut_one_b: true,
            reverse_cut_one_b_3: true,
            cut_two_b: true,
            cut_all_b: true,
            cut_ac_separate_components: true,
            edge_protection: true,
            edge_protection_two_b: true,
            deepest_protected_order: true,
            deepest_order: true,
            near_preorder_sibling_pairs: true,
            leaf_reduction: true,
            leaf_reduction2: true,
            approx_cut_one_b: true,
            approx_cut_two_b: true,
            approx_reverse_cut_one_b: true,
            prefer_rho: true,
            prefer_nonbranching: true,
        }
    }
}

impl BBConfig {
    /// No optimizations — pure 3-way branching (rspr's `-noopt`).
    #[allow(dead_code)]
    pub fn noopt() -> Self {
        Self {
            bb: false,
            cut_one_b: false,
            reverse_cut_one_b: false,
            reverse_cut_one_b_3: false,
            cut_two_b: false,
            cut_all_b: false,
            cut_ac_separate_components: false,
            edge_protection: false,
            edge_protection_two_b: false,
            deepest_protected_order: false,
            deepest_order: false,
            near_preorder_sibling_pairs: false,
            leaf_reduction: false,
            leaf_reduction2: false,
            approx_cut_one_b: false,
            approx_cut_two_b: false,
            approx_reverse_cut_one_b: false,
            prefer_rho: false,
            prefer_nonbranching: false,
        }
    }

    /// Only BB pruning enabled (current baseline).
    #[allow(dead_code)]
    pub fn bb_only() -> Self {
        Self {
            bb: true,
            ..Self::noopt()
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn solve(instance: &Instance, stats: &mut SolverStats) -> Option<Vec<Tree>> {
    solve_with_config(instance, stats, &BBConfig::default())
}

pub fn solve_with_config(
    instance: &Instance,
    stats: &mut SolverStats,
    config: &BBConfig,
) -> Option<Vec<Tree>> {
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

    // Build once; reuse across k iterations (undo rewinds to initial state).
    let mut tf = TwinForest::from_trees(&instance.trees[0], &instance.trees[1], n);
    let mut um = UndoMachine::new();

    for k in lb_k..=ub_k {
        let cp = um.checkpoint();
        let result = branch_and_bound(&mut tf, k as i32, &mut um, stats, config);

        if result >= 0 {
            return Some(extract_components(&tf));
        }
        um.undo_to(cp, &mut tf);
    }

    None
}

// ---------------------------------------------------------------------------
// Branch-and-bound
// ---------------------------------------------------------------------------

fn branch_and_bound(
    tf: &mut TwinForest,
    mut k: i32,
    um: &mut UndoMachine,
    stats: &mut SolverStats,
    config: &BBConfig,
) -> i32 {

    stats.nodes_explored += 1;

    loop {
        // --- Phase 1: Process singletons ---
        if !process_singletons(tf, &mut k, um) {
            return -1; // k went negative
        }

        // --- Phase 2: Find sibling pair in T1 ---
        match find_sibling_pair(tf) {
            PairResult::NoPairs => {
                return k;
            }
            PairResult::Case2 { t1_parent, t2_parent } => {
                do_case2_contract(tf, t1_parent, t2_parent, um);
                // Loop back to check for new singletons
                continue;
            }
            PairResult::Case3 { t1_a, t1_c, t2_a, t2_b, t2_c } => {
                if k <= 0 {
                    return -1;
                }
                // BB: 3-approximation lower bound prune
                if config.bb && approx_3(tf, um) > 3 * k {
                    return -1;
                }
                return do_case3_branch(
                    tf, k, um, stats, config,
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
    tf: &mut TwinForest,
    _k: &mut i32, // reserved for future singleton-charging optimizations
    um: &mut UndoMachine,
) -> bool {
    loop {
        let singleton = find_singleton(tf);
        if singleton == NONE { return true; }

        // singleton is a T2 leaf that is a component root (singleton in T2)
        let t2_node = singleton;
        let t1_node = tf.twin[T2][t2_node as usize];
        if t1_node == NONE { continue; }

        let t1_parent = tf.parent[T1][t1_node as usize];
        if t1_parent == NONE {
            // T1_a is already a component root — skip
            continue;
        }

        // Cut T1_a from its parent → T1_a becomes a new component
        undo::cut_parent(tf, T1, t1_node, um);
        undo::add_component(tf, T1, t1_node, um);

        // Contract the parent (may become degree-1 and get spliced)
        undo::contract(tf, T1, t1_parent, um);
    }
}

/// Find a singleton: a T2 component that is a single leaf.
/// Skip singletons whose twin is already a component root in T1.
fn find_singleton(tf: &TwinForest) -> NodeId {
    for &root in &tf.components[T2] {
        if tf.is_leaf(T2, root) {
            let twin = tf.twin[T2][root as usize];
            if twin != NONE && tf.parent[T1][twin as usize] != NONE {
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
/// Walks from T1 component roots — only visits live nodes.
fn find_sibling_pair(tf: &TwinForest) -> PairResult {
    for &root in &tf.components[T1] {
        let result = find_pair_in(tf, root);
        if !matches!(result, PairResult::NoPairs) {
            return result;
        }
    }
    PairResult::NoPairs
}

/// Recursive DFS within a T1 component to find a sibling pair.
fn find_pair_in(tf: &TwinForest, node: NodeId) -> PairResult {
    let lc = tf.left[T1][node as usize];
    let rc = tf.right[T1][node as usize];

    if lc == NONE { return PairResult::NoPairs; } // leaf

    // Check if this node is a sibling pair
    if rc != NONE && tf.is_leaf(T1, lc) && tf.is_leaf(T1, rc) {
        if let Some(result) = classify_pair(tf, node, lc, rc) {
            return result;
        }
    }

    // Recurse into children
    if lc != NONE {
        let r = find_pair_in(tf, lc);
        if !matches!(r, PairResult::NoPairs) { return r; }
    }
    if rc != NONE {
        let r = find_pair_in(tf, rc);
        if !matches!(r, PairResult::NoPairs) { return r; }
    }
    PairResult::NoPairs
}

/// Classify a confirmed T1 sibling pair as Case 2 or Case 3.
fn classify_pair(
    tf: &TwinForest,
    t1_parent: NodeId,
    t1_a: NodeId,
    t1_c: NodeId,
) -> Option<PairResult> {
    let t2_a = tf.twin[T1][t1_a as usize];
    let t2_c = tf.twin[T1][t1_c as usize];
    if t2_a == NONE || t2_c == NONE { return None; }

    // Case 2: T2_a and T2_c share a parent in T2
    let t2_a_parent = tf.parent[T2][t2_a as usize];
    let t2_c_parent = tf.parent[T2][t2_c as usize];
    if t2_a_parent != NONE && t2_a_parent == t2_c_parent {
        return Some(PairResult::Case2 {
            t1_parent,
            t2_parent: t2_a_parent,
        });
    }

    // Case 3: orient so T2_a is deeper (from current component root)
    let da = depth_to_root(tf, T2, t2_a);
    let dc = depth_to_root(tf, T2, t2_c);
    let (t1_a, t1_c, t2_a, t2_c) = if da >= dc {
        (t1_a, t1_c, t2_a, t2_c)
    } else {
        (t1_c, t1_a, t2_c, t2_a)
    };

    let t2_b = tf.sibling(T2, t2_a);
    if t2_b == NONE { return None; }

    Some(PairResult::Case3 { t1_a, t1_c, t2_a, t2_b, t2_c })
}

/// Distance from node to its component root (via parent pointers).
fn depth_to_root(tf: &TwinForest, ti: usize, mut node: NodeId) -> u16 {
    let mut d: u16 = 0;
    loop {
        let p = tf.parent[ti][node as usize];
        if p == NONE { return d; }
        d += 1;
        node = p;
    }
}

// ---------------------------------------------------------------------------
// 3-approximation lower bound (rspr's BB pruning)
// ---------------------------------------------------------------------------

/// Greedy 3-approximation of rSPR distance on the current forest state.
/// Faithful port of rspr's `rSPR_worse_3_approx_binary_hlpr`.
///
/// Each Case 3 round: cut T1_a, T1_c from T1; cut T2_a (and maybe T2_b,
/// T2_c) from T2; count 3. Resolves the pair in one round.
/// Guarantee: num_cut ≤ 3 × optimal.
///
/// Non-destructive: uses checkpoint/undo on the live TwinForest.
fn approx_3(tf: &mut TwinForest, um: &mut UndoMachine) -> i32 {
    let cp = um.checkpoint();
    let mut num_cut: i32 = 0;

    loop {
        // Process singletons (free — no contribution to num_cut)
        loop {
            let singleton = find_singleton(tf);
            if singleton == NONE { break; }
            let t2_node = singleton;
            let t1_node = tf.twin[T2][t2_node as usize];
            if t1_node == NONE { continue; }
            let t1_parent = tf.parent[T1][t1_node as usize];
            if t1_parent == NONE { continue; }
            undo::cut_parent(tf, T1, t1_node, um);
            undo::add_component(tf, T1, t1_node, um);
            undo::contract(tf, T1, t1_parent, um);
        }

        // Find sibling pair in T1
        match find_sibling_pair(tf) {
            PairResult::NoPairs => break,
            PairResult::Case2 { t1_parent, t2_parent } => {
                do_case2_contract(tf, t1_parent, t2_parent, um);
            }
            PairResult::Case3 { t1_a, t1_c, t2_a, t2_b: _, t2_c } => {
                let t1_parent = tf.parent[T1][t1_a as usize];

                // Check cut_b_only: T2_a.parent.parent == T2_c.parent
                // (T2_ab and T2_c are siblings → only need to cut T2_b)
                let t2_a_parent = tf.parent[T2][t2_a as usize];
                let t2_c_parent = tf.parent[T2][t2_c as usize];
                let cut_b_only = t2_a_parent != NONE
                    && tf.parent[T2][t2_a_parent as usize] != NONE
                    && tf.parent[T2][t2_a_parent as usize] == t2_c_parent;

                if cut_b_only {
                    // Only cut T2_b (sibling of T2_a). The pair will become
                    // Case 2 in the next iteration after T2_ab contracts.
                    let t2_b = tf.sibling(T2, t2_a);
                    if t2_b != NONE {
                        let t2_b_parent = tf.parent[T2][t2_b as usize];
                        undo::cut_parent(tf, T2, t2_b, um);
                        undo::add_component(tf, T2, t2_b, um);
                        if t2_b_parent != NONE {
                            undo::contract(tf, T2, t2_b_parent, um);
                        }
                    }
                } else {
                    // Full Case 3: cut T1_a, T1_c from T1
                    undo::cut_parent(tf, T1, t1_a, um);
                    undo::add_component(tf, T1, t1_a, um);
                    if t1_parent != NONE {
                        undo::contract(tf, T1, t1_parent, um);
                    }
                    // T1_c may have moved up after contracting t1_parent
                    undo::cut_parent(tf, T1, t1_c, um);
                    undo::add_component(tf, T1, t1_c, um);
                    let t1_c_parent = tf.parent[T1][t1_c as usize];
                    if t1_c_parent != NONE {
                        undo::contract(tf, T1, t1_c_parent, um);
                    }

                    // Cut T2_a from T2
                    if t2_a_parent != NONE {
                        undo::cut_parent(tf, T2, t2_a, um);
                        undo::add_component(tf, T2, t2_a, um);
                        undo::contract(tf, T2, t2_a_parent, um);
                    }

                    // Cut T2_c from T2 (re-read twin in case it moved)
                    let t2_c_now = tf.twin[T1][t1_c as usize];
                    if t2_c_now != NONE {
                        let t2_c_p = tf.parent[T2][t2_c_now as usize];
                        if t2_c_p != NONE {
                            undo::cut_parent(tf, T2, t2_c_now, um);
                            undo::add_component(tf, T2, t2_c_now, um);
                            undo::contract(tf, T2, t2_c_p, um);
                        }
                    }
                }

                num_cut += 3;
            }
        }
    }

    um.undo_to(cp, tf);
    num_cut
}

// ---------------------------------------------------------------------------
// Case 2: Contract matching sibling pair
// ---------------------------------------------------------------------------

fn do_case2_contract(
    tf: &mut TwinForest,
    t1_parent: NodeId,
    t2_parent: NodeId,
    um: &mut UndoMachine,
) {
    // Get children labels BEFORE detaching (right child's label is "kept")
    let t1_lc = tf.left[T1][t1_parent as usize];
    let t1_rc = tf.right[T1][t1_parent as usize];
    let kept_label = tf.label[T1][t1_rc as usize];
    let removed_label = tf.label[T1][t1_lc as usize];

    // Contract in T1: detach both children, parent becomes leaf
    undo::contract_sibling_pair(tf, T1, t1_parent, um);
    // Contract in T2: detach both children, parent becomes leaf
    undo::contract_sibling_pair(tf, T2, t2_parent, um);

    // Parent takes on "kept" label so it's visible to label-based operations
    undo::set_label(tf, T1, t1_parent, kept_label, um);
    undo::set_label(tf, T2, t2_parent, kept_label, um);
    // Update label_to_node: kept label → parent node
    undo::set_label_to_node(tf, T1, kept_label, t1_parent, um);
    undo::set_label_to_node(tf, T2, kept_label, t2_parent, um);
    // Removed label → NONE
    if removed_label != 0 {
        undo::set_label_to_node(tf, T1, removed_label, NONE, um);
        undo::set_label_to_node(tf, T2, removed_label, NONE, um);
        // Track: removed_label collapses into kept_label
        undo::set_collapsed(tf, removed_label, kept_label, um);
    }

    // Update twins: parents now represent the contracted pair
    undo::set_twin(tf, T1, t1_parent, t2_parent, um);
    undo::set_twin(tf, T2, t2_parent, t1_parent, um);
}

// ---------------------------------------------------------------------------
// Case 3: 3-way branching
// ---------------------------------------------------------------------------

fn do_case3_branch(
    tf: &mut TwinForest,
    k: i32,
    um: &mut UndoMachine,
    stats: &mut SolverStats,
    config: &BBConfig,
    _t1_a: NodeId,
    _t1_c: NodeId,
    t2_a: NodeId,
    t2_b: NodeId,
    t2_c: NodeId,
) -> i32 {
    // --- Branch A: cut T2_a ---
    {
        let cp = um.checkpoint();
        let t2_a_parent = tf.parent[T2][t2_a as usize];
        undo::cut_parent(tf, T2, t2_a, um);
        undo::add_component(tf, T2, t2_a, um);
        if t2_a_parent != NONE {
            undo::contract(tf, T2, t2_a_parent, um);
        }

        let result = branch_and_bound(tf, k - 1, um, stats, config);
        if result >= 0 { return result; }
        um.undo_to(cp, tf);
    }

    // --- Branch B: cut T2_b ---
    {
        let cp = um.checkpoint();
        let t2_b_parent = tf.parent[T2][t2_b as usize];
        undo::cut_parent(tf, T2, t2_b, um);
        undo::add_component(tf, T2, t2_b, um);
        if t2_b_parent != NONE {
            undo::contract(tf, T2, t2_b_parent, um);
        }

        let result = branch_and_bound(tf, k - 1, um, stats, config);
        if result >= 0 { return result; }
        um.undo_to(cp, tf);
    }

    // --- Branch C: cut T2_c ---
    {
        let cp = um.checkpoint();
        let t2_c_parent = tf.parent[T2][t2_c as usize];
        if t2_c_parent != NONE {
            undo::cut_parent(tf, T2, t2_c, um);
            undo::add_component(tf, T2, t2_c, um);
            undo::contract(tf, T2, t2_c_parent, um);
        }

        let result = branch_and_bound(tf, k - 1, um, stats, config);
        if result >= 0 { return result; }
        um.undo_to(cp, tf);
    }

    -1
}

// ---------------------------------------------------------------------------
// Solution extraction
// ---------------------------------------------------------------------------

/// Extract MAF components from the solved forest state.
fn extract_components(tf: &TwinForest) -> Vec<Tree> {
    let n = tf.num_leaves;

    // Resolve collapsed_into to final representatives (transitive closure)
    let mut collapsed: Vec<Label> = tf.collapsed_into[..=n as usize].to_vec();
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
    let orig_tree = tree_from_original(tf);
    let mut result = Vec::new();
    for &root in &tf.components[T1] {
        let mut current_labels = Vec::new();
        collect_labels(tf, root, &mut current_labels);
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
fn collect_labels(tf: &TwinForest, node: NodeId, out: &mut Vec<Label>) {
    let lbl = tf.label[T1][node as usize];
    if lbl != 0 {
        out.push(lbl);
        return;
    }
    let lc = tf.left[T1][node as usize];
    let rc = tf.right[T1][node as usize];
    if lc != NONE { collect_labels(tf, lc, out); }
    if rc != NONE { collect_labels(tf, rc, out); }
}


/// Reconstruct original Tree from TwinForest's immutable orig_* arrays.
fn tree_from_original(tf: &TwinForest) -> Tree {
    let mut t = Tree::with_capacity(tf.num_leaves);
    t.parent = tf.orig_parent.clone();
    t.left = tf.orig_left.clone();
    t.right = tf.orig_right.clone();
    t.label = tf.orig_label.clone();
    t.label_to_node = vec![NONE; tf.num_leaves as usize + 1];
    for node in 0..tf.num_nodes[T1] as NodeId {
        let lbl = tf.orig_label[node as usize];
        if lbl != 0 && tf.orig_left[node as usize] == NONE {
            t.label_to_node[lbl as usize] = node;
        }
    }
    t.num_leaves = tf.num_leaves;
    t.root = tf.root[T1];
    t.depth = vec![0; tf.num_nodes[T1]];
    t.subtree_size = vec![0; tf.num_nodes[T1]];
    t
}
