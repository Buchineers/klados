//! Core Whidden branch-and-bound algorithm for 2-tree rSPR distance.
//!
//! Implements iterative deepening with the 3-way sibling-pair branching
//! from Whidden & Zeh, optimized with COB, RCOB, and edge protection.

use klados_core::{Instance, NONE, NodeId, SolverStats, Tree, XForest};

use super::search_state::SearchState;
use crate::lower_bound::maf_bounds;
use crate::shi_mestel::extraction::extract_maf_components;
use crate::shi_mestel::forest_nav::{active_children_xf, descend_to_effective};

/// Solve a 2-tree instance using Whidden's algorithm.
pub fn solve(instance: &Instance, stats: &mut SolverStats) -> Option<Vec<Tree>> {
    debug_assert!(instance.num_trees() == 2);
    let num_leaves = instance.num_leaves;

    if num_leaves <= 1 {
        // Return a single component (the first tree), not all input trees.
        return Some(vec![instance.trees[0].clone()]);
    }

    let bounds = maf_bounds(&instance.trees, num_leaves);
    let lb = bounds.lower;
    let ub = bounds.upper.min(num_leaves as usize);

    stats.lower_bound = lb;
    stats.upper_bound = Some(ub);

    // Bounds are in terms of components. rSPR distance = components - 1.
    // k is the rSPR distance (number of Case 3 branching cuts allowed).
    // Singletons (Case 1) are free — they don't consume k.
    let lb_k = if lb > 0 { lb - 1 } else { 0 };
    let ub_k = if ub > 0 { ub - 1 } else { 0 };

    if trace_enabled() {
        eprintln!("[whidden] n={} lb_components={} ub_components={} lb_k={} ub_k={}", num_leaves, lb, ub, lb_k, ub_k);
    }

    // Iterative deepening on rSPR distance k.
    for k in lb_k..=ub_k {
        let forests: Vec<XForest> = instance
            .trees
            .iter()
            .map(|t| XForest::from_tree(t.clone()))
            .collect();
        let num_nodes = forests[0].tree.num_nodes();
        let mut state = SearchState::new(forests);
        init_sibling_pairs(&mut state);
        state.init_singletons(num_leaves as usize);
        let mut scratch = ScratchPad::new(num_nodes);

        if trace_enabled() {
            eprintln!("[whidden] trying k={}", k);
        }

        let result = branch_and_bound(
            &mut state,
            k as i32,
            num_leaves as usize,
            stats,
            &mut scratch,
        );

        if result >= 0 {
            if trace_enabled() {
                eprintln!("[whidden] SUCCESS at k={}, cuts in f0={}, collapses={}",
                    k, state.forests[0].cut_edges.count_ones(..), state.collapses.len());
                let cuts: Vec<usize> = state.forests[0].cut_edges.ones().collect();
                eprintln!("[whidden] f0 cut nodes: {:?}", cuts);
                let cuts2: Vec<usize> = state.forests[1].cut_edges.ones().collect();
                eprintln!("[whidden] f1 cut nodes: {:?}", cuts2);
                eprintln!("[whidden] collapses: {:?}", state.collapses);
            }
            if trace_enabled() {
                eprintln!("[whidden] nodes_explored={} k={}", stats.nodes_explored, k);
            }
            let components = extract_maf_components(
                &state.forests[0],
                &state.collapses,
                num_leaves as usize,
                num_leaves,
            );
            return Some(components);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// ScratchPad: per-invocation caches for effective_parent and effective_depth
// ---------------------------------------------------------------------------

/// Reusable scratch space for caching effective_parent and effective_depth.
/// Allocated once per iterative-deepening round, cleared per find_next_action call.
struct ScratchPad {
    /// Cached effective_parent per (forest_idx, node). NONE = not yet computed.
    /// Index: [forest_idx * num_nodes + node_id].
    /// We use NONE as "uncomputed" and NONE-1 as "computed, result is NONE".
    eff_parent: Vec<NodeId>,
    /// Precomputed effective_depth per (forest_idx, node). u16::MAX = not computed.
    eff_depth: Vec<u16>,
    num_nodes: usize,
    /// Generation counter — bumped on each invalidate(), entries with old
    /// generation are treated as uncomputed.
    generation: u32,
    eff_parent_gen: Vec<u32>,
    eff_depth_gen: Vec<u32>,
}

/// Sentinel for "computed, result is NONE" (distinct from NONE = "not computed").
const CACHED_NONE: NodeId = NONE - 1;

impl ScratchPad {
    fn new(num_nodes: usize) -> Self {
        let total = 2 * num_nodes; // 2 forests
        Self {
            eff_parent: vec![NONE; total],
            eff_depth: vec![u16::MAX; total],
            num_nodes,
            generation: 1,
            eff_parent_gen: vec![0; total],
            eff_depth_gen: vec![0; total],
        }
    }

    /// Invalidate all caches (O(1) via generation bump).
    #[inline]
    fn invalidate(&mut self) {
        self.generation += 1;
    }

    /// Get cached effective_parent, or compute and cache it.
    #[inline]
    fn effective_parent(&mut self, f: &XForest, fi: usize, node: NodeId) -> NodeId {
        let idx = fi * self.num_nodes + node as usize;
        if self.eff_parent_gen[idx] == self.generation {
            let v = self.eff_parent[idx];
            return if v == CACHED_NONE { NONE } else { v };
        }
        let result = effective_parent_raw(f, node);
        self.eff_parent[idx] = if result == NONE { CACHED_NONE } else { result };
        self.eff_parent_gen[idx] = self.generation;
        result
    }

    /// Get cached effective_depth, or compute and cache it.
    #[inline]
    fn effective_depth(&mut self, f: &XForest, fi: usize, node: NodeId) -> u16 {
        let idx = fi * self.num_nodes + node as usize;
        if self.eff_depth_gen[idx] == self.generation {
            return self.eff_depth[idx];
        }
        let mut depth: u16 = 0;
        let mut cur = node;
        loop {
            let ep = self.effective_parent(f, fi, cur);
            if ep == NONE {
                break;
            }
            depth += 1;
            cur = ep;
        }
        self.eff_depth[idx] = depth;
        self.eff_depth_gen[idx] = self.generation;
        depth
    }

    /// Compute effective_sibling using the cache for effective_parent.
    fn effective_sibling(&mut self, f: &XForest, fi: usize, node: NodeId) -> NodeId {
        let eff_parent = self.effective_parent(f, fi, node);
        if eff_parent == NONE {
            return NONE;
        }
        let children = active_children_xf(f, eff_parent);
        if children.len() != 2 {
            return NONE;
        }
        let left = descend_to_effective(f, children[0]);
        let right = descend_to_effective(f, children[1]);

        if is_ancestor_or_equal(f, left, node) {
            right
        } else if is_ancestor_or_equal(f, right, node) {
            left
        } else {
            NONE
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

fn trace_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("WHIDDEN_TRACE").ok().as_deref() == Some("1"))
}

fn cut_all_b_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("WHIDDEN_CUT_ALL_B").ok().as_deref() != Some("0"))
}

fn optimizations_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("WHIDDEN_NO_OPT").ok().as_deref() != Some("1"))
}

fn cob_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| optimizations_enabled() && std::env::var("WHIDDEN_NO_COB").ok().as_deref() != Some("1"))
}

fn rcob_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| optimizations_enabled() && std::env::var("WHIDDEN_NO_RCOB").ok().as_deref() != Some("1"))
}

fn ep_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| optimizations_enabled() && std::env::var("WHIDDEN_NO_EP").ok().as_deref() != Some("1"))
}



/// Build the initial T1 sibling pair set by scanning the tree once.
fn init_sibling_pairs(state: &mut SearchState) {
    state.sibling_pairs.clear();
    let f1 = &state.forests[0];
    for node in f1.tree.post_order() {
        if f1.tree.is_leaf(node) || f1.is_cut(node) {
            continue;
        }
        if f1.live_leaf_count[node as usize] == 0 {
            continue;
        }
        if descend_to_effective(f1, node) != node {
            continue;
        }
        let children = active_children_xf(f1, node);
        if children.len() != 2 {
            continue;
        }
        let left = descend_to_effective(f1, children[0]);
        let right = descend_to_effective(f1, children[1]);
        if !is_effective_leaf(f1, left) || !is_effective_leaf(f1, right) {
            continue;
        }
        let ll = single_live_label(f1, left);
        let lr = single_live_label(f1, right);
        if ll != 0 && lr != 0 {
            state.sibling_pairs.push((ll, lr));
        }
    }
}

/// Check if `t1_node` is the effective parent of a new sibling pair; if so, add it.
fn try_add_pair_at(state: &mut SearchState, t1_node: NodeId) {
    let pair = {
        let f1 = &state.forests[0];
        if t1_node == NONE {
            return;
        }
        let eff = descend_to_effective(f1, t1_node);
        let children = active_children_xf(f1, eff);
        if children.len() != 2 {
            return;
        }
        let left = descend_to_effective(f1, children[0]);
        let right = descend_to_effective(f1, children[1]);
        if !is_effective_leaf(f1, left) || !is_effective_leaf(f1, right) {
            return;
        }
        let ll = single_live_label(f1, left);
        let lr = single_live_label(f1, right);
        if ll == 0 || lr == 0 {
            return;
        }
        (ll, lr)
    };
    state.sibling_pairs.push(pair);
}

fn branch_and_bound(
    state: &mut SearchState,
    k: i32,
    label_space: usize,
    stats: &mut SolverStats,
    scratch: &mut ScratchPad,
) -> i32 {
    branch_and_bound_inner(state, k, label_space, stats, false, None, scratch)
}

/// Inner branch-and-bound.
///
/// When `cut_b_only` is true (CUT_ALL_B), only branch B is tried on the
/// next Case 3 branching step. `forced_pair` specifies the T1 sibling pair
/// that should be re-examined first (matching rspr's prev_T1_a/prev_T1_c).
fn branch_and_bound_inner(
    state: &mut SearchState,
    k: i32,
    label_space: usize,
    stats: &mut SolverStats,
    cut_b_only: bool,
    forced_pair: Option<(NodeId, NodeId)>,
    scratch: &mut ScratchPad,
) -> i32 {
    stats.nodes_explored += 1;

    let mut cut_b_only = cut_b_only;
    let mut forced_pair = forced_pair;

    loop {
        // Invalidate scratch caches — state has changed since last call.
        scratch.invalidate();

        // When cut_b_only with a forced pair, check the forced pair first
        // (after processing singletons). This matches rspr's prev_T1_a/prev_T1_c.
        let action = if cut_b_only && forced_pair.is_some() {
            find_next_action_with_forced(state, label_space, forced_pair.unwrap(), scratch)
        } else {
            find_next_action_mut(state, label_space, scratch)
        };

        match action {
            Action::Done => {
                if trace_enabled() {
                    eprintln!("[whidden] Done, k={}", k);
                }
                return k;
            }
            Action::Singleton { t1_node } => {
                if trace_enabled() {
                    let lbl = state.forests[0].tree.label[t1_node as usize];
                    eprintln!("[whidden] Singleton: t1_node={} label={}, k={}", t1_node, lbl, k);
                }
                state.cut_node(0, t1_node);
                // After cutting, the sibling's effective parent may form a new pair.
                let sib_eff_parent = {
                    let f1 = &state.forests[0];
                    let sib = sibling_in_tree(&f1.tree, t1_node);
                    if sib != NONE {
                        effective_parent_raw(f1, sib)
                    } else {
                        NONE
                    }
                };
                if sib_eff_parent != NONE {
                    try_add_pair_at(state, sib_eff_parent);
                }
            }
            Action::Contract { label_a, label_b } => {
                if trace_enabled() {
                    eprintln!("[whidden] Contract: {} into {}, k={}", label_a, label_b, k);
                }
                // Pop protected_stack when contraction involves a protected node
                // (matches rspr lines 3858-3870).
                if !state.protected_stack.is_empty() {
                    let top = *state.protected_stack.last().unwrap();
                    let top_label = state.forests[1].tree.label[top as usize];
                    if top_label == label_a || top_label == label_b {
                        state.pop_protected_stack();
                    }
                    // rspr checks twice ("CAN THIS HAPPEN TWICE?")
                    if !state.protected_stack.is_empty() {
                        let top2 = *state.protected_stack.last().unwrap();
                        let top2_label = state.forests[1].tree.label[top2 as usize];
                        if top2_label == label_a || top2_label == label_b {
                            state.pop_protected_stack();
                        }
                    }
                }
                state.add_collapse(label_a, label_b);
                // After contraction, the kept label's effective parent may form a new pair.
                let eff_parent = {
                    let f1 = &state.forests[0];
                    let t1_kept = f1.tree.label_to_node[label_b as usize];
                    if t1_kept != NONE {
                        effective_parent_raw(f1, t1_kept)
                    } else {
                        NONE
                    }
                };
                if eff_parent != NONE {
                    try_add_pair_at(state, eff_parent);
                }
                cut_b_only = false;
                forced_pair = None;
            }
            Action::Branch {
                t1_a,
                t1_c,
                t2_a,
                t2_b,
                t2_c,
            } => {
                if k <= 0 {
                    return -1;
                }
                return do_branch(
                    state, k, label_space, stats,
                    t1_a, t1_c, t2_a, t2_b, t2_c, cut_b_only, scratch,
                );
            }
            Action::Failure => {
                return -1;
            }
        }
    }
}

/// The 3-way branching with COB/RCOB and edge protection.
///
/// When `incoming_b_only` is true (CUT_ALL_B), branches A and C are skipped.
/// COB/RCOB are also skipped in b-only mode (matching rspr: `!cut_b_only`).
fn do_branch(
    state: &mut SearchState,
    k: i32,
    label_space: usize,
    stats: &mut SolverStats,
    t1_a: NodeId,
    t1_c: NodeId,
    t2_a: NodeId,
    t2_b: NodeId,
    t2_c: NodeId,
    incoming_b_only: bool,
    scratch: &mut ScratchPad,
) -> i32 {
    if trace_enabled() {
        eprintln!("[whidden] ENTER do_branch k={} b_only={}", k, incoming_b_only);
    }
    let mut cut_a = !incoming_b_only;
    let mut cut_b = true;
    let mut cut_c = !incoming_b_only;
    let mut cut_c_only = false;
    let mut _cut_a_only = false;

    // Compute tree-structural info with immutable borrows, then drop borrows
    // before accessing state.protected / state.protected_stack.
    let t2_a_eff_parent;
    let t2_c_eff_parent;
    let same_component;
    let branch_c_sib_blocked;
    let branch_b_root_blocked;
    {
        let t2 = &state.forests[1];
        let t1 = &state.forests[0];
        let t1_ac = scratch.effective_parent(t1, 0, t1_a);

        t2_a_eff_parent = scratch.effective_parent(t2, 1, t2_a);
        t2_c_eff_parent = scratch.effective_parent(t2, 1, t2_c);

        if optimizations_enabled() {
            // --- COB: Cut-One-B ---
            if cob_enabled() && !incoming_b_only && t2_a_eff_parent != NONE && t2_c_eff_parent != NONE {
                let t2_a_grandparent = scratch.effective_parent(t2, 1, t2_a_eff_parent);
                if t2_a_grandparent != NONE && t2_a_grandparent == t2_c_eff_parent {
                    cut_a = false;
                    cut_c = false;
                    if trace_enabled() {
                        eprintln!("[whidden]   COB: only cut_b");
                    }
                }
            }

            // --- RCOB: Reverse Cut-One-B ---
            if rcob_enabled() && !incoming_b_only && cut_a && cut_c && t1_ac != NONE {
                let t1_s = scratch.effective_sibling(t1, 0, t1_ac);
                if t1_s != NONE {
                    let t1_s_label = single_live_label(t1, t1_s);
                    if t1_s_label != 0 {
                        let t2_s = t2.tree.label_to_node[t1_s_label as usize];
                        if t2_s != NONE {
                            let t2_s_eff_parent = scratch.effective_parent(t2, 1, t2_s);
                            if trace_enabled() {
                                let la = t2.tree.label[t2_a as usize];
                                let lc = t2.tree.label[t2_c as usize];
                                eprintln!("[whidden]   RCOB check: T1_s_label={} T2_a_label={} T2_c_label={} t2_s_ep={:?} t2_a_ep={:?} t2_c_ep={:?}",
                                    t1_s_label, la, lc, t2_s_eff_parent, t2_a_eff_parent, t2_c_eff_parent);
                            }
                            if t2_s_eff_parent != NONE && t2_a_eff_parent != NONE
                                && t2_s_eff_parent == t2_a_eff_parent
                            {
                                cut_a = false;
                                cut_b = false;
                                cut_c_only = true;
                                if trace_enabled() {
                                    eprintln!("[whidden]   RCOB-1: only cut_c");
                                }
                            } else if t2_s_eff_parent != NONE && t2_c_eff_parent != NONE
                                && t2_s_eff_parent == t2_c_eff_parent
                            {
                                cut_a = true;
                                cut_b = false;
                                cut_c = false;
                                _cut_a_only = true;
                                if trace_enabled() {
                                    eprintln!("[whidden]   RCOB-2: only cut_a");
                                }
                            }
                        }

                        // --- CUT_TWO_B (rspr lines 3987-4022) ---
                        // If uncle of (a,c) in T1 is leaf `s`, and in T2 the
                        // grandparent of `a` has `s`'s twin as sibling → cut_b_only.
                        if cut_a && cut_c && t1_s_label != 0 {
                            let t2_s_ctb = t2.tree.label_to_node[t1_s_label as usize];
                            if t2_s_ctb != NONE && t2_a_eff_parent != NONE {
                                let t2_l = scratch.effective_parent(t2, 1, t2_a_eff_parent);
                                if t2_l != NONE && t2_c_eff_parent != NONE {
                                    let t2_c_gp = scratch.effective_parent(t2, 1, t2_c_eff_parent);
                                    if (t2_c_gp != NONE && t2_c_gp == t2_l)
                                        || t2_c_eff_parent == t2_l
                                    {
                                        let t2_l_sib = scratch.effective_sibling(t2, 1, t2_l);
                                        if t2_l_sib != NONE && t2_l_sib == t2_s_ctb {
                                            cut_a = false;
                                            cut_c = false;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // CUT_AC_SEPARATE_COMPONENTS: are T2_a and T2_c in the same component?
        same_component = t2.component_root(t2_a) == t2.component_root(t2_c);

        // Branch C sibling-protection guard: when T2_c is at the root of its
        // component and T2_c's sibling is protected, skip Branch C.
        branch_c_sib_blocked = {
            let t2_c_parent = scratch.effective_parent(t2, 1, t2_c);
            if t2_c_parent != NONE {
                let comp_root = t2.component_root(t2_c);
                let parent_parent = scratch.effective_parent(t2, 1, t2_c_parent);
                if parent_parent == NONE && t2_c_parent == comp_root {
                    let t2_c_sib = scratch.effective_sibling(t2, 1, t2_c);
                    t2_c_sib != NONE && state.is_protected(1, t2_c_sib)
                } else {
                    false
                }
            } else {
                false
            }
        };

        // Branch B root guard: when T2_a.parent is the root and T2_a is protected.
        branch_b_root_blocked = {
            let t2_a_parent = scratch.effective_parent(t2, 1, t2_a);
            t2_a_parent != NONE
                && scratch.effective_parent(t2, 1, t2_a_parent) == NONE
                && state.is_protected(1, t2_a)
        };
    }
    // Immutable borrows of state.forests are now dropped.

    if optimizations_enabled() {
        // --- Edge protection ---
        if ep_enabled() {
            if cut_a && state.is_protected(1, t2_a) {
                cut_a = false;
            }
            if cut_b && state.is_protected(1, t2_b) {
                cut_b = false;
            }
            if cut_c && state.is_protected(1, t2_c) {
                cut_c = false;
            }
        }

        // CUT_AC_SEPARATE_COMPONENTS: skip Branch B when different components.
        if cut_b && !same_component {
            cut_b = false;
        }

        // Branch B root guard.
        if ep_enabled() && cut_b && branch_b_root_blocked {
            cut_b = false;
        }

        // Branch C sibling-protection guard.
        if ep_enabled() && cut_c && branch_c_sib_blocked {
            cut_c = false;
        }
    }

    if !cut_a && !cut_b && !cut_c {
        return -1;
    }

    if trace_enabled() {
        let la = state.forests[1].tree.label[t2_a as usize];
        let lb = state.forests[1].tree.label[t2_b as usize];
        let lc = state.forests[1].tree.label[t2_c as usize];
        eprintln!("[whidden]   do_branch: a={} b={} c={} cut_a={} cut_b={} cut_c={} k={}",
            la, lb, lc, cut_a, cut_b, cut_c, k);
    }

    // --- Branch A: cut T2_a ---
    if cut_a {
        state.checkpoint();
        state.cut_node(1, t2_a);
        let result = branch_and_bound(state, k - 1, label_space, stats, scratch);
        if result >= 0 {
            return result;
        }
        state.rollback();
        if trace_enabled() {
            eprintln!("[whidden]   branch A failed, k={}", k);
        }
    }

    // --- Branch B: cut T2_b ---
    if cut_b {
        state.checkpoint();
        state.cut_node(1, t2_b);
        let use_cut_all_b = if optimizations_enabled() { cut_all_b_enabled() } else { false };
        let forced = if use_cut_all_b { Some((t1_a, t1_c)) } else { None };
        let result = branch_and_bound_inner(
            state, k - 1, label_space, stats, use_cut_all_b, forced, scratch,
        );
        if result >= 0 {
            return result;
        }
        state.rollback();
        if trace_enabled() {
            eprintln!("[whidden]   branch B failed, k={}", k);
        }
    }

    // --- Branch C: cut T2_c ---
    if cut_c {
        state.checkpoint();
        // Edge protection: protect T2_a after cutting T2_c.
        // Skipped when cut_c_only (from RCOB), matching rspr's behavior.
        if ep_enabled() && !cut_c_only && !state.is_protected(1, t2_a) {
            state.protect_edge(1, t2_a);
            state.push_protected_stack(t2_a);
        }
        state.cut_node(1, t2_c);
        let result = branch_and_bound(state, k - 1, label_space, stats, scratch);
        if result >= 0 {
            return result;
        }
        state.rollback();
        if trace_enabled() {
            eprintln!("[whidden]   branch C failed, k={}", k);
        }
    }

    -1
}


// ---------------------------------------------------------------------------
// Action classification
// ---------------------------------------------------------------------------

enum Action {
    /// Agreement forest found — no more work to do.
    Done,
    /// Case 1: singleton — cut above t1_node in T1.
    Singleton { t1_node: NodeId },
    /// Case 2: matching sibling pair — collapse label_a into label_b.
    Contract { label_a: u32, label_b: u32 },
    /// Case 3: non-matching sibling pair — must branch.
    Branch {
        t1_a: NodeId,
        t1_c: NodeId,
        t2_a: NodeId,
        t2_b: NodeId,
        t2_c: NodeId,
    },
    /// No valid action — algorithm stuck (infeasible at this k).
    Failure,
}

/// Like `find_next_action`, but when a forced pair is given (from CUT_ALL_B),
/// check that pair first after processing singletons. This matches rspr's
/// prev_T1_a/prev_T1_c mechanism.
fn find_next_action_with_forced(
    state: &SearchState,
    label_space: usize,
    forced: (NodeId, NodeId),
    scratch: &mut ScratchPad,
) -> Action {
    let f1 = &state.forests[0];
    let f2 = &state.forests[1];

    // Singletons first (same as normal).
    if let Some(action) = find_singleton_incremental(state) {
        return action;
    }

    // Check the forced pair specifically.
    let (t1_a, t1_c) = forced;
    let label_a = single_live_label(f1, descend_to_effective(f1, t1_a));
    let label_c = single_live_label(f1, descend_to_effective(f1, t1_c));

    if label_a != 0 && label_c != 0 {
        let t2_a_node = f2.tree.label_to_node[label_a as usize];
        let t2_c_node = f2.tree.label_to_node[label_c as usize];
        if t2_a_node != NONE && t2_c_node != NONE {
            let t2_a_ep = scratch.effective_parent(f2, 1, t2_a_node);
            let t2_c_ep = scratch.effective_parent(f2, 1, t2_c_node);
            if t2_a_ep != NONE && t2_c_ep != NONE && t2_a_ep == t2_c_ep {
                if trace_enabled() {
                    eprintln!("[whidden]   forced pair ({},{}) now Case 2 → contract", label_a, label_c);
                }
                return Action::Contract {
                    label_a,
                    label_b: label_c,
                };
            }

            // Still Case 3: determine a/c orientation and find T2_b.
            let (t1_a_o, t1_c_o, t2_a_o, t2_c_o) = {
                let depth_a = scratch.effective_depth(f2, 1, t2_a_node);
                let depth_c = scratch.effective_depth(f2, 1, t2_c_node);
                if depth_a >= depth_c {
                    (t1_a, t1_c, t2_a_node, t2_c_node)
                } else {
                    (t1_c, t1_a, t2_c_node, t2_a_node)
                }
            };
            let t2_b_node = scratch.effective_sibling(f2, 1, t2_a_o);
            if trace_enabled() {
                let lb = if t2_b_node != NONE { f2.tree.label[t2_b_node as usize] } else { 0 };
                eprintln!("[whidden]   forced pair ({},{}) still Case 3, T2_b={}",
                    label_a, label_c, lb);
            }
            if t2_b_node != NONE {
                return Action::Branch {
                    t1_a: descend_to_effective(f1, t1_a_o),
                    t1_c: descend_to_effective(f1, t1_c_o),
                    t2_a: t2_a_o,
                    t2_b: t2_b_node,
                    t2_c: t2_c_o,
                };
            }
        }
    }

    // Forced pair is no longer valid — fall through to normal scan.
    // Do full singleton scan here since we don't go through find_next_action_mut.
    if let Some(action) = find_singleton_full(state, label_space) {
        return action;
    }
    if trace_enabled() {
        eprintln!("[whidden]   forced pair invalid, falling through to normal scan");
    }
    find_next_action(state, label_space, scratch)
}

/// Wrapper that tries the incremental pair set first, then rebuilds it from a
/// full tree scan if no valid pair was found.  This ensures correctness even
/// when incremental updates miss a pair: the full rebuild is amortised over
/// the lifetime of the rebuilt set.
fn find_next_action_mut(state: &mut SearchState, label_space: usize, scratch: &mut ScratchPad) -> Action {
    let action = find_next_action(state, label_space, scratch);
    match action {
        Action::Done | Action::Failure => {
            // Incremental sets may have missed entries — rebuild from scratch.
            init_sibling_pairs(state);
            state.init_singletons(label_space);
            find_next_action(state, label_space, scratch)
        }
        other => other,
    }
}

/// Find a singleton using the incremental candidate list only.
/// Scans candidates from the back; stale entries are skipped.
fn find_singleton_incremental(state: &SearchState) -> Option<Action> {
    let f1 = &state.forests[0];
    let f2 = &state.forests[1];

    for &lbl in state.singleton_candidates.iter().rev() {
        if let Some(action) = validate_singleton(f1, f2, lbl) {
            return Some(action);
        }
    }
    None
}

/// Full scan for singletons — used as fallback when incremental misses.
fn find_singleton_full(state: &SearchState, label_space: usize) -> Option<Action> {
    let f1 = &state.forests[0];
    let f2 = &state.forests[1];

    for lbl in 1..=label_space as u32 {
        if lbl as usize >= f1.tree.label_to_node.len() {
            continue;
        }
        if let Some(action) = validate_singleton(f1, f2, lbl) {
            return Some(action);
        }
    }
    None
}

/// Check if a label is currently a valid singleton.
#[inline]
fn validate_singleton(f1: &XForest, f2: &XForest, lbl: u32) -> Option<Action> {
    let t1_node = f1.tree.label_to_node[lbl as usize];
    let t2_node = f2.tree.label_to_node[lbl as usize];
    if t1_node == NONE || t2_node == NONE {
        return None;
    }
    if f1.live_leaf_count[t1_node as usize] == 0 {
        return None;
    }
    // Quick rejection: if T2 leaf's parent exists and is not cut, the component
    // has at least the parent + leaf (so size >= 2, not a singleton).
    let t2_parent = f2.tree.parent[t2_node as usize];
    if t2_parent != NONE && !f2.is_cut(t2_node) {
        // Parent exists and the edge to parent is not cut.
        // Check if parent has other live children — if so, not a singleton.
        if f2.live_leaf_count[t2_parent as usize] > 1 {
            return None;
        }
    }
    // Full component root check for edge cases.
    let t2_comp_root = f2.component_root(t2_node);
    if f2.live_leaf_count[t2_comp_root as usize] != 1 {
        return None;
    }
    // T1: the twin must NOT be a singleton (must be in a component with >1 leaves)
    let t1_comp_root = f1.component_root(t1_node);
    if f1.live_leaf_count[t1_comp_root as usize] <= 1 {
        return None;
    }
    Some(Action::Singleton { t1_node })
}

/// Scan forests for the next action using the incrementally maintained pair set.
///
/// Priority: singletons first, then sibling pairs (Case 2 before Case 3).
/// Returns Done only when no sibling pairs and no singletons remain.
fn find_next_action(state: &SearchState, label_space: usize, scratch: &mut ScratchPad) -> Action {
    let f1 = &state.forests[0];
    let f2 = &state.forests[1];

    // 1. Singleton check (incremental only — full scan in find_next_action_mut fallback)
    if let Some(action) = find_singleton_incremental(state) {
        return action;
    }

    // Protected stack disconnect guard (rspr lines 3716-3728):
    // If the protected stack top's T1 twin has been disconnected
    // (cut off from the main component), this subtree is infeasible.
    if ep_enabled() && !state.protected_stack.is_empty() {
        let top = *state.protected_stack.last().unwrap();
        let top_label = f2.tree.label[top as usize];
        if top_label != 0 {
            let t1_twin = f1.tree.label_to_node[top_label as usize];
            if t1_twin != NONE
                && f1.live_leaf_count[t1_twin as usize] > 0
            {
                let t1_twin_parent = f1.tree.parent[t1_twin as usize];
                if t1_twin_parent == NONE || f1.is_cut(t1_twin) {
                    // T1 twin is a component root by itself — disconnected
                    if t1_twin != f1.tree.root {
                        return Action::Failure;
                    }
                }
            }
        }
    }

    // 2. Iterate the maintained sibling pair set instead of a full tree scan.
    //    DEEPEST_ORDER selects the Case 3 pair with greatest effective depth.
    //    DEEPEST_PROTECTED_ORDER constrains to pairs within the DPO scope.
    let mut best_branch: Option<Action> = None;
    let mut best_t1_depth: u16 = 0;
    let mut best_t2_depth: u16 = 0;

    let dpo_scope: NodeId = if ep_enabled() {
        if let Some(&last_protected_t2) = state.protected_stack.last() {
            let label = f2.tree.label[last_protected_t2 as usize];
            if label != 0 {
                let t1_twin = f1.tree.label_to_node[label as usize];
                if t1_twin != NONE {
                    scratch.effective_parent(f1, 0, t1_twin)
                } else {
                    NONE
                }
            } else {
                NONE
            }
        } else {
            NONE
        }
    } else {
        NONE
    };

    if trace_enabled() && dpo_scope != NONE {
        eprintln!("[whidden]   DPO scope: node {} (stack len={})", dpo_scope, state.protected_stack.len());
    }

    for &(label_l, label_r) in &state.sibling_pairs {
        // Validity: both labels still active
        let t1_l = f1.tree.label_to_node[label_l as usize];
        let t1_r = f1.tree.label_to_node[label_r as usize];
        if t1_l == NONE || t1_r == NONE {
            continue;
        }
        if f1.live_leaf_count[t1_l as usize] == 0
            || f1.live_leaf_count[t1_r as usize] == 0
        {
            continue;
        }

        // Check they're still T1 siblings (same effective parent)
        let ep_l = scratch.effective_parent(f1, 0, t1_l);
        let ep_r = scratch.effective_parent(f1, 0, t1_r);
        if ep_l != ep_r || ep_l == NONE {
            continue;
        }

        // T2 counterparts
        let t2_l = f2.tree.label_to_node[label_l as usize];
        let t2_r = f2.tree.label_to_node[label_r as usize];
        if t2_l == NONE || t2_r == NONE {
            continue;
        }

        let t2_l_eff_parent = scratch.effective_parent(f2, 1, t2_l);
        let t2_r_eff_parent = scratch.effective_parent(f2, 1, t2_r);

        if t2_l_eff_parent != NONE
            && t2_r_eff_parent != NONE
            && t2_l_eff_parent == t2_r_eff_parent
        {
            return Action::Contract {
                label_a: label_l,
                label_b: label_r,
            };
        }

        // PREFER_NONBRANCHING: select immediately if branching is trivially determined.
        if is_nonbranching(f1, f2, t1_l, t1_r, t2_l, t2_r, state, scratch) {
            let (t1_a, t1_c, t2_a_node, t2_c_node) = {
                let dl = scratch.effective_depth(f2, 1, t2_l);
                let dr = scratch.effective_depth(f2, 1, t2_r);
                if dl >= dr {
                    (t1_l, t1_r, t2_l, t2_r)
                } else {
                    (t1_r, t1_l, t2_r, t2_l)
                }
            };
            let t2_b_node = scratch.effective_sibling(f2, 1, t2_a_node);
            if t2_b_node != NONE {
                return Action::Branch {
                    t1_a,
                    t1_c,
                    t2_a: t2_a_node,
                    t2_b: t2_b_node,
                    t2_c: t2_c_node,
                };
            }
        }

        // Case 3: DEEPEST_ORDER — track best by (T1 depth, T2 depth).
        let t1_depth = scratch.effective_depth(f1, 0, t1_l).max(scratch.effective_depth(f1, 0, t1_r));
        let t2_depth_l = scratch.effective_depth(f2, 1, t2_l);
        let t2_depth_r = scratch.effective_depth(f2, 1, t2_r);
        let t2_depth = t2_depth_l.max(t2_depth_r);

        if dpo_scope != NONE
            && !is_effective_descendant_or_equal(f1, dpo_scope, t1_l)
        {
            if trace_enabled() {
                eprintln!("[whidden]   DPO: rejecting pair ({},{}) - not in scope", label_l, label_r);
            }
            continue;
        }

        if best_branch.is_none()
            || t1_depth > best_t1_depth
            || (t1_depth == best_t1_depth && t2_depth > best_t2_depth)
        {
            let (t1_a, t1_c, t2_a_node, t2_c_node) = if t2_depth_l >= t2_depth_r {
                (t1_l, t1_r, t2_l, t2_r)
            } else {
                (t1_r, t1_l, t2_r, t2_l)
            };

            let t2_b_node = scratch.effective_sibling(f2, 1, t2_a_node);
            if t2_b_node != NONE {
                if trace_enabled() {
                    let la = f2.tree.label[t2_a_node as usize];
                    let lc = f2.tree.label[t2_c_node as usize];
                    let lb = f2.tree.label[t2_b_node as usize];
                    eprintln!("[whidden]   Case 3 candidate: labels ({}, {}) in T1, T1d={} T2d={}", label_l, label_r, t1_depth, t2_depth);
                    eprintln!("[whidden]   Branch setup: T2_a={} T2_b={} T2_c={} (nodes {},{},{})",
                        la, lb, lc, t2_a_node, t2_b_node, t2_c_node);
                }
                best_branch = Some(Action::Branch {
                    t1_a,
                    t1_c,
                    t2_a: t2_a_node,
                    t2_b: t2_b_node,
                    t2_c: t2_c_node,
                });
                best_t1_depth = t1_depth;
                best_t2_depth = t2_depth;
            }
        }
    }

    if let Some(branch) = best_branch {
        if trace_enabled() { eprintln!("[whidden]   returning Branch action"); }
        return branch;
    }

    if trace_enabled() { eprintln!("[whidden]   no branch found, checking components_match"); }

    use crate::shi_mestel::forest_nav::component_leaf_sets_xf;
    let comps1 = component_leaf_sets_xf(f1, label_space);
    let comps2 = component_leaf_sets_xf(f2, label_space);
    if components_match(&comps1, &comps2) {
        if trace_enabled() { eprintln!("[whidden]   components MATCH -> Done"); }
        return Action::Done;
    }

    if trace_enabled() {
        eprintln!("[whidden] BUG: no action found but components don't match!");
        eprintln!("[whidden]   comps1: {:?}", comps1.iter().map(|c| c.ones().collect::<Vec<_>>()).collect::<Vec<_>>());
        eprintln!("[whidden]   comps2: {:?}", comps2.iter().map(|c| c.ones().collect::<Vec<_>>()).collect::<Vec<_>>());
    }
    Action::Failure
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn components_match(a: &[fixedbitset::FixedBitSet], b: &[fixedbitset::FixedBitSet]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut used = vec![false; b.len()];
    for comp_a in a {
        let mut found = false;
        for (j, comp_b) in b.iter().enumerate() {
            if !used[j] && comp_a == comp_b {
                used[j] = true;
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }
    true
}

fn is_effective_leaf(f: &XForest, node: NodeId) -> bool {
    if f.tree.is_leaf(node) {
        return true;
    }
    active_children_xf(f, node).is_empty()
}

fn single_live_label(f: &XForest, node: NodeId) -> u32 {
    if f.live_leaf_count[node as usize] != 1 {
        return 0;
    }
    f.live_leafsets[node as usize].ones().next().unwrap() as u32
}

/// Raw (uncached) effective_parent — used when scratch is not available.
fn effective_parent_raw(f: &XForest, node: NodeId) -> NodeId {
    if f.is_cut(node) || f.tree.parent[node as usize] == NONE {
        return NONE;
    }
    let mut cur = f.tree.parent[node as usize];
    loop {
        if cur == NONE {
            return NONE;
        }
        let children = active_children_xf(f, cur);
        if children.len() >= 2 {
            return cur;
        }
        if f.is_cut(cur) || f.tree.parent[cur as usize] == NONE {
            return cur;
        }
        cur = f.tree.parent[cur as usize];
    }
}

/// Check if a Case 3 pair is "nonbranching" — i.e., the branching is trivially
/// determined (only one branch is viable).  Matches rspr's `is_nonbranching`.
/// When true, the pair is selected immediately (before deeper Case 3 pairs) via
/// PREFER_NONBRANCHING, ensuring EP-protected nodes don't block forced branches.
fn is_nonbranching(
    f1: &XForest,
    f2: &XForest,
    t1_left: NodeId,
    _t1_right: NodeId,
    t2_l: NodeId,
    t2_r: NodeId,
    state: &SearchState,
    scratch: &mut ScratchPad,
) -> bool {
    // Orient: T2_a = deeper twin.
    let (t2_a, t2_c) = {
        let dl = scratch.effective_depth(f2, 1, t2_l);
        let dr = scratch.effective_depth(f2, 1, t2_r);
        if dl >= dr { (t2_l, t2_r) } else { (t2_r, t2_l) }
    };
    let t2_a_ep = scratch.effective_parent(f2, 1, t2_a);
    let t2_c_ep = scratch.effective_parent(f2, 1, t2_c);

    // 1. Two or more of {T2_a, T2_b, T2_c} are protected → nonbranching.
    let mut num_prot = state.is_protected(1, t2_a) as i32
        + state.is_protected(1, t2_c) as i32;
    let t2_b = scratch.effective_sibling(f2, 1, t2_a);
    if t2_b != NONE {
        num_prot += state.is_protected(1, t2_b) as i32;
    }
    if num_prot >= 2 {
        return true;
    }

    // 2. COB applies: T2_a.parent.parent == T2_c.parent → only Branch B.
    if cob_enabled() && t2_a_ep != NONE && t2_c_ep != NONE {
        let t2_a_gp = scratch.effective_parent(f2, 1, t2_a_ep);
        if t2_a_gp != NONE && t2_a_gp == t2_c_ep {
            return true;
        }
    }

    // 3. RCOB applies: T1_s is an effective leaf whose T2 twin shares a parent
    //    with T2_a or T2_c → only one branch.
    let t1_left_ep = scratch.effective_parent(f1, 0, t1_left);
    if rcob_enabled() && t1_left_ep != NONE {
        let t1_ac = t1_left_ep;
        let t1_ac_parent = scratch.effective_parent(f1, 0, t1_ac);
        if t1_ac_parent != NONE {
            let t1_s = scratch.effective_sibling(f1, 0, t1_ac);
            if t1_s != NONE {
                let t1_s_label = single_live_label(f1, t1_s);
                if t1_s_label != 0 {
                    let t2_s = f2.tree.label_to_node[t1_s_label as usize];
                    if t2_s != NONE {
                        let t2_s_ep = scratch.effective_parent(f2, 1, t2_s);
                        if (t2_s_ep != NONE && t2_a_ep != NONE && t2_s_ep == t2_a_ep)
                            || (t2_s_ep != NONE && t2_c_ep != NONE && t2_s_ep == t2_c_ep)
                        {
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}

fn is_ancestor_or_equal(f: &XForest, ancestor: NodeId, mut node: NodeId) -> bool {
    loop {
        if node == ancestor {
            return true;
        }
        if node == NONE || f.is_cut(node) {
            return false;
        }
        node = f.tree.parent[node as usize];
    }
}

/// Check if `node` is an effective descendant of `ancestor` in the forest.
/// Unlike `is_ancestor_or_equal`, this navigates through the PHYSICAL tree
/// structure (following parent pointers including through cut/contracted nodes)
/// since the effective subtree of `ancestor` contains all physical descendants.
fn is_effective_descendant_or_equal(f: &XForest, ancestor: NodeId, node: NodeId) -> bool {
    is_ancestor_or_equal(f, ancestor, node)
}

fn sibling_in_tree(tree: &klados_core::Tree, node: NodeId) -> NodeId {
    let p = tree.parent[node as usize];
    if p == NONE {
        return NONE;
    }
    let left = tree.left[p as usize];
    if left == node {
        tree.right[p as usize]
    } else {
        left
    }
}

// ---------------------------------------------------------------------------
// NOTE: Dynamic cluster decomposition was attempted here but removed.
// The static cluster reduction in mod.rs already handles common clades
// before the B&B starts. Per-branch-node decomposition was too expensive
// (~O(n) per branch for cluster detection + O(n·k) for recursive solve).
// A cheaper lower bound (component count) could be used for pruning in future.
