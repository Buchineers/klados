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
        let mut state = SearchState::new(forests);

        if trace_enabled() {
            eprintln!("[whidden] trying k={}", k);
        }

        let result = branch_and_bound(
            &mut state,
            k as i32,
            num_leaves as usize,
            stats,
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

/// Main branch-and-bound loop.
///
/// Returns the remaining k if successful (>= 0), or -1 if no solution.
/// `k` is the number of cuts still allowed.
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


fn branch_and_bound(
    state: &mut SearchState,
    k: i32,
    label_space: usize,
    stats: &mut SolverStats,
) -> i32 {
    branch_and_bound_inner(state, k, label_space, stats, false, None)
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
) -> i32 {
    stats.nodes_explored += 1;

    let mut cut_b_only = cut_b_only;
    let mut forced_pair = forced_pair;

    loop {
        // When cut_b_only with a forced pair, check the forced pair first
        // (after processing singletons). This matches rspr's prev_T1_a/prev_T1_c.
        let action = if cut_b_only && forced_pair.is_some() {
            find_next_action_with_forced(state, label_space, forced_pair.unwrap())
        } else {
            find_next_action(state, label_space)
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
                    t1_a, t1_c, t2_a, t2_b, t2_c, cut_b_only,
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
        let t1_ac = effective_parent(t1, t1_a);

        t2_a_eff_parent = effective_parent(t2, t2_a);
        t2_c_eff_parent = effective_parent(t2, t2_c);

        if optimizations_enabled() {
            // --- COB: Cut-One-B ---
            if cob_enabled() && !incoming_b_only && t2_a_eff_parent != NONE && t2_c_eff_parent != NONE {
                let t2_a_grandparent = effective_parent(t2, t2_a_eff_parent);
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
                let t1_s = effective_sibling(t1, t1_ac);
                if t1_s != NONE {
                    let t1_s_label = single_live_label(t1, t1_s);
                    if t1_s_label != 0 {
                        let t2_s = t2.tree.label_to_node[t1_s_label as usize];
                        if t2_s != NONE {
                            let t2_s_eff_parent = effective_parent(t2, t2_s);
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
                    }
                }
            }
        }

        // CUT_AC_SEPARATE_COMPONENTS: are T2_a and T2_c in the same component?
        same_component = t2.component_root(t2_a) == t2.component_root(t2_c);

        // Branch C sibling-protection guard: when T2_c is at the root of its
        // component and T2_c's sibling is protected, skip Branch C.
        branch_c_sib_blocked = {
            let t2_c_parent = effective_parent(t2, t2_c);
            if t2_c_parent != NONE {
                let comp_root = t2.component_root(t2_c);
                let parent_parent = effective_parent(t2, t2_c_parent);
                if parent_parent == NONE && t2_c_parent == comp_root {
                    let t2_c_sib = effective_sibling(t2, t2_c);
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
            let t2_a_parent = effective_parent(t2, t2_a);
            t2_a_parent != NONE
                && effective_parent(t2, t2_a_parent) == NONE
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
        let result = branch_and_bound(state, k - 1, label_space, stats);
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
            state, k - 1, label_space, stats, use_cut_all_b, forced,
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
        let result = branch_and_bound(state, k - 1, label_space, stats);
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
) -> Action {
    let f1 = &state.forests[0];
    let f2 = &state.forests[1];

    // Singletons first (same as normal).
    for lbl in 1..=label_space as u32 {
        if lbl as usize >= f1.tree.label_to_node.len() {
            continue;
        }
        let t1_node = f1.tree.label_to_node[lbl as usize];
        let t2_node = f2.tree.label_to_node[lbl as usize];
        if t1_node == NONE || t2_node == NONE {
            continue;
        }
        if !f1.live_leafsets[t1_node as usize].contains(lbl as usize) {
            continue;
        }
        let t2_comp_root = f2.component_root(t2_node);
        let t2_comp_size = f2.live_leafsets[t2_comp_root as usize].count_ones(..);
        let t1_comp_root = f1.component_root(t1_node);
        let t1_comp_size = f1.live_leafsets[t1_comp_root as usize].count_ones(..);
        if trace_enabled() {
            eprintln!("[whidden]   singleton check: lbl={} t1_comp_size={} t2_comp_size={}", lbl, t1_comp_size, t2_comp_size);
        }
        if t2_comp_size == 1 && t1_comp_size > 1 {
            return Action::Singleton { t1_node };
        }
    }

    // Check the forced pair specifically.
    let (t1_a, t1_c) = forced;
    let label_a = single_live_label(f1, descend_to_effective(f1, t1_a));
    let label_c = single_live_label(f1, descend_to_effective(f1, t1_c));

    if label_a != 0 && label_c != 0 {
        let t2_a_node = f2.tree.label_to_node[label_a as usize];
        let t2_c_node = f2.tree.label_to_node[label_c as usize];
        if t2_a_node != NONE && t2_c_node != NONE {
            let t2_a_ep = effective_parent(f2, t2_a_node);
            let t2_c_ep = effective_parent(f2, t2_c_node);
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
                let depth_a = effective_depth(f2, t2_a_node);
                let depth_c = effective_depth(f2, t2_c_node);
                if depth_a >= depth_c {
                    (t1_a, t1_c, t2_a_node, t2_c_node)
                } else {
                    (t1_c, t1_a, t2_c_node, t2_a_node)
                }
            };
            let t2_b_node = effective_sibling(f2, t2_a_o);
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
    if trace_enabled() {
        eprintln!("[whidden]   forced pair invalid, falling through to normal scan");
    }
    find_next_action(state, label_space)
}

/// Scan forests for the next action.
///
/// Priority: singletons first, then sibling pairs (Case 2 before Case 3).
/// Returns Done only when no sibling pairs and no singletons remain.
fn find_next_action(state: &SearchState, label_space: usize) -> Action {
    let f1 = &state.forests[0];
    let f2 = &state.forests[1];

    // First, check for singletons: a leaf whose component in one forest
    // is a single leaf while its component in the other has multiple leaves.
    // In rspr, a singleton in T2 means: a leaf in T2 that is NOT in the
    // first component, and has become isolated (component = just itself).
    // Equivalently: cut above the twin in T1.
    //
    // We check: for each label, if it's in a 1-leaf component in T2 but
    // a multi-leaf component in T1 (or vice versa).
    for lbl in 1..=label_space as u32 {
        if lbl as usize >= f1.tree.label_to_node.len() {
            continue;
        }
        let t1_node = f1.tree.label_to_node[lbl as usize];
        let t2_node = f2.tree.label_to_node[lbl as usize];
        if t1_node == NONE || t2_node == NONE {
            continue;
        }
        if !f1.live_leafsets[t1_node as usize].contains(lbl as usize) {
            continue; // deactivated
        }

        let t2_comp_root = f2.component_root(t2_node);
        let t2_comp_size = f2.live_leafsets[t2_comp_root as usize].count_ones(..);
        let t1_comp_root = f1.component_root(t1_node);
        let t1_comp_size = f1.live_leafsets[t1_comp_root as usize].count_ones(..);
        if trace_enabled() {
            eprintln!("[whidden]   singleton check: lbl={} t1_comp_size={} t2_comp_size={}", lbl, t1_comp_size, t2_comp_size);
        }
        if t2_comp_size == 1 {
            if t1_comp_size > 1 {
                return Action::Singleton { t1_node };
            }
        }
    }

    // Next, find sibling pairs in T1.
    // A sibling pair is two effective leaves that share an effective parent.
    //
    // DEEPEST_ORDER (matching rspr): select the Case 3 pair with the greatest
    // effective depth.  Primary key = max T1 depth of the pair, secondary =
    // max T2 depth of the twins.  This ordering is required for edge-protection
    // soundness: processing deeper pairs first ensures that protecting T2_a
    // after Branch C does not prune valid cuts at shallower levels.
    //
    // DEEPEST_PROTECTED_ORDER: when the protected stack is non-empty, only
    // accept pairs where T1_a is a descendant of the T1 parent of the last
    // protected T2 node's twin.  This confines EP's scope to the subtree
    // where the protection is semantically valid.
    let mut best_branch: Option<Action> = None;
    let mut best_t1_depth: usize = 0;
    let mut best_t2_depth: usize = 0;

    // Compute the DEEPEST_PROTECTED_ORDER constraint scope.
    let dpo_scope: NodeId = if ep_enabled() {
        if let Some(&last_protected_t2) = state.protected_stack.last() {
            let label = f2.tree.label[last_protected_t2 as usize];
            if label != 0 {
                let t1_twin = f1.tree.label_to_node[label as usize];
                if t1_twin != NONE {
                    effective_parent(f1, t1_twin)
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

    for node in f1.tree.post_order() {
        if f1.tree.is_leaf(node) {
            continue;
        }
        if f1.is_cut(node) {
            continue;
        }
        if f1.live_leafsets[node as usize].count_ones(..) == 0 {
            continue;
        }

        let eff = descend_to_effective(f1, node);
        let children = active_children_xf(f1, eff);
        if children.len() != 2 {
            continue;
        }

        let left = descend_to_effective(f1, children[0]);
        let right = descend_to_effective(f1, children[1]);

        if !is_effective_leaf(f1, left) || !is_effective_leaf(f1, right) {
            continue;
        }

        let label_l = single_live_label(f1, left);
        let label_r = single_live_label(f1, right);
        if label_l == 0 || label_r == 0 {
            continue;
        }

        // Found sibling pair (label_l, label_r) in T1.
        let t2_l = f2.tree.label_to_node[label_l as usize];
        let t2_r = f2.tree.label_to_node[label_r as usize];
        if t2_l == NONE || t2_r == NONE {
            continue;
        }

        let t2_l_eff_parent = effective_parent(f2, t2_l);
        let t2_r_eff_parent = effective_parent(f2, t2_r);

        if t2_l_eff_parent != NONE
            && t2_r_eff_parent != NONE
            && t2_l_eff_parent == t2_r_eff_parent
        {
            // Case 2: matching sibling pair → contract immediately.
            return Action::Contract {
                label_a: label_l,
                label_b: label_r,
            };
        }

        // PREFER_NONBRANCHING (rspr): if this Case 3 pair has trivially
        // determined branching, select it immediately (before deeper pairs).
        if is_nonbranching(f1, f2, left, right, t2_l, t2_r, state) {
            let (t1_a, t1_c, t2_a_node, t2_c_node) = {
                let dl = effective_depth(f2, t2_l);
                let dr = effective_depth(f2, t2_r);
                if dl >= dr { (left, right, t2_l, t2_r) } else { (right, left, t2_r, t2_l) }
            };
            let t2_b_node = effective_sibling(f2, t2_a_node);
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

        // Case 3: non-matching → potential branch point.
        // DEEPEST_ORDER: pick the deepest pair by (T1 depth, T2 depth).
        let t1_depth = effective_depth(f1, left).max(effective_depth(f1, right));
        let t2_depth_l = effective_depth(f2, t2_l);
        let t2_depth_r = effective_depth(f2, t2_r);
        let t2_depth = t2_depth_l.max(t2_depth_r);

        // DEEPEST_PROTECTED_ORDER: only accept pairs within the DPO scope.
        if dpo_scope != NONE
            && !is_effective_descendant_or_equal(f1, dpo_scope, left)
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
            // T2_a = the deeper twin in T2 (convention from rspr).
            let (t1_a, t1_c, t2_a_node, t2_c_node) = {
                if t2_depth_l >= t2_depth_r {
                    (left, right, t2_l, t2_r)
                } else {
                    (right, left, t2_r, t2_l)
                }
            };

            let t2_b_node = effective_sibling(f2, t2_a_node);
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

    // If we found a branch point, return it.
    if let Some(branch) = best_branch {
        if trace_enabled() { eprintln!("[whidden]   returning Branch action"); }
        return branch;
    }

    if trace_enabled() { eprintln!("[whidden]   no branch found, checking components_match"); }

    // No singletons and no sibling pairs → check if done.
    use crate::shi_mestel::forest_nav::component_leaf_sets_xf;
    let comps1 = component_leaf_sets_xf(f1, label_space);
    let comps2 = component_leaf_sets_xf(f2, label_space);
    if components_match(&comps1, &comps2) {
        if trace_enabled() { eprintln!("[whidden]   components MATCH -> Done"); }
        return Action::Done;
    }

    // Components don't match but no actionable sibling pair found.
    // This shouldn't happen if the algorithm is correct.
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
    let ls = &f.live_leafsets[node as usize];
    if ls.count_ones(..) != 1 {
        return 0;
    }
    ls.ones().next().unwrap() as u32
}

fn effective_parent(f: &XForest, node: NodeId) -> NodeId {
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

fn effective_sibling(f: &XForest, node: NodeId) -> NodeId {
    let eff_parent = effective_parent(f, node);
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

/// Compute effective depth of a node in the forest.
/// Counts the number of effective ancestors (nodes with >= 2 active children)
/// between the node and its component root. This is the depth in the
/// "contracted" tree that rspr would physically build.
fn effective_depth(f: &XForest, node: NodeId) -> usize {
    let mut depth = 0;
    let mut cur = node;
    loop {
        let ep = effective_parent(f, cur);
        if ep == NONE {
            break;
        }
        depth += 1;
        cur = ep;
    }
    depth
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
) -> bool {
    // Orient: T2_a = deeper twin.
    let (t2_a, t2_c) = {
        let dl = effective_depth(f2, t2_l);
        let dr = effective_depth(f2, t2_r);
        if dl >= dr { (t2_l, t2_r) } else { (t2_r, t2_l) }
    };
    let t2_a_ep = effective_parent(f2, t2_a);
    let t2_c_ep = effective_parent(f2, t2_c);

    // 1. Two or more of {T2_a, T2_b, T2_c} are protected → nonbranching.
    let mut num_prot = state.is_protected(1, t2_a) as i32
        + state.is_protected(1, t2_c) as i32;
    let t2_b = effective_sibling(f2, t2_a);
    if t2_b != NONE {
        num_prot += state.is_protected(1, t2_b) as i32;
    }
    if num_prot >= 2 {
        return true;
    }

    // 2. COB applies: T2_a.parent.parent == T2_c.parent → only Branch B.
    if cob_enabled() && t2_a_ep != NONE && t2_c_ep != NONE {
        let t2_a_gp = effective_parent(f2, t2_a_ep);
        if t2_a_gp != NONE && t2_a_gp == t2_c_ep {
            return true;
        }
    }

    // 3. RCOB applies: T1_s is an effective leaf whose T2 twin shares a parent
    //    with T2_a or T2_c → only one branch.
    let t1_left_ep = effective_parent(f1, t1_left);
    if rcob_enabled() && t1_left_ep != NONE {
        let t1_ac = t1_left_ep;
        let t1_ac_parent = effective_parent(f1, t1_ac);
        if t1_ac_parent != NONE {
            let t1_s = effective_sibling(f1, t1_ac);
            if t1_s != NONE {
                let t1_s_label = single_live_label(f1, t1_s);
                if t1_s_label != 0 {
                    let t2_s = f2.tree.label_to_node[t1_s_label as usize];
                    if t2_s != NONE {
                        let t2_s_ep = effective_parent(f2, t2_s);
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
