//! Branching rules: BR-LSI step and Case 2 sibling-pair branching.

use fixedbitset::FixedBitSet;
use fxhash::FxHashMap;
use klados_core::{NONE, NodeId, XForest};

use super::forest_nav::{active_children_xf, forest_lca};
use super::reduction::{find_all_sibling_pairs, find_violating_pair_cached};
use super::search_state::SearchState;
use super::transposition::{TTEntry, ZobristTable};
use super::utils::has_intersection;
use crate::SolverStats;

pub fn br_lsi_step(
    state: &mut SearchState,
    target_s: usize,
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    comp_sets: &[Vec<FixedBitSet>],
    zobrist: &ZobristTable,
    tt: &mut FxHashMap<u64, TTEntry>,
) -> Option<Vec<klados_core::Tree>> {
    let (i, j) = find_violating_pair_cached(comp_sets)?;

    let (target_idx, v1, v2) = if let Some((_v, v1, v2)) =
        find_branching_vertex_cached(&state.forests[i], &state.forests[j], &comp_sets[j])
    {
        (i, v1, v2)
    } else if let Some((_v, v1, v2)) =
        find_branching_vertex_cached(&state.forests[j], &state.forests[i], &comp_sets[i])
    {
        (j, v1, v2)
    } else {
        super::trace!("no branching vertex found for pair ({}, {})", i, j);
        stats.branches_pruned += 1;
        return None;
    };

    super::trace!("BR1: forest={}, v1={}, v2={}", target_idx, v1, v2);

    if v1 != state.forests[target_idx].tree.root && !state.forests[target_idx].is_cut(v1) {
        state.checkpoint();
        state.cut_node(target_idx, v1);
        if let Some(result) =
            super::algorithm::alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    if v2 != state.forests[target_idx].tree.root && !state.forests[target_idx].is_cut(v2) {
        state.checkpoint();
        state.cut_node(target_idx, v2);
        if let Some(result) =
            super::algorithm::alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    stats.branches_pruned += 1;
    None
}

pub fn find_branching_vertex_cached(
    fi: &XForest,
    _fj: &XForest,
    fj_components: &[FixedBitSet],
) -> Option<(NodeId, NodeId, NodeId)> {
    for node in fi.tree.pre_order() {
        if fi.live_leafsets[node as usize].count_ones(..) == 0 {
            continue;
        }
        let children = active_children_xf(fi, node);
        if children.len() < 2 {
            continue;
        }

        let (c1, c2) = (children[0], children[1]);
        let ls1 = &fi.live_leafsets[c1 as usize];
        let ls2 = &fi.live_leafsets[c2 as usize];

        for comp in fj_components {
            let c1_inter = has_intersection(ls1, comp);
            let c2_inter = has_intersection(ls2, comp);

            if c1_inter && !c2_inter && super::utils::is_subset(ls1, comp) {
                return Some((node, c1, c2));
            }
            if c2_inter && !c1_inter && super::utils::is_subset(ls2, comp) {
                return Some((node, c2, c1));
            }
        }
    }
    None
}

pub fn find_best_sibling_pair(forests: &[XForest], label_space: usize) -> Option<(u32, u32)> {
    let mut seen = fxhash::FxHashSet::default();
    let mut all_pairs: Vec<(u32, u32)> = Vec::new();
    for forest in forests {
        for pair in find_all_sibling_pairs(forest, label_space) {
            if seen.insert(pair) {
                all_pairs.push(pair);
            }
        }
    }

    if all_pairs.len() <= 1 {
        return all_pairs.first().copied();
    }

    let mut best_pair = all_pairs[0];
    let mut best_score: i32 = i32::MIN;

    for &(a, b) in &all_pairs {
        let score = score_sibling_pair(forests, a, b);
        if score > best_score {
            best_score = score;
            best_pair = (a, b);
        }
    }

    Some(best_pair)
}

fn score_sibling_pair(forests: &[XForest], a: u32, b: u32) -> i32 {
    let e_sizes: Vec<usize> = forests.iter().map(|f| compute_e_f(f, a, b).len()).collect();
    let max_e = e_sizes.iter().copied().max().unwrap_or(0);

    let omega1: Vec<usize> = e_sizes
        .iter()
        .enumerate()
        .filter(|(_, e)| **e == 1)
        .map(|(i, _)| i)
        .collect();

    let base_score;

    if !omega1.is_empty() {
        let lca_sets: Vec<FixedBitSet> = omega1
            .iter()
            .map(|&i| lca_leafset(&forests[i], a, b))
            .collect();
        let all_same_lca = lca_sets.windows(2).all(|w| w[0] == w[1]);
        if all_same_lca {
            base_score = 10000;
        } else if max_e >= 3 {
            base_score = 5000 + max_e as i32;
        } else if max_e >= 2 {
            base_score = 3000;
        } else {
            base_score = 0;
        }
    } else if max_e >= 3 {
        base_score = 5000 + max_e as i32;
    } else if max_e >= 2 {
        base_score = 3000;
    } else {
        base_score = -1000;
    }

    // DEEPEST_ORDER: prefer deeper sibling pairs (deeper = more specific = better pruning).
    // Use max depth across all forests as a tiebreaker.
    let mut max_depth = 0i32;
    for f in forests {
        let a_node = f.tree.label_to_node[a as usize];
        let b_node = f.tree.label_to_node[b as usize];
        if f.live_leafsets[a_node as usize].count_ones(..) > 0 {
            max_depth = max_depth.max(f.tree.depth[a_node as usize] as i32);
        }
        if f.live_leafsets[b_node as usize].count_ones(..) > 0 {
            max_depth = max_depth.max(f.tree.depth[b_node as usize] as i32);
        }
    }

    base_score + max_depth * 100
}

pub fn compute_e_f(forest: &XForest, a: u32, b: u32) -> Vec<NodeId> {
    let a_node = forest.tree.label_to_node[a as usize];
    let b_node = forest.tree.label_to_node[b as usize];

    if forest.live_leafsets[a_node as usize].count_ones(..) == 0
        || forest.live_leafsets[b_node as usize].count_ones(..) == 0
    {
        return Vec::new();
    }

    let lca = forest_lca(forest, a_node, b_node);
    if lca == NONE {
        return Vec::new();
    }

    let n = forest.tree.num_nodes();
    let mut on_path = vec![false; n];
    let mut path_nodes_buf = Vec::with_capacity(32);

    on_path[a_node as usize] = true;
    on_path[b_node as usize] = true;
    on_path[lca as usize] = true;
    path_nodes_buf.push(lca);

    let mut cur = a_node;
    while cur != lca {
        if forest.is_cut(cur) {
            break;
        }
        let p = forest.tree.parent[cur as usize];
        if p == NONE {
            break;
        }
        if !on_path[p as usize] {
            on_path[p as usize] = true;
            path_nodes_buf.push(p);
        }
        cur = p;
    }
    cur = b_node;
    while cur != lca {
        if forest.is_cut(cur) {
            break;
        }
        let p = forest.tree.parent[cur as usize];
        if p == NONE {
            break;
        }
        if !on_path[p as usize] {
            on_path[p as usize] = true;
            path_nodes_buf.push(p);
        }
        cur = p;
    }

    let mut e_f = Vec::new();
    for &path_node in &path_nodes_buf {
        if let Some((left, right)) = forest.tree.children(path_node) {
            if left != NONE
                && !forest.is_cut(left)
                && forest.live_leafsets[left as usize].count_ones(..) > 0
                && !on_path[left as usize]
            {
                e_f.push(left);
            }
            if right != NONE
                && !forest.is_cut(right)
                && forest.live_leafsets[right as usize].count_ones(..) > 0
                && !on_path[right as usize]
            {
                e_f.push(right);
            }
        }
    }
    e_f
}

fn lca_leafset(forest: &XForest, a: u32, b: u32) -> FixedBitSet {
    let a_node = forest.tree.label_to_node[a as usize];
    let b_node = forest.tree.label_to_node[b as usize];
    let lca = forest_lca(forest, a_node, b_node);
    if lca == NONE {
        FixedBitSet::new()
    } else {
        forest.live_leafsets[lca as usize].clone()
    }
}

struct CutInfo {
    forest_idx: usize,
    node: NodeId,
}

/// CUT_ONE_B: For m=2, check if a 3-way BR-2.1 branch can be reduced to a single cut.
///
/// Given sibling pair (a, b) that are siblings in forest q, check in the other forest r:
/// - If grandparent of a == parent of b in r, we only need to cut the sibling of a in r.
/// - Symmetric: if grandparent of b == parent of a in r, cut the sibling of b in r.
fn check_cut_one_b(state: &SearchState, a: u32, b: u32) -> Option<CutInfo> {
    for q in 0..state.forests.len() {
        let fq = &state.forests[q];
        let a_node_q = fq.tree.label_to_node[a as usize];
        let b_node_q = fq.tree.label_to_node[b as usize];

        // Check if a_node_q and b_node_q are live
        if fq.live_leafsets[a_node_q as usize].count_ones(..) == 0
            || fq.live_leafsets[b_node_q as usize].count_ones(..) == 0
        {
            continue;
        }

        let parent_a_q = fq.tree.parent[a_node_q as usize];
        if parent_a_q == NONE {
            continue;
        }

        // Check if a and b are siblings in forest q (same parent)
        let sib_of_a_q = if fq.tree.left[parent_a_q as usize] == a_node_q {
            fq.tree.right[parent_a_q as usize]
        } else {
            fq.tree.left[parent_a_q as usize]
        };
        if sib_of_a_q != b_node_q {
            continue;
        }

        // They're siblings in forest q. Check CUT_ONE_B against other forests.
        for r in 0..state.forests.len() {
            if r == q {
                continue;
            }
            let fr = &state.forests[r];
            let a_node_r = fr.tree.label_to_node[a as usize];
            let b_node_r = fr.tree.label_to_node[b as usize];

            // Check both nodes are live in forest r
            if fr.live_leafsets[a_node_r as usize].count_ones(..) == 0
                || fr.live_leafsets[b_node_r as usize].count_ones(..) == 0
            {
                continue;
            }

            let parent_a_r = fr.tree.parent[a_node_r as usize];
            let parent_b_r = fr.tree.parent[b_node_r as usize];
            if parent_a_r == NONE || parent_b_r == NONE {
                continue;
            }

            // Check: grandparent_a == parent_b in forest r
            let grandparent_a_r = fr.tree.parent[parent_a_r as usize];
            if grandparent_a_r != NONE && grandparent_a_r == parent_b_r {
                // CUT_ONE_B: cut the sibling of a in forest r
                let sib_a_r = if fr.tree.left[parent_a_r as usize] == a_node_r {
                    fr.tree.right[parent_a_r as usize]
                } else {
                    fr.tree.left[parent_a_r as usize]
                };
                if sib_a_r != NONE && !fr.is_cut(sib_a_r) {
                    super::trace!(
                        "CUT_ONE_B: sibling pair ({},{}) in forest {}, cut sib_a={} in forest {}",
                        a, b, q, sib_a_r, r
                    );
                    return Some(CutInfo {
                        forest_idx: r,
                        node: sib_a_r,
                    });
                }
            }

            // Symmetric: grandparent_b == parent_a in forest r
            let grandparent_b_r = fr.tree.parent[parent_b_r as usize];
            if grandparent_b_r != NONE && grandparent_b_r == parent_a_r {
                // CUT_ONE_B: cut the sibling of b in forest r
                let sib_b_r = if fr.tree.left[parent_b_r as usize] == b_node_r {
                    fr.tree.right[parent_b_r as usize]
                } else {
                    fr.tree.left[parent_b_r as usize]
                };
                if sib_b_r != NONE && !fr.is_cut(sib_b_r) {
                    super::trace!(
                        "CUT_ONE_B: sibling pair ({},{}) in forest {}, cut sib_b={} in forest {}",
                        a, b, q, sib_b_r, r
                    );
                    return Some(CutInfo {
                        forest_idx: r,
                        node: sib_b_r,
                    });
                }
            }

            // REVERSE_CUT_ONE_B: check the sibling of (a,b)'s parent in forest q.
            // If that sibling is a leaf `s`, check s's twin in forest r.
            let grandparent_q = fq.tree.parent[parent_a_q as usize];
            if grandparent_q != NONE {
                let uncle_q = if fq.tree.left[grandparent_q as usize] == parent_a_q {
                    fq.tree.right[grandparent_q as usize]
                } else {
                    fq.tree.left[grandparent_q as usize]
                };

                if uncle_q != NONE && fq.tree.is_leaf(uncle_q) {
                    let s_label = fq.tree.label[uncle_q as usize];
                    if s_label != 0 {
                        let s_node_r = fr.tree.label_to_node[s_label as usize];
                        if fr.live_leafsets[s_node_r as usize].count_ones(..) > 0 {
                            let s_parent_r = fr.tree.parent[s_node_r as usize];
                            if s_parent_r != NONE {
                                // If s and a share a parent in forest r → must cut b
                                if s_parent_r == parent_a_r {
                                    let b_node_r2 = fr.tree.label_to_node[b as usize];
                                    if !fr.is_cut(b_node_r2) {
                                        super::trace!(
                                            "REVERSE_CUT_ONE_B: ({},{}) forest {}, s={} shares parent with a in forest {} → cut b={}",
                                            a, b, q, s_label, r, b_node_r2
                                        );
                                        return Some(CutInfo {
                                            forest_idx: r,
                                            node: b_node_r2,
                                        });
                                    }
                                }
                                // If s and b share a parent in forest r → must cut a
                                if s_parent_r == parent_b_r {
                                    let a_node_r2 = fr.tree.label_to_node[a as usize];
                                    if !fr.is_cut(a_node_r2) {
                                        super::trace!(
                                            "REVERSE_CUT_ONE_B: ({},{}) forest {}, s={} shares parent with b in forest {} → cut a={}",
                                            a, b, q, s_label, r, a_node_r2
                                        );
                                        return Some(CutInfo {
                                            forest_idx: r,
                                            node: a_node_r2,
                                        });
                                    }
                                }
                            }

                            // CUT_TWO_B: check if uncle relationship forces a single cut.
                            // In forest r: if grandparent of a == grandparent of b (= l),
                            // and s's twin in r is the sibling of l, then cut sibling of a in r.
                            let grandparent_a_r2 = fr.tree.parent[parent_a_r as usize];
                            if grandparent_a_r2 != NONE {
                                let grandparent_b_r2 = fr.tree.parent[parent_b_r as usize];
                                if grandparent_b_r2 == grandparent_a_r2 {
                                    let l_r = grandparent_a_r2;
                                    let l_parent_r = fr.tree.parent[l_r as usize];
                                    if l_parent_r != NONE {
                                        let l_sibling_r = if fr.tree.left[l_parent_r as usize] == l_r {
                                            fr.tree.right[l_parent_r as usize]
                                        } else {
                                            fr.tree.left[l_parent_r as usize]
                                        };
                                        if l_sibling_r != NONE && l_sibling_r == s_node_r {
                                            // Cut sibling of a in r
                                            let sib_a_r = if fr.tree.left[parent_a_r as usize] == a_node_r {
                                                fr.tree.right[parent_a_r as usize]
                                            } else {
                                                fr.tree.left[parent_a_r as usize]
                                            };
                                            if sib_a_r != NONE && !fr.is_cut(sib_a_r) {
                                                super::trace!(
                                                    "CUT_TWO_B: ({},{}) forest {}, s={} is sibling of l in forest {} → cut sib_a={}",
                                                    a, b, q, s_label, r, sib_a_r
                                                );
                                                return Some(CutInfo {
                                                    forest_idx: r,
                                                    node: sib_a_r,
                                                });
                                            }
                                        }
                                    }
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

pub fn apply_case_2_branching(
    state: &mut SearchState,
    target_s: usize,
    a: u32,
    b: u32,
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    zobrist: &ZobristTable,
    tt: &mut FxHashMap<u64, TTEntry>,
) -> Option<Vec<klados_core::Tree>> {
    // CUT_ONE_B: for m=2, check if we can reduce 3-way branching to a single cut.
    if state.forests.len() == 2 {
        if let Some(cut_info) = check_cut_one_b(state, a, b) {
            state.checkpoint();
            state.cut_node(cut_info.forest_idx, cut_info.node);
            if let Some(result) = super::algorithm::alg_maf(
                state,
                target_s,
                label_space,
                num_leaves,
                stats,
                zobrist,
                tt,
            ) {
                state.rollback();
                return Some(result);
            }
            state.rollback();
            return None;
        }
    }

    let e_sets: Vec<Vec<NodeId>> = state.forests.iter().map(|f| compute_e_f(f, a, b)).collect();
    let max_e = e_sets.iter().map(|e| e.len()).max().unwrap_or(0);

    if max_e >= 2 {
        // BB: Approximation-based pruning before 3-way BR-2.1 branch (m=2 only).
        // Use red-blue 2-approximation on pruned trees from current live leaves.
        // Only worth the overhead for larger instances.
        if state.forests.len() == 2 {
            let live = &state.forests[0].live_leafsets[state.forests[0].tree.root as usize];
            let live_count = live.count_ones(..);
            if live_count >= 15 {
                let t1_pruned = state.forests[0].tree.prune_to_leafset(live);
                let t2_pruned = state.forests[1].tree.prune_to_leafset(live);
                let approx = klados_core::lower_bound::red_blue_approx(&t1_pruned, &t2_pruned);
                // approx is an upper bound on rSPR distance. MAF components = distance + 1.
                // But this is a 2-approx: OPT >= ceil(approx / 2), so LB on components = ceil(approx/2) + 1.
                let approx_lb_comps = approx.div_ceil(2) + 1;
                if approx_lb_comps > target_s {
                    stats.branches_pruned += 1;
                    super::trace!(
                        "BB prune: approx={}, lb_comps={}, target_s={}",
                        approx, approx_lb_comps, target_s
                    );
                    return None;
                }
            }
        }

        return apply_branching_rule_2_1(
            state,
            target_s,
            a,
            b,
            &e_sets,
            label_space,
            num_leaves,
            stats,
            zobrist,
            tt,
        );
    }

    let omega1: Vec<usize> = e_sets
        .iter()
        .enumerate()
        .filter(|(_, e)| e.len() == 1)
        .map(|(i, _)| i)
        .collect();

    if omega1.is_empty() {
        return None;
    }

    let lca_sets: Vec<FixedBitSet> = omega1
        .iter()
        .map(|&i| lca_leafset(&state.forests[i], a, b))
        .collect();
    let all_same_lca = lca_sets.windows(2).all(|w| w[0] == w[1]);

    if all_same_lca {
        super::trace!("Case 2.2.1: reduction for ({}, {})", a, b);
        return apply_reduction_rule_2_2_1(
            state,
            target_s,
            &e_sets,
            label_space,
            num_leaves,
            stats,
            zobrist,
            tt,
        );
    }

    super::trace!("Case 2.2.2: branching for ({}, {})", a, b);
    apply_branching_rule_2_2_2(
        state,
        target_s,
        a,
        b,
        &e_sets,
        label_space,
        num_leaves,
        stats,
        zobrist,
        tt,
    )
}

fn apply_branching_rule_2_1(
    state: &mut SearchState,
    target_s: usize,
    a: u32,
    b: u32,
    e_sets: &[Vec<NodeId>],
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    zobrist: &ZobristTable,
    tt: &mut FxHashMap<u64, TTEntry>,
) -> Option<Vec<klados_core::Tree>> {
    super::trace!("BR 2.1: a={}, b={}", a, b);

    {
        state.checkpoint();
        for idx in 0..state.forests.len() {
            let a_node = state.forests[idx].tree.label_to_node[a as usize];
            state.cut_node(idx, a_node);
        }
        if let Some(result) =
            super::algorithm::alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    {
        state.checkpoint();
        for idx in 0..state.forests.len() {
            let b_node = state.forests[idx].tree.label_to_node[b as usize];
            state.cut_node(idx, b_node);
        }
        if let Some(result) =
            super::algorithm::alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    {
        state.checkpoint();
        let mut any_cut = false;
        for (i, e_nodes) in e_sets.iter().enumerate() {
            for &node in e_nodes {
                if node != state.forests[i].tree.root && !state.forests[i].is_cut(node) {
                    state.cut_node(i, node);
                    any_cut = true;
                }
            }
        }
        if any_cut
            && let Some(result) = super::algorithm::alg_maf(
                state,
                target_s,
                label_space,
                num_leaves,
                stats,
                zobrist,
                tt,
            )
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    None
}

fn apply_reduction_rule_2_2_1(
    state: &mut SearchState,
    target_s: usize,
    e_sets: &[Vec<NodeId>],
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    zobrist: &ZobristTable,
    tt: &mut FxHashMap<u64, TTEntry>,
) -> Option<Vec<klados_core::Tree>> {
    for (i, e_nodes) in e_sets.iter().enumerate() {
        for &node in e_nodes {
            state.cut_node(i, node);
        }
    }
    super::algorithm::alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
}

fn apply_branching_rule_2_2_2(
    state: &mut SearchState,
    target_s: usize,
    a: u32,
    b: u32,
    e_sets: &[Vec<NodeId>],
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    zobrist: &ZobristTable,
    tt: &mut FxHashMap<u64, TTEntry>,
) -> Option<Vec<klados_core::Tree>> {
    super::trace!("BR 2.2.2: a={}, b={}", a, b);

    {
        state.checkpoint();
        for idx in 0..state.forests.len() {
            let a_node = state.forests[idx].tree.label_to_node[a as usize];
            state.cut_node(idx, a_node);
        }
        if let Some(result) =
            super::algorithm::alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    {
        state.checkpoint();
        for idx in 0..state.forests.len() {
            let b_node = state.forests[idx].tree.label_to_node[b as usize];
            state.cut_node(idx, b_node);
        }
        if let Some(result) =
            super::algorithm::alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    {
        state.checkpoint();
        let mut any_cut = false;
        for (i, e_nodes) in e_sets.iter().enumerate() {
            for &node in e_nodes {
                if node != state.forests[i].tree.root && !state.forests[i].is_cut(node) {
                    state.cut_node(i, node);
                    any_cut = true;
                }
            }
        }
        if any_cut
            && let Some(result) = super::algorithm::alg_maf(
                state,
                target_s,
                label_space,
                num_leaves,
                stats,
                zobrist,
                tt,
            )
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    None
}
