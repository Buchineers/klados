//! Branching rules: BR-LSI step and Case 2 sibling-pair branching.

use fixedbitset::FixedBitSet;
use fxhash::FxHashMap;
use klados_core::{NodeId, XForest, NONE};

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
    let (i, j) = match find_violating_pair_cached(comp_sets) {
        Some(pair) => pair,
        None => return None,
    };

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

    if !omega1.is_empty() {
        let lca_sets: Vec<FixedBitSet> = omega1
            .iter()
            .map(|&i| lca_leafset(&forests[i], a, b))
            .collect();
        let all_same_lca = lca_sets.windows(2).all(|w| w[0] == w[1]);
        if all_same_lca {
            return 10000;
        }
    }

    if max_e >= 3 {
        return 5000 + max_e as i32;
    }
    if max_e >= 2 {
        return 3000;
    }

    if !omega1.is_empty() {
        return 0;
    }

    -1000
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
    let e_sets: Vec<Vec<NodeId>> = state.forests.iter().map(|f| compute_e_f(f, a, b)).collect();
    let max_e = e_sets.iter().map(|e| e.len()).max().unwrap_or(0);

    if max_e >= 2 {
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
        if any_cut {
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
        if any_cut {
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
        }
        state.rollback();
    }

    None
}
