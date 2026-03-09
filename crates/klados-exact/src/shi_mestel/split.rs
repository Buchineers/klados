//! Split-or-decompose: Overlapping component detection and splitting core computation.

use fixedbitset::FixedBitSet;
use klados_core::{NONE, NodeId, Tree, XForest};

use super::search_state::SearchState;
use super::transposition::{TTEntry, ZobristTable};
use super::utils::{count_intersection, is_subset};
use crate::SolverStats;

pub fn apply_split_branching_cached(
    state: &mut SearchState,
    target_s: usize,
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    comps: &[FixedBitSet],
    zobrist: &ZobristTable,
    tt: &mut fxhash::FxHashMap<u64, TTEntry>,
    profile_enabled: bool,
    split_stats: &mut SplitStats,
) -> (bool, Option<Vec<Tree>>) {
    let start = if profile_enabled {
        Some(std::time::Instant::now())
    } else {
        None
    };
    if profile_enabled {
        split_stats.attempts += 1;
    }
    if comps.len() <= 1 {
        return (false, None);
    }

    if let Some((forest_idx, comp_a, comp_b, edge_child)) =
        find_best_overlap(&state.forests, comps, profile_enabled, split_stats)
    {
        if profile_enabled {
            split_stats.triggered += 1;
        }
        super::trace!(
            "SPLIT: forest={}, comp_a={}, comp_b={}, edge_child={}",
            forest_idx,
            comp_a,
            comp_b,
            edge_child
        );

        if let Some(result) = split_component_branch(
            state,
            target_s,
            label_space,
            num_leaves,
            stats,
            forest_idx,
            &comps[comp_a],
            edge_child,
            zobrist,
            tt,
        ) {
            if let Some(t0) = start {
                let dt = t0.elapsed().as_nanos();
                split_stats.split_nanos += dt;
            }
            return (true, Some(result));
        }

        if let Some(result) = split_component_branch(
            state,
            target_s,
            label_space,
            num_leaves,
            stats,
            forest_idx,
            &comps[comp_b],
            edge_child,
            zobrist,
            tt,
        ) {
            if let Some(t0) = start {
                let dt = t0.elapsed().as_nanos();
                split_stats.split_nanos += dt;
            }
            return (true, Some(result));
        }

        stats.branches_pruned += 1;
        if let Some(t0) = start {
            let dt = t0.elapsed().as_nanos();
            split_stats.split_nanos += dt;
        }
        return (true, None);
    }

    if let Some(t0) = start {
        let dt = t0.elapsed().as_nanos();
        split_stats.split_nanos += dt;
    }
    (false, None)
}

pub fn split_component_branch(
    state: &mut SearchState,
    target_s: usize,
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    forest_idx: usize,
    comp: &FixedBitSet,
    edge_child: NodeId,
    zobrist: &ZobristTable,
    tt: &mut fxhash::FxHashMap<u64, TTEntry>,
) -> Option<Vec<Tree>> {
    let tree = &state.forests[forest_idx].tree;
    let full_leafsets = &state.forests[forest_idx].full_leafsets;

    let mut y = full_leafsets[edge_child as usize].clone();
    y.intersect_with(comp);
    if y.count_ones(..) == 0 {
        return None;
    }
    let mut z = comp.clone();
    z.difference_with(&y);
    if z.count_ones(..) == 0 {
        return None;
    }

    let core = splitting_core(tree, full_leafsets, comp, &y, &z);
    if core.is_empty() {
        return None;
    }

    for cut in core {
        state.checkpoint();
        for child in cut {
            state.cut_node(forest_idx, child);
        }
        if let Some(result) =
            super::algorithm::alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    None
}

pub fn find_best_overlap(
    forests: &[XForest],
    comps: &[FixedBitSet],
    profile_enabled: bool,
    split_stats: &mut SplitStats,
) -> Option<(usize, usize, usize, NodeId)> {
    if profile_enabled {
        split_stats.trees_scanned += forests.len() as u64;
    }

    let mut best: Option<(usize, usize, usize, NodeId)> = None;
    let mut best_score = usize::MAX;

    for (forest_idx, forest) in forests.iter().enumerate() {
        if let Some((a, b, edge_child)) = find_overlap_in_tree(forest, comps) {
            let score = overlap_score(forest, comps, a, b, edge_child);
            if score < best_score {
                best_score = score;
                best = Some((forest_idx, a, b, edge_child));
                if best_score <= 1 {
                    break;
                }
            }
        }
    }
    best
}

pub fn overlap_score(
    forest: &XForest,
    comps: &[FixedBitSet],
    comp_a: usize,
    comp_b: usize,
    edge_child: NodeId,
) -> usize {
    let tree = &forest.tree;
    let full_leafsets = &forest.full_leafsets;

    let mut best = usize::MAX;
    for &comp_idx in &[comp_a, comp_b] {
        let comp = &comps[comp_idx];
        let mut y = full_leafsets[edge_child as usize].clone();
        y.intersect_with(comp);
        if y.count_ones(..) == 0 {
            continue;
        }
        let mut z = comp.clone();
        z.difference_with(&y);
        if z.count_ones(..) == 0 {
            continue;
        }
        let core = splitting_core(tree, full_leafsets, comp, &y, &z);
        if !core.is_empty() && core.len() < best {
            best = core.len();
        }
    }
    best
}

pub fn find_overlap_in_tree(
    forest: &XForest,
    comps: &[FixedBitSet],
) -> Option<(usize, usize, NodeId)> {
    let num_nodes = forest.tree.num_nodes();
    let mut edge_owner: Vec<Option<usize>> = vec![None; num_nodes];
    let comp_sizes: Vec<usize> = comps.iter().map(|c| c.count_ones(..)).collect();

    for child in forest.tree.pre_order() {
        if child == forest.tree.root {
            continue;
        }
        if comp_sizes.iter().all(|&s| s <= 1) {
            break;
        }
        let child_ls = &forest.full_leafsets[child as usize];
        for (idx, comp) in comps.iter().enumerate() {
            if comp_sizes[idx] <= 1 {
                continue;
            }
            let inter = count_intersection(comp, child_ls);
            if inter == 0 || inter == comp_sizes[idx] {
                continue;
            }
            if let Some(other) = edge_owner[child as usize] {
                if other != idx {
                    return Some((other, idx, child));
                }
            } else {
                edge_owner[child as usize] = Some(idx);
            }
        }
    }
    None
}

pub fn splitting_core(
    tree: &Tree,
    full_leafsets: &[FixedBitSet],
    x: &FixedBitSet,
    y: &FixedBitSet,
    z: &FixedBitSet,
) -> Vec<Vec<NodeId>> {
    for child in tree.pre_order() {
        if child == tree.root {
            continue;
        }
        let mut side = full_leafsets[child as usize].clone();
        side.intersect_with(x);
        if side.count_ones(..) == 0 || side == *x {
            continue;
        }
        if side == *y || side == *z {
            return vec![vec![child]];
        }
    }

    for v in tree.pre_order() {
        let mut pure_y_edge: Option<NodeId> = None;
        let mut pure_y_side: Option<FixedBitSet> = None;
        let mut pure_z_edge: Option<NodeId> = None;
        let mut pure_z_side: Option<FixedBitSet> = None;
        let mut has_mixed = false;

        for neighbor in neighbors(tree, v) {
            let side = side_leafset(tree, full_leafsets, x, v, neighbor);
            if side.count_ones(..) == 0 {
                continue;
            }
            if is_subset(&side, y) {
                if pure_y_edge.is_none() {
                    pure_y_edge = Some(edge_child(tree, v, neighbor));
                    pure_y_side = Some(side);
                }
            } else if is_subset(&side, z) {
                if pure_z_edge.is_none() {
                    pure_z_edge = Some(edge_child(tree, v, neighbor));
                    pure_z_side = Some(side);
                }
            } else {
                has_mixed = true;
            }
        }

        if let (Some(e1), Some(e2), Some(side_y), Some(side_z)) =
            (pure_y_edge, pure_z_edge, pure_y_side, pure_z_side)
            && has_mixed
        {
            let mut x1 = x.clone();
            x1.difference_with(&side_y);
            let mut y1 = y.clone();
            y1.difference_with(&side_y);
            let z1 = z.clone();

            let mut x2 = x.clone();
            x2.difference_with(&side_z);
            let y2 = y.clone();
            let mut z2 = z.clone();
            z2.difference_with(&side_z);

            let mut out = Vec::new();
            for mut k in splitting_core(tree, full_leafsets, &x1, &y1, &z1) {
                k.push(e1);
                out.push(k);
            }
            for mut k in splitting_core(tree, full_leafsets, &x2, &y2, &z2) {
                k.push(e2);
                out.push(k);
            }
            return out;
        }
    }

    Vec::new()
}

fn neighbors(tree: &Tree, node: NodeId) -> Vec<NodeId> {
    let mut out = Vec::with_capacity(3);
    let parent = tree.parent[node as usize];
    if parent != NONE {
        out.push(parent);
    }
    if let Some((l, r)) = tree.children(node) {
        if l != NONE {
            out.push(l);
        }
        if r != NONE {
            out.push(r);
        }
    }
    out
}

fn edge_child(tree: &Tree, from: NodeId, to: NodeId) -> NodeId {
    if tree.parent[to as usize] == from {
        to
    } else if tree.parent[from as usize] == to {
        from
    } else {
        to
    }
}

fn side_leafset(
    tree: &Tree,
    full_leafsets: &[FixedBitSet],
    x: &FixedBitSet,
    node: NodeId,
    neighbor: NodeId,
) -> FixedBitSet {
    if tree.parent[neighbor as usize] == node {
        let mut side = full_leafsets[neighbor as usize].clone();
        side.intersect_with(x);
        side
    } else if tree.parent[node as usize] == neighbor {
        let mut side = x.clone();
        side.difference_with(&full_leafsets[node as usize]);
        side
    } else {
        FixedBitSet::new()
    }
}

#[derive(Default)]
pub struct SplitStats {
    pub attempts: u64,
    pub triggered: u64,
    pub trees_scanned: u64,
    pub overlap_checks: u64,
    pub core_calls: u64,
    pub core_branches: u64,
    pub split_nanos: u128,
}
