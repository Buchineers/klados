//! Reduction rules: R1 (LSI-based cuts) and R2 (sibling pair collapse).

use fixedbitset::FixedBitSet;
use klados_core::{NodeId, XForest, NONE};

use super::forest_nav::{
    component_leaf_sets_xf, forest_children, forest_is_leaf,
    forest_parent_leaf, forest_resolves_to,
};
use super::search_state::SearchState;

pub fn apply_reduction_rules_state(
    state: &mut SearchState,
    label_space: usize,
) -> Vec<Vec<FixedBitSet>> {
    let mut scratch = FixedBitSet::with_capacity(label_space + 1);
    let nf = state.forests.len();

    let mut comp_sets: Vec<Vec<FixedBitSet>> = state
        .forests
        .iter()
        .map(|f| component_leaf_sets_xf(f, label_space))
        .collect();

    loop {
        let mut changed = false;
        'outer: for i in 0..nf {
            for j in 0..nf {
                if i == j {
                    continue;
                }
                if let Some(node) =
                    find_r1_cut(&state.forests[i], &comp_sets[j], label_space, &mut scratch)
                {
                    super::trace!("R1: cut node {} in forest {}", node, i);
                    state.cut_node(i, node);
                    comp_sets[i] = component_leaf_sets_xf(&state.forests[i], label_space);
                    changed = true;
                    break 'outer;
                }
            }
        }
        if !changed {
            return comp_sets;
        }
    }
}

pub fn find_r1_cut(
    forest_i: &XForest,
    fj_components: &[FixedBitSet],
    label_space: usize,
    scratch: &mut FixedBitSet,
) -> Option<NodeId> {
    for node in forest_i.tree.pre_order() {
        if forest_i.is_cut(node) || node == forest_i.tree.root {
            continue;
        }
        if forest_i.live_leafsets[node as usize].count_ones(..) == 0 {
            continue;
        }
        let parent = forest_i.tree.parent[node as usize];
        if parent == NONE {
            continue;
        }

        let node_ls = &forest_i.live_leafsets[node as usize];
        let node_count = node_ls.count_ones(..);
        let comp_root = forest_i.component_root(node);
        let comp_ls = &forest_i.live_leafsets[comp_root as usize];
        let comp_count = comp_ls.count_ones(..);

        if node_count == 0 || node_count >= comp_count {
            continue;
        }

        scratch.clear();
        scratch.grow(label_space + 1);
        let mut union_count = 0usize;

        for fj_comp in fj_components {
            let fj_sl = fj_comp.as_slice();
            let comp_sl = comp_ls.as_slice();
            let node_sl = node_ls.as_slice();
            let len = fj_sl.len().min(comp_sl.len());

            let mut has_inter = false;
            let mut subset_of_node = true;
            for k in 0..len {
                let inter_word = fj_sl[k] & comp_sl[k];
                if inter_word != 0 {
                    has_inter = true;
                    let node_word = if k < node_sl.len() { node_sl[k] } else { 0 };
                    if inter_word & !node_word != 0 {
                        subset_of_node = false;
                        break;
                    }
                }
            }

            if has_inter && subset_of_node {
                let scratch_sl = scratch.as_mut_slice();
                for k in 0..len {
                    let inter_word = fj_sl[k] & comp_sl[k];
                    if k < scratch_sl.len() {
                        let old = scratch_sl[k];
                        scratch_sl[k] |= inter_word;
                        union_count += (scratch_sl[k].count_ones() - old.count_ones()) as usize;
                    }
                }
                if union_count > node_count {
                    break;
                }
            }
        }

        if union_count == node_count && union_count > 0 {
            if scratch.as_slice() == node_ls.as_slice()
                || (scratch.as_slice().len() >= node_ls.as_slice().len()
                    && scratch.as_slice()[..node_ls.as_slice().len()] == *node_ls.as_slice()
                    && scratch.as_slice()[node_ls.as_slice().len()..]
                        .iter()
                        .all(|&w| w == 0))
            {
                return Some(node);
            }
        }
    }
    None
}

pub fn find_common_sibling_pair(forests: &[XForest], label_space: usize) -> Option<(u32, u32)> {
    if forests.is_empty() {
        return None;
    }
    let pairs = find_all_sibling_pairs(&forests[0], label_space);
    'outer: for (a, b) in &pairs {
        for forest in &forests[1..] {
            if !is_sibling_pair_in_forest(forest, *a, *b) {
                continue 'outer;
            }
        }
        return Some((*a, *b));
    }
    None
}

pub fn find_all_sibling_pairs(forest: &XForest, _label_space: usize) -> Vec<(u32, u32)> {
    let mut pairs = Vec::new();
    for node in forest.tree.pre_order() {
        if forest.live_leafsets[node as usize].count_ones(..) == 0 {
            continue;
        }
        if forest.tree.is_leaf(node) {
            continue;
        }
        let children = forest_children(forest, node);
        if children.len() == 2 {
            let c1 = children[0];
            let c2 = children[1];
            let c1_leaf = forest_is_leaf(forest, c1);
            let c2_leaf = forest_is_leaf(forest, c2);
            if c1_leaf && c2_leaf {
                let lbl1 = forest.tree.leaf_label(c1);
                let lbl2 = forest.tree.leaf_label(c2);
                if let (Some(l1), Some(l2)) = (lbl1, lbl2) {
                    pairs.push((l1.min(l2), l1.max(l2)));
                }
            }
        }
    }
    pairs
}

pub fn is_sibling_pair_in_forest(forest: &XForest, a: u32, b: u32) -> bool {
    let a_node = forest.tree.label_to_node[a as usize];
    let b_node = forest.tree.label_to_node[b as usize];
    if forest.live_leafsets[a_node as usize].count_ones(..) == 0
        || forest.live_leafsets[b_node as usize].count_ones(..) == 0
    {
        return false;
    }
    let pa = forest_parent_leaf(forest, a_node);
    let pb = forest_parent_leaf(forest, b_node);
    if pa == NONE || pa != pb {
        return false;
    }
    let children = forest_children(forest, pa);
    if children.len() != 2 {
        return false;
    }
    let c1_is_a = forest_resolves_to(forest, children[0], a_node);
    let c2_is_b = forest_resolves_to(forest, children[1], b_node);
    let c1_is_b = forest_resolves_to(forest, children[0], b_node);
    let c2_is_a = forest_resolves_to(forest, children[1], a_node);
    (c1_is_a && c2_is_b) || (c1_is_b && c2_is_a)
}

pub fn all_pairs_lsi_cached(comp_sets: &[Vec<FixedBitSet>]) -> bool {
    if comp_sets.len() <= 1 {
        return true;
    }
    let ref_sorted = super::utils::sorted_partition_hashes(&comp_sets[0]);
    for cs in &comp_sets[1..] {
        if super::utils::sorted_partition_hashes(cs) != ref_sorted {
            return false;
        }
    }
    true
}

pub fn lsi_pair(a: &[FixedBitSet], b: &[FixedBitSet]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    super::utils::sorted_partition_hashes(a) == super::utils::sorted_partition_hashes(b)
}

pub fn find_violating_pair_cached(comp_sets: &[Vec<FixedBitSet>]) -> Option<(usize, usize)> {
    for i in 0..comp_sets.len() {
        for j in (i + 1)..comp_sets.len() {
            if !lsi_pair(&comp_sets[i], &comp_sets[j]) {
                return Some((i, j));
            }
        }
    }
    None
}
