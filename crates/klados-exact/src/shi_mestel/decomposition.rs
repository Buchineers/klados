//! Component decomposition for independent sub-problem solving.

use fixedbitset::FixedBitSet;
use fxhash::FxHashMap;
use klados_core::{Tree, XForest};

use super::extraction::{build_collapsed_into, build_component_tree, expand_leafset};
use super::search_state::SearchState;
use super::transposition::{TTEntry, ZobristTable};
use super::utils::hash_bitset;
use crate::SolverStats;

pub fn solve_decomposed(
    forests: &[XForest],
    target_s: usize,
    collapses: &super::search_state::Collapses,
    label_space: usize,
    num_leaves: u32,
    non_iso_comps: &[FixedBitSet],
    all_comps: &[FixedBitSet],
    stats: &mut SolverStats,
) -> Option<Vec<Tree>> {
    let cur_order = all_comps.len();
    let remaining = target_s.saturating_sub(cur_order);

    super::trace!(
        "Decomposing: {} components, {} non-isomorphic, remaining={}",
        all_comps.len(),
        non_iso_comps.len(),
        remaining
    );

    let collapsed_into = build_collapsed_into(collapses, num_leaves);
    let nc = non_iso_comps.len();

    let mut sub_forest_cache: Vec<Vec<XForest>> = Vec::with_capacity(nc);
    let mut shallow_results: Vec<Option<Vec<Tree>>> = Vec::with_capacity(nc);
    let mut lower_bounds: Vec<usize> = Vec::with_capacity(nc);
    let mut total_lower_bound: usize = 0;
    let mut comp_zobrist: Vec<ZobristTable> = Vec::with_capacity(nc);
    let mut comp_tt: Vec<FxHashMap<u64, TTEntry>> = Vec::with_capacity(nc);

    for comp_ls in non_iso_comps {
        let sub_trees: Vec<Tree> = forests
            .iter()
            .map(|f| f.tree.prune_to_leafset(comp_ls))
            .collect();
        let sub_forests: Vec<XForest> = sub_trees
            .iter()
            .map(|t| XForest::from_tree(t.clone()))
            .collect();

        let zobrist = ZobristTable::new(label_space);
        let mut tt: FxHashMap<u64, TTEntry> = FxHashMap::default();

        let approx_lb = crate::lower_bound::maf_bounds(&sub_trees, num_leaves).lower;

        if approx_lb <= 2 {
            let mut sub_state = SearchState::new(sub_forests.clone());
            if let Some(result) = super::algorithm::alg_maf(
                &mut sub_state,
                2,
                label_space,
                num_leaves,
                stats,
                &zobrist,
                &mut tt,
            ) {
                shallow_results.push(Some(result));
                lower_bounds.push(1);
                total_lower_bound += 1;
            } else {
                shallow_results.push(None);
                lower_bounds.push(2);
                total_lower_bound += 2;
            }
        } else {
            shallow_results.push(None);
            lower_bounds.push(approx_lb - 1);
            total_lower_bound += approx_lb - 1;
        }
        sub_forest_cache.push(sub_forests);
        comp_zobrist.push(zobrist);
        comp_tt.push(tt);
    }

    if total_lower_bound > target_s {
        stats.branches_pruned += 1;
        return None;
    }

    let mut total_cost: usize = 0;
    let mut component_results: Vec<Vec<Tree>> = Vec::with_capacity(nc);

    for (idx, _comp_ls) in non_iso_comps.iter().enumerate() {
        if let Some(ref result) = shallow_results[idx] {
            total_cost += 1;
            let mut trees = Vec::new();
            for sub_tree in result {
                let mut sub_ls = FixedBitSet::with_capacity(label_space + 1);
                sub_ls.grow(label_space + 1);
                for node in sub_tree.pre_order() {
                    if sub_tree.is_leaf(node) {
                        let lbl = sub_tree.label[node as usize];
                        if lbl > 0 {
                            sub_ls.insert(lbl as usize);
                        }
                    }
                }
                let expanded = expand_leafset(&sub_ls, &collapsed_into, num_leaves, label_space);
                trees.push(build_component_tree(
                    &expanded,
                    &forests[0].tree,
                    num_leaves,
                ));
            }
            component_results.push(trees);
        } else {
            let other_lb: usize = lower_bounds
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != idx)
                .map(|(_, &lb)| lb)
                .sum();
            let budget = remaining.saturating_sub(other_lb);
            let comp_num_labels = non_iso_comps[idx].count_ones(..);

            let mut found = false;
            let start_cost = lower_bounds[idx].max(2);
            for cost in start_cost..=budget.min(comp_num_labels) {
                let mut sub_state = SearchState::new(sub_forest_cache[idx].clone());
                if let Some(result) = super::algorithm::alg_maf(
                    &mut sub_state,
                    1 + cost,
                    label_space,
                    num_leaves,
                    stats,
                    &comp_zobrist[idx],
                    &mut comp_tt[idx],
                ) {
                    total_cost += cost;
                    lower_bounds[idx] = cost;
                    let mut trees = Vec::new();
                    for sub_tree in &result {
                        let mut sub_ls = FixedBitSet::with_capacity(label_space + 1);
                        sub_ls.grow(label_space + 1);
                        for node in sub_tree.pre_order() {
                            if sub_tree.is_leaf(node) {
                                let lbl = sub_tree.label[node as usize];
                                if lbl > 0 {
                                    sub_ls.insert(lbl as usize);
                                }
                            }
                        }
                        let expanded =
                            expand_leafset(&sub_ls, &collapsed_into, num_leaves, label_space);
                        trees.push(build_component_tree(
                            &expanded,
                            &forests[0].tree,
                            num_leaves,
                        ));
                    }
                    component_results.push(trees);
                    found = true;
                    break;
                }
            }
            if !found {
                return None;
            }
        }
    }

    if total_cost > remaining {
        return None;
    }

    let mut result_trees: Vec<Tree> = Vec::new();
    let non_iso_hashes: Vec<u64> = non_iso_comps.iter().map(hash_bitset).collect();

    for comp_ls in all_comps {
        let h = hash_bitset(comp_ls);
        if non_iso_comps
            .iter()
            .zip(non_iso_hashes.iter())
            .any(|(c, &ch)| ch == h && c.as_slice() == comp_ls.as_slice())
        {
            continue;
        }
        let expanded = expand_leafset(comp_ls, &collapsed_into, num_leaves, label_space);
        result_trees.push(build_component_tree(
            &expanded,
            &forests[0].tree,
            num_leaves,
        ));
    }

    for trees in component_results {
        result_trees.extend(trees);
    }

    Some(result_trees)
}
