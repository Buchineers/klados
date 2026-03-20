//! Solution expansion: map kernelized solutions back to original label space.

use fixedbitset::FixedBitSet;
use fxhash::FxHashMap;
use crate::Tree;
use super::KernelizeResult;

/// Expand a solution on a kernelized instance back to the original label space.
///
/// Takes the component trees from solving the reduced instance, and expands each
/// by re-inserting the collapsed leaves (from subtree/chain reduction) and appending
/// singleton trees for parameter-reducing rule deleted leaves.
pub fn expand_solution(
    reduced_components: Vec<Tree>,
    result: &KernelizeResult,
    original_ref_tree: &Tree,
    original_num_leaves: u32,
) -> Vec<Tree> {
    let collapses = &result.collapses_original;
    let reverse_map = &result.reverse_map;

    let mut components = if collapses.is_empty() {
        expand_with_reverse_map(
            reduced_components,
            reverse_map,
            original_ref_tree,
            original_num_leaves,
        )
    } else {
        expand_with_collapses(
            reduced_components,
            collapses,
            reverse_map,
            original_ref_tree,
            original_num_leaves,
        )
    };

    // Append components for each deleted leaf (forced singletons).
    let rep_to_all = build_rep_to_all(collapses);
    for &orig_label in &result.stats.deleted_labels {
        if let Some(all_labels) = rep_to_all.get(&orig_label) {
            let mut ls = FixedBitSet::with_capacity(original_num_leaves as usize + 1);
            for &l in all_labels {
                ls.insert(l as usize);
            }
            components.push(Tree::component_from_leafset(
                &ls,
                original_ref_tree,
                original_num_leaves,
            ));
        } else {
            components.push(Tree::singleton(orig_label, original_num_leaves));
        }
    }

    components
}

/// Build map from collapse representative -> all labels it represents (itself + removed),
/// with transitive closure.
pub fn build_rep_to_all(collapses: &[(u32, Vec<u32>)]) -> FxHashMap<u32, Vec<u32>> {
    let mut rep_to_all: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for (rep, removed) in collapses {
        let mut to_add: Vec<u32> = Vec::new();
        for &r in removed {
            to_add.push(r);
            if let Some(sub_group) = rep_to_all.remove(&r) {
                for &l in &sub_group {
                    if l != r {
                        to_add.push(l);
                    }
                }
            }
        }

        let entry = rep_to_all.entry(*rep).or_insert_with(|| vec![*rep]);
        for l in to_add {
            if !entry.contains(&l) {
                entry.push(l);
            }
        }
    }
    rep_to_all
}

/// Expand solution when there are collapses (subtree/chain reductions).
fn expand_with_collapses(
    reduced_components: Vec<Tree>,
    collapses: &[(u32, Vec<u32>)],
    reverse_map: &[u32],
    original_ref_tree: &Tree,
    original_num_leaves: u32,
) -> Vec<Tree> {
    let label_space = original_num_leaves as usize;
    let rep_to_all = build_rep_to_all(collapses);

    let mut result = Vec::with_capacity(reduced_components.len());
    for comp in &reduced_components {
        let mut expanded_ls = FixedBitSet::with_capacity(label_space + 1);
        for new_lbl in comp.leaves() {
            let old_lbl = reverse_map[new_lbl as usize];
            if let Some(all_labels) = rep_to_all.get(&old_lbl) {
                for &l in all_labels {
                    expanded_ls.insert(l as usize);
                }
            } else {
                expanded_ls.insert(old_lbl as usize);
            }
        }

        result.push(Tree::component_from_leafset(
            &expanded_ls,
            original_ref_tree,
            original_num_leaves,
        ));
    }
    result
}

/// Expand solution when there are no collapses (only relabeling via reverse_map).
fn expand_with_reverse_map(
    reduced_components: Vec<Tree>,
    reverse_map: &[u32],
    original_ref_tree: &Tree,
    original_num_leaves: u32,
) -> Vec<Tree> {
    let label_space = original_num_leaves as usize;
    let mut result = Vec::with_capacity(reduced_components.len());
    for comp in &reduced_components {
        let mut expanded_ls = FixedBitSet::with_capacity(label_space + 1);
        for new_lbl in comp.leaves() {
            let old_lbl = reverse_map[new_lbl as usize];
            expanded_ls.insert(old_lbl as usize);
        }
        result.push(Tree::component_from_leafset(
            &expanded_ls,
            original_ref_tree,
            original_num_leaves,
        ));
    }
    result
}
