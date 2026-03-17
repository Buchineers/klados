//! MAF component extraction from solved state.

use fixedbitset::FixedBitSet;
use klados_core::{Tree, XForest};

use super::forest_nav::component_leaf_sets_xf;
use super::search_state::Collapses;

pub fn extract_maf_components(
    forest: &XForest,
    collapses: &Collapses,
    label_space: usize,
    num_leaves: u32,
) -> Vec<Tree> {
    let collapsed_into = build_collapsed_into(collapses, num_leaves);
    let comps = component_leaf_sets_xf(forest, label_space);
    let mut result = Vec::new();
    for comp_ls in &comps {
        if comp_ls.count_ones(..) == 0 {
            continue;
        }
        let expanded = expand_leafset(comp_ls, &collapsed_into, num_leaves, label_space);
        result.push(build_component_tree(&expanded, &forest.tree, num_leaves));
    }
    result
}

pub fn build_collapsed_into(collapses: &Collapses, num_leaves: u32) -> Vec<u32> {
    let mut collapsed_into: Vec<u32> = (0..=num_leaves).collect();
    for &(removed, kept) in collapses {
        collapsed_into[removed as usize] = kept;
    }
    for lbl in 1..=num_leaves {
        let mut cur = lbl;
        while collapsed_into[cur as usize] != cur {
            cur = collapsed_into[cur as usize];
        }
        collapsed_into[lbl as usize] = cur;
    }
    collapsed_into
}

pub fn expand_leafset(
    comp_ls: &FixedBitSet,
    collapsed_into: &[u32],
    num_leaves: u32,
    label_space: usize,
) -> FixedBitSet {
    let mut expanded = FixedBitSet::with_capacity(label_space + 1);
    expanded.grow(label_space + 1);
    for lbl in 1..=num_leaves {
        let target = collapsed_into[lbl as usize];
        if comp_ls.contains(target as usize) {
            expanded.insert(lbl as usize);
        }
    }
    expanded
}

pub fn build_component_tree(
    expanded: &FixedBitSet,
    reference_tree: &Tree,
    num_leaves: u32,
) -> Tree {
    Tree::component_from_leafset(expanded, reference_tree, num_leaves)
}

pub fn classify_components_cached(
    forests: &[XForest],
    all_comps: &[FixedBitSet],
) -> (Vec<FixedBitSet>, Vec<FixedBitSet>) {
    let all_comps = all_comps.to_vec();
    let mut non_iso = Vec::new();

    for comp_ls in &all_comps {
        if comp_ls.count_ones(..) <= 1 {
            continue;
        }
        let ref_canon = super::utils::tree_canonical_for_labels(&forests[0].tree, comp_ls);
        let mut all_same = true;
        for forest in &forests[1..] {
            if super::utils::tree_canonical_for_labels(&forest.tree, comp_ls) != ref_canon {
                all_same = false;
                break;
            }
        }
        if !all_same {
            non_iso.push(comp_ls.clone());
        }
    }

    non_iso.sort_by_key(|c| c.count_ones(..));
    (all_comps, non_iso)
}
