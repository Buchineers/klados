//! Component decomposition for independent sub-problem solving.
//!
//! After split branching exhausts all overlapping components, each remaining
//! non-isomorphic component can be solved independently. This module builds
//! compact sub-instances for each component and delegates to a solver callback.

use fixedbitset::FixedBitSet;
use klados_core::{Instance, Tree, XForest};

use super::extraction::{build_collapsed_into, expand_leafset};
use super::search_state::Collapses;

/// Solve each non-isomorphic component independently by building compact
/// sub-instances and delegating to `solve_sub`.
///
/// Isomorphic components (same topology in all trees) are already solved and
/// included directly.
///
/// Returns the combined MAF components (one `Tree` per component), or `None`
/// if any sub-problem is infeasible.
#[allow(dead_code)]
pub fn solve_decomposed_simple<S>(
    forests: &[XForest],
    collapses: &Collapses,
    label_space: usize,
    num_leaves: u32,
    non_iso_comps: &[FixedBitSet],
    all_comps: &[FixedBitSet],
    solve_sub: &mut S,
) -> Option<Vec<Tree>>
where
    S: FnMut(&Instance) -> Option<Vec<Tree>>,
{
    let collapsed_into = build_collapsed_into(collapses, num_leaves);

    super::trace!(
        "decompose_simple: {} total components, {} non-isomorphic",
        all_comps.len(),
        non_iso_comps.len(),
    );

    let mut result_trees: Vec<Tree> = Vec::new();

    // 1. Collect isomorphic components directly — they are already agreement
    //    subtrees (same topology in every tree).
    let non_iso_slices: Vec<&[usize]> = non_iso_comps.iter().map(|c| c.as_slice()).collect();
    for comp_ls in all_comps {
        let is_non_iso = non_iso_slices.iter().any(|&s| s == comp_ls.as_slice());
        if is_non_iso {
            continue;
        }
        // Isomorphic component: expand collapsed labels and build tree from reference.
        let expanded = expand_leafset(comp_ls, &collapsed_into, num_leaves, label_space);
        result_trees.push(Tree::component_from_leafset(
            &expanded,
            &forests[0].tree,
            num_leaves,
        ));
    }

    // 2. Solve each non-isomorphic component independently.
    for comp_ls in non_iso_comps {
        let comp_leaves: Vec<u32> = comp_ls.ones().map(|x| x as u32).collect();
        let sub_num_leaves = comp_leaves.len() as u32;

        if sub_num_leaves <= 1 {
            // Singleton or empty — add directly.
            if comp_leaves.first().is_some() {
                let expanded = expand_leafset(comp_ls, &collapsed_into, num_leaves, label_space);
                result_trees.push(Tree::component_from_leafset(
                    &expanded,
                    &forests[0].tree,
                    num_leaves,
                ));
            }
            continue;
        }

        // Build compact label mapping: old_label -> new_label (1-based).
        let max_old_label = comp_leaves.iter().copied().max().unwrap_or(0) as usize;
        let mut old_to_new = vec![0u32; max_old_label + 1];
        let mut new_to_old = vec![0u32; sub_num_leaves as usize + 1];
        for (i, &old_lbl) in comp_leaves.iter().enumerate() {
            let new_lbl = (i + 1) as u32;
            old_to_new[old_lbl as usize] = new_lbl;
            new_to_old[new_lbl as usize] = old_lbl;
        }

        // For each tree: prune to this component's leaves, then relabel compactly.
        let sub_trees: Vec<Tree> = forests
            .iter()
            .map(|f| {
                let pruned = f.tree.prune_to_leafset(comp_ls);
                pruned.relabel(&old_to_new, sub_num_leaves)
            })
            .collect();

        let sub_instance = Instance::new(sub_trees, sub_num_leaves);

        super::trace!(
            "  sub-instance: {} leaves from component with labels {:?}",
            sub_num_leaves,
            &comp_leaves,
        );

        // Solve the sub-instance.
        let sub_result = solve_sub(&sub_instance)?;

        // Remap sub-solver output: relabel compact → original labels, then expand
        // outer collapses. Use relabel() on the sub-solver's actual tree structure
        // (not component_from_leafset, which uses the wrong reference tree).
        let mut reverse_map = vec![0u32; sub_num_leaves as usize + 1];
        for (i, &old_lbl) in comp_leaves.iter().enumerate() {
            let new_lbl = (i + 1) as u32;
            // Map compact label back to the FULL original label space:
            // compact → search-state label → all original labels collapsed into it.
            reverse_map[new_lbl as usize] = old_lbl;
        }

        // Build a full relabel map: for each compact label, find ALL original
        // labels that were collapsed into the corresponding search-state label.
        // We need to map compact → num_leaves space for relabel().
        let mut full_relabel = vec![0u32; sub_num_leaves as usize + 1];
        for compact_lbl in 1..=sub_num_leaves {
            let search_lbl = reverse_map[compact_lbl as usize];
            // The search_lbl IS a valid label in the outer state.
            // We keep it as-is for the relabel — expand_solution in the
            // outer shi-mestel solve() handles collapse expansion.
            full_relabel[compact_lbl as usize] = search_lbl;
        }

        for sub_tree in &sub_result {
            let remapped = sub_tree.relabel(&full_relabel, num_leaves);
            result_trees.push(remapped);
        }
    }

    Some(result_trees)
}
