//! Kernelization pipeline for MAF instances.
//!
//! Implements three reduction rules (all applied iteratively until fixpoint):
//! 1. **Subtree reduction** — collapse common cherries iteratively (parameter-preserving)
//! 2. **Chain reduction** — compress common caterpillar chains (parameter-preserving)
//! 3. **3-2 chain reduction** — delete leaves in 3-2 patterns (parameter-reducing, 2-tree only)
//!
//! The rules interleave: each cherry reduction may create new chain opportunities,
//! and each chain/3-2 reduction may create new common cherries.

use fixedbitset::FixedBitSet;
use fxhash::FxHashMap;
use klados_core::{Instance, Tree, NONE};

use crate::shi_mestel::extraction::build_component_tree;
use crate::shi_mestel::utils::make_singleton_tree;

/// Which reduction rules are enabled.
#[derive(Clone, Debug)]
pub struct KernelizeConfig {
    pub subtree: bool,
    pub chain: bool,
    pub chain32: bool,
    /// Labels that must survive kernelization unchanged (used by cluster decomposition
    /// to protect ghost representative labels). If a protected label would be collapsed
    /// away, it is swapped to the representative position.
    pub protected_labels: Vec<u32>,
}

impl Default for KernelizeConfig {
    fn default() -> Self {
        Self {
            subtree: true,
            chain: true,
            chain32: true,
            protected_labels: Vec::new(),
        }
    }
}

/// Statistics from a kernelization run.
#[derive(Clone, Debug, Default)]
pub struct KernelizeStats {
    pub original_leaves: u32,
    pub reduced_leaves: u32,
    pub subtree_removed: usize,
    pub chain_removed: usize,
    pub chain32_removed: usize,
    /// Labels deleted by 3-2 chain reduction, in original label space.
    pub chain32_deleted_labels: Vec<u32>,
    /// For each surviving label in the reduced instance: (reduced_label, original_labels_it_represents)
    pub surviving_taxa: Vec<(u32, Vec<u32>)>,
}

/// Result of kernelization.
pub struct KernelizeResult {
    pub instance: Instance,
    pub stats: KernelizeStats,
    /// Maps reduced label → original label.
    pub reverse_map: Vec<u32>,
    /// All collapses in original label space (for solution expansion).
    pub collapses_original: Vec<(u32, Vec<u32>)>,
    /// Parameter reduction from 3-2 chain deletions.
    pub param_reduction: usize,
}

/// Run the full kernelization pipeline.
pub fn kernelize(instance: &Instance, config: &KernelizeConfig) -> KernelizeResult {
    let original_leaves = instance.num_leaves;
    let mut reduced = instance.clone();

    // Composite reverse map: current reduced label → original label
    let mut composite_rev: Vec<u32> = (0..=instance.num_leaves).collect();
    // All collapses in original label space
    let mut all_collapses_original: Vec<(u32, Vec<u32>)> = Vec::new();

    let mut subtree_removed: usize = 0;
    let mut chain_removed: usize = 0;
    let mut chain32_removed: usize = 0;
    let mut chain32_deleted_labels: Vec<u32> = Vec::new();

    // Outer fixpoint loop: interleave cherry, chain, and 3-2 reductions.
    // Priority: cherry (cheap) > chain > 3-2. Restart from cherry after any progress.
    loop {
        let mut progress = false;

        // Iterative cherry reduction: collapse one common cherry at a time
        if config.subtree {
            loop {
                let cherry = find_common_cherry(&reduced.trees, reduced.num_leaves);
                if let Some((mut keep_label, mut remove_label)) = cherry {
                    // Protect labels: if the remove_label is protected, swap it to be the representative
                    if is_protected(&config.protected_labels, &composite_rev, remove_label)
                        && !is_protected(&config.protected_labels, &composite_rev, keep_label)
                    {
                        std::mem::swap(&mut keep_label, &mut remove_label);
                    }

                    let orig_keep = composite_rev[keep_label as usize];
                    let orig_remove = composite_rev[remove_label as usize];
                    all_collapses_original.push((orig_keep, vec![orig_remove]));
                    subtree_removed += 1;

                    let collapses = vec![(keep_label, vec![remove_label])];
                    let (r, rev) = reduce_instance(&reduced, &collapses);
                    composite_rev = compose_reverse_maps(&composite_rev, &rev, r.num_leaves);
                    reduced = r;
                    progress = true;
                } else {
                    break;
                }
            }
        }

        // Chain reduction
        if config.chain {
            let found: Option<(u32, u32)> = if reduced.num_trees() == 2 {
                // Use Kelk-style 4-chain walk for 2-tree (handles pendancy)
                // Returns (victim, chain_head)
                find_4chain_candidate(&reduced.trees[0], &reduced.trees[1], reduced.num_leaves)
            } else {
                // For multi-tree: use sequence-based chain detection
                let truncate_to = reduced.trees.len() + 1;
                let collapses =
                    find_common_chains(&reduced.trees, reduced.num_leaves, truncate_to);
                // Return the first victim and chain representative
                collapses.into_iter().next().and_then(|(rep, removed)| {
                    removed.into_iter().next().map(|victim| (victim, rep))
                })
            };

            if let Some((victim, chain_neighbor)) = found {
                // Record collapse: the victim is absorbed by a surviving chain neighbor.
                let orig_victim = composite_rev[victim as usize];
                let orig_neighbor = composite_rev[chain_neighbor as usize];
                all_collapses_original.push((orig_neighbor, vec![orig_victim]));
                chain_removed += 1;

                let mut keep = FixedBitSet::with_capacity(reduced.num_leaves as usize + 1);
                for lbl in 1..=reduced.num_leaves {
                    keep.insert(lbl as usize);
                }
                keep.set(victim as usize, false);
                let (r, rev) = restrict_instance_simple(&reduced, &keep);
                composite_rev = compose_reverse_maps(&composite_rev, &rev, r.num_leaves);
                reduced = r;
                progress = true;
                continue; // chain may enable new cherries
            }
        }

        // 3-2 chain reduction (2-tree only)
        if config.chain32 && reduced.num_trees() == 2 && reduced.num_leaves >= 3 {
            if let Some(victim) =
                find_32_chain_candidate(&reduced.trees[0], &reduced.trees[1], reduced.num_leaves)
            {
                let orig_label = composite_rev[victim as usize];
                chain32_deleted_labels.push(orig_label);
                chain32_removed += 1;

                let mut keep = FixedBitSet::with_capacity(reduced.num_leaves as usize + 1);
                for lbl in 1..=reduced.num_leaves {
                    keep.insert(lbl as usize);
                }
                keep.set(victim as usize, false);
                let (r, rev) = restrict_instance_simple(&reduced, &keep);
                composite_rev = compose_reverse_maps(&composite_rev, &rev, r.num_leaves);
                reduced = r;
                progress = true;
                continue; // 3-2 may enable new cherries/chains
            }
        }

        if !progress {
            break;
        }
    }

    // Build surviving taxa info
    let surviving_taxa = build_surviving_taxa(&composite_rev, &all_collapses_original, reduced.num_leaves, original_leaves);

    let stats = KernelizeStats {
        original_leaves,
        reduced_leaves: reduced.num_leaves,
        subtree_removed,
        chain_removed,
        chain32_removed,
        chain32_deleted_labels: chain32_deleted_labels.clone(),
        surviving_taxa,
    };

    KernelizeResult {
        instance: reduced,
        stats,
        reverse_map: composite_rev,
        collapses_original: all_collapses_original,
        param_reduction: chain32_removed,
    }
}

/// Check whether a label in current reduced space maps to a protected original label.
fn is_protected(protected: &[u32], composite_rev: &[u32], label: u32) -> bool {
    if protected.is_empty() {
        return false;
    }
    let orig = composite_rev[label as usize];
    protected.contains(&orig)
}

// ═══════════════════════════════════════════════════════════════
// Solution expansion
// ═══════════════════════════════════════════════════════════════

/// Expand a solution on a kernelized instance back to the original label space.
///
/// Takes the component trees from solving the reduced instance, and expands each
/// by re-inserting the collapsed leaves (from subtree/chain reduction) and appending
/// singleton trees for 3-2 chain deleted leaves.
pub fn expand_solution(
    reduced_components: Vec<Tree>,
    result: &KernelizeResult,
    original_ref_tree: &Tree,
    original_num_leaves: u32,
) -> Vec<Tree> {
    let collapses = &result.collapses_original;
    let reverse_map = &result.reverse_map;

    let mut components = if collapses.is_empty() {
        expand_with_reverse_map(reduced_components, reverse_map, original_ref_tree, original_num_leaves)
    } else {
        expand_with_collapses(
            reduced_components,
            collapses,
            reverse_map,
            original_ref_tree,
            original_num_leaves,
        )
    };

    // Append components for each 3-2 chain deleted leaf.
    // A deleted label may have been a collapse representative that absorbed other labels,
    // so we must include those absorbed labels in the component.
    let rep_to_all = build_rep_to_all(collapses);
    for &orig_label in &result.stats.chain32_deleted_labels {
        if let Some(all_labels) = rep_to_all.get(&orig_label) {
            let mut ls = FixedBitSet::with_capacity(original_num_leaves as usize + 1);
            for &l in all_labels {
                ls.insert(l as usize);
            }
            components.push(build_component_tree(&ls, original_ref_tree, original_num_leaves));
        } else {
            components.push(make_singleton_tree(orig_label, original_num_leaves));
        }
    }

    components
}

/// Build map from collapse representative → all labels it represents (itself + removed),
/// with transitive closure: if rep A absorbs B, and B previously absorbed C, then A gets C too.
fn build_rep_to_all(collapses: &[(u32, Vec<u32>)]) -> FxHashMap<u32, Vec<u32>> {
    let mut rep_to_all: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for (rep, removed) in collapses {
        // Gather all labels to add: each removed label + any labels it had previously absorbed
        let mut to_add: Vec<u32> = Vec::new();
        for &r in removed {
            to_add.push(r);
            // Transitive: if r was itself a representative, absorb its group
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

        result.push(build_component_tree(
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
        result.push(build_component_tree(
            &expanded_ls,
            original_ref_tree,
            original_num_leaves,
        ));
    }
    result
}

// ═══════════════════════════════════════════════════════════════
// Instance reduction
// ═══════════════════════════════════════════════════════════════

/// Reduce an instance by collapsing groups of leaves.
/// Each collapse (representative, removed_labels) prunes the removed labels from all trees.
/// Returns (reduced_instance, reverse_map) where reverse_map maps new labels → old labels.
pub fn reduce_instance(instance: &Instance, collapses: &[(u32, Vec<u32>)]) -> (Instance, Vec<u32>) {
    if collapses.is_empty() {
        let reverse = (0..=instance.num_leaves).collect();
        return (instance.clone(), reverse);
    }

    let label_space = instance.num_leaves as usize;
    let mut keep = FixedBitSet::with_capacity(label_space + 1);
    for lbl in 1..=instance.num_leaves {
        keep.insert(lbl as usize);
    }
    for (_, removed) in collapses {
        for &lbl in removed {
            keep.set(lbl as usize, false);
        }
    }

    let kept_labels: Vec<u32> = keep.ones().map(|i| i as u32).collect();
    let new_num_leaves = kept_labels.len() as u32;

    let mut label_map = vec![0u32; label_space + 1];
    let mut reverse_map = vec![0u32; new_num_leaves as usize + 1];

    for (new_idx, &old_lbl) in kept_labels.iter().enumerate() {
        let new_lbl = (new_idx + 1) as u32;
        label_map[old_lbl as usize] = new_lbl;
        reverse_map[new_lbl as usize] = old_lbl;
    }

    let reduced_trees: Vec<Tree> = instance
        .trees
        .iter()
        .map(|t| {
            let pruned = t.prune_to_leafset(&keep);
            pruned.relabel(&label_map, new_num_leaves)
        })
        .collect();

    (Instance::new(reduced_trees, new_num_leaves), reverse_map)
}

// ═══════════════════════════════════════════════════════════════
// Cherry detection
// ═══════════════════════════════════════════════════════════════

/// Find a single common cherry (two sibling leaves with the same parent in all trees).
/// Returns Some((keep_label, remove_label)) if found.
fn find_common_cherry(trees: &[Tree], num_leaves: u32) -> Option<(u32, u32)> {
    if trees.is_empty() || num_leaves < 2 {
        return None;
    }

    let ref_tree = &trees[0];
    for node in ref_tree.post_order() {
        if let Some((l, r)) = ref_tree.children(node) {
            if ref_tree.is_leaf(l) && ref_tree.is_leaf(r) {
                let ll = ref_tree.label[l as usize];
                let rl = ref_tree.label[r as usize];
                if ll == 0 || rl == 0 {
                    continue;
                }

                // Check if this cherry exists in all other trees
                let mut common = true;
                for other in &trees[1..] {
                    let nl = other.node_by_label(ll);
                    let nr = other.node_by_label(rl);
                    if nl == NONE || nr == NONE {
                        common = false;
                        break;
                    }
                    let pl = other.parent[nl as usize];
                    let pr = other.parent[nr as usize];
                    if pl == NONE || pl != pr {
                        common = false;
                        break;
                    }
                }

                if common {
                    let (keep, remove) = if ll < rl { (ll, rl) } else { (rl, ll) };
                    return Some((keep, remove));
                }
            }
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════
// 4-chain reduction (Kelk-style, 2-tree)
// ═══════════════════════════════════════════════════════════════

/// Find a single 4-chain reduction candidate (2-tree only).
///
/// Walks down from each leaf, checking if 4 consecutive chain nodes match in both trees.
/// Handles pendancy: at level 3, one tree may have a cherry while the other has a chain node.
/// Returns `Some((victim_to_delete, chain_head))` if found.
fn find_4chain_candidate(t1: &Tree, t2: &Tree, num_leaves: u32) -> Option<(u32, u32)> {
    for x in 1..=num_leaves {
        let x_t1 = t1.node_by_label(x);
        let x_t2 = t2.node_by_label(x);
        if x_t1 == NONE || x_t2 == NONE {
            continue;
        }

        // x must be a leaf with parent that has exactly 1 leaf child (chain node)
        let p_t1 = t1.parent[x_t1 as usize];
        let p_t2 = t2.parent[x_t2 as usize];
        if p_t1 == NONE || p_t2 == NONE {
            continue;
        }
        if num_leaf_children(t1, p_t1) != 1 || num_leaf_children(t2, p_t2) != 1 {
            continue;
        }

        // Level 2: move to non-leaf child
        let at_t1 = non_leaf_child(t1, p_t1);
        let at_t2 = non_leaf_child(t2, p_t2);
        let (at_t1, at_t2) = match (at_t1, at_t2) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        if num_leaf_children(t1, at_t1) != 1 || num_leaf_children(t2, at_t2) != 1 {
            continue;
        }

        // Level 2 leaf children must match
        let c1 = leaf_child(t1, at_t1);
        let c2 = leaf_child(t2, at_t2);
        let (c1, c2) = match (c1, c2) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        let lbl1 = t1.label[c1 as usize];
        let lbl2 = t2.label[c2 as usize];
        if lbl1 != lbl2 {
            continue;
        }

        // Level 3: move to non-leaf child
        let at_t1 = non_leaf_child(t1, at_t1);
        let at_t2 = non_leaf_child(t2, at_t2);
        let (at_t1, at_t2) = match (at_t1, at_t2) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };

        // Level 3: handle pendancy — at most one tree can have 2 leaf children
        let nlc1 = num_leaf_children(t1, at_t1);
        let nlc2 = num_leaf_children(t2, at_t2);
        if nlc1 == 0 && nlc2 == 0 {
            continue;
        }

        let (at_t1_next, at_t2_next, chain3_label);
        if nlc1 == 1 && nlc2 == 1 {
            let lc1 = leaf_child(t1, at_t1).unwrap();
            let lc2 = leaf_child(t2, at_t2).unwrap();
            if t1.label[lc1 as usize] != t2.label[lc2 as usize] {
                continue;
            }
            chain3_label = t1.label[lc1 as usize];
            at_t1_next = non_leaf_child(t1, at_t1);
            at_t2_next = non_leaf_child(t2, at_t2);
        } else if nlc1 == 1 && nlc2 == 2 {
            let lc1 = leaf_child(t1, at_t1).unwrap();
            let lbl = t1.label[lc1 as usize];
            if !has_leaf_child_with_label(t2, at_t2, lbl) {
                continue;
            }
            chain3_label = lbl;
            at_t1_next = non_leaf_child(t1, at_t1);
            at_t2_next = None;
        } else if nlc1 == 2 && nlc2 == 1 {
            let lc2 = leaf_child(t2, at_t2).unwrap();
            let lbl = t2.label[lc2 as usize];
            if !has_leaf_child_with_label(t1, at_t1, lbl) {
                continue;
            }
            chain3_label = lbl;
            at_t1_next = None;
            at_t2_next = non_leaf_child(t2, at_t2);
        } else {
            continue;
        }

        let _ = chain3_label;

        // Level 4: find a common leaf child
        let check_t1 = at_t1_next.unwrap_or(at_t1);
        let check_t2 = at_t2_next.unwrap_or(at_t2);

        for y in 1..=num_leaves {
            if y == x || y == lbl1 || y == chain3_label {
                continue;
            }
            if has_leaf_child_with_label(t1, check_t1, y)
                && has_leaf_child_with_label(t2, check_t2, y)
            {
                return Some((y, x));
            }
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════
// 3-2 chain reduction (2-tree only)
// ═══════════════════════════════════════════════════════════════

/// Find a single 3-2 chain reduction candidate in a 2-tree instance.
///
/// A 3-2 chain is: p is a cherry sibling in both T1 and T2, with siblings q (in T1)
/// and r (in T2), where q ≠ r, and parent(q in T2) == grandparent(r in T2).
/// Deleting r reduces the MAF by exactly 1.
///
/// Returns `Some(label_to_delete)` if found, `None` otherwise.
fn find_32_chain_candidate(t1: &Tree, t2: &Tree, num_leaves: u32) -> Option<u32> {
    for p in 1..=num_leaves {
        let node_p_t1 = t1.node_by_label(p);
        let node_p_t2 = t2.node_by_label(p);
        if node_p_t1 == NONE || node_p_t2 == NONE {
            continue;
        }

        let sib_t1 = match t1.sibling(node_p_t1) {
            Some(s) if t1.is_leaf(s) => s,
            _ => continue,
        };
        let sib_t2 = match t2.sibling(node_p_t2) {
            Some(s) if t2.is_leaf(s) => s,
            _ => continue,
        };

        let q = t1.label[sib_t1 as usize];
        let r = t2.label[sib_t2 as usize];
        if q == r {
            continue;
        }

        // Forward: check parent(q in T2) == grandparent(r in T2)
        let node_q_t2 = t2.node_by_label(q);
        if node_q_t2 != NONE {
            let parent_q_t2 = t2.parent[node_q_t2 as usize];
            let parent_r_t2 = t2.parent[sib_t2 as usize];
            if parent_r_t2 != NONE {
                let gp_r_t2 = t2.parent[parent_r_t2 as usize];
                if parent_q_t2 != NONE && gp_r_t2 != NONE && parent_q_t2 == gp_r_t2 {
                    return Some(r);
                }
            }
        }

        // Symmetric: check parent(r in T1) == grandparent(q in T1)
        let node_r_t1 = t1.node_by_label(r);
        if node_r_t1 != NONE {
            let parent_r_t1 = t1.parent[node_r_t1 as usize];
            let parent_q_t1 = t1.parent[sib_t1 as usize];
            if parent_q_t1 != NONE {
                let gp_q_t1 = t1.parent[parent_q_t1 as usize];
                if parent_r_t1 != NONE && gp_q_t1 != NONE && parent_r_t1 == gp_q_t1 {
                    return Some(q);
                }
            }
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════
// Chain reduction (multi-tree)
// ═══════════════════════════════════════════════════════════════

/// Find common chains across all trees and return collapses.
/// Chains longer than `truncate_to` are compressed to that length.
fn find_common_chains(trees: &[Tree], num_leaves: u32, truncate_to: usize) -> Vec<(u32, Vec<u32>)> {
    let n = num_leaves as usize;
    if n < 5 || trees.len() < 2 {
        return Vec::new();
    }

    let all_chains: Vec<Vec<Vec<u32>>> = trees
        .iter()
        .map(|t| extract_chains(t, n))
        .collect();

    let mut collapses = Vec::new();
    'outer: for chain in &all_chains[0] {
        if chain.len() <= truncate_to {
            continue;
        }

        for other_chains in &all_chains[1..] {
            let found = other_chains.iter().any(|oc| oc == chain);
            if !found {
                continue 'outer;
            }
        }

        let representative = chain[0];
        let removed: Vec<u32> = chain[truncate_to..].to_vec();
        if !removed.is_empty() {
            collapses.push((representative, removed));
        }
    }

    collapses
}

/// Extract all maximal chains from a tree.
fn extract_chains(tree: &Tree, n: usize) -> Vec<Vec<u32>> {
    let mut chains = Vec::new();
    let mut visited = vec![false; tree.num_nodes()];

    for start_node in tree.pre_order() {
        if tree.is_leaf(start_node) || visited[start_node as usize] {
            continue;
        }

        let mut chain = Vec::new();
        let mut cur = start_node;

        loop {
            if tree.is_leaf(cur) {
                break;
            }

            let (left, right) = match tree.children(cur) {
                Some(lr) => lr,
                None => break,
            };

            let (leaf_child, next_node) = if tree.is_leaf(left) && !tree.is_leaf(right) {
                (left, right)
            } else if tree.is_leaf(right) && !tree.is_leaf(left) {
                (right, left)
            } else {
                break;
            };

            let lbl = tree.label[leaf_child as usize];
            if lbl > 0 && (lbl as usize) <= n {
                chain.push(lbl);
                visited[leaf_child as usize] = true;
            }
            cur = next_node;
        }

        if !tree.is_leaf(cur) {
            if let Some((left, right)) = tree.children(cur) {
                if tree.is_leaf(left) && tree.is_leaf(right) {
                    let ll = tree.label[left as usize];
                    let rl = tree.label[right as usize];
                    if ll > 0 {
                        chain.push(ll);
                    }
                    if rl > 0 {
                        chain.push(rl);
                    }
                }
            }
        }

        if chain.len() >= 3 {
            chains.push(chain);
        }
    }

    chains
}

// ═══════════════════════════════════════════════════════════════
// Tree helpers
// ═══════════════════════════════════════════════════════════════

fn num_leaf_children(tree: &Tree, node: u32) -> u32 {
    if let Some((l, r)) = tree.children(node) {
        let mut count = 0;
        if tree.is_leaf(l) {
            count += 1;
        }
        if tree.is_leaf(r) {
            count += 1;
        }
        count
    } else {
        0
    }
}

fn non_leaf_child(tree: &Tree, node: u32) -> Option<u32> {
    if let Some((l, r)) = tree.children(node) {
        if !tree.is_leaf(l) {
            Some(l)
        } else if !tree.is_leaf(r) {
            Some(r)
        } else {
            None
        }
    } else {
        None
    }
}

fn leaf_child(tree: &Tree, node: u32) -> Option<u32> {
    if let Some((l, r)) = tree.children(node) {
        if tree.is_leaf(l) {
            Some(l)
        } else if tree.is_leaf(r) {
            Some(r)
        } else {
            None
        }
    } else {
        None
    }
}

fn has_leaf_child_with_label(tree: &Tree, node: u32, label: u32) -> bool {
    if let Some((l, r)) = tree.children(node) {
        (tree.is_leaf(l) && tree.label[l as usize] == label)
            || (tree.is_leaf(r) && tree.label[r as usize] == label)
    } else {
        false
    }
}

// ═══════════════════════════════════════════════════════════════
// Utility functions
// ═══════════════════════════════════════════════════════════════

/// Compose reverse maps: new_label → intermediate_label → original_label
fn compose_reverse_maps(outer: &[u32], inner_rev: &[u32], new_n: u32) -> Vec<u32> {
    (0..=new_n)
        .map(|lbl| {
            if lbl == 0 {
                0
            } else {
                outer[inner_rev[lbl as usize] as usize]
            }
        })
        .collect()
}

/// Restrict instance to a subset of leaves.
fn restrict_instance_simple(instance: &Instance, keep: &FixedBitSet) -> (Instance, Vec<u32>) {
    let kept_labels: Vec<u32> = keep.ones().map(|i| i as u32).collect();
    let new_n = kept_labels.len() as u32;

    let mut label_map = vec![0u32; instance.num_leaves as usize + 1];
    let mut reverse_map = vec![0u32; new_n as usize + 1];
    for (new_idx, &old_lbl) in kept_labels.iter().enumerate() {
        let new_lbl = (new_idx + 1) as u32;
        label_map[old_lbl as usize] = new_lbl;
        reverse_map[new_lbl as usize] = old_lbl;
    }

    let trees: Vec<Tree> = instance
        .trees
        .iter()
        .map(|t| {
            let pruned = t.prune_to_leafset(keep);
            pruned.relabel(&label_map, new_n)
        })
        .collect();

    (Instance::new(trees, new_n), reverse_map)
}

/// Build info about which original taxa each surviving label represents.
fn build_surviving_taxa(
    composite_rev: &[u32],
    all_collapses: &[(u32, Vec<u32>)],
    reduced_leaves: u32,
    _original_leaves: u32,
) -> Vec<(u32, Vec<u32>)> {
    let mut label_groups: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for reduced_lbl in 1..=reduced_leaves {
        let orig = composite_rev[reduced_lbl as usize];
        label_groups.entry(orig).or_default();
    }
    for (rep, removed) in all_collapses {
        let group = label_groups.entry(*rep).or_default();
        if group.is_empty() {
            group.push(*rep);
        }
        for &r in removed {
            if !group.contains(&r) {
                group.push(r);
            }
        }
    }
    for reduced_lbl in 1..=reduced_leaves {
        let orig = composite_rev[reduced_lbl as usize];
        let group = label_groups.entry(orig).or_default();
        if group.is_empty() {
            group.push(orig);
        } else if !group.contains(&orig) {
            group.insert(0, orig);
        }
    }

    let mut result: Vec<(u32, Vec<u32>)> = label_groups.into_iter().collect();
    result.sort_by_key(|(k, _)| *k);
    result
}

// ═══════════════════════════════════════════════════════════════
// Output
// ═══════════════════════════════════════════════════════════════

/// Print kernelization results in a format comparable to Kelk's RMAFKernel output.
pub fn print_stats(stats: &KernelizeStats) {
    let total_removed =
        stats.subtree_removed + stats.chain_removed + stats.chain32_removed;
    eprintln!("// --- Kernelization is finished.");
    eprintln!("// Leaves in original instance: {}", stats.original_leaves);
    eprintln!("// Leaves in reduced instance: {}", stats.reduced_leaves);
    eprintln!(
        "// So {} leaves removed. Breakdown is as follows:",
        total_removed
    );
    eprintln!("// --- Due to subtree reduction: {}", stats.subtree_removed);
    eprintln!("// --- Due to chain reduction: {}", stats.chain_removed);
    eprintln!(
        "// --- Due to 3-2 chain reduction: {}",
        stats.chain32_removed
    );
}

/// Print detailed taxon info matching Kelk's verbose output.
pub fn print_taxa_detail(stats: &KernelizeStats) {
    eprintln!("// Explaining the semantics of the surviving and deleted taxa:");
    for (rep, group) in &stats.surviving_taxa {
        let labels: Vec<String> = group.iter().map(|l| l.to_string()).collect();
        eprintln!(
            "// Taxon {} represents the following subset of taxa from the original trees: {}",
            rep,
            labels.join(",")
        );
    }
    for &del in &stats.chain32_deleted_labels {
        eprintln!(
            "// 3-2 chain reduction deleted taxon {}",
            del
        );
    }
}
