//! Preprocessing: Bordewich-Semple subtree reduction and instance reduction/expansion.

use fixedbitset::FixedBitSet;
use fxhash::{FxHashMap, FxHashSet};
use klados_core::{Instance, Tree, NONE};

use super::extraction::build_component_tree;

pub fn subtree_hashes(tree: &Tree) -> Vec<u64> {
    let n = tree.num_nodes();
    let mut hashes = vec![0u64; n];
    for node in tree.post_order() {
        if tree.is_leaf(node) {
            let lbl = tree.label[node as usize] as u64;
            let mut h = lbl.wrapping_mul(0x9e3779b97f4a7c15);
            h ^= h >> 30;
            hashes[node as usize] = h;
        } else {
            let (l, r) = tree.children(node).unwrap();
            let lh = hashes[l as usize];
            let rh = hashes[r as usize];
            let (lo, hi) = if lh <= rh { (lh, rh) } else { (rh, lh) };
            let mut h = lo.wrapping_mul(0xbf58476d1ce4e5b9);
            h ^= hi;
            h = h.wrapping_mul(0x94d049bb133111eb);
            h ^= h >> 31;
            hashes[node as usize] = h;
        }
    }
    hashes
}

pub fn find_common_subtrees(trees: &[Tree], num_leaves: u32) -> Vec<(u32, Vec<u32>)> {
    if trees.len() < 2 || num_leaves < 4 {
        return Vec::new();
    }

    let ref_tree = &trees[0];
    let n = num_leaves as usize;
    let mut ref_sibling = vec![0u32; n + 1];
    for node in ref_tree.post_order() {
        if let Some((l, r)) = ref_tree.children(node)
            && ref_tree.is_leaf(l) && ref_tree.is_leaf(r) {
                let ll = ref_tree.label[l as usize];
                let rl = ref_tree.label[r as usize];
                if ll != 0 && rl != 0 {
                    ref_sibling[ll as usize] = rl;
                    ref_sibling[rl as usize] = ll;
                }
            }
    }

    let has_any_pair = ref_sibling[1..].iter().any(|&s| s != 0);
    if !has_any_pair {
        return Vec::new();
    }

    for other in &trees[1..] {
        let mut other_sibling = vec![0u32; n + 1];
        for node in other.post_order() {
            if let Some((l, r)) = other.children(node)
                && other.is_leaf(l) && other.is_leaf(r) {
                    let ll = other.label[l as usize];
                    let rl = other.label[r as usize];
                    if ll != 0 && rl != 0 {
                        other_sibling[ll as usize] = rl;
                        other_sibling[rl as usize] = ll;
                    }
                }
        }

        let mut any_survives = false;
        for lbl in 1..=n {
            let rs = ref_sibling[lbl];
            if rs != 0 && other_sibling[lbl] == rs {
                any_survives = true;
                break;
            }
        }
        if !any_survives {
            return Vec::new();
        }
    }

    let all_hashes: Vec<Vec<u64>> = trees.iter().map(subtree_hashes).collect();

    let other_hash_sets: Vec<FxHashSet<(u64, u32)>> = trees[1..]
        .iter()
        .enumerate()
        .map(|(ti, other_tree)| {
            let mut set: FxHashSet<(u64, u32)> = FxHashSet::default();
            for node in other_tree.post_order() {
                if !other_tree.is_leaf(node) {
                    let h = all_hashes[ti + 1][node as usize];
                    let sz = other_tree.subtree_size[node as usize];
                    set.insert((h, sz));
                }
            }
            set
        })
        .collect();

    let num_nodes = ref_tree.num_nodes();
    let mut is_common = vec![false; num_nodes];

    for node in ref_tree.post_order() {
        if ref_tree.is_leaf(node) {
            is_common[node as usize] = true;
            continue;
        }

        let ref_size = ref_tree.subtree_size[node as usize];
        if ref_size < 2 || ref_size == num_leaves {
            continue;
        }

        let ref_hash = all_hashes[0][node as usize];
        let key = (ref_hash, ref_size);

        let all_match = other_hash_sets.iter().all(|set| set.contains(&key));
        is_common[node as usize] = all_match;
    }

    let mut collapses = Vec::new();
    for node in ref_tree.post_order() {
        if ref_tree.is_leaf(node) || !is_common[node as usize] {
            continue;
        }
        let ref_size = ref_tree.subtree_size[node as usize];
        if ref_size < 2 {
            continue;
        }

        let parent = ref_tree.parent[node as usize];
        if parent != NONE && is_common[parent as usize] {
            continue;
        }

        let mut labels = Vec::with_capacity(ref_size as usize);
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            if ref_tree.is_leaf(n) {
                labels.push(ref_tree.label[n as usize]);
            } else if let Some((l, r)) = ref_tree.children(n) {
                stack.push(r);
                stack.push(l);
            }
        }
        labels.sort_unstable();

        let representative = labels[0];
        let removed: Vec<u32> = labels[1..].to_vec();
        if !removed.is_empty() {
            collapses.push((representative, removed));
        }
    }

    collapses
}

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

pub fn expand_solution(
    reduced_components: Vec<Tree>,
    collapses: &[(u32, Vec<u32>)],
    reverse_map: &[u32],
    original_ref_tree: &Tree,
    original_num_leaves: u32,
) -> Vec<Tree> {
    if collapses.is_empty() {
        return reduced_components;
    }

    let label_space = original_num_leaves as usize;

    let mut rep_to_all: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for (rep, removed) in collapses {
        let mut all = vec![*rep];
        all.extend_from_slice(removed);
        rep_to_all.insert(*rep, all);
    }

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

/// Find common chains across all trees and return collapses.
/// A chain is a sequence of leaves appearing in identical caterpillar order.
/// Chains longer than 3 are compressed to length 3.
pub fn find_common_chains(trees: &[Tree], num_leaves: u32) -> Vec<(u32, Vec<u32>)> {
    let n = num_leaves as usize;
    if n < 5 || trees.len() < 2 {
        return Vec::new();
    }

    // Extract chains from each tree
    let all_chains: Vec<Vec<Vec<u32>>> = trees
        .iter()
        .map(|t| extract_chains(t, n))
        .collect();

    // Find chains common to ALL trees
    let mut collapses = Vec::new();
    'outer: for chain in &all_chains[0] {
        if chain.len() <= 3 {
            continue;
        }

        // Check if this chain appears in all other trees
        for other_chains in &all_chains[1..] {
            let found = other_chains.iter().any(|oc| oc == chain);
            if !found {
                continue 'outer;
            }
        }

        // Common chain found - keep first 3, remove the rest
        let representative = chain[0];
        let removed: Vec<u32> = chain[3..].to_vec();
        if !removed.is_empty() {
            collapses.push((representative, removed));
        }
    }

    collapses
}

/// Extract all maximal chains from a tree.
/// A chain is a path where each internal node has exactly one leaf child.
fn extract_chains(tree: &Tree, n: usize) -> Vec<Vec<u32>> {
    let mut chains = Vec::new();
    let mut visited = vec![false; tree.num_nodes()];

    for start_node in tree.pre_order() {
        if tree.is_leaf(start_node) || visited[start_node as usize] {
            continue;
        }

        // Try to extend a chain from this node
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

            // Check if this is a chain node (one leaf, one internal)
            let (leaf_child, next_node) = if tree.is_leaf(left) && !tree.is_leaf(right) {
                (left, right)
            } else if tree.is_leaf(right) && !tree.is_leaf(left) {
                (right, left)
            } else {
                // Not a chain node
                break;
            };

            let lbl = tree.label[leaf_child as usize];
            if lbl > 0 && (lbl as usize) <= n {
                chain.push(lbl);
                visited[leaf_child as usize] = true;
            }
            cur = next_node;
        }

        // Add any remaining leaves at the end of the chain
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

        if chain.len() >= 4 {
            chains.push(chain);
        }
    }

    chains
}
