//! Instance reduction and restriction operations.

use fixedbitset::FixedBitSet;
use crate::{Instance, Tree};

/// Reduce an instance by collapsing groups of leaves.
/// Each collapse (representative, removed_labels) prunes the removed labels from all trees.
/// Returns (reduced_instance, reverse_map) where reverse_map maps new labels -> old labels.
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

/// Restrict instance to a subset of leaves.
pub fn restrict_instance_simple(instance: &Instance, keep: &FixedBitSet) -> (Instance, Vec<u32>) {
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

/// Compose reverse maps: new_label -> intermediate_label -> original_label
pub fn compose_reverse_maps(outer: &[u32], inner_rev: &[u32], new_n: u32) -> Vec<u32> {
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
