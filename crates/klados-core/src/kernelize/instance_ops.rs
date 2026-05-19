//! Instance reduction and restriction operations.

use crate::{Instance, NodeId, Tree, NONE};
use fixedbitset::FixedBitSet;

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
    keep.insert_range(1..(label_space + 1));
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

    // `Tree::relabel` already prunes labels mapped to 0 and suppresses
    // degree-1 internal nodes. Doing `prune_to_leafset` first rebuilds every
    // tree twice, which dominates kernelization when thousands of single-leaf
    // reductions are applied.
    let reduced_trees: Vec<Tree> = instance
        .trees
        .iter()
        .map(|t| t.relabel(&label_map, new_num_leaves))
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

    // `Tree::relabel` is a combined prune+compact operation: kept labels map
    // to dense labels and removed labels map to 0. Avoid the older two-pass
    // prune-then-relabel rebuild.
    let trees: Vec<Tree> = instance
        .trees
        .iter()
        .map(|t| t.relabel(&label_map, new_n))
        .collect();

    (Instance::new(trees, new_n), reverse_map)
}

/// Restrict an instance by removing exactly one current label and compacting
/// labels with the standard order-preserving map.
///
/// This is the hot path for iterative chain/3-2 reductions. It avoids building
/// a `FixedBitSet` and full label map for every single-label deletion.
pub fn restrict_instance_remove_label(instance: &Instance, remove: u32) -> (Instance, Vec<u32>) {
    debug_assert!(remove >= 1 && remove <= instance.num_leaves);
    let new_n = instance.num_leaves - 1;

    let mut reverse_map = vec![0u32; new_n as usize + 1];
    for new_lbl in 1..=new_n {
        reverse_map[new_lbl as usize] = if new_lbl < remove {
            new_lbl
        } else {
            new_lbl + 1
        };
    }

    let trees: Vec<Tree> = instance
        .trees
        .iter()
        .map(|t| relabel_remove_label(t, remove, new_n))
        .collect();

    (Instance::new(trees, new_n), reverse_map)
}

fn relabel_remove_label(src: &Tree, remove: u32, new_num_leaves: u32) -> Tree {
    let mut out = Tree::with_capacity(new_num_leaves);

    fn build(src: &Tree, remove: u32, out: &mut Tree, node: NodeId) -> Option<NodeId> {
        if node == NONE {
            return None;
        }
        if src.is_leaf(node) {
            let old_lbl = src.label[node as usize];
            if old_lbl == 0 || old_lbl == remove {
                return None;
            }
            let new_lbl = if old_lbl < remove {
                old_lbl
            } else {
                old_lbl - 1
            };
            let id = out.parent.len() as NodeId;
            out.parent.push(NONE);
            out.left.push(NONE);
            out.right.push(NONE);
            out.label.push(new_lbl);
            out.label_to_node[new_lbl as usize] = id;
            return Some(id);
        }

        let (left, right) = src.children_pair(node);
        let l = build(src, remove, out, left);
        let r = build(src, remove, out, right);

        match (l, r) {
            (None, None) => None,
            (Some(child), None) | (None, Some(child)) => Some(child),
            (Some(lc), Some(rc)) => {
                let id = out.parent.len() as NodeId;
                out.parent.push(NONE);
                out.left.push(lc);
                out.right.push(rc);
                out.label.push(0);
                out.parent[lc as usize] = id;
                out.parent[rc as usize] = id;
                Some(id)
            }
        }
    }

    if src.root != NONE {
        if let Some(root) = build(src, remove, &mut out, src.root) {
            out.root = root;
            out.parent[root as usize] = NONE;
        } else {
            out.root = NONE;
        }
    }

    out.compute_metadata();
    out
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
