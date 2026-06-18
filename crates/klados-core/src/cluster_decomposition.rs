//! rspr-style cluster decomposition for MAF instances.
//!
//! Implements Whidden's cluster decomposition generalized to m >= 2 trees:
//! identifies "cluster points" in a reference tree T0 whose subtree leaf-sets
//! can be solved independently (verified against ALL other trees), then
//! recombines the sub-solutions with remapped labels.

use fixedbitset::FixedBitSet;

use crate::Instance;
use crate::tree::{Label, NONE, NodeId, Tree};

/// A selected cluster: a disjoint leaf set that can be solved independently.
#[derive(Clone, Debug)]
pub struct SelectedCluster {
    /// The leaf set of this cluster (1-based labels).
    pub leaves: FixedBitSet,
}

/// Find disjoint rspr-style clusters for any m >= 2 instance.
///
/// Uses T0 (first tree) as reference. A node in T0 is a cluster point if the
/// round-trip depth check passes against ALL other trees.
///
/// Returns None if m < 2, n < 5, or no useful clusters found.
pub fn find_clusters(instance: &Instance) -> Option<Vec<SelectedCluster>> {
    let m = instance.num_trees();
    if m < 2 {
        return None;
    }
    let n = instance.num_leaves as usize;
    if n < 5 {
        return None;
    }

    let t0 = &instance.trees[0];
    let leaf_sets = compute_leaf_sets(t0, n);

    // Compute twins from T0 to each other tree, and back.
    let mut twins_0_to_t: Vec<Vec<NodeId>> = Vec::with_capacity(m);
    let mut twins_t_to_0: Vec<Vec<NodeId>> = Vec::with_capacity(m);
    for t in 0..m {
        twins_0_to_t.push(compute_twins(t0, &instance.trees[t]));
        twins_t_to_0.push(compute_twins(&instance.trees[t], t0));
    }

    // Find cluster points: must pass round-trip depth check against ALL other trees.
    let cluster_points =
        find_cluster_points_multi(t0, &twins_0_to_t, &twins_t_to_0, &leaf_sets, n, m);

    if cluster_points.is_empty() {
        return None;
    }
    let selected = select_disjoint_clusters(t0, &cluster_points, &leaf_sets, n);
    if selected.is_empty() {
        return None;
    }
    Some(
        selected
            .iter()
            .map(|&node| SelectedCluster {
                leaves: leaf_sets[node as usize].clone(),
            })
            .collect(),
    )
}

/// Attempt rspr-style cluster decomposition on an instance with m >= 2 trees.
///
/// Returns `None` if no useful clusters are found or a sub-problem
/// cannot be solved. Otherwise returns the combined agreement-forest
/// components with original labels restored.
pub fn try_rspr_cluster_decomposition<S>(
    instance: &Instance,
    solve_subproblem: &mut S,
) -> Option<Vec<Tree>>
where
    S: FnMut(&Instance) -> Option<Vec<Tree>>,
{
    let m = instance.num_trees();
    if m < 2 {
        return None;
    }

    let n = instance.num_leaves as usize;
    if n < 5 {
        return None;
    }

    let t0 = &instance.trees[0];

    // Step 1: compute twins between T0 and all other trees
    let mut twins_0_to_t: Vec<Vec<NodeId>> = Vec::with_capacity(m);
    let mut twins_t_to_0: Vec<Vec<NodeId>> = Vec::with_capacity(m);
    for t in 0..m {
        twins_0_to_t.push(compute_twins(t0, &instance.trees[t]));
        twins_t_to_0.push(compute_twins(&instance.trees[t], t0));
    }

    // Step 2: compute leaf-sets for each node in T0
    let leaf_sets = compute_leaf_sets(t0, n);

    // Step 3: find cluster points (must pass for ALL trees)
    let cluster_points =
        find_cluster_points_multi(t0, &twins_0_to_t, &twins_t_to_0, &leaf_sets, n, m);

    if cluster_points.is_empty() {
        return None;
    }

    // Step 4: select disjoint clusters
    let selected = select_disjoint_clusters(t0, &cluster_points, &leaf_sets, n);

    if selected.is_empty() {
        return None;
    }

    // Step 5: build and solve each cluster sub-instance, plus the remainder
    let mut all_components: Vec<Tree> = Vec::new();

    // Track which leaves are consumed by clusters
    let mut consumed = FixedBitSet::with_capacity(n + 1);

    for &node in &selected {
        let cluster_leaves = &leaf_sets[node as usize];
        let cluster_size = cluster_leaves.count_ones(..);

        // Build compact sub-instance
        let (sub_instance, old_labels) = build_sub_instance(instance, cluster_leaves, cluster_size);

        // Solve sub-instance
        let sub_solution = solve_subproblem(&sub_instance)?;

        // Remap labels back and collect components
        for component in &sub_solution {
            let remapped = remap_component(component, &old_labels, instance.num_leaves);
            all_components.push(remapped);
        }

        consumed.union_with(cluster_leaves);
    }

    // Build remainder instance (leaves not in any cluster)
    let mut remainder_leaves = FixedBitSet::with_capacity(n + 1);
    for lbl in 1..=n {
        if !consumed.contains(lbl) {
            remainder_leaves.insert(lbl);
        }
    }

    let remainder_size = remainder_leaves.count_ones(..);
    if remainder_size > 0 {
        if remainder_size == 1 {
            // Single leaf remainder is a trivial singleton component
            let lbl = remainder_leaves.ones().next().unwrap() as Label;
            all_components.push(Tree::singleton(lbl, instance.num_leaves));
        } else {
            let (rem_instance, rem_old_labels) =
                build_sub_instance(instance, &remainder_leaves, remainder_size);

            let rem_solution = solve_subproblem(&rem_instance)?;

            for component in &rem_solution {
                let remapped = remap_component(component, &rem_old_labels, instance.num_leaves);
                all_components.push(remapped);
            }
        }
    }

    Some(all_components)
}

/// Compute twin mapping: for each node in `src`, find its twin (LCA of
/// descendant leaves) in `dst`. Returns a vec indexed by src NodeId.
fn compute_twins(src: &Tree, dst: &Tree) -> Vec<NodeId> {
    let num_nodes = src.num_nodes();
    let mut twin = vec![NONE; num_nodes];

    for node in src.post_order() {
        if src.is_leaf(node) {
            let lbl = src.label[node as usize];
            if lbl > 0 && (lbl as usize) <= dst.num_leaves as usize {
                let dst_node = dst.label_to_node[lbl as usize];
                if dst_node != NONE {
                    twin[node as usize] = dst_node;
                }
            }
        } else {
            let (left, right) = src
                .children(node)
                .expect("internal node must have children");
            let twin_left = twin[left as usize];
            let twin_right = twin[right as usize];

            if twin_left != NONE && twin_right != NONE {
                twin[node as usize] = dst.nearest_common_ancestor(twin_left, twin_right);
            } else if twin_left != NONE {
                twin[node as usize] = twin_left;
            } else if twin_right != NONE {
                twin[node as usize] = twin_right;
            }
            // else remains NONE
        }
    }

    twin
}

/// Compute the leaf-set (as FixedBitSet) for every node in the tree via
/// post-order traversal.
fn compute_leaf_sets(tree: &Tree, n: usize) -> Vec<FixedBitSet> {
    let num_nodes = tree.num_nodes();
    let mut leaf_sets = vec![FixedBitSet::with_capacity(n + 1); num_nodes];

    for node in tree.post_order() {
        if tree.is_leaf(node) {
            let lbl = tree.label[node as usize];
            if lbl > 0 && (lbl as usize) <= n {
                leaf_sets[node as usize].insert(lbl as usize);
            }
        } else {
            let (left, right) = tree
                .children(node)
                .expect("internal node must have children");
            let mut set = leaf_sets[left as usize].clone();
            set.union_with(&leaf_sets[right as usize]);
            leaf_sets[node as usize] = set;
        }
    }

    leaf_sets
}

/// Identify cluster points in T0 that pass the round-trip depth check
/// against ALL other trees.
fn find_cluster_points_multi(
    t0: &Tree,
    twins_0_to_t: &[Vec<NodeId>],
    twins_t_to_0: &[Vec<NodeId>],
    leaf_sets: &[FixedBitSet],
    n: usize,
    m: usize,
) -> Vec<NodeId> {
    let mut is_cluster_point = vec![false; t0.num_nodes()];
    let mut candidates = Vec::new();

    for node in t0.post_order() {
        if t0.is_leaf(node) || t0.is_root(node) {
            continue;
        }

        let size = leaf_sets[node as usize].count_ones(..);
        if size < 3 || size > n - 2 {
            continue;
        }

        // Check round-trip depth condition against EVERY other tree.
        let depth_node = t0.depth[node as usize];
        let mut passes_all = true;

        for t in 1..m {
            let twin_in_t = twins_0_to_t[t][node as usize];
            if twin_in_t == NONE {
                passes_all = false;
                break;
            }

            let round_trip = twins_t_to_0[t][twin_in_t as usize];
            if round_trip == NONE {
                passes_all = false;
                break;
            }

            if depth_node > t0.depth[round_trip as usize] {
                passes_all = false;
                break;
            }
        }

        if passes_all {
            is_cluster_point[node as usize] = true;
            candidates.push(node);
        }
    }

    // Remove redundant: skip nodes where ALL children are also cluster points.
    let mut result = Vec::new();
    for &node in &candidates {
        if let Some((left, right)) = t0.children(node)
            && is_cluster_point[left as usize]
            && is_cluster_point[right as usize]
        {
            continue;
        }
        result.push(node);
    }

    result
}

/// Select disjoint clusters from the candidates, processing in post-order.
/// Skip clusters that overlap with already-selected ones.
fn select_disjoint_clusters(
    t1: &Tree,
    candidates: &[NodeId],
    leaf_sets: &[FixedBitSet],
    n: usize,
) -> Vec<NodeId> {
    // Build a set of candidates for quick lookup
    let mut is_candidate = vec![false; t1.num_nodes()];
    for &node in candidates {
        is_candidate[node as usize] = true;
    }

    let mut selected = Vec::new();
    let mut used_leaves = FixedBitSet::with_capacity(n + 1);

    // Process in post-order so deeper (smaller) clusters are selected first
    for node in t1.post_order() {
        if !is_candidate[node as usize] {
            continue;
        }

        let ls = &leaf_sets[node as usize];

        // Check for overlap with already-selected clusters
        let mut overlaps = false;
        for lbl in ls.ones() {
            if used_leaves.contains(lbl) {
                overlaps = true;
                break;
            }
        }

        if overlaps {
            continue;
        }

        selected.push(node);
        used_leaves.union_with(ls);
    }

    selected
}

/// Build a compact sub-Instance from a leaf subset, relabeling to 1..=k.
/// Returns the sub-instance and a mapping from new labels to old labels.
fn build_sub_instance(
    instance: &Instance,
    leaves: &FixedBitSet,
    count: usize,
) -> (Instance, Vec<Label>) {
    let new_num_leaves = count as u32;

    // old_labels[new_label] = old_label (1-indexed: old_labels[1] is the first)
    let mut old_labels: Vec<Label> = vec![0; count + 1];
    // label_map[old_label] = new_label
    let mut label_map: Vec<Label> = vec![0; instance.num_leaves as usize + 1];

    for (new_label, old_lbl) in (1u32..).zip(leaves.ones()) {
        label_map[old_lbl] = new_label;
        old_labels[new_label as usize] = old_lbl as Label;
    }

    let sub_trees: Vec<Tree> = instance
        .trees
        .iter()
        .map(|tree| tree.relabel(&label_map, new_num_leaves))
        .collect();

    (Instance::new(sub_trees, new_num_leaves), old_labels)
}

/// Remap a component tree from compact labels back to original labels.
fn remap_component(component: &Tree, old_labels: &[Label], original_num_leaves: u32) -> Tree {
    // Build reverse map: new_label -> old_label
    let mut label_map: Vec<Label> = vec![0; component.num_leaves as usize + 1];
    for new_lbl in 1..=component.num_leaves {
        if (new_lbl as usize) < old_labels.len() {
            label_map[new_lbl as usize] = old_labels[new_lbl as usize];
        }
    }

    component.relabel(&label_map, original_num_leaves)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_leaf(tree: &mut Tree, label: Label) -> NodeId {
        let id = tree.parent.len() as NodeId;
        tree.parent.push(NONE);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(label);
        tree.label_to_node[label as usize] = id;
        id
    }

    fn push_internal(tree: &mut Tree, left: NodeId, right: NodeId) -> NodeId {
        let id = tree.parent.len() as NodeId;
        tree.parent.push(NONE);
        tree.left.push(left);
        tree.right.push(right);
        tree.label.push(0);
        tree.parent[left as usize] = id;
        tree.parent[right as usize] = id;
        id
    }

    #[derive(Clone)]
    enum Shape {
        Leaf(u32),
        Node(Box<Shape>, Box<Shape>),
    }

    fn make_tree(shape: Shape, num_leaves: u32) -> Tree {
        fn build(shape: &Shape, tree: &mut Tree) -> NodeId {
            match shape {
                Shape::Leaf(lbl) => push_leaf(tree, *lbl),
                Shape::Node(left, right) => {
                    let l = build(left, tree);
                    let r = build(right, tree);
                    push_internal(tree, l, r)
                }
            }
        }

        let mut tree = Tree::with_capacity(num_leaves);
        let root = build(&shape, &mut tree);
        tree.root = root;
        tree.parent[root as usize] = NONE;
        tree.compute_metadata();
        tree
    }

    fn l(x: u32) -> Shape {
        Shape::Leaf(x)
    }

    fn n(left: Shape, right: Shape) -> Shape {
        Shape::Node(Box::new(left), Box::new(right))
    }

    #[test]
    fn test_compute_twins_identity() {
        // Two identical trees: twin of every node should be the corresponding
        // node in the other tree (same structure).
        let t = make_tree(n(n(l(1), l(2)), n(l(3), l(4))), 4);
        let twins = compute_twins(&t, &t);

        // Every leaf maps to itself
        for lbl in 1..=4u32 {
            let node = t.label_to_node[lbl as usize];
            assert_eq!(twins[node as usize], node);
        }
        // Root's twin is root
        assert_eq!(twins[t.root as usize], t.root);
    }

    #[test]
    fn test_cluster_decomposition_identical_trees() {
        // Identical trees with a clear cluster: ((1,2),(3,(4,5)))
        // Cluster {1,2} and cluster {4,5} are both valid.
        let t1 = make_tree(n(n(l(1), l(2)), n(l(3), n(l(4), l(5)))), 5);
        let t2 = make_tree(n(n(l(1), l(2)), n(l(3), n(l(4), l(5)))), 5);
        let instance = Instance::new(vec![t1, t2], 5);

        let result = try_rspr_cluster_decomposition(&instance, &mut |sub| {
            // For identical trees the MAF is the single tree itself
            Some(vec![sub.trees[0].clone()])
        });

        // Should return Some with components covering all leaves
        let components = result.expect("decomposition should succeed");
        let mut all_leaves = FixedBitSet::with_capacity(6);
        for comp in &components {
            for lbl in comp.leaves() {
                all_leaves.insert(lbl as usize);
            }
        }
        for lbl in 1..=5 {
            assert!(
                all_leaves.contains(lbl),
                "leaf {} missing from components",
                lbl
            );
        }
    }

    #[test]
    fn test_cluster_decomposition_returns_none_for_3_trees() {
        let t = make_tree(n(n(l(1), l(2)), n(l(3), l(4))), 4);
        let instance = Instance::new(vec![t.clone(), t.clone(), t.clone()], 4);

        let result = try_rspr_cluster_decomposition(&instance, &mut |_| None);
        assert!(result.is_none());
    }

    #[test]
    fn test_cluster_decomposition_small_instance() {
        // Instance too small (< 5 leaves)
        let t = make_tree(n(n(l(1), l(2)), l(3)), 3);
        let instance = Instance::new(vec![t.clone(), t.clone()], 3);

        let result = try_rspr_cluster_decomposition(&instance, &mut |_| None);
        assert!(result.is_none());
    }

    #[test]
    fn test_cluster_decomposition_with_different_trees() {
        // T1: ((1,2),(3,(4,5)))
        // T2: ((1,3),(2,(4,5)))
        // Cluster {4,5} is shared and valid; {1,2} is NOT a cluster in T2.
        let t1 = make_tree(n(n(l(1), l(2)), n(l(3), n(l(4), l(5)))), 5);
        let t2 = make_tree(n(n(l(1), l(3)), n(l(2), n(l(4), l(5)))), 5);
        let instance = Instance::new(vec![t1, t2], 5);

        let result = try_rspr_cluster_decomposition(&instance, &mut |sub| {
            // Trivial solver: return the first tree as a single component
            Some(vec![sub.trees[0].clone()])
        });

        if let Some(components) = result {
            // Verify all leaves are covered
            let mut all_leaves = FixedBitSet::with_capacity(6);
            for comp in &components {
                for lbl in comp.leaves() {
                    all_leaves.insert(lbl as usize);
                }
            }
            for lbl in 1..=5 {
                assert!(
                    all_leaves.contains(lbl),
                    "leaf {} missing from components",
                    lbl
                );
            }
        }
        // It's also valid for the decomposition to find no clusters and return None
    }
}
