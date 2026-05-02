//! Cluster reduction for rooted multi-tree MAF.
//!
//! This follows Kelk's four-subinstance reduction, generalized to m trees:
//! - `TP1`: cluster subtree with a marker leaf at the cluster root (all trees)
//! - `TP2`: cluster subtree without the marker (all trees)
//! - `TP3`: top part with a placeholder leaf for the cluster (all trees)
//! - `TP4`: top part without the placeholder (all trees)
//!
//! If `OPT(TP1) = OPT(TP2)` and `OPT(TP3) = OPT(TP4)`, then the original
//! instance has optimum `OPT(TP2) + OPT(TP4) - 1`; otherwise it is
//! `OPT(TP2) + OPT(TP4)`.

use fixedbitset::FixedBitSet;
use fxhash::FxHashMap;
use crate::Instance;
use crate::tree::{Label, NONE, NodeId, Tree};

#[derive(Clone, Debug)]
pub struct CommonCluster {
    pub leaves: FixedBitSet,
    nodes: Vec<NodeId>,
}

#[derive(Clone, Debug)]
pub struct ClusterReductionSolution {
    pub components: Vec<Tree>,
    pub cluster_size: usize,
    pub rest_size: usize,
}

#[derive(Clone, Debug)]
pub enum ClusterReductionResult {
    NotApplicable,
    Solved(ClusterReductionSolution),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SubproblemLabel {
    Unused,
    Original(u32),
    MarkerDown,
    MarkerUp,
}

#[derive(Clone, Debug)]
struct PackedSubproblem {
    instance: Instance,
    labels: Vec<SubproblemLabel>,
}

#[derive(Clone, Debug)]
struct ClusterSubproblems {
    tp1: PackedSubproblem,
    tp2: PackedSubproblem,
    tp3: PackedSubproblem,
    tp4: PackedSubproblem,
}

#[derive(Clone, Debug)]
struct DecodedForest {
    plain_components: Vec<FixedBitSet>,
    marker_component: Option<FixedBitSet>,
}

pub fn find_best_common_cluster(instance: &Instance) -> Option<CommonCluster> {
    let m = instance.num_trees();
    if m < 2 || instance.num_leaves < 5 {
        return None;
    }

    let n = instance.num_leaves as usize;

    // Build per-tree lookup: leafset-key -> NodeId
    let per_tree: Vec<FxHashMap<Vec<usize>, NodeId>> = instance
        .trees
        .iter()
        .map(|tree| {
            tree_clusters_with_nodes(tree, n)
                .into_iter()
                .map(|(leaves, node)| (leafset_key(&leaves), node))
                .collect()
        })
        .collect();

    // Use tree 0 as base, intersect with all others
    let target = n / 2;
    let mut best: Option<(CommonCluster, usize)> = None;

    for (key, &node0) in &per_tree[0] {
        // Check all other trees have this cluster
        let mut nodes = vec![node0];
        let mut all_present = true;
        for lookup in &per_tree[1..] {
            if let Some(&node_i) = lookup.get(key) {
                nodes.push(node_i);
            } else {
                all_present = false;
                break;
            }
        }
        if !all_present {
            continue;
        }

        // Reconstruct leaf bitset from key
        let mut leaves = FixedBitSet::with_capacity(n + 1);
        for &lbl in key {
            leaves.insert(lbl);
        }
        let size = leaves.count_ones(..);
        if size <= 2 || size >= n - 1 {
            continue;
        }

        let dist = (size as isize - target as isize).unsigned_abs();
        let candidate = CommonCluster { leaves, nodes };

        match &best {
            None => best = Some((candidate, dist)),
            Some((current, current_dist)) => {
                let current_size = current.leaves.count_ones(..);
                let better = dist < *current_dist
                    || (dist == *current_dist && size > current_size)
                    || (dist == *current_dist
                        && size == current_size
                        && key < &leafset_key(&current.leaves));
                if better {
                    best = Some((candidate, dist));
                }
            }
        }
    }

    best.map(|(cluster, _)| cluster)
}

pub fn try_cluster_reduction<S>(
    instance: &Instance,
    solve_subproblem: &mut S,
) -> Option<ClusterReductionResult>
where
    S: FnMut(&Instance) -> Option<Vec<Tree>>,
{
    let Some(cluster) = find_best_common_cluster(instance) else {
        return Some(ClusterReductionResult::NotApplicable);
    };

    let subproblems = build_cluster_subproblems(instance, &cluster);

    let tp2_solution = solve_subproblem(&subproblems.tp2.instance)?;

    // Adding the marker leaf cannot lower the optimum: deleting that leaf from
    // any marker-instance forest yields a forest for the closed instance. If we
    // can attach the marker to an existing optimal closed component and validate
    // the resulting forest, equality is proven and the marker solve is skipped.
    let (tp1_solution, tp1_equal) = match try_attach_marker_to_solution(
        &tp2_solution,
        &subproblems.tp2.labels,
        &subproblems.tp1,
        SubproblemLabel::MarkerDown,
    ) {
        Some(solution) => (Some(solution), true),
        None => {
            let solution = solve_subproblem(&subproblems.tp1.instance)?;
            let equal = solution.len() == tp2_solution.len();
            (Some(solution), equal)
        }
    };

    let tp4_solution = solve_subproblem(&subproblems.tp4.instance)?;

    let (tp3_solution, tp3_equal) = if tp1_equal {
        match try_attach_marker_to_solution(
            &tp4_solution,
            &subproblems.tp4.labels,
            &subproblems.tp3,
            SubproblemLabel::MarkerUp,
        ) {
            Some(solution) => (Some(solution), true),
            None => {
                let solution = solve_subproblem(&subproblems.tp3.instance)?;
                let equal = solution.len() == tp4_solution.len();
                (Some(solution), equal)
            }
        }
    } else {
        (None, false)
    };

    let merge_across_boundary = tp1_equal && tp3_equal;

    let final_leafsets = if merge_across_boundary {
        let tp1 = decode_forest_with_marker(
            tp1_solution.as_ref()?,
            &subproblems.tp1.labels,
            SubproblemLabel::MarkerDown,
            instance.num_leaves,
        )?;
        let tp3 = decode_forest_with_marker(
            tp3_solution.as_ref()?,
            &subproblems.tp3.labels,
            SubproblemLabel::MarkerUp,
            instance.num_leaves,
        )?;

        let mut merged = tp1.marker_component?;
        let top = tp3.marker_component?;
        if merged.count_ones(..) == 0 || top.count_ones(..) == 0 {
            return None;
        }
        merged.union_with(&top);

        let mut result = tp1.plain_components;
        result.extend(tp3.plain_components);
        result.push(merged);
        result
    } else {
        let mut result =
            decode_forest(&tp2_solution, &subproblems.tp2.labels, instance.num_leaves)?;
        result.extend(decode_forest(
            &tp4_solution,
            &subproblems.tp4.labels,
            instance.num_leaves,
        )?);
        result
    };

    let components =
        build_components_from_leafsets(&final_leafsets, &instance.trees[0], instance.num_leaves);

    let cluster_size = cluster.leaves.count_ones(..);
    let rest_size = instance.num_leaves as usize - cluster_size + 1;

    Some(ClusterReductionResult::Solved(ClusterReductionSolution {
        components,
        cluster_size,
        rest_size,
    }))
}

fn try_attach_marker_to_solution(
    closed_solution: &[Tree],
    closed_labels: &[SubproblemLabel],
    marker_subproblem: &PackedSubproblem,
    marker: SubproblemLabel,
) -> Option<Vec<Tree>> {
    if closed_solution.is_empty() {
        return None;
    }

    let n = marker_subproblem.instance.num_leaves as usize;
    let marker_label = marker_subproblem
        .labels
        .iter()
        .position(|desc| *desc == marker)?;
    let mut original_to_marker_label = FxHashMap::default();
    for (label, desc) in marker_subproblem.labels.iter().enumerate().skip(1) {
        if let SubproblemLabel::Original(original) = desc {
            original_to_marker_label.insert(*original, label);
        }
    }

    let mut base_leafsets = Vec::with_capacity(closed_solution.len());
    for component in closed_solution {
        let mut leafset = FixedBitSet::with_capacity(n + 1);
        for label in component.leaves() {
            let SubproblemLabel::Original(original) = closed_labels.get(label as usize)? else {
                return None;
            };
            let mapped = *original_to_marker_label.get(original)?;
            leafset.insert(mapped);
        }
        if leafset.count_ones(..) == 0 {
            return None;
        }
        base_leafsets.push(leafset);
    }

    let base_nodes = component_nodes_for_instance(&base_leafsets, &marker_subproblem.instance)?;
    let mut used_nodes: Vec<FixedBitSet> = marker_subproblem
        .instance
        .trees
        .iter()
        .map(|tree| FixedBitSet::with_capacity(tree.num_nodes()))
        .collect();
    for component in &base_nodes {
        for (ti, nodes) in component.iter().enumerate() {
            for node in nodes.ones() {
                if used_nodes[ti].contains(node) {
                    return None;
                }
                used_nodes[ti].insert(node);
            }
        }
    }

    for attach_idx in 0..base_leafsets.len() {
        let mut augmented = base_leafsets[attach_idx].clone();
        augmented.insert(marker_label);
        if marker_attachment_valid(
            &augmented,
            &marker_subproblem.instance,
            &base_nodes[attach_idx],
            &used_nodes,
        ) {
            let mut candidate = base_leafsets.clone();
            candidate[attach_idx] = augmented;
            return Some(build_components_from_leafsets(
                &candidate,
                marker_subproblem.instance.reference_tree(),
                marker_subproblem.instance.num_leaves,
            ));
        }
    }

    None
}

fn component_nodes_for_instance(
    leafsets: &[FixedBitSet],
    instance: &Instance,
) -> Option<Vec<Vec<FixedBitSet>>> {
    let mut out = Vec::with_capacity(leafsets.len());
    for leafset in leafsets {
        let Some(reference_signature) = induced_tree_signature(&instance.trees[0], leafset) else {
            return None;
        };

        let mut component = Vec::with_capacity(instance.num_trees());
        for tree in &instance.trees {
            let Some(signature) = induced_tree_signature(tree, leafset) else {
                return None;
            };
            if signature != reference_signature {
                return None;
            }

            let Some(nodes) = covered_internal_nodes_for_leafset(tree, leafset) else {
                return None;
            };
            let mut bitset = FixedBitSet::with_capacity(tree.num_nodes());
            for node in nodes {
                bitset.insert(node);
            }
            component.push(bitset);
        }
        out.push(component);
    }

    Some(out)
}

fn marker_attachment_valid(
    augmented_leafset: &FixedBitSet,
    instance: &Instance,
    old_component_nodes: &[FixedBitSet],
    used_nodes: &[FixedBitSet],
) -> bool {
    let Some(reference_signature) = induced_tree_signature(&instance.trees[0], augmented_leafset) else {
        return false;
    };

    for (ti, tree) in instance.trees.iter().enumerate() {
        let Some(signature) = induced_tree_signature(tree, augmented_leafset) else {
            return false;
        };
        if signature != reference_signature {
            return false;
        }

        let Some(nodes) = covered_internal_nodes_for_leafset(tree, augmented_leafset) else {
            return false;
        };
        for node in nodes {
            if used_nodes[ti].contains(node) && !old_component_nodes[ti].contains(node) {
                return false;
            }
        }
    }

    true
}

fn covered_internal_nodes_for_leafset(tree: &Tree, leafset: &FixedBitSet) -> Option<Vec<usize>> {
    let mut labels = leafset.ones();
    let first = labels.next()? as Label;
    let mut lca_node = tree.node_by_label(first);
    if lca_node == NONE {
        return None;
    }
    let mut component_labels = vec![first];
    for label in labels {
        let label = label as Label;
        let node = tree.node_by_label(label);
        if node == NONE {
            return None;
        }
        lca_node = tree.nearest_common_ancestor(lca_node, node);
        component_labels.push(label);
    }

    let mut seen = FixedBitSet::with_capacity(tree.num_nodes());
    let mut covered = Vec::new();
    for label in component_labels {
        let mut cur = tree.node_by_label(label);
        loop {
            let idx = cur as usize;
            if seen.contains(idx) {
                break;
            }
            seen.insert(idx);
            if !tree.is_leaf(cur) {
                covered.push(idx);
            }
            if cur == lca_node {
                break;
            }
            cur = tree.parent[idx];
        }
    }
    Some(covered)
}

fn induced_tree_signature(tree: &Tree, leafset: &FixedBitSet) -> Option<String> {
    fn build(tree: &Tree, leafset: &FixedBitSet, node: NodeId) -> Option<String> {
        if tree.is_leaf(node) {
            let label = tree.label[node as usize] as usize;
            return leafset.contains(label).then(|| label.to_string());
        }

        let (left, right) = tree.children(node)?;
        match (build(tree, leafset, left), build(tree, leafset, right)) {
            (None, None) => None,
            (Some(child), None) | (None, Some(child)) => Some(child),
            (Some(left_sig), Some(right_sig)) => {
                if left_sig <= right_sig {
                    Some(format!("({},{})", left_sig, right_sig))
                } else {
                    Some(format!("({},{})", right_sig, left_sig))
                }
            }
        }
    }

    build(tree, leafset, tree.root)
}

fn tree_clusters_with_nodes(tree: &Tree, num_leaves: usize) -> Vec<(FixedBitSet, NodeId)> {
    let mut leaf_sets = vec![FixedBitSet::with_capacity(num_leaves + 1); tree.num_nodes()];
    let mut clusters = Vec::new();

    for node in tree.post_order() {
        if tree.is_leaf(node) {
            let lbl = tree.label[node as usize];
            if lbl > 0 && (lbl as usize) <= num_leaves {
                leaf_sets[node as usize].insert(lbl as usize);
            }
            continue;
        }

        let (left, right) = tree
            .children(node)
            .expect("internal node must have children");
        let mut set = leaf_sets[left as usize].clone();
        set.union_with(&leaf_sets[right as usize]);
        leaf_sets[node as usize] = set;
    }

    for node in 0..tree.num_nodes() as NodeId {
        if tree.is_leaf(node) {
            continue;
        }
        let leaf_count = leaf_sets[node as usize].count_ones(..);
        if leaf_count >= 2 && leaf_count < num_leaves {
            clusters.push((leaf_sets[node as usize].clone(), node));
        }
    }

    clusters
}

fn leafset_key(leaves: &FixedBitSet) -> Vec<usize> {
    leaves.ones().collect()
}

fn build_cluster_subproblems(instance: &Instance, cluster: &CommonCluster) -> ClusterSubproblems {
    let total_label_space = instance.num_leaves + 2;
    let marker_down = instance.num_leaves + 1;
    let marker_up = instance.num_leaves + 2;

    let cluster_labels: Vec<u32> = cluster.leaves.ones().map(|i| i as u32).collect();
    let top_labels: Vec<u32> = (1..=instance.num_leaves)
        .filter(|&lbl| !cluster.leaves.contains(lbl as usize))
        .collect();

    let tp1_trees: Vec<Tree> = instance
        .trees
        .iter()
        .zip(&cluster.nodes)
        .map(|(tree, &node)| {
            attach_marker_to_cluster_root(tree, node, marker_down, total_label_space)
        })
        .collect();
    let tp2_trees: Vec<Tree> = instance
        .trees
        .iter()
        .map(|tree| tree.prune_to_leafset(&cluster.leaves))
        .collect();
    let tp3_trees: Vec<Tree> = instance
        .trees
        .iter()
        .zip(&cluster.nodes)
        .map(|(tree, &node)| {
            replace_cluster_with_marker(tree, node, marker_up, total_label_space)
        })
        .collect();
    let tp4_trees: Vec<Tree> = instance
        .trees
        .iter()
        .map(|tree| prune_complement(tree, &cluster.leaves))
        .collect();

    let tp1_labels = cluster_labels
        .iter()
        .copied()
        .map(|lbl| (lbl, SubproblemLabel::Original(lbl)))
        .chain(std::iter::once((marker_down, SubproblemLabel::MarkerDown)))
        .collect::<Vec<_>>();
    let tp2_labels = cluster_labels
        .iter()
        .copied()
        .map(|lbl| (lbl, SubproblemLabel::Original(lbl)))
        .collect::<Vec<_>>();
    let tp3_labels = top_labels
        .iter()
        .copied()
        .map(|lbl| (lbl, SubproblemLabel::Original(lbl)))
        .chain(std::iter::once((marker_up, SubproblemLabel::MarkerUp)))
        .collect::<Vec<_>>();
    let tp4_labels = top_labels
        .iter()
        .copied()
        .map(|lbl| (lbl, SubproblemLabel::Original(lbl)))
        .collect::<Vec<_>>();

    ClusterSubproblems {
        tp1: pack_subproblem(tp1_trees, &tp1_labels, total_label_space),
        tp2: pack_subproblem(tp2_trees, &tp2_labels, total_label_space),
        tp3: pack_subproblem(tp3_trees, &tp3_labels, total_label_space),
        tp4: pack_subproblem(tp4_trees, &tp4_labels, total_label_space),
    }
}

fn attach_marker_to_cluster_root(
    tree: &Tree,
    cluster_node: NodeId,
    marker_label: Label,
    total_label_space: u32,
) -> Tree {
    let cluster_subtree = clone_subtree(tree, cluster_node, total_label_space);
    let mut out = Tree::with_capacity(total_label_space);

    let cluster_root = copy_subtree_into(&cluster_subtree, cluster_subtree.root, &mut out);
    let marker = push_leaf(&mut out, marker_label);
    let root = push_internal(&mut out, cluster_root, marker);

    out.root = root;
    out.parent[root as usize] = NONE;
    out.compute_metadata();
    out
}

fn replace_cluster_with_marker(
    tree: &Tree,
    cluster_node: NodeId,
    marker_label: Label,
    total_label_space: u32,
) -> Tree {
    fn build(
        src: &Tree,
        node: NodeId,
        target: NodeId,
        marker_label: Label,
        out: &mut Tree,
    ) -> NodeId {
        if node == target {
            return push_leaf(out, marker_label);
        }

        if src.is_leaf(node) {
            return push_leaf(out, src.label[node as usize]);
        }

        let (left, right) = src
            .children(node)
            .expect("internal node must have children");
        let left_id = build(src, left, target, marker_label, out);
        let right_id = build(src, right, target, marker_label, out);
        push_internal(out, left_id, right_id)
    }

    let mut out = Tree::with_capacity(total_label_space);
    let root = build(tree, tree.root, cluster_node, marker_label, &mut out);
    out.root = root;
    out.parent[root as usize] = NONE;
    out.compute_metadata();
    out
}

fn prune_complement(tree: &Tree, cluster: &FixedBitSet) -> Tree {
    let mut keep = FixedBitSet::with_capacity(tree.num_leaves as usize + 1);
    for lbl in 1..=tree.num_leaves {
        if !cluster.contains(lbl as usize) {
            keep.insert(lbl as usize);
        }
    }
    tree.prune_to_leafset(&keep)
}

fn clone_subtree(tree: &Tree, root: NodeId, total_label_space: u32) -> Tree {
    let mut out = Tree::with_capacity(total_label_space);
    let new_root = copy_subtree_into(tree, root, &mut out);
    out.root = new_root;
    out.parent[new_root as usize] = NONE;
    out.compute_metadata();
    out
}

fn copy_subtree_into(src: &Tree, node: NodeId, out: &mut Tree) -> NodeId {
    if src.is_leaf(node) {
        return push_leaf(out, src.label[node as usize]);
    }

    let (left, right) = src
        .children(node)
        .expect("internal node must have children");
    let left_id = copy_subtree_into(src, left, out);
    let right_id = copy_subtree_into(src, right, out);
    push_internal(out, left_id, right_id)
}

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

fn pack_subproblem(
    trees: Vec<Tree>,
    label_spec: &[(u32, SubproblemLabel)],
    total_label_space: u32,
) -> PackedSubproblem {
    let mut label_map = vec![0u32; total_label_space as usize + 1];
    let mut reverse = vec![SubproblemLabel::Unused; label_spec.len() + 1];

    for (new_idx, (old_label, desc)) in label_spec.iter().enumerate() {
        let new_label = (new_idx + 1) as u32;
        label_map[*old_label as usize] = new_label;
        reverse[new_label as usize] = desc.clone();
    }

    let relabeled = trees
        .iter()
        .map(|tree| tree.relabel(&label_map, label_spec.len() as u32))
        .collect::<Vec<_>>();

    PackedSubproblem {
        instance: Instance::new(relabeled, label_spec.len() as u32),
        labels: reverse,
    }
}

fn decode_forest(
    forest: &[Tree],
    labels: &[SubproblemLabel],
    original_num_leaves: u32,
) -> Option<Vec<FixedBitSet>> {
    let mut result = Vec::new();
    for tree in forest {
        let (leafset, has_marker) = decode_component(tree, labels, original_num_leaves)?;
        if has_marker || leafset.count_ones(..) == 0 {
            return None;
        }
        result.push(leafset);
    }
    Some(result)
}

fn decode_forest_with_marker(
    forest: &[Tree],
    labels: &[SubproblemLabel],
    marker: SubproblemLabel,
    original_num_leaves: u32,
) -> Option<DecodedForest> {
    let mut plain_components = Vec::new();
    let mut marker_component = None;

    for tree in forest {
        let (leafset, has_marker) = decode_component(tree, labels, original_num_leaves)?;
        if has_marker {
            if marker_component.is_some() {
                return None;
            }

            let contains_requested_marker = tree
                .leaves()
                .any(|lbl| labels.get(lbl as usize).is_some_and(|x| *x == marker));
            if !contains_requested_marker {
                return None;
            }
            marker_component = Some(leafset);
        } else if leafset.count_ones(..) > 0 {
            plain_components.push(leafset);
        }
    }

    Some(DecodedForest {
        plain_components,
        marker_component,
    })
}

fn decode_component(
    tree: &Tree,
    labels: &[SubproblemLabel],
    original_num_leaves: u32,
) -> Option<(FixedBitSet, bool)> {
    let mut leafset = FixedBitSet::with_capacity(original_num_leaves as usize + 1);
    let mut has_marker = false;

    for lbl in tree.leaves() {
        match labels.get(lbl as usize)? {
            SubproblemLabel::Unused => return None,
            SubproblemLabel::Original(orig) => {
                leafset.insert(*orig as usize);
            }
            SubproblemLabel::MarkerDown | SubproblemLabel::MarkerUp => {
                has_marker = true;
            }
        }
    }

    Some((leafset, has_marker))
}

fn build_components_from_leafsets(
    leafsets: &[FixedBitSet],
    reference_tree: &Tree,
    num_leaves: u32,
) -> Vec<Tree> {
    leafsets
        .iter()
        .filter(|ls| ls.count_ones(..) > 0)
        .map(|ls| build_component_tree(reference_tree, num_leaves, ls))
        .collect()
}

fn build_component_tree(reference_tree: &Tree, num_leaves: u32, leafset: &FixedBitSet) -> Tree {
    Tree::component_from_leafset(leafset, reference_tree, num_leaves)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn make_cluster_instance() -> Instance {
        let t1 = make_tree(
            Shape::Node(
                Box::new(Shape::Node(
                    Box::new(Shape::Node(
                        Box::new(Shape::Leaf(1)),
                        Box::new(Shape::Leaf(2)),
                    )),
                    Box::new(Shape::Leaf(3)),
                )),
                Box::new(Shape::Node(
                    Box::new(Shape::Leaf(4)),
                    Box::new(Shape::Node(
                        Box::new(Shape::Leaf(5)),
                        Box::new(Shape::Leaf(6)),
                    )),
                )),
            ),
            6,
        );
        let t2 = make_tree(
            Shape::Node(
                Box::new(Shape::Node(
                    Box::new(Shape::Leaf(1)),
                    Box::new(Shape::Node(
                        Box::new(Shape::Leaf(2)),
                        Box::new(Shape::Leaf(3)),
                    )),
                )),
                Box::new(Shape::Node(
                    Box::new(Shape::Node(
                        Box::new(Shape::Leaf(4)),
                        Box::new(Shape::Leaf(5)),
                    )),
                    Box::new(Shape::Leaf(6)),
                )),
            ),
            6,
        );
        Instance::new(vec![t1, t2], 6)
    }

    fn tree_signature(tree: &Tree, node: NodeId) -> String {
        if tree.is_leaf(node) {
            return tree.label[node as usize].to_string();
        }
        let (left, right) = tree.children(node).unwrap();
        format!(
            "({},{})",
            tree_signature(tree, left),
            tree_signature(tree, right)
        )
    }

    fn instance_signature(instance: &Instance) -> Vec<String> {
        instance
            .trees
            .iter()
            .map(|tree| tree_signature(tree, tree.root))
            .collect()
    }

    fn forest_from_groups(instance: &Instance, groups: &[Vec<u32>]) -> Vec<Tree> {
        groups
            .iter()
            .map(|group| {
                let mut keep = FixedBitSet::with_capacity(instance.num_leaves as usize + 1);
                for &lbl in group {
                    keep.insert(lbl as usize);
                }
                if group.len() == 1 {
                    Tree::singleton(group[0], instance.num_leaves)
                } else {
                    instance.trees[0].prune_to_leafset(&keep)
                }
            })
            .collect()
    }

    #[test]
    fn test_find_best_common_cluster() {
        let instance = make_cluster_instance();
        let cluster = find_best_common_cluster(&instance).expect("expected a common cluster");
        let labels: Vec<_> = cluster.leaves.ones().collect();
        assert_eq!(labels, vec![1, 2, 3]);
    }

    #[test]
    fn test_cluster_reduction_k_minus_one_case() {
        let instance = make_cluster_instance();
        let cluster = find_best_common_cluster(&instance).unwrap();
        let subproblems = build_cluster_subproblems(&instance, &cluster);

        let tp1_sig = instance_signature(&subproblems.tp1.instance);
        let tp2_sig = instance_signature(&subproblems.tp2.instance);
        let tp3_sig = instance_signature(&subproblems.tp3.instance);
        let tp4_sig = instance_signature(&subproblems.tp4.instance);

        let result = try_cluster_reduction(&instance, &mut |subinstance| {
            let sig = instance_signature(subinstance);
            if sig == tp1_sig {
                Some(forest_from_groups(subinstance, &[vec![1, 2, 3, 4]]))
            } else if sig == tp2_sig {
                Some(forest_from_groups(subinstance, &[vec![1, 2, 3]]))
            } else if sig == tp3_sig {
                Some(forest_from_groups(subinstance, &[vec![1, 2, 3, 4]]))
            } else if sig == tp4_sig {
                Some(forest_from_groups(subinstance, &[vec![1, 2, 3]]))
            } else {
                None
            }
        })
        .unwrap();

        let ClusterReductionResult::Solved(solution) = result else {
            panic!("expected solved reduction");
        };
        assert_eq!(solution.components.len(), 1);
        let labels: Vec<_> = solution.components[0].leaves().collect();
        assert_eq!(labels, vec![1, 2, 3, 4, 5, 6]);
    }
}
