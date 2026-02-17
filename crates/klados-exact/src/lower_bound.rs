//! Bound computation for MAF via the Red-Blue 2-approximation algorithm
//! (Olver, Schalekamp, van der Ster, Stougie, van Zuylen, 2018) for lower
//! bounds, and cherry-based heuristic for upper bounds.
//!
//! The 2-approximation guarantees: true_dist <= approx_cost <= 2 * true_dist.
//! This gives a valid lower bound: OPT >= ceil(approx_cost / 2).
//!
//! The cherry-picking heuristic gives a valid upper bound on rSPR distance
//! (but NOT a proven approximation — its ratio can exceed 3x).

use fixedbitset::FixedBitSet;
use klados_core::tree::{Label, NodeId, Tree, NONE};

/// Bounds on the optimal number of MAF components.
pub struct MafBounds {
    /// Lower bound on optimal component count (safe to start iterative
    /// deepening here).
    pub lower: usize,
    /// Upper bound on optimal component count (can stop iterative deepening
    /// here). Only tight for 2-tree instances.
    pub upper: usize,
}

/// Compute bounds on the optimal number of MAF components.
pub fn maf_bounds(trees: &[Tree], num_leaves: u32) -> MafBounds {
    if trees.len() <= 1 {
        return MafBounds { lower: 1, upper: 1 };
    }

    let m = trees.len();
    let mut best_lb = 1usize;
    let mut best_ub_pair = usize::MAX;

    for i in 0..m {
        for j in (i + 1)..m {
            // 2-approximation: OPT <= approx_2 <= 2*OPT
            // Used for BOTH lower bound (ceil(approx_2/2)) and upper bound (approx_2 itself)
            let approx_2 = red_blue_approx(&trees[i], &trees[j]);
            let lb_cost = (approx_2 + 1) / 2; // ceil(approx_2 / 2)
            let lb_components = lb_cost + 1;
            if lb_components > best_lb {
                best_lb = lb_components;
            }
            // The 2-approx value is a valid upper bound on the pairwise rSPR distance
            let approx_ub_components = approx_2 + 1;
            if approx_ub_components < best_ub_pair {
                best_ub_pair = approx_ub_components;
            }

            // Cherry reduction upper bound (often looser, but sometimes tighter)
            let cherry_ub = cherry_reduce_ub(&trees[i], &trees[j]);
            let ub_components = cherry_ub + 1;
            if ub_components < best_ub_pair {
                best_ub_pair = ub_components;
            }
        }
    }

    // Multi-tree additive lower bound using 2-approx pairwise distances
    if m >= 3 {
        let mut pairwise = vec![vec![0usize; m]; m];
        for i in 0..m {
            for j in (i + 1)..m {
                let a = red_blue_approx(&trees[i], &trees[j]);
                pairwise[i][j] = a;
                pairwise[j][i] = a;
            }
        }
        for i in 0..m {
            let sum_d: usize = pairwise[i].iter().sum();
            let denom = 2 * (m - 1);
            let lb_cuts = (sum_d + denom - 1) / denom;
            let lb_components = lb_cuts + 1;
            if lb_components > best_lb {
                best_lb = lb_components;
            }
        }
    }

    let upper = if trees.len() == 2 {
        best_ub_pair.min(num_leaves as usize)
    } else {
        let mut best_multi_ub = num_leaves as usize;
        for ref_idx in 0..m {
            let ub = greedy_multi_tree_ub(trees, ref_idx);
            if ub < best_multi_ub {
                best_multi_ub = ub;
            }
        }
        best_multi_ub.min(num_leaves as usize)
    };

    MafBounds {
        lower: best_lb.min(upper),
        upper,
    }
}

/// Compute a lower bound on the optimal number of MAF components.
pub fn lower_bound_components(trees: &[Tree]) -> usize {
    if trees.len() <= 1 {
        return 1;
    }
    maf_bounds(trees, trees[0].num_leaves).lower
}

/// Public wrapper for pairwise approximate rSPR distance (cherry upper bound).
pub fn approx_rspr_distance_pub(t1: &Tree, t2: &Tree) -> usize {
    cherry_reduce_ub(t1, t2)
}

/// Public wrapper for the 2-approximation.
pub fn red_blue_approx_pub(t1: &Tree, t2: &Tree) -> usize {
    red_blue_approx(t1, t2)
}

// ============================================================================
// Red-Blue 2-Approximation Algorithm — optimized implementation
// ============================================================================

/// Precomputed data for a tree used in the Red-Blue algorithm.
struct TreeData {
    /// The tree
    tree: Tree,
    /// leaf_set[v] = bitset of leaf labels that are descendants of node v
    leaf_set: Vec<FixedBitSet>,
    /// Euler tour data for O(1) LCA
    euler: Vec<NodeId>,
    euler_depth: Vec<u16>,
    first_occ: Vec<u32>,
    sparse: Vec<Vec<u32>>,
    /// Post-order traversal of the tree
    post_order: Vec<NodeId>,
}

impl TreeData {
    fn build(tree: &Tree) -> Self {
        let n = tree.num_nodes();
        let nl = tree.num_leaves as usize;

        // Build leaf sets via post-order
        let post = tree.post_order_vec();
        let mut leaf_set = vec![FixedBitSet::with_capacity(nl + 1); n];
        for &node in &post {
            if tree.is_leaf(node) {
                let lbl = tree.label[node as usize];
                if lbl > 0 {
                    leaf_set[node as usize].insert(lbl as usize);
                }
            } else if let Some((l, r)) = tree.children(node) {
                // Union of children — clone right, then union left into it
                let mut combined = leaf_set[l as usize].clone();
                combined.union_with(&leaf_set[r as usize]);
                leaf_set[node as usize] = combined;
            }
        }

        // Build Euler tour for LCA
        let tour_cap = 2 * n;
        let mut euler = Vec::with_capacity(tour_cap);
        let mut euler_depth = Vec::with_capacity(tour_cap);
        let mut first_occ = vec![u32::MAX; n];

        let mut stack: Vec<(NodeId, bool)> = vec![(tree.root, false)];
        while let Some((node, returning)) = stack.pop() {
            let pos = euler.len() as u32;
            euler.push(node);
            euler_depth.push(tree.depth[node as usize]);
            if first_occ[node as usize] == u32::MAX {
                first_occ[node as usize] = pos;
            }
            if !returning {
                if let Some((left, right)) = tree.children(node) {
                    stack.push((node, true));
                    stack.push((right, false));
                    stack.push((node, true));
                    stack.push((left, false));
                }
            }
        }

        // Sparse table for RMQ
        let len = euler.len();
        let log_len = if len > 1 {
            (usize::BITS - (len - 1).leading_zeros()) as usize + 1
        } else {
            1
        };
        let mut sparse = Vec::with_capacity(log_len);
        let level0: Vec<u32> = (0..len as u32).collect();
        sparse.push(level0);

        for k in 1..log_len {
            let half = 1usize << (k - 1);
            let prev = &sparse[k - 1];
            let level_len = len.saturating_sub((1usize << k) - 1);
            let mut level = Vec::with_capacity(level_len);
            for i in 0..level_len {
                let a = prev[i];
                let b = prev[i + half];
                if euler_depth[a as usize] <= euler_depth[b as usize] {
                    level.push(a);
                } else {
                    level.push(b);
                }
            }
            sparse.push(level);
        }

        TreeData {
            tree: tree.clone(),
            leaf_set,
            euler,
            euler_depth,
            first_occ,
            sparse,
            post_order: post,
        }
    }

    /// O(1) LCA query.
    #[inline]
    fn lca(&self, u: NodeId, v: NodeId) -> NodeId {
        if u == NONE || v == NONE {
            return NONE;
        }
        if u == v {
            return u;
        }
        let mut l = self.first_occ[u as usize] as usize;
        let mut r = self.first_occ[v as usize] as usize;
        if l > r {
            std::mem::swap(&mut l, &mut r);
        }
        let len = r - l + 1;
        if len == 1 {
            return self.euler[l];
        }
        let k = (usize::BITS - len.leading_zeros() - 1) as usize;
        let a = self.sparse[k][l];
        let b = self.sparse[k][r + 1 - (1 << k)];
        if self.euler_depth[a as usize] <= self.euler_depth[b as usize] {
            self.euler[a as usize]
        } else {
            self.euler[b as usize]
        }
    }

    /// LCA of a set of labels. Returns NONE if the set is empty or
    /// all labels map to NONE.
    fn lca_of_labels(&self, labels: &FixedBitSet) -> NodeId {
        let mut result = NONE;
        for lbl in labels.ones() {
            if lbl >= self.tree.label_to_node.len() {
                continue;
            }
            let node = self.tree.label_to_node[lbl];
            if node == NONE {
                continue;
            }
            if result == NONE {
                result = node;
            } else {
                result = self.lca(result, node);
            }
        }
        result
    }

    /// Check if u is an ancestor of v (or equal) using depth + LCA.
    #[inline]
    fn is_ancestor_or_eq(&self, u: NodeId, v: NodeId) -> bool {
        self.lca(u, v) == u
    }
}

/// Partition of labels 1..=n into components.
/// Uses a flat array: comp[label] = component_id.
/// Also maintains reverse mapping: members[comp_id] = bitset of labels.
struct Partition {
    comp: Vec<u32>,            // comp[label] = component_id (indexed 0..=n, 0 unused)
    members: Vec<FixedBitSet>, // members[comp_id] = set of labels
    n: u32,
}

impl Partition {
    fn new_single(n: u32) -> Self {
        let mut all = FixedBitSet::with_capacity(n as usize + 1);
        for i in 1..=n as usize {
            all.insert(i);
        }
        Partition {
            comp: vec![0; n as usize + 1],
            members: vec![all],
            n,
        }
    }

    #[inline]
    fn component_of(&self, label: Label) -> u32 {
        self.comp[label as usize]
    }

    /// Split off a subset from its current component into a new component.
    /// All labels in `subset` must currently belong to the same component.
    fn split_off(&mut self, subset: &FixedBitSet) -> u32 {
        if subset.count_ones(..) == 0 {
            return u32::MAX;
        }
        let new_id = self.members.len() as u32;
        let mut new_set = FixedBitSet::with_capacity(self.n as usize + 1);

        let old_id = self.comp[subset.ones().next().unwrap()];

        for lbl in subset.ones() {
            debug_assert_eq!(self.comp[lbl], old_id);
            self.comp[lbl] = new_id;
            self.members[old_id as usize].set(lbl, false);
            new_set.insert(lbl);
        }
        self.members.push(new_set);
        new_id
    }

    /// Merge two components.
    fn merge(&mut self, keep: u32, remove: u32) {
        if keep == remove {
            return;
        }
        let remove_members: Vec<usize> = self.members[remove as usize].ones().collect();
        for lbl in remove_members {
            self.comp[lbl] = keep;
            self.members[keep as usize].insert(lbl);
            self.members[remove as usize].set(lbl, false);
        }
    }

    /// Count non-empty components.
    fn count_components(&self) -> usize {
        self.members.iter().filter(|m| m.count_ones(..) > 0).count()
    }

    /// Iterate over non-empty component IDs.
    fn active_component_ids(&self) -> Vec<u32> {
        (0..self.members.len() as u32)
            .filter(|&id| self.members[id as usize].count_ones(..) > 0)
            .collect()
    }
}

/// Run the Red-Blue 2-approximation on two trees.
/// Returns the MAF cost (|P| - 1).
fn red_blue_approx(t1: &Tree, t2: &Tree) -> usize {
    let n = t1.num_leaves;
    if n <= 1 {
        return 0;
    }

    let td1 = TreeData::build(t1);
    let td2 = TreeData::build(t2);
    let mut partition = Partition::new_single(n);
    let mut pairslist: Vec<(Label, Label)> = Vec::new();

    let max_iterations = 4 * n as usize; // each iteration creates ≥1 new component, max O(n)
    for _ in 0..max_iterations {
        // Find lowest root-of-infeasibility
        let u = match find_lowest_roi(&td1, &td2, &partition) {
            Some(u) => u,
            None => break,
        };

        let (u_l, u_r) = t1.children(u).unwrap();

        // Color: red = L(u_r), blue = L(u_l), white = rest
        let red = &td1.leaf_set[u_r as usize];
        let blue = &td1.leaf_set[u_l as usize];
        // white = everything not in red or blue
        let mut white = FixedBitSet::with_capacity(n as usize + 1);
        for lbl in 1..=n as usize {
            if !red.contains(lbl) && !blue.contains(lbl) {
                white.insert(lbl);
            }
        }

        // Save original partition for Find-Merge-Pair
        let original_comp = partition.comp.clone();

        // Step 1: Make-R∪B-compatible
        make_rub_compatible(&td2, &mut partition, red, blue);

        // Step 2: Make-Splittable
        make_splittable(&td2, &mut partition, red, blue, &white);

        // Step 3: Split
        split_procedure(&td1, &td2, &mut partition, red, blue, &white);

        // Step 4: Find-Merge-Pair
        if let Some(pair) = find_merge_pair(&td2, &partition, red, blue, &original_comp) {
            pairslist.push(pair);
        }
    }

    // Merge-Components
    for &(x1, x2) in &pairslist {
        let c1 = partition.component_of(x1);
        let c2 = partition.component_of(x2);
        partition.merge(c1, c2);
    }

    let nc = partition.count_components();
    if nc == 0 {
        0
    } else {
        nc - 1
    }
}

/// Check if a triple (x1,x2,x3) is compatible across two trees.
#[inline]
fn is_triple_compatible(td1: &TreeData, td2: &TreeData, x1: Label, x2: Label, x3: Label) -> bool {
    let n1 = td1.tree.label_to_node[x1 as usize];
    let n2 = td1.tree.label_to_node[x2 as usize];
    let n3 = td1.tree.label_to_node[x3 as usize];

    let m1 = td2.tree.label_to_node[x1 as usize];
    let m2 = td2.tree.label_to_node[x2 as usize];
    let m3 = td2.tree.label_to_node[x3 as usize];

    // Check all three pairings: (x1,x2), (x1,x3), (x2,x3)
    // For each, check if lca(pair) < lca(all three) matches between trees

    // Pair (x1, x2)
    let lca12_1 = td1.lca(n1, n2);
    let lca123_1 = td1.lca(lca12_1, n3);
    let strict12_1 = lca12_1 != lca123_1;

    let lca12_2 = td2.lca(m1, m2);
    let lca123_2 = td2.lca(lca12_2, m3);
    let strict12_2 = lca12_2 != lca123_2;

    if strict12_1 != strict12_2 {
        return false;
    }

    // Pair (x1, x3)
    let lca13_1 = td1.lca(n1, n3);
    let lca123_1b = td1.lca(lca13_1, n2);
    let strict13_1 = lca13_1 != lca123_1b;

    let lca13_2 = td2.lca(m1, m3);
    let lca123_2b = td2.lca(lca13_2, m2);
    let strict13_2 = lca13_2 != lca123_2b;

    if strict13_1 != strict13_2 {
        return false;
    }

    // Pair (x2, x3)
    let lca23_1 = td1.lca(n2, n3);
    let lca123_1c = td1.lca(lca23_1, n1);
    let strict23_1 = lca23_1 != lca123_1c;

    let lca23_2 = td2.lca(m2, m3);
    let lca123_2c = td2.lca(lca23_2, m1);
    let strict23_2 = lca23_2 != lca123_2c;

    strict23_1 == strict23_2
}

/// Check if a set of labels (given as a bitset) is compatible.
fn is_set_compatible(td1: &TreeData, td2: &TreeData, labels: &FixedBitSet) -> bool {
    let lbls: Vec<usize> = labels.ones().collect();
    if lbls.len() <= 2 {
        return true;
    }
    for i in 0..lbls.len() {
        for j in (i + 1)..lbls.len() {
            for k in (j + 1)..lbls.len() {
                if !is_triple_compatible(
                    td1,
                    td2,
                    lbls[i] as Label,
                    lbls[j] as Label,
                    lbls[k] as Label,
                ) {
                    return false;
                }
            }
        }
    }
    true
}

/// Mark all nodes on paths from each leaf in `labels` to `lca_node` in tree `td`.
/// Returns a bitset over node IDs.
fn mark_v_set(td: &TreeData, labels: &FixedBitSet, lca_node: NodeId) -> FixedBitSet {
    let num_nodes = td.tree.num_nodes();
    let mut marked = FixedBitSet::with_capacity(num_nodes);
    if lca_node == NONE {
        return marked;
    }
    for lbl in labels.ones() {
        if lbl >= td.tree.label_to_node.len() {
            continue;
        }
        let mut cur = td.tree.label_to_node[lbl];
        if cur == NONE {
            continue;
        }
        loop {
            if marked.contains(cur as usize) {
                break;
            }
            marked.insert(cur as usize);
            if cur == lca_node {
                break;
            }
            let p = td.tree.parent[cur as usize];
            if p == NONE {
                break;
            }
            cur = p;
        }
    }
    // Ensure lca_node is marked
    if lca_node != NONE {
        marked.insert(lca_node as usize);
    }
    marked
}

/// Check if the partition overlaps in V2 (two components share a node in T2).
fn partition_overlaps_in_v2(td2: &TreeData, partition: &Partition) -> bool {
    let num_nodes = td2.tree.num_nodes();
    // For each node, track which component covers it
    let mut cover: Vec<u32> = vec![u32::MAX; num_nodes];

    for cid in partition.active_component_ids() {
        let labels = &partition.members[cid as usize];
        if labels.count_ones(..) < 2 {
            continue;
        }
        let lca_node = td2.lca_of_labels(labels);
        if lca_node == NONE {
            continue;
        }
        for lbl in labels.ones() {
            if lbl >= td2.tree.label_to_node.len() {
                continue;
            }
            let mut cur = td2.tree.label_to_node[lbl];
            if cur == NONE {
                continue;
            }
            loop {
                if cover[cur as usize] == cid {
                    break;
                }
                if cover[cur as usize] != u32::MAX && cover[cur as usize] != cid {
                    return true;
                }
                cover[cur as usize] = cid;
                if cur == lca_node {
                    break;
                }
                let p = td2.tree.parent[cur as usize];
                if p == NONE {
                    break;
                }
                cur = p;
            }
        }
    }
    false
}

/// Check Definition 4: is u a root-of-infeasibility?
fn is_roi(td1: &TreeData, td2: &TreeData, partition: &Partition, u: NodeId) -> bool {
    if td1.tree.is_leaf(u) {
        return false;
    }
    let u_leaves = &td1.leaf_set[u as usize];

    // (a) P is not L(u)-compatible
    for cid in partition.active_component_ids() {
        let labels = &partition.members[cid as usize];
        // Intersection with u_leaves
        let mut intersection = labels.clone();
        intersection.intersect_with(u_leaves);
        if intersection.count_ones(..) >= 3 && !is_set_compatible(td1, td2, &intersection) {
            return true;
        }
    }

    // (b) P overlaps in V1[L(u)]
    {
        let num_nodes = td1.tree.num_nodes();
        let mut cover: Vec<u32> = vec![u32::MAX; num_nodes];
        for cid in partition.active_component_ids() {
            let labels = &partition.members[cid as usize];
            // Only care about labels inside L(u)
            let mut inside = labels.clone();
            inside.intersect_with(u_leaves);
            if inside.count_ones(..) < 2 {
                continue;
            }
            let lca_node = td1.lca_of_labels(&inside);
            if lca_node == NONE {
                continue;
            }
            for lbl in inside.ones() {
                if lbl >= td1.tree.label_to_node.len() {
                    continue;
                }
                let mut cur = td1.tree.label_to_node[lbl];
                if cur == NONE {
                    continue;
                }
                loop {
                    if cover[cur as usize] == cid {
                        break;
                    }
                    if cover[cur as usize] != u32::MAX && cover[cur as usize] != cid {
                        return true;
                    }
                    cover[cur as usize] = cid;
                    if cur == lca_node {
                        break;
                    }
                    let p = td1.tree.parent[cur as usize];
                    if p == NONE {
                        break;
                    }
                    cur = p;
                }
            }
        }
    }

    // (c) exists component A with A\L(u)!=∅, and for all w in A\L(u),
    // A∩L(u)∪{w} is not compatible
    for cid in partition.active_component_ids() {
        let labels = &partition.members[cid as usize];
        let mut inside = labels.clone();
        inside.intersect_with(u_leaves);
        if inside.count_ones(..) == 0 {
            continue;
        }
        let mut outside = labels.clone();
        // remove u_leaves
        for lbl in u_leaves.ones() {
            outside.set(lbl, false);
        }
        if outside.count_ones(..) == 0 {
            continue;
        }
        // Check: for ALL w in outside, inside ∪ {w} is not compatible
        let all_incompatible = outside.ones().all(|w| {
            let mut test = inside.clone();
            test.insert(w);
            !is_set_compatible(td1, td2, &test)
        });
        if all_incompatible {
            return true;
        }
    }

    false
}

/// Find the lowest root-of-infeasibility in T1.
fn find_lowest_roi(td1: &TreeData, td2: &TreeData, partition: &Partition) -> Option<NodeId> {
    // Post-order: check from bottom up
    for &node in &td1.post_order {
        if td1.tree.is_leaf(node) {
            continue;
        }
        if is_roi(td1, td2, partition, node) {
            // Verify children are NOT ROIs
            let (left, right) = td1.tree.children(node).unwrap();
            let left_roi = !td1.tree.is_leaf(left) && is_roi(td1, td2, partition, left);
            let right_roi = !td1.tree.is_leaf(right) && is_roi(td1, td2, partition, right);
            if !left_roi && !right_roi {
                return Some(node);
            }
        }
    }
    None
}

/// Make-R∪B-compatible
fn make_rub_compatible(
    td2: &TreeData,
    partition: &mut Partition,
    red: &FixedBitSet,
    blue: &FixedBitSet,
) {
    let mut rub = red.clone();
    rub.union_with(blue);

    loop {
        let mut split_done = false;
        for cid in partition.active_component_ids() {
            let labels = &partition.members[cid as usize];
            let mut a_rub = labels.clone();
            a_rub.intersect_with(&rub);
            if a_rub.count_ones(..) < 2 {
                continue;
            }

            // Check R∪B-compatible: is a_rub compatible?
            // Actually, we need to check if the R∪B intersection of this component is compatible
            // But this is expensive for large sets. We can just check: is there any
            // internal node in T2 where A∩L(û) intersects both R and B?
            // The procedure says: find lowest û in V2[A] where A∩L(û) intersects both R and B
            if let Some(u_hat) = find_rub_split_node(td2, labels, red, blue) {
                let u_hat_leaves = &td2.leaf_set[u_hat as usize];
                let mut inside = labels.clone();
                inside.intersect_with(u_hat_leaves);
                if inside.count_ones(..) > 0 && inside.count_ones(..) < labels.count_ones(..) {
                    partition.split_off(&inside);
                    split_done = true;
                    break;
                }
            }
        }
        if !split_done {
            break;
        }
    }
}

/// Find lowest node û in V2[A] where A∩L(û) intersects both R and B.
fn find_rub_split_node(
    td2: &TreeData,
    comp_labels: &FixedBitSet,
    red: &FixedBitSet,
    blue: &FixedBitSet,
) -> Option<NodeId> {
    if comp_labels.count_ones(..) < 2 {
        return None;
    }
    let comp_lca = td2.lca_of_labels(comp_labels);
    if comp_lca == NONE {
        return None;
    }

    // Find the deepest internal node v in T2's subtree of comp_lca where
    // comp ∩ L(v) intersects both R and B.
    let mut best: Option<NodeId> = None;
    let mut best_depth: u16 = 0;

    // Walk internal nodes in T2 subtree
    let mut stack = vec![comp_lca];
    while let Some(v) = stack.pop() {
        if td2.tree.is_leaf(v) {
            continue;
        }
        // comp ∩ L(v)
        let mut cv = td2.leaf_set[v as usize].clone();
        cv.intersect_with(comp_labels);

        let mut has_red = false;
        let mut has_blue = false;
        for lbl in cv.ones() {
            if red.contains(lbl) {
                has_red = true;
            }
            if blue.contains(lbl) {
                has_blue = true;
            }
            if has_red && has_blue {
                break;
            }
        }

        if has_red && has_blue {
            let d = td2.tree.depth[v as usize];
            if best.is_none() || d > best_depth {
                best = Some(v);
                best_depth = d;
            }
        }

        if let Some((left, right)) = td2.tree.children(v) {
            // Only descend if the child's subtree contains comp labels
            if !td2.leaf_set[left as usize].is_disjoint(comp_labels) {
                stack.push(left);
            }
            if !td2.leaf_set[right as usize].is_disjoint(comp_labels) {
                stack.push(right);
            }
        }
    }
    best
}

/// Make-Splittable
fn make_splittable(
    td2: &TreeData,
    partition: &mut Partition,
    red: &FixedBitSet,
    blue: &FixedBitSet,
    white: &FixedBitSet,
) {
    loop {
        let mut split_done = false;
        for cid in partition.active_component_ids() {
            let labels = &partition.members[cid as usize];
            if is_splittable(td2, labels, red, blue, white) {
                continue;
            }
            if let Some(u_hat) = find_splittable_split_node(td2, labels, red, blue, white) {
                let u_hat_leaves = &td2.leaf_set[u_hat as usize];
                let mut inside = labels.clone();
                inside.intersect_with(u_hat_leaves);
                if inside.count_ones(..) > 0 && inside.count_ones(..) < labels.count_ones(..) {
                    partition.split_off(&inside);
                    split_done = true;
                    break;
                }
            }
        }
        if !split_done {
            break;
        }
    }
}

/// Check if a component is splittable.
fn is_splittable(
    td2: &TreeData,
    labels: &FixedBitSet,
    red: &FixedBitSet,
    blue: &FixedBitSet,
    white: &FixedBitSet,
) -> bool {
    let mut a_r = labels.clone();
    a_r.intersect_with(red);
    let mut a_b = labels.clone();
    a_b.intersect_with(blue);
    let mut a_w = labels.clone();
    a_w.intersect_with(white);

    !sets_overlap_in_v2(td2, &a_r, &a_b)
        && !sets_overlap_in_v2(td2, &a_r, &a_w)
        && !sets_overlap_in_v2(td2, &a_b, &a_w)
}

/// Check if V2[S1] and V2[S2] overlap.
fn sets_overlap_in_v2(td2: &TreeData, s1: &FixedBitSet, s2: &FixedBitSet) -> bool {
    if s1.count_ones(..) < 2 || s2.count_ones(..) < 2 {
        return false;
    }
    let lca1 = td2.lca_of_labels(s1);
    let lca2 = td2.lca_of_labels(s2);
    if lca1 == NONE || lca2 == NONE {
        return false;
    }

    let v1_set = mark_v_set(td2, s1, lca1);
    let v2_set = mark_v_set(td2, s2, lca2);
    !v1_set.is_disjoint(&v2_set)
}

/// Find node û for Make-Splittable.
fn find_splittable_split_node(
    td2: &TreeData,
    labels: &FixedBitSet,
    red: &FixedBitSet,
    blue: &FixedBitSet,
    white: &FixedBitSet,
) -> Option<NodeId> {
    let mut a_r = labels.clone();
    a_r.intersect_with(red);
    let mut a_b = labels.clone();
    a_b.intersect_with(blue);
    let mut a_w = labels.clone();
    a_w.intersect_with(white);

    // Find overlapping pair and lowest node in their overlap
    let pairs: [(&FixedBitSet, &FixedBitSet); 3] = [(&a_r, &a_b), (&a_r, &a_w), (&a_b, &a_w)];
    for (s1, s2) in pairs {
        if s1.count_ones(..) >= 1 && s2.count_ones(..) >= 1 {
            let lca_s1 = td2.lca_of_labels(s1);
            let lca_s2 = td2.lca_of_labels(s2);
            if lca_s1 == NONE || lca_s2 == NONE {
                continue;
            }
            if s1.count_ones(..) >= 2 && s2.count_ones(..) >= 2 {
                let v1_set = mark_v_set(td2, s1, lca_s1);
                let v2_set = mark_v_set(td2, s2, lca_s2);
                if !v1_set.is_disjoint(&v2_set) {
                    // Find deepest node in the intersection
                    let mut best: Option<NodeId> = None;
                    let mut best_depth = 0u16;
                    let intersection = {
                        let mut r = v1_set;
                        r.intersect_with(&v2_set);
                        r
                    };
                    for node_idx in intersection.ones() {
                        let d = td2.tree.depth[node_idx];
                        if best.is_none() || d > best_depth {
                            best = Some(node_idx as NodeId);
                            best_depth = d;
                        }
                    }
                    return best;
                }
            }
        }
    }
    None
}

/// Split procedure
fn split_procedure(
    td1: &TreeData,
    td2: &TreeData,
    partition: &mut Partition,
    red: &FixedBitSet,
    blue: &FixedBitSet,
    white: &FixedBitSet,
) {
    let comp_ids = partition.active_component_ids();
    for cid in comp_ids {
        let labels = partition.members[cid as usize].clone();
        let mut a_r = labels.clone();
        a_r.intersect_with(red);
        let mut a_b = labels.clone();
        a_b.intersect_with(blue);
        let mut a_w = labels.clone();
        a_w.intersect_with(white);

        let num_colors = (a_r.count_ones(..) > 0) as u8
            + (a_b.count_ones(..) > 0) as u8
            + (a_w.count_ones(..) > 0) as u8;
        if num_colors <= 1 {
            continue;
        }

        let is_tricolored =
            a_r.count_ones(..) > 0 && a_b.count_ones(..) > 0 && a_w.count_ones(..) > 0;

        if is_tricolored && has_compatible_tricolored_triple(td1, td2, &a_r, &a_b, &a_w) {
            // Special-Split
            special_split(td1, td2, partition, cid, &a_r, &a_b, &a_w);
        } else {
            // Regular split: separate into R, B, W parts
            // Split off the smaller groups; keep whatever's left in cid
            // We need to split off at least the two smaller groups
            if a_r.count_ones(..) > 0 && (a_b.count_ones(..) > 0 || a_w.count_ones(..) > 0) {
                partition.split_off(&a_r);
            }
            if a_b.count_ones(..) > 0 && a_w.count_ones(..) > 0 {
                partition.split_off(&a_b);
            }
            // The remaining labels stay in cid
        }
    }
}

fn has_compatible_tricolored_triple(
    td1: &TreeData,
    td2: &TreeData,
    a_r: &FixedBitSet,
    a_b: &FixedBitSet,
    a_w: &FixedBitSet,
) -> bool {
    for r in a_r.ones() {
        for b in a_b.ones() {
            for w in a_w.ones() {
                if is_triple_compatible(td1, td2, r as Label, b as Label, w as Label) {
                    return true;
                }
            }
        }
    }
    false
}

fn special_split(
    td1: &TreeData,
    td2: &TreeData,
    partition: &mut Partition,
    _cid: u32,
    a_r: &FixedBitSet,
    a_b: &FixedBitSet,
    a_w: &FixedBitSet,
) {
    // Check if every tricolored triple is compatible
    let all_compatible = a_r.ones().all(|r| {
        a_b.ones().all(|b| {
            a_w.ones()
                .all(|w| is_triple_compatible(td1, td2, r as Label, b as Label, w as Label))
        })
    });

    if all_compatible {
        partition.split_off(a_r);
    } else {
        let mut rub = a_r.clone();
        rub.union_with(a_b);
        if rub.count_ones(..) == 0 {
            return;
        }
        let u_hat = td2.lca_of_labels(&rub);
        let u_hat_leaves = &td2.leaf_set[u_hat as usize];

        // A' = A ∩ L(û)
        let mut all_labels = a_r.clone();
        all_labels.union_with(a_b);
        all_labels.union_with(a_w);

        let mut a_prime = all_labels.clone();
        a_prime.intersect_with(u_hat_leaves);

        let mut a_outside = all_labels.clone();
        for lbl in a_prime.ones() {
            a_outside.set(lbl, false);
        }

        if a_outside.count_ones(..) > 0 {
            partition.split_off(&a_outside);
        }

        let mut apr = a_prime.clone();
        apr.intersect_with(a_r);
        let mut apb = a_prime.clone();
        apb.intersect_with(a_b);

        if apr.count_ones(..) > 0 {
            partition.split_off(&apr);
        }
        if apb.count_ones(..) > 0 {
            partition.split_off(&apb);
        }
    }
}

/// Find-Merge-Pair
fn find_merge_pair(
    td2: &TreeData,
    partition: &Partition,
    red: &FixedBitSet,
    blue: &FixedBitSet,
    original_comp: &[u32],
) -> Option<(Label, Label)> {
    let mut rub = red.clone();
    rub.union_with(blue);
    let rub_labels: Vec<usize> = rub.ones().collect();

    for i in 0..rub_labels.len() {
        for j in (i + 1)..rub_labels.len() {
            let x1 = rub_labels[i] as Label;
            let x2 = rub_labels[j] as Label;

            if original_comp[x1 as usize] != original_comp[x2 as usize] {
                continue;
            }

            let c1 = partition.component_of(x1);
            let c2 = partition.component_of(x2);
            if c1 == c2 {
                continue;
            }

            // Check if merging would keep no overlap in V2
            // Quick check: merged component is compatible
            let mut merged = partition.members[c1 as usize].clone();
            merged.union_with(&partition.members[c2 as usize]);

            // Build test partition
            let mut test = Partition {
                comp: partition.comp.clone(),
                members: partition.members.clone(),
                n: partition.n,
            };
            test.merge(c1, c2);

            if !partition_overlaps_in_v2(td2, &test) {
                return Some((x1, x2));
            }
        }
    }
    None
}

// ============================================================================
// Cherry-based heuristic for upper bounds
// ============================================================================

fn cherry_reduce_ub(t1: &Tree, t2: &Tree) -> usize {
    let c1 = cherry_reduce(t1, t2);
    let c2 = cherry_reduce(t2, t1);
    c1.min(c2)
}

struct MutableTree {
    parent: Vec<NodeId>,
    left: Vec<NodeId>,
    right: Vec<NodeId>,
    label: Vec<Label>,
    label_to_node: Vec<NodeId>,
    alive: Vec<bool>,
    num_alive_leaves: u32,
    root: NodeId,
}

impl MutableTree {
    fn from_tree(t: &Tree) -> Self {
        Self {
            parent: t.parent.clone(),
            left: t.left.clone(),
            right: t.right.clone(),
            label: t.label.clone(),
            label_to_node: t.label_to_node.clone(),
            alive: vec![true; t.num_nodes()],
            num_alive_leaves: t.num_leaves,
            root: t.root,
        }
    }

    #[inline]
    fn is_leaf(&self, node: NodeId) -> bool {
        self.left[node as usize] == NONE
    }

    #[inline]
    fn sibling(&self, node: NodeId) -> NodeId {
        let p = self.parent[node as usize];
        let l = self.left[p as usize];
        if l == node {
            self.right[p as usize]
        } else {
            l
        }
    }

    fn is_cherry(&self, a: Label, b: Label) -> bool {
        let na = self.label_to_node[a as usize];
        let nb = self.label_to_node[b as usize];
        if na == NONE || nb == NONE {
            return false;
        }
        if !self.alive[na as usize] || !self.alive[nb as usize] {
            return false;
        }
        let pa = self.parent[na as usize];
        let pb = self.parent[nb as usize];
        pa != NONE && pa == pb
    }

    fn contract_cherry(&mut self, keep: Label, remove: Label) {
        let keep_node = self.label_to_node[keep as usize];
        let remove_node = self.label_to_node[remove as usize];
        let parent = self.parent[keep_node as usize];
        self.alive[keep_node as usize] = false;
        self.alive[remove_node as usize] = false;
        self.left[parent as usize] = NONE;
        self.right[parent as usize] = NONE;
        self.label[parent as usize] = keep;
        self.label_to_node[keep as usize] = parent;
        self.label_to_node[remove as usize] = NONE;
        self.num_alive_leaves -= 1;
    }

    fn cut_leaf(&mut self, lbl: Label) {
        let node = self.label_to_node[lbl as usize];
        if node == NONE || !self.alive[node as usize] {
            return;
        }
        let parent = self.parent[node as usize];
        if parent == NONE {
            self.alive[node as usize] = false;
            self.label_to_node[lbl as usize] = NONE;
            self.num_alive_leaves -= 1;
            return;
        }
        let sib = self.sibling(node);
        let gp = self.parent[parent as usize];
        self.alive[node as usize] = false;
        self.label_to_node[lbl as usize] = NONE;
        self.num_alive_leaves -= 1;
        self.alive[parent as usize] = false;
        self.parent[sib as usize] = gp;
        if gp == NONE {
            self.root = sib;
        } else if self.left[gp as usize] == parent {
            self.left[gp as usize] = sib;
        } else {
            self.right[gp as usize] = sib;
        }
    }

    fn find_cherries(&self) -> Vec<(Label, Label)> {
        let mut cherries = Vec::new();
        for i in 0..self.parent.len() {
            let node = i as NodeId;
            if !self.alive[node as usize] || self.is_leaf(node) {
                continue;
            }
            let l = self.left[node as usize];
            let r = self.right[node as usize];
            if l != NONE
                && r != NONE
                && self.alive[l as usize]
                && self.alive[r as usize]
                && self.is_leaf(l)
                && self.is_leaf(r)
            {
                let la = self.label[l as usize];
                let lb = self.label[r as usize];
                if la > 0 && lb > 0 {
                    cherries.push((la.min(lb), la.max(lb)));
                }
            }
        }
        cherries
    }
}

fn cherry_reduce(ref_tree: &Tree, other_tree: &Tree) -> usize {
    let mut m_ref = MutableTree::from_tree(ref_tree);
    let mut m_other = MutableTree::from_tree(other_tree);
    let mut cuts = 0;
    loop {
        if m_ref.num_alive_leaves <= 1 {
            break;
        }
        let cherries = m_ref.find_cherries();
        if cherries.is_empty() {
            break;
        }
        let (a, b) = cherries[0];
        if m_other.is_cherry(a, b) {
            m_ref.contract_cherry(a, b);
            m_other.contract_cherry(a, b);
        } else {
            let cut = pick_leaf_to_cut(&m_other, a, b);
            m_ref.cut_leaf(cut);
            m_other.cut_leaf(cut);
            cuts += 1;
        }
    }
    cuts
}

fn pick_leaf_to_cut(t2: &MutableTree, a: Label, b: Label) -> Label {
    let na = t2.label_to_node[a as usize];
    let nb = t2.label_to_node[b as usize];
    if na == NONE {
        return a;
    }
    if nb == NONE {
        return b;
    }
    let da = depth_in_mtree(t2, na);
    let db = depth_in_mtree(t2, nb);
    if da >= db {
        a
    } else {
        b
    }
}

fn depth_in_mtree(t: &MutableTree, mut node: NodeId) -> u32 {
    let mut d = 0;
    while t.parent[node as usize] != NONE {
        node = t.parent[node as usize];
        d += 1;
    }
    d
}

fn greedy_multi_tree_ub(trees: &[Tree], ref_idx: usize) -> usize {
    let mut mtrees: Vec<MutableTree> = trees.iter().map(|t| MutableTree::from_tree(t)).collect();
    let mut cuts = 0;
    loop {
        if mtrees[ref_idx].num_alive_leaves <= 1 {
            break;
        }
        let cherries = mtrees[ref_idx].find_cherries();
        if cherries.is_empty() {
            break;
        }
        let (a, b) = cherries[0];
        if mtrees.iter().all(|t| t.is_cherry(a, b)) {
            for t in &mut mtrees {
                t.contract_cherry(a, b);
            }
        } else {
            let cut = pick_multi_tree_cut(&mtrees, ref_idx, a, b);
            for t in &mut mtrees {
                t.cut_leaf(cut);
            }
            cuts += 1;
        }
    }
    cuts + 1
}

fn pick_multi_tree_cut(mtrees: &[MutableTree], ref_idx: usize, a: Label, b: Label) -> Label {
    let mut a_deeper = 0i32;
    let mut total_diff: i32 = 0;
    for (i, t) in mtrees.iter().enumerate() {
        if i == ref_idx {
            continue;
        }
        let na = t.label_to_node[a as usize];
        let nb = t.label_to_node[b as usize];
        if na == NONE || nb == NONE {
            continue;
        }
        let da = depth_in_mtree(t, na) as i32;
        let db = depth_in_mtree(t, nb) as i32;
        total_diff += da - db;
        if da > db {
            a_deeper += 1;
        } else if db > da {
            a_deeper -= 1;
        }
    }
    if a_deeper > 0 || (a_deeper == 0 && total_diff >= 0) {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tree_2leaves() -> (Tree, Tree) {
        let mut t1 = Tree::with_capacity(2);
        t1.parent.push(2);
        t1.left.push(NONE);
        t1.right.push(NONE);
        t1.label.push(1);
        t1.label_to_node[1] = 0;
        t1.parent.push(2);
        t1.left.push(NONE);
        t1.right.push(NONE);
        t1.label.push(2);
        t1.label_to_node[2] = 1;
        t1.parent.push(NONE);
        t1.left.push(0);
        t1.right.push(1);
        t1.label.push(0);
        t1.root = 2;
        t1.compute_metadata();
        (t1.clone(), t1)
    }

    fn make_3leaf_trees() -> (Tree, Tree) {
        // T1: ((1,2),3), T2: ((1,3),2)
        let mut t1 = Tree::with_capacity(3);
        t1.parent.push(3);
        t1.left.push(NONE);
        t1.right.push(NONE);
        t1.label.push(1);
        t1.label_to_node[1] = 0;
        t1.parent.push(3);
        t1.left.push(NONE);
        t1.right.push(NONE);
        t1.label.push(2);
        t1.label_to_node[2] = 1;
        t1.parent.push(4);
        t1.left.push(NONE);
        t1.right.push(NONE);
        t1.label.push(3);
        t1.label_to_node[3] = 2;
        t1.parent.push(4);
        t1.left.push(0);
        t1.right.push(1);
        t1.label.push(0);
        t1.parent.push(NONE);
        t1.left.push(3);
        t1.right.push(2);
        t1.label.push(0);
        t1.root = 4;
        t1.compute_metadata();

        let mut t2 = Tree::with_capacity(3);
        t2.parent.push(3);
        t2.left.push(NONE);
        t2.right.push(NONE);
        t2.label.push(1);
        t2.label_to_node[1] = 0;
        t2.parent.push(3);
        t2.left.push(NONE);
        t2.right.push(NONE);
        t2.label.push(3);
        t2.label_to_node[3] = 1;
        t2.parent.push(4);
        t2.left.push(NONE);
        t2.right.push(NONE);
        t2.label.push(2);
        t2.label_to_node[2] = 2;
        t2.parent.push(4);
        t2.left.push(0);
        t2.right.push(1);
        t2.label.push(0);
        t2.parent.push(NONE);
        t2.left.push(3);
        t2.right.push(2);
        t2.label.push(0);
        t2.root = 4;
        t2.compute_metadata();
        (t1, t2)
    }

    fn make_4leaf_trees() -> (Tree, Tree) {
        // T1: ((1,2),(3,4)), T2: ((1,3),(2,4))
        let mut t1 = Tree::with_capacity(4);
        t1.parent.push(4);
        t1.left.push(NONE);
        t1.right.push(NONE);
        t1.label.push(1);
        t1.label_to_node[1] = 0;
        t1.parent.push(4);
        t1.left.push(NONE);
        t1.right.push(NONE);
        t1.label.push(2);
        t1.label_to_node[2] = 1;
        t1.parent.push(5);
        t1.left.push(NONE);
        t1.right.push(NONE);
        t1.label.push(3);
        t1.label_to_node[3] = 2;
        t1.parent.push(5);
        t1.left.push(NONE);
        t1.right.push(NONE);
        t1.label.push(4);
        t1.label_to_node[4] = 3;
        t1.parent.push(6);
        t1.left.push(0);
        t1.right.push(1);
        t1.label.push(0);
        t1.parent.push(6);
        t1.left.push(2);
        t1.right.push(3);
        t1.label.push(0);
        t1.parent.push(NONE);
        t1.left.push(4);
        t1.right.push(5);
        t1.label.push(0);
        t1.root = 6;
        t1.compute_metadata();

        let mut t2 = Tree::with_capacity(4);
        t2.parent.push(4);
        t2.left.push(NONE);
        t2.right.push(NONE);
        t2.label.push(1);
        t2.label_to_node[1] = 0;
        t2.parent.push(4);
        t2.left.push(NONE);
        t2.right.push(NONE);
        t2.label.push(3);
        t2.label_to_node[3] = 1;
        t2.parent.push(5);
        t2.left.push(NONE);
        t2.right.push(NONE);
        t2.label.push(2);
        t2.label_to_node[2] = 2;
        t2.parent.push(5);
        t2.left.push(NONE);
        t2.right.push(NONE);
        t2.label.push(4);
        t2.label_to_node[4] = 3;
        t2.parent.push(6);
        t2.left.push(0);
        t2.right.push(1);
        t2.label.push(0);
        t2.parent.push(6);
        t2.left.push(2);
        t2.right.push(3);
        t2.label.push(0);
        t2.parent.push(NONE);
        t2.left.push(4);
        t2.right.push(5);
        t2.label.push(0);
        t2.root = 6;
        t2.compute_metadata();
        (t1, t2)
    }

    #[test]
    fn test_identical_trees() {
        let (t1, t2) = make_tree_2leaves();
        assert_eq!(cherry_reduce_ub(&t1, &t2), 0);
    }

    #[test]
    fn test_lower_bound_identical() {
        let (t1, t2) = make_tree_2leaves();
        assert_eq!(lower_bound_components(&[t1, t2]), 1);
    }

    #[test]
    fn test_red_blue_identical() {
        let (t1, t2) = make_tree_2leaves();
        assert_eq!(red_blue_approx(&t1, &t2), 0);
    }

    #[test]
    fn test_red_blue_3leaf() {
        let (t1, t2) = make_3leaf_trees();
        let cost = red_blue_approx(&t1, &t2);
        assert!(cost >= 1 && cost <= 2, "red_blue cost={}", cost);
    }

    #[test]
    fn test_red_blue_4leaf() {
        let (t1, t2) = make_4leaf_trees();
        let cost = red_blue_approx(&t1, &t2);
        assert!(cost >= 1 && cost <= 2, "red_blue cost={}", cost);
    }

    #[test]
    fn test_lower_bound_3leaf() {
        let (t1, t2) = make_3leaf_trees();
        let lb = lower_bound_components(&[t1, t2]);
        assert!(lb >= 1 && lb <= 2, "lb={}", lb);
    }

    #[test]
    fn test_single_tree() {
        let (t1, _) = make_3leaf_trees();
        assert_eq!(lower_bound_components(&[t1]), 1);
    }

    #[test]
    fn test_lca_basic() {
        let (t1, _) = make_3leaf_trees();
        let td = TreeData::build(&t1);
        assert_eq!(td.lca(0, 1), 3);
        assert_eq!(td.lca(0, 2), 4);
        assert_eq!(td.lca(1, 2), 4);
    }
}
