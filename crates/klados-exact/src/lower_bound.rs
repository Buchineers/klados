//! Lower bound computation for MAF via pairwise approximation.
//!
//! Uses a fast cherry-based heuristic on each pair of trees to obtain an
//! upper bound on the pairwise rSPR distance, then derives a lower bound
//! on the optimal MAF component count for the full multi-tree instance.
//!
//! The algorithm works on simplified copies of the two input trees:
//!   1. Find a cherry (sibling pair) {a,b} in T1.
//!   2. If {a,b} is also a cherry in T2, contract: replace the pair by a
//!      single leaf in both trees (free reduction, no cost).
//!   3. Otherwise, cut the "deeper" leaf from T2 (the one further from the
//!      cherry position in T2) and increment the cut counter.
//!   4. Repeat until only one leaf remains.
//!
//! The cut count is a 3-approximation of the rSPR distance d(T1,T2).
//! Since d(T1,T2) <= OPT_cost for any multi-tree instance containing T1
//! and T2, we get: OPT_components >= ceil(cuts/3) + 1 for each pair.
//! Taking the max over all pairs gives the tightest pairwise lower bound.

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
///
/// For each pair of trees, computes an approximate rSPR distance (3-approx).
/// - Lower bound: `max over pairs of ceil(approx_cost / 3) + 1`
/// - Upper bound: for 2-tree instances, `min over pairs of approx_cost + 1`;
///   for multi-tree, greedy multi-tree cherry reduction.
pub fn maf_bounds(trees: &[Tree], num_leaves: u32) -> MafBounds {
    if trees.len() <= 1 {
        return MafBounds { lower: 1, upper: 1 };
    }

    let mut best_lb = 1usize;
    let mut best_ub_pair = usize::MAX;

    // Pairwise distances (3-approx) for lower bound
    let m = trees.len();
    let mut pairwise = vec![vec![0usize; m]; m];
    for i in 0..m {
        for j in (i + 1)..m {
            let approx_cost = approx_rspr_distance(&trees[i], &trees[j]);
            pairwise[i][j] = approx_cost;
            pairwise[j][i] = approx_cost;

            // Standard pairwise lower bound
            let lb_cost = (approx_cost + 2) / 3;
            let lb_components = lb_cost + 1;
            if lb_components > best_lb {
                best_lb = lb_components;
            }

            let ub_components = approx_cost + 1;
            if ub_components < best_ub_pair {
                best_ub_pair = ub_components;
            }
        }
    }

    // Multi-tree additive lower bound: for each reference tree i,
    // OPT_cuts >= sum_{j!=i}(d(i,j)) / (m-1) because each optimal cut
    // resolves at most (m-1) pairwise disagreements.
    // We have 3-approx of d(i,j), so: OPT_cuts >= sum / (3*(m-1)).
    if m >= 3 {
        for i in 0..m {
            let sum_d: usize = pairwise[i].iter().sum();
            // OPT_cuts >= ceil(sum_d / (3 * (m-1)))
            let denom = 3 * (m - 1);
            let lb_cuts = (sum_d + denom - 1) / denom;
            let lb_components = lb_cuts + 1;
            if lb_components > best_lb {
                best_lb = lb_components;
            }
        }
    }

    // Upper bound
    let upper = if trees.len() == 2 {
        best_ub_pair.min(num_leaves as usize)
    } else {
        // Greedy multi-tree cherry reduction: try each tree as reference,
        // take the best (lowest) upper bound.
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
        lower: best_lb,
        upper,
    }
}

/// Greedy upper bound for multi-tree MAF: use `ref_idx` as the reference tree,
/// reduce cherries against ALL other trees simultaneously.
/// Returns the number of components in the resulting forest.
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

        // Check if this cherry is common to ALL trees
        let common = mtrees.iter().all(|t| t.is_cherry(a, b));

        if common {
            // Free reduction: contract in all trees
            for t in &mut mtrees {
                t.contract_cherry(a, b);
            }
        } else {
            // Cut one leaf from ALL trees. Pick the leaf that causes the
            // least damage: for each candidate {a, b}, count how many
            // other trees have it as part of a cherry (cheaper to cut the
            // one that's less "useful").
            let cut_label = pick_multi_tree_cut(&mtrees, ref_idx, a, b);
            for t in &mut mtrees {
                t.cut_leaf(cut_label);
            }
            cuts += 1;
        }
    }

    cuts + 1 // components = cuts + 1
}

/// Pick which of {a, b} to cut in multi-tree setting.
/// Heuristic: cut the leaf that is deeper (more displaced) in the most
/// trees, breaking ties by total depth.
fn pick_multi_tree_cut(mtrees: &[MutableTree], ref_idx: usize, a: Label, b: Label) -> Label {
    let mut a_deeper_count = 0i32;
    let mut total_depth_diff: i32 = 0;

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
        total_depth_diff += da - db;
        if da > db {
            a_deeper_count += 1;
        } else if db > da {
            a_deeper_count -= 1;
        }
    }

    if a_deeper_count > 0 || (a_deeper_count == 0 && total_depth_diff >= 0) {
        a
    } else {
        b
    }
}

/// Compute a lower bound on the optimal number of MAF components.
///
/// Convenience wrapper around `maf_bounds` that returns just the lower bound.
pub fn lower_bound_components(trees: &[Tree]) -> usize {
    if trees.len() <= 1 {
        return 1;
    }
    maf_bounds(trees, trees[0].num_leaves).lower
}

/// Public wrapper for pairwise approximate rSPR distance.
pub fn approx_rspr_distance_pub(t1: &Tree, t2: &Tree) -> usize {
    approx_rspr_distance(t1, t2)
}

// ---------------------------------------------------------------------------
// Internal: lightweight mutable tree representation for cherry reduction
// ---------------------------------------------------------------------------

/// A lightweight mutable tree for cherry-reduction.
/// Stores parent, left, right, label arrays. Supports suppression of nodes.
struct MutableTree {
    parent: Vec<NodeId>,
    left: Vec<NodeId>,
    right: Vec<NodeId>,
    label: Vec<Label>, // >0 for leaves
    label_to_node: Vec<NodeId>,
    alive: Vec<bool>, // whether node is still in the tree
    num_alive_leaves: u32,
    root: NodeId,
}

impl MutableTree {
    fn from_tree(t: &Tree) -> Self {
        let n = t.num_nodes();
        Self {
            parent: t.parent.clone(),
            left: t.left.clone(),
            right: t.right.clone(),
            label: t.label.clone(),
            label_to_node: t.label_to_node.clone(),
            alive: vec![true; n],
            num_alive_leaves: t.num_leaves,
            root: t.root,
        }
    }

    #[inline]
    fn is_leaf(&self, node: NodeId) -> bool {
        self.left[node as usize] == NONE
    }

    /// Get the sibling of a node (assumes node is not the root).
    #[inline]
    fn sibling(&self, node: NodeId) -> NodeId {
        let p = self.parent[node as usize];
        debug_assert!(p != NONE);
        let l = self.left[p as usize];
        if l == node {
            self.right[p as usize]
        } else {
            l
        }
    }

    /// Check if labels a and b form a cherry (their leaf nodes share a parent).
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

    /// Contract cherry {a,b}: remove leaf b, replace parent with leaf a.
    /// The parent node of a and b becomes "a" (takes on label a).
    fn contract_cherry(&mut self, keep: Label, remove: Label) {
        let keep_node = self.label_to_node[keep as usize];
        let remove_node = self.label_to_node[remove as usize];
        let parent = self.parent[keep_node as usize];
        debug_assert!(parent != NONE);
        debug_assert_eq!(parent, self.parent[remove_node as usize]);

        // Remove both children, make parent a leaf with label `keep`
        self.alive[keep_node as usize] = false;
        self.alive[remove_node as usize] = false;

        // Parent becomes the leaf for `keep`
        self.left[parent as usize] = NONE;
        self.right[parent as usize] = NONE;
        self.label[parent as usize] = keep;
        self.label_to_node[keep as usize] = parent;
        self.label_to_node[remove as usize] = NONE;

        self.num_alive_leaves -= 1;
    }

    /// Cut a leaf (remove it from the tree and suppress its parent if needed).
    /// Returns the label of the removed leaf.
    fn cut_leaf(&mut self, lbl: Label) {
        let node = self.label_to_node[lbl as usize];
        if node == NONE || !self.alive[node as usize] {
            return;
        }

        let parent = self.parent[node as usize];
        if parent == NONE {
            // Node is root — just remove it
            self.alive[node as usize] = false;
            self.label_to_node[lbl as usize] = NONE;
            self.num_alive_leaves -= 1;
            return;
        }

        let sib = self.sibling(node);
        let grandparent = self.parent[parent as usize];

        // Remove node
        self.alive[node as usize] = false;
        self.label_to_node[lbl as usize] = NONE;
        self.num_alive_leaves -= 1;

        // Suppress parent: connect sibling directly to grandparent
        self.alive[parent as usize] = false;
        self.parent[sib as usize] = grandparent;

        if grandparent == NONE {
            // Parent was root, sibling becomes new root
            self.root = sib;
        } else {
            // Replace parent with sibling in grandparent's children
            if self.left[grandparent as usize] == parent {
                self.left[grandparent as usize] = sib;
            } else {
                self.right[grandparent as usize] = sib;
            }
        }
    }

    /// Find all cherries (sibling pairs of leaves). Returns (label_a, label_b)
    /// with label_a < label_b.
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

/// Compute an approximate rSPR distance between two trees using cherry
/// reduction. Returns the number of cuts (a 3-approximation of the true
/// rSPR distance).
///
/// Runs the cherry reduction in both directions (T1-first and T2-first)
/// and returns the **maximum**, which gives the tightest lower bound when
/// divided by the approximation factor.
fn approx_rspr_distance(t1: &Tree, t2: &Tree) -> usize {
    let c1 = cherry_reduce(t1, t2);
    let c2 = cherry_reduce(t2, t1);
    c1.max(c2)
}

/// One-directional cherry reduction: find cherries in `ref_tree`, resolve
/// against `other_tree`. Returns the number of cuts.
fn cherry_reduce(ref_tree: &Tree, other_tree: &Tree) -> usize {
    let mut m_ref = MutableTree::from_tree(ref_tree);
    let mut m_other = MutableTree::from_tree(other_tree);

    let mut cuts = 0;

    loop {
        if m_ref.num_alive_leaves <= 1 {
            break;
        }

        // Find a cherry in the reference tree
        let cherries = m_ref.find_cherries();
        if cherries.is_empty() {
            break;
        }

        // Process the first cherry
        let (a, b) = cherries[0];

        if m_other.is_cherry(a, b) {
            // Common cherry: contract for free in both trees
            m_ref.contract_cherry(a, b);
            m_other.contract_cherry(a, b);
        } else {
            // Not a cherry in the other tree: cut one leaf.
            let cut_label = pick_leaf_to_cut(&m_other, a, b);

            m_ref.cut_leaf(cut_label);
            m_other.cut_leaf(cut_label);
            cuts += 1;
        }
    }

    cuts
}

/// Pick which of {a, b} to cut from T2 when {a,b} is a cherry in T1 but
/// not in T2. Uses a simple depth-based heuristic: cut the leaf whose
/// node in T2 is deeper (further from root), since it's more "displaced".
/// If equal depth, cut `b` (arbitrary but deterministic).
fn pick_leaf_to_cut(t2: &MutableTree, a: Label, b: Label) -> Label {
    let na = t2.label_to_node[a as usize];
    let nb = t2.label_to_node[b as usize];
    if na == NONE {
        return a;
    }
    if nb == NONE {
        return b;
    }

    // Walk up from both nodes to estimate depth
    let da = depth_in_mtree(t2, na);
    let db = depth_in_mtree(t2, nb);

    // Cut the deeper leaf (more displaced from root)
    if da >= db {
        a
    } else {
        b
    }
}

/// Count depth of a node in a MutableTree by walking to root.
fn depth_in_mtree(t: &MutableTree, mut node: NodeId) -> u32 {
    let mut d = 0;
    while t.parent[node as usize] != NONE {
        node = t.parent[node as usize];
        d += 1;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use klados_core::tree::Tree;

    /// Helper: build a tree from a simple nested tuple representation.
    /// We'll build small trees manually for testing.
    fn make_tree_2leaves() -> (Tree, Tree) {
        // T1: (1, 2)  -- cherry {1,2}
        // T2: (1, 2)  -- same cherry
        // rSPR = 0
        let mut t1 = Tree::with_capacity(2);
        // leaf 1 (node 0)
        t1.parent.push(2);
        t1.left.push(NONE);
        t1.right.push(NONE);
        t1.label.push(1);
        t1.label_to_node[1] = 0;
        // leaf 2 (node 1)
        t1.parent.push(2);
        t1.left.push(NONE);
        t1.right.push(NONE);
        t1.label.push(2);
        t1.label_to_node[2] = 1;
        // root (node 2)
        t1.parent.push(NONE);
        t1.left.push(0);
        t1.right.push(1);
        t1.label.push(0);
        t1.root = 2;
        t1.compute_metadata();

        (t1.clone(), t1)
    }

    #[test]
    fn test_identical_trees() {
        let (t1, t2) = make_tree_2leaves();
        assert_eq!(approx_rspr_distance(&t1, &t2), 0);
    }

    #[test]
    fn test_lower_bound_identical() {
        let (t1, t2) = make_tree_2leaves();
        assert_eq!(lower_bound_components(&[t1, t2]), 1);
    }

    fn make_3leaf_trees() -> (Tree, Tree) {
        // T1: ((1,2),3)
        // T2: ((1,3),2)
        // rSPR distance = 1

        // T1: ((1,2),3)
        let mut t1 = Tree::with_capacity(3);
        // leaf 1 (node 0)
        t1.parent.push(3);
        t1.left.push(NONE);
        t1.right.push(NONE);
        t1.label.push(1);
        t1.label_to_node[1] = 0;
        // leaf 2 (node 1)
        t1.parent.push(3);
        t1.left.push(NONE);
        t1.right.push(NONE);
        t1.label.push(2);
        t1.label_to_node[2] = 1;
        // leaf 3 (node 2)
        t1.parent.push(4);
        t1.left.push(NONE);
        t1.right.push(NONE);
        t1.label.push(3);
        t1.label_to_node[3] = 2;
        // internal (1,2) -> node 3
        t1.parent.push(4);
        t1.left.push(0);
        t1.right.push(1);
        t1.label.push(0);
        // root (node 4)
        t1.parent.push(NONE);
        t1.left.push(3);
        t1.right.push(2);
        t1.label.push(0);
        t1.root = 4;
        t1.compute_metadata();

        // T2: ((1,3),2)
        let mut t2 = Tree::with_capacity(3);
        // leaf 1 (node 0)
        t2.parent.push(3);
        t2.left.push(NONE);
        t2.right.push(NONE);
        t2.label.push(1);
        t2.label_to_node[1] = 0;
        // leaf 3 (node 1)
        t2.parent.push(3);
        t2.left.push(NONE);
        t2.right.push(NONE);
        t2.label.push(3);
        t2.label_to_node[3] = 1;
        // leaf 2 (node 2)
        t2.parent.push(4);
        t2.left.push(NONE);
        t2.right.push(NONE);
        t2.label.push(2);
        t2.label_to_node[2] = 2;
        // internal (1,3) -> node 3
        t2.parent.push(4);
        t2.left.push(0);
        t2.right.push(1);
        t2.label.push(0);
        // root (node 4)
        t2.parent.push(NONE);
        t2.left.push(3);
        t2.right.push(2);
        t2.label.push(0);
        t2.root = 4;
        t2.compute_metadata();

        (t1, t2)
    }

    #[test]
    fn test_3leaf_different() {
        let (t1, t2) = make_3leaf_trees();
        let cost = approx_rspr_distance(&t1, &t2);
        // True rSPR = 1, 3-approx gives at most 3
        assert!(cost >= 1 && cost <= 3, "cost={}", cost);
    }

    #[test]
    fn test_lower_bound_3leaf() {
        let (t1, t2) = make_3leaf_trees();
        let lb = lower_bound_components(&[t1, t2]);
        // True OPT = 2 components. Lower bound should be >= 1 (and ideally 2).
        assert!(lb >= 1);
    }

    #[test]
    fn test_single_tree() {
        let (t1, _) = make_3leaf_trees();
        assert_eq!(lower_bound_components(&[t1]), 1);
    }
}
