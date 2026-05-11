//! Arena-based rooted binary phylogenetic tree
//!
//! Trees are stored in a cache-efficient Structure-of-Arrays (SoA) layout.
//! All nodes are indices into contiguous vectors, avoiding pointer chasing.

use fixedbitset::FixedBitSet;
use pace26io::binary_tree::{Label as PaceLabel, TopDownCursor};

/// Node identifier (index into arena vectors)
pub type NodeId = u32;

/// Leaf label (1..=n as per PACE format)
pub type Label = u32;

/// Sentinel value for "no node"
pub const NONE: NodeId = NodeId::MAX;

/// A rooted binary phylogenetic tree stored in arena format
///
/// Leaves are labeled 1..=n (matching PACE format).
/// Internal nodes have indices assigned during construction.
#[derive(Clone, Debug)]
pub struct Tree {
    // Topology (Structure of Arrays)
    /// Parent of each node (NONE for root)
    pub parent: Vec<NodeId>,
    /// Left child of each node (NONE for leaves)
    pub left: Vec<NodeId>,
    /// Right child of each node (NONE for leaves)
    pub right: Vec<NodeId>,

    // Leaf data
    /// Label of leaf nodes (0 for internal nodes)
    pub label: Vec<Label>,
    /// Map from label to node id: label_to_node[label] = node_id
    pub label_to_node: Vec<NodeId>,

    // Precomputed metadata
    /// Depth of each node from root
    pub depth: Vec<u16>,
    /// Number of leaves in subtree rooted at each node
    pub subtree_size: Vec<u32>,

    // Tree metadata
    /// Number of leaves
    pub num_leaves: u32,
    /// Root node id
    pub root: NodeId,
}

impl Tree {
    /// Create an empty tree with capacity for n leaves
    pub fn with_capacity(num_leaves: u32) -> Self {
        // A binary tree with n leaves has 2n-1 nodes
        let num_nodes = (2 * num_leaves).saturating_sub(1) as usize;

        Self {
            parent: Vec::with_capacity(num_nodes),
            left: Vec::with_capacity(num_nodes),
            right: Vec::with_capacity(num_nodes),
            label: Vec::with_capacity(num_nodes),
            label_to_node: vec![NONE; num_leaves as usize + 1], // 1-indexed
            depth: Vec::with_capacity(num_nodes),
            subtree_size: Vec::with_capacity(num_nodes),
            num_leaves,
            root: NONE,
        }
    }

    /// Build a Tree from a pace26io TopDownCursor
    pub fn from_cursor<C: TopDownCursor>(cursor: C, num_leaves: u32) -> Self {
        let mut tree = Self::with_capacity(num_leaves);
        tree.root = tree.build_from_cursor(cursor, None);
        tree.compute_metadata();
        tree
    }

    fn build_from_cursor<C: TopDownCursor>(&mut self, cursor: C, parent: Option<NodeId>) -> NodeId {
        let node_id = self.parent.len() as NodeId;

        // Set parent if provided
        self.parent.push(parent.unwrap_or(NONE));

        if let Some((left_cursor, right_cursor)) = cursor.children() {
            // Internal node
            self.label.push(0);

            // Add placeholder children (will be updated)
            self.left.push(NONE);
            self.right.push(NONE);

            // Recursively build children
            let left_id = self.build_from_cursor(left_cursor, Some(node_id));
            let right_id = self.build_from_cursor(right_cursor, Some(node_id));

            // Update children
            self.left[node_id as usize] = left_id;
            self.right[node_id as usize] = right_id;
        } else if let Some(PaceLabel(lbl)) = cursor.leaf_label() {
            // Leaf node
            self.label.push(lbl);
            self.label_to_node[lbl as usize] = node_id;
            self.left.push(NONE);
            self.right.push(NONE);
        } else {
            panic!("Node is neither internal nor leaf");
        }

        node_id
    }

    /// Check if node is a leaf
    #[inline]
    pub fn is_leaf(&self, node: NodeId) -> bool {
        self.left[node as usize] == NONE
    }

    /// Check if node is the root
    #[inline]
    pub fn is_root(&self, node: NodeId) -> bool {
        self.parent[node as usize] == NONE
    }

    /// Get children of a node, or None if it's a leaf
    #[inline]
    pub fn children(&self, node: NodeId) -> Option<(NodeId, NodeId)> {
        let left = self.left[node as usize];
        if left == NONE {
            None
        } else {
            Some((left, self.right[node as usize]))
        }
    }

    /// Get children of an internal node directly (no Option check).
    /// Caller must ensure the node is internal (not a leaf).
    #[inline]
    pub fn children_pair(&self, node: NodeId) -> (NodeId, NodeId) {
        (self.left[node as usize], self.right[node as usize])
    }

    /// Get the label of a leaf node
    #[inline]
    pub fn leaf_label(&self, node: NodeId) -> Option<Label> {
        let lbl = self.label[node as usize];
        if lbl == 0 { None } else { Some(lbl) }
    }

    /// Get node by leaf label
    #[inline]
    pub fn node_by_label(&self, label: Label) -> NodeId {
        self.label_to_node[label as usize]
    }

    /// Get sibling of a node (None for root)
    pub fn sibling(&self, node: NodeId) -> Option<NodeId> {
        let p = self.parent[node as usize];
        if p == NONE {
            return None;
        }

        let left = self.left[p as usize];
        if left == node {
            Some(self.right[p as usize])
        } else {
            Some(left)
        }
    }

    pub fn nearest_common_ancestor(&self, mut node_a: NodeId, mut node_b: NodeId) -> NodeId {
        // Bring both nodes to the same depth
        let mut da = self.depth[node_a as usize];
        let mut db = self.depth[node_b as usize];

        while da > db {
            node_a = self.parent[node_a as usize];
            da -= 1;
        }
        while db > da {
            node_b = self.parent[node_b as usize];
            db -= 1;
        }

        // Walk up together until they meet
        while node_a != node_b {
            node_a = self.parent[node_a as usize];
            node_b = self.parent[node_b as usize];
        }

        node_a
    }

    /// Number of nodes in the tree
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.parent.len()
    }

    /// Iterate over all leaf labels
    pub fn leaves(&self) -> impl Iterator<Item = Label> + '_ {
        (1..=self.num_leaves).filter(move |&lbl| self.label_to_node[lbl as usize] != NONE)
    }

    /// Iterate over nodes in post-order (leaves before parents)
    pub fn post_order(&self) -> PostOrderIter<'_> {
        PostOrderIter::new(self)
    }

    /// Iterate over nodes in pre-order (parents before children)
    pub fn pre_order(&self) -> PreOrderIter<'_> {
        PreOrderIter::new(self)
    }

    /// Get a cursor for navigating this tree via pace26io's TopDownCursor trait
    pub fn cursor(&self) -> TreeCursor<'_> {
        TreeCursor {
            tree: self,
            node: self.root,
        }
    }

    /// Return post-order traversal as a pre-computed Vec for reuse
    pub fn post_order_vec(&self) -> Vec<NodeId> {
        self.post_order().collect()
    }

    /// Compute and cache depth and subtree_size for all nodes
    pub fn compute_metadata(&mut self) {
        self.depth.clear();
        self.depth.resize(self.num_nodes(), 0);
        self.subtree_size.clear();
        self.subtree_size.resize(self.num_nodes(), 0);

        if self.root == NONE {
            return;
        }

        // Compute depths via BFS from root
        let mut queue = vec![self.root];
        self.depth[self.root as usize] = 0;

        while let Some(node) = queue.pop() {
            if let Some((left, right)) = self.children(node) {
                let d = self.depth[node as usize] + 1;
                self.depth[left as usize] = d;
                self.depth[right as usize] = d;
                queue.push(left);
                queue.push(right);
            }
        }

        // Compute subtree sizes via post-order
        let post_order: Vec<_> = self.post_order().collect();
        for node in post_order {
            if self.is_leaf(node) {
                self.subtree_size[node as usize] = 1;
            } else {
                let (left, right) = self.children(node).unwrap();
                self.subtree_size[node as usize] =
                    self.subtree_size[left as usize] + self.subtree_size[right as usize];
            }
        }
    }

    /// Create a new tree with labels remapped according to the given mapping.
    ///
    /// `new_num_leaves` is the size of the new label space (labels will be 1..=new_num_leaves).
    /// `label_map[old_label]` = new_label for each leaf label present in this tree.
    /// Labels not in the map (value 0) are treated as absent — their leaves are pruned
    /// and degree-1 internal nodes are suppressed.
    pub fn relabel(&self, label_map: &[Label], new_num_leaves: u32) -> Self {
        let mut out = Tree::with_capacity(new_num_leaves);

        fn build(src: &Tree, label_map: &[Label], out: &mut Tree, node: NodeId) -> Option<NodeId> {
            if src.is_leaf(node) {
                let old_lbl = src.label[node as usize];
                if old_lbl == 0 {
                    return None;
                }
                let new_lbl = if (old_lbl as usize) < label_map.len() {
                    label_map[old_lbl as usize]
                } else {
                    0
                };
                if new_lbl == 0 {
                    return None;
                }
                let id = out.parent.len() as NodeId;
                out.parent.push(NONE);
                out.left.push(NONE);
                out.right.push(NONE);
                out.label.push(new_lbl);
                out.label_to_node[new_lbl as usize] = id;
                return Some(id);
            }

            let (left, right) = src.children(node).unwrap();
            let l = build(src, label_map, out, left);
            let r = build(src, label_map, out, right);

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

        if self.root != NONE {
            if let Some(root) = build(self, label_map, &mut out, self.root) {
                out.root = root;
                out.parent[root as usize] = NONE;
            } else {
                out.root = NONE;
            }
        }

        out.compute_metadata();
        out
    }

    /// Create a new tree that contains only leaves in `keep`.
    ///
    /// Suppresses degree-1 internal nodes to keep the tree binary.
    /// The label space (and `num_leaves`) is preserved from `self`.
    pub fn prune_to_leafset(&self, keep: &FixedBitSet) -> Self {
        let mut out = Tree::with_capacity(self.num_leaves);

        fn build(src: &Tree, keep: &FixedBitSet, out: &mut Tree, node: NodeId) -> Option<NodeId> {
            if src.is_leaf(node) {
                let lbl = src.label[node as usize];
                if lbl != 0 && keep.contains(lbl as usize) {
                    let id = out.parent.len() as NodeId;
                    out.parent.push(NONE);
                    out.left.push(NONE);
                    out.right.push(NONE);
                    out.label.push(lbl);
                    out.label_to_node[lbl as usize] = id;
                    return Some(id);
                }
                return None;
            }

            let (left, right) = src.children(node).unwrap();
            let l = build(src, keep, out, left);
            let r = build(src, keep, out, right);

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

        if self.root != NONE {
            if let Some(root) = build(self, keep, &mut out, self.root) {
                out.root = root;
                out.parent[root as usize] = NONE;
            } else {
                out.root = NONE;
            }
        }

        out.compute_metadata();
        out
    }

    /// Create a singleton tree containing exactly one leaf with the given label.
    ///
    /// Ported from `shi_mestel::utils::make_singleton_tree`. The label space
    /// (`num_leaves`) is preserved so the tree is compatible with others on the
    /// same instance.
    pub fn singleton(label: Label, num_leaves: u32) -> Self {
        let mut t = Tree::with_capacity(num_leaves);
        t.parent.push(NONE);
        t.left.push(NONE);
        t.right.push(NONE);
        t.label.push(label);
        t.label_to_node[label as usize] = 0;
        t.root = 0;
        t.compute_metadata();
        t
    }

    /// Build a component tree from a leafset by pruning the reference tree.
    ///
    /// If the leafset contains exactly one leaf, returns a singleton tree.
    /// Otherwise prunes `reference` to only the leaves in `leafset`.
    ///
    /// Ported from `shi_mestel::extraction::build_component_tree`.
    pub fn component_from_leafset(
        leafset: &FixedBitSet,
        reference: &Tree,
        num_leaves: u32,
    ) -> Self {
        if leafset.count_ones(..) == 1 {
            let lbl = leafset.ones().next().unwrap() as Label;
            Tree::singleton(lbl, num_leaves)
        } else {
            reference.prune_to_leafset(leafset)
        }
    }
}

/// Cursor for navigating our arena tree via pace26io's TopDownCursor trait
#[derive(Clone)]
pub struct TreeCursor<'a> {
    tree: &'a Tree,
    node: NodeId,
}

impl<'a> TopDownCursor for TreeCursor<'a> {
    fn children(&self) -> Option<(Self, Self)> {
        self.tree.children(self.node).map(|(l, r)| {
            (
                TreeCursor {
                    tree: self.tree,
                    node: l,
                },
                TreeCursor {
                    tree: self.tree,
                    node: r,
                },
            )
        })
    }

    fn leaf_label(&self) -> Option<PaceLabel> {
        self.tree.leaf_label(self.node).map(PaceLabel)
    }
}

/// Post-order iterator over tree nodes.
///
/// Uses a single-pass leftmost-descent approach: each node is pushed to the
/// stack exactly once (as a `NodeId` rather than `(NodeId, bool)`), eliminating
/// the bool flag overhead. We descend to the leftmost leaf, yield it, then
/// process right subtrees as we unwind.
pub struct PostOrderIter<'a> {
    tree: &'a Tree,
    stack: Vec<NodeId>,
}

impl<'a> PostOrderIter<'a> {
    fn new(tree: &'a Tree) -> Self {
        let mut iter = Self {
            tree,
            stack: Vec::with_capacity(64),
        };
        if tree.root != NONE {
            iter.push_leftmost(tree.root);
        }
        iter
    }

    /// Push a node and all its left descendants onto the stack.
    #[inline]
    fn push_leftmost(&mut self, mut node: NodeId) {
        loop {
            self.stack.push(node);
            match self.tree.left[node as usize] {
                NONE => return,
                left => node = left,
            }
        }
    }
}

impl<'a> Iterator for PostOrderIter<'a> {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.stack.pop()?;

        // If this node has a parent on the stack, and this node is the left
        // child of that parent, we need to process the right subtree first.
        if let Some(&parent) = self.stack.last()
            && self.tree.left[parent as usize] == node
        {
            // We just finished the left subtree; descend into right
            self.push_leftmost(self.tree.right[parent as usize]);
        }

        Some(node)
    }
}

/// Pre-order iterator over tree nodes
pub struct PreOrderIter<'a> {
    tree: &'a Tree,
    stack: Vec<NodeId>,
}

impl<'a> PreOrderIter<'a> {
    fn new(tree: &'a Tree) -> Self {
        let mut iter = Self {
            tree,
            stack: Vec::with_capacity(64),
        };
        if tree.root != NONE {
            iter.stack.push(tree.root);
        }
        iter
    }
}

impl<'a> Iterator for PreOrderIter<'a> {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.stack.pop()?;

        if let Some((left, right)) = self.tree.children(node) {
            self.stack.push(right);
            self.stack.push(left);
        }

        Some(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_tree() -> Tree {
        let mut tree = Tree::with_capacity(3);

        // leaf 1 (node 0)
        tree.parent.push(3);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(1);
        tree.label_to_node[1] = 0;

        // leaf 2 (node 1)
        tree.parent.push(3);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(2);
        tree.label_to_node[2] = 1;

        // leaf 3 (node 2)
        tree.parent.push(4);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(3);
        tree.label_to_node[3] = 2;

        // internal (1,2) (node 3)
        tree.parent.push(4);
        tree.left.push(0);
        tree.right.push(1);
        tree.label.push(0);

        // root (node 4)
        tree.parent.push(NONE);
        tree.left.push(3);
        tree.right.push(2);
        tree.label.push(0);

        tree.root = 4;
        tree.compute_metadata();
        tree
    }

    #[test]
    fn test_tree_structure() {
        let tree = make_simple_tree();

        assert_eq!(tree.num_nodes(), 5);
        assert_eq!(tree.num_leaves, 3);
        assert!(tree.is_leaf(0));
        assert!(tree.is_leaf(1));
        assert!(tree.is_leaf(2));
        assert!(!tree.is_leaf(3));
        assert!(!tree.is_leaf(4));
        assert!(tree.is_root(4));
    }

    #[test]
    fn test_post_order() {
        let tree = make_simple_tree();
        let order: Vec<_> = tree.post_order().collect();

        assert_eq!(order.len(), 5);
        let pos_0 = order.iter().position(|&n| n == 0).unwrap();
        let pos_1 = order.iter().position(|&n| n == 1).unwrap();
        let pos_3 = order.iter().position(|&n| n == 3).unwrap();
        let pos_4 = order.iter().position(|&n| n == 4).unwrap();

        assert!(pos_0 < pos_3);
        assert!(pos_1 < pos_3);
        assert!(pos_3 < pos_4);
    }

    #[test]
    fn test_newick_roundtrip() {
        use pace26io::newick::NewickWriter;
        let tree = make_simple_tree();
        let newick = tree.cursor().to_newick_string();
        assert_eq!(newick, "((1,2),3);");
    }

    #[test]
    fn test_post_order_vec() {
        let tree = make_simple_tree();
        let vec = tree.post_order_vec();
        let iter: Vec<_> = tree.post_order().collect();
        assert_eq!(vec, iter);
    }

    #[test]
    fn test_depths() {
        let tree = make_simple_tree();

        assert_eq!(tree.depth[4], 0); // root
        assert_eq!(tree.depth[3], 1); // internal
        assert_eq!(tree.depth[2], 1); // leaf 3
        assert_eq!(tree.depth[0], 2); // leaf 1
        assert_eq!(tree.depth[1], 2); // leaf 2
    }

    #[test]
    fn test_nearest_common_ancestor() {
        let tree = make_simple_tree();

        // LCA of leaves 1 and 2 is their parent (node 3)
        assert_eq!(tree.nearest_common_ancestor(0, 1), 3);
        // LCA of leaves 1 and 3 is the root (node 4)
        assert_eq!(tree.nearest_common_ancestor(0, 2), 4);
        // LCA of a node with itself is itself
        assert_eq!(tree.nearest_common_ancestor(2, 2), 2);
        // LCA of leaf 2 and internal node 3 is node 3
        assert_eq!(tree.nearest_common_ancestor(1, 3), 3);
    }
}
