//! Extended forest representation with leafsets for MAF algorithms.
//!
//! XForest extends the basic Forest with precomputed leafsets, enabling
//! efficient component analysis and reduction rule applications.

use crate::tree::{NONE, NodeId, Tree};
use fixedbitset::FixedBitSet;

/// A forest representation with precomputed leafsets for efficient component analysis.
///
/// Maintains two leafset arrays:
/// - `full_leafsets`: Static leafsets computed once from the tree structure
/// - `live_leafsets`: Dynamic leafsets updated as edges are cut/uncut
#[derive(Clone, Debug)]
pub struct XForest {
    pub tree: Tree,
    pub cut_edges: FixedBitSet,
    pub full_leafsets: Vec<FixedBitSet>,
    pub live_leafsets: Vec<FixedBitSet>,
    /// Number of live leaves in each node's subtree — O(1) replacement for
    /// `live_leafsets[node].count_ones(..)`.
    pub live_leaf_count: Vec<u32>,
    pub component_roots: Vec<NodeId>,
}

impl XForest {
    pub fn from_tree(tree: Tree) -> Self {
        let num_nodes = tree.num_nodes();
        let num_leaves = tree.num_leaves;
        let mut full_leafsets = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let mut set = FixedBitSet::with_capacity(num_leaves as usize + 1);
            set.grow(num_leaves as usize + 1);
            full_leafsets.push(set);
        }
        for node in tree.post_order() {
            if let Some(lbl) = tree.leaf_label(node) {
                full_leafsets[node as usize].insert(lbl as usize);
            } else if let Some((l, r)) = tree.children(node) {
                let left = full_leafsets[l as usize].clone();
                let right = full_leafsets[r as usize].clone();
                full_leafsets[node as usize].union_with(&left);
                full_leafsets[node as usize].union_with(&right);
            }
        }
        let root = tree.root;
        let live_leafsets = full_leafsets.clone();
        let live_leaf_count: Vec<u32> = live_leafsets
            .iter()
            .map(|ls| ls.count_ones(..) as u32)
            .collect();
        Self {
            tree,
            cut_edges: FixedBitSet::with_capacity(num_nodes),
            full_leafsets,
            live_leafsets,
            live_leaf_count,
            component_roots: vec![root],
        }
    }

    #[inline]
    pub fn is_cut(&self, node: NodeId) -> bool {
        self.cut_edges.contains(node as usize)
    }

    pub fn cut(&mut self, node: NodeId) {
        debug_assert!(node != self.tree.root, "Cannot cut above root");
        if !self.cut_edges.contains(node as usize) {
            self.cut_edges.insert(node as usize);
            self.component_roots.push(node);
            let removed = self.live_leafsets[node as usize].clone();
            let removed_count = self.live_leaf_count[node as usize];
            let mut cur = self.tree.parent[node as usize];
            while cur != NONE {
                self.live_leafsets[cur as usize].difference_with(&removed);
                self.live_leaf_count[cur as usize] -= removed_count;
                if self.is_cut(cur) {
                    break;
                }
                cur = self.tree.parent[cur as usize];
            }
        }
    }

    pub fn uncut(&mut self, node: NodeId) {
        debug_assert!(self.cut_edges.contains(node as usize));
        self.cut_edges.set(node as usize, false);
        if let Some(pos) = self.component_roots.iter().rposition(|&r| r == node) {
            self.component_roots.swap_remove(pos);
        }
        let restored = self.live_leafsets[node as usize].clone();
        let restored_count = self.live_leaf_count[node as usize];
        let mut cur = self.tree.parent[node as usize];
        while cur != NONE {
            self.live_leafsets[cur as usize].union_with(&restored);
            self.live_leaf_count[cur as usize] += restored_count;
            if self.is_cut(cur) {
                break;
            }
            cur = self.tree.parent[cur as usize];
        }
    }

    pub fn reactivate_label(&mut self, lbl: u32) {
        let a_node = self.tree.label_to_node[lbl as usize];
        self.live_leafsets[a_node as usize].insert(lbl as usize);
        self.live_leaf_count[a_node as usize] += 1;
        let mut cur = self.tree.parent[a_node as usize];
        while cur != NONE {
            self.live_leafsets[cur as usize].insert(lbl as usize);
            self.live_leaf_count[cur as usize] += 1;
            if self.is_cut(cur) {
                break;
            }
            cur = self.tree.parent[cur as usize];
        }
    }

    pub fn component_root(&self, mut node: NodeId) -> NodeId {
        while !self.is_cut(node) && self.tree.parent[node as usize] != NONE {
            node = self.tree.parent[node as usize];
        }
        node
    }

    pub fn num_components(&self) -> usize {
        self.cut_edges.count_ones(..) + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_tree() -> Tree {
        let mut tree = Tree::with_capacity(3);
        tree.parent.push(3);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(1);
        tree.label_to_node[1] = 0;
        tree.parent.push(3);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(2);
        tree.label_to_node[2] = 1;
        tree.parent.push(4);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(3);
        tree.label_to_node[3] = 2;
        tree.parent.push(4);
        tree.left.push(0);
        tree.right.push(1);
        tree.label.push(0);
        tree.parent.push(NONE);
        tree.left.push(3);
        tree.right.push(2);
        tree.label.push(0);
        tree.root = 4;
        tree.compute_metadata();
        tree
    }

    #[test]
    fn test_xforest_from_tree() {
        let tree = make_simple_tree();
        let forest = XForest::from_tree(tree);
        assert_eq!(forest.cut_edges.count_ones(..), 0);
        assert_eq!(forest.component_roots.len(), 1);
    }

    #[test]
    fn test_xforest_cut_uncut() {
        let tree = make_simple_tree();
        let mut forest = XForest::from_tree(tree);
        let orig_leafsets: Vec<FixedBitSet> = forest.live_leafsets.clone();
        forest.cut(0);
        assert!(forest.is_cut(0));
        assert_eq!(forest.cut_edges.count_ones(..), 1);
        forest.uncut(0);
        assert!(!forest.is_cut(0));
        assert_eq!(forest.cut_edges.count_ones(..), 0);
        assert_eq!(forest.live_leafsets, orig_leafsets);
    }
}
