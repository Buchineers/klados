//! Forest representation (tree with cut edges)
//!
//! A forest represents a partial solution: the reference tree with some edges cut.

use crate::tree::{NodeId, Tree, NONE};
use fixedbitset::FixedBitSet;

/// A forest is a tree with some edges marked as cut
///
/// Cut edges partition the tree into multiple connected components.
#[derive(Clone, Debug)]
pub struct Forest {
    /// The underlying tree structure
    pub tree: Tree,
    /// Set of cut edges (indexed by child node of the edge)
    pub cut_edges: FixedBitSet,
    /// Roots of current components (including original root if not cut below)
    pub component_roots: Vec<NodeId>,
}

impl Forest {
    /// Create a forest from a tree (initially no cuts)
    pub fn from_tree(tree: Tree) -> Self {
        let num_nodes = tree.num_nodes();
        let root = tree.root;

        Self {
            tree,
            cut_edges: FixedBitSet::with_capacity(num_nodes),
            component_roots: vec![root],
        }
    }

    /// Cut the edge above a node, creating a new component
    pub fn cut(&mut self, node: NodeId) {
        debug_assert!(node != self.tree.root, "Cannot cut above root");
        debug_assert!(
            !self.cut_edges.contains(node as usize),
            "Edge already cut"
        );

        self.cut_edges.insert(node as usize);
        self.component_roots.push(node);
    }

    /// Check if edge above node is cut
    #[inline]
    pub fn is_cut(&self, node: NodeId) -> bool {
        self.cut_edges.contains(node as usize)
    }

    /// Number of components (= number of cuts + 1)
    #[inline]
    pub fn num_components(&self) -> usize {
        self.cut_edges.count_ones(..) + 1
    }

    /// Get the component root for a given node
    pub fn component_root(&self, mut node: NodeId) -> NodeId {
        while !self.is_cut(node) && self.tree.parent[node as usize] != NONE {
            node = self.tree.parent[node as usize];
        }
        node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forest_creation() {
        let tree = Tree::with_capacity(3);
        let forest = Forest::from_tree(tree);
        assert_eq!(forest.num_components(), 1);
    }
}
