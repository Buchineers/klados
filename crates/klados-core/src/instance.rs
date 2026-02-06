//! Problem instance: a collection of trees on the same leaf set

use crate::tree::Tree;

/// A problem instance: t trees on n leaves
#[derive(Clone, Debug)]
pub struct Instance {
    /// The input trees (T₁, T₂, ..., Tₜ)
    pub trees: Vec<Tree>,
    /// Number of leaves (same for all trees)
    pub num_leaves: u32,
    /// Instance name (from STRIDE metadata)
    pub name: Option<String>,
}

impl Instance {
    /// Create a new instance
    pub fn new(trees: Vec<Tree>, num_leaves: u32) -> Self {
        debug_assert!(trees.iter().all(|t| t.num_leaves == num_leaves));
        Self {
            trees,
            num_leaves,
            name: None,
        }
    }

    /// Number of trees in the instance
    #[inline]
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Get reference tree (T₁)
    #[inline]
    pub fn reference_tree(&self) -> &Tree {
        &self.trees[0]
    }
}
