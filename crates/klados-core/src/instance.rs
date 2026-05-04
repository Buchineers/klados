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
    /// Labels that must survive kernelization unchanged.
    /// Each entry names a label in THIS instance's label space 1..=num_leaves.
    /// The kernelization pipeline merges these with any caller-supplied
    /// protected_labels when constructing KernelizeConfig.
    pub protected_labels: Vec<u32>,
}

impl Instance {
    /// Create a new instance
    pub fn new(trees: Vec<Tree>, num_leaves: u32) -> Self {
        debug_assert!(trees.iter().all(|t| t.num_leaves == num_leaves));
        Self {
            trees,
            num_leaves,
            name: None,
            protected_labels: Vec::new(),
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
