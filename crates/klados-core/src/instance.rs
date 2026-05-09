//! Problem instance: a collection of trees on the same leaf set

use crate::tree::Tree;
use pace26io::binary_tree::IndexedBinTreeBuilder;
use pace26io::pace::simplified::Instance as PaceInstance;
use std::io::{self, BufReader, Read};
use std::path::Path;

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

    /// Read a PACE instance from a [`Read`] source.
    ///
    /// This is the canonical entry point for loading instances from stdin,
    /// files, or any byte source. Returns an error if parsing fails.
    pub fn from_reader(reader: impl Read) -> Result<Self, Box<dyn std::error::Error>> {
        let reader = BufReader::new(reader);
        let mut builder = IndexedBinTreeBuilder::default();
        let pace = PaceInstance::try_read(reader, &mut builder)?;
        let num_leaves = pace.num_leaves as u32;
        let trees: Vec<Tree> = pace
            .trees
            .iter()
            .map(|t| Tree::from_cursor(t.top_down(), num_leaves))
            .collect();
        Ok(Self::new(trees, num_leaves))
    }

    /// Read a PACE instance from a file path.
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        Self::from_reader(content.as_bytes())
    }

    /// Read a PACE instance from stdin.
    pub fn from_stdin() -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_reader(io::stdin().lock())
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
