//! klados-core: Core data structures for the klados MAF solver
//!
//! Provides arena-based tree representation for cache-efficient traversal
//! during the FPT search.

pub mod instance;
pub mod tree;
pub mod xforest;

pub use instance::Instance;
pub use tree::{Label, NodeId, Tree, NONE};
pub use xforest::XForest;

/// Solver configuration
#[derive(Clone, Debug)]
pub struct SolverConfig {
    /// Maximum search depth (None = unlimited)
    pub max_depth: Option<usize>,
    /// Verbosity level
    pub verbose: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_depth: None,
            verbose: false,
        }
    }
}

/// Statistics collected during solving
#[derive(Clone, Debug, Default)]
pub struct SolverStats {
    /// Number of search nodes explored
    pub nodes_explored: u64,
    /// Number of branches pruned
    pub branches_pruned: u64,
    /// Current lower bound
    pub lower_bound: usize,
    /// Current upper bound (if found)
    pub upper_bound: Option<usize>,
}
