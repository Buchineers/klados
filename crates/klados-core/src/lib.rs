//! klados-core: Core data structures for the klados MAF solver
//!
//! Provides arena-based tree representation for cache-efficient traversal
//! during the FPT search.

pub mod cluster_decomposition;
pub mod cluster_reduction;
pub mod instance;
pub mod kernelize;
pub mod lower_bound;
pub mod tree;
pub mod xforest;

pub use instance::Instance;
pub use tree::{Label, NONE, NodeId, Tree};
pub use xforest::XForest;

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
