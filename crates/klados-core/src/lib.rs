//! klados-core: Core data structures for the klados MAF solver
//!
//! Provides arena-based tree representation for cache-efficient traversal
//! during the FPT search.
//!
//! Indexed `for i in 0..n` loops over parallel arrays are an intentional
//! pattern in the numerical kernels; clippy's iterator suggestion there is
//! churn that hurts readability, so the lint is silenced crate-wide.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod af_validator;
pub mod brute_maf;
pub mod cluster_decomposition;
pub mod cluster_reduction;
pub mod instance;
pub mod instance_list;
pub mod kernelize;
pub mod lower_bound;
pub mod solve_pipeline;
pub mod tree;
pub mod twin_tree;
pub mod xforest;

pub use instance::Instance;
pub use instance_list::{InstanceEntry, parse_list_file};
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
