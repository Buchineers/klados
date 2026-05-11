//! Shared solve pipeline for exact solvers.
//!
//! All exact solvers follow the same structure:
//! 1. (optional) Kernelize the instance
//! 2. (optional) Try cluster decomposition/reduction
//! 3. Run inner solver on reduced instance
//! 4. Expand solution back to original label space
//!
//! This module provides a reusable pipeline that solvers can delegate to.

use crate::Instance;
use crate::Tree;
use crate::cluster_decomposition;
use crate::cluster_reduction::{self, ClusterReductionResult};
use crate::kernelize::{self, KernelizeConfig, KernelizeResult};

/// Which cluster decomposition algorithm to apply.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ClusterAlgo {
    /// Don't try any cluster decomposition.
    #[default]
    None,
    /// Kelk's 4-subinstance cluster reduction (default for most solvers).
    ClusterReduction,
    /// rSPR-style cluster decomposition (Whidden et al.).
    ClusterDecomposition,
    /// Try cluster reduction first, fall back to rSPR decomposition.
    Both,
}

/// Configuration for the solve pipeline.
#[derive(Clone, Debug)]
pub struct SolveConfig {
    /// Whether to run kernelization before solving.
    pub kernelize: bool,
    /// Detailed kernelization options (only used if `kernelize` is true).
    pub kernelize_config: KernelizeConfig,
    /// Which cluster decomposition algorithm to try.
    pub cluster_algo: ClusterAlgo,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            kernelize: true,
            kernelize_config: KernelizeConfig::default(),
            cluster_algo: ClusterAlgo::ClusterReduction,
        }
    }
}

impl SolveConfig {
    /// No kernelization, no cluster decomposition — just run the inner solver.
    pub fn bare() -> Self {
        Self {
            kernelize: false,
            kernelize_config: KernelizeConfig::default(),
            cluster_algo: ClusterAlgo::None,
        }
    }

    /// Kernelize with subtree-only reduction (fastest).
    pub fn subtree_only() -> Self {
        Self {
            kernelize: true,
            kernelize_config: KernelizeConfig {
                subtree: true,
                chain: false,
                chain32: false,
                chain32_multi: false,
                ..Default::default()
            },
            cluster_algo: ClusterAlgo::None,
        }
    }

    /// Kernelize with all rules but no cluster decomposition.
    pub fn kernelize_only() -> Self {
        Self {
            kernelize: true,
            kernelize_config: KernelizeConfig::default(),
            cluster_algo: ClusterAlgo::None,
        }
    }

    /// Full pipeline with all optimizations.
    pub fn full() -> Self {
        Self::default()
    }

    /// Full pipeline with cluster reduction only (no rSPR decomposition).
    pub fn with_cluster_reduction() -> Self {
        Self {
            kernelize: true,
            kernelize_config: KernelizeConfig::default(),
            cluster_algo: ClusterAlgo::ClusterReduction,
        }
    }
}

/// A function that solves a (possibly kernelized) instance and returns
/// the components of an agreement forest, or `None` on failure.
pub type InnerSolver = dyn FnMut(&Instance) -> Option<Vec<Tree>>;

/// Run the standard solve pipeline on an instance.
///
/// The callback `solve_inner` receives the (possibly kernelized) instance and must
/// return agreement forest components in the REDUCED label space, or `None` if it
/// cannot solve.
///
/// Returns `Some(components)` in the ORIGINAL label space (expanded through kernelization
/// reverse map and deletion singletons), or `None` on failure.
pub fn solve_with_pipeline(
    instance: &Instance,
    config: &SolveConfig,
    solve_inner: &mut InnerSolver,
) -> Option<Vec<Tree>> {
    if instance.trees.is_empty() {
        return None;
    }
    if instance.num_trees() == 1 {
        return Some(instance.trees.clone());
    }

    let kern = if config.kernelize {
        kernelize::kernelize_best(instance, &config.kernelize_config)
    } else {
        empty_kern_result(instance)
    };
    let reduced = &kern.instance;

    let solve = &mut |sub: &Instance| -> Option<Vec<Tree>> { solve_inner(sub) };

    // Try cluster algorithms in order. `try_cluster_reduction` returns
    // `Some(NotApplicable)` when no common cluster exists — we must fall
    // through to the next algorithm or inner solver, NOT return None.
    match config.cluster_algo {
        ClusterAlgo::None => {}
        ClusterAlgo::ClusterReduction => {
            if let Some(ClusterReductionResult::Solved(solution)) =
                cluster_reduction::try_cluster_reduction(reduced, solve)
            {
                return Some(kernelize::expand_solution(
                    solution.components, &kern, &instance.trees[0], instance.num_leaves,
                ));
            }
            // NotApplicable or None → fall through to inner solver.
        }
        ClusterAlgo::ClusterDecomposition => {
            if let Some(components) =
                cluster_decomposition::try_rspr_cluster_decomposition(reduced, solve)
            {
                return Some(kernelize::expand_solution(
                    components, &kern, &instance.trees[0], instance.num_leaves,
                ));
            }
        }
        ClusterAlgo::Both => {
            if let Some(ClusterReductionResult::Solved(solution)) =
                cluster_reduction::try_cluster_reduction(reduced, solve)
            {
                return Some(kernelize::expand_solution(
                    solution.components, &kern, &instance.trees[0], instance.num_leaves,
                ));
            }
            // NotApplicable → fall through to cluster decomposition.
            if let Some(components) =
                cluster_decomposition::try_rspr_cluster_decomposition(reduced, solve)
            {
                return Some(kernelize::expand_solution(
                    components, &kern, &instance.trees[0], instance.num_leaves,
                ));
            }
        }
    }

    // No cluster decomposition worked — run the inner solver directly
    let reduced_components = solve_inner(reduced)?;
    Some(kernelize::expand_solution(
        reduced_components,
        &kern,
        &instance.trees[0],
        instance.num_leaves,
    ))
}

fn empty_kern_result(instance: &Instance) -> KernelizeResult {
    KernelizeResult {
        instance: instance.clone(),
        stats: kernelize::KernelizeStats {
            original_leaves: instance.num_leaves,
            reduced_leaves: instance.num_leaves,
            rule_counts: Default::default(),
            deleted_labels: Vec::new(),
            surviving_taxa: Vec::new(),
        },
        reverse_map: (0..=instance.num_leaves).collect(),
        collapses_original: Vec::new(),
        param_reduction: 0,
        trace: Vec::new(),
    }
}
