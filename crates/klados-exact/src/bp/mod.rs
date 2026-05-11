//! Branch-and-Price solver for multi-tree MAF — type-disciplined rewrite.
//!
//! ## Module map
//! - [`column`] — `AfColumn` (validity-by-construction), builder, dedup set.
//! - [`search`] — `Branchings` (pair-only), `SearchState`, selection, telemetry.
//! - [`rmp`]    — HiGHS-backed RMP, eager rows, branchings-derived bounds.
//! - [`pricer`] — `Pricer` trait + `BrutePairsPricer` (stage 2 default).
//! - [`solver`] — search loop and node solver.

pub mod column;
pub mod pricer;
pub mod rmp;
pub mod search;
pub mod solver;

use std::time::Instant;

use klados_core::solve_pipeline::{ClusterAlgo, SolveConfig, solve_with_pipeline};
use klados_core::{Instance, SolverStats, Tree};
use log::{info, trace};

use crate::ExactSolver;
use crate::whidden_cluster::try_whidden_decomp_2tree;

const LOG_TARGET: &str = "klados::bp";

/// Minimum leaves for which Whidden strict cluster decomp is worth trying.
/// Below this, the pipeline's generic cluster_reduction handles things fine
/// and Whidden's overhead isn't justified.
const WHIDDEN_MIN_LEAVES: u32 = 20;

/// Stage-2 configuration. The defaults match what the current pricer can
/// soundly support: cluster algorithms stay disabled until a sound pricer
/// (m=2 pair-DP / small-m m-DP) lands, since cluster reduction's stitching
/// requires optimal sub-solves.
#[derive(Clone, Debug)]
pub struct BpConfig {
    pub kernelize: bool,
    pub cluster_algo: ClusterAlgo,
}

impl Default for BpConfig {
    fn default() -> Self {
        Self {
            kernelize: true,
            cluster_algo: ClusterAlgo::Both,
        }
    }
}

impl BpConfig {
    /// Configuration with all decomposition disabled — only kernelization
    /// and direct B&P. Used to expose algorithmic issues that would
    /// otherwise be hidden by decomposition.
    pub fn no_decomp() -> Self {
        Self {
            kernelize: true,
            cluster_algo: ClusterAlgo::None,
        }
    }
}

pub struct BpSolver {
    stats: SolverStats,
    config: BpConfig,
}

impl Default for BpSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl BpSolver {
    pub fn new() -> Self {
        // KLADOS_BP_NO_DECOMP=1 disables all decomposition (Whidden, cluster
        // reduction, cluster decomposition). Used to expose algorithmic
        // weaknesses in the core B&P that would otherwise be masked.
        let config = if std::env::var("KLADOS_BP_NO_DECOMP").is_ok() {
            BpConfig::no_decomp()
        } else {
            BpConfig::default()
        };
        Self {
            stats: SolverStats::default(),
            config,
        }
    }

    pub fn with_config(config: BpConfig) -> Self {
        Self {
            stats: SolverStats::default(),
            config,
        }
    }
}

impl ExactSolver for BpSolver {
    fn name(&self) -> &'static str {
        "bp"
    }

    fn description(&self) -> &'static str {
        "Branch & Price for multi-tree MAF (rewrite, in progress)"
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        let t_total = Instant::now();
        let cfg = self.config.clone();
        let components = solve_recursive(instance, &cfg)?;
        self.stats.upper_bound = Some(components.len());
        self.stats.lower_bound = components.len();
        info!(
            target: LOG_TARGET,
            "solved n={} m={} k={} in {:.1}ms",
            instance.num_leaves,
            instance.num_trees(),
            components.len(),
            t_total.elapsed().as_secs_f64() * 1000.0,
        );
        Some(components)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

/// Recursive solve entry: tries strategies in order of effectiveness for
/// each instance shape, falling through on failure.
///
/// Exposed `pub(crate)` so `solver::solve_inner` can call it back as the
/// recursion target for primal heuristics that solve sub-instances.
pub(crate) fn _solve_recursive_alias(instance: &Instance, cfg: &BpConfig) -> Option<Vec<Tree>> {
    solve_recursive(instance, cfg)
}


///
/// Order:
/// 1. **Trivial** (m≤1, n≤1) — short-circuit.
/// 2. **Kernelize** — reduce the instance before decomposition.
/// 3. **Whidden strict cluster decomp** (m=2, n≥WHIDDEN_MIN_LEAVES) —
///    applied on the kernelized instance so cluster points aren't obscured
///    by reducible leaves. Matches bp-multi's kernelize-before-Whidden flow.
/// 4. **Pipeline** (cluster_reduction → cluster_decomposition → inner solve)
///    with kernelization disabled (already done). The inner solver retries
///    Whidden on every sub-instance so the path is available at every
///    recursion level.
fn solve_recursive(instance: &Instance, cfg: &BpConfig) -> Option<Vec<Tree>> {
    if instance.trees.is_empty() {
        return None;
    }
    if instance.num_trees() == 1 {
        return Some(instance.trees.clone());
    }
    if instance.num_leaves <= 1 {
        return Some(instance.trees[0..1].to_vec());
    }

    // Kernelize first so Whidden runs on a reduced instance — matching
    // bp-multi's solve_branch_price_multi_cached which kernelizes before
    // trying any decomposition.
    let kern = if cfg.kernelize {
        klados_core::kernelize::kernelize_best(instance, &Default::default())
    } else {
        klados_core::kernelize::KernelizeResult {
            instance: instance.clone(),
            stats: Default::default(),
            reverse_map: (0..=instance.num_leaves).map(|i| i as u32).collect(),
            collapses_original: vec![],
            param_reduction: 0,
            trace: vec![],
        }
    };
    let reduced = &kern.instance;

    let allow_whidden = !matches!(cfg.cluster_algo, ClusterAlgo::None);
    if allow_whidden && reduced.num_trees() == 2 && reduced.num_leaves >= WHIDDEN_MIN_LEAVES {
        let cfg_inner = cfg.clone();
        if let Some(comps) =
            try_whidden_decomp_2tree(reduced, &mut |sub| solve_recursive(sub, &cfg_inner))
        {
            trace!(
                target: LOG_TARGET,
                "whidden strict decomp solved: n={} k={}",
                instance.num_leaves, comps.len(),
            );
            let expanded = klados_core::kernelize::expand_solution(
                comps, &kern, &instance.trees[0], instance.num_leaves,
            );
            return Some(expanded);
        }
    }

    let pipeline_cfg = SolveConfig {
        kernelize: false, // already kernelized above
        kernelize_config: Default::default(),
        cluster_algo: cfg.cluster_algo.clone(),
    };
    let inner_cfg = cfg.clone();
    solve_with_pipeline(
        reduced,
        &pipeline_cfg,
        &mut move |sub: &Instance| -> Option<Vec<Tree>> {
            if allow_whidden && sub.num_trees() == 2 && sub.num_leaves >= WHIDDEN_MIN_LEAVES {
                let cfg2 = inner_cfg.clone();
                if let Some(comps) =
                    try_whidden_decomp_2tree(sub, &mut |s| solve_recursive(s, &cfg2))
                {
                    return Some(comps);
                }
            }
            solver::solve_inner(sub)
        },
    )
    .map(|comps| {
        klados_core::kernelize::expand_solution(
            comps, &kern, &instance.trees[0], instance.num_leaves,
        )
    })
}
