//! Root-corridor exact hybrid.
//!
//! This is the proof-producing wrapper around the root-price-and-cover idea.
//! Today it uses the part we can certify cheaply:
//!
//! * run a bounded root-pool probe;
//! * if the converged root LP lower bound already meets the incumbent, return
//!   the incumbent with an optimality certificate;
//! * otherwise fall back to the production B&P solver, preserving exactness.
//!
//! The intended next step is to replace more fallbacks with a complete
//! reduced-cost-corridor enumerator (`rc <= U-1-L`).  Keeping this as a
//! separate solver lets us benchmark that path without endangering `bp`.

use klados_core::kernelize::kernelize_best;
use klados_core::{Instance, SolverStats, Tree};
use log::{debug, info};

use crate::solvers::bp::BpSolver;
use crate::solvers::root_pool::RootPoolSolver;

/// Tuning knobs for [`RootCorridorSolver`].
#[derive(Clone, Debug, Default)]
pub struct RootCorridorConfig {
    /// Skip the root probe above this kernel leaf count (0 = never probe).
    pub max_probe_leaves: usize,
    /// Also probe 2-tree instances up to this original leaf count.
    pub max_probe_original_2tree: usize,
}

pub struct RootCorridorSolver {
    stats: SolverStats,
    config: RootCorridorConfig,
}

impl RootCorridorSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
            config: RootCorridorConfig::default(),
        }
    }
}

impl Default for RootCorridorSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for RootCorridorSolver {
    type Config = RootCorridorConfig;
    const SUPPORTED_TRACKS: &'static [Track] = &[Track::Exact];

    fn solve(&mut self, instance: &Instance, cfg: &RunConfig<Self::Config>) -> Option<Vec<Tree>> {
        self.config = cfg.specific.clone();
        let n = instance.num_leaves as usize;

        // The root probe is useful only if the *kernel* is small enough.  Some
        // heuristic-track 2-tree instances have thousands of original leaves
        // but kernelize to a few hundred leaves; those are exactly where the
        // root LP can certify while B&P times out.  Conversely, if the kernel
        // remains large, the LP/MIP setup can eat the whole timeout.
        let probe_leaves = if self.config.max_probe_leaves == 0 {
            usize::MAX
        } else if instance.num_trees() > 1 && instance.num_leaves > 2 {
            let kern = kernelize_best(instance, &Default::default());
            kern.instance.num_leaves as usize
        } else {
            n
        };

        let allow_2tree_probe = instance.num_trees() == 2 && n <= self.config.max_probe_original_2tree;

        if probe_leaves <= self.config.max_probe_leaves || allow_2tree_probe {
            let mut root = RootPoolSolver::for_corridor_probe();
            if let Some(out) = root.solve_with_outcome(instance) {
                let k = out.forest.len();
                if let Some(lb) = out.lower_bound {
                    debug!(
                        "[root-corridor] probe n={} m={} k={} lb={} conv={} ms={:.1}",
                        n,
                        instance.num_trees(),
                        k,
                        lb,
                        out.converged,
                        out.elapsed.as_secs_f64() * 1000.0,
                    );
                    if lb >= k {
                        self.stats.upper_bound = Some(k);
                        self.stats.lower_bound = k;
                        info!(
                            "root-corridor certified at root: n={} m={} k={}",
                            n,
                            instance.num_trees(),
                            k,
                        );
                        return Some(out.forest);
                    }
                } else {
                    debug!(
                        "[root-corridor] probe uncertified n={} m={} k={} conv={} ms={:.1}",
                        n,
                        instance.num_trees(),
                        k,
                        out.converged,
                        out.elapsed.as_secs_f64() * 1000.0,
                    );
                }
            }
        } else {
            debug!(
                "[root-corridor] skip probe n={} kernel_n={} > max_probe_leaves={}",
                n, probe_leaves, self.config.max_probe_leaves
            );
        }

        let mut bp = BpSolver::new();
        let forest = crate::Solver::solve(&mut bp, instance, &crate::RunConfig::default())?;
        self.stats.upper_bound = Some(forest.len());
        self.stats.lower_bound = forest.len();
        Some(forest)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

// ── Unified Solver impl + entry point ───────────────────────────────────────
use crate::{RunConfig, Solver, Track};

pub fn main() {
    crate::run(RootCorridorSolver::new(), RunConfig { track: Track::Exact, ..Default::default() });
}
