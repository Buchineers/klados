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

use klados_core::{Instance, SolverStats, Tree};
use klados_core::kernelize::kernelize_best;
use log::info;

use crate::ExactSolver;
use crate::bp::BpSolver;
use crate::root_pool::RootPoolSolver;

pub struct RootCorridorSolver {
    stats: SolverStats,
    max_probe_leaves: usize,
    max_probe_original_2tree: usize,
    trace: bool,
}

impl RootCorridorSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
            max_probe_leaves: env_usize("KLADOS_ROOT_CORRIDOR_MAX_PROBE_LEAVES", 0),
            max_probe_original_2tree: env_usize("KLADOS_ROOT_CORRIDOR_MAX_PROBE_2TREE_LEAVES", 0),
            trace: std::env::var("KLADOS_ROOT_CORRIDOR_TRACE").is_ok(),
        }
    }
}

impl Default for RootCorridorSolver {
    fn default() -> Self { Self::new() }
}

impl ExactSolver for RootCorridorSolver {
    fn name(&self) -> &'static str { "root-corridor" }

    fn description(&self) -> &'static str {
        "Certified root-LP corridor probe with exact B&P fallback"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("KLADOS_ROOT_CORRIDOR_MAX_PROBE_LEAVES", "skip root probe above this leaf count"),
            ("KLADOS_ROOT_CORRIDOR_MAX_PROBE_2TREE_LEAVES", "also probe 2-tree instances up to this original leaf count"),
            ("KLADOS_ROOT_CORRIDOR_PROBE_MS", "soft root probe wall budget in milliseconds"),
            ("KLADOS_ROOT_CORRIDOR_MIP_TIME_LIMIT", "HiGHS pool-MIP time limit per probe pass"),
            ("KLADOS_ROOT_CORRIDOR_TRACE", "print diagnostics"),
        ]
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        let n = instance.num_leaves as usize;

        // The root probe is useful only if the *kernel* is small enough.  Some
        // heuristic-track 2-tree instances have thousands of original leaves
        // but kernelize to a few hundred leaves; those are exactly where the
        // root LP can certify while B&P times out.  Conversely, if the kernel
        // remains large, the LP/MIP setup can eat the whole timeout.
        let probe_leaves = if self.max_probe_leaves == 0 {
            usize::MAX
        } else if instance.num_trees() > 1 && instance.num_leaves > 2 {
            let kern = kernelize_best(instance, &Default::default());
            kern.instance.num_leaves as usize
        } else {
            n
        };

        let allow_2tree_probe =
            instance.num_trees() == 2 && n <= self.max_probe_original_2tree;

        if probe_leaves <= self.max_probe_leaves || allow_2tree_probe {
            let mut root = RootPoolSolver::for_corridor_probe();
            if let Some(out) = root.solve_with_outcome(instance) {
                let k = out.forest.len();
                if let Some(lb) = out.lower_bound {
                    if self.trace {
                        eprintln!(
                            "[root-corridor] probe n={} m={} k={} lb={} conv={} ms={:.1}",
                            n,
                            instance.num_trees(),
                            k,
                            lb,
                            out.converged,
                            out.elapsed.as_secs_f64() * 1000.0,
                        );
                    }
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
                } else if self.trace {
                    eprintln!(
                        "[root-corridor] probe uncertified n={} m={} k={} conv={} ms={:.1}",
                        n,
                        instance.num_trees(),
                        k,
                        out.converged,
                        out.elapsed.as_secs_f64() * 1000.0,
                    );
                }
            }
        } else if self.trace {
            eprintln!(
                "[root-corridor] skip probe n={} kernel_n={} > max_probe_leaves={}",
                n, probe_leaves, self.max_probe_leaves
            );
        }

        let mut bp = BpSolver::new();
        let forest = bp.solve(instance)?;
        self.stats.upper_bound = Some(forest.len());
        self.stats.lower_bound = forest.len();
        Some(forest)
    }

    fn stats(&self) -> &SolverStats { &self.stats }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
