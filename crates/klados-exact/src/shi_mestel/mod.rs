//! Shi et al. (2018) parameterized algorithm for MAF on multiple rooted trees.
//!
//! Implements Alg-Maf from "A parameterized algorithm for the Maximum Agreement
//! Forest problem on multiple rooted multifurcating trees" (JCSS 97, 2018).

mod algorithm;
mod branching;
mod decomposition;
pub(crate) mod extraction;
mod forest_nav;
mod reduction;
mod search_state;
mod split;
mod transposition;
pub(crate) mod utils;

use klados_core::{Instance, SolverStats, Tree};

use crate::kernelize::{self, KernelizeConfig};

fn trace_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("SHI_MESTEL_TRACE").ok().as_deref() == Some("1"))
}

fn profile_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("SHI_MESTEL_PROFILE").ok().as_deref() == Some("1"))
}

macro_rules! trace {
    ($($arg:tt)*) => {
        if crate::shi_mestel::trace_enabled() {
            eprintln!($($arg)*);
        }
    };
}

pub(crate) use trace;

pub struct ShiMestelSolver {
    stats: SolverStats,
}

impl Default for ShiMestelSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ShiMestelSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
        }
    }

    pub fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        if instance.trees.is_empty() {
            return None;
        }
        if instance.num_trees() == 1 {
            return Some(instance.trees.clone());
        }

        let config = KernelizeConfig::default();
        let kern = kernelize::kernelize(instance, &config);

        if kern.stats.reduced_leaves < instance.num_leaves {
            let total =
                kern.stats.subtree_removed + kern.stats.chain_removed + kern.stats.chain32_removed;
            trace!(
                "kernelized: {} → {} leaves ({} removed: {} subtree, {} chain, {} 3-2)",
                instance.num_leaves,
                kern.stats.reduced_leaves,
                total,
                kern.stats.subtree_removed,
                kern.stats.chain_removed,
                kern.stats.chain32_removed,
            );
            let reduced_result = self.solve_inner(&kern.instance);
            return reduced_result.map(|components| {
                kernelize::expand_solution(
                    components,
                    &kern,
                    &instance.trees[0],
                    instance.num_leaves,
                )
            });
        }

        self.solve_inner(instance)
    }

    fn solve_inner(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        algorithm::solve_with_stats(instance, &mut self.stats)
    }
}

impl super::ExactSolver for ShiMestelSolver {
    fn name(&self) -> &'static str {
        "shi-mestel"
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        ShiMestelSolver::solve(self, instance)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests;
