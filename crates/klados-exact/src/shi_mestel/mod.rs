//! Shi et al. (2018) parameterized algorithm for MAF on multiple rooted trees.
//!
//! Implements Alg-Maf from "A parameterized algorithm for the Maximum Agreement
//! Forest problem on multiple rooted multifurcating trees" (JCSS 97, 2018).

mod algorithm;
mod branching;
mod decomposition;
mod extraction;
mod forest_nav;
pub(crate) mod preprocessing;
mod reduction;
mod search_state;
mod split;
mod transposition;
mod utils;

use klados_core::{Instance, SolverStats, Tree};

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

        let subtree_collapses =
            preprocessing::find_common_subtrees(&instance.trees, instance.num_leaves);
        if !subtree_collapses.is_empty() {
            let removed_count: usize = subtree_collapses.iter().map(|(_, r)| r.len()).sum();
            trace!(
                "subtree reduction: {} maximal common subtrees, removing {} labels ({} -> {} effective leaves)",
                subtree_collapses.len(),
                removed_count,
                instance.num_leaves,
                instance.num_leaves as usize - removed_count,
            );
            let (reduced, reverse_map) =
                preprocessing::reduce_instance(instance, &subtree_collapses);
            let reduced_result = self.solve_inner(&reduced);
            return reduced_result.map(|components| {
                preprocessing::expand_solution(
                    components,
                    &subtree_collapses,
                    &reverse_map,
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
