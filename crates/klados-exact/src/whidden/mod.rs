//! Whidden's FPT algorithm for rSPR distance (2-tree MAF).
//!
//! Faithful port of rspr (Whidden & Zeh) using SoA arrays with physical
//! tree mutations (matching rspr's cut_parent / contract semantics).
//!
//! Phase 1: correct base algorithm without branch-pruning optimizations.
//! Restricted to m=2 (two input trees).

mod forest;
mod undo;
mod algorithm;

use klados_core::{Instance, SolverStats, Tree};

use crate::kernelize::{self, KernelizeConfig};

pub struct WhiddenSolver {
    stats: SolverStats,
}

impl Default for WhiddenSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl WhiddenSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
        }
    }

    pub fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        assert_eq!(instance.num_trees(), 2,
            "Whidden solver requires exactly 2 trees, got {}", instance.num_trees());

        if instance.num_leaves <= 1 {
            return Some(vec![instance.trees[0].clone()]);
        }

        let config = KernelizeConfig::default();
        let kern = kernelize::kernelize_best(instance, &config);
        let reduced = &kern.instance;

        if reduced.num_leaves <= 1 {
            let trivial_components = if reduced.num_leaves == 0 {
                vec![]
            } else {
                vec![reduced.trees[0].clone()]
            };
            return Some(kernelize::expand_solution(
                trivial_components,
                &kern,
                &instance.trees[0],
                instance.num_leaves,
            ));
        }

        // Try cluster decomposition on the reduced instance.
        match crate::cluster_reduction::try_cluster_reduction(reduced, &mut |subinstance| {
            let mut sub_solver = WhiddenSolver::new();
            WhiddenSolver::solve(&mut sub_solver, subinstance)
        })? {
            crate::cluster_reduction::ClusterReductionResult::NotApplicable => {}
            crate::cluster_reduction::ClusterReductionResult::Solved(solution) => {
                return Some(kernelize::expand_solution(
                    solution.components,
                    &kern,
                    &instance.trees[0],
                    instance.num_leaves,
                ));
            }
        }

        let reduced_result = algorithm::solve(reduced, &mut self.stats);
        reduced_result.map(|components| {
            kernelize::expand_solution(
                components,
                &kern,
                &instance.trees[0],
                instance.num_leaves,
            )
        })
    }
}

impl super::ExactSolver for WhiddenSolver {
    fn name(&self) -> &'static str {
        "whidden"
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        WhiddenSolver::solve(self, instance)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}
