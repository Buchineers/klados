//! Whidden's FPT algorithm for rSPR distance (2-tree MAF).
//!
//! Ports the core branching strategy from rspr (Whidden & Zeh) to klados's
//! arena-based tree representation. Uses 3-way branching on sibling pairs
//! with COB/RCOB optimizations and edge protection to achieve O(2^k · n).
//!
//! Currently restricted to m=2 (two input trees).

mod algorithm;
mod search_state;

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
        assert_eq!(instance.num_trees(), 2, "Whidden solver requires exactly 2 trees, got {}", instance.num_trees());

        if instance.num_leaves <= 1 {
            return Some(vec![instance.trees[0].clone()]);
        }

        let config = KernelizeConfig::default();
        let kern = kernelize::kernelize_best(instance, &config);
        let reduced = &kern.instance;

        eprintln!("[whidden-solve] n={} n_reduced={}", instance.num_leaves, reduced.num_leaves);

        // Try cluster decomposition on the reduced instance.
        match crate::cluster_reduction::try_cluster_reduction(reduced, &mut |subinstance| {
            eprintln!("[whidden-solve] cluster sub-instance: n={} m={}", subinstance.num_leaves, subinstance.num_trees());
            let mut sub_solver = WhiddenSolver::new();
            let result = WhiddenSolver::solve(&mut sub_solver, subinstance);
            eprintln!("[whidden-solve] cluster sub-result: {:?} components", result.as_ref().map(|r| r.len()));
            result
        })? {
            crate::cluster_reduction::ClusterReductionResult::NotApplicable => {
                eprintln!("[whidden-solve] cluster reduction: not applicable");
            }
            crate::cluster_reduction::ClusterReductionResult::Solved(solution) => {
                eprintln!("[whidden-solve] cluster reduction solved: {} components", solution.components.len());
                return Some(kernelize::expand_solution(
                    solution.components,
                    &kern,
                    &instance.trees[0],
                    instance.num_leaves,
                ));
            }
        }

        eprintln!("[whidden-solve] running algorithm on n={}", reduced.num_leaves);
        let reduced_result = algorithm::solve(reduced, &mut self.stats);
        eprintln!("[whidden-solve] algorithm result: {:?} components", reduced_result.as_ref().map(|r| r.len()));
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
