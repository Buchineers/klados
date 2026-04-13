//! Whidden's FPT algorithm for rSPR distance (2-tree MAF).
//!
//! Faithful port of rspr (Whidden & Zeh) using SoA arrays with physical
//! tree mutations (matching rspr's cut_parent / contract semantics).
//! Restricted to m=2 (two input trees).

mod algorithm;
mod stats;

pub use algorithm::BBConfig;
pub use algorithm::approx_2_lb_for_instance;
pub use algorithm::approx_3_for_instance;
pub use stats::{WhiddenProgressUpdate, WhiddenRuleStats, WhiddenRunStats};

use std::time::Instant;

use klados_core::{Instance, SolverStats, Tree};

use crate::kernelize::{self, KernelizeConfig};

pub struct WhiddenSolver {
    stats: SolverStats,
    rule_stats: WhiddenRuleStats,
    bb_config: BBConfig,
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
            rule_stats: WhiddenRuleStats::default(),
            bb_config: BBConfig::default(),
        }
    }

    pub fn with_bb_2approx(mut self, enabled: bool) -> Self {
        self.bb_config.bb_2approx = enabled;
        self
    }

    pub fn with_bb(mut self, enabled: bool) -> Self {
        self.bb_config.bb = enabled;
        self
    }

    pub fn with_tt_enabled(mut self, enabled: bool) -> Self {
        self.bb_config.tt_enabled = enabled;
        self
    }

    pub fn with_tt_prune(mut self, enabled: bool) -> Self {
        self.bb_config.tt_prune = enabled;
        self
    }

    pub fn with_tt_size_log2(mut self, size_log2: u8) -> Self {
        self.bb_config.tt_size_log2 = size_log2;
        self
    }

    pub fn with_bound_cache_enabled(mut self, enabled: bool) -> Self {
        self.bb_config.bound_cache_enabled = enabled;
        self
    }

    pub fn with_bound_cache_size_log2(mut self, size_log2: u8) -> Self {
        self.bb_config.bound_cache_size_log2 = size_log2;
        self
    }

    pub fn with_mestel_rule6(mut self, enabled: bool) -> Self {
        self.bb_config.mestel_rule6 = enabled;
        self
    }

    fn merge_subsolver_stats(&mut self, sub: &WhiddenSolver) {
        self.stats.nodes_explored += sub.stats.nodes_explored;
        self.stats.branches_pruned += sub.stats.branches_pruned;
        self.rule_stats += &sub.rule_stats;
    }

    pub fn rule_stats(&self) -> &WhiddenRuleStats {
        &self.rule_stats
    }

    pub fn solver_stats(&self) -> &SolverStats {
        &self.stats
    }

    pub fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        self.solve_with_progress(instance, None)
    }

    pub fn solve_with_progress(
        &mut self,
        instance: &Instance,
        mut progress: Option<&mut dyn FnMut(WhiddenProgressUpdate)>,
    ) -> Option<Vec<Tree>> {
        // Whidden is 2-tree only; fall back to SAT solver for multi-tree instances.
        if instance.num_trees() != 2 {
            eprintln!(
                "[whidden] m={}, falling back to maf-sat",
                instance.num_trees()
            );
            let mut sat = crate::maf_sat::MafSatSolver::new();
            let result = crate::ExactSolver::solve(&mut sat, instance);
            self.stats = crate::ExactSolver::stats(&sat).clone();
            return result;
        }

        if instance.num_leaves <= 1 {
            return Some(vec![instance.trees[0].clone()]);
        }

        self.stats = SolverStats::default();
        self.rule_stats = WhiddenRuleStats::default();

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
            let mut sub_solver = WhiddenSolver {
                stats: SolverStats::default(),
                rule_stats: WhiddenRuleStats::default(),
                bb_config: self.bb_config.clone(),
            };
            let out = WhiddenSolver::solve(&mut sub_solver, subinstance);
            self.merge_subsolver_stats(&sub_solver);
            out
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

        let started = Instant::now();
        let cb = progress.take();
        let reduced_result = algorithm::solve_with_rule_stats_and_progress(
            reduced,
            &mut self.stats,
            &mut self.rule_stats,
            &self.bb_config,
            cb,
        );
        self.rule_stats.k_total_elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

        reduced_result.map(|components| {
            kernelize::expand_solution(components, &kern, &instance.trees[0], instance.num_leaves)
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
