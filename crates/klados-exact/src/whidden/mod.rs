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

std::thread_local! {
    static SPLIT_DIAG_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

struct SplitDiagDepthDecrement;
impl Drop for SplitDiagDepthDecrement {
    fn drop(&mut self) {
        SPLIT_DIAG_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Prints the SPLIT diagnostic on drop. Holds a raw pointer to the
/// solver's rule_stats so we can read final values regardless of which
/// return path was taken. Safe because the guard lives strictly within
/// the call frame that owns the WhiddenSolver.
struct SplitDiagPrinter {
    stats_ptr: *const WhiddenRuleStats,
    n_input: u32,
}
impl Drop for SplitDiagPrinter {
    fn drop(&mut self) {
        let rs = unsafe { &*self.stats_ptr };
        let n = rs.split_diag_nodes.max(1);
        let avg_blocks = rs.split_diag_disjoint_blocks_sum as f64
            / rs.split_diag_disjoint.max(1) as f64;
        let avg_max_block = rs.split_diag_disjoint_max_block_sum as f64
            / rs.split_diag_disjoint.max(1) as f64;
        eprintln!(
            "[split-diag] n_input={} sampled={} overlap={} ({:.1}%) disjoint={} ({:.1}%) single={} ({:.1}%) avg_blocks={:.2} avg_max_block={:.1}",
            self.n_input,
            rs.split_diag_nodes,
            rs.split_diag_overlap,
            100.0 * rs.split_diag_overlap as f64 / n as f64,
            rs.split_diag_disjoint,
            100.0 * rs.split_diag_disjoint as f64 / n as f64,
            rs.split_diag_single_component,
            100.0 * rs.split_diag_single_component as f64 / n as f64,
            avg_blocks,
            avg_max_block,
        );
        // Day 6: report SPLIT rule entry-point firing (only if checked).
        if rs.split_rule_checked > 0 {
            let avg_core_cut = rs.split_rule_core_edges as f64
                / rs.split_rule_core_cutsets.max(1) as f64;
            eprintln!(
                "[split-rule] checked={} overlap_found={} ({:.1}%) disjoint_found={} ({:.1}%) applied={} core_cutsets={} size1={} ({:.1}%) avg_cut_size={:.2}",
                rs.split_rule_checked,
                rs.split_rule_overlap_found,
                100.0 * rs.split_rule_overlap_found as f64 / rs.split_rule_checked as f64,
                rs.split_rule_disjoint_found,
                100.0 * rs.split_rule_disjoint_found as f64 / rs.split_rule_checked as f64,
                rs.split_rule_applied,
                rs.split_rule_core_cutsets,
                rs.split_rule_size1_cutsets,
                100.0 * rs.split_rule_size1_cutsets as f64
                    / rs.split_rule_core_cutsets.max(1) as f64,
                avg_core_cut,
            );
        }
    }
}

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
        let mut bb_config = BBConfig::default();
        // Env-var gate for the new split-or-decompose path (Day 6+).
        if std::env::var("KLADOS_WHIDDEN_SPLIT_OR_DECOMPOSE").is_ok() {
            bb_config.use_split_or_decompose = true;
        }
        Self {
            stats: SolverStats::default(),
            rule_stats: WhiddenRuleStats::default(),
            bb_config,
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

    pub fn with_split_or_decompose(mut self, enabled: bool) -> Self {
        self.bb_config.use_split_or_decompose = enabled;
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
        let entry_depth = SPLIT_DIAG_DEPTH.with(|d| {
            let v = d.get();
            d.set(v + 1);
            v
        });
        let _decr = SplitDiagDepthDecrement;
        let is_outermost = entry_depth == 0;
        // Diagnostic print runs unconditionally on outermost exit via Drop.
        let _diag_guard = if is_outermost && std::env::var("KLADOS_WHIDDEN_SPLIT_DIAG").is_ok() {
            Some(SplitDiagPrinter {
                stats_ptr: &self.rule_stats as *const _,
                n_input: instance.num_leaves,
            })
        } else {
            None
        };
        // Whidden is 2-tree only; fall back to SAT solver for multi-tree instances.
        if instance.num_trees() != 2 {
            eprintln!("[whidden] m={}, falling back to sat", instance.num_trees());
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

        // Diag print happens via _diag_guard Drop on return.
        let _ = is_outermost;

        reduced_result.map(|components| {
            kernelize::expand_solution(components, &kern, &instance.trees[0], instance.num_leaves)
        })
    }
}

impl super::ExactSolver for WhiddenSolver {
    fn name(&self) -> &'static str {
        "whidden"
    }

    fn description(&self) -> &'static str {
        "Whidden 3-way branch-and-bound (2-tree only)"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("KLADOS_WHIDDEN_BATCH_STRICT", "use strict cluster point detection"),
            ("KLADOS_WHIDDEN_RSPR_GREEDY", "use greedy rSPR decomposition"),
            ("KLADOS_WHIDDEN_DEBUG", "enable debug output"),
        ]
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        WhiddenSolver::solve(self, instance)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}
