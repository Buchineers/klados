//! Root-price-and-cover prototype.
//!
//! This is deliberately **not** branch-and-price: it runs only the root column
//! generation loop, then solves one integer cover over the generated pool
//! (with the same lazy node rows used by the RMP).  If the pool MIP times out,
//! it returns the best safe rounding/seed incumbent.
//!
//! The experiment tests whether the useful part of B&P is mostly:
//!   incumbent → sparse duals → tiny relevant column pool → one integer cover
//! rather than the branch tree itself.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use fixedbitset::FixedBitSet;
use klados_core::af_validator::{AfValidation, validate_agreement_forest};
use klados_core::kernelize::{expand_solution, kernelize_best};
use klados_core::lower_bound::{best_randomized_partition, pairwise_refine_ub};
use klados_core::{Instance, SolverStats, Tree};
use log::debug;

use crate::decomp::whidden_cluster::try_whidden_decomp_2tree;
use crate::solvers::bp::column::{AfColumn, ColumnBuilder, ColumnSet};
use crate::solvers::bp::pricer::exact_pair_dp::collect_corridor_candidates_ref;
use crate::solvers::bp::pricer::{
    Pricer, PricerScratch, PricingContext, PricingResult, dispatch_by_m,
};
use crate::solvers::bp::rmp::{Rmp, RmpSolution};
use crate::solvers::bp::search::Branchings;
use crate::solvers::chen_rspr::chen_pair_agreement;

/// Tuning knobs for [`RootPoolSolver`].
#[derive(Clone, Debug)]
pub struct RootPoolConfig {
    pub max_cg_iters: usize,
    pub max_wall_ms: u64,
    pub mip_passes: usize,
    pub mip_time_limit: f64,
    pub seed_budget: usize,
    pub no_kernel: bool,
    pub probe_ms: u64,
    pub probe_mip_time_limit: f64,
    pub dump_core: bool,
    /// Path to append the core-dump TSV row to.
    pub core_dump_file: String,
    pub support_only: bool,
    pub shell_only: bool,
    pub enumerate_shell: bool,
    pub lazy_audit: bool,
    pub shell_enum_max_passes: usize,
    pub anchor_k: u32,
}

impl Default for RootPoolConfig {
    fn default() -> Self {
        Self {
            max_cg_iters: 256,
            max_wall_ms: 2000,
            mip_passes: 8,
            mip_time_limit: 2.0,
            seed_budget: 200,
            no_kernel: false,
            probe_ms: 1000,
            probe_mip_time_limit: 0.5,
            dump_core: false,
            core_dump_file: "core_dump.tsv".to_string(),
            support_only: false,
            shell_only: false,
            enumerate_shell: false,
            lazy_audit: false,
            shell_enum_max_passes: 32,
            anchor_k: 8,
        }
    }
}

pub struct RootPoolSolver {
    stats: SolverStats,
    config: RootPoolConfig,
}

pub struct RootPoolOutcome {
    pub forest: Vec<Tree>,
    pub lower_bound: Option<usize>,
    pub converged: bool,
    pub elapsed: Duration,
}

impl RootPoolSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
            config: RootPoolConfig::default(),
        }
    }

    pub(crate) fn for_corridor_probe() -> Self {
        let mut s = Self::new();
        s.config.max_wall_ms = s.config.probe_ms;
        s.config.mip_time_limit = s.config.probe_mip_time_limit;
        s
    }

    pub(crate) fn solve_with_outcome(&mut self, instance: &Instance) -> Option<RootPoolOutcome> {
        let total_started = Instant::now();
        if !self.config.no_kernel && instance.num_trees() > 1 && instance.num_leaves > 2 {
            let kern = kernelize_best(instance, &Default::default());
            if kern.instance.num_leaves < instance.num_leaves || kern.param_reduction > 0 {
                if kern.instance.num_trees() == 2 && kern.instance.num_leaves >= 20 {
                    let mut all_certified = true;
                    let mut solve_sub = |sub: &Instance| {
                        let mut solver = RootPoolSolver::new();
                        let out = solver.solve_with_outcome(sub)?;
                        if !matches!(out.lower_bound, Some(lb) if lb >= out.forest.len()) {
                            all_certified = false;
                        }
                        Some(out.forest)
                    };
                    if let Some(reduced) = try_whidden_decomp_2tree(
                        &kern.instance,
                        &mut solve_sub,
                        &crate::decomp::whidden_cluster::NEVER_TERMINATE,
                    ) {
                        let expanded = expand_solution(
                            reduced,
                            &kern,
                            instance.reference_tree(),
                            instance.num_leaves,
                        );
                        self.stats.upper_bound = Some(expanded.len());
                        let k = expanded.len();
                        return Some(RootPoolOutcome {
                            forest: expanded,
                            lower_bound: if all_certified { Some(k) } else { None },
                            converged: all_certified,
                            elapsed: total_started.elapsed(),
                        });
                    }
                }
                let reduced = self.solve_core(&kern.instance)?;
                let expanded = expand_solution(
                    reduced.forest,
                    &kern,
                    instance.reference_tree(),
                    instance.num_leaves,
                );
                self.stats.upper_bound = Some(expanded.len());
                return Some(RootPoolOutcome {
                    forest: expanded,
                    lower_bound: reduced.lower_bound.map(|lb| lb + kern.param_reduction),
                    converged: reduced.converged,
                    elapsed: total_started.elapsed(),
                });
            }
        }
        if instance.num_trees() == 2 && instance.num_leaves >= 20 {
            let mut all_certified = true;
            let mut solve_sub = |sub: &Instance| {
                let mut solver = RootPoolSolver::new();
                let out = solver.solve_with_outcome(sub)?;
                if !matches!(out.lower_bound, Some(lb) if lb >= out.forest.len()) {
                    all_certified = false;
                }
                Some(out.forest)
            };
            if let Some(forest) = try_whidden_decomp_2tree(
                instance,
                &mut solve_sub,
                &crate::decomp::whidden_cluster::NEVER_TERMINATE,
            ) {
                self.stats.upper_bound = Some(forest.len());
                let k = forest.len();
                return Some(RootPoolOutcome {
                    forest,
                    lower_bound: if all_certified { Some(k) } else { None },
                    converged: all_certified,
                    elapsed: total_started.elapsed(),
                });
            }
        }
        self.solve_core(instance)
    }
}

impl Default for RootPoolSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for RootPoolSolver {
    type Config = RootPoolConfig;
    const SUPPORTED_TRACKS: &'static [Track] = &[Track::Heuristic];

    fn solve(&mut self, instance: &Instance, cfg: &RunConfig<Self::Config>) -> Option<Vec<Tree>> {
        self.config = cfg.specific.clone();
        self.solve_with_outcome(instance).map(|out| out.forest)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

impl RootPoolSolver {
    fn solve_core(&mut self, instance: &Instance) -> Option<RootPoolOutcome> {
        let started = Instant::now();
        let trees = &instance.trees;
        let n = instance.num_leaves as usize;
        if trees.is_empty() {
            return None;
        }
        if n == 0 {
            return Some(RootPoolOutcome {
                forest: Vec::new(),
                lower_bound: Some(0),
                converged: true,
                elapsed: started.elapsed(),
            });
        }
        if trees.len() == 1 {
            return Some(RootPoolOutcome {
                forest: vec![trees[0].clone()],
                lower_bound: Some(1),
                converged: true,
                elapsed: started.elapsed(),
            });
        }

        let mut builder = ColumnBuilder::new(trees);
        let initial: Vec<AfColumn> = (1..=n as u32)
            .filter_map(|l| builder.try_build(vec![l], trees))
            .collect();
        if initial.len() != n {
            return None;
        }
        let mut columns = initial.clone();
        let mut seen = ColumnSet::new();
        for c in &columns {
            seen.insert(c.labels().to_vec());
        }
        let mut rmp = Rmp::new(&initial, trees, n);

        let mut best_cols: Vec<Vec<u32>> = (1..=n as u32).map(|l| vec![l]).collect();
        seed_columns_and_incumbent(
            instance,
            self.config.seed_budget,
            &mut builder,
            &mut columns,
            &mut seen,
            &mut rmp,
            &mut best_cols,
        );

        let mut scratch = PricerScratch::new(trees);
        let mut pricer = dispatch_by_m(trees);
        let branchings = Branchings::default();
        let mut final_lp: Option<RmpSolution> = None;
        let mut cg_iters = 0usize;
        let mut added_total = 0usize;
        let mut cuts_total = 0usize;
        let mut converged = false;

        while cg_iters < self.config.max_cg_iters
            && started.elapsed() < Duration::from_millis(self.config.max_wall_ms)
        {
            let lp = match rmp.solve() {
                Ok(lp) => lp,
                Err(_) => break,
            };
            cg_iters += 1;

            let cuts = rmp.separate_and_add_cuts(&columns, &lp.column_values, 1.0e-6);
            if cuts > 0 {
                cuts_total += cuts;
                final_lp = Some(lp);
                continue;
            }

            if let Some(labels) = round_lp(&columns, &lp.column_values, n)
                && labels.len() < best_cols.len()
            {
                best_cols = labels;
            }

            let result = pricer.price(
                &PricingContext {
                    trees,
                    num_leaves: n,
                    alpha: &lp.leaf_duals,
                    beta: &lp.node_duals,
                    columns: &columns,
                    seen: &seen,
                    branchings: &branchings,
                    terminate: &crate::solvers::bp::pricer::NEVER_TERMINATE,
                    deadline: None,
                    restrict_side: None,
                    clean_cut: None,
                },
                &mut scratch,
            );
            final_lp = Some(lp);
            match result {
                PricingResult::Found(cols) => {
                    for c in cols {
                        if seen.insert(c.labels().to_vec()) {
                            rmp.add_column(&c);
                            columns.push(c);
                            added_total += 1;
                        }
                    }
                }
                PricingResult::Converged => {
                    converged = true;
                    break;
                }
                PricingResult::Improving => break,
            }
        }

        let root_lower_bound = if converged {
            final_lp
                .as_ref()
                .map(|lp| (lp.objective - 1.0e-6).ceil() as usize)
        } else {
            None
        };

        // Stage-1 obstruction-core diagnostic. After CG converges, dump
        // LP-state shape so we can decide whether the dual-cert + small-ILP
        // architecture is worth pursuing. A leaf is "core" if no single
        // support column claims it (max x_c < 1 − eps). Core columns are
        // support columns touching any core leaf.
        if self.config.dump_core
            && let Some(lp) = final_lp.as_ref()
        {
            let eps = 1.0e-6;
            let col_lim = columns.len().min(lp.column_values.len());
            let support: Vec<usize> = (0..col_lim)
                .filter(|&i| lp.column_values[i] > eps)
                .collect();
            let mut max_label = n;
            for &ci in &support {
                for &l in columns[ci].labels() {
                    if l as usize > max_label {
                        max_label = l as usize;
                    }
                }
            }
            let mut max_x_per_leaf = vec![0.0f64; max_label + 1];
            for &ci in &support {
                let x = lp.column_values[ci];
                for &l in columns[ci].labels() {
                    let li = l as usize;
                    if x > max_x_per_leaf[li] {
                        max_x_per_leaf[li] = x;
                    }
                }
            }
            let mut core_leaf_mask = vec![false; max_label + 1];
            let mut core_leaves_count = 0usize;
            for l in 0..=max_label {
                if max_x_per_leaf[l] > eps && max_x_per_leaf[l] < 1.0 - eps {
                    core_leaf_mask[l] = true;
                    core_leaves_count += 1;
                }
            }
            let mut core_cols_count = 0usize;
            let mut core_sum_x = 0.0f64;
            let mut core_col_sizes: Vec<usize> = Vec::new();
            for &ci in &support {
                if columns[ci]
                    .labels()
                    .iter()
                    .any(|&l| core_leaf_mask[l as usize])
                {
                    core_cols_count += 1;
                    core_sum_x += lp.column_values[ci];
                    core_col_sizes.push(columns[ci].labels().len());
                }
            }
            let core_col_max_size = core_col_sizes.iter().copied().max().unwrap_or(0);
            let core_col_mean_size = if core_col_sizes.is_empty() {
                0.0
            } else {
                core_col_sizes.iter().sum::<usize>() as f64 / core_col_sizes.len() as f64
            };
            let gap = best_cols.len() as f64 - lp.objective;
            let name = instance.name.as_deref().unwrap_or("?");
            let line = format!(
                "{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{:.4}\t{}\t{:.1}\n",
                name,
                n,
                trees.len(),
                best_cols.len(),
                lp.objective,
                gap,
                support.len(),
                core_leaves_count,
                core_cols_count,
                core_sum_x,
                core_col_max_size,
                core_col_mean_size,
            );
            let path = self.config.core_dump_file.clone();
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                use std::io::Write;
                let _ = f.write_all(line.as_bytes());
            }
        }

        // Experimental "dual-face" mode: integerize only the positive
        // support of the converged root LP.  This is the zero reduced-cost
        // face exposed by the duals, and on hard m>=3 instances it is often
        // a much smaller, much more relevant MIP than the full root pool.
        let support_only = self.config.support_only;
        let shell_only = self.config.shell_only;
        if self.config.enumerate_shell
            && let Some(lp) = final_lp.as_ref()
        {
            let added = expand_shell_anchor_best(
                instance,
                &mut builder,
                &mut columns,
                &mut seen,
                &mut rmp,
                &mut scratch,
                lp,
                best_cols.len(),
                self.config.shell_enum_max_passes,
            );
            debug!("[root-pool] shell-enum added={added}");
        }

        // Lazy-audit shell: iterate
        //   {shell MIP → audit via LazyKBest::next() → add unseen → re-MIP}
        // until LazyKBest reports `None` at the current γ.  When the
        // audit returns no column with `score >= 1 − γ`, the K=1 shell
        // is **certified complete** — every column in any improving
        // integer solution is already in the pool, and the MIP's
        // verdict on that pool is the proven optimum (modulo the
        // within-anchor frontier, which lifts K from 1 to ≥2 in a
        // follow-up commit).
        // Per-pass MIP time limit for the standard solve path; audit can
        // bump this when it enlarges the pool, so the time-per-pass budget
        // tracks the size of the work.
        let mut downstream_mip_time_limit = self.config.mip_time_limit;
        if instance.num_trees() == 2
            && self.config.lazy_audit
            && let Some(lp) = final_lp.as_ref()
        {
            let pool_before = columns.len();
            let added = lazy_audit_shell(
                instance,
                n,
                &mut builder,
                &mut columns,
                &mut seen,
                &mut rmp,
                &mut scratch,
                lp,
                &mut best_cols,
                self.config.mip_passes,
                self.config.mip_time_limit,
                self.config.anchor_k,
            );
            if added > 0 && pool_before > 0 {
                // Scale roughly with pool growth, clamped to a sane upper.
                let factor = (columns.len() as f64 / pool_before as f64).max(1.0);
                downstream_mip_time_limit =
                    (self.config.mip_time_limit * factor).min(self.config.mip_time_limit * 16.0);
            }
            debug!(
                "[root-pool] lazy-audit added={added} mip_tl={:.2}s",
                downstream_mip_time_limit,
            );
        }
        if support_only
            && let Some(lp) = final_lp.as_ref()
            && let Some(labels) = solve_support_face_mip(
                trees,
                n,
                &columns,
                &lp.column_values,
                self.config.mip_passes,
                self.config.mip_time_limit,
            )
            && labels.len() < best_cols.len()
        {
            best_cols = labels;
        }
        if shell_only
            && let Some(lp) = final_lp.as_ref()
            && let Some(labels) = solve_reduced_cost_shell_mip(
                trees,
                n,
                &columns,
                lp,
                best_cols.len(),
                self.config.mip_passes,
                self.config.mip_time_limit,
            )
            && labels.len() < best_cols.len()
        {
            best_cols = labels;
        }

        // If the root LP already meets the incumbent, the pool MIP cannot
        // improve anything.  This is the fast certification case root-corridor
        // is looking for.
        if !support_only
            && !shell_only
            && !matches!(root_lower_bound, Some(lb) if lb >= best_cols.len())
        {
            // One integer cover over the generated pool.  We keep adding
            // violated node rows and re-solving; this is still a single
            // root-pool solve, not a branch tree.
            for _ in 0..self.config.mip_passes {
                if started.elapsed() >= Duration::from_millis(self.config.max_wall_ms) {
                    break;
                }
                let Ok(Some(mip)) = rmp.solve_mip_with_time_limit(downstream_mip_time_limit) else {
                    break;
                };
                let cuts = rmp.separate_and_add_cuts(&columns, &mip.column_values, 0.5);
                if cuts > 0 {
                    cuts_total += cuts;
                    continue;
                }
                if let Some(labels) = integral_solution(&columns, &mip.column_values, n)
                    && labels.len() < best_cols.len()
                {
                    best_cols = labels;
                }
                break;
            }
        }

        let mut forest = labels_to_trees(instance, &best_cols);
        if !matches!(
            validate_agreement_forest(instance, &forest),
            AfValidation::Ok
        ) {
            // Defensive fallback: if a lazy-cut or rounding bug slips through,
            // return the singleton forest rather than an invalid solution.
            forest = (1..=instance.num_leaves)
                .map(|l| Tree::singleton(l, instance.num_leaves))
                .collect();
        }
        self.stats.upper_bound = Some(forest.len());
        let lower_bound = root_lower_bound;
        {
            let lp_obj = final_lp.as_ref().map(|lp| lp.objective).unwrap_or(0.0);
            debug!(
                "[root-pool] n={} m={} k={} cols={} added={} cg={} cuts={} lp={:.3} conv={} ms={:.1}",
                n,
                trees.len(),
                forest.len(),
                columns.len(),
                added_total,
                cg_iters,
                cuts_total,
                lp_obj,
                converged,
                started.elapsed().as_secs_f64() * 1000.0,
            );
            let tiers = pricer
                .tier_timings()
                .into_iter()
                .map(|(name, dur, calls)| {
                    format!("{}={:.1}ms/{}", name, dur.as_secs_f64() * 1000.0, calls)
                })
                .collect::<Vec<_>>()
                .join(" ");
            debug!("[root-pool] tiers {tiers}");
        }
        Some(RootPoolOutcome {
            forest,
            lower_bound,
            converged,
            elapsed: started.elapsed(),
        })
    }
}

/// Lazy-audit shell expansion under **fixed root duals**.
///
/// **What this does end-to-end:**
///
/// 1. Compute γ from the root LP and the current incumbent.
/// 2. Drain `LazyKBest` at threshold `1 − γ` using the **root** duals —
///    not refreshed duals from a re-solved LP. This is critical: the
///    corridor theorem speaks about reduced cost under *root* duals;
///    if we re-solved LP between rounds the duals would shift and the
///    enumeration would generate "new" columns indefinitely without
///    representing actual integer-improving moves.
/// 3. Solve the MIP on the enriched pool to optimality. The MIP's
///    answer is the proven optimum (under K=1 enumeration semantics —
///    within-anchor frontier expansion will lift K to ≥2 later).
///
/// Returns the number of new columns added during the audit.
fn lazy_audit_shell(
    instance: &Instance,
    n: usize,
    builder: &mut ColumnBuilder,
    columns: &mut Vec<AfColumn>,
    seen: &mut ColumnSet,
    rmp: &mut Rmp,
    _scratch: &mut PricerScratch,
    initial_lp: &RmpSolution,
    _best_cols: &mut Vec<Vec<u32>>,
    _mip_passes: usize,
    _mip_time_limit: f64,
    anchor_k: u32,
) -> usize {
    use crate::solvers::corridor::lazy_kbest::{LazyKBest, LazyKBestCache, LazyKBestInput};

    let trees = &instance.trees;
    if trees.len() != 2 {
        return 0;
    }
    let n0 = trees[0].num_nodes();
    let n1 = trees[1].num_nodes();
    let mut lazy_cache = LazyKBestCache::new(n0, n1, n);

    let upper = _best_cols.len();
    let lp_obj = initial_lp.objective;
    let lb = (lp_obj - 1.0e-6).ceil() as usize;
    if lb >= upper {
        debug!(
            "[lazy-audit] pre-certified U={} lp={:.4} lb={} (γ < 0)",
            upper, lp_obj, lb,
        );
        return 0;
    }
    let gamma = (upper as f64) - 1.0 - lp_obj;
    let threshold = 1.0 - gamma - 1.0e-8;

    // Drain the lazy enumerator at threshold under the **root** duals.
    // After this loop terminates, every column with rc ≤ γ under root
    // duals has either (a) been added to the pool, or (b) was already
    // in `seen` from the prior CG. Either way the pool is closed under
    // the K=1 corridor at this γ.
    let mut added = 0usize;
    let mut iter = LazyKBest::new(
        LazyKBestInput {
            t0: &trees[0],
            t1: &trees[1],
            alpha: &initial_lp.leaf_duals,
            beta_t0: &initial_lp.node_duals[0],
            beta_t1: &initial_lp.node_duals[1],
            threshold,
        },
        &mut lazy_cache,
        anchor_k,
    );
    while let Some(col) = iter.next_column() {
        if !seen.insert(col.labels.clone()) {
            continue;
        }
        if let Some(c) = builder.try_build(col.labels, trees) {
            rmp.add_column(&c);
            columns.push(c);
            added += 1;
        }
    }
    drop(iter);
    debug!(
        "[lazy-audit] γ={:.3} U={} new={} pool={}",
        gamma,
        upper,
        added,
        columns.len(),
    );
    let _ = n;
    added
}

fn solve_support_face_mip(
    trees: &[Tree],
    n: usize,
    columns: &[AfColumn],
    values: &[f64],
    mip_passes: usize,
    mip_time_limit: f64,
) -> Option<Vec<Vec<u32>>> {
    let support_ids = values
        .iter()
        .enumerate()
        .filter_map(|(ci, &v)| (v > 1.0e-6).then_some(ci))
        .collect::<Vec<_>>();
    if support_ids.is_empty() {
        return None;
    }
    let support_columns = support_ids
        .iter()
        .map(|&ci| columns[ci].clone())
        .collect::<Vec<_>>();
    let mut rmp = Rmp::new(&support_columns, trees, n);
    for _ in 0..mip_passes {
        let Ok(Some(mip)) = rmp.solve_mip_with_time_limit(mip_time_limit) else {
            return None;
        };
        let cuts = rmp.separate_and_add_cuts(&support_columns, &mip.column_values, 0.5);
        if cuts > 0 {
            continue;
        }
        return integral_solution(&support_columns, &mip.column_values, n);
    }
    None
}

fn solve_reduced_cost_shell_mip(
    trees: &[Tree],
    n: usize,
    columns: &[AfColumn],
    lp: &RmpSolution,
    incumbent_k: usize,
    mip_passes: usize,
    mip_time_limit: f64,
) -> Option<Vec<Vec<u32>>> {
    let gamma = incumbent_k as f64 - 1.0 - lp.objective;
    if gamma < -1.0e-6 {
        return None;
    }
    let shell_ids = columns
        .iter()
        .enumerate()
        .filter_map(|(ci, col)| {
            let rc = 1.0 - col.pricing_score(&lp.leaf_duals, &lp.node_duals);
            (rc <= gamma + 1.0e-6).then_some(ci)
        })
        .collect::<Vec<_>>();
    if shell_ids.is_empty() {
        return None;
    }
    let shell_columns = shell_ids
        .iter()
        .map(|&ci| columns[ci].clone())
        .collect::<Vec<_>>();
    let mut rmp = Rmp::new(&shell_columns, trees, n);
    for _ in 0..mip_passes {
        let Ok(Some(mip)) = rmp.solve_mip_with_time_limit(mip_time_limit) else {
            return None;
        };
        let cuts = rmp.separate_and_add_cuts(&shell_columns, &mip.column_values, 0.5);
        if cuts > 0 {
            continue;
        }
        return integral_solution(&shell_columns, &mip.column_values, n);
    }
    None
}

fn expand_shell_anchor_best(
    instance: &Instance,
    builder: &mut ColumnBuilder,
    columns: &mut Vec<AfColumn>,
    seen: &mut ColumnSet,
    rmp: &mut Rmp,
    scratch: &mut PricerScratch,
    lp: &RmpSolution,
    incumbent_k: usize,
    shell_enum_max_passes: usize,
) -> usize {
    let trees = &instance.trees;
    if trees.len() < 3 {
        return 0;
    }
    let gamma = incumbent_k as f64 - 1.0 - lp.objective;
    if gamma < -1.0e-6 {
        return 0;
    }

    let n0 = trees[0].num_nodes();
    let n1 = trees[1].num_nodes();
    let mut cache = scratch
        .exact_dp_cache
        .take()
        .filter(|c| c.fits(n0, n1, instance.num_leaves as usize))
        .unwrap_or_else(|| {
            crate::solvers::bp::pricer::exact_pair_dp::ExactPairDpCache::new(
                n0,
                n1,
                instance.num_leaves as usize,
            )
        });
    let mut forbidden = Vec::<(u32, u32)>::new();
    let mut added = 0usize;

    for _ in 0..shell_enum_max_passes {
        let output = {
            let branchings = Branchings::default();
            let ctx = PricingContext {
                trees,
                num_leaves: instance.num_leaves as usize,
                alpha: &lp.leaf_duals,
                beta: &lp.node_duals,
                columns,
                seen,
                branchings: &branchings,
                terminate: &crate::solvers::bp::pricer::NEVER_TERMINATE,
                deadline: None,
                restrict_side: None,
                clean_cut: None,
            };
            collect_corridor_candidates_ref(&ctx, &mut cache, 1, gamma, &forbidden)
        };
        if output.candidates.is_empty() {
            break;
        }
        let mut added_forbidden = false;
        for cand in output.candidates {
            added_forbidden |= push_forbidden(&mut forbidden, (cand.anchor0, cand.anchor1));
            if seen.contains(&cand.labels) {
                continue;
            }
            let Some(column) = builder.try_build(cand.labels, trees) else {
                continue;
            };
            let rc = 1.0 - column.pricing_score(&lp.leaf_duals, &lp.node_duals);
            if rc > gamma + 1.0e-6 {
                continue;
            }
            if seen.insert(column.labels().to_vec()) {
                rmp.add_column(&column);
                columns.push(column);
                added += 1;
            }
        }
        if !added_forbidden {
            break;
        }
    }
    scratch.exact_dp_cache = Some(cache);
    added
}

fn push_forbidden(forbidden: &mut Vec<(u32, u32)>, anchor: (u32, u32)) -> bool {
    if forbidden.contains(&anchor) {
        false
    } else {
        forbidden.push(anchor);
        true
    }
}

pub(crate) fn seed_columns_and_incumbent(
    instance: &Instance,
    seed_budget: usize,
    builder: &mut ColumnBuilder,
    columns: &mut Vec<AfColumn>,
    seen: &mut ColumnSet,
    rmp: &mut Rmp,
    best_cols: &mut Vec<Vec<u32>>,
) {
    let trees = &instance.trees;
    let n = instance.num_leaves as usize;
    if trees.len() == 2 && n >= 4 {
        let (_, _, leafsets) = chen_pair_agreement(&trees[0], &trees[1]);
        add_partition_like_seed(trees, leafsets, builder, columns, seen, rmp, best_cols);
    }

    if trees.len() > 2 {
        let refs: Vec<usize> = if trees.len() <= 8 {
            (0..trees.len()).collect()
        } else {
            (0..8).map(|i| i * (trees.len() - 1) / 7).collect()
        };
        let seeds_per_ref = (seed_budget / refs.len().max(1)).max(10);
        let (_ub, part) = best_randomized_partition(trees, &refs, seeds_per_ref);
        add_partition_seed(trees, &part, builder, columns, seen, rmp, best_cols);

        if n <= 140 && trees.len() <= 12 {
            let (_ub, part) = pairwise_refine_ub(trees, n);
            add_partition_seed(trees, &part, builder, columns, seen, rmp, best_cols);
        }
    }
}

fn add_partition_seed(
    trees: &[Tree],
    part: &[usize],
    builder: &mut ColumnBuilder,
    columns: &mut Vec<AfColumn>,
    seen: &mut ColumnSet,
    rmp: &mut Rmp,
    best_cols: &mut Vec<Vec<u32>>,
) {
    let mut map: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    for (leaf_idx, &cid) in part.iter().enumerate() {
        map.entry(cid).or_default().push((leaf_idx + 1) as u32);
    }
    let labels: Vec<Vec<u32>> = map.into_values().collect();
    add_partition_like_seed(trees, labels, builder, columns, seen, rmp, best_cols);
}

fn add_partition_like_seed(
    trees: &[Tree],
    labels: Vec<Vec<u32>>,
    builder: &mut ColumnBuilder,
    columns: &mut Vec<AfColumn>,
    seen: &mut ColumnSet,
    rmp: &mut Rmp,
    best_cols: &mut Vec<Vec<u32>>,
) {
    if labels.is_empty() {
        return;
    }
    let mut built = Vec::with_capacity(labels.len());
    for block in labels {
        let Some(col) = builder.try_build(block, trees) else {
            return;
        };
        built.push(col);
    }
    if built.len() < best_cols.len() {
        *best_cols = built.iter().map(|c| c.labels().to_vec()).collect();
    }
    for col in built {
        if seen.insert(col.labels().to_vec()) {
            rmp.add_column(&col);
            columns.push(col);
        }
    }
}

fn integral_solution(columns: &[AfColumn], values: &[f64], n: usize) -> Option<Vec<Vec<u32>>> {
    let mut cover = vec![0u32; n + 1];
    let mut out = Vec::new();
    for (ci, &v) in values.iter().enumerate() {
        if v <= 0.5 {
            continue;
        }
        let col = &columns[ci];
        for &l in col.labels() {
            cover[l as usize] += 1;
            if cover[l as usize] > 1 {
                return None;
            }
        }
        out.push(col.labels().to_vec());
    }
    if (1..=n).any(|l| cover[l] != 1) {
        return None;
    }
    Some(out)
}

fn round_lp(columns: &[AfColumn], values: &[f64], n: usize) -> Option<Vec<Vec<u32>>> {
    let mut indexed: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v > 1.0e-8)
        .map(|(i, &v)| (i, v))
        .collect();
    if indexed.is_empty() {
        return None;
    }
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| columns[b.0].size().cmp(&columns[a.0].size()))
    });

    let num_trees = columns
        .first()
        .map(|c| c.coverage().iter_per_tree().count())
        .unwrap_or(0);
    let mut used_leaves = FixedBitSet::with_capacity(n + 1);
    let caps: Vec<usize> = (0..num_trees)
        .map(|ti| {
            columns
                .iter()
                .filter_map(|c| c.coverage().iter_per_tree().nth(ti))
                .flat_map(|nodes| nodes.iter().copied())
                .max()
                .map_or(1, |v| v + 1)
        })
        .collect();
    let mut used_nodes = caps
        .into_iter()
        .map(FixedBitSet::with_capacity)
        .collect::<Vec<_>>();

    let mut out = Vec::new();
    for (ci, _) in indexed {
        let c = &columns[ci];
        if c.labels().iter().any(|&l| used_leaves.contains(l as usize)) {
            continue;
        }
        let mut ok = true;
        for (ti, nodes) in c.coverage().iter_per_tree().enumerate() {
            if nodes.iter().any(|&v| used_nodes[ti].contains(v)) {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        for &l in c.labels() {
            used_leaves.insert(l as usize);
        }
        for (ti, nodes) in c.coverage().iter_per_tree().enumerate() {
            for &v in nodes {
                used_nodes[ti].insert(v);
            }
        }
        out.push(c.labels().to_vec());
    }
    for l in 1..=n {
        if !used_leaves.contains(l) {
            out.push(vec![l as u32]);
        }
    }
    Some(out)
}

fn labels_to_trees(instance: &Instance, labels: &[Vec<u32>]) -> Vec<Tree> {
    labels
        .iter()
        .map(|block| {
            if block.len() == 1 {
                Tree::singleton(block[0], instance.num_leaves)
            } else {
                let mut leafset = FixedBitSet::with_capacity(instance.num_leaves as usize + 1);
                for &l in block {
                    leafset.insert(l as usize);
                }
                Tree::component_from_leafset(
                    &leafset,
                    instance.reference_tree(),
                    instance.num_leaves,
                )
            }
        })
        .collect()
}

// ── Unified Solver impl + entry point ───────────────────────────────────────
use crate::{RunConfig, Solver, Track};

pub fn main() {
    crate::run(
        RootPoolSolver::new(),
        RunConfig {
            track: Track::Heuristic,
            specific: RootPoolConfig::default(),
            ..Default::default()
        },
    );
}
