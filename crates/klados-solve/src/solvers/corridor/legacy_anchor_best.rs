//! Reduced-cost corridor solver — the corridor theorem in action.
//!
//! ## The corridor theorem (recap)
//!
//! After root column generation converges with LP value `L` and incumbent
//! `U`, by LP duality every column appearing in any improving integer
//! solution (size `< U`) has reduced cost `rc(c) ≤ γ := U − 1 − L`.
//!
//! ## What this solver does
//!
//! 1. Run root CG to convergence — exactly as B&P would at its root.
//! 2. Compute `γ = U − 1 − L`. If `γ < 0` (i.e. `⌈L⌉ ≥ U`), the
//!    incumbent is already provably optimal.
//! 3. Enumerate every anchor-best column with `rc(c) ≤ γ` ("the
//!    corridor"). For m=2 we use [`super::bp::pricer::exact_pair_dp::
//!    collect_corridor_candidates`]; for m≥3 we currently route to B&P.
//! 4. Add the corridor columns to the pool. They automatically join
//!    every existing cut row via the RMP's standard column-add path.
//! 5. Solve an exact MIP over the enriched pool.
//! 6. If the MIP finds `k < U`, update `U` and loop (γ shrinks).
//! 7. If the MIP cannot improve, `U` is optimal (modulo the
//!    anchor-best caveat noted in the corridor enumerator's doc).
//!
//! Unlike a branch-and-price tree, this solver runs the LP a small
//! number of times total (typically 1–3 outer iterations), not once
//! per B&B node. The big wins are concentrated on m=2 instances with
//! a large `n` and a small LP gap, where B&P's per-node cost dominates.
//!
//! ## What this solver intentionally does *not* do
//!
//! - **No branch-and-price search tree.** Branching is the mechanism
//!   B&P uses to expose corridor columns via dual modulation; we
//!   enumerate them directly.
//! - **No time-capped MIP heuristic substitution.** Each MIP call is
//!   exact (subject to HiGHS's own time controls if any). A real
//!   MIP timeout is reported as such, not silently swapped for a
//!   heuristic answer.
//! - **No silent fallback to B&P on m=2.** For m=2 instances the
//!   corridor pipeline is the algorithm; if it fails the failure is
//!   reported. For m≥3 we route to B&P because no corridor enumerator
//!   exists yet for the multi-tree case — that's a routing decision
//!   on instance class, not a failure fallback.

use std::time::Instant;

use log::debug;

use klados_core::af_validator::{AfValidation, validate_agreement_forest};
use klados_core::kernelize::{expand_solution, kernelize_best};
use klados_core::{Instance, SolverStats, Tree};

use crate::decomp::whidden_cluster::try_whidden_decomp_2tree;
use crate::solvers::bp::BpSolver;
use crate::solvers::bp::column::{AfColumn, ColumnBuilder, ColumnSet};
use crate::solvers::bp::pricer::exact_pair_dp::{ExactPairDpCache, collect_corridor_candidates};
use crate::solvers::bp::pricer::{
    Pricer, PricerScratch, PricingContext, PricingResult, dispatch_by_m,
};
use crate::solvers::bp::rmp::Rmp;
use crate::solvers::bp::search::Branchings;
use crate::solvers::corridor::topk_m2;
use crate::solvers::root_pool::seed_columns_and_incumbent;
use crate::{RunConfig, Solver, Track};

const PRICING_EPS: f64 = 1.0e-8;

/// Tuning knobs for [`CorridorSolver`].
#[derive(Clone, Debug)]
pub struct CorridorConfig {
    /// Bound on root-CG iterations *per outer iter* — a runaway CG would
    /// indicate a pricer bug, not a hard problem. Default is loose.
    pub max_cg_iters: usize,
    /// Bound on outer (γ-shrink) iterations. Each outer iter must either
    /// shrink γ via an improving MIP or exhaust the corridor; the bound
    /// is therefore at most `initial_γ` in practice. Default is loose.
    pub max_outer_iters: usize,
    /// Primal-seed budget for randomized cherry partitions.
    pub seed_budget: usize,
    /// Skip kernelization.
    pub no_kernel: bool,
    /// Corridor enumeration width: `<=1` = legacy anchor-best, `>1` = top-K DP.
    pub topk: usize,
}

impl Default for CorridorConfig {
    fn default() -> Self {
        Self {
            max_cg_iters: 4096,
            max_outer_iters: 64,
            seed_budget: 200,
            no_kernel: false,
            topk: 1,
        }
    }
}

pub struct CorridorSolver {
    stats: SolverStats,
    config: CorridorConfig,
}

impl CorridorSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
            config: CorridorConfig::default(),
        }
    }
}

impl Default for CorridorSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for CorridorSolver {
    type Config = CorridorConfig;
    const SUPPORTED_TRACKS: &'static [Track] = &[Track::Exact];

    fn solve(&mut self, instance: &Instance, cfg: &RunConfig<Self::Config>) -> Option<Vec<Tree>> {
        self.config = cfg.specific.clone();
        // m≥3 is not yet supported by the corridor enumerator. Route to
        // B&P as a separate algorithm — this is not a fallback, it's a
        // routing decision: the corridor algorithm is undefined for m≥3
        // until the threshold-K DSSR enumerator is built.
        if instance.num_trees() >= 3 {
            return Solver::solve(&mut BpSolver::new(), instance, &RunConfig::default());
        }
        if instance.num_trees() <= 1 || instance.num_leaves <= 1 {
            return Some(instance.trees.clone());
        }
        self.solve_m2(instance).map(|(forest, _)| forest)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

impl CorridorSolver {
    fn solve_m2(&mut self, instance: &Instance) -> Option<(Vec<Tree>, bool)> {
        // Standard kernelization first.
        let kern = if !self.config.no_kernel {
            kernelize_best(instance, &Default::default())
        } else {
            klados_core::kernelize::KernelizeResult {
                instance: instance.clone(),
                stats: Default::default(),
                reverse_map: (0..=instance.num_leaves).collect(),
                collapses_original: vec![],
                param_reduction: 0,
                trace: vec![],
            }
        };
        let reduced = &kern.instance;
        if reduced.num_trees() != 2 || reduced.num_leaves <= 1 {
            // Kernelization fully resolved the instance → certified.  Return
            // the expanded trivial reduced solution, not the original input
            // trees (which would duplicate leaves in the output forest).
            let reduced_solution = if reduced.num_leaves == 0 {
                Vec::new()
            } else {
                vec![reduced.trees[0].clone()]
            };
            let expanded = expand_solution(
                reduced_solution,
                &kern,
                instance.reference_tree(),
                instance.num_leaves,
            );
            self.stats.upper_bound = Some(expanded.len());
            self.stats.lower_bound = expanded.len();
            return Some((expanded, true));
        }

        // Whidden strict cluster decomposition splits big 2-tree instances
        // into independent sub-problems. Each sub-problem gets its own
        // corridor solve — this is where the corridor really shines on
        // huge n=2000+ instances, which decompose into many small
        // clusters.
        if reduced.num_leaves >= 20 {
            let mut sub_failed = false;
            // The decomposed solve is a certified optimum only if EVERY
            // sub-problem certified; one unproven sub poisons the whole.
            let mut all_certified = true;
            let mut solve_sub = |sub: &Instance| {
                let inner = CorridorSolver {
                    stats: SolverStats::default(),
                    config: self.config.clone(),
                };
                if let Some((comps, cert)) = inner.solve_m2_core(sub) {
                    all_certified &= cert;
                    Some(comps)
                } else {
                    sub_failed = true;
                    None
                }
            };
            if let Some(forest) = try_whidden_decomp_2tree(
                reduced,
                &mut solve_sub,
                &crate::decomp::whidden_cluster::NEVER_TERMINATE,
            ) && !sub_failed
            {
                let expanded = expand_solution(
                    forest,
                    &kern,
                    instance.reference_tree(),
                    instance.num_leaves,
                );
                self.stats.upper_bound = Some(expanded.len());
                self.stats.lower_bound = expanded.len();
                return Some((expanded, all_certified));
            }
        }

        let (reduced_forest, certified) = self.solve_m2_core(reduced)?;
        let expanded = expand_solution(
            reduced_forest,
            &kern,
            instance.reference_tree(),
            instance.num_leaves,
        );
        self.stats.upper_bound = Some(expanded.len());
        self.stats.lower_bound = expanded.len();
        Some((expanded, certified))
    }

    /// Exact-track-safe 2-tree entry: returns the forest **only** when the
    /// corridor closed and proved optimality (`lb >= ub`). Returns `None`
    /// when corridor could only produce an unproven incumbent (which may be
    /// suboptimal — e.g. pub012 returned 224 vs opt 223), so the caller must
    /// fall back to another exact engine. This makes m=2 → corridor routing
    /// sound for the Exact track.
    pub fn solve_m2_certified(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        if instance.num_trees() != 2 {
            return None;
        }
        match self.solve_m2(instance) {
            Some((forest, true)) => Some(forest),
            _ => None,
        }
    }

    /// Core corridor solve on a kernelized, undecomposable 2-tree
    /// sub-instance. Returns `Some((forest, certified))` where `certified`
    /// is `true` only when the LP lower bound met the incumbent (`lb >= ub`)
    /// — a real optimality proof. When the corridor exhausts / aborts without
    /// closing the gap, the forest is the best incumbent found but may be
    /// suboptimal, so `certified` is `false`. `None` on setup failure.
    fn solve_m2_core(&self, instance: &Instance) -> Option<(Vec<Tree>, bool)> {
        debug_assert_eq!(instance.num_trees(), 2);
        let started = Instant::now();
        let trees = &instance.trees;
        let n = instance.num_leaves as usize;

        // ── Setup: seeds + initial pool + RMP ───────────────────────
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

        // ── Outer loop: CG → corridor → MIP → γ-shrink ─────────────
        let mut outer = 0usize;
        let mut total_cg_iters = 0usize;
        let mut total_corridor_added = 0usize;

        loop {
            // Safety bound: outer iter count. Each outer iter either
            // shrinks γ via an improving MIP or exhausts the corridor,
            // so the bound is essentially the initial γ; the cap here
            // just catches infinite loops from a bug.
            if outer >= self.config.max_outer_iters {
                debug!(
                    "[corridor] outer-cap n={} k={} cols={} outer={} ms={:.0}",
                    n,
                    best_cols.len(),
                    columns.len(),
                    outer,
                    started.elapsed().as_secs_f64() * 1000.0,
                );
                // outer-cap (safety, runaway): incumbent unproven.
                return assemble_forest(instance, &best_cols).map(|f| (f, false));
            }
            outer += 1;

            // Phase 1: standard CG (rc < 0 columns only) — runs to
            // convergence with no internal time limit; the safety bound
            // on iter count just catches a runaway pricer.
            let (lp_obj, lp_converged, cg_iters, _cuts) = run_cg(
                trees,
                n,
                &mut rmp,
                &mut columns,
                &mut seen,
                &mut pricer,
                &mut scratch,
                &branchings,
                self.config.max_cg_iters,
            )?;
            total_cg_iters += cg_iters;

            if !lp_converged {
                debug!(
                    "[corridor] cg-not-converged outer={} lp={:.4} cg_iters={}",
                    outer, lp_obj, cg_iters,
                );
                // CG didn't converge: LP bound not valid, incumbent unproven.
                return assemble_forest(instance, &best_cols).map(|f| (f, false));
            }

            // Update incumbent via LP rounding before computing γ —
            // a tighter U gives a smaller corridor.
            if let Some(lp_sol) = solve_lp(&mut rmp)
                && let Some(rounded) = lp_round(&columns, &lp_sol.column_values, n)
                && rounded.len() < best_cols.len()
            {
                best_cols = rounded;
            }

            let upper = best_cols.len();
            let lb = (lp_obj - 1.0e-6).ceil() as usize;
            if lb >= upper {
                // Certified optimal: LP lower bound matches incumbent.
                debug!(
                    "[corridor] certified-lp-bound n={} k={} lp={:.4} lb={} outer={} cg={} corridor_added={} ms={:.0}",
                    n,
                    upper,
                    lp_obj,
                    lb,
                    outer,
                    total_cg_iters,
                    total_corridor_added,
                    started.elapsed().as_secs_f64() * 1000.0,
                );
                // Certified optimal: LP lower bound matches incumbent.
                return assemble_forest(instance, &best_cols).map(|f| (f, true));
            }

            // Phase 2+3: corridor enumeration via iterated anchor-cutting,
            // alternating with MIP solves. The anchor-best DP returns at
            // most one column per anchor; to enumerate the wider corridor
            // we add the found anchors to `forbidden_anchors` and re-run
            // the DP, which exposes the next anchor-best at each
            // surviving anchor. Loop terminates when either:
            //   (a) the MIP finds an improving incumbent (→ shrink γ),
            //   (b) the DP reports no new column with score ≥ 1−γ
            //       (→ corridor exhausted under this enumeration),
            //   (c) the wall budget is exhausted.
            let gamma = (upper as f64) - 1.0 - lp_obj;
            // Fresh duals for the corridor enumeration.
            let lp_sol = solve_lp(&mut rmp)?;
            let alpha = lp_sol.leaf_duals.clone();
            let beta = lp_sol.node_duals.clone();

            let mut forbidden_anchors: Vec<(u32, u32)> = Vec::new();
            let mut improved_this_outer = false;
            let mut inner_pass = 0usize;
            loop {
                inner_pass += 1;
                let mut cache = scratch
                    .exact_dp_cache
                    .take()
                    .filter(|c| c.fits(trees[0].num_nodes(), trees[1].num_nodes(), n))
                    .unwrap_or_else(|| {
                        ExactPairDpCache::new(trees[0].num_nodes(), trees[1].num_nodes(), n)
                    });
                let ctx = PricingContext {
                    trees,
                    num_leaves: n,
                    alpha: &alpha,
                    beta: &beta,
                    columns: &columns,
                    seen: &seen,
                    branchings: &branchings,
                    terminate: &crate::solvers::bp::pricer::NEVER_TERMINATE,
                    deadline: None,
                    restrict_side: None,
                    clean_cut: None,
                };
                // Corridor enumeration: either anchor-best (legacy
                // behaviour, K=1) or top-K threshold DP. The choice is
                // controlled by `CorridorConfig.topk` so we can A/B
                // test the new oracle.
                let topk = self.config.topk;
                let candidates: Vec<(f64, Vec<u32>, u32, u32)> = if topk <= 1 {
                    let corridor =
                        collect_corridor_candidates(&ctx, &mut cache, gamma, &forbidden_anchors);
                    scratch.exact_dp_cache = Some(cache);
                    corridor
                        .candidates
                        .into_iter()
                        .map(|c| (c.score, c.labels, c.anchor0, c.anchor1))
                        .collect()
                } else {
                    scratch.exact_dp_cache = Some(cache);
                    let mut tk_cache = scratch
                        .topk_dp_cache
                        .take()
                        .filter(|c| c.fits(trees[0].num_nodes(), trees[1].num_nodes(), n))
                        .unwrap_or_else(|| {
                            topk_m2::TopKDpCache::new(trees[0].num_nodes(), trees[1].num_nodes(), n)
                        });
                    let cols = topk_m2::enumerate_corridor(
                        &topk_m2::CorridorInput {
                            t0: &trees[0],
                            t1: &trees[1],
                            alpha: &alpha,
                            beta_t0: &beta[0],
                            beta_t1: &beta[1],
                            threshold: 1.0 - gamma - 1.0e-8,
                            max_k: topk,
                        },
                        &mut tk_cache,
                    );
                    scratch.topk_dp_cache = Some(tk_cache);
                    cols.into_iter()
                        .filter(|c| {
                            !forbidden_anchors
                                .iter()
                                .any(|&(a0, a1)| a0 == c.anchor0 && a1 == c.anchor1)
                        })
                        .map(|c| (c.score, c.labels, c.anchor0, c.anchor1))
                        .collect()
                };

                let mut newly_added = 0usize;
                for (_score, labels, anchor0, anchor1) in candidates {
                    forbidden_anchors.push((anchor0, anchor1));
                    if seen.insert(labels.clone())
                        && let Some(col) = builder.try_build(labels, trees)
                    {
                        rmp.add_column(&col);
                        columns.push(col);
                        newly_added += 1;
                    }
                }
                total_corridor_added += newly_added;
                debug!(
                    "[corridor] outer={} pass={} lp={:.4} U={} γ={:.3} new={} pool={} forbidden={}",
                    outer,
                    inner_pass,
                    lp_obj,
                    upper,
                    gamma,
                    newly_added,
                    columns.len(),
                    forbidden_anchors.len(),
                );

                if newly_added == 0 {
                    // Corridor exhausted under anchor-cut enumeration.
                    // Run MIP one final time — if it doesn't improve U,
                    // this anchor-best corridor cannot reach a better
                    // integer solution. (Caveat: still incomplete vs
                    // top-K-per-anchor; documented limitation.)
                    let mip = solve_mip_to_optimality(&mut rmp, &columns, n);
                    if let Some(labels) = mip
                        && labels.len() < best_cols.len()
                    {
                        best_cols = labels;
                        improved_this_outer = true;
                    }
                    break;
                }

                // Run MIP after each enrichment batch. If it improves
                // U we restart from outer (γ shrinks → tighter corridor).
                let mip = solve_mip_to_optimality(&mut rmp, &columns, n);
                if let Some(labels) = mip
                    && labels.len() < best_cols.len()
                {
                    best_cols = labels;
                    improved_this_outer = true;
                    break;
                }
            }

            if improved_this_outer {
                continue;
            }

            // Corridor enumeration was exhausted (all anchor-bests at
            // this γ examined) and no MIP improvement found. Under the
            // anchor-best enumeration this is the best we can certify;
            // returning the current best forest.
            debug!(
                "[corridor] corridor-exhausted n={} k={} γ={:.3} corridor_total={} pool={}",
                n,
                best_cols.len(),
                gamma,
                total_corridor_added,
                columns.len(),
            );
            // Corridor exhausted without closing the gap: incumbent unproven.
            return assemble_forest(instance, &best_cols).map(|f| (f, false));
        }
    }
}

fn assemble_forest(instance: &Instance, label_sets: &[Vec<u32>]) -> Option<Vec<Tree>> {
    let forest: Vec<Tree> = label_sets
        .iter()
        .map(|block| {
            if block.len() == 1 {
                Tree::singleton(block[0], instance.num_leaves)
            } else {
                let mut leafset =
                    fixedbitset::FixedBitSet::with_capacity(instance.num_leaves as usize + 1);
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
        .collect();
    match validate_agreement_forest(instance, &forest) {
        AfValidation::Ok => Some(forest),
        _ => None,
    }
}

fn run_cg<P: Pricer>(
    trees: &[Tree],
    n: usize,
    rmp: &mut Rmp,
    columns: &mut Vec<AfColumn>,
    seen: &mut ColumnSet,
    pricer: &mut P,
    scratch: &mut PricerScratch,
    branchings: &Branchings,
    max_iters: usize,
) -> Option<(f64, bool, usize, usize)> {
    let mut cg_iters = 0usize;
    let mut cuts_total = 0usize;
    let mut final_obj = 0.0;
    let mut converged = false;

    while cg_iters < max_iters {
        let lp = match rmp.solve() {
            Ok(lp) => lp,
            Err(_) => return None,
        };
        cg_iters += 1;
        final_obj = lp.objective;

        let cuts = rmp.separate_and_add_cuts(columns, &lp.column_values, 1.0e-6);
        if cuts > 0 {
            cuts_total += cuts;
            continue;
        }

        let result = pricer.price(
            &PricingContext {
                trees,
                num_leaves: n,
                alpha: &lp.leaf_duals,
                beta: &lp.node_duals,
                columns,
                seen,
                branchings,
                terminate: &crate::solvers::bp::pricer::NEVER_TERMINATE,
                deadline: None,
                restrict_side: None,
                clean_cut: None,
            },
            scratch,
        );
        match result {
            PricingResult::Found(cols) => {
                for c in cols {
                    if seen.insert(c.labels().to_vec()) {
                        rmp.add_column(&c);
                        columns.push(c);
                    }
                }
            }
            PricingResult::Converged | PricingResult::Improving => {
                converged = true;
                break;
            }
        }
    }
    Some((final_obj, converged, cg_iters, cuts_total))
}

fn solve_lp(rmp: &mut Rmp) -> Option<crate::solvers::bp::rmp::RmpSolution> {
    rmp.solve().ok()
}

fn lp_round(columns: &[AfColumn], values: &[f64], n: usize) -> Option<Vec<Vec<u32>>> {
    // Same greedy disjoint cover as root_pool's `round_lp`. Kept local
    // to avoid pub coupling.
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
    let mut used_leaves = fixedbitset::FixedBitSet::with_capacity(n + 1);
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
        .map(fixedbitset::FixedBitSet::with_capacity)
        .collect::<Vec<_>>();

    let mut out: Vec<Vec<u32>> = Vec::new();
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

fn solve_mip_to_optimality(rmp: &mut Rmp, columns: &[AfColumn], n: usize) -> Option<Vec<Vec<u32>>> {
    // Exact MIP solve with lazy cut separation. We loop until either:
    //   (a) the MIP optimum is integer-feasible with no node-cover
    //       violations, or
    //   (b) the MIP reports infeasibility (returns None).
    // Pass `f64::INFINITY` as the HiGHS time_limit so the MIP runs to
    // optimality — the corridor solver as a whole has no internal time
    // budget, matching an exact-solver contract.
    let mut best: Option<Vec<Vec<u32>>> = None;
    for _ in 0..16 {
        let mip = rmp
            .solve_mip_with_time_limit(f64::INFINITY)
            .ok()
            .flatten()?;
        let cuts = rmp.separate_and_add_cuts(columns, &mip.column_values, 0.5);
        if cuts > 0 {
            continue;
        }
        if let Some(labels) = integral_solution(columns, &mip.column_values, n) {
            best = Some(labels);
        }
        break;
    }
    best
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

#[allow(dead_code)]
fn _ensure_pricing_eps_unused() -> f64 {
    PRICING_EPS
}
