//! Branch-and-price search loop.
//!
//! Branches exclusively on leaf pairs (must-link / cannot-link). The column
//! pool grows append-only and is shared across all branches; branchings
//! never reference column ids. See [`crate::solvers::bp::search::branchings`] for
//! the rationale.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use fixedbitset::FixedBitSet;
use klados_core::af_validator::validate_agreement_forest;
use klados_core::lower_bound::{best_randomized_partition, pairwise_refine_ub};
use klados_core::{Instance, Tree};
use log::{debug, info};

use crate::decomp::whidden_cluster::{
    analyze_whidden_decomp_potential, try_whidden_relaxed_incumbent_2tree,
};
use crate::solvers::bp::column::{AfColumn, ColumnBuilder};
use crate::solvers::bp::pricer::{
    Pricer, PricerScratch, PricingContext, PricingResult, dispatch_by_m,
};
use crate::solvers::bp::rmp::Rmp;
use crate::solvers::bp::search::{
    BranchSelector, Branchings, Incumbent, SearchState, SelectionContext, Telemetry,
};
use crate::solvers::chen_rspr::{chen_pair_agreement, chen_pair_bounds};

const LOG_TARGET: &str = "klados::bp";

thread_local! {
    static IN_OBSTRUCTION_PROBE: Cell<bool> = const { Cell::new(false) };
}

struct LocalBounds {
    best_partition: Option<Vec<usize>>,
}

fn sampled_reference_indices(m: usize, limit: usize) -> Vec<usize> {
    if limit >= m {
        return (0..m).collect();
    }
    let mut out = Vec::with_capacity(limit);
    for slot in 0..limit {
        let idx = slot * (m - 1) / (limit - 1).max(1);
        if out.last().copied() != Some(idx) {
            out.push(idx);
        }
    }
    out
}

/// A valid combinatorial lower bound on the MAF component count, from Chen's
/// fast 2-approximation. For m=2 it is the pair's rSPR lower bound; for m≥3
/// the maximum over **every** tree pair — the m-tree MAF must agree with each
/// pair, so its component count is ≥ each pairwise MAF. Conservative on the
/// rSPR-vs-component-count offset (no `+1`) so it can never over-bound.
/// Cheap: each `chen_pair_bounds` is the fast 2-approx, not red-blue.
fn chen_lower_bound(trees: &[Tree], no_chen_lb: bool) -> usize {
    let m = trees.len();
    if m < 2 {
        return 0;
    }
    if no_chen_lb {
        return 0;
    }
    let mut lb = 0usize;
    for i in 0..m {
        for j in (i + 1)..m {
            let (lo, _) = chen_pair_bounds(&trees[i], &trees[j]);
            lb = lb.max(lo);
        }
    }
    lb
}

fn maybe_log_core_decomp_potential(reduced: &Instance, analyze: bool, min_leaves: usize) {
    if !analyze {
        return;
    }
    let n = reduced.num_leaves as usize;
    if n < min_leaves || reduced.num_trees() != 2 {
        return;
    }
    let Some(p) = analyze_whidden_decomp_potential(reduced) else {
        return;
    };
    debug!(
        "BP_CORE_DECOMP\tn={}\tstrict={}\trelaxed={}\tstrict_sel={}\tstrict_clustered={}\tstrict_rem={}\tstrict_largest={}\tbalanced_sel={}\tbalanced_clustered={}\tbalanced_rem={}\tbalanced_largest={}\ttop_strict={:?}\ttop_relaxed={:?}",
        n,
        p.strict_points,
        p.relaxed_points,
        p.strict_selected,
        p.strict_clustered,
        p.strict_remainder,
        p.strict_largest_subproblem,
        p.balanced_selected,
        p.balanced_clustered,
        p.balanced_remainder,
        p.balanced_largest_subproblem,
        p.top_strict_sizes,
        p.top_relaxed_sizes,
    );
}

fn compute_local_bounds(trees: &[Tree], num_leaves: u32) -> LocalBounds {
    if trees.len() <= 2 {
        return LocalBounds {
            best_partition: None,
        };
    }

    let m = trees.len();
    let n = num_leaves as usize;
    // Each greedy run is O(n²·m); pairwise refinement is O(m²·22·n²).
    // We budget total trials: aim for ~200 cherry-greedy runs and only
    // run pairwise refinement when m and n are both moderate (it's the
    // expensive component). The old config (2–5 seeds × 4–m refs ≤ 40
    // trials, deterministic tie-break) systematically missed tighter
    // UBs on hard v2 instances; cut-randomized tie-breaking plus more
    // trials buys real diversity. Total wall ~20-100ms.
    let trial_budget = 200usize;
    let (ref_limit, run_pairwise) = if m >= 20 || n >= 200 {
        (6usize, false)
    } else if m >= 12 || n >= 140 {
        (m.min(8), false)
    } else {
        (m, true)
    };
    let ref_count = ref_limit.min(m).max(1);
    let seed_count = (trial_budget / ref_count).max(20);

    let ref_indices = sampled_reference_indices(m, ref_count);
    let (best_multi_ub, best_partition_multi) =
        best_randomized_partition(trees, &ref_indices, seed_count);

    let best_partition = if run_pairwise {
        let (pr_ub, pr_partition) = pairwise_refine_ub(trees, n);
        if pr_ub < best_multi_ub {
            Some(pr_partition)
        } else {
            Some(best_partition_multi)
        }
    } else {
        Some(best_partition_multi)
    };

    LocalBounds { best_partition }
}

/// Install a feasible AF (returned by a primal heuristic such as Whidden
/// relaxed) as the search incumbent. Adds each component as an `AfColumn`
/// to the pool, marks them x=1 in the (logical) basis, and updates
/// `state.best_ub`. After this, the LP-bound prune fires from the start.
fn install_incumbent(
    state: &mut SearchState,
    rmp: &mut Rmp,
    trees: &[Tree],
    builder: &mut ColumnBuilder,
    forest: Vec<Tree>,
    source: &str,
) {
    let candidate_k = forest.len();
    if candidate_k >= state.best_ub() {
        return;
    }
    let mut component_columns: Vec<usize> = Vec::with_capacity(forest.len());
    for component in &forest {
        let labels: Vec<u32> = component.leaves().collect();
        if labels.is_empty() {
            continue;
        }
        // Find or insert.
        if let Some(existing) = state
            .columns()
            .iter()
            .position(|col| col.labels() == labels.as_slice())
        {
            component_columns.push(existing);
        } else {
            let column = builder.build_unchecked(labels, trees);
            if let Some(id) = state.add_column(column) {
                rmp.add_column(state.columns().last().unwrap());
                component_columns.push(id);
            }
        }
    }
    if component_columns.len() != candidate_k {
        // Shouldn't happen for a valid forest, but if labels collide or
        // dedup conflicts somehow, abort silently — the search will
        // discover the incumbent organically.
        return;
    }
    state.update_incumbent(Incumbent {
        component_columns,
        k: candidate_k,
    });
    log::info!(
        target: LOG_TARGET,
        "primal heuristic ({source}): installed incumbent k={}",
        candidate_k,
    );
}

fn install_partition_incumbent(
    state: &mut SearchState,
    rmp: &mut Rmp,
    trees: &[Tree],
    builder: &mut ColumnBuilder,
    partition: &[usize],
) {
    let mut comp_labels: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    for (leaf_idx, &comp_id) in partition.iter().enumerate() {
        comp_labels
            .entry(comp_id)
            .or_default()
            .push((leaf_idx + 1) as u32);
    }

    let candidate_k = comp_labels.len();
    if candidate_k == 0 || candidate_k >= state.best_ub() {
        return;
    }

    let mut component_columns = Vec::with_capacity(candidate_k);
    for labels in comp_labels.values() {
        if let Some(existing) = state
            .columns()
            .iter()
            .position(|col| col.labels() == labels.as_slice())
        {
            component_columns.push(existing);
            continue;
        }

        let Some(column) = builder.try_build(labels.clone(), trees) else {
            return;
        };
        if let Some(id) = state.add_column(column) {
            rmp.add_column(state.columns().last().unwrap());
            component_columns.push(id);
        } else {
            return;
        }
    }

    if component_columns.len() == candidate_k {
        state.update_incumbent(Incumbent {
            component_columns,
            k: candidate_k,
        });
        log::info!(
            target: LOG_TARGET,
            "local multi-tree UB: installed incumbent k={}",
            candidate_k,
        );
    }
}

enum NodeOutcome {
    Pruned,
    Integral(Incumbent),
    /// LP is fractional; selector produced one or more child branchings.
    /// 2-element Vec is classic must/cannot pair split; longer Vec is
    /// k-way (e.g. 4-way cluster branching on a fractional triple).
    Branch {
        lp_obj: f64,
        children: Vec<Branchings>,
    },
}

/// Decide whether the LP objective allows pruning the current B&B node.
///
/// The RMP is a restriction of the full master LP (same constraints, subset
/// of columns).  In a minimisation problem, restricting variables can only
/// *increase* the optimum, so `RMP_obj ≥ full_master_obj`.  Lazy node-row
/// separation guarantees every violated node constraint is materialised
/// before we read the final objective.  Hence `ceil(RMP_obj) ≥ incumbent` is
/// a sound prune — the integer optimum cannot be lower than the full-master
/// LP optimum, which itself cannot be lower than `RMP_obj`.
fn can_prune_by_bound(lb: usize, best_ub: usize, disable_bound_prune: bool) -> bool {
    if disable_bound_prune {
        return false;
    }
    lb >= best_ub
}

fn is_tiny_two_tree_core(reduced: &Instance, trees: &[Tree]) -> bool {
    trees.len() == 2 && reduced.num_leaves <= 64
}

fn use_bound_prune_shortcuts(reduced: &Instance, trees: &[Tree]) -> bool {
    let _ = (reduced, trees);
    true
}

fn use_rcvf_shortcuts(reduced: &Instance, trees: &[Tree], no_rcvf: bool, tiny_rcvf: bool) -> bool {
    if no_rcvf {
        return false;
    }
    if tiny_rcvf {
        return true;
    }
    !is_tiny_two_tree_core(reduced, trees)
}

/// Solve a kernelized, undecomposable subinstance.
///
/// Caller must guarantee `m ≥ 2` and `n ≥ 2` (the pipeline's
/// `trivial_solution` short-circuit handles the trivial cases).
pub fn solve_inner(reduced: &Instance, terminate: &Arc<AtomicBool>) -> Option<Vec<Tree>> {
    let cancel = crate::solvers::bp::Cancel::new(Arc::clone(terminate));
    let cfg = crate::solvers::bp::BpConfig::default();
    solve_inner_with_subsolver(reduced, &cfg, &cancel, &mut |sub| {
        crate::solvers::bp::solve_subinstance(sub, &cfg, &cancel)
    })
}

/// Variant of [`solve_inner`] that lets the recursive decomposition caller
/// provide the same subproblem solver/memo to primal heuristics.
pub fn solve_inner_with_subsolver<F>(
    reduced: &Instance,
    cfg: &crate::solvers::bp::BpConfig,
    cancel: &crate::solvers::bp::Cancel,
    solve_sub: &mut F,
) -> Option<Vec<Tree>>
where
    F: FnMut(&Instance) -> Option<Vec<Tree>>,
{
    let trees = &reduced.trees;
    let n = reduced.num_leaves as usize;
    if trees.is_empty() {
        return None;
    }
    if trees.len() == 1 {
        return Some(trees.clone());
    }
    if n <= 1 {
        return Some(trees[0..1].to_vec());
    }
    maybe_log_core_decomp_potential(reduced, cfg.core_decomp_analyze, cfg.core_decomp_min_leaves);

    // Chen pairwise lower bound — a sound combinatorial floor on the
    // component count, valid for every B&B node of this (sub)instance.
    // May be raised below by the ncpack mu* floor (gated).
    let mut chen_lb = chen_lower_bound(trees, cfg.no_chen_lb);

    // Seed singletons via a temporary builder; the runtime builder lives in
    // PricerScratch so all pricer tiers share it.
    let mut seed_builder = ColumnBuilder::new(trees);
    let initial: Vec<AfColumn> = (1..=n as u32)
        .map(|l| seed_builder.build_unchecked(vec![l], trees))
        .collect();
    let mut state = SearchState::seed_singletons(n, initial.clone());
    let mut rmp = Rmp::new(&initial, trees, n);

    if trees.len() > 2 {
        let bounds = compute_local_bounds(trees, reduced.num_leaves);
        if let Some(partition) = bounds.best_partition.as_deref() {
            install_partition_incumbent(&mut state, &mut rmp, trees, &mut seed_builder, partition);
        }
    }

    if cfg.ncpack_incumbent
        && trees.len() >= cfg.ncpack_min_trees
        && n <= cfg.ncpack_max_leaves
        && state.best_ub() > 1
    {
        let nc_t = Instant::now();
        if let Some(forest) = crate::solvers::ncpack::pair_matching_forest(
            reduced,
            cfg.ncpack_max_leaves,
            cfg.ncpack_node_budget,
        ) {
            let k = forest.len();
            install_incumbent(
                &mut state,
                &mut rmp,
                trees,
                &mut seed_builder,
                forest,
                "ncpack pairs",
            );
            log::debug!(
                target: LOG_TARGET,
                "ncpack pair incumbent: k={} active_pairs={} ({:.1}ms)",
                k,
                reduced.num_leaves as usize - k,
                nc_t.elapsed().as_secs_f64() * 1000.0,
            );
        } else {
            log::debug!(
                target: LOG_TARGET,
                "ncpack pair incumbent: no forest ({:.1}ms)",
                nc_t.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    // ncpack mu* floor (GATED, KLADOS_BP_NCPACK_LB=1, off by default, EXPERIMENTAL).
    //
    // ⚠ UNSOUND IN GENERAL. ncpack's `value` is an ACHIEVABLE packing, so
    // `value <= mu*` and `n - value >= OPT`: it is fundamentally an UPPER bound on
    // OPT (an incumbent), and a valid LOWER bound (what `chen_lb` is) ONLY when
    // `value == mu*` exactly — i.e. only when ncpack's single-block search is the
    // true optimum. With slack 0 it can undercount mu* (multi-block lifts), making
    // n-value > OPT and this "floor" too high → could prune the true optimum.
    // It happens to equal mu* on the high-frag wall (pub134/pub101), giving a real
    // speedup there, but that rests on the unproven single-block-sufficiency
    // conjecture. Kept only as a research artifact; DO NOT enable for correctness.
    if cfg.ncpack_lb
        && trees.len() >= cfg.ncpack_min_trees
        && n <= cfg.ncpack_max_leaves
    {
        let nc_t = Instant::now();
        let deadline =
            Instant::now() + std::time::Duration::from_secs(cfg.ncpack_lb_budget_secs);
        if let Some(mu) = crate::solvers::ncpack::certified_mu_star(
            reduced,
            0,
            cfg.ncpack_lb_kmax,
            cfg.ncpack_node_budget,
            deadline,
        ) {
            let ncpack_lb = n.saturating_sub(mu);
            if ncpack_lb > chen_lb {
                log::info!(
                    target: LOG_TARGET,
                    "ncpack mu* floor: mu={} -> lb={} (was chen={}) ({:.1}ms)",
                    mu,
                    ncpack_lb,
                    chen_lb,
                    nc_t.elapsed().as_secs_f64() * 1000.0,
                );
                chen_lb = ncpack_lb;
            }
        } else {
            log::debug!(
                target: LOG_TARGET,
                "ncpack mu* floor: not certified within budget ({:.1}ms)",
                nc_t.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    // For m=2, seed multi-leaf columns from Chen's 2-approximation. These are
    // valid AF components (any pair-derived AF component is consistent in 2
    // trees by definition), so they go straight into the pool. Gives the LP
    // a much better starting point than singletons alone.
    //
    // ⚠ m=2 only: a Chen-derived column is a valid AF component for the pair
    // (T_i, T_j) it was computed from, but is NOT necessarily a valid AF
    // component for an m≥3 instance — its restricted shape may disagree with
    // a third tree. For m≥3 we'd have to validate, and in practice the
    // pricer's leaf-pair DP already handles those cases.
    if trees.len() == 2 && n >= 4 {
        let chen_t = Instant::now();
        let (_, _, leafsets) = chen_pair_agreement(&trees[0], &trees[1]);
        let mut chen_added = 0usize;
        for labels in leafsets {
            if labels.len() < 2 {
                continue; // singletons already in pool
            }
            let column = seed_builder.build_unchecked(labels, trees);
            if state.add_column(column).is_some() {
                rmp.add_column(state.columns().last().unwrap());
                chen_added += 1;
            }
        }
        log::debug!(
            target: LOG_TARGET,
            "chen seed: +{} cols ({:.1}ms)",
            chen_added,
            chen_t.elapsed().as_secs_f64() * 1000.0,
        );
    }

    // Primal heuristic via Whidden relaxed cluster decomposition (m=2 only).
    // Runs the relaxed (non-strict) Whidden algorithm, which produces a
    // *feasible* AF whose validity is verified by the AF validator but is
    // not certified optimal. We adopt it as an early incumbent — that's
    // exactly what makes the LP-bound prune effective from iteration 1
    // instead of starting from ub=n. Without this, even when our root LP
    // equals the optimum, fractional LP support prevents extracting an
    // integer solution and we lose by branching needlessly. Matches
    // bp-multi's behavior; this was the missing primal heuristic that
    // explains the recurring "LP=optimum but support fractional" gap.
    let relaxed_incumbent_enabled = cfg.relaxed_incumbent;
    if relaxed_incumbent_enabled
        && trees.len() == 2
        && reduced.num_leaves >= 20
        && let Some(incumbent_forest) =
            try_whidden_relaxed_incumbent_2tree(reduced, solve_sub, false)
    {
        install_incumbent(
            &mut state,
            &mut rmp,
            trees,
            &mut seed_builder,
            incumbent_forest,
            "whidden relaxed",
        );
    }

    let mut scratch = PricerScratch::new(trees);
    scratch.m2_batch = cfg.m2_batch;
    scratch.m2_exact_dp_cells = cfg.m2_exact_dp_cells;
    scratch.m2_exact_reserve_cap = cfg.m2_exact_reserve_cap;
    scratch.use_anchor_cache = cfg.use_anchor_cache;
    let mut pricer = dispatch_by_m(trees);
    // Tried, all reverted with strong negative results — all three
    // amounted to "branching scheme is the bottleneck", which the data
    // refuted each time:
    //
    // 1. Strong branching (every node / depth≤5 / root-only): regressed
    //    both slices. Root LP probe cost dominates the bound gain.
    // 2. Best-first node ordering: regressed both slices ~2×. ΔLP/branch
    //    is too small (~0.2) to make best-first work; behaves like BFS.
    // 3. 4-way cluster branching on fractional triples: regressed easy
    //    (0→21 timeouts) and hard (50→71). The three "isolated" children
    //    only commit 2 cannot-link constraints each, which are weak; the
    //    4× tree multiplier wasn't offset by the actual ΔLP rise.
    //
    // The diagnosis memory and Gemini's literature critique are
    // converging: the LP relaxation on hard m≥3 instances is
    // intrinsically loose, and no branching reform tightens it. Levers
    // that *might* help are speed (per-node cost) and LP tightness
    // itself (cuts that aren't dominated).
    let mut selector = crate::solvers::bp::search::selection::MostFractionalPair;
    let mut tel = Telemetry::default();
    let mut root_regions: Option<RootSupportRegions> = None;

    // DFS stack carrying parent LP bounds. We tried best-first (min-heap
    // by parent_lp) and it regressed badly (~2× on both easy and hard
    // slices) because the LP gap per branch is small (ΔLP ≈ 0.2/level on
    // hard m≥3 instances) — best-first ends up exploring all shallow
    // subtrees within a narrow LP band before driving to any leaf, never
    // finding an integer incumbent fast. DFS naturally drives a single
    // branch to a leaf, finds an incumbent, then prunes via inherited
    // bound. The combination DFS + inherited-bound prune + per-subtree
    // RCVF dominates best-first when ΔLP per branch is small relative to
    // the gap.
    let mut stack: Vec<(Branchings, f64)> = vec![(Branchings::default(), f64::NEG_INFINITY)];
    let mut last_progress_log = std::time::Instant::now();
    let allow_bound_prune = use_bound_prune_shortcuts(reduced, trees);
    let allow_rcvf = use_rcvf_shortcuts(reduced, trees, cfg.no_rcvf, cfg.tiny_rcvf);
    while let Some((b, parent_lp_bound)) = stack.pop() {
        // Periodic progress log so we can see telemetry on timeouts, not
        // just on successful completion. Every 5 seconds is rare enough
        // to be free.
        if last_progress_log.elapsed().as_secs_f64() >= 5.0 {
            let tier_summary = pricer
                .tier_timings()
                .into_iter()
                .filter(|(_, d, _)| !d.is_zero())
                .map(|(name, dur, calls)| {
                    format!("{}={:.0}ms/{}", name, dur.as_secs_f64() * 1000.0, calls)
                })
                .collect::<Vec<_>>()
                .join(",");
            info!(
                target: LOG_TARGET,
                "progress: n={} m={} nodes={} cg={} cols={} prunes={} ub={} stack={} tiers=[{}]",
                reduced.num_leaves,
                trees.len(),
                tel.nodes_explored,
                tel.cg_iters,
                tel.columns_added,
                tel.bound_prunes,
                state.best_ub(),
                stack.len() + 1,
                tier_summary,
            );
            last_progress_log = std::time::Instant::now();
        }
        // Backtrack: drop any per-subtree RCVF fixings made by previously-
        // explored sibling subtrees. Trail entries with depth ≥ b.depth()
        // were placed by deeper nodes that have since been fully explored;
        // their fixings are no longer valid in the subtree we're entering.
        rmp.unfix_above_depth(b.depth());

        // Inherited-bound prune: child_LP ≥ parent_LP, so if the parent's
        // LP already met the prune threshold the child does too.
        let inherited_lb = if parent_lp_bound.is_finite() {
            (parent_lp_bound - 1e-6).ceil() as usize
        } else {
            0
        };
        // The Chen lower bound is a sound floor independent of the LP.
        if allow_bound_prune
            && can_prune_by_bound(
                inherited_lb.max(chen_lb),
                state.best_ub(),
                cfg.disable_bound_prune,
            )
        {
            tel.bound_prunes += 1;
            continue;
        }

        // Graceful abort: return best incumbent (or all-singletons as fallback).
        if cancel.is_cancelled() {
            let components = match state.incumbent() {
                Some(inc) => reconstruct_components(inc, state.columns(), reduced),
                None => {
                    let num_leaves = reduced.num_leaves;
                    (1..=num_leaves)
                        .map(|l| klados_core::Tree::singleton(l, num_leaves))
                        .collect()
                }
            };
            return Some(components);
        }

        let outcome = solve_node(
            &mut state,
            &b,
            reduced,
            trees,
            &mut rmp,
            &mut pricer,
            &mut scratch,
            &mut selector,
            &mut tel,
            &mut root_regions,
            allow_bound_prune,
            allow_rcvf,
            chen_lb,
            cancel,
            cfg,
        );
        match outcome {
            NodeOutcome::Pruned => {}
            NodeOutcome::Integral(inc) => {
                let updated = state.update_incumbent(inc);
                if updated {
                    tel.incumbent_updates += 1;
                    info!(
                        target: LOG_TARGET,
                        "incumbent: k={} (n={} m={} depth={} nodes={})",
                        state.best_ub(),
                        reduced.num_leaves,
                        trees.len(),
                        b.depth(),
                        tel.nodes_explored,
                    );
                    maybe_log_bridge_footprint(
                        "incumbent-update",
                        state.incumbent(),
                        state.columns(),
                        root_regions.as_ref(),
                        cfg.bridge_probe,
                    );
                    // RCVF replay happens at the top of the next solve_node.
                }
            }
            NodeOutcome::Branch { lp_obj, children } => {
                // Push children in reverse so the first one is popped
                // next — matches the prior 2-way DFS ordering where
                // `left` (the must-link side) was explored before
                // `right` (cannot-link). For k-way branching, the
                // selector's natural child ordering is preserved.
                for child in children.into_iter().rev() {
                    stack.push((child, lp_obj));
                }
            }
        }
    }

    let inc = state.incumbent()?;
    let components = reconstruct_components(inc, state.columns(), reduced);
    info!(
        target: LOG_TARGET,
        "bp done: n={} m={} k={} nodes={} cg_iters={} cols={} cuts={}",
        reduced.num_leaves, trees.len(), components.len(),
        tel.nodes_explored, tel.cg_iters, tel.columns_added, tel.cuts_added,
    );
    info!(
        target: LOG_TARGET,
        "bp timings ms: pricing={:.1} lp_solve={:.1} apply_bounds={:.1} cuts={:.1} branching={:.1}",
        tel.timings.pricing.as_secs_f64() * 1000.0,
        tel.timings.lp_solve.as_secs_f64() * 1000.0,
        tel.timings.bounds_apply.as_secs_f64() * 1000.0,
        tel.timings.cut_separation.as_secs_f64() * 1000.0,
        tel.timings.branching.as_secs_f64() * 1000.0,
    );
    // Per-tier pricer breakdown so we can see which tier dominates on
    // hard instances. The tier names match the strings logged at
    // composite-pricer trace level (`reserve`, `leaf-pair-dp`,
    // `dssr-multi-pair-dp`, `small-component`, `exact-pair-dp`).
    let tier_breakdown = pricer
        .tier_timings()
        .into_iter()
        .map(|(name, dur, calls)| format!("{}={:.1}ms/{}", name, dur.as_secs_f64() * 1000.0, calls))
        .collect::<Vec<_>>()
        .join(" ");
    info!(target: LOG_TARGET, "bp pricer tiers: {}", tier_breakdown);
    Some(components)
}

fn solve_node<P: Pricer, S: BranchSelector>(
    state: &mut SearchState,
    branchings: &Branchings,
    reduced: &Instance,
    trees: &[Tree],
    rmp: &mut Rmp,
    pricer: &mut P,
    scratch: &mut PricerScratch,
    selector: &mut S,
    tel: &mut Telemetry,
    root_regions: &mut Option<RootSupportRegions>,
    allow_bound_prune: bool,
    allow_rcvf: bool,
    chen_lb: usize,
    cancel: &crate::solvers::bp::Cancel,
    cfg: &crate::solvers::bp::BpConfig,
) -> NodeOutcome {
    tel.nodes_explored += 1;

    // Replay root-RCVF if the incumbent tightened since the last fixing.
    // No-op when untightened (or before root has been solved), so this is
    // free on the hot path; when an in-node primal heuristic (rounding /
    // MIP-on-pool) cut best_ub, the next node entry picks up the new
    // fixings before we touch the LP.
    let newly = if allow_rcvf {
        rmp.reapply_root_rcvf(state.columns(), state.best_ub())
    } else {
        0
    };
    if newly > 0 {
        debug!(
            target: LOG_TARGET,
            "rcvf replay (incumbent={}, depth={}): fixed {} more columns",
            state.best_ub(), branchings.depth(), newly,
        );
    }

    let t0 = Instant::now();
    rmp.apply_bounds(state.columns(), branchings);
    tel.timings.bounds_apply += t0.elapsed();

    // Column generation.
    // `node_converged` records whether CG ended on a genuine `Converged` from
    // the pricer. The LP bound (bound-prune, RCVF) may be trusted ONLY then.
    let mut node_converged = false;
    let lp = loop {
        let t0 = Instant::now();
        let lp = match rmp.solve() {
            Ok(lp) => lp,
            Err(e) => {
                debug!(target: LOG_TARGET, "node pruned: LP {e}");
                return NodeOutcome::Pruned;
            }
        };
        tel.timings.lp_solve += t0.elapsed();
        tel.cg_iters += 1;

        // Abort check between CG rounds so signal doesn't get stuck
        // waiting for a long pricing phase to complete.
        if cancel.is_cancelled() {
            debug!(target: LOG_TARGET, "cg abort at iter {}", tel.cg_iters);
            return NodeOutcome::Pruned;
        }

        // Cut separation: materialise any violated node ≤1 constraints.
        // Must happen *before* pricing so the duals we feed the pricer
        // reflect the tightened LP — otherwise β≡0 on unmaterialised rows
        // makes the pricer overweight columns covering many internals.
        let t0 = Instant::now();
        let new_cuts = rmp.separate_and_add_cuts(state.columns(), &lp.column_values, 1.0e-6);
        tel.timings.cut_separation += t0.elapsed();
        if new_cuts > 0 {
            tel.cuts_added += new_cuts;
            debug!(
                target: LOG_TARGET,
                "cg iter {}: +{} cuts (total rows tightened); re-solving LP",
                tel.cg_iters, new_cuts,
            );
            continue;
        }

        let t0 = Instant::now();
        let result = pricer.price(
            &PricingContext {
                trees,
                num_leaves: state.num_leaves(),
                alpha: &lp.leaf_duals,
                beta: &lp.node_duals,
                columns: state.columns(),
                seen: state.seen(),
                branchings,
                terminate: cancel.flag(),
                deadline: cancel.deadline(),
            },
            scratch,
        );
        tel.timings.pricing += t0.elapsed();
        let pt = t0.elapsed();

        match result {
            PricingResult::Found(cols) => {
                let ncols = cols.len();
                let nseen = cols
                    .iter()
                    .filter(|c| state.seen().contains(c.labels()))
                    .count();
                let mut added = 0;
                for c in cols {
                    if let Some(_id) = state.add_column(c) {
                        rmp.add_column(state.columns().last().unwrap());
                        added += 1;
                    }
                }
                tel.columns_added += added;
                debug!(
                    target: LOG_TARGET,
                    "cg iter={} lp={:.4} lp_ms={:.1} pricer_ms={:.1} cols_found={} added={} seen={} total_cols={}",
                    tel.cg_iters,
                    lp.objective,
                    tel.timings.lp_solve.as_secs_f64() * 1000.0,
                    pt.as_secs_f64() * 1000.0,
                    ncols,
                    added,
                    nseen,
                    state.columns().len(),
                );
                continue;
            }
            PricingResult::Improving => {
                // An improving column provably exists but none was emittable
                // (all violate branch constraints). The LP is NOT at its true
                // optimum — bound is uncertified. CG stops; the node branches.
                debug!(
                    target: LOG_TARGET,
                    "cg iter={} lp={:.4} IMPROVING (uncertified) pricer_ms={:.1}",
                    tel.cg_iters,
                    lp.objective,
                    pt.as_secs_f64() * 1000.0
                );
                break lp;
            }
            PricingResult::Converged => {
                debug!(
                    target: LOG_TARGET,
                    "cg iter={} lp={:.4} CONVERGED pricer_ms={:.1}",
                    tel.cg_iters,
                    lp.objective,
                    pt.as_secs_f64() * 1000.0
                );
                tel.had_converged = true;
                node_converged = true;
                break lp;
            }
        }
    };

    // The LP bound is a valid lower bound ONLY if CG genuinely converged
    // (`Improving` leaves the LP objective below the true node optimum). The
    // Chen lower bound is a sound combinatorial floor that holds regardless.
    let lp_lb = (lp.objective - 1e-6).ceil() as usize;
    let lb = if node_converged {
        lp_lb.max(chen_lb)
    } else {
        chen_lb
    };

    if allow_bound_prune && can_prune_by_bound(lb, state.best_ub(), cfg.disable_bound_prune) {
        debug!(
            target: LOG_TARGET,
            "node pruned by bound: lb={} (lp={:.4} chen={}) ub={}",
            lb, lp.objective, chen_lb, state.best_ub(),
        );
        tel.bound_prunes += 1;
        return NodeOutcome::Pruned;
    }

    // Reduced-cost variable fixing. Standard B&P result: for every column c,
    // `LP_with_x_c≥1 ≥ lp_obj + rc(c)`, so any improving integer solution
    // (objective ≤ best_ub − 1) cannot use c if `lp_obj + rc(c) > best_ub − 1`.
    //
    // Only safe at the **root** because RCVF fixings live on the shared Rmp
    // and would otherwise poison sibling subtrees. Root duals come from the
    // unrestricted master LP, so the fixings hold globally — every feasible
    // solution of any descendant subtree is feasible for the unrestricted
    // problem too, so columns barred from the unrestricted improving region
    // are barred everywhere. Non-root RCVF would tighten further within its
    // subtree but requires per-subtree undo machinery; deferred.
    // RCVF reduced costs are valid only at the true LP optimum — gate on
    // genuine convergence (root is unconstrained so it always converges).
    if branchings.depth() == 0 && allow_rcvf && node_converged {
        let rcvf_newly_fixed = rmp.apply_rcvf(
            lp.objective,
            state.columns(),
            &lp.leaf_duals,
            &lp.node_duals,
            state.best_ub(),
        );
        if rcvf_newly_fixed > 0 {
            debug!(
                target: LOG_TARGET,
                "rcvf root: fixed {} / {} columns (lp={:.4} ub={})",
                rcvf_newly_fixed,
                state.columns().len(),
                lp.objective,
                state.best_ub(),
            );
        }
        // Cache the root LP solution: every future incumbent improvement
        // makes the RCVF condition strictly tighter under these same duals,
        // so we replay them rather than re-solving root.
        rmp.save_root_lp(
            lp.objective,
            lp.leaf_duals.clone(),
            lp.node_duals.clone(),
            state.best_ub(),
        );

        // Diagnostic only: expose the global fractional support obstruction
        // before the ordinary branch tree starts.  The current Class-B
        // hypothesis is that almost the whole LP gap lives in one connected
        // overlap component of the positive support; if that component's
        // induced subinstance has a substantially larger exact rank than the
        // LP mass currently paid inside it, then the missing proof object is a
        // global obstruction cut rather than another local branch.
        if root_regions.is_none()
            && (cfg.obstruction_probe || cfg.bridge_probe || cfg.root_support_incumbent)
        {
            *root_regions = build_root_support_regions(state, &lp);
        }
        if cfg.root_support_incumbent
            && let Some(regions) = root_regions.as_ref()
        {
            let t0 = Instant::now();
            if let Some(inc) = try_root_support_incumbent(reduced, state, regions) {
                let k = inc.k;
                if state.update_incumbent(inc) {
                    info!(
                        target: LOG_TARGET,
                        "root-support incumbent: k={} support_cols={} solve_ms={:.1}",
                        k,
                        regions.comps.iter().map(|comp| comp.column_ids.len()).sum::<usize>(),
                        t0.elapsed().as_secs_f64() * 1000.0,
                    );
                    rmp.reapply_root_rcvf(state.columns(), state.best_ub());
                }
            }
        }
        maybe_probe_root_obstruction(reduced, state, &lp, root_regions.as_ref(), cfg);
        maybe_probe_root_rank_cuts(state, &lp, cfg);
        maybe_probe_residual_completion_cuts(reduced, state, &lp, cfg);
        maybe_log_bridge_footprint(
            "root-incumbent",
            state.incumbent(),
            state.columns(),
            root_regions.as_ref(),
            cfg.bridge_probe,
        );

        // ── Corridor-enriched B&P (DISABLED by default; ablation only) ─
        // After root CG converges, the pool contains every column with
        // `rc < 0` under root duals; the corridor theorem says any
        // column in an improving integer solution has `rc ≤ γ`. We
        // tried pre-enumerating the root corridor and adding it to the
        // pool to skip B&P's deep-node DP work. **Empirically didn't
        // help**: on v2 50/100 timeouts unchanged, valid-sum got
        // *slower* by ~30%, because (a) at root duals corridor columns
        // have `rc ≥ 0` and don't enter any LP basis, (b) at deep
        // nodes B&P's pricer finds columns that aren't in the *root*
        // corridor (their `rc_root > γ_root` but `rc_local < 0`), so
        // pre-loading doesn't shortcut the deep DP, and (c) the
        // enriched pool slows every LP solve in the search tree.
        // Conclusion: B&P's incremental dual-modulation discovers
        // exactly the columns it needs; the root-corridor theorem
        // is *informative* about completeness but the upfront-add
        // formulation isn't a shortcut.
        // Re-enable via `BpConfig.corridor_enrich` for further
        // experimentation.
        let corridor_enrich = cfg.corridor_enrich;
        if corridor_enrich && trees.len() == 2 {
            let upper = state.best_ub();
            let lb = (lp.objective - 1.0e-6).ceil() as usize;
            // Only enumerate if there's slack to close. `γ < 0` means
            // ⌈L⌉ ≥ U already, no improving column possible.
            if lb < upper {
                let gamma = (upper as f64) - 1.0 - lp.objective;
                let threshold = 1.0 - gamma - 1.0e-8;
                let max_k = cfg.corridor_max_k.max(1);
                let n0 = trees[0].num_nodes();
                let n1 = trees[1].num_nodes();
                let mut cache = scratch
                    .topk_dp_cache
                    .take()
                    .filter(|c| c.fits(n0, n1, state.num_leaves()))
                    .unwrap_or_else(|| {
                        crate::solvers::corridor::topk_m2::TopKDpCache::new(
                            n0,
                            n1,
                            state.num_leaves(),
                        )
                    });
                let cols = crate::solvers::corridor::topk_m2::enumerate_corridor(
                    &crate::solvers::corridor::topk_m2::CorridorInput {
                        t0: &trees[0],
                        t1: &trees[1],
                        alpha: &lp.leaf_duals,
                        beta_t0: &lp.node_duals[0],
                        beta_t1: &lp.node_duals[1],
                        threshold,
                        max_k,
                    },
                    &mut cache,
                );
                scratch.topk_dp_cache = Some(cache);

                let mut added = 0usize;
                let pool_before = state.columns().len();
                let builder = &mut scratch.builder;
                for cand in cols {
                    if state.seen().contains(&cand.labels) {
                        continue;
                    }
                    let column = builder.build_unchecked(cand.labels, trees);
                    if let Some(_id) = state.add_column(column) {
                        rmp.add_column(state.columns().last().unwrap());
                        added += 1;
                    }
                }
                if added > 0 {
                    debug!(
                        target: LOG_TARGET,
                        "corridor enrich: +{} cols (pool {}→{}, γ={:.3}, K={})",
                        added,
                        pool_before,
                        state.columns().len(),
                        gamma,
                        max_k,
                    );
                }
            }
        }
    } else if allow_rcvf && node_converged {
        // Subtree-local RCVF: the LP at a branched node is tighter than
        // root because the branching constraints have raised its optimum,
        // so the same rc-bound condition fixes columns that root duals
        // can't reach. Fixings here are valid only inside this subtree —
        // recorded on the rcvf_trail and undone on backtrack. Correct
        // only under DFS, which is the search order we use (best-first
        // tried and reverted — see the search loop's comment).
        //
        // Gated on `node_converged`: RCVF's rc-bound `lp.objective + rc ≥ ub`
        // is valid only at the true LP optimum. On an uncertified `Improving`
        // node the column generation stopped early — `lp.objective` is not a
        // lower bound and the reduced costs are against incomplete duals, so
        // fixing here could discard a column the optimum needs.
        let rcvf_newly_fixed = rmp.apply_subtree_rcvf(
            lp.objective,
            state.columns(),
            &lp.leaf_duals,
            &lp.node_duals,
            state.best_ub(),
            branchings.depth(),
        );
        if rcvf_newly_fixed > 0 {
            debug!(
                target: LOG_TARGET,
                "rcvf subtree (depth={}): fixed {} more (lp={:.4} ub={})",
                branchings.depth(), rcvf_newly_fixed, lp.objective, state.best_ub(),
            );
        }
    }

    if let Some(inc) = try_integral(state, &lp.column_values) {
        return NodeOutcome::Integral(inc);
    }
    if let Some(inc) = try_support_partition(state, &lp.column_values, reduced) {
        return NodeOutcome::Integral(inc);
    }

    // Greedy primal heuristic: round LP support to a feasible integer AF.
    // Improves ub when applicable; doesn't terminate the subtree. Cheap.
    // Note: for cases where LP=optimum but support is fractional, greedy
    // rounding generally returns ub = optimum + 1 — it doesn't recover the
    // missing integer optimum. A diving / MIP-on-pool heuristic would be
    // stronger but is deferred.
    if let Some(inc) = try_round_primal(state, &lp.column_values)
        && inc.k < state.best_ub()
    {
        let updated = state.update_incumbent(inc);
        if updated {
            tel.incumbent_updates += 1;
            debug!(
                target: LOG_TARGET,
                "primal heuristic improved incumbent: ub={} (cg_iter={})",
                state.best_ub(), tel.cg_iters,
            );

            if allow_bound_prune && can_prune_by_bound(lb, state.best_ub(), cfg.disable_bound_prune)
            {
                tel.bound_prunes += 1;
                return NodeOutcome::Pruned;
            }
        }
    }

    // MIP-on-pool primal heuristic. Disabled by default; enable via
    // `BpConfig.mip_heuristic`. Fires when the LP objective is at an
    // integer boundary but the support is fractional — exactly the case
    // where pure branching nudges the LP by ε per node and a MIP solve
    // over the existing pool finds the missing integer combination
    // directly. Time-capped (100ms by default) so the failure mode is
    // bounded.
    let lp_frac = lp.objective.ceil() - lp.objective;
    if cfg.mip_heuristic && lb < state.best_ub() && lp_frac < 1e-4 {
        debug!(target: LOG_TARGET, "Running MIP heuristic on pool of {} columns (lp_obj={:.4})", state.columns().len(), lp.objective);
        let mut mip_attempts = 0;
        while mip_attempts < 5 {
            mip_attempts += 1;
            if let Ok(Some(mip_sol)) = rmp.solve_mip_with_time_limit(cfg.mip_time_limit) {
                debug!(target: LOG_TARGET, "MIP solve {}: obj={:.4}", mip_attempts, mip_sol.objective);
                let new_cuts =
                    rmp.separate_and_add_cuts(state.columns(), &mip_sol.column_values, 0.5);
                if new_cuts > 0 {
                    tel.cuts_added += new_cuts;
                    debug!(target: LOG_TARGET, "MIP solution violated {} cuts, looping", new_cuts);
                    continue; // Re-solve MIP with new cuts
                }

                if let Some(inc) = try_integral(state, &mip_sol.column_values) {
                    debug!(target: LOG_TARGET, "try_integral found valid incumbent k={}", inc.k);
                    if inc.k < state.best_ub() {
                        let updated = state.update_incumbent(inc);
                        if updated {
                            tel.incumbent_updates += 1;
                            debug!(
                                target: LOG_TARGET,
                                "MIP heuristic improved incumbent: ub={} (cg_iter={}, depth={})",
                                state.best_ub(), tel.cg_iters, branchings.depth(),
                            );

                            if allow_bound_prune
                                && can_prune_by_bound(lb, state.best_ub(), cfg.disable_bound_prune)
                            {
                                tel.bound_prunes += 1;
                                return NodeOutcome::Pruned;
                            }
                        }
                    }
                } else {
                    debug!(target: LOG_TARGET, "try_integral returned None for MIP solution");
                }
            } else {
                debug!(target: LOG_TARGET, "rmp.solve_mip_with_time_limit() failed or returned None");
            }
            break;
        }
    }
    let t0 = Instant::now();
    let children = selector.select(
        &SelectionContext {
            columns: state.columns(),
            values: &lp.column_values,
            num_leaves: state.num_leaves(),
            branchings,
            current_lp_obj: lp.objective,
        },
        rmp,
    );
    tel.timings.branching += t0.elapsed();
    match children {
        Some(children) if !children.is_empty() => NodeOutcome::Branch {
            lp_obj: lp.objective,
            children,
        },
        _ => {
            debug!(target: LOG_TARGET, "selector returned no children, but not integral. Pruning fractional solution!");
            NodeOutcome::Pruned
        }
    }
}

#[derive(Clone, Debug)]
struct SupportComponentSummary {
    column_ids: Vec<usize>,
    leaves: FixedBitSet,
    lp_mass: f64,
    fractional_columns: usize,
}

impl SupportComponentSummary {
    fn leaf_count(&self) -> usize {
        self.leaves.count_ones(..)
    }
}

#[derive(Clone, Debug)]
struct RootSupportRegions {
    comps: Vec<SupportComponentSummary>,
    region_of_leaf: Vec<usize>,
    component_ceil_sum: usize,
    lp_objective: f64,
}

fn build_root_support_regions(
    state: &SearchState,
    lp: &crate::solvers::bp::rmp::RmpSolution,
) -> Option<RootSupportRegions> {
    let mut support_cols = Vec::new();
    for (ci, &value) in lp.column_values.iter().enumerate() {
        if value > 1.0e-6 {
            support_cols.push(ci);
        }
    }
    if support_cols.is_empty() {
        return None;
    }

    // Union support columns whenever they share at least one leaf.
    let mut parent: Vec<usize> = (0..support_cols.len()).collect();
    let mut rank = vec![0u8; support_cols.len()];
    let mut first_owner_by_leaf = vec![None; state.num_leaves() + 1];

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            let root = find(parent, parent[x]);
            parent[x] = root;
        }
        parent[x]
    }
    fn union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
        let mut ra = find(parent, a);
        let mut rb = find(parent, b);
        if ra == rb {
            return;
        }
        if rank[ra] < rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        parent[rb] = ra;
        if rank[ra] == rank[rb] {
            rank[ra] += 1;
        }
    }

    for (local_idx, &ci) in support_cols.iter().enumerate() {
        for &leaf in state.columns()[ci].labels() {
            let slot = &mut first_owner_by_leaf[leaf as usize];
            if let Some(prev_local) = *slot {
                union(&mut parent, &mut rank, local_idx, prev_local);
            } else {
                *slot = Some(local_idx);
            }
        }
    }

    let mut by_root: HashMap<usize, SupportComponentSummary> = HashMap::new();
    for (local_idx, &ci) in support_cols.iter().enumerate() {
        let root = find(&mut parent, local_idx);
        let entry = by_root
            .entry(root)
            .or_insert_with(|| SupportComponentSummary {
                column_ids: Vec::new(),
                leaves: FixedBitSet::with_capacity(state.num_leaves() + 1),
                lp_mass: 0.0,
                fractional_columns: 0,
            });
        entry.column_ids.push(ci);
        entry.lp_mass += lp.column_values[ci];
        if lp.column_values[ci] > 1.0e-6 && lp.column_values[ci] < 1.0 - 1.0e-6 {
            entry.fractional_columns += 1;
        }
        for &leaf in state.columns()[ci].labels() {
            entry.leaves.insert(leaf as usize);
        }
    }

    let mut comps: Vec<_> = by_root.into_values().collect();
    comps.sort_by(|a, b| {
        b.leaf_count()
            .cmp(&a.leaf_count())
            .then_with(|| b.column_ids.len().cmp(&a.column_ids.len()))
    });
    let component_ceil_sum: usize = comps
        .iter()
        .map(|comp| (comp.lp_mass - 1.0e-6).ceil() as usize)
        .sum();
    let mut region_of_leaf = vec![usize::MAX; state.num_leaves() + 1];
    for (rid, comp) in comps.iter().enumerate() {
        for leaf in comp.leaves.ones() {
            region_of_leaf[leaf] = rid;
        }
    }

    Some(RootSupportRegions {
        comps,
        region_of_leaf,
        component_ceil_sum,
        lp_objective: lp.objective,
    })
}

fn maybe_probe_root_obstruction(
    reduced: &Instance,
    state: &SearchState,
    lp: &crate::solvers::bp::rmp::RmpSolution,
    root_regions: Option<&RootSupportRegions>,
    cfg: &crate::solvers::bp::BpConfig,
) {
    if !cfg.obstruction_probe {
        return;
    }

    let already_inside = IN_OBSTRUCTION_PROBE.with(|flag| flag.get());
    if already_inside {
        return;
    }

    IN_OBSTRUCTION_PROBE.with(|flag| flag.set(true));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        probe_root_obstruction_impl(reduced, state, lp, root_regions, cfg)
    }));
    IN_OBSTRUCTION_PROBE.with(|flag| flag.set(false));
    if result.is_err() {
        info!(target: LOG_TARGET, "obstruction-probe: panicked; skipping diagnostic");
    }
}

fn probe_root_obstruction_impl(
    reduced: &Instance,
    state: &SearchState,
    lp: &crate::solvers::bp::rmp::RmpSolution,
    root_regions: Option<&RootSupportRegions>,
    cfg: &crate::solvers::bp::BpConfig,
) {
    let Some(regions) = root_regions else {
        info!(target: LOG_TARGET, "obstruction-probe: empty LP support");
        return;
    };

    info!(
        target: LOG_TARGET,
        "obstruction-probe: root lp={:.4} support_cols={} support_components={}",
        regions.lp_objective,
        regions.comps.iter().map(|comp| comp.column_ids.len()).sum::<usize>(),
        regions.comps.len(),
    );
    info!(
        target: LOG_TARGET,
        "obstruction-probe: component_ceil_sum={} ceil_gap={:.4}",
        regions.component_ceil_sum,
        regions.component_ceil_sum as f64 - regions.lp_objective,
    );
    for (idx, comp) in regions.comps.iter().take(12).enumerate() {
        info!(
            target: LOG_TARGET,
            "obstruction-probe: comp#{idx} cols={} leaves={} frac_cols={} lp_mass={:.4}",
            comp.column_ids.len(),
            comp.leaf_count(),
            comp.fractional_columns,
            comp.lp_mass,
        );
    }

    if cfg.root_support_mip {
        probe_root_support_mip(reduced, state, lp, regions);
    }
    if cfg.all_region_support_mip {
        probe_all_region_support_mips(reduced, state, regions);
    }
    if cfg.all_region_exact_rank {
        probe_all_region_exact_ranks(reduced, regions, cfg.all_region_exact_max_leaves);
    }
    if cfg.tree_side_exact_rank {
        probe_tree_side_exact_ranks(
            reduced,
            state,
            lp,
            cfg.tree_side_exact_max_leaves,
            cfg.tree_side_exact_limit,
            false,
            "tree-side-exact-rank",
        );
    }
    if cfg.tree_laminar_exact_rank {
        probe_tree_side_exact_ranks(
            reduced,
            state,
            lp,
            cfg.tree_side_exact_max_leaves,
            cfg.tree_side_exact_limit,
            true,
            "tree-laminar-exact-rank",
        );
    }

    let Some(largest) = regions.comps.first() else {
        return;
    };
    if largest.fractional_columns == 0 || largest.leaf_count() <= 1 {
        info!(target: LOG_TARGET, "obstruction-probe: largest component is integral/trivial");
        return;
    }

    let want_local_lb = cfg.obstruction_local_lb;
    let want_exact_core = cfg.obstruction_solve_core;
    if !want_local_lb && !want_exact_core {
        return;
    }

    let (core, reverse_map) =
        klados_core::kernelize::restrict_instance_simple(reduced, &largest.leaves);
    let reverse_labels: Vec<u32> = reverse_map.iter().copied().skip(1).collect();
    info!(
        target: LOG_TARGET,
        "obstruction-probe: solving largest core leaves={} lp_mass={:.4} labels={:?}",
        core.num_leaves,
        largest.lp_mass,
        reverse_labels,
    );

    if want_local_lb {
        let mut local = crate::solvers::root_pool::RootPoolSolver::for_corridor_probe();
        let t_lb = Instant::now();
        match local.solve_with_outcome(&core) {
            Some(out) => info!(
                target: LOG_TARGET,
                "obstruction-probe: largest core local_lb={:?} local_k={} local_conv={} support_mass={:.4} lb_ms={:.1}",
                out.lower_bound,
                out.forest.len(),
                out.converged,
                largest.lp_mass,
                t_lb.elapsed().as_secs_f64() * 1000.0,
            ),
            None => info!(
                target: LOG_TARGET,
                "obstruction-probe: largest core local LB solve failed after {:.1}ms",
                t_lb.elapsed().as_secs_f64() * 1000.0,
            ),
        }
    }

    if cfg.region_support_mip {
        probe_region_support_mip(reduced, state, largest);
    }

    if !want_exact_core {
        return;
    }

    let t0 = Instant::now();
    let exact = crate::solvers::bp::solve_subinstance(
        &core,
        &crate::solvers::bp::BpConfig::default(),
        &crate::solvers::bp::Cancel::new(Arc::new(AtomicBool::new(false))),
    );
    match exact {
        Some(forest) => info!(
            target: LOG_TARGET,
            "obstruction-probe: largest core exact_rank={} lp_mass={:.4} gap={:.4} solve_ms={:.1}",
            forest.len(),
            largest.lp_mass,
            forest.len() as f64 - largest.lp_mass,
            t0.elapsed().as_secs_f64() * 1000.0,
        ),
        None => info!(
            target: LOG_TARGET,
            "obstruction-probe: largest core exact solve failed after {:.1}ms",
            t0.elapsed().as_secs_f64() * 1000.0,
        ),
    }
}

fn probe_root_support_mip(
    reduced: &Instance,
    state: &SearchState,
    lp: &crate::solvers::bp::rmp::RmpSolution,
    regions: &RootSupportRegions,
) {
    let t0 = Instant::now();
    let (support_cols, cuts_total, result) = solve_root_support_mip(reduced, state, regions);
    let shell = result
        .as_ref()
        .map(|inc| summarize_root_reduced_cost_shell(state, lp, regions, inc.k));
    info!(
        target: LOG_TARGET,
        "obstruction-probe: root-support-mip support_cols={} lp={:.4} cuts={} result={} solve_ms={:.1}",
        support_cols,
        regions.lp_objective,
        cuts_total,
        result
            .as_ref()
            .map(|inc| format!("k={}", inc.k))
            .unwrap_or_else(|| "none".to_string()),
        t0.elapsed().as_secs_f64() * 1000.0,
    );
    if let Some(shell) = shell {
        info!(
            target: LOG_TARGET,
            "obstruction-probe: root-shell gamma={:.4} generated_total={} generated_in_shell={} support_in_shell={} nonsupport_in_shell={} min_nonsupport_rc={:.4}",
            shell.gamma,
            shell.generated_total,
            shell.generated_in_shell,
            shell.support_in_shell,
            shell.nonsupport_in_shell,
            shell.min_nonsupport_rc,
        );
    }
}

fn try_root_support_incumbent(
    reduced: &Instance,
    state: &SearchState,
    regions: &RootSupportRegions,
) -> Option<Incumbent> {
    let (_, _, result) = solve_root_support_mip(reduced, state, regions);
    result
}

fn solve_root_support_mip(
    reduced: &Instance,
    state: &SearchState,
    regions: &RootSupportRegions,
) -> (usize, usize, Option<Incumbent>) {
    let mut support_ids = regions
        .comps
        .iter()
        .flat_map(|comp| comp.column_ids.iter().copied())
        .collect::<Vec<_>>();
    support_ids.sort_unstable();
    let support_columns = support_ids
        .iter()
        .map(|&ci| state.columns()[ci].clone())
        .collect::<Vec<_>>();
    if support_columns.is_empty() {
        return (0, 0, None);
    }

    let mut rmp = Rmp::new(
        &support_columns,
        &reduced.trees,
        reduced.num_leaves as usize,
    );
    let mut cuts_total = 0usize;
    for _ in 0..32 {
        let Ok(Some(mip)) = rmp.solve_mip_with_time_limit(2.0) else {
            break;
        };
        let cuts = rmp.separate_and_add_cuts(&support_columns, &mip.column_values, 0.5);
        cuts_total += cuts;
        if cuts > 0 {
            continue;
        }
        let chosen = mip
            .column_values
            .iter()
            .enumerate()
            .filter_map(|(local_ci, &v)| (v > 0.5).then_some(support_ids[local_ci]))
            .collect::<Vec<_>>();
        return (
            support_columns.len(),
            cuts_total,
            Some(Incumbent {
                k: chosen.len(),
                component_columns: chosen,
            }),
        );
    }
    (support_columns.len(), cuts_total, None)
}

struct RootShellSummary {
    gamma: f64,
    generated_total: usize,
    generated_in_shell: usize,
    support_in_shell: usize,
    nonsupport_in_shell: usize,
    min_nonsupport_rc: f64,
}

fn summarize_root_reduced_cost_shell(
    state: &SearchState,
    lp: &crate::solvers::bp::rmp::RmpSolution,
    regions: &RootSupportRegions,
    incumbent_k: usize,
) -> RootShellSummary {
    let gamma = incumbent_k as f64 - 1.0 - regions.lp_objective;
    let mut support_ids = FixedBitSet::with_capacity(state.columns().len());
    for comp in &regions.comps {
        for &ci in &comp.column_ids {
            support_ids.insert(ci);
        }
    }
    let mut generated_in_shell = 0usize;
    let mut support_in_shell = 0usize;
    let mut nonsupport_in_shell = 0usize;
    let mut min_nonsupport_rc = f64::INFINITY;
    for (ci, col) in state.columns().iter().enumerate() {
        let rc = 1.0 - col.pricing_score(&lp.leaf_duals, &lp.node_duals);
        if !support_ids.contains(ci) {
            min_nonsupport_rc = min_nonsupport_rc.min(rc);
        }
        if rc <= gamma + 1.0e-6 {
            generated_in_shell += 1;
            if support_ids.contains(ci) {
                support_in_shell += 1;
            } else {
                nonsupport_in_shell += 1;
            }
        }
    }
    RootShellSummary {
        gamma,
        generated_total: state.columns().len(),
        generated_in_shell,
        support_in_shell,
        nonsupport_in_shell,
        min_nonsupport_rc,
    }
}

fn probe_region_support_mip(
    reduced: &Instance,
    state: &SearchState,
    region: &SupportComponentSummary,
) {
    let t0 = Instant::now();
    let (leaves, local_cols, rejected, cuts_total, result) =
        solve_region_support_mip(reduced, state, region);
    info!(
        target: LOG_TARGET,
        "obstruction-probe: support-mip leaves={} support_cols={} local_cols={} rejected={} lp_mass={:.4} cuts={} result={} solve_ms={:.1}",
        leaves,
        region.column_ids.len(),
        local_cols,
        rejected,
        region.lp_mass,
        cuts_total,
        result
            .map(|k| format!("k={k}"))
            .unwrap_or_else(|| "none".to_string()),
        t0.elapsed().as_secs_f64() * 1000.0,
    );
}

fn probe_all_region_support_mips(
    reduced: &Instance,
    state: &SearchState,
    regions: &RootSupportRegions,
) {
    let t0 = Instant::now();
    let mut solved = 0usize;
    let mut sum_k = 0usize;
    let mut failed = 0usize;
    let mut details = Vec::with_capacity(regions.comps.len());
    for (rid, region) in regions.comps.iter().enumerate() {
        let (_, _, _, _, result) = solve_region_support_mip(reduced, state, region);
        if let Some(k) = result {
            solved += 1;
            sum_k += k;
            details.push(format!("{rid}:{k}"));
        } else {
            failed += 1;
            details.push(format!("{rid}:x"));
        }
    }
    info!(
        target: LOG_TARGET,
        "obstruction-probe: all-region-support-mips solved={} failed={} sum_k={} detail=[{}] solve_ms={:.1}",
        solved,
        failed,
        sum_k,
        details.join(","),
        t0.elapsed().as_secs_f64() * 1000.0,
    );
}

fn probe_all_region_exact_ranks(
    reduced: &Instance,
    regions: &RootSupportRegions,
    max_leaves: usize,
) {
    let t0 = Instant::now();
    let mut solved = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;
    let mut sum_rank = 0usize;
    let mut sum_mass = 0.0;
    let mut sum_violation = 0.0;
    let mut details = Vec::with_capacity(regions.comps.len());

    for (rid, region) in regions.comps.iter().enumerate() {
        let leaves = region.leaf_count();
        if leaves > max_leaves {
            skipped += 1;
            details.push(format!("{rid}:skip{leaves}"));
            continue;
        }

        let (core, _) = klados_core::kernelize::restrict_instance_simple(reduced, &region.leaves);
        let t_region = Instant::now();
        let exact = crate::solvers::bp::solve_subinstance(
            &core,
            &crate::solvers::bp::BpConfig::default(),
            &crate::solvers::bp::Cancel::new(Arc::new(AtomicBool::new(false))),
        );
        match exact {
            Some(forest) => {
                let rank = forest.len();
                let violation = rank as f64 - region.lp_mass;
                solved += 1;
                sum_rank += rank;
                sum_mass += region.lp_mass;
                sum_violation += violation;
                details.push(format!(
                    "{rid}:k{}|m{:.3}|v{:.3}|n{}|ms{:.0}",
                    rank,
                    region.lp_mass,
                    violation,
                    leaves,
                    t_region.elapsed().as_secs_f64() * 1000.0,
                ));
            }
            None => {
                failed += 1;
                details.push(format!("{rid}:fail{leaves}"));
            }
        }
    }

    info!(
        target: LOG_TARGET,
        "obstruction-probe: all-region-exact-rank solved={} failed={} skipped={} max_leaves={} sum_rank={} sum_mass={:.4} sum_violation={:.4} root_lp={:.4} rank_minus_root_lp={:.4} detail=[{}] solve_ms={:.1}",
        solved,
        failed,
        skipped,
        max_leaves,
        sum_rank,
        sum_mass,
        sum_violation,
        regions.lp_objective,
        sum_rank as f64 - regions.lp_objective,
        details.join(","),
        t0.elapsed().as_secs_f64() * 1000.0,
    );
}

#[derive(Clone)]
struct TreeSideCandidate {
    leaves: FixedBitSet,
    key: Vec<usize>,
    lp_mass: f64,
    frac_slack: f64,
}

#[derive(Clone)]
struct SolvedTreeSideCut {
    leaves: FixedBitSet,
    rank: usize,
    lp_mass: f64,
    violation: f64,
    detail: String,
}

fn probe_tree_side_exact_ranks(
    reduced: &Instance,
    state: &SearchState,
    lp: &crate::solvers::bp::rmp::RmpSolution,
    max_leaves: usize,
    limit: usize,
    include_nested_differences: bool,
    label: &str,
) {
    let t0 = Instant::now();
    let n = reduced.num_leaves as usize;
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    let mut candidates = Vec::new();

    for tree in &reduced.trees {
        let mut sides = Vec::with_capacity(tree.num_nodes());
        for node in 0..tree.num_nodes() {
            let mut side = FixedBitSet::with_capacity(n + 1);
            collect_subtree_leaves(tree, node as u32, &mut side);
            sides.push(side.clone());
            maybe_add_tree_side_candidate(
                state,
                lp,
                &mut seen,
                &mut candidates,
                &side,
                max_leaves,
                n,
            );

            let side_count = side.count_ones(..);
            if side_count > 0 && side_count < n {
                let mut complement = FixedBitSet::with_capacity(n + 1);
                for leaf in 1..=n {
                    if !side.contains(leaf) {
                        complement.insert(leaf);
                    }
                }
                maybe_add_tree_side_candidate(
                    state,
                    lp,
                    &mut seen,
                    &mut candidates,
                    &complement,
                    max_leaves,
                    n,
                );
            }
        }
        if include_nested_differences {
            for ancestor in 0..tree.num_nodes() {
                let ancestor_side = &sides[ancestor];
                for descendant in 0..tree.num_nodes() {
                    if ancestor == descendant
                        || !subtree_contains_node(tree, ancestor as u32, descendant as u32)
                    {
                        continue;
                    }
                    let descendant_side = &sides[descendant];
                    let mut difference = FixedBitSet::with_capacity(n + 1);
                    for leaf in ancestor_side.ones() {
                        if !descendant_side.contains(leaf) {
                            difference.insert(leaf);
                        }
                    }
                    maybe_add_tree_side_candidate(
                        state,
                        lp,
                        &mut seen,
                        &mut candidates,
                        &difference,
                        max_leaves,
                        n,
                    );
                }
            }
        }
    }

    candidates.sort_by(|a, b| {
        b.frac_slack
            .partial_cmp(&a.frac_slack)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.key.len().cmp(&a.key.len()))
            .then_with(|| a.key.cmp(&b.key))
    });
    let total_candidates = candidates.len();
    candidates.truncate(limit);

    let mut solved = 0usize;
    let mut failed = 0usize;
    let mut best_violation = f64::NEG_INFINITY;
    let mut best_detail = String::from("none");
    let mut positive = Vec::new();
    let mut positive_cuts = Vec::new();

    for (idx, cand) in candidates.iter().enumerate() {
        let (core, _) = klados_core::kernelize::restrict_instance_simple(reduced, &cand.leaves);
        let t_side = Instant::now();
        let exact = crate::solvers::bp::solve_subinstance(
            &core,
            &crate::solvers::bp::BpConfig::default(),
            &crate::solvers::bp::Cancel::new(Arc::new(AtomicBool::new(false))),
        );
        match exact {
            Some(forest) => {
                solved += 1;
                let rank = forest.len();
                let violation = rank as f64 - cand.lp_mass;
                let detail = format!(
                    "#{idx}:k{}|m{:.3}|v{:.3}|n{}|ms{:.0}|leaves={:?}",
                    rank,
                    cand.lp_mass,
                    violation,
                    cand.key.len(),
                    t_side.elapsed().as_secs_f64() * 1000.0,
                    cand.key,
                );
                if violation > best_violation {
                    best_violation = violation;
                    best_detail = detail.clone();
                }
                if violation > 1.0e-6 {
                    positive_cuts.push(SolvedTreeSideCut {
                        leaves: cand.leaves.clone(),
                        rank,
                        lp_mass: cand.lp_mass,
                        violation,
                        detail: detail.clone(),
                    });
                    if positive.len() < 8 {
                        positive.push(detail);
                    }
                }
            }
            None => {
                failed += 1;
            }
        }
    }

    info!(
        target: LOG_TARGET,
        "obstruction-probe: {} candidates={} tested={} solved={} failed={} max_leaves={} best={} positives=[{}] solve_ms={:.1}",
        label,
        total_candidates,
        candidates.len(),
        solved,
        failed,
        max_leaves,
        best_detail,
        positive.join(";"),
        t0.elapsed().as_secs_f64() * 1000.0,
    );

    if !positive_cuts.is_empty() {
        positive_cuts.sort_by(|a, b| {
            b.violation
                .partial_cmp(&a.violation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let positive_before_dedup = positive_cuts.len();
        positive_cuts = dedup_tree_side_cuts_by_pool_signature(state, positive_cuts);
        if positive_cuts.len() < positive_before_dedup {
            info!(
                target: LOG_TARGET,
                "obstruction-probe: {} pool-signature-dedup positive_before={} positive_after={}",
                label,
                positive_before_dedup,
                positive_cuts.len(),
            );
        }
        probe_restricted_pool_with_tree_side_cuts(reduced, state, "top", &positive_cuts, 8);

        // Decisive ceiling test: install ALL positive cuts at once (one round,
        // current pool). restricted-pool-cut-LP >= full-cut-LP, so if even this
        // does not reach the incumbent, the tree-side cut family is provably
        // insufficient and cut-aware pricing cannot help.
        probe_restricted_pool_with_tree_side_cuts(
            reduced,
            state,
            "all",
            &positive_cuts,
            positive_cuts.len(),
        );

        let diverse_cuts = select_diverse_tree_side_cuts(&positive_cuts, 8);
        probe_restricted_pool_with_tree_side_cuts(reduced, state, "diverse", &diverse_cuts, 8);

        let pool_diverse_cuts = select_pool_diverse_tree_side_cuts(state, &positive_cuts, 8);
        probe_restricted_pool_with_tree_side_cuts(
            reduced,
            state,
            "pooldiverse",
            &pool_diverse_cuts,
            8,
        );

        let marginal_cuts = select_marginal_tree_side_cuts(reduced, state, &positive_cuts, 8, 16);
        if !marginal_cuts.is_empty() {
            probe_restricted_pool_with_tree_side_cuts(
                reduced,
                state,
                "marginal",
                &marginal_cuts,
                8,
            );
        }
    }
}

fn dedup_tree_side_cuts_by_pool_signature(
    state: &SearchState,
    cuts: Vec<SolvedTreeSideCut>,
) -> Vec<SolvedTreeSideCut> {
    let mut best_by_signature: HashMap<Vec<usize>, SolvedTreeSideCut> = HashMap::new();
    for cut in cuts {
        let signature = local_rank_pool_signature(state.columns(), &cut.leaves);
        best_by_signature
            .entry(signature)
            .and_modify(|old| {
                if cut.rank > old.rank || (cut.rank == old.rank && cut.violation > old.violation) {
                    *old = cut.clone();
                }
            })
            .or_insert(cut);
    }
    let mut deduped = best_by_signature.into_values().collect::<Vec<_>>();
    deduped.sort_by(|a, b| {
        b.violation
            .partial_cmp(&a.violation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    deduped
}

fn local_rank_pool_signature(columns: &[AfColumn], leaves: &FixedBitSet) -> Vec<usize> {
    columns
        .iter()
        .enumerate()
        .filter_map(|(ci, col)| {
            col.labels()
                .iter()
                .any(|&label| leaves.contains(label as usize))
                .then_some(ci)
        })
        .collect()
}

fn select_diverse_tree_side_cuts(
    positive_cuts: &[SolvedTreeSideCut],
    max_cuts: usize,
) -> Vec<SolvedTreeSideCut> {
    let mut selected: Vec<SolvedTreeSideCut> = Vec::new();
    let mut remaining = positive_cuts.to_vec();

    while selected.len() < max_cuts && !remaining.is_empty() {
        let mut best_idx = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (idx, cand) in remaining.iter().enumerate() {
            let max_overlap = selected
                .iter()
                .map(|chosen| jaccard(&cand.leaves, &chosen.leaves))
                .fold(0.0, f64::max);
            // Keep violation dominant, but prefer cuts that expose a different
            // leaf side when violations are comparable.
            let score = cand.violation * (1.0 - 0.75 * max_overlap);
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        selected.push(remaining.swap_remove(best_idx));
    }
    selected
}

fn jaccard(a: &FixedBitSet, b: &FixedBitSet) -> f64 {
    let mut inter = a.clone();
    inter.intersect_with(b);
    let inter_count = inter.count_ones(..);
    if inter_count == 0 {
        return 0.0;
    }
    let mut union = a.clone();
    union.union_with(b);
    inter_count as f64 / union.count_ones(..) as f64
}

fn select_pool_diverse_tree_side_cuts(
    state: &SearchState,
    positive_cuts: &[SolvedTreeSideCut],
    max_cuts: usize,
) -> Vec<SolvedTreeSideCut> {
    let mut selected: Vec<(SolvedTreeSideCut, Vec<usize>)> = Vec::new();
    let mut remaining = positive_cuts
        .iter()
        .map(|cut| {
            (
                cut.clone(),
                local_rank_pool_signature(state.columns(), &cut.leaves),
            )
        })
        .collect::<Vec<_>>();

    while selected.len() < max_cuts && !remaining.is_empty() {
        let mut best_idx = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (idx, (cand, cand_sig)) in remaining.iter().enumerate() {
            let max_overlap = selected
                .iter()
                .map(|(_, chosen_sig)| sorted_jaccard_usize(cand_sig, chosen_sig))
                .fold(0.0, f64::max);
            let score = cand.violation * (1.0 - 0.75 * max_overlap);
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        selected.push(remaining.swap_remove(best_idx));
    }

    selected.into_iter().map(|(cut, _)| cut).collect()
}

fn sorted_jaccard_usize(a: &[usize], b: &[usize]) -> f64 {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut inter = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
        }
    }
    if inter == 0 {
        return 0.0;
    }
    inter as f64 / (a.len() + b.len() - inter) as f64
}

fn select_marginal_tree_side_cuts(
    reduced: &Instance,
    state: &SearchState,
    positive_cuts: &[SolvedTreeSideCut],
    max_cuts: usize,
    candidate_limit: usize,
) -> Vec<SolvedTreeSideCut> {
    let mut selected = Vec::new();
    let mut remaining = positive_cuts
        .iter()
        .take(candidate_limit)
        .cloned()
        .collect::<Vec<_>>();
    let mut current_obj = match evaluate_restricted_pool_with_tree_side_cuts(reduced, state, &[]) {
        Some(result) => result.side_obj,
        None => return selected,
    };

    while selected.len() < max_cuts && !remaining.is_empty() {
        let mut best_idx = None;
        let mut best_obj = current_obj;
        for idx in 0..remaining.len() {
            let mut trial = selected.clone();
            trial.push(remaining[idx].clone());
            let Some(result) =
                evaluate_restricted_pool_with_tree_side_cuts(reduced, state, &trial)
            else {
                continue;
            };
            if result.side_obj > best_obj + 1.0e-7 {
                best_obj = result.side_obj;
                best_idx = Some(idx);
            }
        }
        let Some(idx) = best_idx else {
            break;
        };
        selected.push(remaining.swap_remove(idx));
        current_obj = best_obj;
    }

    selected
}

struct RestrictedSideLpResult {
    base_obj: f64,
    side_obj: f64,
    cuts_total: usize,
    solve_ms: f64,
}

#[derive(Clone)]
struct ResidualCompletionCut {
    anchor_col: usize,
    compatible_cols: Vec<usize>,
    residual_rank: usize,
    anchor_value: f64,
    compatible_mass: f64,
    violation: f64,
    detail: String,
}

fn maybe_probe_residual_completion_cuts(
    reduced: &Instance,
    state: &SearchState,
    lp: &crate::solvers::bp::rmp::RmpSolution,
    cfg: &crate::solvers::bp::BpConfig,
) {
    if !cfg.residual_completion_probe {
        return;
    }

    let t0 = Instant::now();
    let mut candidates = lp
        .column_values
        .iter()
        .enumerate()
        .filter(|&(ci, value)| {
            *value > 1.0e-6
                && state.columns()[ci].size() >= 3
                && state.columns()[ci].size() < reduced.num_leaves as usize
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|(ai, av), (bi, bv)| {
        bv.partial_cmp(av)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| state.columns()[*bi].size().cmp(&state.columns()[*ai].size()))
            .then_with(|| ai.cmp(bi))
    });

    let total_candidates = candidates.len();
    candidates.truncate(cfg.residual_completion_max_cols);

    let mut solved = 0usize;
    let mut exact_solved = 0usize;
    let mut chen_solved = 0usize;
    let mut failed = 0usize;
    let mut skipped_large = 0usize;
    let mut skipped_empty = 0usize;
    let mut positive_cuts = Vec::new();
    let mut family_candidates = Vec::new();
    let mut best_violation = f64::NEG_INFINITY;
    let mut best_detail = String::from("none");

    for (rank_idx, &(ci, &anchor_value)) in candidates.iter().enumerate() {
        let anchor = &state.columns()[ci];
        let residual_leaves = residual_leafset(reduced.num_leaves as usize, anchor);
        let residual_count = residual_leaves.count_ones(..);
        if residual_count == 0 {
            skipped_empty += 1;
            continue;
        }

        let (core, _) = klados_core::kernelize::restrict_instance_simple(reduced, &residual_leaves);
        let (residual_rank, source) = if residual_count == 1 {
            (Some(1), "single")
        } else if residual_count > cfg.residual_completion_max_residual_leaves {
            skipped_large += 1;
            (Some(chen_lower_bound(&core.trees, false)), "chen-large")
        } else {
            (
                crate::solvers::bp::solve_subinstance(
                    &core,
                    &crate::solvers::bp::BpConfig::default(),
                    &crate::solvers::bp::Cancel::new(Arc::new(AtomicBool::new(false))),
                )
                .map(|forest| forest.len()),
                "exact",
            )
        };

        let Some(residual_rank) = residual_rank else {
            failed += 1;
            continue;
        };
        solved += 1;
        if source == "exact" {
            exact_solved += 1;
        } else if source == "chen-large" {
            chen_solved += 1;
        }

        let compatible_cols = compatible_columns_for_anchor(state.columns(), ci);
        let compatible_mass: f64 = compatible_cols
            .iter()
            .map(|&cj| lp.column_values.get(cj).copied().unwrap_or(0.0))
            .sum();
        let violation = residual_rank as f64 * anchor_value - compatible_mass;
        let max_possible_violation = residual_count as f64 * anchor_value - compatible_mass;
        let detail = format!(
            "#{rank_idx}:c{}|size{}|x{:.3}|res{}|rank{}|{}|cmass{:.3}|v{:.3}|vmax{:.3}|compat{}",
            ci,
            anchor.size(),
            anchor_value,
            residual_count,
            residual_rank,
            source,
            compatible_mass,
            violation,
            max_possible_violation,
            compatible_cols.len(),
        );
        if violation > best_violation {
            best_violation = violation;
            best_detail = detail.clone();
        }
        let cut = ResidualCompletionCut {
            anchor_col: ci,
            compatible_cols,
            residual_rank,
            anchor_value,
            compatible_mass,
            violation,
            detail,
        };
        if cut.violation > 1.0e-6 {
            positive_cuts.push(cut.clone());
        }
        family_candidates.push(cut);
    }

    positive_cuts.sort_by(|a, b| {
        b.violation
            .partial_cmp(&a.violation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let positives = positive_cuts
        .iter()
        .take(8)
        .map(|cut| cut.detail.as_str())
        .collect::<Vec<_>>()
        .join(";");

    info!(
        target: LOG_TARGET,
        "obstruction-probe: residual-completion candidates={} tested={} solved={} exact_solved={} chen_large={} failed={} skipped_large={} skipped_empty={} max_residual_leaves={} best={} positives=[{}] solve_ms={:.1}",
        total_candidates,
        candidates.len(),
        solved,
        exact_solved,
        chen_solved,
        failed,
        skipped_large,
        skipped_empty,
        cfg.residual_completion_max_residual_leaves,
        best_detail,
        positives,
        t0.elapsed().as_secs_f64() * 1000.0,
    );

    if !positive_cuts.is_empty() {
        probe_restricted_pool_with_residual_completion_cuts(reduced, state, &positive_cuts, 8);
    }
    if family_candidates.len() >= 2 {
        probe_anchor_family_residual_completion(reduced, state, lp, &family_candidates, 16);
    }
}

fn residual_leafset(n: usize, anchor: &AfColumn) -> FixedBitSet {
    let mut leaves = FixedBitSet::with_capacity(n + 1);
    for leaf in 1..=n {
        leaves.insert(leaf);
    }
    for &leaf in anchor.labels() {
        leaves.set(leaf as usize, false);
    }
    leaves
}

fn compatible_columns_for_anchor(columns: &[AfColumn], anchor_col: usize) -> Vec<usize> {
    let anchor = &columns[anchor_col];
    columns
        .iter()
        .enumerate()
        .filter_map(|(ci, col)| {
            (ci != anchor_col && !columns_conflict(anchor, col)).then_some(ci)
        })
        .collect()
}

fn evaluate_restricted_pool_with_residual_completion_cuts(
    reduced: &Instance,
    state: &SearchState,
    selected_cuts: &[ResidualCompletionCut],
) -> Option<RestrictedSideLpResult> {
    let t0 = Instant::now();
    let mut rmp = Rmp::new(state.columns(), &reduced.trees, reduced.num_leaves as usize);
    let mut base = None;
    let mut cuts_total = 0usize;
    for _ in 0..64 {
        let Ok(lp) = rmp.solve() else {
            return None;
        };
        let cuts = rmp.separate_and_add_cuts(state.columns(), &lp.column_values, 1.0e-6);
        cuts_total += cuts;
        if cuts == 0 {
            base = Some(lp.objective);
            break;
        }
    }

    let base_obj = base?;
    for cut in selected_cuts {
        rmp.add_diagnostic_residual_completion_row(
            cut.anchor_col,
            &cut.compatible_cols,
            cut.residual_rank,
        );
    }

    let side_obj = match rmp.solve() {
        Ok(lp) => lp.objective,
        Err(_) => return None,
    };

    Some(RestrictedSideLpResult {
        base_obj,
        side_obj,
        cuts_total,
        solve_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

fn probe_restricted_pool_with_residual_completion_cuts(
    reduced: &Instance,
    state: &SearchState,
    positive_cuts: &[ResidualCompletionCut],
    max_cuts: usize,
) {
    let installed = positive_cuts.len().min(max_cuts);
    let Some(result) = evaluate_restricted_pool_with_residual_completion_cuts(
        reduced,
        state,
        &positive_cuts[..installed],
    ) else {
        info!(
            target: LOG_TARGET,
            "obstruction-probe: residual-completion-restricted-lp failed before rows"
        );
        return;
    };

    let details = positive_cuts
        .iter()
        .take(installed)
        .map(|cut| {
            format!(
                "c{}|r{}|x{:.3}|cm{:.3}|v{:.3}|{}",
                cut.anchor_col,
                cut.residual_rank,
                cut.anchor_value,
                cut.compatible_mass,
                cut.violation,
                cut.detail
            )
        })
        .collect::<Vec<_>>()
        .join(";");
    info!(
        target: LOG_TARGET,
        "obstruction-probe: residual-completion-restricted-lp base={:.4} row_obj={:.4} delta={:.4} installed={} node_cuts={} details=[{}] solve_ms={:.1}",
        result.base_obj,
        result.side_obj,
        result.side_obj - result.base_obj,
        installed,
        result.cuts_total,
        details,
        result.solve_ms,
    );
}

fn probe_anchor_family_residual_completion(
    reduced: &Instance,
    state: &SearchState,
    lp: &crate::solvers::bp::rmp::RmpSolution,
    candidates: &[ResidualCompletionCut],
    max_anchors: usize,
) {
    let mut ordered = candidates.to_vec();
    ordered.sort_by(|a, b| {
        let bd = b.residual_rank as f64 * b.anchor_value;
        let ad = a.residual_rank as f64 * a.anchor_value;
        bd.partial_cmp(&ad)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.residual_rank.cmp(&a.residual_rank))
    });

    let mut family = Vec::new();
    for cand in ordered {
        if family.len() >= max_anchors {
            break;
        }
        if family.iter().all(|chosen: &ResidualCompletionCut| {
            columns_conflict(
                &state.columns()[cand.anchor_col],
                &state.columns()[chosen.anchor_col],
            )
        }) {
            family.push(cand);
        }
    }

    if family.len() < 2 {
        info!(
            target: LOG_TARGET,
            "obstruction-probe: anchor-family-residual skipped clique_size={}",
            family.len()
        );
        return;
    }

    let mut compatible_seen = FixedBitSet::with_capacity(state.columns().len());
    for cut in &family {
        for &ci in &cut.compatible_cols {
            compatible_seen.insert(ci);
        }
    }
    let compatible_union = compatible_seen.ones().collect::<Vec<_>>();
    let lhs: f64 = compatible_union
        .iter()
        .map(|&ci| lp.column_values.get(ci).copied().unwrap_or(0.0))
        .sum();
    let rhs: f64 = family
        .iter()
        .map(|cut| cut.residual_rank as f64 * cut.anchor_value)
        .sum();
    let violation = rhs - lhs;

    let anchor_terms = family
        .iter()
        .map(|cut| (cut.anchor_col, cut.residual_rank))
        .collect::<Vec<_>>();
    let details = family
        .iter()
        .map(|cut| {
            format!(
                "c{}|r{}|x{:.3}|d{:.3}",
                cut.anchor_col,
                cut.residual_rank,
                cut.anchor_value,
                cut.residual_rank as f64 * cut.anchor_value
            )
        })
        .collect::<Vec<_>>()
        .join(";");

    info!(
        target: LOG_TARGET,
        "obstruction-probe: anchor-family-residual clique={} union_cols={} lhs={:.4} rhs={:.4} violation={:.4} anchors=[{}]",
        family.len(),
        compatible_union.len(),
        lhs,
        rhs,
        violation,
        details,
    );

    if violation <= 1.0e-6 {
        return;
    }

    let t0 = Instant::now();
    let mut rmp = Rmp::new(state.columns(), &reduced.trees, reduced.num_leaves as usize);
    let mut base = None;
    let mut cuts_total = 0usize;
    for _ in 0..64 {
        let Ok(lp) = rmp.solve() else {
            info!(
                target: LOG_TARGET,
                "obstruction-probe: anchor-family-residual-restricted-lp failed before row"
            );
            return;
        };
        let cuts = rmp.separate_and_add_cuts(state.columns(), &lp.column_values, 1.0e-6);
        cuts_total += cuts;
        if cuts == 0 {
            base = Some(lp.objective);
            break;
        }
    }
    let Some(base_obj) = base else {
        info!(
            target: LOG_TARGET,
            "obstruction-probe: anchor-family-residual-restricted-lp failed to converge base rows"
        );
        return;
    };
    rmp.add_diagnostic_anchor_family_residual_row(&anchor_terms, &compatible_union);
    let row_obj = match rmp.solve() {
        Ok(lp) => lp.objective,
        Err(_) => {
            info!(
                target: LOG_TARGET,
                "obstruction-probe: anchor-family-residual-restricted-lp failed after row"
            );
            return;
        }
    };

    info!(
        target: LOG_TARGET,
        "obstruction-probe: anchor-family-residual-restricted-lp base={:.4} row_obj={:.4} delta={:.4} clique={} node_cuts={} solve_ms={:.1}",
        base_obj,
        row_obj,
        row_obj - base_obj,
        family.len(),
        cuts_total,
        t0.elapsed().as_secs_f64() * 1000.0,
    );
}

fn evaluate_restricted_pool_with_tree_side_cuts(
    reduced: &Instance,
    state: &SearchState,
    selected_cuts: &[SolvedTreeSideCut],
) -> Option<RestrictedSideLpResult> {
    let t0 = Instant::now();
    let mut rmp = Rmp::new(state.columns(), &reduced.trees, reduced.num_leaves as usize);
    let mut base = None;
    let mut cuts_total = 0usize;
    for _ in 0..64 {
        let Ok(lp) = rmp.solve() else {
            return None;
        };
        let cuts = rmp.separate_and_add_cuts(state.columns(), &lp.column_values, 1.0e-6);
        cuts_total += cuts;
        if cuts == 0 {
            base = Some(lp.objective);
            break;
        }
    }

    let base_obj = base?;
    for cut in selected_cuts {
        rmp.add_diagnostic_local_rank_row(state.columns(), &cut.leaves, cut.rank);
    }

    let side_obj = match rmp.solve() {
        Ok(lp) => lp.objective,
        Err(_) => return None,
    };

    Some(RestrictedSideLpResult {
        base_obj,
        side_obj,
        cuts_total,
        solve_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

fn probe_restricted_pool_with_tree_side_cuts(
    reduced: &Instance,
    state: &SearchState,
    label: &str,
    positive_cuts: &[SolvedTreeSideCut],
    max_cuts: usize,
) {
    let installed = positive_cuts.len().min(max_cuts);
    let Some(result) =
        evaluate_restricted_pool_with_tree_side_cuts(reduced, state, &positive_cuts[..installed])
    else {
        info!(
            target: LOG_TARGET,
            "obstruction-probe: tree-side-restricted-lp failed before side rows"
        );
        return;
    };

    let details = positive_cuts
        .iter()
        .take(installed)
        .map(|cut| {
            format!(
                "k{}|m{:.3}|v{:.3}|{}",
                cut.rank, cut.lp_mass, cut.violation, cut.detail
            )
        })
        .collect::<Vec<_>>()
        .join(";");
    info!(
        target: LOG_TARGET,
        "obstruction-probe: tree-side-restricted-lp label={} base={:.4} side_obj={:.4} delta={:.4} installed={} node_cuts={} details=[{}] solve_ms={:.1}",
        label,
        result.base_obj,
        result.side_obj,
        result.side_obj - result.base_obj,
        installed,
        result.cuts_total,
        details,
        result.solve_ms,
    );
}

fn maybe_add_tree_side_candidate(
    state: &SearchState,
    lp: &crate::solvers::bp::rmp::RmpSolution,
    seen: &mut HashSet<Vec<usize>>,
    candidates: &mut Vec<TreeSideCandidate>,
    leaves: &FixedBitSet,
    max_leaves: usize,
    n: usize,
) {
    let count = leaves.count_ones(..);
    if count < 2 || count > max_leaves || count >= n {
        return;
    }
    let key = leaves.ones().collect::<Vec<_>>();
    if !seen.insert(key.clone()) {
        return;
    }
    let lp_mass = lp
        .column_values
        .iter()
        .enumerate()
        .filter_map(|(ci, &value)| {
            (value > 1.0e-9
                && state.columns()[ci]
                    .labels()
                    .iter()
                    .any(|&leaf| leaves.contains(leaf as usize)))
            .then_some(value)
        })
        .sum::<f64>();
    let ceil = (lp_mass - 1.0e-6).ceil();
    candidates.push(TreeSideCandidate {
        leaves: leaves.clone(),
        key,
        lp_mass,
        frac_slack: ceil - lp_mass,
    });
}

fn collect_subtree_leaves(tree: &Tree, root: u32, out: &mut FixedBitSet) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if tree.is_leaf(node) {
            out.insert(tree.label[node as usize] as usize);
        } else {
            let (left, right) = tree.children_pair(node);
            stack.push(left);
            stack.push(right);
        }
    }
}

fn subtree_contains_node(tree: &Tree, root: u32, needle: u32) -> bool {
    if root == needle {
        return true;
    }
    if tree.is_leaf(root) {
        return false;
    }
    let (left, right) = tree.children_pair(root);
    subtree_contains_node(tree, left, needle) || subtree_contains_node(tree, right, needle)
}

fn solve_region_support_mip(
    reduced: &Instance,
    state: &SearchState,
    region: &SupportComponentSummary,
) -> (u32, usize, usize, usize, Option<usize>) {
    let (core, reverse_map) =
        klados_core::kernelize::restrict_instance_simple(reduced, &region.leaves);
    let mut old_to_new = vec![0u32; reduced.num_leaves as usize + 1];
    for (new_label, &old_label) in reverse_map.iter().enumerate().skip(1) {
        old_to_new[old_label as usize] = new_label as u32;
    }

    let mut builder = ColumnBuilder::new(&core.trees);
    let mut local_columns = Vec::with_capacity(region.column_ids.len());
    let mut rejected = 0usize;
    for &ci in &region.column_ids {
        // Reconstruct the region column in the local label space.
        // The caller only passes columns from one support region, so every
        // label is guaranteed to map.
        let labels = state.columns()[ci]
            .labels()
            .iter()
            .map(|&old_label| old_to_new[old_label as usize])
            .collect::<Vec<_>>();
        if labels.contains(&0) {
            rejected += 1;
            continue;
        }
        if let Some(col) = builder.try_build(labels, &core.trees) {
            local_columns.push(col);
        } else {
            rejected += 1;
        }
    }

    if local_columns.is_empty() {
        return (core.num_leaves, 0, rejected, 0, None);
    }

    let mut rmp = Rmp::new(&local_columns, &core.trees, core.num_leaves as usize);
    let mut cuts_total = 0usize;
    for _ in 0..32 {
        let Ok(Some(mip)) = rmp.solve_mip_with_time_limit(2.0) else {
            break;
        };
        let cuts = rmp.separate_and_add_cuts(&local_columns, &mip.column_values, 0.5);
        cuts_total += cuts;
        if cuts > 0 {
            continue;
        }
        let chosen = mip.column_values.iter().filter(|&&v| v > 0.5).count();
        return (
            core.num_leaves,
            local_columns.len(),
            rejected,
            cuts_total,
            Some(chosen),
        );
    }
    (
        core.num_leaves,
        local_columns.len(),
        rejected,
        cuts_total,
        None,
    )
}

struct RankCutProbeBest {
    label: String,
    cols: usize,
    leaves: usize,
    lp_mass: f64,
    alpha: usize,
    violation: f64,
}

fn maybe_probe_root_rank_cuts(
    state: &SearchState,
    lp: &crate::solvers::bp::rmp::RmpSolution,
    cfg: &crate::solvers::bp::BpConfig,
) {
    if !cfg.rank_cut_probe {
        return;
    }
    let t0 = Instant::now();
    let support = lp
        .column_values
        .iter()
        .enumerate()
        .filter_map(|(ci, &v)| (v > 1.0e-6).then_some(ci))
        .collect::<Vec<_>>();
    if support.is_empty() {
        info!(target: LOG_TARGET, "rank-cut-probe: empty LP support");
        return;
    }

    let mut parent: Vec<usize> = (0..support.len()).collect();
    let mut rank = vec![0u8; support.len()];

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            let root = find(parent, parent[x]);
            parent[x] = root;
        }
        parent[x]
    }
    fn union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
        let mut ra = find(parent, a);
        let mut rb = find(parent, b);
        if ra == rb {
            return;
        }
        if rank[ra] < rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        parent[rb] = ra;
        if rank[ra] == rank[rb] {
            rank[ra] += 1;
        }
    }

    let mut resource_owner: HashMap<(usize, usize), usize> = HashMap::new();
    for (local, &ci) in support.iter().enumerate() {
        let col = &state.columns()[ci];
        for &leaf in col.labels() {
            let key = (usize::MAX, leaf as usize);
            if let Some(&prev) = resource_owner.get(&key) {
                union(&mut parent, &mut rank, local, prev);
            } else {
                resource_owner.insert(key, local);
            }
        }
        for (ti, nodes) in col.coverage().iter_per_tree().enumerate() {
            for &node in nodes {
                let key = (ti, node);
                if let Some(&prev) = resource_owner.get(&key) {
                    union(&mut parent, &mut rank, local, prev);
                } else {
                    resource_owner.insert(key, local);
                }
            }
        }
    }

    let mut by_root: HashMap<usize, Vec<usize>> = HashMap::new();
    for local in 0..support.len() {
        let root = find(&mut parent, local);
        by_root.entry(root).or_default().push(local);
    }
    let mut comps = by_root.into_values().collect::<Vec<_>>();
    comps.sort_by(|a, b| b.len().cmp(&a.len()));

    let max_cols = cfg.rank_cut_probe_max_cols.min(96).max(8);
    let mut tested = 0usize;
    let mut skipped_large = 0usize;
    let mut best: Option<RankCutProbeBest> = None;

    for (comp_idx, comp) in comps.iter().enumerate() {
        if comp.len() <= max_cols {
            rank_cut_probe_subset(
                state,
                lp,
                &support,
                comp,
                format!("comp#{comp_idx}"),
                &mut tested,
                &mut best,
            );
        } else {
            skipped_large += 1;
            let mut sample = comp.clone();
            sample.sort_by(|&a, &b| {
                lp.column_values[support[b]]
                    .partial_cmp(&lp.column_values[support[a]])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            sample.truncate(max_cols);
            rank_cut_probe_subset(
                state,
                lp,
                &support,
                &sample,
                format!("comp#{comp_idx}:top{max_cols}"),
                &mut tested,
                &mut best,
            );
        }
    }

    let best_msg = best
        .map(|b| {
            format!(
                "{} cols={} leaves={} lp_mass={:.6} alpha={} violation={:.6}",
                b.label, b.cols, b.leaves, b.lp_mass, b.alpha, b.violation
            )
        })
        .unwrap_or_else(|| "none".to_string());
    info!(
        target: LOG_TARGET,
        "rank-cut-probe: root_lp={:.6} support_cols={} conflict_components={} tested={} skipped_large={} cap={} best={} solve_ms={:.1}",
        lp.objective,
        support.len(),
        comps.len(),
        tested,
        skipped_large,
        max_cols,
        best_msg,
        t0.elapsed().as_secs_f64() * 1000.0,
    );
}

fn rank_cut_probe_subset(
    state: &SearchState,
    lp: &crate::solvers::bp::rmp::RmpSolution,
    support: &[usize],
    subset: &[usize],
    label: String,
    tested: &mut usize,
    best: &mut Option<RankCutProbeBest>,
) {
    if subset.is_empty() {
        return;
    }
    *tested += 1;
    let cols = subset
        .iter()
        .map(|&local| support[local])
        .collect::<Vec<_>>();
    let lp_mass = cols.iter().map(|&ci| lp.column_values[ci]).sum::<f64>();
    let alpha = exact_support_alpha(state.columns(), &cols);
    let violation = lp_mass - alpha as f64;
    let mut leaves = FixedBitSet::with_capacity(state.num_leaves() + 1);
    for &ci in &cols {
        for &leaf in state.columns()[ci].labels() {
            leaves.insert(leaf as usize);
        }
    }

    let replace = best
        .as_ref()
        .is_none_or(|old| violation > old.violation + 1.0e-9);
    if replace {
        *best = Some(RankCutProbeBest {
            label,
            cols: cols.len(),
            leaves: leaves.count_ones(..),
            lp_mass,
            alpha,
            violation,
        });
    }
}

fn exact_support_alpha(columns: &[AfColumn], col_ids: &[usize]) -> usize {
    let n = col_ids.len();
    debug_assert!(n <= 128);
    let mut adj = vec![0u128; n];
    for i in 0..n {
        for j in (i + 1)..n {
            if columns_conflict(&columns[col_ids[i]], &columns[col_ids[j]]) {
                adj[i] |= 1u128 << j;
                adj[j] |= 1u128 << i;
            }
        }
    }
    let all = if n == 128 {
        u128::MAX
    } else {
        (1u128 << n) - 1
    };
    let mut best = 0usize;
    mis_branch_and_bound(&adj, all, 0, &mut best);
    best
}

fn mis_branch_and_bound(adj: &[u128], cand: u128, chosen: usize, best: &mut usize) {
    if cand == 0 {
        *best = (*best).max(chosen);
        return;
    }
    if chosen + cand.count_ones() as usize <= *best {
        return;
    }

    let mut scan = cand;
    let mut pivot = scan.trailing_zeros() as usize;
    let mut pivot_deg = 0u32;
    while scan != 0 {
        let v = scan.trailing_zeros() as usize;
        let deg = (adj[v] & cand).count_ones();
        if deg > pivot_deg {
            pivot = v;
            pivot_deg = deg;
        }
        scan &= scan - 1;
    }

    let without_pivot = cand & !(1u128 << pivot);
    mis_branch_and_bound(adj, without_pivot & !adj[pivot], chosen + 1, best);
    mis_branch_and_bound(adj, without_pivot, chosen, best);
}

fn columns_conflict(a: &AfColumn, b: &AfColumn) -> bool {
    if sorted_slices_intersect_u32(a.labels(), b.labels()) {
        return true;
    }
    a.coverage()
        .iter_per_tree()
        .zip(b.coverage().iter_per_tree())
        .any(|(an, bn)| sorted_slices_intersect_usize(an, bn))
}

fn sorted_slices_intersect_u32(a: &[u32], b: &[u32]) -> bool {
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

fn sorted_slices_intersect_usize(a: &[usize], b: &[usize]) -> bool {
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

fn maybe_log_bridge_footprint(
    label: &str,
    incumbent: Option<&Incumbent>,
    columns: &[AfColumn],
    root_regions: Option<&RootSupportRegions>,
    bridge_probe: bool,
) {
    if !bridge_probe {
        return;
    }
    let (Some(inc), Some(regions)) = (incumbent, root_regions) else {
        return;
    };

    let mut bridge_columns = 0usize;
    let mut bridge_savings = 0usize;
    let mut max_regions_touched = 0usize;
    let mut touched_hist = vec![0usize; regions.comps.len().max(1) + 1];
    let mut off_support_columns = 0usize;
    let mut local_component_counts = vec![0usize; regions.comps.len()];
    let mut support_ids = FixedBitSet::with_capacity(columns.len());
    for comp in &regions.comps {
        for &ci in &comp.column_ids {
            support_ids.insert(ci);
        }
    }

    for &ci in &inc.component_columns {
        if !support_ids.contains(ci) {
            off_support_columns += 1;
        }
        let mut touched = FixedBitSet::with_capacity(regions.comps.len());
        for &leaf in columns[ci].labels() {
            let rid = regions.region_of_leaf[leaf as usize];
            if rid != usize::MAX {
                touched.insert(rid);
            }
        }
        let q = touched.count_ones(..);
        if q < touched_hist.len() {
            touched_hist[q] += 1;
        }
        if q == 1
            && let Some(rid) = touched.ones().next()
        {
            local_component_counts[rid] += 1;
        }
        if q > 1 {
            bridge_columns += 1;
            bridge_savings += q - 1;
            max_regions_touched = max_regions_touched.max(q);
        }
    }

    info!(
        target: LOG_TARGET,
        "bridge-probe: {label} k={} ceil_sum={} delta={} bridge_cols={} bridge_savings={} off_support_cols={} max_regions_touched={} touched_hist={:?}",
        inc.k,
        regions.component_ceil_sum,
        inc.k as isize - regions.component_ceil_sum as isize,
        bridge_columns,
        bridge_savings,
        off_support_columns,
        max_regions_touched,
        touched_hist,
    );
    let per_region = regions
        .comps
        .iter()
        .enumerate()
        .map(|(rid, comp)| {
            let ceil = (comp.lp_mass - 1.0e-6).ceil() as usize;
            format!(
                "{rid}:{}|{}|{:.3}",
                local_component_counts[rid], ceil, comp.lp_mass
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    info!(
        target: LOG_TARGET,
        "bridge-probe: {label} per_region local|ceil|mass=[{}]",
        per_region,
    );
}

/// Greedy primal heuristic: pick columns in descending `x_c` order, skipping
/// any that overlap already-covered leaves; backfill with singletons. Always
/// produces a *feasible* integer AF for an instance where singletons are in
/// the pool. Returns `None` only if a leaf has no singleton column in the
/// pool (shouldn't happen since we seed singletons up front).
///
/// This is the single most-impactful piece of "matching bp-multi" we were
/// missing: when LP=optimum but simplex picks a fractional optimal basis,
/// rounding still recovers the integer solution.
fn try_round_primal(state: &SearchState, values: &[f64]) -> Option<Incumbent> {
    let n = state.num_leaves();
    let mut indexed: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v > 1.0e-6)
        .map(|(i, &v)| (i, v))
        .collect();
    if indexed.is_empty() {
        return None;
    }
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut covered_leaves = vec![false; n + 1];
    let num_trees = state
        .columns()
        .first()
        .map(|c| c.coverage().iter_per_tree().count())
        .unwrap_or(0);
    let mut covered_nodes = vec![std::collections::HashSet::new(); num_trees];

    let mut chosen = Vec::new();
    for &(ci, _) in &indexed {
        let col = &state.columns()[ci];
        let labels = col.labels();
        if labels.iter().any(|&l| covered_leaves[l as usize]) {
            continue;
        }

        let mut node_overlap = false;
        for (ti, nodes) in col.coverage().iter_per_tree().enumerate() {
            if nodes.iter().any(|n| covered_nodes[ti].contains(n)) {
                node_overlap = true;
                break;
            }
        }
        if node_overlap {
            continue;
        }

        for &l in labels {
            covered_leaves[l as usize] = true;
        }
        for (ti, nodes) in col.coverage().iter_per_tree().enumerate() {
            for &n in nodes {
                covered_nodes[ti].insert(n);
            }
        }
        chosen.push(ci);
    }
    // Backfill singletons for uncovered leaves. We assume singletons are
    // seeded first (id l-1 for label l in solve_inner's seed_singletons).
    for label in 1..=n {
        if !covered_leaves[label] {
            // Linear scan to find singleton column for this label. With
            // singletons seeded first, this is just `label - 1`, but we
            // verify defensively.
            let singleton_ci = label - 1;
            if singleton_ci < state.columns().len()
                && state.columns()[singleton_ci].labels() == [label as u32]
            {
                chosen.push(singleton_ci);
                covered_leaves[label] = true;
            } else {
                let pos = state
                    .columns()
                    .iter()
                    .position(|col| col.labels() == [label as u32])?;
                chosen.push(pos);
                covered_leaves[label] = true;
            }
        }
    }
    let k = chosen.len();
    Some(Incumbent {
        component_columns: chosen,
        k,
    })
}

/// If the LP support is an integer partition (every leaf covered exactly once
/// by a column at value 1), return the corresponding incumbent.
fn try_integral(state: &SearchState, values: &[f64]) -> Option<Incumbent> {
    let n = state.num_leaves();
    let mut cover = vec![0u32; n + 1];
    let mut chosen = Vec::new();

    let num_trees = state
        .columns()
        .first()
        .map(|c| c.coverage().iter_per_tree().count())
        .unwrap_or(0);
    let mut covered_nodes = vec![std::collections::HashSet::new(); num_trees];

    for (ci, &v) in values.iter().enumerate() {
        if v <= 1.0e-6 {
            continue;
        }
        if (v - 1.0).abs() > 1.0e-6 {
            debug!(target: LOG_TARGET, "try_integral failed: variable {} is fractional ({})", ci, v);
            return None;
        }
        let col = &state.columns()[ci];

        for (ti, nodes) in col.coverage().iter_per_tree().enumerate() {
            for &n in nodes {
                if !covered_nodes[ti].insert(n) {
                    debug!(target: LOG_TARGET, "try_integral failed: node overlap at tree {}, node {}", ti, n);
                    return None; // Node constraint violated
                }
            }
        }

        for &l in col.labels() {
            cover[l as usize] += 1;
            if cover[l as usize] > 1 {
                debug!(target: LOG_TARGET, "try_integral failed: leaf {} covered multiple times", l);
                return None;
            }
        }
        chosen.push(ci);
    }
    if (1..=n).any(|l| cover[l] == 0) {
        debug!(target: LOG_TARGET, "try_integral failed: some leaves are not covered");
        return None;
    }
    let k = chosen.len();
    Some(Incumbent {
        component_columns: chosen,
        k,
    })
}

/// Old bp-multi accepted an LP support as an incumbent whenever the positive
/// columns formed a leaf partition, even if the LP values themselves were
/// fractional.  That is a rounding heuristic, not an LP-integrality proof, but
/// it is extremely useful on degenerate roots where the simplex returns a
/// 0.5/0.5 basis over disjoint components.  Keep it safe by validating the
/// reconstructed forest before installing the incumbent.
fn try_support_partition(
    state: &SearchState,
    values: &[f64],
    reduced: &Instance,
) -> Option<Incumbent> {
    let n = state.num_leaves();
    let mut cover = vec![0u32; n + 1];
    let mut chosen = Vec::new();

    for (ci, &v) in values.iter().enumerate() {
        if v <= 1.0e-9 {
            continue;
        }
        let col = &state.columns()[ci];
        for &l in col.labels() {
            let idx = l as usize;
            cover[idx] += 1;
            if cover[idx] > 1 {
                return None;
            }
        }
        chosen.push(ci);
    }

    if chosen.is_empty() || (1..=n).any(|l| cover[l] != 1) {
        return None;
    }

    let inc = Incumbent {
        k: chosen.len(),
        component_columns: chosen,
    };
    let components = reconstruct_components(&inc, state.columns(), reduced);
    if validate_agreement_forest(reduced, &components).is_ok() {
        Some(inc)
    } else {
        None
    }
}

/// Convert an incumbent into AF components in the reduced label space.
fn reconstruct_components(inc: &Incumbent, columns: &[AfColumn], reduced: &Instance) -> Vec<Tree> {
    let n = reduced.num_leaves;
    let mut covered = FixedBitSet::with_capacity(n as usize + 1);
    let mut out = Vec::with_capacity(inc.component_columns.len());
    for &ci in &inc.component_columns {
        let labels = columns[ci].labels();
        if labels.len() == 1 {
            covered.insert(labels[0] as usize);
            out.push(Tree::singleton(labels[0], n));
        } else {
            let mut leafset = FixedBitSet::with_capacity(n as usize + 1);
            for &l in labels {
                leafset.insert(l as usize);
            }
            covered.union_with(&leafset);
            out.push(Tree::component_from_leafset(
                &leafset,
                reduced.reference_tree(),
                n,
            ));
        }
    }
    // Defensive: seed any missing leaf as a singleton (shouldn't happen if
    // the LP integer check passed, but cheap insurance).
    for label in 1..=n {
        if !covered.contains(label as usize) {
            out.push(Tree::singleton(label, n));
        }
    }
    out
}
