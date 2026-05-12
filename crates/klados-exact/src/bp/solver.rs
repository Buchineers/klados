//! Branch-and-price search loop.
//!
//! Branches exclusively on leaf pairs (must-link / cannot-link). The column
//! pool grows append-only and is shared across all branches; branchings
//! never reference column ids. See [`crate::bp::search::branchings`] for
//! the rationale.

use std::time::Instant;

use fixedbitset::FixedBitSet;
use klados_core::af_validator::validate_agreement_forest;
use klados_core::{Instance, Tree};
use log::{debug, info, trace};

use crate::bp::column::{AfColumn, ColumnBuilder};
use crate::bp::pricer::{Pricer, PricerScratch, PricingContext, PricingResult, dispatch_by_m};
use crate::bp::rmp::Rmp;
use crate::bp::search::{
    BranchSelector, Branchings, Incumbent, SearchState, SelectionContext, Telemetry,
};
use crate::chen_rspr::chen_pair_agreement;
use crate::whidden_cluster::try_whidden_relaxed_incumbent_2tree;

const LOG_TARGET: &str = "klados::bp";

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
        "primal heuristic (whidden relaxed): installed incumbent k={}",
        candidate_k,
    );
}

enum NodeOutcome {
    Pruned,
    Integral(Incumbent),
    /// LP is fractional; branch on `pair`.
    Branch {
        lp_obj: f64,
        pair: crate::bp::search::LeafPair,
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
fn can_prune_by_bound(lb: usize, best_ub: usize) -> bool {
    if std::env::var("KLADOS_BP_DISABLE_BOUND_PRUNE").is_ok() {
        return false;
    }
    lb >= best_ub
}

/// Solve a kernelized, undecomposable subinstance.
///
/// Caller must guarantee `m ≥ 2` and `n ≥ 2` (the pipeline's
/// `trivial_solution` short-circuit handles the trivial cases).
pub fn solve_inner(reduced: &Instance) -> Option<Vec<Tree>> {
    let trees = &reduced.trees;
    let n = reduced.num_leaves as usize;
    debug_assert!(trees.len() >= 2 && n >= 2);

    // Seed singletons via a temporary builder; the runtime builder lives in
    // PricerScratch so all pricer tiers share it.
    let mut seed_builder = ColumnBuilder::new(trees);
    let initial: Vec<AfColumn> = (1..=n as u32)
        .map(|l| seed_builder.build_unchecked(vec![l], trees))
        .collect();
    let mut state = SearchState::seed_singletons(n, initial.clone());
    let mut rmp = Rmp::new(&initial, trees, n);

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
            if let Some(_) = state.add_column(column) {
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
    if trees.len() >= 2 && reduced.num_leaves >= 20 {
        if let Some(incumbent_forest) = try_whidden_relaxed_incumbent_2tree(reduced, &mut |sub| {
            crate::bp::solve_subinstance(sub, &crate::bp::BpConfig::default())
        }) {
            install_incumbent(
                &mut state,
                &mut rmp,
                trees,
                &mut seed_builder,
                incumbent_forest,
            );
        }
    }

    let mut scratch = PricerScratch::new(trees);
    let mut pricer = dispatch_by_m(trees);
    let mut selector = crate::bp::search::selection::MostFractionalPair;
    let mut tel = Telemetry::default();

    let mut stack: Vec<Branchings> = vec![Branchings::default()];
    while let Some(b) = stack.pop() {
        if b.is_inconsistent() {
            continue;
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
        );
        match outcome {
            NodeOutcome::Pruned => {}
            NodeOutcome::Integral(inc) => {
                let updated = state.update_incumbent(inc);
                if updated {
                    tel.incumbent_updates += 1;
                    info!(
                        target: LOG_TARGET,
                        "incumbent: k={} (depth={}, nodes={})",
                        state.best_ub(), b.depth(), tel.nodes_explored,
                    );
                }
            }
            NodeOutcome::Branch {
                lp_obj,
                pair,
            } => {
                let _ = lp_obj;
                let (left, right) = b.split_on(pair);
                stack.push(right);
                stack.push(left);
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
) -> NodeOutcome {
    tel.nodes_explored += 1;

    let t0 = Instant::now();
    rmp.apply_bounds(state.columns(), branchings);
    tel.timings.bounds_apply += t0.elapsed();

    // Column generation.
    let lp = loop {
        let t0 = Instant::now();
        let lp = match rmp.solve() {
            Ok(lp) => lp,
            Err(e) => {
                trace!(target: LOG_TARGET, "node pruned: LP {e}");
                return NodeOutcome::Pruned;
            }
        };
        tel.timings.lp_solve += t0.elapsed();
        tel.cg_iters += 1;

        // Cut separation: materialise any violated node ≤1 constraints.
        // Must happen *before* pricing so the duals we feed the pricer
        // reflect the tightened LP — otherwise β≡0 on unmaterialised rows
        // makes the pricer overweight columns covering many internals.
        let t0 = Instant::now();
        let new_cuts = rmp.separate_and_add_cuts(state.columns(), &lp.column_values, 1.0e-6);
        tel.timings.cut_separation += t0.elapsed();
        if new_cuts > 0 {
            tel.cuts_added += new_cuts;
            trace!(
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
            },
            scratch,
        );
        tel.timings.pricing += t0.elapsed();
        let pt = t0.elapsed();

        match result {
            PricingResult::Found(cols) => {
                let ncols = cols.len();
                let nseen = cols.iter().filter(|c| state.seen().contains(c.labels())).count();
                let mut added = 0;
                for c in cols {
                    if let Some(_id) = state.add_column(c) {
                        rmp.add_column(state.columns().last().unwrap());
                        added += 1;
                    }
                }
                tel.columns_added += added;
                eprintln!(
                    "[bp-cg] iter={} lp={:.4} lp_ms={:.1} pricer_ms={:.1} cols_found={} added={} seen={} total_cols={}",
                    tel.cg_iters, lp.objective, tel.timings.lp_solve.as_secs_f64()*1000.0,
                    pt.as_secs_f64()*1000.0, ncols, added, nseen, state.columns().len(),
                );
                continue;
            }
            PricingResult::Exhausted => {
                eprintln!("[bp-cg] iter={} lp={:.4} EXHAUSTED pricer_ms={:.1}", tel.cg_iters, lp.objective, pt.as_secs_f64()*1000.0);
                break lp;
            }
            PricingResult::Converged => {
                eprintln!("[bp-cg] iter={} lp={:.4} CONVERGED pricer_ms={:.1}", tel.cg_iters, lp.objective, pt.as_secs_f64()*1000.0);
                tel.had_converged = true;
                break lp;
            }
        }
    };

    let lb = (lp.objective - 1e-6).ceil() as usize;

    if can_prune_by_bound(lb, state.best_ub()) {
        debug!(
            target: LOG_TARGET,
            "node pruned by bound: lp={:.4} ub={}",
            lp.objective, state.best_ub(),
        );
        tel.bound_prunes += 1;
        return NodeOutcome::Pruned;
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
    if let Some(inc) = try_round_primal(state, &lp.column_values) {
        if inc.k < state.best_ub() {
            let updated = state.update_incumbent(inc);
            if updated {
                tel.incumbent_updates += 1;
                trace!(
                    target: LOG_TARGET,
                    "primal heuristic improved incumbent: ub={} (cg_iter={})",
                    state.best_ub(), tel.cg_iters,
                );

                if can_prune_by_bound(lb, state.best_ub()) {
                    tel.bound_prunes += 1;
                    return NodeOutcome::Pruned;
                }
            }
        }
    }

    let lp_frac = lp.objective.ceil() - lp.objective;
    if std::env::var("KLADOS_BP_MIP_HEURISTIC").map_or(false, |v| v != "0")
        && lb < state.best_ub()
        && (lp_frac < 1e-4 || branchings.depth() == 0)
    {
        trace!(target: LOG_TARGET, "Running MIP heuristic on pool of {} columns (lp_obj={:.4})", state.columns().len(), lp.objective);
        let mut mip_attempts = 0;
        while mip_attempts < 5 {
            mip_attempts += 1;
            if let Ok(Some(mip_sol)) = rmp.solve_mip() {
                trace!(target: LOG_TARGET, "MIP solve {}: obj={:.4}", mip_attempts, mip_sol.objective);
                let new_cuts =
                    rmp.separate_and_add_cuts(state.columns(), &mip_sol.column_values, 0.5);
                if new_cuts > 0 {
                    tel.cuts_added += new_cuts;
                    trace!(target: LOG_TARGET, "MIP solution violated {} cuts, looping", new_cuts);
                    continue; // Re-solve MIP with new cuts
                }

                if let Some(inc) = try_integral(state, &mip_sol.column_values) {
                    trace!(target: LOG_TARGET, "try_integral found valid incumbent k={}", inc.k);
                    if inc.k < state.best_ub() {
                        let updated = state.update_incumbent(inc);
                        if updated {
                            tel.incumbent_updates += 1;
                            debug!(
                                target: LOG_TARGET,
                                "MIP heuristic improved incumbent: ub={} (cg_iter={}, depth={})",
                                state.best_ub(), tel.cg_iters, branchings.depth(),
                            );

                            if can_prune_by_bound(lb, state.best_ub()) {
                                tel.bound_prunes += 1;
                                return NodeOutcome::Pruned;
                            }
                        }
                    }
                } else {
                    trace!(target: LOG_TARGET, "try_integral returned None for MIP solution");
                }
            } else {
                trace!(target: LOG_TARGET, "rmp.solve_mip() failed or returned None");
            }
            break;
        }
    }
    let t0 = Instant::now();
    let pair = selector.select(
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
    match pair {
        Some(pair) => NodeOutcome::Branch {
            lp_obj: lp.objective,
            pair,
        },
        None => {
            debug!(target: LOG_TARGET, "selector returned None, but not integral. Pruning fractional solution!");
            NodeOutcome::Pruned
        }
    }
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
            trace!(target: LOG_TARGET, "try_integral failed: variable {} is fractional ({})", ci, v);
            return None;
        }
        let col = &state.columns()[ci];

        for (ti, nodes) in col.coverage().iter_per_tree().enumerate() {
            for &n in nodes {
                if !covered_nodes[ti].insert(n) {
                    trace!(target: LOG_TARGET, "try_integral failed: node overlap at tree {}, node {}", ti, n);
                    return None; // Node constraint violated
                }
            }
        }

        for &l in col.labels() {
            cover[l as usize] += 1;
            if cover[l as usize] > 1 {
                trace!(target: LOG_TARGET, "try_integral failed: leaf {} covered multiple times", l);
                return None;
            }
        }
        chosen.push(ci);
    }
    if (1..=n).any(|l| cover[l] == 0) {
        trace!(target: LOG_TARGET, "try_integral failed: some leaves are not covered");
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
