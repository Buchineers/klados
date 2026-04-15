//! Main MAF algorithm: iterative deepening with transposition table.

use fixedbitset::FixedBitSet;
use fxhash::FxHashMap;
use klados_core::{Instance, SolverStats, Tree, XForest};

use super::branching::{apply_case_2_branching, br_lsi_step, find_best_sibling_pair};
use super::extraction::{classify_components_cached, extract_maf_components};
use super::reduction::{
    all_pairs_lsi_cached, apply_reduction_rules_state, find_common_sibling_pair,
};
use super::search_state::SearchState;
use super::split::{SplitStats, apply_split_branching_cached};
use super::transposition::{TTEntry, ZobristTable, tt_insert};
use super::utils::trivial_forest;
use crate::lower_bound::maf_bounds;

thread_local! {
    static SPLIT_STATS: std::cell::RefCell<SplitStats> = std::cell::RefCell::new(SplitStats::default());
}

pub fn max_order_from_cached(comp_sets: &[Vec<FixedBitSet>]) -> usize {
    comp_sets.iter().map(|cs| cs.len()).max().unwrap_or(1)
}

pub fn alg_maf(
    state: &mut SearchState,
    target_s: usize,
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    zobrist: &ZobristTable,
    tt: &mut FxHashMap<u64, TTEntry>,
) -> Option<Vec<Tree>> {
    stats.nodes_explored += 1;
    state.checkpoint();

    let comp_sets = loop {
        let comp_sets = apply_reduction_rules_state(state, label_space);

        let cur_order = max_order_from_cached(&comp_sets);
        if cur_order > target_s {
            stats.branches_pruned += 1;
            state.rollback();
            return None;
        }

        if !all_pairs_lsi_cached(&comp_sets) {
            let result = br_lsi_step(
                state,
                target_s,
                label_space,
                num_leaves,
                stats,
                &comp_sets,
                zobrist,
                tt,
            );
            state.rollback();
            return result;
        }

        if let Some((a, b)) = find_common_sibling_pair(&state.forests, label_space) {
            super::trace!("R2: collapsing common sibling-pair ({}, {})", a, b);
            state.add_collapse(a, b);
            continue;
        }

        break comp_sets;
    };

    let comps_f0 = &comp_sets[0];
    let tt_hash = zobrist.hash_partition(comps_f0);
    if let Some(entry) = tt.get(&tt_hash)
        && target_s <= entry.infeasible_at
    {
        stats.branches_pruned += 1;
        state.rollback();
        return None;
    }

    let profile_enabled = super::profile_enabled();
    let split_result = SPLIT_STATS.with(|s| {
        let mut st = s.borrow_mut();
        apply_split_branching_cached(
            state,
            target_s,
            label_space,
            num_leaves,
            stats,
            comps_f0,
            zobrist,
            tt,
            profile_enabled,
            &mut st,
        )
    });
    if split_result.0 {
        if split_result.1.is_none() {
            tt_insert(tt, tt_hash, target_s);
        }
        state.rollback();
        return split_result.1;
    }

    let (all_comps, non_iso_comps) = classify_components_cached(&state.forests, comps_f0);

    if non_iso_comps.is_empty() {
        let result =
            extract_maf_components(&state.forests[0], &state.collapses, label_space, num_leaves);
        if result.len() > target_s {
            super::trace!(
                "BUG: extraction produced {} components but target_s={}, all_comps={}",
                result.len(),
                target_s,
                all_comps.len()
            );
            // Extraction disagrees with search state — reject this solution.
            state.rollback();
            return None;
        }
        state.rollback();
        return Some(result);
    }

    let remaining = target_s.saturating_sub(all_comps.len());
    if remaining == 0 {
        stats.branches_pruned += 1;
        tt_insert(tt, tt_hash, target_s);
        state.rollback();
        return None;
    }

    // Component decomposition (Shi Section 4 / Mestel Section 4.2):
    // After LSI is satisfied and overlaps resolved, non-isomorphic components
    // can be solved independently. Each component's sub-forests are restrictions
    // of the current forests to that component's leaf set.
    // Correctness: Shi Lemma 4.1 guarantees that solving each component in the
    // same label space with its own budget produces a valid MAF.
    if non_iso_comps.len() >= 2 {
        let _iso_count = all_comps.len() - non_iso_comps.len();
        let remaining = target_s.saturating_sub(all_comps.len());

        // Solve each non-iso component by creating restricted sub-forests and
        // recursing into alg_maf directly (no re-kernelization, same label space).
        //
        // Budget allocation: each non-iso component needs at least 1 MAF component.
        // Track minimum costs (initially 1 per component) and update with actual
        // costs as sub-problems are solved, giving tighter budgets to later components.
        let mut min_costs: Vec<usize> = vec![1; non_iso_comps.len()];
        let mut sub_results: Vec<Vec<Tree>> = Vec::new();

        // Early check: if the sum of minimum costs exceeds remaining budget, fail fast.
        let total_min: usize = min_costs.iter().sum();
        if total_min > remaining + non_iso_comps.len() {
            tt_insert(tt, tt_hash, target_s);
            state.rollback();
            return None;
        }

        for (idx, comp_ls) in non_iso_comps.iter().enumerate() {
            let comp_size = comp_ls.count_ones(..);
            if comp_size <= 1 {
                // Singleton -- trivial, costs 0 additional components.
                sub_results.push(Vec::new());
                min_costs[idx] = 1;
                continue;
            }

            // Build restricted sub-forests for this component.
            let sub_forests: Vec<XForest> = state
                .forests
                .iter()
                .map(|f| {
                    let pruned = f.tree.prune_to_leafset(comp_ls);
                    XForest::from_tree(pruned)
                })
                .collect();

            let sub_zobrist = ZobristTable::new(label_space);
            let mut sub_tt: FxHashMap<u64, TTEntry> = FxHashMap::default();

            // Budget for this component: total budget for non-iso components minus
            // minimum cost of all other components.
            // Total non-iso budget = remaining + non_iso_comps.len() (since each
            // component contributes at least 1 to the count, already subtracted).
            let other_min: usize = min_costs
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != idx)
                .map(|(_, &c)| c)
                .sum();
            let non_iso_budget = remaining + non_iso_comps.len();
            let max_budget_for_this = non_iso_budget.saturating_sub(other_min);

            let mut found = false;
            for sub_target in 1..=max_budget_for_this.min(comp_size) {
                let mut sub_state = SearchState::new(sub_forests.clone());
                let mut sub_stats = SolverStats::default();
                if let Some(result) = alg_maf(
                    &mut sub_state,
                    sub_target,
                    label_space,
                    num_leaves,
                    &mut sub_stats,
                    &sub_zobrist,
                    &mut sub_tt,
                ) {
                    // Update min_costs with the actual cost for tighter budgets
                    // on subsequent components.
                    min_costs[idx] = sub_target;
                    sub_results.push(result);
                    found = true;
                    break;
                }
            }
            if !found {
                // This component can't be solved within budget -- decomposition fails.
                tt_insert(tt, tt_hash, target_s);
                state.rollback();
                return None;
            }
        }

        // Assemble: isomorphic components + sub-problem results.
        let mut result_trees: Vec<Tree> = Vec::new();

        // Isomorphic components from the current search state.
        let collapsed_into = super::extraction::build_collapsed_into(&state.collapses, num_leaves);
        for comp_ls in &all_comps {
            let is_non_iso = non_iso_comps
                .iter()
                .any(|c| c.as_slice() == comp_ls.as_slice());
            if is_non_iso {
                continue;
            }
            if comp_ls.count_ones(..) == 0 {
                continue;
            }
            let expanded = super::extraction::expand_leafset(
                comp_ls,
                &collapsed_into,
                num_leaves,
                label_space,
            );
            result_trees.push(super::extraction::build_component_tree(
                &expanded,
                &state.forests[0].tree,
                num_leaves,
            ));
        }

        // Non-iso component results: alg_maf returns trees using the pruned forest's
        // label space. We need to expand collapsed labels just like isomorphic components.
        for sub_result in sub_results {
            for sub_tree in &sub_result {
                // Extract leaf labels from the sub-tree.
                let mut sub_ls = FixedBitSet::with_capacity(label_space + 1);
                for node in sub_tree.pre_order() {
                    if sub_tree.is_leaf(node) {
                        let lbl = sub_tree.label[node as usize];
                        if lbl > 0 {
                            sub_ls.insert(lbl as usize);
                        }
                    }
                }
                // Expand through outer collapses and build from reference tree.
                let expanded = super::extraction::expand_leafset(
                    &sub_ls,
                    &collapsed_into,
                    num_leaves,
                    label_space,
                );
                result_trees.push(super::extraction::build_component_tree(
                    &expanded,
                    &state.forests[0].tree,
                    num_leaves,
                ));
            }
        }

        // Verify total component count doesn't exceed target.
        if result_trees.len() > target_s {
            super::trace!(
                "decomposition produced {} components but target_s={}",
                result_trees.len(),
                target_s
            );
            tt_insert(tt, tt_hash, target_s);
            state.rollback();
            return None;
        }

        state.rollback();
        return Some(result_trees);
    }

    let (a, b) = match find_best_sibling_pair(&state.forests, label_space) {
        Some(pair) => pair,
        None => {
            tt_insert(tt, tt_hash, target_s);
            state.rollback();
            return None;
        }
    };

    super::trace!("MSS pair: a={}, b={}, remaining={}", a, b, remaining);
    let result = apply_case_2_branching(
        state,
        target_s,
        a,
        b,
        label_space,
        num_leaves,
        stats,
        zobrist,
        tt,
    );
    if result.is_none() {
        tt_insert(tt, tt_hash, target_s);
    }
    state.rollback();
    result
}

pub fn solve_inner(instance: &Instance, stats: &mut SolverStats) -> Option<Vec<Tree>> {
    let label_space = instance.num_leaves as usize;
    let forests: Vec<XForest> = instance
        .trees
        .iter()
        .map(|t| XForest::from_tree(t.clone()))
        .collect();

    let mut state = SearchState::new(forests);

    let bounds = maf_bounds(&instance.trees, instance.num_leaves);
    super::trace!("maf_bounds: lower={}, upper={}", bounds.lower, bounds.upper);

    // For multi-tree instances, try to tighten LB via exact pairwise distances + additive formula.
    // Uses the SAT solver for pairwise solves (faster than FPT).
    let start_lb = if instance.trees.len() >= 3 && bounds.upper > bounds.lower {
        let exact_lb = klados_core::lower_bound::exact_pairwise_lower_bound(
            &instance.trees,
            instance.num_leaves,
            bounds.lower,
            bounds.upper,
            std::time::Duration::from_secs(3),
            &mut |pair| crate::maf_sat::solve_pair_sat(pair),
        );
        if exact_lb > bounds.lower {
            super::trace!(
                "exact_pairwise_lb tightened: {} → {}",
                bounds.lower,
                exact_lb
            );
        }
        exact_lb
    } else {
        bounds.lower
    };

    let zobrist = ZobristTable::new(label_space);
    let mut tt: FxHashMap<u64, TTEntry> = FxHashMap::default();

    let solve_start = std::time::Instant::now();

    for target_s in start_lb..=bounds.upper {
        *stats = SolverStats::default();
        let round_start = std::time::Instant::now();

        if let Some(result) = alg_maf(
            &mut state,
            target_s,
            label_space,
            instance.num_leaves,
            stats,
            &zobrist,
            &mut tt,
        ) {
            let total_ms = solve_start.elapsed().as_millis();
            let round_ms = round_start.elapsed().as_millis();
            super::trace!(
                "solution found: target_s={}, components={}, round={}ms, total={}ms, tt_size={}, nodes={}",
                target_s,
                result.len(),
                round_ms,
                total_ms,
                tt.len(),
                stats.nodes_explored,
            );
            dump_split_stats();
            return Some(result);
        }

        let round_ms = round_start.elapsed().as_millis();
        super::trace!(
            "target_s={} failed: {}ms, nodes={}, pruned={}, tt_size={}",
            target_s,
            round_ms,
            stats.nodes_explored,
            stats.branches_pruned,
            tt.len(),
        );
    }

    dump_split_stats();
    Some(trivial_forest(&instance.trees[0], instance.num_leaves))
}

pub fn solve_with_stats(instance: &Instance, stats: &mut SolverStats) -> Option<Vec<Tree>> {
    solve_inner(instance, stats)
}

fn dump_split_stats() {
    if !super::profile_enabled() {
        return;
    }
    let line = SPLIT_STATS.with(|s| {
        let st = s.borrow();
        format!(
            "SPLIT stats: attempts={}, triggered={}, trees_scanned={}, overlap_checks={}, core_calls={}, core_branches={}, nanos={}",
            st.attempts, st.triggered, st.trees_scanned, st.overlap_checks, st.core_calls, st.core_branches, st.split_nanos
        )
    });
    eprintln!("{line}");
    if let Ok(path) = std::env::var("SHI_MESTEL_PROFILE_PATH")
        && let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
    {
        use std::io::Write;
        let _ = writeln!(f, "{line}");
    }
}
