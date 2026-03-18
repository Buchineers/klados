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

    if false && non_iso_comps.len() >= 2 {
        let result = super::decomposition::solve_decomposed(
            &state.forests,
            target_s,
            &state.collapses,
            label_space,
            num_leaves,
            &non_iso_comps,
            &all_comps,
            stats,
        );
        if result.is_none() {
            tt_insert(tt, tt_hash, target_s);
        }
        state.rollback();
        return result;
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
            super::trace!("exact_pairwise_lb tightened: {} → {}", bounds.lower, exact_lb);
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
