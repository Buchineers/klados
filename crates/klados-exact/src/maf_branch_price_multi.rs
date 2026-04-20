//! Experimental multi-tree Branch & Price solver.
//!
//! Current shape:
//! - generic master problem over an arbitrary number of rooted trees
//! - exact multi-tree pricing via memoized top-down `M`/`V` recurrences
//! - exhaustive subset oracle available as an optional validation check on tiny instances
//! - generic branch-and-bound on fractional columns

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::time::Instant;

use fixedbitset::FixedBitSet;
use fxhash::{FxHashMap, FxHashSet};
use highs::{ColProblem, HighsModelStatus, Model, Row};
use klados_core::lower_bound::{
    greedy_multi_tree_partition, greedy_multi_tree_ub_seeded, pairwise_refine_ub,
};
use klados_core::{Instance, SolverStats, Tree};

use crate::whidden::approx_3_for_instance;

/// Subset of `MafBounds` that's actually used by this solver.
/// We replace the expensive red_blue_approx_detailed pairwise loop with approx_3
/// (per-pair 3-approximation of rSPR distance) for the LB, since:
///   (a) approx_3 is ~100x cheaper than red_blue_approx_detailed
///   (b) B&P pruning actually uses the LP bound, not the initial LB
///   (c) `MafBounds.lower` is not read on the hot path here — only the partition is.
struct LocalBounds {
    best_partition: Option<Vec<usize>>,
}

fn compute_local_bounds(trees: &[Tree], num_leaves: u32) -> LocalBounds {
    if trees.len() <= 1 {
        return LocalBounds { best_partition: None };
    }

    let m = trees.len();
    let n = num_leaves as usize;
    let mut best_lb = 1usize;

    // --- Pairwise LB via approx_3 (3-approx on rSPR distance).
    // approx_3 ≤ 3 * d(Ti,Tj)  ⇒  d(Ti,Tj) ≥ ⌈approx_3 / 3⌉  ⇒  components ≥ ⌈approx_3/3⌉ + 1.
    // approx_3 is ~3000x faster than red_blue_approx_detailed on n=230, so we always run it.
    let t_pair = Instant::now();
    for i in 0..m {
        for j in (i + 1)..m {
            let approx3 = approx_3_for_instance(&trees[i], &trees[j], num_leaves);
            let pair_lb_cuts = ((approx3 + 2) / 3) as usize; // ceil div
            best_lb = best_lb.max(pair_lb_cuts + 1);
        }
    }
    let pair_ms = t_pair.elapsed().as_secs_f64() * 1000.0;
    if m >= 5 || std::env::var("KLADOS_BP_MULTI_PROFILE").ok().as_deref() == Some("1") {
        eprintln!(
            "[bounds] pairwise m={}: approx_3={:.1}ms for {} pairs, best_lb={}",
            m, pair_ms, m * (m - 1) / 2, best_lb,
        );
    }

    // --- Warm-start partition.
    let best_partition = if m == 2 {
        None
    } else {
        let t_multi = Instant::now();
        let mut best_multi_ub = n;
        let mut best_ref = 0usize;
        let mut best_seed = 0u64;
        for ref_idx in 0..m {
            for seed in 0..=20u64 {
                let ub = greedy_multi_tree_ub_seeded(trees, ref_idx, seed);
                if ub < best_multi_ub {
                    best_multi_ub = ub;
                    best_ref = ref_idx;
                    best_seed = seed;
                }
            }
        }
        let multi_ms = t_multi.elapsed().as_secs_f64() * 1000.0;

        let t_pr = Instant::now();
        let (pr_ub, pr_partition) = pairwise_refine_ub(trees, n);
        eprintln!(
            "[bounds] UB m={}: greedy_multi={} ({:.1}ms, {}x21), pairwise_refine={} ({:.1}ms)",
            m, best_multi_ub, multi_ms, m, pr_ub, t_pr.elapsed().as_secs_f64() * 1000.0,
        );

        if pr_ub < best_multi_ub {
            Some(pr_partition)
        } else {
            let (_, partition) = greedy_multi_tree_partition(trees, best_ref, best_seed);
            Some(partition)
        }
    };

    LocalBounds { best_partition }
}

use crate::cluster_reduction::{self, ClusterReductionResult};
use crate::kernelize::{self, KernelizeConfig};
use crate::ExactSolver;

const ORACLE_ENUM_LEAVES: usize = 24;

#[derive(Default, Clone, Copy)]
struct PricerTiming {
    setup_ms: f64,
    collect_ms: f64,
    filter_ms: f64,
    raw_candidates: usize,
    filtered_candidates: usize,
}

thread_local! {
    static PRICER_TIMING: std::cell::Cell<PricerTiming> = const {
        std::cell::Cell::new(PricerTiming {
            setup_ms: 0.0,
            collect_ms: 0.0,
            filter_ms: 0.0,
            raw_candidates: 0,
            filtered_candidates: 0,
        })
    };
}

fn pricer_timing_reset() -> PricerTiming {
    PRICER_TIMING.with(|c| c.replace(PricerTiming::default()))
}

fn pricer_timing_add_setup(ms: f64) {
    PRICER_TIMING.with(|c| {
        let mut t = c.get();
        t.setup_ms += ms;
        c.set(t);
    });
}

fn pricer_timing_add_collect(ms: f64, raw: usize) {
    PRICER_TIMING.with(|c| {
        let mut t = c.get();
        t.collect_ms += ms;
        t.raw_candidates += raw;
        c.set(t);
    });
}

fn pricer_timing_add_filter(ms: f64, kept: usize) {
    PRICER_TIMING.with(|c| {
        let mut t = c.get();
        t.filter_ms += ms;
        t.filtered_candidates += kept;
        c.set(t);
    });
}
const NEG_INF: f64 = -1.0e100;

fn exact_pricer_trace() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("KLADOS_BP_MULTI_TRACE").ok().as_deref() == Some("1"))
}

fn exact_pricer_profile() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("KLADOS_BP_MULTI_PROFILE").ok().as_deref() == Some("1"))
}

fn pairdp_batch_size() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("KLADOS_BP_MULTI_PAIRDP_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v.max(1))
            .unwrap_or(64)
    })
}

fn exhaustive_oracle_validation_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        matches!(
            std::env::var("KLADOS_BP_MULTI_VALIDATE_PRICER").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
        )
    })
}

fn dual_stabilization_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("KLADOS_BP_MULTI_STABILIZE").ok().as_deref() == Some("1"))
}

fn dual_stabilization_weight() -> f64 {
    use std::sync::OnceLock;
    static CACHED: OnceLock<f64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("KLADOS_BP_MULTI_STABILIZE_WEIGHT")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(0.5)
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PricingBackend {
    Native,
    PairDp,
}

fn pricing_backend() -> PricingBackend {
    use std::sync::OnceLock;
    static CACHED: OnceLock<PricingBackend> = OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("KLADOS_BP_MULTI_PRICER").ok().as_deref() {
            Some("native") => PricingBackend::Native,
            Some("pairdp")
            | Some("ilp")
            | Some("ilp-global")
            | Some("ilp-hybrid") => PricingBackend::PairDp,
            _ => PricingBackend::PairDp,
        }
    })
}

pub struct MafBranchPriceMultiSolver {
    stats: SolverStats,
}

impl Default for MafBranchPriceMultiSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl MafBranchPriceMultiSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
        }
    }
}

impl ExactSolver for MafBranchPriceMultiSolver {
    fn name(&self) -> &'static str {
        "maf-bp-multi"
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        if instance.trees.is_empty() {
            return None;
        }
        if instance.num_trees() == 1 {
            return Some(instance.trees.clone());
        }
        if instance.num_leaves <= 1 {
            return Some(instance.trees[0..1].to_vec());
        }
        solve_branch_price_multi(instance, &mut self.stats)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

struct BpColumn {
    labels: Vec<u32>,
    covered_internal_nodes: Vec<Vec<usize>>,
    total_internal_count: usize,
}

struct BpState {
    columns: Vec<BpColumn>,
    seen: FxHashSet<Vec<u32>>,
    best_ub: usize,
    best_solution: Option<Vec<f64>>,
    nodes_explored: usize,
    cg_iterations_total: usize,
    oracle_matches: usize,
    oracle_mismatches: usize,
    pricing_time_ms: f64,
    lp_solve_time_ms: f64,
    pricing_calls: usize,
    lp_solve_calls: usize,
    columns_added: usize,
    node_time_ms: f64,
    apply_bounds_time_ms: f64,
    add_column_time_ms: f64,
    branch_select_time_ms: f64,
    pricer_setup_ms: f64,
    pricer_collect_ms: f64,
    pricer_filter_ms: f64,
    pricer_candidates_raw: usize,
    pricer_candidates_after_filter: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct LeafPair {
    a: u32,
    b: u32,
}

impl LeafPair {
    fn new(lhs: u32, rhs: u32) -> Self {
        if lhs <= rhs {
            Self { a: lhs, b: rhs }
        } else {
            Self { a: rhs, b: lhs }
        }
    }
}

#[derive(Clone)]
struct BpNode {
    fixed_to_one: Vec<usize>,
    fixed_to_zero: Vec<usize>,
    must_link_pairs: Vec<LeafPair>,
    cannot_link_pairs: Vec<LeafPair>,
    depth: usize,
}

struct ForbiddenColumns {
    fixed_zero_labels: Vec<Vec<u32>>,
}

impl ForbiddenColumns {
    fn new(state: &BpState, node: &BpNode) -> Self {
        let fixed_zero_labels = node
            .fixed_to_zero
            .iter()
            .map(|&ci| state.columns[ci].labels.clone())
            .collect();
        Self { fixed_zero_labels }
    }

    fn contains(&self, seen: &FxHashSet<Vec<u32>>, labels: &[u32]) -> bool {
        seen.contains(labels)
            || self
                .fixed_zero_labels
                .iter()
                .any(|blocked| blocked.as_slice() == labels)
    }
}

enum NodeResult {
    Integral(usize, Vec<f64>),
    BranchPair { lp_obj: f64, pair: LeafPair },
    BranchColumn { lp_obj: f64, branch_col: usize },
    Pruned,
}

fn solve_branch_price_multi(instance: &Instance, stats: &mut SolverStats) -> Option<Vec<Tree>> {
    let t_total = Instant::now();

    let t_kern = Instant::now();
    let config = KernelizeConfig::default();
    let kern = kernelize::kernelize_best(instance, &config);
    let reduced = &kern.instance;
    let kern_ms = t_kern.elapsed().as_secs_f64() * 1000.0;

    eprintln!(
        "[maf-bp-multi] kernelized {} -> {} leaves (m={})",
        instance.num_leaves,
        reduced.num_leaves,
        reduced.num_trees(),
    );

    if reduced.num_leaves <= 1 {
        let trivial = if reduced.num_leaves == 0 {
            vec![]
        } else {
            vec![reduced.trees[0].clone()]
        };
        let components = kernelize::expand_solution(
            trivial,
            &kern,
            instance.reference_tree(),
            instance.num_leaves,
        );
        stats.upper_bound = Some(components.len());
        stats.lower_bound = components.len();
        return Some(components);
    }

    let n = reduced.num_leaves as usize;
    let trees = &reduced.trees;
    let param_reduction_32 = kern.param_reduction;

    let t_cluster = Instant::now();
    let cluster_result = cluster_reduction::try_cluster_reduction(reduced, &mut |subinstance| {
        solve_branch_price_multi(subinstance, &mut SolverStats::default())
    })?;
    let cluster_ms = t_cluster.elapsed().as_secs_f64() * 1000.0;
    match cluster_result {
        ClusterReductionResult::NotApplicable => {}
        ClusterReductionResult::Solved(solution) => {
            let exact_k = solution.components.len() + param_reduction_32;
            stats.lower_bound = exact_k;
            stats.upper_bound = Some(exact_k);
            let components = kernelize::expand_solution(
                solution.components,
                &kern,
                instance.reference_tree(),
                instance.num_leaves,
            );
            eprintln!(
                "[maf-bp-multi] optimal: {} components (cluster decomp), {:.1}ms total",
                components.len(),
                t_total.elapsed().as_secs_f64() * 1000.0,
            );
            return Some(components);
        }
    }

    let t_bounds = Instant::now();
    let bounds = compute_local_bounds(trees, reduced.num_leaves);
    let bounds_ms = t_bounds.elapsed().as_secs_f64() * 1000.0;
    let mut columns: Vec<BpColumn> = (1..=n as u32)
        .map(|label| make_bp_column(vec![label], trees))
        .collect();
    let mut seen: FxHashSet<Vec<u32>> = columns.iter().map(|c| c.labels.clone()).collect();

    let mut best_solution = {
        let mut values = vec![0.0; columns.len()];
        for (ci, col) in columns.iter().enumerate() {
            if col.labels.len() == 1 {
                values[ci] = 1.0;
            }
        }
        Some(values)
    };
    let mut best_ub = n;
    if let Some(partition) = &bounds.best_partition {
        let mut comp_labels: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for (leaf_idx, &comp_id) in partition.iter().enumerate() {
            comp_labels.entry(comp_id).or_default().push((leaf_idx + 1) as u32);
        }
        let mut values = vec![0.0; columns.len()];
        for labels in comp_labels.values() {
            if let Some((ci, _)) = columns.iter().enumerate().find(|(_, col)| col.labels == *labels) {
                values[ci] = 1.0;
            } else {
                columns.push(make_bp_column(labels.clone(), trees));
                values.push(1.0);
                seen.insert(labels.clone());
            }
        }
        best_solution = Some(values);
        best_ub = comp_labels.len().min(n);
    }

    let mut state = BpState {
        columns,
        seen,
        best_ub,
        best_solution,
        nodes_explored: 0,
        cg_iterations_total: 0,
        oracle_matches: 0,
        oracle_mismatches: 0,
        pricing_time_ms: 0.0,
        lp_solve_time_ms: 0.0,
        pricing_calls: 0,
        lp_solve_calls: 0,
        columns_added: 0,
        node_time_ms: 0.0,
        apply_bounds_time_ms: 0.0,
        add_column_time_ms: 0.0,
        branch_select_time_ms: 0.0,
        pricer_setup_ms: 0.0,
        pricer_collect_ms: 0.0,
        pricer_filter_ms: 0.0,
        pricer_candidates_raw: 0,
        pricer_candidates_after_filter: 0,
    };

    let root = BpNode {
        fixed_to_one: vec![],
        fixed_to_zero: vec![],
        must_link_pairs: vec![],
        cannot_link_pairs: vec![],
        depth: 0,
    };

    let t_pricer_ws = Instant::now();
    let mut pricer_ws = MultiPricerWorkspace::new(trees, n);
    let pricer_ws_ms = t_pricer_ws.elapsed().as_secs_f64() * 1000.0;
    let t_rmp_new = Instant::now();
    let mut rmp = match PersistentRmp::new(&state.columns, trees, n) {
        Ok(rmp) => rmp,
        Err(err) => {
            eprintln!("[maf-bp-multi] failed to build persistent RMP: {}", err);
            return None;
        }
    };
    let rmp_new_ms = t_rmp_new.elapsed().as_secs_f64() * 1000.0;
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let node_t0 = Instant::now();
        let result = solve_bp_node(&mut state, &node, trees, n, &mut pricer_ws, &mut rmp);
        state.node_time_ms += node_t0.elapsed().as_secs_f64() * 1000.0;
        match result {
            NodeResult::Integral(obj, values) => {
                if obj < state.best_ub {
                    eprintln!(
                        "[maf-bp-multi] new incumbent: {} components (depth={}, nodes={})",
                        obj, node.depth, state.nodes_explored,
                    );
                    state.best_ub = obj;
                    let mut padded = values;
                    padded.resize(state.columns.len(), 0.0);
                    state.best_solution = Some(padded);
                }
            }
            NodeResult::BranchPair { lp_obj, pair } => {
                let lp_lb = (lp_obj - 1e-6).ceil() as usize;
                if lp_lb >= state.best_ub {
                    continue;
                }

                let mut right = node.clone();
                if !right.cannot_link_pairs.contains(&pair) {
                    right.cannot_link_pairs.push(pair);
                }
                right.depth += 1;

                let mut left = node.clone();
                if !left.must_link_pairs.contains(&pair) {
                    left.must_link_pairs.push(pair);
                }
                left.depth += 1;

                stack.push(right);
                stack.push(left);
            }
            NodeResult::BranchColumn { lp_obj, branch_col } => {
                let lp_lb = (lp_obj - 1e-6).ceil() as usize;
                if lp_lb >= state.best_ub {
                    continue;
                }
                let mut right = node.clone();
                right.fixed_to_zero.push(branch_col);
                right.depth += 1;

                let mut left = node.clone();
                left.fixed_to_one.push(branch_col);
                left.depth += 1;

                stack.push(right);
                stack.push(left);
            }
            NodeResult::Pruned => {}
        }
    }

    let values = state.best_solution.as_ref()?;
    let reduced_components = reconstruct_components(&state.columns, values, reduced);
    let components = kernelize::expand_solution(
        reduced_components,
        &kern,
        instance.reference_tree(),
        instance.num_leaves,
    );
    stats.upper_bound = Some(components.len());
    stats.lower_bound = components.len();
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    let node_in_ms = state.pricing_time_ms
        + state.lp_solve_time_ms
        + state.apply_bounds_time_ms
        + state.add_column_time_ms
        + state.branch_select_time_ms;
    let node_other_ms = (state.node_time_ms - node_in_ms).max(0.0);
    let outside_nodes_ms = (total_ms - state.node_time_ms).max(0.0);
    eprintln!(
        "[maf-bp-multi] optimal: {} components, {} B&B nodes, {} CG iters, {} cols, oracle(m/mm)={}/{}, {:.1}ms total",
        components.len(),
        state.nodes_explored,
        state.cg_iterations_total,
        state.columns_added,
        state.oracle_matches,
        state.oracle_mismatches,
        total_ms,
    );
    eprintln!(
        "[maf-bp-multi] setup: kern={:.1}ms, cluster={:.1}ms, bounds={:.1}ms, pricer_ws={:.1}ms, rmp_new={:.1}ms",
        kern_ms, cluster_ms, bounds_ms, pricer_ws_ms, rmp_new_ms,
    );
    eprintln!(
        "[maf-bp-multi] pricer: setup={:.1}ms, collect={:.1}ms, filter={:.1}ms, raw_cands={}, kept={}",
        state.pricer_setup_ms,
        state.pricer_collect_ms,
        state.pricer_filter_ms,
        state.pricer_candidates_raw,
        state.pricer_candidates_after_filter,
    );
    eprintln!(
        "[maf-bp-multi] timing: node_total={:.1}ms (pricing={:.1}/{}, lp={:.1}/{}, apply_bounds={:.1}, add_col={:.1}, branch_sel={:.1}, node_other={:.1}), outside_nodes={:.1}",
        state.node_time_ms,
        state.pricing_time_ms,
        state.pricing_calls,
        state.lp_solve_time_ms,
        state.lp_solve_calls,
        state.apply_bounds_time_ms,
        state.add_column_time_ms,
        state.branch_select_time_ms,
        node_other_ms,
        outside_nodes_ms,
    );
    Some(components)
}

fn price_columns_for_duals(
    pricer_ws: &mut MultiPricerWorkspace,
    trees: &[Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    seen: &FxHashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
    must_link_pairs: &[LeafPair],
    cannot_link_pairs: &[LeafPair],
) -> Vec<(f64, Vec<u32>)> {
    match pricing_backend() {
        PricingBackend::Native => price_best_new_exact_column(
            pricer_ws,
            trees,
            num_leaves,
            alpha,
            beta,
            blocked_leaves,
            seen,
            forbidden,
            must_link_pairs,
            cannot_link_pairs,
        )
        .into_iter()
        .collect::<Vec<_>>(),
        PricingBackend::PairDp => price_best_new_pairdp_columns(
            pricer_ws,
            trees,
            num_leaves,
            alpha,
            beta,
            blocked_leaves,
            seen,
            forbidden,
            must_link_pairs,
            cannot_link_pairs,
        ),
    }
}

fn blend_duals(current: &[f64], center: &[f64], weight: f64) -> Vec<f64> {
    current
        .iter()
        .zip(center.iter())
        .map(|(&cur, &ctr)| weight * cur + (1.0 - weight) * ctr)
        .collect()
}

fn blend_dual_rows(current: &[Vec<f64>], center: &[Vec<f64>], weight: f64) -> Vec<Vec<f64>> {
    current
        .iter()
        .zip(center.iter())
        .map(|(cur_row, ctr_row)| blend_duals(cur_row, ctr_row, weight))
        .collect()
}

fn rescore_profitable_columns(
    candidates: Vec<(f64, Vec<u32>)>,
    trees: &[Tree],
    alpha: &[f64],
    beta: &[Vec<f64>],
) -> Vec<(f64, Vec<u32>)> {
    let mut dedup: FxHashMap<Vec<u32>, f64> = FxHashMap::default();
    for (_, labels) in candidates {
        let score = column_score(&labels, trees, alpha, beta);
        if score <= 1.0 + 1.0e-8 {
            continue;
        }
        dedup
            .entry(labels)
            .and_modify(|best| *best = best.max(score))
            .or_insert(score);
    }
    let mut out = dedup
        .into_iter()
        .map(|(labels, score)| (score, labels))
        .collect::<Vec<_>>();
    out.sort_unstable_by(|lhs, rhs| {
        rhs.0
            .total_cmp(&lhs.0)
            .then_with(|| rhs.1.len().cmp(&lhs.1.len()))
            .then_with(|| lhs.1.cmp(&rhs.1))
    });
    out
}

fn update_stability_center(
    center: &mut Option<(Vec<f64>, Vec<Vec<f64>>)>,
    alpha: &[f64],
    beta: &[Vec<f64>],
    weight: f64,
) {
    match center {
        Some((center_alpha, center_beta)) => {
            *center_alpha = blend_duals(alpha, center_alpha, weight);
            *center_beta = blend_dual_rows(beta, center_beta, weight);
        }
        None => *center = Some((alpha.to_vec(), beta.to_vec())),
    }
}

fn solve_bp_node(
    state: &mut BpState,
    node: &BpNode,
    trees: &[Tree],
    num_leaves: usize,
    pricer_ws: &mut MultiPricerWorkspace,
    rmp: &mut PersistentRmp,
) -> NodeResult {
    state.nodes_explored += 1;
    if node.fixed_to_one.len() >= state.best_ub {
        return NodeResult::Pruned;
    }
    if !node_branchings_self_consistent(&state.columns, node) {
        return NodeResult::Pruned;
    }

    let mut blocked_leaves = vec![false; num_leaves + 1];
    for &forced_ci in &node.fixed_to_one {
        for &label in &state.columns[forced_ci].labels {
            blocked_leaves[label as usize] = true;
        }
    }

    let forbidden = ForbiddenColumns::new(state, node);

    let apply_t0 = Instant::now();
    rmp.apply_node_bounds(
        &state.columns,
        &node.fixed_to_one,
        &node.fixed_to_zero,
        &node.must_link_pairs,
        &node.cannot_link_pairs,
        &blocked_leaves,
    );
    state.apply_bounds_time_ms += apply_t0.elapsed().as_secs_f64() * 1000.0;

    let mut stability_center: Option<(Vec<f64>, Vec<Vec<f64>>)> = None;
    let lp = loop {
        let lp_t0 = Instant::now();
        let lp = match rmp.solve(state.columns.len()) {
            Ok(lp) => lp,
            Err(_) => {
                state.lp_solve_time_ms += lp_t0.elapsed().as_secs_f64() * 1000.0;
                state.lp_solve_calls += 1;
                return NodeResult::Pruned;
            }
        };
        state.lp_solve_time_ms += lp_t0.elapsed().as_secs_f64() * 1000.0;
        state.lp_solve_calls += 1;
        let alpha = lp.leaf_duals.clone();
        let beta = lp.node_duals.clone();
        let pricing_t0 = Instant::now();
        let priced_columns = if dual_stabilization_enabled() {
            let weight = dual_stabilization_weight();
            let (price_alpha, price_beta) = if let Some((center_alpha, center_beta)) = &stability_center {
                (
                    blend_duals(&alpha, center_alpha, weight),
                    blend_dual_rows(&beta, center_beta, weight),
                )
            } else {
                (alpha.clone(), beta.clone())
            };

            let mut stabilized_columns = price_columns_for_duals(
                pricer_ws,
                trees,
                num_leaves,
                &price_alpha,
                &price_beta,
                &blocked_leaves,
                &state.seen,
                &forbidden,
                &node.must_link_pairs,
                &node.cannot_link_pairs,
            );
            stabilized_columns = rescore_profitable_columns(stabilized_columns, trees, &alpha, &beta);
            if stabilized_columns.is_empty() {
                price_columns_for_duals(
                    pricer_ws,
                    trees,
                    num_leaves,
                    &alpha,
                    &beta,
                    &blocked_leaves,
                    &state.seen,
                    &forbidden,
                    &node.must_link_pairs,
                    &node.cannot_link_pairs,
                )
            } else {
                stabilized_columns
            }
        } else {
            price_columns_for_duals(
                pricer_ws,
                trees,
                num_leaves,
                &alpha,
                &beta,
                &blocked_leaves,
                &state.seen,
                &forbidden,
                &node.must_link_pairs,
                &node.cannot_link_pairs,
            )
        };
        state.pricing_time_ms += pricing_t0.elapsed().as_secs_f64() * 1000.0;
        state.pricing_calls += 1;
        let pt = pricer_timing_reset();
        state.pricer_setup_ms += pt.setup_ms;
        state.pricer_collect_ms += pt.collect_ms;
        state.pricer_filter_ms += pt.filter_ms;
        state.pricer_candidates_raw += pt.raw_candidates;
        state.pricer_candidates_after_filter += pt.filtered_candidates;
        let priced = priced_columns.first().cloned();
        if exhaustive_oracle_validation_enabled() && num_leaves <= ORACLE_ENUM_LEAVES {
            let oracle_priced = price_best_column_exhaustive(
                trees,
                num_leaves,
                &alpha,
                &beta,
                &blocked_leaves,
                &state.seen,
                &forbidden,
            );
            let priced_profitable =
                priced
                    .clone()
                    .filter(|(score, _)| *score > 1.0 + 1.0e-8);
            let oracle_profitable =
                oracle_priced
                    .clone()
                    .filter(|(score, _)| *score > 1.0 + 1.0e-8);
            match (&priced_profitable, &oracle_profitable) {
                (Some((s1, lhs)), Some((s2, rhs)))
                    if lhs == rhs && (s1 - s2).abs() <= 1e-8 => state.oracle_matches += 1,
                (None, None) => state.oracle_matches += 1,
                _ => {
                    state.oracle_mismatches += 1;
                    if exact_pricer_trace() {
                        let exact_details = priced.as_ref().map(|(score, labels)| {
                            let valid = is_set_compatible_all(trees, labels);
                            let forbidden_hit = forbidden.contains(&state.seen, labels);
                            let real_score =
                                column_score(labels, trees, &alpha, &beta);
                            (*score, real_score, valid, forbidden_hit, labels.clone())
                        });
                        let oracle_details = oracle_priced.as_ref().map(|(score, labels)| {
                            let real_score =
                                column_score(labels, trees, &alpha, &beta);
                            (*score, real_score, labels.clone())
                        });
                        eprintln!(
                            "[maf-bp-multi] pricer mismatch: m={} n={} depth={} exact={:?} oracle={:?}",
                            trees.len(),
                            num_leaves,
                            node.depth,
                            exact_details,
                            oracle_details
                        );
                    }
                }
            }
        }

        if dual_stabilization_enabled() {
            update_stability_center(&mut stability_center, &alpha, &beta, dual_stabilization_weight());
        }

        let mut added_any = false;
        let add_t0 = Instant::now();
        for (score, labels) in priced_columns {
            if score <= 1.0 + 1e-8 {
                continue;
            }
            let inserted = state.seen.insert(labels.clone());
            if !inserted {
                continue;
            }
            let new_ci = state.columns.len();
            state.columns.push(make_bp_column(labels, trees));
            rmp.add_column(new_ci, &state.columns[new_ci], trees);
            if let Some(best_solution) = state.best_solution.as_mut() {
                best_solution.push(0.0);
            }
            added_any = true;
            state.columns_added += 1;
        }
        state.add_column_time_ms += add_t0.elapsed().as_secs_f64() * 1000.0;

        if added_any {
            state.cg_iterations_total += 1;
            continue;
        }

        break lp;
    };
    let lp_bound = (lp.objective - 1e-6).ceil() as usize;
    if lp_bound >= state.best_ub {
        return NodeResult::Pruned;
    }

    if support_is_integral_partition(&state.columns, &lp.column_values, num_leaves) {
        let obj = lp.column_values.iter().filter(|&&v| v > 1.0e-9).count();
        return NodeResult::Integral(obj, lp.column_values);
    }

    let sel_t0 = Instant::now();
    let pair_opt = select_branch_pair(&state.columns, &lp.column_values, num_leaves);
    state.branch_select_time_ms += sel_t0.elapsed().as_secs_f64() * 1000.0;
    if let Some(pair) = pair_opt {
        return NodeResult::BranchPair {
            lp_obj: lp.objective,
            pair,
        };
    }

    let branch_col = select_branch_column(&state.columns, &lp.column_values, num_leaves);
    match branch_col {
        Some(col_idx) => NodeResult::BranchColumn {
            lp_obj: lp.objective,
            branch_col: col_idx,
        },
        None => NodeResult::Pruned,
    }
}

#[derive(Clone)]
struct PricingPrefixNode {
    prefix_membership: Vec<bool>,
    score: f64,
    labels: Vec<u32>,
}

struct MultiPricerWorkspace {
    descendant_leaves: Vec<Vec<FixedBitSet>>,
    label_stride: usize,
    label_lca: Vec<Vec<u32>>,
    label_side_child: Vec<Vec<u32>>,
    label_node: Vec<Vec<u32>>,
    memo_m: FxHashMap<Box<[u32]>, f64>,
    memo_v: FxHashMap<Box<[u32]>, f64>,
}

impl MultiPricerWorkspace {
    fn new(trees: &[Tree], num_leaves: usize) -> Self {
        let mut descendant_leaves = Vec::with_capacity(trees.len());
        for tree in trees {
            let mut leaves = vec![FixedBitSet::with_capacity(num_leaves + 1); tree.num_nodes()];
            for node in tree.post_order_vec() {
                if tree.is_leaf(node) {
                    let lbl = tree.label[node as usize] as usize;
                    leaves[node as usize].insert(lbl);
                } else {
                    let (l, r) = tree.children_pair(node);
                    let mut bits = leaves[l as usize].clone();
                    bits.union_with(&leaves[r as usize]);
                    leaves[node as usize] = bits;
                }
            }
            descendant_leaves.push(leaves);
        }

        let stride = num_leaves + 1;
        let mut label_lca = Vec::with_capacity(trees.len());
        let mut label_side_child = Vec::with_capacity(trees.len());
        let mut label_node = Vec::with_capacity(trees.len());
        for (ti, tree) in trees.iter().enumerate() {
            let mut node_by_label = vec![0u32; stride];
            for la in 1..=num_leaves as u32 {
                node_by_label[la as usize] = tree.node_by_label(la);
            }
            let mut lca_table = vec![0u32; stride * stride];
            let mut side_table = vec![0u32; stride * stride];
            for la in 1..=num_leaves as u32 {
                let node_a = node_by_label[la as usize];
                let base = (la as usize) * stride;
                for lb in 1..=num_leaves as u32 {
                    let idx = base + lb as usize;
                    if la == lb {
                        lca_table[idx] = node_a;
                        side_table[idx] = node_a;
                        continue;
                    }
                    let node_b = node_by_label[lb as usize];
                    let root = tree.nearest_common_ancestor(node_a, node_b);
                    lca_table[idx] = root;
                    let child = if tree.is_leaf(root) {
                        root
                    } else {
                        let (left, right) = tree.children_pair(root);
                        if descendant_leaves[ti][left as usize].contains(la as usize) {
                            left
                        } else {
                            right
                        }
                    };
                    side_table[idx] = child;
                }
            }
            label_lca.push(lca_table);
            label_side_child.push(side_table);
            label_node.push(node_by_label);
        }

        Self {
            descendant_leaves,
            label_stride: stride,
            label_lca,
            label_side_child,
            label_node,
            memo_m: FxHashMap::default(),
            memo_v: FxHashMap::default(),
        }
    }

    fn reset(&mut self) {
        self.memo_m.clear();
        self.memo_v.clear();
    }

    #[inline]
    fn tuple_key(tuple: &[u32]) -> Box<[u32]> {
        tuple.to_vec().into_boxed_slice()
    }

    fn solve_best(
        &mut self,
        trees: &[Tree],
        alpha: &[f64],
        beta: &[Vec<f64>],
        blocked_leaves: &[bool],
    ) -> Option<(f64, Vec<u32>)> {
        self.reset();
        let root_tuple = trees.iter().map(|t| t.root).collect::<Vec<_>>();
        let score = self.solve_m_score(&root_tuple, trees, alpha, beta, blocked_leaves);
        if score <= NEG_INF / 2.0 {
            return None;
        }
        if exact_pricer_profile() {
            eprintln!(
                "[maf-bp-multi] pricer profile: m={} memo_m={} memo_v={}",
                trees.len(),
                self.memo_m.len(),
                self.memo_v.len()
            );
        }
        let mut labels = Vec::new();
        self.collect_m_labels(&root_tuple, trees, alpha, beta, blocked_leaves, &mut labels);
        labels.sort_unstable();
        labels.dedup();
        (!labels.is_empty()).then_some((score, labels))
    }

    fn solve_m_score(
        &mut self,
        tuple: &[u32],
        trees: &[Tree],
        alpha: &[f64],
        beta: &[Vec<f64>],
        blocked_leaves: &[bool],
    ) -> f64 {
        if let Some(best) = self.memo_m.get(tuple) {
            return *best;
        }

        let inter = self.intersection_bits(tuple);
        let (has_leaf, ub, best_leaf) = self.intersection_stats(&inter, alpha, blocked_leaves);
        if !has_leaf {
            self.memo_m.insert(Self::tuple_key(tuple), NEG_INF);
            return NEG_INF;
        }
        let mut best = best_leaf;
        if best >= ub - 1e-12 {
            self.memo_m.insert(Self::tuple_key(tuple), best);
            return best;
        }

        for (ti, &node) in tuple.iter().enumerate() {
            if trees[ti].is_leaf(node) {
                continue;
            }
            let (left, right) = trees[ti].children_pair(node);
            let mut left_tuple = tuple.to_vec();
            left_tuple[ti] = left;
            let left_ub = self.restricted_upper_bound(&inter, ti, left, alpha, blocked_leaves);
            if left_ub > best + 1e-12 {
                let cand_left = self.solve_m_score(&left_tuple, trees, alpha, beta, blocked_leaves);
                if cand_left > best {
                    best = cand_left;
                    if best >= ub - 1e-12 {
                        self.memo_m.insert(Self::tuple_key(tuple), best);
                        return best;
                    }
                }
            }

            let mut right_tuple = tuple.to_vec();
            right_tuple[ti] = right;
            let right_ub = self.restricted_upper_bound(&inter, ti, right, alpha, blocked_leaves);
            if right_ub > best + 1e-12 {
                let cand_right =
                    self.solve_m_score(&right_tuple, trees, alpha, beta, blocked_leaves);
                if cand_right > best {
                    best = cand_right;
                    if best >= ub - 1e-12 {
                        self.memo_m.insert(Self::tuple_key(tuple), best);
                        return best;
                    }
                }
            }
        }

        if ub > best + 1e-12 {
            let rooted = self.solve_v_score(tuple, trees, alpha, beta, blocked_leaves);
            if rooted > best {
                best = rooted;
            }
        }

        self.memo_m.insert(Self::tuple_key(tuple), best);
        best
    }

    fn solve_v_score(
        &mut self,
        tuple: &[u32],
        trees: &[Tree],
        alpha: &[f64],
        beta: &[Vec<f64>],
        blocked_leaves: &[bool],
    ) -> f64 {
        if let Some(best) = self.memo_v.get(tuple) {
            return *best;
        }

        let inter = self.intersection_bits(tuple);
        let (has_leaf, ub, _best_leaf) = self.intersection_stats(&inter, alpha, blocked_leaves);
        if !has_leaf {
            self.memo_v.insert(Self::tuple_key(tuple), NEG_INF);
            return NEG_INF;
        }

        if tuple.iter().enumerate().all(|(ti, &node)| trees[ti].is_leaf(node)) {
            let first_label = trees[0].label[tuple[0] as usize];
            let all_same = tuple.iter().enumerate().all(|(ti, &node)| {
                trees[ti].label[node as usize] == first_label
            });
            let best = if all_same && !blocked_leaves[first_label as usize] {
                alpha[first_label as usize]
            } else {
                NEG_INF
            };
            self.memo_v.insert(Self::tuple_key(tuple), best);
            return best;
        }

        let mut best = NEG_INF;
        if ub <= 0.0 {
            self.memo_v.insert(Self::tuple_key(tuple), best);
            return best;
        }

        for (ti, &node) in tuple.iter().enumerate() {
            if trees[ti].is_leaf(node) {
                continue;
            }
            let skip_penalty = beta[ti][node as usize];
            let (left, right) = trees[ti].children_pair(node);
            let mut left_tuple = tuple.to_vec();
            left_tuple[ti] = left;
            let left_ub = self.restricted_upper_bound(&inter, ti, left, alpha, blocked_leaves);
            if left_ub - skip_penalty > best + 1e-12 {
                let cand_left = self.solve_v_score(&left_tuple, trees, alpha, beta, blocked_leaves);
                if cand_left > NEG_INF / 2.0 {
                    best = best.max(cand_left - skip_penalty);
                    if best >= ub - 1e-12 {
                        self.memo_v.insert(Self::tuple_key(tuple), best);
                        return best;
                    }
                }
            }

            let mut right_tuple = tuple.to_vec();
            right_tuple[ti] = right;
            let right_ub = self.restricted_upper_bound(&inter, ti, right, alpha, blocked_leaves);
            if right_ub - skip_penalty > best + 1e-12 {
                let cand_right =
                    self.solve_v_score(&right_tuple, trees, alpha, beta, blocked_leaves);
                if cand_right > NEG_INF / 2.0 {
                    best = best.max(cand_right - skip_penalty);
                    if best >= ub - 1e-12 {
                        self.memo_v.insert(Self::tuple_key(tuple), best);
                        return best;
                    }
                }
            }
        }

        if tuple.iter().enumerate().all(|(ti, &node)| !trees[ti].is_leaf(node)) {
            let beta_sum = tuple
                .iter()
                .enumerate()
                .map(|(ti, &node)| beta[ti][node as usize])
                .sum::<f64>();
            let orient_count = 1usize << (tuple.len().saturating_sub(1));
            for orient in 0..orient_count {
                let (left_tuple, right_tuple) = oriented_children_tuple(tuple, trees, orient);
                let left_ub =
                    self.oriented_side_upper_bound(tuple, trees, orient, true, alpha, blocked_leaves);
                if left_ub <= NEG_INF / 2.0 {
                    continue;
                }
                let right_ub =
                    self.oriented_side_upper_bound(tuple, trees, orient, false, alpha, blocked_leaves);
                if right_ub <= NEG_INF / 2.0 || left_ub + right_ub - beta_sum <= best + 1e-12 {
                    continue;
                }
                let left = self.solve_v_score(&left_tuple, trees, alpha, beta, blocked_leaves);
                let right = self.solve_v_score(&right_tuple, trees, alpha, beta, blocked_leaves);
                if left <= NEG_INF / 2.0 || right <= NEG_INF / 2.0 {
                    continue;
                }
                best = best.max(left + right - beta_sum);
                if best >= ub - 1e-12 {
                    self.memo_v.insert(Self::tuple_key(tuple), best);
                    return best;
                }
            }
        }

        self.memo_v.insert(Self::tuple_key(tuple), best);
        best
    }

    fn intersection_bits(&self, tuple: &[u32]) -> FixedBitSet {
        let mut inter = self.descendant_leaves[0][tuple[0] as usize].clone();
        for (ti, &node) in tuple.iter().enumerate().skip(1) {
            inter.intersect_with(&self.descendant_leaves[ti][node as usize]);
        }
        inter
    }

    fn intersection_stats(
        &self,
        inter: &FixedBitSet,
        alpha: &[f64],
        blocked_leaves: &[bool],
    ) -> (bool, f64, f64) {
        let mut has_leaf = false;
        let mut ub = 0.0;
        let mut best_leaf = NEG_INF;
        for leaf in inter.ones() {
            if leaf == 0 || blocked_leaves.get(leaf).copied().unwrap_or(false) {
                continue;
            }
            has_leaf = true;
            let score = alpha[leaf];
            if score > best_leaf {
                best_leaf = score;
            }
            if score > 0.0 {
                ub += score;
            }
        }
        (has_leaf, ub, best_leaf)
    }

    fn restricted_upper_bound(
        &self,
        inter: &FixedBitSet,
        tree_idx: usize,
        node: u32,
        alpha: &[f64],
        blocked_leaves: &[bool],
    ) -> f64 {
        let mut child_inter = inter.clone();
        child_inter.intersect_with(&self.descendant_leaves[tree_idx][node as usize]);
        let (has_leaf, ub, _best_leaf) = self.intersection_stats(&child_inter, alpha, blocked_leaves);
        if has_leaf { ub } else { NEG_INF }
    }

    fn oriented_side_upper_bound(
        &self,
        tuple: &[u32],
        trees: &[Tree],
        orient: usize,
        left_side: bool,
        alpha: &[f64],
        blocked_leaves: &[bool],
    ) -> f64 {
        let mut inter: Option<FixedBitSet> = None;
        for (ti, &node) in tuple.iter().enumerate() {
            let (left, right) = trees[ti].children_pair(node);
            let chosen = if ti == 0 || ((orient >> (ti - 1)) & 1) == 0 {
                if left_side { left } else { right }
            } else if left_side {
                right
            } else {
                left
            };
            match &mut inter {
                Some(bits) => bits.intersect_with(&self.descendant_leaves[ti][chosen as usize]),
                None => inter = Some(self.descendant_leaves[ti][chosen as usize].clone()),
            }
        }
        let Some(inter) = inter else {
            return NEG_INF;
        };
        let (has_leaf, ub, _best_leaf) = self.intersection_stats(&inter, alpha, blocked_leaves);
        if has_leaf { ub } else { NEG_INF }
    }

    fn collect_m_labels(
        &mut self,
        tuple: &[u32],
        trees: &[Tree],
        alpha: &[f64],
        beta: &[Vec<f64>],
        blocked_leaves: &[bool],
        out: &mut Vec<u32>,
    ) {
        let best = self.solve_m_score(tuple, trees, alpha, beta, blocked_leaves);
        if best <= NEG_INF / 2.0 {
            return;
        }
        let labels_here = self.intersection_labels(tuple, blocked_leaves);
        if labels_here.is_empty() {
            return;
        }

        if let Some(label) = labels_here
            .iter()
            .copied()
            .filter(|&label| (alpha[label as usize] - best).abs() <= 1e-12)
            .max_by(|&lhs, &rhs| alpha[lhs as usize].total_cmp(&alpha[rhs as usize]))
        {
            out.push(label);
            return;
        }

        for (ti, &node) in tuple.iter().enumerate() {
            if trees[ti].is_leaf(node) {
                continue;
            }
            let (left, right) = trees[ti].children_pair(node);
            let mut left_tuple = tuple.to_vec();
            left_tuple[ti] = left;
            let cand_left = self.solve_m_score(&left_tuple, trees, alpha, beta, blocked_leaves);
            if (cand_left - best).abs() <= 1e-12 {
                self.collect_m_labels(&left_tuple, trees, alpha, beta, blocked_leaves, out);
                return;
            }

            let mut right_tuple = tuple.to_vec();
            right_tuple[ti] = right;
            let cand_right =
                self.solve_m_score(&right_tuple, trees, alpha, beta, blocked_leaves);
            if (cand_right - best).abs() <= 1e-12 {
                self.collect_m_labels(&right_tuple, trees, alpha, beta, blocked_leaves, out);
                return;
            }
        }

        let rooted = self.solve_v_score(tuple, trees, alpha, beta, blocked_leaves);
        if (rooted - best).abs() <= 1e-12 {
            self.collect_v_labels(tuple, trees, alpha, beta, blocked_leaves, out);
        }
    }

    fn collect_v_labels(
        &mut self,
        tuple: &[u32],
        trees: &[Tree],
        alpha: &[f64],
        beta: &[Vec<f64>],
        blocked_leaves: &[bool],
        out: &mut Vec<u32>,
    ) {
        let best = self.solve_v_score(tuple, trees, alpha, beta, blocked_leaves);
        if best <= NEG_INF / 2.0 {
            return;
        }

        if tuple.iter().enumerate().all(|(ti, &node)| trees[ti].is_leaf(node)) {
            let label = trees[0].label[tuple[0] as usize];
            out.push(label);
            return;
        }

        for (ti, &node) in tuple.iter().enumerate() {
            if trees[ti].is_leaf(node) {
                continue;
            }
            let skip_penalty = beta[ti][node as usize];
            let (left, right) = trees[ti].children_pair(node);

            let mut left_tuple = tuple.to_vec();
            left_tuple[ti] = left;
            let cand_left = self.solve_v_score(&left_tuple, trees, alpha, beta, blocked_leaves);
            if cand_left > NEG_INF / 2.0 && (cand_left - skip_penalty - best).abs() <= 1e-12 {
                self.collect_v_labels(&left_tuple, trees, alpha, beta, blocked_leaves, out);
                return;
            }

            let mut right_tuple = tuple.to_vec();
            right_tuple[ti] = right;
            let cand_right =
                self.solve_v_score(&right_tuple, trees, alpha, beta, blocked_leaves);
            if cand_right > NEG_INF / 2.0 && (cand_right - skip_penalty - best).abs() <= 1e-12 {
                self.collect_v_labels(&right_tuple, trees, alpha, beta, blocked_leaves, out);
                return;
            }
        }

        if tuple.iter().enumerate().all(|(ti, &node)| !trees[ti].is_leaf(node)) {
            let beta_sum = tuple
                .iter()
                .enumerate()
                .map(|(ti, &node)| beta[ti][node as usize])
                .sum::<f64>();
            let orient_count = 1usize << (tuple.len().saturating_sub(1));
            for orient in 0..orient_count {
                let (left_tuple, right_tuple) = oriented_children_tuple(tuple, trees, orient);
                let left = self.solve_v_score(&left_tuple, trees, alpha, beta, blocked_leaves);
                let right = self.solve_v_score(&right_tuple, trees, alpha, beta, blocked_leaves);
                if left <= NEG_INF / 2.0 || right <= NEG_INF / 2.0 {
                    continue;
                }
                if (left + right - beta_sum - best).abs() <= 1e-12 {
                    self.collect_v_labels(&left_tuple, trees, alpha, beta, blocked_leaves, out);
                    self.collect_v_labels(&right_tuple, trees, alpha, beta, blocked_leaves, out);
                    return;
                }
            }
        }
    }

    fn intersection_labels(&self, tuple: &[u32], blocked_leaves: &[bool]) -> Vec<u32> {
        let mut inter = self.descendant_leaves[0][tuple[0] as usize].clone();
        for (ti, &node) in tuple.iter().enumerate().skip(1) {
            inter.intersect_with(&self.descendant_leaves[ti][node as usize]);
        }
        inter
            .ones()
            .filter(|&leaf| leaf > 0 && !blocked_leaves.get(leaf).copied().unwrap_or(false))
            .map(|leaf| leaf as u32)
            .collect()
    }
}

fn oriented_children_tuple(tuple: &[u32], trees: &[Tree], orient: usize) -> (Vec<u32>, Vec<u32>) {
    let mut left_tuple = Vec::with_capacity(tuple.len());
    let mut right_tuple = Vec::with_capacity(tuple.len());
    for (ti, &node) in tuple.iter().enumerate() {
        let (left, right) = trees[ti].children_pair(node);
        if ti == 0 || ((orient >> (ti - 1)) & 1) == 0 {
            left_tuple.push(left);
            right_tuple.push(right);
        } else {
            left_tuple.push(right);
            right_tuple.push(left);
        }
    }
    (left_tuple, right_tuple)
}

#[derive(Clone, Copy)]
enum PairDpSideChoice {
    Singleton,
    Split(usize),
}

#[derive(Clone, Copy)]
struct PairDpSideState {
    score: f64,
    choice: PairDpSideChoice,
}

struct PairDpPricer<'a> {
    trees: &'a [Tree],
    workspace: &'a MultiPricerWorkspace,
    active_labels: Vec<u32>,
    active_nodes: Vec<Vec<u32>>,
    alpha: &'a [f64],
    beta: &'a [Vec<f64>],
    prefix_beta: Vec<Vec<f64>>,
    roots: Vec<Vec<u32>>,
    side_child: Vec<Vec<u32>>,
    pair_order: Vec<(usize, usize)>,
    memo_pair: Vec<Option<f64>>,
    memo_side: Vec<Option<PairDpSideState>>,
    memo_pair_labels: Vec<Option<Vec<u32>>>,
    solving_pair: Vec<bool>,
    solving_side: Vec<bool>,
    solved_2tree: bool,
}

impl<'a> PairDpPricer<'a> {
    fn new(
        workspace: &'a MultiPricerWorkspace,
        trees: &'a [Tree],
        num_leaves: usize,
        alpha: &'a [f64],
        beta: &'a [Vec<f64>],
        blocked_leaves: &[bool],
    ) -> Self {
        let active_labels = (1..=num_leaves as u32)
            .filter(|&label| {
                !blocked_leaves[label as usize] && alpha[label as usize] > 1.0e-12
            })
            .collect::<Vec<_>>();
        let p = active_labels.len();
        let pair_count = p * p;
        let stride = workspace.label_stride;

        let mut active_nodes = Vec::with_capacity(trees.len());
        let mut prefix_beta = Vec::with_capacity(trees.len());
        let mut roots = Vec::with_capacity(trees.len());
        let mut side_child = Vec::with_capacity(trees.len());

        for (ti, tree) in trees.iter().enumerate() {
            let node_by_label = &workspace.label_node[ti];
            let nodes: Vec<u32> = active_labels
                .iter()
                .map(|&label| node_by_label[label as usize])
                .collect();
            active_nodes.push(nodes);

            let mut prefix = vec![0.0; tree.num_nodes()];
            for node in tree.pre_order() {
                let parent = tree.parent[node as usize];
                let parent_sum = if parent == klados_core::NONE {
                    0.0
                } else {
                    prefix[parent as usize]
                };
                let own = if tree.is_leaf(node) {
                    0.0
                } else {
                    beta[ti][node as usize]
                };
                prefix[node as usize] = parent_sum + own;
            }
            prefix_beta.push(prefix);

            let lca_table = &workspace.label_lca[ti];
            let side_table = &workspace.label_side_child[ti];
            let mut tree_roots = vec![0u32; pair_count];
            let mut tree_side_child = vec![0u32; pair_count];
            for a in 0..p {
                let la = active_labels[a] as usize;
                let base_ws = la * stride;
                let base_out = a * p;
                for b in 0..p {
                    let lb = active_labels[b] as usize;
                    let ws_idx = base_ws + lb;
                    let out_idx = base_out + b;
                    tree_roots[out_idx] = lca_table[ws_idx];
                    tree_side_child[out_idx] = side_table[ws_idx];
                }
            }
            roots.push(tree_roots);
            side_child.push(tree_side_child);
        }

        let mut pair_order = Vec::new();
        if trees.len() == 2 {
            pair_order.reserve(pair_count.saturating_sub(p));
            for a in 0..p {
                for b in 0..p {
                    if a == b {
                        continue;
                    }
                    let idx = a * p + b;
                    let key = trees
                        .iter()
                        .enumerate()
                        .map(|(ti, tree)| tree.subtree_size[roots[ti][idx] as usize] as usize)
                        .sum::<usize>();
                    pair_order.push((key, idx));
                }
            }
            pair_order
                .sort_unstable_by(|lhs, rhs| lhs.0.cmp(&rhs.0).then_with(|| lhs.1.cmp(&rhs.1)));
        }

        Self {
            trees,
            workspace,
            active_labels,
            active_nodes,
            alpha,
            beta,
            prefix_beta,
            roots,
            side_child,
            pair_order,
            memo_pair: vec![None; pair_count],
            memo_side: vec![None; pair_count],
            memo_pair_labels: vec![None; pair_count],
            solving_pair: vec![false; pair_count],
            solving_side: vec![false; pair_count],
            solved_2tree: false,
        }
    }

    fn solve_best(&mut self) -> Option<(f64, Vec<u32>)> {
        if self.active_labels.is_empty() {
            return None;
        }

        let p = self.active_labels.len();
        let mut best_score = NEG_INF;
        let mut best_labels = Vec::new();

        for a in 0..p {
            let label = self.active_labels[a];
            let score = self.alpha[label as usize];
            if score > best_score + 1.0e-12 {
                best_score = score;
                best_labels.clear();
                best_labels.push(label);
            }
        }

        for a in 0..p {
            for b in (a + 1)..p {
                let score = self.solve_pair(a, b);
                if score <= best_score + 1.0e-12 {
                    continue;
                }
                let mut labels = Vec::new();
                self.collect_pair(a, b, &mut labels);
                labels.sort_unstable();
                labels.dedup();
                if score > best_score + 1.0e-12
                    || ((score - best_score).abs() <= 1.0e-12 && labels.len() > best_labels.len())
                {
                    best_score = score;
                    best_labels = labels;
                }
            }
        }

        if exact_pricer_profile() {
            eprintln!(
                "[maf-bp-multi] pairdp profile: p={} pairs={}",
                self.active_labels.len(),
                self.active_labels.len().saturating_mul(self.active_labels.len().saturating_sub(1)) / 2
            );
        }

        (best_score > NEG_INF / 2.0).then_some((best_score, best_labels))
    }

    fn collect_profitable_columns<F>(
        &mut self,
        limit: usize,
        mut accept: F,
    ) -> (usize, Vec<(f64, Vec<u32>)>)
    where
        F: FnMut(&[u32]) -> bool,
    {
        if self.active_labels.len() < 2 || limit == 0 {
            return (0, Vec::new());
        }

        // First pass: gather profitable (score, a, b) without extracting labels.
        let p = self.active_labels.len();
        let mut scored: Vec<(f64, usize, usize)> =
            Vec::with_capacity(p.saturating_mul(p.saturating_sub(1)) / 2);
        for a in 0..p {
            for b in (a + 1)..p {
                let score = self.solve_pair(a, b);
                if score <= 1.0 + 1.0e-8 {
                    continue;
                }
                scored.push((score, a, b));
            }
        }
        let raw_candidates = scored.len();
        if raw_candidates == 0 {
            return (0, Vec::new());
        }
        // Highest-score pairs first so label extraction focuses on the best columns.
        scored.sort_unstable_by(|l, r| r.0.total_cmp(&l.0));

        // Second pass: extract labels in score order; stop early once we have `limit`
        // filter-accepted unique columns. This avoids materializing labels for the
        // bulk of profitable pairs that would be truncated away.
        let mut dedup: FxHashMap<Vec<u32>, f64> = FxHashMap::default();
        for &(score, a, b) in scored.iter() {
            if dedup.len() >= limit {
                break;
            }
            let labels = self.pair_labels(a, b);
            if labels.len() < 2 {
                continue;
            }
            if !accept(&labels) {
                continue;
            }
            dedup
                .entry(labels)
                .and_modify(|best| *best = best.max(score))
                .or_insert(score);
        }

        let mut out = dedup.into_iter().map(|(labels, score)| (score, labels)).collect::<Vec<_>>();
        out.sort_unstable_by(|lhs, rhs| {
            rhs.0
                .total_cmp(&lhs.0)
                .then_with(|| rhs.1.len().cmp(&lhs.1.len()))
                .then_with(|| lhs.1.cmp(&rhs.1))
        });
        if out.len() > limit {
            out.truncate(limit);
        }
        (raw_candidates, out)
    }

    fn pair_idx(&self, a: usize, b: usize) -> usize {
        a * self.active_labels.len() + b
    }

    fn solve_all_2tree(&mut self) {
        if self.solved_2tree {
            return;
        }

        let p = self.active_labels.len();
        let mut pos = 0usize;
        while pos < self.pair_order.len() {
            let key = self.pair_order[pos].0;
            let mut end = pos + 1;
            while end < self.pair_order.len() && self.pair_order[end].0 == key {
                end += 1;
            }

            for &(_, idx) in &self.pair_order[pos..end] {
                let a = idx / p;
                let b = idx % p;
                let label_a = self.active_labels[a];
                let mut best_score =
                    self.alpha[label_a as usize] - self.singleton_chain_penalty_2tree(a, idx);
                let mut best_choice = PairDpSideChoice::Singleton;

                for c in 0..p {
                    if c == a || c == b || !self.is_same_side_2tree(idx, c) {
                        continue;
                    }
                    let idx_ac = self.pair_idx(a, c);
                    let child_score = self.memo_pair[idx_ac]
                        .expect("child pair must be solved earlier in pair DP order");
                    if child_score <= NEG_INF / 2.0 {
                        continue;
                    }
                    let cand = child_score - self.transition_chain_penalty_2tree(idx, idx_ac);
                    if cand > best_score + 1.0e-12 {
                        best_score = cand;
                        best_choice = PairDpSideChoice::Split(c);
                    }
                }

                self.memo_side[idx] = Some(PairDpSideState {
                    score: best_score,
                    choice: best_choice,
                });
            }

            for &(_, idx) in &self.pair_order[pos..end] {
                let a = idx / p;
                let b = idx % p;
                let left = self.memo_side[idx]
                    .expect("left side state present after pair DP side phase")
                    .score;
                let right = self.memo_side[self.pair_idx(b, a)]
                    .expect("right side state present after pair DP side phase")
                    .score;
                let score = if left <= NEG_INF / 2.0 || right <= NEG_INF / 2.0 {
                    NEG_INF
                } else {
                    -self.root_penalty_2tree(idx) + left + right
                };
                self.memo_pair[idx] = Some(score);
            }

            pos = end;
        }

        self.solved_2tree = true;
    }

    fn solve_pair(&mut self, a: usize, b: usize) -> f64 {
        debug_assert!(a != b);
        let idx = self.pair_idx(a, b);
        if self.trees.len() == 2 {
            self.solve_all_2tree();
            return self.memo_pair[idx].unwrap_or(NEG_INF);
        }
        if let Some(score) = self.memo_pair[idx] {
            return score;
        }
        if self.solving_pair[idx] {
            return NEG_INF;
        }
        self.solving_pair[idx] = true;

        let left = self.solve_side(a, b);
        let right = self.solve_side(b, a);
        let score = if left <= NEG_INF / 2.0 || right <= NEG_INF / 2.0 {
            NEG_INF
        } else {
            -self.root_penalty(a, b) + left + right
        };

        self.solving_pair[idx] = false;
        self.memo_pair[idx] = Some(score);
        score
    }

    fn solve_side(&mut self, a: usize, b: usize) -> f64 {
        debug_assert!(a != b);
        let idx = self.pair_idx(a, b);
        if self.trees.len() == 2 {
            self.solve_all_2tree();
            return self.memo_side[idx]
                .expect("side state present after 2-tree pair DP solve")
                .score;
        }
        if let Some(state) = self.memo_side[idx] {
            return state.score;
        }
        if self.solving_side[idx] {
            return NEG_INF;
        }
        self.solving_side[idx] = true;

        let label_a = self.active_labels[a];
        let mut best_score = self.alpha[label_a as usize] - self.singleton_chain_penalty(a, b);
        let mut best_choice = PairDpSideChoice::Singleton;

        for c in 0..self.active_labels.len() {
            if c == a || c == b || !self.is_same_side(a, b, c) {
                continue;
            }
            let child_score = self.solve_pair(a, c);
            if child_score <= NEG_INF / 2.0 {
                continue;
            }
            let cand = child_score - self.transition_chain_penalty(a, b, c);
            if cand > best_score + 1.0e-12 {
                best_score = cand;
                best_choice = PairDpSideChoice::Split(c);
            }
        }

        self.solving_side[idx] = false;
        self.memo_side[idx] = Some(PairDpSideState {
            score: best_score,
            choice: best_choice,
        });
        best_score
    }

    fn collect_pair(&mut self, a: usize, b: usize, out: &mut Vec<u32>) {
        self.collect_side(a, b, out);
        self.collect_side(b, a, out);
    }

    fn pair_labels(&mut self, a: usize, b: usize) -> Vec<u32> {
        let idx = self.pair_idx(a, b);
        if let Some(labels) = self.memo_pair_labels[idx].clone() {
            return labels;
        }
        let mut labels = Vec::new();
        self.collect_pair(a, b, &mut labels);
        labels.sort_unstable();
        labels.dedup();
        self.memo_pair_labels[idx] = Some(labels.clone());
        labels
    }

    fn collect_side(&mut self, a: usize, b: usize, out: &mut Vec<u32>) {
        let idx = self.pair_idx(a, b);
        let state = self.memo_side[idx].unwrap_or_else(|| {
            let _ = self.solve_side(a, b);
            self.memo_side[idx].expect("side state present after solve")
        });
        match state.choice {
            PairDpSideChoice::Singleton => out.push(self.active_labels[a]),
            PairDpSideChoice::Split(c) => self.collect_pair(a, c, out),
        }
    }

    fn root_penalty(&self, a: usize, b: usize) -> f64 {
        let idx = self.pair_idx(a, b);
        self.trees
            .iter()
            .enumerate()
            .map(|(ti, _)| self.beta[ti][self.roots[ti][idx] as usize])
            .sum()
    }

    fn singleton_chain_penalty(&self, a: usize, b: usize) -> f64 {
        let idx = self.pair_idx(a, b);
        self.trees
            .iter()
            .enumerate()
            .map(|(ti, tree)| {
                self.path_internal_penalty(
                    ti,
                    tree,
                    self.side_child[ti][idx],
                    self.active_nodes[ti][a],
                )
            })
            .sum()
    }

    fn transition_chain_penalty(&self, a: usize, b: usize, c: usize) -> f64 {
        let idx_ab = self.pair_idx(a, b);
        let idx_ac = self.pair_idx(a, c);
        self.trees
            .iter()
            .enumerate()
            .map(|(ti, tree)| {
                self.path_internal_penalty(
                    ti,
                    tree,
                    self.side_child[ti][idx_ab],
                    self.roots[ti][idx_ac],
                )
            })
            .sum()
    }

    #[inline(always)]
    fn root_penalty_2tree(&self, idx: usize) -> f64 {
        self.beta[0][self.roots[0][idx] as usize] + self.beta[1][self.roots[1][idx] as usize]
    }

    #[inline(always)]
    fn singleton_chain_penalty_2tree(&self, a: usize, idx: usize) -> f64 {
        self.path_internal_penalty(0, &self.trees[0], self.side_child[0][idx], self.active_nodes[0][a])
            + self.path_internal_penalty(1, &self.trees[1], self.side_child[1][idx], self.active_nodes[1][a])
    }

    #[inline(always)]
    fn transition_chain_penalty_2tree(&self, idx_ab: usize, idx_ac: usize) -> f64 {
        self.path_internal_penalty(0, &self.trees[0], self.side_child[0][idx_ab], self.roots[0][idx_ac])
            + self.path_internal_penalty(1, &self.trees[1], self.side_child[1][idx_ab], self.roots[1][idx_ac])
    }

    fn path_internal_penalty(&self, ti: usize, tree: &Tree, anc: u32, desc: u32) -> f64 {
        if anc == desc {
            return 0.0;
        }
        let desc_parent = tree.parent[desc as usize];
        if desc_parent == klados_core::NONE {
            return 0.0;
        }
        let anc_parent = tree.parent[anc as usize];
        let upper = self.prefix_beta[ti][desc_parent as usize];
        let lower = if anc_parent == klados_core::NONE {
            0.0
        } else {
            self.prefix_beta[ti][anc_parent as usize]
        };
        (upper - lower).max(0.0)
    }

    fn is_same_side(&self, a: usize, b: usize, c: usize) -> bool {
        let idx = self.pair_idx(a, b);
        let label_c = self.active_labels[c] as usize;
        self.trees.iter().enumerate().all(|(ti, _)| {
            self.workspace.descendant_leaves[ti][self.side_child[ti][idx] as usize]
                .contains(label_c)
        })
    }

    #[inline(always)]
    fn is_same_side_2tree(&self, idx: usize, c: usize) -> bool {
        let label_c = self.active_labels[c] as usize;
        self.workspace.descendant_leaves[0][self.side_child[0][idx] as usize].contains(label_c)
            && self.workspace.descendant_leaves[1][self.side_child[1][idx] as usize]
                .contains(label_c)
    }
}

fn solve_best_pairdp_column(
    workspace: &MultiPricerWorkspace,
    trees: &[Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
) -> Option<(f64, Vec<u32>)> {
    let mut pricer = PairDpPricer::new(
        workspace,
        trees,
        num_leaves,
        alpha,
        beta,
        blocked_leaves,
    );
    pricer.solve_best()
}

fn solve_best_pairdp_columns<F>(
    workspace: &MultiPricerWorkspace,
    trees: &[Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    limit: usize,
    accept: F,
) -> Vec<(f64, Vec<u32>)>
where
    F: FnMut(&[u32]) -> bool,
{
    let t_setup = Instant::now();
    let mut pricer = PairDpPricer::new(
        workspace,
        trees,
        num_leaves,
        alpha,
        beta,
        blocked_leaves,
    );
    pricer_timing_add_setup(t_setup.elapsed().as_secs_f64() * 1000.0);
    let t_collect = Instant::now();
    let (raw, out) = pricer.collect_profitable_columns(limit, accept);
    pricer_timing_add_collect(t_collect.elapsed().as_secs_f64() * 1000.0, raw);
    out
}

fn price_best_new_pairdp_column_single(
    pricer_ws: &mut MultiPricerWorkspace,
    trees: &[Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    seen: &FxHashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
    must_link_pairs: &[LeafPair],
    cannot_link_pairs: &[LeafPair],
) -> Option<(f64, Vec<u32>)> {
    let base = solve_best_pairdp_column(
        pricer_ws,
        trees,
        num_leaves,
        alpha,
        beta,
        blocked_leaves,
    )?;
    let (base_score, base_labels) = base;
    if pricing_candidate_allowed(
        &base_labels,
        seen,
        forbidden,
        must_link_pairs,
        cannot_link_pairs,
    ) {
        return Some((base_score, base_labels));
    }

    let free_labels = (1..=num_leaves as u32)
        .filter(|&label| !blocked_leaves[label as usize])
        .collect::<Vec<_>>();
    let ordinary_upper = alpha.iter().map(|&value| value.max(0.0)).sum::<f64>();
    let beta_sum = beta
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .sum::<f64>();
    let required_bonus = ordinary_upper + beta_sum + 1.0;
    let mut incumbent: Option<(f64, Vec<u32>)> = None;
    search_pricing_prefix_dfs_pairdp(
        pricer_ws,
        trees,
        num_leaves,
        alpha,
        beta,
        blocked_leaves,
        seen,
        forbidden,
        must_link_pairs,
        cannot_link_pairs,
        &free_labels,
        &[],
        required_bonus,
        &mut incumbent,
    );
    incumbent
}

fn price_best_new_pairdp_columns(
    pricer_ws: &mut MultiPricerWorkspace,
    trees: &[Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    seen: &FxHashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
    must_link_pairs: &[LeafPair],
    cannot_link_pairs: &[LeafPair],
) -> Vec<(f64, Vec<u32>)> {
    if pairdp_batch_size() <= 1 {
        return price_best_new_pairdp_column_single(
            pricer_ws,
            trees,
            num_leaves,
            alpha,
            beta,
            blocked_leaves,
            seen,
            forbidden,
            must_link_pairs,
            cannot_link_pairs,
        )
        .into_iter()
        .collect();
    }

    let candidates = solve_best_pairdp_columns(
        pricer_ws,
        trees,
        num_leaves,
        alpha,
        beta,
        blocked_leaves,
        pairdp_batch_size(),
        |labels| {
            pricing_candidate_allowed(
                labels,
                seen,
                forbidden,
                must_link_pairs,
                cannot_link_pairs,
            )
        },
    );
    pricer_timing_add_filter(0.0, candidates.len());

    if exact_pricer_profile() && !candidates.is_empty() {
        eprintln!(
            "[maf-bp-multi] pairdp batch: emitted={} batch_size={}",
            candidates.len(),
            pairdp_batch_size()
        );
    }

    candidates
}

fn search_pricing_prefix_dfs_pairdp(
    pricer_ws: &mut MultiPricerWorkspace,
    trees: &[Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    seen: &FxHashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
    must_link_pairs: &[LeafPair],
    cannot_link_pairs: &[LeafPair],
    free_labels: &[u32],
    prefix_membership: &[bool],
    required_bonus: f64,
    incumbent: &mut Option<(f64, Vec<u32>)>,
) {
    let Some(candidate) = solve_pricing_prefix_subproblem_pairdp(
        pricer_ws,
        trees,
        num_leaves,
        alpha,
        beta,
        blocked_leaves,
        free_labels,
        prefix_membership,
        required_bonus,
    ) else {
        return;
    };

    if let Some((best_score, best_labels)) = incumbent.as_ref() {
        if candidate.score < *best_score - 1e-12 {
            return;
        }
        if (candidate.score - *best_score).abs() <= 1e-12
            && candidate.labels.len() <= best_labels.len()
        {
            return;
        }
    }

    if pricing_candidate_allowed(
        &candidate.labels,
        seen,
        forbidden,
        must_link_pairs,
        cannot_link_pairs,
    ) {
        match incumbent {
            Some((best_score, best_labels))
                if candidate.score < *best_score - 1e-12
                    || ((candidate.score - *best_score).abs() <= 1e-12
                        && candidate.labels.len() <= best_labels.len()) => {}
            _ => *incumbent = Some((candidate.score, candidate.labels)),
        }
        return;
    }

    let membership = membership_over_free_labels(free_labels, &candidate.labels);
    let fixed_prefix_len = candidate.prefix_membership.len();
    for split_idx in (fixed_prefix_len..free_labels.len()).rev() {
        let mut child_prefix = candidate.prefix_membership.clone();
        child_prefix.extend_from_slice(&membership[fixed_prefix_len..split_idx]);
        child_prefix.push(!membership[split_idx]);
        search_pricing_prefix_dfs_pairdp(
            pricer_ws,
            trees,
            num_leaves,
            alpha,
            beta,
            blocked_leaves,
            seen,
            forbidden,
            must_link_pairs,
            cannot_link_pairs,
            free_labels,
            &child_prefix,
            required_bonus,
            incumbent,
        );
    }
}

fn solve_pricing_prefix_subproblem_pairdp(
    pricer_ws: &mut MultiPricerWorkspace,
    trees: &[Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    free_labels: &[u32],
    prefix_membership: &[bool],
    required_bonus: f64,
) -> Option<PricingPrefixNode> {
    let mut blocked = blocked_leaves.to_vec();
    let mut alpha_mod = alpha.to_vec();
    let mut required_count = 0usize;
    for (idx, &present) in prefix_membership.iter().enumerate() {
        let label = free_labels[idx] as usize;
        if present {
            alpha_mod[label] += required_bonus;
            required_count += 1;
        } else {
            blocked[label] = true;
        }
    }

    let (score_with_bonus, labels) = solve_best_pairdp_column(
        pricer_ws,
        trees,
        num_leaves,
        &alpha_mod,
        beta,
        &blocked,
    )?;
    for (idx, &present) in prefix_membership.iter().enumerate() {
        let label = free_labels[idx];
        let actually_present = labels.binary_search(&label).is_ok();
        if actually_present != present {
            return None;
        }
    }
    Some(PricingPrefixNode {
        prefix_membership: prefix_membership.to_vec(),
        score: score_with_bonus - required_bonus * required_count as f64,
        labels,
    })
}

fn price_best_new_exact_column(
    pricer_ws: &mut MultiPricerWorkspace,
    trees: &[Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    seen: &FxHashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
    must_link_pairs: &[LeafPair],
    cannot_link_pairs: &[LeafPair],
) -> Option<(f64, Vec<u32>)> {
    let base = pricer_ws.solve_best(trees, alpha, beta, blocked_leaves);
    let Some((base_score, base_labels)) = base else {
        return None;
    };
    if pricing_candidate_allowed(
        &base_labels,
        seen,
        forbidden,
        must_link_pairs,
        cannot_link_pairs,
    ) {
        return Some((base_score, base_labels));
    }

    let free_labels = (1..=num_leaves as u32)
        .filter(|&label| !blocked_leaves[label as usize])
        .collect::<Vec<_>>();
    let ordinary_upper = alpha.iter().map(|&value| value.max(0.0)).sum::<f64>();
    let beta_sum = beta
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .sum::<f64>();
    let required_bonus = ordinary_upper + beta_sum + 1.0;
    let mut incumbent: Option<(f64, Vec<u32>)> = None;
    search_pricing_prefix_dfs(
        pricer_ws,
        trees,
        alpha,
        beta,
        blocked_leaves,
        seen,
        forbidden,
        must_link_pairs,
        cannot_link_pairs,
        &free_labels,
        &[],
        required_bonus,
        &mut incumbent,
    );
    incumbent
}

fn search_pricing_prefix_dfs(
    pricer_ws: &mut MultiPricerWorkspace,
    trees: &[Tree],
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    seen: &FxHashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
    must_link_pairs: &[LeafPair],
    cannot_link_pairs: &[LeafPair],
    free_labels: &[u32],
    prefix_membership: &[bool],
    required_bonus: f64,
    incumbent: &mut Option<(f64, Vec<u32>)>,
) {
    let Some(candidate) = solve_pricing_prefix_subproblem(
        pricer_ws,
        trees,
        alpha,
        beta,
        blocked_leaves,
        free_labels,
        prefix_membership,
        required_bonus,
    ) else {
        return;
    };

    if let Some((best_score, best_labels)) = incumbent.as_ref() {
        if candidate.score < *best_score - 1e-12 {
            return;
        }
        if (candidate.score - *best_score).abs() <= 1e-12
            && candidate.labels.len() <= best_labels.len()
        {
            return;
        }
    }

    if pricing_candidate_allowed(
        &candidate.labels,
        seen,
        forbidden,
        must_link_pairs,
        cannot_link_pairs,
    ) {
        match incumbent {
            Some((best_score, best_labels))
                if candidate.score < *best_score - 1e-12
                    || ((candidate.score - *best_score).abs() <= 1e-12
                        && candidate.labels.len() <= best_labels.len()) => {}
            _ => *incumbent = Some((candidate.score, candidate.labels)),
        }
        return;
    }

    let membership = membership_over_free_labels(free_labels, &candidate.labels);
    let fixed_prefix_len = candidate.prefix_membership.len();
    for split_idx in (fixed_prefix_len..free_labels.len()).rev() {
        let mut child_prefix = candidate.prefix_membership.clone();
        child_prefix.extend_from_slice(&membership[fixed_prefix_len..split_idx]);
        child_prefix.push(!membership[split_idx]);
        search_pricing_prefix_dfs(
            pricer_ws,
            trees,
            alpha,
            beta,
            blocked_leaves,
            seen,
            forbidden,
            must_link_pairs,
            cannot_link_pairs,
            free_labels,
            &child_prefix,
            required_bonus,
            incumbent,
        );
    }
}

fn solve_pricing_prefix_subproblem(
    pricer_ws: &mut MultiPricerWorkspace,
    trees: &[Tree],
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    free_labels: &[u32],
    prefix_membership: &[bool],
    required_bonus: f64,
) -> Option<PricingPrefixNode> {
    let mut blocked = blocked_leaves.to_vec();
    let mut alpha_mod = alpha.to_vec();
    let mut required_count = 0usize;
    for (idx, &present) in prefix_membership.iter().enumerate() {
        let label = free_labels[idx] as usize;
        if present {
            alpha_mod[label] += required_bonus;
            required_count += 1;
        } else {
            blocked[label] = true;
        }
    }

    let priced = pricer_ws.solve_best(trees, &alpha_mod, beta, &blocked)?;
    let (score_with_bonus, labels) = priced;
    for (idx, &present) in prefix_membership.iter().enumerate() {
        let label = free_labels[idx];
        let actually_present = labels.binary_search(&label).is_ok();
        if actually_present != present {
            return None;
        }
    }
    Some(PricingPrefixNode {
        prefix_membership: prefix_membership.to_vec(),
        score: score_with_bonus - required_bonus * required_count as f64,
        labels,
    })
}

fn membership_over_free_labels(free_labels: &[u32], labels: &[u32]) -> Vec<bool> {
    let mut membership = vec![false; free_labels.len()];
    let mut i = 0usize;
    let mut j = 0usize;
    while i < free_labels.len() && j < labels.len() {
        match free_labels[i].cmp(&labels[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                membership[i] = true;
                i += 1;
                j += 1;
            }
        }
    }
    membership
}

struct RmpLpResult {
    objective: f64,
    column_values: Vec<f64>,
    leaf_duals: Vec<f64>,
    node_duals: Vec<Vec<f64>>,
}

struct PersistentRmp {
    model: Option<Model>,
    leaf_rows: Vec<Row>,
    leaf_row_idx: Vec<usize>,
    node_rows: Vec<Vec<Option<Row>>>,
    node_row_idx: Vec<Vec<Option<usize>>>,
    /// HiGHS column index for each global column (parallel to state.columns).
    col_idx: Vec<i32>,
    /// Current lower bound applied in HiGHS for each added column.
    current_lower: Vec<f64>,
    /// Current upper bound applied in HiGHS for each added column.
    current_upper: Vec<f64>,
}

impl PersistentRmp {
    fn new(
        columns: &[BpColumn],
        trees: &[Tree],
        num_leaves: usize,
    ) -> Result<Self, String> {
        let mut model = Model::new(ColProblem::default());
        model.make_quiet();
        model.set_option("threads", 1_i32);
        model.set_option("presolve", "on");
        model.set_option("solver", "simplex");

        let mut next_row = 0usize;
        let leaf_rows: Vec<Row> = (0..=num_leaves)
            .map(|leaf| {
                let row = if leaf == 0 {
                    model.add_row(0.0..=0.0, Vec::new())
                } else {
                    model.add_row(1.0..=1.0, Vec::new())
                };
                next_row += 1;
                row
            })
            .collect();
        let leaf_row_idx: Vec<usize> = (0..=num_leaves).collect();

        let mut node_rows = Vec::new();
        let mut node_row_idx = Vec::new();
        for tree in trees {
            let mut rows = Vec::new();
            let mut idxs = Vec::new();
            for node in 0..tree.num_nodes() as u32 {
                if tree.is_leaf(node) {
                    rows.push(None);
                    idxs.push(None);
                } else {
                    let row = model.add_row(..=1.0, Vec::new());
                    let ri = next_row;
                    next_row += 1;
                    rows.push(Some(row));
                    idxs.push(Some(ri));
                }
            }
            node_rows.push(rows);
            node_row_idx.push(idxs);
        }

        let mut rmp = Self {
            model: Some(model),
            leaf_rows,
            leaf_row_idx,
            node_rows,
            node_row_idx,
            col_idx: Vec::new(),
            current_lower: Vec::new(),
            current_upper: Vec::new(),
        };
        for (ci, col) in columns.iter().enumerate() {
            rmp.add_column(ci, col, trees);
        }
        Ok(rmp)
    }

    fn add_column(&mut self, global_ci: usize, col: &BpColumn, trees: &[Tree]) {
        debug_assert_eq!(global_ci, self.col_idx.len());
        let mut rows = col
            .labels
            .iter()
            .map(|&label| (self.leaf_rows[label as usize], 1.0))
            .collect::<Vec<_>>();
        if col.labels.len() >= 2 {
            for (ti, tree) in trees.iter().enumerate() {
                for &node in &col.covered_internal_nodes[ti] {
                    debug_assert!(!tree.is_leaf(node as u32));
                    if let Some(row) = self.node_rows[ti][node] {
                        rows.push((row, 1.0));
                    }
                }
            }
        }
        self.model
            .as_mut()
            .expect("RMP model present")
            .add_col(1.0, 0.0.., rows);
        self.col_idx.push(global_ci as i32);
        self.current_lower.push(0.0);
        self.current_upper.push(f64::INFINITY);
    }

    fn apply_node_bounds(
        &mut self,
        columns: &[BpColumn],
        fixed_to_one: &[usize],
        fixed_to_zero: &[usize],
        must_link_pairs: &[LeafPair],
        cannot_link_pairs: &[LeafPair],
        blocked_leaves: &[bool],
    ) {
        let fixed_one_set: FxHashSet<usize> = fixed_to_one.iter().copied().collect();
        let fixed_zero_set: FxHashSet<usize> = fixed_to_zero.iter().copied().collect();
        let ptr = self
            .model
            .as_mut()
            .expect("RMP model present")
            .as_mut_ptr();
        for ci in 0..self.col_idx.len() {
            let (desired_lo, desired_hi) = if fixed_one_set.contains(&ci) {
                (1.0, 1.0)
            } else if fixed_zero_set.contains(&ci) {
                (0.0, 0.0)
            } else if columns[ci]
                .labels
                .iter()
                .any(|&l| blocked_leaves[l as usize])
                || !column_respects_branchings(
                    columns,
                    ci,
                    fixed_to_one,
                    fixed_to_zero,
                    must_link_pairs,
                    cannot_link_pairs,
                )
            {
                (0.0, 0.0)
            } else {
                (0.0, f64::INFINITY)
            };
            if (self.current_lower[ci] - desired_lo).abs() > 0.0
                || (self.current_upper[ci].is_finite() != desired_hi.is_finite())
                || ((self.current_upper[ci] - desired_hi).abs() > 0.0
                    && self.current_upper[ci].is_finite()
                    && desired_hi.is_finite())
            {
                unsafe {
                    highs_sys::Highs_changeColBounds(
                        ptr,
                        self.col_idx[ci],
                        desired_lo,
                        desired_hi,
                    );
                }
                self.current_lower[ci] = desired_lo;
                self.current_upper[ci] = desired_hi;
            }
        }
    }

    fn solve(&mut self, total_columns: usize) -> Result<RmpLpResult, String> {
        let solved = self.model.take().expect("RMP model present").solve();
        let status = solved.status();
        if status != HighsModelStatus::Optimal {
            self.model = Some(Model::from(solved));
            return Err(format!("LP status: {:?}", status));
        }
        let solution = solved.get_solution();
        let mut column_values = vec![0.0; total_columns];
        let solution_cols = solution.columns();
        for (local_idx, &global_ci) in self.col_idx.iter().enumerate() {
            let ci = global_ci as usize;
            if ci < total_columns {
                column_values[ci] = solution_cols[local_idx];
            }
        }
        let objective = solved.objective_value();
        let dual_rows = solution.dual_rows();
        let leaf_duals = self
            .leaf_row_idx
            .iter()
            .map(|&ri| clean_dual(dual_rows[ri]))
            .collect();
        let node_duals = self
            .node_row_idx
            .iter()
            .map(|tree_idxs| {
                tree_idxs
                    .iter()
                    .map(|opt| opt.map(|ri| clean_dual(-dual_rows[ri])).unwrap_or(0.0))
                    .collect()
            })
            .collect();

        self.model = Some(Model::from(solved));
        Ok(RmpLpResult {
            objective,
            column_values,
            leaf_duals,
            node_duals,
        })
    }
}

fn price_best_column_exhaustive(
    trees: &[Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    seen: &FxHashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
) -> Option<(f64, Vec<u32>)> {
    let available = (1..=num_leaves as u32)
        .filter(|&label| !blocked_leaves[label as usize])
        .collect::<Vec<_>>();
    if available.is_empty() || available.len() >= u64::BITS as usize {
        return None;
    }

    let mut best: Option<(f64, Vec<u32>)> = None;
    let limit = 1u64 << available.len();
    for mask in 1..limit {
        let labels = labels_from_mask(mask, &available);
        if forbidden.contains(seen, &labels) {
            continue;
        }
        if !is_set_compatible_all(trees, &labels) {
            continue;
        }
        let score = column_score(&labels, trees, alpha, beta);
        match &best {
            Some((best_score, best_labels)) => {
                if score > *best_score + 1e-12
                    || ((score - *best_score).abs() <= 1e-12 && labels.len() > best_labels.len())
                {
                    best = Some((score, labels));
                }
            }
            None => best = Some((score, labels)),
        }
    }
    best
}

fn labels_from_mask(mask: u64, available: &[u32]) -> Vec<u32> {
    let mut labels = Vec::new();
    for (idx, &label) in available.iter().enumerate() {
        if (mask & (1u64 << idx)) != 0 {
            labels.push(label);
        }
    }
    labels
}

fn column_score(labels: &[u32], trees: &[Tree], alpha: &[f64], beta: &[Vec<f64>]) -> f64 {
    let leaf_sum = labels.iter().map(|&label| alpha[label as usize]).sum::<f64>();
    let beta_sum = if labels.len() >= 2 {
        trees.iter().enumerate().map(|(ti, tree)| {
            mark_component_nodes(tree, labels)
                .ones()
                .filter(|&node| !tree.is_leaf(node as u32))
                .map(|node| beta[ti][node])
                .sum::<f64>()
        }).sum::<f64>()
    } else {
        0.0
    };
    leaf_sum - beta_sum
}

fn support_is_integral_partition(columns: &[BpColumn], values: &[f64], num_leaves: usize) -> bool {
    let mut cover_count = vec![0usize; num_leaves + 1];
    for (ci, &value) in values.iter().enumerate() {
        if value <= 1.0e-9 {
            continue;
        }
        for &label in &columns[ci].labels {
            let leaf = label as usize;
            cover_count[leaf] += 1;
            if cover_count[leaf] > 1 {
                return false;
            }
        }
    }
    (1..=num_leaves).all(|leaf| cover_count[leaf] == 1)
}

fn reconstruct_components(columns: &[BpColumn], values: &[f64], instance: &Instance) -> Vec<Tree> {
    let n = instance.num_leaves;
    let mut covered = FixedBitSet::with_capacity(n as usize + 1);
    let mut components = Vec::new();
    for (ci, col) in columns.iter().enumerate() {
        if values.get(ci).copied().unwrap_or(0.0) < 0.5 {
            continue;
        }
        if col.labels.len() == 1 {
            covered.insert(col.labels[0] as usize);
            components.push(Tree::singleton(col.labels[0], n));
        } else {
            let leafset = make_leafset(&col.labels, n);
            covered.union_with(&leafset);
            components.push(Tree::component_from_leafset(
                &leafset,
                instance.reference_tree(),
                n,
            ));
        }
    }
    for label in 1..=n {
        if !covered.contains(label as usize) {
            components.push(Tree::singleton(label, n));
        }
    }
    components
}

fn select_branch_pair(columns: &[BpColumn], values: &[f64], num_leaves: usize) -> Option<LeafPair> {
    const ACTIVE_EPS: f64 = 1.0e-9;
    const FRACTIONAL_EPS: f64 = 1.0e-6;

    let stride = num_leaves + 1;
    let mut together = vec![0.0; stride * stride];
    let mut support_count = vec![0usize; stride * stride];

    for (ci, &value) in values.iter().enumerate() {
        if value <= ACTIVE_EPS {
            continue;
        }
        let labels = &columns[ci].labels;
        if labels.len() < 2 {
            continue;
        }
        for i in 0..labels.len() {
            let a = labels[i] as usize;
            let row = a * stride;
            for &label_b in &labels[(i + 1)..] {
                let b = label_b as usize;
                let idx = row + b;
                together[idx] += value;
                support_count[idx] += 1;
            }
        }
    }

    let mut best_pair = None;
    let mut best_balance = f64::NEG_INFINITY;
    let mut best_support = 0usize;
    for a in 1..=num_leaves {
        let row = a * stride;
        for b in (a + 1)..=num_leaves {
            let idx = row + b;
            let together_mass = together[idx];
            if together_mass <= FRACTIONAL_EPS || together_mass >= 1.0 - FRACTIONAL_EPS {
                continue;
            }
            let balance = 0.5 - (together_mass - 0.5).abs();
            let support = support_count[idx];
            if balance > best_balance + 1.0e-12
                || ((balance - best_balance).abs() <= 1.0e-12 && support > best_support)
            {
                best_pair = Some(LeafPair::new(a as u32, b as u32));
                best_balance = balance;
                best_support = support;
            }
        }
    }

    best_pair
}

fn select_branch_column(columns: &[BpColumn], values: &[f64], num_leaves: usize) -> Option<usize> {
    let mut seen = vec![false; num_leaves + 1];
    let mut duplicated = vec![false; num_leaves + 1];
    for (ci, &value) in values.iter().enumerate() {
        if value <= 1.0e-9 || value >= 1.0 - 1.0e-9 {
            continue;
        }
        for &label in &columns[ci].labels {
            let leaf = label as usize;
            if seen[leaf] {
                duplicated[leaf] = true;
            } else {
                seen[leaf] = true;
            }
        }
    }

    let mut best_idx = None;
    let mut best_score = f64::NEG_INFINITY;
    for (ci, &value) in values.iter().enumerate() {
        if value <= 1.0e-9 || value >= 1.0 - 1.0e-9 {
            continue;
        }
        if !columns[ci].labels.iter().any(|&label| duplicated[label as usize]) {
            continue;
        }
        if columns[ci].labels.len() <= 1 || columns[ci].total_internal_count == 0 {
            continue;
        }
        let score = columns[ci].labels.len() as f64 / columns[ci].total_internal_count as f64;
        if score > best_score {
            best_score = score;
            best_idx = Some(ci);
        }
    }
    best_idx
}

fn labels_disjoint(a: &[u32], b: &[u32]) -> bool {
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => return false,
        }
    }
    true
}

fn labels_contain(labels: &[u32], target: u32) -> bool {
    labels.binary_search(&target).is_ok()
}

fn labels_satisfy_pair_constraints(
    labels: &[u32],
    must_link_pairs: &[LeafPair],
    cannot_link_pairs: &[LeafPair],
) -> bool {
    for &pair in must_link_pairs {
        if labels_contain(labels, pair.a) != labels_contain(labels, pair.b) {
            return false;
        }
    }
    for &pair in cannot_link_pairs {
        if labels_contain(labels, pair.a) && labels_contain(labels, pair.b) {
            return false;
        }
    }
    true
}

fn pricing_candidate_allowed(
    labels: &[u32],
    seen: &FxHashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
    must_link_pairs: &[LeafPair],
    cannot_link_pairs: &[LeafPair],
) -> bool {
    !forbidden.contains(seen, labels)
        && labels_satisfy_pair_constraints(labels, must_link_pairs, cannot_link_pairs)
}

fn node_branchings_self_consistent(columns: &[BpColumn], node: &BpNode) -> bool {
    if node
        .must_link_pairs
        .iter()
        .any(|pair| node.cannot_link_pairs.contains(pair))
    {
        return false;
    }
    if node
        .fixed_to_one
        .iter()
        .any(|ci| node.fixed_to_zero.contains(ci))
    {
        return false;
    }
    for (idx, &forced_ci) in node.fixed_to_one.iter().enumerate() {
        if !labels_satisfy_pair_constraints(
            &columns[forced_ci].labels,
            &node.must_link_pairs,
            &node.cannot_link_pairs,
        ) {
            return false;
        }
        for &other_ci in &node.fixed_to_one[(idx + 1)..] {
            if !labels_disjoint(&columns[forced_ci].labels, &columns[other_ci].labels) {
                return false;
            }
        }
    }
    true
}

fn column_respects_branchings(
    columns: &[BpColumn],
    ci: usize,
    fixed_to_one: &[usize],
    fixed_to_zero: &[usize],
    must_link_pairs: &[LeafPair],
    cannot_link_pairs: &[LeafPair],
) -> bool {
    if fixed_to_zero
        .iter()
        .any(|&blocked_ci| columns[blocked_ci].labels == columns[ci].labels)
    {
        return false;
    }
    for &forced_ci in fixed_to_one {
        if forced_ci == ci {
            continue;
        }
        if !labels_disjoint(&columns[forced_ci].labels, &columns[ci].labels) {
            return false;
        }
    }
    labels_satisfy_pair_constraints(
        &columns[ci].labels,
        must_link_pairs,
        cannot_link_pairs,
    )
}

fn make_leafset(labels: &[u32], num_leaves: u32) -> FixedBitSet {
    let mut bits = FixedBitSet::with_capacity(num_leaves as usize + 1);
    for &label in labels {
        bits.insert(label as usize);
    }
    bits
}

fn clean_dual(value: f64) -> f64 {
    if value.abs() <= 1.0e-9 { 0.0 } else { value }
}

fn make_bp_column(mut labels: Vec<u32>, trees: &[Tree]) -> BpColumn {
    labels.sort_unstable();
    labels.dedup();
    let covered_internal_nodes = if labels.len() >= 2 {
        trees.iter()
            .map(|tree| {
                mark_component_nodes(tree, &labels)
                    .ones()
                    .filter(|&node| !tree.is_leaf(node as u32))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    } else {
        trees.iter().map(|_| Vec::new()).collect::<Vec<_>>()
    };
    let total_internal_count = covered_internal_nodes.iter().map(|nodes| nodes.len()).sum();
    BpColumn {
        labels,
        covered_internal_nodes,
        total_internal_count,
    }
}

fn mark_component_nodes(tree: &Tree, labels: &[u32]) -> FixedBitSet {
    let mut bits = FixedBitSet::with_capacity(tree.num_nodes());
    if labels.is_empty() {
        return bits;
    }
    let mut lca_node = tree.node_by_label(labels[0]);
    for &label in &labels[1..] {
        lca_node = tree.nearest_common_ancestor(lca_node, tree.node_by_label(label));
    }
    for &label in labels {
        let mut cur = tree.node_by_label(label);
        loop {
            bits.insert(cur as usize);
            if cur == lca_node {
                break;
            }
            cur = tree.parent[cur as usize];
        }
    }
    bits
}

fn triplet_topology(tree: &Tree, x: u32, y: u32, z: u32) -> u8 {
    let nx = tree.node_by_label(x);
    let ny = tree.node_by_label(y);
    let nz = tree.node_by_label(z);
    let lxy = tree.nearest_common_ancestor(nx, ny);
    let lxz = tree.nearest_common_ancestor(nx, nz);
    let lyz = tree.nearest_common_ancestor(ny, nz);
    let dxy = tree.depth[lxy as usize];
    let dxz = tree.depth[lxz as usize];
    let dyz = tree.depth[lyz as usize];
    if dxy > dxz && dxy > dyz {
        0
    } else if dxz > dxy && dxz > dyz {
        1
    } else {
        2
    }
}

fn is_set_compatible_all(trees: &[Tree], labels: &[u32]) -> bool {
    if labels.len() <= 2 {
        return true;
    }
    for i in 0..labels.len() {
        for j in (i + 1)..labels.len() {
            for k in (j + 1)..labels.len() {
                let a = labels[i];
                let b = labels[j];
                let c = labels[k];
                let topo0 = triplet_topology(&trees[0], a, b, c);
                if trees[1..]
                    .iter()
                    .any(|tree| triplet_topology(tree, a, b, c) != topo0)
                {
                    return false;
                }
            }
        }
    }
    true
}
