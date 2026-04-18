//! Experimental multi-tree Branch & Price solver.
//!
//! Current shape:
//! - generic master problem over an arbitrary number of rooted trees
//! - exact multi-tree pricing via memoized top-down `M`/`V` recurrences
//! - exhaustive subset oracle retained only for validation on tiny instances
//! - generic branch-and-bound on fractional columns

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use fixedbitset::FixedBitSet;
use fxhash::FxHashMap;
use highs::{Col, ColProblem, HighsModelStatus, Model, Row, RowProblem, Sense};
use klados_core::lower_bound::maf_bounds;
use klados_core::{Instance, SolverStats, Tree};

use crate::cluster_reduction::{self, ClusterReductionResult};
use crate::kernelize::{self, KernelizeConfig};
use crate::maf_branch_price::MafBranchPriceSolver;
use crate::ExactSolver;

const ORACLE_ENUM_LEAVES: usize = 24;
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum PricingBackend {
    Native,
    PairDp,
    IlpGlobal,
    IlpHybrid,
}

fn pricing_backend() -> PricingBackend {
    use std::sync::OnceLock;
    static CACHED: OnceLock<PricingBackend> = OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("KLADOS_BP_MULTI_PRICER").ok().as_deref() {
            Some("native") => PricingBackend::Native,
            Some("pairdp") => PricingBackend::PairDp,
            Some("ilp-global") => PricingBackend::IlpGlobal,
            Some("ilp") | Some("ilp-hybrid") => PricingBackend::IlpHybrid,
            _ => PricingBackend::PairDp,
        }
    })
}

const ILP_REGION_TOP_POS_LEAVES: usize = 24;
const ILP_REGION_MAX_LEAVES: usize = 40;
const ILP_REGION_MAX_CANDIDATES: usize = 32;

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
        if instance.num_trees() == 2 {
            let mut solver = MafBranchPriceSolver::new();
            let result = solver.solve(instance);
            self.stats = solver.stats().clone();
            return result;
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
    seen: HashSet<Vec<u32>>,
    best_ub: usize,
    best_solution: Option<Vec<f64>>,
    nodes_explored: usize,
    cg_iterations_total: usize,
    oracle_matches: usize,
    oracle_mismatches: usize,
}

#[derive(Clone)]
struct BpNode {
    fixed_to_one: Vec<usize>,
    fixed_to_zero: Vec<usize>,
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

    fn contains(&self, seen: &HashSet<Vec<u32>>, labels: &[u32]) -> bool {
        seen.contains(labels)
            || self
                .fixed_zero_labels
                .iter()
                .any(|blocked| blocked.as_slice() == labels)
    }
}

enum NodeResult {
    Integral(usize, Vec<f64>),
    Branch { lp_obj: f64, branch_col: usize },
    Pruned,
}

thread_local! {
    static CALL_DEPTH: Cell<usize> = Cell::new(0);
}

struct CallDepthGuard(usize);
impl Drop for CallDepthGuard {
    fn drop(&mut self) {
        CALL_DEPTH.set(self.0);
    }
}

fn solve_branch_price_multi(instance: &Instance, stats: &mut SolverStats) -> Option<Vec<Tree>> {
    let depth = CALL_DEPTH.get();
    CALL_DEPTH.set(depth + 1);
    let _depth_guard = CallDepthGuard(depth);
    let t_total = Instant::now();

    let config = KernelizeConfig::default();
    let kern = kernelize::kernelize_best(instance, &config);
    let reduced = &kern.instance;

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

    match cluster_reduction::try_cluster_reduction(reduced, &mut |subinstance| {
        solve_branch_price_multi(subinstance, &mut SolverStats::default())
    })? {
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

    let bounds = maf_bounds(trees, reduced.num_leaves);
    let mut columns: Vec<BpColumn> = (1..=n as u32)
        .map(|label| make_bp_column(vec![label], trees))
        .collect();
    let mut seen: HashSet<Vec<u32>> = columns.iter().map(|c| c.labels.clone()).collect();

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
    };

    let root = BpNode {
        fixed_to_one: vec![],
        fixed_to_zero: vec![],
        depth: 0,
    };

    let mut pricer_ws = MultiPricerWorkspace::new(trees, n);
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        match solve_bp_node(&mut state, &node, trees, n, &mut pricer_ws) {
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
            NodeResult::Branch { lp_obj, branch_col } => {
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
    eprintln!(
        "[maf-bp-multi] optimal: {} components, {} B&B nodes, {} CG iterations, oracle_matches={}, oracle_mismatches={}, {:.1}ms total",
        components.len(),
        state.nodes_explored,
        state.cg_iterations_total,
        state.oracle_matches,
        state.oracle_mismatches,
        t_total.elapsed().as_secs_f64() * 1000.0,
    );
    Some(components)
}

fn solve_bp_node(
    state: &mut BpState,
    node: &BpNode,
    trees: &[Tree],
    num_leaves: usize,
    pricer_ws: &mut MultiPricerWorkspace,
) -> NodeResult {
    state.nodes_explored += 1;
    if node.fixed_to_one.len() >= state.best_ub {
        return NodeResult::Pruned;
    }

    let mut blocked_leaves = vec![false; num_leaves + 1];
    for &forced_ci in &node.fixed_to_one {
        for &label in &state.columns[forced_ci].labels {
            blocked_leaves[label as usize] = true;
        }
    }

    let forbidden = ForbiddenColumns::new(state, node);

    let mut node_rmp = match NodeRmp::build(
        &state.columns,
        trees,
        num_leaves,
        &node.fixed_to_one,
        &node.fixed_to_zero,
    ) {
        Ok(rmp) => rmp,
        Err(_) => return NodeResult::Pruned,
    };

    let mut final_lp: Option<RmpLpResult> = None;
    loop {
        let lp = match node_rmp.solve(state.columns.len()) {
            Ok(lp) => lp,
            Err(_) => return NodeResult::Pruned,
        };
        let alpha = lp.leaf_duals.clone();
        let beta = lp.node_duals.clone();
        let priced = match pricing_backend() {
            PricingBackend::Native => price_best_new_exact_column(
                pricer_ws,
                trees,
                num_leaves,
                &alpha,
                &beta,
                &blocked_leaves,
                &state.seen,
                &forbidden,
            ),
            PricingBackend::PairDp => price_best_new_pairdp_column(
                pricer_ws,
                trees,
                num_leaves,
                &alpha,
                &beta,
                &blocked_leaves,
                &state.seen,
                &forbidden,
            ),
            PricingBackend::IlpGlobal | PricingBackend::IlpHybrid => price_best_new_ilp_column(
                pricer_ws,
                trees,
                num_leaves,
                &alpha,
                &beta,
                &blocked_leaves,
                &state.seen,
                &forbidden,
            ),
        };
        if num_leaves <= ORACLE_ENUM_LEAVES {
            let oracle_priced = price_best_column_exhaustive(
                trees,
                num_leaves,
                &alpha,
                &beta,
                &blocked_leaves,
                &state.seen,
                &forbidden,
            );
            match (&priced, &oracle_priced) {
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

        match priced {
            Some((score, labels)) if score > 1.0 + 1e-8 => {
                let inserted = state.seen.insert(labels.clone());
                if !inserted {
                    final_lp = Some(lp);
                    break;
                }
                let new_ci = state.columns.len();
                state.columns.push(make_bp_column(labels, trees));
                node_rmp.add_column(new_ci, &state.columns[new_ci], trees);
                if let Some(best_solution) = state.best_solution.as_mut() {
                    best_solution.push(0.0);
                }
                state.cg_iterations_total += 1;
            }
            _ => {
                final_lp = Some(lp);
                break;
            }
        }
    }

    let lp = match final_lp {
        Some(lp) => lp,
        None => match node_rmp.solve(state.columns.len()) {
            Ok(lp) => lp,
            Err(_) => return NodeResult::Pruned,
        },
    };

    let lp_bound = (lp.objective - 1e-6).ceil() as usize;
    if lp_bound >= state.best_ub {
        return NodeResult::Pruned;
    }

    if support_is_integral_partition(&state.columns, &lp.column_values, num_leaves) {
        let obj = lp.column_values.iter().filter(|&&v| v > 1.0e-9).count();
        return NodeResult::Integral(obj, lp.column_values);
    }

    let branch_col = select_branch_column(&state.columns, &lp.column_values, num_leaves);
    match branch_col {
        Some(col_idx) => NodeResult::Branch {
            lp_obj: lp.objective,
            branch_col: col_idx,
        },
        None => NodeResult::Pruned,
    }
}

#[derive(Clone)]
struct PricedColumn {
    score: f64,
    labels: Vec<u32>,
}

impl PricedColumn {
    fn none() -> Self {
        Self {
            score: NEG_INF,
            labels: Vec::new(),
        }
    }

    fn is_some(&self) -> bool {
        self.score > NEG_INF / 2.0 && !self.labels.is_empty()
    }
}

fn better_column(lhs: &PricedColumn, rhs: &PricedColumn) -> bool {
    lhs.score > rhs.score + 1e-12
        || ((lhs.score - rhs.score).abs() <= 1e-12 && lhs.labels.len() > rhs.labels.len())
}

#[derive(Clone)]
struct PricingPrefixNode {
    prefix_membership: Vec<bool>,
    score: f64,
    labels: Vec<u32>,
}

struct MultiPricerWorkspace {
    descendant_leaves: Vec<Vec<FixedBitSet>>,
    memo_m: FxHashMap<Vec<u32>, f64>,
    memo_v: FxHashMap<Vec<u32>, f64>,
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
                    let left_bits = leaves[l as usize].clone();
                    let right_bits = leaves[r as usize].clone();
                    leaves[node as usize].union_with(&left_bits);
                    leaves[node as usize].union_with(&right_bits);
                }
            }
            descendant_leaves.push(leaves);
        }
        Self {
            descendant_leaves,
            memo_m: FxHashMap::default(),
            memo_v: FxHashMap::default(),
        }
    }

    fn reset(&mut self) {
        self.memo_m.clear();
        self.memo_v.clear();
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
            self.memo_m.insert(tuple.to_vec(), NEG_INF);
            return NEG_INF;
        }
        let mut best = best_leaf;
        if best >= ub - 1e-12 {
            self.memo_m.insert(tuple.to_vec(), best);
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
                        self.memo_m.insert(tuple.to_vec(), best);
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
                        self.memo_m.insert(tuple.to_vec(), best);
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

        self.memo_m.insert(tuple.to_vec(), best);
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
            self.memo_v.insert(tuple.to_vec(), NEG_INF);
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
            self.memo_v.insert(tuple.to_vec(), best);
            return best;
        }

        let mut best = NEG_INF;
        if ub <= 0.0 {
            self.memo_v.insert(tuple.to_vec(), best);
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
                        self.memo_v.insert(tuple.to_vec(), best);
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
                        self.memo_v.insert(tuple.to_vec(), best);
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
                    self.memo_v.insert(tuple.to_vec(), best);
                    return best;
                }
            }
        }

        self.memo_v.insert(tuple.to_vec(), best);
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

fn merge_sorted_labels(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let mut merged = Vec::with_capacity(lhs.len() + rhs.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < lhs.len() && j < rhs.len() {
        match lhs[i].cmp(&rhs[j]) {
            Ordering::Less => {
                merged.push(lhs[i]);
                i += 1;
            }
            Ordering::Greater => {
                merged.push(rhs[j]);
                j += 1;
            }
            Ordering::Equal => {
                merged.push(lhs[i]);
                i += 1;
                j += 1;
            }
        }
    }
    merged.extend_from_slice(&lhs[i..]);
    merged.extend_from_slice(&rhs[j..]);
    merged
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
    descendant_leaves: &'a [Vec<FixedBitSet>],
    active_labels: Vec<u32>,
    active_nodes: Vec<Vec<u32>>,
    alpha: &'a [f64],
    beta: &'a [Vec<f64>],
    prefix_beta: Vec<Vec<f64>>,
    roots: Vec<Vec<u32>>,
    side_child: Vec<Vec<u32>>,
    memo_pair: Vec<Option<f64>>,
    memo_side: Vec<Option<PairDpSideState>>,
    solving_pair: Vec<bool>,
    solving_side: Vec<bool>,
}

impl<'a> PairDpPricer<'a> {
    fn new(
        descendant_leaves: &'a [Vec<FixedBitSet>],
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

        let mut active_nodes = Vec::with_capacity(trees.len());
        let mut prefix_beta = Vec::with_capacity(trees.len());
        let mut roots = Vec::with_capacity(trees.len());
        let mut side_child = Vec::with_capacity(trees.len());

        for (ti, tree) in trees.iter().enumerate() {
            let nodes = active_labels
                .iter()
                .map(|&label| tree.node_by_label(label))
                .collect::<Vec<_>>();
            active_nodes.push(nodes.clone());

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

            let mut tree_roots = vec![0u32; pair_count];
            let mut tree_side_child = vec![0u32; pair_count];
            for a in 0..p {
                for b in 0..p {
                    let idx = a * p + b;
                    if a == b {
                        tree_roots[idx] = nodes[a];
                        tree_side_child[idx] = nodes[a];
                        continue;
                    }
                    let root = tree.nearest_common_ancestor(nodes[a], nodes[b]);
                    tree_roots[idx] = root;
                    let child = if tree.is_leaf(root) {
                        root
                    } else {
                        let (left, right) = tree.children_pair(root);
                        if descendant_leaves[ti][left as usize]
                            .contains(active_labels[a] as usize)
                        {
                            left
                        } else {
                            right
                        }
                    };
                    tree_side_child[idx] = child;
                }
            }
            roots.push(tree_roots);
            side_child.push(tree_side_child);
        }

        Self {
            trees,
            descendant_leaves,
            active_labels,
            active_nodes,
            alpha,
            beta,
            prefix_beta,
            roots,
            side_child,
            memo_pair: vec![None; pair_count],
            memo_side: vec![None; pair_count],
            solving_pair: vec![false; pair_count],
            solving_side: vec![false; pair_count],
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
            for b in 0..p {
                if a == b {
                    continue;
                }
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
                self.active_labels.len() * self.active_labels.len()
            );
        }

        (best_score > NEG_INF / 2.0).then_some((best_score, best_labels))
    }

    fn pair_idx(&self, a: usize, b: usize) -> usize {
        a * self.active_labels.len() + b
    }

    fn solve_pair(&mut self, a: usize, b: usize) -> f64 {
        debug_assert!(a != b);
        let idx = self.pair_idx(a, b);
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
            self.descendant_leaves[ti][self.side_child[ti][idx] as usize].contains(label_c)
        })
    }
}

fn solve_best_pairdp_column(
    descendant_leaves: &[Vec<FixedBitSet>],
    trees: &[Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
) -> Option<(f64, Vec<u32>)> {
    let mut pricer = PairDpPricer::new(
        descendant_leaves,
        trees,
        num_leaves,
        alpha,
        beta,
        blocked_leaves,
    );
    pricer.solve_best()
}

fn price_best_new_pairdp_column(
    pricer_ws: &mut MultiPricerWorkspace,
    trees: &[Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    seen: &HashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
) -> Option<(f64, Vec<u32>)> {
    let base = solve_best_pairdp_column(
        &pricer_ws.descendant_leaves,
        trees,
        num_leaves,
        alpha,
        beta,
        blocked_leaves,
    )?;
    let (base_score, base_labels) = base;
    if !forbidden.contains(seen, &base_labels) {
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
        &free_labels,
        &[],
        required_bonus,
        &mut incumbent,
    );
    incumbent
}

fn search_pricing_prefix_dfs_pairdp(
    pricer_ws: &mut MultiPricerWorkspace,
    trees: &[Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    seen: &HashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
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

    if !forbidden.contains(seen, &candidate.labels) {
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
        &pricer_ws.descendant_leaves,
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
    seen: &HashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
) -> Option<(f64, Vec<u32>)> {
    let base = pricer_ws.solve_best(trees, alpha, beta, blocked_leaves);
    let Some((base_score, base_labels)) = base else {
        return None;
    };
    if !forbidden.contains(seen, &base_labels) {
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
        &free_labels,
        &[],
        required_bonus,
        &mut incumbent,
    );
    incumbent
}

fn price_best_new_ilp_column(
    pricer_ws: &mut MultiPricerWorkspace,
    trees: &[Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    seen: &HashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
) -> Option<(f64, Vec<u32>)> {
    if pricing_backend() == PricingBackend::IlpHybrid {
        if let Some(priced) = price_best_pair_region_ilp_column(
            &pricer_ws.descendant_leaves,
            trees,
            num_leaves,
            alpha,
            beta,
            blocked_leaves,
            seen,
            forbidden,
        ) {
            return Some(priced);
        }
    }

    price_best_column_ilp(
        &pricer_ws.descendant_leaves,
        trees,
        num_leaves,
        alpha,
        beta,
        blocked_leaves,
        seen,
        forbidden,
    )
}

#[derive(Clone)]
struct PairRegionCandidate {
    upper_bound: f64,
    labels: Vec<u32>,
}

fn price_best_pair_region_ilp_column(
    descendant_leaves: &[Vec<FixedBitSet>],
    trees: &[Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    seen: &HashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
) -> Option<(f64, Vec<u32>)> {
    let mut positive_labels = (1..=num_leaves as u32)
        .filter(|&label| !blocked_leaves[label as usize] && alpha[label as usize] > 1.0e-12)
        .collect::<Vec<_>>();
    if positive_labels.is_empty() {
        return None;
    }
    positive_labels.sort_unstable_by(|&lhs, &rhs| {
        alpha[rhs as usize].total_cmp(&alpha[lhs as usize])
    });

    for &label in &positive_labels {
        let singleton = vec![label];
        if !forbidden.contains(seen, &singleton) && alpha[label as usize] > 1.0 + 1.0e-8 {
            return Some((alpha[label as usize], singleton));
        }
    }

    if positive_labels.len() < 2 {
        return None;
    }
    positive_labels.truncate(ILP_REGION_TOP_POS_LEAVES);

    let mut dedup: FxHashMap<Vec<u32>, f64> = FxHashMap::default();
    for i in 0..positive_labels.len() {
        for j in (i + 1)..positive_labels.len() {
            let a = positive_labels[i];
            let b = positive_labels[j];
            let mut region = {
                let lca = trees[0]
                    .nearest_common_ancestor(trees[0].node_by_label(a), trees[0].node_by_label(b));
                descendant_leaves[0][lca as usize].clone()
            };
            for ti in 1..trees.len() {
                let lca = trees[ti]
                    .nearest_common_ancestor(trees[ti].node_by_label(a), trees[ti].node_by_label(b));
                region.intersect_with(&descendant_leaves[ti][lca as usize]);
            }
            let labels = region
                .ones()
                .filter(|&leaf| leaf > 0 && !blocked_leaves[leaf])
                .map(|leaf| leaf as u32)
                .collect::<Vec<_>>();
            if labels.len() < 2 || labels.len() > ILP_REGION_MAX_LEAVES {
                continue;
            }
            let upper_bound = labels
                .iter()
                .map(|&label| alpha[label as usize].max(0.0))
                .sum::<f64>();
            if upper_bound <= 1.0 + 1.0e-8 {
                continue;
            }
            dedup
                .entry(labels)
                .and_modify(|ub| *ub = ub.max(upper_bound))
                .or_insert(upper_bound);
        }
    }

    if dedup.is_empty() {
        return None;
    }

    let mut candidates = dedup
        .into_iter()
        .map(|(labels, upper_bound)| PairRegionCandidate { upper_bound, labels })
        .collect::<Vec<_>>();
    candidates.sort_unstable_by(|lhs, rhs| rhs.upper_bound.total_cmp(&lhs.upper_bound));
    if candidates.len() > ILP_REGION_MAX_CANDIDATES {
        candidates.truncate(ILP_REGION_MAX_CANDIDATES);
    }

    if exact_pricer_profile() {
        eprintln!(
            "[maf-bp-multi] pair-region heuristic: positive={} candidates={}",
            positive_labels.len(),
            candidates.len()
        );
    }

    for candidate in candidates {
        let mut allowed = vec![false; num_leaves + 1];
        for &label in &candidate.labels {
            allowed[label as usize] = true;
        }
        let mut blocked = blocked_leaves.to_vec();
        for label in 1..=num_leaves {
            if !allowed[label] {
                blocked[label] = true;
            }
        }
        if let Some((score, labels)) = price_best_column_ilp(
            descendant_leaves,
            trees,
            num_leaves,
            alpha,
            beta,
            &blocked,
            seen,
            forbidden,
        ) {
            if score > 1.0 + 1.0e-8 {
                if exact_pricer_profile() {
                    eprintln!(
                        "[maf-bp-multi] pair-region hit: score={:.6} size={}",
                        score,
                        labels.len()
                    );
                }
                return Some((score, labels));
            }
        }
    }

    None
}

fn price_best_column_ilp(
    descendant_leaves: &[Vec<FixedBitSet>],
    trees: &[Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    seen: &HashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
) -> Option<(f64, Vec<u32>)> {
    let free_labels = (1..=num_leaves as u32)
        .filter(|&label| !blocked_leaves[label as usize])
        .collect::<Vec<_>>();
    if free_labels.is_empty() {
        return None;
    }

    let conflict_triples = find_conflict_triples_multi(trees, &free_labels);
    let mut pb = RowProblem::default();

    let mut x_vars = vec![None; num_leaves + 1];
    for &label in &free_labels {
        let col = pb.add_integer_column(alpha[label as usize], 0.0..=1.0);
        x_vars[label as usize] = Some(col);
    }

    pb.add_row(
        1.0..,
        free_labels
            .iter()
            .filter_map(|&label| x_vars[label as usize].map(|col| (col, 1.0)))
            .collect::<Vec<_>>(),
    );

    let mut used_vars: Vec<Vec<Option<Col>>> = Vec::with_capacity(trees.len());
    for (ti, tree) in trees.iter().enumerate() {
        let mut tree_used_vars = vec![None; tree.num_nodes()];
        for node in 0..tree.num_nodes() as u32 {
            if tree.is_leaf(node) {
                continue;
            }
            let y = pb.add_integer_column(-beta[ti][node as usize], 0.0..=1.0);
            let (left, right) = tree.children_pair(node);
            let left_labels = descendant_leaves[ti][left as usize].ones().collect::<Vec<_>>();
            let right_labels = descendant_leaves[ti][right as usize].ones().collect::<Vec<_>>();
            let p_left = add_presence_var(&mut pb, &x_vars, &left_labels);
            let p_right = add_presence_var(&mut pb, &x_vars, &right_labels);
            if tree.is_root(node) {
                // Root is used iff both child directions contain selected leaves.
                pb.add_row(..=0.0, [(y, 1.0), (p_left, -1.0)]);
                pb.add_row(..=0.0, [(y, 1.0), (p_right, -1.0)]);
                pb.add_row(-1.0.., [(y, 1.0), (p_left, -1.0), (p_right, -1.0)]);
            } else {
                // A non-root Steiner node is used iff at least two of its three
                // incident directions are active: left, right, or upward/outside.
                let outside_labels = free_labels
                    .iter()
                    .copied()
                    .filter(|&label| !descendant_leaves[ti][node as usize].contains(label as usize))
                    .map(|label| label as usize)
                    .collect::<Vec<_>>();
                let p_up = add_presence_var(&mut pb, &x_vars, &outside_labels);

                pb.add_row(..=0.0, [(y, 1.0), (p_left, -1.0), (p_right, -1.0)]);
                pb.add_row(..=0.0, [(y, 1.0), (p_left, -1.0), (p_up, -1.0)]);
                pb.add_row(..=0.0, [(y, 1.0), (p_right, -1.0), (p_up, -1.0)]);

                pb.add_row(-1.0.., [(y, 1.0), (p_left, -1.0), (p_right, -1.0)]);
                pb.add_row(-1.0.., [(y, 1.0), (p_left, -1.0), (p_up, -1.0)]);
                pb.add_row(-1.0.., [(y, 1.0), (p_right, -1.0), (p_up, -1.0)]);
            }
            tree_used_vars[node as usize] = Some(y);
        }
        used_vars.push(tree_used_vars);
    }

    for &(a, b, c) in &conflict_triples {
        let xa = x_vars[a as usize].expect("free leaf var exists");
        let xb = x_vars[b as usize].expect("free leaf var exists");
        let xc = x_vars[c as usize].expect("free leaf var exists");
        pb.add_row(..=2.0, [(xa, 1.0), (xb, 1.0), (xc, 1.0)]);
    }

    for blocked in forbidden
        .fixed_zero_labels
        .iter()
        .chain(seen.iter())
    {
        if blocked.iter().any(|&label| blocked_leaves[label as usize]) {
            continue;
        }
        let blocked_set = blocked.iter().copied().collect::<HashSet<_>>();
        let mut coeffs = Vec::with_capacity(free_labels.len());
        for &label in &free_labels {
            let x = x_vars[label as usize].expect("free leaf var exists");
            if blocked_set.contains(&label) {
                coeffs.push((x, 1.0));
            } else {
                coeffs.push((x, -1.0));
            }
        }
        if blocked.len() > free_labels.len() {
            continue;
        }
        if blocked.iter().all(|&label| x_vars[label as usize].is_some()) {
            pb.add_row(..=(blocked.len() as f64 - 1.0), coeffs);
        }
    }

    let mut model = pb.optimise(Sense::Maximise);
    model.make_quiet();
    model.set_option("threads", 1_i32);
    model.set_option("presolve", "on");

    let solved = model.solve();
    if solved.status() != HighsModelStatus::Optimal {
        return None;
    }

    let values = solved.get_solution().columns().to_vec();
    let mut labels = Vec::new();
    let mut x_index = 0usize;
    for &label in &free_labels {
        if values[x_index] > 0.5 {
            labels.push(label);
        }
        x_index += 1;
    }

    if labels.is_empty() || forbidden.contains(seen, &labels) || !is_set_compatible_all(trees, &labels)
    {
        return None;
    }
    Some((column_score(&labels, trees, alpha, beta), labels))
}

fn find_conflict_triples_multi(trees: &[Tree], labels: &[u32]) -> Vec<(u32, u32, u32)> {
    let mut conflicts = Vec::new();
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
                    conflicts.push((a, b, c));
                }
            }
        }
    }
    conflicts
}

fn add_presence_var(pb: &mut RowProblem, x_vars: &[Option<Col>], labels: &[usize]) -> Col {
    let p = pb.add_integer_column(0.0, 0.0..=1.0);
    for &leaf in labels {
        if let Some(x) = x_vars[leaf] {
            pb.add_row(0.0.., [(p, 1.0), (x, -1.0)]);
        }
    }
    let mut coeffs = Vec::with_capacity(labels.len() + 1);
    coeffs.push((p, 1.0));
    for &leaf in labels {
        if let Some(x) = x_vars[leaf] {
            coeffs.push((x, -1.0));
        }
    }
    pb.add_row(..=0.0, coeffs);
    p
}

fn search_pricing_prefix_dfs(
    pricer_ws: &mut MultiPricerWorkspace,
    trees: &[Tree],
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    seen: &HashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
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

    if !forbidden.contains(seen, &candidate.labels) {
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

struct NodeRmp {
    model: Option<Model>,
    active_global_cols: Vec<usize>,
    leaf_rows: Vec<Row>,
    leaf_row_idx: Vec<usize>,
    node_rows: Vec<Vec<Option<Row>>>,
    node_row_idx: Vec<Vec<Option<usize>>>,
}

impl NodeRmp {
    fn build(
        columns: &[BpColumn],
        trees: &[Tree],
        num_leaves: usize,
        fixed_to_one: &[usize],
        fixed_to_zero: &[usize],
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
            active_global_cols: Vec::new(),
            leaf_rows,
            leaf_row_idx,
            node_rows,
            node_row_idx,
        };
        for (ci, col) in columns.iter().enumerate() {
            if column_respects_branchings(columns, ci, fixed_to_one, fixed_to_zero) {
                rmp.add_column(ci, col, trees);
            }
        }
        Ok(rmp)
    }

    fn add_column(&mut self, global_ci: usize, col: &BpColumn, trees: &[Tree]) {
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
            .expect("node RMP model present")
            .add_col(1.0, 0.0.., rows);
        self.active_global_cols.push(global_ci);
    }

    fn solve(&mut self, total_columns: usize) -> Result<RmpLpResult, String> {
        let solved = self.model.take().expect("node RMP model present").solve();
        let status = solved.status();
        if status != HighsModelStatus::Optimal {
            self.model = Some(Model::from(solved));
            return Err(format!("LP status: {:?}", status));
        }
        let solution = solved.get_solution();
        let mut column_values = vec![0.0; total_columns];
        for (local_idx, &global_ci) in self.active_global_cols.iter().enumerate() {
            column_values[global_ci] = solution.columns()[local_idx];
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
    seen: &HashSet<Vec<u32>>,
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

fn column_respects_branchings(
    columns: &[BpColumn],
    ci: usize,
    fixed_to_one: &[usize],
    fixed_to_zero: &[usize],
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
    true
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

fn make_bp_column(labels: Vec<u32>, trees: &[Tree]) -> BpColumn {
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
