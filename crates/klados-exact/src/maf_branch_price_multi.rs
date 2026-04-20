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
use highs::{ColProblem, HighsModelStatus, Model};
use klados_core::lower_bound::{
    greedy_multi_tree_partition, greedy_multi_tree_ub_seeded, pairwise_refine_ub,
};
use klados_core::{Instance, SolverStats, Tree};

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

fn compute_local_bounds(trees: &[Tree], num_leaves: u32) -> LocalBounds {
    if trees.len() <= 1 {
        return LocalBounds { best_partition: None };
    }

    let m = trees.len();
    let n = num_leaves as usize;

    let best_partition = if m == 2 {
        None
    } else {
        let (ref_limit, seed_limit, run_pairwise) = if m >= 20 || n >= 200 {
            (4usize, 2u64, false)
        } else if m >= 12 || n >= 140 {
            (6usize, 3u64, true)
        } else {
            (m, 5u64, true)
        };
        let ref_indices = sampled_reference_indices(m, ref_limit.min(m));
        let mut best_multi_ub = n;
        let mut best_ref = 0usize;
        let mut best_seed = 0u64;
        for ref_idx in ref_indices {
            for seed in 0..seed_limit {
                let ub = greedy_multi_tree_ub_seeded(trees, ref_idx, seed);
                if ub < best_multi_ub {
                    best_multi_ub = ub;
                    best_ref = ref_idx;
                    best_seed = seed;
                }
            }
        }
        if run_pairwise {
            let (pr_ub, pr_partition) = pairwise_refine_ub(trees, n);
            if pr_ub < best_multi_ub {
                Some(pr_partition)
            } else {
                let (_, partition) = greedy_multi_tree_partition(trees, best_ref, best_seed);
                Some(partition)
            }
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

const NEG_INF: f64 = -1.0e100;
const PAIRDP_BATCH_SIZE: usize = 64;

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

struct ColumnBuildScratch {
    marks: Vec<Vec<u32>>,
    epochs: Vec<u32>,
}

impl ColumnBuildScratch {
    fn new(trees: &[Tree]) -> Self {
        Self {
            marks: trees.iter().map(|tree| vec![0; tree.num_nodes()]).collect(),
            epochs: vec![1; trees.len()],
        }
    }

    fn next_stamp(&mut self, ti: usize) -> u32 {
        let stamp = self.epochs[ti];
        if stamp == u32::MAX {
            self.marks[ti].fill(0);
            self.epochs[ti] = 2;
            1
        } else {
            self.epochs[ti] += 1;
            stamp
        }
    }

    fn build_column(&mut self, mut labels: Vec<u32>, trees: &[Tree]) -> BpColumn {
        labels.sort_unstable();
        labels.dedup();
        let covered_internal_nodes = if labels.len() >= 2 {
            trees.iter()
                .enumerate()
                .map(|(ti, tree)| self.mark_component_nodes(ti, tree, &labels))
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

    fn mark_component_nodes(&mut self, ti: usize, tree: &Tree, labels: &[u32]) -> Vec<usize> {
        let stamp = self.next_stamp(ti);
        let marks = &mut self.marks[ti];
        let mut lca_node = tree.node_by_label(labels[0]);
        for &label in &labels[1..] {
            lca_node = tree.nearest_common_ancestor(lca_node, tree.node_by_label(label));
        }

        let mut covered = Vec::new();
        for &label in labels {
            let mut cur = tree.node_by_label(label);
            loop {
                let idx = cur as usize;
                if marks[idx] == stamp {
                    break;
                }
                marks[idx] = stamp;
                if !tree.is_leaf(cur) {
                    covered.push(idx);
                }
                if cur == lca_node {
                    break;
                }
                cur = tree.parent[idx];
            }
        }
        covered.sort_unstable();
        covered
    }
}

struct BpState {
    columns: Vec<BpColumn>,
    seen: FxHashSet<Vec<u32>>,
    best_ub: usize,
    best_solution: Option<Vec<f64>>,
    nodes_explored: usize,
    cg_iterations_total: usize,
    columns_added: usize,
    t_pricer_new: f64,
    t_pricer_solve: f64,
    t_pricer_collect: f64,
    t_lp_apply_bounds: f64,
    t_lp_solve: f64,
    t_add_col: f64,
    t_cuts: f64,
    cuts_added: usize,
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
    let mut column_builder = ColumnBuildScratch::new(trees);

    let cluster_result = cluster_reduction::try_cluster_reduction(reduced, &mut |subinstance| {
        solve_branch_price_multi(subinstance, &mut SolverStats::default())
    })?;
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

    let bounds = compute_local_bounds(trees, reduced.num_leaves);
    let mut columns: Vec<BpColumn> = (1..=n as u32)
        .map(|label| column_builder.build_column(vec![label], trees))
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
                columns.push(column_builder.build_column(labels.clone(), trees));
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
        columns_added: 0,
        t_pricer_new: 0.0,
        t_pricer_solve: 0.0,
        t_pricer_collect: 0.0,
        t_lp_apply_bounds: 0.0,
        t_lp_solve: 0.0,
        t_add_col: 0.0,
        t_cuts: 0.0,
        cuts_added: 0,
    };

    let root = BpNode {
        fixed_to_one: vec![],
        fixed_to_zero: vec![],
        must_link_pairs: vec![],
        cannot_link_pairs: vec![],
        depth: 0,
    };

    let mut pricer_ws = MultiPricerWorkspace::new(trees, n);
    let mut rmp = match PersistentRmp::new(&state.columns, trees, n) {
        Ok(rmp) => rmp,
        Err(err) => {
            eprintln!("[maf-bp-multi] failed to build persistent RMP: {}", err);
            return None;
        }
    };
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let result = solve_bp_node(
            &mut state,
            &node,
            trees,
            n,
            &mut pricer_ws,
            &mut rmp,
            &mut column_builder,
        );
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
    eprintln!(
        "[maf-bp-multi] optimal: {} components, {} B&B nodes, {} CG iters, {} cols, {:.1}ms total",
        components.len(),
        state.nodes_explored,
        state.cg_iterations_total,
        state.columns_added,
        total_ms,
    );
    eprintln!(
        "[maf-bp-multi] timings ms: pricer_new={:.1} pricer_solve={:.1} collect={:.1} apply_bounds={:.1} lp_solve={:.1} add_col={:.1} cuts={:.1} cuts_added={}",
        state.t_pricer_new * 1000.0,
        state.t_pricer_solve * 1000.0,
        state.t_pricer_collect * 1000.0,
        state.t_lp_apply_bounds * 1000.0,
        state.t_lp_solve * 1000.0,
        state.t_add_col * 1000.0,
        state.t_cuts * 1000.0,
        state.cuts_added,
    );
    Some(components)
}

fn solve_bp_node(
    state: &mut BpState,
    node: &BpNode,
    trees: &[Tree],
    num_leaves: usize,
    pricer_ws: &mut MultiPricerWorkspace,
    rmp: &mut PersistentRmp,
    column_builder: &mut ColumnBuildScratch,
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

    let t0 = Instant::now();
    rmp.apply_node_bounds(
        &state.columns,
        &node.fixed_to_one,
        &node.fixed_to_zero,
        &node.must_link_pairs,
        &node.cannot_link_pairs,
        &blocked_leaves,
    );
    state.t_lp_apply_bounds += t0.elapsed().as_secs_f64();

    let lp = loop {
        let t_solve = Instant::now();
        let lp = match rmp.solve(state.columns.len()) {
            Ok(lp) => lp,
            Err(_) => return NodeResult::Pruned,
        };
        state.t_lp_solve += t_solve.elapsed().as_secs_f64();

        // Separate violated node ≤1 cuts FIRST so the dual values we feed to the
        // pricer reflect the tightened LP — otherwise β≡0 on unmaterialized rows
        // causes the pricer to overweight columns covering many internals.
        let t_sep = Instant::now();
        let new_cuts = rmp.separate_and_add_cuts(&state.columns, &lp.column_values, 1.0e-6);
        state.t_cuts += t_sep.elapsed().as_secs_f64();
        if new_cuts > 0 {
            state.cuts_added += new_cuts;
            state.cg_iterations_total += 1;
            continue;
        }

        let alpha = &lp.leaf_duals;
        let beta = &lp.node_duals;
        let t_price = Instant::now();
        let priced_columns = price_best_new_pairdp_columns(
            pricer_ws,
            trees,
            num_leaves,
            alpha,
            beta,
            &blocked_leaves,
            &state.seen,
            &forbidden,
            &node.must_link_pairs,
            &node.cannot_link_pairs,
            &mut state.t_pricer_new,
            &mut state.t_pricer_solve,
            &mut state.t_pricer_collect,
        );
        let _ = t_price; // individual phase timings already captured below

        let mut added_any = false;
        for (score, labels) in priced_columns {
            if score <= 1.0 + 1e-8 {
                continue;
            }
            let inserted = state.seen.insert(labels.clone());
            if !inserted {
                continue;
            }
            let new_ci = state.columns.len();
            state.columns.push(column_builder.build_column(labels, trees));
            let t_add = Instant::now();
            rmp.add_column(new_ci, &state.columns[new_ci], trees);
            state.t_add_col += t_add.elapsed().as_secs_f64();
            if let Some(best_solution) = state.best_solution.as_mut() {
                best_solution.push(0.0);
            }
            added_any = true;
            state.columns_added += 1;
        }

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

    let pair_opt = select_branch_pair(&state.columns, &lp.column_values, num_leaves);
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

struct MultiPricerWorkspace {
    label_stride: usize,
    label_lca: Vec<Vec<u32>>,
    label_side_child: Vec<Vec<u32>>,
    label_node: Vec<Vec<u32>>,
    /// Per-tree, per-node bitset (width = num_leaves + 1) of descendant leaf labels.
    /// Persistent across all CG iterations and B&B nodes — depends only on tree structure.
    descendant_leaves: Vec<Vec<FixedBitSet>>,
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
            label_stride: stride,
            label_lca,
            label_side_child,
            label_node,
            descendant_leaves,
        }
    }
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
    active_labels: Vec<u32>,
    active_nodes: Vec<Vec<u32>>,
    /// Borrowed reference to workspace's persistent per-tree per-node descendant-leaf bitsets.
    /// Each inner FixedBitSet has width num_leaves+1 and is indexed by original leaf label.
    descendant_leaves: &'a [Vec<FixedBitSet>],
    /// Mask (width num_leaves+1) of currently-active labels (alpha>0 and not blocked).
    active_mask: FixedBitSet,
    /// label -> active index; u32::MAX for inactive labels.
    label_to_active_idx: Vec<u32>,
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

        let mut active_nodes = Vec::with_capacity(trees.len());
        let mut prefix_beta = Vec::with_capacity(trees.len());
        let mut roots = Vec::with_capacity(trees.len());
        let mut side_child = Vec::with_capacity(trees.len());

        let mut active_mask = FixedBitSet::with_capacity(num_leaves + 1);
        let mut label_to_active_idx = vec![u32::MAX; num_leaves + 1];
        for (ai, &label) in active_labels.iter().enumerate() {
            active_mask.insert(label as usize);
            label_to_active_idx[label as usize] = ai as u32;
        }

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

            let stride = workspace.label_stride;
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
            active_labels,
            active_nodes,
            descendant_leaves: &workspace.descendant_leaves,
            active_mask,
            label_to_active_idx,
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

    #[inline(always)]
    fn root_of(&self, ti: usize, a: usize, b: usize) -> u32 {
        self.roots[ti][self.pair_idx(a, b)]
    }

    #[inline(always)]
    fn side_of(&self, ti: usize, a: usize, b: usize) -> u32 {
        self.side_child[ti][self.pair_idx(a, b)]
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
                let label_b = self.active_labels[b];
                let mut best_score =
                    self.alpha[label_a as usize] - self.singleton_chain_penalty_2tree(a, b);
                let mut best_choice = PairDpSideChoice::Singleton;

                let side0 = self.side_of(0, a, b) as usize;
                let side1 = self.side_of(1, a, b) as usize;
                let d0 = self.descendant_leaves[0][side0].as_slice();
                let d1 = self.descendant_leaves[1][side1].as_slice();
                let am = self.active_mask.as_slice();
                let la_w = label_a as usize >> 6;
                let la_m = 1usize << (label_a as usize & 63);
                let lb_w = label_b as usize >> 6;
                let lb_m = 1usize << (label_b as usize & 63);
                for wi in 0..d0.len() {
                    let mut w = d0[wi] & d1[wi] & am[wi];
                    if wi == la_w { w &= !la_m; }
                    if wi == lb_w { w &= !lb_m; }
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        w &= w - 1;
                        let c_label = (wi << 6) + bit;
                        let c = self.label_to_active_idx[c_label] as usize;
                        let idx_ac = self.pair_idx(a, c);
                        let child_score = self.memo_pair[idx_ac]
                            .expect("child pair must be solved earlier in pair DP order");
                        if child_score <= NEG_INF / 2.0 {
                            continue;
                        }
                        let cand = child_score - self.transition_chain_penalty_2tree(a, b, c);
                        if cand > best_score + 1.0e-12 {
                            best_score = cand;
                            best_choice = PairDpSideChoice::Split(c);
                        }
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
                    -self.root_penalty_2tree(a, b) + left + right
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
        let label_b = self.active_labels[b];
        let mut best_score = self.alpha[label_a as usize] - self.singleton_chain_penalty(a, b);
        let mut best_choice = PairDpSideChoice::Singleton;

        // Collect candidate labels via word-level AND across all trees' descendant bitsets
        // intersected with the active mask, then clear a and b.
        let num_trees = self.trees.len();
        let side_nodes: Vec<usize> = (0..num_trees)
            .map(|ti| self.side_of(ti, a, b) as usize)
            .collect();
        let d0 = self.descendant_leaves[0][side_nodes[0]].as_slice();
        let am = self.active_mask.as_slice();
        let la_w = label_a as usize >> 6;
        let la_m = 1usize << (label_a as usize & 63);
        let lb_w = label_b as usize >> 6;
        let lb_m = 1usize << (label_b as usize & 63);
        let mut candidate_labels: Vec<usize> = Vec::new();
        for wi in 0..d0.len() {
            let mut w = d0[wi] & am[wi];
            for ti in 1..num_trees {
                w &= self.descendant_leaves[ti][side_nodes[ti]].as_slice()[wi];
            }
            if wi == la_w { w &= !la_m; }
            if wi == lb_w { w &= !lb_m; }
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                w &= w - 1;
                candidate_labels.push((wi << 6) + bit);
            }
        }
        for c_label in candidate_labels {
            let c = self.label_to_active_idx[c_label] as usize;
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
        self.trees
            .iter()
            .enumerate()
            .map(|(ti, _)| self.beta[ti][self.root_of(ti, a, b) as usize])
            .sum()
    }

    fn singleton_chain_penalty(&self, a: usize, b: usize) -> f64 {
        self.trees
            .iter()
            .enumerate()
            .map(|(ti, tree)| {
                self.path_internal_penalty(
                    ti,
                    tree,
                    self.side_of(ti, a, b),
                    self.active_nodes[ti][a],
                )
            })
            .sum()
    }

    fn transition_chain_penalty(&self, a: usize, b: usize, c: usize) -> f64 {
        self.trees
            .iter()
            .enumerate()
            .map(|(ti, tree)| {
                self.path_internal_penalty(
                    ti,
                    tree,
                    self.side_of(ti, a, b),
                    self.root_of(ti, a, c),
                )
            })
            .sum()
    }

    #[inline(always)]
    fn root_penalty_2tree(&self, a: usize, b: usize) -> f64 {
        self.beta[0][self.root_of(0, a, b) as usize]
            + self.beta[1][self.root_of(1, a, b) as usize]
    }

    #[inline(always)]
    fn singleton_chain_penalty_2tree(&self, a: usize, b: usize) -> f64 {
        self.path_internal_penalty(0, &self.trees[0], self.side_of(0, a, b), self.active_nodes[0][a])
            + self.path_internal_penalty(1, &self.trees[1], self.side_of(1, a, b), self.active_nodes[1][a])
    }

    #[inline(always)]
    fn transition_chain_penalty_2tree(&self, a: usize, b: usize, c: usize) -> f64 {
        self.path_internal_penalty(0, &self.trees[0], self.side_of(0, a, b), self.root_of(0, a, c))
            + self.path_internal_penalty(1, &self.trees[1], self.side_of(1, a, b), self.root_of(1, a, c))
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
    t_new: &mut f64,
    t_solve: &mut f64,
    t_collect: &mut f64,
) -> Vec<(f64, Vec<u32>)> {
    let t0 = Instant::now();
    let mut pricer = PairDpPricer::new(
        pricer_ws,
        trees,
        num_leaves,
        alpha,
        beta,
        blocked_leaves,
    );
    *t_new += t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    if trees.len() == 2 {
        pricer.solve_all_2tree();
    }
    *t_solve += t1.elapsed().as_secs_f64();
    let t2 = Instant::now();
    let (_raw, out) = pricer.collect_profitable_columns(PAIRDP_BATCH_SIZE, |labels| {
        pricing_candidate_allowed(labels, seen, forbidden, must_link_pairs, cannot_link_pairs)
    });
    *t_collect += t2.elapsed().as_secs_f64();
    out
}

struct RmpLpResult {
    objective: f64,
    column_values: Vec<f64>,
    leaf_duals: Vec<f64>,
    node_duals: Vec<Vec<f64>>,
}

struct PersistentRmp {
    model: Option<Model>,
    leaf_row_idx: Vec<usize>,
    /// Row index for each (tree, internal-node). `None` until the row is materialized
    /// lazily via cut separation when violated.
    node_row_idx: Vec<Vec<Option<usize>>>,
    /// Total number of rows currently in the HiGHS model.
    num_rows: usize,
    /// Reverse index: node_to_cols[ti][node_idx] = list of global column indices whose
    /// column covers that internal node. Populated for every column regardless of whether
    /// the row has been materialized — used to build the row's coefficient vector when
    /// materialized later.
    node_to_cols: Vec<Vec<Vec<usize>>>,
    /// HiGHS column index for each global column (parallel to state.columns).
    col_idx: Vec<i32>,
    /// Current lower bound applied in HiGHS for each added column.
    current_lower: Vec<f64>,
    /// Current upper bound applied in HiGHS for each added column.
    current_upper: Vec<f64>,
    fixed_one_mark: Vec<u32>,
    fixed_zero_mark: Vec<u32>,
    mark_epoch: u32,
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
        // Presolve is wasteful when we warm-start with an existing basis on every solve.
        // We change only column bounds / add columns between solves, so dual simplex
        // can resume from the stored basis directly.
        model.set_option("presolve", "off");
        model.set_option("solver", "simplex");
        model.set_option("simplex_strategy", 1_i32); // 1 = dual simplex (best for warm-start with bound changes)

        // Leaf rows are added eagerly — every column necessarily covers at least one leaf.
        for leaf in 0..=num_leaves {
            if leaf == 0 {
                model.add_row(0.0..=0.0, Vec::new());
            } else {
                model.add_row(1.0..=1.0, Vec::new());
            }
        }
        let leaf_row_idx: Vec<usize> = (0..=num_leaves).collect();
        let num_rows = num_leaves + 1;

        // Internal-node rows are materialized lazily — only if a LP solution violates
        // the ≤1 constraint for that node. Most internal nodes never become tight.
        let node_row_idx: Vec<Vec<Option<usize>>> = trees
            .iter()
            .map(|tree| vec![None; tree.num_nodes()])
            .collect();
        let node_to_cols: Vec<Vec<Vec<usize>>> = trees
            .iter()
            .map(|tree| vec![Vec::new(); tree.num_nodes()])
            .collect();

        let mut rmp = Self {
            model: Some(model),
            leaf_row_idx,
            node_row_idx,
            num_rows,
            node_to_cols,
            col_idx: Vec::new(),
            current_lower: Vec::new(),
            current_upper: Vec::new(),
            fixed_one_mark: Vec::new(),
            fixed_zero_mark: Vec::new(),
            mark_epoch: 1,
        };
        for (ci, col) in columns.iter().enumerate() {
            rmp.add_column(ci, col, trees);
        }
        Ok(rmp)
    }

    fn add_column(&mut self, global_ci: usize, col: &BpColumn, trees: &[Tree]) {
        debug_assert_eq!(global_ci, self.col_idx.len());
        // Build row-index/value vectors for the C API.
        let mut row_indices: Vec<i32> = col
            .labels
            .iter()
            .map(|&label| self.leaf_row_idx[label as usize] as i32)
            .collect();
        if col.labels.len() >= 2 {
            for (ti, tree) in trees.iter().enumerate() {
                for &node in &col.covered_internal_nodes[ti] {
                    debug_assert!(!tree.is_leaf(node as u32));
                    // Always record the coverage in the reverse index — the row may be
                    // materialized later and will need to know which columns cover it.
                    self.node_to_cols[ti][node].push(global_ci);
                    if let Some(ri) = self.node_row_idx[ti][node] {
                        row_indices.push(ri as i32);
                    }
                }
            }
        }
        let values: Vec<f64> = vec![1.0; row_indices.len()];
        let ptr = self
            .model
            .as_mut()
            .expect("RMP model present")
            .as_mut_ptr();
        unsafe {
            highs_sys::Highs_addCol(
                ptr,
                1.0,               // cost
                0.0,               // lower bound
                f64::INFINITY,     // upper bound
                row_indices.len() as i32,
                row_indices.as_ptr(),
                values.as_ptr(),
            );
        }
        self.col_idx.push(global_ci as i32);
        self.current_lower.push(0.0);
        self.current_upper.push(f64::INFINITY);
        self.fixed_one_mark.push(0);
        self.fixed_zero_mark.push(0);
    }

    /// Materialize an internal-node row lazily. Pulls all currently-alive columns that
    /// cover (ti, node) from the reverse index and adds a ≤1 constraint over them.
    fn add_node_row_lazy(&mut self, ti: usize, node: usize) {
        debug_assert!(self.node_row_idx[ti][node].is_none());
        let cols_covering = &self.node_to_cols[ti][node];
        let indices: Vec<i32> = cols_covering
            .iter()
            .map(|&ci| self.col_idx[ci])
            .collect();
        let values: Vec<f64> = vec![1.0; indices.len()];
        let ptr = self
            .model
            .as_mut()
            .expect("RMP model present")
            .as_mut_ptr();
        unsafe {
            highs_sys::Highs_addRow(
                ptr,
                -f64::INFINITY,
                1.0,
                indices.len() as i32,
                indices.as_ptr(),
                values.as_ptr(),
            );
        }
        self.node_row_idx[ti][node] = Some(self.num_rows);
        self.num_rows += 1;
    }

    /// Scan the LP support for violated node ≤1 constraints and materialize them.
    /// Returns number of rows added.
    fn separate_and_add_cuts(
        &mut self,
        columns: &[BpColumn],
        column_values: &[f64],
        eps: f64,
    ) -> usize {
        let mut tally: FxHashMap<(usize, usize), f64> = FxHashMap::default();
        for (ci, &val) in column_values.iter().enumerate() {
            if val <= 1.0e-9 {
                continue;
            }
            if ci >= columns.len() {
                continue;
            }
            let col = &columns[ci];
            for (ti, nodes) in col.covered_internal_nodes.iter().enumerate() {
                for &node in nodes {
                    if self.node_row_idx[ti][node].is_none() {
                        *tally.entry((ti, node)).or_insert(0.0) += val;
                    }
                }
            }
        }
        let mut added = 0usize;
        for ((ti, node), sum) in tally {
            if sum > 1.0 + eps {
                self.add_node_row_lazy(ti, node);
                added += 1;
            }
        }
        added
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
        if self.mark_epoch == u32::MAX {
            self.fixed_one_mark.fill(0);
            self.fixed_zero_mark.fill(0);
            self.mark_epoch = 1;
        }
        self.mark_epoch += 1;
        let epoch = self.mark_epoch;
        for &ci in fixed_to_one {
            if ci < self.fixed_one_mark.len() {
                self.fixed_one_mark[ci] = epoch;
            }
        }
        for &ci in fixed_to_zero {
            if ci < self.fixed_zero_mark.len() {
                self.fixed_zero_mark[ci] = epoch;
            }
        }
        let ptr = self
            .model
            .as_mut()
            .expect("RMP model present")
            .as_mut_ptr();
        for ci in 0..self.col_idx.len() {
            let labels = &columns[ci].labels;
            let (desired_lo, desired_hi) = if self.fixed_one_mark[ci] == epoch {
                (1.0, 1.0)
            } else if self.fixed_zero_mark[ci] == epoch {
                (0.0, 0.0)
            } else if labels.iter().any(|&l| blocked_leaves[l as usize])
                || !labels_satisfy_pair_constraints(
                    labels,
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
