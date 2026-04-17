//! Experimental multi-tree Branch & Price solver.
//!
//! This is a simple theory-checking implementation:
//! - generic master problem over an arbitrary number of rooted trees
//! - exact column pricing by exhaustive subset enumeration
//! - generic branch-and-bound on fractional columns
//!
//! The exhaustive pricer is only practical on small reduced instances, but it
//! lets us verify the multi-tree master problem and pricing logic without any
//! fallback solver.

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use fixedbitset::FixedBitSet;
use highs::{ColProblem, HighsModelStatus, Model, Row};
use klados_core::lower_bound::maf_bounds;
use klados_core::{Instance, SolverStats, Tree};

use crate::cluster_reduction::{self, ClusterReductionResult};
use crate::kernelize::{self, KernelizeConfig};
use crate::ExactSolver;

const MAX_ENUM_LEAVES: usize = 24;

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
    seen: HashSet<Vec<u32>>,
    best_ub: usize,
    best_solution: Option<Vec<f64>>,
    nodes_explored: usize,
    cg_iterations_total: usize,
}

#[derive(Clone)]
struct BpNode {
    fixed_to_one: Vec<usize>,
    fixed_to_zero: Vec<usize>,
    depth: usize,
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

    if reduced.num_leaves as usize > MAX_ENUM_LEAVES {
        eprintln!(
            "[maf-bp-multi] reduced instance still has {} leaves; exhaustive pricing capped at {}",
            reduced.num_leaves, MAX_ENUM_LEAVES
        );
        return None;
    }

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
    };

    let root = BpNode {
        fixed_to_one: vec![],
        fixed_to_zero: vec![],
        depth: 0,
    };

    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        match solve_bp_node(&mut state, &node, trees, n) {
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
        "[maf-bp-multi] optimal: {} components, {} B&B nodes, {} CG iterations, {:.1}ms total",
        components.len(),
        state.nodes_explored,
        state.cg_iterations_total,
        t_total.elapsed().as_secs_f64() * 1000.0,
    );
    Some(components)
}

fn solve_bp_node(
    state: &mut BpState,
    node: &BpNode,
    trees: &[Tree],
    num_leaves: usize,
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

    let mut forbidden_labels = state.seen.clone();
    for &ci in &node.fixed_to_zero {
        forbidden_labels.insert(state.columns[ci].labels.clone());
    }

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
        let priced = price_best_column_exhaustive(
            trees,
            num_leaves,
            &alpha,
            &beta,
            &blocked_leaves,
            &forbidden_labels,
        );

        match priced {
            Some((score, labels)) if score > 1.0 + 1e-8 => {
                let inserted = state.seen.insert(labels.clone());
                if !inserted {
                    final_lp = Some(lp);
                    break;
                }
                forbidden_labels.insert(labels.clone());
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
    forbidden: &HashSet<Vec<u32>>,
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
        if forbidden.contains(&labels) {
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
