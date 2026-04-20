//! Rooted Branch & Price solver for 2-tree Maximum Agreement Forest.
//!
//! Implements the rooted specialization of Frohn 2025 "Branch & Price":
//! - Set cover master problem (Formulation 1): min components subject to
//!   leaf covering (equality) and internal-node packing (≤ 1).
//!   NOTE: leaf cover uses =1 (exact cover) vs paper's ≥1 — stronger LP.
//! - Pricing via O(n²) rooted WMAST DP (V/M/W recurrences).
//!   Rooted simplification: discards paper's Eq. 6 edge-cut maximization.
//! - Integrality check at each B&B node (~99% solved at root per paper).
//! - RATIO branching strategy (default), SIZE available via env var.
//! - Column generation re-run at every B&B node with branching fixings.
//!
//! Deviations from paper:
//! - No epsilon stabilization (Formulation 3/4) — leaves degeneracy on table.
//! - Constrained unseen-column search is heuristic (1000-iter cap).
//!
//! Restricted to m = 2 trees; falls back to maf-sat for multi-tree instances.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::time::Instant;

use fixedbitset::FixedBitSet;
use highs::{ColProblem, HighsModelStatus, Model, Row};
use klados_core::lower_bound::maf_bounds;
use klados_core::{Instance, SolverStats, Tree};

use crate::cluster_reduction::{self, ClusterReductionResult};
use crate::kernelize::{self, KernelizeConfig};
use crate::ExactSolver;

// ---------------------------------------------------------------------------
// Solver struct
// ---------------------------------------------------------------------------

pub struct MafBranchPriceSolver {
    stats: SolverStats,
}

impl Default for MafBranchPriceSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl MafBranchPriceSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
        }
    }
}

impl ExactSolver for MafBranchPriceSolver {
    fn name(&self) -> &'static str {
        "maf-bp"
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
        // B&P pricer is 2-tree only; fall back to SAT for multi-tree.
        if instance.num_trees() != 2 {
            eprintln!(
                "[maf-bp] m={}, falling back to maf-sat",
                instance.num_trees()
            );
            let mut sat = crate::maf_sat::MafSatSolver::new();
            let result = crate::ExactSolver::solve(&mut sat, instance);
            self.stats = crate::ExactSolver::stats(&sat).clone();
            return result;
        }
        solve_branch_price(instance, &mut self.stats)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// Column representation
// ---------------------------------------------------------------------------

struct BpColumn {
    labels: Vec<u32>,            // sorted leaf labels in this block
    covered_internal_nodes: Vec<Vec<usize>>,
    total_internal_count: usize, // cached sum of covered internal nodes across trees
}

// ---------------------------------------------------------------------------
// Branch-and-bound types
// ---------------------------------------------------------------------------

/// Global state shared across all B&B nodes.
struct BpState {
    columns: Vec<BpColumn>,
    seen: HashSet<Vec<u32>>,
    best_ub: usize,
    best_solution: Option<Vec<f64>>,
    nodes_explored: usize,
    cg_iterations_total: usize,
}

/// Per-node branching decisions. Column indices are stable (append-only pool).
#[derive(Clone)]
struct BpNode {
    fixed_to_one: Vec<usize>,  // column indices forced to a_Y = 1
    fixed_to_zero: Vec<usize>, // column indices forced to a_Y = 0
    depth: usize,
}

enum NodeResult {
    /// LP is integral at this node. (objective, column_values)
    Integral(usize, Vec<f64>),
    /// LP is fractional. Branch on this column index.
    Branch { lp_obj: f64, branch_col: usize },
    /// Node pruned (LP bound ≥ incumbent) or infeasible.
    Pruned,
}

// ---------------------------------------------------------------------------
// Main B&P pipeline
// ---------------------------------------------------------------------------

fn solve_branch_price(instance: &Instance, stats: &mut SolverStats) -> Option<Vec<Tree>> {
    let t_total = Instant::now();

    let config = KernelizeConfig::default();
    let kern = kernelize::kernelize_best(instance, &config);
    let reduced = &kern.instance;

    eprintln!(
        "[maf-bp] kernelized {} -> {} leaves (param_reduction={})",
        instance.num_leaves, reduced.num_leaves, kern.param_reduction,
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

    // Try Kelk common-cluster decomposition (works for any m).
    match cluster_reduction::try_cluster_reduction(reduced, &mut |subinstance| {
        solve_branch_price(subinstance, &mut SolverStats::default())
    })? {
        ClusterReductionResult::NotApplicable => {}
        ClusterReductionResult::Solved(solution) => {
            eprintln!(
                "[maf-bp] Cluster decomposition: {} = {} + {}",
                n, solution.cluster_size, solution.rest_size
            );
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
                "[maf-bp] optimal: {} components (cluster decomp), {:.1}ms total",
                components.len(),
                t_total.elapsed().as_secs_f64() * 1000.0,
            );
            return Some(components);
        }
    }

    // rspr-style cluster decomposition is intentionally NOT used here.
    // Its round-trip depth check finds structural clusters valid for RSPR distance,
    // but these are not necessarily agreement clusters (clades in all trees),
    // which is required for MAF correctness. Kelk decomposition above handles
    // valid common-cluster splits.

    // Compute greedy UB for tighter initial bound (enables more B&B pruning).
    let bounds = maf_bounds(trees, reduced.num_leaves);
    let _ = &bounds; // used for best_partition below

    // Initialize column pool with singletons.
    let mut columns: Vec<BpColumn> = (1..=n as u32)
        .map(|label| make_bp_column(vec![label], trees))
        .collect();
    let mut seen: HashSet<Vec<u32>> = columns.iter().map(|c| c.labels.clone()).collect();

    // Seed columns and best_solution from the greedy partition so LP pruning works from node 0.
    // Without this, the solver can't prune fractional nodes until it stumbles upon
    // an integral LP solution naturally — causing the tree to explode in depth.
    let mut best_solution: Option<Vec<f64>> = None;
    let mut best_ub = n; // default: singleton solution
    if let Some(partition) = &bounds.best_partition {
        let mut comp_labels: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for (leaf_idx, &comp_id) in partition.iter().enumerate() {
            comp_labels.entry(comp_id).or_default().push((leaf_idx + 1) as u32);
        }
        let num_components = comp_labels.len();
        let mut values = vec![0.0; columns.len()];
        for (_comp_id, labels) in &comp_labels {
            let mut found = false;
            for (ci, col) in columns.iter().enumerate() {
                if col.labels == *labels {
                    values[ci] = 1.0;
                    found = true;
                    break;
                }
            }
            if !found {
                // Add the missing column to the pool.
                columns.push(make_bp_column(labels.clone(), trees));
                values.push(1.0);
                seen.insert(labels.clone());
            }
        }
        best_solution = Some(values);
        best_ub = num_components.min(n);
        eprintln!("[maf-bp] seeded best_solution from greedy partition (UB={})", best_ub);
    } else {
        // Fallback: seed best_solution with the trivial all-singletons solution
        // so LP pruning works even when the greedy partition isn't available.
        let mut values = vec![0.0; columns.len()];
        for (ci, col) in columns.iter().enumerate() {
            if col.labels.len() == 1 {
                values[ci] = 1.0;
            }
        }
        best_solution = Some(values);
        best_ub = n;
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

    // DFS branch-and-bound
    let mut pricer_ws = PricerWorkspace::new(&trees[0], &trees[1]);
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let result = solve_bp_node(&mut state, &node, trees, n, &mut pricer_ws);
        match result {
            NodeResult::Integral(obj, values) => {
                if obj < state.best_ub {
                    eprintln!(
                        "[maf-bp] new incumbent: {} components (depth={}, nodes={})",
                        obj, node.depth, state.nodes_explored,
                    );
                    state.best_ub = obj;
                    let mut padded_values = values;
                    padded_values.resize(state.columns.len(), 0.0);
                    state.best_solution = Some(padded_values);
                }
            }
            NodeResult::Branch { lp_obj, branch_col } => {
                let lp_lb = (lp_obj - 1e-6).ceil() as usize;
                if lp_lb >= state.best_ub {
                    continue; // pruned after CG (race with incumbent update)
                }
                let branch_labels = &state.columns[branch_col].labels;
                eprintln!(
                    "[maf-bp] branching on column {} (|Y|={}, depth={})",
                    branch_col, branch_labels.len(), node.depth,
                );

                // Right child: exclude branch_col (a_Y = 0)
                let mut right = node.clone();
                right.fixed_to_zero.push(branch_col);
                right.depth += 1;

                // Left child: include branch_col (a_Y = 1)
                let mut left = node.clone();
                left.fixed_to_one.push(branch_col);
                left.depth += 1;

                // Push right first so DFS explores left first (include-first heuristic)
                stack.push(right);
                stack.push(left);
            }
            NodeResult::Pruned => {}
        }
    }

    // Reconstruct from best solution
    let result = if let Some(values) = &state.best_solution {
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
            "[maf-bp] optimal: {} components, {} B&B nodes, {} CG iterations, {:.1}ms total, returning Some",
            components.len(),
            state.nodes_explored,
            state.cg_iterations_total,
            t_total.elapsed().as_secs_f64() * 1000.0,
        );
        Some(components)
    } else {
        eprintln!("[maf-bp] no solution found, returning None");
        None
    };
    eprintln!(
        "[maf-bp] solve_branch_price returning: {}",
        if result.is_some() { "Some" } else { "None" }
    );
    result
}

// ---------------------------------------------------------------------------
// Solve a single B&B node: CG loop + integrality check
// ---------------------------------------------------------------------------

fn solve_bp_node(
    state: &mut BpState,
    node: &BpNode,
    trees: &[Tree],
    num_leaves: usize,
    pricer_ws: &mut PricerWorkspace,
) -> NodeResult {
    state.nodes_explored += 1;

    // Forced-one columns each contribute 1 to the objective; prune hopeless nodes early.
    if node.fixed_to_one.len() >= state.best_ub {
        return NodeResult::Pruned;
    }

    let mut blocked_leaves = vec![false; num_leaves + 1];
    for &forced_ci in &node.fixed_to_one {
        for &label in &state.columns[forced_ci].labels {
            blocked_leaves[label as usize] = true;
        }
    }
    let banned_zero_labels = node
        .fixed_to_zero
        .iter()
        .map(|&ci| state.columns[ci].labels.clone())
        .collect::<HashSet<_>>();
    // Forbidden labels for this node = globally seen columns ∪ branch-fixed-to-zero columns.
    // We keep this set incrementally updated to avoid cloning state.seen on every CG iteration.
    let mut forbidden_labels = state.seen.clone();
    forbidden_labels.extend(banned_zero_labels.iter().cloned());

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
    // Column generation loop
    let mut cg_iters_this_node = 0usize;
    let mut final_lp: Option<RmpLpResult> = None;
    loop {
        let lp = match node_rmp.solve(state.columns.len()) {
            Ok(lp) => lp,
            Err(_) => return NodeResult::Pruned, // infeasible
        };

        // Extract true alpha and beta from the same RMP row duals.
        // Using duals from the same LP ensures consistency: the pricer's
        // reduced cost = sum(alpha) - sum(beta) is meaningful.
        let alpha = lp.leaf_duals.clone();
        let beta = lp.node_duals.clone();

        // Fast path for rooted instances: run the standard pricer first.
        // Only when the best unconstrained label-set is forbidden at this node
        // do we pay for constrained unseen-column separation.
        let priced = match run_rooted_paper_pricer(
            pricer_ws,
            &trees[0],
            &trees[1],
            &alpha,
            &beta,
            &blocked_leaves,
        ) {
            Ok(Some((score, labels))) if score > 1.0 + 1e-8 && forbidden_labels.contains(&labels) => {
                match price_best_new_compatible_column(
                    pricer_ws,
                    &trees[0],
                    &trees[1],
                    &alpha,
                    &beta,
                    &blocked_leaves,
                    &forbidden_labels,
                ) {
                    Ok(Some((alt_score, alt_labels))) => (alt_score, alt_labels),
                    Ok(None) => {
                        final_lp = Some(lp);
                        break;
                    }
                    Err(_) => break,
                }
            }
            Ok(Some((score, labels))) => (score, labels),
            Ok(None) => {
                final_lp = Some(lp);
                break;
            }
            Err(_) => break,
        };

        let (score, labels) = priced;
        if score <= 1.0 + 1e-8 {
            final_lp = Some(lp);
            break; // CG converged: no improving column
        }

        // For rooted trees, the WMAST DP guarantees the priced column is
        // a valid agreement subtree — no triplet post-check needed.

        let inserted = state.seen.insert(labels.clone());
        if !inserted {
            eprintln!(
                "[maf-bp]   INTERNAL BUG depth={}: constrained pricing returned duplicate column {:?}; stopping CG at this node",
                node.depth, labels
            );
            break;
        }
        forbidden_labels.insert(labels.clone());
        let new_ci = state.columns.len();
        state.columns.push(make_bp_column(labels.clone(), trees));
        node_rmp.add_column(new_ci, &state.columns[new_ci], trees);
        if let Some(best_solution) = state.best_solution.as_mut() {
            best_solution.push(0.0);
        }
        state.cg_iterations_total += 1;
        cg_iters_this_node += 1;
    }

    // Final LP solve at this node (reuse the terminal LP if we already have it).
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

    // Check integrality
    let integral = support_is_integral_partition(&state.columns, &lp.column_values, num_leaves);
    if integral {
        let obj = lp.column_values.iter().filter(|&&v| v > 1.0e-9).count();
        return NodeResult::Integral(obj, lp.column_values);
    }

    // Find branching column: paper/reference SIZE strategy on the positive support.
    let branch_col = select_branch_column(&state.columns, &lp.column_values, num_leaves);
    match branch_col {
        Some(col_idx) => NodeResult::Branch {
            lp_obj: lp.objective,
            branch_col: col_idx,
        },
        None => NodeResult::Pruned, // shouldn't happen if LP is fractional
    }
}

#[derive(Clone, Copy)]
enum BranchStrategy {
    Size,
    Ratio,
}

fn branch_strategy() -> BranchStrategy {
    match std::env::var("KLADOS_MAF_BP_BRANCH") {
        Ok(value) if value.trim().eq_ignore_ascii_case("size") => BranchStrategy::Size,
        Ok(value) if value.trim().eq_ignore_ascii_case("ratio") => BranchStrategy::Ratio,
        Ok(_) | Err(_) => BranchStrategy::Ratio,
    }
}

/// SIZE branching as in the paper/reference implementation:
/// among positive-support columns that hit duplicated leaves, pick the largest block.
fn select_branch_column(columns: &[BpColumn], values: &[f64], num_leaves: usize) -> Option<usize> {
    const ACTIVE_EPS: f64 = 1.0e-9;
    let strategy = branch_strategy();

    let mut seen = vec![false; num_leaves + 1];
    let mut duplicated = vec![false; num_leaves + 1];
    for (ci, &value) in values.iter().enumerate() {
        if value <= ACTIVE_EPS || value >= 1.0 - ACTIVE_EPS {
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
        if value <= ACTIVE_EPS || value >= 1.0 - ACTIVE_EPS {
            continue;
        }
        if !columns[ci]
            .labels
            .iter()
            .any(|&label| duplicated[label as usize])
        {
            continue;
        }
        let size = columns[ci].labels.len() as f64;
        let total_internal = columns[ci].total_internal_count as f64;
        let score = match strategy {
            BranchStrategy::Size => size,
            // In the paper, RATIO is defined as |Y| / |V[Y]|.
            // For singleton columns V[Y] = ∅, so the ratio is undefined.
            // Those are poor branch candidates anyway; when the LP is fractional,
            // the useful branching signal comes from duplicated multi-leaf blocks.
            BranchStrategy::Ratio => {
                if columns[ci].labels.len() <= 1 || total_internal <= 0.0 {
                    continue;
                }
                size / total_internal
            }
        };
        if score > best_score {
            best_idx = Some(ci);
            best_score = score;
        }
    }
    if best_idx.is_none() && matches!(strategy, BranchStrategy::Ratio) {
        for (ci, &value) in values.iter().enumerate() {
            if value <= ACTIVE_EPS || value >= 1.0 - ACTIVE_EPS {
                continue;
            }
            if !columns[ci]
                .labels
                .iter()
                .any(|&label| duplicated[label as usize])
            {
                continue;
            }
            let size = columns[ci].labels.len() as f64;
            if size > best_score {
                best_idx = Some(ci);
                best_score = size;
            }
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// LP formulation with branching fixings
// ---------------------------------------------------------------------------

struct RmpLpResult {
    objective: f64,
    column_values: Vec<f64>,
    /// Dual values for leaf covering constraints (=1). These are the true alpha
    /// values for the pricing problem. Extracted from HiGHS row duals.
    leaf_duals: Vec<f64>,
    /// Dual values for internal-node packing constraints (<=1). These are the
    /// true beta values. For <=1 rows, beta = -row_dual (HiGHS sign convention).
    node_duals: Vec<Vec<f64>>,
}

struct NodeRmp {
    model: Option<Model>,
    active_global_cols: Vec<usize>,
    leaf_rows: Vec<Row>,
    leaf_row_idx: Vec<usize>,    // usize indices for dual extraction
    node_rows: Vec<Vec<Option<Row>>>,
    node_row_idx: Vec<Vec<Option<usize>>>, // usize indices for dual extraction
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
        for tree in trees.iter() {
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

        // Extract dual values from row constraints.
        // HiGHS row duals follow the active row-bound sign in minimization:
        // lower-bound rows are nonnegative, upper-bound rows nonpositive.
        // Leaf cover rows are = 1 (both lb and ub), dual is direct.
        // Node pack rows are <= 1 (ub only), beta = -row_dual.
        let dual_rows = solution.dual_rows();
        let leaf_duals = self.leaf_row_idx.iter().map(|&ri| clean_dual(dual_rows[ri])).collect();
        let node_duals = self
            .node_row_idx
            .iter()
            .map(|tree_idxs| {
                tree_idxs
                    .iter()
                    .map(|opt| {
                        opt.map(|ri| clean_dual(-dual_rows[ri]))
                            .unwrap_or(0.0)
                    })
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

// ---------------------------------------------------------------------------
// Solution reconstruction
// ---------------------------------------------------------------------------

fn support_is_integral_partition(
    columns: &[BpColumn],
    values: &[f64],
    num_leaves: usize,
) -> bool {
    const ACTIVE_EPS: f64 = 1.0e-9;

    let mut cover_count = vec![0usize; num_leaves + 1];
    for (ci, &value) in values.iter().enumerate() {
        if value <= ACTIVE_EPS {
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

fn reconstruct_components(
    columns: &[BpColumn],
    values: &[f64],
    instance: &Instance,
) -> Vec<Tree> {
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

// ---------------------------------------------------------------------------
// Index helpers
// ---------------------------------------------------------------------------

fn labels_disjoint(a: &[u32], b: &[u32]) -> bool {
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return false,
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
    if value.abs() <= 1.0e-9 {
        0.0
    } else {
        value
    }
}

fn make_bp_column(labels: Vec<u32>, trees: &[Tree]) -> BpColumn {
    let covered_internal_nodes = if labels.len() >= 2 {
        trees.iter()
            .map(|tree| {
                let cover = mark_component_nodes(tree, &labels);
                cover
                    .ones()
                    .filter(|&node| !tree.is_leaf(node as u32))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    } else {
        trees.iter()
            .map(|_| Vec::new())
            .collect::<Vec<_>>()
    };
    let total_internal_count = covered_internal_nodes.iter().map(|n| n.len()).sum();
    BpColumn {
        labels,
        covered_internal_nodes,
        total_internal_count,
    }
}

#[derive(Clone)]
struct PricingPrefixNode {
    prefix_membership: Vec<bool>,
    score: f64,
    labels: Vec<u32>,
}

impl PartialEq for PricingPrefixNode {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Equal
    }
}
impl Eq for PricingPrefixNode {}
impl PartialOrd for PricingPrefixNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PricingPrefixNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score)
    }
}

fn price_best_new_compatible_column(
    pricer_ws: &mut PricerWorkspace,
    t1: &Tree,
    t2: &Tree,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    forbidden: &HashSet<Vec<u32>>,
) -> Result<Option<(f64, Vec<u32>)>, Box<dyn std::error::Error>> {
    // Fast path: run the base DP once. If the best column is not forbidden, return it.
    // This avoids the entire prefix search in the common case (no forbidden columns
    // or best column is valid).
    let base_priced = run_rooted_paper_pricer(pricer_ws, t1, t2, alpha, beta, blocked_leaves)?;
    let Some((base_score, base_labels)) = base_priced else {
        return Ok(None);
    };
    if !forbidden.contains(&base_labels) {
        return Ok(Some((base_score, base_labels)));
    }

    // The best column is forbidden. Prepare for prefix search.
    let free_labels = (1..=t1.num_leaves)
        .filter(|&label| !blocked_leaves[label as usize])
        .collect::<Vec<_>>();
    let ordinary_upper = alpha.iter().map(|&value| value.max(0.0)).sum::<f64>();
    let beta_sum = beta
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .sum::<f64>();
    let required_bonus = ordinary_upper + beta_sum + 1.0;

    // Single-flip heuristic: force the first free label in/out and run the
    // constrained DP. This is a cheap 2-call heuristic that often finds an
    // improving column before the full prefix search.
    if !free_labels.is_empty() {
        // Force label IN
        if let Some(node) = solve_pricing_prefix_subproblem(
            pricer_ws,
            t1, t2, alpha, beta, blocked_leaves,
            &free_labels, &[true], required_bonus,
        )? {
            if !forbidden.contains(&node.labels) {
                return Ok(Some((node.score, node.labels)));
            }
        }
        // Force label OUT
        if let Some(node) = solve_pricing_prefix_subproblem(
            pricer_ws,
            t1, t2, alpha, beta, blocked_leaves,
            &free_labels, &[false], required_bonus,
        )? {
            if !forbidden.contains(&node.labels) {
                return Ok(Some((node.score, node.labels)));
            }
        }
    }

    // Heuristics exhausted. Fall back to exact prefix search to certify
    // that no improving column exists (or find one).
    let mut frontier = BinaryHeap::new();
    if let Some(root) = solve_pricing_prefix_subproblem(
        pricer_ws,
        t1,
        t2,
        alpha,
        beta,
        blocked_leaves,
        &free_labels,
        &[],
        required_bonus,
    )? {
        frontier.push(root);
    }

    while let Some(candidate) = frontier.pop() {
        if !forbidden.contains(&candidate.labels) {
            return Ok(Some((candidate.score, candidate.labels)));
        }

        let membership = membership_over_free_labels(&free_labels, &candidate.labels);
        let fixed_prefix_len = candidate.prefix_membership.len();
        for split_idx in fixed_prefix_len..free_labels.len() {
            let mut child_prefix = candidate.prefix_membership.clone();
            child_prefix.extend_from_slice(&membership[fixed_prefix_len..split_idx]);
            child_prefix.push(!membership[split_idx]);
            if let Some(child) = solve_pricing_prefix_subproblem(
                pricer_ws,
                t1,
                t2,
                alpha,
                beta,
                blocked_leaves,
                &free_labels,
                &child_prefix,
                required_bonus,
            )? {
                frontier.push(child);
            }
        }
    }

    Ok(None)
}

fn solve_pricing_prefix_subproblem(
    pricer_ws: &mut PricerWorkspace,
    t1: &Tree,
    t2: &Tree,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    free_labels: &[u32],
    prefix_membership: &[bool],
    required_bonus: f64,
) -> Result<Option<PricingPrefixNode>, Box<dyn std::error::Error>> {
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

    let priced = run_rooted_paper_pricer(pricer_ws, t1, t2, &alpha_mod, beta, &blocked)?;
    let Some((score_with_bonus, labels)) = priced else {
        return Ok(None);
    };

    for (idx, &present) in prefix_membership.iter().enumerate() {
        let label = free_labels[idx];
        let actually_present = labels.binary_search(&label).is_ok();
        if actually_present != present {
            return Ok(None);
        }
    }

    Ok(Some(PricingPrefixNode {
        prefix_membership: prefix_membership.to_vec(),
        score: score_with_bonus - required_bonus * required_count as f64,
        labels,
    }))
}

fn membership_over_free_labels(free_labels: &[u32], labels: &[u32]) -> Vec<bool> {
    let mut membership = vec![false; free_labels.len()];
    let mut i = 0usize;
    let mut j = 0usize;
    while i < free_labels.len() && j < labels.len() {
        match free_labels[i].cmp(&labels[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                membership[i] = true;
                i += 1;
                j += 1;
            }
        }
    }
    membership
}

// ---------------------------------------------------------------------------
// Tree helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// PricerWorkspace: reusable DP tables + structural metadata
// ---------------------------------------------------------------------------
// Cache structure, not scores. Dual values (alpha/beta) change every LP solve,
// so full DP value caching across iterations is useless. But post-order lists,
// node type classification, and the DP table buffers themselves are stable.

const NEG_INF: f64 = -1.0e100;

pub(crate) struct PricerWorkspace {
    // DP tables (flat, indexed as u * n2 + v)
    v_score: Vec<f64>,
    v_choice: Vec<VChoice>,
    m_score: Vec<f64>,
    m_choice: Vec<MChoice>,
    split_choice: Vec<SplitChoice>,
    // Structural metadata (computed once, reused)
    leaves1: Vec<u32>,
    leaves2: Vec<u32>,
    internal1: Vec<u32>,
    internal2: Vec<u32>,
    // Scratch buffers (reused, not reallocated)
    live1: Vec<u32>,  // unblocked leaf count per node in t1
    live2: Vec<u32>,  // unblocked leaf count per node in t2
    n2: usize,
    table_size: usize,
}

impl PricerWorkspace {
    pub(crate) fn new(t1: &Tree, t2: &Tree) -> Self {
        let n2 = t2.num_nodes();
        let table_size = t1.num_nodes() * n2;
        let post1 = t1.post_order_vec();
        let post2 = t2.post_order_vec();

        // Segregate nodes by type for branchless DP phases
        let mut leaves1 = Vec::new();
        let mut internal1 = Vec::new();
        for &u in &post1 {
            if t1.is_leaf(u) {
                leaves1.push(u);
            } else {
                internal1.push(u);
            }
        }
        let mut leaves2 = Vec::new();
        let mut internal2 = Vec::new();
        for &v in &post2 {
            if t2.is_leaf(v) {
                leaves2.push(v);
            } else {
                internal2.push(v);
            }
        }

        Self {
            v_score: vec![NEG_INF; table_size],
            v_choice: vec![VChoice::None; table_size],
            m_score: vec![0.0; table_size],
            m_choice: vec![MChoice::None; table_size],
            split_choice: vec![SplitChoice::None; table_size],
            leaves1,
            leaves2,
            internal1,
            internal2,
            live1: vec![0; t1.num_nodes()],
            live2: vec![0; t2.num_nodes()],
            n2,
            table_size,
        }
    }

    /// Reset score tables before a fresh DP fill. Only touches the region we use.
    #[inline(always)]
    fn reset_scores(&mut self) {
        for s in &mut self.v_score[..self.table_size] {
            *s = NEG_INF;
        }
        for s in &mut self.m_score[..self.table_size] {
            *s = 0.0;
        }
        for c in &mut self.split_choice[..self.table_size] {
            *c = SplitChoice::None;
        }
    }

    #[inline(always)]
    fn idx(&self, u: u32, v: u32) -> usize {
        u as usize * self.n2 + v as usize
    }

    /// Recompute live-leaf counts for the current blocked set.
    /// Returns true if the root pair has any live leaves at all.
    fn compute_live_counts(&mut self, t1: &Tree, t2: &Tree, blocked: &[bool]) -> bool {
        // t1: leaves first (already in post-order, leaves come before internals)
        for &u in &self.leaves1 {
            let lbl = t1.label[u as usize];
            self.live1[u as usize] = if blocked.get(lbl as usize).copied().unwrap_or(false) { 0 } else { 1 };
        }
        for &u in &self.internal1 {
            let (ul, ur) = t1.children_pair(u);
            self.live1[u as usize] = self.live1[ul as usize] + self.live1[ur as usize];
        }
        // t2
        for &v in &self.leaves2 {
            let lbl = t2.label[v as usize];
            self.live2[v as usize] = if blocked.get(lbl as usize).copied().unwrap_or(false) { 0 } else { 1 };
        }
        for &v in &self.internal2 {
            let (vl, vr) = t2.children_pair(v);
            self.live2[v as usize] = self.live2[vl as usize] + self.live2[vr as usize];
        }
        self.live1[t1.root as usize] > 0 && self.live2[t2.root as usize] > 0
    }
}

// ---------------------------------------------------------------------------
// WMAST pricer: O(n²) DP with workspace reuse + segregated loops
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum SplitChoice {
    None,
    Straight,
    Cross,
}

#[derive(Clone, Copy)]
enum VChoice {
    None,
    LeafMatch(u32),
    UseRooted,
    SkipLeftU,
    SkipRightU,
    SkipLeftV,
    SkipRightV,
}

#[derive(Clone, Copy)]
enum MChoice {
    None,
    LeafMatch(u32),
    UseRooted,
    SkipLeftU,
    SkipRightU,
    SkipLeftV,
    SkipRightV,
}

/// Run the rooted WMAST pricer using a reusable workspace.
/// The workspace avoids re-allocating O(n²) tables on every call.
pub(crate) fn run_rooted_paper_pricer(
    ws: &mut PricerWorkspace,
    t1: &Tree,
    t2: &Tree,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
) -> Result<Option<(f64, Vec<u32>)>, Box<dyn std::error::Error>> {
    ws.reset_scores();

    // Phase 0: compute live-leaf counts for the current blocked set.
    // If no live leaves remain in either tree, bail out immediately.
    if !ws.compute_live_counts(t1, t2, blocked_leaves) {
        return Ok(None);
    }

    let idx = |u: u32, v: u32| -> usize { u as usize * ws.n2 + v as usize };

    // Phase 1: leaf vs leaf — only matching labels produce non-trivial scores
    for &u in &ws.leaves1 {
        let lbl = t1.label[u as usize];
        let blocked = blocked_leaves.get(lbl as usize).copied().unwrap_or(false);
        for &v in &ws.leaves2 {
            let pair = idx(u, v);
            if t2.label[v as usize] != lbl || blocked {
                // v_score stays NEG_INF, m_score stays 0.0
                continue;
            }
            let score = alpha[lbl as usize];
            ws.v_score[pair] = score;
            ws.v_choice[pair] = VChoice::LeafMatch(lbl);
            ws.m_score[pair] = score.max(0.0);
            ws.m_choice[pair] = if score > 0.0 {
                MChoice::LeafMatch(lbl)
            } else {
                MChoice::None
            };
        }
    }

    // Phase 2: leaf vs internal — skip-left/skip-right for V, max-propagation for M
    for &u in &ws.leaves1 {
        let lbl = t1.label[u as usize];
        let blocked = blocked_leaves.get(lbl as usize).copied().unwrap_or(false);
        for &v in &ws.internal2 {
            let pair = idx(u, v);
            // Skip if t2 subtree has no live leaves
            if ws.live2[v as usize] == 0 {
                continue;
            }
            let (vl, vr) = t2.children_pair(v);
            let beta_v = beta[1][v as usize];

            // V: can only skip in t2 (t1 is a leaf, can't split)
            let left = -beta_v + ws.v_score[idx(u, vl)];
            let right = -beta_v + ws.v_score[idx(u, vr)];
            if left >= right {
                ws.v_score[pair] = left;
                ws.v_choice[pair] = VChoice::SkipLeftV;
            } else {
                ws.v_score[pair] = right;
                ws.v_choice[pair] = VChoice::SkipRightV;
            }

            // M: propagate best from children (no rooted option since t1 is leaf)
            let ml = ws.m_score[idx(u, vl)];
            let mr = ws.m_score[idx(u, vr)];
            if ml >= mr && ml > 0.0 {
                ws.m_score[pair] = ml;
                ws.m_choice[pair] = ws.m_choice[idx(u, vl)];
            } else if mr > 0.0 {
                ws.m_score[pair] = mr;
                ws.m_choice[pair] = ws.m_choice[idx(u, vr)];
            } else {
                ws.m_score[pair] = 0.0;
                ws.m_choice[pair] = MChoice::None;
            }
        }
    }

    // Phase 3: internal vs leaf — symmetric to phase 2
    for &u in &ws.internal1 {
        // Skip if t1 subtree has no live leaves
        if ws.live1[u as usize] == 0 {
            continue;
        }
        let beta_u = beta[0][u as usize];
        for &v in &ws.leaves2 {
            let pair = idx(u, v);
            let (ul, ur) = t1.children_pair(u);
            let lbl = t2.label[v as usize];
            let blocked = blocked_leaves.get(lbl as usize).copied().unwrap_or(false);

            // V: can only skip in t1
            let left = -beta_u + ws.v_score[idx(ul, v)];
            let right = -beta_u + ws.v_score[idx(ur, v)];
            if left >= right {
                ws.v_score[pair] = left;
                ws.v_choice[pair] = VChoice::SkipLeftU;
            } else {
                ws.v_score[pair] = right;
                ws.v_choice[pair] = VChoice::SkipRightU;
            }

            // M: propagate best from children
            let ml = ws.m_score[idx(ul, v)];
            let mr = ws.m_score[idx(ur, v)];
            if ml >= mr && ml > 0.0 {
                ws.m_score[pair] = ml;
                ws.m_choice[pair] = ws.m_choice[idx(ul, v)];
            } else if mr > 0.0 {
                ws.m_score[pair] = mr;
                ws.m_choice[pair] = ws.m_choice[idx(ur, v)];
            } else {
                ws.m_score[pair] = 0.0;
                ws.m_choice[pair] = MChoice::None;
            }
        }
    }

    // Phase 4: internal vs internal — the heavy part with straight/cross splits
    for &u in &ws.internal1 {
        // Skip entire row if t1 subtree has no live leaves
        if ws.live1[u as usize] == 0 {
            continue;
        }
        let beta_u = beta[0][u as usize];
        let (ul, ur) = t1.children_pair(u);
        for &v in &ws.internal2 {
            // Skip if t2 subtree has no live leaves
            if ws.live2[v as usize] == 0 {
                continue;
            }
            let pair = idx(u, v);
            let (vl, vr) = t2.children_pair(v);
            let beta_v = beta[1][v as usize];

            // W: straight vs cross split
            let straight = ws.v_score[idx(ul, vl)] + ws.v_score[idx(ur, vr)];
            let cross = ws.v_score[idx(ul, vr)] + ws.v_score[idx(ur, vl)];
            let (best_split, split_pick) = if straight >= cross {
                (straight, SplitChoice::Straight)
            } else {
                (cross, SplitChoice::Cross)
            };
            ws.split_choice[pair] = if best_split > NEG_INF / 2.0 {
                split_pick
            } else {
                SplitChoice::None
            };

            // V: rooted split + skip candidates
            let rooted = if best_split > NEG_INF / 2.0 {
                -beta_u - beta_v + best_split
            } else {
                NEG_INF
            };

            let mut best_v = rooted;
            let mut best_v_choice = VChoice::UseRooted;

            let skip_left_u = -beta_u + ws.v_score[idx(ul, v)];
            if skip_left_u > best_v {
                best_v = skip_left_u;
                best_v_choice = VChoice::SkipLeftU;
            }
            let skip_right_u = -beta_u + ws.v_score[idx(ur, v)];
            if skip_right_u > best_v {
                best_v = skip_right_u;
                best_v_choice = VChoice::SkipRightU;
            }
            let skip_left_v = -beta_v + ws.v_score[idx(u, vl)];
            if skip_left_v > best_v {
                best_v = skip_left_v;
                best_v_choice = VChoice::SkipLeftV;
            }
            let skip_right_v = -beta_v + ws.v_score[idx(u, vr)];
            if skip_right_v > best_v {
                best_v = skip_right_v;
                best_v_choice = VChoice::SkipRightV;
            }

            ws.v_score[pair] = best_v;
            ws.v_choice[pair] = if best_v > NEG_INF / 2.0 {
                best_v_choice
            } else {
                VChoice::None
            };

            // M: rooted split + skip candidates (no beta penalty)
            let mut best_m = 0.0;
            let mut best_m_choice = MChoice::None;

            if rooted > best_m {
                best_m = rooted;
                best_m_choice = MChoice::UseRooted;
            }
            let ml_u = ws.m_score[idx(ul, v)];
            if ml_u > best_m {
                best_m = ml_u;
                best_m_choice = MChoice::SkipLeftU;
            }
            let mr_u = ws.m_score[idx(ur, v)];
            if mr_u > best_m {
                best_m = mr_u;
                best_m_choice = MChoice::SkipRightU;
            }
            let ml_v = ws.m_score[idx(u, vl)];
            if ml_v > best_m {
                best_m = ml_v;
                best_m_choice = MChoice::SkipLeftV;
            }
            let mr_v = ws.m_score[idx(u, vr)];
            if mr_v > best_m {
                best_m = mr_v;
                best_m_choice = MChoice::SkipRightV;
            }

            ws.m_score[pair] = best_m;
            ws.m_choice[pair] = best_m_choice;
        }
    }

    let root_score = ws.m_score[idx(t1.root, t2.root)];
    // NOTE: Paper Proposition 1 adds max(0, Σ_{v∈P1(ρ)∪P2(ρ)} w(v)) to M(root,root).
    // We omit this because our column model (mark_component_nodes) only charges nodes
    // from selected leaves up to their LCA — the path above the component root is
    // intentionally excluded. M(root,root) directly gives the rooted component score.
    if root_score <= 1e-9 {
        return Ok(None);
    }

    let mut labels = Vec::new();
    collect_m_labels(
        t1,
        t2,
        &ws.m_choice,
        &ws.v_choice,
        &ws.split_choice,
        t1.root,
        t2.root,
        &mut labels,
    );
    labels.sort_unstable();
    labels.dedup();
    Ok(Some((root_score, labels)))
}

fn collect_m_labels(
    t1: &Tree,
    t2: &Tree,
    m_choice: &[MChoice],
    v_choice: &[VChoice],
    split_choice: &[SplitChoice],
    u: u32,
    v: u32,
    out: &mut Vec<u32>,
) {
    let n2 = t2.num_nodes();
    let pair = u as usize * n2 + v as usize;
    match m_choice[pair] {
        MChoice::None => {}
        MChoice::LeafMatch(lbl) => out.push(lbl),
        MChoice::UseRooted => collect_split_labels(t1, t2, v_choice, split_choice, u, v, out),
        MChoice::SkipLeftU => {
            let (ul, _) = t1.children(u).expect("skip-left-u requires internal u");
            collect_m_labels(t1, t2, m_choice, v_choice, split_choice, ul, v, out);
        }
        MChoice::SkipRightU => {
            let (_, ur) = t1.children(u).expect("skip-right-u requires internal u");
            collect_m_labels(t1, t2, m_choice, v_choice, split_choice, ur, v, out);
        }
        MChoice::SkipLeftV => {
            let (vl, _) = t2.children(v).expect("skip-left-v requires internal v");
            collect_m_labels(t1, t2, m_choice, v_choice, split_choice, u, vl, out);
        }
        MChoice::SkipRightV => {
            let (_, vr) = t2.children(v).expect("skip-right-v requires internal v");
            collect_m_labels(t1, t2, m_choice, v_choice, split_choice, u, vr, out);
        }
    }
}

fn collect_v_labels(
    t1: &Tree,
    t2: &Tree,
    v_choice: &[VChoice],
    split_choice: &[SplitChoice],
    u: u32,
    v: u32,
    out: &mut Vec<u32>,
) {
    let n2 = t2.num_nodes();
    let pair = u as usize * n2 + v as usize;
    match v_choice[pair] {
        VChoice::None => {}
        VChoice::LeafMatch(lbl) => out.push(lbl),
        VChoice::UseRooted => collect_split_labels(t1, t2, v_choice, split_choice, u, v, out),
        VChoice::SkipLeftU => {
            let (ul, _) = t1.children(u).expect("skip-left-u requires internal u");
            collect_v_labels(t1, t2, v_choice, split_choice, ul, v, out);
        }
        VChoice::SkipRightU => {
            let (_, ur) = t1.children(u).expect("skip-right-u requires internal u");
            collect_v_labels(t1, t2, v_choice, split_choice, ur, v, out);
        }
        VChoice::SkipLeftV => {
            let (vl, _) = t2.children(v).expect("skip-left-v requires internal v");
            collect_v_labels(t1, t2, v_choice, split_choice, u, vl, out);
        }
        VChoice::SkipRightV => {
            let (_, vr) = t2.children(v).expect("skip-right-v requires internal v");
            collect_v_labels(t1, t2, v_choice, split_choice, u, vr, out);
        }
    }
}

fn collect_split_labels(
    t1: &Tree,
    t2: &Tree,
    v_choice: &[VChoice],
    split_choice: &[SplitChoice],
    u: u32,
    v: u32,
    out: &mut Vec<u32>,
) {
    let n2 = t2.num_nodes();
    let pair = u as usize * n2 + v as usize;
    match split_choice[pair] {
        SplitChoice::None => {}
        SplitChoice::Straight => {
            let (ul, ur) = t1.children(u).expect("straight split requires internal u");
            let (vl, vr) = t2.children(v).expect("straight split requires internal v");
            collect_v_labels(t1, t2, v_choice, split_choice, ul, vl, out);
            collect_v_labels(t1, t2, v_choice, split_choice, ur, vr, out);
        }
        SplitChoice::Cross => {
            let (ul, ur) = t1.children(u).expect("cross split requires internal u");
            let (vl, vr) = t2.children(v).expect("cross split requires internal v");
            collect_v_labels(t1, t2, v_choice, split_choice, ul, vr, out);
            collect_v_labels(t1, t2, v_choice, split_choice, ur, vl, out);
        }
    }
}
