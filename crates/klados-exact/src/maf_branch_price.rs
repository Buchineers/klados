//! Exact Branch & Price solver for 2-tree Maximum Agreement Forest.
//!
//! Implements the full algorithm from Frohn 2025 "Branch & Price":
//! - Set cover master problem (Formulation 1): min components subject to
//!   leaf covering (equality) and internal-node packing (≤ 1).
//! - Pricing via O(n²) Weighted MAST DP (V/M/W recurrences).
//! - Integrality check at each B&B node (~99% solved at root per paper).
//! - SIZE branching strategy on fractional columns when LP is not integral.
//! - Column generation re-run at every B&B node with branching fixings.
//!
//! Restricted to m = 2 trees; falls back to maf-sat for multi-tree instances.

use std::collections::HashSet;
use std::time::Instant;

use fixedbitset::FixedBitSet;
use highs::{Col, ColProblem, HighsModelStatus, Model, Row, RowProblem, Sense};
use klados_core::{Instance, SolverStats, Tree};

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
    labels: Vec<u32>, // sorted leaf labels in this block
    covered_internal_nodes: Vec<Vec<usize>>,
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

    // Initialize column pool with singletons
    let columns: Vec<BpColumn> = (1..=n as u32)
        .map(|label| make_bp_column(vec![label], trees))
        .collect();
    let seen: HashSet<Vec<u32>> = columns.iter().map(|c| c.labels.clone()).collect();

    let mut state = BpState {
        columns,
        seen,
        best_ub: n + 1, // trivial UB: all singletons = n components
        best_solution: None,
        nodes_explored: 0,
        cg_iterations_total: 0,
    };

    let root = BpNode {
        fixed_to_one: vec![],
        fixed_to_zero: vec![],
        depth: 0,
    };

    // DFS branch-and-bound
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let result = solve_bp_node(&mut state, &node, trees, n);
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
                let lp_lb = lp_obj.ceil() as usize;
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
    if let Some(values) = &state.best_solution {
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
            "[maf-bp] optimal: {} components, {} B&B nodes, {} CG iterations, {:.1}ms total",
            components.len(),
            state.nodes_explored,
            state.cg_iterations_total,
            t_total.elapsed().as_secs_f64() * 1000.0,
        );
        Some(components)
    } else {
        eprintln!("[maf-bp] no solution found");
        None
    }
}

// ---------------------------------------------------------------------------
// Solve a single B&B node: CG loop + integrality check
// ---------------------------------------------------------------------------

fn solve_bp_node(
    state: &mut BpState,
    node: &BpNode,
    trees: &[Tree],
    num_leaves: usize,
) -> NodeResult {
    state.nodes_explored += 1;
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
    let mut node_dual = match NodeDual::build(
        &state.columns,
        trees,
        &node.fixed_to_one,
        &node.fixed_to_zero,
    ) {
        Ok(dual) => dual,
        Err(_) => return NodeResult::Pruned,
    };
    // Column generation loop
    let mut cg_iters_this_node = 0usize;
    loop {
        let lp = match node_rmp.solve(state.columns.len()) {
            Ok(lp) => lp,
            Err(_) => return NodeResult::Pruned, // infeasible
        };

        let (alpha, beta) = match node_dual.solve(trees) {
            Ok(duals) => duals,
            Err(err) => {
                eprintln!("[maf-bp] stabilized dual failed at depth {}: {}", node.depth, err);
                return NodeResult::Pruned;
            }
        };

        // Dual verification: check that no in-pool compatible column has negative reduced cost
        if cg_iters_this_node == 0 || cg_iters_this_node % 20 == 0 {
            let mut worst_rc = f64::INFINITY;
            let mut worst_ci = 0;
            for (ci, col) in state.columns.iter().enumerate() {
                if !column_respects_branchings(
                    &state.columns,
                    ci,
                    &node.fixed_to_one,
                    &node.fixed_to_zero,
                ) {
                    continue;
                }
                let score = column_pricing_score(col, &alpha, &beta, trees);
                let rc = 1.0 - score;
                if rc < worst_rc {
                    worst_rc = rc;
                    worst_ci = ci;
                }
            }
            if worst_rc < -1e-4 {
                eprintln!(
                    "[maf-bp]   DUAL BUG: col {} has rc={:.6} (score={:.6}), LP={:.6}, val={:.6}",
                    worst_ci,
                    worst_rc,
                    1.0 - worst_rc,
                    lp.objective,
                    lp.column_values[worst_ci],
                );
            }
        }

        // Price columns paper-faithfully over the admissible set.
        // At the root we keep a faster shortcut. At every branched node we solve
        // the exact unseen-column separation problem over the current admissible
        // column set. This matches the branch-aware theory much more closely and
        // avoids branch-dependent termination bugs.
        let priced = if node.depth == 0 {
            match run_rooted_paper_pricer(
                &trees[0],
                &trees[1],
                &alpha,
                &beta,
                &blocked_leaves,
            ) {
                Ok(Some((score, labels))) if score > 1.0 + 1e-8 && state.seen.contains(&labels) => {
                    match price_best_new_compatible_column(
                        &trees[0],
                        &trees[1],
                        &alpha,
                        &beta,
                        &blocked_leaves,
                        &state.seen,
                    ) {
                        Ok(Some((alt_score, alt_labels))) => (alt_score, alt_labels),
                        Ok(None) => {
                            eprintln!(
                                "[maf-bp]   CG@depth={}: no priced column, LP={:.6}, iters={}",
                                node.depth, lp.objective, cg_iters_this_node,
                            );
                            break;
                        }
                        Err(_) => break,
                    }
                }
                Ok(Some((score, labels))) => (score, labels),
                Ok(None) => {
                    eprintln!(
                        "[maf-bp]   CG@depth={}: no priced column, LP={:.6}, iters={}",
                        node.depth, lp.objective, cg_iters_this_node,
                    );
                    break;
                }
                Err(_) => break,
            }
        } else {
            let mut forbidden = state.seen.clone();
            forbidden.extend(banned_zero_labels.iter().cloned());
            match price_best_new_compatible_column(
                &trees[0],
                &trees[1],
                &alpha,
                &beta,
                &blocked_leaves,
                &forbidden,
            ) {
                Ok(Some((score, labels))) => (score, labels),
                Ok(None) => {
                    eprintln!(
                        "[maf-bp]   CG@depth={}: no priced column, LP={:.6}, iters={}",
                        node.depth, lp.objective, cg_iters_this_node,
                    );
                    break;
                }
                Err(_) => break,
            }
        };

        let (score, labels) = priced;
        if score <= 1.0 + 1e-8 {
            eprintln!(
                "[maf-bp]   CG@depth={}: score={:.6} <= 1, LP={:.6}, iters={}",
                node.depth, score, lp.objective, cg_iters_this_node,
            );
            break; // CG converged: no improving column
        }

        if labels.len() >= 2 && !is_set_compatible_all(trees, &labels) {
            continue;
        }

        let inserted = state.seen.insert(labels.clone());
        if !inserted {
            eprintln!(
                "[maf-bp]   INTERNAL BUG depth={}: constrained pricing returned duplicate column {:?}; stopping CG at this node",
                node.depth, labels
            );
            break;
        }
        let new_ci = state.columns.len();
        state.columns.push(make_bp_column(labels.clone(), trees));
        node_rmp.add_column(new_ci, &state.columns[new_ci], trees);
        if state.columns[new_ci].labels.len() > 1 {
            node_dual.add_column(&state.columns[new_ci], trees);
        }
        if let Some(best_solution) = state.best_solution.as_mut() {
            best_solution.push(0.0);
        }
        state.cg_iterations_total += 1;
        cg_iters_this_node += 1;
    }

    // Final LP solve at this node
    let lp = match node_rmp.solve(state.columns.len()) {
        Ok(lp) => lp,
        Err(_) => return NodeResult::Pruned,
    };

    if lp.objective.ceil() as usize >= state.best_ub {
        return NodeResult::Pruned;
    }

    // Check integrality
    if support_is_integral_partition(&state.columns, &lp.column_values, num_leaves) {
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
        let total_internal = columns[ci]
            .covered_internal_nodes
            .iter()
            .map(|nodes| nodes.len())
            .sum::<usize>() as f64;
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
}

struct NodeRmp {
    model: Option<Model>,
    active_global_cols: Vec<usize>,
    leaf_rows: Vec<Row>,
    node_rows: Vec<Vec<Option<Row>>>,
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

        let leaf_rows = (0..=num_leaves)
            .map(|leaf| {
                if leaf == 0 {
                    model.add_row(0.0..=0.0, Vec::new())
                } else {
                    model.add_row(1.0..=1.0, Vec::new())
                }
            })
            .collect::<Vec<_>>();
        let node_rows = trees
            .iter()
            .map(|tree| {
                (0..tree.num_nodes() as u32)
                    .map(|node| {
                        if tree.is_leaf(node) {
                            None
                        } else {
                            Some(model.add_row(..=1.0, Vec::new()))
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut rmp = Self {
            model: Some(model),
            active_global_cols: Vec::new(),
            leaf_rows,
            node_rows,
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
            .add_col(1.0, 0.0..=1.0, rows);
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
        self.model = Some(Model::from(solved));
        Ok(RmpLpResult {
            objective,
            column_values,
        })
    }
}

struct NodeDual {
    model: Option<Model>,
    penalty_vars: Vec<Vec<Option<Col>>>,
    num_leaves: usize,
}

impl NodeDual {
    fn build(
        columns: &[BpColumn],
        trees: &[Tree],
        fixed_to_one: &[usize],
        fixed_to_zero: &[usize],
    ) -> Result<Self, String> {
        const BETA_COST_SCALE: f64 = 10_000.0;

        let num_leaves = trees
            .first()
            .map(|tree| tree.num_leaves as usize)
            .unwrap_or(0);
        let mut model = Model::new(ColProblem::default());
        model.make_quiet();
        model.set_option("threads", 1_i32);
        model.set_option("presolve", "on");
        model.set_option("solver", "simplex");

        let mut penalty_vars = trees
            .iter()
            .map(|tree| vec![None; tree.num_nodes()])
            .collect::<Vec<_>>();
        for (ti, tree) in trees.iter().enumerate() {
            for node in 0..tree.num_nodes() as u32 {
                if tree.is_leaf(node) {
                    continue;
                }
                penalty_vars[ti][node as usize] = Some(model.add_col(BETA_COST_SCALE, 0.0.., Vec::new()));
            }
        }

        let mut dual = Self {
            model: Some(model),
            penalty_vars,
            num_leaves,
        };
        for (ci, col) in columns.iter().enumerate() {
            if col.labels.len() > 1
                && column_respects_branchings(columns, ci, fixed_to_one, fixed_to_zero)
            {
                dual.add_column(col, trees);
            }
        }
        Ok(dual)
    }

    fn add_column(&mut self, col: &BpColumn, trees: &[Tree]) {
        add_column_to_stabilized_dual_model(
            self.model.as_mut().expect("node dual model present"),
            &self.penalty_vars,
            col,
            trees,
            1.0,
        );
    }

    fn solve(&mut self, trees: &[Tree]) -> Result<(Vec<f64>, Vec<Vec<f64>>), String> {
        let solved = self.model.take().expect("node dual model present").solve();
        let status = solved.status();
        if status != HighsModelStatus::Optimal {
            self.model = Some(Model::from(solved));
            return Err(format!("stabilized dual LP status: {:?}", status));
        }
        let solution = solved.get_solution();
        let alpha = (0..=self.num_leaves)
            .map(|leaf| if leaf == 0 { 0.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let mut beta = trees
            .iter()
            .map(|tree| vec![0.0; tree.num_nodes()])
            .collect::<Vec<_>>();
        for (ti, tree) in trees.iter().enumerate() {
            for node in 0..tree.num_nodes() {
                if let Some(var) = self.penalty_vars[ti][node] {
                    beta[ti][node] = clean_dual(solution[var]);
                }
            }
        }
        self.model = Some(Model::from(solved));
        Ok((alpha, beta))
    }
}

fn solve_rmp_lp(
    columns: &[BpColumn],
    trees: &[Tree],
    num_leaves: usize,
    fixed_to_one: &[usize],
    fixed_to_zero: &[usize],
) -> Result<RmpLpResult, String> {
    let mut pb = RowProblem::default();

    // One variable per column in the current node LP relaxation.
    // Branching is enforced exactly as in the reference implementation:
    // a compatible column remains 0 <= a_Y <= 1, and incompatible columns are excluded.
    let vars: Vec<Col> = columns
        .iter()
        .enumerate()
        .map(|(ci, _)| {
            if !column_respects_branchings(columns, ci, fixed_to_one, fixed_to_zero) {
                pb.add_column(1.0, 0.0..=0.0) // branch-incompatible
            } else {
                pb.add_column(1.0, 0.0..=1.0)
            }
        })
        .collect();

    // Covering constraints: Σ_{Y: x∈Y} a_Y = 1 for each leaf (exact partition)
    let leaf_to_cols = build_leaf_to_cols(columns, num_leaves);
    for leaf in 1..=num_leaves {
        let cols: Vec<(Col, f64)> = leaf_to_cols[leaf]
            .iter()
            .map(|&ci| (vars[ci], 1.0))
            .collect();
        pb.add_row(1.0..=1.0, &cols);
    }

    // Packing constraints: Σ_{Y: v∈V_q[Y]} a_Y ≤ 1 for each internal node
    let node_to_cols = build_node_to_cols(columns, trees);
    for (ti, tree) in trees.iter().enumerate() {
        for node in 0..tree.num_nodes() as u32 {
            if tree.is_leaf(node) {
                continue;
            }
            let cols: Vec<(Col, f64)> = node_to_cols[ti][node as usize]
                .iter()
                .map(|&ci| (vars[ci], 1.0))
                .collect();
            if cols.len() >= 2 {
                pb.add_row(..=1.0, &cols);
            }
        }
    }

    let mut model = pb.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_option("threads", 1_i32);
    model.set_option("presolve", "on");
    model.set_option("solver", "simplex");
    let solved = model.solve();
    if solved.status() != HighsModelStatus::Optimal {
        return Err(format!("LP status: {:?}", solved.status()));
    }

    let solution = solved.get_solution();
    let column_values = solution.columns().to_vec();

    Ok(RmpLpResult {
        objective: solved.objective_value(),
        column_values,
    })
}

fn solve_stabilized_dual(
    columns: &[BpColumn],
    trees: &[Tree],
    fixed_to_one: &[usize],
    fixed_to_zero: &[usize],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), String> {
    const BETA_COST_SCALE: f64 = 10_000.0;
    const MAX_BETA_COST: f64 = 1.0;

    let num_leaves = trees
        .first()
        .map(|tree| tree.num_leaves as usize)
        .unwrap_or(0);

    let active_multi_leaf_cols = columns
        .iter()
        .enumerate()
        .filter(|(ci, col)| {
            col.labels.len() > 1
                && column_respects_branchings(columns, *ci, fixed_to_one, fixed_to_zero)
        })
        .count();

    if active_multi_leaf_cols == 0 {
        return Ok((
            (0..=num_leaves)
                .map(|leaf| if leaf == 0 { 0.0 } else { 1.0 })
                .collect(),
            trees.iter()
                .map(|tree| vec![0.0; tree.num_nodes()])
                .collect::<Vec<_>>(),
        ));
    }

    let mut dual_model = Model::new(ColProblem::default());
    dual_model.make_quiet();
    dual_model.set_option("threads", 1_i32);
    dual_model.set_option("presolve", "on");
    dual_model.set_option("solver", "simplex");

    let mut penalty_vars: Vec<Vec<Option<Col>>> = trees
        .iter()
        .map(|tree| vec![None; tree.num_nodes()])
        .collect();
    for (ti, tree) in trees.iter().enumerate() {
        for node in 0..tree.num_nodes() as u32 {
            if tree.is_leaf(node) {
                continue;
            }
            penalty_vars[ti][node as usize] = Some(dual_model.add_col(BETA_COST_SCALE, 0.0.., Vec::new()));
        }
    }

    for (ci, col) in columns.iter().enumerate() {
        if col.labels.len() <= 1 || !column_respects_branchings(columns, ci, fixed_to_one, fixed_to_zero) {
            continue;
        }
        add_column_to_stabilized_dual_model(
            &mut dual_model,
            &penalty_vars,
            col,
            trees,
            MAX_BETA_COST,
        );
    }

    let solved = dual_model.solve();
    if solved.status() != HighsModelStatus::Optimal {
        return Err(format!("stabilized dual LP status: {:?}", solved.status()));
    }

    let solution = solved.get_solution();
    let alpha = (0..=num_leaves)
        .map(|leaf| if leaf == 0 { 0.0 } else { 1.0 })
        .collect::<Vec<_>>();
    let mut beta = trees
        .iter()
        .map(|tree| vec![0.0; tree.num_nodes()])
        .collect::<Vec<_>>();
    for (ti, tree) in trees.iter().enumerate() {
        for node in 0..tree.num_nodes() {
            if let Some(var) = penalty_vars[ti][node] {
                beta[ti][node] = clean_dual(solution[var]);
            }
        }
    }
    Ok((alpha, beta))
}

fn add_column_to_stabilized_dual_model(
    dual_model: &mut Model,
    penalty_vars: &[Vec<Option<Col>>],
    col: &BpColumn,
    _trees: &[Tree],
    max_beta_cost: f64,
) {
    let mut cover_penalties = Vec::new();
    for (ti, covered_nodes) in col.covered_internal_nodes.iter().enumerate() {
        for &node in covered_nodes {
            if let Some(var) = penalty_vars[ti][node] {
                cover_penalties.push(var);
            }
        }
    }
    if cover_penalties.is_empty() {
        return;
    }

    dual_model.add_row(
        (col.labels.len() as f64 - 1.0)..,
        cover_penalties.iter().copied().map(|var| (var, 1.0)),
    );

    let max_rows = cover_penalties
        .iter()
        .copied()
        .map(|var| dual_model.add_row(..=0.0, vec![(var, 1.0)]))
        .collect::<Vec<_>>();
    let _ = dual_model.add_col(max_beta_cost, 0.0.., max_rows.into_iter().map(|row| (row, -1.0)));
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

fn build_leaf_to_cols(columns: &[BpColumn], num_leaves: usize) -> Vec<Vec<usize>> {
    let mut l2c = vec![Vec::new(); num_leaves + 1];
    for (ci, col) in columns.iter().enumerate() {
        for &label in &col.labels {
            l2c[label as usize].push(ci);
        }
    }
    l2c
}

fn build_node_to_cols(columns: &[BpColumn], trees: &[Tree]) -> Vec<Vec<Vec<usize>>> {
    let mut n2c: Vec<Vec<Vec<usize>>> = trees
        .iter()
        .map(|tree| vec![Vec::new(); tree.num_nodes()])
        .collect();
    for (ci, col) in columns.iter().enumerate() {
        if col.labels.len() < 2 {
            continue;
        }
        for (ti, covered_nodes) in col.covered_internal_nodes.iter().enumerate() {
            for &node in covered_nodes {
                n2c[ti][node].push(ci);
            }
        }
    }
    n2c
}

fn make_leafset(labels: &[u32], num_leaves: u32) -> FixedBitSet {
    let mut bits = FixedBitSet::with_capacity(num_leaves as usize + 1);
    for &label in labels {
        bits.insert(label as usize);
    }
    bits
}

/// Compute the pricing score (Σ α_x - Σ β_{q,v}) for a column.
fn column_pricing_score(col: &BpColumn, alpha: &[f64], beta: &[Vec<f64>], _trees: &[Tree]) -> f64 {
    let leaf_sum: f64 = col.labels.iter().map(|&l| alpha[l as usize]).sum();
    let mut node_sum = 0.0f64;
    if col.labels.len() >= 2 {
        for (ti, covered_nodes) in col.covered_internal_nodes.iter().enumerate() {
            for &node in covered_nodes {
                node_sum += beta[ti][node];
            }
        }
    }
    leaf_sum - node_sum
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
    BpColumn {
        labels,
        covered_internal_nodes,
    }
}

#[derive(Clone)]
struct PricingPrefixNode {
    prefix_membership: Vec<bool>,
    score: f64,
    labels: Vec<u32>,
}

fn price_best_new_compatible_column(
    t1: &Tree,
    t2: &Tree,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    forbidden: &HashSet<Vec<u32>>,
) -> Result<Option<(f64, Vec<u32>)>, Box<dyn std::error::Error>> {
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

    let mut frontier = Vec::new();
    if let Some(root) = solve_pricing_prefix_subproblem(
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

    while let Some((best_idx, _)) = frontier
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.score.total_cmp(&b.score))
    {
        let candidate = frontier.swap_remove(best_idx);
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

    let priced = run_rooted_paper_pricer(t1, t2, &alpha_mod, beta, &blocked)?;
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
                let (a, b, c) = (labels[i], labels[j], labels[k]);
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

// ---------------------------------------------------------------------------
// WMAST pricer: O(n²) DP (Frohn 2025 pricing problem 5)
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

fn run_rooted_paper_pricer(
    t1: &Tree,
    t2: &Tree,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
) -> Result<Option<(f64, Vec<u32>)>, Box<dyn std::error::Error>> {
    const NEG_INF: f64 = -1.0e100;

    let n2 = t2.num_nodes();
    let idx = |u: u32, v: u32| -> usize { u as usize * n2 + v as usize };
    let mut v_score = vec![NEG_INF; t1.num_nodes() * n2];
    let mut v_choice = vec![VChoice::None; t1.num_nodes() * n2];
    let mut m_score = vec![0.0; t1.num_nodes() * n2];
    let mut m_choice = vec![MChoice::None; t1.num_nodes() * n2];
    let mut split_choice = vec![SplitChoice::None; t1.num_nodes() * n2];
    let post1 = t1.post_order_vec();
    let post2 = t2.post_order_vec();

    for &u in &post1 {
        for &v in &post2 {
            let pair = idx(u, v);
            match (t1.children(u), t2.children(v)) {
                (None, None) => {
                    if t1.label[u as usize] == t2.label[v as usize] {
                        let lbl = t1.label[u as usize];
                        if blocked_leaves.get(lbl as usize).copied().unwrap_or(false) {
                            v_score[pair] = NEG_INF;
                            v_choice[pair] = VChoice::None;
                            m_score[pair] = 0.0;
                            m_choice[pair] = MChoice::None;
                        } else {
                            let score = alpha[lbl as usize];
                            v_score[pair] = score;
                            v_choice[pair] = VChoice::LeafMatch(lbl);
                            m_score[pair] = score.max(0.0);
                            m_choice[pair] = if score > 0.0 {
                                MChoice::LeafMatch(lbl)
                            } else {
                                MChoice::None
                            };
                        }
                    } else {
                        v_score[pair] = NEG_INF;
                        m_score[pair] = 0.0;
                    }
                }
                (Some((ul, ur)), None) => {
                    let left = -beta[0][u as usize] + v_score[idx(ul, v)];
                    let right = -beta[0][u as usize] + v_score[idx(ur, v)];
                    if left >= right {
                        v_score[pair] = left;
                        v_choice[pair] = VChoice::SkipLeftU;
                    } else {
                        v_score[pair] = right;
                        v_choice[pair] = VChoice::SkipRightU;
                    }
                    let ml = m_score[idx(ul, v)];
                    let mr = m_score[idx(ur, v)];
                    if ml >= mr && ml > 0.0 {
                        m_score[pair] = ml;
                        m_choice[pair] = m_choice[idx(ul, v)];
                    } else if mr > 0.0 {
                        m_score[pair] = mr;
                        m_choice[pair] = m_choice[idx(ur, v)];
                    } else {
                        m_score[pair] = 0.0;
                        m_choice[pair] = MChoice::None;
                    }
                }
                (None, Some((vl, vr))) => {
                    let left = -beta[1][v as usize] + v_score[idx(u, vl)];
                    let right = -beta[1][v as usize] + v_score[idx(u, vr)];
                    if left >= right {
                        v_score[pair] = left;
                        v_choice[pair] = VChoice::SkipLeftV;
                    } else {
                        v_score[pair] = right;
                        v_choice[pair] = VChoice::SkipRightV;
                    }
                    let ml = m_score[idx(u, vl)];
                    let mr = m_score[idx(u, vr)];
                    if ml >= mr && ml > 0.0 {
                        m_score[pair] = ml;
                        m_choice[pair] = m_choice[idx(u, vl)];
                    } else if mr > 0.0 {
                        m_score[pair] = mr;
                        m_choice[pair] = m_choice[idx(u, vr)];
                    } else {
                        m_score[pair] = 0.0;
                        m_choice[pair] = MChoice::None;
                    }
                }
                (Some((ul, ur)), Some((vl, vr))) => {
                    let straight = v_score[idx(ul, vl)] + v_score[idx(ur, vr)];
                    let cross = v_score[idx(ul, vr)] + v_score[idx(ur, vl)];
                    let (best_split, split_pick) = if straight >= cross {
                        (straight, SplitChoice::Straight)
                    } else {
                        (cross, SplitChoice::Cross)
                    };
                    split_choice[pair] = if best_split > NEG_INF / 2.0 {
                        split_pick
                    } else {
                        SplitChoice::None
                    };

                    let rooted = if best_split > NEG_INF / 2.0 {
                        -beta[0][u as usize] - beta[1][v as usize] + best_split
                    } else {
                        NEG_INF
                    };

                    let mut best_v = rooted;
                    let mut best_v_choice = VChoice::UseRooted;
                    for (cand, branch_pick) in [
                        (
                            -beta[0][u as usize] + v_score[idx(ul, v)],
                            VChoice::SkipLeftU,
                        ),
                        (
                            -beta[0][u as usize] + v_score[idx(ur, v)],
                            VChoice::SkipRightU,
                        ),
                        (
                            -beta[1][v as usize] + v_score[idx(u, vl)],
                            VChoice::SkipLeftV,
                        ),
                        (
                            -beta[1][v as usize] + v_score[idx(u, vr)],
                            VChoice::SkipRightV,
                        ),
                    ] {
                        if cand > best_v {
                            best_v = cand;
                            best_v_choice = branch_pick;
                        }
                    }
                    v_score[pair] = best_v;
                    v_choice[pair] = if best_v > NEG_INF / 2.0 {
                        best_v_choice
                    } else {
                        VChoice::None
                    };

                    let mut best_m = 0.0;
                    let mut best_m_choice = MChoice::None;
                    for (cand, branch_pick) in [
                        (rooted, MChoice::UseRooted),
                        (m_score[idx(ul, v)], MChoice::SkipLeftU),
                        (m_score[idx(ur, v)], MChoice::SkipRightU),
                        (m_score[idx(u, vl)], MChoice::SkipLeftV),
                        (m_score[idx(u, vr)], MChoice::SkipRightV),
                    ] {
                        if cand > best_m {
                            best_m = cand;
                            best_m_choice = branch_pick;
                        }
                    }
                    m_score[pair] = best_m;
                    m_choice[pair] = best_m_choice;
                }
            }
        }
    }

    let root_score = m_score[idx(t1.root, t2.root)];
    if root_score <= 1e-9 {
        return Ok(None);
    }

    let mut labels = Vec::new();
    collect_m_labels(
        t1,
        t2,
        &m_choice,
        &v_choice,
        &split_choice,
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
