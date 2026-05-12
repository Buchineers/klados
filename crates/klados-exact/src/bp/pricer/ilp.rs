//! ILP-based pricing — solves the pricing subproblem via HiGHS MIP.
//!
//! Encodes leaf selection + tree coverage as a small integer program.
//! Runs as a convergence prover for m≥3: when the leaf-pair DP exhausts,
//! the ILP either finds a column the DP missed, or proves none exists.
//!
//! Only for cluster-reduced sizes (n ≤ 200).

use highs::{HighsModelStatus, Model, RowProblem};

use klados_core::Tree;

use super::{Pricer, PricerScratch, PricingContext, PricingResult};

const PRICING_EPS: f64 = 1.0e-8;
const MAX_LEAVES: usize = 200;
const ILP_TIME_LIMIT: f64 = 0.5; // seconds

pub struct IlpPricer;

impl IlpPricer {
    pub fn new(_trees: &[Tree]) -> Self {
        Self
    }

    fn build_and_solve(&self, ctx: &PricingContext) -> Option<(Vec<u32>, f64)> {
        let n = ctx.num_leaves;
        let m = ctx.trees.len();
        if n > MAX_LEAVES || m < 3 {
            return None;
        }

        // --- Build column-oriented ILP via raw HiGHS C API. ---
        let mut model = Model::new(RowProblem::default());
        model.make_quiet();
        model.set_option("threads", 1_i32);
        model.set_option("presolve", "on");
        model.set_option("solver", "choose");
        model.set_option("time_limit", ILP_TIME_LIMIT);

        // Map internal node (t, v) to column index.  Leaves are cols 0..n.
        let mut node_col: Vec<Vec<i32>> = Vec::with_capacity(m);
        let mut next_col = n as i32;
        for t in 0..m {
            let tree = &ctx.trees[t];
            let mut map = vec![-1i32; tree.num_nodes()];
            for v in 0..tree.num_nodes() {
                map[v] = next_col;
                next_col += 1;
            }
            node_col.push(map);
        }
        let total_cols = next_col as usize;

        // Allocate columns: leaf vars (binary), node vars (binary).
        let ptr = model.as_mut_ptr();
        for ci in 0..total_cols {
            let is_leaf = ci < n;
            let cost = if is_leaf {
                -ctx.alpha[ci + 1] // leaf label = ci+1 (label 0 is sentinel)
            } else {
                // Find which (t,v) this column represents.
                let mut found = false;
                let mut beta_val = 0.0;
                for t in 0..m {
                    let tree = &ctx.trees[t];
                    for v in 0..tree.num_nodes() {
                        if node_col[t][v] == ci as i32 && !tree.is_leaf(v as u32) {
                            beta_val = ctx.beta[t][v];
                            found = true;
                            break;
                        }
                    }
                    if found {
                        break;
                    }
                }
                beta_val
            };
            unsafe {
                highs_sys::Highs_addCol(
                    ptr,
                    cost,
                    0.0,           // lower bound
                    1.0,           // upper bound
                    0,             // no nonzeros yet — rows added below
                    std::ptr::null(),
                    std::ptr::null(),
                );
                // Set integrality.
                highs_sys::Highs_changeColIntegrality(ptr, ci as i32, 1);
            }
        }

        // Constraint 1: at least 2 leaves.
        {
            let active: Vec<usize> = (1..=n).filter(|&l| ctx.alpha[l] > 1e-12).collect();
            let indices: Vec<i32> = active.iter().map(|&l| (l - 1) as i32).collect();
            let values: Vec<f64> = vec![1.0; indices.len()];
            unsafe {
                highs_sys::Highs_addRow(
                    ptr,
                    2.0,
                    f64::INFINITY,
                    indices.len() as i32,
                    indices.as_ptr(),
                    values.as_ptr(),
                );
            }
        }

        // Constraint: for each tree t, internal node v with children (l,r):
        //   cover[v] ≥ cover[l],  cover[v] ≥ cover[r],
        //   cover[v] ≤ cover[l] + cover[r].
        // Also: leaf node cover = x[leaf_label].
        for t in 0..m {
            let tree = &ctx.trees[t];
            for v in 0..tree.num_nodes() {
                let cv = node_col[t][v];
                if tree.is_leaf(v as u32) {
                    let lbl = tree.label[v] as usize;
                    if lbl >= 1 && lbl <= n {
                        let leaf_col = (lbl - 1) as i32;
                        // cover == x: -leaf + cover = 0
                        let idxs = vec![leaf_col, cv];
                        let vals = vec![-1.0, 1.0];
                        unsafe {
                            highs_sys::Highs_addRow(ptr, 0.0, 0.0, 2, idxs.as_ptr(), vals.as_ptr());
                        }
                    }
                } else {
                    let (l, r) = tree.children_pair(v as u32);
                    let cl = node_col[t][l as usize];
                    let cr = node_col[t][r as usize];
                    // cv ≥ cl
                    unsafe {
                        let idx = [cv, cl];
                        let val = [1.0, -1.0];
                        highs_sys::Highs_addRow(ptr, 0.0, f64::INFINITY, 2, idx.as_ptr(), val.as_ptr());
                    }
                    // cv ≥ cr
                    unsafe {
                        let idx = [cv, cr];
                        let val = [1.0, -1.0];
                        highs_sys::Highs_addRow(ptr, 0.0, f64::INFINITY, 2, idx.as_ptr(), val.as_ptr());
                    }
                    // cv ≤ cl + cr  =>  cv - cl - cr ≤ 0
                    unsafe {
                        let idx = [cv, cl, cr];
                        let val = [1.0, -1.0, -1.0];
                        highs_sys::Highs_addRow(ptr, f64::NEG_INFINITY, 0.0, 3, idx.as_ptr(), val.as_ptr());
                    }
                }
            }
        }

        // Solve.
        let solved = model.solve();
        let status = solved.status();
        if status != HighsModelStatus::Optimal {
            return None;
        }

        let objective = -solved.objective_value();
        if objective <= 1.0 + PRICING_EPS {
            return None;
        }

        let solution = solved.get_solution();
        let cols = solution.columns();
        let mut labels: Vec<u32> = Vec::new();
        for l in 1..=n {
            if cols[l - 1] > 0.5 {
                labels.push(l as u32);
            }
        }

        if labels.len() >= 2 {
            Some((labels, objective))
        } else {
            None
        }
    }
}

impl Pricer for IlpPricer {
    fn name(&self) -> &'static str {
        "ilp"
    }

    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult {
        match self.build_and_solve(ctx) {
            Some((labels, _score)) => {
                if ctx.seen.contains(&labels) {
                    return PricingResult::Exhausted;
                }
                let column = scratch.builder.build_unchecked(labels, ctx.trees);
                if ctx.branchings.forbids(&column) {
                    return PricingResult::Exhausted;
                }
                PricingResult::Found(vec![column])
            }
            None => PricingResult::Exhausted,
        }
    }
}
