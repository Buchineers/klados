//! Restricted Master Problem — set-cover LP relaxation, HiGHS-backed.
//!
//! ## Formulation
//! ```text
//! min  Σ_c x_c
//! s.t. Σ_{c ∋ leaf l} x_c == 1            ∀ leaf l           (leaf cover)
//!      Σ_{c covers (t,v)} x_c <= 1        ∀ tree t, internal v
//!      lo_c ≤ x_c ≤ hi_c                                    (branching)
//! ```
//!
//! ## Lazy node rows
//!
//! Internal-node `≤1` constraints are materialised **lazily**: a row is
//! added only when the current LP solution violates it (Σ x_c > 1 over
//! columns covering that node). On most instances <10% of node rows ever
//! need to exist, so LP solves get a 5–10× speedup vs. eager materialisation.
//!
//! The reverse index [`Rmp::node_to_cols`] tracks every column that covers
//! `(t, v)` regardless of whether the row is materialised, so when a row is
//! created we can build its coefficient vector in O(|columns covering (t,v)|).
//!
//! Branching never references column ids — bounds are derived from a
//! [`Branchings`] via [`bound_for`] each time we enter a B&B node,
//! and the dual simplex warm-starts from the previous basis.

pub mod bounds;

use highs::{ColProblem, HighsModelStatus, Model};
use klados_core::Tree;

use crate::bp::column::AfColumn;
use crate::bp::search::Branchings;
use bounds::{Bound, bound_for};

pub struct RmpSolution {
    pub objective: f64,
    pub column_values: Vec<f64>,
    /// `leaf_duals[label]` for label 1..=num_leaves. Index 0 is a sentinel.
    pub leaf_duals: Vec<f64>,
    /// `node_duals[t][v]` — already negated so reduced cost is `α − β`.
    /// `0.0` for nodes whose row is not materialised (i.e. constraint isn't
    /// in the LP, so its dual contribution is zero).
    pub node_duals: Vec<Vec<f64>>,
}

pub struct Rmp {
    model: Option<Model>,
    leaf_row_idx: Vec<usize>,
    /// `None` until materialised. The row exists in HiGHS only after lazy
    /// separation determines a column-value violation.
    node_row_idx: Vec<Vec<Option<usize>>>,
    /// Total rows currently in HiGHS (leaf + materialised node rows).
    num_rows: usize,
    /// Reverse index — `node_to_cols[t][v]` = list of global column ids whose
    /// labelset's coverage in tree `t` includes node `v`. Populated on every
    /// `add_column` regardless of row materialisation; used to build the
    /// coefficient vector when a row is materialised later.
    node_to_cols: Vec<Vec<Vec<usize>>>,
    /// HiGHS column handle per global column id.
    col_handle: Vec<i32>,
    cur_lo: Vec<f64>,
    cur_hi: Vec<f64>,
}

impl Rmp {
    pub fn new(initial: &[AfColumn], trees: &[Tree], num_leaves: usize) -> Self {
        let mut model = Model::new(ColProblem::default());
        model.make_quiet();
        model.set_option("threads", 1_i32);
        model.set_option("presolve", "off");
        model.set_option("solver", "simplex");
        model.set_option("simplex_strategy", 1_i32); // dual simplex

        // Leaf rows: every column covers ≥1 leaf, so all of these will be
        // referenced — eager materialisation is correct. Index 0 is a
        // sentinel so we can index by label directly.
        let mut leaf_row_idx = vec![0usize; num_leaves + 1];
        for leaf in 0..=num_leaves {
            let row = if leaf == 0 { 0.0..=0.0 } else { 1.0..=1.0 };
            model.add_row(row, Vec::<(highs::Col, f64)>::new());
            leaf_row_idx[leaf] = leaf;
        }
        let num_rows = num_leaves + 1;

        // Node rows: lazy. Reverse index allocated up front; rows materialise
        // on demand via `separate_and_add_cuts`.
        let node_row_idx: Vec<Vec<Option<usize>>> =
            trees.iter().map(|t| vec![None; t.num_nodes()]).collect();
        let node_to_cols: Vec<Vec<Vec<usize>>> = trees
            .iter()
            .map(|t| vec![Vec::new(); t.num_nodes()])
            .collect();

        let mut rmp = Self {
            model: Some(model),
            leaf_row_idx,
            node_row_idx,
            num_rows,
            node_to_cols,
            col_handle: Vec::new(),
            cur_lo: Vec::new(),
            cur_hi: Vec::new(),
        };
        for c in initial {
            rmp.add_column(c);
        }
        rmp
    }

    pub fn num_columns(&self) -> usize {
        self.col_handle.len()
    }

    pub fn add_column(&mut self, column: &AfColumn) {
        let global_ci = self.col_handle.len();
        let mut row_indices: Vec<i32> = column
            .labels()
            .iter()
            .map(|&l| self.leaf_row_idx[l as usize] as i32)
            .collect();
        // For each (t, v) the column covers: always update the reverse index.
        // Add to row_indices only if the row is currently materialised.
        for (ti, nodes) in column.coverage().iter_per_tree().enumerate() {
            for &v in nodes {
                self.node_to_cols[ti][v].push(global_ci);
                if let Some(ri) = self.node_row_idx[ti][v] {
                    row_indices.push(ri as i32);
                }
            }
        }
        let values = vec![1.0_f64; row_indices.len()];
        let ptr = self.model.as_mut().expect("model present").as_mut_ptr();
        unsafe {
            highs_sys::Highs_addCol(
                ptr,
                1.0,           // cost
                0.0,           // lower
                f64::INFINITY, // upper
                row_indices.len() as i32,
                row_indices.as_ptr(),
                values.as_ptr(),
            );
        }
        self.col_handle.push(global_ci as i32);
        self.cur_lo.push(0.0);
        self.cur_hi.push(f64::INFINITY);
    }

    /// Materialise the row for `(t, v)` lazily, populating its coefficients
    /// from `node_to_cols[t][v]`. Caller must ensure the row is currently
    /// `None` in `node_row_idx`.
    fn add_node_row_lazy(&mut self, ti: usize, v: usize) {
        debug_assert!(self.node_row_idx[ti][v].is_none());
        let cols_covering = &self.node_to_cols[ti][v];
        let indices: Vec<i32> = cols_covering
            .iter()
            .map(|&ci| self.col_handle[ci])
            .collect();
        let values = vec![1.0_f64; indices.len()];
        let ptr = self.model.as_mut().expect("model present").as_mut_ptr();
        unsafe {
            highs_sys::Highs_addRow(
                ptr,
                f64::NEG_INFINITY,
                1.0,
                indices.len() as i32,
                indices.as_ptr(),
                values.as_ptr(),
            );
        }
        self.node_row_idx[ti][v] = Some(self.num_rows);
        self.num_rows += 1;
    }

    /// Scan the current LP support for violated `≤1` node constraints.
    /// Materialises all violated rows and returns the count added. Caller
    /// should re-solve the LP if the return is non-zero.
    pub fn separate_and_add_cuts(
        &mut self,
        columns: &[AfColumn],
        column_values: &[f64],
        eps: f64,
    ) -> usize {
        // Tally Σ x_c for each unmaterialised (t, v).
        use fxhash::FxHashMap;
        let mut tally: FxHashMap<(usize, usize), f64> = FxHashMap::default();
        for (ci, &v) in column_values.iter().enumerate() {
            if v <= 1.0e-9 {
                continue;
            }
            if ci >= columns.len() {
                continue;
            }
            let col = &columns[ci];
            for (ti, nodes) in col.coverage().iter_per_tree().enumerate() {
                for &node in nodes {
                    if self.node_row_idx[ti][node].is_none() {
                        *tally.entry((ti, node)).or_insert(0.0) += v;
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

    /// Apply per-column bounds derived from `branchings`.
    pub fn apply_bounds(&mut self, columns: &[AfColumn], branchings: &Branchings) {
        debug_assert_eq!(columns.len(), self.col_handle.len());
        let ptr = self.model.as_mut().expect("model present").as_mut_ptr();
        for (ci, column) in columns.iter().enumerate() {
            let Bound { lo, hi } = bound_for(column, branchings);
            if Self::bounds_changed(self.cur_lo[ci], self.cur_hi[ci], lo, hi) {
                unsafe {
                    highs_sys::Highs_changeColBounds(ptr, self.col_handle[ci], lo, hi);
                }
                self.cur_lo[ci] = lo;
                self.cur_hi[ci] = hi;
            }
        }
    }

    pub fn solve(&mut self) -> Result<RmpSolution, String> {
        let solved = self.model.take().expect("model present").solve();
        let status = solved.status();
        if status != HighsModelStatus::Optimal {
            self.model = Some(Model::from(solved));
            return Err(format!("LP status: {status:?}"));
        }
        let solution = solved.get_solution();
        let cols = solution.columns();
        let column_values: Vec<f64> = (0..self.col_handle.len()).map(|ci| cols[ci]).collect();
        let dual_rows = solution.dual_rows();
        let leaf_duals = self
            .leaf_row_idx
            .iter()
            .map(|&ri| clean_dual(dual_rows[ri]))
            .collect();
        // Unmaterialised node rows contribute 0 to RC (the constraint isn't
        // in the LP, so its dual is implicitly zero).
        let node_duals: Vec<Vec<f64>> = self
            .node_row_idx
            .iter()
            .map(|tree_idxs| {
                tree_idxs
                    .iter()
                    .map(|opt| match opt {
                        Some(ri) => clean_dual(-dual_rows[*ri]),
                        None => 0.0,
                    })
                    .collect()
            })
            .collect();
        let objective = solved.objective_value();
        self.model = Some(Model::from(solved));
        Ok(RmpSolution {
            objective,
            column_values,
            leaf_duals,
            node_duals,
        })
    }

    pub fn solve_mip(&mut self) -> Result<Option<RmpSolution>, String> {
        let num_cols = self.col_handle.len();
        if num_cols == 0 {
            return Ok(None);
        }
        let ptr = self.model.as_mut().expect("model present").as_mut_ptr();
        let integrality = vec![1_i32; num_cols]; // 1 = kHighsVarTypeInteger
        unsafe {
            highs_sys::Highs_changeColsIntegralityByRange(
                ptr,
                0,
                num_cols as i32 - 1,
                integrality.as_ptr(),
            );
        }

        let mut model = self.model.take().expect("model present");
        model.set_option("presolve", "on");
        model.set_option("solver", "choose");

        let solved = model.solve();
        let status = solved.status();
        let objective = if status == HighsModelStatus::Optimal {
            solved.objective_value()
        } else {
            0.0
        };
        
        let (solution_cols, has_solution) = if status == HighsModelStatus::Optimal {
            let solution = solved.get_solution();
            (solution.columns().to_vec(), true)
        } else {
            (Vec::new(), false)
        };
        
        let mut model = Model::from(solved);
        model.set_option("presolve", "off");
        model.set_option("solver", "simplex");
        self.model = Some(model);

        let ptr = self.model.as_mut().unwrap().as_mut_ptr();
        let continuous = vec![0_i32; num_cols]; // 0 = kHighsVarTypeContinuous
        unsafe {
            highs_sys::Highs_changeColsIntegralityByRange(
                ptr,
                0,
                num_cols as i32 - 1,
                continuous.as_ptr(),
            );
        }

        if !has_solution {
            return Ok(None);
        }
        
        let column_values: Vec<f64> = (0..self.col_handle.len()).map(|ci| solution_cols[ci]).collect();
        
        Ok(Some(RmpSolution {
            objective,
            column_values,
            leaf_duals: Vec::new(),
            node_duals: Vec::new(),
        }))
    }

    fn bounds_changed(prev_lo: f64, prev_hi: f64, lo: f64, hi: f64) -> bool {
        (prev_lo - lo).abs() > 0.0
            || prev_hi.is_finite() != hi.is_finite()
            || (prev_hi.is_finite() && hi.is_finite() && (prev_hi - hi).abs() > 0.0)
    }
}

fn clean_dual(value: f64) -> f64 {
    if value.abs() <= 1.0e-9 { 0.0 } else { value }
}
