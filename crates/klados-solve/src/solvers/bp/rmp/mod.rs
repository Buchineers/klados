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

use crate::solvers::bp::column::AfColumn;
use crate::solvers::bp::search::Branchings;
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
    /// Reduced-cost variable fixing flag, indexed by global column id.
    /// Once a column is RCVF-fixed it remains x=0 for the entire subtree
    /// — the fixing is monotone-valid because best_ub only decreases as
    /// the search progresses, and tighter best_ub fixes strictly more
    /// columns. Once true, `apply_bounds` ignores the branching-derived
    /// bound and keeps the column pinned at zero.
    rcvf_zero: Vec<bool>,
    /// Cached root-node LP solution. Stored once the root CG converges so
    /// that whenever the incumbent tightens we can re-derive RCVF fixings
    /// from the *unrestricted* duals (the only ones whose fixings hold
    /// globally) without re-solving the root.
    root_lp: Option<RootLp>,
    /// Per-subtree RCVF trail. Each entry `(depth, ci)` records a column
    /// that was fixed by `apply_subtree_rcvf` while exploring a node at
    /// the given branching depth. On backtrack above that depth the entry
    /// is popped and the column is unfixed. Root fixings (depth=0) are
    /// **never** placed on the trail — they hold globally.
    rcvf_trail: Vec<(usize, usize)>,
    tally_scratch: Vec<f64>,
    tally_dirty: Vec<(usize, usize)>,
    max_nodes: usize,
}

struct RootLp {
    objective: f64,
    leaf_duals: Vec<f64>,
    node_duals: Vec<Vec<f64>>,
    /// Smallest `best_ub` we've already RCVF'd against using this root LP.
    /// Replaying with the same or looser bound is a no-op; replaying with
    /// a tighter bound can only fix more columns.
    last_applied_best_ub: usize,
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

        let max_nodes = trees.iter().map(|t| t.num_nodes()).max().unwrap_or(0);
        let num_trees = trees.len();
        let tally_scratch = vec![0.0; num_trees * max_nodes];

        let mut rmp = Self {
            model: Some(model),
            leaf_row_idx,
            node_row_idx,
            num_rows,
            node_to_cols,
            col_handle: Vec::new(),
            cur_lo: Vec::new(),
            cur_hi: Vec::new(),
            rcvf_zero: Vec::new(),
            root_lp: None,
            rcvf_trail: Vec::new(),
            tally_scratch,
            tally_dirty: Vec::new(),
            max_nodes,
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
        self.rcvf_zero.push(false);
    }

    /// Reduced-Cost Variable Fixing.
    ///
    /// Standard branch-and-price result: after CG converges, for every column
    /// `c` the reduced cost `rc(c) = 1 − pricing_score(c)` satisfies the bound
    /// `LP_with_x_c≥1 ≥ lp_obj + rc(c)`. Therefore any integer solution using
    /// `c` has objective ≥ ⌈lp_obj + rc(c)⌉. An improving integer solution has
    /// objective ≤ `best_ub − 1`, so if `lp_obj + rc(c) > best_ub − 1` then
    /// `c` cannot appear in any improving solution and `x_c = 0` is forced
    /// throughout the remaining subtree.
    ///
    /// Returns the number of columns newly fixed by this call. The fixing
    /// is monotone: future calls (with tighter `best_ub` or different LP)
    /// can only fix more, never unfix.
    pub fn apply_rcvf(
        &mut self,
        lp_obj: f64,
        columns: &[AfColumn],
        alpha: &[f64],
        beta: &[Vec<f64>],
        best_ub: usize,
    ) -> usize {
        // depth=0 → root: fixings are global, never trail-tracked.
        self.apply_rcvf_inner(lp_obj, columns, alpha, beta, best_ub, 0)
    }

    /// Subtree-local RCVF. Fixings made here are sound only inside the
    /// subtree rooted at the current node; they are pushed onto
    /// `rcvf_trail` so [`unfix_above_depth`] can flip them back when the
    /// search backtracks above `depth`.
    ///
    /// `depth` must be ≥ 1 — the root path uses [`apply_rcvf`] /
    /// [`reapply_root_rcvf`] for permanent fixings instead.
    pub fn apply_subtree_rcvf(
        &mut self,
        lp_obj: f64,
        columns: &[AfColumn],
        alpha: &[f64],
        beta: &[Vec<f64>],
        best_ub: usize,
        depth: usize,
    ) -> usize {
        debug_assert!(depth >= 1, "subtree RCVF must run at depth >= 1");
        self.apply_rcvf_inner(lp_obj, columns, alpha, beta, best_ub, depth)
    }

    /// Pop and unfix every trail entry whose recorded depth is ≥
    /// `min_depth`. Called when the DFS pops a node at depth `min_depth`:
    /// any per-subtree fixings made by previously-explored sibling
    /// subtrees no longer apply, so we restore those columns to FREE.
    pub fn unfix_above_depth(&mut self, min_depth: usize) {
        let ptr = self.model.as_mut().expect("model present").as_mut_ptr();
        while let Some(&(d, _)) = self.rcvf_trail.last() {
            if d < min_depth {
                break;
            }
            let (_, ci) = self.rcvf_trail.pop().unwrap();
            // Mark as not-fixed; the next apply_bounds will re-derive the
            // bound from the active branchings and call HiGHS only if
            // changed. But the column is currently pinned at [0,0]; we
            // need to restore it to FREE here so the LP sees it again.
            self.rcvf_zero[ci] = false;
            unsafe {
                highs_sys::Highs_changeColBounds(ptr, self.col_handle[ci], 0.0, f64::INFINITY);
            }
            self.cur_lo[ci] = 0.0;
            self.cur_hi[ci] = f64::INFINITY;
        }
    }

    /// Cache the converged root LP solution so [`reapply_root_rcvf`] can
    /// re-derive fixings under tighter incumbents without re-solving.
    pub fn save_root_lp(
        &mut self,
        lp_obj: f64,
        leaf_duals: Vec<f64>,
        node_duals: Vec<Vec<f64>>,
        applied_at_best_ub: usize,
    ) {
        self.root_lp = Some(RootLp {
            objective: lp_obj,
            leaf_duals,
            node_duals,
            last_applied_best_ub: applied_at_best_ub,
        });
    }

    /// Re-derive RCVF fixings using the cached root LP solution and the
    /// current incumbent. The whole point: as best_ub tightens during
    /// search, more columns become fixable *under the original root duals*,
    /// and those fixings still hold globally. No-op if `best_ub` hasn't
    /// tightened since the last call (or no root LP is cached). Returns
    /// the number of columns newly fixed.
    pub fn reapply_root_rcvf(&mut self, columns: &[AfColumn], best_ub: usize) -> usize {
        let Some(mut root) = self.root_lp.take() else {
            return 0;
        };
        if best_ub >= root.last_applied_best_ub {
            self.root_lp = Some(root);
            return 0;
        }
        let newly = self.apply_rcvf_inner(
            root.objective,
            columns,
            &root.leaf_duals,
            &root.node_duals,
            best_ub,
            0,
        );
        root.last_applied_best_ub = best_ub;
        self.root_lp = Some(root);
        newly
    }

    fn apply_rcvf_inner(
        &mut self,
        lp_obj: f64,
        columns: &[AfColumn],
        alpha: &[f64],
        beta: &[Vec<f64>],
        best_ub: usize,
        depth: usize,
    ) -> usize {
        debug_assert_eq!(columns.len(), self.col_handle.len());
        // We can fix x_c = 0 when ⌈lp_obj + rc(c)⌉ ≥ best_ub. Equivalently,
        // since best_ub is integer, when lp_obj + rc(c) > best_ub − 1.
        // A small numerical slack guards against LP solver tolerance.
        let threshold = (best_ub as f64) - 1.0 + 1.0e-6;
        let ptr = self.model.as_mut().expect("model present").as_mut_ptr();
        let mut newly_fixed = 0;
        for (ci, column) in columns.iter().enumerate() {
            if self.rcvf_zero[ci] {
                continue;
            }
            let rc = 1.0 - column.pricing_score(alpha, beta);
            if lp_obj + rc > threshold {
                unsafe {
                    highs_sys::Highs_changeColBounds(ptr, self.col_handle[ci], 0.0, 0.0);
                }
                self.cur_lo[ci] = 0.0;
                self.cur_hi[ci] = 0.0;
                self.rcvf_zero[ci] = true;
                if depth > 0 {
                    self.rcvf_trail.push((depth, ci));
                }
                newly_fixed += 1;
            }
        }
        newly_fixed
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
        let max_nodes = self.max_nodes;
        for (ci, &v) in column_values.iter().enumerate() {
            if v <= 1.0e-9 {
                continue;
            }
            if ci >= columns.len() {
                continue;
            }
            let col = &columns[ci];
            for (ti, nodes) in col.coverage().nodes_per_tree.iter().enumerate() {
                let offset = ti * max_nodes;
                for &node in nodes {
                    if self.node_row_idx[ti][node].is_none() {
                        let idx = offset + node;
                        if self.tally_scratch[idx] == 0.0 {
                            self.tally_dirty.push((ti, node));
                        }
                        self.tally_scratch[idx] += v;
                    }
                }
            }
        }

        let mut dirty = std::mem::take(&mut self.tally_dirty);
        let mut added = 0usize;
        for &(ti, node) in &dirty {
            let idx = ti * max_nodes + node;
            let sum = self.tally_scratch[idx];
            if sum > 1.0 + eps {
                self.add_node_row_lazy(ti, node);
                added += 1;
            }
            self.tally_scratch[idx] = 0.0;
        }
        dirty.clear();
        self.tally_dirty = dirty;
        added
    }

    /// Apply per-column bounds derived from `branchings`. RCVF-fixed columns
    /// stay pinned at zero regardless of the branching state.
    pub fn apply_bounds(&mut self, columns: &[AfColumn], branchings: &Branchings) {
        debug_assert_eq!(columns.len(), self.col_handle.len());
        let ptr = self.model.as_mut().expect("model present").as_mut_ptr();
        for (ci, column) in columns.iter().enumerate() {
            let Bound { lo, hi } = if self.rcvf_zero[ci] {
                Bound::ZERO
            } else {
                bound_for(column, branchings)
            };
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

    pub fn solve_mip_with_time_limit(
        &mut self,
        mip_time_limit: f64,
    ) -> Result<Option<RmpSolution>, String> {
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
        model.set_option("time_limit", mip_time_limit);

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
        // Reset the time limit so subsequent LP solves aren't capped at 0.1s.
        model.set_option("time_limit", f64::INFINITY);
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

        let column_values: Vec<f64> = (0..self.col_handle.len())
            .map(|ci| solution_cols[ci])
            .collect();

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
