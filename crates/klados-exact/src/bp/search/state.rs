//! Search state — append-only column pool, dedup set, incumbent tracking.

use crate::bp::column::{AfColumn, ColumnSet};

#[derive(Clone)]
pub struct Incumbent {
    /// Indices into `SearchState::columns` that form the integer solution.
    pub component_columns: Vec<usize>,
    pub k: usize,
}

pub struct SearchState {
    columns: Vec<AfColumn>,
    seen: ColumnSet,
    incumbent: Option<Incumbent>,
    /// Best upper bound found so far. Initialised to `n` (singleton forest).
    best_ub: usize,
    num_leaves: usize,
}

impl SearchState {
    /// Seeds with one singleton column per leaf. Sets `best_ub = n` from the
    /// trivial all-singletons partition. Caller should add real columns after.
    pub fn seed_singletons(num_leaves: usize, columns: Vec<AfColumn>) -> Self {
        debug_assert_eq!(columns.len(), num_leaves);
        let mut seen = ColumnSet::new();
        for c in &columns {
            seen.insert(c.labels().to_vec());
        }
        let component_columns: Vec<usize> = (0..num_leaves).collect();
        Self {
            columns,
            seen,
            incumbent: Some(Incumbent {
                component_columns,
                k: num_leaves,
            }),
            best_ub: num_leaves,
            num_leaves,
        }
    }

    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    pub fn columns(&self) -> &[AfColumn] {
        &self.columns
    }

    pub fn seen(&self) -> &ColumnSet {
        &self.seen
    }

    pub fn incumbent(&self) -> Option<&Incumbent> {
        self.incumbent.as_ref()
    }

    pub fn best_ub(&self) -> usize {
        self.best_ub
    }

    /// Append a column. Returns its global id, or `None` if a column with the
    /// same labels was already in the pool (no insertion).
    pub fn add_column(&mut self, column: AfColumn) -> Option<usize> {
        if !self.seen.insert(column.labels().to_vec()) {
            return None;
        }
        let id = self.columns.len();
        self.columns.push(column);
        Some(id)
    }

    /// Record a new integer incumbent if it improves on the best known.
    pub fn update_incumbent(&mut self, candidate: Incumbent) -> bool {
        if candidate.k < self.best_ub {
            self.best_ub = candidate.k;
            self.incumbent = Some(candidate);
            true
        } else {
            false
        }
    }
}
