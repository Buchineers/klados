//! State shared across pricer tiers and CG iterations.
//!
//! [`PricerScratch`] is threaded through every [`super::Pricer::price`] call.
//! Each tier reads what it needs and writes to fields that later tiers (or
//! later CG iterations) can use. This is what lets a "less strict" pricer
//! cheaply pre-compute work the "more strict" pricer can build on, instead
//! of every tier starting from scratch.
//!
//! Concrete sharing:
//! - `builder` — single [`ColumnBuilder`] for all column construction.
//! - `candidate_pool` — Tier 1 (anchor-and-extend) deposits promising
//!   anchors here even when it returns `Exhausted`. Tier 3 (sound oracle,
//!   future) consumes them as warm-start columns.
//! - `pair_dp_table` — Tier 2 (pair-DP) keeps its DP table allocated
//!   across CG iterations within a B&B node. Refilled from updated duals;
//!   storage reused.

use crate::bp::column::{AfColumn, ColumnBuilder};

pub struct PricerScratch {
    pub builder: ColumnBuilder,
    /// Promising columns or anchors found by an earlier tier; later tiers
    /// may consume as warm-starts. Stage 5 keeps this empty by default —
    /// hook for future LP/SAT oracles.
    pub candidate_pool: Vec<AfColumn>,
    /// Pair-DP cell table, kept allocated across calls within a single
    /// B&B node so we don't re-allocate ~n² f64s every CG iter.
    /// `None` until first use; `take()`/`replace()` to mutate borrow-safely.
    pub pair_dp_table: Option<PairDpTable>,
    /// Exact bottom-up 2-tree DP tables, reused across CG iterations
    /// to match the legacy `Dp2TreeCache` pattern (allocated once,
    /// filled per call with current duals).
    pub exact_dp_cache: Option<super::exact_pair_dp::ExactPairDpCache>,
    pub column_reserve: Vec<AfColumn>,
}

impl PricerScratch {
    pub fn new(trees: &[klados_core::Tree]) -> Self {
        Self {
            builder: ColumnBuilder::new(trees),
            candidate_pool: Vec::new(),
            pair_dp_table: None,
            exact_dp_cache: None,
            column_reserve: Vec::new(),
        }
    }
}

/// Persistent storage for the pair-DP. Cells are refilled each call but the
/// vector storage is reused.
pub struct PairDpTable {
    pub cells: Vec<PairDpCell>,
    pub n0: usize,
    pub n1: usize,
}

#[derive(Clone, Copy, Default)]
pub struct PairDpCell {
    pub open_score: f64,
    pub closed_score: f64,
    pub open_choice: u8,
    pub closed_choice: u8,
}

impl PairDpTable {
    pub fn new(n0: usize, n1: usize) -> Self {
        Self {
            cells: vec![PairDpCell::default(); n0 * n1],
            n0,
            n1,
        }
    }

    pub fn fits(&self, n0: usize, n1: usize) -> bool {
        self.n0 == n0 && self.n1 == n1
    }

    pub fn idx(&self, u: usize, v: usize) -> usize {
        u * self.n1 + v
    }

    /// Reset all cells to default (NEG_INF / infeasible). The reuse is
    /// purely for storage — every call recomputes from current duals.
    pub fn reset(&mut self) {
        // f64::NEG_INFINITY isn't Default-compatible; use a sentinel via fill.
        for c in &mut self.cells {
            c.open_score = f64::NEG_INFINITY;
            c.closed_score = f64::NEG_INFINITY;
            c.open_choice = 0;
            c.closed_choice = 0;
        }
    }
}
