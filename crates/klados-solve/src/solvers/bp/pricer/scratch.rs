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

use crate::solvers::bp::column::{AfColumn, ColumnBuilder};

use super::PricingContext;

const PRICING_EPS: f64 = 1.0e-8;

#[derive(Clone, Debug, Default)]
pub struct PricingStats {
    pub reserve_before: usize,
    pub reserve_drained: usize,
    pub reserve_discarded: usize,
    pub reserve_kept: usize,
    pub reserve_stashed: usize,
    pub dssr_passes: usize,
    pub dssr_candidates: usize,
    pub dssr_seen: usize,
    pub dssr_branch_blocked: usize,
    pub dssr_invalid: usize,
    pub dssr_nonprofitable: usize,
    pub dssr_found: usize,
    pub dssr_alt_refs_tried: usize,
    pub small_cache_cols: usize,
    pub small_profitable: usize,
    pub leaf_pair_scanned: usize,
    pub leaf_pair_positive: usize,
    pub leaf_pair_found: usize,
    pub leaf_pair_seen: usize,
    pub leaf_pair_repair_failed: usize,
    pub leaf_pair_repair_nonprofitable: usize,
    pub leaf_pair_branch_blocked: usize,
    pub leaf_pair_blocked_must: usize,
    pub leaf_pair_blocked_cannot: usize,
    pub leaf_pair_completed: bool,
    pub leaf_pair_trial_limited: bool,
    pub leaf_pair_global_max: f64,
    /// Branch-feasible certification quantity: the max of `solve_pair` over the
    /// scanned anchors `(a,b)` that can co-occur in some branch-feasible column
    /// (their must-link classes are not cannot-link-conflicting). This is a
    /// sound upper bound on `max_{C ∈ C(B)} score(C)` — see
    /// `leaf_pair_dp::LeafPairDpPricer::anchor_feasible`. `≤ global_max` always.
    pub leaf_pair_feasible_global_max: f64,
    // ── Must-link-closure emission telemetry (multi-tree branch pricing) ──
    /// Positive raw DP candidates for which must-link closure was attempted
    /// (i.e. the branch state had at least one must-link constraint).
    pub must_closure_attempted: usize,
    /// Closure was a valid AF component, repaired, and the repaired column
    /// emitted as an improving (profitable) branch-feasible column.
    pub must_closure_valid_positive: usize,
    /// Closure was a valid AF component and repaired, but the repaired column
    /// scored at or below the 1+ε threshold and was dropped.
    pub must_closure_valid_nonprofitable: usize,
    /// Closure was not a valid AF component, so the raw-subset repair fallback
    /// was used instead.
    pub must_closure_invalid: usize,
    /// The closure path did not yield the emitted labels and the old
    /// raw-subset repair fallback was taken (invalid closure or
    /// repair-stripped-below-2 closure).
    pub must_closure_fallback: usize,
}

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
    /// Top-K threshold DP cache (m=2) for corridor enumeration. Lives
    /// here so it can be reused across γ-iterations within one solver
    /// call; allocated lazily.
    pub topk_dp_cache: Option<crate::solvers::corridor::topk_m2::TopKDpCache>,
    /// Cached-positive reuse for the leaf-pair-DP pricer (m=2). Gated by
    /// `BpConfig.use_anchor_cache` at the pricer level. Lives here so it
    /// persists across CG iterations within a B&P node and is dropped when
    /// the scratch is reset between nodes.
    pub anchor_cache: Option<super::anchor_cache::AnchorCache>,
    pub column_reserve: Vec<AfColumn>,
    pub pricing_stats: PricingStats,
    // --- BpConfig fields threaded through scratch ---
    pub m2_batch: usize,
    pub m2_exact_dp_cells: usize,
    pub m2_exact_reserve_cap: usize,
    pub use_anchor_cache: bool,
}

impl PricerScratch {
    pub fn new(trees: &[klados_core::Tree]) -> Self {
        Self {
            builder: ColumnBuilder::new(trees),
            candidate_pool: Vec::new(),
            pair_dp_table: None,
            exact_dp_cache: None,
            topk_dp_cache: None,
            anchor_cache: None,
            column_reserve: Vec::new(),
            pricing_stats: PricingStats::default(),
            m2_batch: 0,
            m2_exact_dp_cells: 64_000_000,
            m2_exact_reserve_cap: 0,
            use_anchor_cache: false,
        }
    }

    pub fn reset_pricing_stats(&mut self) {
        self.pricing_stats = PricingStats::default();
    }

    pub fn drain_reserve(&mut self, ctx: &PricingContext, limit: usize) -> Vec<AfColumn> {
        self.pricing_stats.reserve_before = self.column_reserve.len();
        if self.column_reserve.is_empty() || limit == 0 {
            return Vec::new();
        }

        let mut scored: Vec<(f64, usize)> = Vec::new();
        let mut keep = vec![true; self.column_reserve.len()];
        for (i, col) in self.column_reserve.iter().enumerate() {
            if ctx.seen.contains(col.labels()) || ctx.branchings.forbids(col) {
                keep[i] = false;
                self.pricing_stats.reserve_discarded += 1;
                continue;
            }
            let score = col.pricing_score(ctx.alpha, ctx.beta);
            if score > 1.0 + PRICING_EPS {
                scored.push((score, i));
            } else {
                keep[i] = false;
                self.pricing_stats.reserve_discarded += 1;
            }
        }

        if scored.is_empty() {
            self.compact_reserve(&keep);
            self.pricing_stats.reserve_kept = self.column_reserve.len();
            return Vec::new();
        }

        scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        scored.truncate(limit);
        for &(_, i) in &scored {
            keep[i] = false;
        }

        let mut selected = vec![false; self.column_reserve.len()];
        for &(_, i) in &scored {
            selected[i] = true;
        }

        let mut out = Vec::with_capacity(scored.len());
        let mut old = std::mem::take(&mut self.column_reserve);
        for (i, col) in old.drain(..).enumerate() {
            if selected[i] {
                out.push(col);
            } else if keep[i] {
                self.column_reserve.push(col);
            }
        }
        self.pricing_stats.reserve_drained = out.len();
        self.pricing_stats.reserve_kept = self.column_reserve.len();
        out
    }

    pub fn emit_with_reserve(
        &mut self,
        mut cols: Vec<AfColumn>,
        ctx: &PricingContext,
        limit: usize,
    ) -> Vec<AfColumn> {
        cols.sort_unstable_by(|a, b| {
            let sa = a.pricing_score(ctx.alpha, ctx.beta);
            let sb = b.pricing_score(ctx.alpha, ctx.beta);
            sb.total_cmp(&sa)
                .then_with(|| b.size().cmp(&a.size()))
                .then_with(|| a.labels().cmp(b.labels()))
        });
        cols.dedup_by(|a, b| a.labels() == b.labels());

        let split = cols.len().min(limit);
        let reserve = cols.split_off(split);
        for col in reserve {
            if !ctx.seen.contains(col.labels())
                && !self
                    .column_reserve
                    .iter()
                    .any(|existing| existing.labels() == col.labels())
            {
                self.column_reserve.push(col);
                self.pricing_stats.reserve_stashed += 1;
            }
        }
        cols
    }

    fn compact_reserve(&mut self, keep: &[bool]) {
        let mut old = std::mem::take(&mut self.column_reserve);
        for (i, col) in old.drain(..).enumerate() {
            if keep[i] {
                self.column_reserve.push(col);
            }
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
