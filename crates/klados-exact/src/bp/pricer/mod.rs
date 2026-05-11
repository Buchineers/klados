//! Pricing — find columns with positive reduced cost, or prove none exist.
//!
//! ## Tiered architecture
//!
//! Pricers are composable through the [`Pricer`] trait. Each tier runs only
//! when the previous one returned [`PricingResult::Exhausted`]; `Found` and
//! `Converged` short-circuit. The cheap tier handles the easy case; the
//! sound oracle tier provides the rigorous bound proof when needed.
//!
//! Soundness composes cleanly: if any tier proves convergence, that's a
//! valid proof — even if it's the cheap tier hitting an easy state.
//!
//! ## State sharing
//!
//! All pricers receive a `&mut` [`PricerScratch`]. Tiers can deposit work
//! (anchors, partial columns, DP tables) into scratch for later tiers or
//! later CG iterations to reuse, avoiding redundant computation.
//!
//! ## Soundness signalling
//!
//! [`PricingResult`] distinguishes three states:
//! - `Found(cols)` — at least one positive-RC column found; add to RMP, re-solve.
//! - `Exhausted` — no columns found by this (possibly heuristic) pricer.
//!   The LP-bound prune is still sound (see [`crate::bp::solver`] for proof).
//! - `Converged` — proved no positive-RC column exists in the entire valid
//!   space, given current branchings. Strongest guarantee.

pub mod anchor_extend;
pub mod composite;
pub mod exact_pair_dp;
pub mod leaf_pair_dp;
pub mod pair_dp;
pub mod pair_dp_filter;
pub mod scratch;

// --- Used in the default dispatch_by_m() ---
pub use composite::{CompositePricer, dispatch_by_m};
pub use exact_pair_dp::ExactPairDpPricer;
pub use leaf_pair_dp::LeafPairDpPricer;

// --- Tier infrastructure ---
pub use scratch::{PairDpTable, PricerScratch};

// --- Experimental / heuristic extras (not in default dispatch) ---
pub use anchor_extend::AnchorExtendPricer;
pub use pair_dp::PairDpPricer;
pub use pair_dp_filter::PairDpFilterPricer;

use klados_core::Tree;

use crate::bp::column::{AfColumn, ColumnSet};
use crate::bp::search::Branchings;

pub struct PricingContext<'a> {
    pub trees: &'a [Tree],
    pub num_leaves: usize,
    pub alpha: &'a [f64],
    pub beta: &'a [Vec<f64>],
    pub columns: &'a [AfColumn],
    pub seen: &'a ColumnSet,
    pub branchings: &'a Branchings,
}

pub enum PricingResult {
    /// Positive-RC columns found. Caller adds and re-prices.
    Found(Vec<AfColumn>),
    /// No columns found but no convergence proof — LP-bound prune unsound.
    Exhausted,
    /// Proved no positive-RC valid column exists — LP-bound prune sound.
    Converged,
}

pub trait Pricer {
    fn name(&self) -> &'static str;
    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult;
}
