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

pub mod anchor_cache;
pub mod exact_pair_dp;
pub mod leaf_pair_dp;
pub mod maf_pricer;
pub mod scratch;

pub use exact_pair_dp::ExactPairDpPricer;
pub use leaf_pair_dp::LeafPairDpPricer;
pub use maf_pricer::{dispatch_by_m, MafPricer};

pub use anchor_cache::{AnchorCache, AnchorEntry, CacheResult};
pub use scratch::{PairDpTable, PricerScratch};

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

/// Outcome of a pricing call. The solver trusts the LP bound (bound-prune,
/// optimality certification) **only** on `Converged`.
pub enum PricingResult {
    /// At least one emittable improving column was found. CG continues.
    Found(Vec<AfColumn>),
    /// Proven that no improving column exists (valid or not). CG stops, the
    /// LP bound is certified — bound-prune is sound.
    Converged,
    /// An improving column provably exists, but none was emittable (every
    /// candidate violates a branch constraint). CG stops, the LP bound is
    /// NOT certified — the solver must branch, never bound-prune.
    Improving,
}

pub trait Pricer {
    fn name(&self) -> &'static str;
    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult;
}

/// Legacy-compatible batch cap for the m=2 column generator.
///
/// The old monolithic solver deliberately emitted smaller batches on
/// medium-sized two-tree subproblems (and at most 16 away from the root).
/// The rewrite had grown to 64-column batches everywhere, which tends to
/// flood the RMP with weaker columns and makes the LP/search loop much larger
/// on the hard decomposed heuristic subinstances.
pub(crate) fn adaptive_m2_batch_size(ctx: &PricingContext) -> usize {
    if let Ok(raw) = std::env::var("KLADOS_BP_M2_BATCH") {
        if let Ok(n) = raw.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }

    let active = ctx
        .alpha
        .iter()
        .skip(1)
        .filter(|&&a| a > 1.0e-12)
        .count();
    let mut batch = if active >= 1200 {
        64
    } else if active >= 768 {
        48
    } else if active >= 384 {
        32
    } else if active >= 256 {
        24
    } else {
        16
    };
    if ctx.branchings.depth() > 0 {
        batch = batch.min(16);
    }
    batch
}
