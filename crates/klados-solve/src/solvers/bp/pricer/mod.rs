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
//!   The LP-bound prune is still sound (see [`crate::solvers::bp::solver`] for proof).
//! - `Converged` — proved no positive-RC column exists in the entire valid
//!   space, given current branchings. Strongest guarantee.

pub mod anchor_cache;
pub mod exact_pair_dp;
pub mod leaf_pair_dp;
pub mod maf_pricer;
pub mod scratch;

pub use exact_pair_dp::ExactPairDpPricer;
pub use leaf_pair_dp::LeafPairDpPricer;
pub use maf_pricer::{MafPricer, dispatch_by_m};

pub use anchor_cache::{AnchorCache, AnchorEntry, CacheResult};
pub use scratch::{PairDpTable, PricerScratch};

use klados_core::Tree;
use std::time::Instant;

use crate::solvers::bp::column::{AfColumn, ColumnSet};
use crate::solvers::bp::search::Branchings;
use fixedbitset::FixedBitSet;

pub struct PricingContext<'a> {
    pub trees: &'a [Tree],
    pub num_leaves: usize,
    pub alpha: &'a [f64],
    pub beta: &'a [Vec<f64>],
    pub rank_cut_groups: &'a [Vec<FixedBitSet>],
    pub rank_cut_duals: &'a [f64],
    pub columns: &'a [AfColumn],
    pub seen: &'a ColumnSet,
    pub branchings: &'a Branchings,
    /// Cooperative cancellation for the (otherwise uninterruptible) inner DP.
    /// The pricing recurrence on a large core can run for seconds; checking this
    /// flag lets a SIGTERM abort it promptly. Callers without a real flag pass
    /// [`NEVER_TERMINATE`].
    pub terminate: &'a core::sync::atomic::AtomicBool,
    /// Optional wall-clock deadline. This is checked by pricing loops that can
    /// run for a long time so capped exact solves do not wait until pricing
    /// returns before noticing that their budget expired.
    pub deadline: Option<Instant>,
}

impl<'a> PricingContext<'a> {
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.terminate.load(core::sync::atomic::Ordering::Relaxed)
            || self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    #[inline]
    pub fn rank_bonus(&self, column: &AfColumn) -> f64 {
        rank_bonus_for_labels(column.labels(), self.rank_cut_groups, self.rank_cut_duals)
    }

    #[inline]
    pub fn pricing_score(&self, column: &AfColumn) -> f64 {
        column.pricing_score(self.alpha, self.beta) + self.rank_bonus(column)
    }

    #[inline]
    pub fn max_rank_bonus(&self) -> f64 {
        self.rank_cut_duals
            .iter()
            .copied()
            .zip(self.rank_cut_groups.iter())
            .filter(|(dual, _)| *dual > 0.0)
            .map(|(dual, group)| dual * group.len() as f64)
            .sum()
    }
}

pub fn rank_bonus_for_labels(
    labels: &[u32],
    rank_cut_groups: &[Vec<FixedBitSet>],
    rank_cut_duals: &[f64],
) -> f64 {
    rank_cut_groups
        .iter()
        .zip(rank_cut_duals.iter().copied())
        .filter(|(_, dual)| *dual > 0.0)
        .map(|(group, dual)| {
            let coeff = group
                .iter()
                .filter(|cut| labels.iter().any(|&label| cut.contains(label as usize)))
                .count() as f64;
            coeff * dual
        })
        .sum()
}

/// Never-set cancellation flag for pricing callers that don't supply their own.
pub static NEVER_TERMINATE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

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
pub(crate) fn adaptive_m2_batch_size(ctx: &PricingContext, m2_batch_override: usize) -> usize {
    if m2_batch_override > 0 {
        return m2_batch_override;
    }

    let active = ctx.alpha.iter().skip(1).filter(|&&a| a > 1.0e-12).count();
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
