//! Tier-2 pricer for **m ≥ 3** — pair-DP between `(T₀, T₁)` plus AF-validity
//! filter against `T₂..Tₘ₋₁`.
//!
//! ## Heuristic, not sound
//!
//! Runs the same DP machinery as [`super::PairDpPricer`] but:
//! 1. The DP optimises reduced cost using only `α` and `β` from trees 0 and 1.
//!    `β` from trees 2..m is not in the objective, so the DP can miss
//!    columns whose true RC (over all m trees) is positive.
//! 2. Emitted columns are **filtered** by topology agreement in trees 2..m
//!    via [`crate::bp::column::ColumnBuilder::try_build`]. Only columns that
//!    are valid AF components for *every* tree are kept.
//!
//! Returns [`super::PricingResult::Found`] when at least one column survives
//! filtering, [`super::PricingResult::Exhausted`] otherwise. **Never returns
//! `Converged`** — for m≥3 we don't certify the absence of positive-RC
//! columns; the LP-bound prune is unsound at every node where this tier was
//! the last to run.
//!
//! Matches bp-multi's `m≥3` pricer pattern. A future tier-3 LP oracle is
//! what would lift this to soundness.

use klados_core::Tree;

use super::{Pricer, PricerScratch, PricingContext, PricingResult, pair_dp::run_pair_dp};

pub struct PairDpFilterPricer {
    max_per_call: usize,
}

impl PairDpFilterPricer {
    pub fn new(trees: &[Tree]) -> Self {
        assert!(trees.len() >= 2, "PairDpFilterPricer requires m≥2");
        Self { max_per_call: 64 }
    }
}

impl Pricer for PairDpFilterPricer {
    fn name(&self) -> &'static str {
        "pair-dp-filter"
    }

    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult {
        // run_pair_dp may legitimately return Converged when m=2, but for
        // m≥3 the convergence claim isn't sound (see module docs). Downgrade
        // Converged to Exhausted in the m≥3 path.
        match run_pair_dp(ctx, scratch, /*allow_filter=*/ true, self.max_per_call) {
            PricingResult::Found(c) => PricingResult::Found(c),
            PricingResult::Converged if ctx.trees.len() == 2 => PricingResult::Converged,
            _ => PricingResult::Exhausted,
        }
    }
}
