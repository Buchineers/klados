//! Pair-DP filter pricer for **m ≥ 3** — heuristic, not in default dispatch.
//!
//! Runs pair-DP between `(T₀, T₁)` and post-filters against `T₂..Tₘ₋₁` for
//! AF validity.  The DP only sees `β` from trees 0 and 1, so it can miss
//! columns whose true multi-tree RC is positive.
//!
//! Never returns `Converged`.  For m≥3 the default dispatch uses
//! [`super::LeafPairDpPricer`] (multi-tree bitmask intersection) instead.

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
