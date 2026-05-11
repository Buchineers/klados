//! Composite tiered pricer.
//!
//! Wraps a list of tier pricers. Calls them in order until one of:
//! - returns [`PricingResult::Found`] (continue CG with new columns)
//! - returns [`PricingResult::Converged`] (sound proof — short-circuit)
//!
//! Tiers returning [`PricingResult::Exhausted`] cascade to the next tier.
//! If every tier exhausts, the composite returns `Exhausted` (heuristic).
//!
//! Soundness composes naturally: any tier proving convergence is itself a
//! valid proof. If the cheap tier hits an easy state (`max α-β ≤ 1` over
//! some restricted view), that's still a sound proof when applicable.

use klados_core::Tree;
use log::trace;

use super::{
    ExactPairDpPricer, LeafPairDpPricer, Pricer, PricerScratch, PricingContext, PricingResult,
};

const LOG_TARGET: &str = "klados::bp::composite";

pub struct CompositePricer {
    tiers: Vec<Box<dyn Pricer>>,
}

impl CompositePricer {
    pub fn new(tiers: Vec<Box<dyn Pricer>>) -> Self {
        Self { tiers }
    }
}

impl Pricer for CompositePricer {
    fn name(&self) -> &'static str {
        "composite"
    }

    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult {
        for tier in self.tiers.iter_mut() {
            match tier.price(ctx, scratch) {
                PricingResult::Found(cols) => {
                    trace!(target: LOG_TARGET, "{}: Found {}", tier.name(), cols.len());
                    return PricingResult::Found(cols);
                }
                PricingResult::Converged => {
                    trace!(target: LOG_TARGET, "{}: Converged", tier.name());
                    return PricingResult::Converged;
                }
                PricingResult::Exhausted => {
                    trace!(target: LOG_TARGET, "{}: Exhausted (cascading)", tier.name());
                    continue;
                }
            }
        }
        PricingResult::Exhausted
    }
}

/// Build a default tier list for an instance.
/// - m=2: `[LeafPairDp(trial_limit=256), ExactPairDp]`. Tier 1 (leaf-pair DP
///   with limited pair evaluation, matching bp-multi's FastPricer) provides
///   column diversity without flooding the LP. Tier 2 is the faithful port of
///   bp-multi's exact two-tree bottom-up DP and provides the convergence proof.
/// - m≥3: `[LeafPairDp]`. Leaf-pair DP with multi-tree bitmask intersection
///   emits valid columns by construction across all m trees. Heuristic —
///   does not prove convergence. Full pair evaluation (no trial limit) since
///   there's no sound fallback tier.
pub fn dispatch_by_m(trees: &[Tree]) -> CompositePricer {
    let mut tiers: Vec<Box<dyn Pricer>> = Vec::new();
    if trees.len() == 2 {
        tiers.push(Box::new(
            LeafPairDpPricer::new(trees)
                .with_pair_trial_limit(256)
                .with_max_per_call(16),
        ));
        tiers.push(Box::new(
            LeafPairDpPricer::new(trees)
                .with_pair_trial_limit(2048)
                .with_max_per_call(32),
        ));
        tiers.push(Box::new(ExactPairDpPricer::new(trees)));
        // Unlimited LeafPairDpPricer matches bp-multi's
        // collect_profitable_columns when constraints are present.
        tiers.push(Box::new(LeafPairDpPricer::new(trees)));
    } else {
        tiers.push(Box::new(LeafPairDpPricer::new(trees)));
    }
    CompositePricer::new(tiers)
}
