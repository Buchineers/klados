//! Composite tiered pricer.
//!
//! Two-tier design for m=2:
//! 1. [`LeafPairDpPricer`] — column generation.  Evaluates all pairs with
//!    `pair_ub − root_penalty > 1+ε` (sound upper bound on RC).  No
//!    artificial trial limits.  Branch-aware (checks cannot-link/must-link
//!    during solve_side).
//! 2. [`ExactPairDpPricer`] — convergence proof.  Only fires when the
//!    leaf-pair DP returns `Exhausted`.  Proves no positive-RC column
//!    exists in O(n²) time, terminating column generation.
//!
//! For m≥3: single tier `LeafPairDpPricer` with multi-tree bitmask
//! intersection.  Heuristic — no convergence proof.

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

pub fn dispatch_by_m(trees: &[Tree]) -> CompositePricer {
    let mut tiers: Vec<Box<dyn Pricer>> = Vec::new();
    if trees.len() == 2 {
        tiers.push(Box::new(LeafPairDpPricer::new(trees).with_max_per_call(32)));
        // The exact DP's O(n²) table is infeasible above ~1000 leaves
        // (~1M cells, 16MB).  On large instances the leaf-pair DP's
        // `pair_ub` bound provides the convergence proof.
        let n0 = trees[0].num_nodes();
        let n1 = trees[1].num_nodes();
        if n0 * n1 <= 1_000_000 {
            tiers.push(Box::new(ExactPairDpPricer::new(trees)));
        }
    } else {
        tiers.push(Box::new(LeafPairDpPricer::new(trees)));
    }
    CompositePricer::new(tiers)
}
