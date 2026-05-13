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
//! For m≥3: DSSR exact relaxed 2-tree DP with full multi-tree validation.
//! Leaf-pair DP is kept as a heuristic column generator only; it must not
//! certify convergence at branched multi-tree nodes.

use klados_core::Tree;
use log::trace;

use super::{
    ExactPairDpPricer, LeafPairDpPricer, MultiTreePairDpPricer, Pricer, PricerScratch,
    PricingContext, PricingResult, SmallComponentPricer,
};

const LOG_TARGET: &str = "klados::bp::composite";

struct HeuristicOnlyPricer {
    inner: Box<dyn Pricer>,
}

impl HeuristicOnlyPricer {
    fn new(inner: Box<dyn Pricer>) -> Self {
        Self { inner }
    }
}

impl Pricer for HeuristicOnlyPricer {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult {
        match self.inner.price(ctx, scratch) {
            PricingResult::Converged => PricingResult::Exhausted,
            other => other,
        }
    }
}

pub struct CompositePricer {
    tiers: Vec<Box<dyn Pricer>>,
    portfolio_batch: Option<usize>,
}

impl CompositePricer {
    pub fn new(tiers: Vec<Box<dyn Pricer>>) -> Self {
        Self {
            tiers,
            portfolio_batch: None,
        }
    }

    pub fn with_portfolio_batch(mut self, batch: usize) -> Self {
        self.portfolio_batch = Some(batch);
        self
    }
}

impl Pricer for CompositePricer {
    fn name(&self) -> &'static str {
        "composite"
    }

    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult {
        scratch.reset_pricing_stats();
        let batch = adaptive_batch_size(ctx);

        if let Some(batch) = self.portfolio_batch {
            return self.price_portfolio(ctx, scratch, batch);
        }

        let reserve = scratch.drain_reserve(ctx, batch);
        if !reserve.is_empty() {
            trace!(
                target: LOG_TARGET,
                "reserve: Found {} (before={} kept={} discarded={})",
                reserve.len(),
                scratch.pricing_stats.reserve_before,
                scratch.pricing_stats.reserve_kept,
                scratch.pricing_stats.reserve_discarded,
            );
            return PricingResult::Found(reserve);
        }

        for tier in self.tiers.iter_mut() {
            let t0 = std::time::Instant::now();
            match tier.price(ctx, scratch) {
                PricingResult::Found(cols) => {
                    trace!(
                        target: LOG_TARGET,
                        "{}: Found {} in {:.3}ms stats={:?}",
                        tier.name(),
                        cols.len(),
                        t0.elapsed().as_secs_f64() * 1000.0,
                        scratch.pricing_stats,
                    );
                    return PricingResult::Found(cols);
                }
                PricingResult::Converged => {
                    trace!(
                        target: LOG_TARGET,
                        "{}: Converged in {:.3}ms stats={:?}",
                        tier.name(),
                        t0.elapsed().as_secs_f64() * 1000.0,
                        scratch.pricing_stats
                    );
                    return PricingResult::Converged;
                }
                PricingResult::Exhausted => {
                    trace!(
                        target: LOG_TARGET,
                        "{}: Exhausted in {:.3}ms (cascading)",
                        tier.name(),
                        t0.elapsed().as_secs_f64() * 1000.0,
                    );
                    continue;
                }
            }
        }
        PricingResult::Exhausted
    }
}

impl CompositePricer {
    /// Portfolio mode for m≥3.
    ///
    /// The pricing theory says the RMP wants the columns with largest *true*
    /// reduced cost, not just the first column family that happens to find
    /// something.  In multi-tree mode the tiers are complementary:
    ///
    /// - DSSR gives exact relaxed-DP candidates and repairs illegal anchors.
    /// - small-component enumeration catches high-value tiny components.
    /// - leaf-pair DP remains a heuristic broad search safety net.
    ///
    /// Running the later tiers only when the current portfolio is under-filled
    /// keeps easy nodes cheap while giving hard nodes a stronger batch per LP
    /// solve.  The final `emit_with_reserve` re-scores every candidate by exact
    /// full reduced cost and stashes overflow for the next CG call.
    fn price_portfolio(
        &mut self,
        ctx: &PricingContext,
        scratch: &mut PricerScratch,
        batch: usize,
    ) -> PricingResult {
        let mut portfolio = scratch.drain_reserve(ctx, batch);
        if portfolio.len() >= batch {
            trace!(
                target: LOG_TARGET,
                "reserve portfolio: Found {} stats={:?}",
                portfolio.len(),
                scratch.pricing_stats,
            );
            return PricingResult::Found(portfolio);
        }

        let mut saw_converged = false;
        for tier in self.tiers.iter_mut() {
            let t0 = std::time::Instant::now();
            match tier.price(ctx, scratch) {
                PricingResult::Found(cols) => {
                    trace!(
                        target: LOG_TARGET,
                        "{}: Portfolio found {} in {:.3}ms stats={:?}",
                        tier.name(),
                        cols.len(),
                        t0.elapsed().as_secs_f64() * 1000.0,
                        scratch.pricing_stats,
                    );
                    portfolio.extend(cols);
                    if portfolio.len() >= batch {
                        let out = scratch.emit_with_reserve(portfolio, ctx, batch);
                        return PricingResult::Found(out);
                    }
                }
                PricingResult::Converged => {
                    trace!(
                        target: LOG_TARGET,
                        "{}: Portfolio converged in {:.3}ms stats={:?}",
                        tier.name(),
                        t0.elapsed().as_secs_f64() * 1000.0,
                        scratch.pricing_stats,
                    );
                    saw_converged = true;
                    break;
                }
                PricingResult::Exhausted => {
                    trace!(
                        target: LOG_TARGET,
                        "{}: Exhausted in {:.3}ms (portfolio)",
                        tier.name(),
                        t0.elapsed().as_secs_f64() * 1000.0,
                    );
                }
            }
        }

        if !portfolio.is_empty() {
            let out = scratch.emit_with_reserve(portfolio, ctx, batch);
            PricingResult::Found(out)
        } else if saw_converged {
            PricingResult::Converged
        } else {
            PricingResult::Exhausted
        }
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
        let batch = adaptive_batch_size_for(trees.len(), trees[0].num_leaves as usize);
        if use_fast_leaf_first(trees) {
            // Old bp-multi's speed on easy high-m benchmark instances comes
            // from this leaf-pair generator. Run it first as a heuristic only:
            // profitable columns short-circuit cheaply, but "Converged" is
            // downgraded so DSSR/small still get a chance to repair the
            // one-best-per-state pathology at hard/branched nodes.
            tiers.push(Box::new(HeuristicOnlyPricer::new(Box::new(
                LeafPairDpPricer::new(trees)
                    .with_pair_trial_limit(256)
                    .with_fallback_full_when_empty(true)
                    .with_max_per_call(batch),
            ))));
        }

        // m≥3: DSSR is the exact relaxed-pair repair tier. It uses the exact
        // 2-tree DP as a relaxation, then decrementally cuts invalid / illegal
        // / already-seen anchors.
        tiers.push(Box::new(
            MultiTreePairDpPricer::new(trees).with_batch_size(batch),
        ));
        tiers.push(Box::new(SmallComponentPricer::new(trees)));
        // Keep the multi-tree leaf-pair DP as a column-generation safety net,
        // but never as a convergence oracle: its one-best-per-state behavior
        // is exactly what DSSR is meant to repair.
        if !use_fast_leaf_first(trees) {
            tiers.push(Box::new(LeafPairDpPricer::new(trees).with_max_per_call(batch)));
        }
    }
    let pricer = CompositePricer::new(tiers);
    if trees.len() >= 3 && std::env::var("KLADOS_BP_PORTFOLIO").is_ok() {
        pricer.with_portfolio_batch(64)
    } else {
        pricer
    }
}

fn use_fast_leaf_first(trees: &[Tree]) -> bool {
    let n = trees[0].num_leaves as usize;
    let m = trees.len();
    // Keep DSSR first on small/medium instances where exact_pub exposed
    // one-best suboptimality; use the legacy-fast leaf-pair generator first
    // on large/high-m cases where it is empirically the dominant win.
    n >= 80 || m >= 8
}

fn adaptive_batch_size(ctx: &PricingContext) -> usize {
    adaptive_batch_size_for(ctx.trees.len(), ctx.num_leaves)
}

fn adaptive_batch_size_for(num_trees: usize, num_leaves: usize) -> usize {
    if num_trees < 3 {
        return 64;
    }
    if num_leaves >= 1200 {
        64
    } else if num_leaves >= 768 {
        48
    } else if num_leaves >= 384 {
        32
    } else if num_leaves >= 256 {
        24
    } else {
        16
    }
}
