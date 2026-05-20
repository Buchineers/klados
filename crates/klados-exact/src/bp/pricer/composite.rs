//! Composite tiered pricer.
//!
//! m=2 dispatch:
//! - Use [`ExactPairDpPricer`] first when its O(n²) table fits. Empirically
//!   this is far faster than the recursive leaf-pair generator on the
//!   reduced m=2 subinstances that dominate pricing.
//! - Fall back to [`LeafPairDpPricer`] for large trees where the exact
//!   table would exceed the memory budget. It evaluates promising pairs
//!   first and falls back to a full scan when needed for convergence.
//!
//! For m≥3: DSSR exact relaxed 2-tree DP with full multi-tree validation.
//! Leaf-pair DP is kept as a heuristic column generator only; it must not
//! certify convergence at branched multi-tree nodes.

use klados_core::Tree;
use log::trace;

use super::{
    adaptive_m2_batch_size, ExactPairDpPricer, LeafPairDpPricer, MultiTreePairDpPricer, Pricer,
    PricerScratch, PricingContext, PricingResult, SmallComponentPricer,
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
    /// Aggregate per-tier wall-time across the full search, indexed by
    /// tier position (mirrors `tiers`). Exposed via [`tier_timings`] so
    /// the solver can report which tier dominates at `bp done`.
    tier_total: Vec<std::time::Duration>,
    /// Per-tier invocation count.
    tier_calls: Vec<u64>,
    /// Wall-time spent in `drain_reserve` short-circuits at the top of
    /// every CG iter — counted separately because it bypasses the tier
    /// dispatch loop.
    reserve_total: std::time::Duration,
}

impl CompositePricer {
    pub fn new(tiers: Vec<Box<dyn Pricer>>) -> Self {
        let n = tiers.len();
        Self {
            tiers,
            portfolio_batch: None,
            tier_total: vec![std::time::Duration::ZERO; n],
            tier_calls: vec![0; n],
            reserve_total: std::time::Duration::ZERO,
        }
    }

    pub fn with_portfolio_batch(mut self, batch: usize) -> Self {
        self.portfolio_batch = Some(batch);
        self
    }

    /// `(tier_name, total_wall, invocation_count)` per tier plus the
    /// reserve-drain total at index 0 of the returned vector under the
    /// synthetic name `"reserve"`. The solver's `bp done` summary logs
    /// this so we can see at a glance which tier dominates per instance.
    pub fn tier_timings(&self) -> Vec<(&'static str, std::time::Duration, u64)> {
        let mut out: Vec<(&'static str, std::time::Duration, u64)> = Vec::new();
        out.push(("reserve", self.reserve_total, 0));
        for (i, tier) in self.tiers.iter().enumerate() {
            out.push((tier.name(), self.tier_total[i], self.tier_calls[i]));
        }
        out
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

        let t_reserve = std::time::Instant::now();
        let reserve = scratch.drain_reserve(ctx, batch);
        self.reserve_total += t_reserve.elapsed();
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

        for (i, tier) in self.tiers.iter_mut().enumerate() {
            let t0 = std::time::Instant::now();
            let result = tier.price(ctx, scratch);
            let elapsed = t0.elapsed();
            self.tier_total[i] += elapsed;
            self.tier_calls[i] += 1;
            match result {
                PricingResult::Found(cols) => {
                    trace!(
                        target: LOG_TARGET,
                        "{}: Found {} in {:.3}ms stats={:?}",
                        tier.name(),
                        cols.len(),
                        elapsed.as_secs_f64() * 1000.0,
                        scratch.pricing_stats,
                    );
                    return PricingResult::Found(cols);
                }
                PricingResult::Converged => {
                    trace!(
                        target: LOG_TARGET,
                        "{}: Converged in {:.3}ms stats={:?}",
                        tier.name(),
                        elapsed.as_secs_f64() * 1000.0,
                        scratch.pricing_stats
                    );
                    return PricingResult::Converged;
                }
                PricingResult::Exhausted => {
                    trace!(
                        target: LOG_TARGET,
                        "{}: Exhausted in {:.3}ms (cascading)",
                        tier.name(),
                        elapsed.as_secs_f64() * 1000.0,
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
        // The exact DP's O(n²) table is infeasible above ~1000 leaves
        // (~1M cells, 16MB).  On large instances the leaf-pair DP's
        // `pair_ub` bound provides the convergence proof.
        let n0 = trees[0].num_nodes();
        let n1 = trees[1].num_nodes();
        if use_m2_leaf_first() {
            // Legacy bp-multi tried the fast leaf-pair generator before the
            // exact anchor DP.  This can seed the RMP with structurally better
            // columns and shrink the B&B tree on some hard 2-tree subproblems.
            tiers.push(Box::new(HeuristicOnlyPricer::new(Box::new(
                LeafPairDpPricer::new(trees)
                    .with_pair_trial_limit(m2_leaf_pair_trial_limit())
                    .with_fallback_full_when_empty(false)
                    .with_max_per_call(64),
            ))));
        }
        if n0 * n1 <= m2_exact_dp_cell_cap() {
            tiers.push(Box::new(ExactPairDpPricer::new(trees)));
        }
        tiers.push(Box::new(
            LeafPairDpPricer::new(trees)
                .with_pair_trial_limit(m2_leaf_pair_trial_limit())
                .with_fallback_full_when_empty(m2_leaf_pair_full_fallback(trees))
                .with_max_per_call(32),
        ));
    } else {
        let batch = adaptive_batch_size_for(trees.len(), trees[0].num_leaves as usize);
        if use_m3_leaf_only() {
            tiers.push(Box::new(
                LeafPairDpPricer::new(trees)
                    .with_pair_trial_limit(256)
                    .with_fallback_full_when_empty(true)
                    .with_max_per_call(batch),
            ));
            return CompositePricer::new(tiers);
        }
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

fn m2_leaf_pair_trial_limit() -> u32 {
    std::env::var("KLADOS_BP_M2_LEAF_PAIR_TRIALS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(64)
}

fn m2_exact_dp_cell_cap() -> usize {
    std::env::var("KLADOS_BP_M2_EXACT_DP_CELLS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        // Old bp-multi always keeps the exact 2-tree DP available.  The
        // rewrite's former 1M-cell cutoff accidentally disabled the exact
        // convergence tier on the remaining hard ~550-620 leaf subcores,
        // leaving only the leaf-pair repair pricer and causing huge branch
        // trees/timeouts.  4M cells is still modest memory for one active
        // B&P solve while covering those legacy-fast cores.
        .unwrap_or(4_000_000)
}

fn use_m2_leaf_first() -> bool {
    std::env::var("KLADOS_BP_M2_LEAF_FIRST")
        .ok()
        .map(|v| v != "0")
        .unwrap_or(false)
}

fn m2_leaf_pair_full_fallback(trees: &[Tree]) -> bool {
    std::env::var("KLADOS_BP_M2_LEAF_FULL_FALLBACK")
        .ok()
        .map(|v| v != "0")
        // The full p² scan is still needed on some exact-track cores: e.g.
        // reduced n=290 from the 350-leaf public instance 05/5d needs a
        // branch-feasible same-anchor alternative to hit k=223.  But at
        // n≈343+ it dominates the remaining hard heuristic cases (07/e9 spent
        // ~79s there).  Keep it only below that empirical correctness/speed
        // boundary; exact-DP remains the convergence tier above it.
        .unwrap_or_else(|| trees[0].num_leaves <= 300)
}

fn use_m3_leaf_only() -> bool {
    std::env::var("KLADOS_BP_M3_LEAF_ONLY")
        .ok()
        .map(|v| v != "0")
        .unwrap_or(true)
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
    if ctx.trees.len() == 2 {
        return adaptive_m2_batch_size(ctx);
    }
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
