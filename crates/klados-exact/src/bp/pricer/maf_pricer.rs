//! `MafPricer` — the pricing entry point.
//!
//! A pricing call has two distinct jobs, and this pricer names them as two
//! roles with crisp contracts (see
//! `papers/ours/Pricing Architecture Rework - Implementation Spec.md`):
//!
//! - **Generator** — find emittable improving columns, fast. Constraint-aware.
//!   May come up empty. *Never* claims convergence.
//! - **Certifier** — prove no improving column exists. Sound, the sole
//!   authority for `Converged`.
//!
//! The generator is the multi-tree leaf-pair DP (builds columns valid across
//! all `m` trees by construction, branch-constraint-aware).
//!
//! The certifier is the exact `(T₀,T₁)` anchor DP. Key fact: a column valid
//! across all `m` trees is valid in `(T₀,T₁)`, and its `m`-tree score (more β
//! penalties) is ≤ its `(T₀,T₁)` score. So the `(T₀,T₁)` DP maximum is an
//! upper bound on the true `m`-tree global maximum — `max ≤ 1+ε` certifies
//! convergence for **any** `m`. No relaxation repair, no DSSR.
//!
//! There is no tier cascade and no `Exhausted`: `price` returns `Found`
//! (CG continues), `Converged` (bound trusted), or `Improving` (an improving
//! column provably exists but none was emittable — the solver must branch).

use klados_core::Tree;

use super::exact_pair_dp::{
    collect_positive_candidates, exact_dp_cell_cap, ExactPairDpCache, ExactPairDpPricer,
};
use super::leaf_pair_dp::LeafPairDpPricer;
use super::{adaptive_m2_batch_size, Pricer, PricerScratch, PricingContext, PricingResult};

const PRICING_EPS: f64 = 1.0e-8;

pub struct MafPricer {
    /// Generation: the multi-tree leaf-pair DP. Returns `Found` or defers;
    /// never certifies.
    generator: LeafPairDpPricer,
    /// Certification for m=2 only: the exact 2-tree anchor DP, which for m=2
    /// is the complete pricer (`Found`/`Converged`/`Improving` all exact).
    /// `None` for m≥3, where certification goes through `certify_relaxation`.
    m2_certifier: Option<ExactPairDpPricer>,
}

impl MafPricer {
    pub fn new(trees: &[Tree]) -> Self {
        let m = trees.len();
        let batch = adaptive_batch_size_for(m, trees[0].num_leaves as usize);
        let generator = LeafPairDpPricer::new(trees)
            .with_pair_trial_limit(if m == 2 { 64 } else { 256 })
            .with_fallback_full_when_empty(true)
            .with_max_per_call(batch);
        let m2_certifier = if m == 2 {
            Some(ExactPairDpPricer::new(trees))
        } else {
            None
        };
        Self {
            generator,
            m2_certifier,
        }
    }

    /// Per-tier wall-time breakdown — kept for solver telemetry compatibility.
    /// `MafPricer` is a single component, so there are no tiers to report.
    pub fn tier_timings(&self) -> Vec<(&'static str, std::time::Duration, u64)> {
        Vec::new()
    }

    /// m≥3 certification: run the exact `(T₀,T₁)` anchor DP. Its global
    /// maximum upper-bounds the true m-tree maximum, so `max ≤ 1+ε` is a
    /// sound convergence proof. If the O(n²) table exceeds the cell budget
    /// the DP cannot run — return `Improving` (uncertified), never a false
    /// `Converged`.
    fn certify_relaxation(ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult {
        let n0 = ctx.trees[0].num_nodes();
        let n1 = ctx.trees[1].num_nodes();
        let nl = ctx.num_leaves;
        if n0.saturating_mul(n1) > exact_dp_cell_cap() {
            return PricingResult::Improving;
        }
        let mut cache = scratch
            .exact_dp_cache
            .take()
            .filter(|c| c.fits(n0, n1, nl))
            .unwrap_or_else(|| ExactPairDpCache::new(n0, n1, nl));
        let out = collect_positive_candidates(ctx, &mut cache, &[]);
        scratch.exact_dp_cache = Some(cache);
        if out.max_allowed_closed <= 1.0 + PRICING_EPS {
            PricingResult::Converged
        } else {
            PricingResult::Improving
        }
    }
}

impl Pricer for MafPricer {
    fn name(&self) -> &'static str {
        "maf"
    }

    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult {
        scratch.reset_pricing_stats();

        // Fast path: drain the reserve (columns banked by an earlier DP pass,
        // re-scored against the current duals).
        let reserve = scratch.drain_reserve(ctx, adaptive_m2_batch_size(ctx));
        if !reserve.is_empty() {
            return PricingResult::Found(reserve);
        }

        match &mut self.m2_certifier {
            // m=2: the exact O(n²) anchor DP both generates and certifies, and
            // is the fast path. Run it first. It only fails to generate when
            // an anchor's best column is branch-blocked (`Improving`); only
            // then does the constraint-aware leaf-pair generator run.
            Some(cert) => match cert.price(ctx, scratch) {
                PricingResult::Found(cols) => PricingResult::Found(cols),
                PricingResult::Converged => PricingResult::Converged,
                PricingResult::Improving => {
                    match self.generator.price(ctx, scratch) {
                        PricingResult::Found(cols) => PricingResult::Found(cols),
                        _ => PricingResult::Improving,
                    }
                }
            },
            // m≥3: the leaf-pair DP is the only generator of m-tree-valid
            // columns; run it first. If it comes up empty, certify via the
            // (T₀,T₁) relaxation maximum.
            None => match self.generator.price(ctx, scratch) {
                PricingResult::Found(cols) => PricingResult::Found(cols),
                _ => Self::certify_relaxation(ctx, scratch),
            },
        }
    }
}

/// The pricer for a B&P (sub)instance solve — one `MafPricer` for any `m`.
pub fn dispatch_by_m(trees: &[Tree]) -> MafPricer {
    MafPricer::new(trees)
}

/// Per-call batch size. Larger for big active sets / many trees; capped tight
/// at branched nodes (matching the legacy generator's behaviour).
fn adaptive_batch_size_for(m: usize, n: usize) -> usize {
    let base = if m >= 8 {
        64
    } else if n >= 384 {
        32
    } else {
        16
    };
    base
}
