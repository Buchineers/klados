//! `MafPricer` — the pricing entry point.
//!
//! (see `papers/ours/Pricing Architecture Rework - Implementation Spec.md`.)
//!
//! - **m = 2**: the exact `(T₀,T₁)` anchor DP is the complete pricer — it both
//!   generates and certifies. The leaf-pair DP runs only as a constraint-aware
//!   generator when the exact DP defers (a branch-blocked positive).
//!
//! - **m ≥ 3**: the multi-tree leaf-pair DP is *both* generator and certifier.
//!   It builds columns valid across all `m` trees by construction and is
//!   constraint-aware (cannot-link masking in the recurrence, drop-repair at
//!   emission), so the per-anchor best it emits is node-valid. Its full
//!   all-anchor scan certifies directly: `solve_pair` dominates every column
//!   at its anchor, so a global max ≤ 1+ε proves no improving column exists.
//!
//! `price` returns `Found` (CG continues), `Converged` (bound trusted), or
//! `Improving` (an improving column provably exists but none was emittable —
//! the solver must branch). No tier cascade, no `Exhausted`.

use klados_core::Tree;

use super::exact_pair_dp::ExactPairDpPricer;
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
            // m≥3: the multi-tree leaf-pair DP is both generator and
            // certifier. It builds columns valid across all m trees by
            // construction, and its full all-anchor scan certifies
            // convergence directly (§3): `solve_pair` dominates every column
            // at its anchor, so a constraint-blind global max ≤ 1+ε proves no
            // improving column exists for any m.
            None => self.generator.price(ctx, scratch),
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
