//! Small-component enumerating pricer.
//!
//! Many exact_pub optima for high-m instances are mostly singleton forests
//! plus a handful of pairs/triples/quartets. The DP pricers can miss these at
//! branched nodes because their state stores one best representative. This
//! tier enumerates valid small components once, then re-scores them under the
//! current duals. It is a column generator only and never certifies
//! convergence.

use fxhash::FxHashSet;
use klados_core::Tree;

use crate::bp::column::{AfColumn, ColumnBuilder};
use crate::bp::search::Branchings;

use super::{Pricer, PricerScratch, PricingContext, PricingResult};

const PRICING_EPS: f64 = 1.0e-8;
const MAX_PER_CALL: usize = 64;

pub struct SmallComponentPricer {
    cache: Option<Vec<AfColumn>>,
    max_size: usize,
}

impl SmallComponentPricer {
    pub fn new(trees: &[Tree]) -> Self {
        let n = trees
            .iter()
            .flat_map(|t| t.label.iter().copied())
            .max()
            .unwrap_or(0) as usize;
        let max_size = if n <= 40 {
            4
        } else if n <= 90 {
            3
        } else {
            0
        };
        Self {
            cache: None,
            max_size,
        }
    }

    fn ensure_cache(&mut self, ctx: &PricingContext) {
        if self.cache.is_some() {
            return;
        }
        let mut builder = ColumnBuilder::new(ctx.trees);
        let n = ctx.num_leaves as u32;
        let mut cols = Vec::new();

        if self.max_size < 2 {
            self.cache = Some(cols);
            return;
        }

        // Pairs are cheap and cover the max_size=2 large-n case.
        for a in 1..=n {
            for b in (a + 1)..=n {
                if let Some(c) = builder.try_build(vec![a, b], ctx.trees) {
                    cols.push(c);
                }
            }
        }

        if self.max_size >= 3 {
            for a in 1..=n {
                for b in (a + 1)..=n {
                    for c in (b + 1)..=n {
                        if let Some(col) = builder.try_build(vec![a, b, c], ctx.trees) {
                            cols.push(col);
                        }
                    }
                }
            }
        }

        if self.max_size >= 4 {
            for a in 1..=n {
                for b in (a + 1)..=n {
                    for c in (b + 1)..=n {
                        for d in (c + 1)..=n {
                            if let Some(col) = builder.try_build(vec![a, b, c, d], ctx.trees) {
                                cols.push(col);
                            }
                        }
                    }
                }
            }
        }

        self.cache = Some(cols);
    }
}

impl Pricer for SmallComponentPricer {
    fn name(&self) -> &'static str {
        "small-component"
    }

    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult {
        self.ensure_cache(ctx);
        let cache = self.cache.as_ref().unwrap();
        scratch.pricing_stats.small_cache_cols = cache.len();

        // Cheap O(1)-lookup membership sets for the branching constraints,
        // built once per call. Replaces `Branchings::forbids`'s ~2·k
        // binary-searches per column (where k = |must_link| + |cannot_link|).
        // On deep nodes with k ≈ 10–15 this cuts the per-column reject cost
        // from ~30 ops to ~6 — across ~5000 cache entries × ~1000 calls per
        // hard Class B instance, real savings.
        let constraints = BranchingSets::from(ctx.branchings);

        // Pass 1: index-only scan. We use a *per-column* leaf-gain ceiling
        // (Σ α[l] over the column's labels) rather than the looser global
        // `k · max(α)`: leaf_gain is exactly the positive part of
        // `pricing_score` and node_penalty is non-negative, so any column
        // whose leaf_gain ≤ 1+ε can be safely skipped without touching its
        // coverage / β tables.
        let mut scored: Vec<(f64, usize)> = Vec::new();
        for (i, col) in cache.iter().enumerate() {
            // Tight reject: sum α over the column's own labels. Most
            // cache entries fail this in any given iter because alpha
            // is sparse / weighted toward a few "hot" leaves.
            let leaf_gain: f64 = col
                .labels()
                .iter()
                .map(|&l| ctx.alpha[l as usize])
                .sum();
            if leaf_gain <= 1.0 + PRICING_EPS {
                continue;
            }
            if ctx.seen.contains(col.labels()) || constraints.forbids(col.labels()) {
                continue;
            }
            // node_penalty (sum over coverage·β) is the only remaining
            // ingredient; compute the full pricing_score.
            let score = col.pricing_score(ctx.alpha, ctx.beta);
            if score > 1.0 + PRICING_EPS {
                scratch.pricing_stats.small_profitable += 1;
                scored.push((score, i));
            }
        }
        if scored.is_empty() {
            return PricingResult::Exhausted;
        }

        // Sort by score (high first); only the top MAX_PER_CALL get
        // cloned into the result. We deliberately *do not* feed leftovers
        // into the column-reserve scratch: the cache is fixed across CG
        // iterations, so any column we leave behind here will reappear in
        // the next call's pass 1 at the same cost. Stashing them would
        // just double-store the same labelsets.
        scored.sort_unstable_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| cache[b.1].size().cmp(&cache[a.1].size()))
                .then_with(|| cache[a.1].labels().cmp(cache[b.1].labels()))
        });
        let take = scored.len().min(MAX_PER_CALL);
        let cols: Vec<AfColumn> =
            scored.iter().take(take).map(|&(_, i)| cache[i].clone()).collect();
        PricingResult::Found(cols)
    }
}

/// Per-call cache of the current B&B node's branching constraints in a form
/// optimized for `O(1)` lookups against small column labelsets.
///
/// `Branchings::forbids` does a binary search per pair per check, which is
/// fine when called sparsely but becomes the dominant cost when the small-
/// component cache contains thousands of entries and forbids is called
/// once per entry per CG iter. The membership-set form here costs the
/// same in big-O but cuts the constant ~5× on Class B's deep-node calls.
struct BranchingSets {
    /// Pairs `(min, max)` of cannot-linked leaves.
    cannot: FxHashSet<(u32, u32)>,
    /// For each leaf that is must-linked to another, the partner leaf.
    /// We store both directions so a single lookup suffices. Multiple
    /// must-link entries on the same leaf are rare in this code path
    /// (each branch adds one new pair); we overwrite on conflict, which
    /// is fine because `Branchings::is_inconsistent` would have caught
    /// that case upstream.
    must_partner: Vec<Option<u32>>,
}

impl BranchingSets {
    fn from(branchings: &Branchings) -> Self {
        let cannot: FxHashSet<(u32, u32)> = branchings
            .cannot_link()
            .iter()
            .map(|p| (p.a.min(p.b), p.a.max(p.b)))
            .collect();
        // Bound by the largest label seen in must-link pairs; safer is to
        // size to max(cannot.max, must.max) + 1, but for small_component
        // the columns store sorted u32 labels and a Vec-indexed lookup
        // outperforms a hash for the tiny per-call total leaf count.
        let max_label = branchings
            .must_link()
            .iter()
            .flat_map(|p| [p.a, p.b])
            .max()
            .unwrap_or(0);
        let mut must_partner = vec![None; (max_label as usize) + 1];
        for p in branchings.must_link() {
            must_partner[p.a as usize] = Some(p.b);
            must_partner[p.b as usize] = Some(p.a);
        }
        Self { cannot, must_partner }
    }

    /// True if any pair in `labels` is cannot-linked, or if any leaf in
    /// `labels` is must-linked to a leaf NOT in `labels`.
    fn forbids(&self, labels: &[u32]) -> bool {
        // Cannot-link: check each pair within the column. With k = 2..4
        // labels there are at most C(4,2)=6 pairs — cheap.
        for i in 0..labels.len() {
            let a = labels[i];
            for j in (i + 1)..labels.len() {
                let b = labels[j];
                let key = if a < b { (a, b) } else { (b, a) };
                if self.cannot.contains(&key) {
                    return true;
                }
            }
        }
        // Must-link: for each leaf in the column, if it has a partner,
        // the partner must also be in the column. `labels` is sorted, so
        // we can binary-search for the partner.
        for &l in labels {
            let l_idx = l as usize;
            if l_idx < self.must_partner.len() {
                if let Some(partner) = self.must_partner[l_idx] {
                    if labels.binary_search(&partner).is_err() {
                        return true;
                    }
                }
            }
        }
        false
    }
}
