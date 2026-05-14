//! Small-component enumerating pricer.
//!
//! Many exact_pub optima for high-m instances are mostly singleton forests
//! plus a handful of pairs/triples/quartets. The DP pricers can miss these at
//! branched nodes because their state stores one best representative. This
//! tier enumerates valid small components once, then re-scores them under the
//! current duals. It is a column generator only and never certifies
//! convergence.

use klados_core::Tree;

use crate::bp::column::{AfColumn, ColumnBuilder};

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

        // Pass 1: index-only scan. Compute pricing_score for every
        // surviving cache entry and collect `(score, idx)` pairs.
        // Avoids cloning the AfColumn until we know it's in the top-K.
        let mut scored: Vec<(f64, usize)> = Vec::new();
        for (i, col) in cache.iter().enumerate() {
            if ctx.seen.contains(col.labels()) || ctx.branchings.forbids(col) {
                continue;
            }
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
