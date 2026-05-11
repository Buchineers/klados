//! Anchor-and-extend pricer — cheap heuristic, not in default dispatch.
//!
//! Two passes per call: enumerate all leaf pairs, then greedily extend
//! existing columns.  Useful for experiments; the default resolve path
//! uses [`super::dispatch_by_m`] which does not include this tier.

use klados_core::Tree;

use crate::bp::column::AfColumn;
use crate::bp::search::branchings::LeafPair;

use super::{Pricer, PricerScratch, PricingContext, PricingResult};

const PRICING_EPS: f64 = 1.0e-8;
const MAX_PER_CALL: usize = 64;

pub struct AnchorExtendPricer;

impl AnchorExtendPricer {
    pub fn new(_trees: &[Tree]) -> Self {
        Self
    }
}

impl Pricer for AnchorExtendPricer {
    fn name(&self) -> &'static str {
        "anchor-extend"
    }

    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult {
        let mut scored: Vec<(f64, AfColumn)> = Vec::new();
        price_pairs(ctx, scratch, &mut scored);
        price_extensions(ctx, scratch, &mut scored);

        if scored.is_empty() {
            return PricingResult::Exhausted;
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(MAX_PER_CALL);
        PricingResult::Found(scored.into_iter().map(|(_, c)| c).collect())
    }
}

fn price_pairs(ctx: &PricingContext, scratch: &mut PricerScratch, out: &mut Vec<(f64, AfColumn)>) {
    let n = ctx.num_leaves;
    for a in 1..=n as u32 {
        for b in (a + 1)..=n as u32 {
            let labels = vec![a, b];
            if ctx.seen.contains(&labels) {
                continue;
            }
            if ctx.branchings.cannot_link().contains(&LeafPair::new(a, b)) {
                continue;
            }
            let column = scratch.builder.build_unchecked(labels, ctx.trees);
            let score = column.pricing_score(ctx.alpha, ctx.beta);
            if score > 1.0 + PRICING_EPS {
                out.push((score, column));
            }
        }
    }
}

fn price_extensions(
    ctx: &PricingContext,
    scratch: &mut PricerScratch,
    out: &mut Vec<(f64, AfColumn)>,
) {
    let n = ctx.num_leaves;
    for parent in ctx.columns {
        if parent.size() < 2 {
            continue;
        }
        let parent_score = parent.pricing_score(ctx.alpha, ctx.beta);
        if parent_score <= 1.0 + PRICING_EPS {
            continue;
        }

        let mut in_parent = vec![false; n + 1];
        for &l in parent.labels() {
            in_parent[l as usize] = true;
        }
        for new_leaf in 1..=n as u32 {
            if in_parent[new_leaf as usize] {
                continue;
            }
            let mut labels = parent.labels().to_vec();
            labels.push(new_leaf);
            labels.sort_unstable();
            if ctx.seen.contains(&labels) {
                continue;
            }
            let Some(column) = scratch.builder.try_build(labels, ctx.trees) else {
                continue;
            };
            if ctx.branchings.forbids(&column) {
                continue;
            }
            let score = column.pricing_score(ctx.alpha, ctx.beta);
            if score > parent_score + PRICING_EPS && score > 1.0 + PRICING_EPS {
                out.push((score, column));
            }
        }
    }
}
