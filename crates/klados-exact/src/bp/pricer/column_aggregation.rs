//! Column-aggregation pricer — generates new columns by merging existing
//! ones from the LP support.  O(k²) per CG call where k = number of
//! columns with x_c > 0 in the LP solution.
//!
//! For each pair of LP-active columns with disjoint leaf sets, try their
//! union as a new column.  If the union is a valid AF component and has
//! positive RC, add it.  Also tries replacing individual leaves.
//!
//! Placed as Tier 0 — runs before the leaf-pair DP.  Catches easy wins
//! without evaluating thousands of pairs.

use klados_core::Tree;

use crate::bp::column::AfColumn;

use super::{Pricer, PricerScratch, PricingContext, PricingResult};

const PRICING_EPS: f64 = 1.0e-8;

pub struct ColumnAggregationPricer;

impl ColumnAggregationPricer {
    pub fn new(_trees: &[Tree]) -> Self {
        Self
    }
}

impl Pricer for ColumnAggregationPricer {
    fn name(&self) -> &'static str {
        "column-aggregation"
    }

    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult {
        let cols = ctx.columns;
        let n = cols.len();
        if n < 2 {
            return PricingResult::Exhausted;
        }

        // Find columns with x_c > 0 in the LP (we don't have values here,
        // so use the dual-based RC check as a proxy).
        let mut found = Vec::new();

        for i in 0..n {
            let ci = &cols[i];
            if ci.size() < 2 {
                continue;
            }
            let sci = ci.pricing_score(ctx.alpha, ctx.beta);
            if sci <= 1.0 + PRICING_EPS {
                continue;
            }

            // Try extending ci with every other LP-active column.
            for j in (i + 1)..n {
                let cj = &cols[j];
                if cj.size() < 1 {
                    continue;
                }
                let scj = cj.pricing_score(ctx.alpha, ctx.beta);
                if scj <= 1.0 + PRICING_EPS {
                    continue;
                }

                // Union of leaf sets if disjoint.
                if !are_disjoint(ci.labels(), cj.labels()) {
                    continue;
                }
                let mut labels = ci.labels().to_vec();
                labels.extend_from_slice(cj.labels());
                labels.sort_unstable();
                labels.dedup();

                if labels.len() < 3 {
                    continue;
                }
                if ctx.seen.contains(&labels) {
                    continue;
                }

                // Validate as AF component.
                let column = match scratch.builder.try_build(labels, ctx.trees) {
                    Some(c) => c,
                    None => continue,
                };
                if ctx.branchings.forbids(&column) {
                    continue;
                }
                let score = column.pricing_score(ctx.alpha, ctx.beta);
                if score > 1.0 + PRICING_EPS {
                    found.push(column);
                    if found.len() >= 16 {
                        return PricingResult::Found(found);
                    }
                }
            }
        }

        if !found.is_empty() {
            return PricingResult::Found(found);
        }
        PricingResult::Exhausted
    }
}

fn are_disjoint(a: &[u32], b: &[u32]) -> bool {
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return false,
        }
    }
    true
}
