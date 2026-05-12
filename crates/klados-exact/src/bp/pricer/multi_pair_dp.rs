//! DSSR multi-tree pricer for m ≥ 3.
//!
//! The engine runs the exact O(n²) 2-tree DP on `(T₀,T₁)` as a relaxation.
//! Its score is an upper bound on the true m-tree reduced cost because it
//! omits non-negative β penalties from `T₂..`. Candidate columns are then
//! audited against the full instance and current B&B branchings. Whenever the
//! relaxed best column is unusable, the responsible relaxed anchor is
//! forbidden and the DP is re-run, forcing the next-best state to surface.

use klados_core::Tree;

use crate::bp::column::{AfColumn, ViolatingTriplet};
use crate::bp::search::LeafPair;

use super::{Pricer, PricerScratch, PricingContext, PricingResult};

const PRICING_EPS: f64 = 1.0e-8;
const COLUMN_BATCH: usize = 64;

pub struct MultiTreePairDpPricer {
    batch_size: usize,
}

impl MultiTreePairDpPricer {
    pub fn new(_trees: &[Tree]) -> Self {
        Self {
            batch_size: COLUMN_BATCH,
        }
    }
}

impl Pricer for MultiTreePairDpPricer {
    fn name(&self) -> &'static str {
        "dssr-multi-pair-dp"
    }

    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult {
        if ctx.trees.len() < 3 {
            return PricingResult::Exhausted;
        }

        if let Some(cols) = drain_reserve(ctx, scratch, self.batch_size) {
            return PricingResult::Found(cols);
        }

        let mut forbidden: Vec<(u32, u32)> = Vec::new();
        let mut found: Vec<AfColumn> = Vec::new();
        let mut used_decremental_cuts = false;

        loop {
            let n0 = ctx.trees[0].num_nodes();
            let n1 = ctx.trees[1].num_nodes();
            let mut cache = scratch
                .exact_dp_cache
                .take()
                .filter(|c| c.fits(n0, n1, ctx.num_leaves))
                .unwrap_or_else(|| {
                    super::exact_pair_dp::ExactPairDpCache::new(n0, n1, ctx.num_leaves)
                });

            let output =
                super::exact_pair_dp::collect_positive_candidates(ctx, &mut cache, &forbidden);
            scratch.exact_dp_cache = Some(cache);

            if output.candidates.is_empty() {
                if found.is_empty() {
                    return if !used_decremental_cuts
                        && scratch.column_reserve.is_empty()
                        && output.max_allowed_closed <= 1.0 + PRICING_EPS
                    {
                        PricingResult::Converged
                    } else {
                        // We may have over-approximated a structural cut, or
                        // branch constraints may require a richer state than
                        // pure anchor forbidding. Let later safety tiers try.
                        PricingResult::Exhausted
                    };
                }
                return emit_with_reserve(found, ctx, scratch, self.batch_size);
            }

            let mut added_forbidden = false;

            for cand in output.candidates {
                if ctx.seen.contains(&cand.labels) {
                    continue;
                }

                if let Some(anchor) = branching_forbidden_anchor(&cand.labels, ctx) {
                    added_forbidden |= push_forbidden(&mut forbidden, anchor);
                    added_forbidden |=
                        push_forbidden(&mut forbidden, (cand.anchor0, cand.anchor1));
                    used_decremental_cuts = true;
                    continue;
                }

                let column = match scratch
                    .builder
                    .try_build_with_violation(cand.labels.clone(), ctx.trees)
                {
                    Ok(c) => c,
                    Err(v) => {
                        let triplet_anchor = triplet_anchor_01(v, ctx.trees);
                        added_forbidden |= push_forbidden(&mut forbidden, triplet_anchor);
                        // Always cut the emitted closed anchor too. The
                        // semantic triplet/pair anchor can be below the
                        // relaxed DP state and would otherwise let the exact
                        // same unusable candidate reappear forever.
                        added_forbidden |=
                            push_forbidden(&mut forbidden, (cand.anchor0, cand.anchor1));
                        used_decremental_cuts = true;
                        continue;
                    }
                };

                if ctx.branchings.forbids(&column) {
                    added_forbidden |= push_forbidden(&mut forbidden, (cand.anchor0, cand.anchor1));
                    used_decremental_cuts = true;
                    continue;
                }

                let full_score = column.pricing_score(ctx.alpha, ctx.beta);
                if full_score > 1.0 + PRICING_EPS {
                    found.push(column);
                } else {
                    // Relaxed-positive but full-nonprofitable columns can
                    // hide the full-profitable second-best at the same DP
                    // state. Cut this relaxed optimum and continue.
                    added_forbidden |= push_forbidden(&mut forbidden, (cand.anchor0, cand.anchor1));
                    used_decremental_cuts = true;
                }
            }

            if !found.is_empty() {
                return emit_with_reserve(found, ctx, scratch, self.batch_size);
            }

            if !added_forbidden {
                return PricingResult::Exhausted;
            }
        }
    }
}

fn drain_reserve(
    ctx: &PricingContext,
    scratch: &mut PricerScratch,
    limit: usize,
) -> Option<Vec<AfColumn>> {
    if scratch.column_reserve.is_empty() || limit == 0 {
        return None;
    }

    let mut scored: Vec<(f64, usize)> = Vec::new();
    let mut keep = vec![true; scratch.column_reserve.len()];
    for (i, col) in scratch.column_reserve.iter().enumerate() {
        if ctx.seen.contains(col.labels()) || ctx.branchings.forbids(col) {
            keep[i] = false;
            continue;
        }
        let score = col.pricing_score(ctx.alpha, ctx.beta);
        if score > 1.0 + PRICING_EPS {
            scored.push((score, i));
        } else {
            // Stale under the current duals. If the dual landscape shifts
            // back later, exact DSSR can rediscover it.
            keep[i] = false;
        }
    }

    if scored.is_empty() {
        compact_reserve(&mut scratch.column_reserve, &keep);
        return None;
    }

    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored.truncate(limit);
    for &(_, i) in &scored {
        keep[i] = false;
    }

    let mut selected = vec![false; scratch.column_reserve.len()];
    for &(_, i) in &scored {
        selected[i] = true;
    }

    let mut out = Vec::with_capacity(scored.len());
    let mut old = std::mem::take(&mut scratch.column_reserve);
    for (i, col) in old.drain(..).enumerate() {
        if selected[i] {
            out.push(col);
        } else if keep[i] {
            scratch.column_reserve.push(col);
        }
    }
    Some(out)
}

fn compact_reserve(reserve: &mut Vec<AfColumn>, keep: &[bool]) {
    let mut old = std::mem::take(reserve);
    for (i, col) in old.drain(..).enumerate() {
        if keep[i] {
            reserve.push(col);
        }
    }
}

fn emit_with_reserve(
    mut cols: Vec<AfColumn>,
    ctx: &PricingContext,
    scratch: &mut PricerScratch,
    limit: usize,
) -> PricingResult {
    cols.sort_unstable_by(|a, b| {
        let sa = a.pricing_score(ctx.alpha, ctx.beta);
        let sb = b.pricing_score(ctx.alpha, ctx.beta);
        sb.total_cmp(&sa)
            .then_with(|| b.size().cmp(&a.size()))
            .then_with(|| a.labels().cmp(b.labels()))
    });
    cols.dedup_by(|a, b| a.labels() == b.labels());

    let split = cols.len().min(limit);
    let reserve = cols.split_off(split);
    for col in reserve {
        if !ctx.seen.contains(col.labels())
            && !scratch
                .column_reserve
                .iter()
                .any(|existing| existing.labels() == col.labels())
        {
            scratch.column_reserve.push(col);
        }
    }
    PricingResult::Found(cols)
}

fn push_forbidden(forbidden: &mut Vec<(u32, u32)>, anchor: (u32, u32)) -> bool {
    if forbidden.contains(&anchor) {
        false
    } else {
        forbidden.push(anchor);
        true
    }
}

fn branching_forbidden_anchor(labels: &[u32], ctx: &PricingContext) -> Option<(u32, u32)> {
    for &pair in ctx.branchings.cannot_link() {
        if has_label(labels, pair.a) && has_label(labels, pair.b) {
            return Some(pair_anchor_01(pair, ctx.trees));
        }
    }
    for &pair in ctx.branchings.must_link() {
        let has_a = has_label(labels, pair.a);
        let has_b = has_label(labels, pair.b);
        if has_a != has_b {
            // There is no safe pure LCA-pair cut for "contains exactly one".
            // Cutting the candidate anchor is conservative for generation and
            // we only let safety tiers certify convergence after such cuts.
            return Some(full_labels_anchor_01(labels, ctx.trees));
        }
    }
    None
}

fn has_label(labels: &[u32], label: u32) -> bool {
    labels.binary_search(&label).is_ok()
}

fn pair_anchor_01(pair: LeafPair, trees: &[Tree]) -> (u32, u32) {
    let u = lca2(&trees[0], pair.a, pair.b);
    let v = lca2(&trees[1], pair.a, pair.b);
    (u, v)
}

fn triplet_anchor_01(v: ViolatingTriplet, trees: &[Tree]) -> (u32, u32) {
    (lca3(&trees[0], v.a, v.b, v.c), lca3(&trees[1], v.a, v.b, v.c))
}

fn full_labels_anchor_01(labels: &[u32], trees: &[Tree]) -> (u32, u32) {
    (lca_labels(&trees[0], labels), lca_labels(&trees[1], labels))
}

fn lca2(tree: &Tree, a: u32, b: u32) -> u32 {
    tree.nearest_common_ancestor(tree.node_by_label(a), tree.node_by_label(b))
}

fn lca3(tree: &Tree, a: u32, b: u32, c: u32) -> u32 {
    let ab = lca2(tree, a, b);
    tree.nearest_common_ancestor(ab, tree.node_by_label(c))
}

fn lca_labels(tree: &Tree, labels: &[u32]) -> u32 {
    let mut lca = tree.node_by_label(labels[0]);
    for &label in &labels[1..] {
        lca = tree.nearest_common_ancestor(lca, tree.node_by_label(label));
    }
    lca
}
