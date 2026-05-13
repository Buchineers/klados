//! DSSR multi-tree pricer for m ≥ 3.
//!
//! The engine runs the exact O(n²) 2-tree DP on `(T₀,T₁)` as a relaxation.
//! Its score is an upper bound on the true m-tree reduced cost because it
//! omits non-negative β penalties from `T₂..`. Candidate columns are then
//! audited against the full instance and current B&B branchings. Whenever the
//! relaxed best column is unusable, the responsible relaxed anchor is
//! forbidden and the DP is re-run, forcing the next-best state to surface.

use klados_core::Tree;

use crate::bp::column::ViolatingTriplet;
use crate::bp::search::LeafPair;

use super::{Pricer, PricerScratch, PricingContext, PricingResult};

const PRICING_EPS: f64 = 1.0e-8;
const COLUMN_BATCH: usize = 64;
const MAX_DSSR_PASSES_PER_REF: usize = 12;

pub struct MultiTreePairDpPricer {
    batch_size: usize,
    ref_tree_indices: Vec<usize>,
}

impl MultiTreePairDpPricer {
    pub fn new(trees: &[Tree]) -> Self {
        Self {
            batch_size: COLUMN_BATCH,
            ref_tree_indices: choose_reference_trees(trees),
        }
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
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

        let primary = self.ref_tree_indices.first().copied().unwrap_or(1);
        let before = DssrSnapshot::from_scratch(scratch);
        let primary_result = self.price_ref_pair(ctx, scratch, primary);
        match primary_result {
            PricingResult::Found(cols) => return PricingResult::Found(cols),
            PricingResult::Converged => return PricingResult::Converged,
            PricingResult::Exhausted => {}
        }

        if !should_try_alt_refs(ctx, scratch, before) {
            return PricingResult::Exhausted;
        }

        let refs = self.ref_tree_indices.clone();
        for ref_tree_idx in refs.into_iter().skip(1) {
            scratch.pricing_stats.dssr_alt_refs_tried += 1;
            match self.price_ref_pair(ctx, scratch, ref_tree_idx) {
                PricingResult::Found(cols) => return PricingResult::Found(cols),
                PricingResult::Converged => return PricingResult::Converged,
                PricingResult::Exhausted => {}
            }
        }
        PricingResult::Exhausted
    }
}

#[derive(Clone, Copy)]
struct DssrSnapshot {
    candidates: usize,
    invalid: usize,
    found: usize,
    blocked: usize,
    nonprofitable: usize,
}

impl DssrSnapshot {
    fn from_scratch(scratch: &PricerScratch) -> Self {
        let s = &scratch.pricing_stats;
        Self {
            candidates: s.dssr_candidates,
            invalid: s.dssr_invalid,
            found: s.dssr_found,
            blocked: s.dssr_branch_blocked,
            nonprofitable: s.dssr_nonprofitable,
        }
    }
}

fn should_try_alt_refs(
    ctx: &PricingContext,
    scratch: &PricerScratch,
    before: DssrSnapshot,
) -> bool {
    if ctx.num_leaves > 100 || ctx.trees.len() < 4 {
        return false;
    }
    let after = DssrSnapshot::from_scratch(scratch);
    let candidates = after.candidates.saturating_sub(before.candidates);
    if candidates == 0 {
        return false;
    }
    let invalid = after.invalid.saturating_sub(before.invalid);
    let found = after.found.saturating_sub(before.found);
    let blocked = after.blocked.saturating_sub(before.blocked);
    let nonprofitable = after.nonprofitable.saturating_sub(before.nonprofitable);
    let rejected = invalid + blocked + nonprofitable;

    // Alternate reference pairs are for cases where the primary relaxation is
    // loose: many candidates appear, but nearly all are invalid/blocked/full
    // nonprofitable. Avoid extra O(n²) DP passes for tiny/noisy calls.
    if candidates < 24 {
        return false;
    }
    if found > 0 {
        return false;
    }
    rejected * 3 >= candidates * 2
}

impl MultiTreePairDpPricer {
    fn price_ref_pair(
        &mut self,
        ctx: &PricingContext,
        scratch: &mut PricerScratch,
        ref_tree_idx: usize,
    ) -> PricingResult {
        let mut forbidden: Vec<(u32, u32)> = Vec::new();
        let mut found = Vec::new();
        let mut used_decremental_cuts = false;
        let mut passes = 0usize;

        loop {
            passes += 1;
            scratch.pricing_stats.dssr_passes += 1;
            let n0 = ctx.trees[0].num_nodes();
            let n1 = ctx.trees[ref_tree_idx].num_nodes();
            let mut cache = scratch
                .exact_dp_cache
                .take()
                .filter(|c| c.fits(n0, n1, ctx.num_leaves))
                .unwrap_or_else(|| {
                    super::exact_pair_dp::ExactPairDpCache::new(n0, n1, ctx.num_leaves)
                });

            let output = super::exact_pair_dp::collect_positive_candidates_ref(
                ctx,
                &mut cache,
                ref_tree_idx,
                &forbidden,
            );
            scratch.exact_dp_cache = Some(cache);
            scratch.pricing_stats.dssr_candidates += output.candidates.len();

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
                let cols = scratch.emit_with_reserve(found, ctx, self.batch_size);
                return PricingResult::Found(cols);
            }

            let mut added_forbidden = false;

            for cand in output.candidates {
                if ctx.seen.contains(&cand.labels) {
                    scratch.pricing_stats.dssr_seen += 1;
                    // Seen columns are already present in the RMP. For
                    // pricing we need the best *not-yet-present* column, so a
                    // seen relaxed optimum has the same "one-best-per-state"
                    // pathology as an invalid/branch-blocked optimum: cut its
                    // closed anchor and rerun to expose the second-best state.
                    added_forbidden |= push_forbidden(&mut forbidden, (cand.anchor0, cand.anchor1));
                    used_decremental_cuts = true;
                    continue;
                }

                if let Some(anchor) = branching_forbidden_anchor(&cand.labels, ctx, ref_tree_idx) {
                    scratch.pricing_stats.dssr_branch_blocked += 1;
                    added_forbidden |= push_forbidden(&mut forbidden, anchor);
                    added_forbidden |= push_forbidden(&mut forbidden, (cand.anchor0, cand.anchor1));
                    used_decremental_cuts = true;
                    continue;
                }

                let column = match scratch
                    .builder
                    .try_build_with_violation(cand.labels.clone(), ctx.trees)
                {
                    Ok(c) => c,
                    Err(v) => {
                        scratch.pricing_stats.dssr_invalid += 1;
                        let triplet_anchor = triplet_anchor_ref(v, ctx.trees, ref_tree_idx);
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
                    scratch.pricing_stats.dssr_branch_blocked += 1;
                    added_forbidden |= push_forbidden(&mut forbidden, (cand.anchor0, cand.anchor1));
                    used_decremental_cuts = true;
                    continue;
                }

                let full_score = column.pricing_score(ctx.alpha, ctx.beta);
                if full_score > 1.0 + PRICING_EPS {
                    scratch.pricing_stats.dssr_found += 1;
                    found.push(column);
                    // Continue k-best enumeration within this CG call instead
                    // of returning a trickle of columns. The found column will
                    // be emitted from `found`; forbidding its relaxed anchor
                    // lets the DP surface the next-best alternative.
                    if found.len() < self.batch_size {
                        added_forbidden |=
                            push_forbidden(&mut forbidden, (cand.anchor0, cand.anchor1));
                        used_decremental_cuts = true;
                    }
                } else {
                    scratch.pricing_stats.dssr_nonprofitable += 1;
                    // Relaxed-positive but full-nonprofitable columns can
                    // hide the full-profitable second-best at the same DP
                    // state. Cut this relaxed optimum and continue.
                    added_forbidden |= push_forbidden(&mut forbidden, (cand.anchor0, cand.anchor1));
                    used_decremental_cuts = true;
                }
            }

            if found.len() >= self.batch_size
                || (!found.is_empty() && (!added_forbidden || passes >= MAX_DSSR_PASSES_PER_REF))
            {
                let cols = scratch.emit_with_reserve(found, ctx, self.batch_size);
                return PricingResult::Found(cols);
            }

            if passes >= MAX_DSSR_PASSES_PER_REF {
                return PricingResult::Exhausted;
            }

            if !added_forbidden {
                return PricingResult::Exhausted;
            }
        }
    }
}

fn choose_reference_trees(trees: &[Tree]) -> Vec<usize> {
    let m = trees.len();
    if m <= 1 {
        return Vec::new();
    }
    if m == 2 {
        return vec![1];
    }

    let n = trees[0].num_leaves as usize;
    // Extra reference pairs are valuable on small/medium hard branched
    // instances, but on large high-m cases they multiply the O(n²) DP cost
    // at exactly the nodes where the old benchmark is already dominated by
    // pricing. Keep large cases on the primary pair until we have a cheaper
    // adaptive trigger.
    if n > 100 {
        return vec![1];
    }

    let mut sample: Vec<u32> = if n <= 24 {
        (1..=n as u32).collect()
    } else {
        let step = (n / 24).max(1);
        (1..=n).step_by(step).take(24).map(|x| x as u32).collect()
    };
    sample.sort_unstable();
    sample.dedup();

    let mut scored = Vec::new();
    for k in 1..m {
        let mut disagreements = 0usize;
        let mut total = 0usize;
        for i in 0..sample.len() {
            for j in (i + 1)..sample.len() {
                for l in (j + 1)..sample.len() {
                    total += 1;
                    let a = sample[i];
                    let b = sample[j];
                    let c = sample[l];
                    if triplet_outgroup(&trees[0], a, b, c) != triplet_outgroup(&trees[k], a, b, c)
                    {
                        disagreements += 1;
                    }
                }
            }
        }
        scored.push((disagreements, total, k));
    }

    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.2.cmp(&b.2)));
    let mut refs = Vec::new();
    for &(_, _, k) in &scored {
        refs.push(k);
        if refs.len() >= 3 {
            break;
        }
    }
    if refs.is_empty() {
        refs.push(1);
    }
    refs
}

fn push_forbidden(forbidden: &mut Vec<(u32, u32)>, anchor: (u32, u32)) -> bool {
    if forbidden.contains(&anchor) {
        false
    } else {
        forbidden.push(anchor);
        true
    }
}

fn branching_forbidden_anchor(
    labels: &[u32],
    ctx: &PricingContext,
    ref_tree_idx: usize,
) -> Option<(u32, u32)> {
    for &pair in ctx.branchings.cannot_link() {
        if has_label(labels, pair.a) && has_label(labels, pair.b) {
            return Some(pair_anchor_ref(pair, ctx.trees, ref_tree_idx));
        }
    }
    for &pair in ctx.branchings.must_link() {
        let has_a = has_label(labels, pair.a);
        let has_b = has_label(labels, pair.b);
        if has_a != has_b {
            // There is no safe pure LCA-pair cut for "contains exactly one".
            // Cutting the candidate anchor is conservative for generation and
            // we only let safety tiers certify convergence after such cuts.
            return Some(full_labels_anchor_ref(labels, ctx.trees, ref_tree_idx));
        }
    }
    None
}

fn has_label(labels: &[u32], label: u32) -> bool {
    labels.binary_search(&label).is_ok()
}

fn pair_anchor_ref(pair: LeafPair, trees: &[Tree], ref_tree_idx: usize) -> (u32, u32) {
    let u = lca2(&trees[0], pair.a, pair.b);
    let v = lca2(&trees[ref_tree_idx], pair.a, pair.b);
    (u, v)
}

fn triplet_anchor_ref(v: ViolatingTriplet, trees: &[Tree], ref_tree_idx: usize) -> (u32, u32) {
    (
        lca3(&trees[0], v.a, v.b, v.c),
        lca3(&trees[ref_tree_idx], v.a, v.b, v.c),
    )
}

fn full_labels_anchor_ref(labels: &[u32], trees: &[Tree], ref_tree_idx: usize) -> (u32, u32) {
    (
        lca_labels(&trees[0], labels),
        lca_labels(&trees[ref_tree_idx], labels),
    )
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

fn triplet_outgroup(tree: &Tree, a: u32, b: u32, c: u32) -> u32 {
    let na = tree.node_by_label(a);
    let nb = tree.node_by_label(b);
    let nc = tree.node_by_label(c);
    let nab = tree.nearest_common_ancestor(na, nb);
    let nac = tree.nearest_common_ancestor(na, nc);
    let nbc = tree.nearest_common_ancestor(nb, nc);
    if nab == nac {
        a
    } else if nab == nbc {
        b
    } else {
        c
    }
}
