//! Faithful port of the legacy `ExactPricer2Tree` from `maf_branch_price_multi.rs`.
//!
//! This is the **exact bottom-up DP for m=2** — the same recurrence used by
//! the proven-correct `bp-multi` solver.  It is the convergence-proving tier
//! in [`super::dispatch_by_m`] for m=2.
//!
//! Separate from the heuristic [`super::PairDpPricer`] (Steel-Warnow style)
//! whose recurrence is not equivalent and can miss columns.

use klados_core::{NONE, Tree};

use super::{Pricer, PricerScratch, PricingContext, PricingResult, adaptive_m2_batch_size};

const PRICING_EPS: f64 = 1.0e-8;
const NEG_INF: f64 = f64::NEG_INFINITY;

#[derive(Clone, Debug)]
pub(crate) struct PairDpCandidate {
    pub score: f64,
    pub labels: Vec<u32>,
    pub anchor0: u32,
    pub anchor1: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct PairDpOutput {
    pub candidates: Vec<PairDpCandidate>,
}

#[derive(Clone, Copy)]
struct DpClosed {
    score: f64,
    v_l: u32,
    v_r: u32,
}

impl Default for DpClosed {
    fn default() -> Self {
        Self {
            score: NEG_INF,
            v_l: 0,
            v_r: 0,
        }
    }
}

#[derive(Clone, Copy)]
struct DpOpen {
    score: f64,
    choice: u8,
}

impl Default for DpOpen {
    fn default() -> Self {
        Self {
            score: NEG_INF,
            choice: 0,
        }
    }
}

/// Persistent DP-table storage reused across CG iterations within a B&B node.
/// Avoids the ~20 MB allocation + zeroing of `n0 × n1` cell arrays on every
/// CG call — exactly matching the legacy `Dp2TreeCache` pattern.
#[derive(Clone)]
pub struct ExactPairDpCache {
    dp_closed: Vec<DpClosed>,
    dp_open: Vec<DpOpen>,
    active_labels: Vec<bool>,
    t0_active: Vec<bool>,
    t1_active: Vec<bool>,
    best_l0: Vec<(f64, u32)>,
    best_r0: Vec<(f64, u32)>,
}

impl ExactPairDpCache {
    pub fn new(n0: usize, n1: usize, num_leaves: usize) -> Self {
        Self {
            dp_closed: vec![DpClosed::default(); n0 * n1],
            dp_open: vec![DpOpen::default(); n0 * n1],
            active_labels: vec![false; num_leaves + 1],
            t0_active: vec![false; n0],
            t1_active: vec![false; n1],
            best_l0: vec![(NEG_INF, 0u32); n1],
            best_r0: vec![(NEG_INF, 0u32); n1],
        }
    }

    /// Whether this cache can serve a `(n0, n1, num_leaves)` problem.
    ///
    /// A max-sized cache may be reused for smaller windows, but all DP indexing
    /// must use the cache's allocated stride (`self.stride()`), not the current
    /// problem's `n1`. Reconstruction follows stored child pointers, so mixing
    /// strides can read stale cells from an older window and turn their pointers
    /// into wild indices.
    pub(crate) fn fits(&self, n0: usize, n1: usize, num_leaves: usize) -> bool {
        self.t0_active.len() >= n0
            && self.t1_active.len() >= n1
            && self.active_labels.len() > num_leaves
    }

    #[inline]
    fn stride(&self) -> usize {
        self.t1_active.len()
    }
}

pub struct ExactPairDpPricer {
    max_per_call: usize,
}

impl ExactPairDpPricer {
    pub fn new(trees: &[Tree]) -> Self {
        assert_eq!(trees.len(), 2, "ExactPairDpPricer requires m=2");
        Self { max_per_call: 64 }
    }
}

impl Pricer for ExactPairDpPricer {
    fn name(&self) -> &'static str {
        "exact-pair-dp"
    }

    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult {
        debug_assert_eq!(ctx.trees.len(), 2);
        price_exact_pair_dp(
            ctx,
            scratch,
            self.max_per_call
                .min(adaptive_m2_batch_size(ctx, scratch.m2_batch)),
        )
    }
}

/// Cell budget for the exact DP's `n0 × n1` table. Each cell costs 32 bytes
/// (`DpClosed` + `DpOpen`), so the default 64M cells ≈ 2 GB — comfortably
/// within the 8 GB platform budget while covering every exact-track m=2 core.
/// Above this the pricer declines (returns `Exhausted`) and the exhaustive
/// leaf-pair tier behind it provides the sound convergence proof instead.
pub(crate) fn exact_dp_cell_cap(scratch: &PricerScratch) -> usize {
    scratch.m2_exact_dp_cells
}

fn price_exact_pair_dp(
    ctx: &PricingContext,
    scratch: &mut PricerScratch,
    max_per_call: usize,
) -> PricingResult {
    let t0 = &ctx.trees[0];
    let t1 = &ctx.trees[1];
    let n0 = t0.num_nodes();
    let n1 = t1.num_nodes();
    let nl = ctx.num_leaves;

    // Memory guard: when the O(n²) table would exceed the cell budget, decline
    // rather than allocate. Returning `Exhausted` (never `Converged`) keeps
    // this tier sound — it cascades to the exhaustive leaf-pair verifier,
    // which proves convergence without the dense table.
    if n0.saturating_mul(n1) > exact_dp_cell_cap(scratch) {
        return PricingResult::Improving;
    }

    let mut cache = scratch
        .exact_dp_cache
        .take()
        .filter(|c| c.fits(n0, n1, nl))
        .unwrap_or_else(|| ExactPairDpCache::new(n0, n1, nl));
    let candidates = collect_positive_columns(ctx, &mut cache);
    scratch.exact_dp_cache = Some(cache);
    if ctx.is_cancelled() {
        return PricingResult::Improving;
    }

    if candidates.is_empty() {
        return PricingResult::Converged;
    }

    let mut found = Vec::new();
    let mut blocked_positive = false;
    let mut seen_positive = false;
    let reserve_cap = exact_reserve_cap(max_per_call, scratch);
    for (_, labels) in candidates {
        let column = scratch.builder.build_unchecked(labels, ctx.trees);
        // `forbids` MUST be checked before `seen`. A column that is both
        // already-in-pool and forbidden by branching is bounded to zero in
        // this node's RMP — it does not serve the LP. Counting it as benign
        // `seen` would let the pricer declare `Converged` while an allowed,
        // improving column may still exist at the same anchor → unsound bound.
        if ctx.branchings.forbids(&column) {
            blocked_positive = true;
            continue;
        }
        if ctx.seen.contains(column.labels()) {
            seen_positive = true;
            continue;
        }
        found.push(column);
        // Keep a bounded reserve of exact-DP positives. The composite pricer
        // drains this before invoking any tier on later CG iterations, so a
        // batch of useful columns can avoid several full exact-DP calls.
        if found.len() >= reserve_cap {
            break;
        }
    }

    if !found.is_empty() {
        let cols = scratch.emit_with_reserve(found, ctx, max_per_call);
        PricingResult::Found(cols)
    } else if blocked_positive {
        PricingResult::Improving
    } else if seen_positive {
        // Match the legacy bp-multi behaviour on unconstrained 2-tree CG:
        // if the exact anchor DP found only columns that are already in the
        // global pool, there is no *new* positive column to add.  Falling
        // through to the leaf-pair tier here is both redundant and extremely
        // expensive on hard decomposed subproblems (it repeatedly full-scans
        // p² leaf anchors just to rediscover pool columns).  Under branch
        // constraints we still return Exhausted for blocked positives above,
        // because a different same-anchor column may satisfy the branch.
        PricingResult::Converged
    } else {
        PricingResult::Converged
    }
}

fn exact_reserve_cap(max_per_call: usize, scratch: &PricerScratch) -> usize {
    let default_cap = max_per_call.saturating_mul(8).max(max_per_call);
    if scratch.m2_exact_reserve_cap > 0 {
        scratch.m2_exact_reserve_cap.max(max_per_call)
    } else {
        default_cap
    }
}

pub(crate) fn collect_positive_columns(
    ctx: &PricingContext,
    cache: &mut ExactPairDpCache,
) -> Vec<(f64, Vec<u32>)> {
    collect_positive_candidates(ctx, cache, &[])
        .candidates
        .into_iter()
        .map(|c| (c.score, c.labels))
        .collect()
}

pub(crate) fn collect_positive_candidates(
    ctx: &PricingContext,
    cache: &mut ExactPairDpCache,
    forbidden_anchors: &[(u32, u32)],
) -> PairDpOutput {
    collect_positive_candidates_ref(ctx, cache, 1, forbidden_anchors)
}

/// Corridor enumeration for m=2: returns every (anchor-best) AF column
/// whose reduced cost under the current duals is ≤ `gamma`. Equivalently,
/// every column with `pricing_score(c) ≥ 1 − gamma`.
///
/// **Soundness.** After root CG converges with LP value `L` and incumbent
/// `U`, by LP duality every column in any improving integer solution
/// (size < U) has reduced cost `≤ γ := U − 1 − L`. So the union of
/// (a) every column already in the pool (rc < 0 after CG) and
/// (b) every column returned here (rc ≤ γ, anchor-best)
/// contains the columns of any improving solution **at the anchor level**:
/// any anchor (u, v) whose best column has rc ≤ γ is captured here.
///
/// **What's not enumerated.** If multiple valid AF columns share the
/// same `(LCA_{T0}, LCA_{T1})` anchor and a non-best one has rc ≤ γ but
/// the best at that anchor has rc < threshold-eligible, the non-best is
/// missed. In practice the DP picks the smallest valid column at each
/// anchor (the leaf-minimal one), so larger columns at the same anchor
/// have strictly higher score — meaning when an anchor enters the
/// corridor, its biggest valid column does. This characterization is
/// not a completeness proof; for proven-optimal corridor solving the
/// reconstruction also needs to enumerate top-K per anchor (deferred).
pub(crate) fn collect_corridor_candidates(
    ctx: &PricingContext,
    cache: &mut ExactPairDpCache,
    gamma: f64,
    forbidden_anchors: &[(u32, u32)],
) -> PairDpOutput {
    collect_corridor_candidates_ref(ctx, cache, 1, gamma, forbidden_anchors)
}

pub(crate) fn collect_corridor_candidates_ref(
    ctx: &PricingContext,
    cache: &mut ExactPairDpCache,
    ref_tree_idx: usize,
    gamma: f64,
    forbidden_anchors: &[(u32, u32)],
) -> PairDpOutput {
    let threshold = 1.0 - gamma - PRICING_EPS;
    collect_candidates_above(ctx, cache, ref_tree_idx, forbidden_anchors, threshold)
}

pub(crate) fn collect_positive_candidates_ref(
    ctx: &PricingContext,
    cache: &mut ExactPairDpCache,
    ref_tree_idx: usize,
    forbidden_anchors: &[(u32, u32)],
) -> PairDpOutput {
    // Original pricer behaviour: enumerate every anchor-best column with
    // strictly positive reduced cost (`score > 1 + ε`), i.e. score > the
    // "strictly improving" threshold. Routed through the generic threshold
    // path with `threshold = 1 + ε`.
    collect_candidates_above(
        ctx,
        cache,
        ref_tree_idx,
        forbidden_anchors,
        1.0 + PRICING_EPS,
    )
}

/// Run the m=2 DP and return every anchor-best column whose score is
/// strictly above `threshold`. The DP and reconstruction logic is the
/// same; only the inclusion filter is parametrised.
fn collect_candidates_above(
    ctx: &PricingContext,
    cache: &mut ExactPairDpCache,
    ref_tree_idx: usize,
    forbidden_anchors: &[(u32, u32)],
    threshold: f64,
) -> PairDpOutput {
    let t0 = &ctx.trees[0];
    let t1 = &ctx.trees[ref_tree_idx];
    let n0 = t0.num_nodes();
    let n1 = t1.num_nodes();
    debug_assert!(cache.fits(n0, n1, ctx.num_leaves));
    let stride = cache.stride();
    let idx = |u: usize, v: usize| -> usize { u * stride + v };
    let is_forbidden =
        |u: u32, v: u32| -> bool { forbidden_anchors.iter().any(|&(fu, fv)| fu == u && fv == v) };

    let active_labels = &mut cache.active_labels;
    let t0_active = &mut cache.t0_active;
    let t1_active = &mut cache.t1_active;
    let dp_closed = &mut cache.dp_closed;
    let dp_open = &mut cache.dp_open;
    let best_l0 = &mut cache.best_l0;
    let best_r0 = &mut cache.best_r0;

    let nl = ctx.num_leaves;
    active_labels[..=nl].fill(false);
    t0_active.fill(false);
    t1_active.fill(false);
    best_l0.fill((NEG_INF, 0u32));
    best_r0.fill((NEG_INF, 0u32));
    // Must-linked leaves stay active even with alpha <= 0. A must-link
    // constraint can force such a leaf into an improving column; excluding it
    // would make the pricer unable to build that column and falsely report
    // convergence — an unsound LP bound.
    for pair in ctx.branchings.must_link() {
        if (pair.a as usize) <= nl {
            active_labels[pair.a as usize] = true;
        }
        if (pair.b as usize) <= nl {
            active_labels[pair.b as usize] = true;
        }
    }
    for label in 1..=nl {
        if ctx.alpha[label] <= 1.0e-12 && !active_labels[label] {
            continue;
        }
        active_labels[label] = true;

        let mut cur = t0.node_by_label(label as u32);
        while cur != NONE && !t0_active[cur as usize] {
            t0_active[cur as usize] = true;
            cur = t0.parent[cur as usize];
        }

        let mut cur = t1.node_by_label(label as u32);
        while cur != NONE && !t1_active[cur as usize] {
            t1_active[cur as usize] = true;
            cur = t1.parent[cur as usize];
        }
    }

    let t0_post: Vec<u32> = t0.post_order().filter(|&u| t0_active[u as usize]).collect();
    let t1_post: Vec<u32> = t1.post_order().filter(|&v| t1_active[v as usize]).collect();
    if t0_post.is_empty() || t1_post.is_empty() {
        return PairDpOutput {
            candidates: Vec::new(),
        };
    }

    let beta_ref = &ctx.beta[ref_tree_idx];
    let beta_0 = &ctx.beta[0];

    let mut cancelled = false;
    for &u in &t0_post {
        // Cooperative cancellation: this O(n0·n1) DP is the longest
        // uninterruptible step in the solver; bail promptly on SIGTERM. The
        // partial result is discarded by the caller on termination, so leaving
        // the DP incomplete is safe.
        if ctx.is_cancelled() {
            cancelled = true;
            break;
        }
        let u_idx = u as usize;

        if t0.is_leaf(u) {
            let lbl = t0.label[u_idx] as usize;
            let u_offset = u_idx * stride;
            for &v in &t1_post {
                let i = u_offset + v as usize;
                dp_closed[i] = DpClosed::default();
                dp_open[i] = DpOpen {
                    score: NEG_INF,
                    choice: 0,
                };
            }
            if active_labels[lbl] {
                let v = t1.node_by_label(lbl as u32);
                let i = u_offset + v as usize;
                dp_closed[i].score = ctx.alpha[lbl];
                dp_open[i].score = ctx.alpha[lbl];
            }
            continue;
        }

        let (l0, r0) = t0.children_pair(u);
        let l0_idx = l0 as usize;
        let r0_idx = r0 as usize;
        let l0_active = t0_active[l0_idx];
        let r0_active = t0_active[r0_idx];
        let l0_offset = l0_idx * stride;
        let r0_offset = r0_idx * stride;
        let u_offset = u_idx * stride;

        let beta_0_l0 = beta_0[l0_idx];
        let beta_0_r0 = beta_0[r0_idx];
        let c_u = beta_0[u_idx] + beta_0_l0 + beta_0_r0;

        for &v in &t1_post {
            let v_idx = v as usize;
            let is_leaf = t1.is_leaf(v);
            let children = if !is_leaf {
                Some(t1.children_pair(v))
            } else {
                None
            };

            // Compute best_l0 bottom-up
            let mut max_s_l = if l0_active {
                dp_open[l0_offset + v_idx].score
            } else {
                NEG_INF
            };
            let mut best_v_l = v;
            if let Some((l1, r1)) = children {
                let s_l = best_l0[l1 as usize].0 - beta_ref[l1 as usize];
                if s_l > max_s_l {
                    max_s_l = s_l;
                    best_v_l = best_l0[l1 as usize].1;
                }
                let s_r = best_l0[r1 as usize].0 - beta_ref[r1 as usize];
                if s_r > max_s_l {
                    max_s_l = s_r;
                    best_v_l = best_l0[r1 as usize].1;
                }
            }
            best_l0[v_idx] = (max_s_l, best_v_l);

            // Compute best_r0 bottom-up
            let mut max_s_r = if r0_active {
                dp_open[r0_offset + v_idx].score
            } else {
                NEG_INF
            };
            let mut best_v_r = v;
            if let Some((l1, r1)) = children {
                let s_l = best_r0[l1 as usize].0 - beta_ref[l1 as usize];
                if s_l > max_s_r {
                    max_s_r = s_l;
                    best_v_r = best_r0[l1 as usize].1;
                }
                let s_r = best_r0[r1 as usize].0 - beta_ref[r1 as usize];
                if s_r > max_s_r {
                    max_s_r = s_r;
                    best_v_r = best_r0[r1 as usize].1;
                }
            }
            best_r0[v_idx] = (max_s_r, best_v_r);
        }

        for &v in &t1_post {
            let v_idx = v as usize;
            let i = u_offset + v_idx;

            dp_closed[i] = DpClosed::default();

            if !t1.is_leaf(v) && !is_forbidden(u, v) {
                let (l1, r1) = t1.children_pair(v);

                let mut best_c_score = NEG_INF;
                let mut v_l = 0;
                let mut v_r = 0;

                let s_l0_l1 = best_l0[l1 as usize].0 - beta_ref[l1 as usize];
                let s_r0_r1 = best_r0[r1 as usize].0 - beta_ref[r1 as usize];
                if s_l0_l1 > NEG_INF / 2.0 && s_r0_r1 > NEG_INF / 2.0 {
                    let s = s_l0_l1 + s_r0_r1 - c_u - beta_ref[v_idx];
                    if s > best_c_score {
                        best_c_score = s;
                        v_l = best_l0[l1 as usize].1;
                        v_r = best_r0[r1 as usize].1;
                    }
                }

                let s_l0_r1 = best_l0[r1 as usize].0 - beta_ref[r1 as usize];
                let s_r0_l1 = best_r0[l1 as usize].0 - beta_ref[l1 as usize];
                if s_l0_r1 > NEG_INF / 2.0 && s_r0_l1 > NEG_INF / 2.0 {
                    let s = s_l0_r1 + s_r0_l1 - c_u - beta_ref[v_idx];
                    if s > best_c_score {
                        best_c_score = s;
                        v_l = best_l0[r1 as usize].1;
                        v_r = best_r0[l1 as usize].1;
                    }
                }

                if best_c_score > NEG_INF / 2.0 {
                    dp_closed[i] = DpClosed {
                        score: best_c_score,
                        v_l,
                        v_r,
                    };
                }
            }

            let mut best_o_score = NEG_INF;
            let mut choice = 0;

            let closed = dp_closed[i].score;
            if closed > NEG_INF / 2.0 {
                best_o_score = closed + beta_0[u_idx] + beta_ref[v_idx];
            }

            let s_l0 = if l0_active {
                dp_open[l0_offset + v_idx].score - beta_0_l0
            } else {
                NEG_INF
            };
            if s_l0 > best_o_score {
                best_o_score = s_l0;
                choice = 1;
            }

            let s_r0 = if r0_active {
                dp_open[r0_offset + v_idx].score - beta_0_r0
            } else {
                NEG_INF
            };
            if s_r0 > best_o_score {
                best_o_score = s_r0;
                choice = 2;
            }

            dp_open[i] = DpOpen {
                score: best_o_score,
                choice,
            };
        }
    }
    if cancelled {
        return PairDpOutput {
            candidates: Vec::new(),
        };
    }

    let mut results = Vec::new();
    for &u in &t0_post {
        if t0.is_leaf(u) {
            continue;
        }
        for &v in &t1_post {
            if t1.is_leaf(v) {
                continue;
            }
            if is_forbidden(u, v) {
                continue;
            }
            let score = dp_closed[idx(u as usize, v as usize)].score;
            if score > threshold {
                let mut labels = Vec::new();
                extract_closed(u, v, t0, dp_closed, dp_open, stride, &mut labels);
                labels.sort_unstable();
                labels.dedup();
                if labels.len() >= 2 {
                    results.push(PairDpCandidate {
                        score,
                        labels,
                        anchor0: u,
                        anchor1: v,
                    });
                }
            }
        }
    }

    results.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.labels.len().cmp(&a.labels.len()))
            .then_with(|| a.labels.cmp(&b.labels))
    });
    PairDpOutput {
        candidates: results,
    }
}

fn extract_closed(
    u: u32,
    v: u32,
    t0: &Tree,
    dp_closed: &[DpClosed],
    dp_open: &[DpOpen],
    n1: usize,
    out: &mut Vec<u32>,
) {
    let state = &dp_closed[u as usize * n1 + v as usize];
    let (l0, r0) = t0.children_pair(u);
    extract_open(l0, state.v_l, t0, dp_closed, dp_open, n1, out);
    extract_open(r0, state.v_r, t0, dp_closed, dp_open, n1, out);
}

fn extract_open(
    u: u32,
    v: u32,
    t0: &Tree,
    dp_closed: &[DpClosed],
    dp_open: &[DpOpen],
    n1: usize,
    out: &mut Vec<u32>,
) {
    let state = &dp_open[u as usize * n1 + v as usize];
    if t0.is_leaf(u) && state.choice == 0 {
        out.push(t0.label[u as usize]);
        return;
    }
    match state.choice {
        0 => extract_closed(u, v, t0, dp_closed, dp_open, n1, out),
        1 => {
            let (l0, _) = t0.children_pair(u);
            extract_open(l0, v, t0, dp_closed, dp_open, n1, out);
        }
        2 => {
            let (_, r0) = t0.children_pair(u);
            extract_open(r0, v, t0, dp_closed, dp_open, n1, out);
        }
        _ => debug_assert!(false, "invalid exact-pair-dp open choice"),
    }
}
