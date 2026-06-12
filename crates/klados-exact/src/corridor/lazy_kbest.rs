//! Lazy k-best enumerator over the m=2 pair-DP DAG.
//!
//! ## What this module provides
//!
//! `LazyKBest::next()` returns the next valid AF column in **strictly
//! descending pricing-score order**, lazily expanding the DP DAG only as
//! the caller asks for more columns. Combined with a threshold
//! `1 − γ` it gives a *complete reduced-cost shell oracle*:
//!
//! * If `next()` returns `Some(c)` with `c.score ≥ 1 − γ`: a corridor
//!   column was found.
//! * If `next()` returns a column with `c.score < 1 − γ`, or `None`: no
//!   unseen column with `rc ≤ γ` exists, i.e. the shell at γ is
//!   *certified complete*.
//!
//! ## Why this matters
//!
//! Empirically the integer optimum lies on (or extremely close to) the
//! root LP support face. The "dual-face shell" architecture solves the
//! MIP over the support first, expands to the reduced-cost shell only
//! if needed, and uses this enumerator to *prove* the shell is
//! complete. Unlike top-K-per-cell storage (which has `O(n²·K)` memory
//! and dies at n=2000+), this is output-sensitive: memory grows only
//! with the number of columns the caller asks for.
//!
//! ## Algorithm sketch (Eppstein-style, specialized for the pair-DP)
//!
//! The standard pair-DP fills `dp_closed[u, v]` and `dp_open[u, v]`
//! with **the single best** sub-column at each cell. For lazy k-best
//! we keep that "best" computation as the starting point and grow it
//! per-cell on demand.
//!
//! Each cell maintains a *frontier*: a min/max-heap of "candidate
//! derivations" that haven't yet been emitted at that cell. The
//! `i`-th alternative at cell `C` is produced by popping the
//! top of `C`'s frontier and replacing it with its successors —
//! candidates that differ by exactly one incremented child-alternative
//! index.
//!
//! At the top level, a global heap holds the current best across all
//! **root anchor** cells (`dp_closed[u, v]` for internal `u`/`v`).
//! Calling `next()` pops this heap, reconstructs the column, and
//! pushes the just-popped cell's *next* alternative into the global
//! heap so the next call finds it.
//!
//! ## Status: scaffolding only
//!
//! This module establishes the data shape and exposes a `next()` API
//! that currently returns the **same anchor-best** sequence as the
//! existing `collect_corridor_candidates_ref`. The within-anchor top-K
//! frontier expansion is the next engineering step; the scaffolding is
//! in place so we can profile end-to-end performance with the audit
//! integrated before paying the within-anchor-K complexity cost.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use klados_core::{NONE, Tree};

const NEG_INF: f64 = f64::NEG_INFINITY;
const PRICING_EPS: f64 = 1.0e-8;

#[inline]
fn cell(n1: usize, u: usize, v: usize) -> usize {
    u * n1 + v
}

/// Single-best entries — same layout as the existing pair-DP, kept
/// internal so we control the lifetime via the lazy iterator.
///
/// **Default score is `NEG_INFINITY`**, not `0`: "no valid column at
/// this cell" must propagate as an unreachable score so combination
/// arithmetic at higher levels skips it. A naïve `derive(Default)`
/// would give `f64 = 0.0` and silently corrupt the DP.
#[derive(Clone, Copy, Debug)]
struct ClosedBest {
    score: f64,
    /// Grounded `T₁` node on the left (for `dp_open[l0, ·]`).
    grounded_v_l: u32,
    /// Grounded `T₁` node on the right (for `dp_open[r0, ·]`).
    grounded_v_r: u32,
}
impl Default for ClosedBest {
    fn default() -> Self {
        Self {
            score: NEG_INF,
            grounded_v_l: 0,
            grounded_v_r: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct OpenBest {
    score: f64,
    /// 0 = closed at (u, v), 1 = left T₀-child of `u`, 2 = right T₀-child of `u`.
    choice: u8,
}
impl Default for OpenBest {
    fn default() -> Self {
        Self {
            score: NEG_INF,
            choice: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BestSideBest {
    score: f64,
    /// Grounded `T₁` node where `dp_open[child_t0, grounded_v]` was consulted.
    grounded_v: u32,
}
impl Default for BestSideBest {
    fn default() -> Self {
        Self {
            score: NEG_INF,
            grounded_v: 0,
        }
    }
}

/// A complete column emitted by [`LazyKBest::next`].
#[derive(Clone, Debug)]
pub struct CorridorColumn {
    pub score: f64,
    pub labels: Vec<u32>,
    pub anchor0: u32,
    pub anchor1: u32,
}

/// Persistent storage for the lazy enumerator. Allocated once, reused
/// across CG iterations within the solver.
pub struct LazyKBestCache {
    n0: usize,
    n1: usize,
    num_leaves: usize,

    dp_closed: Vec<ClosedBest>,
    dp_open: Vec<OpenBest>,
    best_l0: Vec<BestSideBest>,
    best_r0: Vec<BestSideBest>,

    active_labels: Vec<bool>,
    t0_active: Vec<bool>,
    t1_active: Vec<bool>,

    /// Per-anchor within-cell K-best state, materialized on first request
    /// past alt-0. Index by `cell(n1, u, v)`.
    anchor_kbest: Vec<Option<Box<AnchorKBest>>>,
}

/// Within-anchor K-best state. For each pairing, holds the sorted list of
/// `(v_l, score)` and `(v_r, score)` candidates, plus a 2D Eppstein heap
/// over `(l_rank, r_rank)`. Top-level merge across the two pairings.
struct AnchorKBest {
    /// `[pairing]` → list of `(score, grounded_v)` sorted descending by
    /// `score`, where score = `dp_open[l0_or_r0, v] − chain_pen(v, partner_top)`.
    sorted_l: [Vec<(f64, u32)>; 2],
    sorted_r: [Vec<(f64, u32)>; 2],
    /// Constant offset subtracted from `(l_score + r_score)` to get the
    /// closed-cell score, per pairing. (Pen + β₁[l1] + β₁[r1].)
    pen: [f64; 2],
    /// 2D-heap frontier merged across pairings.
    frontier: BinaryHeap<AnchorFrontier>,
    /// Emitted alternatives in score-descending order.
    emitted: Vec<AnchorAlt>,
}

#[derive(Clone, Copy, Debug)]
struct AnchorAlt {
    score: f64,
    pairing: u8,
    grounded_v_l: u32,
    grounded_v_r: u32,
}

#[derive(Clone, Copy, Debug)]
struct AnchorFrontier {
    score: f64,
    pairing: u8,
    l_rank: u32,
    r_rank: u32,
}
impl PartialEq for AnchorFrontier {
    fn eq(&self, o: &Self) -> bool {
        self.score == o.score
            && self.pairing == o.pairing
            && self.l_rank == o.l_rank
            && self.r_rank == o.r_rank
    }
}
impl Eq for AnchorFrontier {}
impl PartialOrd for AnchorFrontier {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for AnchorFrontier {
    fn cmp(&self, o: &Self) -> Ordering {
        self.score
            .total_cmp(&o.score)
            .then_with(|| self.pairing.cmp(&o.pairing))
            .then_with(|| self.l_rank.cmp(&o.l_rank))
            .then_with(|| self.r_rank.cmp(&o.r_rank))
    }
}

impl LazyKBestCache {
    pub fn new(n0: usize, n1: usize, num_leaves: usize) -> Self {
        Self {
            n0,
            n1,
            num_leaves,
            dp_closed: vec![ClosedBest::default(); n0 * n1],
            dp_open: vec![OpenBest::default(); n0 * n1],
            best_l0: vec![BestSideBest::default(); n1],
            best_r0: vec![BestSideBest::default(); n1],
            active_labels: vec![false; num_leaves + 1],
            t0_active: vec![false; n0],
            t1_active: vec![false; n1],
            anchor_kbest: (0..n0 * n1).map(|_| None).collect(),
        }
    }

    pub fn fits(&self, n0: usize, n1: usize, num_leaves: usize) -> bool {
        self.n0 == n0 && self.n1 == n1 && self.num_leaves == num_leaves
    }
}

/// One entry in the global heap of "current top from each anchor".
///
/// Ordered by score so `BinaryHeap` pops the highest-scoring first.
#[derive(Clone, Copy, Debug)]
struct RootEntry {
    score: f64,
    /// Internal-node anchor in `T₀`.
    u: u32,
    /// Internal-node anchor in `T₁`.
    v: u32,
    /// Within-anchor alternative index (0 = anchor-best).
    alt: u32,
}

impl PartialEq for RootEntry {
    fn eq(&self, o: &Self) -> bool {
        self.score == o.score && self.u == o.u && self.v == o.v && self.alt == o.alt
    }
}
impl Eq for RootEntry {}
impl PartialOrd for RootEntry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for RootEntry {
    fn cmp(&self, o: &Self) -> Ordering {
        self.score
            .total_cmp(&o.score)
            .then_with(|| self.u.cmp(&o.u))
            .then_with(|| self.v.cmp(&o.v))
            .then_with(|| o.alt.cmp(&self.alt))
    }
}

/// Inputs to the lazy enumerator — duals, threshold, reference tree.
pub struct LazyKBestInput<'a> {
    pub t0: &'a Tree,
    pub t1: &'a Tree,
    pub alpha: &'a [f64],
    pub beta_t0: &'a [f64],
    pub beta_t1: &'a [f64],
    /// Score threshold τ. Iterator stops emitting once `score < τ`.
    pub threshold: f64,
}

/// The lazy k-best iterator. Build with [`LazyKBest::new`] and call
/// [`LazyKBest::next`] repeatedly.
pub struct LazyKBest<'a, 'c> {
    input: LazyKBestInput<'a>,
    cache: &'c mut LazyKBestCache,
    /// Anchors not yet emitted, ordered by current best score at that
    /// anchor.
    heap: BinaryHeap<RootEntry>,
    /// Per-anchor hard cap on within-anchor alts. Many `(v_l, v_r)`
    /// tuples collapse to the same labelset under `reconstruct_open`,
    /// so unbounded expansion wastes effort. Configurable via
    /// `KLADOS_LAZY_KBEST_ANCHOR_K`.
    max_alt: u32,
}

impl<'a, 'c> LazyKBest<'a, 'c> {
    /// Run the forward best-only DP and seed the root heap.
    ///
    /// After this returns, calling `next()` enumerates anchor-best
    /// columns in score-descending order until threshold cuts off.
    pub fn new(input: LazyKBestInput<'a>, cache: &'c mut LazyKBestCache) -> Self {
        Self::run_forward_dp(&input, cache);
        let heap = Self::seed_heap(&input, cache);
        let max_alt = std::env::var("KLADOS_LAZY_KBEST_ANCHOR_K")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(8);
        Self {
            input,
            cache,
            heap,
            max_alt,
        }
    }

    fn run_forward_dp(input: &LazyKBestInput, cache: &mut LazyKBestCache) {
        let t0 = input.t0;
        let t1 = input.t1;
        let n0 = t0.num_nodes();
        let n1 = t1.num_nodes();

        for c in cache.dp_closed.iter_mut() {
            *c = ClosedBest::default();
        }
        for c in cache.dp_open.iter_mut() {
            *c = OpenBest::default();
        }
        cache.active_labels.fill(false);
        cache.t0_active.fill(false);
        cache.t1_active.fill(false);
        // Within-anchor state is stale once duals change; clear lazy slots.
        for slot in cache.anchor_kbest.iter_mut() {
            *slot = None;
        }

        let nl = cache.num_leaves;
        for label in 1..=nl {
            if input.alpha[label] <= 1.0e-12 {
                continue;
            }
            cache.active_labels[label] = true;
            let mut cur = t0.node_by_label(label as u32);
            while cur != NONE && !cache.t0_active[cur as usize] {
                cache.t0_active[cur as usize] = true;
                cur = t0.parent[cur as usize];
            }
            let mut cur = t1.node_by_label(label as u32);
            while cur != NONE && !cache.t1_active[cur as usize] {
                cache.t1_active[cur as usize] = true;
                cur = t1.parent[cur as usize];
            }
        }

        let t0_post: Vec<u32> = t0
            .post_order()
            .filter(|&u| cache.t0_active[u as usize])
            .collect();
        let t1_post: Vec<u32> = t1
            .post_order()
            .filter(|&v| cache.t1_active[v as usize])
            .collect();
        if t0_post.is_empty() || t1_post.is_empty() {
            return;
        }

        for &u in &t0_post {
            let u_idx = u as usize;
            if t0.is_leaf(u) {
                fill_leaf(t0, t1, u, &t1_post, input, cache);
                continue;
            }
            let (l0, r0) = t0.children_pair(u);
            let l0_idx = l0 as usize;
            let r0_idx = r0 as usize;
            let l0_active = cache.t0_active[l0_idx];
            let r0_active = cache.t0_active[r0_idx];

            // Reset dp_closed for cells we'll write at this u.
            for &v in &t1_post {
                cache.dp_closed[cell(n1, u_idx, v as usize)] = ClosedBest::default();
            }

            compute_best_side_pass(
                t1,
                &t1_post,
                l0_idx,
                l0_active,
                n1,
                input,
                &cache.dp_open,
                &mut cache.best_l0,
            );
            compute_best_side_pass(
                t1,
                &t1_post,
                r0_idx,
                r0_active,
                n1,
                input,
                &cache.dp_open,
                &mut cache.best_r0,
            );

            for &v in &t1_post {
                if t1.is_leaf(v) {
                    continue;
                }
                let v_idx = v as usize;
                let (l1, r1) = t1.children_pair(v);
                let pen = input.beta_t0[u_idx]
                    + input.beta_t1[v_idx]
                    + input.beta_t0[l0_idx]
                    + input.beta_t0[r0_idx];

                // Two pairings: (l0↔l1, r0↔r1) and (l0↔r1, r0↔l1). For
                // best-only we just take whichever combo has the higher
                // sum; the lazy within-anchor frontier will eventually
                // expose both alternatives.
                let mut best_score = NEG_INF;
                let mut best_v_l: u32 = 0;
                let mut best_v_r: u32 = 0;
                {
                    let a = cache.best_l0[l1 as usize];
                    let b = cache.best_r0[r1 as usize];
                    if a.score.is_finite() && b.score.is_finite() {
                        let s = a.score + b.score
                            - pen
                            - input.beta_t1[l1 as usize]
                            - input.beta_t1[r1 as usize];
                        if s > best_score {
                            best_score = s;
                            best_v_l = a.grounded_v;
                            best_v_r = b.grounded_v;
                        }
                    }
                }
                {
                    let a = cache.best_l0[r1 as usize];
                    let b = cache.best_r0[l1 as usize];
                    if a.score.is_finite() && b.score.is_finite() {
                        let s = a.score + b.score
                            - pen
                            - input.beta_t1[r1 as usize]
                            - input.beta_t1[l1 as usize];
                        if s > best_score {
                            best_score = s;
                            best_v_l = a.grounded_v;
                            best_v_r = b.grounded_v;
                        }
                    }
                }
                if best_score > NEG_INF / 2.0 {
                    cache.dp_closed[cell(n1, u_idx, v_idx)] = ClosedBest {
                        score: best_score,
                        grounded_v_l: best_v_l,
                        grounded_v_r: best_v_r,
                    };
                }
            }

            // dp_open[u, v] = best of (closed at (u,v), open at children).
            for &v in &t1_post {
                let v_idx = v as usize;
                let cell_idx = cell(n1, u_idx, v_idx);
                let closed_score = cache.dp_closed[cell_idx].score;

                let mut best = OpenBest::default();
                if closed_score > NEG_INF / 2.0 {
                    let s = closed_score + input.beta_t0[u_idx] + input.beta_t1[v_idx];
                    if s > best.score {
                        best.score = s;
                        best.choice = 0;
                    }
                }
                if l0_active {
                    let s = cache.dp_open[cell(n1, l0_idx, v_idx)].score - input.beta_t0[l0_idx];
                    if s > best.score {
                        best.score = s;
                        best.choice = 1;
                    }
                }
                if r0_active {
                    let s = cache.dp_open[cell(n1, r0_idx, v_idx)].score - input.beta_t0[r0_idx];
                    if s > best.score {
                        best.score = s;
                        best.choice = 2;
                    }
                }
                cache.dp_open[cell_idx] = best;
            }
        }
    }

    fn seed_heap(input: &LazyKBestInput, cache: &LazyKBestCache) -> BinaryHeap<RootEntry> {
        let t0 = input.t0;
        let t1 = input.t1;
        let n0 = t0.num_nodes();
        let n1 = t1.num_nodes();
        let mut heap = BinaryHeap::new();
        for u in 0..n0 {
            if t0.is_leaf(u as u32) || !cache.t0_active[u] {
                continue;
            }
            for v in 0..n1 {
                if t1.is_leaf(v as u32) || !cache.t1_active[v] {
                    continue;
                }
                let entry = cache.dp_closed[cell(n1, u, v)];
                if entry.score < input.threshold - PRICING_EPS {
                    continue;
                }
                if !entry.score.is_finite() {
                    continue;
                }
                heap.push(RootEntry {
                    score: entry.score,
                    u: u as u32,
                    v: v as u32,
                    alt: 0,
                });
            }
        }
        heap
    }

    /// Return the next column with score `≥ self.input.threshold`, or
    /// `None` once the shell is exhausted.
    pub fn next(&mut self) -> Option<CorridorColumn> {
        loop {
            let entry = self.heap.pop()?;
            if entry.score < self.input.threshold - PRICING_EPS {
                return None;
            }
            // Resolve the within-anchor alt. For alt-0, prefer the
            // forward-DP K=1 groundings (and only build the lazy state
            // when alt-1 is actually requested, which keeps the easy
            // path zero-allocation). For alt ≥ 1, materialize and look
            // up.
            let (pairing, grounded_v_l, grounded_v_r, score) = if entry.alt == 0 {
                let n1 = self.cache.n1;
                let closed = self.cache.dp_closed[cell(n1, entry.u as usize, entry.v as usize)];
                (0u8, closed.grounded_v_l, closed.grounded_v_r, closed.score)
            } else {
                let alt = self.get_or_compute_alt(entry.u, entry.v, entry.alt as usize);
                match alt {
                    Some(a) => (a.pairing, a.grounded_v_l, a.grounded_v_r, a.score),
                    None => continue,
                }
            };
            let _ = pairing;
            let _ = score;

            let mut labels = Vec::new();
            self.reconstruct_at(entry.u, grounded_v_l, grounded_v_r, &mut labels);
            labels.sort_unstable();
            labels.dedup();

            // Push the next within-anchor alt if it still lives above threshold.
            let next_alt = entry.alt + 1;
            if next_alt < self.max_alt
                && let Some(next_score) =
                    self.peek_or_compute_alt_score(entry.u, entry.v, next_alt as usize)
                && next_score >= self.input.threshold - PRICING_EPS
            {
                self.heap.push(RootEntry {
                    score: next_score,
                    u: entry.u,
                    v: entry.v,
                    alt: next_alt,
                });
            }

            if labels.len() < 2 {
                continue;
            }
            return Some(CorridorColumn {
                score: entry.score,
                labels,
                anchor0: entry.u,
                anchor1: entry.v,
            });
        }
    }

    /// Ensure `anchor_kbest[u, v]` has at least `k+1` emitted alts.
    /// Returns `None` if the anchor has fewer than `k+1` alternatives
    /// (frontier exhausted).
    fn get_or_compute_alt(&mut self, u: u32, v: u32, k: usize) -> Option<AnchorAlt> {
        self.materialize_anchor(u, v);
        let n1 = self.cache.n1;
        let idx = cell(n1, u as usize, v as usize);
        let state = self.cache.anchor_kbest[idx].as_mut()?;
        while state.emitted.len() <= k {
            if !state.advance() {
                return None;
            }
        }
        Some(state.emitted[k])
    }

    fn peek_or_compute_alt_score(&mut self, u: u32, v: u32, k: usize) -> Option<f64> {
        self.get_or_compute_alt(u, v, k).map(|a| a.score)
    }

    /// First-time materialization of within-anchor state: build sorted
    /// `(score, grounded_v)` lists for each pairing and seed the
    /// frontier.
    fn materialize_anchor(&mut self, u: u32, v: u32) {
        let n1 = self.cache.n1;
        let idx = cell(n1, u as usize, v as usize);
        if self.cache.anchor_kbest[idx].is_some() {
            return;
        }
        let t0 = self.input.t0;
        let t1 = self.input.t1;
        let (l0, r0) = t0.children_pair(u);
        let (l1, r1) = t1.children_pair(v);
        let pen_common = self.input.beta_t0[u as usize]
            + self.input.beta_t1[v as usize]
            + self.input.beta_t0[l0 as usize]
            + self.input.beta_t0[r0 as usize];
        let pen0 = pen_common + self.input.beta_t1[l1 as usize] + self.input.beta_t1[r1 as usize];
        let pen1 = pen0; // same offset, different (v_l, v_r) sources

        // Pairing 0: v_l ∈ subtree(l1) using dp_open[l0, ·]; v_r ∈ subtree(r1) using dp_open[r0, ·].
        let mut sorted_l0 = collect_subtree_scores(
            t1,
            l1,
            l0 as usize,
            n1,
            &self.cache.dp_open,
            self.input.beta_t1,
            &self.cache.t1_active,
        );
        let mut sorted_r0 = collect_subtree_scores(
            t1,
            r1,
            r0 as usize,
            n1,
            &self.cache.dp_open,
            self.input.beta_t1,
            &self.cache.t1_active,
        );
        // Pairing 1: v_l ∈ subtree(r1) using dp_open[l0, ·]; v_r ∈ subtree(l1) using dp_open[r0, ·].
        let mut sorted_l1 = collect_subtree_scores(
            t1,
            r1,
            l0 as usize,
            n1,
            &self.cache.dp_open,
            self.input.beta_t1,
            &self.cache.t1_active,
        );
        let mut sorted_r1 = collect_subtree_scores(
            t1,
            l1,
            r0 as usize,
            n1,
            &self.cache.dp_open,
            self.input.beta_t1,
            &self.cache.t1_active,
        );
        for v in [
            &mut sorted_l0,
            &mut sorted_r0,
            &mut sorted_l1,
            &mut sorted_r1,
        ] {
            v.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        }

        let mut frontier = BinaryHeap::new();
        if !sorted_l0.is_empty() && !sorted_r0.is_empty() {
            frontier.push(AnchorFrontier {
                score: sorted_l0[0].0 + sorted_r0[0].0 - pen0,
                pairing: 0,
                l_rank: 0,
                r_rank: 0,
            });
        }
        if !sorted_l1.is_empty() && !sorted_r1.is_empty() {
            frontier.push(AnchorFrontier {
                score: sorted_l1[0].0 + sorted_r1[0].0 - pen1,
                pairing: 1,
                l_rank: 0,
                r_rank: 0,
            });
        }

        self.cache.anchor_kbest[idx] = Some(Box::new(AnchorKBest {
            sorted_l: [sorted_l0, sorted_l1],
            sorted_r: [sorted_r0, sorted_r1],
            pen: [pen0, pen1],
            frontier,
            emitted: Vec::new(),
        }));
    }

    fn reconstruct_at(&self, u: u32, v_l: u32, v_r: u32, out: &mut Vec<u32>) {
        let (l0, r0) = self.input.t0.children_pair(u);
        self.reconstruct_open(l0, v_l, out);
        self.reconstruct_open(r0, v_r, out);
    }

    /// Recurse into closed (u, v) using the forward-DP K=1 groundings.
    /// Used only below the root anchor — within-anchor K-best is rooted
    /// at the top-level anchor and uses K=1 oracles below.
    fn reconstruct_closed_best(&self, u: u32, v: u32, out: &mut Vec<u32>) {
        let n1 = self.cache.n1;
        let entry = self.cache.dp_closed[cell(n1, u as usize, v as usize)];
        let (l0, r0) = self.input.t0.children_pair(u);
        self.reconstruct_open(l0, entry.grounded_v_l, out);
        self.reconstruct_open(r0, entry.grounded_v_r, out);
    }

    fn reconstruct_open(&self, u: u32, v: u32, out: &mut Vec<u32>) {
        if self.input.t0.is_leaf(u) {
            out.push(self.input.t0.label[u as usize]);
            return;
        }
        let n1 = self.cache.n1;
        let entry = self.cache.dp_open[cell(n1, u as usize, v as usize)];
        match entry.choice {
            0 => self.reconstruct_closed_best(u, v, out),
            1 => {
                let (l0, _) = self.input.t0.children_pair(u);
                self.reconstruct_open(l0, v, out);
            }
            2 => {
                let (_, r0) = self.input.t0.children_pair(u);
                self.reconstruct_open(r0, v, out);
            }
            _ => debug_assert!(false, "invalid open choice"),
        }
    }
}

impl AnchorKBest {
    /// Pop one frontier entry, append to `emitted`, push successors.
    /// Returns `false` if the frontier was empty.
    fn advance(&mut self) -> bool {
        let Some(top) = self.frontier.pop() else {
            return false;
        };
        let p = top.pairing as usize;
        let li = top.l_rank as usize;
        let ri = top.r_rank as usize;
        let (l_score, l_v) = self.sorted_l[p][li];
        let (r_score, r_v) = self.sorted_r[p][ri];
        self.emitted.push(AnchorAlt {
            score: top.score,
            pairing: top.pairing,
            grounded_v_l: l_v,
            grounded_v_r: r_v,
        });
        // 2D Eppstein: push (li+1, ri) only when ri==0; always push (li, ri+1).
        // Each (i, j) thus has exactly one path: from (i-1, j) if j==0 else (i, j-1).
        if ri == 0 && li + 1 < self.sorted_l[p].len() {
            let (ls, _) = self.sorted_l[p][li + 1];
            self.frontier.push(AnchorFrontier {
                score: ls + r_score - self.pen[p],
                pairing: top.pairing,
                l_rank: (li + 1) as u32,
                r_rank: 0,
            });
        }
        if ri + 1 < self.sorted_r[p].len() {
            let (rs, _) = self.sorted_r[p][ri + 1];
            self.frontier.push(AnchorFrontier {
                score: l_score + rs - self.pen[p],
                pairing: top.pairing,
                l_rank: li as u32,
                r_rank: (ri + 1) as u32,
            });
        }
        true
    }
}

/// Collect `(score, v)` for all `v ∈ subtree(top)` of T₁ where
/// `dp_open[x, v]` is finite. `score = dp_open[x, v] − chain_pen(v, top)`,
/// where `chain_pen(top, top) = 0` and descending from `p` to `c` adds
/// `β₁[c]`.
fn collect_subtree_scores(
    t1: &Tree,
    top: u32,
    x: usize,
    n1: usize,
    dp_open: &[OpenBest],
    beta_t1: &[f64],
    t1_active: &[bool],
) -> Vec<(f64, u32)> {
    let mut out = Vec::new();
    let mut stack: Vec<(u32, f64)> = vec![(top, 0.0)];
    while let Some((v, cpen)) = stack.pop() {
        if !t1_active[v as usize] {
            continue;
        }
        let dp_val = dp_open[x * n1 + v as usize].score;
        if dp_val.is_finite() {
            out.push((dp_val - cpen, v));
        }
        if !t1.is_leaf(v) {
            let (lc, rc) = t1.children_pair(v);
            stack.push((lc, cpen + beta_t1[lc as usize]));
            stack.push((rc, cpen + beta_t1[rc as usize]));
        }
    }
    out
}

fn fill_leaf(
    t0: &Tree,
    t1: &Tree,
    u: u32,
    t1_post: &[u32],
    input: &LazyKBestInput,
    cache: &mut LazyKBestCache,
) {
    let n1 = cache.n1;
    let u_idx = u as usize;
    let lbl = t0.label[u_idx] as usize;
    for &v in t1_post {
        let i = cell(n1, u_idx, v as usize);
        cache.dp_closed[i] = ClosedBest::default();
        cache.dp_open[i] = OpenBest::default();
    }
    if cache.active_labels[lbl] {
        let v = t1.node_by_label(lbl as u32);
        let v_idx = v as usize;
        let score = input.alpha[lbl];
        cache.dp_closed[cell(n1, u_idx, v_idx)] = ClosedBest {
            score,
            grounded_v_l: 0,
            grounded_v_r: 0,
        };
        cache.dp_open[cell(n1, u_idx, v_idx)] = OpenBest { score, choice: 0 };
    }
}

fn compute_best_side_pass(
    t1: &Tree,
    t1_post: &[u32],
    child_t0_idx: usize,
    child_t0_active: bool,
    n1: usize,
    input: &LazyKBestInput,
    dp_open: &[OpenBest],
    out: &mut [BestSideBest],
) {
    for v in t1_post {
        out[*v as usize] = BestSideBest {
            score: NEG_INF,
            grounded_v: *v,
        };
    }
    for &v in t1_post {
        let v_idx = v as usize;
        let mut best = BestSideBest {
            score: NEG_INF,
            grounded_v: v,
        };
        if child_t0_active {
            let score = dp_open[child_t0_idx * n1 + v_idx].score;
            if score > best.score {
                best.score = score;
                best.grounded_v = v;
            }
        }
        if !t1.is_leaf(v) {
            let (l1, r1) = t1.children_pair(v);
            let l_entry = out[l1 as usize];
            let s = l_entry.score - input.beta_t1[l1 as usize];
            if s > best.score {
                best.score = s;
                best.grounded_v = l_entry.grounded_v;
            }
            let r_entry = out[r1 as usize];
            let s = r_entry.score - input.beta_t1[r1 as usize];
            if s > best.score {
                best.score = s;
                best.grounded_v = r_entry.grounded_v;
            }
        }
        out[v_idx] = best;
    }
}

// ──────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bp::column::{AfColumn, ColumnBuilder, ColumnSet};
    use crate::bp::pricer::PricingContext;
    use crate::bp::pricer::exact_pair_dp::{ExactPairDpCache, collect_corridor_candidates_ref};
    use crate::bp::search::Branchings;
    use klados_core::tree::{Label, NodeId};

    /// Lazy K-best must be a *superset* of the K=1 reference: every
    /// labelset emitted by `collect_corridor_candidates_ref` must also
    /// be emitted by `LazyKBest::next()` (possibly more, since the lazy
    /// enumerator explores within-anchor alternatives).
    #[test]
    fn lazy_kbest_k1_matches_pair_dp_on_balanced_4() {
        // T0 = ((1,2),(3,4))  ;  T1 = ((1,3),(2,4))
        let trees = vec![build_balanced_4(1, 2, 3, 4), build_balanced_4(1, 3, 2, 4)];
        check_k1_equivalence(&trees, 4);
    }

    #[test]
    fn lazy_kbest_k1_matches_pair_dp_on_identical_4() {
        // Identical trees: every leaf-subset is a valid AF column,
        // so the corridor at large γ should be substantial.
        let trees = vec![build_balanced_4(1, 2, 3, 4), build_balanced_4(1, 2, 3, 4)];
        check_k1_equivalence(&trees, 4);
    }

    fn check_k1_equivalence(trees: &[Tree], n: usize) {
        let n0 = trees[0].num_nodes();
        let n1 = trees[1].num_nodes();
        let nl = n;

        let alpha: Vec<f64> = (0..=nl).map(|l| if l == 0 { 0.0 } else { 1.0 }).collect();
        let beta: Vec<Vec<f64>> = trees.iter().map(|t| vec![0.0; t.num_nodes()]).collect();
        let gamma = (n as f64) - 1.0;

        let mut builder = ColumnBuilder::new(trees);
        let columns: Vec<AfColumn> = (1..=n as u32)
            .filter_map(|l| builder.try_build(vec![l], trees))
            .collect();
        let seen = ColumnSet::new();
        let branchings = Branchings::default();
        let ctx = PricingContext {
            trees,
            num_leaves: nl,
            alpha: &alpha,
            beta: &beta,
            columns: &columns,
            seen: &seen,
            branchings: &branchings,
            terminate: &crate::bp::pricer::NEVER_TERMINATE,
            deadline: None,
        };

        let mut existing_cache = ExactPairDpCache::new(n0, n1, nl);
        let existing = collect_corridor_candidates_ref(&ctx, &mut existing_cache, 1, gamma, &[]);
        let mut existing_keys: Vec<(Vec<u32>, f64)> = existing
            .candidates
            .into_iter()
            .map(|c| (c.labels, c.score))
            .collect();
        existing_keys.sort_by(|a, b| a.0.cmp(&b.0));

        let threshold = 1.0 - gamma - 1e-8;
        let mut lazy_cache = LazyKBestCache::new(n0, n1, nl);
        let mut lazy = LazyKBest::new(
            LazyKBestInput {
                t0: &trees[0],
                t1: &trees[1],
                alpha: &alpha,
                beta_t0: &beta[0],
                beta_t1: &beta[1],
                threshold,
            },
            &mut lazy_cache,
        );
        let mut lazy_keys: Vec<(Vec<u32>, f64)> = Vec::new();
        while let Some(col) = lazy.next() {
            lazy_keys.push((col.labels, col.score));
        }
        lazy_keys.sort_by(|a, b| a.0.cmp(&b.0));

        // Superset check: every distinct labelset in `existing_keys`
        // must appear (with matching score) somewhere in `lazy_keys`.
        let mut missing = Vec::new();
        for (labels, score) in &existing_keys {
            let found = lazy_keys
                .iter()
                .any(|(lb, sb)| lb == labels && (sb - score).abs() < 1e-6);
            if !found {
                missing.push((labels.clone(), *score));
            }
        }
        if !missing.is_empty() {
            eprintln!("existing ({}):", existing_keys.len());
            for (l, s) in &existing_keys {
                eprintln!("  score={:.4} labels={:?}", s, l);
            }
            eprintln!("lazy ({}):", lazy_keys.len());
            for (l, s) in &lazy_keys {
                eprintln!("  score={:.4} labels={:?}", s, l);
            }
            eprintln!("missing in lazy:");
            for (l, s) in &missing {
                eprintln!("  score={:.4} labels={:?}", s, l);
            }
            panic!("lazy enumerator did not cover the K=1 reference");
        }
    }

    fn build_balanced_4(a: Label, b: Label, c: Label, d: Label) -> Tree {
        let mut t = Tree::with_capacity(4);
        let na = push_leaf(&mut t, a);
        let nb = push_leaf(&mut t, b);
        let nc = push_leaf(&mut t, c);
        let nd = push_leaf(&mut t, d);
        let n_ab = push_internal(&mut t, na, nb);
        let n_cd = push_internal(&mut t, nc, nd);
        let n_root = push_internal(&mut t, n_ab, n_cd);
        t.root = n_root;
        t.compute_metadata();
        t
    }
    fn push_leaf(t: &mut Tree, label: Label) -> NodeId {
        let id = t.parent.len() as NodeId;
        t.parent.push(NONE);
        t.left.push(NONE);
        t.right.push(NONE);
        t.label.push(label);
        t.label_to_node[label as usize] = id;
        id
    }
    fn push_internal(t: &mut Tree, l: NodeId, r: NodeId) -> NodeId {
        let id = t.parent.len() as NodeId;
        t.parent.push(NONE);
        t.left.push(l);
        t.right.push(r);
        t.label.push(0);
        t.parent[l as usize] = id;
        t.parent[r as usize] = id;
        id
    }
}
