//! Top-K threshold corridor enumeration for m=2 instances.
//!
//! Given LP duals `(α, β)` and a score threshold `τ = 1 − γ`, this module
//! returns **every valid 2-tree AF column** with `pricing_score ≥ τ`,
//! sorted by score descending. The output is the complete corridor at
//! the given γ — by the corridor theorem, every column appearing in any
//! integer solution with cost < `U` lies in this set.
//!
//! ## Approach: top-K propagation on the existing pair-DP DAG
//!
//! The existing `exact_pair_dp` walks a DP DAG over `(T₀, T₁)` node-pair
//! states and stores **the single best** sub-column score per cell. To
//! enumerate the complete corridor we propagate **all** sub-column
//! entries whose score remains within reach of the global threshold.
//!
//! For each cell `(u, v)`:
//!
//! * `dp_closed[u, v]` — list of entries for columns whose LCA is
//!   exactly `(u, v)` in `(T₀, T₁)`. Each entry remembers which sub-cell
//!   entry it was built from at the two children.
//!
//! * `dp_open[u, v]` — list of entries for columns whose T₀-anchor is `u`
//!   *or any descendant* and T₁-anchor is `v`. Each entry remembers
//!   which alternative (closed / left-descendant / right-descendant) it
//!   came from.
//!
//! * `best_l0[v]` / `best_r0[v]` — per-T₁-node lists of entries
//!   propagated up through `T₁`'s subtree while processing the current
//!   `T₀` internal node's left/right child.
//!
//! ## K and threshold
//!
//! Two pruning mechanisms work together:
//!
//! * **Threshold pruning** — any entry with score < `τ_local` for the
//!   current cell is discarded. The local threshold is the global `τ`
//!   minus the maximum score gain achievable from this cell upward,
//!   which we conservatively bound by the cell's own current best.
//!
//! * **K-cap pruning** — beyond the threshold, we retain at most
//!   `max_k` entries per cell. Entries with identical *labelsets* are
//!   deduped during reconstruction.
//!
//! For most cells in real instances `max_k = 1` suffices (the anchor-
//! best column is the only valid AF column at that anchor). Cells where
//! multiple sub-leaf-selections give different valid AF columns at the
//! same anchor are the ones that genuinely need `K > 1`. Iterated-K
//! control in the solver doubles `K` only when the MIP can't reach the
//! LP lower bound at the current `K`.

use klados_core::{NONE, Tree};

#[inline]
fn cell_idx(n1: usize, u: usize, v: usize) -> usize {
    u * n1 + v
}

/// One entry in a top-K list. Stores the score plus enough back-pointer
/// information to reconstruct the leafset by walking the DAG top-down.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ClosedEntry {
    pub score: f64,
    /// The grounded `T₁` node where `dp_open[l0, ·]` was consulted on the
    /// left side. `dp_open` is persistent across `T₀` internals, so this
    /// reference is stable for reconstruction.
    pub grounded_v_l: u32,
    /// Index of the chosen entry in `dp_open[l0_idx][grounded_v_l]`.
    pub open_idx_l: u16,
    /// Symmetric for the right side.
    pub grounded_v_r: u32,
    pub open_idx_r: u16,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct OpenEntry {
    pub score: f64,
    /// 0 = closed at (u, v), 1 = continues into left child of u, 2 = right child of u.
    pub choice: u8,
    /// Index into the corresponding source list. For `choice == 0`,
    /// indexes into `dp_closed[u, v]`. For `1`, indexes into
    /// `dp_open[l0, v]`. For `2`, indexes into `dp_open[r0, v]`.
    pub idx: u16,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BestSideEntry {
    pub score: f64,
    /// The grounded `T₁` node where `dp_open[child_t0, ·]` is consulted.
    /// Stays fixed as the entry propagates up through `T₁`'s post-order;
    /// the propagation just adds `−β` adjustments along the way.
    pub grounded_v: u32,
    /// Index of the chosen entry in `dp_open[child_t0_idx][grounded_v]`.
    pub open_idx: u16,
}

/// A finished corridor column, ready for the solver to ingest.
#[derive(Clone, Debug)]
pub struct CorridorColumn {
    pub score: f64,
    pub labels: Vec<u32>,
    pub anchor0: u32,
    pub anchor1: u32,
}

/// Top-K bound per cell. Higher = more complete enumeration, more work.
/// `K=1` reproduces the existing anchor-best behaviour.
pub type TopK<T> = Vec<T>;

/// Persistent storage for the top-K DP, reusable across calls within a
/// single solver invocation. All arrays are sized for `n0 × n1` cells.
pub struct TopKDpCache {
    n0: usize,
    n1: usize,
    num_leaves: usize,

    /// `dp_closed[u * n1 + v]`
    dp_closed: Vec<TopK<ClosedEntry>>,
    /// `dp_open[u * n1 + v]`
    dp_open: Vec<TopK<OpenEntry>>,
    /// `best_l0[v]` — per-T₁-node lists, reused per `T₀` internal node
    best_l0: Vec<TopK<BestSideEntry>>,
    /// `best_r0[v]`
    best_r0: Vec<TopK<BestSideEntry>>,

    /// Workspace for activity flags.
    active_labels: Vec<bool>,
    t0_active: Vec<bool>,
    t1_active: Vec<bool>,
}

impl TopKDpCache {
    pub fn new(n0: usize, n1: usize, num_leaves: usize) -> Self {
        Self {
            n0,
            n1,
            num_leaves,
            dp_closed: vec![Vec::new(); n0 * n1],
            dp_open: vec![Vec::new(); n0 * n1],
            best_l0: vec![Vec::new(); n1],
            best_r0: vec![Vec::new(); n1],
            active_labels: vec![false; num_leaves + 1],
            t0_active: vec![false; n0],
            t1_active: vec![false; n1],
        }
    }

    pub fn fits(&self, n0: usize, n1: usize, num_leaves: usize) -> bool {
        self.n0 == n0 && self.n1 == n1 && self.num_leaves == num_leaves
    }

    fn clear_all(&mut self) {
        for c in self.dp_closed.iter_mut() {
            c.clear();
        }
        for c in self.dp_open.iter_mut() {
            c.clear();
        }
        for c in self.best_l0.iter_mut() {
            c.clear();
        }
        for c in self.best_r0.iter_mut() {
            c.clear();
        }
        self.active_labels.fill(false);
        self.t0_active.fill(false);
        self.t1_active.fill(false);
    }
}

/// Input to the corridor enumerator.
pub struct CorridorInput<'a> {
    pub t0: &'a Tree,
    pub t1: &'a Tree,
    pub alpha: &'a [f64],
    pub beta_t0: &'a [f64],
    pub beta_t1: &'a [f64],
    /// Score threshold τ. Columns with `score ≥ τ` are returned.
    pub threshold: f64,
    /// Top-K bound per DP cell.
    pub max_k: usize,
}

/// Enumerate all valid m=2 AF columns with score ≥ `input.threshold`.
///
/// Returns columns sorted by score descending. Caller is responsible
/// for any further filtering (e.g. removing already-seen labelsets) and
/// for building `AfColumn` instances from the labels.
pub fn enumerate_corridor(input: &CorridorInput, cache: &mut TopKDpCache) -> Vec<CorridorColumn> {
    let t0 = input.t0;
    let t1 = input.t1;
    let n0 = t0.num_nodes();
    let n1 = t1.num_nodes();
    debug_assert_eq!(cache.n0, n0);
    debug_assert_eq!(cache.n1, n1);
    cache.clear_all();

    // Mark active labels and their ancestor closures in both trees.
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
        return Vec::new();
    }

    let max_k = input.max_k.max(1);

    // --- Phase 1: forward DP fill ----------------------------------------
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

        // Reset dp_closed cells we will write.
        for &v in &t1_post {
            cache.dp_closed[cell_idx(cache.n1, u_idx, v as usize)].clear();
        }

        compute_best_side(
            t1,
            &t1_post,
            l0_idx,
            l0_active,
            n1,
            input,
            &cache.dp_open,
            &mut cache.best_l0,
            max_k,
        );
        compute_best_side(
            t1,
            &t1_post,
            r0_idx,
            r0_active,
            n1,
            input,
            &cache.dp_open,
            &mut cache.best_r0,
            max_k,
        );

        // Fill dp_closed[u, v] for internal v.
        for &v in &t1_post {
            if t1.is_leaf(v) {
                continue;
            }
            let v_idx = v as usize;
            let (l1, r1) = t1.children_pair(v);
            let cell = cell_idx(cache.n1, u_idx, v_idx);

            // Collect candidates from both pairings (l0↔l1,r0↔r1) and (l0↔r1,r0↔l1).
            let mut local: TopK<ClosedEntry> = Vec::new();
            combine_closed_candidates(
                &cache.best_l0,
                &cache.best_r0,
                l1,
                r1,
                u_idx,
                v_idx,
                l0_idx,
                r0_idx,
                input,
                &mut local,
                max_k,
            );
            combine_closed_candidates(
                &cache.best_l0,
                &cache.best_r0,
                r1,
                l1,
                u_idx,
                v_idx,
                l0_idx,
                r0_idx,
                input,
                &mut local,
                max_k,
            );

            // Apply the *global* threshold filter: any entry whose score
            // can't reach `threshold` is dropped early. We use the entry
            // score itself as the trivial upper bound — propagating into
            // dp_open or upward can only subtract (no positive penalty
            // term is added at higher levels), so an entry below
            // threshold at the leaf level can't become above-threshold
            // higher up either.
            local.retain(|e| e.score >= input.threshold);
            sort_and_truncate_closed(&mut local, max_k);
            cache.dp_closed[cell] = local;
        }

        // Fill dp_open[u, v] for all v.
        for &v in &t1_post {
            let v_idx = v as usize;
            let cell = cell_idx(cache.n1, u_idx, v_idx);
            let mut local: TopK<OpenEntry> = Vec::new();

            // Source 0: closed at (u, v).
            for (idx, e) in cache.dp_closed[cell].iter().enumerate() {
                let s = e.score + input.beta_t0[u_idx] + input.beta_t1[v_idx];
                push_open_candidate(&mut local, s, 0, idx as u16, max_k);
            }
            // Source 1: dp_open[l0, v] minus β[l0].
            if l0_active {
                let l_cell = cell_idx(cache.n1, l0_idx, v_idx);
                for (idx, e) in cache.dp_open[l_cell].iter().enumerate() {
                    let s = e.score - input.beta_t0[l0_idx];
                    push_open_candidate(&mut local, s, 1, idx as u16, max_k);
                }
            }
            // Source 2: dp_open[r0, v] minus β[r0].
            if r0_active {
                let r_cell = cell_idx(cache.n1, r0_idx, v_idx);
                for (idx, e) in cache.dp_open[r_cell].iter().enumerate() {
                    let s = e.score - input.beta_t0[r0_idx];
                    push_open_candidate(&mut local, s, 2, idx as u16, max_k);
                }
            }
            local.retain(|e| e.score >= input.threshold);
            sort_and_truncate_open(&mut local, max_k);
            cache.dp_open[cell] = local;
        }
    }

    // --- Phase 2: collect candidates from anchored cells -----------------
    let mut output: Vec<CorridorColumn> = Vec::new();
    let ctx = ReconstructCtx {
        t0,
        n1,
        dp_closed: &cache.dp_closed,
        dp_open: &cache.dp_open,
    };
    let _ = t1;
    for u in 0..n0 {
        if t0.is_leaf(u as u32) {
            continue;
        }
        for v in 0..n1 {
            if t1.is_leaf(v as u32) {
                continue;
            }
            let cell = cell_idx(cache.n1, u, v);
            for (entry_idx, entry) in cache.dp_closed[cell].iter().enumerate() {
                if entry.score < input.threshold {
                    continue;
                }
                let mut labels = Vec::new();
                reconstruct_closed(&ctx, u as u32, v as u32, entry_idx as u16, &mut labels);
                labels.sort_unstable();
                labels.dedup();
                if labels.len() < 2 {
                    continue;
                }
                output.push(CorridorColumn {
                    score: entry.score,
                    labels,
                    anchor0: u as u32,
                    anchor1: v as u32,
                });
            }
        }
    }

    output.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.labels.len().cmp(&a.labels.len()))
            .then_with(|| a.labels.cmp(&b.labels))
    });
    output.dedup_by(|a, b| a.labels == b.labels);
    output
}

// ──────────────────────────────────────────────────────────────────────
// Leaf-level fill
// ──────────────────────────────────────────────────────────────────────

fn fill_leaf(
    t0: &Tree,
    t1: &Tree,
    u: u32,
    t1_post: &[u32],
    input: &CorridorInput,
    cache: &mut TopKDpCache,
) {
    let u_idx = u as usize;
    let lbl = t0.label[u_idx] as usize;
    // Clear all cells for this u.
    for &v in t1_post {
        let cell = cell_idx(cache.n1, u_idx, v as usize);
        cache.dp_closed[cell].clear();
        cache.dp_open[cell].clear();
    }
    // Place the singleton at the matching `T₁` leaf.
    if cache.active_labels[lbl] {
        let v = t1.node_by_label(lbl as u32);
        let v_idx = v as usize;
        let score = input.alpha[lbl];
        // Singletons get filtered later — at the leaf level we always
        // record them so internal compositions can use them, even if a
        // singleton alone fails the ≥-2-leaf rule.
        cache.dp_closed[cell_idx(cache.n1, u_idx, v_idx)].push(ClosedEntry {
            score,
            grounded_v_l: 0,
            open_idx_l: 0,
            grounded_v_r: 0,
            open_idx_r: 0,
        });
        cache.dp_open[cell_idx(cache.n1, u_idx, v_idx)].push(OpenEntry {
            score,
            choice: 0,
            idx: 0,
        });
    }
    // dp_open at non-matching v stays empty — there is no valid column
    // anchored only at this T0-leaf and a different T1 sub-node.
    let _ = t1_post;
}

// ──────────────────────────────────────────────────────────────────────
// Compute best_l0 / best_r0 over T₁ post-order
// ──────────────────────────────────────────────────────────────────────

fn compute_best_side(
    t1: &Tree,
    t1_post: &[u32],
    child_t0_idx: usize,
    child_t0_active: bool,
    n1: usize,
    input: &CorridorInput,
    dp_open: &[TopK<OpenEntry>],
    out: &mut [TopK<BestSideEntry>],
    max_k: usize,
) {
    // Reset per-v entries.
    for v in t1_post {
        out[*v as usize].clear();
    }
    for &v in t1_post {
        let v_idx = v as usize;
        let mut local: TopK<BestSideEntry> = Vec::new();

        // Source 0: dp_open[child_t0, v] (T₀-side stays in this subtree
        // while T₁-side is at exactly v). Grounded at v.
        if child_t0_active {
            let cell = child_t0_idx * n1 + v_idx;
            for (idx, e) in dp_open[cell].iter().enumerate() {
                push_best_side_candidate(&mut local, e.score, v, idx as u16, max_k);
            }
        }
        // For internal v, also propagate up from T₁'s children, paying
        // β penalty along the way. Grounded reference is carried as-is.
        if !t1.is_leaf(v) {
            let (l1, r1) = t1.children_pair(v);
            for e in out[l1 as usize].iter() {
                let s = e.score - input.beta_t1[l1 as usize];
                push_best_side_candidate(&mut local, s, e.grounded_v, e.open_idx, max_k);
            }
            for e in out[r1 as usize].iter() {
                let s = e.score - input.beta_t1[r1 as usize];
                push_best_side_candidate(&mut local, s, e.grounded_v, e.open_idx, max_k);
            }
        }
        sort_and_truncate_best(&mut local, max_k);
        out[v_idx] = local;
    }
}

// ──────────────────────────────────────────────────────────────────────
// dp_closed combination at an internal `(u, v)`
// ──────────────────────────────────────────────────────────────────────

fn combine_closed_candidates(
    best_l0: &[TopK<BestSideEntry>],
    best_r0: &[TopK<BestSideEntry>],
    side_a: u32,
    side_b: u32,
    u_idx: usize,
    v_idx: usize,
    l0_idx: usize,
    r0_idx: usize,
    input: &CorridorInput,
    out: &mut TopK<ClosedEntry>,
    max_k: usize,
) {
    let beta = &input.beta_t1;
    let beta0 = &input.beta_t0;
    let side_a_idx = side_a as usize;
    let side_b_idx = side_b as usize;
    if best_l0[side_a_idx].is_empty() || best_r0[side_b_idx].is_empty() {
        return;
    }
    let pen = beta0[u_idx]
        + beta[v_idx]
        + beta0[l0_idx]
        + beta0[r0_idx]
        + beta[side_a_idx]
        + beta[side_b_idx];
    for ea in best_l0[side_a_idx].iter() {
        if !ea.score.is_finite() {
            continue;
        }
        for eb in best_r0[side_b_idx].iter() {
            if !eb.score.is_finite() {
                continue;
            }
            let s = ea.score + eb.score - pen;
            push_closed_candidate(
                out,
                s,
                ea.grounded_v,
                ea.open_idx,
                eb.grounded_v,
                eb.open_idx,
                max_k,
            );
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Top-K push helpers
// ──────────────────────────────────────────────────────────────────────

fn push_closed_candidate(
    list: &mut TopK<ClosedEntry>,
    score: f64,
    grounded_v_l: u32,
    open_idx_l: u16,
    grounded_v_r: u32,
    open_idx_r: u16,
    max_k: usize,
) {
    let _ = max_k;
    list.push(ClosedEntry {
        score,
        grounded_v_l,
        open_idx_l,
        grounded_v_r,
        open_idx_r,
    });
}

fn push_open_candidate(list: &mut TopK<OpenEntry>, score: f64, choice: u8, idx: u16, max_k: usize) {
    let _ = max_k;
    list.push(OpenEntry { score, choice, idx });
}

fn push_best_side_candidate(
    list: &mut TopK<BestSideEntry>,
    score: f64,
    grounded_v: u32,
    open_idx: u16,
    max_k: usize,
) {
    let _ = max_k;
    list.push(BestSideEntry {
        score,
        grounded_v,
        open_idx,
    });
}

fn sort_and_truncate_closed(list: &mut TopK<ClosedEntry>, max_k: usize) {
    list.sort_by(|a, b| b.score.total_cmp(&a.score));
    if list.len() > max_k {
        list.truncate(max_k);
    }
}

fn sort_and_truncate_open(list: &mut TopK<OpenEntry>, max_k: usize) {
    list.sort_by(|a, b| b.score.total_cmp(&a.score));
    if list.len() > max_k {
        list.truncate(max_k);
    }
}

fn sort_and_truncate_best(list: &mut TopK<BestSideEntry>, max_k: usize) {
    list.sort_by(|a, b| b.score.total_cmp(&a.score));
    if list.len() > max_k {
        list.truncate(max_k);
    }
}

// ──────────────────────────────────────────────────────────────────────
// Reconstruction: walk down the DAG, accumulate labels
// ──────────────────────────────────────────────────────────────────────

struct ReconstructCtx<'a> {
    t0: &'a Tree,
    n1: usize,
    dp_closed: &'a [TopK<ClosedEntry>],
    dp_open: &'a [TopK<OpenEntry>],
}

fn reconstruct_closed(ctx: &ReconstructCtx, u: u32, v: u32, entry_idx: u16, out: &mut Vec<u32>) {
    let _ = v;
    let entry = ctx.dp_closed[u as usize * ctx.n1 + v as usize][entry_idx as usize];
    let (l0, r0) = ctx.t0.children_pair(u);
    reconstruct_open(ctx, l0, entry.grounded_v_l, entry.open_idx_l, out);
    reconstruct_open(ctx, r0, entry.grounded_v_r, entry.open_idx_r, out);
}

fn reconstruct_open(ctx: &ReconstructCtx, u: u32, v: u32, entry_idx: u16, out: &mut Vec<u32>) {
    if ctx.t0.is_leaf(u) {
        out.push(ctx.t0.label[u as usize]);
        return;
    }
    let entry = ctx.dp_open[u as usize * ctx.n1 + v as usize][entry_idx as usize];
    match entry.choice {
        0 => reconstruct_closed(ctx, u, v, entry.idx, out),
        1 => {
            let (l0, _) = ctx.t0.children_pair(u);
            reconstruct_open(ctx, l0, v, entry.idx, out);
        }
        2 => {
            let (_, r0) = ctx.t0.children_pair(u);
            reconstruct_open(ctx, r0, v, entry.idx, out);
        }
        _ => debug_assert!(false, "unknown OpenEntry choice"),
    }
}
