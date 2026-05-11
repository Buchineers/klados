//! Tier-2 pricer — exact pair-DP for **m = 2** (Steel-Warnow style).
//!
//! Two DP tables in shared scratch (storage reused across calls):
//!
//! - `closed(u, v)` — max α-β score over leafsets `L` with `LCA_{T₀}(L) = u`
//!   **and** `LCA_{T₁}(L) = v` (anchors exact in both trees).
//! - `open(u, v)` — max α-β score over leafsets `L ⊆ desc(u) ∩ desc(v)`,
//!   anchors free; the score is computed as if `L`'s covered range extends
//!   up to `u` in T₀ and `v` in T₁. The "extends up" framing makes `open`
//!   additive across siblings — `closed`'s recurrence sums children's
//!   `open` values and adds its own β contributions exactly once.
//!
//! ## Recurrences
//!
//! - `open(leaf u, leaf v)`: `α_ℓ` if same label, else infeasible.
//! - `open(leaf u, internal v)`: `max(open(u, v_L), open(u, v_R)) − β_{T₁}(v)`.
//! - `open(internal u, leaf v)`: `max(open(u_L, v), open(u_R, v)) − β_{T₀}(u)`.
//! - `open(internal u, internal v)`: max of `closed(u, v)` plus the four
//!   "confine to one child" cases — each subtracts `β_{T₀}(u)` or `β_{T₁}(v)`.
//! - `closed(internal u, internal v)`:
//!   `max{open(u_L, v_L) + open(u_R, v_R), open(u_L, v_R) + open(u_R, v_L)}
//!     − β_{T₀}(u) − β_{T₁}(v)`.
//!
//! ## Validity by construction & soundness
//!
//! Every state corresponds to a leafset whose topology is consistent in both
//! trees — invalid columns can't appear in the DP. After filling the table
//! we report `Converged` iff `max_{u,v} closed(u, v) ≤ 1` (and singletons
//! are bounded too). When branchings forbid the best column at every state,
//! we conservatively return `Exhausted` instead of `Converged`.

use klados_core::Tree;

use super::{
    PairDpTable, Pricer, PricerScratch, PricingContext, PricingResult, scratch::PairDpCell,
};

const PRICING_EPS: f64 = 1.0e-8;
const NEG_INF: f64 = f64::NEG_INFINITY;
const LOG_TARGET: &str = "klados::bp::pair_dp";

mod choice {
    pub const INFEASIBLE: u8 = 0;
    pub const PAIR_LL_RR: u8 = 2;
    pub const PAIR_LR_RL: u8 = 3;
    pub const LEAF: u8 = 10;
    pub const VIA_CLOSED: u8 = 11;
    pub const SKIP_UR: u8 = 12;
    pub const SKIP_UL: u8 = 13;
    pub const SKIP_VR: u8 = 14;
    pub const SKIP_VL: u8 = 15;
}

pub struct PairDpPricer {
    max_per_call: usize,
}

impl PairDpPricer {
    pub fn new(trees: &[Tree]) -> Self {
        assert_eq!(trees.len(), 2, "PairDpPricer requires m=2");
        Self { max_per_call: 64 }
    }
}

impl Pricer for PairDpPricer {
    fn name(&self) -> &'static str {
        "pair-dp"
    }

    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult {
        debug_assert_eq!(ctx.trees.len(), 2);
        run_pair_dp(
            ctx,
            scratch,
            /*allow_filter=*/ false,
            self.max_per_call,
        )
    }
}

/// Run the pair-DP on `(ctx.trees[0], ctx.trees[1])`. When `allow_filter` is
/// true, additionally filter every emitted column by AF-validity in
/// `ctx.trees[2..]` — used by [`super::PairDpFilterPricer`] to recycle this
/// machinery for m≥3 as a heuristic.
pub(super) fn run_pair_dp(
    ctx: &PricingContext,
    scratch: &mut PricerScratch,
    allow_filter: bool,
    max_per_call: usize,
) -> PricingResult {
    let t0 = &ctx.trees[0];
    let t1 = &ctx.trees[1];
    let n0 = t0.num_nodes();
    let n1 = t1.num_nodes();

    // Acquire / size the cached DP table.
    let mut table = scratch
        .pair_dp_table
        .take()
        .unwrap_or_else(|| PairDpTable::new(n0, n1));
    if !table.fits(n0, n1) {
        table = PairDpTable::new(n0, n1);
    }
    table.reset();

    let post0 = t0.post_order_vec();
    let post1 = t1.post_order_vec();

    for &u in &post0 {
        for &v in &post1 {
            fill_cell(&mut table, u as usize, v as usize, t0, t1, ctx);
        }
    }

    // Collect closed-state candidates with positive RC.
    let mut candidates: Vec<(usize, usize, f64)> = Vec::new();
    let mut max_closed = NEG_INF;
    for u in 0..n0 {
        for v in 0..n1 {
            let s = table.cells[table.idx(u, v)].closed_score;
            if s > max_closed {
                max_closed = s;
            }
            if s > 1.0 + PRICING_EPS {
                candidates.push((u, v, s));
            }
        }
    }
    let max_singleton = ctx.alpha.iter().copied().fold(NEG_INF, f64::max);
    let any_positive = max_closed > 1.0 + PRICING_EPS || max_singleton > 1.0 + PRICING_EPS;

    log::trace!(
        target: LOG_TARGET,
        "DP m={}: max_closed={:.4} max_singleton={:.4} candidates={}",
        ctx.trees.len(), max_closed, max_singleton, candidates.len(),
    );

    if !any_positive {
        scratch.pair_dp_table = Some(table);
        return PricingResult::Converged;
    }

    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut found = Vec::new();
    let mut tried_any = false;
    for (u, v, _) in candidates.into_iter() {
        let labels = backtrack(&table, u, v, t0, t1);
        if labels.is_empty() {
            continue;
        }
        if ctx.seen.contains(&labels) {
            continue;
        }
        tried_any = true;

        // For m=2 the DP guarantees validity. For m≥3 (filter mode), validate
        // against the other trees explicitly.
        let column = if allow_filter && ctx.trees.len() > 2 {
            // Validate against trees 2..m via the public path.
            match scratch.builder.try_build(labels.clone(), ctx.trees) {
                Some(c) => c,
                None => continue,
            }
        } else {
            scratch.builder.build_unchecked(labels, ctx.trees)
        };

        if ctx.branchings.forbids(&column) {
            continue;
        }
        found.push(column);
        if found.len() >= max_per_call {
            break;
        }
    }

    scratch.pair_dp_table = Some(table);

    if !found.is_empty() {
        PricingResult::Found(found)
    } else if tried_any {
        // Positive-RC states existed but every emitted column was filtered
        // (forbidden, already seen, or invalid for trees 2..m).
        PricingResult::Exhausted
    } else {
        // No state had positive RC after filtering branchings/seen — at this
        // node, no new positive-RC column exists in the unconstrained space.
        PricingResult::Converged
    }
}

fn fill_cell(
    table: &mut PairDpTable,
    u: usize,
    v: usize,
    t0: &Tree,
    t1: &Tree,
    ctx: &PricingContext,
) {
    let u_leaf = t0.is_leaf(u as u32);
    let v_leaf = t1.is_leaf(v as u32);

    if u_leaf && v_leaf {
        if t0.label[u] == t1.label[v] && t0.label[u] != 0 {
            let lbl = t0.label[u] as usize;
            let i = table.idx(u, v);
            table.cells[i] = PairDpCell {
                open_score: ctx.alpha[lbl],
                closed_score: NEG_INF,
                open_choice: choice::LEAF,
                closed_choice: choice::INFEASIBLE,
            };
        }
        return;
    }

    if u_leaf && !v_leaf {
        let (vl, vr) = t1.children_pair(v as u32);
        let bv = ctx.beta[1][v];
        let s_l = table.cells[table.idx(u, vl as usize)].open_score;
        let s_r = table.cells[table.idx(u, vr as usize)].open_score;
        if s_l == NEG_INF && s_r == NEG_INF {
            return;
        }
        let (best, ch) = if s_l >= s_r {
            (s_l - bv, choice::SKIP_VR)
        } else {
            (s_r - bv, choice::SKIP_VL)
        };
        let i = table.idx(u, v);
        table.cells[i].open_score = best;
        table.cells[i].open_choice = ch;
        return;
    }

    if !u_leaf && v_leaf {
        let (ul, ur) = t0.children_pair(u as u32);
        let bu = ctx.beta[0][u];
        let s_l = table.cells[table.idx(ul as usize, v)].open_score;
        let s_r = table.cells[table.idx(ur as usize, v)].open_score;
        if s_l == NEG_INF && s_r == NEG_INF {
            return;
        }
        let (best, ch) = if s_l >= s_r {
            (s_l - bu, choice::SKIP_UR)
        } else {
            (s_r - bu, choice::SKIP_UL)
        };
        let i = table.idx(u, v);
        table.cells[i].open_score = best;
        table.cells[i].open_choice = ch;
        return;
    }

    let (ul, ur) = t0.children_pair(u as u32);
    let (vl, vr) = t1.children_pair(v as u32);
    let bu = ctx.beta[0][u];
    let bv = ctx.beta[1][v];

    let s_ll_rr = combine(
        table.cells[table.idx(ul as usize, vl as usize)].open_score,
        table.cells[table.idx(ur as usize, vr as usize)].open_score,
    );
    let s_lr_rl = combine(
        table.cells[table.idx(ul as usize, vr as usize)].open_score,
        table.cells[table.idx(ur as usize, vl as usize)].open_score,
    );
    let (closed_score, closed_ch) = if s_ll_rr == NEG_INF && s_lr_rl == NEG_INF {
        (NEG_INF, choice::INFEASIBLE)
    } else if s_ll_rr >= s_lr_rl {
        (s_ll_rr - bu - bv, choice::PAIR_LL_RR)
    } else {
        (s_lr_rl - bu - bv, choice::PAIR_LR_RL)
    };
    {
        let i = table.idx(u, v);
        table.cells[i].closed_score = closed_score;
        table.cells[i].closed_choice = closed_ch;
    }

    // open: max over closed at (u,v) and four "confine to one side" cases.
    let mut best = closed_score;
    let mut best_ch = if closed_score > NEG_INF {
        choice::VIA_CLOSED
    } else {
        choice::INFEASIBLE
    };
    let try_update = |best: &mut f64, best_ch: &mut u8, cand: f64, ch: u8| {
        if cand > *best {
            *best = cand;
            *best_ch = ch;
        }
    };
    // SKIP_UR/UL: confine to ul or ur subtree; subtract beta[u] (the parent)
    // because we're "stepping over" the current node u.
    let s_skip_ur = table.cells[table.idx(ul as usize, v)].open_score;
    if s_skip_ur != NEG_INF {
        try_update(&mut best, &mut best_ch, s_skip_ur - bu, choice::SKIP_UR);
    }
    let s_skip_ul = table.cells[table.idx(ur as usize, v)].open_score;
    if s_skip_ul != NEG_INF {
        try_update(&mut best, &mut best_ch, s_skip_ul - bu, choice::SKIP_UL);
    }
    // SKIP_VR/VL: confine to vl or vr subtree; subtract beta[v] (the parent).
    let s_skip_vr = table.cells[table.idx(u, vl as usize)].open_score;
    if s_skip_vr != NEG_INF {
        try_update(&mut best, &mut best_ch, s_skip_vr - bv, choice::SKIP_VR);
    }
    let s_skip_vl = table.cells[table.idx(u, vr as usize)].open_score;
    if s_skip_vl != NEG_INF {
        try_update(&mut best, &mut best_ch, s_skip_vl - bv, choice::SKIP_VL);
    }
    let i = table.idx(u, v);
    table.cells[i].open_score = best;
    table.cells[i].open_choice = best_ch;
}

fn combine(a: f64, b: f64) -> f64 {
    if a == NEG_INF || b == NEG_INF {
        NEG_INF
    } else {
        a + b
    }
}

fn backtrack(table: &PairDpTable, u: usize, v: usize, t0: &Tree, t1: &Tree) -> Vec<u32> {
    enum Frame {
        Closed(usize, usize),
        Open(usize, usize),
    }
    let mut labels = Vec::new();
    let mut stack = vec![Frame::Closed(u, v)];

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Closed(u, v) => {
                let cell = table.cells[table.idx(u, v)];
                let (ul, ur) = t0.children_pair(u as u32);
                let (vl, vr) = t1.children_pair(v as u32);
                match cell.closed_choice {
                    choice::PAIR_LL_RR => {
                        stack.push(Frame::Open(ul as usize, vl as usize));
                        stack.push(Frame::Open(ur as usize, vr as usize));
                    }
                    choice::PAIR_LR_RL => {
                        stack.push(Frame::Open(ul as usize, vr as usize));
                        stack.push(Frame::Open(ur as usize, vl as usize));
                    }
                    _ => debug_assert!(false, "backtrack closed with infeasible cell"),
                }
            }
            Frame::Open(u, v) => {
                let cell = table.cells[table.idx(u, v)];
                match cell.open_choice {
                    choice::LEAF => labels.push(t0.label[u]),
                    choice::VIA_CLOSED => stack.push(Frame::Closed(u, v)),
                    choice::SKIP_UR => {
                        let (ul, _) = t0.children_pair(u as u32);
                        stack.push(Frame::Open(ul as usize, v));
                    }
                    choice::SKIP_UL => {
                        let (_, ur) = t0.children_pair(u as u32);
                        stack.push(Frame::Open(ur as usize, v));
                    }
                    choice::SKIP_VR => {
                        let (vl, _) = t1.children_pair(v as u32);
                        stack.push(Frame::Open(u, vl as usize));
                    }
                    choice::SKIP_VL => {
                        let (_, vr) = t1.children_pair(v as u32);
                        stack.push(Frame::Open(u, vr as usize));
                    }
                    _ => debug_assert!(false, "backtrack open with infeasible cell"),
                }
            }
        }
    }
    labels.sort_unstable();
    labels.dedup();
    labels
}
