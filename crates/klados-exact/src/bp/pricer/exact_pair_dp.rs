//! Exact bottom-up DP pricer for two trees.
//!
//! This is the faithful module-split port of `ExactPricer2Tree` from
//! `maf_branch_price_multi.rs`.  It is intentionally kept separate from the
//! newer Steel-Warnow-style `pair_dp` implementation: the latter is useful as
//! a heuristic under branching, but its recurrence is not equivalent to the
//! old exact root pricer and can miss columns needed by the pool MIP.

use klados_core::{NONE, Tree};

use super::{Pricer, PricerScratch, PricingContext, PricingResult};

const PRICING_EPS: f64 = 1.0e-8;
const NEG_INF: f64 = f64::NEG_INFINITY;

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
        price_exact_pair_dp(ctx, scratch, self.max_per_call)
    }
}

fn price_exact_pair_dp(
    ctx: &PricingContext,
    scratch: &mut PricerScratch,
    max_per_call: usize,
) -> PricingResult {
    let candidates = collect_positive_columns(ctx);
    if candidates.is_empty() {
        return PricingResult::Converged;
    }

    let mut found = Vec::new();
    let mut blocked_positive = false;
    for (_, labels) in candidates {
        let column = scratch.builder.build_unchecked(labels, ctx.trees);
        if ctx.branchings.forbids(&column) {
            // Even if this labelset is already in the global pool, it is
            // bound to zero in the current branch.  A positive best state
            // hidden behind a forbidden seen column is not a convergence
            // proof: there may be a second-best allowed column for the same
            // state that the one-best DP did not emit.
            blocked_positive = true;
            continue;
        }
        if ctx.seen.contains(column.labels()) {
            continue;
        }
        found.push(column);
        if found.len() >= max_per_call {
            break;
        }
    }

    if !found.is_empty() {
        PricingResult::Found(found)
    } else if blocked_positive {
        // There were positive columns in the unconstrained two-tree space,
        // but every newly emitted one was forbidden by branch constraints.
        // The one-best-per-state DP is not a proof that no alternative
        // positive allowed column exists for those states.
        PricingResult::Exhausted
    } else {
        // All positive candidates were already present in the master.
        PricingResult::Converged
    }
}

fn collect_positive_columns(ctx: &PricingContext) -> Vec<(f64, Vec<u32>)> {
    let t0 = &ctx.trees[0];
    let t1 = &ctx.trees[1];
    let n0 = t0.num_nodes();
    let n1 = t1.num_nodes();

    let mut active_labels = vec![false; ctx.num_leaves + 1];
    let mut t0_active = vec![false; n0];
    let mut t1_active = vec![false; n1];
    for label in 1..=ctx.num_leaves {
        if ctx.alpha[label] <= 1.0e-12 {
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
        return Vec::new();
    }

    let mut dp_closed = vec![DpClosed::default(); n0 * n1];
    let mut dp_open = vec![DpOpen::default(); n0 * n1];
    let idx = |u: usize, v: usize| -> usize { u * n1 + v };
    let mut best_l0 = vec![(NEG_INF, 0u32); n1];
    let mut best_r0 = vec![(NEG_INF, 0u32); n1];

    for &u in &t0_post {
        let u_idx = u as usize;

        if t0.is_leaf(u) {
            let lbl = t0.label[u_idx] as usize;
            for &v in &t1_post {
                let i = idx(u_idx, v as usize);
                dp_closed[i] = DpClosed::default();
                dp_open[i] = DpOpen::default();
            }
            if active_labels[lbl] {
                let v = t1.node_by_label(lbl as u32);
                dp_closed[idx(u_idx, v as usize)].score = ctx.alpha[lbl];
            }
            for &v in &t1_post {
                let i = idx(u_idx, v as usize);
                dp_open[i] = DpOpen {
                    score: dp_closed[i].score,
                    choice: 0,
                };
            }
            continue;
        }

        let (l0, r0) = t0.children_pair(u);
        let l0_idx = l0 as usize;
        let r0_idx = r0 as usize;
        let l0_active = t0_active[l0_idx];
        let r0_active = t0_active[r0_idx];

        for &v in &t1_post {
            dp_closed[idx(u_idx, v as usize)] = DpClosed::default();
        }

        for &v in &t1_post {
            let v_idx = v as usize;
            let mut max_s = if l0_active {
                dp_open[idx(l0_idx, v_idx)].score
            } else {
                NEG_INF
            };
            let mut best_v = v;
            if !t1.is_leaf(v) {
                let (l1, r1) = t1.children_pair(v);
                let s_l = best_l0[l1 as usize].0 - ctx.beta[1][l1 as usize];
                if s_l > max_s {
                    max_s = s_l;
                    best_v = best_l0[l1 as usize].1;
                }
                let s_r = best_l0[r1 as usize].0 - ctx.beta[1][r1 as usize];
                if s_r > max_s {
                    max_s = s_r;
                    best_v = best_l0[r1 as usize].1;
                }
            }
            best_l0[v_idx] = (max_s, best_v);
        }

        for &v in &t1_post {
            let v_idx = v as usize;
            let mut max_s = if r0_active {
                dp_open[idx(r0_idx, v_idx)].score
            } else {
                NEG_INF
            };
            let mut best_v = v;
            if !t1.is_leaf(v) {
                let (l1, r1) = t1.children_pair(v);
                let s_l = best_r0[l1 as usize].0 - ctx.beta[1][l1 as usize];
                if s_l > max_s {
                    max_s = s_l;
                    best_v = best_r0[l1 as usize].1;
                }
                let s_r = best_r0[r1 as usize].0 - ctx.beta[1][r1 as usize];
                if s_r > max_s {
                    max_s = s_r;
                    best_v = best_r0[r1 as usize].1;
                }
            }
            best_r0[v_idx] = (max_s, best_v);
        }

        for &v in &t1_post {
            if t1.is_leaf(v) {
                continue;
            }
            let v_idx = v as usize;
            let (l1, r1) = t1.children_pair(v);

            let mut best_c_score = NEG_INF;
            let mut v_l = 0;
            let mut v_r = 0;

            let s_l0_l1 = best_l0[l1 as usize].0 - ctx.beta[1][l1 as usize];
            let s_r0_r1 = best_r0[r1 as usize].0 - ctx.beta[1][r1 as usize];
            if s_l0_l1 > NEG_INF / 2.0 && s_r0_r1 > NEG_INF / 2.0 {
                let s = s_l0_l1 + s_r0_r1
                    - ctx.beta[0][u_idx]
                    - ctx.beta[1][v_idx]
                    - ctx.beta[0][l0_idx]
                    - ctx.beta[0][r0_idx];
                if s > best_c_score {
                    best_c_score = s;
                    v_l = best_l0[l1 as usize].1;
                    v_r = best_r0[r1 as usize].1;
                }
            }

            let s_l0_r1 = best_l0[r1 as usize].0 - ctx.beta[1][r1 as usize];
            let s_r0_l1 = best_r0[l1 as usize].0 - ctx.beta[1][l1 as usize];
            if s_l0_r1 > NEG_INF / 2.0 && s_r0_l1 > NEG_INF / 2.0 {
                let s = s_l0_r1 + s_r0_l1
                    - ctx.beta[0][u_idx]
                    - ctx.beta[1][v_idx]
                    - ctx.beta[0][l0_idx]
                    - ctx.beta[0][r0_idx];
                if s > best_c_score {
                    best_c_score = s;
                    v_l = best_l0[r1 as usize].1;
                    v_r = best_r0[l1 as usize].1;
                }
            }

            if best_c_score > NEG_INF / 2.0 {
                dp_closed[idx(u_idx, v_idx)] = DpClosed {
                    score: best_c_score,
                    v_l,
                    v_r,
                };
            }
        }

        for &v in &t1_post {
            let v_idx = v as usize;
            let mut best_o_score = NEG_INF;
            let mut choice = 0;

            let closed = dp_closed[idx(u_idx, v_idx)].score;
            if closed > NEG_INF / 2.0 {
                best_o_score = closed + ctx.beta[0][u_idx] + ctx.beta[1][v_idx];
            }

            let s_l0 = if l0_active {
                dp_open[idx(l0_idx, v_idx)].score - ctx.beta[0][l0_idx]
            } else {
                NEG_INF
            };
            if s_l0 > best_o_score {
                best_o_score = s_l0;
                choice = 1;
            }

            let s_r0 = if r0_active {
                dp_open[idx(r0_idx, v_idx)].score - ctx.beta[0][r0_idx]
            } else {
                NEG_INF
            };
            if s_r0 > best_o_score {
                best_o_score = s_r0;
                choice = 2;
            }

            dp_open[idx(u_idx, v_idx)] = DpOpen {
                score: best_o_score,
                choice,
            };
        }
    }

    let mut results = Vec::new();
    for u in 0..n0 {
        if t0.is_leaf(u as u32) {
            continue;
        }
        for v in 0..n1 {
            if t1.is_leaf(v as u32) {
                continue;
            }
            let score = dp_closed[idx(u, v)].score;
            if score > 1.0 + PRICING_EPS {
                let mut labels = Vec::new();
                extract_closed(
                    u as u32,
                    v as u32,
                    t0,
                    &dp_closed,
                    &dp_open,
                    n1,
                    &mut labels,
                );
                labels.sort_unstable();
                labels.dedup();
                if labels.len() >= 2 {
                    results.push((score, labels));
                }
            }
        }
    }

    results.sort_unstable_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| b.1.len().cmp(&a.1.len()))
            .then_with(|| a.1.cmp(&b.1))
    });
    results
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
