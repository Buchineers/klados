//! Branch-pair selection. Pluggable so we can experiment with strong
//! branching, pseudo-cost, etc. — the search loop only sees [`BranchSelector`].

use crate::bp::column::AfColumn;
use crate::bp::rmp::Rmp;
use crate::bp::search::Branchings;
use crate::bp::search::branchings::LeafPair;

pub struct SelectionContext<'a> {
    pub columns: &'a [AfColumn],
    pub values: &'a [f64],
    pub num_leaves: usize,
    pub branchings: &'a Branchings,
    pub current_lp_obj: f64,
}

pub trait BranchSelector {
    /// Choose a leaf pair to branch on, or `None` if the LP support is
    /// already an integer partition (no fractional choice exists).
    ///
    /// `rmp` is provided mutable so selectors can perform speculative LP
    /// resolves (strong branching). The selector **must** restore the RMP
    /// to the caller's branching state before returning.
    fn select(&mut self, ctx: &SelectionContext, rmp: &mut Rmp) -> Option<LeafPair>;
}

const ACTIVE_EPS: f64 = 1.0e-9;
const FRACTIONAL_EPS: f64 = 1.0e-6;

/// Compute, for each leaf pair (a,b), `Σ_{c ⊇ {a,b}} x_c` and the number
/// of columns contributing to that sum (support count for tie-breaking).
/// Indexed `together[a*stride + b]` with `stride = num_leaves + 1`.
fn pair_mass_and_support(
    columns: &[AfColumn],
    values: &[f64],
    num_leaves: usize,
) -> (Vec<f64>, Vec<usize>) {
    let stride = num_leaves + 1;
    let mut together = vec![0.0_f64; stride * stride];
    let mut support = vec![0usize; stride * stride];
    for (ci, &x) in values.iter().enumerate() {
        if x <= ACTIVE_EPS {
            continue;
        }
        let labels = columns[ci].labels();
        for i in 0..labels.len() {
            let a = labels[i] as usize;
            let row = a * stride;
            for j in (i + 1)..labels.len() {
                let b = labels[j] as usize;
                let idx = row + b;
                together[idx] += x;
                support[idx] += 1;
                together[b * stride + a] += x;
                support[b * stride + a] += 1;
            }
        }
    }
    (together, support)
}

/// Pairs in the fractional band, sorted by closeness to 0.5, then by
/// support count (more columns → more structurally informative branch).
fn fractional_pairs(
    together: &[f64],
    support: &[usize],
    num_leaves: usize,
) -> Vec<(LeafPair, f64, usize)> {
    let stride = num_leaves + 1;
    let mut pairs: Vec<(LeafPair, f64, usize)> = Vec::new();
    for a in 1..=num_leaves {
        for b in (a + 1)..=num_leaves {
            let idx = a * stride + b;
            let mass = together[idx];
            if mass <= FRACTIONAL_EPS || mass >= 1.0 - FRACTIONAL_EPS {
                continue;
            }
            pairs.push((
                LeafPair::new(a as u32, b as u32),
                (mass - 0.5).abs(),
                support[idx],
            ));
        }
    }
    pairs.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.2.cmp(&a.2))
    });
    pairs
}

/// Default: pick the leaf pair `(a,b)` whose "together mass"
/// `Σ_{c ⊇ {a,b}} x_c` is closest to `0.5`, with support-count
/// tie-breaking (more columns supporting the pair → more informative branch).
pub struct MostFractionalPair;

impl BranchSelector for MostFractionalPair {
    fn select(&mut self, ctx: &SelectionContext, _rmp: &mut Rmp) -> Option<LeafPair> {
        let (together, support) =
            pair_mass_and_support(ctx.columns, ctx.values, ctx.num_leaves);
        let pairs = fractional_pairs(&together, &support, ctx.num_leaves);
        pairs.into_iter().next().map(|(p, _, _)| p)
    }
}

/// Strong branching: take the top-K most-fractional pairs as candidates;
/// for each, **simulate** the must-link and cannot-link child LPs (without
/// pricing), score by `min(LP_must, LP_cannot)` (worst-case child LP), pick
/// the pair maximising this score. The pair that drives both children's LP
/// up the most is the most informative branch.
///
/// Strong branching at every node is expensive (`2K` extra LP solves per
/// branch decision), so we keep K small. The RMP is **always restored** to
/// the caller's branching state on return.
pub struct StrongBranching {
    /// Number of candidate pairs to simulate. K=3..5 typical.
    pub candidates: usize,
    /// If `Some(d)`, only run strong branching at branching depths `< d`;
    /// fall back to most-fractional at deeper nodes. The choice at shallow
    /// nodes shapes the entire subtree below, so the 2K-LP probe cost
    /// amortizes; deep nodes inherit a tight LP from ancestors and
    /// branching choice matters less.
    pub max_depth: Option<usize>,
}

impl StrongBranching {
    pub fn new(candidates: usize) -> Self {
        Self {
            candidates: candidates.max(1),
            max_depth: None,
        }
    }

    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }
}

impl Default for StrongBranching {
    fn default() -> Self {
        Self::new(4)
    }
}

impl BranchSelector for StrongBranching {
    fn select(&mut self, ctx: &SelectionContext, rmp: &mut Rmp) -> Option<LeafPair> {
        let (together, support) =
            pair_mass_and_support(ctx.columns, ctx.values, ctx.num_leaves);
        let candidates = fractional_pairs(&together, &support, ctx.num_leaves);
        if candidates.is_empty() {
            return None;
        }
        if candidates.len() == 1 {
            return Some(candidates[0].0);
        }

        // Past the depth cap, fall back to most-fractional (the first
        // entry, which is already sorted by closeness to 0.5).
        if let Some(d) = self.max_depth {
            if ctx.branchings.depth() >= d {
                return Some(candidates[0].0);
            }
        }

        let pool = std::cmp::min(self.candidates, candidates.len());
        let mut best: Option<(LeafPair, f64)> = None;

        for &(pair, _, _) in candidates.iter().take(pool) {
            let lp_ml = simulate_child(rmp, ctx, pair, true);
            let lp_cl = simulate_child(rmp, ctx, pair, false);
            let lp_min = match (lp_ml, lp_cl) {
                (Some(a), Some(b)) => a.min(b),
                (Some(a), None) | (None, Some(a)) => a,
                (None, None) => f64::NEG_INFINITY,
            };
            if best.map_or(true, |(_, s)| lp_min > s) {
                best = Some((pair, lp_min));
            }
        }

        rmp.apply_bounds(ctx.columns, ctx.branchings);
        best.map(|(p, _)| p)
    }
}

/// Simulate: apply the branching `pair`-must-link (or cannot-link) on top of
/// the parent branchings, solve the LP (no pricing), return objective. Does
/// **not** restore — caller must call `apply_bounds` with the parent's
/// branchings before relying on RMP state.
fn simulate_child(
    rmp: &mut Rmp,
    ctx: &SelectionContext,
    pair: LeafPair,
    must_link: bool,
) -> Option<f64> {
    let mut child = ctx.branchings.clone();
    if must_link {
        child.push_must_link(pair);
    } else {
        child.push_cannot_link(pair);
    }
    if child.is_inconsistent() {
        return None;
    }
    rmp.apply_bounds(ctx.columns, &child);
    rmp.solve().ok().map(|sol| sol.objective)
}
