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
    /// Choose how to branch from the current node. Returns the children's
    /// fully-formed [`Branchings`] (each = parent + new constraints), or
    /// `None` if the LP support is already an integer partition.
    ///
    /// Returning a single-element Vec is unusual but valid (no real
    /// branching, e.g. an enumeration fallback). Returning a 2-element Vec
    /// is the classic Ryan-Foster must/cannot split. Returning a k-element
    /// Vec is k-way branching (e.g. cluster branching on a fractional
    /// triple). The search loop is arity-agnostic and pushes all children.
    ///
    /// `rmp` is provided mutable so selectors can perform speculative LP
    /// resolves (strong branching). The selector **must** restore the RMP
    /// to the caller's branching state before returning.
    fn select(
        &mut self,
        ctx: &SelectionContext,
        rmp: &mut Rmp,
    ) -> Option<Vec<Branchings>>;
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
    fn select(
        &mut self,
        ctx: &SelectionContext,
        _rmp: &mut Rmp,
    ) -> Option<Vec<Branchings>> {
        let (together, support) =
            pair_mass_and_support(ctx.columns, ctx.values, ctx.num_leaves);
        let pairs = fractional_pairs(&together, &support, ctx.num_leaves);
        let pair = pairs.into_iter().next().map(|(p, _, _)| p)?;
        let (left, right) = ctx.branchings.split_on(pair);
        Some(vec![left, right])
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
    fn select(
        &mut self,
        ctx: &SelectionContext,
        rmp: &mut Rmp,
    ) -> Option<Vec<Branchings>> {
        let (together, support) =
            pair_mass_and_support(ctx.columns, ctx.values, ctx.num_leaves);
        let candidates = fractional_pairs(&together, &support, ctx.num_leaves);
        if candidates.is_empty() {
            return None;
        }
        if candidates.len() == 1 {
            let (l, r) = ctx.branchings.split_on(candidates[0].0);
            return Some(vec![l, r]);
        }

        // Past the depth cap, fall back to most-fractional (the first
        // entry, which is already sorted by closeness to 0.5).
        if let Some(d) = self.max_depth {
            if ctx.branchings.depth() >= d {
                let (l, r) = ctx.branchings.split_on(candidates[0].0);
                return Some(vec![l, r]);
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
        best.map(|(p, _)| {
            let (l, r) = ctx.branchings.split_on(p);
            vec![l, r]
        })
    }
}

/// 4-way **cluster branching** on a fractional triple `(a, b, c)`.
///
/// Motivation: classic Ryan-Foster pair branching commits one bit of
/// information per branch. On the hard m≥3 instances in `exact_pub_v2.lst`
/// the LP increment per branch is ≈ 0.2 against an integrality gap of
/// 3–4 units, so proving optimality needs ~18 levels (tree ~2¹⁸ pre-prune).
/// Branching on a triple commits 2–3 pair constraints per child, which
/// (in theory) should raise ΔLP/branch toward 0.5+ and collapse the tree
/// from 18 to 4–8 levels.
///
/// The branching rule for a fractional triple `S = {a, b, c}`:
///
/// 1. **All-together** — all three must be in one AF component. Add
///    must-link(a,b), must-link(a,c), must-link(b,c).
/// 2. **a-isolated** — `a` is in a different component than {b, c}, while
///    leaving the (b,c) relationship open. Add cannot-link(a,b),
///    cannot-link(a,c).
/// 3. **b-isolated** — symmetric.
/// 4. **c-isolated** — symmetric.
///
/// These four cases partition the integer feasible region: any AF either
/// has all three together (case 1), or at least one of them is alone
/// w.r.t. the other two (cases 2–4). The partitioning is exhaustive and
/// disjoint, so the branching is sound.
///
/// **Triple selection**: pick the triple whose three pairwise mass values
/// are all above `MIN_TRIPLE_PAIR_MASS` (so all three pairs are genuinely
/// "in question") and whose minimum pairwise mass is highest — this
/// concentrates the branching where the LP is most ambiguous.
///
/// **Fallback**: if no triple has all three pairs above the threshold,
/// fall back to most-fractional pair branching. This handles early /
/// late stages of the search where the LP is mostly integer-feasible
/// modulo one pair.
pub struct ClusterBranching {
    /// Each of the three pairwise masses in the chosen triple must
    /// exceed this for cluster branching to fire. Below it, fall back
    /// to pair branching to avoid 4-way splits on near-integer LP
    /// support (where the 4× tree multiplier wouldn't pay off).
    pub min_triple_pair_mass: f64,
}

const DEFAULT_MIN_TRIPLE_PAIR_MASS: f64 = 0.3;

impl ClusterBranching {
    pub fn new() -> Self {
        Self {
            min_triple_pair_mass: DEFAULT_MIN_TRIPLE_PAIR_MASS,
        }
    }

    pub fn with_min_triple_pair_mass(mut self, m: f64) -> Self {
        self.min_triple_pair_mass = m;
        self
    }
}

impl Default for ClusterBranching {
    fn default() -> Self {
        Self::new()
    }
}

impl BranchSelector for ClusterBranching {
    fn select(
        &mut self,
        ctx: &SelectionContext,
        _rmp: &mut Rmp,
    ) -> Option<Vec<Branchings>> {
        let (together, support) =
            pair_mass_and_support(ctx.columns, ctx.values, ctx.num_leaves);
        let stride = ctx.num_leaves + 1;

        // Collect all leaves that appear in at least one fractional pair —
        // any leaf in the chosen triple must have at least two qualifying
        // partners, so we only need to consider these.
        let mut hot_leaves: Vec<usize> = Vec::new();
        for a in 1..=ctx.num_leaves {
            let row = a * stride;
            let any = (1..=ctx.num_leaves)
                .filter(|&b| b != a)
                .any(|b| together[row + b] >= self.min_triple_pair_mass);
            if any {
                hot_leaves.push(a);
            }
        }

        // Score every viable triple. We iterate hot_leaves × hot_leaves ×
        // hot_leaves with a < b < c; the inner pair-mass lookups are O(1).
        // Even with all ~num_leaves hot, the work is O(n³) and bounded by
        // n³/6 lookups — for n=140 that's ~457K ops, ~µs class.
        let mut best: Option<(usize, usize, usize, f64)> = None;
        for i in 0..hot_leaves.len() {
            let a = hot_leaves[i];
            for j in (i + 1)..hot_leaves.len() {
                let b = hot_leaves[j];
                let mab = together[a * stride + b];
                if mab < self.min_triple_pair_mass {
                    continue;
                }
                for k in (j + 1)..hot_leaves.len() {
                    let c = hot_leaves[k];
                    let mac = together[a * stride + c];
                    let mbc = together[b * stride + c];
                    if mac < self.min_triple_pair_mass
                        || mbc < self.min_triple_pair_mass
                    {
                        continue;
                    }
                    let m_min = mab.min(mac).min(mbc);
                    if best.map_or(true, |(_, _, _, s)| m_min > s) {
                        best = Some((a, b, c, m_min));
                    }
                }
            }
        }

        if let Some((a, b, c, _)) = best {
            return Some(build_triple_children(
                ctx.branchings,
                a as u32,
                b as u32,
                c as u32,
            ));
        }

        // No qualifying triple — fall back to most-fractional pair so we
        // never get stuck. Use the unified pairs scan we already did.
        let _ = support;
        let pairs = fractional_pairs(&together, &support, ctx.num_leaves);
        let pair = pairs.into_iter().next().map(|(p, _, _)| p)?;
        let (left, right) = ctx.branchings.split_on(pair);
        Some(vec![left, right])
    }
}

/// Build the four child branchings for cluster branching on `(a, b, c)`.
/// The four cases partition every AF: either all three together, or
/// exactly one of {a, b, c} is in a different component from the other
/// two. Inconsistent branchings (e.g. a must-link that contradicts an
/// inherited cannot-link) are dropped; the search loop is fine with
/// `< 4` children.
fn build_triple_children(parent: &Branchings, a: u32, b: u32, c: u32) -> Vec<Branchings> {
    let ab = LeafPair::new(a, b);
    let ac = LeafPair::new(a, c);
    let bc = LeafPair::new(b, c);

    let mut children = Vec::with_capacity(4);

    // 1. All-together.
    let mut all = parent.clone();
    all.push_must_link(ab);
    all.push_must_link(ac);
    all.push_must_link(bc);
    if !all.is_inconsistent() {
        children.push(all);
    }

    // 2. a-isolated.
    let mut a_iso = parent.clone();
    a_iso.push_cannot_link(ab);
    a_iso.push_cannot_link(ac);
    if !a_iso.is_inconsistent() {
        children.push(a_iso);
    }

    // 3. b-isolated.
    let mut b_iso = parent.clone();
    b_iso.push_cannot_link(ab);
    b_iso.push_cannot_link(bc);
    if !b_iso.is_inconsistent() {
        children.push(b_iso);
    }

    // 4. c-isolated.
    let mut c_iso = parent.clone();
    c_iso.push_cannot_link(ac);
    c_iso.push_cannot_link(bc);
    if !c_iso.is_inconsistent() {
        children.push(c_iso);
    }

    children
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
