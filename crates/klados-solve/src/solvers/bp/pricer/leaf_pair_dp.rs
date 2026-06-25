//! Tier-2 pricer for **m ≥ 3** — leaf-pair DP with multi-tree bitmask
//! intersection (ported from `bp-multi`).
//!
//! ## Why this exists
//!
//! Our previous tier-2 for m≥3 (`PairDpFilterPricer`) ran Steel-Warnow
//! pair-DP on `(T₀, T₁)` and post-filtered against `T₂..Tₘ₋₁`. For
//! topologically-disagreeing trees, *every* DP candidate gets filtered out
//! and the pricer reports `Exhausted` even when valid columns of higher RC
//! exist — they're just not the best at any (u₀, u₁) anchor.
//!
//! This pricer takes a different approach: state is a **leaf pair**
//! `(label_a, label_b)`, not a node pair. The column is anchored at
//! `LCA(a, b)` in **every** tree simultaneously, with leaves split into
//! "a-side" and "b-side" by the tree's own children of LCA. Extension
//! candidates `c` must lie in the descendant set of "a-side" in **every**
//! tree (computed via bitmask intersection of `desc[ti]` per tree). So
//! emitted columns are valid AF components by construction across all m
//! trees — no post-filter, no rejection cascade.
//!
//! ## Recurrence
//!
//! ```text
//! solve_pair(a, b) = -root_penalty(a, b) + solve_side(a, b) + solve_side(b, a)
//!
//! solve_side(a, b):
//!   best = α[a] − pair_singleton_penalty[a, b]    (just leaf a alone)
//!   for c in (descendants of a-side in EVERY tree, c ≠ a, c ≠ b):
//!       cand = solve_pair(a, c) − pen[a, c]
//!       best = max(best, cand)
//!   return best
//! ```
//!
//! Memoized on `(a, b)` pairs of active labels — state space O(p²) where
//! `p` = active labels (independent of n). Multi-tree cost lives in the
//! bitmask intersection inside `solve_side`'s extension loop, not in the
//! state itself, so this scales cleanly to large m.
//!
//! ## Soundness — generator *and* certifier
//!
//! This pricer is both. `Found` when an emittable improving column exists;
//! `Converged` when a **full** all-anchor scan proves none exists anywhere;
//! `Improving` otherwise (improving column exists but none emittable, or the
//! scan was trial-limited so convergence is undecided).
//!
//! The certification is the spec's load-bearing fact (§3 of the pricing
//! rework spec): for any column `C` and any leaf pair `(a,b) ⊆ C`,
//! `solve_pair(a,b) ≥ score(C)`. So the constraint-blind max of `solve_pair`
//! over **every** anchor pair upper-bounds the score of every column. If that
//! max is `≤ 1+ε`, no improving column exists — `Converged` is sound for any
//! `m`. This requires the full scan: no `pair_trial_limit` cut and no early
//! `target` break (the `completed` flag tracks this). The α-filtered active
//! set is sound here because any improving column shrinks to an all-α>0
//! sub-column of no-lesser score, and must-linked leaves are force-activated.

use fixedbitset::FixedBitSet;
use klados_core::{NONE, Tree};

use crate::solvers::bp::column::{AfColumn, is_valid_af_component};

use super::{Pricer, PricerScratch, PricingContext, PricingResult, adaptive_m2_batch_size};

const PRICING_EPS: f64 = 1.0e-8;
const NEG_INF: f64 = f64::NEG_INFINITY;

/// Returns true if anchor cache is enabled via config.
fn anchor_cache_enabled(scratch: &PricerScratch) -> bool {
    scratch.use_anchor_cache
}
/// Sentinel in `memo_side_split` meaning "leaf-only side, no extension chosen".
const SPLIT_LEAF_ONLY: u32 = u32::MAX - 1;
/// `solve_side` hoists per-tree bitset slice pointers into a stack array of
/// this fixed size to avoid a heap allocation per call. Instances with more
/// trees than this fall back to a `Vec`; on the v2 benchmark the maximum
/// observed m is 36, so we set this generously above the long-tail observed
/// distribution.
const MAX_TREES_INLINE_HOIST: usize = 40;

pub struct LeafPairDpPricer {
    // ── Instance-fixed precompute (independent of duals) ──
    /// stride = `num_leaves + 1`; tables index by `label * stride + label`.
    stride: usize,
    /// `label_lca[ti][a*stride + b]` = LCA(a, b) in tree ti.
    label_lca: Vec<Vec<u32>>,
    /// `label_side_child[ti][a*stride + b]` = child of LCA(a, b) in T_ti that
    /// contains leaf a (the "a-side" anchor).
    label_side_child: Vec<Vec<u32>>,
    /// `label_node[ti][label]` = node id of that leaf in tree ti.
    label_node: Vec<Vec<u32>>,
    /// `descendant_leaves[ti][node]` = bitmask of leaf labels descending from
    /// `node` in tree ti. Width num_leaves+1.
    descendant_leaves: Vec<Vec<FixedBitSet>>,

    // ── Per-call dual-dependent (rebuilt every price()) ──
    active_labels: Vec<u32>,
    label_to_active_idx: Vec<u32>,
    active_mask: FixedBitSet,
    /// Indexed by `pair_idx(a, b) = a * p + b`.
    pair_root: Vec<Vec<u32>>, // per tree
    pair_side_child: Vec<Vec<u32>>, // per tree
    /// `pair_side_parent_prefix_beta[ti][a*p+b]` = `prefix_beta[ti][parent(side_child[ti][a,b])]`,
    /// or 0.0 if side_child is the root. Used to correct `pair_penalty` for
    /// extension columns anchored below root (subtract β path from side_child
    /// to root).
    pair_side_parent_prefix_beta: Vec<Vec<f64>>,
    /// Sum of beta along path-to-root for each tree node (precomputed).
    prefix_beta: Vec<Vec<f64>>,
    /// Subtree-α sum for each tree node (over active labels only).
    sum_alpha: Vec<Vec<f64>>,
    pair_penalty: Vec<f64>,
    pair_ub: Vec<f64>,
    pair_singleton_penalty: Vec<f64>,

    // ── Memo (reset each call after table rebuild) ──
    memo_pair: Vec<f64>,
    memo_side_score: Vec<f64>,
    memo_side_split: Vec<u32>,
    /// Cached reconstructed label vectors per pair, to avoid redundant
    /// reconstruction when the same pair is visited from multiple parents.
    memo_pair_labels: Vec<Option<Vec<u32>>>,

    num_leaves: usize,
    num_trees: usize,
    max_per_call: usize,
    /// Maximum number of (a, b) leaf-pair candidates to evaluate per pricing
    /// call (unlimited = u32::MAX). Limits the DP work and matches bp-multi's
    /// FASTPRICER_PAIR_TRIALS / WIDEPRICER_PAIR_TRIALS behaviour.
    pair_trial_limit: u32,
    /// If the limited proxy scan finds no columns, continue with the full
    /// leaf-pair scan using the same refreshed tables and memoization.  This
    /// matches bp-multi's fast-then-exact pipeline without paying for two
    /// separate pricer objects and two O(p²·m) dual refreshes.
    fallback_full_when_empty: bool,
    /// Skip the pricer entirely when active_labels < this threshold (0 = always
    /// active).  Matches bp-multi's WIDEPRICER_MIN_ACTIVE_LABELS gate.
    min_active_labels: usize,
    /// ── Branch-awareness (set at each `price()` call) ──
    /// `cannot_sets[l]` = bitset of every leaf cannot-linked to `l`. The
    /// `solve_side` extension masks these out, so a column never extends a
    /// side anchored at `l` with one of `l`'s cannot-partners — constraint-
    /// aware generation along the recursion spine. Cross-anchor cannot-link
    /// (both endpoints deep extensions on opposite sides) is repaired at
    /// emission by [`Self::repair_to_valid`].
    cannot_sets: Vec<FixedBitSet>,
    /// Leaves whose `cannot_sets` entry is non-empty this call — used to
    /// clear only the touched bitsets between calls.
    cannot_dirty: Vec<u32>,
    /// `has_cannot[l]` mirrors `!cannot_sets[l].is_clear()` for a cheap
    /// per-anchor skip in the hot `solve_side` loop.
    has_cannot: Vec<bool>,
    /// Reusable scratch for `solve_side`'s phase-1 candidate collection.
    /// Avoids a Vec allocation per call (~2500 per CG iter on Class B).
    solve_side_candidates: Vec<(u32, u32)>,
    /// ── Branch-feasible certification (rebuilt each `price()` call) ──
    /// `class_root[l]` = representative leaf of `l`'s transitive must-link
    /// class (union-find over `branchings.must_link()`, flattened). A
    /// singleton class is its own root.
    class_root: Vec<u32>,
    /// Unordered `(root, root)` pairs of must-link classes that are
    /// cannot-link-conflicting: some leaf of one class cannot-links some leaf
    /// of the other (or, when `root_a == root_b`, a cannot-link lies inside one
    /// class — an inconsistent node). An anchor whose two leaves' class roots
    /// form such a pair cannot appear in any branch-feasible column.
    conflict_class_pairs: std::collections::HashSet<(u32, u32)>,
    /// ── Lagrangian must-link certification ──
    /// When set, a node that would otherwise exit `Improving` (completed scan,
    /// must-link active, feasible bound > 1+ε) instead runs a subgradient loop
    /// that relaxes the must-link equalities into the pricing objective and may
    /// prove convergence. Sound for any multiplier (see
    /// [`Self::lagrangian_must_certify`]); `μ = 0` reproduces the base bound.
    lagrangian_certify_enabled: bool,
    lagrangian_max_iters: usize,
    /// Reusable buffers for the subgradient loop (avoid per-iter allocation).
    lagrangian_mu: Vec<f64>,
    lagrangian_alpha_buf: Vec<f64>,
}

impl LeafPairDpPricer {
    pub fn new(trees: &[Tree]) -> Self {
        assert!(trees.len() >= 2, "LeafPairDpPricer requires m≥2");
        let mut p = Self::new_raw(trees);
        p.pair_trial_limit = u32::MAX;
        p
    }

    pub fn with_pair_trial_limit(mut self, limit: u32) -> Self {
        self.pair_trial_limit = limit;
        self
    }

    pub fn with_fallback_full_when_empty(mut self, enabled: bool) -> Self {
        self.fallback_full_when_empty = enabled;
        self
    }

    pub fn with_min_active_labels(mut self, n: usize) -> Self {
        self.min_active_labels = n;
        self
    }

    pub fn with_max_per_call(mut self, n: usize) -> Self {
        self.max_per_call = n;
        self
    }

    fn new_raw(trees: &[Tree]) -> Self {
        // We assume all trees share the same `num_leaves`. Pull from the first.
        let num_leaves = (trees
            .iter()
            .flat_map(|t| t.label.iter().copied())
            .max()
            .unwrap_or(0)) as usize;
        let stride = num_leaves + 1;
        let num_trees = trees.len();

        // ── descendant_leaves: per tree, post-order union of children ──
        let descendant_leaves: Vec<Vec<FixedBitSet>> = trees
            .iter()
            .map(|tree| {
                let mut leaves = vec![FixedBitSet::with_capacity(num_leaves + 1); tree.num_nodes()];
                for node in tree.post_order_vec() {
                    if tree.is_leaf(node) {
                        let lbl = tree.label[node as usize] as usize;
                        leaves[node as usize].insert(lbl);
                    } else {
                        let (l, r) = tree.children_pair(node);
                        let mut bits = leaves[l as usize].clone();
                        bits.union_with(&leaves[r as usize]);
                        leaves[node as usize] = bits;
                    }
                }
                leaves
            })
            .collect();

        // ── label_node, label_lca, label_side_child ──
        let mut label_node: Vec<Vec<u32>> = Vec::with_capacity(num_trees);
        let mut label_lca: Vec<Vec<u32>> = Vec::with_capacity(num_trees);
        let mut label_side_child: Vec<Vec<u32>> = Vec::with_capacity(num_trees);
        for (ti, tree) in trees.iter().enumerate() {
            let mut node_by_label = vec![0u32; stride];
            for la in 1..=num_leaves as u32 {
                node_by_label[la as usize] = tree.node_by_label(la);
            }
            let mut lca_table = vec![0u32; stride * stride];
            let mut side_table = vec![0u32; stride * stride];
            for la in 1..=num_leaves as u32 {
                let node_a = node_by_label[la as usize];
                let base = (la as usize) * stride;
                for lb in 1..=num_leaves as u32 {
                    let idx = base + lb as usize;
                    if la == lb {
                        lca_table[idx] = node_a;
                        side_table[idx] = node_a;
                        continue;
                    }
                    let node_b = node_by_label[lb as usize];
                    let root = tree.nearest_common_ancestor(node_a, node_b);
                    lca_table[idx] = root;
                    let child = if tree.is_leaf(root) {
                        root
                    } else {
                        let (left, right) = tree.children_pair(root);
                        if descendant_leaves[ti][left as usize].contains(la as usize) {
                            left
                        } else {
                            right
                        }
                    };
                    side_table[idx] = child;
                }
            }
            label_node.push(node_by_label);
            label_lca.push(lca_table);
            label_side_child.push(side_table);
        }

        Self {
            stride,
            label_lca,
            label_side_child,
            label_node,
            descendant_leaves,
            active_labels: Vec::new(),
            label_to_active_idx: vec![u32::MAX; stride],
            active_mask: FixedBitSet::with_capacity(num_leaves + 1),
            pair_root: vec![Vec::new(); num_trees],
            pair_side_child: vec![Vec::new(); num_trees],
            pair_side_parent_prefix_beta: vec![Vec::new(); num_trees],
            prefix_beta: trees.iter().map(|t| vec![0.0; t.num_nodes()]).collect(),
            sum_alpha: trees.iter().map(|t| vec![0.0; t.num_nodes()]).collect(),
            pair_penalty: Vec::new(),
            pair_ub: Vec::new(),
            pair_singleton_penalty: Vec::new(),
            memo_pair: Vec::new(),
            memo_side_score: Vec::new(),
            memo_side_split: Vec::new(),
            memo_pair_labels: Vec::new(),
            num_leaves,
            num_trees,
            max_per_call: 64,
            pair_trial_limit: u32::MAX,
            fallback_full_when_empty: false,
            min_active_labels: 0,
            cannot_sets: Vec::new(),
            cannot_dirty: Vec::new(),
            has_cannot: Vec::new(),
            solve_side_candidates: Vec::new(),
            class_root: Vec::new(),
            conflict_class_pairs: std::collections::HashSet::new(),
            lagrangian_certify_enabled: false,
            lagrangian_max_iters: 0,
            lagrangian_mu: Vec::new(),
            lagrangian_alpha_buf: Vec::new(),
        }
    }

    /// Enable Lagrangian must-link certification with `iters` subgradient
    /// steps (0 disables). See [`Self::lagrangian_must_certify`].
    pub fn with_lagrangian_certify(mut self, iters: usize) -> Self {
        self.lagrangian_certify_enabled = iters > 0;
        self.lagrangian_max_iters = iters;
        self
    }

    /// Collect active labels from alpha. Returns `true` if the set changed
    /// since the last call.
    fn ensure_active_labels(&mut self, alpha: &[f64], ctx: &PricingContext) -> bool {
        let prev: Vec<u32> = std::mem::take(&mut self.active_labels);
        self.active_mask.clear();
        for v in self.label_to_active_idx.iter_mut() {
            *v = u32::MAX;
        }
        let activate = |me: &mut Self, label: u32| {
            if me.active_mask.contains(label as usize) {
                return;
            }
            me.active_mask.insert(label as usize);
            me.label_to_active_idx[label as usize] = me.active_labels.len() as u32;
            me.active_labels.push(label);
        };
        for label in 1..=self.num_leaves as u32 {
            if alpha[label as usize] > 1.0e-12 {
                activate(self, label);
            }
        }
        // Must-linked leaves stay active even with alpha <= 0: a must-link
        // constraint can force such a leaf into an improving column. Excluding
        // it would make the pricer unable to build that column and falsely
        // report convergence — an unsound LP bound.
        for pair in ctx.branchings.must_link() {
            if (pair.a as usize) <= self.num_leaves {
                activate(self, pair.a);
            }
            if (pair.b as usize) <= self.num_leaves {
                activate(self, pair.b);
            }
        }
        self.active_labels != prev
    }

    /// Rebuild per-pair LCA / side-child tables (expensive — O(p²·m)).
    /// Only called when the active label set changes.
    fn rebuild_pair_tables(&mut self, _trees: &[Tree]) {
        let p = self.active_labels.len();
        let pair_count = p * p;
        for ti in 0..self.num_trees {
            self.pair_root[ti].clear();
            self.pair_root[ti].resize(pair_count, 0);
            self.pair_side_child[ti].clear();
            self.pair_side_child[ti].resize(pair_count, 0);
            for a in 0..p {
                let la = self.active_labels[a] as usize;
                let base_ws = la * self.stride;
                let base_out = a * p;
                for b in 0..p {
                    let lb = self.active_labels[b] as usize;
                    let ws_idx = base_ws + lb;
                    self.pair_root[ti][base_out + b] = self.label_lca[ti][ws_idx];
                    self.pair_side_child[ti][base_out + b] = self.label_side_child[ti][ws_idx];
                }
            }
        }
    }

    /// Recompute dual-dependent tables in-place (prefix-β, subtree-α, pair
    /// penalties).  Much cheaper than `rebuild_pair_tables` — O(n·m + p²·m)
    /// but no LCA lookups.  Called on every CG call.
    fn refresh_dual_tables(&mut self, trees: &[Tree], alpha: &[f64], beta: &[Vec<f64>]) {
        let p = self.active_labels.len();
        let pair_count = p * p;

        // --- prefix_beta & sum_alpha (per tree, O(n)) ---
        for ti in 0..self.num_trees {
            let tree = &trees[ti];
            for node in tree.post_order_vec() {
                if tree.is_leaf(node) {
                    self.sum_alpha[ti][node as usize] =
                        alpha[tree.label[node as usize] as usize].max(0.0);
                } else {
                    let (l, r) = tree.children_pair(node);
                    self.sum_alpha[ti][node as usize] =
                        self.sum_alpha[ti][l as usize] + self.sum_alpha[ti][r as usize];
                }
            }
            for node in tree.pre_order() {
                let parent = tree.parent[node as usize];
                let parent_sum = if parent == NONE {
                    0.0
                } else {
                    self.prefix_beta[ti][parent as usize]
                };
                let own = if tree.is_leaf(node) {
                    0.0
                } else {
                    beta[ti][node as usize]
                };
                self.prefix_beta[ti][node as usize] = parent_sum + own;
            }
        }

        // pair_penalty, pair_ub, pair_singleton_penalty and
        // pair_side_parent_prefix_beta in one O(p²·m) pass (same arithmetic as
        // the old two-pass form). `pair_side_parent_prefix_beta` is the `lower`
        // term, computed once; `upper` depends only on `a` so it is hoisted out
        // of the `c` loop; `nps[r]` is read directly instead of via a per-tree
        // scratch vector (which allocated every call).
        self.pair_penalty.clear();
        self.pair_penalty.resize(pair_count, 0.0);
        self.pair_ub.clear();
        self.pair_ub.resize(pair_count, f64::INFINITY);
        self.pair_singleton_penalty.clear();
        self.pair_singleton_penalty.resize(pair_count, 0.0);

        for ti in 0..self.num_trees {
            let tree = &trees[ti];
            self.pair_side_parent_prefix_beta[ti].clear();
            self.pair_side_parent_prefix_beta[ti].resize(pair_count, 0.0);
            for a in 0..p {
                let base = a * p;
                let leaf_a_node = self.label_node[ti][self.active_labels[a] as usize];
                // `upper` = prefix-β at the parent of leaf a; depends only on a.
                let upper = if leaf_a_node != NONE {
                    let dp = tree.parent[leaf_a_node as usize];
                    if dp != NONE {
                        self.prefix_beta[ti][dp as usize]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                for c in 0..p {
                    let idx = base + c;
                    let r = self.pair_root[ti][idx] as usize;
                    let pr = tree.parent[r];
                    let nps_r = if pr != NONE {
                        self.prefix_beta[ti][pr as usize]
                    } else {
                        0.0
                    };
                    self.pair_penalty[idx] += nps_r;
                    if self.sum_alpha[ti][r] < self.pair_ub[idx] {
                        self.pair_ub[idx] = self.sum_alpha[ti][r];
                    }

                    let anc = self.pair_side_child[ti][idx];
                    let ap = tree.parent[anc as usize];
                    let lower = if ap != NONE {
                        self.prefix_beta[ti][ap as usize]
                    } else {
                        0.0
                    };
                    self.pair_side_parent_prefix_beta[ti][idx] = lower;
                    let diff = upper - lower;
                    if anc != leaf_a_node && diff > 0.0 {
                        self.pair_singleton_penalty[idx] += diff;
                    }
                }
            }
        }
    }

    fn reset_memos(&mut self) {
        let pair_count = self.active_labels.len() * self.active_labels.len();
        self.memo_pair.clear();
        self.memo_pair.resize(pair_count, f64::NAN);
        self.memo_side_score.clear();
        self.memo_side_score.resize(pair_count, f64::NAN);
        self.memo_side_split.clear();
        self.memo_side_split.resize(pair_count, u32::MAX);
        self.memo_pair_labels.clear();
        self.memo_pair_labels.resize(pair_count, None);
    }

    fn pair_idx(&self, a: usize, b: usize) -> usize {
        a * self.active_labels.len() + b
    }

    /// Rebuild the must-link-class union-find and the cannot-link-conflicting
    /// class-pair set for the current branch state. O(n + |must| + |cannot|).
    ///
    /// This drives [`Self::anchor_feasible`], the branch-feasible certification
    /// filter. With no branch constraints the conflict set is empty and every
    /// anchor is feasible, so `feasible_global_max == global_max` and behaviour
    /// is identical to the unconstrained pricer.
    fn rebuild_branch_classes(&mut self, ctx: &PricingContext) {
        let n = self.num_leaves;
        if self.class_root.len() != n + 1 {
            self.class_root = (0..=n as u32).collect();
        } else {
            for (i, r) in self.class_root.iter_mut().enumerate() {
                *r = i as u32;
            }
        }
        for pair in ctx.branchings.must_link() {
            if (pair.a as usize) <= n && (pair.b as usize) <= n {
                self.union(pair.a, pair.b);
            }
        }
        // Flatten so `class_root[l]` is the ultimate representative.
        for l in 1..=n as u32 {
            let r = self.find(l);
            self.class_root[l as usize] = r;
        }
        self.conflict_class_pairs.clear();
        for pair in ctx.branchings.cannot_link() {
            if (pair.a as usize) <= n && (pair.b as usize) <= n {
                let ra = self.class_root[pair.a as usize];
                let rb = self.class_root[pair.b as usize];
                self.conflict_class_pairs
                    .insert(if ra <= rb { (ra, rb) } else { (rb, ra) });
            }
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.class_root[x as usize] != x {
            // Path halving.
            let gp = self.class_root[self.class_root[x as usize] as usize];
            self.class_root[x as usize] = gp;
            x = gp;
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.class_root[ra as usize] = rb;
        }
    }

    /// True iff leaves `la`, `lb` can co-occur in some branch-feasible AF
    /// component, i.e. their must-link classes are not cannot-link-conflicting.
    ///
    /// Soundness of the certification rests on this being a *sufficient*
    /// condition for infeasibility (it may return `true` for some pairs that
    /// happen to be jointly infeasible for other reasons — that only loosens
    /// the bound, never unsounds it). The key property used by the `Converged`
    /// proof: for any branch-feasible column `C` and any two leaves
    /// `la, lb ∈ C`, their whole classes lie in `C` (must-closure), so if those
    /// classes were cannot-conflicting `C` would violate a cannot-link — hence
    /// `anchor_feasible(la, lb)` holds for every leaf pair of every feasible
    /// column.
    fn anchor_feasible(&self, la: u32, lb: u32) -> bool {
        if self.conflict_class_pairs.is_empty() {
            return true;
        }
        let ra = self.class_root[la as usize];
        let rb = self.class_root[lb as usize];
        let key = if ra <= rb { (ra, rb) } else { (rb, ra) };
        !self.conflict_class_pairs.contains(&key)
    }

    /// Recompute the branch-feasible certification bound under a (possibly
    /// shifted) `alpha`, doing a **full** scan over all feasible anchors with
    /// tight UB > 1+ε. Returns `(bound, argmax_anchor)` where `bound` is the
    /// max of `solve_pair` over those anchors and `argmax_anchor` is the
    /// `(a,b)` achieving it (active indices), or `None` if no feasible anchor
    /// exceeds the threshold.
    ///
    /// The caller must have set `alpha` so the active-label *set* is unchanged
    /// (the Lagrangian shift only touches force-activated must-linked leaves,
    /// so it is). This lets us refresh the dual tables without an active-set
    /// rebuild.
    fn relaxed_feasible_cert(
        &mut self,
        ctx: &PricingContext,
        alpha: &[f64],
    ) -> (f64, Option<(usize, usize)>) {
        self.refresh_dual_tables(ctx.trees, alpha, ctx.beta);
        self.reset_memos();
        let p = self.active_labels.len();
        let mut best = NEG_INF;
        let mut arg = None;
        for a in 0..p {
            for b in (a + 1)..p {
                let idx = a * p + b;
                let tight = self.pair_ub[idx] - self.root_penalty(a, b, ctx.beta);
                if tight <= 1.0 + PRICING_EPS {
                    continue;
                }
                let la = self.active_labels[a];
                let lb = self.active_labels[b];
                if !self.anchor_feasible(la, lb) {
                    continue;
                }
                let s = self.solve_pair(a, b, alpha, ctx.beta);
                if s > best {
                    best = s;
                    arg = Some((a, b));
                }
            }
        }
        (best, arg)
    }

    /// Lagrangian certification of must-link convergence.
    ///
    /// The branch-feasible pricing problem maximises `score(C)` over must-link-
    /// closed, cannot-respecting valid AF components. Relax each must-link
    /// equality `1_x = 1_y` with a multiplier `μ`:
    ///
    /// ```text
    /// L(C, μ) = score(C) + Σ_{(x,y) ∈ must} μ_xy (1_x[C] − 1_y[C]).
    /// ```
    ///
    /// For **any** `μ` and any must-closed `C` the added term is zero, so
    /// `max_C L(C, μ) ≥ max_{must-closed C} score(C)`. The term is linear in the
    /// leaf indicators, so `L` is just `score` with `α` shifted by
    /// `Δ_l = Σ_{(l,·)} μ − Σ_{(·,l)} μ`; the relaxed max is therefore computed
    /// by the existing DP on shifted duals ([`Self::relaxed_feasible_cert`]).
    ///
    /// Hence if the relaxed feasible bound is `≤ 1+ε` for *some* `μ`, no
    /// must-closed feasible column is improving → convergence is certified.
    /// This is sound for every `μ` (so it can never over-certify); the
    /// subgradient steps only try to tighten the bound. Returns `true` iff
    /// convergence is proven.
    fn lagrangian_must_certify(&mut self, ctx: &PricingContext) -> bool {
        const STEP0: f64 = 1.0;
        let must = ctx.branchings.must_link();
        if must.is_empty() {
            return false;
        }
        let n = self.num_leaves;
        self.lagrangian_mu.clear();
        self.lagrangian_mu.resize(must.len(), 0.0);
        let mut mu = std::mem::take(&mut self.lagrangian_mu);
        let mut alpha = std::mem::take(&mut self.lagrangian_alpha_buf);

        let mut certified = false;
        for iter in 0..self.lagrangian_max_iters {
            // Each iteration is a full DP scan; honour cancellation/deadline so
            // the relaxation can never push a hard node past its time budget.
            // Bailing early just yields `Improving` (never a false certificate).
            if ctx.terminate.load(std::sync::atomic::Ordering::Relaxed)
                || ctx.deadline.is_some_and(|d| std::time::Instant::now() >= d)
            {
                break;
            }
            // shifted α = base α + Δ(μ).
            alpha.clear();
            alpha.extend_from_slice(ctx.alpha);
            if alpha.len() < n + 1 {
                alpha.resize(n + 1, 0.0);
            }
            for (k, pair) in must.iter().enumerate() {
                if (pair.a as usize) <= n {
                    alpha[pair.a as usize] += mu[k];
                }
                if (pair.b as usize) <= n {
                    alpha[pair.b as usize] -= mu[k];
                }
            }

            let (bound, arg) = self.relaxed_feasible_cert(ctx, &alpha);
            if bound <= 1.0 + PRICING_EPS {
                certified = true;
                break;
            }
            let Some((a, b)) = arg else {
                break;
            };
            // Subgradient of `max_C L` w.r.t. μ is the must-link violation of the
            // maximiser; step against it to *minimise* the bound.
            let labels = self.pair_labels(a, b, &alpha, ctx.beta);
            let step = STEP0 / (iter as f64 + 1.0);
            for (k, pair) in must.iter().enumerate() {
                let gx = i32::from(labels.binary_search(&pair.a).is_ok());
                let gy = i32::from(labels.binary_search(&pair.b).is_ok());
                mu[k] -= step * f64::from(gx - gy);
            }
        }

        self.lagrangian_mu = mu;
        self.lagrangian_alpha_buf = alpha;
        certified
    }

    fn root_penalty(&self, a: usize, b: usize, beta: &[Vec<f64>]) -> f64 {
        (0..self.num_trees)
            .map(|ti| beta[ti][self.pair_root[ti][self.pair_idx(a, b)] as usize])
            .sum()
    }

    /// Diagnostic gap from the chosen side extensions to the leaf-only
    /// alternatives.
    ///
    /// The leaf-pair DP decomposes `s(a, b) = -root_penalty + solve_side(a, b)
    /// + solve_side(b, a)`. On each side, the optimum is either "leaf-only"
    /// (just `a` alone, scored `α[a] − pair_singleton_penalty`) or some
    /// extension `c` (`solve_pair(a, c) − pen(a, c)`), and the "leaf-only" alternative is always a valid competing column at this anchor.
    ///
    /// This is *not* the true second-best gap and must not be used as a
    /// sound skip/optimality certificate. It is retained only as a cheap
    /// statistic while the cache acts as positive-column reuse.
    ///
    /// When the optimum on either side is "leaf-only", the diagnostic value
    /// is 0.
    ///
    /// Must be called AFTER `solve_pair(a, b)` so memos are populated.
    fn anchor_gap_diagnostic(&self, a: usize, b: usize, alpha: &[f64]) -> f64 {
        let side_ab_gap = self.side_gap_to_leaf_only(a, b, alpha);
        let side_ba_gap = self.side_gap_to_leaf_only(b, a, alpha);
        // Either side might be leaf-only-optimal (gap = 0).
        side_ab_gap.min(side_ba_gap).max(0.0)
    }

    /// Gap from `solve_side(a, b)`'s optimum to the leaf-only alternative
    /// at the same side. Returns 0 if leaf-only IS the optimum.
    fn side_gap_to_leaf_only(&self, a: usize, b: usize, alpha: &[f64]) -> f64 {
        let idx = self.pair_idx(a, b);
        let split = self.memo_side_split[idx];
        if split == SPLIT_LEAF_ONLY {
            return 0.0;
        }
        let side_score = self.memo_side_score[idx];
        if !side_score.is_finite() {
            return 0.0;
        }
        let label_a = self.active_labels[a] as usize;
        let leaf_only = alpha[label_a] - self.pair_singleton_penalty[idx];
        (side_score - leaf_only).max(0.0)
    }

    fn solve_pair(&mut self, a: usize, b: usize, alpha: &[f64], beta: &[Vec<f64>]) -> f64 {
        debug_assert!(a != b);
        let idx = self.pair_idx(a, b);
        let val = self.memo_pair[idx];
        if !val.is_nan() {
            return if val.is_infinite() && val.is_sign_positive() {
                NEG_INF
            } else {
                val
            };
        }
        // Sentinel during recursion to break cycles.
        self.memo_pair[idx] = f64::INFINITY;

        let left = self.solve_side(a, b, alpha, beta);
        let right = self.solve_side(b, a, alpha, beta);
        let score = if left <= NEG_INF / 2.0 || right <= NEG_INF / 2.0 {
            NEG_INF
        } else {
            -self.root_penalty(a, b, beta) + left + right
        };
        self.memo_pair[idx] = score;
        score
    }

    fn solve_side(&mut self, a: usize, b: usize, alpha: &[f64], beta: &[Vec<f64>]) -> f64 {
        debug_assert!(a != b);
        let _ = beta; // beta used only via pre-computed prefix_beta tables
        let idx = self.pair_idx(a, b);
        let val = self.memo_side_score[idx];
        if !val.is_nan() {
            return if val.is_infinite() && val.is_sign_positive() {
                NEG_INF
            } else {
                val
            };
        }
        self.memo_side_score[idx] = f64::INFINITY;

        let label_a = self.active_labels[a];
        let mut best_score = alpha[label_a as usize] - self.pair_singleton_penalty[idx];
        let mut best_choice = SPLIT_LEAF_ONLY;

        let p = self.active_labels.len();

        // Side anchors per tree. `b_penalty_sum` is the β cost of the path
        // from each tree's side-child up to its root, summed across trees.
        // We subtract this from `pair_penalty[a, c]` when evaluating an
        // extension because extension columns are anchored at LCA(a, b),
        // not at the root — so they don't pay the β between side-child and
        // root.
        let mut b_penalty_sum = 0.0;
        let mut side_nodes: Vec<u32> = Vec::with_capacity(self.num_trees);
        for ti in 0..self.num_trees {
            let anc = self.pair_side_child[ti][idx];
            side_nodes.push(anc);
            // Caller must ensure trees are aligned with our precompute.
            // Use the first tree from the labels' view: actually we need the
            // tree itself. Pull from the `Tree` slice via parent indirection
            // we have from trees passed at refresh time. We don't have trees
            // here — we stored prefix_beta indexed by node directly, so:
            //   parent = ?
            // We need parent(anc). Store it during refresh.
            // For now: the parent can be looked up from anc->parent... but
            // we don't have the tree slice here. Workaround: precompute
            // `pair_side_parent_prefix_beta` during refresh.
            let pb_parent = self.pair_side_parent_prefix_beta[ti][idx];
            b_penalty_sum += pb_parent;
            let _ = anc;
        }

        const BLOCK_BITS: usize = std::mem::size_of::<usize>() * 8;
        const BLOCK_SHIFT: usize = BLOCK_BITS.trailing_zeros() as usize;
        const BLOCK_MASK: usize = BLOCK_BITS - 1;

        let la = label_a as usize;
        let lb = self.active_labels[b] as usize;
        let la_w = la >> BLOCK_SHIFT;
        let la_m = 1usize << (la & BLOCK_MASK);
        let lb_w = lb >> BLOCK_SHIFT;
        let lb_m = 1usize << (lb & BLOCK_MASK);

        // Bitmask intersection of "descendants of side_child in every tree",
        // restricted to the active mask, excluding a and b themselves.
        //
        // We split this into two phases to enable hoisting the per-tree
        // bitset slice pointers out of the per-block inner loop.
        //
        // Phase 1 (immutable self borrow): walk the m-way bitmask
        // intersection, collect the candidate set `(c_label, c)` that
        // survives the cannot-link filter and the static UB-vs-best-score
        // filter against the *initial* `best_score`.
        // Phase 2 (mutable self borrow): for each surviving candidate,
        // re-check the UB filter against the *current* `best_score`
        // (which may have grown via recursive solve_pair calls), then
        // recurse / read memo. This re-introduces the dynamic pruning the
        // single-phase code did inline; the cost is iterating the
        // candidate list twice in the worst case.
        //
        // Hoist motivation: `descendant_leaves[ti][side_nodes[ti] as
        // usize].as_slice()` walks 3 levels of Vec/FixedBitSet indirection
        // and `side_nodes[ti]` is constant for fixed (a,b). On Class B
        // (m=8–36) this function is invoked ~2500× per CG iter, so the
        // hoisted slice pointers save tens of thousands of pointer chases
        // per iter without changing the algorithm.
        // Constraint-aware extension: mask out every leaf cannot-linked to the
        // side anchor `label_a`. A column that extends `a`'s side is invalid
        // if it contains a cannot-partner of `a`, so we never even enumerate
        // those candidates — generation stays node-valid along the recursion
        // spine, no post-filter rejection.
        let cannot_a_blocks: &[usize] = if self.has_cannot[label_a as usize] {
            self.cannot_sets[label_a as usize].as_slice()
        } else {
            &[]
        };
        // Reuse the scratch candidates buffer; swap it out so recursive
        // solve_pair → solve_side calls get their own empty buffer.
        let mut candidates: Vec<(u32, u32)> = std::mem::take(&mut self.solve_side_candidates);
        candidates.clear();
        {
            // Inline-storage slice array for the common case (m ≤
            // MAX_TREES_INLINE_HOIST); falls back to a heap Vec for
            // larger m. Stack storage avoids an allocation in the inner
            // call path.
            let mut tree_slices_inline: [&[usize]; MAX_TREES_INLINE_HOIST] =
                [&[][..]; MAX_TREES_INLINE_HOIST];
            let mut tree_slices_overflow: Vec<&[usize]> = Vec::new();
            let slices: &[&[usize]] = if self.num_trees <= MAX_TREES_INLINE_HOIST {
                for ti in 0..self.num_trees {
                    tree_slices_inline[ti] =
                        self.descendant_leaves[ti][side_nodes[ti] as usize].as_slice();
                }
                &tree_slices_inline[..self.num_trees]
            } else {
                tree_slices_overflow.reserve(self.num_trees);
                for ti in 0..self.num_trees {
                    tree_slices_overflow
                        .push(self.descendant_leaves[ti][side_nodes[ti] as usize].as_slice());
                }
                &tree_slices_overflow
            };
            let active_mask_slice = self.active_mask.as_slice();
            let n_blocks = slices[0].len();
            for wi in 0..n_blocks {
                let mut w = slices[0][wi] & active_mask_slice[wi];
                for ti in 1..slices.len() {
                    w &= slices[ti][wi];
                }
                if wi == la_w {
                    w &= !la_m;
                }
                if wi == lb_w {
                    w &= !lb_m;
                }
                if let Some(&cb) = cannot_a_blocks.get(wi) {
                    w &= !cb;
                }
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    w &= w - 1;
                    let c_label = (wi << BLOCK_SHIFT) + bit;
                    let c = self.label_to_active_idx[c_label] as usize;
                    let idx_c = a * p + c;
                    let pen = self.pair_penalty[idx_c] - b_penalty_sum;
                    let ub = self.pair_ub[idx_c];
                    // Static filter against initial best_score; dynamic
                    // re-filter in phase 2.
                    if ub - pen <= best_score + 1.0e-12 {
                        continue;
                    }
                    candidates.push((c_label as u32, c as u32));
                }
            }
        }

        for &(_c_label, c) in &candidates {
            let c = c as usize;
            let idx_c = a * p + c;
            let pen = self.pair_penalty[idx_c] - b_penalty_sum;
            let ub = self.pair_ub[idx_c];
            // Dynamic re-filter: best_score may have grown via earlier
            // recursive solve_pair calls in this same phase 2 loop.
            if ub - pen <= best_score + 1.0e-12 {
                continue;
            }
            let cached = self.memo_pair[idx_c];
            let child = if !cached.is_nan() {
                if cached.is_infinite() && cached.is_sign_positive() {
                    NEG_INF
                } else {
                    cached
                }
            } else {
                self.solve_pair(a, c, alpha, beta)
            };
            if child <= NEG_INF / 2.0 {
                continue;
            }
            let cand = child - pen;
            if cand > best_score + 1.0e-12 {
                best_score = cand;
                best_choice = c as u32;
            }
        }

        // Return the candidates buffer to scratch so future calls reuse
        // its capacity instead of reallocating.
        candidates.clear();
        self.solve_side_candidates = candidates;

        self.memo_side_score[idx] = best_score;
        self.memo_side_split[idx] = best_choice;
        best_score
    }

    fn collect_pair(
        &mut self,
        a: usize,
        b: usize,
        alpha: &[f64],
        beta: &[Vec<f64>],
        out: &mut Vec<u32>,
    ) {
        self.collect_side(a, b, alpha, beta, out);
        self.collect_side(b, a, alpha, beta, out);
    }

    fn collect_side(
        &mut self,
        a: usize,
        b: usize,
        alpha: &[f64],
        beta: &[Vec<f64>],
        out: &mut Vec<u32>,
    ) {
        let idx = self.pair_idx(a, b);
        let s = self.memo_side_score[idx];
        if s.is_nan() || s.is_infinite() {
            self.solve_side(a, b, alpha, beta);
        }
        let choice = self.memo_side_split[idx];
        if choice == SPLIT_LEAF_ONLY {
            out.push(self.active_labels[a]);
        } else {
            self.collect_pair(a, choice as usize, alpha, beta, out);
        }
    }

    fn pair_labels(&mut self, a: usize, b: usize, alpha: &[f64], beta: &[Vec<f64>]) -> Vec<u32> {
        let idx = self.pair_idx(a, b);
        if let Some(cached) = self.memo_pair_labels[idx].clone() {
            return cached;
        }
        let mut labels = Vec::new();
        self.collect_pair(a, b, alpha, beta, &mut labels);
        labels.sort_unstable();
        labels.dedup();
        self.memo_pair_labels[idx] = Some(labels.clone());
        labels
    }

    fn quick_proxy(&self, a: usize, b: usize, alpha: &[f64], beta: &[Vec<f64>]) -> f64 {
        let la = self.active_labels[a] as usize;
        let lb = self.active_labels[b] as usize;
        let p = self.active_labels.len();
        alpha[la] + alpha[lb]
            - self.root_penalty(a, b, beta)
            - self.pair_singleton_penalty[a * p + b]
            - self.pair_singleton_penalty[b * p + a]
    }

    fn collect_from_order(
        &mut self,
        order: &[(f64, usize, usize)],
        trial_limit: usize,
        target: usize,
        ctx: &PricingContext,
        scratch: &mut PricerScratch,
    ) -> CollectResult {
        let p = self.active_labels.len();
        let mut found: Vec<(f64, AfColumn)> = Vec::new();
        // Constraint-blind global maximum of `solve_pair` over every anchor
        // actually scanned. This is the certification quantity (§3 of the
        // pricing rework spec): `solve_pair(a,b)` dominates every column at
        // anchor `(a,b)`, so once every order-pair is scanned and this max is
        // `≤ 1+ε`, no improving column exists anywhere → sound `Converged`.
        let mut global_max: f64 = NEG_INF;
        // Branch-feasible certification quantity: the same max restricted to
        // anchors that can co-occur in a branch-feasible column. Sound upper
        // bound on `max_{C ∈ C(B)} score(C)` and `≤ global_max` — this is what
        // gates `Converged` (see `anchor_feasible` and the `price()` gate).
        let mut feasible_global_max: f64 = NEG_INF;

        let cache_active =
            anchor_cache_enabled(scratch) && ctx.trees.len() == 2 && scratch.anchor_cache.is_some();

        let scan = order.len().min(trial_limit);
        scratch.pricing_stats.leaf_pair_scanned = 0;
        scratch.pricing_stats.leaf_pair_positive = 0;
        scratch.pricing_stats.leaf_pair_found = 0;
        scratch.pricing_stats.leaf_pair_seen = 0;
        scratch.pricing_stats.leaf_pair_repair_failed = 0;
        scratch.pricing_stats.leaf_pair_repair_nonprofitable = 0;
        scratch.pricing_stats.leaf_pair_branch_blocked = 0;
        scratch.pricing_stats.leaf_pair_blocked_must = 0;
        scratch.pricing_stats.leaf_pair_blocked_cannot = 0;
        scratch.pricing_stats.must_closure_attempted = 0;
        scratch.pricing_stats.must_closure_valid_positive = 0;
        scratch.pricing_stats.must_closure_valid_nonprofitable = 0;
        scratch.pricing_stats.must_closure_invalid = 0;
        scratch.pricing_stats.must_closure_fallback = 0;
        scratch.pricing_stats.leaf_pair_completed = false;
        scratch.pricing_stats.leaf_pair_trial_limited = scan < order.len();
        scratch.pricing_stats.leaf_pair_global_max = NEG_INF;
        scratch.pricing_stats.leaf_pair_feasible_global_max = NEG_INF;
        // `completed` ⇒ every pair in `order` had its `solve_pair` evaluated;
        // only then is `global_max` a valid convergence certificate.
        let mut completed = scan == order.len();
        for &(_, a, b) in order.iter().take(scan) {
            if found.len() >= target {
                // Early stop: enough emittable columns. The scan is now
                // incomplete, so `global_max` is not a valid convergence
                // certificate — the caller must not certify on this result.
                completed = false;
                break;
            }
            let la = self.active_labels[a];
            let lb = self.active_labels[b];
            scratch.pricing_stats.leaf_pair_scanned += 1;
            // Whether this anchor can occur in any branch-feasible column. Only
            // such anchors contribute to `feasible_global_max`, the quantity
            // that gates `Converged`. Computed once and reused at both
            // `solve_pair`/cache update sites below.
            let anchor_feasible = self.anchor_feasible(la, lb);

            // `pair_ub` may have tightened since we built the order (the
            // bound is recomputed each call). If it dropped to `≤ 1+ε` then
            // `solve_pair(a,b) ≤ pair_ub − root_penalty ≤ 1+ε`, so this anchor
            // cannot beat the threshold — skip it without affecting the
            // certification max.
            if self.pair_ub[a * p + b] <= 1.0 + PRICING_EPS {
                continue;
            }

            // --- Anchor cache fast path ---
            if cache_active {
                let cache_hit = {
                    let cache = scratch.anchor_cache.as_mut().unwrap();
                    cache.try_emit(la, lb, ctx.alpha, &ctx.beta[0], &ctx.beta[1], PRICING_EPS)
                };
                match cache_hit {
                    super::anchor_cache::CacheResult::Emit {
                        score: cached_score,
                    } => {
                        global_max = global_max.max(cached_score);
                        if anchor_feasible {
                            feasible_global_max = feasible_global_max.max(cached_score);
                        }
                        // Rebuild column from cached leaves. If the cached
                        // column is already seen or blocked, do NOT skip this
                        // anchor: a different column at the same anchor may
                        // still be improving, so fall through to the DP.
                        let labels = scratch
                            .anchor_cache
                            .as_ref()
                            .unwrap()
                            .entry_for(la, lb)
                            .unwrap()
                            .column_leaves
                            .clone();
                        if labels.len() >= 2 && !ctx.seen.contains(&labels) {
                            let column = scratch.builder.build_unchecked(labels, ctx.trees);
                            if !ctx.branchings.forbids(&column) {
                                // Re-score via the canonical column coverage as
                                // a final exact guard before returning.
                                let full_score = column.pricing_score(ctx.alpha, ctx.beta);
                                if full_score > 1.0 + PRICING_EPS {
                                    found.push((full_score, column));
                                    continue;
                                }
                            }
                        }
                        // Fall through to DP.
                    }
                    super::anchor_cache::CacheResult::Stale
                    | super::anchor_cache::CacheResult::Miss => {
                        // Fall through to DP.
                    }
                }
            }

            // `solve_pair` runs constraint-blind for *every* scanned anchor —
            // it is the certification quantity. Constraint filtering applies
            // only below, to emission.
            let score = self.solve_pair(a, b, ctx.alpha, ctx.beta);
            global_max = global_max.max(score);
            if anchor_feasible {
                feasible_global_max = feasible_global_max.max(score);
            }
            if score <= 1.0 + PRICING_EPS {
                continue;
            }
            scratch.pricing_stats.leaf_pair_positive += 1;
            // An improving column exists at this anchor. The DP builds it
            // constraint-blind; `solve_side`'s cannot-masking already keeps it
            // node-valid along the recursion spine, but a cross-anchor
            // cannot-link or a half-present must-group can still slip in.
            // `repair_to_valid` drops the offending leaves — any subset of an
            // agreement component is itself a valid agreement component — so
            // the emitted column is node-valid by construction. If repair
            // empties it below 2 leaves, this anchor contributes to the
            // `Improving` residue and we skip it.
            let raw_labels = self.pair_labels(a, b, ctx.alpha, ctx.beta);
            if raw_labels.len() < 2 {
                scratch.pricing_stats.leaf_pair_repair_failed += 1;
                continue;
            }
            let raw_violations = branch_violation_kinds(&raw_labels, ctx.branchings);
            if raw_violations.0 || raw_violations.1 {
                scratch.pricing_stats.leaf_pair_branch_blocked += 1;
                if raw_violations.0 {
                    scratch.pricing_stats.leaf_pair_blocked_must += 1;
                }
                if raw_violations.1 {
                    scratch.pricing_stats.leaf_pair_blocked_cannot += 1;
                }
            }
            let (labels_opt, closure_outcome) =
                branch_feasible_labels(&raw_labels, ctx.branchings, ctx.alpha, ctx.trees);
            match closure_outcome {
                MustClosureOutcome::NotAttempted => {}
                MustClosureOutcome::Used => {
                    scratch.pricing_stats.must_closure_attempted += 1;
                }
                MustClosureOutcome::InvalidFallback => {
                    scratch.pricing_stats.must_closure_attempted += 1;
                    scratch.pricing_stats.must_closure_invalid += 1;
                    scratch.pricing_stats.must_closure_fallback += 1;
                }
                MustClosureOutcome::RepairFallback => {
                    scratch.pricing_stats.must_closure_attempted += 1;
                    scratch.pricing_stats.must_closure_fallback += 1;
                }
            }
            let from_closure = closure_outcome == MustClosureOutcome::Used;
            let labels = match labels_opt {
                Some(l) => l,
                None => {
                    scratch.pricing_stats.leaf_pair_repair_failed += 1;
                    continue;
                }
            };
            if ctx.seen.contains(&labels) {
                scratch.pricing_stats.leaf_pair_seen += 1;
                continue;
            }
            let column = scratch.builder.build_unchecked(labels, ctx.trees);
            // Repair drops leaves, so the score changes — re-score exactly and
            // re-check that the repaired column is still improving.
            let score = column.pricing_score(ctx.alpha, ctx.beta);
            if score <= 1.0 + PRICING_EPS {
                scratch.pricing_stats.leaf_pair_repair_nonprofitable += 1;
                if from_closure {
                    scratch.pricing_stats.must_closure_valid_nonprofitable += 1;
                }
                continue;
            }
            // Guard: the repaired column must satisfy every branch constraint.
            debug_assert!(!ctx.branchings.forbids(&column));
            if ctx.branchings.forbids(&column) {
                let violations = branch_violation_kinds(column.labels(), ctx.branchings);
                scratch.pricing_stats.leaf_pair_branch_blocked += 1;
                if violations.0 {
                    scratch.pricing_stats.leaf_pair_blocked_must += 1;
                }
                if violations.1 {
                    scratch.pricing_stats.leaf_pair_blocked_cannot += 1;
                }
                continue;
            }

            // Refresh cache entry with the freshly computed column.
            // The stored gap is diagnostic only. The cache's hot path
            // re-scores cached columns exactly and never uses this value to
            // certify a skip or anchor optimality.
            if cache_active {
                let gap = self.anchor_gap_diagnostic(a, b, ctx.alpha);
                let nodes_t0: Vec<u32> = column
                    .coverage()
                    .iter_per_tree()
                    .nth(0)
                    .map(|s| s.iter().map(|&x| x as u32).collect())
                    .unwrap_or_default();
                let nodes_t1: Vec<u32> = column
                    .coverage()
                    .iter_per_tree()
                    .nth(1)
                    .map(|s| s.iter().map(|&x| x as u32).collect())
                    .unwrap_or_default();
                let leaves_sorted: Vec<u32> = column.labels().to_vec();
                let cache = scratch.anchor_cache.as_mut().unwrap();
                cache.refresh(
                    la,
                    lb,
                    leaves_sorted,
                    nodes_t0,
                    nodes_t1,
                    score,
                    gap,
                    ctx.alpha,
                    &ctx.beta[0],
                    &ctx.beta[1],
                );
            }

            if from_closure {
                scratch.pricing_stats.must_closure_valid_positive += 1;
            }
            found.push((score, column));
            scratch.pricing_stats.leaf_pair_found += 1;
        }

        scratch.pricing_stats.leaf_pair_completed = completed;
        scratch.pricing_stats.leaf_pair_global_max = global_max;
        scratch.pricing_stats.leaf_pair_feasible_global_max = feasible_global_max;

        CollectResult {
            found,
            global_max,
            feasible_global_max,
            completed,
        }
    }
}

/// Which must-link-closure path produced the labels returned by
/// [`branch_feasible_labels`]. Used purely for telemetry; the emitted labels
/// are sound on every variant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MustClosureOutcome {
    /// No must-link constraints active — closure was not attempted.
    NotAttempted,
    /// Closure was a valid AF component and its repair succeeded; the returned
    /// labels are the (repaired) closure.
    Used,
    /// Closure was not a valid AF component; fell back to raw-subset repair.
    InvalidFallback,
    /// Closure was a valid AF component but its repair stripped it below two
    /// leaves; fell back to raw-subset repair.
    RepairFallback,
}

fn branch_feasible_labels(
    raw_labels: &[u32],
    branchings: &crate::solvers::bp::search::Branchings,
    alpha: &[f64],
    trees: &[Tree],
) -> (Option<Vec<u32>>, MustClosureOutcome) {
    if branchings.must_link().is_empty() {
        return (
            repair_to_valid(raw_labels, branchings, alpha),
            MustClosureOutcome::NotAttempted,
        );
    }

    let closed = must_link_closure(raw_labels, branchings);
    if closed.len() >= 2 && is_valid_af_component(&closed, trees) {
        if let Some(labels) = repair_to_valid(&closed, branchings, alpha) {
            return (Some(labels), MustClosureOutcome::Used);
        }
        // Valid AF component, but repair dropped it below two leaves. Fall back
        // to the raw DP candidate, which is valid by construction.
        return (
            repair_to_valid(raw_labels, branchings, alpha),
            MustClosureOutcome::RepairFallback,
        );
    }

    // Closure is not a valid AF component. Fall back to the old safe behavior:
    // the raw DP candidate is valid by construction, and `repair_to_valid`
    // only takes subsets of it.
    (
        repair_to_valid(raw_labels, branchings, alpha),
        MustClosureOutcome::InvalidFallback,
    )
}

fn must_link_closure(
    labels: &[u32],
    branchings: &crate::solvers::bp::search::Branchings,
) -> Vec<u32> {
    let mut out = labels.to_vec();
    if branchings.must_link().is_empty() {
        return out;
    }
    loop {
        let mut changed = false;
        for pair in branchings.must_link() {
            let has_a = out.binary_search(&pair.a).is_ok();
            let has_b = out.binary_search(&pair.b).is_ok();
            if has_a && !has_b {
                out.push(pair.b);
                changed = true;
            } else if has_b && !has_a {
                out.push(pair.a);
                changed = true;
            }
        }
        if !changed {
            break;
        }
        out.sort_unstable();
        out.dedup();
    }
    out
}

/// Drop leaves from `labels` (sorted) until the set satisfies every branch
/// constraint: no cannot-linked pair both present, no must-linked pair
/// half-present. Returns the repaired sorted label set, or `None` if it
/// shrinks below two leaves.
///
/// Soundness: any subset of a valid agreement component is itself a valid
/// agreement component (restriction preserves cross-tree isomorphism), so the
/// repaired set is always a constructible, node-valid column. When a leaf is
/// dropped for one constraint it may strand a must-partner — the loop re-scans
/// until a fixpoint, so a must-group is always wholly kept or wholly dropped.
fn repair_to_valid(
    labels: &[u32],
    branchings: &crate::solvers::bp::search::Branchings,
    alpha: &[f64],
) -> Option<Vec<u32>> {
    let cl = branchings.cannot_link();
    let ml = branchings.must_link();
    let mut out = labels.to_vec();
    if cl.is_empty() && ml.is_empty() {
        return Some(out);
    }
    loop {
        let mut drop_label: Option<u32> = None;
        for p in cl {
            if out.binary_search(&p.a).is_ok() && out.binary_search(&p.b).is_ok() {
                // Drop the lower-α endpoint — it costs the column less gain.
                drop_label = Some(if alpha[p.a as usize] <= alpha[p.b as usize] {
                    p.a
                } else {
                    p.b
                });
                break;
            }
        }
        if drop_label.is_none() {
            for p in ml {
                let ha = out.binary_search(&p.a).is_ok();
                let hb = out.binary_search(&p.b).is_ok();
                if ha != hb {
                    drop_label = Some(if ha { p.a } else { p.b });
                    break;
                }
            }
        }
        match drop_label {
            Some(d) => {
                if let Ok(pos) = out.binary_search(&d) {
                    out.remove(pos);
                }
            }
            None => break,
        }
    }
    if out.len() < 2 { None } else { Some(out) }
}

/// Return `(must_link_violated, cannot_link_violated)` for a sorted label set.
fn branch_violation_kinds(
    labels: &[u32],
    branchings: &crate::solvers::bp::search::Branchings,
) -> (bool, bool) {
    let mut must = false;
    let mut cannot = false;
    for p in branchings.must_link() {
        let ha = labels.binary_search(&p.a).is_ok();
        let hb = labels.binary_search(&p.b).is_ok();
        if ha != hb {
            must = true;
            break;
        }
    }
    for p in branchings.cannot_link() {
        if labels.binary_search(&p.a).is_ok() && labels.binary_search(&p.b).is_ok() {
            cannot = true;
            break;
        }
    }
    (must, cannot)
}

/// Result of a `collect_from_order` scan.
struct CollectResult {
    /// Emittable improving columns (constraint-valid, not already seen).
    found: Vec<(f64, AfColumn)>,
    /// Constraint-blind max of `solve_pair` over every anchor scanned. Valid
    /// as an *unconstrained* convergence certificate only when `completed`.
    global_max: f64,
    /// Branch-feasible max of `solve_pair`: the same max restricted to anchors
    /// that can co-occur in a branch-feasible column. A sound upper bound on
    /// `max_{C ∈ C(B)} score(C)` and `≤ global_max`. This is the quantity that
    /// gates `Converged` — valid as a certificate only when `completed`.
    feasible_global_max: f64,
    /// True ⇔ every pair in `order` was scanned (no trial-limit cut, no
    /// early `target` break).
    completed: bool,
}

impl Pricer for LeafPairDpPricer {
    fn name(&self) -> &'static str {
        "leaf-pair-dp"
    }

    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult {
        // --- Pricer caching: only rebuild when active set changes ---
        let changed = self.ensure_active_labels(ctx.alpha, ctx);
        if changed {
            self.rebuild_pair_tables(ctx.trees);
        }
        self.refresh_dual_tables(ctx.trees, ctx.alpha, ctx.beta);
        self.reset_memos();

        // --- Anchor cache (cached-positive reuse) ---
        // Gated by `BpConfig.use_anchor_cache`. Lazy-init; allocate label-pair storage once.
        // Indexing is by leaf label, so structural data is invariant
        // across active-label-set changes.
        if anchor_cache_enabled(scratch) && ctx.trees.len() == 2 {
            if scratch.anchor_cache.is_none() {
                scratch.anchor_cache = Some(super::anchor_cache::AnchorCache::new(self.num_leaves));
            }
            let cache = scratch.anchor_cache.as_mut().unwrap();
            if !cache.is_built() {
                cache.refresh_static(ctx.trees);
            }
        }

        // --- Branch-awareness: cannot-link partner bitsets ---
        // Generation is constraint-aware: `solve_side` masks out the side
        // anchor's cannot-partners. Clear only the bitsets dirtied last call.
        if self.cannot_sets.len() != self.num_leaves + 1 {
            self.cannot_sets = (0..=self.num_leaves)
                .map(|_| FixedBitSet::with_capacity(self.num_leaves + 1))
                .collect();
            self.has_cannot = vec![false; self.num_leaves + 1];
        }
        for &l in &self.cannot_dirty {
            self.cannot_sets[l as usize].clear();
            self.has_cannot[l as usize] = false;
        }
        self.cannot_dirty.clear();
        for pair in ctx.branchings.cannot_link() {
            let (a, b) = (pair.a as usize, pair.b as usize);
            if a <= self.num_leaves && b <= self.num_leaves {
                self.cannot_sets[a].insert(b);
                self.cannot_sets[b].insert(a);
                if !self.has_cannot[a] {
                    self.has_cannot[a] = true;
                    self.cannot_dirty.push(a as u32);
                }
                if !self.has_cannot[b] {
                    self.has_cannot[b] = true;
                    self.cannot_dirty.push(b as u32);
                }
            }
        }

        // --- Branch-feasible certification: must-link classes + conflicts ---
        // Drives `anchor_feasible`, which restricts the certification max to
        // anchors that can occur in a branch-feasible column.
        self.rebuild_branch_classes(ctx);

        let p = self.active_labels.len();
        if p < 2 || p < self.min_active_labels {
            return PricingResult::Improving;
        }

        // Collect all pairs where the tightest upper bound on RC exceeds
        // 1+ε.  The bound `pair_ub[a,b] − root_penalty(a,b)` is sound:
        // leaf gain ≤ pair_ub, node cost ≥ root_penalty (at minimum the
        // LCA nodes in every tree).  Pairs below this threshold cannot
        // yield a positive-RC column — skip them.
        let mut order: Vec<(f64, usize, usize)> = Vec::with_capacity(p * (p - 1) / 2);
        let mut max_bound: f64 = NEG_INF;
        for a in 0..p {
            for b in (a + 1)..p {
                let idx = a * p + b;
                let ub = self.pair_ub[idx];
                let rp = self.root_penalty(a, b, ctx.beta);
                let tight = ub - rp;
                if tight > max_bound {
                    max_bound = tight;
                }
                if tight <= 1.0 + PRICING_EPS {
                    continue;
                }
                let q = self.quick_proxy(a, b, ctx.alpha, ctx.beta);
                order.push((q, a, b));
            }
        }
        if order.is_empty() {
            // Every anchor pair has a tight UB `pair_ub − root_penalty ≤ 1+ε`.
            // That UB dominates `solve_pair` (§3), so no column anywhere — at
            // any anchor, constraint-valid or not — can have score > 1+ε.
            // No improving column exists: convergence is proven.
            return PricingResult::Converged;
        }
        let target = if ctx.trees.len() == 2 {
            self.max_per_call
                .min(adaptive_m2_batch_size(ctx, scratch.m2_batch))
        } else {
            self.max_per_call
        };
        let trial_limit = (self.pair_trial_limit as usize).min(order.len());

        if trial_limit < order.len() {
            order.select_nth_unstable_by(trial_limit, |l, r| {
                r.0.partial_cmp(&l.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            let (head, _) = order.split_at_mut(trial_limit);
            head.sort_unstable_by(|l, r| {
                r.0.partial_cmp(&l.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            order.sort_unstable_by(|l, r| {
                r.0.partial_cmp(&l.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let mut result = self.collect_from_order(&order, trial_limit, target, ctx, scratch);

        // §3: the trial-limited pass is a fast *generation* shortcut. If it
        // emitted nothing, the full all-anchor scan must run before pricing
        // may certify (`Converged`) or rule out (`Improving`) convergence.
        if result.found.is_empty() && self.fallback_full_when_empty && trial_limit < order.len() {
            order.sort_unstable_by(|l, r| {
                r.0.partial_cmp(&l.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            result = self.collect_from_order(&order, order.len(), target, ctx, scratch);
        }

        // Soundness invariant: the branch-feasible certification max never
        // exceeds the constraint-blind max (it is a sub-max over fewer anchors).
        // A violation would mean `feasible_global_max` saw an anchor the
        // constraint-blind max did not — a bug that could over-certify.
        debug_assert!(
            result.feasible_global_max <= result.global_max + PRICING_EPS,
            "feasible_global_max {} exceeds global_max {}",
            result.feasible_global_max,
            result.global_max
        );

        // Optional anchor-cache stats logging.
        if anchor_cache_enabled(scratch)
            && let Some(cache) = scratch.anchor_cache.as_ref()
        {
            let avg_gap = if cache.gap_positive_refreshes > 0 {
                cache.gap_sum / cache.gap_positive_refreshes as f64
            } else {
                0.0
            };
            log::debug!(
                target: "klados::bp",
                "anchor-cache: hits={} skips={} stales={} misses={} refreshes={} gap_pos={} gap_zero={} avg_pos_gap={:.4}",
                cache.hits, cache.skips, cache.stales, cache.misses,
                cache.refreshes, cache.gap_positive_refreshes,
                cache.gap_zero_refreshes, avg_gap,
            );
        }

        if !result.found.is_empty() {
            if ctx.branchings.depth() > 0 && scratch.pricing_stats.leaf_pair_branch_blocked > 0 {
                let s = &scratch.pricing_stats;
                log::debug!(
                    target: "klados::bp",
                    "leaf-pair branch stats depth={} outcome=found scanned={} positive={} found={} blocked={} must={} cannot={} seen={} repair_failed={} repair_nonprofit={} mc_attempted={} mc_valid_pos={} mc_valid_nonprofit={} mc_invalid={} mc_fallback={} completed={} trial_limited={} global_max={:.4} feasible_global_max={:.4}",
                    ctx.branchings.depth(),
                    s.leaf_pair_scanned,
                    s.leaf_pair_positive,
                    s.leaf_pair_found,
                    s.leaf_pair_branch_blocked,
                    s.leaf_pair_blocked_must,
                    s.leaf_pair_blocked_cannot,
                    s.leaf_pair_seen,
                    s.leaf_pair_repair_failed,
                    s.leaf_pair_repair_nonprofitable,
                    s.must_closure_attempted,
                    s.must_closure_valid_positive,
                    s.must_closure_valid_nonprofitable,
                    s.must_closure_invalid,
                    s.must_closure_fallback,
                    s.leaf_pair_completed,
                    s.leaf_pair_trial_limited,
                    s.leaf_pair_global_max,
                    s.leaf_pair_feasible_global_max,
                );
            }
            // Sort by RC descending, cap at 128 to prevent RMP flooding
            // while still returning far more columns than the old limit of 32.
            let mut found = result.found;
            found.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let cap = 128usize.min(found.len());
            return PricingResult::Found(found.into_iter().take(cap).map(|(_, c)| c).collect());
        }

        // No emittable column. Decide between a proven `Converged` and an
        // uncertified `Improving`.
        //
        // Certification gates on `feasible_global_max`, the max of `solve_pair`
        // over anchors that can occur in a branch-feasible column. This is a
        // sound upper bound on `max_{C ∈ C(B)} score(C)`:
        //
        //   * `solve_pair(a,b) ≥ score(C)` for every column `C ∋ a,b` (§3), so
        //     the bound dominates every branch-feasible column;
        //   * every branch-feasible `C` with `|C| ≥ 2` has a pair `(a,b) ⊆ C`
        //     whose classes lie wholly in `C` (must-closure) and so cannot be
        //     cannot-conflicting — hence `anchor_feasible(a,b)` holds, the pair
        //     was scanned (its tight UB exceeds 1+ε since `solve_pair ≥
        //     score(C) > 1+ε`), and it contributed to `feasible_global_max`.
        //
        // Therefore `completed ∧ feasible_global_max ≤ 1+ε` ⇒ no improving
        // branch-feasible column exists ⇒ the branch-node LP bound is certified.
        // `feasible_global_max ≤ global_max`, so this subsumes the old
        // constraint-blind gate and only ever certifies more (never wrongly).
        if result.completed && result.feasible_global_max <= 1.0 + PRICING_EPS {
            // The full all-anchor scan completed and the branch-feasible max
            // anchor score is ≤ 1+ε: no branch-feasible improving column exists.
            if ctx.branchings.depth() > 0 && scratch.pricing_stats.leaf_pair_positive > 0 {
                let s = &scratch.pricing_stats;
                log::debug!(
                    target: "klados::bp",
                    "leaf-pair branch stats depth={} outcome=converged scanned={} positive={} found={} blocked={} must={} cannot={} seen={} repair_failed={} repair_nonprofit={} mc_attempted={} mc_valid_pos={} mc_valid_nonprofit={} mc_invalid={} mc_fallback={} completed={} trial_limited={} global_max={:.4} feasible_global_max={:.4}",
                    ctx.branchings.depth(),
                    s.leaf_pair_scanned,
                    s.leaf_pair_positive,
                    s.leaf_pair_found,
                    s.leaf_pair_branch_blocked,
                    s.leaf_pair_blocked_must,
                    s.leaf_pair_blocked_cannot,
                    s.leaf_pair_seen,
                    s.leaf_pair_repair_failed,
                    s.leaf_pair_repair_nonprofitable,
                    s.must_closure_attempted,
                    s.must_closure_valid_positive,
                    s.must_closure_valid_nonprofitable,
                    s.must_closure_invalid,
                    s.must_closure_fallback,
                    s.leaf_pair_completed,
                    s.leaf_pair_trial_limited,
                    s.leaf_pair_global_max,
                    s.leaf_pair_feasible_global_max,
                );
            }
            PricingResult::Converged
        } else {
            // The constraint-blind/anchor-level bound did not certify. If the
            // scan completed and must-link constraints are active, try the
            // Lagrangian must-link relaxation: it can prove convergence on
            // nodes whose residual is must-link-blocked (the dominant case),
            // and is sound for any multiplier (`μ = 0` reproduces the bound
            // just computed). Gated to completed scans because the relaxed
            // bound, like the base one, only certifies after a full scan.
            if self.lagrangian_certify_enabled
                && result.completed
                && ctx.branchings.depth() > 0
                && !ctx.branchings.must_link().is_empty()
                && self.lagrangian_must_certify(ctx)
            {
                log::debug!(
                    target: "klados::bp",
                    "leaf-pair branch stats depth={} outcome=converged-lagrangian global_max={:.4} feasible_global_max={:.4}",
                    ctx.branchings.depth(),
                    result.global_max,
                    result.feasible_global_max,
                );
                return PricingResult::Converged;
            }
            // Either the scan was incomplete (trial-limited, no fallback), or
            // improving columns provably exist but every one is branch-blocked
            // or already pooled. The LP bound is NOT certified — the solver
            // must branch, never bound-prune.
            if ctx.branchings.depth() > 0 {
                let s = &scratch.pricing_stats;
                log::debug!(
                    target: "klados::bp",
                    "leaf-pair branch stats depth={} active={} outcome=improving scanned={} positive={} found={} blocked={} must={} cannot={} seen={} repair_failed={} repair_nonprofit={} mc_attempted={} mc_valid_pos={} mc_valid_nonprofit={} mc_invalid={} mc_fallback={} completed={} trial_limited={} global_max={:.4} feasible_global_max={:.4}",
                    ctx.branchings.depth(),
                    self.active_labels.len(),
                    s.leaf_pair_scanned,
                    s.leaf_pair_positive,
                    s.leaf_pair_found,
                    s.leaf_pair_branch_blocked,
                    s.leaf_pair_blocked_must,
                    s.leaf_pair_blocked_cannot,
                    s.leaf_pair_seen,
                    s.leaf_pair_repair_failed,
                    s.leaf_pair_repair_nonprofitable,
                    s.must_closure_attempted,
                    s.must_closure_valid_positive,
                    s.must_closure_valid_nonprofitable,
                    s.must_closure_invalid,
                    s.must_closure_fallback,
                    s.leaf_pair_completed,
                    s.leaf_pair_trial_limited,
                    s.leaf_pair_global_max,
                    s.leaf_pair_feasible_global_max,
                );
            }
            PricingResult::Improving
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solvers::bp::search::Branchings;
    use crate::solvers::bp::search::branchings::LeafPair;
    use klados_core::tree::{Label, NONE, NodeId};

    fn parse(nw: &str, n: u32) -> Tree {
        let mut t = Tree::with_capacity(n);
        let b = nw.as_bytes();
        let mut pos = 0usize;
        fn rec(b: &[u8], pos: &mut usize, t: &mut Tree) -> NodeId {
            if b[*pos] == b'(' {
                *pos += 1;
                let l = rec(b, pos, t);
                assert_eq!(b[*pos], b',');
                *pos += 1;
                let r = rec(b, pos, t);
                assert_eq!(b[*pos], b')');
                *pos += 1;
                let id = t.parent.len() as NodeId;
                t.parent.push(NONE);
                t.left.push(l);
                t.right.push(r);
                t.label.push(0);
                t.parent[l as usize] = id;
                t.parent[r as usize] = id;
                id
            } else {
                let start = *pos;
                while *pos < b.len() && b[*pos].is_ascii_digit() {
                    *pos += 1;
                }
                let lbl: u32 = std::str::from_utf8(&b[start..*pos])
                    .unwrap()
                    .parse()
                    .unwrap();
                let id = t.parent.len() as NodeId;
                t.parent.push(NONE);
                t.left.push(NONE);
                t.right.push(NONE);
                t.label.push(lbl as Label);
                t.label_to_node[lbl as usize] = id;
                id
            }
        }
        t.root = rec(b, &mut pos, &mut t);
        t.compute_metadata();
        t
    }

    #[test]
    fn must_link_closure_is_transitive() {
        let mut b = Branchings::default();
        b.push_must_link(LeafPair::new(1, 2));
        b.push_must_link(LeafPair::new(2, 3));

        assert_eq!(must_link_closure(&[1], &b), vec![1, 2, 3]);
    }

    #[test]
    fn branch_feasible_labels_prefers_valid_must_closure() {
        let mut b = Branchings::default();
        b.push_must_link(LeafPair::new(1, 2));
        let trees = vec![parse("((1,2),3)", 3), parse("((1,2),3)", 3)];
        let alpha = vec![0.0, 1.0, 1.0, 1.0];

        let (labels, outcome) = branch_feasible_labels(&[1, 3], &b, &alpha, &trees);
        assert_eq!(labels, Some(vec![1, 2, 3]));
        assert_eq!(outcome, MustClosureOutcome::Used);
    }

    #[test]
    fn repair_drops_whole_must_class_after_cannot_conflict() {
        let mut b = Branchings::default();
        b.push_must_link(LeafPair::new(1, 2));
        b.push_cannot_link(LeafPair::new(2, 3));
        let alpha = vec![0.0, 1.0, 1.0, 10.0];

        assert_eq!(repair_to_valid(&[1, 2, 3], &b, &alpha), None);
    }

    /// `anchor_feasible` lifts cannot-link to whole must-link classes: an anchor
    /// is infeasible if the two leaves' classes are cannot-conflicting, even
    /// when the two anchor leaves are not themselves directly cannot-linked.
    #[test]
    fn anchor_feasibility_lifts_cannot_link_to_must_classes() {
        use crate::solvers::bp::column::ColumnSet;
        let trees = vec![parse("((1,2),(3,4))", 4), parse("((1,2),(3,4))", 4)];
        let alpha = vec![0.0, 1.0, 1.0, 1.0, 1.0];
        let beta: Vec<Vec<f64>> = trees.iter().map(|t| vec![0.0; t.num_nodes()]).collect();
        let mut br = Branchings::default();
        br.push_must_link(LeafPair::new(1, 2)); // class {1, 2}
        br.push_cannot_link(LeafPair::new(2, 3)); // {1, 2} conflicts with {3}
        let columns: Vec<AfColumn> = Vec::new();
        let seen = ColumnSet::new();
        let ctx = PricingContext {
            trees: &trees,
            num_leaves: 4,
            alpha: &alpha,
            beta: &beta,
            columns: &columns,
            seen: &seen,
            branchings: &br,
            terminate: &super::super::NEVER_TERMINATE,
            deadline: None,
        };
        let mut pricer = LeafPairDpPricer::new(&trees);
        pricer.rebuild_branch_classes(&ctx);

        // (1,3): classes {1,2} and {3} conflict via cannot(2,3) — infeasible
        // even though 1 and 3 are not directly cannot-linked.
        assert!(!pricer.anchor_feasible(1, 3));
        // (2,3): direct cannot-link — infeasible.
        assert!(!pricer.anchor_feasible(2, 3));
        // (1,2): same must-link class — always feasible.
        assert!(pricer.anchor_feasible(1, 2));
        // No conflict — feasible.
        assert!(pricer.anchor_feasible(1, 4));
        assert!(pricer.anchor_feasible(3, 4));
    }

    /// The Lagrangian must-link relaxation certifies a node that the
    /// constraint-blind / anchor-level bound cannot. Construction: must-link
    /// `(2,3)`, with `α = [_, 1, 1, −5]` and `β ≡ 0` on two copies of
    /// `((1,2),3)`. The must-**violating** column `{1,2}` scores `2.0` at the
    /// feasible anchor `(1,2)`, so the base bound is `2.0 > 1+ε` → `Improving`.
    /// But every must-closed feasible column (`{2,3}`, `{1,3}`, `{1,2,3}`)
    /// scores `≤ −3`, so the node is genuinely converged. Relaxing `1_2 = 1_3`
    /// with `μ → −1` shifts `α₂ → 0`, pulling the relaxed bound to `1.0 ≤ 1+ε`
    /// → `Converged`.
    #[test]
    fn lagrangian_certifies_must_blocked_node_base_bound_misses() {
        use crate::solvers::bp::column::ColumnSet;
        let trees = vec![parse("((1,2),3)", 3), parse("((1,2),3)", 3)];
        let alpha = vec![0.0, 1.0, 1.0, -5.0];
        let beta: Vec<Vec<f64>> = trees.iter().map(|t| vec![0.0; t.num_nodes()]).collect();
        let mut br = Branchings::default();
        br.push_must_link(LeafPair::new(2, 3));
        let columns: Vec<AfColumn> = Vec::new();
        let seen = ColumnSet::new();
        let ctx = PricingContext {
            trees: &trees,
            num_leaves: 3,
            alpha: &alpha,
            beta: &beta,
            columns: &columns,
            seen: &seen,
            branchings: &br,
            terminate: &super::super::NEVER_TERMINATE,
            deadline: None,
        };

        // Lagrangian OFF: base bound sees `{1,2}` at a feasible anchor → cannot
        // certify → Improving.
        let mut off = LeafPairDpPricer::new(&trees);
        let mut sc = PricerScratch::new(&trees);
        assert!(matches!(off.price(&ctx, &mut sc), PricingResult::Improving));

        // Lagrangian ON: relaxing must(2,3) drives the bound to ≤ 1+ε.
        let mut on = LeafPairDpPricer::new(&trees).with_lagrangian_certify(6);
        let mut sc2 = PricerScratch::new(&trees);
        assert!(matches!(on.price(&ctx, &mut sc2), PricingResult::Converged));
    }

    /// Exact-track soundness guard for branch-feasible certification.
    ///
    /// Over many random small instances, dual vectors, and consistent branch
    /// states, brute-force the true branch-feasible column space `C(B)` and its
    /// maximum pricing score, then run the pricer's full scan. Two properties
    /// must hold:
    ///
    /// 1. **No over-certification.** If a branch-feasible improving column
    ///    exists (`brute_max > 1+ε`), the pricer must never return `Converged`.
    ///    This is the load-bearing exactness check: a violation would mean the
    ///    LP bound is wrongly certified and the search could prune an optimum.
    /// 2. **Bound dominance.** On a completed scan, `feasible_global_max`
    ///    upper-bounds the true branch-feasible max.
    #[test]
    fn certification_never_over_certifies_vs_brute_force() {
        use crate::solvers::bp::column::{ColumnBuilder, ColumnSet};

        // Deterministic LCG so failures are reproducible from the seed.
        fn next(rng: &mut u64) -> u64 {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *rng >> 33
        }
        fn rrange(rng: &mut u64, lo: u64, hi: u64) -> u64 {
            lo + next(rng) % (hi - lo)
        }
        fn random_tree(labels: &[u32], n_cap: u32, rng: &mut u64) -> Tree {
            let mut t = Tree::with_capacity(n_cap);
            let mut pool: Vec<NodeId> = Vec::new();
            for &lbl in labels {
                let id = t.parent.len() as NodeId;
                t.parent.push(NONE);
                t.left.push(NONE);
                t.right.push(NONE);
                t.label.push(lbl as Label);
                t.label_to_node[lbl as usize] = id;
                pool.push(id);
            }
            while pool.len() > 1 {
                let i = (next(rng) % pool.len() as u64) as usize;
                let a = pool.swap_remove(i);
                let j = (next(rng) % pool.len() as u64) as usize;
                let b = pool.swap_remove(j);
                let id = t.parent.len() as NodeId;
                t.parent.push(NONE);
                t.left.push(a);
                t.right.push(b);
                t.label.push(0);
                t.parent[a as usize] = id;
                t.parent[b as usize] = id;
                pool.push(id);
            }
            t.root = pool[0];
            t.compute_metadata();
            t
        }

        const EPS: f64 = PRICING_EPS;
        let mut saw_converged = 0usize;
        let mut saw_feasible_improving = 0usize;

        for seed in 0..500u64 {
            let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            let n = rrange(&mut rng, 4, 8) as u32; // 4..=7 leaves
            let m = rrange(&mut rng, 2, 4) as usize; // 2 or 3 trees
            let labels: Vec<u32> = (1..=n).collect();
            let trees: Vec<Tree> = (0..m).map(|_| random_tree(&labels, n, &mut rng)).collect();

            // Half the seeds use heavy penalties / small gains to exercise the
            // `Converged` path; the rest keep many improving columns.
            let heavy = seed % 2 == 0;
            let mut alpha = vec![0.0f64; n as usize + 1];
            for a in alpha.iter_mut().take(n as usize + 1).skip(1) {
                *a = if heavy {
                    0.3 + (rrange(&mut rng, 0, 60) as f64) / 100.0
                } else {
                    0.8 + (rrange(&mut rng, 0, 200) as f64) / 100.0
                };
            }
            let beta: Vec<Vec<f64>> = trees
                .iter()
                .map(|t| {
                    (0..t.num_nodes())
                        .map(|v| {
                            if t.is_leaf(v as NodeId) {
                                0.0
                            } else {
                                let scale = if heavy { 120 } else { 40 };
                                (rrange(&mut rng, 0, scale) as f64) / 100.0
                            }
                        })
                        .collect()
                })
                .collect();

            // Random consistent branchings.
            let mut br = Branchings::default();
            let rand_pair = |rng: &mut u64| -> LeafPair {
                loop {
                    let a = rrange(rng, 1, n as u64 + 1) as u32;
                    let b = rrange(rng, 1, n as u64 + 1) as u32;
                    if a != b {
                        return LeafPair::new(a, b);
                    }
                }
            };
            for _ in 0..rrange(&mut rng, 0, 3) {
                br.push_must_link(rand_pair(&mut rng));
            }
            for _ in 0..rrange(&mut rng, 0, 3) {
                let p = rand_pair(&mut rng);
                if !br.must_link().contains(&p) {
                    br.push_cannot_link(p);
                }
            }

            // Brute-force the branch-feasible max pricing score.
            let mut builder = ColumnBuilder::new(&trees);
            let feasible = |s: &[u32]| -> bool {
                for ml in br.must_link() {
                    if s.binary_search(&ml.a).is_ok() != s.binary_search(&ml.b).is_ok() {
                        return false;
                    }
                }
                for cl in br.cannot_link() {
                    if s.binary_search(&cl.a).is_ok() && s.binary_search(&cl.b).is_ok() {
                        return false;
                    }
                }
                true
            };
            let mut brute_max = f64::NEG_INFINITY;
            for mask in 0u32..(1u32 << n) {
                let s: Vec<u32> = (1..=n).filter(|&l| mask & (1 << (l - 1)) != 0).collect();
                if s.len() < 2 || !feasible(&s) || !is_valid_af_component(&s, &trees) {
                    continue;
                }
                let col = builder.build_unchecked(s, &trees);
                brute_max = brute_max.max(col.pricing_score(&alpha, &beta));
            }

            // Full-scan pricing with Lagrangian must-link certification on —
            // this is the path that could over-certify if the relaxation were
            // unsound, so the brute guard below must exercise it.
            let mut pricer = LeafPairDpPricer::new(&trees).with_lagrangian_certify(6);
            let mut scratch = PricerScratch::new(&trees);
            let seen = ColumnSet::new();
            let columns: Vec<AfColumn> = Vec::new();
            let ctx = PricingContext {
                trees: &trees,
                num_leaves: n as usize,
                alpha: &alpha,
                beta: &beta,
                columns: &columns,
                seen: &seen,
                branchings: &br,
                terminate: &super::super::NEVER_TERMINATE,
                deadline: None,
            };
            let result = pricer.price(&ctx, &mut scratch);
            let fgm = scratch.pricing_stats.leaf_pair_feasible_global_max;
            let gm = scratch.pricing_stats.leaf_pair_global_max;
            let completed = scratch.pricing_stats.leaf_pair_completed;
            let is_converged = matches!(result, PricingResult::Converged);
            if is_converged {
                saw_converged += 1;
            }

            assert!(
                fgm <= gm + EPS,
                "seed {seed}: feasible_max {fgm} > global_max {gm}"
            );

            if brute_max > 1.0 + EPS {
                saw_feasible_improving += 1;
                // (1) No over-certification.
                assert!(
                    !is_converged,
                    "seed {seed}: over-certified Converged but a branch-feasible \
                     improving column exists (brute_max={brute_max})"
                );
                // (2) Bound dominance on a completed scan.
                if completed {
                    assert!(
                        fgm + 1e-6 >= brute_max,
                        "seed {seed}: feasible_global_max {fgm} < brute feasible max {brute_max}"
                    );
                }
            }
        }

        // Sanity: the corpus actually exercised both regimes.
        assert!(saw_converged > 0, "no Converged outcomes — corpus too easy");
        assert!(
            saw_feasible_improving > 0,
            "no feasible-improving cases — corpus too hard"
        );
    }
}
