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

use crate::solvers::bp::column::AfColumn;

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

    // ── Clean-cut "sided" DP (only allocated/used when a cut is active; the
    //    cut-absent production path never touches any of this) ──
    /// `leaf_pat[active_idx]` = 1 if the active leaf is in `C`, 2 if in `Cᶜ`.
    /// Built per call from the cut. Empty when no cut.
    leaf_pat: Vec<u8>,
    /// Best score per touch-pattern, indexed `[pair_idx][pattern]` where pattern
    /// is a 2-bit set (bit0 = touches C, bit1 = touches Cᶜ): 1=C-only, 2=Cᶜ-only,
    /// 3=straddle. Index 0 is unused.
    memo_pair_sided: Vec<[f64; 4]>,
    memo_side_sided: Vec<[f64; 4]>,
    /// Reconstruction: per pattern, the side's chosen extension (`SPLIT_LEAF_ONLY`
    /// or an active index).
    memo_side_split_sided: Vec<[u32; 4]>,
    /// Reconstruction: per pattern, the `(left_pattern, right_pattern)` split.
    memo_pair_split_sided: Vec<[(u8, u8); 4]>,
    /// Cycle-break / memo state: 0 = unvisited, 1 = in-progress, 2 = done.
    sided_pair_state: Vec<u8>,
    sided_side_state: Vec<u8>,
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
            leaf_pat: Vec::new(),
            memo_pair_sided: Vec::new(),
            memo_side_sided: Vec::new(),
            memo_side_split_sided: Vec::new(),
            memo_pair_split_sided: Vec::new(),
            sided_pair_state: Vec::new(),
            sided_side_state: Vec::new(),
        }
    }

    /// Collect active labels from alpha. Returns `true` if the set changed
    /// since the last call.
    fn ensure_active_labels(&mut self, alpha: &[f64], ctx: &PricingContext) -> bool {
        let prev: Vec<u32> = std::mem::take(&mut self.active_labels);
        self.active_mask.clear();
        for v in self.label_to_active_idx.iter_mut() {
            *v = u32::MAX;
        }
        // Clean-cut side restriction: when set, a label is eligible only if it
        // lies in the restricted side. This confines generated columns to one
        // side (the no-straddle solve). `None` ⇒ every label eligible ⇒ exact
        // current behavior.
        let side = ctx.restrict_side;
        let eligible = |label: u32| side.is_none_or(|s| s.contains(label as usize));
        let activate = |me: &mut Self, label: u32| {
            if me.active_mask.contains(label as usize) {
                return;
            }
            me.active_mask.insert(label as usize);
            me.label_to_active_idx[label as usize] = me.active_labels.len() as u32;
            me.active_labels.push(label);
        };
        for label in 1..=self.num_leaves as u32 {
            // Without rank rows, the usual α>0 filter is sound: any improving
            // column can drop non-positive-α leaves without reducing score.
            //
            // With clean-cut rank rows, a non-positive-α leaf can still be
            // useful because it may be the only leaf that earns a positive
            // side-touch dual γ.  Therefore, while a γ-side is active, keep
            // every eligible leaf on that side in the DP universe.  This is
            // the load-bearing fix that makes the sided pricer a real
            // certifier instead of an α-filtered generator.
            let rank_touch_active = ctx.clean_cut.is_some_and(|cut| {
                if cut.side_c.contains(label as usize) {
                    cut.gamma_c > 1.0e-12
                } else {
                    cut.gamma_cc > 1.0e-12
                }
            });
            if (alpha[label as usize] > 1.0e-12 || rank_touch_active) && eligible(label) {
                activate(self, label);
            }
        }
        // Must-linked leaves stay active even with alpha <= 0: a must-link
        // constraint can force such a leaf into an improving column. Excluding
        // it would make the pricer unable to build that column and falsely
        // report convergence — an unsound LP bound. Under a side restriction a
        // must-link partner outside the side cannot appear in any (single-side)
        // column, so it stays inactive — consistent with the restricted space.
        for pair in ctx.branchings.must_link() {
            if (pair.a as usize) <= self.num_leaves && eligible(pair.a) {
                activate(self, pair.a);
            }
            if (pair.b as usize) <= self.num_leaves && eligible(pair.b) {
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

    // ───────────────────────── Sided DP (clean-cut) ─────────────────────────
    //
    // A parallel recurrence that tracks, per pair/side, the best score for each
    // touch-pattern p ∈ {1=C-only, 2=Cᶜ-only, 3=straddle}. This lets pricing
    // optimize the EFFECTIVE reduced cost `1 − score − bonus(p)` exactly, where
    // `bonus(p) = γ_C·[p touches C] + γ_Cᶜ·[p touches Cᶜ]`. Only invoked when a
    // clean cut is active; the scalar path above is byte-for-byte unchanged.
    //
    // The recurrence mirrors `solve_pair`/`solve_side` exactly (same column
    // space, same penalties), so `max_p sided[p]` equals the scalar
    // `solve_pair`. Candidate pruning is intentionally dropped here (the scalar
    // dynamic `best_score` filter is unsound per-pattern), so it enumerates the
    // full candidate set — correctness over speed for the gated path.

    /// Prepare the sided tables for a call with cut side `C` (`side_c` is
    /// 1-indexed membership). Builds `leaf_pat` and resets the sided memos.
    fn setup_sided(&mut self, side_c: &FixedBitSet) {
        let p = self.active_labels.len();
        self.leaf_pat.clear();
        self.leaf_pat.resize(p, 0);
        for a in 0..p {
            let label = self.active_labels[a] as usize;
            self.leaf_pat[a] = if side_c.contains(label) { 1 } else { 2 };
        }
        let pc = p * p;
        self.memo_pair_sided.clear();
        self.memo_pair_sided.resize(pc, [NEG_INF; 4]);
        self.memo_side_sided.clear();
        self.memo_side_sided.resize(pc, [NEG_INF; 4]);
        self.memo_side_split_sided.clear();
        self.memo_side_split_sided.resize(pc, [SPLIT_LEAF_ONLY; 4]);
        self.memo_pair_split_sided.clear();
        self.memo_pair_split_sided.resize(pc, [(0u8, 0u8); 4]);
        self.sided_pair_state.clear();
        self.sided_pair_state.resize(pc, 0);
        self.sided_side_state.clear();
        self.sided_side_state.resize(pc, 0);
    }

    fn solve_pair_sided(&mut self, a: usize, b: usize, alpha: &[f64], beta: &[Vec<f64>]) -> [f64; 4] {
        debug_assert!(a != b);
        let idx = self.pair_idx(a, b);
        match self.sided_pair_state[idx] {
            2 => return self.memo_pair_sided[idx],
            1 => return [NEG_INF; 4], // recursion cycle — this path can't complete
            _ => {}
        }
        self.sided_pair_state[idx] = 1;
        let left = self.solve_side_sided(a, b, alpha, beta);
        let right = self.solve_side_sided(b, a, alpha, beta);
        let rp = self.root_penalty(a, b, beta);
        let mut res = [NEG_INF; 4];
        let mut split = [(0u8, 0u8); 4];
        for pl in 1..4usize {
            if left[pl] <= NEG_INF / 2.0 {
                continue;
            }
            for pr in 1..4usize {
                if right[pr] <= NEG_INF / 2.0 {
                    continue;
                }
                let p = pl | pr; // union of touched sides
                let val = -rp + left[pl] + right[pr];
                if val > res[p] {
                    res[p] = val;
                    split[p] = (pl as u8, pr as u8);
                }
            }
        }
        self.memo_pair_sided[idx] = res;
        self.memo_pair_split_sided[idx] = split;
        self.sided_pair_state[idx] = 2;
        res
    }

    fn solve_side_sided(&mut self, a: usize, b: usize, alpha: &[f64], beta: &[Vec<f64>]) -> [f64; 4] {
        debug_assert!(a != b);
        let idx = self.pair_idx(a, b);
        match self.sided_side_state[idx] {
            2 => return self.memo_side_sided[idx],
            1 => return [NEG_INF; 4],
            _ => {}
        }
        self.sided_side_state[idx] = 1;

        let p = self.active_labels.len();
        let la = self.active_labels[a] as usize;
        let pat_a = self.leaf_pat[a] as usize;
        let mut res = [NEG_INF; 4];
        let mut split = [SPLIT_LEAF_ONLY; 4];
        // Leaf-only option: the side is just `a`, pattern = a's own side.
        res[pat_a] = alpha[la] - self.pair_singleton_penalty[idx];

        let b_penalty_sum: f64 = (0..self.num_trees)
            .map(|ti| self.pair_side_parent_prefix_beta[ti][idx])
            .sum();
        let candidates = self.collect_sided_candidates(a, b);
        for c in candidates {
            let idx_c = a * p + c;
            let pen = self.pair_penalty[idx_c] - b_penalty_sum;
            let child = self.solve_pair_sided(a, c, alpha, beta);
            for q in 1..4usize {
                if child[q] <= NEG_INF / 2.0 {
                    continue;
                }
                let cand = child[q] - pen;
                if cand > res[q] {
                    res[q] = cand;
                    split[q] = c as u32;
                }
            }
        }
        self.memo_side_sided[idx] = res;
        self.memo_side_split_sided[idx] = split;
        self.sided_side_state[idx] = 2;
        res
    }

    /// Candidate extensions for `solve_side_sided(a, b)`: descendants of the
    /// a-side child of LCA(a,b) in EVERY tree, intersected with the active mask,
    /// excluding `a`, `b`, and `a`'s cannot-partners. Mirrors the scalar phase-1
    /// enumeration (without the per-pattern-unsound dynamic score filter).
    fn collect_sided_candidates(&self, a: usize, b: usize) -> Vec<usize> {
        const BLOCK_BITS: usize = std::mem::size_of::<usize>() * 8;
        const BLOCK_SHIFT: usize = BLOCK_BITS.trailing_zeros() as usize;
        const BLOCK_MASK: usize = BLOCK_BITS - 1;
        let idx = self.pair_idx(a, b);
        let la = self.active_labels[a] as usize;
        let lb = self.active_labels[b] as usize;
        let la_w = la >> BLOCK_SHIFT;
        let la_m = 1usize << (la & BLOCK_MASK);
        let lb_w = lb >> BLOCK_SHIFT;
        let lb_m = 1usize << (lb & BLOCK_MASK);
        let cannot_a_blocks: &[usize] = if self.has_cannot[la] {
            self.cannot_sets[la].as_slice()
        } else {
            &[]
        };
        let active_mask_slice = self.active_mask.as_slice();
        let mut out = Vec::new();
        let n_blocks = self.descendant_leaves[0][self.pair_side_child[0][idx] as usize]
            .as_slice()
            .len();
        for wi in 0..n_blocks {
            let mut w = active_mask_slice[wi];
            for ti in 0..self.num_trees {
                let node = self.pair_side_child[ti][idx] as usize;
                w &= self.descendant_leaves[ti][node].as_slice()[wi];
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
                out.push(self.label_to_active_idx[c_label] as usize);
            }
        }
        out
    }

    /// Reconstruct the leaf set of the best column at pair `(a, b)` with touch
    /// pattern `pat`. Mirrors `collect_pair`/`collect_side` along the per-pattern
    /// split choices recorded by the sided DP.
    fn collect_pair_sided(&self, a: usize, b: usize, pat: usize, out: &mut Vec<u32>) {
        let (pl, pr) = self.memo_pair_split_sided[self.pair_idx(a, b)][pat];
        self.collect_side_sided(a, b, pl as usize, out);
        self.collect_side_sided(b, a, pr as usize, out);
    }

    fn collect_side_sided(&self, a: usize, b: usize, pat: usize, out: &mut Vec<u32>) {
        let idx = self.pair_idx(a, b);
        let choice = self.memo_side_split_sided[idx][pat];
        if choice == SPLIT_LEAF_ONLY {
            out.push(self.active_labels[a]);
        } else {
            self.collect_pair_sided(a, choice as usize, pat, out);
        }
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

        let cache_active =
            anchor_cache_enabled(scratch) && ctx.trees.len() == 2 && scratch.anchor_cache.is_some();

        let scan = order.len().min(trial_limit);
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
            if score <= 1.0 + PRICING_EPS {
                continue;
            }
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
                continue;
            }
            let labels = match repair_to_valid(&raw_labels, ctx.branchings, ctx.alpha) {
                Some(l) => l,
                None => continue,
            };
            if ctx.seen.contains(&labels) {
                continue;
            }
            let column = scratch.builder.build_unchecked(labels, ctx.trees);
            // Repair drops leaves, so the score changes — re-score exactly and
            // re-check that the repaired column is still improving.
            let score = column.pricing_score(ctx.alpha, ctx.beta);
            if score <= 1.0 + PRICING_EPS {
                continue;
            }
            // Guard: the repaired column must satisfy every branch constraint.
            debug_assert!(!ctx.branchings.forbids(&column));
            if ctx.branchings.forbids(&column) {
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

            found.push((score, column));
        }

        CollectResult {
            found,
            global_max,
            completed,
        }
    }

    /// Sided counterpart of [`Self::collect_from_order`]: certifies/generates
    /// against the EFFECTIVE reduced cost under the clean-cut rank rows. For each
    /// anchor it evaluates the per-pattern best `solve_pair_sided` and adds the
    /// pattern's `bonus`; `global_max` is the max effective score (the sound
    /// `Converged` certificate when `≤ 1+ε` and the scan completed). Emission
    /// reconstructs the winning pattern's column, repairs it, and re-checks the
    /// EXACT effective reduced cost (`pricing_score + bonus(actual sides)`).
    fn collect_from_order_sided(
        &mut self,
        order: &[(f64, usize, usize)],
        trial_limit: usize,
        target: usize,
        ctx: &PricingContext,
        scratch: &mut PricerScratch,
    ) -> CollectResult {
        let cut = ctx.clean_cut.expect("collect_from_order_sided requires a cut");
        let side_c = cut.side_c;
        let bonus = [0.0, cut.gamma_c, cut.gamma_cc, cut.gamma_c + cut.gamma_cc];
        // Skip-widening uses max(0,γ) per side (sound upper bound on the bonus);
        // the per-pattern `bonus` above uses the actual duals for exact scoring.
        let gamma_total = cut.gamma_c.max(0.0) + cut.gamma_cc.max(0.0);
        let tau = 1.0 + PRICING_EPS;
        let p = self.active_labels.len();

        let mut found: Vec<(f64, AfColumn)> = Vec::new();
        let mut global_max: f64 = NEG_INF;
        let repair_cert = std::env::var("KLADOS_BP_REPAIR_CERT").as_deref() != Ok("0");
        let scan = order.len().min(trial_limit);
        let mut completed = scan == order.len();

        let mut labels_buf: Vec<u32> = Vec::new();
        for &(_, a, b) in order.iter().take(scan) {
            if found.len() >= target {
                completed = false;
                break;
            }
            // Per-anchor skip: `pair_ub` upper-bounds the score, so the effective
            // score is ≤ pair_ub + Γ; if that is ≤ τ no improving column exists
            // here. (Stricter than the scalar `≤ τ` skip — scans more anchors.)
            if self.pair_ub[a * p + b] + gamma_total <= tau {
                continue;
            }
            let sided = self.solve_pair_sided(a, b, ctx.alpha, ctx.beta);
            // Invariant: the sided DP partitions the SAME column space by touch
            // pattern, so the best over patterns must equal the scalar score.
            #[cfg(debug_assertions)]
            {
                let scalar = self.solve_pair(a, b, ctx.alpha, ctx.beta);
                let smax = sided.iter().copied().fold(NEG_INF, f64::max);
                if scalar > NEG_INF / 2.0 {
                    debug_assert!(
                        (smax - scalar).abs() < 1.0e-6,
                        "sided/scalar mismatch at ({a},{b}): sided_max={smax} scalar={scalar}"
                    );
                }
                // STRONGER: each finite pattern's reconstructed column must (1)
                // actually have that touch pattern and (2) re-score to sided[pat].
                for pp in 1..4usize {
                    if sided[pp] <= NEG_INF / 2.0 {
                        continue;
                    }
                    let mut lb: Vec<u32> = Vec::new();
                    self.collect_pair_sided(a, b, pp, &mut lb);
                    lb.sort_unstable();
                    lb.dedup();
                    if lb.len() < 2 {
                        continue;
                    }
                    let tc = lb.iter().any(|&l| ctx.clean_cut.unwrap().side_c.contains(l as usize));
                    let tcc = lb.iter().any(|&l| !ctx.clean_cut.unwrap().side_c.contains(l as usize));
                    let actual_pat = (tc as usize) | ((tcc as usize) << 1);
                    let col = scratch.builder.build_unchecked(lb.clone(), ctx.trees);
                    let sc = col.pricing_score(ctx.alpha, ctx.beta);
                    debug_assert!(
                        actual_pat == pp,
                        "PATTERN MISLABEL at ({a},{b}) pat={pp}: reconstructed labels {lb:?} have actual_pat={actual_pat} score sided={} recon={sc}",
                        sided[pp]
                    );
                    debug_assert!(
                        sc >= sided[pp] - 1.0e-6,
                        "SCORE UNDERFLOW at ({a},{b}) pat={pp}: sided={} but reconstructed col scores {sc}",
                        sided[pp]
                    );
                }
            }
            for pat in 1..4usize {
                if sided[pat] <= NEG_INF / 2.0 {
                    continue;
                }
                let eff = sided[pat] + bonus[pat];
                if !repair_cert && eff > global_max {
                    global_max = eff;
                }
                if eff <= tau {
                    if repair_cert && eff > global_max {
                        global_max = eff;
                    }
                    continue;
                }
                // Improving under pattern `pat`: reconstruct, repair, re-score.
                labels_buf.clear();
                self.collect_pair_sided(a, b, pat, &mut labels_buf);
                labels_buf.sort_unstable();
                labels_buf.dedup();
                if labels_buf.len() < 2 {
                    continue;
                }
                let repaired = match repair_to_valid(&labels_buf, ctx.branchings, ctx.alpha) {
                    Some(l) => l,
                    None => {
                        if !repair_cert && eff > global_max {
                            global_max = eff;
                        }
                        continue;
                    }
                };
                if ctx.seen.contains(&repaired) {
                    if !repair_cert && eff > global_max {
                        global_max = eff;
                    }
                    continue;
                }
                let column = scratch.builder.build_unchecked(repaired.clone(), ctx.trees);
                if ctx.branchings.forbids(&column) {
                    if !repair_cert && eff > global_max {
                        global_max = eff;
                    }
                    continue;
                }
                // Exact effective reduced cost on the (possibly repaired) column:
                // bonus from the actual sides its labels touch.
                let touches_c = repaired.iter().any(|&l| side_c.contains(l as usize));
                let touches_cc = repaired.iter().any(|&l| !side_c.contains(l as usize));
                let bonus_actual = if touches_c { cut.gamma_c } else { 0.0 }
                    + if touches_cc { cut.gamma_cc } else { 0.0 };
                let eff_score = column.pricing_score(ctx.alpha, ctx.beta) + bonus_actual;
                if repair_cert && eff_score > global_max {
                    global_max = eff_score;
                }
                if eff_score > tau {
                    found.push((eff_score, column));
                }
            }
        }

        if std::env::var("KLADOS_CLEAN_LB_TRACE").as_deref() == Ok("1") && found.is_empty() {
            log::info!(
                target: "klados::bp",
                "sided-converge: global_max={:.6} tau={:.6} (margin {:.2e}) gc={:.6} gcc={:.6} completed={} scanned={}/{}",
                global_max, tau, tau - global_max, cut.gamma_c, cut.gamma_cc, completed, scan, order.len(),
            );
        }

        CollectResult {
            found,
            global_max,
            completed,
        }
    }
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

/// Result of a `collect_from_order` scan.
struct CollectResult {
    /// Emittable improving columns (constraint-valid, not already seen).
    found: Vec<(f64, AfColumn)>,
    /// Constraint-blind max of `solve_pair` over every anchor scanned. Valid
    /// as a convergence certificate only when `completed` is true.
    global_max: f64,
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

        // --- Clean-cut sided DP setup (only when a rank-row cut is active) ---
        // `gamma_total = max(0,γ_C) + max(0,γ_Cᶜ)` widens the anchor-skip
        // thresholds (a sound upper bound on the per-pattern bonus) so no anchor
        // with an improving column is dropped. Zero when no cut → thresholds
        // identical to production.
        let gamma_total =
            ctx.clean_cut.map_or(0.0, |c| c.gamma_c.max(0.0) + c.gamma_cc.max(0.0));
        if let Some(c) = ctx.clean_cut.as_ref() {
            self.setup_sided(c.side_c);
        }

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
                // Under a clean cut the effective score is ≤ tight + Γ, so an
                // anchor can be dropped only when tight + Γ ≤ 1+ε. `gamma_total`
                // is 0 with no cut → identical to production.
                if tight + gamma_total <= 1.0 + PRICING_EPS {
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

        let cut_active = ctx.clean_cut.is_some();
        let mut result = if cut_active {
            self.collect_from_order_sided(&order, trial_limit, target, ctx, scratch)
        } else {
            self.collect_from_order(&order, trial_limit, target, ctx, scratch)
        };

        // §3: the trial-limited pass is a fast *generation* shortcut. If it
        // emitted nothing, the full all-anchor scan must run before pricing
        // may certify (`Converged`) or rule out (`Improving`) convergence.
        if result.found.is_empty() && self.fallback_full_when_empty && trial_limit < order.len() {
            order.sort_unstable_by(|l, r| {
                r.0.partial_cmp(&l.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            result = if cut_active {
                self.collect_from_order_sided(&order, order.len(), target, ctx, scratch)
            } else {
                self.collect_from_order(&order, order.len(), target, ctx, scratch)
            };
        }

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
            // Sort by RC descending, cap at 128 to prevent RMP flooding
            // while still returning far more columns than the old limit of 32.
            let mut found = result.found;
            found.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let cap = 128usize.min(found.len());
            return PricingResult::Found(found.into_iter().take(cap).map(|(_, c)| c).collect());
        }

        // No emittable column. Decide between a proven `Converged` and an
        // uncertified `Improving`.
        if result.completed && result.global_max <= 1.0 + PRICING_EPS {
            // The full all-anchor scan completed and the constraint-blind max
            // anchor score is ≤ 1+ε: `solve_pair` dominates every column at
            // its anchor (§3), so no improving column exists anywhere.
            PricingResult::Converged
        } else {
            // Either the scan was incomplete (trial-limited, no fallback), or
            // improving columns provably exist but every one is branch-blocked
            // or already pooled. The LP bound is NOT certified — the solver
            // must branch, never bound-prune.
            PricingResult::Improving
        }
    }
}
