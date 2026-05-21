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
//! ## Soundness
//!
//! Heuristic: returns `Found` when any anchor pair (a, b) has positive
//! score, `Exhausted` when none do. **Never returns `Converged`** — the DP
//! optimises over leaf-pair anchors, and there might be valid columns
//! whose best anchor pair is outside the active label set. Matching
//! bp-multi's behaviour here.

use fixedbitset::FixedBitSet;
use klados_core::{NONE, Tree};

use crate::bp::column::AfColumn;

use super::{adaptive_m2_batch_size, Pricer, PricerScratch, PricingContext, PricingResult};

const PRICING_EPS: f64 = 1.0e-8;
const NEG_INF: f64 = f64::NEG_INFINITY;

/// Read the `KLADOS_BP_USE_ANCHOR_CACHE` env var once and cache the
/// result. Cached-positive reuse is opt-in via this flag.
fn anchor_cache_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KLADOS_BP_USE_ANCHOR_CACHE").is_ok())
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
    /// `cannot_pair[l]` = partner that cannot appear in the same column as l,
    /// or 0 if none.  Checked during `solve_side` to skip extension candidates
    /// that would produce a forbidden column.
    cannot_pair: Vec<u32>,
    /// `must_pair[l]` = partner that must appear in the same column as l,
    /// or 0 if none.  Checked during `solve_pair` to reject anchor pairs that
    /// separate must-linked leaves.
    must_pair: Vec<u32>,
    /// Reusable scratch for `solve_side`'s phase-1 candidate collection.
    /// Avoids a Vec allocation per call (~2500 per CG iter on Class B).
    solve_side_candidates: Vec<(u32, u32)>,
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
            cannot_pair: Vec::new(),
            must_pair: Vec::new(),
            solve_side_candidates: Vec::new(),
        }
    }

    /// Collect active labels from alpha. Returns `true` if the set changed
    /// since the last call.
    fn ensure_active_labels(&mut self, alpha: &[f64]) -> bool {
        let prev: Vec<u32> = std::mem::take(&mut self.active_labels);
        self.active_mask.clear();
        for v in self.label_to_active_idx.iter_mut() {
            *v = u32::MAX;
        }
        for label in 1..=self.num_leaves as u32 {
            if alpha[label as usize] > 1.0e-12 {
                self.active_mask.insert(label as usize);
                self.label_to_active_idx[label as usize] = self.active_labels.len() as u32;
                self.active_labels.push(label);
            }
        }
        self.active_labels != prev
    }

    /// Rebuild per-pair LCA / side-child tables (expensive — O(p²·m)).
    /// Only called when the active label set changes.
    fn rebuild_pair_tables(&mut self, trees: &[Tree]) {
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
    fn refresh_dual_tables(
        &mut self,
        trees: &[Tree],
        alpha: &[f64],
        beta: &[Vec<f64>],
    ) {
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

        // pair_side_parent_prefix_beta (O(p²·m), no tree traversal)
        for ti in 0..self.num_trees {
            self.pair_side_parent_prefix_beta[ti].clear();
            self.pair_side_parent_prefix_beta[ti].resize(pair_count, 0.0);
            let tree = &trees[ti];
            for idx in 0..pair_count {
                let anc = self.pair_side_child[ti][idx] as usize;
                let parent = tree.parent[anc];
                self.pair_side_parent_prefix_beta[ti][idx] = if parent == NONE {
                    0.0
                } else {
                    self.prefix_beta[ti][parent as usize]
                };
            }
        }

        // pair_penalty, pair_ub, pair_singleton_penalty (O(p²·m))
        self.pair_penalty.clear();
        self.pair_penalty.resize(pair_count, 0.0);
        self.pair_ub.clear();
        self.pair_ub.resize(pair_count, f64::INFINITY);
        self.pair_singleton_penalty.clear();
        self.pair_singleton_penalty.resize(pair_count, 0.0);

        for ti in 0..self.num_trees {
            let tree = &trees[ti];
            let mut nps = vec![0.0; tree.num_nodes()];
            for node in 0..tree.num_nodes() {
                let parent = tree.parent[node];
                nps[node] = if parent == NONE {
                    0.0
                } else {
                    self.prefix_beta[ti][parent as usize]
                };
            }
            for a in 0..p {
                let base = a * p;
                let leaf_a_node = self.label_node[ti][self.active_labels[a] as usize];
                for c in 0..p {
                    let idx = base + c;
                    let r = self.pair_root[ti][idx] as usize;
                    self.pair_penalty[idx] += nps[r];
                    if self.sum_alpha[ti][r] < self.pair_ub[idx] {
                        self.pair_ub[idx] = self.sum_alpha[ti][r];
                    }

                    let anc = self.pair_side_child[ti][idx];
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
                    let lower = if anc != NONE {
                        let ap = tree.parent[anc as usize];
                        if ap != NONE {
                            self.prefix_beta[ti][ap as usize]
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
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
    /// extension `c` (`solve_pair(a, c) − pen(a, c)`). The "leaf-only"
    /// alternative is always a valid competing column at this anchor.
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
        // Constant across wi.
        let cannot_a = self.cannot_pair[label_a as usize] as u32;
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
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    w &= w - 1;
                    let c_label = (wi << BLOCK_SHIFT) + bit;
                    if cannot_a != 0 && cannot_a == c_label as u32 {
                        continue;
                    }
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
    ) -> Vec<(f64, AfColumn)> {
        let p = self.active_labels.len();
        let mut found: Vec<(f64, AfColumn)> = Vec::new();

        let cache_active =
            anchor_cache_enabled() && ctx.trees.len() == 2 && scratch.anchor_cache.is_some();

        for &(_, a, b) in order.iter().take(trial_limit) {
            if found.len() >= target {
                break;
            }
            // `pair_ub` may have tightened since we built the order (the
            // bound is recomputed each call).  Double-check here.
            if self.pair_ub[a * p + b] <= 1.0 + PRICING_EPS {
                continue;
            }

            // Branch-aware: skip anchor pairs that separate must-linked
            // leaves or that are themselves cannot-linked.
            let la = self.active_labels[a];
            let lb = self.active_labels[b];
            if self.must_pair[la as usize] != 0 && self.must_pair[la as usize] != lb {
                continue;
            }
            if self.must_pair[lb as usize] != 0 && self.must_pair[lb as usize] != la {
                continue;
            }
            if self.cannot_pair[la as usize] == lb {
                continue;
            }

            // --- Anchor cache fast path ---
            if cache_active {
                let cache_hit = {
                    let cache = scratch.anchor_cache.as_mut().unwrap();
                    cache.try_emit(la, lb, ctx.alpha, &ctx.beta[0], &ctx.beta[1], PRICING_EPS)
                };
                match cache_hit {
                    super::anchor_cache::CacheResult::Emit { score: _ } => {
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

            let score = self.solve_pair(a, b, ctx.alpha, ctx.beta);
            if score <= 1.0 + PRICING_EPS {
                // Refresh cache with a "non-improving" entry (gap=0,
                // empty column would be wrong; instead just skip the
                // refresh — the cache will treat this anchor as a Miss
                // next time and re-run the DP if duals change).
                continue;
            }
            let labels = self.pair_labels(a, b, ctx.alpha, ctx.beta);
            if labels.len() < 2 {
                continue;
            }
            if ctx.seen.contains(&labels) {
                continue;
            }
            let column = scratch.builder.build_unchecked(labels, ctx.trees);
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

        found
    }
}

impl Pricer for LeafPairDpPricer {
    fn name(&self) -> &'static str {
        "leaf-pair-dp"
    }

    fn price(&mut self, ctx: &PricingContext, scratch: &mut PricerScratch) -> PricingResult {
        // --- Pricer caching: only rebuild when active set changes ---
        let changed = self.ensure_active_labels(ctx.alpha);
        if changed {
            self.rebuild_pair_tables(ctx.trees);
        }
        self.refresh_dual_tables(ctx.trees, ctx.alpha, ctx.beta);
        self.reset_memos();

        // --- Anchor cache (cached-positive reuse) ---
        // Gated by env var. Lazy-init; allocate label-pair storage once.
        // Indexing is by leaf label, so structural data is invariant
        // across active-label-set changes.
        if anchor_cache_enabled() && ctx.trees.len() == 2 {
            if scratch.anchor_cache.is_none() {
                scratch.anchor_cache = Some(super::anchor_cache::AnchorCache::new(self.num_leaves));
            }
            let cache = scratch.anchor_cache.as_mut().unwrap();
            if !cache.is_built() {
                cache.refresh_static(ctx.trees);
            }
        }

        // --- Branch-awareness: build lookup maps from current branchings ---
        self.cannot_pair.resize(self.num_leaves + 1, 0);
        self.cannot_pair.fill(0);
        self.must_pair.resize(self.num_leaves + 1, 0);
        self.must_pair.fill(0);
        for pair in ctx.branchings.cannot_link() {
            self.cannot_pair[pair.a as usize] = pair.b;
            self.cannot_pair[pair.b as usize] = pair.a;
        }
        for pair in ctx.branchings.must_link() {
            self.must_pair[pair.a as usize] = pair.b;
            self.must_pair[pair.b as usize] = pair.a;
        }

        let p = self.active_labels.len();
        if p < 2 || p < self.min_active_labels {
            return PricingResult::Exhausted;
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
            let max_alpha = ctx.alpha.iter().copied().fold(NEG_INF, f64::max);
            if max_alpha <= 1.0 + PRICING_EPS && max_bound <= 1.0 + PRICING_EPS {
                return PricingResult::Converged;
            }
            return PricingResult::Exhausted;
        }
        let target = if ctx.trees.len() == 2 {
            self.max_per_call.min(adaptive_m2_batch_size(ctx))
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

        let mut found = self.collect_from_order(&order, trial_limit, target, ctx, scratch);

        if found.is_empty() && self.fallback_full_when_empty && trial_limit < order.len() {
            order.sort_unstable_by(|l, r| {
                r.0.partial_cmp(&l.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            found = self.collect_from_order(&order, order.len(), target, ctx, scratch);
        }

        // Optional anchor-cache stats logging.
        if anchor_cache_enabled() {
            if let Some(cache) = scratch.anchor_cache.as_ref() {
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
        }

        if found.is_empty() {
            let max_alpha = ctx.alpha.iter().copied().fold(NEG_INF, f64::max);
            // `Converged` may only be claimed when the scan was exhaustive —
            // every UB-surviving anchor pair actually evaluated. A partial
            // (trial-limited) scan that found nothing proves nothing, so it
            // must report `Exhausted` and never certify convergence.
            let exhaustive = trial_limit >= order.len() || self.fallback_full_when_empty;
            if exhaustive && max_alpha <= 1.0 + PRICING_EPS {
                // No positive singleton and no positive multi-leaf column
                // anchored at any pair → truly converged.
                return PricingResult::Converged;
            }
            return PricingResult::Exhausted;
        }
        // Sort by RC descending, cap at 128 to prevent RMP flooding
        // while still returning far more columns than the old limit of 32.
        found.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let cap = 128usize.min(found.len());
        PricingResult::Found(found.into_iter().take(cap).map(|(_, c)| c).collect())
    }
}
