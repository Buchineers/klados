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

use super::{Pricer, PricerScratch, PricingContext, PricingResult};

const PRICING_EPS: f64 = 1.0e-8;
const NEG_INF: f64 = f64::NEG_INFINITY;
/// Sentinel in `memo_side_split` meaning "leaf-only side, no extension chosen".
const SPLIT_LEAF_ONLY: u32 = u32::MAX - 1;

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
            min_active_labels: 0,
            cannot_pair: Vec::new(),
            must_pair: Vec::new(),
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

        let am = self.active_mask.clone();
        let am_blocks: Vec<usize> = am.as_slice().to_vec();
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
        let n_blocks = self.descendant_leaves[0][side_nodes[0] as usize]
            .as_slice()
            .len();
        for wi in 0..n_blocks {
            let mut w =
                self.descendant_leaves[0][side_nodes[0] as usize].as_slice()[wi] & am_blocks[wi];
            for ti in 1..self.num_trees {
                w &= self.descendant_leaves[ti][side_nodes[ti] as usize].as_slice()[wi];
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
                let c = self.label_to_active_idx[c_label] as usize;

                // Branch-aware: skip extension candidate that would violate
                // cannot-link with the anchor leaf of this side.
                let cannot_a = self.cannot_pair[label_a as usize] as u32;
                if cannot_a != 0 && cannot_a == c_label as u32 {
                    continue;
                }

                let idx_c = a * p + c;
                let pen = self.pair_penalty[idx_c] - b_penalty_sum;
                let ub = self.pair_ub[idx_c];
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
        }

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
        order.sort_unstable_by(|l, r| r.0.partial_cmp(&l.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut found: Vec<(f64, AfColumn)> = Vec::new();
        let target = self.max_per_call;
        let trial_limit = (self.pair_trial_limit as usize).min(order.len());

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

            let score = self.solve_pair(a, b, ctx.alpha, ctx.beta);
            if score <= 1.0 + PRICING_EPS {
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
            found.push((score, column));
        }

        if found.is_empty() {
            let max_alpha = ctx.alpha.iter().copied().fold(NEG_INF, f64::max);
            if max_alpha <= 1.0 + PRICING_EPS {
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
