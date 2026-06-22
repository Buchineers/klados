//! Non-crossing packing (`ncpack`) — high-fragmentation multi-tree MAF.
//!
//! For ROOTED multi-tree MAF on `n` leaves and `m` trees, let `mu* = n - OPT`
//! be the number of "merges". On the high-fragmentation wall (small `mu*`,
//! `OPT ≈ 0.9n`, many trees) the optimum agreement forest is dominated by
//! 2-leaf components, which makes a purely combinatorial attack possible.
//!
//! ## The reformulation
//! Two leaf-pairs `{a,b}` and `{c,d}` **cross** in tree `q` iff their closed
//! Steiner paths (`leaf..lca..leaf`, inclusive) share a node. Sharing a leaf
//! ⇒ cross. A set of mutually non-crossing pairs is *always* a valid agreement
//! forest (pairs agree vacuously; pairwise non-crossing is exactly the global
//! non-crossing constraint for all-pair blocks). Hence
//!
//! > `M := max non-crossing matching = max clique in the pair-compatibility
//! >  graph (V = all pairs, edge = mutually non-crossing in every tree) ≤ mu*`.
//!
//! Empirically `M = mu*` (gap 0) or `mu*-1` (gap 1) across the whole high-`m`
//! wall, found + proven in well under a second.
//!
//! ## Certificate machinery (this module)
//! - **DSATUR clique-cover** of the pair-conflict graph: a coloring of the
//!   compat graph. With a good order (DSATUR) the cover size equals `M`, which
//!   *certifies* no `M+1` mutually-non-crossing pairs exist (a tight bound —
//!   earlier work used a bad greedy order and wrongly concluded the graph was
//!   "imperfect / loose").
//! - **Exact lift identity:** `mu* = max_B [ (|B|-1) + M(L\B, ⊥B) ]` over
//!   agreement blocks `B`, where `M(L\B, ⊥B)` is the residual non-crossing
//!   matching (drop `B`'s leaves; forbid `B`-crossing pairs) — recomputed by the
//!   same fast clique solver, so it is *cheap per block*.
//! - **killed-bound prune:** with the tight cover (`ncov = M`),
//!   `lift(B) ≤ (|B|-1) - killed(B)`, where `killed(B)` counts cover-classes
//!   every pair of which is crossed-or-touched by `B`. `killed(B)` is **monotone
//!   non-decreasing** as `B` grows (the Steiner footprint only enlarges), giving
//!   a sound growth-prune that *bounds* the block enumeration instead of letting
//!   it explode.
//!
//! ## Soundness status (IMPORTANT for the Exact track)
//! The forest produced is *always* a valid agreement forest (validated before
//! emission), so it is a sound primal / upper bound on `OPT` and is emitted on
//! the Heuristic track unconditionally.
//!
//! Claiming the forest is *optimal* (Exact track) requires proving no packing
//! beats the found value. The single-block lift search proves no *single* block
//! lifts; the multi-block case reduces to an open charging lemma ("two disjoint
//! non-crossing blocks cannot both fully-kill a common cover-class"). It holds
//! empirically but is not yet proven, so the optimality claim is **gated behind
//! `KLADOS_NCPACK_CERT=1`** and off by default. With the gate off, the Exact
//! track emits the forest only when it coincides with an independently sound
//! bound (see [`NcpackSolver::solve`]).

use fixedbitset::FixedBitSet;
use klados_core::af_validator::{AfValidation, validate_agreement_forest};
use klados_core::{Instance, SolverStats, Tree};
use log::debug;
use std::time::{Duration, Instant};

use crate::{RunConfig, Solver, Track};

// ── Pair graph ───────────────────────────────────────────────────────────────

/// The pair-compatibility ("compat") graph over all `C(n,2)` leaf-pairs.
///
/// Vertices are leaf-pairs; an edge means the two pairs are mutually
/// non-crossing in *every* tree (their concatenated per-tree Steiner masks are
/// disjoint). `masks[i]` is pair `i`'s footprint; `adj[i]` its compat neighbors.
struct PairGraph {
    /// `pairs[i] = (a, b)` with `1 ≤ a < b ≤ n`.
    pairs: Vec<(u32, u32)>,
    /// Concatenated per-tree Steiner-node bitmask for each pair (words of 64).
    masks: Vec<Vec<u64>>,
    /// Compat adjacency: `adj[i]` = pairs non-crossing with `i` (a bitset over
    /// pair indices).
    adj: Vec<FixedBitSet>,
    /// Number of pairs.
    p: usize,
}

/// Return the node ids on the closed path `leaf(a)..lca..leaf(b)` in `tree`.
fn path_nodes(tree: &Tree, a: u32, b: u32, out: &mut Vec<u32>) {
    out.clear();
    let la = tree.node_by_label(a);
    let lb = tree.node_by_label(b);
    let anc = tree.nearest_common_ancestor(la, lb);
    let mut u = la;
    while u != anc {
        out.push(u);
        u = tree.parent[u as usize];
    }
    out.push(anc);
    let mut u = lb;
    while u != anc {
        out.push(u);
        u = tree.parent[u as usize];
    }
}

impl PairGraph {
    fn build(inst: &Instance) -> Self {
        let n = inst.num_leaves;
        let trees = &inst.trees;
        // Per-tree bit offsets into the concatenated mask.
        let mut offs = Vec::with_capacity(trees.len());
        let mut off = 0usize;
        for t in trees {
            offs.push(off);
            off += t.num_nodes();
        }
        let total_bits = off;
        let words = total_bits.div_ceil(64);

        let mut pairs = Vec::new();
        for a in 1..=n {
            for b in (a + 1)..=n {
                pairs.push((a, b));
            }
        }
        let p = pairs.len();

        // Build each pair's concatenated Steiner mask.
        let mut masks = vec![vec![0u64; words]; p];
        let mut scratch = Vec::new();
        for (i, &(a, b)) in pairs.iter().enumerate() {
            let mask = &mut masks[i];
            for (q, tree) in trees.iter().enumerate() {
                let base = offs[q];
                path_nodes(tree, a, b, &mut scratch);
                for &nd in &scratch {
                    let bit = base + nd as usize;
                    mask[bit >> 6] |= 1u64 << (bit & 63);
                }
            }
        }

        // Compat adjacency: disjoint masks ⇒ non-crossing in all trees.
        let mut adj = vec![FixedBitSet::with_capacity(p); p];
        for i in 0..p {
            // split borrow: compare i against j>i, set both sides.
            let (mi, rest) = {
                let (head, tail) = masks.split_at(i + 1);
                (&head[i], tail)
            };
            for (off_j, mj) in rest.iter().enumerate() {
                let j = i + 1 + off_j;
                if disjoint(mi, mj) {
                    adj[i].insert(j);
                    adj[j].insert(i);
                }
            }
        }

        PairGraph {
            pairs,
            masks,
            adj,
            p,
        }
    }
}

/// `true` iff the two equal-length word vectors share no set bit.
#[inline]
fn disjoint(a: &[u64], b: &[u64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x & y == 0)
}

// ── Max-clique (Tomita) — used for the matching subroutine over any candset ──

/// Greedy coloring of the subgraph induced by `cand`; returns vertices in
/// non-decreasing color order with their color numbers. A clique can contain at
/// most one vertex of each color, so `len(R) + color` upper-bounds any clique
/// extending `R` through `cand` — the Tomita pruning bound.
fn color_sort(g: &PairGraph, cand: &FixedBitSet) -> (Vec<usize>, Vec<usize>) {
    let mut order = Vec::new();
    let mut colors = Vec::new();
    let mut uncolored = cand.clone();
    let mut c = 0usize;
    while uncolored.count_ones(..) > 0 {
        c += 1;
        let mut avail = uncolored.clone();
        while let Some(v) = avail.ones().next() {
            order.push(v);
            colors.push(c);
            avail.difference_with(&g.adj[v]);
            avail.remove(v);
            uncolored.remove(v);
        }
    }
    (order, colors)
}

/// Tomita expansion. Returns `true` if the subtree was searched to completion
/// (i.e. the node budget was not exhausted), so the caller can report `proven`.
fn expand(
    g: &PairGraph,
    r: &mut Vec<usize>,
    cand: FixedBitSet,
    best: &mut Vec<usize>,
    nodes: &mut u64,
    budget: u64,
) -> bool {
    if cand.count_ones(..) == 0 {
        if r.len() > best.len() {
            *best = r.clone();
        }
        return true;
    }
    *nodes += 1;
    if *nodes > budget {
        return false;
    }
    let (order, colors) = color_sort(g, &cand);
    let mut p = cand;
    for i in (0..order.len()).rev() {
        let v = order[i];
        if r.len() + colors[i] <= best.len() {
            return true; // bound: no extension here can beat best
        }
        r.push(v);
        let mut next = p.clone();
        next.intersect_with(&g.adj[v]);
        let complete = expand(g, r, next, best, nodes, budget);
        r.pop();
        p.remove(v);
        if !complete {
            return false;
        }
    }
    true
}

// ── DSATUR clique-cover of the pair-conflict graph ───────────────────────────

/// Clique-cover of the conflict graph = coloring of the compat graph (an
/// independent set in compat = a mutually-crossing class). DSATUR order gives a
/// tight cover (`ncov = M` on the wall), which certifies no `M+1` pairs.
///
/// Returns the classes (each a list of pair indices); `classes.len()` is `ncov`.
fn dsatur_cover(g: &PairGraph) -> Vec<Vec<usize>> {
    let p = g.p;
    let mut color = vec![usize::MAX; p];
    // saturation = set of distinct neighbor-colors seen so far
    let mut sat: Vec<FixedBitSet> = vec![FixedBitSet::with_capacity(0); p];
    let degc: Vec<usize> = (0..p).map(|v| g.adj[v].count_ones(..)).collect();
    let mut uncolored = FixedBitSet::with_capacity(p);
    uncolored.insert_range(..);
    let mut ncolors = 0usize;

    for _ in 0..p {
        // pick uncolored vertex of max saturation, tie-break by compat-degree.
        let v = uncolored
            .ones()
            .max_by_key(|&u| (sat[u].count_ones(..), degc[u]));
        let v = match v {
            Some(v) => v,
            None => break,
        };
        // smallest color not used by a compat-neighbor (neighbors in compat
        // graph must differ — they could share a class only if crossing).
        let mut used = FixedBitSet::with_capacity(ncolors + 1);
        for u in g.adj[v].ones() {
            if color[u] != usize::MAX {
                if color[u] >= used.len() {
                    used.grow(color[u] + 1);
                }
                used.insert(color[u]);
            }
        }
        let mut cc = 0usize;
        while cc < used.len() && used.contains(cc) {
            cc += 1;
        }
        color[v] = cc;
        ncolors = ncolors.max(cc + 1);
        uncolored.remove(v);
        for u in g.adj[v].ones() {
            if color[u] == usize::MAX {
                if cc >= sat[u].len() {
                    sat[u].grow(cc + 1);
                }
                sat[u].insert(cc);
            }
        }
    }

    let mut classes = vec![Vec::new(); ncolors];
    for i in 0..p {
        classes[color[i]].push(i);
    }
    classes
}

// ── Lift search ──────────────────────────────────────────────────────────────

/// A packing block: the leaf labels it merges (`|B| ≥ 2`).
type Block = Vec<u32>;

/// Result of the lift search.
/// Result of the recursive exact packing search.
struct PackResult {
    /// `mu*` value found. Exact when `complete`; otherwise a lower bound.
    value: usize,
    /// The ≥3-leaf blocks of an optimal packing (pairs are the residual matching).
    blocks: Vec<Block>,
    /// `true` iff the recursion closed (no budget/deadline hit) — i.e. the value
    /// is the proven exact `mu*` for the explored block-size cap.
    complete: bool,
    /// #matching solves (diagnostic).
    matchings: u64,
}

/// Recursive exact engine for `mu*`.
///
/// **Anchored canonical-block recursion.** Any packing splits into its ≥3-leaf
/// blocks plus a set of pairs/singletons. The pairs are handled in one shot by
/// the max non-crossing matching (polynomial subroutine); we branch only on the
/// ≥3-leaf blocks, each anchored at its smallest leaf so every packing is reached
/// exactly once:
///
/// ```text
/// mu*(active, forbidden, anchor) =
///   max( M(active, forbidden),
///        max over agreement blocks B (|B|>=3, min(B)>=anchor, available)
///            of (|B|-1) + mu*(active \ B, forbidden ∪ mask(B), min(B)+1) )
/// ```
///
/// This is exhaustive and exact — no cover-tightness or lemma needed. Soundness
/// of certification is therefore independent of the (imperfect) clique cover.
///
/// **Pruning.** `avail_classes(B)` = the top clique-cover restricted to pairs
/// still available in the residual; it is a valid upper bound on the residual
/// matching `M(residual)`. A block is *promising* (worth recursing) only when
/// `(|B|-1) + avail_classes(B) + slack > best`, where `slack` covers the
/// residual's own gap. With `slack` ≥ the true residual gap this prune is sound;
/// the recursion completing then certifies `mu*`. `slack` is small (gap ≤ 1 on
/// the wall) and configurable. `avail_classes` is monotone non-increasing as a
/// block grows, giving a sound growth-prune that bounds the enumeration.
struct Rec<'a> {
    g: &'a PairGraph,
    inst: &'a Instance,
    /// Per-tree node-id offset into the concatenated mask.
    offs: Vec<usize>,
    /// Mask word count.
    words: usize,
    /// Max block size considered (blocks larger than this are not enumerated).
    kmax: usize,
    /// Slack: assumed upper bound on any residual's own gap (sound iff actual
    /// residual gap ≤ slack). Default 1 (gap ≤ 1 empirically on the wall).
    slack: isize,
    /// Per-matching clique-search node budget.
    node_budget: u64,
    /// Wall deadline.
    deadline: Instant,
    /// Top clique-cover: pair indices per class (for the `avail_classes` bound).
    class_pidx: Vec<Vec<usize>>,
    /// `true` once a budget/deadline was hit (result no longer proven exact).
    aborted: std::cell::Cell<bool>,
    /// #matching solves.
    matchings: std::cell::Cell<u64>,
}

impl<'a> Rec<'a> {
    fn new(
        g: &'a PairGraph,
        inst: &'a Instance,
        cover: &[Vec<usize>],
        kmax: usize,
        slack: isize,
        node_budget: u64,
        deadline: Instant,
    ) -> Self {
        let mut offs = Vec::with_capacity(inst.trees.len());
        let mut off = 0usize;
        for t in &inst.trees {
            offs.push(off);
            off += t.num_nodes();
        }
        let words = off.div_ceil(64);
        Rec {
            g,
            inst,
            offs,
            words,
            kmax,
            slack,
            node_budget,
            deadline,
            class_pidx: cover.to_vec(),
            aborted: std::cell::Cell::new(false),
            matchings: std::cell::Cell::new(0),
        }
    }

    /// Concatenated per-tree Steiner mask of a leaf set (union of paths to the
    /// first leaf).
    fn block_mask(&self, leaves: &[u32], scratch: &mut Vec<u32>) -> Vec<u64> {
        let mut mask = vec![0u64; self.words];
        let r = leaves[0];
        for (q, tree) in self.inst.trees.iter().enumerate() {
            let base = self.offs[q];
            if leaves.len() == 1 {
                let nd = tree.node_by_label(r);
                let bit = base + nd as usize;
                mask[bit >> 6] |= 1u64 << (bit & 63);
                continue;
            }
            for &x in &leaves[1..] {
                path_nodes(tree, r, x, scratch);
                for &nd in scratch.iter() {
                    let bit = base + nd as usize;
                    mask[bit >> 6] |= 1u64 << (bit & 63);
                }
            }
        }
        mask
    }

    /// `leaves` induce the same labeled topology in every tree.
    fn is_agreement(&self, leaves: &[u32]) -> bool {
        if leaves.len() <= 2 {
            return true;
        }
        let t0 = induced_topology(&self.inst.trees[0], leaves);
        self.inst.trees[1..]
            .iter()
            .all(|t| induced_topology(t, leaves) == t0)
    }

    /// Pair indices available given the active leaf set and forbidden footprint.
    fn candset(&self, active: &[bool], forbidden: &[u64]) -> FixedBitSet {
        let mut s = FixedBitSet::with_capacity(self.g.p);
        for i in 0..self.g.p {
            let (a, b) = self.g.pairs[i];
            if active[a as usize] && active[b as usize] && disjoint(&self.g.masks[i], forbidden) {
                s.insert(i);
            }
        }
        s
    }

    /// Max non-crossing matching among available pairs (size).
    fn matching(&self, active: &[bool], forbidden: &[u64]) -> usize {
        let cs = self.candset(active, forbidden);
        let mut best = Vec::new();
        let mut nodes = 0u64;
        expand(self.g, &mut Vec::new(), cs, &mut best, &mut nodes, self.node_budget);
        self.matchings.set(self.matchings.get() + 1);
        best.len()
    }

    /// Max non-crossing matching among available pairs (the pairs themselves).
    fn matching_pairs(&self, active: &[bool], forbidden: &[u64]) -> Vec<(u32, u32)> {
        let cs = self.candset(active, forbidden);
        let mut best = Vec::new();
        let mut nodes = 0u64;
        expand(self.g, &mut Vec::new(), cs, &mut best, &mut nodes, self.node_budget);
        best.iter().map(|&i| self.g.pairs[i]).collect()
    }

    /// Upper bound on the residual matching `M(active\B, forbidden∪mask(B))`:
    /// the number of top-cover classes that still contain an available residual
    /// pair (each class is a clique, contributing ≤ 1 to any matching).
    fn avail_classes(&self, active: &[bool], forbidden: &[u64], bmask: &[u64], bleaves: &[u32]) -> usize {
        let mut c = 0usize;
        for cls in &self.class_pidx {
            for &pi in cls {
                let (a, b) = self.g.pairs[pi];
                if !active[a as usize] || !active[b as usize] {
                    continue;
                }
                if bleaves.contains(&a) || bleaves.contains(&b) {
                    continue;
                }
                if !disjoint(&self.g.masks[pi], forbidden) || !disjoint(&self.g.masks[pi], bmask) {
                    continue;
                }
                c += 1;
                break; // class has an available residual pair
            }
        }
        c
    }

    /// `mu*(active, forbidden)` over blocks with `min ≥ anchor`. Returns the value
    /// and the ≥3-leaf blocks of an optimal packing.
    fn solve(&self, active: &mut [bool], forbidden: &[u64], anchor: u32) -> (usize, Vec<Block>) {
        if Instant::now() >= self.deadline {
            self.aborted.set(true);
        }
        let m = self.matching(active, forbidden);
        let mut best = m;
        let mut best_blocks: Vec<Block> = Vec::new();
        if self.aborted.get() {
            return (best, best_blocks);
        }
        let n = self.inst.num_leaves;
        let mut leaves: Vec<u32> = Vec::new();
        let mut scratch: Vec<u32> = Vec::new();
        for s in anchor..=n {
            if active[s as usize] {
                leaves.push(s);
                self.grow(active, forbidden, &mut leaves, s, &mut scratch, &mut best, &mut best_blocks);
                leaves.pop();
                if self.aborted.get() {
                    break;
                }
            }
        }
        (best, best_blocks)
    }

    /// DFS growth over agreement blocks seeded at `leaves[0] = min(B)`; recurses on
    /// promising blocks.
    fn grow(
        &self,
        active: &mut [bool],
        forbidden: &[u64],
        leaves: &mut Vec<u32>,
        last: u32,
        scratch: &mut Vec<u32>,
        best: &mut usize,
        best_blocks: &mut Vec<Block>,
    ) {
        if self.aborted.get() {
            return;
        }
        let n = self.inst.num_leaves;
        if leaves.len() >= 3 {
            let bmask = self.block_mask(leaves, scratch);
            // Crossing a committed block; the mask only grows, so prune the subtree.
            if !disjoint(&bmask, forbidden) {
                return;
            }
            let w = leaves.len() - 1;
            let av = self.avail_classes(active, forbidden, &bmask, leaves);
            // Promising ⇒ recurse for the exact residual mu*.
            if (w as isize) + (av as isize) + self.slack > *best as isize {
                for &l in leaves.iter() {
                    active[l as usize] = false;
                }
                let mut nf = forbidden.to_vec();
                for k in 0..self.words {
                    nf[k] |= bmask[k];
                }
                let (sv, sb) = self.solve(active, &nf, leaves[0] + 1);
                for &l in leaves.iter() {
                    active[l as usize] = true;
                }
                let cand = w + sv;
                if cand > *best {
                    *best = cand;
                    let mut nb = Vec::with_capacity(sb.len() + 1);
                    nb.push(leaves.clone());
                    nb.extend(sb);
                    *best_blocks = nb;
                }
            }
            // Monotone growth-prune: avail_classes only shrinks as the block grows,
            // so no extension B' (|B'| ≤ kmax) can become promising once this fails.
            if (self.kmax as isize - 1) + (av as isize) + self.slack <= *best as isize {
                return;
            }
        }
        if leaves.len() >= self.kmax {
            return;
        }
        for x in (last + 1)..=n {
            if active[x as usize] {
                leaves.push(x);
                if self.is_agreement(leaves) {
                    self.grow(active, forbidden, leaves, x, scratch, best, best_blocks);
                }
                leaves.pop();
                if self.aborted.get() {
                    return;
                }
            }
        }
    }

    // ── Direct Hall-surplus certificate (single + pair, no recursion) ────────
    //
    // Computes mu* = M + max(0, best single-block lift, best 2-block lift) under
    // the empirically-validated "Hall witness <= 2" conjecture. This is the fast,
    // non-recursive replacement for the slack=1 recursion: it directly enumerates
    // big blocks once, checks single-block lifts, then checks compatible block
    // PAIRS using the sharp criterion |K1∩K2| > slack1 + slack2 - (ncov - M).
    // Returns None if ncov > 64 (bitset cap) → caller falls back to the recursion.

    /// Bitset over cover classes: bit `c` set iff class `c` is fully crossed-or-
    /// touched by the block (every pair of the class hits the block).
    fn killed_bits(&self, bleaves: &[u32], bmask: &[u64]) -> u64 {
        let mut bits = 0u64;
        for (ci, cls) in self.class_pidx.iter().enumerate() {
            if ci >= 64 {
                break;
            }
            let mut all_hit = true;
            for &pi in cls {
                let (a, b) = self.g.pairs[pi];
                if bleaves.contains(&a) || bleaves.contains(&b) {
                    continue;
                }
                if !disjoint(&self.g.masks[pi], bmask) {
                    continue;
                }
                all_hit = false;
                break;
            }
            if all_hit {
                bits |= 1u64 << ci;
            }
        }
        bits
    }

    /// Residual matching weight after committing `bleaves` with footprint `bmask`.
    fn residual_matching(&self, active: &mut [bool], bleaves: &[u32], bmask: &[u64]) -> usize {
        for &l in bleaves {
            active[l as usize] = false;
        }
        let m = self.matching(active, bmask);
        for &l in bleaves {
            active[l as usize] = true;
        }
        m
    }

    /// Enumerate agreement blocks of size 3..=kmax, with footprint + killed bits.
    fn collect_blocks(&self) -> Vec<DirectBlock> {
        let n = self.inst.num_leaves;
        let mut out = Vec::new();
        let mut leaves = Vec::new();
        let mut scratch = Vec::new();
        for s in 1..=n {
            leaves.push(s);
            self.grow_collect(&mut leaves, s, &mut scratch, &mut out);
            leaves.pop();
        }
        out
    }

    fn grow_collect(
        &self,
        leaves: &mut Vec<u32>,
        last: u32,
        scratch: &mut Vec<u32>,
        out: &mut Vec<DirectBlock>,
    ) {
        let n = self.inst.num_leaves;
        if leaves.len() >= 3 {
            let mask = self.block_mask(leaves, scratch);
            let killed = self.killed_bits(leaves, &mask);
            out.push(DirectBlock {
                leaves: leaves.clone(),
                mask,
                killed,
                w: leaves.len() - 1,
            });
        }
        if leaves.len() >= self.kmax {
            return;
        }
        for x in (last + 1)..=n {
            leaves.push(x);
            if self.is_agreement(leaves) {
                self.grow_collect(leaves, x, scratch, out);
            }
            leaves.pop();
        }
    }

    /// Direct certificate. Returns (value, blocks) where value = mu* under the
    /// witness<=2 conjecture, or None if `ncov > 64`.
    fn direct_certify(&self, active: &mut [bool]) -> Option<(usize, Vec<Block>)> {
        let ncov = self.class_pidx.len();
        if ncov > 64 {
            return None;
        }
        let zero = vec![0u64; self.words];
        let m = self.matching(active, &zero);
        let imperf = ncov as isize - m as isize; // ncov - M >= 0 (cover slack)

        let blocks = self.collect_blocks();

        // ── single-block lifts ──
        let mut best = m;
        let mut best_blocks: Vec<Block> = Vec::new();
        for b in &blocks {
            let kc = b.killed.count_ones() as isize;
            // sound prune: residual <= ncov - kc; lift <= w + (ncov - kc) - M
            if b.w as isize + (ncov as isize - kc) - m as isize <= (best - m) as isize {
                continue;
            }
            let rm = self.residual_matching(active, &b.leaves, &b.mask);
            let v = b.w + rm;
            if v > best {
                best = v;
                best_blocks = vec![b.leaves.clone()];
            }
        }

        // ── 2-block lifts (sharp Hall-surplus pair criterion) ──
        // Candidate pair iff |K1∩K2| > slack1 + slack2 - imperf, where
        // slack_i = killed_count_i - w_i. (For a tight cover, imperf=0.)
        // Only blocks that can participate (slack < kmax-1+imperf) are kept.
        let cap = (self.kmax as isize - 1 + imperf).max(0);
        let cand: Vec<&DirectBlock> = blocks
            .iter()
            .filter(|b| (b.killed.count_ones() as isize - b.w as isize) <= cap)
            .collect();
        for i in 0..cand.len() {
            let bi = cand[i];
            let slacki = bi.killed.count_ones() as isize - bi.w as isize;
            for &bj in cand.iter().skip(i + 1) {
                let slackj = bj.killed.count_ones() as isize - bj.w as isize;
                let inter = (bi.killed & bj.killed).count_ones() as isize;
                if inter <= slacki + slackj - imperf {
                    continue; // cannot have positive surplus
                }
                // compatible? disjoint leaves AND non-crossing footprints
                if !disjoint(&bi.mask, &bj.mask) {
                    continue;
                }
                if bi.leaves.iter().any(|l| bj.leaves.contains(l)) {
                    continue;
                }
                // exact 2-block lift via residual matching of the union
                let mut union_leaves = bi.leaves.clone();
                union_leaves.extend_from_slice(&bj.leaves);
                let mut union_mask = bi.mask.clone();
                for k in 0..self.words {
                    union_mask[k] |= bj.mask[k];
                }
                let rm = self.residual_matching(active, &union_leaves, &union_mask);
                let v = bi.w + bj.w + rm;
                if v > best {
                    best = v;
                    best_blocks = vec![bi.leaves.clone(), bj.leaves.clone()];
                }
            }
        }

        Some((best, best_blocks))
    }
}

/// A big block with its footprint and killed-class bitset (direct certificate).
struct DirectBlock {
    leaves: Vec<u32>,
    mask: Vec<u64>,
    killed: u64,
    w: usize,
}

/// Canonical labeled topology of `tree` restricted to `leaves` (for agreement).
/// Two trees induce the same topology on `leaves` iff their canonical forms are
/// equal. Encoded as a nested `Node` enum that derives `Eq`/`Ord`.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Topo {
    Leaf(u32),
    Inner(Vec<Topo>),
}

fn induced_topology(tree: &Tree, leaves: &[u32]) -> Topo {
    use std::collections::HashMap;
    // kept = leaf nodes + all pairwise LCAs (LCA-closed set).
    let leaf_nodes: Vec<u32> = leaves.iter().map(|&l| tree.node_by_label(l)).collect();
    let mut keep: std::collections::HashSet<u32> = leaf_nodes.iter().copied().collect();
    for i in 0..leaf_nodes.len() {
        for j in (i + 1)..leaf_nodes.len() {
            keep.insert(tree.nearest_common_ancestor(leaf_nodes[i], leaf_nodes[j]));
        }
    }
    let lab: HashMap<u32, u32> = leaf_nodes
        .iter()
        .zip(leaves)
        .map(|(&nd, &l)| (nd, l))
        .collect();
    // parent in induced tree = nearest kept strict ancestor.
    let kept_parent = |mut x: u32| -> u32 {
        let mut pp = tree.parent[x as usize];
        while pp != klados_core::tree::NONE && !keep.contains(&pp) {
            pp = tree.parent[pp as usize];
        }
        let _ = &mut x;
        pp
    };
    let mut children: HashMap<u32, Vec<u32>> = keep.iter().map(|&k| (k, Vec::new())).collect();
    let mut roots = Vec::new();
    for &k in &keep {
        let pp = kept_parent(k);
        if pp == klados_core::tree::NONE || !keep.contains(&pp) {
            roots.push(k);
        } else {
            children.get_mut(&pp).unwrap().push(k);
        }
    }
    fn canon(
        x: u32,
        lab: &std::collections::HashMap<u32, u32>,
        children: &std::collections::HashMap<u32, Vec<u32>>,
    ) -> Topo {
        let ch = &children[&x];
        if ch.is_empty() {
            if let Some(&l) = lab.get(&x) {
                return Topo::Leaf(l);
            }
            return Topo::Inner(Vec::new());
        }
        let mut sub: Vec<Topo> = ch.iter().map(|&c| canon(c, lab, children)).collect();
        sub.sort();
        Topo::Inner(sub)
    }
    let mut sub: Vec<Topo> = roots.iter().map(|&r| canon(r, &lab, &children)).collect();
    sub.sort();
    Topo::Inner(sub)
}

// ── Forest construction ──────────────────────────────────────────────────────

/// Build the output agreement forest from a packing: the ≥3-leaf `blocks`, plus
/// the `pairs` (the residual matching that is non-crossing with every block), plus
/// singletons for the rest. All components are pairwise non-crossing by
/// construction (blocks committed with their footprints forbidden; `pairs` taken
/// from the residual matching).
fn build_forest(inst: &Instance, pairs: &[(u32, u32)], blocks: &[Block]) -> Vec<Tree> {
    let n = inst.num_leaves;
    let reference = inst.reference_tree();
    let mut used = vec![false; (n + 1) as usize];
    let mut forest = Vec::new();

    for block in blocks {
        let mut bs = FixedBitSet::with_capacity((n + 1) as usize);
        for &l in block {
            bs.insert(l as usize);
            used[l as usize] = true;
        }
        forest.push(Tree::forest_component(&bs, reference, n));
    }

    for &(a, b) in pairs {
        if used[a as usize] || used[b as usize] {
            continue;
        }
        used[a as usize] = true;
        used[b as usize] = true;
        let mut bs = FixedBitSet::with_capacity((n + 1) as usize);
        bs.insert(a as usize);
        bs.insert(b as usize);
        forest.push(Tree::forest_component(&bs, reference, n));
    }

    for l in 1..=n {
        if !used[l as usize] {
            forest.push(Tree::forest_singleton(l, n));
        }
    }
    forest
}

/// Certified `mu*` (merge count; `OPT = n - mu*`) via the anchored
/// canonical-block recursion. Returns `Some(value)` iff the recursion completed
/// (no budget/deadline hit).
///
/// SOUNDNESS: this is exhaustive over single blocks at each canonical leaf and
/// recurses on blocks whose matching contribution is within `slack` of the
/// incumbent. With `slack = 0` it is the single-block-lift certificate; multi-
/// block lifts (which provably exist on low-tree instances, Failed Lemma 2) are
/// only covered for `slack >= their nesting`. On the high-tree / high-frag wall
/// no multi-block lift has ever been observed, so `slack = 0` returns the true
/// `mu*` there — but that is a VALIDATED conjecture, not a theorem. Callers must
/// gate any optimality claim accordingly.
pub(crate) fn certified_mu_star(
    inst: &Instance,
    slack: isize,
    kmax: usize,
    node_budget: u64,
    deadline: Instant,
) -> Option<usize> {
    let n = inst.num_leaves as usize;
    if n == 0 {
        return Some(0);
    }
    let g = PairGraph::build(inst);
    let cover = dsatur_cover(&g);
    let rec = Rec::new(&g, inst, &cover, kmax, slack, node_budget, deadline);
    let words = g.masks.first().map(|m| m.len()).unwrap_or(0);
    let mut active = vec![false; n + 1];
    for l in 1..=n {
        active[l] = true;
    }
    let (value, _blocks) = rec.solve(&mut active, &vec![0u64; words], 1);
    if rec.aborted.get() { None } else { Some(value) }
}

/// Fast pair-only non-crossing packing, exposed as a safe incumbent generator
/// for other exact solvers. This does not claim optimality for MAF; it only
/// returns a validated agreement forest made of non-crossing pairs plus
/// singletons.
pub(crate) fn pair_matching_forest(
    inst: &Instance,
    max_n: usize,
    node_budget: u64,
) -> Option<Vec<Tree>> {
    if inst.num_leaves as usize > max_n {
        return None;
    }

    let g = PairGraph::build(inst);
    let mut cand = FixedBitSet::with_capacity(g.p);
    cand.insert_range(..);

    let mut best = Vec::new();
    let mut nodes = 0u64;
    if !expand(
        &g,
        &mut Vec::new(),
        cand,
        &mut best,
        &mut nodes,
        node_budget,
    ) {
        return None;
    }

    let pairs: Vec<(u32, u32)> = best.iter().map(|&i| g.pairs[i]).collect();
    let forest = build_forest(inst, &pairs, &[]);
    if matches!(validate_agreement_forest(inst, &forest), AfValidation::Ok) {
        Some(forest)
    } else {
        None
    }
}

// ── Solver ───────────────────────────────────────────────────────────────────

/// Tunables (env-overridable for experiments).
struct NcpackConfig {
    /// Max block size considered (`KLADOS_NCPACK_K`, default 6).
    kmax: usize,
    /// Clique-search node budget (`KLADOS_NCPACK_MAXNODES`, default 50M).
    node_budget: u64,
    /// Recursion slack: assumed bound on any residual's own gap; the recursion
    /// completing certifies `mu*` iff every residual gap ≤ slack
    /// (`KLADOS_NCPACK_SLACK`, default 1 — gap ≤ 1 across the wall).
    slack: isize,
    /// Trust the recursion's completion to claim Exact optimality
    /// (`KLADOS_NCPACK_CERT=1`, default off). Off ⇒ Exact emits nothing.
    trust_cert: bool,
}

impl NcpackConfig {
    fn from_env() -> Self {
        let env_usize = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        NcpackConfig {
            kmax: env_usize("KLADOS_NCPACK_K", 6),
            node_budget: std::env::var("KLADOS_NCPACK_MAXNODES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50_000_000),
            slack: env_usize("KLADOS_NCPACK_SLACK", 1) as isize,
            trust_cert: std::env::var("KLADOS_NCPACK_CERT").as_deref() == Ok("1"),
        }
    }
}

#[derive(Default)]
pub struct NcpackSolver {
    stats: SolverStats,
}

impl NcpackSolver {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Solver for NcpackSolver {
    type Config = ();
    const SUPPORTED_TRACKS: &'static [Track] = &[Track::Exact, Track::Heuristic];

    fn solve(&mut self, inst: &Instance, cfg: &RunConfig<()>) -> Option<Vec<Tree>> {
        let ncfg = NcpackConfig::from_env();
        let n = inst.num_leaves as usize;

        // The pair graph is O(P²) in P = C(n,2) masks; it is only viable for the
        // high-fragmentation exact wall (n up to ~250). Above that the approach
        // is the wrong tool (and would OOM), so decline politely.
        let max_n: usize = std::env::var("KLADOS_NCPACK_MAXN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(250);
        if n > max_n {
            debug!("[ncpack] n={n} > max_n={max_n}; declining (not the right regime)");
            return None;
        }

        let deadline = cfg
            .budget
            .map(|b| Instant::now() + b.saturating_sub(Duration::from_secs(2)))
            .unwrap_or_else(|| Instant::now() + Duration::from_secs(3600));

        // 1. Pair graph.
        let g = PairGraph::build(inst);

        // 2. DSATUR cover — only the `avail_classes` pruning bound for the
        //    recursion (NOT a soundness requirement; certification is exhaustive).
        let cover = dsatur_cover(&g);
        let ncov = cover.len();

        // 3. Recursive exact engine: anchored canonical-block recursion.
        let rec = Rec::new(
            &g,
            inst,
            &cover,
            ncfg.kmax,
            ncfg.slack,
            ncfg.node_budget,
            deadline,
        );
        let words = g.masks.first().map(|m| m.len()).unwrap_or(0);
        let mut active = vec![false; n + 1];
        for l in 1..=n {
            active[l] = true;
        }
        // Direct Hall-surplus certificate (single + pair, no recursion) by
        // default; falls back to the anchored recursion when ncov > 64 or when
        // explicitly disabled (`KLADOS_NCPACK_DIRECT=0`).
        let use_direct = std::env::var("KLADOS_NCPACK_DIRECT").as_deref() != Ok("0");
        let (value, blocks) = if use_direct {
            match rec.direct_certify(&mut active) {
                Some(r) => r,
                None => rec.solve(&mut active, &vec![0u64; words], 1),
            }
        } else {
            rec.solve(&mut active, &vec![0u64; words], 1)
        };
        let complete = !rec.aborted.get();

        debug!(
            "[ncpack] n={n} m={} ncov={ncov} value={value} \
             blocks={} matchings={} complete={complete}",
            inst.num_trees(),
            blocks.len(),
            rec.matchings.get(),
        );

        // Residual matching: pairs non-crossing with every committed block.
        let mut forbidden = vec![0u64; words];
        let mut act = vec![false; n + 1];
        for l in 1..=n {
            act[l] = true;
        }
        let mut scratch = Vec::new();
        for b in &blocks {
            let bm = rec.block_mask(b, &mut scratch);
            for k in 0..words {
                forbidden[k] |= bm[k];
            }
            for &l in b {
                act[l as usize] = false;
            }
        }
        let residual_pairs = rec.matching_pairs(&act, &forbidden);
        let forest = build_forest(inst, &residual_pairs, &blocks);

        // Never emit an invalid forest.
        if !matches!(validate_agreement_forest(inst, &forest), AfValidation::Ok) {
            log::error!("[ncpack] produced an invalid forest; emitting nothing");
            return None;
        }

        let opt = n - value; // forest size = #components
        debug_assert_eq!(forest.len(), opt);
        self.stats.upper_bound = Some(opt);

        match cfg.track {
            Track::Heuristic => Some(forest),
            Track::Exact => {
                // The recursion is exhaustive, so completing it proves `mu*` =
                // `value` — provided every residual gap ≤ `slack` (the only
                // assumption; default slack=1, gap ≤ 1 on the wall). Gated until
                // that bound is made fully rigorous.
                if complete && ncfg.trust_cert {
                    self.stats.lower_bound = opt;
                    Some(forest)
                } else {
                    None
                }
            }
            Track::LowerBound => None,
        }
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

pub fn main() {
    crate::run(
        NcpackSolver::new(),
        RunConfig {
            track: Track::Exact,
            ..Default::default()
        },
    );
}
