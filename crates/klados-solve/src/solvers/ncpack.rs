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
        expand(
            self.g,
            &mut Vec::new(),
            cs,
            &mut best,
            &mut nodes,
            self.node_budget,
        );
        self.matchings.set(self.matchings.get() + 1);
        best.len()
    }

    /// Max non-crossing matching among available pairs (the pairs themselves).
    fn matching_pairs(&self, active: &[bool], forbidden: &[u64]) -> Vec<(u32, u32)> {
        let cs = self.candset(active, forbidden);
        let mut best = Vec::new();
        let mut nodes = 0u64;
        expand(
            self.g,
            &mut Vec::new(),
            cs,
            &mut best,
            &mut nodes,
            self.node_budget,
        );
        best.iter().map(|&i| self.g.pairs[i]).collect()
    }

    /// Upper bound on the residual matching `M(active\B, forbidden∪mask(B))`:
    /// the number of top-cover classes that still contain an available residual
    /// pair (each class is a clique, contributing ≤ 1 to any matching).
    fn avail_classes(
        &self,
        active: &[bool],
        forbidden: &[u64],
        bmask: &[u64],
        bleaves: &[u32],
    ) -> usize {
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
                self.grow(
                    active,
                    forbidden,
                    &mut leaves,
                    s,
                    &mut scratch,
                    &mut best,
                    &mut best_blocks,
                );
                leaves.pop();
                if self.aborted.get() {
                    break;
                }
            }
        }
        (best, best_blocks)
    }

    /// Solve, deepening `slack` until the search is provably exhaustive, then
    /// `kmax` until no optimal block sits at the size cap. Returns
    /// `(value, blocks, certified)`.
    ///
    /// The recursion only ever builds real forests, so its value is always a valid
    /// LOWER bound on `mu*`, non-decreasing in `slack` and `kmax`. The old fixed
    /// `slack = 1` was UNSOUND: its prune (`w + avail_classes + slack > best`) uses
    /// a bound on the residual *matching*, not the residual `mu*`, so a residual
    /// whose block-gap exceeded `slack` had a beneficial block skipped and `mu*`
    /// under-reported (pub042: 20 vs ≥21).
    ///
    /// SOUND certificate: a completed solve at slack `s` returning `v` is
    /// *exhaustive* (no prune ever fired) iff `s > v`. Every ≥3-leaf block has
    /// `w ≥ 2`, so the prune term `w + av + s ≥ s + 2`; once `s > v ≥ best` it can
    /// never reach `≤ best`, so nothing is pruned and `v` is the exact `mu*` for
    /// blocks up to `kmax`. We deepen `slack` past the value to reach that state.
    /// (Exactness is then modulo `kmax`: we bump `kmax` whenever an optimal block
    /// sits at the cap; if none does, larger blocks are assumed unhelpful — the
    /// one remaining heuristic, inherited from the original fixed `kmax`.)
    ///
    /// Exhaustive solves can be slow on large instances (the prune is what made
    /// them fast); this is cheap precisely on small instances — which is what the
    /// decomposition produces. The deadline bounds the cost: on timeout it returns
    /// the best primal found with `certified = false`.
    fn solve_fixpoint(&mut self, n: usize, words: usize) -> (usize, Vec<Block>, bool) {
        let zero = vec![0u64; words];
        let mut active = vec![false; n + 1];
        let kmax_cap = self.kmax + 8;
        let mut best_v = 0usize;
        let mut best_blocks: Vec<Block> = Vec::new();
        loop {
            for l in 1..=n {
                active[l] = true;
            }
            let (v, blocks) = self.solve(&mut active, &zero, 1);
            if v >= best_v {
                best_v = v;
                best_blocks = blocks.clone();
            }
            if self.aborted.get() {
                return (best_v, best_blocks, false); // deadline → best primal, uncertified
            }
            let used_kmax = blocks.iter().any(|b| b.len() >= self.kmax);
            if used_kmax && self.kmax < kmax_cap {
                self.kmax += 1; // an optimal block hit the cap; allow bigger blocks
                continue;
            }
            // Exhaustive iff slack strictly exceeded the value: certify.
            if self.slack > best_v as isize && !used_kmax {
                return (best_v, best_blocks, true);
            }
            if self.kmax >= kmax_cap {
                return (best_v, best_blocks, false); // give up the kmax-side proof
            }
            // Make the next solve exhaustive: push slack past the current value.
            self.slack = best_v as isize + 1;
        }
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

/// DECOMPOSITION PROBE. Given a near-optimal forest (its ≥3 blocks + residual
/// pairs; everything else is a singleton), measure for every clade of every input
/// tree how many forest components straddle the clade boundary (the "interface" =
/// the number of components that would have to be branched over to split the
/// instance at that cut). Reports the minimum interface over all clades and over
/// *balanced* clades (n/4 ≤ |clade| ≤ 3n/4), plus the interface histogram.
///
/// interface(clade C) = #{ component K : K ∩ C ≠ ∅ and K ⊄ C }.
/// A clean cut (interface 0) that is not a common clade would give an exact,
/// non-trivial recursive decomposition.
fn interface_probe(
    inst: &Instance,
    nn: usize,
    blocks: &[Block],
    pairs: &[(u32, u32)],
    value: usize,
) {
    let n = nn as u32;
    // comp_of[leaf] = component id (0-based); singletons get unique ids.
    let mut comp_of = vec![usize::MAX; nn + 1];
    let mut next_id = 0usize;
    let mut comp_size: Vec<usize> = Vec::new();
    for b in blocks {
        for &l in b {
            comp_of[l as usize] = next_id;
        }
        comp_size.push(b.len());
        next_id += 1;
    }
    for &(a, b) in pairs {
        if comp_of[a as usize] != usize::MAX || comp_of[b as usize] != usize::MAX {
            continue;
        }
        comp_of[a as usize] = next_id;
        comp_of[b as usize] = next_id;
        comp_size.push(2);
        next_id += 1;
    }
    for l in 1..=nn {
        if comp_of[l] == usize::MAX {
            comp_of[l] = next_id;
            comp_size.push(1);
            next_id += 1;
        }
    }
    let ncomp = next_id;

    // For each tree, compute the leaf set (over labels 1..=n) under every node by
    // walking each leaf up to the root.
    let mut best_all = usize::MAX;
    let mut best_bal = usize::MAX;
    let mut best_bal_size = 0usize;
    let mut hist = [0usize; 6]; // interface 0,1,2,3,4,>=5 over balanced clades
    let mut n_balanced = 0usize;
    // Best clean (interface-0) cut by mu* balance: maximize min(mu_in, mu_out).
    let mut best_split = (0usize, value); // (min_side_merges, recorded as (lo,hi))
    let mut best_split_score = -1i64;
    let total_merges = value; // mu* = sum(size-1) over all comps

    for tree in &inst.trees {
        let nnodes = tree.num_nodes();
        // leaves_under[node] as a Vec<bool> over labels.
        let mut under: Vec<FixedBitSet> = vec![FixedBitSet::with_capacity(nn + 1); nnodes];
        for lbl in 1..=n {
            let leaf = tree.node_by_label(lbl);
            let mut u = leaf;
            loop {
                under[u as usize].insert(lbl as usize);
                if tree.is_root(u) {
                    break;
                }
                u = tree.parent[u as usize];
            }
        }
        for node in 0..nnodes as u32 {
            if tree.is_leaf(node) || tree.is_root(node) {
                continue; // trivial clades
            }
            let clade = &under[node as usize];
            let csize = clade.count_ones(..);
            if csize <= 1 || csize >= nn {
                continue;
            }
            // count, per component, how many of its leaves are inside the clade
            let mut inside_cnt: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for lbl in clade.ones() {
                *inside_cnt.entry(comp_of[lbl]).or_insert(0) += 1;
            }
            let mut interface = 0usize;
            let mut merges_in = 0usize; // sum(size-1) of comps fully inside C
            for (&cid, &cnt) in &inside_cnt {
                if cnt < comp_size[cid] {
                    interface += 1; // component has leaves both in and out
                } else {
                    merges_in += comp_size[cid] - 1; // comp fully inside C
                }
            }
            if interface < best_all {
                best_all = interface;
            }
            if interface == 0 {
                let merges_out = total_merges - merges_in;
                let score = merges_in.min(merges_out) as i64;
                if score > best_split_score {
                    best_split_score = score;
                    best_split = (merges_in.min(merges_out), merges_in.max(merges_out));
                }
            }
            let balanced = csize >= nn / 4 && csize <= 3 * nn / 4;
            if balanced {
                n_balanced += 1;
                hist[interface.min(5)] += 1;
                if interface < best_bal {
                    best_bal = interface;
                    best_bal_size = csize;
                }
            }
        }
    }

    debug!(
        "[ncpack-INTERFACE] n={n} m={} mu*={value} ncomp={ncomp} \
         min_interface_any_clade={best_all} \
         min_interface_balanced={best_bal} (|clade|={best_bal_size}) \
         balanced_clades={n_balanced} hist[0..5,>=5]={hist:?} \
         best_clean_merge_split={best_split:?}/mu{total_merges}",
        inst.num_trees(),
    );
}

/// Find the leaf set of the best clean (interface-0) balanced tree-clade w.r.t.
/// the ncpack forest, maximizing `min(merges_in, merges_out)`. Returns a
/// 1-indexed `Vec<bool>` membership mask, or `None` if no clean balanced clade.
fn best_clean_balanced_clade(
    inst: &Instance,
    nn: usize,
    comp_of: &[usize],
    comp_size: &[usize],
    total_merges: usize,
) -> Option<Vec<bool>> {
    let n = nn as u32;
    let mut best_in_c: Option<Vec<bool>> = None;
    let mut best_score = -1i64;
    for tree in &inst.trees {
        let nnodes = tree.num_nodes();
        let mut under: Vec<FixedBitSet> = vec![FixedBitSet::with_capacity(nn + 1); nnodes];
        for lbl in 1..=n {
            let mut u = tree.node_by_label(lbl);
            loop {
                under[u as usize].insert(lbl as usize);
                if tree.is_root(u) {
                    break;
                }
                u = tree.parent[u as usize];
            }
        }
        for node in 0..nnodes as u32 {
            if tree.is_leaf(node) || tree.is_root(node) {
                continue;
            }
            let clade = &under[node as usize];
            let csize = clade.count_ones(..);
            if csize < nn / 4 || csize > 3 * nn / 4 {
                continue;
            }
            let mut inside_cnt: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for lbl in clade.ones() {
                *inside_cnt.entry(comp_of[lbl]).or_insert(0) += 1;
            }
            let mut interface = 0usize;
            let mut merges_in = 0usize;
            for (&cid, &cnt) in &inside_cnt {
                if cnt < comp_size[cid] {
                    interface += 1;
                } else {
                    merges_in += comp_size[cid] - 1;
                }
            }
            if interface != 0 {
                continue;
            }
            let merges_out = total_merges - merges_in;
            let score = merges_in.min(merges_out) as i64;
            if score > best_score {
                best_score = score;
                let mut mask = vec![false; nn + 1];
                for lbl in clade.ones() {
                    mask[lbl] = true;
                }
                best_in_c = Some(mask);
            }
        }
    }
    best_in_c
}

/// Find the leaf set `C` of the best clean (interface-0) balanced tree-clade
/// w.r.t. ncpack's near-optimal pair-matching forest, as a 1-indexed
/// [`FixedBitSet`]. `None` when there's no clean balanced clade (or the forest
/// can't be built within `node_budget`).
///
/// This is the cut feeding the clean-cut lower-bound rank rows
/// (`clean-cut-lower-bound-spec.md`). The theorem `|F| ≥ OPT(C)+OPT(Cᶜ) − s`
/// holds for ANY cut, so soundness never depends on the cut's quality — a
/// "clean" (low-interface) balanced cut just yields a tighter bound.
pub(crate) fn clean_cut_leaves(inst: &Instance, node_budget: u64) -> Option<FixedBitSet> {
    let nn = inst.num_leaves as usize;
    let forest = pair_matching_forest(inst, nn, node_budget)?;
    // comp_of / comp_size / total_merges (mu*) from the forest components.
    let mut comp_of = vec![usize::MAX; nn + 1];
    let mut comp_size: Vec<usize> = Vec::with_capacity(forest.len());
    let mut total_merges = 0usize;
    for comp in &forest {
        let leaves: Vec<u32> = comp.leaves().collect();
        if leaves.is_empty() {
            continue;
        }
        let cid = comp_size.len();
        for &l in &leaves {
            comp_of[l as usize] = cid;
        }
        comp_size.push(leaves.len());
        total_merges += leaves.len() - 1;
    }
    // Any leaf the forest didn't cover becomes its own singleton (robustness).
    for l in 1..=nn {
        if comp_of[l] == usize::MAX {
            comp_of[l] = comp_size.len();
            comp_size.push(1);
        }
    }
    let in_c = best_clean_balanced_clade(inst, nn, &comp_of, &comp_size, total_merges)?;
    let mut bits = FixedBitSet::with_capacity(nn + 1);
    for l in 1..=nn {
        if in_c[l] {
            bits.insert(l);
        }
    }
    Some(bits)
}

/// BOUNDARY-ADJACENCY PROBE (`KLADOS_NCPACK_BOUNDARY=1`). For the best clean
/// balanced cut `C`, count how many active (non-singleton) components are
/// "boundary-adjacent": their Steiner tree reaches the cut's mixed region in some
/// tree. A component `K ⊆ C` can only conflict with a `Cᶜ`-component (the source
/// of the coupling defect γ) if its LCA in some tree is a *mixed* node (subtree
/// holds both a C- and a Cᶜ-leaf); a component buried in a monochromatic region
/// can never cross to the other side. Conjecture 4: this count is a small
/// constant on the wall, so γ = O(1). No recursion ⇒ runs on every instance.
fn boundary_probe(inst: &Instance, nn: usize, blocks: &[Block], pairs: &[(u32, u32)]) {
    let n = nn as u32;
    // Active components (size ≥ 2) as leaf-label bitsets.
    let mut comps: Vec<Vec<u32>> = blocks.to_vec();
    for &(a, b) in pairs {
        comps.push(vec![a, b]);
    }
    let ncomp_active = comps.len();
    let mut comp_bits: Vec<FixedBitSet> = Vec::with_capacity(ncomp_active);
    let mut comp_merge: Vec<usize> = Vec::with_capacity(ncomp_active);
    for k in &comps {
        let mut bs = FixedBitSet::with_capacity(nn + 1);
        for &l in k {
            bs.insert(l as usize);
        }
        comp_bits.push(bs);
        comp_merge.push(k.len() - 1);
    }
    let total_merges: usize = comp_merge.iter().sum();

    // Per (active comp, tree): the leaf set of the comp's LCA-clade in that tree
    // (under the LCA of the comp's leaves). A comp is "boundary-adjacent" w.r.t a
    // cut C iff one of its LCA-clades is bi-chromatic (holds a C- and a Cᶜ-leaf).
    // These are cut-independent, so we precompute them once.
    let m = inst.trees.len();
    let mut lca_clade: Vec<Vec<FixedBitSet>> = vec![Vec::with_capacity(m); ncomp_active];
    // Also collect all candidate clades (leaf sets under each internal node).
    let mut clades: Vec<FixedBitSet> = Vec::new();
    for tree in &inst.trees {
        let nnodes = tree.num_nodes();
        let mut under: Vec<FixedBitSet> = vec![FixedBitSet::with_capacity(nn + 1); nnodes];
        for lbl in 1..=n {
            let mut u = tree.node_by_label(lbl);
            loop {
                under[u as usize].insert(lbl as usize);
                if tree.is_root(u) {
                    break;
                }
                u = tree.parent[u as usize];
            }
        }
        for (ci, k) in comps.iter().enumerate() {
            let mut lca = tree.node_by_label(k[0]);
            for &lbl in &k[1..] {
                lca = tree.nearest_common_ancestor(lca, tree.node_by_label(lbl));
            }
            lca_clade[ci].push(under[lca as usize].clone());
        }
        for node in 0..nnodes as u32 {
            if tree.is_leaf(node) || tree.is_root(node) {
                continue;
            }
            let csize = under[node as usize].count_ones(..);
            // All non-trivial clades (≥2 leaves each side); balance is judged later.
            if csize >= 2 && csize <= nn - 2 {
                clades.push(under[node as usize].clone());
            }
        }
    }

    // Scan clean clades (no active comp straddles). Track two minima of the
    // boundary-adjacency (coupling) count:
    //   - best_adj_bal: over LEAF-balanced cuts (|C| in [n/4, 3n/4]);
    //   - best_adj_split: over any cut that MEANINGFULLY splits merges
    //     (mu_in ≥ 2 and mu_out ≥ 2), i.e. a real decomposition (possibly
    //     leaf-unbalanced peel).
    let mut best_adj_bal = usize::MAX;
    let mut best_split_bal = (0usize, 0usize);
    let mut best_adj_split = usize::MAX;
    let mut best_split_split = (0usize, 0usize);
    let mut n_clean = 0usize;
    let mut tmp = FixedBitSet::with_capacity(nn + 1);
    for c in &clades {
        // clean: every active comp ⊆ C or disjoint from C.
        let mut clean = true;
        let mut merges_in = 0usize;
        for (ci, kb) in comp_bits.iter().enumerate() {
            tmp.clone_from(kb);
            tmp.intersect_with(c);
            let inter = tmp.count_ones(..);
            if inter == 0 {
                continue; // disjoint → outside
            }
            if inter != kb.count_ones(..) {
                clean = false;
                break;
            }
            merges_in += comp_merge[ci]; // comp fully inside C
        }
        if !clean {
            continue;
        }
        let merges_out = total_merges - merges_in;
        let csize = c.count_ones(..);
        let balanced = csize >= nn / 4 && csize <= 3 * nn / 4;
        let meaningful = merges_in >= 2 && merges_out >= 2;
        if !balanced && !meaningful {
            continue;
        }
        n_clean += 1;
        // boundary-adjacency: #active comps with a bi-chromatic LCA-clade.
        let mut adj = 0usize;
        for ci in 0..ncomp_active {
            let mut is_adj = false;
            for clade_kt in &lca_clade[ci] {
                tmp.clone_from(clade_kt);
                tmp.intersect_with(c);
                let inter = tmp.count_ones(..);
                let tot = clade_kt.count_ones(..);
                if inter > 0 && inter < tot {
                    is_adj = true;
                    break;
                }
            }
            if is_adj {
                adj += 1;
            }
        }
        let split = (merges_in.min(merges_out), merges_in.max(merges_out));
        if balanced && adj < best_adj_bal {
            best_adj_bal = adj;
            best_split_bal = split;
        }
        if meaningful && adj < best_adj_split {
            best_adj_split = adj;
            best_split_split = split;
        }
    }

    if n_clean == 0 {
        debug!("[ncpack-BOUNDARY] n={nn} m={m} mu*={total_merges} no clean balanced clade");
        return;
    }
    debug!(
        "[ncpack-BOUNDARY] n={nn} m={m} mu*={total_merges} active_comps={ncomp_active} \
         clean_cuts={n_clean} min_coupling_balanced={best_adj_bal}@{best_split_bal:?} \
         min_coupling_anysplit={best_adj_split}@{best_split_split:?}",
    );
}

/// Per-node analysis for the hierarchy probe: build the near-optimal forest,
/// report local μ* and (if one exists) the min-coupling clean cut that
/// MEANINGFULLY splits merges (both sides keep ≥1 merge ⇒ μ* strictly drops).
/// Returns `(mu_star, Some((cut_leaves, coupling_width)))`, or `..None` if the
/// node cannot be cleanly split (the DP would solve it directly at cost exp(μ*)).
fn hier_analyze(
    inst: &Instance,
    node_budget: u64,
) -> Option<(usize, Option<(FixedBitSet, usize)>)> {
    let nn = inst.num_leaves as usize;
    let n = nn as u32;
    let forest = pair_matching_forest(inst, nn, node_budget)?;

    let mut comps: Vec<Vec<u32>> = Vec::new();
    for comp in &forest {
        let leaves: Vec<u32> = comp.leaves().collect();
        if leaves.len() >= 2 {
            comps.push(leaves);
        }
    }
    let total_merges: usize = comps.iter().map(|k| k.len() - 1).sum();
    if total_merges == 0 {
        return Some((0, None));
    }
    let ncomp = comps.len();
    let mut comp_bits: Vec<FixedBitSet> = Vec::with_capacity(ncomp);
    let mut comp_merge: Vec<usize> = Vec::with_capacity(ncomp);
    for k in &comps {
        let mut bs = FixedBitSet::with_capacity(nn + 1);
        for &l in k {
            bs.insert(l as usize);
        }
        comp_bits.push(bs);
        comp_merge.push(k.len() - 1);
    }

    let m = inst.trees.len();
    let mut lca_clade: Vec<Vec<FixedBitSet>> = vec![Vec::with_capacity(m); ncomp];
    let mut clades: Vec<FixedBitSet> = Vec::new();
    for tree in &inst.trees {
        let nnodes = tree.num_nodes();
        let mut under: Vec<FixedBitSet> = vec![FixedBitSet::with_capacity(nn + 1); nnodes];
        for lbl in 1..=n {
            let mut u = tree.node_by_label(lbl);
            loop {
                under[u as usize].insert(lbl as usize);
                if tree.is_root(u) {
                    break;
                }
                u = tree.parent[u as usize];
            }
        }
        for (ci, k) in comps.iter().enumerate() {
            let mut lca = tree.node_by_label(k[0]);
            for &lbl in &k[1..] {
                lca = tree.nearest_common_ancestor(lca, tree.node_by_label(lbl));
            }
            lca_clade[ci].push(under[lca as usize].clone());
        }
        for node in 0..nnodes as u32 {
            if tree.is_leaf(node) || tree.is_root(node) {
                continue;
            }
            let csize = under[node as usize].count_ones(..);
            if csize >= 2 && csize <= nn - 2 {
                clades.push(under[node as usize].clone());
            }
        }
    }

    let mut best: Option<(FixedBitSet, usize)> = None;
    let mut tmp = FixedBitSet::with_capacity(nn + 1);
    for c in &clades {
        let mut clean = true;
        let mut merges_in = 0usize;
        for (ci, kb) in comp_bits.iter().enumerate() {
            tmp.clone_from(kb);
            tmp.intersect_with(c);
            let inter = tmp.count_ones(..);
            if inter == 0 {
                continue;
            }
            if inter != kb.count_ones(..) {
                clean = false;
                break;
            }
            merges_in += comp_merge[ci];
        }
        if !clean {
            continue;
        }
        let merges_out = total_merges - merges_in;
        if merges_in < 1 || merges_out < 1 {
            continue; // not a meaningful merge-split
        }
        let mut adj = 0usize;
        for ci in 0..ncomp {
            let mut is_adj = false;
            for clade_kt in &lca_clade[ci] {
                tmp.clone_from(clade_kt);
                tmp.intersect_with(c);
                let inter = tmp.count_ones(..);
                let tot = clade_kt.count_ones(..);
                if inter > 0 && inter < tot {
                    is_adj = true;
                    break;
                }
            }
            if is_adj {
                adj += 1;
            }
        }
        match &best {
            None => best = Some((c.clone(), adj)),
            Some((_, bw)) if adj < *bw => best = Some((c.clone(), adj)),
            _ => {}
        }
    }
    Some((total_merges, best))
}

/// HIERARCHY PROBE (`KLADOS_NCPACK_HIERARCHY=1`). The make-or-break measurement
/// for the coupling-width DP. The boundary probe measured only the TOP cut; the
/// DP needs coupling bounded along the WHOLE branch decomposition. This bisects
/// recursively by min-coupling clean merge-splits and reports, over every node:
/// the max coupling width at split nodes and the max residual μ* at "stuck"
/// nodes (μ* > BASE_MU but no clean split). The DP is feasible on an instance
/// iff BOTH stay small (table ~ exp(coupling); base cost ~ exp(stuck μ*)).
fn hierarchy_probe(inst: &Instance, node_budget: u64) {
    const BASE_MU: usize = 4; // exp(BASE_MU) solved directly ⇒ cheap base case
    const MAX_NODES: usize = 4_000;

    let mut max_coupling = 0usize;
    let mut max_stuck_mu = 0usize;
    let mut couplings: Vec<usize> = Vec::new();
    let mut stuck_mus: Vec<usize> = Vec::new();
    let (mut n_split, mut n_base, mut n_stuck, mut max_depth, mut total_nodes) =
        (0usize, 0usize, 0usize, 0usize, 0usize);

    // Cap the per-node clique budget hard: the probe only needs a near-optimal
    // forest to define the cut/coupling, not the global best matching. A tiny
    // budget bounds each call so no single node can blow the wall-clock.
    let probe_budget = node_budget.min(20_000);
    let start = Instant::now();
    let mut truncated = false;
    let mut stack: Vec<(Instance, usize)> = vec![(inst.clone(), 0)];
    while let Some((cur, depth)) = stack.pop() {
        total_nodes += 1;
        if total_nodes > MAX_NODES || start.elapsed() > Duration::from_secs(45) {
            truncated = true;
            break;
        }
        max_depth = max_depth.max(depth);
        let nn = cur.num_leaves as usize;
        if nn < 4 {
            n_base += 1;
            continue;
        }
        let Some((mu, best)) = hier_analyze(&cur, probe_budget) else {
            n_base += 1;
            continue;
        };
        if mu <= BASE_MU {
            n_base += 1;
            continue;
        }
        let Some((cut, coupling)) = best else {
            n_stuck += 1;
            max_stuck_mu = max_stuck_mu.max(mu);
            stuck_mus.push(mu);
            continue;
        };
        n_split += 1;
        max_coupling = max_coupling.max(coupling);
        couplings.push(coupling);
        let (ci, _) = klados_core::kernelize::restrict_instance_simple(&cur, &cut);
        let mut ccbits = FixedBitSet::with_capacity(nn + 1);
        for l in 1..=nn {
            if !cut.contains(l) {
                ccbits.insert(l);
            }
        }
        let (cci, _) = klados_core::kernelize::restrict_instance_simple(&cur, &ccbits);
        stack.push((ci, depth + 1));
        stack.push((cci, depth + 1));
    }
    couplings.sort_unstable();
    stuck_mus.sort_unstable();
    log::info!(
        "[ncpack-HIERARCHY] n={} m={} nodes={}(split={} base={} stuck={}) depth={} \
         MAX_COUPLING={} MAX_STUCK_MU={} elapsed_ms={} truncated={} | couplings={:?} stuck_mus={:?}",
        inst.num_leaves,
        inst.trees.len(),
        total_nodes,
        n_split,
        n_base,
        n_stuck,
        max_depth,
        max_coupling,
        max_stuck_mu,
        start.elapsed().as_millis(),
        truncated,
        couplings,
        stuck_mus,
    );
}

/// T0-HIERARCHY PROBE (`KLADOS_NCPACK_T0HIER=1`). Measures the coupling width of
/// the *actual* fixed-reference-tree DP (every `C_u` a clade of `T_0`, so Lemma
/// 2.1 gives ≤1 open block). For each candidate reference tree `T_0`, over every
/// internal-node clade `C_u`, count boundary-active CLOSED blocks (footprint
/// nonempty but not straddling) = `W_dp`; report the min over reference-tree
/// choices. This is the parameter that bounds the DP table; the adaptive
/// hierarchy probe was only a proxy (its `X∖C` side is not a clade).
fn t0_hierarchy_probe(inst: &Instance, nn: usize) {
    let n = nn as u32;
    let Some(forest) = pair_matching_forest(inst, nn, 20_000) else {
        log::info!("[ncpack-T0HIER] n={nn}: no forest");
        return;
    };
    let mut comps: Vec<Vec<u32>> = Vec::new();
    for c in &forest {
        let l: Vec<u32> = c.leaves().collect();
        if l.len() >= 2 {
            comps.push(l);
        }
    }
    let total_merges: usize = comps.iter().map(|k| k.len() - 1).sum();
    let ncomp = comps.len();
    if ncomp == 0 {
        log::info!("[ncpack-T0HIER] n={nn} mu*=0 trivial");
        return;
    }
    let mut comp_bits: Vec<FixedBitSet> = Vec::with_capacity(ncomp);
    for k in &comps {
        let mut b = FixedBitSet::with_capacity(nn + 1);
        for &l in k {
            b.insert(l as usize);
        }
        comp_bits.push(b);
    }
    let m = inst.trees.len();
    let mut lca_clade: Vec<Vec<FixedBitSet>> = vec![Vec::with_capacity(m); ncomp];
    let mut under_per_tree: Vec<Vec<FixedBitSet>> = Vec::with_capacity(m);
    for tree in &inst.trees {
        let nnodes = tree.num_nodes();
        let mut under = vec![FixedBitSet::with_capacity(nn + 1); nnodes];
        for lbl in 1..=n {
            let mut u = tree.node_by_label(lbl);
            loop {
                under[u as usize].insert(lbl as usize);
                if tree.is_root(u) {
                    break;
                }
                u = tree.parent[u as usize];
            }
        }
        for (ci, k) in comps.iter().enumerate() {
            let mut lca = tree.node_by_label(k[0]);
            for &lbl in &k[1..] {
                lca = tree.nearest_common_ancestor(lca, tree.node_by_label(lbl));
            }
            lca_clade[ci].push(under[lca as usize].clone());
        }
        under_per_tree.push(under);
    }
    let mut tmp = FixedBitSet::with_capacity(nn + 1);
    let mut per_t_w: Vec<usize> = Vec::with_capacity(m);
    let mut max_interface = 0usize;
    for (ti, tree) in inst.trees.iter().enumerate() {
        let under = &under_per_tree[ti];
        let mut maxw = 0usize;
        for node in 0..tree.num_nodes() as u32 {
            if tree.is_leaf(node) || tree.is_root(node) {
                continue;
            }
            let c = &under[node as usize];
            let csize = c.count_ones(..);
            if csize < 2 || csize > nn - 2 {
                continue;
            }
            let mut interface = 0usize;
            for kb in &comp_bits {
                tmp.clone_from(kb);
                tmp.intersect_with(c);
                let i = tmp.count_ones(..);
                if i > 0 && i < kb.count_ones(..) {
                    interface += 1;
                }
            }
            max_interface = max_interface.max(interface);
            let mut adj = 0usize;
            for ci in 0..ncomp {
                let mut is = false;
                for ck in &lca_clade[ci] {
                    tmp.clone_from(ck);
                    tmp.intersect_with(c);
                    let it = tmp.count_ones(..);
                    let tot = ck.count_ones(..);
                    if it > 0 && it < tot {
                        is = true;
                        break;
                    }
                }
                if is {
                    adj += 1;
                }
            }
            // Exclude the ≤1 straddler (the open block, tracked separately).
            maxw = maxw.max(adj.saturating_sub(interface));
        }
        per_t_w.push(maxw);
    }
    let best_w = per_t_w.iter().copied().min().unwrap_or(usize::MAX);
    let worst_w = per_t_w.iter().copied().max().unwrap_or(0);
    log::info!(
        "[ncpack-T0HIER] n={nn} m={m} mu*={total_merges} ncomp={ncomp} \
         BEST_T0_W_dp={best_w} worst_T0_W={worst_w} max_interface(should_be<=1)={max_interface} \
         per_tree_W={per_t_w:?}",
    );
}

// ── Coupling-width DP state-counter (KLADOS_NCPACK_DPCOUNT=1) ─────────────────
// Faithful enumeration of the DP's reachable states, keyed by block leaf-sets
// (finer than footprints ⇒ an UPPER BOUND on the real table). If this stays
// small, the footprint-keyed DP table is ≤ it ⇒ the speed risk does not break
// the idea. Validated by checking the DP returns the true μ*.

fn dpc_steiner(tree: &Tree, leaves: &[u32]) -> FixedBitSet {
    let mut lca = tree.node_by_label(leaves[0]);
    for &l in &leaves[1..] {
        lca = tree.nearest_common_ancestor(lca, tree.node_by_label(l));
    }
    let mut bs = FixedBitSet::with_capacity(tree.num_nodes());
    for &l in leaves {
        let mut c = tree.node_by_label(l);
        loop {
            bs.insert(c as usize);
            if c == lca {
                break;
            }
            c = tree.parent[c as usize];
        }
    }
    bs
}

fn dpc_agrees(inst: &Instance, block: &[u32]) -> bool {
    if block.len() <= 2 {
        return true;
    }
    let set: std::collections::HashSet<u32> = block.iter().copied().collect();
    fn sig(tree: &Tree, set: &std::collections::HashSet<u32>, node: u32) -> Option<String> {
        if tree.is_leaf(node) {
            let l = tree.label[node as usize];
            return set.contains(&l).then(|| l.to_string());
        }
        let (a, b) = tree.children(node).unwrap();
        match (sig(tree, set, a), sig(tree, set, b)) {
            (None, None) => None,
            (Some(x), None) | (None, Some(x)) => Some(x),
            (Some(x), Some(y)) => Some(if x <= y {
                format!("({x},{y})")
            } else {
                format!("({y},{x})")
            }),
        }
    }
    let s0 = sig(&inst.trees[0], &set, inst.trees[0].root);
    inst.trees[1..].iter().all(|t| sig(t, &set, t.root) == s0)
}

fn dpc_noncross(inst: &Instance, blocks: &[&Vec<u32>]) -> bool {
    if blocks.len() < 2 {
        return true;
    }
    for tree in &inst.trees {
        let sts: Vec<FixedBitSet> = blocks.iter().map(|b| dpc_steiner(tree, b)).collect();
        for i in 0..sts.len() {
            for j in (i + 1)..sts.len() {
                if sts[i].intersection(&sts[j]).next().is_some() {
                    return false;
                }
            }
        }
    }
    true
}

/// Will the open block `open`, growing upward through pure-`C_u` ancestors,
/// cross block `b`? If so `b` cannot be retired (Lemma 6.1 only separates `b`
/// from blocks OUTSIDE `C_u`; the open block is inside `C_u` and its upward
/// growth traverses pure-`C_u` vertices that may lie on `b`'s Steiner tree).
fn dpc_open_ray_crosses(
    inst: &Instance,
    b: &[u32],
    open: &[u32],
    cu: &FixedBitSet,
    under: &[Vec<FixedBitSet>],
) -> bool {
    for (q, tree) in inst.trees.iter().enumerate() {
        let mut lca = tree.node_by_label(open[0]);
        for &l in &open[1..] {
            lca = tree.nearest_common_ancestor(lca, tree.node_by_label(l));
        }
        let stb = dpc_steiner(tree, b);
        let mut a = lca;
        loop {
            let clade = &under[q][a as usize];
            // stop once we leave the pure-C_u region (a is mixed or outside)
            if clade.intersection(cu).count() != clade.count_ones(..) {
                break;
            }
            if stb.contains(a as usize) {
                return true;
            }
            if tree.is_root(a) {
                break;
            }
            a = tree.parent[a as usize];
        }
    }
    false
}

fn dpc_boundary_active(
    inst: &Instance,
    block: &[u32],
    cu: &FixedBitSet,
    under: &[Vec<FixedBitSet>],
) -> bool {
    for (q, tree) in inst.trees.iter().enumerate() {
        let mut lca = tree.node_by_label(block[0]);
        for &l in &block[1..] {
            lca = tree.nearest_common_ancestor(lca, tree.node_by_label(l));
        }
        let clade = &under[q][lca as usize];
        let inter = clade.intersection(cu).count();
        let csz = clade.count_ones(..);
        if inter > 0 && inter < csz {
            return true;
        }
    }
    false
}

type DpcKey = (Vec<u32>, Vec<Vec<u32>>); // (open-or-empty, sorted closed-active)

/// Run the DP over reference tree `t0_idx`, counting states. Returns
/// `(max_table_size, mu_star)` or `None` if a table exceeds `cap` (RISK).
fn dpc_run_t0(
    inst: &Instance,
    t0_idx: usize,
    under: &[Vec<FixedBitSet>],
    cap: usize,
    deadline: Instant,
) -> Option<(usize, usize, usize, usize)> {
    let t0 = &inst.trees[t0_idx];
    let mut tables: Vec<std::collections::HashMap<DpcKey, usize>> =
        vec![std::collections::HashMap::new(); t0.num_nodes()];
    let mut max_table = 0usize;
    let mut max_fp = 0usize;
    let mut max_sound = 0usize;

    for node in t0.post_order() {
        if Instant::now() > deadline {
            return None;
        }
        if t0.is_leaf(node) {
            let x = t0.label[node as usize];
            let tbl = &mut tables[node as usize];
            tbl.insert((vec![x], vec![]), 0); // x open (will merge)
            tbl.insert((vec![], vec![]), 0); // x final singleton
            continue;
        }
        let (v, w) = t0.children(node).unwrap();
        let cu = &under[t0_idx][node as usize];
        // move children tables out
        let tv = std::mem::take(&mut tables[v as usize]);
        let tw = std::mem::take(&mut tables[w as usize]);
        let mut out: std::collections::HashMap<DpcKey, usize> = std::collections::HashMap::new();

        for ((ov, cv), &rv) in &tv {
            for ((ow, cw), &rw) in &tw {
                if Instant::now() > deadline {
                    return None;
                }
                let base_ret = rv + rw;
                // carried committed blocks
                let mut base_closed: Vec<Vec<u32>> = Vec::with_capacity(cv.len() + cw.len());
                base_closed.extend(cv.iter().cloned());
                base_closed.extend(cw.iter().cloned());

                // seam: (new_open, extra_committed)
                let ov_some = !ov.is_empty();
                let ow_some = !ow.is_empty();
                let mut candidates: Vec<(Vec<u32>, Vec<Vec<u32>>)> = Vec::new();
                if !ov_some && !ow_some {
                    candidates.push((vec![], vec![]));
                } else if ov_some && !ow_some {
                    candidates.push((ov.clone(), vec![]));
                } else if !ov_some && ow_some {
                    candidates.push((ow.clone(), vec![]));
                } else {
                    // both open ⇒ must merge (Lemma 2.1)
                    let mut b: Vec<u32> = ov.clone();
                    b.extend_from_slice(ow);
                    b.sort_unstable();
                    if dpc_agrees(inst, &b) {
                        candidates.push((b.clone(), vec![])); // B continues open
                        candidates.push((vec![], vec![b])); // B commits
                    }
                }

                for (new_open, extra) in candidates {
                    // all live blocks for the non-crossing check
                    let mut refs: Vec<&Vec<u32>> = base_closed.iter().chain(extra.iter()).collect();
                    if !new_open.is_empty() {
                        refs.push(&new_open);
                    }
                    if !dpc_noncross(inst, &refs) {
                        continue;
                    }
                    // retire committed blocks that can no longer interact:
                    // not boundary-active (Lemma 6.1, vs outside blocks) AND not
                    // on the open block's upward pure-C_u growth ray.
                    let mut ret = base_ret;
                    let mut closed: Vec<Vec<u32>> = Vec::new();
                    for b in base_closed.iter().chain(extra.iter()) {
                        let keep = b.len() >= 2
                            && (dpc_boundary_active(inst, b, cu, under)
                                || (!new_open.is_empty()
                                    && dpc_open_ray_crosses(inst, b, &new_open, cu, under)));
                        if keep {
                            closed.push(b.clone());
                        } else {
                            ret += b.len().saturating_sub(1);
                        }
                    }
                    closed.sort();
                    let key = (new_open, closed);
                    let e = out.entry(key).or_insert(0);
                    *e = (*e).max(ret);
                }
                if out.len() > cap {
                    return None;
                }
            }
        }
        max_table = max_table.max(out.len());
        // Footprint-keyed count: how far would the real (footprint) DP collapse
        // the leaf-set table? A LOWER bound on the sound footprint+agreement table.
        let clade = &under[t0_idx][node as usize];
        let block_fp = |b: &[u32]| -> u64 {
            let mut bh: u64 = 14695981039346656037;
            for (q, tree) in inst.trees.iter().enumerate() {
                for z in dpc_steiner(tree, b).ones() {
                    let zc = &under[q][z];
                    let inter = zc.intersection(clade).count();
                    if inter > 0 && inter < zc.count_ones(..) {
                        bh ^= (q as u64).wrapping_mul(1000003).wrapping_add(z as u64);
                        bh = bh.wrapping_mul(1099511628211);
                    }
                }
            }
            bh
        };
        let mut fp_keys: std::collections::HashSet<u64> = std::collections::HashSet::new();
        // SOUND key: open block by full leaves (it grows/merges → needs full info),
        // closed blocks by footprint (they don't grow; footprint decides crossing
        // & future retirement; merge counts are already folded into the value).
        let mut sound_keys: std::collections::HashSet<(Vec<u32>, Vec<u64>)> =
            std::collections::HashSet::new();
        for (open, closed) in out.keys() {
            let mut h: u64 = 1469598103934665603;
            if !open.is_empty() {
                h ^= block_fp(open).wrapping_add(0xABCD);
                h = h.wrapping_mul(1099511628211);
            }
            let mut cs: Vec<u64> = closed.iter().map(|b| block_fp(b)).collect();
            cs.sort_unstable();
            for c in &cs {
                h ^= *c;
                h = h.wrapping_mul(1099511628211);
            }
            fp_keys.insert(h);
            sound_keys.insert((open.clone(), cs));
        }
        max_fp = max_fp.max(fp_keys.len());
        max_sound = max_sound.max(sound_keys.len());
        tables[node as usize] = out;
    }

    // root: close any dangling open block; all blocks retire (C=X).
    let root = t0.root as usize;
    let mu = tables[root]
        .iter()
        .map(|((open, closed), &ret)| {
            let open_m = open.len().saturating_sub(1);
            let closed_m: usize = closed.iter().map(|b| b.len().saturating_sub(1)).sum();
            ret + open_m + closed_m
        })
        .max()
        .unwrap_or(0);
    Some((max_table, max_fp, max_sound, mu))
}

/// Test entry points for the primitive checks.
pub fn dpc_agrees_pub(inst: &Instance, block: &[u32]) -> bool {
    dpc_agrees(inst, block)
}
pub fn dpc_noncross_pub(inst: &Instance, blocks: &[&Vec<u32>]) -> bool {
    dpc_noncross(inst, blocks)
}

/// Test entry point: build `under` and run the DP for one reference tree,
/// returning `(max_table, mu_star)`.
pub fn dpc_mu_for_t0(
    inst: &Instance,
    t0: usize,
    cap: usize,
) -> Option<(usize, usize, usize, usize)> {
    let nn = inst.num_leaves as usize;
    let n = inst.num_leaves;
    let mut under: Vec<Vec<FixedBitSet>> = Vec::with_capacity(inst.trees.len());
    for tree in &inst.trees {
        let mut u = vec![FixedBitSet::with_capacity(nn + 1); tree.num_nodes()];
        for lbl in 1..=n {
            let mut c = tree.node_by_label(lbl);
            loop {
                u[c as usize].insert(lbl as usize);
                if tree.is_root(c) {
                    break;
                }
                c = tree.parent[c as usize];
            }
        }
        under.push(u);
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    dpc_run_t0(inst, t0, &under, cap, deadline)
}

/// DP STATE-COUNT PROBE (`KLADOS_NCPACK_DPCOUNT=1`).
fn dpcount_probe(inst: &Instance, nn: usize) {
    let n = nn as u32;
    let cap: usize = std::env::var("KLADOS_NCPACK_DPCAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    let deadline = Instant::now() + Duration::from_secs(120);
    // precompute under[tree][node] = leaf bitset
    let mut under: Vec<Vec<FixedBitSet>> = Vec::with_capacity(inst.trees.len());
    for tree in &inst.trees {
        let mut u = vec![FixedBitSet::with_capacity(nn + 1); tree.num_nodes()];
        for lbl in 1..=n {
            let mut c = tree.node_by_label(lbl);
            loop {
                u[c as usize].insert(lbl as usize);
                if tree.is_root(c) {
                    break;
                }
                c = tree.parent[c as usize];
            }
        }
        under.push(u);
    }

    let mut best: Option<(usize, usize, usize)> = None; // (max_table, mu, t0)
    let mut mus: Vec<usize> = Vec::new();
    let mut completed = 0usize;
    let mut best_fp: Option<usize> = None;
    let mut best_sound: Option<usize> = None;
    for t0 in 0..inst.trees.len() {
        match dpc_run_t0(inst, t0, &under, cap, deadline) {
            Some((mt, mfp, msound, mu)) => {
                completed += 1;
                mus.push(mu);
                if best.is_none() || mt < best.unwrap().0 {
                    best = Some((mt, mu, t0));
                }
                best_fp = Some(best_fp.map_or(mfp, |b| b.min(mfp)));
                best_sound = Some(best_sound.map_or(msound, |b| b.min(msound)));
            }
            None => {}
        }
        if Instant::now() > deadline {
            break;
        }
    }
    let mu_consistent = mus.iter().all(|&x| Some(x) == mus.iter().max().copied());
    {
        let mut sorted = mus.clone();
        sorted.sort_unstable();
        log::info!("[ncpack-DPCOUNT-mus] n={nn} per_t0_mu={sorted:?}");
    }
    match best {
        Some((mt, mu, t0)) => log::info!(
            "[ncpack-DPCOUNT] n={nn} m={} completed={completed}/{} leafset={mt} \
             fp_only(unsound_lb)={} SOUND_TABLE={} mu*={mu} (OPT={}) best_t0={t0} \
             mu_consistent={mu_consistent} cap={cap}",
            inst.trees.len(),
            inst.trees.len(),
            best_fp.unwrap_or(0),
            best_sound.unwrap_or(0),
            nn - mu,
        ),
        None => log::info!(
            "[ncpack-DPCOUNT] n={nn} m={} ALL CAPPED (>{cap}) — RISK CONFIRMED",
            inst.trees.len()
        ),
    }
}

/// COMBINE TEST (`KLADOS_NCPACK_COMBINE=1`). Pick a clean balanced clade `C`
/// w.r.t. the ncpack forest, solve `L|C` and `L|Cᶜ` INDEPENDENTLY with the exact
/// recursion (activating only one side's leaves), union the two sub-forests, and
/// check (a) the union is a valid AF and (b) its merge count equals the ncpack
/// value. This is the make-or-break check for independent clean-cut
/// decomposition: if `mu*(C) + mu*(Cᶜ) == mu*(L)` and the union validates, the
/// instance splits exactly; if not, independent solving over-merges and the
/// sound form is sequential conditioning (solve Cᶜ with C's footprint forbidden).
fn combine_test(
    rec: &Rec,
    inst: &Instance,
    nn: usize,
    words: usize,
    blocks: &[Block],
    pairs: &[(u32, u32)],
    value: usize,
) {
    // comp_of / comp_size from the ncpack forest.
    let mut comp_of = vec![usize::MAX; nn + 1];
    let mut next_id = 0usize;
    let mut comp_size: Vec<usize> = Vec::new();
    for b in blocks {
        for &l in b {
            comp_of[l as usize] = next_id;
        }
        comp_size.push(b.len());
        next_id += 1;
    }
    for &(a, b) in pairs {
        if comp_of[a as usize] != usize::MAX || comp_of[b as usize] != usize::MAX {
            continue;
        }
        comp_of[a as usize] = next_id;
        comp_of[b as usize] = next_id;
        comp_size.push(2);
        next_id += 1;
    }
    for l in 1..=nn {
        if comp_of[l] == usize::MAX {
            comp_of[l] = next_id;
            comp_size.push(1);
            next_id += 1;
        }
    }

    let Some(in_c) = best_clean_balanced_clade(inst, nn, &comp_of, &comp_size, value) else {
        debug!("[ncpack-COMBINE] no clean balanced clade found");
        return;
    };

    // Solve one side (leaves with in_c[l]==want) exactly, forbidding `extra`
    // (the other side's footprint, for sequential conditioning; `&[]`-zero for
    // the independent solve). Returns (mu, ≥3-blocks, residual pairs).
    let solve_side = |want: bool, extra: &[u64]| -> (usize, Vec<Block>, Vec<(u32, u32)>) {
        let mut active = vec![false; nn + 1];
        for l in 1..=nn {
            active[l] = in_c[l] == want;
        }
        let (mu, blks) = rec.solve(&mut active, extra, 1);
        let mut forbidden = extra.to_vec();
        let mut scratch = Vec::new();
        for b in &blks {
            let bm = rec.block_mask(b, &mut scratch);
            for k in 0..words {
                forbidden[k] |= bm[k];
            }
            for &l in b {
                active[l as usize] = false;
            }
        }
        let prs = rec.matching_pairs(&active, &forbidden);
        (mu, blks, prs)
    };

    // Full footprint (concatenated Steiner mask over all trees) of a side's forest.
    let side_fp = |blks: &[Block], prs: &[(u32, u32)]| -> Vec<u64> {
        let mut fp = vec![0u64; words];
        let mut scratch = Vec::new();
        for b in blks {
            let bm = rec.block_mask(b, &mut scratch);
            for k in 0..words {
                fp[k] |= bm[k];
            }
        }
        for &(a, b) in prs {
            let bm = rec.block_mask(&[a, b], &mut scratch);
            for k in 0..words {
                fp[k] |= bm[k];
            }
        }
        fp
    };

    let zero = vec![0u64; words];
    // Timing: whole-instance recursion vs the two half-solves. This is the
    // make-or-break for decomposition — halves must be much faster than the whole.
    let t0 = Instant::now();
    {
        let mut all_active = vec![false; nn + 1];
        for l in 1..=nn {
            all_active[l] = true;
        }
        let _ = rec.solve(&mut all_active, &zero, 1);
    }
    let t_whole = t0.elapsed();

    // Independent solve (timed).
    let tc = Instant::now();
    let (mu_c, blocks_c, pairs_c) = solve_side(true, &zero);
    let t_c = tc.elapsed();
    let tcc = Instant::now();
    let (mu_cc, blocks_cc, pairs_cc) = solve_side(false, &zero);
    let t_cc = tcc.elapsed();
    let sum = mu_c + mu_cc;
    debug!(
        "[ncpack-DECOMPTIME] mu*={value} t_whole={:.1}ms t_C={:.1}ms t_Cc={:.1}ms \
         t_halves={:.1}ms speedup={:.1}x",
        t_whole.as_secs_f64() * 1e3,
        t_c.as_secs_f64() * 1e3,
        t_cc.as_secs_f64() * 1e3,
        (t_c + t_cc).as_secs_f64() * 1e3,
        t_whole.as_secs_f64() / (t_c + t_cc).as_secs_f64().max(1e-9),
    );

    let mut all_blocks = blocks_c.clone();
    all_blocks.extend(blocks_cc.iter().cloned());
    let mut all_pairs = pairs_c.clone();
    all_pairs.extend(pairs_cc.iter().cloned());
    let forest = build_forest(inst, &all_pairs, &all_blocks);
    let valid = matches!(validate_agreement_forest(inst, &forest), AfValidation::Ok);

    // Sequential conditioning: commit one side, forbid its footprint, solve the
    // other. seq1 = C first, seq2 = Cᶜ first. Build + validate each combined
    // forest so we can distinguish a sound forest (proves mu* >= seq) from a
    // footprint bug (invalid forest).
    let build_validate =
        |b1: &[Block], p1: &[(u32, u32)], b2: &[Block], p2: &[(u32, u32)]| -> bool {
            let mut bb = b1.to_vec();
            bb.extend(b2.iter().cloned());
            let mut pp = p1.to_vec();
            pp.extend(p2.iter().cloned());
            let f = build_forest(inst, &pp, &bb);
            matches!(validate_agreement_forest(inst, &f), AfValidation::Ok)
        };

    let fp_c = side_fp(&blocks_c, &pairs_c);
    let (mu_cc_cond, b_cc_cond, p_cc_cond) = solve_side(false, &fp_c);
    let seq1 = mu_c + mu_cc_cond;
    let seq1_valid = build_validate(&blocks_c, &pairs_c, &b_cc_cond, &p_cc_cond);

    let fp_cc = side_fp(&blocks_cc, &pairs_cc);
    let (mu_c_cond, b_c_cond, p_c_cond) = solve_side(true, &fp_cc);
    let seq2 = mu_cc + mu_c_cond;
    let seq2_valid = build_validate(&blocks_cc, &pairs_cc, &b_c_cond, &p_c_cond);

    // Best VALID forest merge count we can construct (a sound lower bound on the
    // true mu*). The coupling defect γ = indep_sum − true_mu* ≤ indep_sum − best_valid.
    let mut best_valid = value; // the ncpack forest is always valid
    if valid {
        best_valid = best_valid.max(sum);
    }
    if seq1_valid {
        best_valid = best_valid.max(seq1);
    }
    if seq2_valid {
        best_valid = best_valid.max(seq2);
    }
    let coupling_ub = sum as isize - best_valid as isize;

    let nc = in_c[1..=nn].iter().filter(|&&b| b).count();
    debug!(
        "[ncpack-COMBINE] n={nn} |C|={nc} mu_L={value} mu_C={mu_c} mu_Cc={mu_cc} \
         indep_sum={sum} indep_valid={valid} seq1={seq1}(v={seq1_valid}) \
         seq2={seq2}(v={seq2_valid}) best_valid={best_valid} coupling_defect<={coupling_ub}",
    );
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
    let mut rec = Rec::new(&g, inst, &cover, kmax, slack, node_budget, deadline);
    let words = g.masks.first().map(|m| m.len()).unwrap_or(0);
    // Deepen slack/kmax to a fixed point; only return a value when it certifies.
    let (value, _blocks, certified) = rec.solve_fixpoint(n, words);
    if certified { Some(value) } else { None }
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

        // Hierarchy probe is a self-contained measurement (builds its own
        // forests); run it BEFORE the expensive exhaustive certification, which
        // times out on the wall, and decline (diagnostic-only run).
        if std::env::var("KLADOS_NCPACK_HIERARCHY").as_deref() == Ok("1") {
            hierarchy_probe(inst, ncfg.node_budget);
            return None;
        }
        if std::env::var("KLADOS_NCPACK_T0HIER").as_deref() == Ok("1") {
            t0_hierarchy_probe(inst, n);
            return None;
        }
        if std::env::var("KLADOS_NCPACK_DPCOUNT").as_deref() == Ok("1") {
            dpcount_probe(inst, n);
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
        let mut rec = Rec::new(
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
        // Fast direct primal (single + 2-block Hall lifts, no recursion) by
        // default; falls back to the anchored recursion when `ncov > 64` or when
        // explicitly disabled (`KLADOS_NCPACK_DIRECT=0`).
        //
        // NOTE: `direct_certify` only enumerates Hall witnesses of ≤2 big blocks.
        // That "witness ≤ 2" conjecture is FALSE off the high-tree wall: e.g.
        // pub042 (m=3, n=44) has mu*=20 via a 4-block witness, but direct returns
        // 19. So the direct value can UNDERSHOOT mu* — it is a sound primal (the
        // forest exists ⇒ OPT ≤ n − value is a valid upper bound) but NOT an
        // exact certificate. Only the exhaustive recursion may certify mu*.
        // The Exact track needs a certificate, which the direct primal cannot
        // provide, so it defaults to the (exhaustive) recursion. The Heuristic
        // track wants a fast valid forest, so it takes the direct primal.
        // `KLADOS_NCPACK_DIRECT=1` force-enables the direct primal even on the
        // Exact track (for probes); this stays sound because the direct path
        // always reports `certified=false`, so no certificate is emitted.
        let use_direct = match std::env::var("KLADOS_NCPACK_DIRECT").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => cfg.track != Track::Exact,
        };
        let (value, blocks, certified) = if use_direct {
            match rec.direct_certify(&mut active) {
                Some((v, b)) => (v, b, false),
                None => rec.solve_fixpoint(n, words),
            }
        } else {
            // Exact track: deepen slack/kmax to a fixed point for a sound value.
            rec.solve_fixpoint(n, words)
        };

        debug!(
            "[ncpack] n={n} m={} ncov={ncov} value={value} \
             blocks={} matchings={} certified={certified}",
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

        // DECOMPOSITION PROBE (KLADOS_NCPACK_INTERFACE=1): measure, on this
        // near-optimal forest, the minimum number of components crossing any
        // tree-clade cut. Tests whether good (low-interface) decomposition cuts
        // exist on the wall.
        if std::env::var("KLADOS_NCPACK_INTERFACE").as_deref() == Ok("1") {
            interface_probe(inst, n, &blocks, &residual_pairs, value);
        }
        if std::env::var("KLADOS_NCPACK_COMBINE").as_deref() == Ok("1") {
            combine_test(&rec, inst, n, words, &blocks, &residual_pairs, value);
        }
        if std::env::var("KLADOS_NCPACK_BOUNDARY").as_deref() == Ok("1") {
            boundary_probe(inst, n, &blocks, &residual_pairs);
        }

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
                // Only the exhaustive recursion certifies `mu*`; the direct
                // primal can undershoot it (witness ≤ 2 is false off the wall),
                // so `certified` is false on that path. Even when the recursion
                // closes, the proof still assumes every residual gap ≤ `slack`
                // (default 1), so emission stays gated behind `trust_cert`.
                if certified && ncfg.trust_cert {
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
