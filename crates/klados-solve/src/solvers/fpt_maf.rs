//! Fresh multi-tree FPT branch-and-bound Maximum Agreement Forest solver for
//! **binary rooted** trees.
//!
//! Method (Shi–Chen–Feng–Wang, arXiv:1608.02709, specialized to binary rooted
//! trees — Case 2 only, the sibling-pair/cherry branching):
//!
//! Maintain `m` working forests, all over the same competition label set (the
//! `m` input rooted trees; no artificial root sentinel is added). Repeatedly:
//!   * **Reduction (Rule 2 / MSS):** if a cherry `(a,b)` is a cherry in *every*
//!     forest, contract it into one super-leaf — forced, no branching.
//!   * **Terminate:** if no forest has a cherry, all forests are label-set
//!     isomorphic; the common partition is an agreement forest. Its order is a
//!     candidate optimum.
//!   * **Branch** on a cherry `(a,b)` of some forest, 3 ways:
//!       1. cut `a` (make it a singleton in every forest),
//!       2. cut `b`,
//!       3. keep `a,b` together: in every other forest remove the pendant edges
//!          `E_F(a,b)` that separate them, making them a cherry, then contract.
//!     Branch 3 is taken only when `a,b` lie in the same component of *every*
//!     forest (else it is infeasible and we branch 2 ways).
//!
//! The order (number of components) only grows along the search, so the running
//! `max_i Ord(F_i)` is a valid lower bound; we branch-and-bound on it against
//! the best agreement forest found. Exact: returns the minimum-order agreement
//! forest = the MAF.

use std::time::{Duration, Instant};

use fixedbitset::FixedBitSet;
use klados_core::{Instance, SolverStats, Tree};

use crate::{RunConfig, Solver, Track};

const NONE: u32 = u32::MAX;

/// One working forest: an arena of binary nodes plus the set of component roots.
#[derive(Clone)]
struct Forest {
    par: Vec<u32>,     // parent node id, or NONE for a component root
    ch: Vec<[u32; 2]>, // two children, or [NONE, NONE] for a leaf
    lab: Vec<u32>,     // label for a leaf, 0 for an internal node
    alive: Vec<bool>,  // false once a node is orphaned by a cut/contract
    node_of: Vec<u32>, // label -> node id (NONE if the label is absent)
    roots: Vec<u32>,   // component roots
}

impl Forest {
    fn new(maxlab: usize) -> Self {
        Forest {
            par: Vec::new(),
            ch: Vec::new(),
            lab: Vec::new(),
            alive: Vec::new(),
            node_of: vec![NONE; maxlab + 1],
            roots: Vec::new(),
        }
    }

    #[inline]
    fn is_leaf(&self, n: u32) -> bool {
        self.ch[n as usize][0] == NONE
    }

    fn push_leaf(&mut self, label: u32) -> u32 {
        let id = self.par.len() as u32;
        self.par.push(NONE);
        self.ch.push([NONE, NONE]);
        self.lab.push(label);
        self.alive.push(true);
        self.node_of[label as usize] = id;
        id
    }

    fn push_internal(&mut self, l: u32, r: u32) -> u32 {
        let id = self.par.len() as u32;
        self.par.push(NONE);
        self.ch.push([l, r]);
        self.lab.push(0);
        self.alive.push(true);
        self.par[l as usize] = id;
        self.par[r as usize] = id;
        id
    }

    /// Copy a subtree of an input `Tree` (binary, left/right) into this arena,
    /// returning the new node id.
    fn copy_subtree(&mut self, t: &Tree, node: u32) -> u32 {
        if t.is_leaf(node) {
            self.push_leaf(t.label[node as usize])
        } else {
            let l = self.copy_subtree(t, t.left[node as usize]);
            let r = self.copy_subtree(t, t.right[node as usize]);
            self.push_internal(l, r)
        }
    }

    /// Build the working forest for one input rooted binary tree.
    fn from_tree(t: &Tree, maxlab: usize) -> Self {
        let mut f = Forest::new(maxlab);
        let troot = f.copy_subtree(t, t.root);
        f.roots.push(troot);
        f
    }

    #[inline]
    fn sibling(&self, n: u32) -> u32 {
        let p = self.par[n as usize];
        let c = self.ch[p as usize];
        if c[0] == n { c[1] } else { c[0] }
    }

    /// Component root of `n` (walk to the top).
    fn root_of(&self, mut n: u32) -> u32 {
        while self.par[n as usize] != NONE {
            n = self.par[n as usize];
        }
        n
    }

    fn replace_root(&mut self, old: u32, new: u32) {
        for r in self.roots.iter_mut() {
            if *r == old {
                *r = new;
                return;
            }
        }
        self.roots.push(new);
    }

    /// Detach the edge above `node`: `node` becomes a new component root, and
    /// its old parent is forced-contracted (its remaining child takes its
    /// place). No-op if `node` is already a component root.
    fn cut_above(&mut self, node: u32) {
        let p = self.par[node as usize];
        if p == NONE {
            return; // already a root
        }
        let s = self.sibling(node);
        // node becomes a root
        self.par[node as usize] = NONE;
        self.roots.push(node);
        // forced-contract p: s takes p's place
        let gp = self.par[p as usize];
        self.par[s as usize] = gp;
        if gp == NONE {
            self.replace_root(p, s);
        } else {
            let gc = &mut self.ch[gp as usize];
            if gc[0] == p {
                gc[0] = s;
            } else {
                gc[1] = s;
            }
        }
        // The old binary parent was suppressed by forced contraction. It is
        // still present in the arena, so mark it dead to prevent later scans
        // from treating its stale child pair as a real cherry.
        self.alive[p as usize] = false;
        self.ch[p as usize] = [NONE, NONE];
        self.par[p as usize] = NONE;
    }

    /// Pendant nodes whose removal makes leaves `a` and `b` a cherry (they must
    /// be in the same component). Bottom-up order so each pendant's parent is
    /// intact when cut.
    fn pendants(&self, a: u32, b: u32, out: &mut Vec<u32>) {
        let na = self.node_of[a as usize];
        let nb = self.node_of[b as usize];
        let mut on_path = vec![false; self.par.len()];
        let lca = self.marked_lca(na, nb, &mut on_path);
        self.collect_pendants_to_lca(na, lca, out);
        self.collect_pendants_to_lca(nb, lca, out);
    }

    fn pendant_count(&self, a: u32, b: u32) -> usize {
        let na = self.node_of[a as usize];
        let nb = self.node_of[b as usize];
        let mut on_path = vec![false; self.par.len()];
        let lca = self.marked_lca(na, nb, &mut on_path);
        self.count_pendants_to_lca(na, lca) + self.count_pendants_to_lca(nb, lca)
    }

    fn marked_lca(&self, na: u32, nb: u32, on_path: &mut [bool]) -> u32 {
        let mut x = na;
        loop {
            on_path[x as usize] = true;
            let p = self.par[x as usize];
            if p == NONE {
                break;
            }
            x = p;
        }
        let mut lca = nb;
        while !on_path[lca as usize] {
            lca = self.par[lca as usize];
        }
        lca
    }

    fn collect_pendants_to_lca(&self, mut x: u32, lca: u32, out: &mut Vec<u32>) {
        while self.par[x as usize] != lca {
            out.push(self.sibling(x));
            x = self.par[x as usize];
        }
    }

    fn count_pendants_to_lca(&self, mut x: u32, lca: u32) -> usize {
        let mut count = 0;
        while self.par[x as usize] != lca {
            count += 1;
            x = self.par[x as usize];
        }
        count
    }

    fn cherry_pairs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        (0..self.par.len() as u32).filter_map(|n| {
            if !self.alive[n as usize] || self.lab[n as usize] != 0 {
                return None;
            }
            let c = self.ch[n as usize];
            if c[0] != NONE && self.is_leaf(c[0]) && self.is_leaf(c[1]) {
                Some((self.lab[c[0] as usize], self.lab[c[1] as usize]))
            } else {
                None
            }
        })
    }
}

/// The whole branch-and-bound state: all forests plus the contracted-block map.
#[derive(Clone)]
struct State {
    forests: Vec<Forest>,
    /// block[label] = original leaf labels represented by this (super-)leaf.
    block: Vec<Vec<u32>>,
    next_label: u32,
}

impl State {
    fn order(&self) -> usize {
        // valid lower bound on the final order: order only grows
        self.forests
            .iter()
            .map(|f| f.roots.len())
            .max()
            .unwrap_or(0)
    }

    /// Find a cherry that is a cherry in *every* forest (agreed); contract it.
    /// Returns true if it contracted something.
    fn reduce_once(&mut self) -> bool {
        // scan forest 0 for cherries, check agreement in the rest
        let f0 = &self.forests[0];
        let mut found: Option<(u32, u32)> = None;
        'outer: for (a, b) in f0.cherry_pairs() {
            // agreed in all others?
            for f in &self.forests[1..] {
                let na = f.node_of[a as usize];
                let nb = f.node_of[b as usize];
                if na == NONE || nb == NONE {
                    continue 'outer;
                }
                if f.par[na as usize] == NONE || f.par[na as usize] != f.par[nb as usize] {
                    continue 'outer;
                }
            }
            found = Some((a, b));
            break;
        }
        if let Some((a, b)) = found {
            self.contract(a, b);
            true
        } else {
            false
        }
    }

    /// Contract cherry `(a,b)` — a cherry in *every* forest — into a fresh
    /// super-leaf in all forests.
    fn contract(&mut self, a: u32, b: u32) {
        let l = self.next_label;
        self.next_label += 1;
        let mut blk = std::mem::take(&mut self.block[a as usize]);
        blk.extend_from_slice(&self.block[b as usize]);
        self.block[l as usize] = blk;
        for f in &mut self.forests {
            let na = f.node_of[a as usize];
            let nb = f.node_of[b as usize];
            let p = f.par[na as usize];
            // p becomes a leaf labeled l
            f.ch[p as usize] = [NONE, NONE];
            f.lab[p as usize] = l;
            f.alive[na as usize] = false;
            f.alive[nb as usize] = false;
            f.node_of[l as usize] = p;
            f.node_of[a as usize] = NONE;
            f.node_of[b as usize] = NONE;
        }
    }

    /// Cut leaf `label` to a singleton component in every forest.
    fn cut(&mut self, label: u32) {
        for f in &mut self.forests {
            let n = f.node_of[label as usize];
            f.cut_above(n);
        }
    }

    /// Pick a branching cherry. Any cherry is safe; this chooses a constrained
    /// one to shrink the search tree: return immediately if the keep-branch is
    /// impossible (binary branch), otherwise maximize pendant cuts in the keep
    /// branch.
    fn find_cherry(&self) -> Option<(usize, u32, u32)> {
        let mut best_together: Option<(usize, u32, u32, usize)> = None;
        for (fi, f) in self.forests.iter().enumerate() {
            for (a, b) in f.cherry_pairs() {
                if !self.together_everywhere(a, b) {
                    return Some((fi, a, b));
                }
                let mut cuts = 0;
                for (j, other) in self.forests.iter().enumerate() {
                    if j != fi {
                        cuts += other.pendant_count(a, b);
                    }
                }
                if best_together.is_none_or(|(_, _, _, best_cuts)| cuts > best_cuts) {
                    best_together = Some((fi, a, b, cuts));
                }
            }
        }
        best_together.map(|(fi, a, b, _)| (fi, a, b))
    }

    fn keep_pair(&mut self, fi: usize, a: u32, b: u32) {
        for j in 0..self.forests.len() {
            if j == fi {
                continue;
            }
            let mut pend = Vec::new();
            self.forests[j].pendants(a, b, &mut pend);
            for p in pend {
                self.forests[j].cut_above(p);
            }
        }
        self.contract(a, b);
    }

    /// Are `a` and `b` in the same component of every forest?
    fn together_everywhere(&self, a: u32, b: u32) -> bool {
        self.forests.iter().all(|f| {
            let na = f.node_of[a as usize];
            let nb = f.node_of[b as usize];
            na != NONE && nb != NONE && f.root_of(na) == f.root_of(nb)
        })
    }

    /// Fast deterministic incumbent: always apply reductions, keep a chosen
    /// cherry when feasible, otherwise take the cheaper immediate cut branch.
    /// This is only an upper bound; exactness still comes from DFS.
    fn greedy_finish(mut self) -> (usize, Vec<Vec<u32>>) {
        loop {
            while self.reduce_once() {}
            let Some((fi, a, b)) = self.find_cherry() else {
                return (self.order(), self.blocks());
            };
            if self.together_everywhere(a, b) {
                self.keep_pair(fi, a, b);
            } else {
                let mut cut_a = self.clone();
                cut_a.cut(a);
                while cut_a.reduce_once() {}

                let mut cut_b = self;
                cut_b.cut(b);
                while cut_b.reduce_once() {}

                self = if cut_a.order() <= cut_b.order() {
                    cut_a
                } else {
                    cut_b
                };
            }
        }
    }

    /// The partition (list of original-label blocks) at a terminal state.
    fn blocks(&self) -> Vec<Vec<u32>> {
        let f = &self.forests[0];
        let mut out = Vec::new();
        for &r in &f.roots {
            // terminal: every component is a single (super-)leaf
            let lbl = f.lab[r as usize];
            let mut b = self.block[lbl as usize].clone();
            b.sort_unstable();
            out.push(b);
        }
        out
    }
}

struct Search {
    best: usize,
    best_blocks: Vec<Vec<u32>>,
    deadline: Option<Instant>,
    nodes: u64,
    aborted: bool,
    depth_cap: u32,
}

impl Search {
    fn dfs(&mut self, mut st: State, depth: u32) {
        if self.aborted {
            return;
        }
        if depth > self.depth_cap {
            if !self.aborted {
                let f = &st.forests[0];
                eprintln!(
                    "DEPTH-CAP depth={} order={} roots0={} leaves_present(f0 cherry?)={:?}",
                    depth,
                    st.order(),
                    f.roots.len(),
                    st.find_cherry(),
                );
            }
            self.aborted = true;
            return;
        }
        self.nodes += 1;
        if self.nodes % 4096 == 0
            && let Some(dl) = self.deadline
            && Instant::now() >= dl
        {
            self.aborted = true;
            return;
        }

        // Reduction: contract all agreed cherries.
        while st.reduce_once() {}

        // Lower-bound prune.
        if st.order() >= self.best {
            return;
        }

        let Some((fi, a, b)) = st.find_cherry() else {
            // terminal: a complete agreement forest
            let ord = st.order();
            if ord < self.best {
                self.best = ord;
                self.best_blocks = st.blocks();
            }
            return;
        };

        let together = st.together_everywhere(a, b);

        // Branch 3 (keep a,b together): remove pendants in every other forest,
        // then contract. Do the most-constraining branch first.
        if together {
            let mut s3 = st.clone();
            s3.keep_pair(fi, a, b);
            self.dfs(s3, depth + 1);
        }

        // Branch 1: cut a.
        let mut s1 = st.clone();
        s1.cut(a);
        self.dfs(s1, depth + 1);

        // Branch 2: cut b.
        st.cut(b);
        self.dfs(st, depth + 1);
    }
}

/// Solve the exact multi-tree MAF on binary rooted trees. Returns the partition
/// into label blocks, or `None` if aborted (budget) before proving optimum.
fn solve_maf(trees: &[Tree], num_leaves: u32, deadline: Option<Instant>) -> Option<Vec<Vec<u32>>> {
    let n = num_leaves;
    let maxlab = (2 * n + 1) as usize;

    let forests: Vec<Forest> = trees.iter().map(|t| Forest::from_tree(t, maxlab)).collect();

    let mut block = vec![Vec::<u32>::new(); maxlab + 1];
    for l in 1..=n {
        block[l as usize] = vec![l];
    }

    let st = State {
        forests,
        block,
        next_label: n + 1,
    };

    let (incumbent_order, incumbent_blocks) = st.clone().greedy_finish();

    let mut search = Search {
        best: incumbent_order,
        best_blocks: incumbent_blocks,
        deadline,
        nodes: 0,
        aborted: false,
        depth_cap: 4 * num_leaves + 20,
    };
    search.dfs(st, 0);

    if search.aborted && search.best == usize::MAX {
        None
    } else if search.best == usize::MAX {
        None
    } else if search.aborted {
        // had an incumbent but not proven optimal
        None
    } else {
        Some(search.best_blocks)
    }
}

pub struct FptMafSolver {
    stats: SolverStats,
}

impl FptMafSolver {
    pub fn new() -> Self {
        FptMafSolver {
            stats: SolverStats::default(),
        }
    }
}

impl Default for FptMafSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for FptMafSolver {
    type Config = ();
    const SUPPORTED_TRACKS: &'static [Track] = &[Track::Exact];

    fn solve(&mut self, inst: &Instance, cfg: &RunConfig<Self::Config>) -> Option<Vec<Tree>> {
        if inst.trees.is_empty() {
            return None;
        }
        let deadline = cfg.budget.map(|b: Duration| Instant::now() + b);
        let blocks = solve_maf(&inst.trees, inst.num_leaves, deadline)?;

        let reference = &inst.trees[0];
        let n = inst.num_leaves;
        let forest: Vec<Tree> = blocks
            .iter()
            .map(|b| {
                let mut ls = FixedBitSet::with_capacity(n as usize + 1);
                for &l in b {
                    ls.insert(l as usize);
                }
                Tree::forest_component(&ls, reference, n)
            })
            .collect();
        Some(forest)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

pub fn main() {
    crate::run(
        FptMafSolver::new(),
        RunConfig {
            track: Track::Exact,
            ..Default::default()
        },
    );
}
