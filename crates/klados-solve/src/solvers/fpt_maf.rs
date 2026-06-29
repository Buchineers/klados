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

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::time::{Duration, Instant};

use fixedbitset::FixedBitSet;
use klados_core::af_validator::{canonical_newick, validate_agreement_forest};
use klados_core::{
    Instance, SolverStats, Tree, cluster_decomposition, cluster_reduction, kernelize,
};

use crate::solvers::chen_rspr::chen_pair_bounds;
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
    lower_bound: usize,
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

        // Lower-bound prune: current component order plus any global bound
        // inherited from the reduced instance.
        if st.order().max(self.lower_bound) >= self.best {
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
fn solve_maf(
    trees: &[Tree],
    num_leaves: u32,
    deadline: Option<Instant>,
    initial_blocks: Option<Vec<Vec<u32>>>,
    lower_bound: usize,
) -> Option<Vec<Vec<u32>>> {
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

    let (greedy_order, greedy_blocks) = st.clone().greedy_finish();
    let (incumbent_order, incumbent_blocks) = match initial_blocks {
        Some(blocks) if blocks.len() < greedy_order => (blocks.len(), blocks),
        _ => (greedy_order, greedy_blocks),
    };

    let mut search = Search {
        best: incumbent_order,
        best_blocks: incumbent_blocks,
        lower_bound,
        deadline,
        nodes: 0,
        aborted: false,
        depth_cap: 4 * num_leaves + 20,
    };
    if search.best <= search.lower_bound {
        return Some(search.best_blocks);
    }
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

fn solve_recursive(inst: &Instance, deadline: Option<Instant>) -> Option<Vec<Tree>> {
    let memo = Rc::new(RefCell::new(HashMap::new()));
    solve_recursive_memo(inst, deadline, &memo)
}

fn solve_recursive_memo(
    inst: &Instance,
    deadline: Option<Instant>,
    memo: &Rc<RefCell<HashMap<String, Vec<Vec<u32>>>>>,
) -> Option<Vec<Tree>> {
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return None;
    }
    if inst.trees.is_empty() {
        return None;
    }
    if inst.num_trees() == 1 {
        return Some(inst.trees.clone());
    }
    if inst.num_leaves <= 1 {
        return Some(inst.trees[0..1].to_vec());
    }

    let kern = kernelize::kernelize_best(inst, &kernelize::KernelizeConfig::default());
    let reduced = &kern.instance;

    if reduced.num_leaves <= 1 {
        let reduced_solution = if reduced.num_leaves == 0 {
            Vec::new()
        } else {
            vec![reduced.trees[0].clone()]
        };
        return Some(kernelize::expand_solution(
            reduced_solution,
            &kern,
            &inst.trees[0],
            inst.num_leaves,
        ));
    }

    let memo_key = instance_key(reduced);
    if let Some(cached_blocks) = memo.borrow().get(&memo_key).cloned() {
        let reduced_components =
            blocks_to_trees(&cached_blocks, reduced.reference_tree(), reduced.num_leaves);
        return Some(kernelize::expand_solution(
            reduced_components,
            &kern,
            &inst.trees[0],
            inst.num_leaves,
        ));
    }

    if let Some(cluster_reduction::ClusterReductionResult::Solved(solution)) =
        cluster_reduction::try_cluster_reduction(reduced, &mut |sub: &Instance| {
            solve_recursive_memo(sub, deadline, memo)
        })
    {
        let blocks = forest_to_blocks(&solution.components);
        memo.borrow_mut().insert(memo_key.clone(), blocks);
        return Some(kernelize::expand_solution(
            solution.components,
            &kern,
            &inst.trees[0],
            inst.num_leaves,
        ));
    }

    if reduced.num_trees() >= 3
        && let Some(solution) =
            cluster_decomposition::try_rspr_cluster_decomposition(reduced, &mut |sub: &Instance| {
                solve_recursive_memo(sub, deadline, memo)
            })
        && validate_agreement_forest(reduced, &solution).is_ok()
    {
        let blocks = forest_to_blocks(&solution);
        memo.borrow_mut().insert(memo_key.clone(), blocks);
        return Some(kernelize::expand_solution(
            solution,
            &kern,
            &inst.trees[0],
            inst.num_leaves,
        ));
    }

    if reduced.num_trees() == 2
        && reduced.num_leaves >= 20
        && let Some(solution) = crate::decomp::whidden_cluster::try_whidden_decomp_2tree(
            reduced,
            &mut |sub: &Instance| solve_recursive_memo(sub, deadline, memo),
            &crate::decomp::whidden_cluster::NEVER_TERMINATE,
        )
    {
        let blocks = forest_to_blocks(&solution);
        memo.borrow_mut().insert(memo_key.clone(), blocks);
        return Some(kernelize::expand_solution(
            solution,
            &kern,
            &inst.trees[0],
            inst.num_leaves,
        ));
    }

    let initial = initial_incumbent_blocks(reduced);
    let lower = chen_component_lower_bound(&reduced.trees);
    let blocks = solve_maf(&reduced.trees, reduced.num_leaves, deadline, initial, lower)?;
    let reduced_components = blocks_to_trees(&blocks, reduced.reference_tree(), reduced.num_leaves);
    memo.borrow_mut().insert(memo_key, blocks);
    Some(kernelize::expand_solution(
        reduced_components,
        &kern,
        &inst.trees[0],
        inst.num_leaves,
    ))
}

fn instance_key(inst: &Instance) -> String {
    let mut key = format!("m{}n{}:", inst.num_trees(), inst.num_leaves);
    for t in &inst.trees {
        key.push_str(&canonical_newick(t));
        key.push('|');
    }
    key
}

fn forest_to_blocks(forest: &[Tree]) -> Vec<Vec<u32>> {
    let mut blocks: Vec<Vec<u32>> = forest
        .iter()
        .map(|component| {
            let mut labels: Vec<u32> = component.leaves().collect();
            labels.sort_unstable();
            labels
        })
        .filter(|labels| !labels.is_empty())
        .collect();
    blocks.sort_unstable();
    blocks
}

fn blocks_to_trees(blocks: &[Vec<u32>], reference: &Tree, n: u32) -> Vec<Tree> {
    blocks
        .iter()
        .filter(|b| !b.is_empty())
        .map(|b| {
            let mut ls = FixedBitSet::with_capacity(n as usize + 1);
            for &l in b {
                ls.insert(l as usize);
            }
            Tree::forest_component(&ls, reference, n)
        })
        .collect()
}

fn partition_to_blocks(partition: &[usize]) -> Vec<Vec<u32>> {
    let mut comp_labels: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    for (leaf_idx, &comp_id) in partition.iter().enumerate() {
        comp_labels
            .entry(comp_id)
            .or_default()
            .push((leaf_idx + 1) as u32);
    }
    comp_labels.into_values().collect()
}

fn initial_incumbent_blocks(inst: &Instance) -> Option<Vec<Vec<u32>>> {
    let m = inst.num_trees();
    let n = inst.num_leaves as usize;
    if m <= 2 || n < 8 {
        return None;
    }

    // Same idea as BP's local multi-tree UB, but kept modest because FPT's
    // intended niche is small reduced cores.
    let trial_budget = 80usize;
    let ref_count = m.min(6).max(1);
    let seed_count = (trial_budget / ref_count).max(8);
    let refs = sampled_reference_indices(m, ref_count);
    let (mut best_ub, mut best_partition) =
        klados_core::lower_bound::best_randomized_partition(&inst.trees, &refs, seed_count);

    if m < 12 && n < 80 {
        let (pr_ub, pr_partition) = klados_core::lower_bound::pairwise_refine_ub(&inst.trees, n);
        if pr_ub < best_ub {
            best_ub = pr_ub;
            best_partition = pr_partition;
        }
    }

    if best_ub >= n {
        return None;
    }
    let blocks = partition_to_blocks(&best_partition);
    let forest = blocks_to_trees(&blocks, inst.reference_tree(), inst.num_leaves);
    if validate_agreement_forest(inst, &forest).is_ok() {
        Some(blocks)
    } else {
        None
    }
}

fn sampled_reference_indices(m: usize, limit: usize) -> Vec<usize> {
    if limit >= m {
        return (0..m).collect();
    }
    let mut out = Vec::with_capacity(limit);
    for slot in 0..limit {
        let idx = slot * (m - 1) / (limit - 1).max(1);
        if out.last().copied() != Some(idx) {
            out.push(idx);
        }
    }
    out
}

fn chen_component_lower_bound(trees: &[Tree]) -> usize {
    let m = trees.len();
    if m < 2 {
        return 1;
    }
    let mut lb = 1usize;
    for i in 0..m {
        for j in (i + 1)..m {
            let (pair_cut_lb, _) = chen_pair_bounds(&trees[i], &trees[j]);
            lb = lb.max(pair_cut_lb + 1);
        }
    }
    lb
}

fn should_use_root_pool_escape(inst: &Instance) -> bool {
    if std::env::var("KLADOS_FPT_NO_ROOT_POOL").as_deref() == Ok("1") {
        return false;
    }
    // The cherry FPT is excellent on tiny reduced public cores, but the
    // exact-public tail contains many medium/large multi-tree instances where
    // the root column pool closes immediately while the FPT tree explodes.  For
    // two-tree instances, the corridor path is usually better/certified up to a
    // few hundred leaves, so only divert the huge cases.
    if inst.num_trees() == 2 {
        inst.num_leaves >= 350
    } else {
        inst.num_leaves > 20
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
        // Exact-track invariant: emit a forest ONLY when it is proven optimal.
        // The root-pool escape is a heuristic incumbent generator that also
        // returns a (sometimes loose) lower bound; trust its forest only when it
        // *certifies* optimality (`lb >= ub`). Otherwise we must NOT return its
        // unproven incumbent — fall through to the exact recursive solver, which
        // returns `None` rather than an unproven forest when it cannot finish in
        // budget. This keeps the fast certified path while never lying.
        if cfg.track == Track::Exact
            && should_use_root_pool_escape(inst)
            && let Some(outcome) =
                crate::solvers::root_pool::RootPoolSolver::new().solve_with_outcome(inst)
            && matches!(outcome.lower_bound, Some(lb) if lb >= outcome.forest.len())
            && validate_agreement_forest(inst, &outcome.forest).is_ok()
        {
            self.stats.upper_bound = Some(outcome.forest.len());
            self.stats.lower_bound = outcome.forest.len();
            return Some(outcome.forest);
        }
        if cfg.track == Track::Exact
            && inst.num_trees() == 2
            && inst.num_leaves >= 64
            && std::env::var("KLADOS_FPT_NO_M2_CORRIDOR").as_deref() != Ok("1")
            && let Some(forest) =
                crate::solvers::corridor::CorridorSolver::new().solve_m2_certified(inst)
        {
            self.stats.upper_bound = Some(forest.len());
            self.stats.lower_bound = forest.len();
            return Some(forest);
        }
        let forest = solve_recursive(inst, deadline)?;
        self.stats.upper_bound = Some(forest.len());
        self.stats.lower_bound = forest.len();
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
