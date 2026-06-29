//! Merge-space branch-and-bound for multi-tree MAF (exact).
//!
//! Standard solvers search in OPT-space (cut edges / pick blocks), whose depth
//! is OPT ≈ 0.9n on the hard high-m cores — and whose bound (LP / pairwise) is
//! loose, so the tree explodes. This solver searches in **merge-space**: it
//! maximizes the number of merges `μ = n − OPT`, which is *tiny* on exactly
//! those cores. It never enumerates the (exploding) complete component set.
//!
//! State: a partition of the leaves into valid, pairwise non-crossing blocks,
//! plus a set of cannot-link leaf pairs. Branch on a leaf pair `(a,b)`:
//!   - must-link: merge their blocks, propagating forced merges whenever the
//!     grown block crosses another block; prune if the block stops being a valid
//!     agreement component or violates a cannot-link;
//!   - cannot-link: forbid `a,b` from ever sharing a block.
//!
//! Bound (sound): build the "mergeable graph" on current blocks — an edge `i—j`
//! iff `leaves(i)∪leaves(j)` is a valid component with no cannot-link pair
//! between them. If two blocks lie in the same final block `C`, then their union
//! ⊆ C is valid and cannot-link-free, so they are in one mergeable-component.
//! Hence #final-blocks ≥ #mergeable-components = c, so total merges ≤ n − c.
//! Cannot-link decisions fragment the graph, tightening the bound.

use std::time::{Duration, Instant};

use fixedbitset::FixedBitSet;
use klados_core::af_validator::validate_agreement_forest;
use klados_core::kernelize::{self, KernelizeConfig};
use klados_core::{Instance, SolverStats, Tree};

use crate::solvers::bp::column::{AfColumn, ColumnBuilder};
use crate::{RunConfig, Solver, Track};

/// Two leaf-disjoint components cross iff their embeddings share an internal
/// node in some tree (sorted-list intersection per tree).
fn crosses(a: &AfColumn, b: &AfColumn) -> bool {
    for (na, nb) in a.coverage().iter_per_tree().zip(b.coverage().iter_per_tree()) {
        let (mut i, mut j) = (0usize, 0usize);
        while i < na.len() && j < nb.len() {
            match na[i].cmp(&nb[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => return true,
            }
        }
    }
    false
}

struct UnionFind {
    parent: Vec<usize>,
}
impl UnionFind {
    fn new(k: usize) -> Self {
        Self { parent: (0..k).collect() }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

struct MergeBb<'a> {
    trees: &'a [Tree],
    n: u32,
    builder: ColumnBuilder,
    best_merges: usize,
    best_partition: Vec<Vec<u32>>,
    deadline: Option<Instant>,
    nodes: u64,
    aborted: bool,
}

impl<'a> MergeBb<'a> {
    /// Is `la ∪ lb` a valid agreement component? (Necessary for sharing a block.)
    fn union_valid(&mut self, la: &[u32], lb: &[u32]) -> bool {
        let mut leaves = Vec::with_capacity(la.len() + lb.len());
        leaves.extend_from_slice(la);
        leaves.extend_from_slice(lb);
        leaves.sort_unstable();
        leaves.dedup();
        self.builder.try_build(leaves, self.trees).is_some()
    }

    /// Lower bound on the number of final blocks among `active`: connected
    /// components of the "can share a block" graph (edge = union is a valid
    /// component). Two blocks in one final block ⇒ their union is valid ⇒ same
    /// component, so #final-blocks ≥ #components. (Sound, cheap.)
    fn active_components(&mut self, active: &[AfColumn]) -> usize {
        let k = active.len();
        let mut uf = UnionFind::new(k);
        for i in 0..k {
            let li = active[i].labels().to_vec();
            for j in (i + 1)..k {
                if self.union_valid(&li, active[j].labels()) {
                    uf.union(i, j);
                }
            }
        }
        let mut roots = std::collections::HashSet::new();
        for i in 0..k {
            roots.insert(uf.find(i));
        }
        roots.len()
    }

    /// Grow the block group `seed` (indices into `active`) by repeatedly pulling
    /// in any active block it crosses (a cross between separate blocks is illegal
    /// in an agreement forest, so it is a *forced* merge). Returns the new active
    /// set with the grown block first, or `None` if the grown block stops being a
    /// valid component or crosses a finalized (`done`) block.
    fn grow(
        &mut self,
        active: &[AfColumn],
        done: &[AfColumn],
        seed: &[usize],
    ) -> Option<Vec<AfColumn>> {
        let mut in_group = vec![false; active.len()];
        for &i in seed {
            in_group[i] = true;
        }
        loop {
            let mut leaves = Vec::new();
            for (i, b) in active.iter().enumerate() {
                if in_group[i] {
                    leaves.extend_from_slice(b.labels());
                }
            }
            leaves.sort_unstable();
            leaves.dedup();
            let col = self.builder.try_build(leaves, self.trees)?;
            if done.iter().any(|d| crosses(&col, d)) {
                return None; // would cross a finalized block
            }
            let mut added = false;
            for (i, b) in active.iter().enumerate() {
                if !in_group[i] && crosses(&col, b) {
                    in_group[i] = true;
                    added = true;
                }
            }
            if !added {
                let mut na: Vec<AfColumn> = Vec::with_capacity(active.len());
                na.push(col);
                for (i, b) in active.iter().enumerate() {
                    if !in_group[i] {
                        na.push(b.clone());
                    }
                }
                na.sort_by(|x, y| x.labels()[0].cmp(&y.labels()[0]));
                return Some(na);
            }
        }
    }

    fn record(&mut self, done: &[AfColumn], active: &[AfColumn]) {
        let merges = self.n as usize - done.len() - active.len();
        if merges > self.best_merges {
            self.best_merges = merges;
            self.best_partition = done
                .iter()
                .chain(active.iter())
                .map(|b| b.labels().to_vec())
                .collect();
        }
    }

    /// Canonical branching: always grow the active block with the smallest leaf
    /// (`B0 = active[0]`). Either finalize it (it will never merge again) or merge
    /// it with one of its valid partners. Finalizing in smallest-leaf order makes
    /// every partition reachable while keeping the fan-out to B0's partners.
    fn search(&mut self, done: &mut Vec<AfColumn>, active: Vec<AfColumn>) {
        if self.aborted {
            return;
        }
        self.nodes += 1;
        if self.nodes & 0x3ff == 0
            && let Some(dl) = self.deadline
            && Instant::now() >= dl
        {
            self.aborted = true;
            return;
        }

        // Bound: final blocks ≥ |done| + components(active) ⇒ merges ≤ n − that.
        let c = self.active_components(&active);
        if self.n as usize - done.len() - c <= self.best_merges {
            return;
        }
        if active.is_empty() {
            self.record(done, &active);
            return;
        }

        // Branch (a): finalize B0.
        let b0 = active[0].clone();
        let rest: Vec<AfColumn> = active[1..].to_vec();
        done.push(b0);
        self.search(done, rest);
        done.pop();
        if self.aborted {
            return;
        }

        // Branch (b): grow B0 by merging with a valid partner k. Restricting to
        // partners after the previous one is unnecessary; redundant orderings are
        // bounded by (block size)! which is tiny on these high-frag cores.
        let l0 = active[0].labels().to_vec();
        for k in 1..active.len() {
            if !self.union_valid(&l0, active[k].labels()) {
                continue;
            }
            if let Some(new_active) = self.grow(&active, done, &[0, k]) {
                self.search(done, new_active);
                if self.aborted {
                    return;
                }
            }
        }
    }

    /// Greedy maximal merge for an initial incumbent (grow B0 to completion,
    /// finalize, repeat).
    fn greedy(&mut self, active: Vec<AfColumn>) {
        let mut done: Vec<AfColumn> = Vec::new();
        let mut active = active;
        loop {
            if active.is_empty() {
                break;
            }
            let l0 = active[0].labels().to_vec();
            let mut grew = false;
            for k in 1..active.len() {
                if self.union_valid(&l0, active[k].labels())
                    && let Some(na) = self.grow(&active, &done, &[0, k])
                {
                    active = na;
                    grew = true;
                    break;
                }
            }
            if !grew {
                done.push(active.remove(0));
            }
        }
        self.record(&done, &[]);
    }
}

fn singleton_blocks(reduced: &Instance, builder: &mut ColumnBuilder) -> Vec<AfColumn> {
    (1..=reduced.num_leaves)
        .map(|l| {
            builder
                .try_build(vec![l], &reduced.trees)
                .expect("singleton is always a valid component")
        })
        .collect()
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

fn solve_core(reduced: &Instance, deadline: Option<Instant>) -> Option<Vec<Vec<u32>>> {
    let mut builder = ColumnBuilder::new(&reduced.trees);
    let blocks = singleton_blocks(reduced, &mut builder);

    let mut search = MergeBb {
        trees: &reduced.trees,
        n: reduced.num_leaves,
        builder,
        best_merges: 0,
        best_partition: blocks.iter().map(|b| b.labels().to_vec()).collect(),
        deadline,
        nodes: 0,
        aborted: false,
    };

    // Initial incumbent (greedy), then the exact canonical search.
    search.greedy(blocks.clone());
    let mut done: Vec<AfColumn> = Vec::new();
    search.search(&mut done, blocks);

    if search.aborted {
        None // not proven optimal
    } else {
        Some(search.best_partition)
    }
}

pub struct MergeBbSolver {
    stats: SolverStats,
}

impl MergeBbSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
        }
    }
}

impl Default for MergeBbSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for MergeBbSolver {
    type Config = ();
    const SUPPORTED_TRACKS: &'static [Track] = &[Track::Exact];

    fn solve(&mut self, inst: &Instance, cfg: &RunConfig<Self::Config>) -> Option<Vec<Tree>> {
        if inst.trees.is_empty() {
            return None;
        }
        let deadline = cfg.budget.map(|b: Duration| Instant::now() + b);

        let kern = kernelize::kernelize_best(inst, &KernelizeConfig::default());
        let reduced = &kern.instance;

        let reduced_components: Vec<Tree> = if reduced.num_leaves == 0 {
            Vec::new()
        } else if reduced.num_leaves == 1 {
            vec![reduced.trees[0].clone()]
        } else {
            let blocks = solve_core(reduced, deadline)?;
            blocks_to_trees(&blocks, reduced.reference_tree(), reduced.num_leaves)
        };

        let forest = kernelize::expand_solution(
            reduced_components,
            &kern,
            &inst.trees[0],
            inst.num_leaves,
        );

        // Exact: the B&B proves optimality when it finishes (deadline returns
        // None above). Validate defensively before claiming the bound.
        if !validate_agreement_forest(inst, &forest).is_ok() {
            return None;
        }
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
        MergeBbSolver::new(),
        RunConfig {
            track: Track::Exact,
            ..Default::default()
        },
    );
}
