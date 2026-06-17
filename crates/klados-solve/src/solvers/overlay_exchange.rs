//! Incumbent-overlay exchange prototype.
//!
//! This solver does **not** search the global component universe.  It starts
//! from a full incumbent forest and searches small replacement neighborhoods in
//! the incumbent overlay: open `h` incumbent components and ask whether their
//! leaves can be repartitioned into at most `h-1` valid components while all
//! unopened components remain fixed as boundary constraints.
//!
//! This is currently a viability prototype, not a complete proof engine.

use fixedbitset::FixedBitSet;
use klados_core::lower_bound::best_randomized_partition;
use klados_core::{Instance, SolverStats, Tree};
use log::{debug, info};

use crate::solvers::bp::column::{AfColumn, ColumnBuilder};

/// Tuning knobs for [`OverlayExchangeSolver`].
#[derive(Clone, Debug)]
pub struct OverlayConfig {
    /// Maximum incumbent neighborhood size.
    pub max_h: usize,
    /// Maximum improvement rounds.
    pub max_rounds: usize,
    /// Skip split neighborhoods above this many leaves.
    pub local_leaf_cap: usize,
    /// Neighborhood checks per round cap.
    pub max_neighborhoods: u64,
    /// Per-neighborhood local candidate generation cap.
    pub gen_cap: u64,
    /// Randomized-partition seed budget for the initial upper bound.
    pub ub_seeds: usize,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            max_h: 4,
            max_rounds: 100,
            local_leaf_cap: 24,
            max_neighborhoods: 200_000,
            gen_cap: 500_000,
            ub_seeds: 64,
        }
    }
}

pub struct OverlayExchangeSolver {
    stats: SolverStats,
    config: OverlayConfig,
}

impl OverlayExchangeSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
            config: OverlayConfig::default(),
        }
    }
}

impl Default for OverlayExchangeSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for OverlayExchangeSolver {
    type Config = OverlayConfig;
    const SUPPORTED_TRACKS: &'static [Track] = &[Track::Heuristic];

    fn solve(&mut self, instance: &Instance, cfg: &RunConfig<Self::Config>) -> Option<Vec<Tree>> {
        self.config = cfg.specific.clone();
        let n = instance.num_leaves as usize;
        if n == 0 {
            return Some(Vec::new());
        }
        if instance.num_trees() <= 1 {
            let mut all = FixedBitSet::with_capacity(n + 1);
            for l in 1..=n {
                all.insert(l);
            }
            return Some(vec![Tree::component_from_leafset(
                &all,
                &instance.trees[0],
                instance.num_leaves,
            )]);
        }

        let seed_count = self.config.ub_seeds;
        let refs: Vec<usize> = (0..instance.trees.len()).collect();
        let (ub, part) = best_randomized_partition(&instance.trees, &refs, seed_count);
        let mut builder = ColumnBuilder::new(&instance.trees);
        let mut comps = comps_from_partition(instance, &part, &mut builder)?;
        info!(
            "[overlay] initial ub={} comps={} n={} m={} max_h={} leaf_cap={}",
            ub,
            comps.len(),
            n,
            instance.num_trees(),
            self.config.max_h,
            self.config.local_leaf_cap,
        );

        for round in 0..self.config.max_rounds {
            let before = comps.len();
            let mut improved = false;
            let mut checks = 0u64;
            for h in 2..=self.config.max_h.min(comps.len()) {
                let mut choice = Vec::with_capacity(h);
                if let Some(rep) = self.find_exchange(
                    instance,
                    &comps,
                    h,
                    0,
                    &mut choice,
                    &mut builder,
                    &mut checks,
                ) {
                    let opened = rep.open.clone();
                    let new_blocks = rep.blocks.len();
                    apply_replacement_with_builder(instance, &mut comps, rep, &mut builder)?;
                    info!(
                        "[overlay] round={} h={} improved {} -> {} opened={:?} new_blocks={}",
                        round,
                        h,
                        before,
                        comps.len(),
                        opened,
                        new_blocks,
                    );
                    improved = true;
                    break;
                }
            }
            if !improved {
                info!(
                    "[overlay] stopped comps={} rounds={} checked={}",
                    comps.len(),
                    round,
                    checks
                );
                break;
            }
        }

        self.stats.upper_bound = Some(comps.len());
        Some(comps_to_trees(instance, &comps))
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

#[derive(Clone)]
struct OComp {
    labels: Vec<u32>,
    leaf_mask: FixedBitSet,
    node_masks: Vec<FixedBitSet>,
}

struct Replacement {
    open: Vec<usize>,
    blocks: Vec<Vec<u32>>,
}

impl OverlayExchangeSolver {
    fn find_exchange(
        &self,
        instance: &Instance,
        comps: &[OComp],
        h: usize,
        start: usize,
        choice: &mut Vec<usize>,
        builder: &mut ColumnBuilder,
        checks: &mut u64,
    ) -> Option<Replacement> {
        if choice.len() == h {
            *checks += 1;
            if *checks > self.config.max_neighborhoods {
                return None;
            }
            return self.try_choice(instance, comps, choice, builder);
        }
        if *checks > self.config.max_neighborhoods {
            return None;
        }
        let need = h - choice.len();
        for i in start..=comps.len().saturating_sub(need) {
            choice.push(i);
            if let Some(r) = self.find_exchange(instance, comps, h, i + 1, choice, builder, checks)
            {
                return Some(r);
            }
            choice.pop();
            if *checks > self.config.max_neighborhoods {
                break;
            }
        }
        None
    }

    fn try_choice(
        &self,
        instance: &Instance,
        comps: &[OComp],
        open: &[usize],
        builder: &mut ColumnBuilder,
    ) -> Option<Replacement> {
        let n = instance.num_leaves as usize;
        let open_set = open
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let mut labels = Vec::<u32>::new();
        for &idx in open {
            labels.extend_from_slice(&comps[idx].labels);
        }
        labels.sort_unstable();
        labels.dedup();

        let outside = outside_masks(instance, comps, &open_set);

        // Zero-split/coarsening fast path: all opened leaves become one block.
        if let Some(col) = builder.try_build(labels.clone(), &instance.trees) {
            if coverage_disjoint(&outside, col.coverage()) {
                return Some(Replacement {
                    open: open.to_vec(),
                    blocks: vec![labels],
                });
            }
        }

        if labels.len() > self.config.local_leaf_cap {
            return None;
        }
        let target_saving = labels.len().saturating_sub(open.len() - 1);
        if target_saving == 0 {
            return None;
        }

        let mut lgen = LocalGen {
            instance,
            builder,
            outside: &outside,
            labels_universe: &labels,
            out: Vec::new(),
            generated: 0,
            gen_cap: self.config.gen_cap,
            aborted: false,
        };
        let mut cur = Vec::new();
        lgen.enumerate(0, &mut cur);
        if lgen.aborted {
            return None;
        }
        debug!(
            "[overlay] try open={:?} leaves={} target_saving={} candidates={}",
            open,
            labels.len(),
            target_saving,
            lgen.out.len(),
        );
        let mut search = LocalSearch::new(n, instance, labels.clone(), lgen.out, target_saving);
        search.run().map(|blocks| Replacement {
            open: open.to_vec(),
            blocks,
        })
    }
}

struct LocalGen<'a, 'b> {
    instance: &'a Instance,
    builder: &'b mut ColumnBuilder,
    outside: &'a [FixedBitSet],
    labels_universe: &'a [u32],
    out: Vec<OComp>,
    generated: u64,
    gen_cap: u64,
    aborted: bool,
}

impl<'a, 'b> LocalGen<'a, 'b> {
    fn enumerate(&mut self, pos: usize, cur: &mut Vec<u32>) {
        if self.aborted {
            return;
        }
        for i in pos..self.labels_universe.len() {
            cur.push(self.labels_universe[i]);
            if cur.len() >= 2 {
                self.generated += 1;
                if self.generated > self.gen_cap {
                    self.aborted = true;
                    cur.pop();
                    return;
                }
                if let Some(col) = self.builder.try_build(cur.clone(), &self.instance.trees) {
                    if coverage_disjoint(self.outside, col.coverage()) {
                        self.out.push(ocomp_from_col(self.instance, col));
                        self.enumerate(i + 1, cur);
                    }
                }
            } else {
                self.enumerate(i + 1, cur);
            }
            cur.pop();
        }
    }
}

struct LocalSearch<'a> {
    n: usize,
    instance: &'a Instance,
    universe: Vec<u32>,
    candidates: Vec<OComp>,
    by_leaf: Vec<Vec<usize>>,
    target: usize,
}

impl<'a> LocalSearch<'a> {
    fn new(
        n: usize,
        instance: &'a Instance,
        universe: Vec<u32>,
        mut candidates: Vec<OComp>,
        target: usize,
    ) -> Self {
        candidates.sort_by(|a, b| b.labels.len().cmp(&a.labels.len()));
        let mut by_leaf = vec![Vec::new(); n + 1];
        for (i, c) in candidates.iter().enumerate() {
            for &l in &c.labels {
                by_leaf[l as usize].push(i);
            }
        }
        Self {
            n,
            instance,
            universe,
            candidates,
            by_leaf,
            target,
        }
    }

    fn run(&mut self) -> Option<Vec<Vec<u32>>> {
        let used_nodes = self
            .instance
            .trees
            .iter()
            .map(|t| FixedBitSet::with_capacity(t.num_nodes()))
            .collect::<Vec<_>>();
        let used_leaves = FixedBitSet::with_capacity(self.n + 1);
        self.dfs(used_leaves, used_nodes, 0, Vec::new())
    }

    fn dfs(
        &self,
        used_leaves: FixedBitSet,
        used_nodes: Vec<FixedBitSet>,
        saving: usize,
        selected: Vec<Vec<u32>>,
    ) -> Option<Vec<Vec<u32>>> {
        if saving >= self.target {
            return Some(selected);
        }
        let free = self
            .universe
            .iter()
            .filter(|&&l| !used_leaves.contains(l as usize))
            .count();
        if saving + free.saturating_sub(1) < self.target {
            return None;
        }
        let mut best_leaf = None;
        let mut best_list: Vec<usize> = Vec::new();
        for &lbl in &self.universe {
            let l = lbl as usize;
            if used_leaves.contains(l) {
                continue;
            }
            let list = self.by_leaf[l]
                .iter()
                .copied()
                .filter(|&idx| self.compatible(idx, &used_leaves, &used_nodes))
                .collect::<Vec<_>>();
            if best_leaf.is_none() || list.len() < best_list.len() {
                best_leaf = Some(l);
                best_list = list;
            }
        }
        let leaf = best_leaf?;
        for idx in best_list {
            let c = &self.candidates[idx];
            let mut nl = used_leaves.clone();
            nl.union_with(&c.leaf_mask);
            let mut nn = used_nodes.clone();
            for (ti, mask) in c.node_masks.iter().enumerate() {
                nn[ti].union_with(mask);
            }
            let mut ns = selected.clone();
            ns.push(c.labels.clone());
            if let Some(r) = self.dfs(nl, nn, saving + c.labels.len() - 1, ns) {
                return Some(r);
            }
        }
        // Leaf remains singleton.
        let mut nl = used_leaves;
        nl.insert(leaf);
        self.dfs(nl, used_nodes, saving, selected)
    }

    fn compatible(
        &self,
        idx: usize,
        used_leaves: &FixedBitSet,
        used_nodes: &[FixedBitSet],
    ) -> bool {
        let c = &self.candidates[idx];
        if used_leaves.intersection(&c.leaf_mask).next().is_some() {
            return false;
        }
        for (ti, mask) in c.node_masks.iter().enumerate() {
            if used_nodes[ti].intersection(mask).next().is_some() {
                return false;
            }
        }
        true
    }
}

fn apply_replacement_with_builder(
    instance: &Instance,
    comps: &mut Vec<OComp>,
    rep: Replacement,
    builder: &mut ColumnBuilder,
) -> Option<()> {
    let open = rep
        .open
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut used = FixedBitSet::with_capacity(instance.num_leaves as usize + 1);
    for block in &rep.blocks {
        for &l in block {
            used.insert(l as usize);
        }
    }
    let mut all_open = Vec::new();
    for &idx in &rep.open {
        all_open.extend_from_slice(&comps[idx].labels);
    }
    all_open.sort_unstable();
    all_open.dedup();
    let mut blocks = rep.blocks;
    for l in all_open {
        if !used.contains(l as usize) {
            blocks.push(vec![l]);
        }
    }
    let mut next = Vec::new();
    for (i, c) in comps.iter().enumerate() {
        if !open.contains(&i) {
            next.push(c.clone());
        }
    }
    for block in blocks {
        let col = builder.try_build(block, &instance.trees)?;
        next.push(ocomp_from_col(instance, col));
    }
    *comps = next;
    Some(())
}

fn comps_from_partition(
    instance: &Instance,
    part: &[usize],
    builder: &mut ColumnBuilder,
) -> Option<Vec<OComp>> {
    let n = instance.num_leaves as usize;
    let max_comp = part.iter().copied().max().unwrap_or(0);
    let mut sets = vec![Vec::<u32>::new(); max_comp + 1];
    for (idx, &c) in part.iter().enumerate() {
        let label = idx + 1;
        if label <= n {
            sets[c].push(label as u32);
        }
    }
    let mut out = Vec::new();
    for mut labels in sets {
        if labels.is_empty() {
            continue;
        }
        labels.sort_unstable();
        let col = builder.try_build(labels, &instance.trees)?;
        out.push(ocomp_from_col(instance, col));
    }
    Some(out)
}

fn ocomp_from_col(instance: &Instance, col: AfColumn) -> OComp {
    let n = instance.num_leaves as usize;
    let mut leaf_mask = FixedBitSet::with_capacity(n + 1);
    for &l in col.labels() {
        leaf_mask.insert(l as usize);
    }
    let node_masks = instance
        .trees
        .iter()
        .enumerate()
        .map(|(ti, t)| {
            let mut mask = FixedBitSet::with_capacity(t.num_nodes());
            for &v in col.coverage().nodes(ti) {
                mask.insert(v);
            }
            mask
        })
        .collect();
    OComp {
        labels: col.labels().to_vec(),
        leaf_mask,
        node_masks,
    }
}

fn outside_masks(
    instance: &Instance,
    comps: &[OComp],
    open: &std::collections::BTreeSet<usize>,
) -> Vec<FixedBitSet> {
    let mut masks = instance
        .trees
        .iter()
        .map(|t| FixedBitSet::with_capacity(t.num_nodes()))
        .collect::<Vec<_>>();
    for (i, c) in comps.iter().enumerate() {
        if open.contains(&i) {
            continue;
        }
        for (ti, m) in c.node_masks.iter().enumerate() {
            masks[ti].union_with(m);
        }
    }
    masks
}

fn coverage_disjoint(
    used_nodes: &[FixedBitSet],
    coverage: &crate::solvers::bp::column::ColumnCoverage,
) -> bool {
    for (ti, nodes) in coverage.iter_per_tree().enumerate() {
        for &v in nodes {
            if used_nodes[ti].contains(v) {
                return false;
            }
        }
    }
    true
}

fn comps_to_trees(instance: &Instance, comps: &[OComp]) -> Vec<Tree> {
    comps
        .iter()
        .map(|c| {
            let mut set = FixedBitSet::with_capacity(instance.num_leaves as usize + 1);
            for &l in &c.labels {
                set.insert(l as usize);
            }
            Tree::component_from_leafset(&set, &instance.trees[0], instance.num_leaves)
        })
        .collect()
}

// ── Unified Solver impl + entry point ───────────────────────────────────────
use crate::{RunConfig, Solver, Track};

pub fn main() {
    crate::run(
        OverlayExchangeSolver::new(),
        RunConfig {
            track: Track::Heuristic,
            specific: OverlayConfig::default(),
            ..Default::default()
        },
    );
}
