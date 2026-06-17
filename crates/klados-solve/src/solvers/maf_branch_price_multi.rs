//! Experimental multi-tree Branch & Price solver.
//!
//! Current shape:
//! - generic master problem over an arbitrary number of rooted trees
//! - exact multi-tree pricing via memoized top-down `M`/`V` recurrences
//! - exhaustive subset oracle available as an optional validation check on tiny instances
//! - generic branch-and-bound on fractional columns

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::time::Instant;

use fixedbitset::FixedBitSet;
use fxhash::{FxHashMap, FxHashSet};
use highs::{ColProblem, HighsModelStatus, Model};
use klados_core::lower_bound::{
    greedy_multi_tree_partition, greedy_multi_tree_ub_seeded, pairwise_refine_ub,
};
use klados_core::{Instance, SolverStats, Tree};
use log::{error, info};

struct LocalBounds {
    best_partition: Option<Vec<usize>>,
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

fn compute_local_bounds(trees: &[Tree], num_leaves: u32) -> LocalBounds {
    if trees.len() <= 1 {
        return LocalBounds {
            best_partition: None,
        };
    }

    let m = trees.len();
    let n = num_leaves as usize;

    let best_partition = if m == 2 {
        None
    } else {
        let (ref_limit, seed_limit, run_pairwise) = if m >= 20 || n >= 200 {
            (4usize, 2u64, false)
        } else if m >= 12 || n >= 140 {
            (6usize, 3u64, true)
        } else {
            (m, 5u64, true)
        };
        let ref_indices = sampled_reference_indices(m, ref_limit.min(m));
        let mut best_multi_ub = n;
        let mut best_ref = 0usize;
        let mut best_seed = 0u64;
        for ref_idx in ref_indices {
            for seed in 0..seed_limit {
                let ub = greedy_multi_tree_ub_seeded(trees, ref_idx, seed);
                if ub < best_multi_ub {
                    best_multi_ub = ub;
                    best_ref = ref_idx;
                    best_seed = seed;
                }
            }
        }
        if run_pairwise {
            let (pr_ub, pr_partition) = pairwise_refine_ub(trees, n);
            if pr_ub < best_multi_ub {
                Some(pr_partition)
            } else {
                let (_, partition) = greedy_multi_tree_partition(trees, best_ref, best_seed);
                Some(partition)
            }
        } else {
            let (_, partition) = greedy_multi_tree_partition(trees, best_ref, best_seed);
            Some(partition)
        }
    };

    LocalBounds { best_partition }
}

use klados_core::cluster_decomposition;

use crate::solvers::chen_rspr::chen_pair_agreement;
use crate::cluster_reduction::{self, ClusterReductionResult};
use crate::kernelize::{self, KernelizeConfig};

const NEG_INF: f64 = -1.0e100;
const PAIRDP_BATCH_SIZE: usize = 64;
const FASTPRICER_BATCH_SIZE: usize = 16;
const FASTPRICER_ANCHOR_LABELS: usize = 64;
const FASTPRICER_PARTNERS_PER_ANCHOR: usize = 8;
const FASTPRICER_PAIR_TRIALS: usize = 256;
const WIDEPRICER_BATCH_SIZE: usize = 32;
const WIDEPRICER_ANCHOR_LABELS: usize = 160;
const WIDEPRICER_PARTNERS_PER_ANCHOR: usize = 16;
const WIDEPRICER_PAIR_TRIALS: usize = 2048;
const WIDEPRICER_MIN_ACTIVE_LABELS: usize = 330;
const MEMO_MIN_LEAVES: u32 = 4;
const MEMO_MAX_LEAVES: u32 = 512;
const COLUMN_RESERVE_CAP: usize = 0;
const COLUMN_RESERVE_REFILL: usize = 0;
// The current Rust rSPR decomposition treats selected clusters as independent
// closed subinstances. Whidden's original code carries boundary/rho state when
// joining clusters; without that, this path returned 513 vs the known 495 on
// heuristic instance 070bfd..., so keep it off until boundary states are ported.
const RSPR_CLUSTER_DECOMP_EXPERIMENTAL: bool = false;
const RSPR_CLUSTER_MIN_LEAVES: u32 = 128;
// Strict Whidden/rSPR-style 2-tree common-cluster decomposition kicks in above
// this size.  Relaxed/batch variants remain opt-in inside whidden_cluster until
// the full rspr boundary/rho join state is represented.
const WHIDDEN_DECOMP_MIN_LEAVES: u32 = 20;

// === Exact Bottom-Up DP Pricer (m = 2) ===

#[derive(Clone, Copy)]
struct DpClosed {
    score: f64,
    v_l: u32,
    v_r: u32,
}

impl Default for DpClosed {
    fn default() -> Self {
        Self {
            score: NEG_INF,
            v_l: 0,
            v_r: 0,
        }
    }
}

#[derive(Clone, Copy)]
struct DpOpen {
    score: f64,
    choice: u8,
}

impl Default for DpOpen {
    fn default() -> Self {
        Self {
            score: NEG_INF,
            choice: 0,
        }
    }
}

struct Dp2TreeCache {
    dp_closed: Vec<Vec<DpClosed>>,
    dp_open: Vec<Vec<DpOpen>>,
    t0_active: Vec<bool>,
    t1_active: Vec<bool>,
    t0_post_order: Vec<u32>,
    t1_post_order: Vec<u32>,
    max_score_under: Vec<Vec<(f64, u32)>>,
}

impl Dp2TreeCache {
    fn new(trees: &[Tree]) -> Self {
        let mut dp_closed = Vec::new();
        let mut dp_open = Vec::new();
        let mut t0_active = Vec::new();
        let mut t1_active = Vec::new();
        let mut t0_post_order = Vec::new();
        let mut t1_post_order = Vec::new();
        let mut max_score_under = Vec::new();

        if trees.len() == 2 {
            dp_closed = vec![vec![DpClosed::default(); trees[1].num_nodes()]; trees[0].num_nodes()];
            dp_open = vec![vec![DpOpen::default(); trees[1].num_nodes()]; trees[0].num_nodes()];
            t0_active = vec![false; trees[0].num_nodes()];
            t1_active = vec![false; trees[1].num_nodes()];
            max_score_under = vec![vec![(NEG_INF, 0); trees[1].num_nodes()]; 2];
        }

        Self {
            dp_closed,
            dp_open,
            t0_active,
            t1_active,
            t0_post_order,
            t1_post_order,
            max_score_under,
        }
    }
}

struct ExactPricer2Tree<'a> {
    trees: &'a [Tree],
    alpha: &'a [f64],
    beta: &'a [Vec<f64>],
    active_labels: Vec<bool>,
    dp_closed: &'a mut Vec<Vec<DpClosed>>,
    dp_open: &'a mut Vec<Vec<DpOpen>>,
    t0_post_order: &'a Vec<u32>,
    t1_post_order: &'a Vec<u32>,
    t0_active: &'a Vec<bool>,
    max_score_under: &'a mut Vec<Vec<(f64, u32)>>,
}

impl<'a> ExactPricer2Tree<'a> {
    fn new(
        trees: &'a [Tree],
        num_leaves: usize,
        alpha: &'a [f64],
        beta: &'a [Vec<f64>],
        blocked_leaves: &[bool],
        cache: &'a mut Dp2TreeCache,
    ) -> Self {
        assert_eq!(trees.len(), 2);
        let mut active_labels = vec![false; num_leaves + 1];

        cache.t0_active.fill(false);
        cache.t1_active.fill(false);

        for i in 1..=num_leaves {
            if !blocked_leaves[i] && alpha[i] > 1.0e-12 {
                active_labels[i] = true;

                let mut curr = trees[0].label_to_node[i] as u32;
                while curr != klados_core::NONE && !cache.t0_active[curr as usize] {
                    cache.t0_active[curr as usize] = true;
                    curr = trees[0].parent[curr as usize];
                }

                let mut curr = trees[1].label_to_node[i] as u32;
                while curr != klados_core::NONE && !cache.t1_active[curr as usize] {
                    cache.t1_active[curr as usize] = true;
                    curr = trees[1].parent[curr as usize];
                }
            }
        }

        cache.t0_post_order.clear();
        for u in trees[0].post_order() {
            if cache.t0_active[u as usize] {
                cache.t0_post_order.push(u);
            }
        }

        cache.t1_post_order.clear();
        for v in trees[1].post_order() {
            if cache.t1_active[v as usize] {
                cache.t1_post_order.push(v);
            }
        }

        Self {
            trees,
            alpha,
            beta,
            active_labels,
            dp_closed: &mut cache.dp_closed,
            dp_open: &mut cache.dp_open,
            t0_post_order: &cache.t0_post_order,
            t1_post_order: &cache.t1_post_order,
            t0_active: &cache.t0_active,
            max_score_under: &mut cache.max_score_under,
        }
    }

    fn solve(&mut self) -> Vec<(f64, Vec<u32>)> {
        let t0 = &self.trees[0];
        let t1 = &self.trees[1];

        let t1_nodes = t1.num_nodes();

        let mut best_l0 = vec![(NEG_INF, 0u32); t1_nodes];
        let mut best_r0 = vec![(NEG_INF, 0u32); t1_nodes];

        for &u in self.t0_post_order {
            let u_idx = u as usize;

            if t0.is_leaf(u) {
                let lbl = t0.label[u_idx];
                for &v in self.t1_post_order {
                    self.dp_closed[u_idx][v as usize] = DpClosed::default();
                    self.dp_open[u_idx][v as usize] = DpOpen::default();
                }
                if self.active_labels[lbl as usize] {
                    let v = t1.label_to_node[lbl as usize];
                    self.dp_closed[u_idx][v as usize].score = self.alpha[lbl as usize];
                }

                for &v in self.t1_post_order {
                    self.dp_open[u_idx][v as usize] = DpOpen {
                        score: self.dp_closed[u_idx][v as usize].score,
                        choice: 0,
                    };
                }
                continue;
            }

            let (l0, r0) = t0.children_pair(u);
            let l0_idx = l0 as usize;
            let r0_idx = r0 as usize;

            let l0_active = self.t0_active[l0_idx];
            let r0_active = self.t0_active[r0_idx];

            // Reset closed for u
            for &v in self.t1_post_order {
                self.dp_closed[u_idx][v as usize] = DpClosed::default();
            }

            // Compute best_l0_in_t1
            for &v in self.t1_post_order {
                let v_idx = v as usize;
                let mut max_s = if l0_active {
                    self.dp_open[l0_idx][v_idx].score
                } else {
                    NEG_INF
                };
                let mut best_v = v;

                if !t1.is_leaf(v) {
                    let (l1, r1) = t1.children_pair(v);

                    let s_l = best_l0[l1 as usize].0 - self.beta[1][l1 as usize];
                    if s_l > max_s {
                        max_s = s_l;
                        best_v = best_l0[l1 as usize].1;
                    }

                    let s_r = best_l0[r1 as usize].0 - self.beta[1][r1 as usize];
                    if s_r > max_s {
                        max_s = s_r;
                        best_v = best_l0[r1 as usize].1;
                    }
                }
                best_l0[v_idx] = (max_s, best_v);
            }

            // Compute best_r0_in_t1
            for &v in self.t1_post_order {
                let v_idx = v as usize;
                let mut max_s = if r0_active {
                    self.dp_open[r0_idx][v_idx].score
                } else {
                    NEG_INF
                };
                let mut best_v = v;

                if !t1.is_leaf(v) {
                    let (l1, r1) = t1.children_pair(v);

                    let s_l = best_r0[l1 as usize].0 - self.beta[1][l1 as usize];
                    if s_l > max_s {
                        max_s = s_l;
                        best_v = best_r0[l1 as usize].1;
                    }

                    let s_r = best_r0[r1 as usize].0 - self.beta[1][r1 as usize];
                    if s_r > max_s {
                        max_s = s_r;
                        best_v = best_r0[r1 as usize].1;
                    }
                }
                best_r0[v_idx] = (max_s, best_v);
            }

            // Combine to form dp_closed[u]
            for &v in self.t1_post_order {
                if t1.is_leaf(v) {
                    continue;
                }
                let v_idx = v as usize;
                let (l1, r1) = t1.children_pair(v);

                let mut best_c_score = NEG_INF;
                let mut v_l = 0;
                let mut v_r = 0;

                // Case A: l0 -> l1, r0 -> r1
                let s_l0_l1 = best_l0[l1 as usize].0 - self.beta[1][l1 as usize];
                let s_r0_r1 = best_r0[r1 as usize].0 - self.beta[1][r1 as usize];
                if s_l0_l1 > NEG_INF / 2.0 && s_r0_r1 > NEG_INF / 2.0 {
                    let s = s_l0_l1 + s_r0_r1
                        - self.beta[0][u_idx]
                        - self.beta[1][v_idx]
                        - self.beta[0][l0_idx]
                        - self.beta[0][r0_idx];
                    if s > best_c_score {
                        best_c_score = s;
                        v_l = best_l0[l1 as usize].1;
                        v_r = best_r0[r1 as usize].1;
                    }
                }

                // Case B: l0 -> r1, r0 -> l1
                let s_l0_r1 = best_l0[r1 as usize].0 - self.beta[1][r1 as usize];
                let s_r0_l1 = best_r0[l1 as usize].0 - self.beta[1][l1 as usize];
                if s_l0_r1 > NEG_INF / 2.0 && s_r0_l1 > NEG_INF / 2.0 {
                    let s = s_l0_r1 + s_r0_l1
                        - self.beta[0][u_idx]
                        - self.beta[1][v_idx]
                        - self.beta[0][l0_idx]
                        - self.beta[0][r0_idx];
                    if s > best_c_score {
                        best_c_score = s;
                        v_l = best_l0[r1 as usize].1;
                        v_r = best_r0[l1 as usize].1;
                    }
                }

                if best_c_score > NEG_INF / 2.0 {
                    self.dp_closed[u_idx][v_idx] = DpClosed {
                        score: best_c_score,
                        v_l,
                        v_r,
                    };
                }
            }

            // Compute dp_open[u]
            for &v in self.t1_post_order {
                let v_idx = v as usize;
                let mut best_o_score = NEG_INF;
                let mut choice = 0;

                if self.dp_closed[u_idx][v_idx].score > NEG_INF / 2.0 {
                    best_o_score = self.dp_closed[u_idx][v_idx].score
                        + self.beta[0][u_idx]
                        + self.beta[1][v_idx];
                }

                let s_l0 = if l0_active {
                    self.dp_open[l0_idx][v_idx].score - self.beta[0][l0_idx]
                } else {
                    NEG_INF
                };
                if s_l0 > best_o_score {
                    best_o_score = s_l0;
                    choice = 1;
                }

                let s_r0 = if r0_active {
                    self.dp_open[r0_idx][v_idx].score - self.beta[0][r0_idx]
                } else {
                    NEG_INF
                };
                if s_r0 > best_o_score {
                    best_o_score = s_r0;
                    choice = 2;
                }

                self.dp_open[u_idx][v_idx] = DpOpen {
                    score: best_o_score,
                    choice,
                };
            }
        }

        // Collect results
        let mut results = Vec::new();
        for u in 0..t0.num_nodes() {
            if t0.is_leaf(u as u32) {
                continue;
            }
            for v in 0..t1.num_nodes() {
                if t1.is_leaf(v as u32) {
                    continue;
                }
                let score = self.dp_closed[u][v].score;
                if score > 1.0 + 1.0e-8 {
                    let mut labels = Vec::new();
                    self.extract_closed(u as u32, v as u32, &mut labels);
                    labels.sort_unstable();
                    if labels.len() >= 2 {
                        results.push((score, labels));
                    }
                }
            }
        }

        results.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        results
    }

    fn extract_closed(&self, u: u32, v: u32, out: &mut Vec<u32>) {
        let state = &self.dp_closed[u as usize][v as usize];
        let (l0, r0) = self.trees[0].children_pair(u);
        self.extract_open(l0, state.v_l, out);
        self.extract_open(r0, state.v_r, out);
    }

    fn extract_open(&self, u: u32, v: u32, out: &mut Vec<u32>) {
        let state = &self.dp_open[u as usize][v as usize];
        if self.trees[0].is_leaf(u) && state.choice == 0 {
            out.push(self.trees[0].label[u as usize]);
            return;
        }
        match state.choice {
            0 => self.extract_closed(u, v, out),
            1 => {
                let (l0, _) = self.trees[0].children_pair(u);
                self.extract_open(l0, v, out);
            }
            2 => {
                let (_, r0) = self.trees[0].children_pair(u);
                self.extract_open(r0, v, out);
            }
            _ => unreachable!(),
        }
    }
}

fn adaptive_exact_batch_size(active_labels: usize, root_node: bool) -> usize {
    let mut batch = if active_labels >= 1200 {
        64
    } else if active_labels >= 768 {
        48
    } else if active_labels >= 384 {
        32
    } else if active_labels >= 256 {
        24
    } else {
        16
    };
    if !root_node {
        batch = batch.min(16);
    }
    batch
}
pub struct MafBranchPriceMultiSolver {
    stats: SolverStats,
}

impl Default for MafBranchPriceMultiSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl MafBranchPriceMultiSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
        }
    }
}

impl Solver for MafBranchPriceMultiSolver {
    type Config = ();
    const SUPPORTED_TRACKS: &'static [Track] = &[Track::Exact];

    fn solve(&mut self, instance: &Instance, _cfg: &RunConfig<Self::Config>) -> Option<Vec<Tree>> {
        if instance.trees.is_empty() {
            return None;
        }
        if instance.num_trees() == 1 {
            return Some(instance.trees.clone());
        }
        if instance.num_leaves <= 1 {
            return Some(instance.trees[0..1].to_vec());
        }
        solve_branch_price_multi(instance, &mut self.stats)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

#[derive(Default)]
struct ExactSubinstanceMemo {
    solutions: FxHashMap<String, Vec<Vec<u32>>>,
    hits: usize,
    stores: usize,
    skipped_ambiguous: usize,
}

struct CanonicalMemoView {
    key: String,
    label_to_canonical: Vec<u32>,
    canonical_to_label: Vec<u32>,
}

struct BpColumn {
    labels: Vec<u32>,
    covered_internal_nodes: Vec<Vec<usize>>,
    total_internal_count: usize,
}

struct ColumnBuildScratch {
    marks: Vec<Vec<u32>>,
    epochs: Vec<u32>,
}

impl ColumnBuildScratch {
    fn new(trees: &[Tree]) -> Self {
        Self {
            marks: trees.iter().map(|tree| vec![0; tree.num_nodes()]).collect(),
            epochs: vec![1; trees.len()],
        }
    }

    fn next_stamp(&mut self, ti: usize) -> u32 {
        let stamp = self.epochs[ti];
        if stamp == u32::MAX {
            self.marks[ti].fill(0);
            self.epochs[ti] = 2;
            1
        } else {
            self.epochs[ti] += 1;
            stamp
        }
    }

    fn build_column(&mut self, mut labels: Vec<u32>, trees: &[Tree]) -> BpColumn {
        labels.sort_unstable();
        labels.dedup();
        let covered_internal_nodes = if labels.len() >= 2 {
            trees
                .iter()
                .enumerate()
                .map(|(ti, tree)| self.mark_component_nodes(ti, tree, &labels))
                .collect::<Vec<_>>()
        } else {
            trees.iter().map(|_| Vec::new()).collect::<Vec<_>>()
        };
        let total_internal_count = covered_internal_nodes.iter().map(|nodes| nodes.len()).sum();
        BpColumn {
            labels,
            covered_internal_nodes,
            total_internal_count,
        }
    }

    fn mark_component_nodes(&mut self, ti: usize, tree: &Tree, labels: &[u32]) -> Vec<usize> {
        let stamp = self.next_stamp(ti);
        let marks = &mut self.marks[ti];
        let mut lca_node = tree.node_by_label(labels[0]);
        for &label in &labels[1..] {
            lca_node = tree.nearest_common_ancestor(lca_node, tree.node_by_label(label));
        }

        let mut covered = Vec::new();
        for &label in labels {
            let mut cur = tree.node_by_label(label);
            loop {
                let idx = cur as usize;
                if marks[idx] == stamp {
                    break;
                }
                marks[idx] = stamp;
                if !tree.is_leaf(cur) {
                    covered.push(idx);
                }
                if cur == lca_node {
                    break;
                }
                cur = tree.parent[idx];
            }
        }
        covered.sort_unstable();
        covered
    }
}

struct BpState {
    columns: Vec<BpColumn>,
    seen: FxHashSet<Vec<u32>>,
    reserve_columns: Vec<ReservedColumn>,
    reserve_seen: FxHashSet<Vec<u32>>,
    best_ub: usize,
    best_solution: Option<Vec<f64>>,
    nodes_explored: usize,
    cg_iterations_total: usize,
    columns_added: usize,
    t_pricer_new: f64,
    t_pricer_solve: f64,
    t_pricer_collect: f64,
    t_lp_apply_bounds: f64,
    t_lp_solve: f64,
    t_add_col: f64,
    t_cuts: f64,
    cuts_added: usize,
}

struct ReservedColumn {
    score_hint: f64,
    column: BpColumn,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct LeafPair {
    a: u32,
    b: u32,
}

impl LeafPair {
    fn new(lhs: u32, rhs: u32) -> Self {
        if lhs <= rhs {
            Self { a: lhs, b: rhs }
        } else {
            Self { a: rhs, b: lhs }
        }
    }
}

#[derive(Clone)]
struct BpNode {
    fixed_to_one: Vec<usize>,
    fixed_to_zero: Vec<usize>,
    must_link_pairs: Vec<LeafPair>,
    cannot_link_pairs: Vec<LeafPair>,
    depth: usize,
}

struct ForbiddenColumns {
    fixed_zero_labels: Vec<Vec<u32>>,
}

impl ForbiddenColumns {
    fn new(state: &BpState, node: &BpNode) -> Self {
        let fixed_zero_labels = node
            .fixed_to_zero
            .iter()
            .map(|&ci| state.columns[ci].labels.clone())
            .collect();
        Self { fixed_zero_labels }
    }

    fn contains(&self, seen: &FxHashSet<Vec<u32>>, labels: &[u32]) -> bool {
        seen.contains(labels)
            || self
                .fixed_zero_labels
                .iter()
                .any(|blocked| blocked.as_slice() == labels)
    }
}

enum NodeResult {
    Integral(usize, Vec<f64>),
    BranchPair { lp_obj: f64, pair: LeafPair },
    BranchColumn { lp_obj: f64, branch_col: usize },
    Pruned,
}

fn solve_branch_price_multi(instance: &Instance, stats: &mut SolverStats) -> Option<Vec<Tree>> {
    let mut memo = ExactSubinstanceMemo::default();
    let result = solve_branch_price_multi_cached(instance, stats, &mut memo);
    if memo.hits > 0 || memo.stores > 0 || memo.skipped_ambiguous > 0 {
        info!(
            "[bp-multi] memo: hits={} stores={} entries={} skipped_ambiguous={}",
            memo.hits,
            memo.stores,
            memo.solutions.len(),
            memo.skipped_ambiguous,
        );
    }
    result
}

fn solve_branch_price_multi_cached(
    instance: &Instance,
    stats: &mut SolverStats,
    memo: &mut ExactSubinstanceMemo,
) -> Option<Vec<Tree>> {
    let t_total = Instant::now();

    let mut config = KernelizeConfig::default();
    if !instance.protected_labels.is_empty() {
        config.protected_labels = instance.protected_labels.clone();
    }
    let kern = kernelize::kernelize_best(instance, &config);
    let reduced = &kern.instance;

    info!(
        "[bp-multi] kernelized {} -> {} leaves (m={})",
        instance.num_leaves,
        reduced.num_leaves,
        reduced.num_trees(),
    );

    if reduced.num_leaves <= 1 {
        let trivial = if reduced.num_leaves == 0 {
            vec![]
        } else {
            vec![reduced.trees[0].clone()]
        };
        let components = kernelize::expand_solution(
            trivial,
            &kern,
            instance.reference_tree(),
            instance.num_leaves,
        );
        stats.upper_bound = Some(components.len());
        stats.lower_bound = components.len();
        return Some(components);
    }

    let memo_view = if reduced.num_trees() == 2
        && (MEMO_MIN_LEAVES..=MEMO_MAX_LEAVES).contains(&reduced.num_leaves)
    {
        match canonicalize_two_tree_instance(reduced) {
            Some(view) => {
                if let Some(cached_partition) = memo.solutions.get(&view.key) {
                    memo.hits += 1;
                    let reduced_components =
                        reconstruct_cached_components(cached_partition, &view, reduced);
                    let components = kernelize::expand_solution(
                        reduced_components,
                        &kern,
                        instance.reference_tree(),
                        instance.num_leaves,
                    );
                    stats.upper_bound = Some(components.len());
                    stats.lower_bound = components.len();
                    return Some(components);
                }
                Some(view)
            }
            None => {
                memo.skipped_ambiguous += 1;
                None
            }
        }
    } else {
        None
    };

    let n = reduced.num_leaves as usize;
    let trees = &reduced.trees;
    let param_reduction_32 = kern.param_reduction;
    let mut column_builder = ColumnBuildScratch::new(trees);

    // Whidden/rspr cluster decomposition for 2-tree instances.  The default
    // path in whidden_cluster is the exact strict common-cluster split
    // (inner + outer - 1), applied recursively through this callback.  The
    // previously-prototyped batch/relaxed join variants remain opt-in inside
    // whidden_cluster because a valid AF from those variants is not by itself
    // an optimality certificate.
    if reduced.num_trees() == 2 && reduced.num_leaves >= WHIDDEN_DECOMP_MIN_LEAVES {
        if let Some(solution) =
            crate::decomp::whidden_cluster::try_whidden_decomp_2tree(
                reduced,
                &mut |subinstance| {
                    solve_branch_price_multi_cached(subinstance, &mut SolverStats::default(), memo)
                },
                &crate::decomp::whidden_cluster::NEVER_TERMINATE,
            )
        {
            if let Some(view) = memo_view.as_ref() {
                store_cached_solution(memo, view, &solution);
            }
            let exact_k = solution.len() + param_reduction_32;
            stats.lower_bound = exact_k;
            stats.upper_bound = Some(exact_k);
            let components = kernelize::expand_solution(
                solution,
                &kern,
                instance.reference_tree(),
                instance.num_leaves,
            );
            info!(
                "[bp-multi] optimal: {} components (whidden strict cluster decomp, n={}), {:.1}ms total",
                components.len(),
                reduced.num_leaves,
                t_total.elapsed().as_secs_f64() * 1000.0,
            );
            return Some(components);
        }
    }

    if RSPR_CLUSTER_DECOMP_EXPERIMENTAL && reduced.num_leaves >= RSPR_CLUSTER_MIN_LEAVES {
        if let Some(solution) =
            cluster_decomposition::try_rspr_cluster_decomposition(reduced, &mut |subinstance| {
                solve_branch_price_multi_cached(subinstance, &mut SolverStats::default(), memo)
            })
        {
            if let Some(view) = memo_view.as_ref() {
                store_cached_solution(memo, view, &solution);
            }
            let exact_k = solution.len() + param_reduction_32;
            stats.lower_bound = exact_k;
            stats.upper_bound = Some(exact_k);
            let components = kernelize::expand_solution(
                solution,
                &kern,
                instance.reference_tree(),
                instance.num_leaves,
            );
            info!(
                "[bp-multi] optimal: {} components (rspr cluster decomp), {:.1}ms total",
                components.len(),
                t_total.elapsed().as_secs_f64() * 1000.0,
            );
            return Some(components);
        }
    }

    let cluster_result = cluster_reduction::try_cluster_reduction(reduced, &mut |subinstance| {
        solve_branch_price_multi_cached(subinstance, &mut SolverStats::default(), memo)
    })?;
    match cluster_result {
        ClusterReductionResult::NotApplicable => {}
        ClusterReductionResult::Solved(solution) => {
            if let Some(view) = memo_view.as_ref() {
                store_cached_solution(memo, view, &solution.components);
            }
            let exact_k = solution.components.len() + param_reduction_32;
            stats.lower_bound = exact_k;
            stats.upper_bound = Some(exact_k);
            let components = kernelize::expand_solution(
                solution.components,
                &kern,
                instance.reference_tree(),
                instance.num_leaves,
            );
            info!(
                "[bp-multi] optimal: {} components (cluster decomp), {:.1}ms total",
                components.len(),
                t_total.elapsed().as_secs_f64() * 1000.0,
            );
            return Some(components);
        }
    }

    let bounds = compute_local_bounds(trees, reduced.num_leaves);
    let mut columns: Vec<BpColumn> = (1..=n as u32)
        .map(|label| column_builder.build_column(vec![label], trees))
        .collect();
    let mut seen: FxHashSet<Vec<u32>> = columns.iter().map(|c| c.labels.clone()).collect();

    let mut best_solution = {
        let mut values = vec![0.0; columns.len()];
        for (ci, col) in columns.iter().enumerate() {
            if col.labels.len() == 1 {
                values[ci] = 1.0;
            }
        }
        Some(values)
    };
    let mut best_ub = n;
    if let Some(partition) = &bounds.best_partition {
        let mut comp_labels: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for (leaf_idx, &comp_id) in partition.iter().enumerate() {
            comp_labels
                .entry(comp_id)
                .or_default()
                .push((leaf_idx + 1) as u32);
        }
        let mut values = vec![0.0; columns.len()];
        for labels in comp_labels.values() {
            if let Some((ci, _)) = columns
                .iter()
                .enumerate()
                .find(|(_, col)| col.labels == *labels)
            {
                values[ci] = 1.0;
            } else {
                columns.push(column_builder.build_column(labels.clone(), trees));
                values.push(1.0);
                seen.insert(labels.clone());
            }
        }
        best_solution = Some(values);
        best_ub = comp_labels.len().min(n);
    }

    // Seed the RMP with multi-leaf agreement-forest components produced by
    // Chen's 2-approximation. Only safe in the 2-tree case: a column that is
    // a valid AF component for pair (T_i, T_j) need NOT be a valid component
    // for any other tree T_k. The master LP enforces per-tree node-cover
    // (≤1 covering column per internal node) but does not check that the
    // restricted shapes T_k|X agree across all trees, so a pair-derived
    // column can be selected as part of an integer solution that fails the
    // AF validator. With m == 2 the agreement is a property of the only
    // tree pair, so the seed columns are always feasible.
    if trees.len() == 2 {
        let chen_t0 = Instant::now();
        let (_, _, leafsets) = chen_pair_agreement(&trees[0], &trees[1]);
        let mut chen_columns_added = 0usize;
        for labels in leafsets {
            if labels.len() < 2 {
                continue; // singletons are already in the initial set
            }
            if seen.contains(&labels) {
                continue;
            }
            seen.insert(labels.clone());
            columns.push(column_builder.build_column(labels, trees));
            if let Some(values) = best_solution.as_mut() {
                values.push(0.0);
            }
            chen_columns_added += 1;
        }
        info!(
            "[bp-multi] chen seed: {} columns in {:.1}ms",
            chen_columns_added,
            chen_t0.elapsed().as_secs_f64() * 1000.0,
        );
    }

    // Feed the relaxed rspr-style decomposition into B&P as an incumbent only.
    // This ports the useful "solve clusters first" idea without repeating the
    // previous correctness bug: a validated relaxed join is a feasible UB, not
    // a proof that lower_bound == upper_bound until ClusterInstance::join_cluster
    // is ported in full.
    if reduced.num_trees() == 2 && reduced.num_leaves >= WHIDDEN_DECOMP_MIN_LEAVES {
        let relaxed_t0 = Instant::now();
        if let Some(incumbent) =
            crate::decomp::whidden_cluster::try_whidden_relaxed_incumbent_2tree(reduced, &mut |sub| {
                solve_branch_price_multi_cached(sub, &mut SolverStats::default(), memo)
            }, false)
        {
            if incumbent.len() < best_ub {
                let mut values = vec![0.0; columns.len()];
                let mut ok = true;
                let mut added = 0usize;
                for component in &incumbent {
                    let labels: Vec<u32> = component.leaves().collect();
                    if labels.is_empty() {
                        continue;
                    }
                    let ci = match columns.iter().position(|col| col.labels == labels) {
                        Some(ci) => ci,
                        None => {
                            if !seen.insert(labels.clone()) {
                                ok = false;
                                break;
                            }
                            let ci = columns.len();
                            columns.push(column_builder.build_column(labels, trees));
                            values.push(0.0);
                            added += 1;
                            ci
                        }
                    };
                    if ci >= values.len() {
                        values.resize(columns.len(), 0.0);
                    }
                    values[ci] = 1.0;
                }
                if ok {
                    best_ub = incumbent.len();
                    best_solution = Some(values);
                    info!(
                        "[bp-multi] relaxed whidden incumbent: {} components, {} cols added, {:.1}ms",
                        best_ub,
                        added,
                        relaxed_t0.elapsed().as_secs_f64() * 1000.0,
                    );
                }
            }
        }
    }

    let mut state = BpState {
        columns,
        seen,
        reserve_columns: Vec::new(),
        reserve_seen: FxHashSet::default(),
        best_ub,
        best_solution,
        nodes_explored: 0,
        cg_iterations_total: 0,
        columns_added: 0,
        t_pricer_new: 0.0,
        t_pricer_solve: 0.0,
        t_pricer_collect: 0.0,
        t_lp_apply_bounds: 0.0,
        t_lp_solve: 0.0,
        t_add_col: 0.0,
        t_cuts: 0.0,
        cuts_added: 0,
    };

    let root = BpNode {
        fixed_to_one: vec![],
        fixed_to_zero: vec![],
        must_link_pairs: vec![],
        cannot_link_pairs: vec![],
        depth: 0,
    };

    let pricer_ws = MultiPricerWorkspace::new(trees, n);
    let mut dp2_cache = Dp2TreeCache::new(trees);
    let mut rmp = match PersistentRmp::new(&state.columns, trees, n) {
        Ok(rmp) => rmp,
        Err(err) => {
            error!("[bp-multi] failed to build persistent RMP: {}", err);
            return None;
        }
    };
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let result = solve_bp_node(
            &mut state,
            &node,
            trees,
            n,
            &pricer_ws,
            &mut dp2_cache,
            &mut rmp,
            &mut column_builder,
        );
        match result {
            NodeResult::Integral(obj, values) => {
                if obj < state.best_ub {
                    info!(
                        "[bp-multi] new incumbent: {} components (depth={}, nodes={})",
                        obj, node.depth, state.nodes_explored,
                    );
                    state.best_ub = obj;
                    let mut padded = values;
                    padded.resize(state.columns.len(), 0.0);
                    state.best_solution = Some(padded);
                }
            }
            NodeResult::BranchPair { lp_obj, pair } => {
                let lp_lb = (lp_obj - 1e-6).ceil() as usize;
                if lp_lb >= state.best_ub {
                    continue;
                }

                let mut right = node.clone();
                if !right.cannot_link_pairs.contains(&pair) {
                    right.cannot_link_pairs.push(pair);
                }
                right.depth += 1;

                let mut left = node.clone();
                if !left.must_link_pairs.contains(&pair) {
                    left.must_link_pairs.push(pair);
                }
                left.depth += 1;

                stack.push(right);
                stack.push(left);
            }
            NodeResult::BranchColumn { lp_obj, branch_col } => {
                let lp_lb = (lp_obj - 1e-6).ceil() as usize;
                if lp_lb >= state.best_ub {
                    continue;
                }
                let mut right = node.clone();
                right.fixed_to_zero.push(branch_col);
                right.depth += 1;

                let mut left = node.clone();
                left.fixed_to_one.push(branch_col);
                left.depth += 1;

                stack.push(right);
                stack.push(left);
            }
            NodeResult::Pruned => {}
        }
    }

    let values = state.best_solution.as_ref()?;
    let reduced_components = reconstruct_components(&state.columns, values, reduced);
    if let Some(view) = memo_view.as_ref() {
        store_cached_solution(memo, view, &reduced_components);
    }
    let components = kernelize::expand_solution(
        reduced_components,
        &kern,
        instance.reference_tree(),
        instance.num_leaves,
    );
    stats.upper_bound = Some(components.len());
    stats.lower_bound = components.len();
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    info!(
        "[bp-multi] optimal: {} components, {} B&B nodes, {} CG iters, {} cols, {:.1}ms total",
        components.len(),
        state.nodes_explored,
        state.cg_iterations_total,
        state.columns_added,
        total_ms,
    );
    info!(
        "[bp-multi] timings ms: pricer_new={:.1} pricer_solve={:.1} collect={:.1} apply_bounds={:.1} lp_solve={:.1} add_col={:.1} cuts={:.1} cuts_added={}",
        state.t_pricer_new * 1000.0,
        state.t_pricer_solve * 1000.0,
        state.t_pricer_collect * 1000.0,
        state.t_lp_apply_bounds * 1000.0,
        state.t_lp_solve * 1000.0,
        state.t_add_col * 1000.0,
        state.t_cuts * 1000.0,
        state.cuts_added,
    );
    Some(components)
}

const CANON_WL_MAX_ROUNDS: usize = 6;

fn canonicalize_two_tree_instance(instance: &Instance) -> Option<CanonicalMemoView> {
    debug_assert_eq!(instance.num_trees(), 2);
    let t0 = &instance.trees[0];
    let t1 = &instance.trees[1];
    let n = instance.num_leaves as usize;

    // WL-refinement: iteratively sharpen per-leaf colors using the colored subtree
    // codes from both trees. Two leaves converge to the same color iff they're
    // indistinguishable under the joint (T0 structure, T1 structure, path context)
    // after all refinement rounds. Most "ambiguous" leaf pairs under the pure
    // structural signature are resolved by round 2 because of cross-tree context.
    let mut leaf_color: Vec<u32> = vec![0; n + 1];
    let mut prev_classes: usize = 1;

    for _round in 0..CANON_WL_MAX_ROUNDS {
        let codes0 = colored_subtree_codes_ids(t0, &leaf_color);
        let codes1 = colored_subtree_codes_ids(t1, &leaf_color);

        let mut entries: Vec<(Vec<u32>, Vec<u32>, u32)> = Vec::with_capacity(n);
        for label in 1..=n as u32 {
            let p0 = leaf_path_codes_ids(t0, label, &codes0);
            let p1 = leaf_path_codes_ids(t1, label, &codes1);
            entries.push((p0, p1, label));
        }
        entries.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        let mut new_color = vec![0u32; n + 1];
        let mut cur_id: u32 = 0;
        for i in 0..entries.len() {
            if i > 0 && (entries[i].0 != entries[i - 1].0 || entries[i].1 != entries[i - 1].1) {
                cur_id += 1;
            }
            new_color[entries[i].2 as usize] = cur_id;
        }
        let classes = cur_id as usize + 1;

        let stable = new_color == leaf_color;
        leaf_color = new_color;
        if classes == n || stable || classes == prev_classes {
            break;
        }
        prev_classes = classes;
    }

    // Order leaves by refined color (ties broken by original label for determinism).
    // We still build the final cache key from the fully relabeled tree signatures,
    // so residual WL ambiguity can only reduce cache sharing, not create false hits.
    let mut entries: Vec<(u32, u32)> = (1..=n as u32)
        .map(|l| (leaf_color[l as usize], l))
        .collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut label_to_canonical = vec![0u32; n + 1];
    let mut canonical_to_label = vec![0u32; n + 1];
    for (new_idx, (_, label)) in entries.iter().enumerate() {
        let canon = (new_idx + 1) as u32;
        label_to_canonical[*label as usize] = canon;
        canonical_to_label[canon as usize] = *label;
    }

    let relabeled0 = t0.relabel(&label_to_canonical, instance.num_leaves);
    let relabeled1 = t1.relabel(&label_to_canonical, instance.num_leaves);
    let key = format!(
        "{}||{}",
        labeled_tree_signature(&relabeled0, relabeled0.root),
        labeled_tree_signature(&relabeled1, relabeled1.root)
    );

    Some(CanonicalMemoView {
        key,
        label_to_canonical,
        canonical_to_label,
    })
}

/// Per-round colored subtree codes. Leaves get their current `leaf_color` as ID
/// (offset by a sentinel so they can't collide with internal-node IDs); internal
/// nodes get a fresh ID per distinct (sorted) child-code pair.
fn colored_subtree_codes_ids(tree: &Tree, leaf_color: &[u32]) -> Vec<u32> {
    const INTERNAL_ID_OFFSET: u32 = 1_000_000_000;
    let mut codes = vec![0u32; tree.num_nodes()];
    let mut mapper: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    let mut next_internal: u32 = INTERNAL_ID_OFFSET;
    for node in tree.post_order() {
        codes[node as usize] = if tree.is_leaf(node) {
            let lbl = tree.label[node as usize] as usize;
            leaf_color[lbl]
        } else {
            let (l, r) = tree.children_pair(node);
            let a = codes[l as usize];
            let b = codes[r as usize];
            let key = if a <= b { (a, b) } else { (b, a) };
            *mapper.entry(key).or_insert_with(|| {
                let id = next_internal;
                next_internal += 1;
                id
            })
        };
    }
    codes
}

/// Collect sibling subtree IDs along the path from leaf → root (leaf-first order).
fn leaf_path_codes_ids(tree: &Tree, label: u32, subtree_codes: &[u32]) -> Vec<u32> {
    let mut cur = tree.node_by_label(label);
    let mut parts: Vec<u32> = Vec::new();
    while !tree.is_root(cur) {
        let parent = tree.parent[cur as usize];
        let sibling = if tree.left[parent as usize] == cur {
            tree.right[parent as usize]
        } else {
            tree.left[parent as usize]
        };
        parts.push(subtree_codes[sibling as usize]);
        cur = parent;
    }
    parts
}

fn labeled_tree_signature(tree: &Tree, node: u32) -> String {
    if tree.is_leaf(node) {
        return tree.label[node as usize].to_string();
    }
    let (left, right) = tree.children_pair(node);
    let left_sig = labeled_tree_signature(tree, left);
    let right_sig = labeled_tree_signature(tree, right);
    let (a, b) = if left_sig <= right_sig {
        (left_sig, right_sig)
    } else {
        (right_sig, left_sig)
    };
    format!("({},{})", a, b)
}

fn reconstruct_cached_components(
    cached_partition: &[Vec<u32>],
    view: &CanonicalMemoView,
    instance: &Instance,
) -> Vec<Tree> {
    let actual_groups = cached_partition
        .iter()
        .map(|group| {
            let mut labels = group
                .iter()
                .map(|&label| view.canonical_to_label[label as usize])
                .collect::<Vec<_>>();
            labels.sort_unstable();
            labels
        })
        .collect::<Vec<_>>();
    build_component_forest(
        &actual_groups,
        instance.reference_tree(),
        instance.num_leaves,
    )
}

fn store_cached_solution(
    memo: &mut ExactSubinstanceMemo,
    view: &CanonicalMemoView,
    components: &[Tree],
) {
    let mut canonical_groups = components
        .iter()
        .map(|component| {
            let mut labels = component
                .leaves()
                .map(|label| view.label_to_canonical[label as usize])
                .collect::<Vec<_>>();
            labels.sort_unstable();
            labels
        })
        .collect::<Vec<_>>();
    canonical_groups.sort_unstable();
    memo.solutions.entry(view.key.clone()).or_insert_with(|| {
        memo.stores += 1;
        canonical_groups
    });
}

fn build_component_forest(groups: &[Vec<u32>], reference: &Tree, num_leaves: u32) -> Vec<Tree> {
    groups
        .iter()
        .map(|labels| {
            let leafset = make_leafset(labels, num_leaves);
            Tree::component_from_leafset(&leafset, reference, num_leaves)
        })
        .collect()
}

fn solve_bp_node(
    state: &mut BpState,
    node: &BpNode,
    trees: &[Tree],
    num_leaves: usize,
    pricer_ws: &MultiPricerWorkspace,
    dp2_cache: &mut Dp2TreeCache,
    rmp: &mut PersistentRmp,
    column_builder: &mut ColumnBuildScratch,
) -> NodeResult {
    state.nodes_explored += 1;
    if node.fixed_to_one.len() >= state.best_ub {
        return NodeResult::Pruned;
    }
    if !node_branchings_self_consistent(&state.columns, node) {
        return NodeResult::Pruned;
    }

    let mut blocked_leaves = vec![false; num_leaves + 1];
    for &forced_ci in &node.fixed_to_one {
        for &label in &state.columns[forced_ci].labels {
            blocked_leaves[label as usize] = true;
        }
    }

    let forbidden = ForbiddenColumns::new(state, node);

    let t0 = Instant::now();
    rmp.apply_node_bounds(
        &state.columns,
        &node.fixed_to_one,
        &node.fixed_to_zero,
        &node.must_link_pairs,
        &node.cannot_link_pairs,
        &blocked_leaves,
    );
    state.t_lp_apply_bounds += t0.elapsed().as_secs_f64();

    let mut pricer_cache: Option<PairDpPricer<'_>> = None;
    let lp = loop {
        let t_solve = Instant::now();
        let lp = match rmp.solve(state.columns.len()) {
            Ok(lp) => lp,
            Err(_) => return NodeResult::Pruned,
        };
        state.t_lp_solve += t_solve.elapsed().as_secs_f64();
        if num_leaves > 500 && state.cg_iterations_total % 50 == 0 {
            info!(
                "[bp-multi] CG iter {} cols={} obj={:.4} (lp_solve={:.1}ms)",
                state.cg_iterations_total,
                state.columns.len(),
                lp.objective,
                state.t_lp_solve * 1000.0
            );
        }

        // Separate violated node ≤1 cuts FIRST so the dual values we feed to the
        // pricer reflect the tightened LP — otherwise β≡0 on unmaterialized rows
        // causes the pricer to overweight columns covering many internals.
        let t_sep = Instant::now();
        let new_cuts = rmp.separate_and_add_cuts(&state.columns, &lp.column_values, 1.0e-6);
        state.t_cuts += t_sep.elapsed().as_secs_f64();
        if new_cuts > 0 {
            state.cuts_added += new_cuts;
            state.cg_iterations_total += 1;
            continue;
        }

        let alpha = &lp.leaf_duals;
        let beta = &lp.node_duals;
        let t_reserve = Instant::now();
        let reserved_columns = collect_reserved_columns(
            state,
            alpha,
            beta,
            &blocked_leaves,
            &forbidden,
            &node.must_link_pairs,
            &node.cannot_link_pairs,
            PAIRDP_BATCH_SIZE,
        );
        state.t_pricer_collect += t_reserve.elapsed().as_secs_f64();

        let mut added_any = false;
        for (score, column) in reserved_columns {
            if score <= 1.0 + 1e-8 {
                continue;
            }
            let labels = column.labels.clone();
            let inserted = state.seen.insert(labels);
            if !inserted {
                continue;
            }
            let new_ci = state.columns.len();
            state.columns.push(column);
            let t_add = Instant::now();
            rmp.add_column(new_ci, &state.columns[new_ci], trees);
            state.t_add_col += t_add.elapsed().as_secs_f64();
            if let Some(best_solution) = state.best_solution.as_mut() {
                best_solution.push(0.0);
            }
            added_any = true;
            state.columns_added += 1;
        }
        if added_any {
            state.cg_iterations_total += 1;
            continue;
        }

        let priced_columns = price_best_new_pairdp_columns(
            &mut pricer_cache,
            pricer_ws,
            dp2_cache,
            trees,
            num_leaves,
            alpha,
            beta,
            &blocked_leaves,
            &state.seen,
            &forbidden,
            &node.must_link_pairs,
            &node.cannot_link_pairs,
            &mut state.t_pricer_new,
            &mut state.t_pricer_solve,
            &mut state.t_pricer_collect,
            node.depth == 0,
        );
        stash_reserved_columns(state, priced_columns.reserve, trees, column_builder);

        for (score, labels) in priced_columns.immediate {
            if score <= 1.0 + 1e-8 {
                continue;
            }
            let inserted = state.seen.insert(labels.clone());
            if !inserted {
                continue;
            }
            let new_ci = state.columns.len();
            state
                .columns
                .push(column_builder.build_column(labels, trees));
            let t_add = Instant::now();
            rmp.add_column(new_ci, &state.columns[new_ci], trees);
            state.t_add_col += t_add.elapsed().as_secs_f64();
            if let Some(best_solution) = state.best_solution.as_mut() {
                best_solution.push(0.0);
            }
            added_any = true;
            state.columns_added += 1;
        }

        if added_any {
            state.cg_iterations_total += 1;
            continue;
        }

        break lp;
    };
    let lp_bound = (lp.objective - 1e-6).ceil() as usize;
    if lp_bound >= state.best_ub {
        return NodeResult::Pruned;
    }

    if support_is_integral_partition(&state.columns, &lp.column_values, num_leaves) {
        let obj = lp.column_values.iter().filter(|&&v| v > 1.0e-9).count();
        return NodeResult::Integral(obj, lp.column_values);
    }

    let pair_opt = select_branch_pair(&state.columns, &lp.column_values, num_leaves);
    if let Some(pair) = pair_opt {
        return NodeResult::BranchPair {
            lp_obj: lp.objective,
            pair,
        };
    }

    let branch_col = select_branch_column(&state.columns, &lp.column_values, num_leaves);
    match branch_col {
        Some(col_idx) => NodeResult::BranchColumn {
            lp_obj: lp.objective,
            branch_col: col_idx,
        },
        None => NodeResult::Pruned,
    }
}

struct MultiPricerWorkspace {
    label_stride: usize,
    label_lca: Vec<Vec<u32>>,
    label_side_child: Vec<Vec<u32>>,
    label_node: Vec<Vec<u32>>,
    /// Per-tree, per-node bitset (width = num_leaves + 1) of descendant leaf labels.
    /// Persistent across all CG iterations and B&B nodes — depends only on tree structure.
    descendant_leaves: Vec<Vec<FixedBitSet>>,
}

impl MultiPricerWorkspace {
    fn new(trees: &[Tree], num_leaves: usize) -> Self {
        let mut descendant_leaves = Vec::with_capacity(trees.len());
        for tree in trees {
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
            descendant_leaves.push(leaves);
        }

        let stride = num_leaves + 1;
        let mut label_lca = Vec::with_capacity(trees.len());
        let mut label_side_child = Vec::with_capacity(trees.len());
        let mut label_node = Vec::with_capacity(trees.len());
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
            label_lca.push(lca_table);
            label_side_child.push(side_table);
            label_node.push(node_by_label);
        }

        Self {
            label_stride: stride,
            label_lca,
            label_side_child,
            label_node,
            descendant_leaves,
        }
    }
}

struct PairDpPricer<'a> {
    trees: &'a [Tree],
    active_labels: Vec<u32>,
    active_nodes: Vec<Vec<u32>>,
    /// Borrowed reference to workspace's persistent per-tree per-node descendant-leaf bitsets.
    /// Each inner FixedBitSet has width num_leaves+1 and is indexed by original leaf label.
    descendant_leaves: &'a [Vec<FixedBitSet>],
    /// Mask (width num_leaves+1) of currently-active labels (alpha>0 and not blocked).
    active_mask: FixedBitSet,
    /// label -> active index; u32::MAX for inactive labels.
    label_to_active_idx: Vec<u32>,
    alpha: Vec<f64>,
    beta: Vec<Vec<f64>>,
    prefix_beta: Vec<Vec<f64>>,
    sum_alpha: Vec<Vec<f64>>,
    roots: Vec<Vec<u32>>,
    side_child: Vec<Vec<u32>>,
    pair_penalty: Vec<f64>,
    pair_ub: Vec<f64>,
    pair_singleton_penalty: Vec<f64>,
    memo_pair: Vec<f64>,
    memo_side_score: Vec<f64>,
    memo_side_split: Vec<u32>,
    memo_pair_labels: Vec<Option<Vec<u32>>>,
}

struct PricingColumns {
    immediate: Vec<(f64, Vec<u32>)>,
    reserve: Vec<(f64, Vec<u32>)>,
}

impl<'a> PairDpPricer<'a> {
    fn collect_active_labels(
        num_leaves: usize,
        alpha: &[f64],
        blocked_leaves: &[bool],
    ) -> Vec<u32> {
        (1..=num_leaves as u32)
            .filter(|&label| !blocked_leaves[label as usize] && alpha[label as usize] > 1.0e-12)
            .collect()
    }

    fn with_active_labels(
        workspace: &'a MultiPricerWorkspace,
        trees: &'a [Tree],
        num_leaves: usize,
        active_labels: Vec<u32>,
        alpha: &[f64],
        beta: &[Vec<f64>],
    ) -> Self {
        let p = active_labels.len();
        let pair_count = p * p;

        let mut active_nodes = Vec::with_capacity(trees.len());
        let mut prefix_beta = Vec::with_capacity(trees.len());
        let mut sum_alpha = Vec::with_capacity(trees.len());
        let mut roots = Vec::with_capacity(trees.len());
        let mut side_child = Vec::with_capacity(trees.len());

        let mut active_mask = FixedBitSet::with_capacity(num_leaves + 1);
        let mut label_to_active_idx = vec![u32::MAX; num_leaves + 1];
        for (ai, &label) in active_labels.iter().enumerate() {
            active_mask.insert(label as usize);
            label_to_active_idx[label as usize] = ai as u32;
        }

        for (ti, tree) in trees.iter().enumerate() {
            let node_by_label = &workspace.label_node[ti];
            let nodes: Vec<u32> = active_labels
                .iter()
                .map(|&label| node_by_label[label as usize])
                .collect();
            active_nodes.push(nodes);

            let mut prefix = vec![0.0; tree.num_nodes()];
            let mut s_alpha = vec![0.0; tree.num_nodes()];
            for node in tree.post_order() {
                if tree.is_leaf(node) {
                    s_alpha[node as usize] = alpha[tree.label[node as usize] as usize].max(0.0);
                } else {
                    let (l, r) = tree.children_pair(node);
                    s_alpha[node as usize] = s_alpha[l as usize] + s_alpha[r as usize];
                }
            }
            for node in tree.pre_order() {
                let parent = tree.parent[node as usize];
                let parent_sum = if parent == klados_core::NONE {
                    0.0
                } else {
                    prefix[parent as usize]
                };
                let own = if tree.is_leaf(node) {
                    0.0
                } else {
                    beta[ti][node as usize]
                };
                prefix[node as usize] = parent_sum + own;
            }
            prefix_beta.push(prefix);
            sum_alpha.push(s_alpha);

            let stride = workspace.label_stride;
            let lca_table = &workspace.label_lca[ti];
            let side_table = &workspace.label_side_child[ti];
            let mut tree_roots = vec![0u32; pair_count];
            let mut tree_side_child = vec![0u32; pair_count];
            for a in 0..p {
                let la = active_labels[a] as usize;
                let base_ws = la * stride;
                let base_out = a * p;
                for b in 0..p {
                    let lb = active_labels[b] as usize;
                    let ws_idx = base_ws + lb;
                    let out_idx = base_out + b;
                    tree_roots[out_idx] = lca_table[ws_idx];
                    tree_side_child[out_idx] = side_table[ws_idx];
                }
            }
            roots.push(tree_roots);
            side_child.push(tree_side_child);
        }

        let mut pricer = Self {
            trees,
            active_labels,
            active_nodes,
            descendant_leaves: &workspace.descendant_leaves,
            active_mask,
            label_to_active_idx,
            alpha: alpha.to_vec(),
            beta: beta.to_vec(),
            prefix_beta,
            sum_alpha,
            roots,
            side_child,
            pair_penalty: vec![0.0; pair_count],
            pair_ub: vec![f64::INFINITY; pair_count],
            pair_singleton_penalty: vec![0.0; pair_count],
            memo_pair: vec![f64::NAN; pair_count],
            memo_side_score: vec![f64::NAN; pair_count],
            memo_side_split: vec![u32::MAX; pair_count],
            memo_pair_labels: vec![None; pair_count],
        };
        pricer.recompute_pair_arrays();
        pricer
    }

    fn same_active_set(&self, active_labels: &[u32]) -> bool {
        self.active_labels.as_slice() == active_labels
    }

    fn refresh_duals(&mut self, alpha: &[f64], beta: &[Vec<f64>]) {
        self.alpha.clear();
        self.alpha.extend_from_slice(alpha);

        if self.beta.len() != beta.len() {
            self.beta = beta.to_vec();
        } else {
            for (dst, src) in self.beta.iter_mut().zip(beta.iter()) {
                dst.clear();
                dst.extend_from_slice(src);
            }
        }

        self.refresh_prefix_beta();
        self.recompute_pair_arrays();
        self.reset_memos();
    }

    fn recompute_pair_arrays(&mut self) {
        let p = self.active_labels.len();
        self.pair_penalty.fill(0.0);
        self.pair_ub.fill(f64::INFINITY);
        self.pair_singleton_penalty.fill(0.0);
        for (ti, tree) in self.trees.iter().enumerate() {
            let mut nps = vec![0.0; tree.num_nodes()];
            for node in 0..tree.num_nodes() {
                let parent = tree.parent[node];
                nps[node] = if parent == klados_core::NONE {
                    0.0
                } else {
                    self.prefix_beta[ti][parent as usize]
                };
            }
            for a in 0..p {
                let base = a * p;
                let desc = self.active_nodes[ti][a];
                for c in 0..p {
                    let idx = base + c;
                    let r = self.roots[ti][idx] as usize;
                    self.pair_penalty[idx] += nps[r];
                    if self.sum_alpha[ti][r] < self.pair_ub[idx] {
                        self.pair_ub[idx] = self.sum_alpha[ti][r];
                    }

                    // Singleton penalty calculation
                    let anc = self.side_child[ti][idx];
                    let upper = if desc != klados_core::NONE {
                        let dp = tree.parent[desc as usize];
                        if dp != klados_core::NONE {
                            self.prefix_beta[ti][dp as usize]
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    let lower = if anc != klados_core::NONE {
                        let ap = tree.parent[anc as usize];
                        if ap != klados_core::NONE {
                            self.prefix_beta[ti][ap as usize]
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    let diff = upper - lower;
                    if anc != desc && diff > 0.0 {
                        self.pair_singleton_penalty[idx] += diff;
                    }
                }
            }
        }
    }

    fn refresh_prefix_beta(&mut self) {
        if self.prefix_beta.len() != self.trees.len() {
            self.prefix_beta = self
                .trees
                .iter()
                .map(|tree| vec![0.0; tree.num_nodes()])
                .collect();
            self.sum_alpha = self
                .trees
                .iter()
                .map(|tree| vec![0.0; tree.num_nodes()])
                .collect();
        }
        for (ti, tree) in self.trees.iter().enumerate() {
            let prefix = &mut self.prefix_beta[ti];
            let s_alpha = &mut self.sum_alpha[ti];
            if prefix.len() != tree.num_nodes() {
                prefix.resize(tree.num_nodes(), 0.0);
                s_alpha.resize(tree.num_nodes(), 0.0);
            }
            for node in tree.post_order() {
                if tree.is_leaf(node) {
                    s_alpha[node as usize] =
                        self.alpha[tree.label[node as usize] as usize].max(0.0);
                } else {
                    let (l, r) = tree.children_pair(node);
                    s_alpha[node as usize] = s_alpha[l as usize] + s_alpha[r as usize];
                }
            }
            for node in tree.pre_order() {
                let parent = tree.parent[node as usize];
                let parent_sum = if parent == klados_core::NONE {
                    0.0
                } else {
                    prefix[parent as usize]
                };
                let own = if tree.is_leaf(node) {
                    0.0
                } else {
                    self.beta[ti][node as usize]
                };
                prefix[node as usize] = parent_sum + own;
            }
        }
    }

    fn reset_memos(&mut self) {
        self.memo_pair.fill(f64::NAN);
        self.memo_side_score.fill(f64::NAN);
        self.memo_side_split.fill(u32::MAX);
        for slot in &mut self.memo_pair_labels {
            *slot = None;
        }
    }

    #[inline(always)]
    fn root_of(&self, ti: usize, a: usize, b: usize) -> u32 {
        self.roots[ti][self.pair_idx(a, b)]
    }

    #[inline(always)]
    fn side_of(&self, ti: usize, a: usize, b: usize) -> u32 {
        self.side_child[ti][self.pair_idx(a, b)]
    }

    fn collect_profitable_columns<F>(
        &mut self,
        limit: usize,
        stash_limit: usize,
        mut accept: F,
    ) -> Vec<(f64, Vec<u32>)>
    where
        F: FnMut(&[u32]) -> bool,
    {
        if self.active_labels.len() < 2 || limit == 0 {
            return Vec::new();
        }

        let p = self.active_labels.len();
        let mut proxy_order: Vec<(f64, usize, usize)> =
            Vec::with_capacity(p.saturating_mul(p.saturating_sub(1)) / 2);
        for a in 0..p {
            for b in (a + 1)..p {
                proxy_order.push((self.quick_pair_proxy(a, b), a, b));
            }
        }
        let target = limit.saturating_add(stash_limit);

        proxy_order.sort_unstable_by(|lhs, rhs| {
            rhs.0
                .total_cmp(&lhs.0)
                .then_with(|| lhs.1.cmp(&rhs.1))
                .then_with(|| lhs.2.cmp(&rhs.2))
        });

        let mut dedup: FxHashMap<Vec<u32>, f64> = FxHashMap::default();
        let mut evaluated = 0;
        for &(_, a, b) in proxy_order.iter() {
            if a >= p || b >= p {
                continue;
            }
            if self.pair_ub[a * p + b] <= 1.0 + 1.0e-8 {
                continue;
            }

            evaluated += 1;
            let score = self.solve_pair(a, b);
            if score > 1.0 + 1.0e-8 {
                let labels = self.pair_labels(a, b);
                if labels.len() >= 2 && accept(&labels) {
                    dedup
                        .entry(labels)
                        .and_modify(|best| *best = best.max(score))
                        .or_insert(score);
                }
            }
            if dedup.len() >= target && evaluated >= 16 {
                break;
            }
        }

        let mut out = dedup
            .into_iter()
            .map(|(labels, score)| (score, labels))
            .collect::<Vec<_>>();
        out.sort_unstable_by(|lhs, rhs| {
            rhs.0
                .total_cmp(&lhs.0)
                .then_with(|| rhs.1.len().cmp(&lhs.1.len()))
                .then_with(|| lhs.1.cmp(&rhs.1))
        });
        if out.len() > target {
            out.truncate(target);
        }
        out
    }

    fn collect_proxy_columns<F>(
        &mut self,
        limit: usize,
        stash_limit: usize,
        _anchor_budget: usize,
        _partners_per_anchor: usize,
        pair_trial_limit: usize,
        mut accept: F,
    ) -> Vec<(f64, Vec<u32>)>
    where
        F: FnMut(&[u32]) -> bool,
    {
        if self.active_labels.len() < 2 || limit == 0 {
            return Vec::new();
        }

        let p = self.active_labels.len();
        let mut proxy_order: Vec<(f64, usize, usize)> =
            Vec::with_capacity(p.saturating_mul(p.saturating_sub(1)) / 2);
        for a in 0..p {
            for b in (a + 1)..p {
                proxy_order.push((self.quick_pair_proxy(a, b), a, b));
            }
        }

        let target = limit.saturating_add(stash_limit);
        let trial_limit = pair_trial_limit.min(proxy_order.len());

        if trial_limit < proxy_order.len() {
            proxy_order.select_nth_unstable_by(trial_limit, |lhs, rhs| {
                rhs.0
                    .total_cmp(&lhs.0)
                    .then_with(|| lhs.1.cmp(&rhs.1))
                    .then_with(|| lhs.2.cmp(&rhs.2))
            });
            let (head, _) = proxy_order.split_at_mut(trial_limit);
            head.sort_unstable_by(|lhs, rhs| {
                rhs.0
                    .total_cmp(&lhs.0)
                    .then_with(|| lhs.1.cmp(&rhs.1))
                    .then_with(|| lhs.2.cmp(&rhs.2))
            });
        } else {
            proxy_order.sort_unstable_by(|lhs, rhs| {
                rhs.0
                    .total_cmp(&lhs.0)
                    .then_with(|| lhs.1.cmp(&rhs.1))
                    .then_with(|| lhs.2.cmp(&rhs.2))
            });
        }

        let mut dedup: FxHashMap<Vec<u32>, f64> = FxHashMap::default();
        for &(_, a, b) in proxy_order.iter().take(trial_limit) {
            if a >= p || b >= p {
                continue;
            }
            let score = self.solve_pair(a, b);
            if score <= 1.0 + 1.0e-8 {
                continue;
            }
            let labels = self.pair_labels(a, b);
            if labels.len() < 2 || !accept(&labels) {
                continue;
            }
            dedup
                .entry(labels)
                .and_modify(|best| *best = best.max(score))
                .or_insert(score);
            if dedup.len() >= target {
                break;
            }
        }

        let mut out = dedup
            .into_iter()
            .map(|(labels, score)| (score, labels))
            .collect::<Vec<_>>();
        out.sort_unstable_by(|lhs, rhs| {
            rhs.0
                .total_cmp(&lhs.0)
                .then_with(|| rhs.1.len().cmp(&lhs.1.len()))
                .then_with(|| lhs.1.cmp(&rhs.1))
        });
        if out.len() > target {
            out.truncate(target);
        }
        out
    }

    fn collect_fast_columns<F>(
        &mut self,
        limit: usize,
        stash_limit: usize,
        accept: F,
    ) -> Vec<(f64, Vec<u32>)>
    where
        F: FnMut(&[u32]) -> bool,
    {
        self.collect_proxy_columns(
            limit,
            stash_limit,
            FASTPRICER_ANCHOR_LABELS,
            FASTPRICER_PARTNERS_PER_ANCHOR,
            FASTPRICER_PAIR_TRIALS,
            accept,
        )
    }

    fn collect_wide_columns<F>(
        &mut self,
        limit: usize,
        stash_limit: usize,
        accept: F,
    ) -> Vec<(f64, Vec<u32>)>
    where
        F: FnMut(&[u32]) -> bool,
    {
        self.collect_proxy_columns(
            limit,
            stash_limit,
            WIDEPRICER_ANCHOR_LABELS,
            WIDEPRICER_PARTNERS_PER_ANCHOR,
            WIDEPRICER_PAIR_TRIALS,
            accept,
        )
    }

    fn pair_idx(&self, a: usize, b: usize) -> usize {
        a * self.active_labels.len() + b
    }

    fn solve_pair(&mut self, a: usize, b: usize) -> f64 {
        debug_assert!(a != b);
        let idx = self.pair_idx(a, b);
        let val = self.memo_pair[idx];
        if !val.is_nan() {
            if val.is_infinite() && val.is_sign_positive() {
                return NEG_INF;
            }
            return val;
        }
        self.memo_pair[idx] = f64::INFINITY;

        let left = self.solve_side(a, b);
        let right = self.solve_side(b, a);
        let score = if left <= NEG_INF / 2.0 || right <= NEG_INF / 2.0 {
            NEG_INF
        } else {
            -self.root_penalty(a, b) + left + right
        };

        self.memo_pair[idx] = score;
        score
    }

    fn solve_side(&mut self, a: usize, b: usize) -> f64 {
        debug_assert!(a != b);
        let idx = self.pair_idx(a, b);
        let val = self.memo_side_score[idx];
        if !val.is_nan() {
            if val.is_infinite() && val.is_sign_positive() {
                return NEG_INF;
            }
            return val;
        }
        self.memo_side_score[idx] = f64::INFINITY;

        let label_a = self.active_labels[a];
        let label_b = self.active_labels[b];
        let mut best_score = self.alpha[label_a as usize] - self.pair_singleton_penalty[idx];
        let mut best_choice = u32::MAX - 1;

        let num_trees = self.trees.len();
        let pair_ix = a * self.active_labels.len() + b;
        let p = self.active_labels.len();

        let mut b_penalty_sum = 0.0;
        let mut side_nodes = [0u32; 64];
        debug_assert!(num_trees <= 64);
        for ti in 0..num_trees {
            let anc = self.side_child[ti][pair_ix];
            side_nodes[ti] = anc;
            let parent = self.trees[ti].parent[anc as usize];
            if parent != klados_core::NONE {
                b_penalty_sum += self.prefix_beta[ti][parent as usize];
            }
        }

        // `descendant_leaves` is `&'a [...]` (Copy), so this does not reborrow self.
        let desc: &[Vec<FixedBitSet>] = self.descendant_leaves;
        // Clone the active mask so we can borrow it while calling &mut self methods.
        let am = self.active_mask.clone();
        let am_slice = am.as_slice();

        const BLOCK_BITS: usize = std::mem::size_of::<usize>() * 8;
        const BLOCK_SHIFT: usize = BLOCK_BITS.trailing_zeros() as usize;
        const BLOCK_MASK: usize = BLOCK_BITS - 1;

        let la = label_a as usize;
        let lb = label_b as usize;
        let la_w = la >> BLOCK_SHIFT;
        let la_m = 1usize << (la & BLOCK_MASK);
        let lb_w = lb >> BLOCK_SHIFT;
        let lb_m = 1usize << (lb & BLOCK_MASK);
        let d0 = desc[0][side_nodes[0] as usize].as_slice();

        for wi in 0..d0.len() {
            let mut w = d0[wi] & am_slice[wi];
            for ti in 1..num_trees {
                w &= desc[ti][side_nodes[ti] as usize].as_slice()[wi];
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

                let idx_c = a * p + c;
                let pen = self.pair_penalty[idx_c] - b_penalty_sum;
                let ub = self.pair_ub[idx_c];
                let max_possible = ub - pen;
                if max_possible <= best_score + 1.0e-12 {
                    continue;
                }

                let cached_val = self.memo_pair[idx_c];
                let child_score = if !cached_val.is_nan() {
                    if cached_val.is_infinite() && cached_val.is_sign_positive() {
                        NEG_INF
                    } else {
                        cached_val
                    }
                } else {
                    self.solve_pair(a, c)
                };

                if child_score <= NEG_INF / 2.0 {
                    continue;
                }
                let cand = child_score - pen;
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

    fn collect_pair(&mut self, a: usize, b: usize, out: &mut Vec<u32>) {
        self.collect_side(a, b, out);
        self.collect_side(b, a, out);
    }

    fn pair_labels(&mut self, a: usize, b: usize) -> Vec<u32> {
        let idx = self.pair_idx(a, b);
        if let Some(labels) = self.memo_pair_labels[idx].clone() {
            return labels;
        }
        let mut labels = Vec::new();
        self.collect_pair(a, b, &mut labels);
        labels.sort_unstable();
        labels.dedup();
        self.memo_pair_labels[idx] = Some(labels.clone());
        labels
    }

    fn collect_side(&mut self, a: usize, b: usize, out: &mut Vec<u32>) {
        let idx = self.pair_idx(a, b);
        let score = self.memo_side_score[idx];
        if score.is_nan() || score.is_infinite() {
            self.solve_side(a, b);
        }
        let choice = self.memo_side_split[idx];
        if choice == u32::MAX - 1 {
            out.push(self.active_labels[a]);
        } else {
            self.collect_pair(a, choice as usize, out);
        }
    }

    fn root_penalty(&self, a: usize, b: usize) -> f64 {
        self.trees
            .iter()
            .enumerate()
            .map(|(ti, _)| self.beta[ti][self.root_of(ti, a, b) as usize])
            .sum()
    }

    fn quick_pair_proxy(&self, a: usize, b: usize) -> f64 {
        let label_a = self.active_labels[a] as usize;
        let label_b = self.active_labels[b] as usize;
        let p = self.active_labels.len();
        self.alpha[label_a] + self.alpha[label_b]
            - self.root_penalty(a, b)
            - self.pair_singleton_penalty[a * p + b]
            - self.pair_singleton_penalty[b * p + a]
    }
}

fn price_best_new_pairdp_columns<'a>(
    pricer_cache: &mut Option<PairDpPricer<'a>>,
    pricer_ws: &'a MultiPricerWorkspace,
    dp2_cache: &mut Dp2TreeCache,
    trees: &'a [Tree],
    num_leaves: usize,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    seen: &FxHashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
    must_link_pairs: &[LeafPair],
    cannot_link_pairs: &[LeafPair],
    t_new: &mut f64,
    t_solve: &mut f64,
    t_collect: &mut f64,
    allow_wide: bool,
) -> PricingColumns {
    let t0 = Instant::now();
    let active_labels = PairDpPricer::collect_active_labels(num_leaves, alpha, blocked_leaves);
    let pricer = if let Some(pricer) = pricer_cache.as_mut() {
        if pricer.same_active_set(&active_labels) {
            pricer.refresh_duals(alpha, beta);
            pricer
        } else {
            *pricer_cache = Some(PairDpPricer::with_active_labels(
                pricer_ws,
                trees,
                num_leaves,
                active_labels,
                alpha,
                beta,
            ));
            pricer_cache.as_mut().expect("pricer cache present")
        }
    } else {
        *pricer_cache = Some(PairDpPricer::with_active_labels(
            pricer_ws,
            trees,
            num_leaves,
            active_labels,
            alpha,
            beta,
        ));
        pricer_cache.as_mut().expect("pricer cache present")
    };
    *t_new += t0.elapsed().as_secs_f64();
    let exact_batch_size = adaptive_exact_batch_size(pricer.active_labels.len(), allow_wide);

    let t1 = Instant::now();
    let fast = pricer.collect_fast_columns(
        FASTPRICER_BATCH_SIZE.min(exact_batch_size),
        COLUMN_RESERVE_REFILL,
        |labels| {
            pricing_candidate_allowed(labels, seen, forbidden, must_link_pairs, cannot_link_pairs)
        },
    );
    *t_solve += t1.elapsed().as_secs_f64();
    if !fast.is_empty() {
        let mut reserve = fast;
        let immediate = reserve
            .drain(
                ..reserve
                    .len()
                    .min(FASTPRICER_BATCH_SIZE.min(exact_batch_size)),
            )
            .collect();
        return PricingColumns { immediate, reserve };
    }

    if allow_wide && pricer.active_labels.len() >= WIDEPRICER_MIN_ACTIVE_LABELS {
        let t1b = Instant::now();
        let wide = pricer.collect_wide_columns(
            WIDEPRICER_BATCH_SIZE.min(exact_batch_size),
            COLUMN_RESERVE_REFILL,
            |labels| {
                pricing_candidate_allowed(
                    labels,
                    seen,
                    forbidden,
                    must_link_pairs,
                    cannot_link_pairs,
                )
            },
        );
        *t_solve += t1b.elapsed().as_secs_f64();
        if !wide.is_empty() {
            let mut reserve = wide;
            let immediate = reserve
                .drain(
                    ..reserve
                        .len()
                        .min(WIDEPRICER_BATCH_SIZE.min(exact_batch_size)),
                )
                .collect();
            return PricingColumns { immediate, reserve };
        }
    }

    let has_constraints = !forbidden.fixed_zero_labels.is_empty()
        || !must_link_pairs.is_empty()
        || !cannot_link_pairs.is_empty();

    let t2 = Instant::now();
    let exact = if trees.len() == 2 && !has_constraints {
        let mut dp2 =
            ExactPricer2Tree::new(trees, num_leaves, alpha, beta, blocked_leaves, dp2_cache);
        let mut all_results = dp2.solve();
        all_results.retain(|(_, labels)| {
            pricing_candidate_allowed(labels, seen, forbidden, must_link_pairs, cannot_link_pairs)
        });
        all_results.truncate(exact_batch_size + COLUMN_RESERVE_REFILL);
        all_results
    } else {
        pricer.collect_profitable_columns(exact_batch_size, COLUMN_RESERVE_REFILL, |labels| {
            pricing_candidate_allowed(labels, seen, forbidden, must_link_pairs, cannot_link_pairs)
        })
    };
    *t_solve += t2.elapsed().as_secs_f64();

    let t_col = Instant::now();
    let mut reserve = exact;
    let immediate = reserve
        .drain(..reserve.len().min(exact_batch_size))
        .collect();
    *t_collect += t_col.elapsed().as_secs_f64();
    PricingColumns { immediate, reserve }
}

fn column_pricing_score(col: &BpColumn, alpha: &[f64], beta: &[Vec<f64>]) -> f64 {
    let leaf_gain: f64 = col.labels.iter().map(|&label| alpha[label as usize]).sum();
    let node_penalty: f64 = col
        .covered_internal_nodes
        .iter()
        .enumerate()
        .map(|(ti, nodes)| nodes.iter().map(|&node| beta[ti][node]).sum::<f64>())
        .sum();
    leaf_gain - node_penalty
}

fn stash_reserved_columns(
    state: &mut BpState,
    reserve: Vec<(f64, Vec<u32>)>,
    trees: &[Tree],
    column_builder: &mut ColumnBuildScratch,
) {
    if COLUMN_RESERVE_CAP == 0 || reserve.is_empty() {
        return;
    }
    for (score_hint, labels) in reserve {
        if state.seen.contains(&labels) || state.reserve_seen.contains(&labels) {
            continue;
        }
        let col = column_builder.build_column(labels.clone(), trees);
        state.reserve_seen.insert(labels);
        state.reserve_columns.push(ReservedColumn {
            score_hint,
            column: col,
        });
    }
    if state.reserve_columns.len() > COLUMN_RESERVE_CAP {
        state.reserve_columns.sort_unstable_by(|lhs, rhs| {
            rhs.score_hint
                .total_cmp(&lhs.score_hint)
                .then_with(|| rhs.column.labels.len().cmp(&lhs.column.labels.len()))
                .then_with(|| lhs.column.labels.cmp(&rhs.column.labels))
        });
        while state.reserve_columns.len() > COLUMN_RESERVE_CAP {
            if let Some(removed) = state.reserve_columns.pop() {
                state.reserve_seen.remove(&removed.column.labels);
            }
        }
    }
}

fn collect_reserved_columns(
    state: &mut BpState,
    alpha: &[f64],
    beta: &[Vec<f64>],
    blocked_leaves: &[bool],
    forbidden: &ForbiddenColumns,
    must_link_pairs: &[LeafPair],
    cannot_link_pairs: &[LeafPair],
    limit: usize,
) -> Vec<(f64, BpColumn)> {
    if limit == 0 || state.reserve_columns.is_empty() {
        return Vec::new();
    }

    let mut stale = Vec::new();
    let mut scored = Vec::new();
    for (ri, entry) in state.reserve_columns.iter().enumerate() {
        let labels = &entry.column.labels;
        if state.seen.contains(labels) {
            stale.push(ri);
            continue;
        }
        if labels.iter().any(|&label| blocked_leaves[label as usize]) {
            continue;
        }
        if forbidden.contains(&state.seen, labels)
            || !labels_satisfy_pair_constraints(labels, must_link_pairs, cannot_link_pairs)
        {
            continue;
        }
        let score = column_pricing_score(&entry.column, alpha, beta);
        if score > 1.0 + 1.0e-8 {
            scored.push((score, ri));
        }
    }

    if scored.is_empty() && stale.is_empty() {
        return Vec::new();
    }

    scored.sort_unstable_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0).then_with(|| lhs.1.cmp(&rhs.1)));
    if scored.len() > limit {
        scored.truncate(limit);
    }

    let mut selected_scores: FxHashMap<usize, f64> = FxHashMap::default();
    for (score, ri) in scored {
        selected_scores.insert(ri, score);
    }

    let mut remove = stale;
    remove.extend(selected_scores.keys().copied());
    remove.sort_unstable();
    remove.dedup();

    let mut out = Vec::with_capacity(selected_scores.len());
    for &ri in remove.iter().rev() {
        let removed = state.reserve_columns.swap_remove(ri);
        state.reserve_seen.remove(&removed.column.labels);
        if let Some(score) = selected_scores.remove(&ri) {
            out.push((score, removed.column));
        }
    }
    out.sort_unstable_by(|lhs, rhs| {
        rhs.0
            .total_cmp(&lhs.0)
            .then_with(|| rhs.1.labels.len().cmp(&lhs.1.labels.len()))
            .then_with(|| lhs.1.labels.cmp(&rhs.1.labels))
    });
    out
}

struct RmpLpResult {
    objective: f64,
    column_values: Vec<f64>,
    leaf_duals: Vec<f64>,
    node_duals: Vec<Vec<f64>>,
}

struct PersistentRmp {
    model: Option<Model>,
    leaf_row_idx: Vec<usize>,
    /// Row index for each (tree, internal-node). `None` until the row is materialized
    /// lazily via cut separation when violated.
    node_row_idx: Vec<Vec<Option<usize>>>,
    /// Total number of rows currently in the HiGHS model.
    num_rows: usize,
    /// Reverse index: node_to_cols[ti][node_idx] = list of global column indices whose
    /// column covers that internal node. Populated for every column regardless of whether
    /// the row has been materialized — used to build the row's coefficient vector when
    /// materialized later.
    node_to_cols: Vec<Vec<Vec<usize>>>,
    /// HiGHS column index for each global column (parallel to state.columns).
    col_idx: Vec<i32>,
    /// Current lower bound applied in HiGHS for each added column.
    current_lower: Vec<f64>,
    /// Current upper bound applied in HiGHS for each added column.
    current_upper: Vec<f64>,
    fixed_one_mark: Vec<u32>,
    fixed_zero_mark: Vec<u32>,
    mark_epoch: u32,
}

impl PersistentRmp {
    fn new(columns: &[BpColumn], trees: &[Tree], num_leaves: usize) -> Result<Self, String> {
        let mut model = Model::new(ColProblem::default());
        model.make_quiet();
        model.set_option("threads", 1_i32);
        // Presolve is wasteful when we warm-start with an existing basis on every solve.
        // We change only column bounds / add columns between solves, so dual simplex
        // can resume from the stored basis directly.
        model.set_option("presolve", "off");
        model.set_option("solver", "simplex");
        model.set_option("simplex_strategy", 1_i32); // 1 = dual simplex (best for warm-start with bound changes)

        // Leaf rows are added eagerly — every column necessarily covers at least one leaf.
        for leaf in 0..=num_leaves {
            if leaf == 0 {
                model.add_row(0.0..=0.0, Vec::new());
            } else {
                model.add_row(1.0..=1.0, Vec::new());
            }
        }
        let leaf_row_idx: Vec<usize> = (0..=num_leaves).collect();
        let num_rows = num_leaves + 1;

        // Internal-node rows are materialized lazily — only if a LP solution violates
        // the ≤1 constraint for that node. Most internal nodes never become tight.
        let node_row_idx: Vec<Vec<Option<usize>>> = trees
            .iter()
            .map(|tree| vec![None; tree.num_nodes()])
            .collect();
        let node_to_cols: Vec<Vec<Vec<usize>>> = trees
            .iter()
            .map(|tree| vec![Vec::new(); tree.num_nodes()])
            .collect();

        let mut rmp = Self {
            model: Some(model),
            leaf_row_idx,
            node_row_idx,
            num_rows,
            node_to_cols,
            col_idx: Vec::new(),
            current_lower: Vec::new(),
            current_upper: Vec::new(),
            fixed_one_mark: Vec::new(),
            fixed_zero_mark: Vec::new(),
            mark_epoch: 1,
        };
        for (ci, col) in columns.iter().enumerate() {
            rmp.add_column(ci, col, trees);
        }
        Ok(rmp)
    }

    fn add_column(&mut self, global_ci: usize, col: &BpColumn, trees: &[Tree]) {
        debug_assert_eq!(global_ci, self.col_idx.len());
        // Build row-index/value vectors for the C API.
        let mut row_indices: Vec<i32> = col
            .labels
            .iter()
            .map(|&label| self.leaf_row_idx[label as usize] as i32)
            .collect();
        if col.labels.len() >= 2 {
            for (ti, tree) in trees.iter().enumerate() {
                for &node in &col.covered_internal_nodes[ti] {
                    debug_assert!(!tree.is_leaf(node as u32));
                    // Always record the coverage in the reverse index — the row may be
                    // materialized later and will need to know which columns cover it.
                    self.node_to_cols[ti][node].push(global_ci);
                    if let Some(ri) = self.node_row_idx[ti][node] {
                        row_indices.push(ri as i32);
                    }
                }
            }
        }
        let values: Vec<f64> = vec![1.0; row_indices.len()];
        let ptr = self.model.as_mut().expect("RMP model present").as_mut_ptr();
        unsafe {
            highs_sys::Highs_addCol(
                ptr,
                1.0,           // cost
                0.0,           // lower bound
                f64::INFINITY, // upper bound
                row_indices.len() as i32,
                row_indices.as_ptr(),
                values.as_ptr(),
            );
        }
        self.col_idx.push(global_ci as i32);
        self.current_lower.push(0.0);
        self.current_upper.push(f64::INFINITY);
        self.fixed_one_mark.push(0);
        self.fixed_zero_mark.push(0);
    }

    /// Materialize an internal-node row lazily. Pulls all currently-alive columns that
    /// cover (ti, node) from the reverse index and adds a ≤1 constraint over them.
    fn add_node_row_lazy(&mut self, ti: usize, node: usize) {
        debug_assert!(self.node_row_idx[ti][node].is_none());
        let cols_covering = &self.node_to_cols[ti][node];
        let indices: Vec<i32> = cols_covering.iter().map(|&ci| self.col_idx[ci]).collect();
        let values: Vec<f64> = vec![1.0; indices.len()];
        let ptr = self.model.as_mut().expect("RMP model present").as_mut_ptr();
        unsafe {
            highs_sys::Highs_addRow(
                ptr,
                -f64::INFINITY,
                1.0,
                indices.len() as i32,
                indices.as_ptr(),
                values.as_ptr(),
            );
        }
        self.node_row_idx[ti][node] = Some(self.num_rows);
        self.num_rows += 1;
    }

    /// Scan the LP support for violated node ≤1 constraints and materialize them.
    /// Returns number of rows added.
    fn separate_and_add_cuts(
        &mut self,
        columns: &[BpColumn],
        column_values: &[f64],
        eps: f64,
    ) -> usize {
        let mut tally: FxHashMap<(usize, usize), f64> = FxHashMap::default();
        for (ci, &val) in column_values.iter().enumerate() {
            if val <= 1.0e-9 {
                continue;
            }
            if ci >= columns.len() {
                continue;
            }
            let col = &columns[ci];
            for (ti, nodes) in col.covered_internal_nodes.iter().enumerate() {
                for &node in nodes {
                    if self.node_row_idx[ti][node].is_none() {
                        *tally.entry((ti, node)).or_insert(0.0) += val;
                    }
                }
            }
        }
        let mut added = 0usize;
        for ((ti, node), sum) in tally {
            if sum > 1.0 + eps {
                self.add_node_row_lazy(ti, node);
                added += 1;
            }
        }
        added
    }

    fn apply_node_bounds(
        &mut self,
        columns: &[BpColumn],
        fixed_to_one: &[usize],
        fixed_to_zero: &[usize],
        must_link_pairs: &[LeafPair],
        cannot_link_pairs: &[LeafPair],
        blocked_leaves: &[bool],
    ) {
        if self.mark_epoch == u32::MAX {
            self.fixed_one_mark.fill(0);
            self.fixed_zero_mark.fill(0);
            self.mark_epoch = 1;
        }
        self.mark_epoch += 1;
        let epoch = self.mark_epoch;
        for &ci in fixed_to_one {
            if ci < self.fixed_one_mark.len() {
                self.fixed_one_mark[ci] = epoch;
            }
        }
        for &ci in fixed_to_zero {
            if ci < self.fixed_zero_mark.len() {
                self.fixed_zero_mark[ci] = epoch;
            }
        }
        let ptr = self.model.as_mut().expect("RMP model present").as_mut_ptr();
        for ci in 0..self.col_idx.len() {
            let labels = &columns[ci].labels;
            let (desired_lo, desired_hi) = if self.fixed_one_mark[ci] == epoch {
                (1.0, 1.0)
            } else if self.fixed_zero_mark[ci] == epoch {
                (0.0, 0.0)
            } else if labels.iter().any(|&l| blocked_leaves[l as usize])
                || !labels_satisfy_pair_constraints(labels, must_link_pairs, cannot_link_pairs)
            {
                (0.0, 0.0)
            } else {
                (0.0, f64::INFINITY)
            };
            if (self.current_lower[ci] - desired_lo).abs() > 0.0
                || (self.current_upper[ci].is_finite() != desired_hi.is_finite())
                || ((self.current_upper[ci] - desired_hi).abs() > 0.0
                    && self.current_upper[ci].is_finite()
                    && desired_hi.is_finite())
            {
                unsafe {
                    highs_sys::Highs_changeColBounds(ptr, self.col_idx[ci], desired_lo, desired_hi);
                }
                self.current_lower[ci] = desired_lo;
                self.current_upper[ci] = desired_hi;
            }
        }
    }

    fn solve(&mut self, total_columns: usize) -> Result<RmpLpResult, String> {
        let solved = self.model.take().expect("RMP model present").solve();
        let status = solved.status();
        if status != HighsModelStatus::Optimal {
            self.model = Some(Model::from(solved));
            return Err(format!("LP status: {:?}", status));
        }
        let solution = solved.get_solution();
        let mut column_values = vec![0.0; total_columns];
        let solution_cols = solution.columns();
        for (local_idx, &global_ci) in self.col_idx.iter().enumerate() {
            let ci = global_ci as usize;
            if ci < total_columns {
                column_values[ci] = solution_cols[local_idx];
            }
        }
        let objective = solved.objective_value();
        let dual_rows = solution.dual_rows();
        let leaf_duals = self
            .leaf_row_idx
            .iter()
            .map(|&ri| clean_dual(dual_rows[ri]))
            .collect();
        let node_duals = self
            .node_row_idx
            .iter()
            .map(|tree_idxs| {
                tree_idxs
                    .iter()
                    .map(|opt| opt.map(|ri| clean_dual(-dual_rows[ri])).unwrap_or(0.0))
                    .collect()
            })
            .collect();

        self.model = Some(Model::from(solved));
        Ok(RmpLpResult {
            objective,
            column_values,
            leaf_duals,
            node_duals,
        })
    }
}

fn support_is_integral_partition(columns: &[BpColumn], values: &[f64], num_leaves: usize) -> bool {
    let mut cover_count = vec![0usize; num_leaves + 1];
    for (ci, &value) in values.iter().enumerate() {
        if value <= 1.0e-9 {
            continue;
        }
        for &label in &columns[ci].labels {
            let leaf = label as usize;
            cover_count[leaf] += 1;
            if cover_count[leaf] > 1 {
                return false;
            }
        }
    }
    (1..=num_leaves).all(|leaf| cover_count[leaf] == 1)
}

fn reconstruct_components(columns: &[BpColumn], values: &[f64], instance: &Instance) -> Vec<Tree> {
    let n = instance.num_leaves;
    let mut covered = FixedBitSet::with_capacity(n as usize + 1);
    let mut components = Vec::new();
    for (ci, col) in columns.iter().enumerate() {
        if values.get(ci).copied().unwrap_or(0.0) < 0.5 {
            continue;
        }
        if col.labels.len() == 1 {
            covered.insert(col.labels[0] as usize);
            components.push(Tree::singleton(col.labels[0], n));
        } else {
            let leafset = make_leafset(&col.labels, n);
            covered.union_with(&leafset);
            components.push(Tree::component_from_leafset(
                &leafset,
                instance.reference_tree(),
                n,
            ));
        }
    }
    for label in 1..=n {
        if !covered.contains(label as usize) {
            components.push(Tree::singleton(label, n));
        }
    }
    components
}

fn select_branch_pair(columns: &[BpColumn], values: &[f64], num_leaves: usize) -> Option<LeafPair> {
    const ACTIVE_EPS: f64 = 1.0e-9;
    const FRACTIONAL_EPS: f64 = 1.0e-6;

    let stride = num_leaves + 1;
    let mut together = vec![0.0; stride * stride];
    let mut support_count = vec![0usize; stride * stride];

    for (ci, &value) in values.iter().enumerate() {
        if value <= ACTIVE_EPS {
            continue;
        }
        let labels = &columns[ci].labels;
        if labels.len() < 2 {
            continue;
        }
        for i in 0..labels.len() {
            let a = labels[i] as usize;
            let row = a * stride;
            for &label_b in &labels[(i + 1)..] {
                let b = label_b as usize;
                let idx = row + b;
                together[idx] += value;
                support_count[idx] += 1;
            }
        }
    }

    let mut best_pair = None;
    let mut best_balance = f64::NEG_INFINITY;
    let mut best_support = 0usize;
    for a in 1..=num_leaves {
        let row = a * stride;
        for b in (a + 1)..=num_leaves {
            let idx = row + b;
            let together_mass = together[idx];
            if together_mass <= FRACTIONAL_EPS || together_mass >= 1.0 - FRACTIONAL_EPS {
                continue;
            }
            let balance = 0.5 - (together_mass - 0.5).abs();
            let support = support_count[idx];
            if balance > best_balance + 1.0e-12
                || ((balance - best_balance).abs() <= 1.0e-12 && support > best_support)
            {
                best_pair = Some(LeafPair::new(a as u32, b as u32));
                best_balance = balance;
                best_support = support;
            }
        }
    }

    best_pair
}

fn select_branch_column(columns: &[BpColumn], values: &[f64], num_leaves: usize) -> Option<usize> {
    let mut seen = vec![false; num_leaves + 1];
    let mut duplicated = vec![false; num_leaves + 1];
    for (ci, &value) in values.iter().enumerate() {
        if value <= 1.0e-9 || value >= 1.0 - 1.0e-9 {
            continue;
        }
        for &label in &columns[ci].labels {
            let leaf = label as usize;
            if seen[leaf] {
                duplicated[leaf] = true;
            } else {
                seen[leaf] = true;
            }
        }
    }

    let mut best_idx = None;
    let mut best_score = f64::NEG_INFINITY;
    for (ci, &value) in values.iter().enumerate() {
        if value <= 1.0e-9 || value >= 1.0 - 1.0e-9 {
            continue;
        }
        if !columns[ci]
            .labels
            .iter()
            .any(|&label| duplicated[label as usize])
        {
            continue;
        }
        if columns[ci].labels.len() <= 1 || columns[ci].total_internal_count == 0 {
            continue;
        }
        let score = columns[ci].labels.len() as f64 / columns[ci].total_internal_count as f64;
        if score > best_score {
            best_score = score;
            best_idx = Some(ci);
        }
    }
    best_idx
}

fn labels_disjoint(a: &[u32], b: &[u32]) -> bool {
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => return false,
        }
    }
    true
}

fn labels_contain(labels: &[u32], target: u32) -> bool {
    labels.binary_search(&target).is_ok()
}

fn labels_satisfy_pair_constraints(
    labels: &[u32],
    must_link_pairs: &[LeafPair],
    cannot_link_pairs: &[LeafPair],
) -> bool {
    for &pair in must_link_pairs {
        if labels_contain(labels, pair.a) != labels_contain(labels, pair.b) {
            return false;
        }
    }
    for &pair in cannot_link_pairs {
        if labels_contain(labels, pair.a) && labels_contain(labels, pair.b) {
            return false;
        }
    }
    true
}

fn pricing_candidate_allowed(
    labels: &[u32],
    seen: &FxHashSet<Vec<u32>>,
    forbidden: &ForbiddenColumns,
    must_link_pairs: &[LeafPair],
    cannot_link_pairs: &[LeafPair],
) -> bool {
    !forbidden.contains(seen, labels)
        && labels_satisfy_pair_constraints(labels, must_link_pairs, cannot_link_pairs)
}

fn node_branchings_self_consistent(columns: &[BpColumn], node: &BpNode) -> bool {
    if node
        .must_link_pairs
        .iter()
        .any(|pair| node.cannot_link_pairs.contains(pair))
    {
        return false;
    }
    if node
        .fixed_to_one
        .iter()
        .any(|ci| node.fixed_to_zero.contains(ci))
    {
        return false;
    }
    for (idx, &forced_ci) in node.fixed_to_one.iter().enumerate() {
        if !labels_satisfy_pair_constraints(
            &columns[forced_ci].labels,
            &node.must_link_pairs,
            &node.cannot_link_pairs,
        ) {
            return false;
        }
        for &other_ci in &node.fixed_to_one[(idx + 1)..] {
            if !labels_disjoint(&columns[forced_ci].labels, &columns[other_ci].labels) {
                return false;
            }
        }
    }
    true
}

fn make_leafset(labels: &[u32], num_leaves: u32) -> FixedBitSet {
    let mut bits = FixedBitSet::with_capacity(num_leaves as usize + 1);
    for &label in labels {
        bits.insert(label as usize);
    }
    bits
}

fn clean_dual(value: f64) -> f64 {
    if value.abs() <= 1.0e-9 { 0.0 } else { value }
}


// ── Unified Solver impl + entry point ───────────────────────────────────────
use crate::{RunConfig, Solver, Track};

pub fn main() {
    crate::run(MafBranchPriceMultiSolver::new(), RunConfig { track: Track::Exact, ..Default::default() });
}
