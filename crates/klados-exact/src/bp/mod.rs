//! Branch-and-Price solver for multi-tree MAF.
//!
//! ## Architecture
//!
//! The solver wraps the core B&P in a recursive decomposition pipeline:
//! kernelize → Whidden strict cluster decomp → pipeline (cluster reduction
//! → cluster decomposition) → inner B&P.  Each sub-instance that can't be
//! decomposed further is handed to [`solver::solve_inner`].
//!
//! The inner B&P uses a tiered pricer ([`pricer`]), a HiGHS-backed RMP with
//! lazy node-row separation ([`rmp`]), leaf-pair-only B&B ([`search`]), and
//! validity-by-construction column types ([`column`]).
//!
//! ## Module map
//! - [`column`] — `AfColumn` (validity-by-construction), `ColumnBuilder`, `ColumnSet`.
//! - [`search`] — `Branchings` (pair-only), `SearchState`, `Incumbent`, selection, telemetry.
//! - [`rmp`]    — HiGHS-backed restricted master, lazy node rows, branchings-derived bounds.
//! - [`pricer`] — `Pricer` trait, tiered `CompositePricer`, per-tier implementations.
//! - [`solver`] — search loop, node solver, primal heuristics, incumbent construction.
//!
//! ## Solvers
//!
//! [`BpSolver`] implements [`crate::ExactSolver`].  By default it enables
//! kernelization and cluster reduction.  Set `KLADOS_BP_NO_DECOMP=1` to
//! disable all decomposition (useful for debugging the core algorithm).

pub mod column;
pub mod pricer;
pub mod rmp;
pub mod search;
pub mod solver;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use fixedbitset::FixedBitSet;
use fxhash::FxHashMap;
use klados_core::af_validator::validate_agreement_forest;
use klados_core::solve_pipeline::{ClusterAlgo, SolveConfig, solve_with_pipeline};
use klados_core::{Instance, SolverStats, Tree};
use log::{info, trace};

use crate::chen_rspr::chen_pair_agreement;
use crate::ExactSolver;
use crate::whidden_cluster::try_whidden_decomp_2tree;

const LOG_TARGET: &str = "klados::bp";

/// Minimum leaves for which Whidden strict cluster decomp is worth trying.
/// Below this, the pipeline's generic cluster_reduction handles things fine
/// and Whidden's overhead isn't justified.
const WHIDDEN_MIN_LEAVES: u32 = 20;
const DIRECT_M2_SMALL_CORE_MAX_LEAVES: u32 = 64;
const MEMO_MIN_LEAVES: u32 = 4;
const MEMO_MAX_LEAVES: u32 = 512;
/// Cap on individualization-refinement search nodes. If a subinstance is so
/// symmetric that canonicalization would exceed this, we abort and skip the
/// memo for it (correctness preserved, just no caching).
const CANON_IR_BUDGET: usize = 2000;

thread_local! {
    static KERN_NANOS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static CANON_NANOS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[derive(Default)]
struct SubinstanceMemo {
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

/// Stage-2 configuration. The defaults match what the current pricer can
/// soundly support: cluster algorithms stay disabled until a sound pricer
/// (m=2 pair-DP / small-m m-DP) lands, since cluster reduction's stitching
/// requires optimal sub-solves.
#[derive(Clone, Debug)]
pub struct BpConfig {
    pub kernelize: bool,
    pub cluster_algo: ClusterAlgo,
}

impl Default for BpConfig {
    fn default() -> Self {
        Self {
            kernelize: true,
            cluster_algo: ClusterAlgo::ClusterReduction,
        }
    }
}

impl BpConfig {
    /// Configuration with all decomposition disabled — only kernelization
    /// and direct B&P. Used to expose algorithmic issues that would
    /// otherwise be hidden by decomposition.
    pub fn no_decomp() -> Self {
        Self {
            kernelize: true,
            cluster_algo: ClusterAlgo::None,
        }
    }
}

pub struct BpSolver {
    stats: SolverStats,
    config: BpConfig,
    terminated: Arc<AtomicBool>,
}

impl Default for BpSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl BpSolver {
    pub fn new() -> Self {
        // KLADOS_BP_NO_DECOMP=1 disables all decomposition (Whidden, cluster
        // reduction, cluster decomposition). Used to expose algorithmic
        // weaknesses in the core B&P that would otherwise be masked.
        let config = if std::env::var("KLADOS_BP_NO_DECOMP").is_ok() {
            BpConfig::no_decomp()
        } else {
            BpConfig::default()
        };
        Self {
            stats: SolverStats::default(),
            config,
            terminated: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn with_config(config: BpConfig) -> Self {
        Self {
            stats: SolverStats::default(),
            config,
            terminated: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl ExactSolver for BpSolver {
    fn name(&self) -> &'static str {
        "bp"
    }

    fn description(&self) -> &'static str {
        "Branch & Price for multi-tree MAF (rewrite, in progress)"
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        let t_total = Instant::now();
        let cfg = self.config.clone();
        let memo = Rc::new(RefCell::new(SubinstanceMemo::default()));
        let mut components = solve_recursive_memo(instance, &cfg, &memo, &self.terminated)?;

        // Post-validate: if Whidden decomp assembled invalid results
        // (subproblems aborted), fall back to Chen 2-approximation.
        if instance.num_trees() == 2
            && !validate_agreement_forest(instance, &components).is_ok()
        {
            let (_, _, leafsets) = chen_pair_agreement(&instance.trees[0], &instance.trees[1]);
            components = leafsets_to_trees(&leafsets, instance);
        }
        self.stats.upper_bound = Some(components.len());
        self.stats.lower_bound = components.len();
        info!(
            target: LOG_TARGET,
            "solved n={} m={} k={} in {:.1}ms",
            instance.num_leaves,
            instance.num_trees(),
            components.len(),
            t_total.elapsed().as_secs_f64() * 1000.0,
        );
        let memo_stats = memo.borrow();
        if memo_stats.hits > 0 || memo_stats.stores > 0 || memo_stats.skipped_ambiguous > 0 {
            info!(
                target: LOG_TARGET,
                "bp memo: hits={} stores={} entries={} skipped_ambiguous={} kern={:.1}ms canon={:.1}ms",
                memo_stats.hits,
                memo_stats.stores,
                memo_stats.solutions.len(),
                memo_stats.skipped_ambiguous,
                KERN_NANOS.with(|c| c.get()) as f64 / 1e6,
                CANON_NANOS.with(|c| c.get()) as f64 / 1e6,
            );
        }
        Some(components)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }

    #[cfg(feature = "early-termination")]
    fn sigterm_handler(&self) {
        self.terminated.store(true, Ordering::SeqCst);
    }
}

/// Re-entry point for primal heuristics that need to recursively solve
/// sub-instances (e.g. Whidden relaxed decomposition).  Exposed `pub(crate)`
/// so [`solver::solve_inner`] can call it.
pub(crate) fn solve_subinstance(instance: &Instance, cfg: &BpConfig, terminate: &Arc<AtomicBool>) -> Option<Vec<Tree>> {
    let memo = Rc::new(RefCell::new(SubinstanceMemo::default()));
    solve_recursive_memo(instance, cfg, &memo, terminate)
}

fn solve_recursive(instance: &Instance, cfg: &BpConfig, terminate: &Arc<AtomicBool>) -> Option<Vec<Tree>> {
    let memo = Rc::new(RefCell::new(SubinstanceMemo::default()));
    solve_recursive_memo(instance, cfg, &memo, terminate)
}

/// Recursive solve: tries decomposition strategies in effectiveness order,
/// falling through to the inner B&P when no decomposition applies.
///
/// Order:
/// 1. **Trivial** (m≤1, n≤1) — short-circuit.
/// 2. **Kernelize** — reduce the instance before decomposition.
/// 3. **Whidden strict cluster decomp** (m=2, n≥WHIDDEN_MIN_LEAVES) —
///    applied on the kernelized instance so cluster points aren't obscured
///    by reducible leaves. Matches bp-multi's kernelize-before-Whidden flow.
/// 4. **Pipeline** (cluster_reduction → cluster_decomposition → inner solve)
///    with kernelization disabled (already done). The inner solver retries
///    Whidden on every sub-instance so the path is available at every
///    recursion level.
fn solve_recursive_memo(
    instance: &Instance,
    cfg: &BpConfig,
    memo: &Rc<RefCell<SubinstanceMemo>>,
    terminate: &Arc<AtomicBool>,
) -> Option<Vec<Tree>> {
    if instance.trees.is_empty() {
        return None;
    }
    if instance.num_trees() == 1 {
        return Some(instance.trees.clone());
    }
    if instance.num_leaves <= 1 {
        return Some(instance.trees[0..1].to_vec());
    }

    if terminate.load(Ordering::Acquire) {
        let forest: Vec<Tree> = (1..=instance.num_leaves)
            .map(|l| klados_core::Tree::singleton(l, instance.num_leaves))
            .collect();
        return Some(forest);
    }

    // Kernelize first so Whidden runs on a reduced instance — matching
    // bp-multi's solve_branch_price_multi_cached which kernelizes before
    // trying any decomposition.
    let kern = if cfg.kernelize {
        let mut kernel_cfg = klados_core::kernelize::KernelizeConfig::default();
        if !instance.protected_labels.is_empty() {
            kernel_cfg.protected_labels = instance.protected_labels.clone();
        }
        let t = Instant::now();
        let r = klados_core::kernelize::kernelize_best(instance, &kernel_cfg);
        KERN_NANOS.with(|c| c.set(c.get() + t.elapsed().as_nanos() as u64));
        r
    } else {
        klados_core::kernelize::KernelizeResult {
            instance: instance.clone(),
            stats: Default::default(),
            reverse_map: (0..=instance.num_leaves).map(|i| i as u32).collect(),
            collapses_original: vec![],
            param_reduction: 0,
            trace: vec![],
        }
    };
    let reduced = &kern.instance;

    if reduced.num_leaves <= 1 {
        let trivial = if reduced.num_leaves == 0 {
            vec![]
        } else {
            vec![reduced.trees[0].clone()]
        };
        return Some(klados_core::kernelize::expand_solution(
            trivial,
            &kern,
            &instance.trees[0],
            instance.num_leaves,
        ));
    }

    let memo_view = if reduced.num_trees() == 2
        && (MEMO_MIN_LEAVES..=MEMO_MAX_LEAVES).contains(&reduced.num_leaves)
    {
        let t_canon = Instant::now();
        let canon = canonicalize_two_tree_instance(reduced);
        CANON_NANOS.with(|c| c.set(c.get() + t_canon.elapsed().as_nanos() as u64));
        match canon {
            Some(view) => {
                if std::env::var("KLADOS_BP_DUMP_MEMO_KEYS").is_ok() {
                    eprintln!(
                        "MEMOKEY\tn={}\tkey={}",
                        reduced.num_leaves, view.key
                    );
                }
                let cached_partition = {
                    let mut memo_ref = memo.borrow_mut();
                    if let Some(cached) = memo_ref.solutions.get(&view.key).cloned() {
                        memo_ref.hits += 1;
                        Some(cached)
                    } else {
                        None
                    }
                };
                if let Some(cached_partition) = cached_partition {
                    let reduced_components =
                        reconstruct_cached_components(&cached_partition, &view, reduced);
                    return Some(klados_core::kernelize::expand_solution(
                        reduced_components,
                        &kern,
                        &instance.trees[0],
                        instance.num_leaves,
                    ));
                }
                Some(view)
            }
            None => {
                memo.borrow_mut().skipped_ambiguous += 1;
                None
            }
        }
    } else {
        None
    };

    // Small two-tree cores are cheap enough for the new B&P itself.  Look them
    // up in the canonical subinstance memo first (matching bp-multi's useful
    // reuse on repeated anchor cores), then solve directly on cache misses.
    if reduced.num_trees() == 2 && reduced.num_leaves <= DIRECT_M2_SMALL_CORE_MAX_LEAVES {
        let cfg_inner = cfg.clone();
        let memo_inner = Rc::clone(memo);
        let reduced_components =
            solver::solve_inner_with_subsolver(reduced, terminate, &mut |sub| {
                solve_recursive_memo(sub, &cfg_inner, &memo_inner, terminate)
            })?;
        if let Some(view) = memo_view.as_ref() {
            store_cached_solution(&mut memo.borrow_mut(), view, &reduced_components);
        }
        return Some(klados_core::kernelize::expand_solution(
            reduced_components,
            &kern,
            &instance.trees[0],
            instance.num_leaves,
        ));
    }

    let allow_whidden = !matches!(cfg.cluster_algo, ClusterAlgo::None);
    if allow_whidden && reduced.num_trees() == 2 && reduced.num_leaves >= WHIDDEN_MIN_LEAVES {
        let cfg_inner = cfg.clone();
        let memo_inner = Rc::clone(memo);
        if let Some(comps) =
            try_whidden_decomp_2tree(reduced, &mut |sub| solve_recursive_memo(sub, &cfg_inner, &memo_inner, terminate))
        {
            trace!(
                target: LOG_TARGET,
                "whidden strict decomp solved: n={} k={}",
                instance.num_leaves, comps.len(),
            );
            if let Some(view) = memo_view.as_ref() {
                store_cached_solution(&mut memo.borrow_mut(), view, &comps);
            }
            let expanded = klados_core::kernelize::expand_solution(
                comps,
                &kern,
                &instance.trees[0],
                instance.num_leaves,
            );
            return Some(expanded);
        }
    }

    let pipeline_cfg = SolveConfig {
        kernelize: false, // already kernelized above
        kernelize_config: Default::default(),
        cluster_algo: cfg.cluster_algo.clone(),
    };
    let inner_cfg = cfg.clone();
    let reduced_num_leaves = reduced.num_leaves;
    let memo_pipeline = Rc::clone(memo);
    let t = Arc::clone(terminate);
    let reduced_components = solve_with_pipeline(
        reduced,
        &pipeline_cfg,
        &mut move |sub: &Instance| -> Option<Vec<Tree>> {
            if allow_whidden
                && sub.num_trees() == 2
                && sub.num_leaves >= WHIDDEN_MIN_LEAVES
                && sub.num_leaves > DIRECT_M2_SMALL_CORE_MAX_LEAVES
            {
                let cfg2 = inner_cfg.clone();
                let memo2 = Rc::clone(&memo_pipeline);
                if let Some(comps) =
                    try_whidden_decomp_2tree(sub, &mut |s| solve_recursive_memo(s, &cfg2, &memo2, &t))
                {
                    return Some(comps);
                }
            }
            if sub.num_leaves < reduced_num_leaves {
                solve_recursive_memo(sub, &inner_cfg, &memo_pipeline, &t)
            } else {
                let cfg3 = inner_cfg.clone();
                let memo3 = Rc::clone(&memo_pipeline);
                solver::solve_inner_with_subsolver(sub, &t, &mut |s| {
                    solve_recursive_memo(s, &cfg3, &memo3, &t)
                })
            }
        },
    )?;
    if let Some(view) = memo_view.as_ref() {
        store_cached_solution(&mut memo.borrow_mut(), view, &reduced_components);
    }
    Some(klados_core::kernelize::expand_solution(
        reduced_components,
        &kern,
        &instance.trees[0],
        instance.num_leaves,
    ))
}

/// Weisfeiler-Leman colour refinement on the leaf set, shared by both trees.
/// Refines `leaf_color` in place to a stable colouring; returns the class
/// count. The leaf's own current colour is part of its signature, so an
/// externally-imposed split (individualization) is preserved across rounds.
fn wl_refine(t0: &Tree, t1: &Tree, leaf_color: &mut [u32], n: usize) -> usize {
    let mut classes = {
        let mut seen = leaf_color[1..=n].to_vec();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    };
    for _round in 0..=n {
        let (codes0, codes1) = canonical_subtree_codes(t0, t1, leaf_color);

        let mut entries: Vec<(u32, Vec<u32>, Vec<u32>, u32)> = Vec::with_capacity(n);
        for label in 1..=n as u32 {
            let p0 = leaf_path_codes_ids(t0, label, &codes0);
            let p1 = leaf_path_codes_ids(t1, label, &codes1);
            entries.push((leaf_color[label as usize], p0, p1, label));
        }
        entries.sort_unstable();

        let mut new_color = vec![0u32; n + 1];
        let mut cur_id: u32 = 0;
        for i in 0..entries.len() {
            if i > 0
                && (entries[i].0 != entries[i - 1].0
                    || entries[i].1 != entries[i - 1].1
                    || entries[i].2 != entries[i - 1].2)
            {
                cur_id += 1;
            }
            new_color[entries[i].3 as usize] = cur_id;
        }
        let new_classes = cur_id as usize + 1;
        let stable = new_color[1..=n] == leaf_color[1..=n];
        leaf_color[1..=n].copy_from_slice(&new_color[1..=n]);
        classes = new_classes;
        if classes == n || stable {
            break;
        }
    }
    classes
}

struct IrState {
    best_key: Option<String>,
    best_l2c: Vec<u32>,
    best_c2l: Vec<u32>,
    budget: usize,
    aborted: bool,
}

/// Individualization-refinement search. Explores every WL-consistent complete
/// leaf labelling and keeps the one with the lexicographically smallest key.
/// That minimum is a true canonical form: isomorphic instances explore
/// isomorphic search trees and therefore agree on the minimum. If the search
/// exceeds its node budget the canonicalization is aborted (memo skipped).
fn ir_search(t0: &Tree, t1: &Tree, n: usize, color: &[u32], classes: usize, st: &mut IrState) {
    if st.aborted {
        return;
    }
    if st.budget == 0 {
        st.aborted = true;
        return;
    }
    st.budget -= 1;

    if classes == n {
        let mut entries: Vec<(u32, u32)> =
            (1..=n as u32).map(|l| (color[l as usize], l)).collect();
        entries.sort_unstable();
        let mut l2c = vec![0u32; n + 1];
        let mut c2l = vec![0u32; n + 1];
        for (idx, (_, label)) in entries.iter().enumerate() {
            let canon = (idx + 1) as u32;
            l2c[*label as usize] = canon;
            c2l[canon as usize] = *label;
        }
        let r0 = t0.relabel(&l2c, n as u32);
        let r1 = t1.relabel(&l2c, n as u32);
        let key = format!(
            "{}||{}",
            labeled_tree_signature(&r0, r0.root),
            labeled_tree_signature(&r1, r1.root)
        );
        if st.best_key.as_ref().map_or(true, |bk| key < *bk) {
            st.best_key = Some(key);
            st.best_l2c = l2c;
            st.best_c2l = c2l;
        }
        return;
    }

    let mut counts = vec![0u32; classes];
    for l in 1..=n {
        counts[color[l] as usize] += 1;
    }
    let target = (0..classes).find(|&c| counts[c] > 1).unwrap() as u32;
    let members: Vec<u32> = (1..=n as u32)
        .filter(|&l| color[l as usize] == target)
        .collect();

    for &v in &members {
        if st.aborted {
            return;
        }
        // Split `target`: every leaf's colour is doubled, then the non-`v`
        // members of the class are bumped — individualizing `v`. wl_refine
        // renormalizes colours afterwards.
        let mut nc = color.to_vec();
        for l in 1..=n {
            nc[l] *= 2;
        }
        for &w in &members {
            if w != v {
                nc[w as usize] += 1;
            }
        }
        let classes2 = wl_refine(t0, t1, &mut nc, n);
        ir_search(t0, t1, n, &nc, classes2, st);
    }
}

fn canonicalize_two_tree_instance(instance: &Instance) -> Option<CanonicalMemoView> {
    debug_assert_eq!(instance.num_trees(), 2);
    let t0 = &instance.trees[0];
    let t1 = &instance.trees[1];
    let n = instance.num_leaves as usize;

    let mut leaf_color: Vec<u32> = vec![0; n + 1];
    let classes = wl_refine(t0, t1, &mut leaf_color, n);

    let mut st = IrState {
        best_key: None,
        best_l2c: Vec::new(),
        best_c2l: Vec::new(),
        budget: CANON_IR_BUDGET,
        aborted: false,
    };
    ir_search(t0, t1, n, &leaf_color, classes, &mut st);

    if st.aborted {
        return None;
    }
    let key = st.best_key?;
    Some(CanonicalMemoView {
        key,
        label_to_canonical: st.best_l2c,
        canonical_to_label: st.best_c2l,
    })
}

fn node_heights(tree: &Tree) -> Vec<u32> {
    let mut h = vec![0u32; tree.num_nodes()];
    for node in tree.post_order() {
        if !tree.is_leaf(node) {
            let (l, r) = tree.children_pair(node);
            h[node as usize] = 1 + h[l as usize].max(h[r as usize]);
        }
    }
    h
}

/// Assign each node of both trees an integer code that depends only on the
/// colored subtree shape (leaf colours + topology), not on traversal order.
/// Codes are allocated in sorted order, level by level, so the code ordering
/// is itself isomorphism-canonical — essential for canonical cell ordering.
fn canonical_subtree_codes(t0: &Tree, t1: &Tree, leaf_color: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut codes0 = vec![0u32; t0.num_nodes()];
    let mut codes1 = vec![0u32; t1.num_nodes()];
    let h0 = node_heights(t0);
    let h1 = node_heights(t1);
    let max_h = h0
        .iter()
        .chain(h1.iter())
        .copied()
        .max()
        .unwrap_or(0);

    let mut next_code: u32 = 0;
    for level in 0..=max_h {
        let shape_key = |tree: &Tree, codes: &[u32], node: u32| -> (u32, u32, u32) {
            if tree.is_leaf(node) {
                (0, leaf_color[tree.label[node as usize] as usize], 0)
            } else {
                let (l, r) = tree.children_pair(node);
                let a = codes[l as usize];
                let b = codes[r as usize];
                (1, a.min(b), a.max(b))
            }
        };
        let mut keys: Vec<(u32, u32, u32)> = Vec::new();
        for node in 0..t0.num_nodes() as u32 {
            if h0[node as usize] == level {
                keys.push(shape_key(t0, &codes0, node));
            }
        }
        for node in 0..t1.num_nodes() as u32 {
            if h1[node as usize] == level {
                keys.push(shape_key(t1, &codes1, node));
            }
        }
        keys.sort_unstable();
        keys.dedup();
        let mut map: FxHashMap<(u32, u32, u32), u32> = FxHashMap::default();
        for k in keys {
            map.insert(k, next_code);
            next_code += 1;
        }
        for node in 0..t0.num_nodes() as u32 {
            if h0[node as usize] == level {
                codes0[node as usize] = map[&shape_key(t0, &codes0, node)];
            }
        }
        for node in 0..t1.num_nodes() as u32 {
            if h1[node as usize] == level {
                codes1[node as usize] = map[&shape_key(t1, &codes1, node)];
            }
        }
    }
    (codes0, codes1)
}

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
    memo: &mut SubinstanceMemo,
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

fn make_leafset(labels: &[u32], num_leaves: u32) -> FixedBitSet {
    let mut bits = FixedBitSet::with_capacity(num_leaves as usize + 1);
    for &label in labels {
        bits.insert(label as usize);
    }
    bits
}

#[cfg(test)]
mod canon_tests {
    use super::*;
    use klados_core::tree::{Label, NodeId, NONE};

    fn parse(nw: &str, n: u32) -> Tree {
        let mut t = Tree::with_capacity(n);
        let b = nw.as_bytes();
        let mut pos = 0usize;
        fn rec(b: &[u8], pos: &mut usize, t: &mut Tree) -> NodeId {
            if b[*pos] == b'(' {
                *pos += 1;
                let l = rec(b, pos, t);
                assert_eq!(b[*pos], b',');
                *pos += 1;
                let r = rec(b, pos, t);
                assert_eq!(b[*pos], b')');
                *pos += 1;
                let id = t.parent.len() as NodeId;
                t.parent.push(NONE);
                t.left.push(l);
                t.right.push(r);
                t.label.push(0);
                t.parent[l as usize] = id;
                t.parent[r as usize] = id;
                id
            } else {
                let start = *pos;
                while *pos < b.len() && b[*pos].is_ascii_digit() {
                    *pos += 1;
                }
                let lbl: u32 = std::str::from_utf8(&b[start..*pos]).unwrap().parse().unwrap();
                let id = t.parent.len() as NodeId;
                t.parent.push(NONE);
                t.left.push(NONE);
                t.right.push(NONE);
                t.label.push(lbl as Label);
                t.label_to_node[lbl as usize] = id;
                id
            }
        }
        t.root = rec(b, &mut pos, &mut t);
        t.compute_metadata();
        t
    }

    /// Rebuild `src` with the left/right children of every internal node
    /// swapped (a full mirror) — a rotation that canonicalization must ignore.
    fn mirror(src: &Tree) -> Tree {
        let mut t = Tree::with_capacity(src.num_leaves);
        fn rec(src: &Tree, node: NodeId, t: &mut Tree) -> NodeId {
            if src.is_leaf(node) {
                let lbl = src.label[node as usize];
                let id = t.parent.len() as NodeId;
                t.parent.push(NONE);
                t.left.push(NONE);
                t.right.push(NONE);
                t.label.push(lbl);
                t.label_to_node[lbl as usize] = id;
                id
            } else {
                let (l, r) = src.children_pair(node);
                let rr = rec(src, r, t);
                let ll = rec(src, l, t);
                let id = t.parent.len() as NodeId;
                t.parent.push(NONE);
                t.left.push(rr);
                t.right.push(ll);
                t.label.push(0);
                t.parent[rr as usize] = id;
                t.parent[ll as usize] = id;
                id
            }
        }
        t.root = rec(src, src.root, &mut t);
        t.compute_metadata();
        t
    }

    fn check_invariant(nw0: &str, nw1: &str, n: u32) {
        let t0 = parse(nw0, n);
        let t1 = parse(nw1, n);
        let base = canonicalize_two_tree_instance(&Instance::new(vec![t0.clone(), t1.clone()], n))
            .expect("base canonicalizes")
            .key;
        for shift in 1..n {
            let mut map = vec![0 as Label; n as usize + 1];
            for l in 1..=n {
                map[l as usize] = ((l - 1 + shift) % n) + 1;
            }
            let r0 = t0.relabel(&map, n);
            let r1 = t1.relabel(&map, n);
            // mirror tree 0 only: tests rotation + relabeling together.
            let inst = Instance::new(vec![mirror(&r0), r1], n);
            let k = canonicalize_two_tree_instance(&inst).expect("perm canonicalizes").key;
            assert_eq!(base, k, "key not invariant at shift={shift}");
        }
    }

    #[test]
    fn canon_invariant_generic() {
        check_invariant(
            "(((1,2),(3,4)),((5,6),(7,8)))",
            "((1,(3,(5,7))),((2,4),(6,8)))",
            8,
        );
    }

    #[test]
    fn canon_invariant_symmetric() {
        // Highly symmetric: both trees fully balanced — exercises the
        // individualization-refinement branching.
        check_invariant(
            "(((1,2),(3,4)),((5,6),(7,8)))",
            "(((1,2),(3,4)),((5,6),(7,8)))",
            8,
        );
    }

    #[test]
    fn canon_invariant_caterpillar() {
        check_invariant(
            "(1,(2,(3,(4,(5,6)))))",
            "(((((1,6),2),5),3),4)",
            6,
        );
    }
}

/// Convert Chen leaf-sets to agreement forest trees.
fn leafsets_to_trees(leafsets: &[Vec<u32>], instance: &Instance) -> Vec<Tree> {
    let t1 = &instance.trees[0];
    let n = instance.num_leaves;
    leafsets
        .iter()
        .map(|labels| {
            if labels.len() == 1 {
                Tree::singleton(labels[0], n)
            } else {
                let mut bitset = fixedbitset::FixedBitSet::with_capacity(n as usize + 1);
                for &l in labels {
                    bitset.insert(l as usize);
                }
                Tree::component_from_leafset(&bitset, t1, n)
            }
        })
        .collect()
}
