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
use std::time::Instant;

use fixedbitset::FixedBitSet;
use fxhash::FxHashMap;
use klados_core::solve_pipeline::{ClusterAlgo, SolveConfig, solve_with_pipeline};
use klados_core::{Instance, SolverStats, Tree};
use log::{info, trace};

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
const CANON_WL_MAX_ROUNDS: usize = 6;

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
        }
    }

    pub fn with_config(config: BpConfig) -> Self {
        Self {
            stats: SolverStats::default(),
            config,
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
        let components = solve_recursive_memo(instance, &cfg, &memo)?;
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
                "bp memo: hits={} stores={} entries={} skipped_ambiguous={}",
                memo_stats.hits,
                memo_stats.stores,
                memo_stats.solutions.len(),
                memo_stats.skipped_ambiguous,
            );
        }
        Some(components)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

/// Re-entry point for primal heuristics that need to recursively solve
/// sub-instances (e.g. Whidden relaxed decomposition).  Exposed `pub(crate)`
/// so [`solver::solve_inner`] can call it.
pub(crate) fn solve_subinstance(instance: &Instance, cfg: &BpConfig) -> Option<Vec<Tree>> {
    solve_recursive(instance, cfg)
}

fn solve_recursive(instance: &Instance, cfg: &BpConfig) -> Option<Vec<Tree>> {
    let memo = Rc::new(RefCell::new(SubinstanceMemo::default()));
    solve_recursive_memo(instance, cfg, &memo)
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

    // Kernelize first so Whidden runs on a reduced instance — matching
    // bp-multi's solve_branch_price_multi_cached which kernelizes before
    // trying any decomposition.
    let kern = if cfg.kernelize {
        let mut kernel_cfg = klados_core::kernelize::KernelizeConfig::default();
        if !instance.protected_labels.is_empty() {
            kernel_cfg.protected_labels = instance.protected_labels.clone();
        }
        klados_core::kernelize::kernelize_best(instance, &kernel_cfg)
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
        match canonicalize_two_tree_instance(reduced) {
            Some(view) => {
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
            solver::solve_inner_with_subsolver(reduced, &mut |sub| {
                solve_recursive_memo(sub, &cfg_inner, &memo_inner)
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
            try_whidden_decomp_2tree(reduced, &mut |sub| solve_recursive_memo(sub, &cfg_inner, &memo_inner))
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
                    try_whidden_decomp_2tree(sub, &mut |s| solve_recursive_memo(s, &cfg2, &memo2))
                {
                    return Some(comps);
                }
            }
            if sub.num_leaves < reduced_num_leaves {
                solve_recursive_memo(sub, &inner_cfg, &memo_pipeline)
            } else {
                let cfg3 = inner_cfg.clone();
                let memo3 = Rc::clone(&memo_pipeline);
                solver::solve_inner_with_subsolver(sub, &mut |s| {
                    solve_recursive_memo(s, &cfg3, &memo3)
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

fn canonicalize_two_tree_instance(instance: &Instance) -> Option<CanonicalMemoView> {
    debug_assert_eq!(instance.num_trees(), 2);
    let t0 = &instance.trees[0];
    let t1 = &instance.trees[1];
    let n = instance.num_leaves as usize;

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
