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
//! [`BpSolver`] implements [`crate::Solver`].  By default it enables
//! kernelization and cluster reduction.  Set `BpConfig.cluster_algo = ClusterAlgo::None`
//! to disable all decomposition (useful for debugging the core algorithm).

pub mod column;
pub mod pricer;
pub mod rmp;
pub mod search;
pub mod solver;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use fixedbitset::FixedBitSet;
use fxhash::FxHashMap;
use klados_core::af_validator::validate_agreement_forest;
use klados_core::solve_pipeline::{ClusterAlgo, SolveConfig, solve_with_pipeline};
use klados_core::{Instance, SolverStats, Tree};
use log::{debug, info};

use crate::decomp::whidden_cluster::try_whidden_decomp_2tree;
use crate::solvers::chen_rspr::chen_pair_agreement;

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
    pub mip_time_limit: f64,
    pub m2_batch: usize,
    pub m2_exact_dp_cells: usize,
    pub m2_exact_reserve_cap: usize,
    pub use_anchor_cache: bool,
    pub no_chen_lb: bool,
    pub core_decomp_analyze: bool,
    pub core_decomp_min_leaves: usize,
    pub disable_bound_prune: bool,
    pub no_rcvf: bool,
    pub tiny_rcvf: bool,
    pub relaxed_incumbent: bool,
    pub obstruction_probe: bool,
    pub bridge_probe: bool,
    pub root_support_incumbent: bool,
    pub corridor_enrich: bool,
    pub corridor_max_k: usize,
    pub mip_heuristic: bool,
    pub root_support_mip: bool,
    pub obstruction_local_lb: bool,
    pub obstruction_solve_core: bool,
    pub region_support_mip: bool,
    pub all_region_support_mip: bool,
    pub all_region_exact_rank: bool,
    pub all_region_exact_max_leaves: usize,
    /// Diagnostic: at the root, dump the LP fractional merges (leaf pairs with
    /// fractional together-mass) and, for the top few, simulate committing them
    /// (must/cannot-link) to measure ΔLP and how much the fractional support
    /// collapses — i.e. whether the LP usefully ORDERS which merge to commit.
    pub merge_order_probe: bool,
    /// Diagnostic: at the root, for each fractional tree-node activity
    /// `y_{t,v} = Σ_{c covers (t,v)} x_c`, simulate the `y=0` branch (forbid all
    /// columns covering that node) and measure ΔLP — the embedding-branching
    /// analog of cannot-link, to test whether it moves the LP or redistributes.
    pub node_branch_probe: bool,
    /// Prototype: at a stuck core root, solve the candidate-merge MWIS directly
    /// (combinatorial), reporting the exact core opt vs the LP bound and time —
    /// the "core-finisher" that replaces the LP cannot-link tree.
    pub mwis_finish: bool,
    /// Gate probe: at root convergence, install Σ_{touch region} x ≥ exact_rank
    /// rank rows for every support region and re-solve the LP over the existing
    /// pool, logging the lift. An UPPER bound on the achievable rank-cut closure
    /// (no re-pricing) — if even this can't reach the integer optimum, the
    /// support-region rank cut cannot close the last unit. Diagnostic only.
    pub support_rank_resolve: bool,
    pub tree_side_exact_rank: bool,
    pub tree_laminar_exact_rank: bool,
    pub tree_side_exact_max_leaves: usize,
    pub tree_side_exact_limit: usize,
    pub residual_completion_probe: bool,
    pub residual_completion_max_cols: usize,
    pub residual_completion_max_residual_leaves: usize,
    pub rank_cut_probe: bool,
    pub rank_cut_probe_max_cols: usize,
    pub ncpack_incumbent: bool,
    pub ncpack_min_trees: usize,
    pub ncpack_max_leaves: usize,
    pub ncpack_node_budget: u64,
    /// GATED ncpack mu* lower-bound floor (single-block certificate; validated
    /// but unproven). `KLADOS_BP_NCPACK_LB=1`.
    pub ncpack_lb: bool,
    pub ncpack_lb_kmax: usize,
    pub ncpack_lb_budget_secs: u64,
    /// Clean-cut rank-row lower bound.  Kept in config (rather than read
    /// directly in the solver) so recursive side proofs use exactly the same
    /// pricing/certification path as the top-level solve.  Enabled by default;
    /// set `KLADOS_BP_CLEAN_LB=0` for an A/B baseline.
    pub clean_lb: bool,
}

impl Default for BpConfig {
    fn default() -> Self {
        Self {
            kernelize: true,
            cluster_algo: ClusterAlgo::ClusterReduction,
            mip_time_limit: 0.1,
            m2_batch: 0,
            m2_exact_dp_cells: 64_000_000,
            m2_exact_reserve_cap: 0,
            use_anchor_cache: false,
            no_chen_lb: false,
            core_decomp_analyze: false,
            core_decomp_min_leaves: 150,
            disable_bound_prune: false,
            no_rcvf: false,
            tiny_rcvf: false,
            relaxed_incumbent: true,
            obstruction_probe: false,
            bridge_probe: false,
            root_support_incumbent: false,
            corridor_enrich: false,
            corridor_max_k: 0,
            mip_heuristic: false,
            root_support_mip: false,
            obstruction_local_lb: false,
            obstruction_solve_core: false,
            region_support_mip: false,
            all_region_support_mip: false,
            all_region_exact_rank: false,
            all_region_exact_max_leaves: 48,
            support_rank_resolve: false,
            merge_order_probe: false,
            node_branch_probe: false,
            // Default ON: the kernelized complete-component MWIS finisher is the
            // proven engine for the high-m wall (cracks pub101/107/122/134 — see
            // memory). It is sound (validated 0 mismatches) and self-gating: it
            // only fires on m≥3 cores under the leaf/component caps whose root LP
            // is fractional, and falls back to B&P on any cap/budget miss.
            // Disable with `KLADOS_BP_MWIS_FINISH=0`.
            mwis_finish: true,
            tree_side_exact_rank: false,
            tree_laminar_exact_rank: false,
            tree_side_exact_max_leaves: 48,
            tree_side_exact_limit: 64,
            residual_completion_probe: false,
            residual_completion_max_cols: 24,
            residual_completion_max_residual_leaves: 48,
            rank_cut_probe: false,
            rank_cut_probe_max_cols: 48,
            ncpack_incumbent: true,
            ncpack_min_trees: 8,
            ncpack_max_leaves: 250,
            ncpack_node_budget: 50_000_000,
            ncpack_lb: false,
            ncpack_lb_kmax: 6,
            ncpack_lb_budget_secs: 60,
            clean_lb: true,
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
            ..Default::default()
        }
    }

    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if std::env::var("KLADOS_BP_NO_DECOMP").as_deref() == Ok("1") {
            cfg.cluster_algo = ClusterAlgo::None;
        }
        if std::env::var("KLADOS_BP_NO_KERNEL").as_deref() == Ok("1") {
            cfg.kernelize = false;
        }
        if std::env::var("KLADOS_BP_OBSTRUCTION_PROBE").as_deref() == Ok("1") {
            cfg.obstruction_probe = true;
        }
        if std::env::var("KLADOS_BP_BRIDGE_PROBE").as_deref() == Ok("1") {
            cfg.bridge_probe = true;
        }
        if std::env::var("KLADOS_BP_ROOT_SUPPORT_INCUMBENT").as_deref() == Ok("1") {
            cfg.root_support_incumbent = true;
        }
        if std::env::var("KLADOS_BP_ROOT_SUPPORT_MIP").as_deref() == Ok("1") {
            cfg.root_support_mip = true;
        }
        if std::env::var("KLADOS_BP_OBSTRUCTION_LOCAL_LB").as_deref() == Ok("1") {
            cfg.obstruction_local_lb = true;
        }
        if std::env::var("KLADOS_BP_OBSTRUCTION_SOLVE_CORE").as_deref() == Ok("1") {
            cfg.obstruction_solve_core = true;
        }
        if std::env::var("KLADOS_BP_REGION_SUPPORT_MIP").as_deref() == Ok("1") {
            cfg.region_support_mip = true;
        }
        if std::env::var("KLADOS_BP_ALL_REGION_SUPPORT_MIP").as_deref() == Ok("1") {
            cfg.all_region_support_mip = true;
        }
        if std::env::var("KLADOS_BP_ALL_REGION_EXACT_RANK").as_deref() == Ok("1") {
            cfg.all_region_exact_rank = true;
        }
        if std::env::var("KLADOS_BP_SUPPORT_RANK_RESOLVE").as_deref() == Ok("1") {
            cfg.support_rank_resolve = true;
        }
        if std::env::var("KLADOS_BP_MERGE_ORDER_PROBE").as_deref() == Ok("1") {
            cfg.merge_order_probe = true;
        }
        if std::env::var("KLADOS_BP_NODE_BRANCH_PROBE").as_deref() == Ok("1") {
            cfg.node_branch_probe = true;
        }
        // Default ON (set in `default()`); allow explicit opt-out for A/B.
        match std::env::var("KLADOS_BP_MWIS_FINISH").as_deref() {
            Ok("1") => cfg.mwis_finish = true,
            Ok("0") => cfg.mwis_finish = false,
            _ => {}
        }
        if let Ok(raw) = std::env::var("KLADOS_BP_ALL_REGION_EXACT_MAX_LEAVES")
            && let Ok(value) = raw.parse::<usize>()
        {
            cfg.all_region_exact_max_leaves = value.clamp(1, 256);
        }
        if std::env::var("KLADOS_BP_TREE_SIDE_EXACT_RANK").as_deref() == Ok("1") {
            cfg.tree_side_exact_rank = true;
        }
        if std::env::var("KLADOS_BP_TREE_LAMINAR_EXACT_RANK").as_deref() == Ok("1") {
            cfg.tree_laminar_exact_rank = true;
        }
        if let Ok(raw) = std::env::var("KLADOS_BP_TREE_SIDE_EXACT_MAX_LEAVES")
            && let Ok(value) = raw.parse::<usize>()
        {
            cfg.tree_side_exact_max_leaves = value.clamp(1, 256);
        }
        if let Ok(raw) = std::env::var("KLADOS_BP_TREE_SIDE_EXACT_LIMIT")
            && let Ok(value) = raw.parse::<usize>()
        {
            cfg.tree_side_exact_limit = value.clamp(1, 4096);
        }
        if std::env::var("KLADOS_BP_RESIDUAL_COMPLETION_PROBE").as_deref() == Ok("1") {
            cfg.residual_completion_probe = true;
        }
        if let Ok(raw) = std::env::var("KLADOS_BP_RESIDUAL_COMPLETION_MAX_COLS")
            && let Ok(value) = raw.parse::<usize>()
        {
            cfg.residual_completion_max_cols = value.clamp(1, 256);
        }
        if let Ok(raw) = std::env::var("KLADOS_BP_RESIDUAL_COMPLETION_MAX_RESIDUAL_LEAVES")
            && let Ok(value) = raw.parse::<usize>()
        {
            cfg.residual_completion_max_residual_leaves = value.clamp(1, 256);
        }
        if std::env::var("KLADOS_BP_NCPACK_INCUMBENT").as_deref() == Ok("0") {
            cfg.ncpack_incumbent = false;
        }
        if std::env::var("KLADOS_BP_NCPACK_LB").as_deref() == Ok("1") {
            cfg.ncpack_lb = true;
        }
        if std::env::var("KLADOS_BP_CLEAN_LB").as_deref() == Ok("0") {
            cfg.clean_lb = false;
        } else if std::env::var("KLADOS_BP_CLEAN_LB").as_deref() == Ok("1") {
            cfg.clean_lb = true;
        }
        if let Ok(raw) = std::env::var("KLADOS_BP_NCPACK_LB_BUDGET_SECS")
            && let Ok(value) = raw.parse::<u64>()
        {
            cfg.ncpack_lb_budget_secs = value.clamp(1, 600);
        }
        if std::env::var("KLADOS_BP_RANK_CUT_PROBE").as_deref() == Ok("1") {
            cfg.rank_cut_probe = true;
        }
        if let Ok(raw) = std::env::var("KLADOS_BP_RANK_CUT_MAX_COLS")
            && let Ok(value) = raw.parse::<usize>()
        {
            cfg.rank_cut_probe_max_cols = value.clamp(8, 96);
        }
        cfg
    }
}

pub struct BpSolver {
    stats: SolverStats,
    terminated: Arc<AtomicBool>,
}

impl Default for BpSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl BpSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
            terminated: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Solver for BpSolver {
    /// Stage-2 knobs ([`BpConfig`]); production defaults are set in
    /// [`main`].
    type Config = BpConfig;
    const SUPPORTED_TRACKS: &'static [Track] = &[Track::Exact, Track::Heuristic];

    fn solve(&mut self, instance: &Instance, cfg: &RunConfig<Self::Config>) -> Option<Vec<Tree>> {
        // m=2 routing (Exact track): the corridor solver is the 2-tree-native
        // exact engine and certifies optimality by closing its reduced-cost
        // window. Use its result ONLY when it certifies (`lb >= ub`); an
        // unproven corridor incumbent can be suboptimal (e.g. 224 vs opt 223),
        // so on a non-certifying instance we fall through to B&P. B&P times out
        // on the large 2-tree instances anyway, so this is pure upside.
        // Default on; disable with `KLADOS_BP_M2_CORRIDOR=0`.
        if cfg.track == Track::Exact
            && instance.num_trees() == 2
            && std::env::var("KLADOS_BP_M2_CORRIDOR").as_deref() != Ok("0")
            && let Some(forest) =
                crate::solvers::corridor::CorridorSolver::new().solve_m2_certified(instance)
        {
            self.stats.upper_bound = Some(forest.len());
            self.stats.lower_bound = forest.len();
            return Some(forest);
        }

        let t_total = Instant::now();
        let memo = Rc::new(RefCell::new(SubinstanceMemo::default()));
        let cancel = Cancel::new(Arc::clone(&self.terminated));
        let mut components = solve_recursive_memo(instance, &cfg.specific, &memo, &cancel)?;

        // Post-validate: if Whidden decomp assembled invalid results
        // (subproblems aborted), fall back to Chen 2-approximation.
        if instance.num_trees() == 2 && !validate_agreement_forest(instance, &components).is_ok() {
            let (_, _, leafsets) = chen_pair_agreement(&instance.trees[0], &instance.trees[1]);
            components = crate::solvers::chen_rspr::leafsets_to_trees(&leafsets, instance);
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

    fn sigterm_handler(&self, track: Track) -> Option<Box<dyn Fn() + Send + Sync>> {
        match track {
            // Exact emits nothing unless proven optimal, so SIGTERM is ignored:
            // aborting the search could only yield an unproven partial. The
            // harness SIGKILLs at the hard deadline.
            Track::Exact => None,
            // Heuristic: flip the existing stop flag; bp polls it at its cancel
            // points and returns its best incumbent.
            Track::Heuristic => {
                let flag = Arc::clone(&self.terminated);
                Some(Box::new(move || flag.store(true, Ordering::SeqCst)))
            }
            Track::LowerBound => None,
        }
    }
}

/// Cancellation for an exact solve: a shared SIGTERM flag plus an OPTIONAL
/// wall-clock deadline. It is polled at the search's existing abort points, so
/// a per-cluster time cap needs **no watchdog thread** — the inner B&P checks
/// the clock itself instead of a separate thread flipping the flag (the old
/// watchdog slept on a poll and the caller `join()`ed it, idling the main
/// thread up to one poll interval per cluster probe).
#[derive(Clone)]
pub struct Cancel {
    flag: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

impl Cancel {
    /// Cancel on the shared flag only (no deadline).
    pub fn new(flag: Arc<AtomicBool>) -> Self {
        Self {
            flag,
            deadline: None,
        }
    }

    /// Cancel on the shared flag OR once `deadline` passes.
    pub fn with_deadline(flag: Arc<AtomicBool>, deadline: Option<Instant>) -> Self {
        Self { flag, deadline }
    }

    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire) || self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    #[inline]
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// The underlying flag, for inner callers (e.g. the pricer) that take a bare
    /// `&AtomicBool`. Pricing callers should also pass [`Self::deadline`] into
    /// [`crate::solvers::bp::pricer::PricingContext`] so long DP calls can honor wall
    /// caps without waiting for the next node/CG poll.
    #[inline]
    pub fn flag(&self) -> &AtomicBool {
        &self.flag
    }
}

/// Re-entry point for primal heuristics that need to recursively solve
/// sub-instances (e.g. Whidden relaxed decomposition).  Exposed `pub(crate)`
/// so [`solver::solve_inner`] can call it.
pub(crate) fn solve_subinstance(
    instance: &Instance,
    cfg: &BpConfig,
    cancel: &Cancel,
) -> Option<Vec<Tree>> {
    let memo = Rc::new(RefCell::new(SubinstanceMemo::default()));
    solve_recursive_memo(instance, cfg, &memo, cancel)
}

/// Solve a 2-tree sub-instance exactly under an external termination flag.
/// Returns only validated agreement forests; `None` means the solve was cut
/// short or failed validation.
pub fn bp_solve_capped(instance: &Instance, terminate: &Arc<AtomicBool>) -> Option<Vec<Tree>> {
    bp_solve_capped_until(instance, terminate, None)
}

/// Like [`bp_solve_capped`] but also caps the solve at `deadline` (the search
/// polls the deadline itself — no watchdog thread).
pub fn bp_solve_capped_until(
    instance: &Instance,
    terminate: &Arc<AtomicBool>,
    deadline: Option<Instant>,
) -> Option<Vec<Tree>> {
    if instance.num_trees() != 2 || instance.num_leaves < 2 {
        return None;
    }
    let cfg = BpConfig::default();
    let cancel = Cancel::with_deadline(Arc::clone(terminate), deadline);
    let comps = solve_subinstance(instance, &cfg, &cancel)?;
    if !validate_agreement_forest(instance, &comps).is_ok() {
        return None;
    }
    Some(comps)
}

#[allow(dead_code)]
fn solve_recursive(instance: &Instance, cfg: &BpConfig, cancel: &Cancel) -> Option<Vec<Tree>> {
    let memo = Rc::new(RefCell::new(SubinstanceMemo::default()));
    solve_recursive_memo(instance, cfg, &memo, cancel)
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
    cancel: &Cancel,
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

    if cancel.is_cancelled() {
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
            reverse_map: (0..=instance.num_leaves).collect(),
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
                debug!("MEMOKEY\tn={}\tkey={}", reduced.num_leaves, view.key);
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
            solver::solve_inner_with_subsolver(reduced, &cfg_inner, cancel, &mut |sub| {
                solve_recursive_memo(sub, &cfg_inner, &memo_inner, cancel)
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
        if let Some(comps) = try_whidden_decomp_2tree(
            reduced,
            &mut |sub| solve_recursive_memo(sub, &cfg_inner, &memo_inner, cancel),
            cancel.flag(),
        ) {
            debug!(
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

    if matches!(cfg.cluster_algo, ClusterAlgo::None) {
        let cfg_inner = cfg.clone();
        let memo_inner = Rc::clone(memo);
        let reduced_components =
            solver::solve_inner_with_subsolver(reduced, &cfg_inner, cancel, &mut |sub| {
                solve_recursive_memo(sub, &cfg_inner, &memo_inner, cancel)
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

    let pipeline_cfg = SolveConfig {
        kernelize: false, // already kernelized above
        kernelize_config: Default::default(),
        cluster_algo: cfg.cluster_algo.clone(),
    };
    let inner_cfg = cfg.clone();
    let reduced_num_leaves = reduced.num_leaves;
    let memo_pipeline = Rc::clone(memo);
    let c = cancel.clone();
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
                if let Some(comps) = try_whidden_decomp_2tree(
                    sub,
                    &mut |s| solve_recursive_memo(s, &cfg2, &memo2, &c),
                    c.flag(),
                ) {
                    return Some(comps);
                }
            }
            if sub.num_leaves < reduced_num_leaves {
                solve_recursive_memo(sub, &inner_cfg, &memo_pipeline, &c)
            } else {
                let cfg3 = inner_cfg.clone();
                let memo3 = Rc::clone(&memo_pipeline);
                solver::solve_inner_with_subsolver(sub, &cfg3, &c, &mut |s| {
                    solve_recursive_memo(s, &cfg3, &memo3, &c)
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
        let mut entries: Vec<(u32, u32)> = (1..=n as u32).map(|l| (color[l as usize], l)).collect();
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
        if st.best_key.as_ref().is_none_or(|bk| key < *bk) {
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
    let max_h = h0.iter().chain(h1.iter()).copied().max().unwrap_or(0);

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

// ── entry point ─────────────────────────────────────────────────────────────
use crate::{RunConfig, Solver, Track};

pub fn main() {
    crate::run(
        BpSolver::new(),
        RunConfig {
            track: Track::Exact,
            specific: BpConfig::from_env(),
            ..Default::default()
        },
    );
}

#[cfg(test)]
mod canon_tests {
    use super::*;
    use klados_core::tree::{Label, NONE, NodeId};

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
                let lbl: u32 = std::str::from_utf8(&b[start..*pos])
                    .unwrap()
                    .parse()
                    .unwrap();
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
            let k = canonicalize_two_tree_instance(&inst)
                .expect("perm canonicalizes")
                .key;
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
        check_invariant("(1,(2,(3,(4,(5,6)))))", "(((((1,6),2),5),3),4)", 6);
    }
}
