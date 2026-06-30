//! Dual-guided set-packing heuristic (design doc R1 core).
//!
//! Reproduces the B&P RMP dual signal **without an LP** via Lagrangian
//! relaxation (full dualization → per-column box subproblem has the
//! integrality property → by Geoffrion the multipliers equal the LP duals).
//! The production anchor DP is the Lagrangian separation oracle; a
//! dual-guided greedy node-disjoint packing is the anytime primal.
//!
//! Pipeline:
//!   1. Chen 2-approx forest = instant valid incumbent + seed columns.
//!   2. Subgradient loop: price (anchor DP at current α,β) → enrich pool →
//!      dual-guided greedy packing → keep best → subgradient multiplier update.
//!   3. On SIGTERM / time budget, return the best forest seen.
//!
//! The anchor DP declines (no columns) when its dense n₀·n₁ table exceeds the
//! cell cap (~15k leaves) — that memory-lean pricing is the separate R2 item.
//! Here the loop degrades gracefully to the Chen+seed pool at that scale.

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::decomp::whidden_cluster::try_whidden_decomp_2tree;
use crate::solvers::bp::column::{AfColumn, ColumnBuilder, ColumnSet, is_valid_af_component};
use crate::solvers::bp::pricer::exact_pair_dp::ExactPairDpCache;
use crate::solvers::bp::pricer::{
    ExactPairDpPricer, MafPricer, Pricer, PricerScratch, PricingContext, PricingResult,
    dispatch_by_m,
};
use crate::solvers::bp::rmp::{Rmp, RmpSolution};
use crate::solvers::bp::search::{
    BranchSelector, Branchings, LeafPair, MostFractionalPair, SelectionContext,
};
use crate::solvers::chen_rspr::chen_pair_agreement;
use fixedbitset::FixedBitSet;
use fxhash::FxHashSet;
use klados_core::af_validator::validate_agreement_forest;
use klados_core::tree::{NONE, NodeId, Tree};
use klados_core::{Instance, SolverStats};
use log::debug;

const POOL_HARD_CAP: usize = 120_000;
const POOL_PRUNE_TO: usize = 80_000;
/// Memory ceiling for the column pool, measured in u32 "cells" (label + cover
/// entries). On big cluster-free cores each column stores a full V-set cover
/// (~thousands of cells), so the 120k COUNT cap alone let the pool reach 4-7 GB
/// and OOM the 8 GB-limited instances. We bound the pool by total cells instead
/// (~4 bytes each); 500M cells ≈ 2 GB, leaving headroom for trees/windows/seen
/// under the limit. Pruning is by Lagrangian score, so the best columns survive.
const POOL_CELL_BUDGET: usize = 500_000_000;

/// Number of u32 cells a column occupies (labels + per-tree cover V-sets).
fn block_cells(b: &Block) -> usize {
    b.labels.len() + b.cover.iter().map(|c| c.len()).sum::<usize>()
}
/// B&P-whole tier: only try the full exact solver (with its internal Whidden
/// decomposition) on instances up to this many leaves — beyond it B&P will just
/// burn the cap. The cap itself bounds wasted time on within-range timeouts.
const BP_TRY_MAX_LEAVES: u32 = 6000;
/// Cluster router: don't attempt to split a sub-instance below this many leaves
/// (decomposition overhead isn't worth it; just solve it).
const DECOMP_MIN_LEAVES: u32 = 50;
/// Above this reduced-leaf count we skip the decomposition attempt entirely:
/// decomposition descends into a deep cluster tree whose O(n²) recombination
/// and per-cluster exact solves can burn the *entire* budget without proving,
/// emitting only the Chen baseline — the flat engine does far better with that
/// time (n=10037→5422 core: decomp 5000 vs flat 4277, best 4194). The attempt's
/// budget can't reclaim this because the descent already built the deep tree and
/// the recombination is uninterruptible, so we gate by size: only small cores,
/// where the tree stays shallow and proving is genuinely fast, get the attempt.
/// 4000 excludes the observed grinder (5422-leaf core: decomp 5000 → flat 4281)
/// while keeping mid cores that decompose well (3295-leaf: decomp 2381 beats
/// flat's 2455). The exact cut is benchmark-tunable — it trades the grinder fix
/// against small regressions on any well-decomposing core in the 4000–6000 band.
const DECOMP_TRY_MAX_LEAVES: u32 = 4_000;
/// Wall-clock the decomposition attempt is allowed before we conclude the
/// instance has a hard core and hand off to the flat engine. Well-decomposing
/// instances prove well within this; hard ones forfeit only this much.
const DECOMP_ATTEMPT: Duration = Duration::from_secs(25);
/// Only attempt the certifying MIP when the incumbent is within this many
/// components of the LP bound (a wide gap won't close and risks a HiGHS
/// time-limit overrun that blows the SIGTERM grace window).
const MIP_GAP_LIMIT: usize = 4;

/// Safe ceiling on the anchor DP's dense `n₀·n₁` table (kept under the
/// pricer's own ~64M-cell cap). Above this we price in tree-local windows.
const CELL_CAP_SAFE: u64 = 60_000_000;

/// A validated agreement-forest column with its per-tree V-set internal nodes.
struct Block {
    labels: Vec<u32>,
    weight: usize,        // |labels| - 1
    cover: Vec<Vec<u32>>, // internal node ids per tree (the embedding V-set)
}

/// V-set internal nodes of `labels` in `tree` (nodes on the leaf→LCA paths).
fn vset_internal(tree: &Tree, labels: &[u32]) -> Vec<u32> {
    if labels.len() < 2 {
        return Vec::new();
    }
    let mut lca = tree.node_by_label(labels[0]);
    for &l in &labels[1..] {
        lca = tree.nearest_common_ancestor(lca, tree.node_by_label(l));
    }
    let mut mark = FixedBitSet::with_capacity(tree.num_nodes());
    for &l in labels {
        let mut cur = tree.parent[tree.node_by_label(l) as usize];
        while cur != NONE && !mark.contains(cur as usize) {
            mark.insert(cur as usize);
            if cur == lca {
                break;
            }
            cur = tree.parent[cur as usize];
        }
    }
    mark.ones().map(|v| v as u32).collect()
}

fn make_block(trees: &[Tree], mut labels: Vec<u32>) -> Option<Block> {
    labels.sort_unstable();
    labels.dedup();
    if labels.len() < 2 {
        return None;
    }
    let cover = trees.iter().map(|t| vset_internal(t, &labels)).collect();
    Some(Block {
        weight: labels.len() - 1,
        labels,
        cover,
    })
}

#[inline]
fn block_score(b: &Block, alpha: &[f64], beta: &[Vec<f64>]) -> f64 {
    let leaf_gain: f64 = b.labels.iter().map(|&l| alpha[l as usize]).sum();
    let mut node_pen = 0.0;
    for (t, nodes) in b.cover.iter().enumerate() {
        for &v in nodes {
            node_pen += beta[t][v as usize];
        }
    }
    leaf_gain - node_pen
}

#[derive(Clone, Debug)]
pub struct LagrangianConfig {
    pub whole_bp: bool,
    pub bp_cap_ms: u64,
    pub no_decomp: bool,
    pub lns_on: bool,
    pub lbnb: bool,
    pub dive: bool,
    pub force_lp: bool,
    pub no_rmp: bool,
    pub rmp_cap_ms: u64,
    pub no_rmp_warm: bool,
    /// Max leaves per T₀-subtree pricing window. `(2·W)² ≤ CELL_CAP_SAFE` so each
    /// window's restricted DP fits; per-window cache ≈ 32·(2W)² bytes.
    pub window_max: usize,
    pub volume: bool,
    pub volume_a: f64,
    pub hybrid: bool,
    pub refresh_every: usize,
    pub no_ls: bool,
    pub profile: bool,
    pub reprice_eps: f64,
    pub reprice_maxstale: usize,
    pub hybrid_blend: f64,
    pub stall_thresh: usize,
    pub branch: bool,
    pub plan_ms: u64,
    /// Cluster router: irreducible clusters at or below this many leaves are
    /// probed with the exact B&P solver (used only if it proves optimal in the
    /// cap); larger ones, and probes that don't finish, go to the anytime cascade.
    pub decomp_exact: usize,
    pub decomp_exact_cap_ms: u64,
    pub lbnb_root_ms: u64,
    pub lbnb_node_ms: u64,
    pub dive_reopt: usize,
    pub dive_warmup: f64,
    pub lns_max: usize,
    pub lns_cap_ms: u64,
    pub bnb: bool,
    pub topdown_windows: bool,
}

impl Default for LagrangianConfig {
    fn default() -> Self {
        Self {
            whole_bp: false,
            bp_cap_ms: 0,
            no_decomp: false,
            lns_on: true,
            lbnb: false,
            dive: false,
            force_lp: false,
            no_rmp: false,
            rmp_cap_ms: 15_000,
            no_rmp_warm: false,
            window_max: 1_200,
            volume: false,
            volume_a: 0.1,
            hybrid: false,
            refresh_every: 15,
            no_ls: false,
            profile: false,
            reprice_eps: 0.05,
            reprice_maxstale: 10,
            hybrid_blend: 1.0,
            stall_thresh: 20,
            branch: false,
            plan_ms: 290_000,
            decomp_exact: 600,
            decomp_exact_cap_ms: 10_000,
            lbnb_root_ms: 20_000,
            lbnb_node_ms: 2_500,
            dive_reopt: 3,
            dive_warmup: 0.4,
            lns_max: 250,
            lns_cap_ms: 1_000,
            bnb: false,
            topdown_windows: false,
        }
    }
}

pub struct LagrangianSolver {
    config: LagrangianConfig,
    terminate: Arc<AtomicBool>,
    stats: SolverStats,
    /// True while the current solve is the flat top-level (depth 0), whose
    /// forests are terminal (→ LNS → expand → output) and never fed to
    /// `whidden_cluster` recombination. Set at each `solve_reduced_core` entry.
    /// Lets the primal packers (`solve_rmp`/`bnb_anytime`) build their
    /// per-iteration forests WITHOUT the O(num_leaves) index — the bulk of the
    /// during-solve forest memory on large flat cores. Cluster solves (depth ≥
    /// 1) leave it false so their pieces stay label-queryable for recombination.
    flat_terminal: AtomicBool,
    /// Soft wall-time budget for the anytime cascade, from `cfg.budget` (the
    /// run-track wall limit) or set directly by the lower-track racer so it
    /// always returns even without a SIGTERM. `None` ⇒ run until SIGTERM.
    budget_override: Option<Duration>,
    /// Ceil of the dual lower bound from the most recent depth-0 flat solve, in
    /// REDUCED units. `solve` lifts it to original units (adding the kernel
    /// delta) and stores it in `stats.lower_bound`. Lets the lower-track racer
    /// raise its acceptance threshold from this tight bound instead of Chen's.
    reduced_dual_lb: AtomicUsize,
    /// Lower-bound track acceptance parameters `(a, b)`. When set, the flat core
    /// stops the instant its current forest satisfies `k <= floor(a*LB) + b`
    /// against its OWN dual bound (the tightest available) — the speed-bonus
    /// early-abort. `None` (default) ⇒ run to budget as usual (other tracks).
    approx_target: Option<(f64, usize)>,
    /// Kernel parameter reduction for the active solve: `expanded = reduced +
    /// param_reduction`. Lets the core convert reduced sizes to original units
    /// for the approx-target check without an expansion.
    param_reduction: AtomicUsize,
    /// Frozen early-abort target in REDUCED units: once the subgradient loop's
    /// dual bound is final, any reduced forest of `<= abort_k_reduced` components
    /// satisfies the track bound. The primal-improvement phases (local search,
    /// LNS) watch this and stop the instant they reach it. `usize::MAX` ⇒ no
    /// approx target / not yet frozen.
    abort_k_reduced: AtomicUsize,
    /// True only while the genuine flat top-level solve (and its LNS) is running.
    /// The early-abort must NOT fire inside the decomposition attempt's cluster
    /// solves — their local forest/bound describe a slice, not the whole
    /// instance, and tripping the global `terminate` there would abandon the
    /// real flat solve. Gates every approx-target check.
    abort_armed: AtomicBool,
}

impl LagrangianSolver {
    pub fn new() -> Self {
        Self {
            config: LagrangianConfig::default(),
            terminate: Arc::new(AtomicBool::new(false)),
            stats: SolverStats::default(),
            flat_terminal: AtomicBool::new(false),
            budget_override: None,
            reduced_dual_lb: AtomicUsize::new(0),
            approx_target: None,
            param_reduction: AtomicUsize::new(0),
            abort_k_reduced: AtomicUsize::new(usize::MAX),
            abort_armed: AtomicBool::new(false),
        }
    }

    /// Set a soft wall-time budget for this solve.
    pub fn set_budget(&mut self, budget: Duration) {
        self.budget_override = Some(budget);
    }

    /// Enable the lower-bound track early-abort: stop as soon as the current
    /// forest satisfies `k <= floor(a * LB) + b` against the solver's own dual
    /// bound. Used by the lower-track racer to claim the speed bonus instead of
    /// grinding the full budget after a valid forest is already in hand.
    pub fn set_approx_target(&mut self, a: f64, b: usize) {
        self.approx_target = Some((a, b));
    }

    pub fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        let orig_n = instance.num_leaves;
        if instance.num_trees() < 2 {
            return Some(instance.trees.clone());
        }
        if orig_n <= 1 {
            return Some(instance.trees[0..1].to_vec());
        }
        // This solver is specialised for the 2-tree heuristic track.
        if instance.num_trees() != 2 {
            return Some((1..=orig_n).map(|l| Tree::singleton(l, orig_n)).collect());
        }

        let start = Instant::now();
        let budget = self.budget_override;
        self.reduced_dual_lb.store(0, Ordering::Relaxed);
        self.abort_k_reduced.store(usize::MAX, Ordering::Relaxed);
        self.abort_armed.store(false, Ordering::Relaxed);
        self.stats.lower_bound = 0;

        // Compute the Chen 2-approx forest up front (original labels): it is O(n)
        // and drives the lower-bound-track fast path below. It is no longer
        // *published* as an early SIGTERM baseline — the harness is single-threaded
        // and the SIGTERM handler only flips the stop flag, so `solve()` must (and
        // does) notice it and return its best forest promptly on its own.
        {
            let (chen_lb_dist, _up, sets) =
                chen_pair_agreement(&instance.trees[0], &instance.trees[1]);
            let base = forest_from_partition(&sets, &instance.trees, orig_n, true);
            let (base, _) = repair_forest(base, &instance.trees, orig_n);
            // Lower-bound track fast path: Chen's own forest may already clear the
            // approximation bound against Chen's (sound) lower bound `lb_dist + 1`.
            // If so there is nothing to do — return it before any heavy work. This
            // subsumes the racer's old separate "tier-1 Chen" pass.
            if let Some((a, b)) = self.approx_target {
                let chen_lb = chen_lb_dist + 1;
                if base.len() <= (a * chen_lb as f64).floor() as usize + b {
                    self.stats.lower_bound = chen_lb;
                    self.stats.upper_bound = Some(base.len());
                    return Some(base);
                }
            }
        }

        // ---- Kernelize first (optimality-preserving), solve the reduced core,
        //      expand at the end. Shrinks the instance so global pricing fits
        //      more often and the pool is over the conflict core, not agreeing
        //      pendant structure. ----
        let mut kern_cfg = klados_core::kernelize::KernelizeConfig::default();
        if !instance.protected_labels.is_empty() {
            kern_cfg.protected_labels = instance.protected_labels.clone();
        }
        let kern = klados_core::kernelize::kernelize_best(instance, &kern_cfg);
        self.param_reduction
            .store(kern.param_reduction, Ordering::Relaxed);
        let reduced = &kern.instance;
        debug!(
            "[lagr] kernelize {} -> {} leaves ({:.0}ms)",
            orig_n,
            reduced.num_leaves,
            start.elapsed().as_secs_f64() * 1000.0
        );
        if reduced.num_leaves <= 1 {
            let reduced_forest = if reduced.num_leaves == 0 {
                Vec::new()
            } else {
                vec![reduced.trees[0].clone()]
            };
            let expanded = klados_core::kernelize::expand_solution_unindexed(
                reduced_forest,
                &kern,
                &instance.trees[0],
                orig_n,
            );
            self.stats.upper_bound = Some(expanded.len());
            return Some(expanded);
        }

        let deadline = budget.map(|b| start + b);

        // ---- B&P-whole tier ----
        // Full exact Branch & Price, WITH its internal Whidden cluster
        // decomposition (decompose → solve clusters exactly → recombine). Where
        // it can fully solve the instance it returns the OPTIMUM, often beating
        // best-known/loK10 (n=4465: 2710 vs loK10's 2717). Capped, and
        // `solve_cluster_exact` returns None unless B&P PROVES optimality within
        // the cap — a timed-out garbage incumbent is never used. On timeout we
        // fall through to the anytime cascade. This is decomposition delivering
        // the right way (inside B&P), not the lossy external split.
        // DISABLED BY DEFAULT (opt-in via `LagrangianConfig.whole_bp`). Measured data
        // shows the whole-instance B&P gamble HURTS: it can burn up to half the
        // budget getting stuck on an instance whose difficulty we can't predict,
        // then hands the Lagrangian a fraction of the time (n4465: default 3012
        // vs pure-Lagrangian 2955). B&P now runs only where it can't get stuck —
        // the bounded per-cluster probe inside the decomposition cascade.
        if self.config.whole_bp
            && reduced.num_trees() == 2
            && reduced.num_leaves <= BP_TRY_MAX_LEAVES
        {
            let bp_cap = if self.config.bp_cap_ms > 0 {
                Duration::from_millis(self.config.bp_cap_ms)
            } else {
                match budget {
                    Some(b) => b.mul_f64(0.5).min(Duration::from_secs(90)),
                    None => Duration::from_secs(90),
                }
            };
            let bp_deadline = match deadline {
                Some(d) => (start + bp_cap).min(d),
                None => start + bp_cap,
            };
            if let Some(reduced_forest) = self.solve_cluster_exact(reduced, bp_deadline) {
                debug!(
                    "[lagr] B&P solved n={} k={} (OPTIMAL) in {:.1}s",
                    reduced.num_leaves,
                    reduced_forest.len(),
                    start.elapsed().as_secs_f64()
                );
                let expanded = klados_core::kernelize::expand_solution_unindexed(
                    reduced_forest,
                    &kern,
                    &instance.trees[0],
                    orig_n,
                );
                let (expanded, _) = repair_forest(expanded, &instance.trees, orig_n);
                self.stats.upper_bound = Some(expanded.len());
                return Some(expanded);
            }
            debug!(
                "[lagr] B&P did not finish in {:.0}s cap — anytime cascade",
                bp_cap.as_secs_f64()
            );
        }

        // ---- Phase 1: construct a complete forest — decomposition router ----
        //
        // Route by PROVABILITY, not size. The cluster decomposition's strength
        // is exact B&P on tiny clusters: on instances that split into provable
        // clusters it certifies the optimum almost instantly. Its weakness is
        // the hard-core / large case, where its recombination is O(n²) and it
        // ends up emitting little better than Chen — there the flat anytime
        // Lagrangian is far better (n=4669: 2789 vs 3253; n=15692: 7158 vs
        // 8349) and maintains a complete incumbent that scales.
        //
        // So we give decomposition a *budgeted* attempt (it then slices the
        // budget across clusters and RETURNS rather than grinding). If it proves
        // the optimum → use it (the many fast-proving instances). If it cannot
        // prove within the attempt, the instance has a hard core → discard and
        // hand the whole reduced core to the flat engine for the rest of the
        // time. Giants skip the attempt outright (decomposition can't help and
        // its recombination would waste the budget).
        let no_decomp = self.config.no_decomp;
        let try_decomp = !no_decomp
            && reduced.num_trees() == 2
            && reduced.num_leaves >= DECOMP_MIN_LEAVES
            && reduced.num_leaves <= DECOMP_TRY_MAX_LEAVES;
        let (mut reduced_forest, proved) = {
            let mut proven = None;
            if try_decomp {
                let attempt = budget
                    .map(|b| b.min(DECOMP_ATTEMPT))
                    .unwrap_or(DECOMP_ATTEMPT);
                let (forest, p) = self.solve_decomposed(reduced, Some(attempt), start);
                if p {
                    proven = Some((forest, true)); // certified optimal — done
                } else {
                    debug!(
                        "[lagr] decomp attempt did not prove (hard core) — flat engine, t={:.1}s",
                        start.elapsed().as_secs_f64()
                    );
                }
            }
            proven.unwrap_or_else(|| {
                // Arm the lower-track early-abort: the flat top-level solve (and
                // the LNS that follows) is the only place a forest describes the
                // whole instance, so it is the only place the abort may fire.
                self.abort_armed.store(true, Ordering::Relaxed);
                self.solve_reduced_core(reduced, deadline, start, 0)
            })
        };
        debug!(
            "[lagr] construction done: k={} proved={} t={:.1}s",
            reduced_forest.len(),
            proved,
            start.elapsed().as_secs_f64()
        );
        // ---- Phase 2: refine until proven optimal or SIGTERM ----
        // If construction PROVED the optimum there is nothing to improve — stop.
        // Otherwise the optimum is unproven (LP↔IP gap on the hard cores): spend
        // EVERY remaining second re-solving incumbent regions exactly (LNS) and
        // splicing back any that shrink — monotone, never worsens the forest.
        // Default ON (`LagrangianConfig.lns_on = false` disables). Runs until deadline/SIGTERM.
        let lns_on = self.config.lns_on;
        if lns_on && !proved && reduced.num_trees() == 2 && !self.terminate.load(Ordering::Relaxed)
        {
            reduced_forest = self.lns_improve(reduced, reduced_forest, deadline, start);
        }
        let reduced_len = reduced_forest.len();
        let expanded = klados_core::kernelize::expand_solution_unindexed(
            reduced_forest,
            &kern,
            &instance.trees[0],
            orig_n,
        );
        let (expanded, _) = repair_forest(expanded, &instance.trees, orig_n);
        // Lift the reduced dual bound to original units. Kernelization lowers the
        // optimum by a fixed delta (the pendant/chain components re-added on
        // expansion), so `k* >= ceil(reduced_dual_lb) + delta`. When the optimum
        // was proven, LB = UB = expanded size.
        let kernel_delta = expanded.len().saturating_sub(reduced_len);
        self.stats.lower_bound = if proved {
            expanded.len()
        } else {
            self.reduced_dual_lb.load(Ordering::Relaxed) + kernel_delta
        };
        self.stats.upper_bound = Some(expanded.len());
        Some(expanded)
    }

    /// The per-instance anytime cascade (RMP tier + subgradient) over an
    /// already-reduced 2-tree instance. Returns a forest over `reduced`'s labels
    /// (NOT expanded). `start` is when this solve began; `deadline` bounds it
    /// (`None` = run until SIGTERM). `&self` so the cluster router can recurse.
    fn solve_reduced_core(
        &self,
        reduced: &Instance,
        deadline: Option<Instant>,
        start: Instant,
        depth: usize,
    ) -> (Vec<Tree>, bool) {
        let trees = &reduced.trees;
        let n = reduced.num_leaves;
        let nl = n as usize;
        // Indent trace by decomposition depth so nested cluster solves are
        // visually attributable to their level.
        let ind = "  ".repeat(depth);

        // depth == 0 is the flat top-level solve: its forest is TERMINAL — it
        // flows to LNS → expand → output and is never fed to whidden_cluster
        // recombination (that only consumes sub-cluster forests at depth ≥ 1).
        // So at depth 0 we build the forest pieces WITHOUT the O(num_leaves)
        // `label_to_node` index, turning a k-component forest from O(k·n) ≈
        // O(n²) memory into O(n). depth ≥ 1 (cluster solves) MUST stay indexed —
        // `whidden_cluster` label-queries those pieces during recombination.
        let unindexed = depth == 0;
        // Publish to the field so the primal packers (called via &self deep in
        // the CG loop) pick the same indexing without threading a param through
        // every signature. Single-threaded within a solve; nested cluster solves
        // overwrite it and the depth-0 frame never has a nested core in-stack.
        self.flat_terminal.store(unindexed, Ordering::Relaxed);

        // Anytime Lagrangian branch-and-bound: branch on the dual (subgradient),
        // NOT the LP. The flat subgradient plateaus at the LP↔integer gap because
        // the unconstrained pricer never emits the columns the integer optimum
        // needs; branching on a contended leaf-pair forces them, anytime.
        if self.config.lbnb {
            return (self.solve_lagrangian_bnb(reduced, deadline, start), false);
        }
        // Lagrangian dive: warm the dual, then commit the best column and let
        // the dual re-approximate on the residual, repeatedly — a direct descent
        // that subdivides the solution space and is anytime by construction.
        if self.config.dive {
            return (self.solve_lagrangian_dive(reduced, deadline, start), false);
        }

        // ---- Tier cascade ----
        // When global pricing fits, the warm exact-LP RMP proves small/integral
        // instances in milliseconds (bp's small-instance speed). Try it first
        // with a capped budget; if it certifies optimality, return. Otherwise
        // (integrality gap, or it didn't converge in the cap) fall through to
        // the subgradient anytime solver, which wins the primal on gap/large
        // instances via its diverse pool. Routing is by PROVABILITY, not size.
        let global_fits = (reduced.trees[0].num_nodes() as u64)
            * (reduced.trees[1].num_nodes() as u64)
            <= CELL_CAP_SAFE;
        let force_lp = self.config.force_lp;
        let no_rmp = self.config.no_rmp;
        // Columns the (capped) RMP tier priced; fed to the subgradient pool below
        // so its CG warm-starts instead of rediscovering them from Chen.
        let mut rmp_seed_labels: Vec<Vec<u32>> = Vec::new();
        if (global_fits || force_lp) && !no_rmp {
            // Cap the RMP attempt so the subgradient ALWAYS gets the bulk of the
            // budget. Proving happens fast or not at all (n=60, pub049 prove in
            // <1s); a small-but-gappy instance can otherwise run CG for the whole
            // window, starving the subgradient (which wins the primal on gap
            // instances) AND risking the SIGTERM grace. So: cap at a modest
            // absolute ceiling, and at ≤¼ of a known budget. `LagrangianConfig.force_lp`
            // forces the full budget (the standalone RMP-engine path, testing).
            let cap_ceiling = Duration::from_millis(self.config.rmp_cap_ms);
            let rmp_deadline = if force_lp {
                deadline
            } else {
                let dur = match deadline {
                    Some(d) => (d.saturating_duration_since(start) / 4).min(cap_ceiling),
                    None => cap_ceiling,
                };
                Some(start + dur)
            };
            let (rmp_forest, rmp_proved, rmp_pool) = self.solve_rmp(reduced, rmp_deadline, start);
            if force_lp || rmp_proved {
                return (rmp_forest, rmp_proved);
            }
            rmp_seed_labels = rmp_pool;
            debug!(
                "{ind}[lagr] RMP did not certify (gap) — handing off to subgradient at {:.1}s",
                start.elapsed().as_secs_f64()
            );
        }

        // ---- Pool + dedup ----
        let mut pool: Vec<Block> = Vec::new();
        let mut seen = ColumnSet::new();
        let add_block = |labels: Vec<u32>, pool: &mut Vec<Block>, seen: &mut ColumnSet| -> bool {
            let mut l = labels;
            l.sort_unstable();
            l.dedup();
            if l.len() < 2 {
                return false;
            }
            if seen.contains(&l) {
                return false;
            }
            if !is_valid_af_component(&l, trees) {
                return false;
            }
            if let Some(b) = make_block(trees, l.clone()) {
                seen.insert(l);
                pool.push(b);
                true
            } else {
                false
            }
        };

        // ---- Warm start: Chen 2-approx forest ----
        let (_chen_lo, _chen_up, chen_sets) = chen_pair_agreement(&trees[0], &trees[1]);
        let mut best_forest = forest_from_partition(&chen_sets, trees, n, unindexed);
        let mut best_components = best_forest.len();
        // Column indices of the best packing so far (seed for the improvement
        // loop). Empty ⇒ the Chen warm-start (all singletons w.r.t. the pool).
        let mut best_sel: Vec<usize> = Vec::new();
        for s in &chen_sets {
            add_block(s.clone(), &mut pool, &mut seen);
        }
        debug!(
            "{ind}[lagr] n={} chen incumbent={} ({:.0}ms)",
            n,
            best_components,
            start.elapsed().as_secs_f64() * 1000.0
        );

        // ---- Seed pool with a few overlapping greedy partitions ----
        let num_seeds: u64 = if n <= 2_000 {
            12
        } else if n <= 6_000 {
            5
        } else {
            2
        };
        for ref_idx in 0..2usize {
            for seed in 0..num_seeds {
                if self.terminate.load(Ordering::Relaxed) {
                    break;
                }
                let (_k, part) =
                    klados_core::lower_bound::greedy_multi_tree_partition(trees, ref_idx, seed);
                for g in groups_from_partition(&part, nl) {
                    add_block(g, &mut pool, &mut seen);
                }
            }
        }
        // Warm-start with the columns the RMP tier already priced (avoids the
        // subgradient rediscovering them from cold via column generation).
        // `LagrangianConfig.no_rmp_warm` disables it (for A/B measurement).
        let rmp_warm = if !self.config.no_rmp_warm {
            let n_warm = rmp_seed_labels.len();
            for labels in std::mem::take(&mut rmp_seed_labels) {
                add_block(labels, &mut pool, &mut seen);
            }
            n_warm
        } else {
            0
        };
        debug!(
            "{ind}[lagr] seeded pool={} (rmp_warm={}) ({:.0}ms)",
            pool.len(),
            rmp_warm,
            start.elapsed().as_secs_f64() * 1000.0
        );

        // ---- Multipliers: α per leaf (free), β per node per tree (≥0) ----
        let mut alpha = vec![0.0f64; nl + 1];
        for a in alpha.iter_mut().skip(1) {
            *a = 1.0;
        }
        let mut beta: Vec<Vec<f64>> = trees.iter().map(|t| vec![0.0f64; t.num_nodes()]).collect();

        // ---- Pricer state ----
        let mut pricer = ExactPairDpPricer::new(trees);
        let mut scratch = PricerScratch::new(trees);
        let branchings = Branchings::default();

        // ---- R2: windowed pricing when the global DP table is too large ----
        let global_fits =
            (trees[0].num_nodes() as u64) * (trees[1].num_nodes() as u64) <= CELL_CAP_SAFE;
        let window_max = self.config.window_max;
        let mut windows: Vec<Window> = Vec::new();
        if !global_fits {
            for leaves in split_t0_windows(&trees[0], window_max, self.config.topdown_windows) {
                if self.terminate.load(Ordering::Relaxed) {
                    break;
                }
                let mut keep = FixedBitSet::with_capacity(nl + 1);
                for &l in &leaves {
                    keep.insert(l as usize);
                }
                let (inst, rev) = klados_core::kernelize::restrict_instance_simple(reduced, &keep);
                if inst.num_leaves < 2 || inst.num_trees() != 2 {
                    continue;
                }
                let img: Vec<Vec<u32>> = (0..2)
                    .map(|ti| node_images(&inst.trees[ti], &trees[ti], &rev))
                    .collect();
                windows.push(Window::new(inst, rev, img));
            }
            let sizes: Vec<usize> = windows.iter().map(|w| w.inst.num_leaves as usize).collect();
            let (mn, mx) = (
                sizes.iter().copied().min().unwrap_or(0),
                sizes.iter().copied().max().unwrap_or(0),
            );
            let avg = if sizes.is_empty() {
                0
            } else {
                sizes.iter().sum::<usize>() / sizes.len()
            };
            debug!(
                "{ind}[lagr] windowed pricing: {} windows (cap={}, leaves min/avg/max={}/{}/{}) ({:.0}ms)",
                windows.len(),
                window_max,
                mn,
                avg,
                mx,
                start.elapsed().as_secs_f64() * 1000.0
            );
        }

        // The Lagrangian L is a valid global lower bound only when pricing is
        // global (the dense DP sees the whole column space). Windowed pricing
        // is local, so its bound is not an optimality certificate.
        let global = windows.is_empty();
        let mut lambda = 2.0f64;
        let mut best_lb = 0.0f64;
        let mut stall = 0usize;
        let mut no_new = 0usize;
        let mut iter = 0usize;
        let mut proved = false;
        // Running pool memory (u32 cells) for the budget-based prune.
        let mut pool_cells: usize = pool.iter().map(block_cells).sum();
        // Convergence bookkeeping for the deadline-free stall exit.
        const REENERGISE_DRY_LIMIT: usize = 3;
        let mut reenergise_dry = 0usize;
        let mut best_at_reenergise = best_components;
        // Iteration-stall handoff to local search: exit once the primal
        // incumbent has not improved for this many consecutive iterations
        // (after a warm-up). Generous so a slow-but-real descent isn't cut off.
        const PRIMAL_WARMUP: usize = 30;
        const PRIMAL_STALL_LIMIT: usize = 120;
        let mut since_primal_improve = 0usize;
        // ---- Volume algorithm (Barahona–Anbil) state ----
        // The pure subgradient thrashes (the per-iter dual point and thus the
        // primal scores oscillate). The volume algorithm fixes this with a
        // *stability centre* (the best-bound dual point) that the step departs
        // from, a *running-average primal estimate* `x̄` (per column) that
        // drives a smooth descent direction, and serious/null step control.
        // The primal is then rounded from `x̄` (stable) rather than the
        // thrashing instantaneous reduced-cost scores.
        let volume = self.config.volume;
        let avg_a = self.config.volume_a;
        let mut xbar: Vec<f64> = Vec::new(); // per-column averaged selection
        let mut xbar_sing = vec![0.0f64; nl + 1]; // per-leaf averaged singleton selection
        let mut center_alpha: Vec<f64> = alpha.clone();
        let mut center_beta: Vec<Vec<f64>> = beta.clone();
        let mut center_lb = f64::NEG_INFINITY;
        let mut serious_run = 0usize;
        let mut null_run = 0usize;

        // ---- Hybrid: refresh subgradient duals from a warm exact RMP ----
        // The subgradient's oscillating duals build a DIVERSE pool (which wins
        // the integer primal on gap instances), but its dual *center* drifts.
        // Periodically solve the exact LP over the current pool and overwrite
        // α/β with the LP duals: the pricer + greedy then aim at the true LP
        // optimum while the subgradient keeps diversifying around it.
        let hybrid = self.config.hybrid;
        let refresh_every = self.config.refresh_every.max(1);
        let mut h_builder = ColumnBuilder::new(trees);
        let mut h_afpool: Vec<AfColumn> = Vec::new();
        let mut h_in_rmp = ColumnSet::new();
        let mut h_rmp: Option<Rmp> = None;

        // First primal from the seed pool (dual-guided with the initial α=1).
        {
            let scores: Vec<f64> = pool.iter().map(|b| block_score(b, &alpha, &beta)).collect();
            self.try_primal(
                trees,
                n,
                &pool,
                &scores,
                &mut best_forest,
                &mut best_components,
                &mut best_sel,
            );
        }

        // Reserve a tail of the budget for the primal improvement loop: the
        // subgradient generates the columns; the local search then relentlessly
        // re-selects a better packing over the same pool. ON by default
        // (`LagrangianConfig.no_ls = true` disables) — it helps
        // the hard fallback cores (n4465: 2910→2829) and never hurts (it only
        // refines the subgradient's own plateaued incumbent).
        // Local search runs AFTER the subgradient converges (stall-based), not
        // on a reserved time fraction — so it works with no deadline (the
        // default, SIGTERM-driven) instead of needing a hardcoded horizon to
        // carve a tail from. `LagrangianConfig.no_ls` disables it.
        let ls_on = !self.config.no_ls;

        // Config-gated per-center profiling of the hot loop.
        let profile = self.config.profile;
        let (mut t_price, mut t_add, mut t_score, mut t_sg, mut t_primal) = (
            Duration::ZERO,
            Duration::ZERO,
            Duration::ZERO,
            Duration::ZERO,
            Duration::ZERO,
        );

        // Adaptive windowed re-pricing: skip a window's dense DP when its mapped
        // duals moved by ≤ `reprice_eps` (L∞) since it was last priced, unless it
        // has been stale for `reprice_maxstale` iterations (round-robin floor so
        // no window starves). The subgradient moves slowly, so most windows are
        // stable per iteration — this removes the redundant re-pricing that is
        // ~83% of the giant budget, WITHOUT shrinking windows (no column loss).
        let reprice_eps = self.config.reprice_eps;
        let reprice_maxstale = self.config.reprice_maxstale;
        let mut win_priced_total = 0usize;
        let mut win_skipped_total = 0usize;
        // Windows are priced SEQUENTIALLY, so one DP table suffices for all of
        // them: this single max-sized cache is swapped into each window's
        // scratch around its price call, instead of every window holding its own
        // ~(2W)²·32-byte table. On giants (bottom-up packing → big windows) this
        // cuts the DP-cache footprint from ~Σ(2Wᵢ)² to one ~(2·maxW)² table
        // (e.g. ~1.2 GB → ~0.15 GB), which is the dominant giant RAM consumer.
        let mut shared_dp_cache: Option<ExactPairDpCache> = None;

        loop {
            if self.terminate.load(Ordering::Relaxed)
                || deadline.is_some_and(|d| Instant::now() >= d)
            {
                break;
            }
            iter += 1;

            // ---- Price at current duals (drain banked reserve first) ----
            let tp = Instant::now();
            let mut new_cols: Vec<Vec<u32>> = Vec::new();
            if windows.is_empty() {
                // Global pricing: the dense DP fits.
                let ctx = PricingContext {
                    trees,
                    num_leaves: nl,
                    alpha: &alpha,
                    beta: &beta,
                    columns: &[],
                    seen: &seen,
                    branchings: &branchings,
                    terminate: self.terminate.as_ref(),
                    deadline: None,
                };
                for col in scratch.drain_reserve(&ctx, 64) {
                    new_cols.push(col.labels().to_vec());
                }
                match pricer.price(&ctx, &mut scratch) {
                    PricingResult::Found(cols) => {
                        for c in cols {
                            new_cols.push(c.labels().to_vec());
                        }
                    }
                    PricingResult::Converged | PricingResult::Improving => {}
                }
            } else {
                // Windowed pricing: restrict to each T₀ subtree, map α/β,
                // run the DP, lift columns back to original labels.
                for w in windows.iter_mut() {
                    if self.terminate.load(Ordering::Relaxed)
                        || deadline.is_some_and(|d| Instant::now() >= d)
                    {
                        break;
                    }
                    let rn = w.inst.num_leaves as usize;

                    // ---- Adaptive skip: re-price only if the window's mapped
                    //      duals moved (L∞ > eps) since the last price, or the
                    //      round-robin staleness cap is hit. The O(rn+nodes)
                    //      check is cheap vs the O(rn²) DP it guards. ----
                    let mut reprice = w.stale >= reprice_maxstale;
                    if !reprice {
                        for rl in 1..=rn {
                            if (alpha[w.rev[rl] as usize] - w.last_a[rl]).abs() > reprice_eps {
                                reprice = true;
                                break;
                            }
                        }
                    }
                    if !reprice {
                        'beta: for ti in 0..2 {
                            let imgti = &w.img[ti];
                            for (node, &lb) in w.last_b[ti].iter().enumerate() {
                                let o = imgti[node];
                                let cur = if o != NONE { beta[ti][o as usize] } else { 0.0 };
                                if (cur - lb).abs() > reprice_eps {
                                    reprice = true;
                                    break 'beta;
                                }
                            }
                        }
                    }
                    if !reprice {
                        w.stale += 1;
                        win_skipped_total += 1;
                        continue;
                    }

                    let mut a_r = vec![0.0f64; rn + 1];
                    for rl in 1..=rn {
                        a_r[rl] = alpha[w.rev[rl] as usize];
                    }
                    let mut b_r: Vec<Vec<f64>> = w
                        .inst
                        .trees
                        .iter()
                        .map(|t| vec![0.0f64; t.num_nodes()])
                        .collect();
                    for ti in 0..2 {
                        let imgti = &w.img[ti];
                        for (node, b) in b_r[ti].iter_mut().enumerate() {
                            let o = imgti[node];
                            if o != NONE {
                                *b = beta[ti][o as usize];
                            }
                        }
                    }
                    // Lend the single shared DP table to this window for the
                    // price, then take it back (windows priced sequentially).
                    std::mem::swap(&mut w.scratch.exact_dp_cache, &mut shared_dp_cache);
                    let got: Vec<Vec<u32>> = {
                        let ctx = PricingContext {
                            trees: &w.inst.trees,
                            num_leaves: rn,
                            alpha: &a_r,
                            beta: &b_r,
                            columns: &[],
                            seen: &w.seen,
                            branchings: &branchings,
                            terminate: self.terminate.as_ref(),
                            deadline: None,
                        };
                        let mut g = Vec::new();
                        for col in w.scratch.drain_reserve(&ctx, 64) {
                            g.push(col.labels().to_vec());
                        }
                        match pricer.price(&ctx, &mut w.scratch) {
                            PricingResult::Found(cols) => {
                                for c in cols {
                                    g.push(c.labels().to_vec());
                                }
                            }
                            PricingResult::Converged | PricingResult::Improving => {}
                        }
                        g
                    };
                    std::mem::swap(&mut w.scratch.exact_dp_cache, &mut shared_dp_cache);
                    for rl_labels in got {
                        w.seen.insert(rl_labels.clone());
                        new_cols.push(rl_labels.iter().map(|&rl| w.rev[rl as usize]).collect());
                    }
                    // Record the duals just priced at; reset staleness.
                    w.last_a = a_r;
                    w.last_b = b_r;
                    w.stale = 0;
                    win_priced_total += 1;
                }
            }
            if profile {
                t_price += tp.elapsed();
            }
            let t_add0 = Instant::now();
            let mut added = 0usize;
            for c in new_cols {
                if add_block(c, &mut pool, &mut seen) {
                    added += 1;
                    pool_cells += block_cells(pool.last().unwrap());
                }
            }
            if profile {
                t_add += t_add0.elapsed();
            }
            if pool.len() > POOL_HARD_CAP || pool_cells > POOL_CELL_BUDGET {
                prune_pool(
                    &mut pool,
                    &mut seen,
                    &alpha,
                    &beta,
                    POOL_PRUNE_TO,
                    POOL_CELL_BUDGET,
                );
                pool_cells = pool.iter().map(block_cells).sum();
            }

            // ---- Hybrid dual refresh: overwrite α/β with exact LP duals ----
            if hybrid && iter.is_multiple_of(refresh_every) {
                // Sync the warm RMP with any blocks not yet in it (singletons
                // first, for leaf-row =1 feasibility). Pruned blocks already in
                // the RMP stay there — extra columns only sharpen the duals.
                let need_init = h_rmp.is_none();
                if need_init {
                    for l in 1..=n {
                        if h_in_rmp.insert(vec![l])
                            && let Some(c) = h_builder.try_build(vec![l], trees)
                        {
                            h_afpool.push(c);
                        }
                    }
                }
                let mut fresh: Vec<AfColumn> = Vec::new();
                for b in &pool {
                    if b.labels.len() >= 2
                        && h_in_rmp.insert(b.labels.clone())
                        && let Some(c) = h_builder.try_build(b.labels.clone(), trees)
                    {
                        fresh.push(c);
                    }
                }
                if need_init {
                    h_afpool.extend(fresh);
                    h_rmp = Some(Rmp::new(&h_afpool, trees, nl));
                } else if let Some(rmp) = h_rmp.as_mut() {
                    for c in &fresh {
                        rmp.add_column(c);
                    }
                    h_afpool.extend(fresh);
                }

                if let Some(rmp) = h_rmp.as_mut() {
                    rmp.apply_bounds(&h_afpool, &branchings);
                    if let Ok(mut sol) = rmp.solve() {
                        loop {
                            let cuts =
                                rmp.separate_and_add_cuts(&h_afpool, &sol.column_values, 1e-6);
                            if cuts == 0 {
                                break;
                            }
                            rmp.apply_bounds(&h_afpool, &branchings);
                            match rmp.solve() {
                                Ok(s) => sol = s,
                                Err(_) => break,
                            }
                        }
                        // Pull the subgradient's dual center toward the LP duals.
                        // blend=1 → full snap; <1 keeps some subgradient drift.
                        let blend = self.config.hybrid_blend.clamp(0.0, 1.0);
                        let na = alpha.len().min(sol.leaf_duals.len());
                        for l in 0..na {
                            alpha[l] = (1.0 - blend) * alpha[l] + blend * sol.leaf_duals[l];
                        }
                        for ti in 0..beta.len().min(sol.node_duals.len()) {
                            let nb = beta[ti].len().min(sol.node_duals[ti].len());
                            for nd in 0..nb {
                                beta[ti][nd] =
                                    (1.0 - blend) * beta[ti][nd] + blend * sol.node_duals[ti][nd];
                            }
                        }
                        debug!(
                            "{ind}[lagr][hybrid] refresh iter={} rmp_cols={} lp={:.2} best={} t={:.1}s",
                            iter,
                            h_afpool.len(),
                            sol.objective,
                            best_components,
                            start.elapsed().as_secs_f64()
                        );
                    }
                }
            }

            // Score every block once per round (against the current duals) and
            // reuse it for both the subgradient and the packing — avoids the
            // O(P·log P) score re-evaluation that dominated each round at scale.
            let t_sc0 = Instant::now();
            let scores: Vec<f64> = pool.iter().map(|b| block_score(b, &alpha, &beta)).collect();
            if profile {
                t_score += t_sc0.elapsed();
            }

            // ---- Dual multiplier update (subgradient, or volume) over the pool ----
            let t_sg0 = Instant::now();
            let lb = if volume {
                self.volume_step(
                    trees,
                    nl,
                    &pool,
                    &scores,
                    &mut alpha,
                    &mut beta,
                    &mut xbar,
                    &mut xbar_sing,
                    &mut center_alpha,
                    &mut center_beta,
                    &mut center_lb,
                    &mut serious_run,
                    &mut null_run,
                    &mut lambda,
                    avg_a,
                    best_components,
                )
            } else {
                self.subgradient_step(
                    trees,
                    nl,
                    &pool,
                    &scores,
                    &mut alpha,
                    &mut beta,
                    lambda,
                    best_components,
                )
            };
            if lb > best_lb + 1e-6 {
                best_lb = lb;
                stall = 0;
            } else if !volume {
                // Volume adapts its own step via serious/null counters; only the
                // plain subgradient uses the stall-halving rule here.
                stall += 1;
                if stall >= self.config.stall_thresh {
                    lambda *= 0.5;
                    stall = 0;
                }
            }

            if profile {
                t_sg += t_sg0.elapsed();
            }

            // ---- Dual-guided primal ----
            // Volume: pack the stable per-column averaged estimate x̄ (the
            // running fractional solution). Subgradient: the instantaneous
            // reduced-cost scores. Both greedy, node-disjoint.
            let t_pr0 = Instant::now();
            let primal_scores: &[f64] = if volume { &xbar } else { &scores };
            let improved = self.try_primal(
                trees,
                n,
                &pool,
                primal_scores,
                &mut best_forest,
                &mut best_components,
                &mut best_sel,
            );
            if profile {
                t_primal += t_pr0.elapsed();
            }
            // ---- Lower-bound track early-abort ----
            // Stop the instant the current forest already satisfies the track's
            // approximation bound against our OWN dual lower bound (the tightest
            // available). Both sides lifted to original units via the kernel
            // delta. Armed only for the flat top-level solve (see `abort_armed`).
            if self.abort_armed.load(Ordering::Relaxed)
                && let Some((a, b)) = self.approx_target
            {
                let delta = self.param_reduction.load(Ordering::Relaxed);
                let lb_orig = best_lb.ceil().max(0.0) as usize + delta;
                let k_orig = best_components + delta;
                if lb_orig > 0 && k_orig <= (a * lb_orig as f64).floor() as usize + b {
                    self.terminate.store(true, Ordering::Relaxed);
                }
            }
            // Primal-stall convergence: the greedy primal has plateaued (the LB
            // may still trickle up, but the incumbent isn't moving). Hand off to
            // the local search. Keys on the PRIMAL, not the bound, so it fires
            // even while the dual keeps tightening — the robust deadline-free
            // exit. Skipped until the dual has had a warm-up so we don't cut a
            // still-improving early ascent short.
            if improved {
                since_primal_improve = 0;
            } else {
                since_primal_improve += 1;
            }
            if iter >= PRIMAL_WARMUP && since_primal_improve >= PRIMAL_STALL_LIMIT {
                debug!(
                    "{ind}[lagr] primal converged at iter={} (best={}, {} stalled iters)",
                    iter, best_components, since_primal_improve
                );
                break;
            }

            if iter <= 5 || iter.is_multiple_of(25) || improved {
                debug!(
                    "{ind}[lagr] iter={} pool={} +{} lb={:.1} lambda={:.4} best={} gap={:.1}% t={:.1}s",
                    iter,
                    pool.len(),
                    added,
                    lb,
                    lambda,
                    best_components,
                    if lb > 0.0 {
                        100.0 * (best_components as f64 - lb) / lb
                    } else {
                        f64::NAN
                    },
                    start.elapsed().as_secs_f64(),
                );
            }

            // Terminate ONLY when the optimum is proven: global pricing with
            // the complete column set (added == 0) gives a valid LB, and the
            // incumbent meets it. (OPT ≥ ⌈lb⌉ and best ≥ OPT, so best ≤ ⌈lb⌉
            // ⇒ best = OPT.) Windowed pricing never certifies.
            if global && added == 0 && best_components <= lb.ceil() as usize {
                debug!(
                    "{ind}[lagr] PROVED optimal at iter={}: best={} lb={:.2}",
                    iter, best_components, lb
                );
                proved = true;
                break;
            }

            // Otherwise keep using the budget. When the subgradient has settled
            // (λ tiny) but the optimum isn't proven, re-energise: reset the step
            // and pull α back toward 1 so leaves the pricer had pruned (α→0)
            // re-enter column generation — diversifies pool + packing.
            if added == 0 {
                no_new += 1;
            } else {
                no_new = 0;
            }
            if no_new >= 25 && lambda < 1e-3 {
                // The dual has settled without proving optimality. Re-energise
                // to escape the plateau — but bound the attempts: if several
                // consecutive re-energises yield no primal improvement, the
                // subgradient has converged to its best and we hand off to the
                // local search. This is what lets the loop terminate on its own
                // (no deadline needed) instead of spinning until SIGTERM.
                if best_components < best_at_reenergise {
                    reenergise_dry = 0;
                } else {
                    reenergise_dry += 1;
                }
                best_at_reenergise = best_components;
                if reenergise_dry >= REENERGISE_DRY_LIMIT {
                    debug!(
                        "{ind}[lagr] subgradient converged at iter={} (best={}, {} dry re-energises)",
                        iter, best_components, reenergise_dry
                    );
                    break;
                }
                lambda = 1.0;
                for a in alpha.iter_mut().skip(1) {
                    *a = 0.5 * *a + 0.5;
                }
                no_new = 0;
                debug!(
                    "{ind}[lagr] re-energise at iter={} (unproven, best={})",
                    iter, best_components
                );
            }
        }

        if profile {
            let ms = |d: Duration| d.as_secs_f64() * 1000.0;
            debug!(
                "{ind}[lagr][profile] iters={} pool={} | price={:.0}ms add={:.0}ms score={:.0}ms sg={:.0}ms primal={:.0}ms | win_priced={} win_skipped={}",
                iter,
                pool.len(),
                ms(t_price),
                ms(t_add),
                ms(t_score),
                ms(t_sg),
                ms(t_primal),
                win_priced_total,
                win_skipped_total,
            );
        }

        // ---- Branching-lite (prototype, gated by `LagrangianConfig.branch`) ----
        // When the bound can't prove the incumbent (LP↔IP integrality gap), the
        // unconstrained pricer never generates the columns the optimum needs.
        // Branch on contended leaf-pairs: force {a,b} together (must-link),
        // RE-PRICE under that constraint so the anchor DP emits {a,b}-together
        // columns (the gap columns), then re-pack. Keep any improvement.
        if global && !proved && self.config.branch {
            // Incumbent leaf → component map.
            let mut comp_of = vec![usize::MAX; nl + 1];
            for (ci, comp) in best_forest.iter().enumerate() {
                for l in comp.leaves() {
                    if (l as usize) <= nl {
                        comp_of[l as usize] = ci;
                    }
                }
            }
            // Candidate branch pairs: from the highest-score columns, the first
            // leaf-pair the incumbent currently splits across components.
            let fscores: Vec<f64> = pool.iter().map(|b| block_score(b, &alpha, &beta)).collect();
            let mut order: Vec<usize> = (0..pool.len()).collect();
            order.sort_unstable_by(|&i, &j| fscores[j].total_cmp(&fscores[i]));
            let mut pairs: Vec<(u32, u32)> = Vec::new();
            let mut seen_pairs: FxHashSet<(u32, u32)> = FxHashSet::default();
            for &i in &order {
                if pairs.len() >= 16 {
                    break;
                }
                let lbls = &pool[i].labels;
                'find: for wi in 0..lbls.len() {
                    for wj in (wi + 1)..lbls.len() {
                        let (a, b) = (lbls[wi], lbls[wj]);
                        if comp_of[a as usize] != comp_of[b as usize] && seen_pairs.insert((a, b)) {
                            pairs.push((a, b));
                            break 'find;
                        }
                    }
                }
            }
            let n_pairs = pairs.len();
            for (a, b) in pairs {
                if self.terminate.load(Ordering::Relaxed)
                    || deadline.is_some_and(|d| Instant::now() >= d)
                {
                    break;
                }
                let mut br = Branchings::default();
                br.push_must_link(LeafPair::new(a, b));
                // Encourage the constrained pricer to want {a,b}.
                let mut a2 = alpha.clone();
                a2[a as usize] = a2[a as usize].max(2.0);
                a2[b as usize] = a2[b as usize].max(2.0);
                for _ in 0..20 {
                    let ctx = PricingContext {
                        trees,
                        num_leaves: nl,
                        alpha: &a2,
                        beta: &beta,
                        columns: &[],
                        seen: &seen,
                        branchings: &br,
                        terminate: self.terminate.as_ref(),
                        deadline: None,
                    };
                    let mut got: Vec<Vec<u32>> = Vec::new();
                    if let PricingResult::Found(cols) = pricer.price(&ctx, &mut scratch) {
                        for c in cols {
                            got.push(c.labels().to_vec());
                        }
                    }
                    let mut any = false;
                    for c in got {
                        if add_block(c, &mut pool, &mut seen) {
                            any = true;
                        }
                    }
                    if !any {
                        break;
                    }
                }
                // Re-pack honouring the must-link (filter columns that split a|b).
                let sc2: Vec<f64> = pool.iter().map(|bk| block_score(bk, &a2, &beta)).collect();
                let (comps, sel) = greedy_pack(&pool, &sc2, trees, n, &br);
                if comps < best_components {
                    best_components = comps;
                    best_forest = build_forest(&pool, &sel, trees, n, unindexed);
                    debug!(
                        "{ind}[lagr] branch must-link({},{}) improved: best={}",
                        a, b, comps
                    );
                }
            }
            debug!(
                "{ind}[lagr] branching-lite: tried {} pairs, best={} pool={}",
                n_pairs,
                best_components,
                pool.len()
            );
        }

        // Freeze the lower-track early-abort target for the primal-improvement
        // phases. The subgradient's dual bound is final now, so any reduced
        // forest reaching `t_orig - delta` components satisfies the track bound
        // once expanded. `improve_packing`/`lns_improve` watch `abort_k_reduced`
        // and stop the instant they hit it (claims the speed bonus).
        if self.abort_armed.load(Ordering::Relaxed)
            && let Some((a, b)) = self.approx_target
        {
            let delta = self.param_reduction.load(Ordering::Relaxed);
            let lb_orig = best_lb.ceil().max(0.0) as usize + delta;
            let t_orig = (a * lb_orig as f64).floor() as usize + b;
            self.abort_k_reduced
                .store(t_orig.saturating_sub(delta), Ordering::Relaxed);
            if best_components <= t_orig.saturating_sub(delta) {
                self.terminate.store(true, Ordering::Relaxed);
            }
        }

        // ---- Primal improvement loop (iterated local search over the pool) ----
        // The subgradient has converged; now relentlessly re-select a better
        // node-disjoint packing over the columns it generated (the pool already
        // contains the optimum's columns). Runs until its own stall (or the
        // deadline/SIGTERM) — see improve_packing.
        if ls_on && !proved && !self.terminate.load(Ordering::Relaxed) {
            let scores: Vec<f64> = pool.iter().map(|b| block_score(b, &alpha, &beta)).collect();
            // Seed from the subgradient's BEST incumbent (best over all
            // iterations), not a final-dual greedy — the local search refines
            // the strongest packing found, not a one-shot one.
            let sel1 = self.improve_packing(&pool, trees, n, &scores, &best_sel, deadline, start);
            let savings1: usize = sel1.iter().map(|&i| pool[i].labels.len() - 1).sum();
            let k1 = nl - savings1;
            if k1 < best_components {
                best_components = k1;
                best_forest = build_forest(&pool, &sel1, trees, n, unindexed);
            }
        }

        debug!(
            "{ind}[lagr] DONE reduced_n={} reduced_best={} lb={:.1} iters={} pool={} t={:.1}s",
            n,
            best_components,
            best_lb,
            iter,
            pool.len(),
            start.elapsed().as_secs_f64()
        );
        // Publish the top-level reduced dual bound so `solve` can expose a tight
        // lower bound. Only depth 0 covers the whole reduced instance; nested
        // cluster solves (depth ≥ 1) bound only their slice.
        if depth == 0 {
            self.reduced_dual_lb
                .store(best_lb.ceil().max(0.0) as usize, Ordering::Relaxed);
        }
        (best_forest, proved)
    }

    /// Cluster-decomposition driver. Single recursive pass (so the exact Whidden
    /// `−1` anchor-merge recombination stays self-consistent — a two-pass
    /// enumerate/replay desyncs because the anchor merge is solution-dependent).
    /// Budget is shared via leftover-forwarding: a running `remaining` leaf
    /// counter; each leaf cluster gets its leaf-share of the time still left, and
    /// the (fast) exact clusters return their slack to the cores.
    fn solve_decomposed(
        &self,
        reduced: &Instance,
        budget: Option<Duration>,
        start: Instant,
    ) -> (Vec<Tree>, bool) {
        // No internal time horizon by default: the solver is SIGTERM-driven and
        // each cluster runs to its own convergence/stall. A budget is supplied
        // only when `cfg.budget` is set (the run-track wall limit, or the
        // lower-track racer), in which case it is sliced across clusters by
        // leaf-share.
        let plan_deadline: Option<Instant> = budget.map(|b| start + b);
        let remaining = Cell::new(reduced.num_leaves as u64);
        // Tracks whether any cluster finished UNproven (gap or time-capped). The
        // whole instance is proven optimal only if every cluster is.
        let unproved = Cell::new(false);
        let forest = self.solve_cluster(reduced, &remaining, plan_deadline, 0, &unproved);
        (forest, !unproved.get())
    }

    /// Recursively split `sub`; solve each irreducible leaf cluster by size (exact
    /// B&P ≤ threshold, else the anytime cascade) with a leftover-forwarding time
    /// slice. Always returns a valid forest over `sub`'s labels. `depth` is the
    /// decomposition recursion level, used only to indent trace output.
    fn solve_cluster(
        &self,
        sub: &Instance,
        remaining: &Cell<u64>,
        plan_deadline: Option<Instant>,
        depth: usize,
        unproved: &Cell<bool>,
    ) -> Vec<Tree> {
        let ind = "  ".repeat(depth);
        if sub.num_leaves <= 1 {
            return if sub.num_leaves == 0 {
                Vec::new()
            } else {
                vec![sub.trees[0].clone()]
            };
        }
        // On termination, STOP decomposing. The Whidden cluster recursion is
        // itself uninterruptible (a single deep decomposition can take many
        // seconds on a giant instance), so without this guard a SIGTERM landing
        // mid-decomposition leaves the solver unable to emit ANY forest before
        // the harness SIGKILLs it (the cause of "no-response" timeouts on the
        // largest instances). Since the recursion descends through this method,
        // bailing here on every level makes the whole tree unwind within one
        // bounded decomposition step. solve_reduced_core with an elapsed
        // deadline returns this sub's Chen incumbent immediately — a valid (if
        // unrefined) forest for the not-yet-reached part of the instance.
        // Bail the recursion on SIGTERM *or* when the decomposition attempt's
        // budget (`plan_deadline`) is spent. The latter is essential: on a
        // slow-proving instance the split recursion can grind for the entire
        // 300 s and emit only the Chen baseline, when the flat engine would do
        // far better with that time (n=10037: decomp→5000 vs flat→4277). The
        // budget was previously used only to time-SLICE clusters, never to STOP
        // the recursion, so the 25 s attempt overran indefinitely. Bailing here
        // sets `unproved`, so `solve()` discards the partial decomposition and
        // hands the whole core to the flat engine for the remaining time.
        if self.terminate.load(Ordering::Relaxed)
            || plan_deadline.is_some_and(|d| Instant::now() >= d)
        {
            unproved.set(true);
            let now = Instant::now();
            return self.solve_reduced_core(sub, Some(now), now, depth).0;
        }
        if sub.num_trees() == 2 && sub.num_leaves >= DECOMP_MIN_LEAVES {
            debug!("{ind}[lagr][decomp] split n={}", sub.num_leaves);
            let mut cb = |s: &Instance| -> Option<Vec<Tree>> {
                Some(self.solve_cluster(s, remaining, plan_deadline, depth + 1, unproved))
            };
            if let Some(forest) = try_whidden_decomp_2tree(sub, &mut cb, &self.terminate) {
                return forest;
            }
        }

        // Irreducible leaf cluster. With a testing horizon, give it its
        // leaf-share of the time still left; otherwise (the default) it runs to
        // its own convergence/stall (deadline = None).
        let now = Instant::now();
        let rem = remaining.get().max(1);
        remaining.set(rem.saturating_sub(sub.num_leaves as u64));
        let slice_end: Option<Instant> = plan_deadline.map(|pd| {
            let avail = pd.saturating_duration_since(now);
            let dur = avail.mul_f64((sub.num_leaves as f64 / rem as f64).min(1.0));
            (now + dur).min(pd)
        });

        // Probe with exact B&P (only small clusters, short cap): if it FINISHES
        // (proves optimal) use it; otherwise fall to the anytime cascade. A
        // capped B&P returns garbage, so solve_cluster_exact returns None unless
        // it truly finished. The cap is a bounded *attempt*, not a phase
        // timeout — it caps wasted effort on a cluster exact B&P can't crack.
        let exact_threshold = self.config.decomp_exact as u32;
        if sub.num_leaves <= exact_threshold && !self.terminate.load(Ordering::Relaxed) {
            let exact_cap = Duration::from_millis(self.config.decomp_exact_cap_ms);
            let probe = match slice_end {
                Some(se) => (Instant::now() + exact_cap).min(se),
                None => Instant::now() + exact_cap,
            };
            if let Some(forest) = self.solve_cluster_exact(sub, probe) {
                debug!(
                    "{ind}[lagr][decomp] exact n={} k={} (optimal)",
                    sub.num_leaves,
                    forest.len()
                );
                return forest;
            }
        }
        let slice_s =
            slice_end.map(|se| se.saturating_duration_since(Instant::now()).as_secs_f64());
        debug!(
            "{ind}[lagr][decomp] cascade n={} slice={}",
            sub.num_leaves,
            match slice_s {
                Some(s) => format!("{s:.1}s"),
                None => "converge".to_string(),
            }
        );
        let (forest, proved) = self.solve_reduced_core(sub, slice_end, Instant::now(), depth);
        if !proved {
            unproved.set(true);
        }
        forest
    }

    /// Exact B&P on a cluster, wall-capped by a watchdog. Returns `Some` ONLY if
    /// B&P FINISHED (proved optimal) before the cap — a time-capped B&P returns a
    /// garbage near-Chen incumbent, which must NOT be trusted (the caller then
    /// solves the cluster with the anytime cascade instead). The pricer now polls
    /// the shared `term` flag, so a SIGTERM aborts the inner DP promptly.
    fn solve_cluster_exact(&self, sub: &Instance, deadline: Instant) -> Option<Vec<Tree>> {
        // The exact B&P polls the deadline itself (no watchdog thread): the old
        // watchdog slept on a poll and was `join()`ed after each probe, idling
        // the main thread up to one poll interval per cluster probe — on
        // decomp-heavy instances that idled a large fraction of the budget.
        let res = crate::solvers::bp::bp_solve_capped_until(sub, &self.terminate, Some(deadline));
        // A capped/cancelled solve is NOT a proven optimum (a time-capped B&P
        // returns a near-Chen incumbent the caller must not trust); discard it.
        if self.terminate.load(Ordering::Relaxed) || Instant::now() >= deadline {
            return None;
        }
        res
    }

    /// One Lagrangian B&B node: run the subgradient under `branchings` until
    /// `slice_deadline`, growing the shared pool. Constraint-aware pricing emits
    /// the columns the constraints demand (the "gap columns" the flat dual never
    /// generates). Returns the best packing found here: (#components, selected
    /// column indices, best Lagrangian bound).
    fn subgradient_slice(
        &self,
        trees: &[Tree],
        n: u32,
        nl: usize,
        pool: &mut Vec<Block>,
        seen: &mut ColumnSet,
        alpha: &mut [f64],
        beta: &mut Vec<Vec<f64>>,
        branchings: &Branchings,
        pricer: &mut ExactPairDpPricer,
        scratch: &mut PricerScratch,
        slice_deadline: Instant,
        ub: usize,
    ) -> (usize, Vec<usize>, f64) {
        let mut lambda = 1.0f64;
        let mut best_lb = 0.0f64;
        let mut stall = 0usize;
        let mut scores: Vec<f64> = pool.iter().map(|b| block_score(b, alpha, beta)).collect();
        let (mut best_k, mut best_sel) = greedy_pack(pool, &scores, trees, n, branchings);
        loop {
            if self.terminate.load(Ordering::Relaxed) || Instant::now() >= slice_deadline {
                break;
            }
            // Constraint-aware pricing. BOOST the duals of must-linked leaves so
            // the pricer actually emits their joint "gap" column — at the plain
            // duals it has negative reduced cost and is never generated, which
            // starves the must-link branch. The boost is for pricing only; the
            // subgradient step and scoring use the real duals.
            let mut price_alpha = alpha.to_vec();
            for pair in branchings.must_link() {
                if (pair.a as usize) <= nl {
                    price_alpha[pair.a as usize] = price_alpha[pair.a as usize].max(2.0);
                }
                if (pair.b as usize) <= nl {
                    price_alpha[pair.b as usize] = price_alpha[pair.b as usize].max(2.0);
                }
            }
            let mut newc: Vec<Vec<u32>> = Vec::new();
            {
                let ctx = PricingContext {
                    trees,
                    num_leaves: nl,
                    alpha: price_alpha.as_slice(),
                    beta: beta.as_slice(),
                    columns: &[],
                    seen,
                    branchings,
                    terminate: self.terminate.as_ref(),
                    deadline: None,
                };
                for col in scratch.drain_reserve(&ctx, 64) {
                    newc.push(col.labels().to_vec());
                }
                if let PricingResult::Found(cols) = pricer.price(&ctx, scratch) {
                    for c in cols {
                        newc.push(c.labels().to_vec());
                    }
                }
            }
            for c in newc {
                let mut l = c;
                l.sort_unstable();
                l.dedup();
                if l.len() >= 2
                    && !seen.contains(&l)
                    && is_valid_af_component(&l, trees)
                    && let Some(b) = make_block(trees, l.clone())
                {
                    seen.insert(l);
                    pool.push(b);
                }
            }
            scores = pool.iter().map(|b| block_score(b, alpha, beta)).collect();
            let lb = self.subgradient_step(trees, nl, pool, &scores, alpha, beta, lambda, ub);
            if lb > best_lb + 1e-6 {
                best_lb = lb;
                stall = 0;
            } else {
                stall += 1;
                if stall >= 20 {
                    lambda *= 0.5;
                    stall = 0;
                }
            }
            // Re-energise when the step collapses (escape the dual plateau):
            // reset λ and pull α halfway back to 1. Without this the condensed
            // slice stalls ~3 components short of the full flat subgradient.
            if lambda < 1.0e-3 {
                lambda = 1.0;
                for a in alpha.iter_mut().skip(1) {
                    *a = 0.5 * *a + 0.5;
                }
            }
            let scores2: Vec<f64> = pool.iter().map(|b| block_score(b, alpha, beta)).collect();
            let (k, sel) = greedy_pack(pool, &scores2, trees, n, branchings);
            if k < best_k {
                best_k = k;
                best_sel = sel;
            }
        }
        (best_k, best_sel, best_lb)
    }

    /// Anytime Lagrangian branch-and-bound. DFS on must-link/cannot-link leaf
    /// pairs; each node is a subgradient slice under its constraints (the fast
    /// dual — no LP). The root converges the flat dual; each branch forces a
    /// contended pair together so the pricer emits the gap columns the optimum
    /// needs. Keeps the global best incumbent, returns it at the deadline.
    fn solve_lagrangian_bnb(
        &self,
        reduced: &Instance,
        deadline: Option<Instant>,
        start: Instant,
    ) -> Vec<Tree> {
        let trees = &reduced.trees;
        let n = reduced.num_leaves;
        let nl = n as usize;

        // ---- Pool + warm start (Chen 2-approx + greedy seeds) ----
        let mut pool: Vec<Block> = Vec::new();
        let mut seen = ColumnSet::new();
        let add = |labels: Vec<u32>, pool: &mut Vec<Block>, seen: &mut ColumnSet| {
            let mut l = labels;
            l.sort_unstable();
            l.dedup();
            if l.len() < 2 || seen.contains(&l) || !is_valid_af_component(&l, trees) {
                return;
            }
            if let Some(b) = make_block(trees, l.clone()) {
                seen.insert(l);
                pool.push(b);
            }
        };
        let (_, _, chen_sets) = chen_pair_agreement(&trees[0], &trees[1]);
        let mut global_best_forest = forest_from_partition(&chen_sets, trees, n, false);
        let mut global_best_k = global_best_forest.len();
        for s in &chen_sets {
            add(s.clone(), &mut pool, &mut seen);
        }
        let num_seeds: u64 = if n <= 2_000 {
            12
        } else if n <= 6_000 {
            5
        } else {
            2
        };
        for ref_idx in 0..2usize {
            for seed in 0..num_seeds {
                if self.terminate.load(Ordering::Relaxed) {
                    break;
                }
                let (_k, part) =
                    klados_core::lower_bound::greedy_multi_tree_partition(trees, ref_idx, seed);
                for g in groups_from_partition(&part, nl) {
                    add(g, &mut pool, &mut seen);
                }
            }
        }

        let mut pricer = ExactPairDpPricer::new(trees);
        let mut scratch = PricerScratch::new(trees);
        let alpha0: Vec<f64> = (0..=nl).map(|i| if i == 0 { 0.0 } else { 1.0 }).collect();
        let beta0: Vec<Vec<f64>> = trees.iter().map(|t| vec![0.0f64; t.num_nodes()]).collect();

        let plan_deadline = deadline.unwrap_or(start + Duration::from_millis(self.config.plan_ms));
        let root_ms = self.config.lbnb_root_ms;
        let node_ms = self.config.lbnb_node_ms;

        // DFS stack: (branchings, warm duals). Children warm-start from the
        // parent's converged α/β.
        let mut stack: Vec<(Branchings, Vec<f64>, Vec<Vec<f64>>)> =
            vec![(Branchings::default(), alpha0, beta0)];
        let mut nodes = 0usize;
        while let Some((br, mut alpha, mut beta)) = stack.pop() {
            if self.terminate.load(Ordering::Relaxed) || Instant::now() >= plan_deadline {
                break;
            }
            let slice_ms = if nodes == 0 { root_ms } else { node_ms };
            let slice_end = (Instant::now() + Duration::from_millis(slice_ms)).min(plan_deadline);
            let (k, sel, lb) = self.subgradient_slice(
                trees,
                n,
                nl,
                &mut pool,
                &mut seen,
                &mut alpha,
                &mut beta,
                &br,
                &mut pricer,
                &mut scratch,
                slice_end,
                global_best_k,
            );
            if k < global_best_k {
                global_best_k = k;
                global_best_forest = build_forest(&pool, &sel, trees, n, false);
                debug!(
                    "[lagr][lbnb] node={} depth={} k={} (best) t={:.1}s",
                    nodes,
                    br.depth(),
                    global_best_k,
                    start.elapsed().as_secs_f64()
                );
            }
            // The gap columns this branch generated are valid GLOBALLY (a column
            // is a valid AF component regardless of branchings). Pack the
            // enriched pool UNCONSTRAINED too — that's where branching pays off:
            // it feeds the columns the flat dual never produced into the global
            // packing.
            {
                let uscores: Vec<f64> =
                    pool.iter().map(|b| block_score(b, &alpha, &beta)).collect();
                let (uk, usel) = greedy_pack(&pool, &uscores, trees, n, &Branchings::default());
                if uk < global_best_k {
                    global_best_k = uk;
                    global_best_forest = build_forest(&pool, &usel, trees, n, false);
                    debug!(
                        "[lagr][lbnb] node={} (unconstrained pack) k={} t={:.1}s",
                        nodes,
                        global_best_k,
                        start.elapsed().as_secs_f64()
                    );
                }
            }
            nodes += 1;
            let pool_before = pool.len();

            // Prune only at the root: with empty branchings the pool-wide
            // Lagrangian bound is valid for the whole problem. Under branchings
            // the same sum isn't a valid node bound, so we don't prune there
            // (anytime keeps the best regardless).
            if br.depth() == 0 && (lb - 1e-6).ceil() as usize >= global_best_k {
                debug!(
                    "[lagr][lbnb] root certified: k={} lb={:.1}",
                    global_best_k, lb
                );
                break;
            }

            // Branch on the most-contended pair the node's incumbent splits.
            let bp = pick_branch_pair(&pool, &sel, &alpha, &beta, nl, &br);
            debug!(
                "[lagr][lbnb]   node={} depth={} k={} lb={:.1} pool {}→{} pair={:?}",
                nodes - 1,
                br.depth(),
                k,
                lb,
                pool_before,
                pool.len(),
                bp.map(|p| (p.a, p.b))
            );
            if let Some(pair) = bp {
                let (left, right) = br.split_on(pair);
                if !right.is_inconsistent() {
                    stack.push((right, alpha.clone(), beta.clone()));
                }
                if !left.is_inconsistent() {
                    stack.push((left, alpha, beta)); // must-link explored first
                }
            }
        }
        debug!(
            "[lagr][lbnb] done nodes={} k={} pool={} t={:.1}s",
            nodes,
            global_best_k,
            pool.len(),
            start.elapsed().as_secs_f64()
        );
        global_best_forest
    }

    /// One subgradient iteration for the dive: price at the current duals (under
    /// no branching), bank columns into the pool, take a subgradient step, then
    /// FREEZE the duals of already-covered leaves at 0 so the residual dual
    /// re-approximates only what's left to cover. Returns the Lagrangian bound.
    fn dive_sg_iter(
        &self,
        trees: &[Tree],
        nl: usize,
        pool: &mut Vec<Block>,
        seen: &mut ColumnSet,
        alpha: &mut Vec<f64>,
        beta: &mut Vec<Vec<f64>>,
        pricer: &mut ExactPairDpPricer,
        scratch: &mut PricerScratch,
        lambda: f64,
        ub: usize,
        covered: &FixedBitSet,
    ) -> f64 {
        let branchings = Branchings::default();
        let mut newc: Vec<Vec<u32>> = Vec::new();
        {
            let ctx = PricingContext {
                trees,
                num_leaves: nl,
                alpha: alpha.as_slice(),
                beta: beta.as_slice(),
                columns: &[],
                seen,
                branchings: &branchings,
                terminate: self.terminate.as_ref(),
                deadline: None,
            };
            for col in scratch.drain_reserve(&ctx, 64) {
                newc.push(col.labels().to_vec());
            }
            if let PricingResult::Found(cols) = pricer.price(&ctx, scratch) {
                for c in cols {
                    newc.push(c.labels().to_vec());
                }
            }
        }
        for c in newc {
            let mut l = c;
            l.sort_unstable();
            l.dedup();
            if l.len() >= 2
                && !seen.contains(&l)
                && is_valid_af_component(&l, trees)
                && let Some(b) = make_block(trees, l.clone())
            {
                seen.insert(l);
                pool.push(b);
            }
        }
        let scores: Vec<f64> = pool.iter().map(|b| block_score(b, alpha, beta)).collect();
        let lb = self.subgradient_step(trees, nl, pool, &scores, alpha, beta, lambda, ub);
        for l in covered.ones() {
            if l <= nl {
                alpha[l] = 0.0;
            }
        }
        lb
    }

    /// Lagrangian dive. Warm the dual, then repeatedly: commit the best-scored
    /// column that still fits (uncovered leaves, unused nodes), and run a few
    /// subgradient steps with the covered leaves frozen so the dual sharpens for
    /// the residual. Each commit subdivides the space; the partial forest
    /// (committed columns + singletons) is the anytime incumbent.
    fn solve_lagrangian_dive(
        &self,
        reduced: &Instance,
        deadline: Option<Instant>,
        start: Instant,
    ) -> Vec<Tree> {
        let trees = &reduced.trees;
        let n = reduced.num_leaves;
        let nl = n as usize;

        // ---- Pool + warm start ----
        let mut pool: Vec<Block> = Vec::new();
        let mut seen = ColumnSet::new();
        let add = |labels: Vec<u32>, pool: &mut Vec<Block>, seen: &mut ColumnSet| {
            let mut l = labels;
            l.sort_unstable();
            l.dedup();
            if l.len() < 2 || seen.contains(&l) || !is_valid_af_component(&l, trees) {
                return;
            }
            if let Some(b) = make_block(trees, l.clone()) {
                seen.insert(l);
                pool.push(b);
            }
        };
        let (_, _, chen_sets) = chen_pair_agreement(&trees[0], &trees[1]);
        let mut best_forest = forest_from_partition(&chen_sets, trees, n, false);
        let mut best_k = best_forest.len();
        for s in &chen_sets {
            add(s.clone(), &mut pool, &mut seen);
        }
        let num_seeds: u64 = if n <= 2_000 {
            12
        } else if n <= 6_000 {
            5
        } else {
            2
        };
        for ref_idx in 0..2usize {
            for seed in 0..num_seeds {
                if self.terminate.load(Ordering::Relaxed) {
                    break;
                }
                let (_k, part) =
                    klados_core::lower_bound::greedy_multi_tree_partition(trees, ref_idx, seed);
                for g in groups_from_partition(&part, nl) {
                    add(g, &mut pool, &mut seen);
                }
            }
        }

        let mut pricer = ExactPairDpPricer::new(trees);
        let mut scratch = PricerScratch::new(trees);
        let mut alpha: Vec<f64> = (0..=nl).map(|i| if i == 0 { 0.0 } else { 1.0 }).collect();
        let mut beta: Vec<Vec<f64>> = trees.iter().map(|t| vec![0.0f64; t.num_nodes()]).collect();

        let plan_deadline = deadline.unwrap_or(start + Duration::from_millis(self.config.plan_ms));
        let reopt_iters: usize = self.config.dive_reopt;
        // Warm-up gets a fraction of the budget; the dive uses the rest.
        let warmup_frac: f64 = self.config.dive_warmup;

        let empty = FixedBitSet::with_capacity(nl + 1);
        let total = plan_deadline.saturating_duration_since(Instant::now());
        let warmup_end = (Instant::now() + total.mul_f64(warmup_frac)).min(plan_deadline);

        // ---- Warm-up: converge the dual, keep the greedy incumbent ----
        let mut lambda = 1.0f64;
        let mut best_lb = 0.0f64;
        let mut stall = 0usize;
        while !self.terminate.load(Ordering::Relaxed) && Instant::now() < warmup_end {
            let lb = self.dive_sg_iter(
                trees,
                nl,
                &mut pool,
                &mut seen,
                &mut alpha,
                &mut beta,
                &mut pricer,
                &mut scratch,
                lambda,
                best_k,
                &empty,
            );
            if lb > best_lb + 1e-6 {
                best_lb = lb;
                stall = 0;
            } else {
                stall += 1;
                if stall >= 20 {
                    lambda *= 0.5;
                    stall = 0;
                }
            }
            let scores: Vec<f64> = pool.iter().map(|b| block_score(b, &alpha, &beta)).collect();
            let (k, sel) = greedy_pack(&pool, &scores, trees, n, &Branchings::default());
            if k < best_k {
                best_k = k;
                best_forest = build_forest(&pool, &sel, trees, n, false);
            }
        }
        debug!(
            "[lagr][dive] warm-up done: greedy_best={} pool={} t={:.1}s",
            best_k,
            pool.len(),
            start.elapsed().as_secs_f64()
        );

        // ---- Dive: commit best-fitting column, re-approximate the residual ----
        let mut committed: Vec<usize> = Vec::new();
        let mut used: Vec<FixedBitSet> = trees
            .iter()
            .map(|t| FixedBitSet::with_capacity(t.num_nodes()))
            .collect();
        let mut covered = FixedBitSet::with_capacity(nl + 1);
        loop {
            if self.terminate.load(Ordering::Relaxed) || Instant::now() >= plan_deadline {
                break;
            }
            // Residual dual re-approximation (covered leaves frozen).
            for _ in 0..reopt_iters {
                if Instant::now() >= plan_deadline {
                    break;
                }
                self.dive_sg_iter(
                    trees,
                    nl,
                    &mut pool,
                    &mut seen,
                    &mut alpha,
                    &mut beta,
                    &mut pricer,
                    &mut scratch,
                    lambda,
                    best_k,
                    &covered,
                );
            }
            // Best dual-scored column that still fits. Any multi-leaf column
            // saves |labels|−1 components, so we commit the highest-scored
            // *fitting* one regardless of sign — the dual only sets the order.
            let scores: Vec<f64> = pool.iter().map(|b| block_score(b, &alpha, &beta)).collect();
            let mut best_i: Option<usize> = None;
            let mut best_s = f64::NEG_INFINITY;
            'col: for i in 0..pool.len() {
                if scores[i] <= best_s || pool[i].labels.len() < 2 {
                    continue;
                }
                let b = &pool[i];
                for &l in &b.labels {
                    if covered.contains(l as usize) {
                        continue 'col;
                    }
                }
                for (t, nodes) in b.cover.iter().enumerate() {
                    for &v in nodes {
                        if used[t].contains(v as usize) {
                            continue 'col;
                        }
                    }
                }
                best_s = scores[i];
                best_i = Some(i);
            }
            match best_i {
                Some(i) => {
                    committed.push(i);
                    for &l in &pool[i].labels {
                        covered.insert(l as usize);
                    }
                    for (t, nodes) in pool[i].cover.iter().enumerate() {
                        for &v in nodes {
                            used[t].insert(v as usize);
                        }
                    }
                    let k = committed.len() + (nl - covered.count_ones(..));
                    if k < best_k {
                        best_k = k;
                        best_forest = build_forest(&pool, &committed, trees, n, false);
                        debug!(
                            "[lagr][dive] commit #{} k={} covered={}/{} t={:.1}s",
                            committed.len(),
                            best_k,
                            covered.count_ones(..),
                            nl,
                            start.elapsed().as_secs_f64()
                        );
                    }
                }
                None => break, // nothing improving still fits
            }
        }
        debug!(
            "[lagr][dive] done committed={} k={} t={:.1}s",
            committed.len(),
            best_k,
            start.elapsed().as_secs_f64()
        );
        best_forest
    }

    /// Primal improvement loop (iterated local search) over the Lagrangian pool.
    /// The pool already contains the optimum's columns; the greedy just selected
    /// them sub-optimally. Starting from `init_sel`, repeatedly: (1) **swap** —
    /// for each unselected column whose weight exceeds the weight of the
    /// selected columns it conflicts with, drop those and take it; (2) **re-fill**
    /// — greedily add any column that now fits; (3) on a stalled pass, **perturb**
    /// (restore best, eject a few random columns) to escape the local optimum.
    /// Always-valid, anytime (keeps the best), O(pool) memory. Returns the best
    /// node-disjoint selection found by the deadline.
    fn improve_packing(
        &self,
        pool: &[Block],
        trees: &[Tree],
        n: u32,
        scores: &[f64],
        init_sel: &[usize],
        deadline: Option<Instant>,
        start: Instant,
    ) -> Vec<usize> {
        let nl = n as usize;
        let ncol = pool.len();
        let w = |ci: usize| -> i64 { pool[ci].labels.len() as i64 - 1 };

        let mut leaf_owner = vec![usize::MAX; nl + 1];
        let mut node_owner: Vec<Vec<usize>> = trees
            .iter()
            .map(|t| vec![usize::MAX; t.num_nodes()])
            .collect();
        let mut in_sel = vec![false; ncol];
        let mut savings: i64 = 0;

        // Seed the working state from the initial selection.
        for &ci in init_sel {
            if ci >= ncol || in_sel[ci] {
                continue;
            }
            in_sel[ci] = true;
            savings += w(ci);
            for &l in &pool[ci].labels {
                leaf_owner[l as usize] = ci;
            }
            for (t, nodes) in pool[ci].cover.iter().enumerate() {
                for &v in nodes {
                    node_owner[t][v as usize] = ci;
                }
            }
        }

        // Candidate order: biggest columns first (most impactful), dual score
        // breaks ties.
        let mut order: Vec<usize> = (0..ncol).filter(|&i| pool[i].labels.len() >= 2).collect();
        order.sort_by(|&a, &b| {
            pool[b]
                .labels
                .len()
                .cmp(&pool[a].labels.len())
                .then_with(|| scores[b].total_cmp(&scores[a]))
        });

        let mut best_savings = savings;
        let mut best_in_sel = in_sel.clone();
        let mut conflicts: Vec<usize> = Vec::new();
        let mut rng: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut stalls = 0usize;
        // Perturbation rounds without improvement before the ILS is declared
        // converged. Generous so quality isn't lost, but bounded so the search
        // terminates without a deadline.
        const LS_STALL_LIMIT: usize = 400;
        let timed_out = |s: &Self| {
            s.terminate.load(Ordering::Relaxed) || deadline.is_some_and(|d| Instant::now() >= d)
        };

        while !timed_out(self) {
            // ---- (1) improving swaps ----
            for idx in 0..order.len() {
                if idx % 512 == 0 && timed_out(self) {
                    break;
                }
                let c = order[idx];
                if in_sel[c] {
                    continue;
                }
                conflicts.clear();
                for &l in &pool[c].labels {
                    let o = leaf_owner[l as usize];
                    if o != usize::MAX && !conflicts.contains(&o) {
                        conflicts.push(o);
                    }
                }
                for (t, nodes) in pool[c].cover.iter().enumerate() {
                    for &v in nodes {
                        let o = node_owner[t][v as usize];
                        if o != usize::MAX && !conflicts.contains(&o) {
                            conflicts.push(o);
                        }
                    }
                }
                let cw: i64 = conflicts.iter().map(|&s| w(s)).sum();
                if w(c) - cw <= 0 {
                    continue;
                }
                for &s in &conflicts {
                    in_sel[s] = false;
                    savings -= w(s);
                    for &l in &pool[s].labels {
                        if leaf_owner[l as usize] == s {
                            leaf_owner[l as usize] = usize::MAX;
                        }
                    }
                    for (t, nodes) in pool[s].cover.iter().enumerate() {
                        for &v in nodes {
                            if node_owner[t][v as usize] == s {
                                node_owner[t][v as usize] = usize::MAX;
                            }
                        }
                    }
                }
                in_sel[c] = true;
                savings += w(c);
                for &l in &pool[c].labels {
                    leaf_owner[l as usize] = c;
                }
                for (t, nodes) in pool[c].cover.iter().enumerate() {
                    for &v in nodes {
                        node_owner[t][v as usize] = c;
                    }
                }
            }
            // ---- (2) greedy re-fill of freed space ----
            for &d in &order {
                if in_sel[d] {
                    continue;
                }
                let mut ok = true;
                for &l in &pool[d].labels {
                    if leaf_owner[l as usize] != usize::MAX {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    'nd: for (t, nodes) in pool[d].cover.iter().enumerate() {
                        for &v in nodes {
                            if node_owner[t][v as usize] != usize::MAX {
                                ok = false;
                                break 'nd;
                            }
                        }
                    }
                }
                if ok {
                    in_sel[d] = true;
                    savings += w(d);
                    for &l in &pool[d].labels {
                        leaf_owner[l as usize] = d;
                    }
                    for (t, nodes) in pool[d].cover.iter().enumerate() {
                        for &v in nodes {
                            node_owner[t][v as usize] = d;
                        }
                    }
                }
            }
            // ---- (2b) ejection pass: eject each selected column, re-fill the
            //      freed space, keep only if it nets more savings. This is the
            //      "1-out, ≥2-in" move that escapes the swap local optimum
            //      (a random kick rarely lands it). Cheap snapshot-revert.
            let selected_now: Vec<usize> = (0..ncol).filter(|&i| in_sel[i]).collect();
            for (ei, &s) in selected_now.iter().enumerate() {
                if !in_sel[s] {
                    continue; // already removed by an earlier ejection's re-fill
                }
                if ei % 64 == 0 && timed_out(self) {
                    break;
                }
                let before = savings;
                let snap = in_sel.clone();
                // eject s
                in_sel[s] = false;
                savings -= w(s);
                for &l in &pool[s].labels {
                    if leaf_owner[l as usize] == s {
                        leaf_owner[l as usize] = usize::MAX;
                    }
                }
                for (t, nodes) in pool[s].cover.iter().enumerate() {
                    for &v in nodes {
                        if node_owner[t][v as usize] == s {
                            node_owner[t][v as usize] = usize::MAX;
                        }
                    }
                }
                // re-fill freed space
                for &d in &order {
                    if in_sel[d] {
                        continue;
                    }
                    let mut ok = true;
                    for &l in &pool[d].labels {
                        if leaf_owner[l as usize] != usize::MAX {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        'ej: for (t, nodes) in pool[d].cover.iter().enumerate() {
                            for &v in nodes {
                                if node_owner[t][v as usize] != usize::MAX {
                                    ok = false;
                                    break 'ej;
                                }
                            }
                        }
                    }
                    if ok {
                        in_sel[d] = true;
                        savings += w(d);
                        for &l in &pool[d].labels {
                            leaf_owner[l as usize] = d;
                        }
                        for (t, nodes) in pool[d].cover.iter().enumerate() {
                            for &v in nodes {
                                node_owner[t][v as usize] = d;
                            }
                        }
                    }
                }
                if savings <= before {
                    // Not improving — revert to the snapshot and rebuild owners.
                    in_sel.copy_from_slice(&snap);
                    leaf_owner.iter_mut().for_each(|o| *o = usize::MAX);
                    node_owner
                        .iter_mut()
                        .for_each(|t| t.iter_mut().for_each(|o| *o = usize::MAX));
                    savings = 0;
                    for ci in 0..ncol {
                        if in_sel[ci] {
                            savings += w(ci);
                            for &l in &pool[ci].labels {
                                leaf_owner[l as usize] = ci;
                            }
                            for (t, nodes) in pool[ci].cover.iter().enumerate() {
                                for &v in nodes {
                                    node_owner[t][v as usize] = ci;
                                }
                            }
                        }
                    }
                }
            }
            // ---- (3) keep best, else perturb (iterated local search) ----
            if savings > best_savings {
                best_savings = savings;
                best_in_sel.copy_from_slice(&in_sel);
                stalls = 0;
                let k_reduced = nl - best_savings as usize;
                debug!(
                    "[lagr][ls] improved k={} t={:.1}s",
                    k_reduced,
                    start.elapsed().as_secs_f64()
                );
                // Lower-track early-abort: this packing already clears the track
                // bound (target frozen from the final dual). Stop now for the
                // speed bonus rather than perturbing to the deadline. The MAX
                // sentinel means "no target" — never abort then.
                let target = self.abort_k_reduced.load(Ordering::Relaxed);
                if target != usize::MAX && k_reduced <= target {
                    self.terminate.store(true, Ordering::Relaxed);
                    break;
                }
            } else {
                stalls += 1;
                // Converge after many unproductive perturbations so the local
                // search terminates on its own (no deadline needed). Productive
                // use of any leftover time is the outer refinement's job, not
                // an endlessly-perturbing local search.
                if stalls >= LS_STALL_LIMIT {
                    break;
                }
                // Restore the best, rebuild owners, then eject a few columns.
                in_sel.copy_from_slice(&best_in_sel);
                leaf_owner.iter_mut().for_each(|o| *o = usize::MAX);
                node_owner
                    .iter_mut()
                    .for_each(|t| t.iter_mut().for_each(|o| *o = usize::MAX));
                savings = 0;
                let cur: Vec<usize> = (0..ncol).filter(|&i| in_sel[i]).collect();
                for &ci in &cur {
                    savings += w(ci);
                    for &l in &pool[ci].labels {
                        leaf_owner[l as usize] = ci;
                    }
                    for (t, nodes) in pool[ci].cover.iter().enumerate() {
                        for &v in nodes {
                            node_owner[t][v as usize] = ci;
                        }
                    }
                }
                if cur.is_empty() {
                    break;
                }
                let kick = 2 + (stalls % 6);
                for _ in 0..kick {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let s = cur[(rng as usize) % cur.len()];
                    if in_sel[s] {
                        in_sel[s] = false;
                        savings -= w(s);
                        for &l in &pool[s].labels {
                            if leaf_owner[l as usize] == s {
                                leaf_owner[l as usize] = usize::MAX;
                            }
                        }
                        for (t, nodes) in pool[s].cover.iter().enumerate() {
                            for &v in nodes {
                                if node_owner[t][v as usize] == s {
                                    node_owner[t][v as usize] = usize::MAX;
                                }
                            }
                        }
                    }
                }
            }
        }
        (0..ncol).filter(|&i| best_in_sel[i]).collect()
    }

    /// LNS (large-neighbourhood search): repeatedly pick a *clean* region of the
    /// incumbent — a T₁ subtree whose leaves are exactly the union of some whole
    /// incumbent components — re-solve that region **optimally with B&P** (small
    /// → fast), and splice it back if it validates and has fewer components.
    /// This injects the optimum's columns for the region directly (breaking the
    /// pool cap) and produces new bests, while staying anytime and sound (every
    /// accepted move is a validated AF with fewer components).
    fn lns_improve(
        &self,
        reduced: &Instance,
        mut incumbent: Vec<Tree>,
        deadline: Option<Instant>,
        start: Instant,
    ) -> Vec<Tree> {
        let trees = &reduced.trees;
        let n = reduced.num_leaves;
        let nl = n as usize;
        let region_max: usize = self.config.lns_max;
        let cap = Duration::from_millis(self.config.lns_cap_ms);

        // leaf → incumbent component index
        let mut comp_of = vec![usize::MAX; nl + 1];
        for (ci, c) in incumbent.iter().enumerate() {
            for l in c.leaves() {
                if (l as usize) <= nl {
                    comp_of[l as usize] = ci;
                }
            }
        }

        let mut internal: Vec<(usize, NodeId)> = trees
            .iter()
            .enumerate()
            .flat_map(|(ti, t)| {
                (0..t.num_nodes() as u32).filter_map(move |v| {
                    if t.is_leaf(v) {
                        return None;
                    }
                    let sz = t.subtree_size[v as usize] as usize;
                    (4..=region_max).contains(&sz).then_some((ti, v))
                })
            })
            .collect();
        if internal.is_empty() {
            return incumbent;
        }
        // Try every plausible region once before repeating. The old pure-random
        // sampler could spend much of the tail revisiting the same subtrees; an
        // interleaved large/small order balances high-leverage resolves with
        // cheaper neighborhoods that are more likely to prove inside the cap.
        internal.sort_unstable_by(|&(ta, a), &(tb, b)| {
            trees[tb].subtree_size[b as usize]
                .cmp(&trees[ta].subtree_size[a as usize])
                .then_with(|| ta.cmp(&tb))
                .then_with(|| a.cmp(&b))
        });
        let mut ordered = Vec::with_capacity(internal.len());
        let (mut lo, mut hi) = (0usize, internal.len() - 1);
        while lo <= hi {
            ordered.push(internal[lo]);
            lo += 1;
            if lo > hi {
                break;
            }
            ordered.push(internal[hi]);
            hi -= 1;
        }
        internal = ordered;
        let mut cursor = 0usize;
        let (mut tries, mut accepts, mut invalid) = (0usize, 0usize, 0usize);

        while !(self.terminate.load(Ordering::Relaxed)
            || deadline.is_some_and(|d| Instant::now() >= d))
        {
            tries += 1;
            let (ti, v) = internal[cursor];
            let tref = &trees[ti];
            cursor += 1;
            if cursor == internal.len() {
                cursor = 0;
            }

            // Leaves under v in either input tree.
            let mut region: Vec<u32> = Vec::new();
            let mut stack = vec![v];
            while let Some(u) = stack.pop() {
                if tref.is_leaf(u) {
                    region.push(tref.label[u as usize]);
                } else {
                    let (l, r) = tref.children_pair(u);
                    stack.push(l);
                    stack.push(r);
                }
            }
            debug_assert!((4..=region_max).contains(&region.len()));
            let mut in_region = vec![false; nl + 1];
            for &l in &region {
                in_region[l as usize] = true;
            }
            // Components touched by the region; require a CLEAN cut (every
            // touched component is fully inside) so the splice can't overlap
            // the rest.
            let mut touched: Vec<usize> = Vec::new();
            for &l in &region {
                let ci = comp_of[l as usize];
                if ci != usize::MAX && !touched.contains(&ci) {
                    touched.push(ci);
                }
            }
            let mut clean = true;
            let mut l_leaves: Vec<u32> = Vec::new();
            for &ci in &touched {
                let mut inside = true;
                for l in incumbent[ci].leaves() {
                    if (l as usize) > nl || !in_region[l as usize] {
                        inside = false;
                        break;
                    }
                    l_leaves.push(l);
                }
                if !inside {
                    clean = false;
                    break;
                }
            }
            if !clean || touched.len() < 2 || l_leaves.len() < 4 {
                continue;
            }

            // Build the sub-instance (T₁|L, T₂|L) with a compact relabel.
            l_leaves.sort_unstable();
            l_leaves.dedup();
            let m = l_leaves.len();
            let mut orig_to_sub = vec![0u32; nl + 1];
            let mut sub_to_orig = vec![0u32; m + 1];
            for (i, &l) in l_leaves.iter().enumerate() {
                orig_to_sub[l as usize] = (i + 1) as u32;
                sub_to_orig[i + 1] = l;
            }
            let sub_inst = Instance::new(
                vec![
                    trees[0].relabel(&orig_to_sub, m as u32),
                    trees[1].relabel(&orig_to_sub, m as u32),
                ],
                m as u32,
            );

            // Re-solve the region optimally with B&P (capped).
            let sub_sol = match self.solve_cluster_exact(&sub_inst, Instant::now() + cap) {
                Some(s) => s,
                None => continue, // capped / invalid
            };
            if sub_sol.len() >= touched.len() {
                continue; // no improvement available here
            }

            // Splice: replace the touched components with the decoded re-solve.
            let touched_set: std::collections::HashSet<usize> = touched.iter().copied().collect();
            let mut candidate: Vec<Tree> = incumbent
                .iter()
                .enumerate()
                .filter(|(i, _)| !touched_set.contains(i))
                .map(|(_, c)| c.clone())
                .collect();
            for comp in &sub_sol {
                let mut bs = FixedBitSet::with_capacity(nl + 1);
                for sl in comp.leaves() {
                    bs.insert(sub_to_orig[sl as usize] as usize);
                }
                // LNS runs only at the flat top level; its candidate forest is
                // terminal (validated, then spliced into the returned incumbent),
                // never recombined — so build it compact.
                candidate.push(Tree::forest_component(&bs, &trees[0], n));
            }

            if candidate.len() < incumbent.len()
                && validate_agreement_forest(reduced, &candidate).is_ok()
            {
                incumbent = candidate;
                accepts += 1;
                for (ci, c) in incumbent.iter().enumerate() {
                    for l in c.leaves() {
                        if (l as usize) <= nl {
                            comp_of[l as usize] = ci;
                        }
                    }
                }
                debug!(
                    "[lagr][lns] accept k={} (region={} comps {}→{}) t={:.1}s",
                    incumbent.len(),
                    m,
                    touched.len(),
                    sub_sol.len(),
                    start.elapsed().as_secs_f64()
                );
                // Lower-track early-abort: spliced incumbent now clears the bound.
                // MAX sentinel ⇒ no target; never abort then.
                let target = self.abort_k_reduced.load(Ordering::Relaxed);
                if target != usize::MAX && incumbent.len() <= target {
                    self.terminate.store(true, Ordering::Relaxed);
                    break;
                }
            } else {
                invalid += 1;
            }
        }
        debug!(
            "[lagr][lns] done tries={} accepts={} invalid={} k={} t={:.1}s",
            tries,
            accepts,
            invalid,
            incumbent.len(),
            start.elapsed().as_secs_f64()
        );
        incumbent
    }

    /// Warm-started exact-LP column generation (bp's `Rmp`). Each iteration
    /// solves the restricted-master LP exactly (→ exact duals), lazily
    /// separates node `≤1` rows, prices at those duals, and extracts an
    /// integral primal (MIP at convergence, greedy interim — both validated
    /// node-disjoint). This converges in B&P-class iteration counts.
    /// Returns `(best_forest, proved, pool_labels)`. `pool_labels` are the
    /// multi-leaf columns the RMP tier generated; the subgradient fallback seeds
    /// its pool with them so its column generation does not start cold and
    /// rediscover what the (capped) RMP tier already priced.
    fn solve_rmp(
        &self,
        reduced: &Instance,
        deadline: Option<Instant>,
        start: Instant,
    ) -> (Vec<Tree>, bool, Vec<Vec<u32>>) {
        let trees = &reduced.trees;
        let n = reduced.num_leaves;
        let nl = n as usize;
        let mut proved = false;

        let (_lo, _up, chen_sets) = chen_pair_agreement(&trees[0], &trees[1]);
        let mut best_forest = forest_from_partition(&chen_sets, trees, n, false);
        let mut best_components = best_forest.len();

        let mut builder = ColumnBuilder::new(trees);
        let mut seen = ColumnSet::new();
        let mut pool: Vec<AfColumn> = Vec::new();
        // Singleton columns: the RMP leaf-cover rows are equalities (=1), so a
        // leaf coverable only as a singleton needs its own column for LP
        // feasibility. (Ignored when building the forest — uncovered ⇒ singleton.)
        for l in 1..=n {
            if let Some(c) = builder.try_build(vec![l], trees) {
                pool.push(c);
            }
        }
        let add_labels = |labels: Vec<u32>,
                          pool: &mut Vec<AfColumn>,
                          seen: &mut ColumnSet,
                          builder: &mut ColumnBuilder| {
            let mut l = labels;
            l.sort_unstable();
            l.dedup();
            if l.len() < 2 || seen.contains(&l) {
                return;
            }
            if let Some(c) = builder.try_build(l.clone(), trees) {
                seen.insert(l);
                pool.push(c);
            }
        };
        for s in &chen_sets {
            add_labels(s.clone(), &mut pool, &mut seen, &mut builder);
        }
        let num_seeds: u64 = if n <= 2_000 {
            12
        } else if n <= 6_000 {
            5
        } else {
            2
        };
        for ref_idx in 0..2usize {
            for seed in 0..num_seeds {
                let (_k, part) =
                    klados_core::lower_bound::greedy_multi_tree_partition(trees, ref_idx, seed);
                for g in groups_from_partition(&part, nl) {
                    add_labels(g, &mut pool, &mut seen, &mut builder);
                }
            }
        }

        let mut rmp = Rmp::new(&pool, trees, nl);
        let mut pricer = ExactPairDpPricer::new(trees);
        let mut scratch = PricerScratch::new(trees);
        let branchings = Branchings::default();

        // Windowed pricing when the global DP table is too large (n ≳ 2850).
        // The exact LP duals from the RMP are mapped into each T₀-subtree
        // window, the DP prices the window, and columns are lifted back to
        // reduced labels and rebuilt against the full trees.
        let global_fits =
            (trees[0].num_nodes() as u64) * (trees[1].num_nodes() as u64) <= CELL_CAP_SAFE;
        let window_max = self.config.window_max;
        let mut windows: Vec<Window> = Vec::new();
        if !global_fits {
            for leaves in split_t0_windows(&trees[0], window_max, self.config.topdown_windows) {
                if self.terminate.load(Ordering::Relaxed) {
                    break;
                }
                let mut keep = FixedBitSet::with_capacity(nl + 1);
                for &l in &leaves {
                    keep.insert(l as usize);
                }
                let (inst, rev) = klados_core::kernelize::restrict_instance_simple(reduced, &keep);
                if inst.num_leaves < 2 || inst.num_trees() != 2 {
                    continue;
                }
                let img: Vec<Vec<u32>> = (0..2)
                    .map(|ti| node_images(&inst.trees[ti], &trees[ti], &rev))
                    .collect();
                windows.push(Window::new(inst, rev, img));
            }
        }
        // The RMP LP bound is a valid global lower bound only when pricing is
        // global (the DP sees the whole column space). Windowed pricing is
        // local, so its converged objective is not an optimality certificate.
        let global = windows.is_empty();
        debug!(
            "[lagr][rmp] n={} chen={} pool={} pricing={} ({:.0}ms)",
            n,
            best_components,
            pool.len(),
            if global {
                "global".to_string()
            } else {
                format!("windowed({})", windows.len())
            },
            start.elapsed().as_secs_f64() * 1000.0
        );

        let mut iter = 0usize;
        loop {
            if self.terminate.load(Ordering::Relaxed)
                || deadline.is_some_and(|d| Instant::now() >= d)
            {
                break;
            }
            iter += 1;

            // Solve the LP (warm-started), separating lazy node rows until clean.
            rmp.apply_bounds(&pool, &branchings);
            let mut sol = match rmp.solve() {
                Ok(s) => s,
                Err(_) => break,
            };
            loop {
                let cuts = rmp.separate_and_add_cuts(&pool, &sol.column_values, 1e-6);
                if cuts == 0 {
                    break;
                }
                rmp.apply_bounds(&pool, &branchings);
                match rmp.solve() {
                    Ok(s) => sol = s,
                    Err(_) => break,
                }
            }

            // Price at the exact LP duals.
            let mut added = 0usize;
            if global {
                let ctx = PricingContext {
                    trees,
                    num_leaves: nl,
                    alpha: &sol.leaf_duals,
                    beta: &sol.node_duals,
                    columns: &pool,
                    seen: &seen,
                    branchings: &branchings,
                    terminate: self.terminate.as_ref(),
                    deadline: None,
                };
                let mut new_cols: Vec<AfColumn> = scratch.drain_reserve(&ctx, 64);
                if let PricingResult::Found(cols) = pricer.price(&ctx, &mut scratch) {
                    new_cols.extend(cols);
                }
                for c in new_cols {
                    let lbls = c.labels().to_vec();
                    if lbls.len() >= 2 && seen.insert(lbls) {
                        rmp.add_column(&c);
                        pool.push(c);
                        added += 1;
                    }
                }
            } else {
                // Windowed pricing: map the exact LP duals into each window,
                // price, lift labels to reduced space, rebuild & add to the RMP.
                let mut lifted: Vec<Vec<u32>> = Vec::new();
                for w in windows.iter_mut() {
                    if self.terminate.load(Ordering::Relaxed)
                        || deadline.is_some_and(|d| Instant::now() >= d)
                    {
                        break;
                    }
                    let rn = w.inst.num_leaves as usize;
                    let mut a_r = vec![0.0f64; rn + 1];
                    for rl in 1..=rn {
                        a_r[rl] = sol.leaf_duals[w.rev[rl] as usize];
                    }
                    let mut b_r: Vec<Vec<f64>> = w
                        .inst
                        .trees
                        .iter()
                        .map(|t| vec![0.0f64; t.num_nodes()])
                        .collect();
                    for ti in 0..2 {
                        let imgti = &w.img[ti];
                        for (node, b) in b_r[ti].iter_mut().enumerate() {
                            let o = imgti[node];
                            if o != NONE {
                                *b = sol.node_duals[ti][o as usize];
                            }
                        }
                    }
                    let got: Vec<Vec<u32>> = {
                        let ctx = PricingContext {
                            trees: &w.inst.trees,
                            num_leaves: rn,
                            alpha: &a_r,
                            beta: &b_r,
                            columns: &[],
                            seen: &w.seen,
                            branchings: &branchings,
                            terminate: self.terminate.as_ref(),
                            deadline: None,
                        };
                        let mut g = Vec::new();
                        for col in w.scratch.drain_reserve(&ctx, 64) {
                            g.push(col.labels().to_vec());
                        }
                        if let PricingResult::Found(cols) = pricer.price(&ctx, &mut w.scratch) {
                            for c in cols {
                                g.push(c.labels().to_vec());
                            }
                        }
                        g
                    };
                    for rl in got {
                        w.seen.insert(rl.clone());
                        lifted.push(rl.iter().map(|&l| w.rev[l as usize]).collect());
                    }
                }
                for mut lbls in lifted {
                    lbls.sort_unstable();
                    lbls.dedup();
                    if lbls.len() < 2 || !seen.insert(lbls.clone()) {
                        continue;
                    }
                    if let Some(c) = builder.try_build(lbls, trees) {
                        rmp.add_column(&c);
                        pool.push(c);
                        added += 1;
                    }
                }
            }

            // Interim primal: greedy on the exact LP duals (always valid).
            let scores: Vec<f64> = pool
                .iter()
                .map(|c| c.pricing_score(&sol.leaf_duals, &sol.node_duals))
                .collect();
            if let Some((forest, comps)) = greedy_pack_af(
                &pool,
                &scores,
                trees,
                n,
                self.flat_terminal.load(Ordering::Relaxed),
            ) && comps < best_components
            {
                best_components = comps;
                best_forest = forest;
            }

            if iter <= 5 || iter.is_multiple_of(10) || added == 0 {
                debug!(
                    "[lagr][rmp] iter={} cols={} +{} lp={:.2} best={} gap={:.1}% t={:.1}s",
                    iter,
                    pool.len(),
                    added,
                    sol.objective,
                    best_components,
                    100.0 * (best_components as f64 - sol.objective) / sol.objective.max(1.0),
                    start.elapsed().as_secs_f64()
                );
            }

            if added == 0 {
                // CG converged: pool is complete, sol.objective is a valid LB.
                let lb = (sol.objective - 1e-6).ceil() as usize;
                // Integral primal (MIP over the pool) only when it can plausibly
                // CERTIFY: a small, closeable gap to the bound, and we are not
                // already terminating. On the cascade a non-certifying MIP is
                // discarded at handoff, so running it on a wide gap is wasted
                // time — and HiGHS can overrun its time limit (a SIGTERM-grace
                // risk). Gating on the gap removes both hazards while keeping the
                // proving cases (e.g. greedy 47 → MIP 46 = lb).
                let gap = best_components.saturating_sub(lb);
                if global
                    && gap > 0
                    && gap <= MIP_GAP_LIMIT
                    && !self.terminate.load(Ordering::Relaxed)
                    && let Ok(Some(mip)) = rmp.solve_mip_with_time_limit(0.5)
                    && let Some((forest, comps)) = forest_from_lp(
                        &pool,
                        &mip.column_values,
                        trees,
                        n,
                        self.flat_terminal.load(Ordering::Relaxed),
                    )
                    && comps < best_components
                {
                    best_components = comps;
                    best_forest = forest;
                }
                // Global pricing with a complete pool ⇒ sol.objective is a valid
                // LB. best ≤ ⌈lb⌉ certifies optimality. Windowed never certifies.
                proved = global && best_components <= lb;
                let status = if !global {
                    "(windowed)"
                } else if proved {
                    "PROVED"
                } else {
                    "(gap)"
                };
                debug!(
                    "[lagr][rmp] CG converged iter={} lp={:.3} best={} {}",
                    iter, sol.objective, best_components, status
                );
                // Anytime branch-and-price: the root LP converged with an
                // integrality gap. Branch (B&P-parity: most-fractional pair,
                // constraint-aware MafPricer, certified LP-bound prune) to close
                // it, keeping the best incumbent and stopping at the deadline.
                if global && !proved && self.config.bnb {
                    let (bf, _bc, bnb_proved) = self.bnb_anytime(
                        trees,
                        n,
                        &mut rmp,
                        &mut pool,
                        &mut seen,
                        &mut scratch,
                        best_forest,
                        best_components,
                        deadline,
                        start,
                    );
                    best_forest = bf;
                    if bnb_proved {
                        proved = true;
                    }
                }
                break;
            }
        }
        // Hand the multi-leaf columns to the caller so the subgradient fallback
        // can warm-start its pool instead of rediscovering them.
        let pool_labels: Vec<Vec<u32>> = pool
            .iter()
            .filter(|c| c.labels().len() >= 2)
            .map(|c| c.labels().to_vec())
            .collect();
        (best_forest, proved, pool_labels)
    }

    /// Run column generation to convergence under `branchings`, reusing the
    /// global pool/RMP. Returns the final LP solution and whether the pricer
    /// **certified** convergence (`Converged`): only then is `objective` a valid
    /// lower bound. On `Improving` (an improving column exists but is branch-
    /// blocked) the bound is NOT trusted — the node must branch. Returns `None`
    /// on deadline/terminate or LP error. Uses the composite `MafPricer` (with
    /// the constraint-aware leaf-pair fallback) so constrained nodes price
    /// exactly, exactly like bp.
    fn price_node(
        &self,
        rmp: &mut Rmp,
        pool: &mut Vec<AfColumn>,
        seen: &mut ColumnSet,
        pricer: &mut MafPricer,
        scratch: &mut PricerScratch,
        branchings: &Branchings,
        trees: &[Tree],
        nl: usize,
        deadline: Option<Instant>,
    ) -> Option<(RmpSolution, bool)> {
        loop {
            if self.terminate.load(Ordering::Relaxed)
                || deadline.is_some_and(|d| Instant::now() >= d)
            {
                return None;
            }
            rmp.apply_bounds(pool, branchings);
            let mut sol = rmp.solve().ok()?;
            loop {
                let cuts = rmp.separate_and_add_cuts(pool, &sol.column_values, 1e-6);
                if cuts == 0 {
                    break;
                }
                rmp.apply_bounds(pool, branchings);
                sol = rmp.solve().ok()?;
            }

            let (new_cols, converged) = {
                let ctx = PricingContext {
                    trees,
                    num_leaves: nl,
                    alpha: &sol.leaf_duals,
                    beta: &sol.node_duals,
                    columns: pool,
                    seen,
                    branchings,
                    terminate: self.terminate.as_ref(),
                    deadline: None,
                };
                let mut new_cols: Vec<AfColumn> = scratch.drain_reserve(&ctx, 64);
                let r = pricer.price(&ctx, scratch);
                let converged = matches!(r, PricingResult::Converged);
                if let PricingResult::Found(cols) = r {
                    new_cols.extend(cols);
                }
                (new_cols, converged)
            };

            let mut added = 0usize;
            for c in new_cols {
                let lbls = c.labels().to_vec();
                if lbls.len() >= 2 && seen.insert(lbls) {
                    rmp.add_column(&c);
                    pool.push(c);
                    added += 1;
                }
            }
            if added == 0 {
                return Some((sol, converged));
            }
        }
    }

    /// Anytime branch-and-price over the RMP pool. DFS on must-link/cannot-link
    /// leaf-pair branchings (bp's `MostFractionalPair` rule), certified LP-bound
    /// pruning, keeping the best incumbent and returning it at the deadline.
    /// Returns `(best_forest, best_components, proved)`; `proved` is true only if
    /// the tree was fully explored (every leaf integral or pruned by a valid
    /// bound) — i.e. the incumbent is the optimum.
    fn bnb_anytime(
        &self,
        trees: &[Tree],
        n: u32,
        rmp: &mut Rmp,
        pool: &mut Vec<AfColumn>,
        seen: &mut ColumnSet,
        scratch: &mut PricerScratch,
        mut best_forest: Vec<Tree>,
        mut best_components: usize,
        deadline: Option<Instant>,
        start: Instant,
    ) -> (Vec<Tree>, usize, bool) {
        let nl = n as usize;
        let mut pricer = dispatch_by_m(trees);
        let mut selector = MostFractionalPair;
        // DFS stack carrying the parent's *certified* LP bound (−∞ = unknown,
        // never prunes). child_LP ≥ parent_LP, so a parent that met the prune
        // threshold lets the child prune without re-solving.
        let mut stack: Vec<(Branchings, f64)> = vec![(Branchings::default(), f64::NEG_INFINITY)];
        let mut nodes = 0usize;
        let mut hit_deadline = false;

        while let Some((br, parent_lp)) = stack.pop() {
            if self.terminate.load(Ordering::Relaxed)
                || deadline.is_some_and(|d| Instant::now() >= d)
            {
                hit_deadline = true;
                break;
            }
            // Inherited certified-bound prune.
            if parent_lp.is_finite() && (parent_lp - 1e-6).ceil() as usize >= best_components {
                continue;
            }

            let (sol, certified) = match self.price_node(
                rmp,
                pool,
                seen,
                &mut pricer,
                scratch,
                &br,
                trees,
                nl,
                deadline,
            ) {
                Some(x) => x,
                None => {
                    if self.terminate.load(Ordering::Relaxed)
                        || deadline.is_some_and(|d| Instant::now() >= d)
                    {
                        hit_deadline = true;
                        break;
                    }
                    continue; // LP error / infeasible node
                }
            };
            let lp = sol.objective;
            // Certified LP-bound prune (only Converged gives a valid bound).
            if certified && (lp - 1e-6).ceil() as usize >= best_components {
                continue;
            }

            // Incumbent from this node's LP support.
            if let Some((forest, comps)) = forest_from_lp(
                pool,
                &sol.column_values,
                trees,
                n,
                self.flat_terminal.load(Ordering::Relaxed),
            ) && comps < best_components
            {
                best_components = comps;
                best_forest = forest;
                debug!(
                    "[lagr][bnb] node={} incumbent={} lp={:.2} t={:.1}s",
                    nodes,
                    best_components,
                    lp,
                    start.elapsed().as_secs_f64()
                );
            }
            nodes += 1;

            // Branch on the most-fractional leaf-pair. Pass the certified LP
            // bound to children (uncertified ⇒ −∞: children re-derive their own).
            let child_bound = if certified { lp } else { f64::NEG_INFINITY };
            let ctx = SelectionContext {
                columns: pool,
                values: &sol.column_values,
                num_leaves: nl,
                branchings: &br,
                current_lp_obj: lp,
            };
            if let Some(children) = selector.select(&ctx, rmp) {
                for child in children.into_iter().rev() {
                    stack.push((child, child_bound));
                }
            }
            // `None` ⇒ integral LP support: incumbent already captured above.
        }

        let proved = !hit_deadline;
        debug!(
            "[lagr][bnb] done nodes={} best={} proved={} t={:.1}s",
            nodes,
            best_components,
            proved,
            start.elapsed().as_secs_f64()
        );
        (best_forest, best_components, proved)
    }

    /// Dual-guided greedy node-disjoint packing. Returns true if it improved
    /// the incumbent (and updates it).
    fn try_primal(
        &self,
        trees: &[Tree],
        n: u32,
        pool: &[Block],
        scores: &[f64],
        best_forest: &mut Vec<Tree>,
        best_components: &mut usize,
        best_sel: &mut Vec<usize>,
    ) -> bool {
        if pool.is_empty() {
            return false;
        }
        let mut order: Vec<usize> = (0..pool.len()).collect();
        order.sort_unstable_by(|&i, &j| {
            scores[j]
                .total_cmp(&scores[i])
                .then_with(|| pool[j].weight.cmp(&pool[i].weight))
        });

        let mut used: Vec<FixedBitSet> = trees
            .iter()
            .map(|t| FixedBitSet::with_capacity(t.num_nodes()))
            .collect();
        let mut selected: Vec<usize> = Vec::new();
        let mut savings = 0usize;
        'cand: for idx in order {
            let b = &pool[idx];
            for (t, nodes) in b.cover.iter().enumerate() {
                for &v in nodes {
                    if used[t].contains(v as usize) {
                        continue 'cand;
                    }
                }
            }
            for (t, nodes) in b.cover.iter().enumerate() {
                for &v in nodes {
                    used[t].insert(v as usize);
                }
            }
            savings += b.weight;
            selected.push(idx);
        }

        let components = n as usize - savings;
        if components < *best_components {
            *best_components = components;
            *best_forest = build_forest(pool, &selected, trees, n, false);
            *best_sel = selected;
            true
        } else {
            false
        }
    }

    /// One subgradient step over the pool. Returns the Lagrangian bound L(α,β).
    fn subgradient_step(
        &self,
        trees: &[Tree],
        nl: usize,
        pool: &[Block],
        scores: &[f64],
        alpha: &mut [f64],
        beta: &mut [Vec<f64>],
        lambda: f64,
        ub_components: usize,
    ) -> f64 {
        // Lagrangian subproblem: x_c = 1 iff reduced cost < 0 (score > 1).
        let mut cov = vec![0i32; nl + 1];
        let mut use_nodes: Vec<Vec<i32>> =
            trees.iter().map(|t| vec![0i32; t.num_nodes()]).collect();
        let mut sum_rc = 0.0f64;
        for (i, b) in pool.iter().enumerate() {
            let s = scores[i];
            if s > 1.0 {
                sum_rc += 1.0 - s; // negative reduced cost
                for &l in &b.labels {
                    cov[l as usize] += 1;
                }
                for (t, nodes) in b.cover.iter().enumerate() {
                    for &v in nodes {
                        use_nodes[t][v as usize] += 1;
                    }
                }
            }
        }
        // Singleton columns {l} (cost 1, no nodes) are implicit members of the
        // master. Their dual constraint caps α_l ≤ 1: the singleton prices in
        // (rc = 1 − α_l < 0) exactly when α_l > 1, covering leaf l once. Without
        // them, a leaf coverable only as a singleton has an unbounded α
        // subgradient and the dual diverges.
        for l in 1..=nl {
            if alpha[l] > 1.0 {
                sum_rc += 1.0 - alpha[l];
                cov[l] += 1;
            }
        }

        let sum_alpha: f64 = alpha[1..=nl].iter().sum();
        let sum_beta: f64 = beta.iter().flat_map(|b| b.iter()).sum();
        let lagrangian = sum_alpha - sum_beta + sum_rc;

        // Subgradient: g_α[l] = 1 - cov[l]; g_β[t,v] = use[t,v] - 1.
        let mut gnorm2 = 0.0f64;
        for l in 1..=nl {
            let g = 1.0 - cov[l] as f64;
            gnorm2 += g * g;
        }
        for (t, tree) in trees.iter().enumerate() {
            for v in 0..tree.num_nodes() {
                if tree.is_leaf(v as u32) {
                    continue;
                }
                let g = use_nodes[t][v] as f64 - 1.0;
                gnorm2 += g * g;
            }
        }
        if gnorm2 < 1e-12 {
            return lagrangian.max(0.0);
        }

        // Polyak step toward the incumbent upper bound.
        let target = (ub_components as f64 - lagrangian).max(0.5);
        let step = lambda * target / gnorm2;

        for l in 1..=nl {
            let g = 1.0 - cov[l] as f64;
            alpha[l] += step * g; // α is free
        }
        for (t, tree) in trees.iter().enumerate() {
            for v in 0..tree.num_nodes() {
                if tree.is_leaf(v as u32) {
                    continue;
                }
                let g = use_nodes[t][v] as f64 - 1.0;
                let nv = beta[t][v] + step * g;
                beta[t][v] = nv.max(0.0); // β ≥ 0
            }
        }

        lagrangian.max(0.0)
    }

    /// Volume-algorithm step (Barahona–Anbil). Like `subgradient_step` but the
    /// step direction comes from a running average `v` of the subproblem
    /// solutions (the "primal estimate"), not the instantaneous one — this
    /// damps the zigzag and converges to the LP dual in far fewer iterations.
    /// `v_cov`/`v_use` persist across calls; `avg_a` is the averaging weight.
    /// One volume-algorithm iteration (Barahona–Anbil). Maintains a stability
    /// centre (`center_*`, the best-bound dual point), a per-column primal
    /// estimate `xbar` (running average of the subproblem solutions), and
    /// serious/null step control. The dual point `alpha`/`beta` is set to
    /// `centre + step·d`, where `d` is the residual of the *averaged* primal
    /// `xbar` — a far smoother direction than the instantaneous subgradient.
    /// Returns the centre's bound (monotone non-decreasing). The caller packs
    /// the primal from `xbar`, not from the thrashing instantaneous scores.
    fn volume_step(
        &self,
        trees: &[Tree],
        nl: usize,
        pool: &[Block],
        scores: &[f64],
        alpha: &mut [f64],
        beta: &mut [Vec<f64>],
        xbar: &mut Vec<f64>,
        xbar_sing: &mut [f64],
        center_alpha: &mut [f64],
        center_beta: &mut [Vec<f64>],
        center_lb: &mut f64,
        serious_run: &mut usize,
        null_run: &mut usize,
        lambda: &mut f64,
        avg_a: f64,
        ub_components: usize,
    ) -> f64 {
        let first = !center_lb.is_finite();

        // (1) Value L(u_t) of the instantaneous subproblem at the current duals.
        let mut sum_rc = 0.0f64;
        for (c, _) in pool.iter().enumerate() {
            if scores[c] > 1.0 {
                sum_rc += 1.0 - scores[c];
            }
        }
        for l in 1..=nl {
            if alpha[l] > 1.0 {
                sum_rc += 1.0 - alpha[l];
            }
        }
        let sum_alpha: f64 = alpha[1..=nl].iter().sum();
        let sum_beta: f64 = beta.iter().flat_map(|b| b.iter()).sum();
        let l_t = sum_alpha - sum_beta + sum_rc;

        // (2) Serious/null step: the current point becomes the centre iff it
        //     improved the bound. Grow the step after a run of serious steps,
        //     shrink it after a run of nulls.
        if first || l_t > *center_lb + 1e-9 {
            center_alpha.copy_from_slice(alpha);
            for (t, cb) in center_beta.iter_mut().enumerate() {
                cb.copy_from_slice(&beta[t]);
            }
            *center_lb = l_t;
            *serious_run += 1;
            *null_run = 0;
            if *serious_run >= 3 {
                *lambda = (*lambda * 1.1).min(2.0);
                *serious_run = 0;
            }
        } else {
            *null_run += 1;
            *serious_run = 0;
            if *null_run >= 10 {
                *lambda = (*lambda * 0.67).max(1.0e-3);
                *null_run = 0;
            }
        }

        // (3) Running-average primal estimate x̄ ← a·x_t + (1−a)·x̄ (per column
        //     and per implicit singleton). a=1 on the first step (seed = x_t).
        xbar.resize(pool.len(), 0.0);
        let a = if first { 1.0 } else { avg_a };
        for (c, xb) in xbar.iter_mut().enumerate() {
            let xt = if scores[c] > 1.0 { 1.0 } else { 0.0 };
            *xb = a * xt + (1.0 - a) * *xb;
        }
        for l in 1..=nl {
            let xt = if alpha[l] > 1.0 { 1.0 } else { 0.0 };
            xbar_sing[l] = a * xt + (1.0 - a) * xbar_sing[l];
        }

        // (4) Descent direction d = b − A·x̄ from the averaged primal.
        let mut cov = vec![0.0f64; nl + 1];
        let mut use_nodes: Vec<Vec<f64>> =
            trees.iter().map(|t| vec![0.0f64; t.num_nodes()]).collect();
        for (c, b) in pool.iter().enumerate() {
            let x = xbar[c];
            if x <= 1.0e-12 {
                continue;
            }
            for &l in &b.labels {
                cov[l as usize] += x;
            }
            for (t, nodes) in b.cover.iter().enumerate() {
                for &v in nodes {
                    use_nodes[t][v as usize] += x;
                }
            }
        }
        for l in 1..=nl {
            cov[l] += xbar_sing[l];
        }

        let mut gnorm2 = 0.0f64;
        for l in 1..=nl {
            let g = 1.0 - cov[l];
            gnorm2 += g * g;
        }
        for (t, tree) in trees.iter().enumerate() {
            for v in 0..tree.num_nodes() {
                if tree.is_leaf(v as u32) {
                    continue;
                }
                let g = use_nodes[t][v] - 1.0;
                gnorm2 += g * g;
            }
        }
        if gnorm2 < 1.0e-12 {
            return center_lb.max(0.0);
        }

        // (5) Step from the CENTRE along d (Polyak target toward the incumbent).
        let target = (ub_components as f64 - *center_lb).max(0.5);
        let step = *lambda * target / gnorm2;
        for l in 1..=nl {
            alpha[l] = center_alpha[l] + step * (1.0 - cov[l]);
        }
        for (t, tree) in trees.iter().enumerate() {
            for v in 0..tree.num_nodes() {
                if tree.is_leaf(v as u32) {
                    continue;
                }
                let nv = center_beta[t][v] + step * (use_nodes[t][v] - 1.0);
                beta[t][v] = nv.max(0.0);
            }
        }

        center_lb.max(0.0)
    }
}

// ---------------------------------------------------------------------------
// R2: tree-local windowed pricing (so the anchor DP runs at ~15k leaves).
// ---------------------------------------------------------------------------

/// A precomputed pricing window: the instance restricted to a bounded T₀
/// subtree, with maps back to the original problem and reusable pricer state.
struct Window {
    inst: Instance,
    /// `rev[r_label] = original label`.
    rev: Vec<u32>,
    /// `img[ti][r_node] = original node` in tree `ti` (for mapping β).
    img: Vec<Vec<u32>>,
    scratch: PricerScratch,
    seen: ColumnSet,
    /// Mapped duals at the last DP run, for adaptive re-pricing: a window is
    /// re-priced only when its mapped α/β moved materially since (or after a
    /// round-robin staleness cap). The dense pair-DP is 83% of the giant
    /// budget and re-pricing a window whose duals barely changed yields no new
    /// columns — pure waste. `last_a[r_label]`, `last_b[ti][r_node]`.
    last_a: Vec<f64>,
    last_b: Vec<Vec<f64>>,
    /// Iterations since this window was last priced (round-robin floor).
    stale: usize,
}

impl Window {
    fn new(inst: Instance, rev: Vec<u32>, img: Vec<Vec<u32>>) -> Self {
        let rn = inst.num_leaves as usize;
        // NEG_INFINITY + stale=MAX force a price on the first visit.
        let last_a = vec![f64::NEG_INFINITY; rn + 1];
        let last_b: Vec<Vec<f64>> = inst
            .trees
            .iter()
            .map(|t| vec![f64::NEG_INFINITY; t.num_nodes()])
            .collect();
        let scratch = PricerScratch::new(&inst.trees);
        Window {
            inst,
            rev,
            img,
            scratch,
            seen: ColumnSet::new(),
            last_a,
            last_b,
            stale: usize::MAX,
        }
    }
}

/// Split T₀ into leaf groups, each a connected subtree region with ≤
/// `max_leaves` leaves, so any agreement component fully inside the group is
/// findable by the restricted DP.
///
/// Default is BOTTOM-UP packing: post-order, carry each subtree's still-open
/// leaves upward and only close a window when two children together would
/// exceed the cap (closing the larger, carrying the smaller on). This yields
/// ~⌈n/max⌉ near-full windows REGARDLESS of tree balance. The old TOP-DOWN
/// greedy (emit every maximal ≤max subtree) shatters unbalanced trees into many
/// tiny windows (measured: ~137 avg-66-leaf windows on a 9k-leaf core), wasting
/// per-iteration overhead and dropping cross-spine columns the DP never prices.
/// `LagrangianConfig.topdown_windows` restores the old behaviour for A/B testing.
fn split_t0_windows(tree: &Tree, max_leaves: usize, topdown: bool) -> Vec<Vec<u32>> {
    if topdown {
        return split_t0_windows_topdown(tree, max_leaves);
    }
    let max = max_leaves.max(2);
    let mut windows: Vec<Vec<u32>> = Vec::new();
    // `open[node]` = leaves in `node`'s subtree not yet assigned to a window.
    let mut open: Vec<Vec<u32>> = vec![Vec::new(); tree.num_nodes()];
    for v in tree.post_order_vec() {
        if tree.is_leaf(v) {
            let lbl = tree.label[v as usize];
            if lbl > 0 {
                open[v as usize] = vec![lbl];
            }
            continue;
        }
        let (l, r) = tree.children_pair(v);
        let lo = std::mem::take(&mut open[l as usize]);
        let ro = std::mem::take(&mut open[r as usize]);
        if lo.len() + ro.len() <= max {
            // Keep both subtrees open; let the region grow toward the cap.
            let mut merged = lo;
            merged.extend(ro);
            open[v as usize] = merged;
        } else {
            // Closing both would exceed the cap: emit the larger as a window
            // (a connected region), carry the smaller up to keep packing.
            let (keep, close) = if lo.len() >= ro.len() {
                (ro, lo)
            } else {
                (lo, ro)
            };
            if close.len() >= 2 {
                windows.push(close);
            }
            open[v as usize] = keep;
        }
    }
    let root_open = std::mem::take(&mut open[tree.root as usize]);
    if root_open.len() >= 2 {
        windows.push(root_open);
    }
    windows
}

/// Legacy top-down windowing (fragments unbalanced trees); kept for A/B.
fn split_t0_windows_topdown(tree: &Tree, max_leaves: usize) -> Vec<Vec<u32>> {
    let nn = tree.num_nodes();
    let mut cnt = vec![0u32; nn];
    for v in tree.post_order_vec() {
        if tree.is_leaf(v) {
            cnt[v as usize] = 1;
        } else {
            let (l, r) = tree.children_pair(v);
            cnt[v as usize] = cnt[l as usize] + cnt[r as usize];
        }
    }
    let mut windows = Vec::new();
    let mut stack = vec![tree.root];
    while let Some(v) = stack.pop() {
        if (cnt[v as usize] as usize) <= max_leaves {
            let mut leaves = Vec::new();
            let mut s = vec![v];
            while let Some(u) = s.pop() {
                if tree.is_leaf(u) {
                    let lbl = tree.label[u as usize];
                    if lbl > 0 {
                        leaves.push(lbl);
                    }
                } else {
                    let (l, r) = tree.children_pair(u);
                    s.push(l);
                    s.push(r);
                }
            }
            if leaves.len() >= 2 {
                windows.push(leaves);
            }
        } else {
            let (l, r) = tree.children_pair(v);
            stack.push(l);
            stack.push(r);
        }
    }
    windows
}

/// Map each node of `restricted` to its image node in `orig` (the LCA in
/// `orig` of the kept leaves beneath it). `rev[r_label] = orig label`.
fn node_images(restricted: &Tree, orig: &Tree, rev: &[u32]) -> Vec<u32> {
    let mut img = vec![NONE; restricted.num_nodes()];
    for r_node in restricted.post_order_vec() {
        if restricted.is_leaf(r_node) {
            let r_label = restricted.label[r_node as usize];
            if r_label > 0 {
                img[r_node as usize] = orig.node_by_label(rev[r_label as usize]);
            }
        } else {
            let (l, r) = restricted.children_pair(r_node);
            let (ol, or) = (img[l as usize], img[r as usize]);
            img[r_node as usize] = if ol == NONE {
                or
            } else if or == NONE {
                ol
            } else {
                orig.nearest_common_ancestor(ol, or)
            };
        }
    }
    img
}

/// Drop the lowest-scoring blocks when the pool grows past the cap.
/// Prune the pool to the best columns under BOTH a count cap (`keep`) and the
/// cell-memory budget (`max_cells`), keeping highest Lagrangian score first.
/// Rebuilds `seen` from the survivors so the dedup set can't grow without bound
/// (pruned columns may be re-priced later — fine, they were low value).
fn prune_pool(
    pool: &mut Vec<Block>,
    seen: &mut ColumnSet,
    alpha: &[f64],
    beta: &[Vec<f64>],
    keep: usize,
    max_cells: usize,
) {
    pool.sort_unstable_by(|a, b| {
        block_score(b, alpha, beta)
            .total_cmp(&block_score(a, alpha, beta))
            .then_with(|| b.weight.cmp(&a.weight))
    });
    let mut cells = 0usize;
    let mut survivors = 0usize;
    for b in pool.iter() {
        if survivors >= keep {
            break;
        }
        let c = block_cells(b);
        if survivors > 0 && cells + c > max_cells {
            break;
        }
        cells += c;
        survivors += 1;
    }
    pool.truncate(survivors.max(1));
    *seen = ColumnSet::new();
    for b in pool.iter() {
        seen.insert(b.labels.clone());
    }
}

fn groups_from_partition(partition: &[usize], nl: usize) -> Vec<Vec<u32>> {
    let mut by_comp: std::collections::HashMap<usize, Vec<u32>> = std::collections::HashMap::new();
    for (i, &comp) in partition.iter().enumerate().take(nl) {
        by_comp.entry(comp).or_default().push((i + 1) as u32);
    }
    by_comp.into_values().filter(|g| g.len() >= 2).collect()
}

/// Build a singleton component, indexed or compact (unindexed) per `unindexed`.
/// Unindexed pieces are O(1) memory and traverse-only — used for TERMINAL
/// forests (held incumbent, flat depth-0 output) that are never label-queried
/// or fed to `whidden_cluster` recombination. See [`Tree::forest_singleton`].
#[inline]
fn mk_singleton(l: u32, n: u32, unindexed: bool) -> Tree {
    if unindexed {
        Tree::forest_singleton(l, n)
    } else {
        Tree::singleton(l, n)
    }
}

/// Build a leafset component, indexed or compact (unindexed) per `unindexed`.
/// See [`mk_singleton`] for when unindexed is safe.
#[inline]
fn mk_component(bs: &FixedBitSet, tref: &Tree, n: u32, unindexed: bool) -> Tree {
    if unindexed {
        Tree::forest_component(bs, tref, n)
    } else {
        Tree::component_from_leafset(bs, tref, n)
    }
}

fn forest_from_partition(sets: &[Vec<u32>], trees: &[Tree], n: u32, unindexed: bool) -> Vec<Tree> {
    let mut forest = Vec::with_capacity(sets.len());
    let mut covered = FixedBitSet::with_capacity(n as usize + 1);
    for s in sets {
        for &l in s {
            covered.insert(l as usize);
        }
        if s.len() == 1 {
            forest.push(mk_singleton(s[0], n, unindexed));
        } else {
            let mut bs = FixedBitSet::with_capacity(n as usize + 1);
            for &l in s {
                bs.insert(l as usize);
            }
            forest.push(mk_component(&bs, &trees[0], n, unindexed));
        }
    }
    for l in 1..=n {
        if !covered.contains(l as usize) {
            forest.push(mk_singleton(l, n, unindexed));
        }
    }
    // sanity: every leaf covered exactly once (Chen forest is a partition)
    debug_assert!((1..=n).all(|l| covered.contains(l as usize)) || !sets.is_empty());
    forest
}

/// True if `b` violates a branching constraint: splits a must-link pair, or
/// contains a cannot-link pair. (`b.labels` is sorted.)
/// Greedy score-ordered node-disjoint packing over AfColumns (≥2 leaves).
/// Always produces a valid agreement forest. Returns `(forest, components)`.
fn greedy_pack_af(
    pool: &[AfColumn],
    scores: &[f64],
    trees: &[Tree],
    n: u32,
    unindexed: bool,
) -> Option<(Vec<Tree>, usize)> {
    let mut order: Vec<usize> = (0..pool.len())
        .filter(|&i| pool[i].labels().len() >= 2)
        .collect();
    if order.is_empty() {
        return None;
    }
    order.sort_unstable_by(|&i, &j| {
        scores[j]
            .total_cmp(&scores[i])
            .then_with(|| pool[j].labels().len().cmp(&pool[i].labels().len()))
    });
    let mut used: Vec<FixedBitSet> = trees
        .iter()
        .map(|t| FixedBitSet::with_capacity(t.num_nodes()))
        .collect();
    let mut covered = FixedBitSet::with_capacity(n as usize + 1);
    let mut forest: Vec<Tree> = Vec::new();
    'cand: for idx in order {
        let cov = pool[idx].coverage();
        for (t, nodes) in cov.iter_per_tree().enumerate() {
            for &v in nodes {
                if used[t].contains(v) {
                    continue 'cand;
                }
            }
        }
        for (t, nodes) in cov.iter_per_tree().enumerate() {
            for &v in nodes {
                used[t].insert(v);
            }
        }
        let mut bs = FixedBitSet::with_capacity(n as usize + 1);
        for &l in pool[idx].labels() {
            bs.insert(l as usize);
            covered.insert(l as usize);
        }
        forest.push(mk_component(&bs, &trees[0], n, unindexed));
    }
    for l in 1..=n {
        if !covered.contains(l as usize) {
            forest.push(mk_singleton(l, n, unindexed));
        }
    }
    let len = forest.len();
    Some((forest, len))
}

/// Build a forest from an integral LP/MIP solution (columns with x>0.5),
/// validating node-disjointness (lazy node rows may leave a constraint
/// unmaterialised). Returns `None` if the selection is node-infeasible.
fn forest_from_lp(
    pool: &[AfColumn],
    x: &[f64],
    trees: &[Tree],
    n: u32,
    unindexed: bool,
) -> Option<(Vec<Tree>, usize)> {
    let mut used: Vec<FixedBitSet> = trees
        .iter()
        .map(|t| FixedBitSet::with_capacity(t.num_nodes()))
        .collect();
    let mut covered = FixedBitSet::with_capacity(n as usize + 1);
    let mut forest: Vec<Tree> = Vec::new();
    for (ci, &xv) in x.iter().enumerate() {
        if xv <= 0.5 || ci >= pool.len() || pool[ci].labels().len() < 2 {
            continue;
        }
        let cov = pool[ci].coverage();
        for (t, nodes) in cov.iter_per_tree().enumerate() {
            for &v in nodes {
                if used[t].contains(v) {
                    return None; // node conflict ⇒ not a valid AF
                }
                used[t].insert(v);
            }
        }
        let mut bs = FixedBitSet::with_capacity(n as usize + 1);
        for &l in pool[ci].labels() {
            if covered.contains(l as usize) {
                return None; // leaf used twice
            }
            bs.insert(l as usize);
            covered.insert(l as usize);
        }
        forest.push(mk_component(&bs, &trees[0], n, unindexed));
    }
    for l in 1..=n {
        if !covered.contains(l as usize) {
            forest.push(mk_singleton(l, n, unindexed));
        }
    }
    let len = forest.len();
    Some((forest, len))
}

fn block_forbidden(b: &Block, br: &Branchings) -> bool {
    for ml in br.must_link() {
        let ha = b.labels.binary_search(&ml.a).is_ok();
        let hb = b.labels.binary_search(&ml.b).is_ok();
        if ha != hb {
            return true;
        }
    }
    for cl in br.cannot_link() {
        if b.labels.binary_search(&cl.a).is_ok() && b.labels.binary_search(&cl.b).is_ok() {
            return true;
        }
    }
    false
}

/// Pick the leaf-pair to branch on: among positive-reduced-cost columns (the
/// dual "wants" their leaves together), the highest-scoring one that the node's
/// incumbent currently *splits* across components. Forcing it together
/// (must-link) makes the constrained pricer emit that gap column. Skips pairs
/// already constrained on this branch.
fn pick_branch_pair(
    pool: &[Block],
    sel: &[usize],
    alpha: &[f64],
    beta: &[Vec<f64>],
    nl: usize,
    br: &Branchings,
) -> Option<LeafPair> {
    let mut comp_of = vec![usize::MAX; nl + 1];
    for (ci, &pi) in sel.iter().enumerate() {
        for &l in &pool[pi].labels {
            if (l as usize) <= nl {
                comp_of[l as usize] = ci;
            }
        }
    }
    let scores: Vec<f64> = pool.iter().map(|b| block_score(b, alpha, beta)).collect();
    let mut order: Vec<usize> = (0..pool.len()).collect();
    order.sort_unstable_by(|&i, &j| scores[j].total_cmp(&scores[i]));
    // Highest-scored column the incumbent splits. (At dual convergence scores
    // are tight ≈1, so we must NOT require score>1 or branching never starts.)
    for &i in &order {
        let lbls = &pool[i].labels;
        for wi in 0..lbls.len() {
            for wj in (wi + 1)..lbls.len() {
                let (a, b) = (lbls[wi], lbls[wj]);
                let (ca, cb) = (comp_of[a as usize], comp_of[b as usize]);
                if ca == usize::MAX || cb == usize::MAX || ca == cb {
                    continue;
                }
                let pair = LeafPair::new(a, b);
                if br.must_link().contains(&pair) || br.cannot_link().contains(&pair) {
                    continue;
                }
                return Some(pair);
            }
        }
    }
    None
}

/// Score-ordered greedy node-disjoint packing over the pool, skipping columns
/// forbidden by `br`. Returns `(components, selected_indices)`.
fn greedy_pack(
    pool: &[Block],
    scores: &[f64],
    trees: &[Tree],
    n: u32,
    br: &Branchings,
) -> (usize, Vec<usize>) {
    let mut order: Vec<usize> = (0..pool.len())
        .filter(|&i| !block_forbidden(&pool[i], br))
        .collect();
    order.sort_unstable_by(|&i, &j| {
        scores[j]
            .total_cmp(&scores[i])
            .then_with(|| pool[j].weight.cmp(&pool[i].weight))
    });
    let mut used: Vec<FixedBitSet> = trees
        .iter()
        .map(|t| FixedBitSet::with_capacity(t.num_nodes()))
        .collect();
    let mut selected: Vec<usize> = Vec::new();
    let mut savings = 0usize;
    'cand: for idx in order {
        let b = &pool[idx];
        for (t, nodes) in b.cover.iter().enumerate() {
            for &v in nodes {
                if used[t].contains(v as usize) {
                    continue 'cand;
                }
            }
        }
        for (t, nodes) in b.cover.iter().enumerate() {
            for &v in nodes {
                used[t].insert(v as usize);
            }
        }
        savings += b.weight;
        selected.push(idx);
    }
    (n as usize - savings, selected)
}

/// Final safety guard: ensure the emitted forest is a valid agreement forest of
/// the ORIGINAL trees. Kernelization's `expand_solution` re-inserts collapsed
/// leaves and can (intermittently) produce components whose original-tree
/// spanning subtrees interleave — a node-disjointness violation that the
/// validator rejects outright (score 0). We never want to emit that. Keep
/// components largest-first while they remain valid agreement components AND
/// node-disjoint from those already kept; explode any offender into singletons.
/// Returns `(repaired_forest, num_components_exploded)`.
fn repair_forest(forest: Vec<Tree>, trees: &[Tree], n: u32) -> (Vec<Tree>, usize) {
    let comp_leaves: Vec<Vec<u32>> = forest.iter().map(|c| c.leaves().collect()).collect();
    let mut order: Vec<usize> = (0..forest.len()).collect();
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(comp_leaves[i].len()));

    let mut used: Vec<FixedBitSet> = trees
        .iter()
        .map(|t| FixedBitSet::with_capacity(t.num_nodes()))
        .collect();
    // Every leaf must appear in EXACTLY one emitted component. The exact unwind's
    // best-effort join (and any upstream overlap) can repeat a leaf across
    // components without sharing internal nodes, so leaf-disjointness must be
    // enforced separately from node-disjointness — otherwise the emitted forest
    // is not a valid agreement forest (duplicate leaves) and the harness rejects
    // it. `used_leaf` tracks leaves already committed to a kept component.
    let mut used_leaf = FixedBitSet::with_capacity(n as usize + 1);
    let mut out: Vec<Tree> = Vec::with_capacity(forest.len());
    let mut forest: Vec<Option<Tree>> = forest.into_iter().map(Some).collect();
    let mut exploded = 0usize;

    'comp: for &i in &order {
        let labels = &comp_leaves[i];
        let any_dup = labels.iter().any(|&l| used_leaf.contains(l as usize));
        // A genuine, node-disjoint, leaf-disjoint agreement component is kept
        // as-is. Singletons (len<2) likewise, when their leaf is still free.
        if !any_dup {
            if labels.len() < 2 {
                if let Some(&l) = labels.first() {
                    used_leaf.insert(l as usize);
                }
                out.push(forest[i].take().unwrap());
                continue;
            }
            if is_valid_af_component(labels, trees) {
                let cov: Vec<Vec<u32>> = trees.iter().map(|t| vset_internal(t, labels)).collect();
                let conflict = cov
                    .iter()
                    .enumerate()
                    .any(|(t, ns)| ns.iter().any(|&v| used[t].contains(v as usize)));
                if !conflict {
                    for (t, ns) in cov.iter().enumerate() {
                        for &v in ns {
                            used[t].insert(v as usize);
                        }
                    }
                    for &l in labels {
                        used_leaf.insert(l as usize);
                    }
                    out.push(forest[i].take().unwrap());
                    continue 'comp;
                }
            }
        }
        // Offender (overlapping, invalid, or node-conflicting): explode into
        // singletons for its still-free leaves only — never re-emitting a leaf
        // already covered by a kept component.
        let mut any = false;
        for &l in labels {
            if !used_leaf.contains(l as usize) {
                used_leaf.insert(l as usize);
                out.push(Tree::singleton(l, n));
                any = true;
            }
        }
        if any {
            exploded += 1;
        }
    }
    // Safety net: any leaf never covered by the input forest becomes a singleton
    // so the result is a complete partition of 1..=n.
    for l in 1..=n {
        if !used_leaf.contains(l as usize) {
            out.push(Tree::singleton(l, n));
        }
    }
    (out, exploded)
}

fn build_forest(
    pool: &[Block],
    selected: &[usize],
    trees: &[Tree],
    n: u32,
    unindexed: bool,
) -> Vec<Tree> {
    let mut forest = Vec::with_capacity(selected.len());
    let mut covered = FixedBitSet::with_capacity(n as usize + 1);
    for &i in selected {
        let b = &pool[i];
        let mut bs = FixedBitSet::with_capacity(n as usize + 1);
        for &l in &b.labels {
            bs.insert(l as usize);
            covered.insert(l as usize);
        }
        forest.push(mk_component(&bs, &trees[0], n, unindexed));
    }
    for l in 1..=n {
        if !covered.contains(l as usize) {
            forest.push(mk_singleton(l, n, unindexed));
        }
    }
    forest
}

impl Default for LagrangianSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ── Unified Solver impl + entry point ───────────────────────────────────────
use crate::{RunConfig, Solver, Track};

impl Solver for LagrangianSolver {
    type Config = LagrangianConfig;
    const SUPPORTED_TRACKS: &'static [Track] = &[Track::Heuristic];
    fn solve(&mut self, inst: &Instance, cfg: &RunConfig<Self::Config>) -> Option<Vec<Tree>> {
        self.config = cfg.specific.clone();
        LagrangianSolver::solve(self, inst)
    }
    fn stats(&self) -> &SolverStats {
        &self.stats
    }
    fn sigterm_handler(&self, _track: Track) -> Option<Box<dyn Fn() + Send + Sync>> {
        let flag = self.terminate.clone();
        Some(Box::new(move || flag.store(true, Ordering::Relaxed)))
    }
}

pub fn main() {
    crate::run(
        LagrangianSolver::new(),
        RunConfig {
            track: Track::Heuristic,
            specific: LagrangianConfig::default(),
            ..Default::default()
        },
    );
}
