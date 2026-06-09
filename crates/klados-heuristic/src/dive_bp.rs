//! Diving branch-and-price heuristic ("divebp").
//!
//! This is the exact branch-and-price engine of the Klados paper, restructured
//! as an **anytime dive**. It keeps the parts the Lagrangian heuristic gets
//! right — the set-cover LP relaxation's tight gap and its exact `(α, β)`
//! duals — and adds the part the Lagrangian lacks: **Ryan–Foster branching on
//! taxon pairs**. Branching is what escapes the LP↔integer plateau (the
//! "local maxima"): it forces the integral columns the unconstrained pricer
//! never emits.
//!
//! Why this and not just running exact `bp` with early termination:
//!   * The exact solver's primal heuristics are deliberately weak (greedy
//!     rounding typically yields `OPT+1` on a fractional support), so as an
//!     anytime engine it can sit at a poor incumbent for a long time. Here we
//!     extract a complete forest from the **whole shared pool** at every node
//!     and publish it the instant it improves.
//!   * The search order is a *dive* (must-link child first), tuned to reach a
//!     strong integral incumbent fast rather than to prove the bound.
//!
//! Anytime contract (for `--features early-termination`):
//!   1. A Chen 2-approx forest is published before any heavy work.
//!   2. Every node packs the pool into a complete valid forest and publishes
//!      it if smaller (expanded to original labels via the kernel trace).
//!   3. The DFS loop polls `terminate`/`deadline` and returns the best forest.
//!
//! Convergence: under branching column-bounds a *Converged* node LP is a valid
//! node lower bound, so `⌈LP⌉ ≥ incumbent` pruning is sound and exhausting the
//! DFS proves the optimum — unlike the subgradient B&B, whose pool-wide bound
//! is valid only at the root.
//!
//! Scope: 2-tree (the heuristic track). Instances whose dense pricing table
//! exceeds the cell cap degrade to greedy packing of the seed pool (no LP);
//! windowed pricing for giants is a follow-up.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fixedbitset::FixedBitSet;
use klados_core::af_validator::validate_agreement_forest;
use klados_core::kernelize::{
    expand_solution_unindexed, kernelize_best, restrict_instance_simple, KernelizeConfig,
    KernelizeResult,
};
use klados_core::lower_bound::greedy_multi_tree_partition;
use klados_core::tree::NONE;
use klados_core::{Instance, SolverStats, Tree};
use klados_exact::bp::column::{AfColumn, ColumnBuilder, ColumnSet};
use klados_exact::bp::pricer::{
    ExactPairDpPricer, Pricer, PricerScratch, PricingContext, PricingResult,
};
use klados_exact::bp::rmp::Rmp;
use klados_exact::bp::search::{BranchSelector, Branchings, MostFractionalPair, SelectionContext};
use klados_exact::chen_rspr::chen_pair_agreement;

use crate::lagrangian::{node_images, repair_forest, split_t0_windows};
use crate::HeuristicSolver;

/// Above this product of the two trees' node counts the dense pair-DP pricing
/// table is too large to materialise; the dive degrades to greedy packing of
/// the seed pool. Mirrors the Lagrangian's `CELL_CAP_SAFE`.
const CELL_CAP: u64 = 60_000_000;
/// Stop growing the shared column pool past this many columns (memory bound).
/// The LP/branch loop keeps running on the frozen pool.
const POOL_CAP: usize = 120_000;
/// Max leaves per T₀-subtree pricing window on the giant path. Chosen so each
/// window's dense pricing table `(2·W)²` stays well under [`CELL_CAP`].
const WINDOW_MAX_LEAVES: usize = 1_200;

pub struct DiveBpSolver {
    terminate: Arc<AtomicBool>,
    stats: SolverStats,
    /// Best complete forest in ORIGINAL labels, ready to emit. A watcher emits
    /// this the instant SIGTERM arrives — see [`HeuristicSolver::snapshot`].
    /// Seeded with the Chen 2-approx, replaced as the dive improves it.
    incumbent: Arc<Mutex<Vec<Tree>>>,
    /// Caller-supplied soft budget; overrides the `KLADOS_HEUR_TIME_MS` env
    /// fallback (used by the lower-track racer).
    budget_override: Option<Duration>,
}

impl DiveBpSolver {
    pub fn new() -> Self {
        Self {
            terminate: Arc::new(AtomicBool::new(false)),
            stats: SolverStats::default(),
            incumbent: Arc::new(Mutex::new(Vec::new())),
            budget_override: None,
        }
    }

    pub fn set_budget(&mut self, budget: Duration) {
        self.budget_override = Some(budget);
    }

    fn time_budget() -> Option<Duration> {
        std::env::var("KLADOS_HEUR_TIME_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_millis)
    }

    /// Publish `forest` (original labels) as the live incumbent if it is the
    /// first one or strictly smaller than the current.
    fn publish(&self, forest: &[Tree]) {
        if let Ok(mut slot) = self.incumbent.lock() {
            if slot.is_empty() || forest.len() < slot.len() {
                *slot = forest.to_vec();
            }
        }
    }

    /// Expand a reduced-label forest to original labels through the kernel
    /// trace, **repair** it (the unwind's best-effort join can overlap leaves
    /// across components, which the harness rejects — see [`repair_forest`]),
    /// and publish it.
    fn expand_publish(
        &self,
        reduced_forest: &[Tree],
        kern: &KernelizeResult,
        orig: &Instance,
        orig_n: u32,
    ) {
        let expanded =
            expand_solution_unindexed(reduced_forest.to_vec(), kern, &orig.trees[0], orig_n);
        let (expanded, _) = repair_forest(expanded, &orig.trees, orig_n);
        self.publish(&expanded);
    }

    pub fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        let orig_n = instance.num_leaves;
        if instance.num_trees() < 2 {
            return Some(instance.trees.clone());
        }
        if orig_n <= 1 {
            return Some(instance.trees[0..1].to_vec());
        }
        // Specialised for the 2-tree heuristic track.
        if instance.num_trees() != 2 {
            return Some((1..=orig_n).map(|l| Tree::singleton(l, orig_n)).collect());
        }

        let start = Instant::now();
        let budget = self.budget_override.or_else(Self::time_budget);
        let trace = std::env::var("KLADOS_DIVEBP_TRACE").is_ok();

        // ---- Instant valid baseline: Chen 2-approx (original labels) ----
        let (_lo, _up, chen_sets) = chen_pair_agreement(&instance.trees[0], &instance.trees[1]);
        let base = forest_from_sets(&chen_sets, &instance.trees[0], orig_n);
        let (base, _) = repair_forest(base, &instance.trees, orig_n);
        self.publish(&base);

        // ---- Kernelize (optimum-preserving), solve the reduced core, expand ----
        let mut kern_cfg = KernelizeConfig::default();
        if !instance.protected_labels.is_empty() {
            kern_cfg.protected_labels = instance.protected_labels.clone();
        }
        let kern = kernelize_best(instance, &kern_cfg);
        let reduced = &kern.instance;
        if trace {
            eprintln!(
                "[divebp] kernelize {} -> {} leaves ({:.0}ms)",
                orig_n,
                reduced.num_leaves,
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        if reduced.num_leaves <= 1 {
            let rf = if reduced.num_leaves == 0 {
                Vec::new()
            } else {
                vec![reduced.trees[0].clone()]
            };
            let expanded = expand_solution_unindexed(rf, &kern, &instance.trees[0], orig_n);
            let (expanded, _) = repair_forest(expanded, &instance.trees, orig_n);
            self.stats.upper_bound = Some(expanded.len());
            self.publish(&expanded);
            return Some(expanded);
        }

        let deadline = budget.map(|b| start + b);
        let reduced_forest =
            self.dive(reduced, deadline, start, trace, &kern, instance, orig_n);

        let expanded =
            expand_solution_unindexed(reduced_forest, &kern, &instance.trees[0], orig_n);
        let (expanded, _) = repair_forest(expanded, &instance.trees, orig_n);
        self.stats.upper_bound = Some(expanded.len());
        self.publish(&expanded);
        Some(expanded)
    }

    /// The anytime diving branch-and-price over an already-reduced 2-tree
    /// instance. Returns the best forest found over `reduced`'s labels.
    /// Publishes improved incumbents (expanded to original labels) as it goes.
    #[allow(clippy::too_many_arguments)]
    fn dive(
        &self,
        reduced: &Instance,
        deadline: Option<Instant>,
        start: Instant,
        trace: bool,
        kern: &KernelizeResult,
        orig: &Instance,
        orig_n: u32,
    ) -> Vec<Tree> {
        let trees = &reduced.trees;
        let n = reduced.num_leaves;
        let nl = n as usize;

        // ---- Build the initial pool: singletons + Chen components + greedy
        //      seeds. Singletons make the leaf-cover `==1` rows feasible at
        //      every node regardless of branching. ----
        let mut builder = ColumnBuilder::new(trees);
        let mut pool: Vec<AfColumn> = Vec::new();
        let mut seen = ColumnSet::new();
        let push_col =
            |labels: Vec<u32>, pool: &mut Vec<AfColumn>, seen: &mut ColumnSet, b: &mut ColumnBuilder| {
                let mut l = labels;
                l.sort_unstable();
                l.dedup();
                if l.is_empty() || seen.contains(&l) {
                    return;
                }
                if let Some(c) = b.try_build(l.clone(), trees) {
                    seen.insert(l);
                    pool.push(c);
                }
            };
        for leaf in 1..=n {
            push_col(vec![leaf], &mut pool, &mut seen, &mut builder);
        }
        let (chen_lo, _chen_up, chen_sets) = chen_pair_agreement(&trees[0], &trees[1]);
        for s in &chen_sets {
            push_col(s.clone(), &mut pool, &mut seen, &mut builder);
        }
        let num_seeds: u64 = if n <= 2_000 {
            12
        } else if n <= 6_000 {
            5
        } else {
            2
        };
        for seed in 0..num_seeds {
            if self.terminate.load(Ordering::Relaxed) {
                break;
            }
            add_greedy_seed(trees, nl, seed, &mut pool, &mut seen, &mut builder);
        }

        // Best incumbent in REDUCED labels, seeded with the Chen forest.
        let mut best_forest = forest_from_sets(&chen_sets, &trees[0], n);
        let mut best_k = best_forest.len();
        // Sound combinatorial lower bound on the component count: rSPR distance
        // lower bound + 1. Holds globally, so a `chen_lb >= best_k` prune means
        // the incumbent is already optimal.
        let chen_lb = chen_lo + 1;

        // Try the seed pool immediately (largest-first greedy) — often beats Chen.
        self.try_pack(
            &pool, &[], reduced, n, &mut best_forest, &mut best_k, kern, orig, orig_n,
        );

        // ---- Giant path: the dense global pricing table is over cap, but the
        //      RMP LP is sparse and runs fine. Bring the LP duals + the dual-
        //      guided rounding primal to giants via WINDOWED pricing: split T₀
        //      into subtrees small enough to price densely, map the LP duals
        //      into each window, price locally, and lift the columns back to
        //      the reduced label space. Windowed pricing is local (no global
        //      convergence certificate), so we do not branch here — root CG +
        //      LP-rounding pack, re-seeding when CG stalls, until the deadline. -
        let global_fits =
            (trees[0].num_nodes() as u64) * (trees[1].num_nodes() as u64) <= CELL_CAP;
        if !global_fits {
            let window_max = std::env::var("KLADOS_DIVEBP_WINDOW")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(WINDOW_MAX_LEAVES);
            let mut windows = build_windows(reduced, window_max, &self.terminate);
            if trace {
                eprintln!(
                    "[divebp] table over cap (n={}): {} pricing windows (cap={}) k={}",
                    n,
                    windows.len(),
                    window_max,
                    best_k
                );
            }
            if windows.is_empty() {
                // Degenerate (could not window) — keep the seed pack.
                return best_forest;
            }
            let mut rmp = Rmp::new(&pool, trees, nl);
            let mut pricer = ExactPairDpPricer::new(trees);
            let root = Branchings::default();
            let mut seed = num_seeds;
            let mut stale = 0usize;
            while !self.terminate.load(Ordering::Relaxed)
                && !deadline.is_some_and(|d| Instant::now() >= d)
                && best_k > chen_lb
                && pool.len() < POOL_CAP
            {
                rmp.apply_bounds(&pool, &root);
                // Solve the LP, materialising violated node rows.
                let lp = loop {
                    let lp = match rmp.solve() {
                        Ok(lp) => lp,
                        Err(_) => break None,
                    };
                    if rmp.separate_and_add_cuts(&pool, &lp.column_values, 1.0e-6) > 0 {
                        continue;
                    }
                    break Some(lp);
                };
                let Some(lp) = lp else { break };
                // Anytime primal: LP-guided pack of the enriched pool.
                self.try_pack(
                    &pool,
                    &lp.column_values,
                    reduced,
                    n,
                    &mut best_forest,
                    &mut best_k,
                    kern,
                    orig,
                    orig_n,
                );
                if self.terminate.load(Ordering::Relaxed)
                    || deadline.is_some_and(|d| Instant::now() >= d)
                {
                    break;
                }
                // Windowed pricing: enrich the pool with columns priced locally
                // against the current LP duals.
                let priced = self.window_price(
                    &mut windows,
                    &mut pricer,
                    &lp.leaf_duals,
                    &lp.node_duals,
                    &mut pool,
                    &mut seen,
                    &mut builder,
                    &mut rmp,
                    trees,
                    deadline,
                );
                if priced == 0 {
                    // Windowed CG stalled: diversify with a fresh greedy partition.
                    let before = pool.len();
                    let added =
                        add_greedy_seed(trees, nl, seed, &mut pool, &mut seen, &mut builder);
                    seed += 1;
                    if added > 0 {
                        let mut ci = rmp.num_columns();
                        while ci < pool.len() {
                            rmp.add_column(&pool[ci]);
                            ci += 1;
                        }
                    }
                    if pool.len() == before {
                        stale += 1;
                        if stale >= 16 {
                            break; // no new columns from pricing or seeding
                        }
                    } else {
                        stale = 0;
                    }
                }
            }
            if trace {
                eprintln!(
                    "[divebp] giant done k={} pool={} t={:.1}s",
                    best_k,
                    pool.len(),
                    start.elapsed().as_secs_f64()
                );
            }
            return best_forest;
        }

        // ---- RMP + pricer (exact LP gap + duals, lazy node rows) ----
        let mut rmp = Rmp::new(&pool, trees, nl);
        let mut pricer = ExactPairDpPricer::new(trees);
        let mut scratch = PricerScratch::new(trees);
        let mut selector = MostFractionalPair;

        // ---- DFS dive: must-link child first for fast strong incumbents.
        //      Wrapped in an outer loop: when the search tree empties before the
        //      deadline WITHOUT proving optimality (`best_k > chen_lb`), inject a
        //      fresh diversified greedy partition into the warm pool/RMP and
        //      restart from the root rather than returning early and idling the
        //      budget. Stops when proven optimal, out of time, or diversification
        //      stops producing new columns. ----
        let mut stack: Vec<Branchings> = vec![Branchings::default()];
        let mut nodes = 0usize;
        let mut seed_ctr = num_seeds;
        let mut restart_stale = 0usize;
        'outer: loop {
        while let Some(br) = stack.pop() {
            if self.terminate.load(Ordering::Relaxed)
                || deadline.is_some_and(|d| Instant::now() >= d)
            {
                break 'outer;
            }
            nodes += 1;
            rmp.apply_bounds(&pool, &br);

            // ---- Column generation. Each LP iteration also packs the pool
            //      (the paper's LP-guided rounding primal), so the incumbent is
            //      anytime even when CG never converges on a large core — that
            //      is where the deadline-only break previously discarded all of
            //      the column-generation work. ----
            let mut node_converged = false;
            // (objective, column_values) of the last LP solved at this node.
            let mut last: Option<(f64, Vec<f64>)> = None;
            loop {
                let lp = match rmp.solve() {
                    Ok(lp) => lp,
                    // Infeasible under this branching over the current pool —
                    // CG stops and the node is pruned (incumbent unaffected).
                    Err(_) => break,
                };
                // Materialise any violated node `≤1` rows before pricing so the
                // duals reflect the tightened LP, then re-solve.
                if rmp.separate_and_add_cuts(&pool, &lp.column_values, 1.0e-6) > 0 {
                    continue;
                }
                // Anytime primal: pack the (growing) pool at the current LP.
                self.try_pack(
                    &pool,
                    &lp.column_values,
                    reduced,
                    n,
                    &mut best_forest,
                    &mut best_k,
                    kern,
                    orig,
                    orig_n,
                );
                if self.terminate.load(Ordering::Relaxed)
                    || deadline.is_some_and(|d| Instant::now() >= d)
                    || pool.len() >= POOL_CAP
                {
                    last = Some((lp.objective, lp.column_values));
                    break;
                }
                let mut found: Vec<AfColumn> = Vec::new();
                {
                    let ctx = PricingContext {
                        trees,
                        num_leaves: nl,
                        alpha: &lp.leaf_duals,
                        beta: &lp.node_duals,
                        columns: &pool,
                        seen: &seen,
                        branchings: &br,
                        terminate: self.terminate.as_ref(),
                    };
                    for col in scratch.drain_reserve(&ctx, 64) {
                        found.push(col);
                    }
                    match pricer.price(&ctx, &mut scratch) {
                        PricingResult::Found(cols) => found.extend(cols),
                        PricingResult::Converged => node_converged = true,
                        PricingResult::Improving => {}
                    }
                }
                let mut added = 0usize;
                for c in found {
                    if seen.contains(c.labels()) {
                        continue;
                    }
                    seen.insert(c.labels().to_vec());
                    pool.push(c);
                    rmp.add_column(pool.last().unwrap());
                    added += 1;
                }
                if added == 0 {
                    last = Some((lp.objective, lp.column_values));
                    break; // converged or blocked-improving: CG done here
                }
            }

            // Infeasible node (no LP ever solved): pruned.
            let Some((lp_obj, lp_values)) = last else {
                continue;
            };
            // Out of time mid-node: stop the whole search (incumbent is packed).
            if self.terminate.load(Ordering::Relaxed)
                || deadline.is_some_and(|d| Instant::now() >= d)
            {
                break;
            }
            if trace && nodes % 256 == 0 {
                eprintln!(
                    "[divebp] node={} depth={} k={} lp={:.3} pool={} t={:.1}s",
                    nodes,
                    br.depth(),
                    best_k,
                    lp_obj,
                    pool.len(),
                    start.elapsed().as_secs_f64()
                );
            }

            // ---- Bound prune. The LP bound is valid only on a certified
            //      (`Converged`) node; the Chen floor holds unconditionally. ----
            let lp_lb = (lp_obj - 1.0e-6).ceil() as usize;
            let lb = if node_converged {
                lp_lb.max(chen_lb)
            } else {
                chen_lb
            };
            if lb >= best_k {
                continue; // cannot beat the incumbent in this subtree
            }

            // ---- Branch on the most-fractional taxon pair ----
            let children = selector.select(
                &SelectionContext {
                    columns: &pool,
                    values: &lp_values,
                    num_leaves: nl,
                    branchings: &br,
                    current_lp_obj: lp_obj,
                },
                &mut rmp,
            );
            if let Some(ch) = children {
                // `select` returns [must-link, cannot-link]. Push reversed so the
                // must-link child is popped first (deeper, more constraining →
                // reaches an integral leaf faster → stronger early incumbent).
                for child in ch.into_iter().rev() {
                    if !child.is_inconsistent() {
                        stack.push(child);
                    }
                }
            }
            // `None` ⇒ LP support is an integer partition: the node's pack
            // already captured it; nothing to branch on.
        }

        // ---- Search tree exhausted. If the incumbent provably matches the
        //      Chen floor it is optimal — stop. Otherwise diversify and
        //      restart from root rather than idling the remaining budget. ----
        if best_k <= chen_lb
            || self.terminate.load(Ordering::Relaxed)
            || deadline.is_some_and(|d| Instant::now() >= d)
        {
            break 'outer;
        }
        let before = pool.len();
        add_greedy_seed(trees, nl, seed_ctr, &mut pool, &mut seen, &mut builder);
        seed_ctr += 1;
        // Sync newly added pool columns into the warm RMP.
        let mut ci = rmp.num_columns();
        while ci < pool.len() {
            rmp.add_column(&pool[ci]);
            ci += 1;
        }
        if pool.len() == before {
            restart_stale += 1;
            if restart_stale >= 16 {
                break 'outer; // diversification produces no new columns
            }
        } else {
            restart_stale = 0;
        }
        stack.push(Branchings::default());
        } // 'outer

        if trace {
            eprintln!(
                "[divebp] done nodes={} k={} pool={} t={:.1}s",
                nodes,
                best_k,
                pool.len(),
                start.elapsed().as_secs_f64()
            );
        }
        best_forest
    }

    /// Greedy-pack `pool` into a complete valid forest (LP-value then size
    /// order); if it is strictly smaller than `best_k` and validates, adopt it
    /// and publish the expanded forest.
    #[allow(clippy::too_many_arguments)]
    fn try_pack(
        &self,
        pool: &[AfColumn],
        values: &[f64],
        reduced: &Instance,
        n: u32,
        best_forest: &mut Vec<Tree>,
        best_k: &mut usize,
        kern: &KernelizeResult,
        orig: &Instance,
        orig_n: u32,
    ) {
        let packed = greedy_pack(pool, values, &reduced.trees, n);
        if packed.len() < *best_k && validate_agreement_forest(reduced, &packed).is_ok() {
            *best_k = packed.len();
            *best_forest = packed;
            self.expand_publish(best_forest, kern, orig, orig_n);
        }
    }

    /// Price every window against the current LP duals and add the lifted
    /// columns to the (reduced-space) pool and RMP. Returns the number of new
    /// columns added across all windows.
    #[allow(clippy::too_many_arguments)]
    fn window_price(
        &self,
        windows: &mut [Window],
        pricer: &mut ExactPairDpPricer,
        leaf_duals: &[f64],
        node_duals: &[Vec<f64>],
        pool: &mut Vec<AfColumn>,
        seen: &mut ColumnSet,
        builder: &mut ColumnBuilder,
        rmp: &mut Rmp,
        reduced_trees: &[Tree],
        deadline: Option<Instant>,
    ) -> usize {
        let empty = Branchings::default();
        let mut total_added = 0usize;
        for w in windows.iter_mut() {
            if self.terminate.load(Ordering::Relaxed)
                || deadline.is_some_and(|d| Instant::now() >= d)
                || pool.len() >= POOL_CAP
            {
                break;
            }
            let rn = w.inst.num_leaves as usize;
            // Map the reduced-space LP duals into the window's restricted space.
            let mut a_r = vec![0.0f64; rn + 1];
            for rl in 1..=rn {
                a_r[rl] = leaf_duals[w.rev[rl] as usize];
            }
            let mut b_r: Vec<Vec<f64>> =
                w.inst.trees.iter().map(|t| vec![0.0f64; t.num_nodes()]).collect();
            for ti in 0..2 {
                let imgti = &w.img[ti];
                for (node, b) in b_r[ti].iter_mut().enumerate() {
                    let o = imgti[node];
                    if o != NONE {
                        *b = node_duals[ti][o as usize];
                    }
                }
            }
            // Price the window (restricted labels), collect candidate leafsets.
            let mut got: Vec<Vec<u32>> = Vec::new();
            {
                let ctx = PricingContext {
                    trees: &w.inst.trees,
                    num_leaves: rn,
                    alpha: &a_r,
                    beta: &b_r,
                    columns: &[],
                    seen: &w.seen,
                    branchings: &empty,
                    terminate: self.terminate.as_ref(),
                };
                for col in w.scratch.drain_reserve(&ctx, 64) {
                    got.push(col.labels().to_vec());
                }
                if let PricingResult::Found(cols) = pricer.price(&ctx, &mut w.scratch) {
                    for c in cols {
                        got.push(c.labels().to_vec());
                    }
                }
            }
            // Lift restricted labels back to reduced labels and add to the pool.
            for rl_labels in got {
                w.seen.insert(rl_labels.clone());
                let mut lab: Vec<u32> =
                    rl_labels.iter().map(|&rl| w.rev[rl as usize]).collect();
                lab.sort_unstable();
                lab.dedup();
                if lab.len() < 2 || seen.contains(&lab) {
                    continue;
                }
                if let Some(c) = builder.try_build(lab.clone(), reduced_trees) {
                    seen.insert(lab);
                    pool.push(c);
                    rmp.add_column(pool.last().unwrap());
                    total_added += 1;
                }
            }
        }
        total_added
    }
}

impl Default for DiveBpSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// One T₀-subtree pricing window for the giant path: a restricted 2-tree
/// instance plus the maps needed to translate duals in and columns out.
struct Window {
    /// The restricted instance (its leaves are a subtree of T₀, relabelled 1..k).
    inst: Instance,
    /// `rev[restricted_label] = reduced_label`.
    rev: Vec<u32>,
    /// `img[ti][restricted_node] = reduced node` (or [`NONE`]), for mapping β.
    img: Vec<Vec<u32>>,
    /// Per-window pricer scratch (sized to `inst`).
    scratch: PricerScratch,
    /// Per-window dedup set (restricted labels).
    seen: ColumnSet,
}

/// Split T₀ into subtrees of at most `window_max` leaves and build a pricing
/// window for each.
fn build_windows(reduced: &Instance, window_max: usize, terminate: &AtomicBool) -> Vec<Window> {
    let trees = &reduced.trees;
    let nl = reduced.num_leaves as usize;
    let mut windows = Vec::new();
    for leaves in split_t0_windows(&trees[0], window_max) {
        if terminate.load(Ordering::Relaxed) {
            break;
        }
        let mut keep = FixedBitSet::with_capacity(nl + 1);
        for &l in &leaves {
            keep.insert(l as usize);
        }
        let (inst, rev) = restrict_instance_simple(reduced, &keep);
        if inst.num_leaves < 2 || inst.num_trees() != 2 {
            continue;
        }
        let img: Vec<Vec<u32>> = (0..2)
            .map(|ti| node_images(&inst.trees[ti], &trees[ti], &rev))
            .collect();
        let scratch = PricerScratch::new(&inst.trees);
        windows.push(Window {
            inst,
            rev,
            img,
            scratch,
            seen: ColumnSet::new(),
        });
    }
    windows
}

/// Group a `leaf -> component-id` partition into multi-leaf label sets.
fn groups_from_partition(partition: &[usize], nl: usize) -> Vec<Vec<u32>> {
    let mut by_comp: std::collections::HashMap<usize, Vec<u32>> = std::collections::HashMap::new();
    for (i, &comp) in partition.iter().enumerate().take(nl) {
        by_comp.entry(comp).or_default().push((i + 1) as u32);
    }
    by_comp.into_values().filter(|g| g.len() >= 2).collect()
}

/// Build a forest (unindexed, terminal) from a leaf-set partition, filling any
/// uncovered leaf as a singleton.
fn forest_from_sets(sets: &[Vec<u32>], ref_tree: &Tree, n: u32) -> Vec<Tree> {
    let mut forest = Vec::with_capacity(sets.len());
    let mut covered = FixedBitSet::with_capacity(n as usize + 1);
    for s in sets {
        for &l in s {
            covered.insert(l as usize);
        }
        if s.len() == 1 {
            forest.push(Tree::forest_singleton(s[0], n));
        } else {
            let mut bs = FixedBitSet::with_capacity(n as usize + 1);
            for &l in s {
                bs.insert(l as usize);
            }
            forest.push(Tree::forest_component(&bs, ref_tree, n));
        }
    }
    for l in 1..=n {
        if !covered.contains(l as usize) {
            forest.push(Tree::forest_singleton(l, n));
        }
    }
    forest
}

/// Inject one diversified greedy multi-tree partition (both reference trees,
/// the given `seed`) as columns. Returns the number of *new* columns added.
fn add_greedy_seed(
    trees: &[Tree],
    nl: usize,
    seed: u64,
    pool: &mut Vec<AfColumn>,
    seen: &mut ColumnSet,
    builder: &mut ColumnBuilder,
) -> usize {
    let mut added = 0usize;
    for ref_idx in 0..2usize {
        let (_k, part) = greedy_multi_tree_partition(trees, ref_idx, seed);
        for g in groups_from_partition(&part, nl) {
            let mut l = g;
            l.sort_unstable();
            l.dedup();
            if l.len() < 2 || seen.contains(&l) {
                continue;
            }
            if let Some(c) = builder.try_build(l.clone(), trees) {
                seen.insert(l);
                pool.push(c);
                added += 1;
            }
        }
    }
    added
}

/// Greedy node-disjoint, leaf-disjoint packing of `pool`'s multi-leaf columns,
/// with singleton backfill. Columns are ordered by LP value (desc) then size
/// (desc); an empty `values` slice means size-only (used for the seed-pool
/// packs). The result is always a valid agreement forest: every column is a
/// valid component and the disjointness checks make the selection a partition.
fn greedy_pack(pool: &[AfColumn], values: &[f64], trees: &[Tree], n: u32) -> Vec<Tree> {
    let nl = n as usize;
    let mut leaf_used = vec![false; nl + 1];
    let mut node_used: Vec<Vec<bool>> = trees.iter().map(|t| vec![false; t.num_nodes()]).collect();

    let val = |i: usize| -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values[i]
        }
    };
    let mut order: Vec<usize> = (0..pool.len())
        .filter(|&i| pool[i].labels().len() >= 2)
        .collect();
    order.sort_by(|&a, &b| {
        val(b)
            .partial_cmp(&val(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| pool[b].labels().len().cmp(&pool[a].labels().len()))
    });

    let mut forest: Vec<Tree> = Vec::new();
    for ci in order {
        let col = &pool[ci];
        let mut ok = true;
        for &l in col.labels() {
            if leaf_used[l as usize] {
                ok = false;
                break;
            }
        }
        if ok {
            'check: for (t, nodes) in col.coverage().iter_per_tree().enumerate() {
                for &v in nodes {
                    if node_used[t][v] {
                        ok = false;
                        break 'check;
                    }
                }
            }
        }
        if !ok {
            continue;
        }
        let mut bs = FixedBitSet::with_capacity(nl + 1);
        for &l in col.labels() {
            leaf_used[l as usize] = true;
            bs.insert(l as usize);
        }
        for (t, nodes) in col.coverage().iter_per_tree().enumerate() {
            for &v in nodes {
                node_used[t][v] = true;
            }
        }
        forest.push(Tree::forest_component(&bs, &trees[0], n));
    }
    for l in 1..=nl {
        if !leaf_used[l] {
            forest.push(Tree::forest_singleton(l as u32, n));
        }
    }
    forest
}

impl HeuristicSolver for DiveBpSolver {
    fn name(&self) -> &'static str {
        "divebp"
    }

    fn description(&self) -> &'static str {
        "Diving branch-and-price (exact LP duals + Ryan–Foster branching, anytime)"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("KLADOS_HEUR_TIME_MS", "wall-time budget in ms (default: SIGTERM-driven)"),
            ("KLADOS_DIVEBP_TRACE", "print per-node diagnostics"),
        ]
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        DiveBpSolver::solve(self, instance)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }

    fn sigterm_handler(&self) {
        self.terminate.store(true, Ordering::SeqCst);
    }

    fn snapshot(&self) -> Option<Vec<Tree>> {
        match self.incumbent.lock() {
            Ok(slot) if !slot.is_empty() => Some(slot.clone()),
            _ => None,
        }
    }
}
