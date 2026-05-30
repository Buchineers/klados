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

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use fixedbitset::FixedBitSet;
use fxhash::FxHashSet;
use klados_core::tree::{NONE, NodeId, Tree};
use klados_core::{Instance, SolverStats};
use klados_exact::bp::column::{AfColumn, ColumnBuilder, ColumnSet, is_valid_af_component};
use klados_exact::bp::pricer::{ExactPairDpPricer, Pricer, PricerScratch, PricingContext, PricingResult};
use klados_exact::bp::rmp::Rmp;
use klados_exact::bp::search::{Branchings, LeafPair};
use klados_exact::chen_rspr::chen_pair_agreement;

use crate::HeuristicSolver;

const POOL_HARD_CAP: usize = 120_000;
const POOL_PRUNE_TO: usize = 80_000;
/// Only attempt the certifying MIP when the incumbent is within this many
/// components of the LP bound (a wide gap won't close and risks a HiGHS
/// time-limit overrun that blows the SIGTERM grace window).
const MIP_GAP_LIMIT: usize = 4;

/// Safe ceiling on the anchor DP's dense `n₀·n₁` table (kept under the
/// pricer's own ~64M-cell cap). Above this we price in tree-local windows.
const CELL_CAP_SAFE: u64 = 60_000_000;
/// Max leaves per T₀-subtree pricing window. `(2·W)² ≤ CELL_CAP_SAFE` so each
/// window's restricted DP fits; per-window cache ≈ 32·(2W)² bytes.
const WINDOW_MAX_LEAVES: usize = 1_200;

/// A validated agreement-forest column with its per-tree V-set internal nodes.
struct Block {
    labels: Vec<u32>,
    weight: usize, // |labels| - 1
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

pub struct LagrangianSolver {
    terminate: Arc<AtomicBool>,
    stats: SolverStats,
}

impl LagrangianSolver {
    pub fn new() -> Self {
        Self {
            terminate: Arc::new(AtomicBool::new(false)),
            stats: SolverStats::default(),
        }
    }

    /// Optional soft budget. `None` means run until SIGTERM (the real mode);
    /// `KLADOS_HEUR_TIME_MS` sets a budget only for local testing.
    fn time_budget() -> Option<Duration> {
        std::env::var("KLADOS_HEUR_TIME_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_millis)
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
        let budget = Self::time_budget();
        let trace = std::env::var("KLADOS_LAGR_TRACE").is_ok();

        // ---- Kernelize first (optimality-preserving), solve the reduced core,
        //      expand at the end. Shrinks the instance so global pricing fits
        //      more often and the pool is over the conflict core, not agreeing
        //      pendant structure. ----
        let mut kern_cfg = klados_core::kernelize::KernelizeConfig::default();
        if !instance.protected_labels.is_empty() {
            kern_cfg.protected_labels = instance.protected_labels.clone();
        }
        let kern = klados_core::kernelize::kernelize_best(instance, &kern_cfg);
        let reduced = &kern.instance;
        if trace {
            eprintln!(
                "[lagr] kernelize {} -> {} leaves ({:.0}ms)",
                orig_n,
                reduced.num_leaves,
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        if reduced.num_leaves <= 1 {
            let reduced_forest = if reduced.num_leaves == 0 {
                Vec::new()
            } else {
                vec![reduced.trees[0].clone()]
            };
            let expanded = klados_core::kernelize::expand_solution(
                reduced_forest,
                &kern,
                &instance.trees[0],
                orig_n,
            );
            self.stats.upper_bound = Some(expanded.len());
            return Some(expanded);
        }

        let trees = &reduced.trees;
        let n = reduced.num_leaves;
        let nl = n as usize;

        // ---- Tier cascade ----
        // When global pricing fits, the warm exact-LP RMP proves small/integral
        // instances in milliseconds (bp's small-instance speed). Try it first
        // with a capped budget; if it certifies optimality, return. Otherwise
        // (integrality gap, or it didn't converge in the cap) fall through to
        // the subgradient anytime solver, which wins the primal on gap/large
        // instances via its diverse pool. Routing is by PROVABILITY, not size.
        let global_fits =
            (reduced.trees[0].num_nodes() as u64) * (reduced.trees[1].num_nodes() as u64)
                <= CELL_CAP_SAFE;
        let force_lp = std::env::var("KLADOS_LAGR_LP").is_ok();
        let no_rmp = std::env::var("KLADOS_NO_RMP").is_ok();
        if (global_fits || force_lp) && !no_rmp {
            // Cap the RMP attempt so the subgradient ALWAYS gets the bulk of the
            // budget. Proving happens fast or not at all (n=60, pub049 prove in
            // <1s); a small-but-gappy instance can otherwise run CG for the whole
            // window, starving the subgradient (which wins the primal on gap
            // instances) AND risking the SIGTERM grace. So: cap at a modest
            // absolute ceiling, and at ≤¼ of a known budget. KLADOS_LAGR_LP
            // forces the full budget (the standalone RMP-engine path, testing).
            let cap_ceiling = std::env::var("KLADOS_RMP_CAP_MS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .map(Duration::from_millis)
                .unwrap_or_else(|| Duration::from_secs(15));
            let rmp_deadline = if force_lp {
                budget.map(|b| start + b)
            } else {
                match budget {
                    Some(b) => Some(start + (b / 4).min(cap_ceiling)),
                    None => Some(start + cap_ceiling),
                }
            };
            let (reduced_forest, rmp_proved) =
                self.solve_rmp(reduced, rmp_deadline, trace, start);
            if force_lp || rmp_proved {
                let expanded = klados_core::kernelize::expand_solution(
                    reduced_forest,
                    &kern,
                    &instance.trees[0],
                    orig_n,
                );
                let (expanded, _) = repair_forest(expanded, &instance.trees, orig_n);
                self.stats.upper_bound = Some(expanded.len());
                return Some(expanded);
            }
            if trace {
                eprintln!(
                    "[lagr] RMP did not certify (gap) — handing off to subgradient at {:.1}s",
                    start.elapsed().as_secs_f64()
                );
            }
        }

        // ---- Pool + dedup ----
        let mut pool: Vec<Block> = Vec::new();
        let mut seen = ColumnSet::new();
        let mut add_block = |labels: Vec<u32>, pool: &mut Vec<Block>, seen: &mut ColumnSet| -> bool {
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
        let mut best_forest = forest_from_partition(&chen_sets, trees, n);
        let mut best_components = best_forest.len();
        for s in &chen_sets {
            add_block(s.clone(), &mut pool, &mut seen);
        }
        self.stats.upper_bound = Some(best_components);
        if trace {
            eprintln!("[lagr] n={} chen incumbent={} ({:.0}ms)", n, best_components, start.elapsed().as_secs_f64() * 1000.0);
        }

        // ---- Seed pool with a few overlapping greedy partitions ----
        let num_seeds: u64 = if n <= 2_000 { 12 } else if n <= 6_000 { 5 } else { 2 };
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
        if trace {
            eprintln!("[lagr] seeded pool={} ({:.0}ms)", pool.len(), start.elapsed().as_secs_f64() * 1000.0);
        }

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
        let window_max = std::env::var("KLADOS_LAGR_WINDOW")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(WINDOW_MAX_LEAVES);
        let mut windows: Vec<Window> = Vec::new();
        if !global_fits {
            for leaves in split_t0_windows(&trees[0], window_max) {
                if self.terminate.load(Ordering::Relaxed) {
                    break;
                }
                let mut keep = FixedBitSet::with_capacity(nl + 1);
                for &l in &leaves {
                    keep.insert(l as usize);
                }
                let (inst, rev) =
                    klados_core::kernelize::restrict_instance_simple(reduced, &keep);
                if inst.num_leaves < 2 || inst.num_trees() != 2 {
                    continue;
                }
                let img: Vec<Vec<u32>> = (0..2)
                    .map(|ti| node_images(&inst.trees[ti], &trees[ti], &rev))
                    .collect();
                let scratch_w = PricerScratch::new(&inst.trees);
                windows.push(Window {
                    inst,
                    rev,
                    img,
                    scratch: scratch_w,
                    seen: ColumnSet::new(),
                });
            }
            if trace {
                let sizes: Vec<usize> = windows.iter().map(|w| w.inst.num_leaves as usize).collect();
                let (mn, mx) = (
                    sizes.iter().copied().min().unwrap_or(0),
                    sizes.iter().copied().max().unwrap_or(0),
                );
                let avg = if sizes.is_empty() { 0 } else { sizes.iter().sum::<usize>() / sizes.len() };
                eprintln!(
                    "[lagr] windowed pricing: {} windows (cap={}, leaves min/avg/max={}/{}/{}) ({:.0}ms)",
                    windows.len(),
                    window_max,
                    mn,
                    avg,
                    mx,
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
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
        // Volume-algorithm buffers (experimental).
        let volume = std::env::var("KLADOS_LAGR_VOLUME").is_ok();
        let avg_a = std::env::var("KLADOS_LAGR_VOLUME_A")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.2);
        let mut v_cov = vec![0.0f64; nl + 1];
        let mut v_use: Vec<Vec<f64>> =
            trees.iter().map(|t| vec![0.0f64; t.num_nodes()]).collect();

        // ---- Hybrid: refresh subgradient duals from a warm exact RMP ----
        // The subgradient's oscillating duals build a DIVERSE pool (which wins
        // the integer primal on gap instances), but its dual *center* drifts.
        // Periodically solve the exact LP over the current pool and overwrite
        // α/β with the LP duals: the pricer + greedy then aim at the true LP
        // optimum while the subgradient keeps diversifying around it.
        let hybrid = std::env::var("KLADOS_LAGR_HYBRID").is_ok();
        let refresh_every = std::env::var("KLADOS_LAGR_REFRESH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(15)
            .max(1);
        let mut h_builder = ColumnBuilder::new(trees);
        let mut h_afpool: Vec<AfColumn> = Vec::new();
        let mut h_in_rmp = ColumnSet::new();
        let mut h_rmp: Option<Rmp> = None;

        // First primal from the seed pool (dual-guided with the initial α=1).
        {
            let scores: Vec<f64> = pool.iter().map(|b| block_score(b, &alpha, &beta)).collect();
            self.try_primal(trees, n, &pool, &scores, &mut best_forest, &mut best_components);
        }

        loop {
            if self.terminate.load(Ordering::Relaxed)
                || budget.is_some_and(|b| start.elapsed() >= b)
            {
                break;
            }
            iter += 1;

            // ---- Price at current duals (drain banked reserve first) ----
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
                        || budget.is_some_and(|b| start.elapsed() >= b)
                    {
                        break;
                    }
                    let rn = w.inst.num_leaves as usize;
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
                    let got: Vec<Vec<u32>> = {
                        let ctx = PricingContext {
                            trees: &w.inst.trees,
                            num_leaves: rn,
                            alpha: &a_r,
                            beta: &b_r,
                            columns: &[],
                            seen: &w.seen,
                            branchings: &branchings,
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
                    for rl_labels in got {
                        w.seen.insert(rl_labels.clone());
                        new_cols.push(rl_labels.iter().map(|&rl| w.rev[rl as usize]).collect());
                    }
                }
            }
            let mut added = 0usize;
            for c in new_cols {
                if add_block(c, &mut pool, &mut seen) {
                    added += 1;
                }
            }
            if pool.len() > POOL_HARD_CAP {
                prune_pool(&mut pool, &alpha, &beta, POOL_PRUNE_TO);
            }

            // ---- Hybrid dual refresh: overwrite α/β with exact LP duals ----
            if hybrid && iter % refresh_every == 0 {
                // Sync the warm RMP with any blocks not yet in it (singletons
                // first, for leaf-row =1 feasibility). Pruned blocks already in
                // the RMP stay there — extra columns only sharpen the duals.
                let need_init = h_rmp.is_none();
                if need_init {
                    for l in 1..=n {
                        if h_in_rmp.insert(vec![l]) {
                            if let Some(c) = h_builder.try_build(vec![l], trees) {
                                h_afpool.push(c);
                            }
                        }
                    }
                }
                let mut fresh: Vec<AfColumn> = Vec::new();
                for b in &pool {
                    if b.labels.len() >= 2 && h_in_rmp.insert(b.labels.clone()) {
                        if let Some(c) = h_builder.try_build(b.labels.clone(), trees) {
                            fresh.push(c);
                        }
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
                            let cuts = rmp.separate_and_add_cuts(&h_afpool, &sol.column_values, 1e-6);
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
                        let blend = std::env::var("KLADOS_LAGR_HYBRID_BLEND")
                            .ok()
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(1.0)
                            .clamp(0.0, 1.0);
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
                        if trace {
                            eprintln!(
                                "[lagr][hybrid] refresh iter={} rmp_cols={} lp={:.2} best={} t={:.1}s",
                                iter, h_afpool.len(), sol.objective, best_components,
                                start.elapsed().as_secs_f64()
                            );
                        }
                    }
                }
            }

            // Score every block once per round (against the current duals) and
            // reuse it for both the subgradient and the packing — avoids the
            // O(P·log P) score re-evaluation that dominated each round at scale.
            let scores: Vec<f64> = pool.iter().map(|b| block_score(b, &alpha, &beta)).collect();

            // ---- Dual multiplier update (subgradient, or volume) over the pool ----
            let lb = if volume {
                self.volume_step(
                    trees, nl, &pool, &scores, &mut alpha, &mut beta,
                    &mut v_cov, &mut v_use, lambda, avg_a, best_components, iter == 1,
                )
            } else {
                self.subgradient_step(trees, nl, &pool, &scores, &mut alpha, &mut beta, lambda, best_components)
            };
            if lb > best_lb + 1e-6 {
                best_lb = lb;
                stall = 0;
            } else {
                stall += 1;
                let stall_thresh = std::env::var("KLADOS_LAGR_STALL")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(20);
                if stall >= stall_thresh {
                    lambda *= 0.5;
                    stall = 0;
                }
            }

            // ---- Dual-guided primal ----
            let improved =
                self.try_primal(trees, n, &pool, &scores, &mut best_forest, &mut best_components);
            if improved {
                self.stats.upper_bound = Some(best_components);
            }

            if trace && (iter <= 5 || iter % 25 == 0 || improved) {
                eprintln!(
                    "[lagr] iter={} pool={} +{} lb={:.1} lambda={:.4} best={} gap={:.1}% t={:.1}s",
                    iter, pool.len(), added, lb, lambda, best_components,
                    if lb > 0.0 { 100.0 * (best_components as f64 - lb) / lb } else { f64::NAN },
                    start.elapsed().as_secs_f64(),
                );
            }

            // Terminate ONLY when the optimum is proven: global pricing with
            // the complete column set (added == 0) gives a valid LB, and the
            // incumbent meets it. (OPT ≥ ⌈lb⌉ and best ≥ OPT, so best ≤ ⌈lb⌉
            // ⇒ best = OPT.) Windowed pricing never certifies.
            if global && added == 0 && best_components <= lb.ceil() as usize {
                if trace {
                    eprintln!(
                        "[lagr] PROVED optimal at iter={}: best={} lb={:.2}",
                        iter, best_components, lb
                    );
                }
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
                lambda = 1.0;
                for a in alpha.iter_mut().skip(1) {
                    *a = 0.5 * *a + 0.5;
                }
                no_new = 0;
                if trace {
                    eprintln!("[lagr] re-energise at iter={} (unproven, best={})", iter, best_components);
                }
            }
        }

        // ---- Branching-lite (prototype, gated by KLADOS_LAGR_BRANCH) ----
        // When the bound can't prove the incumbent (LP↔IP integrality gap), the
        // unconstrained pricer never generates the columns the optimum needs.
        // Branch on contended leaf-pairs: force {a,b} together (must-link),
        // RE-PRICE under that constraint so the anchor DP emits {a,b}-together
        // columns (the gap columns), then re-pack. Keep any improvement.
        if global && !proved && std::env::var("KLADOS_LAGR_BRANCH").is_ok() {
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
                        if comp_of[a as usize] != comp_of[b as usize]
                            && seen_pairs.insert((a, b))
                        {
                            pairs.push((a, b));
                            break 'find;
                        }
                    }
                }
            }
            let n_pairs = pairs.len();
            for (a, b) in pairs {
                if self.terminate.load(Ordering::Relaxed)
                    || budget.is_some_and(|bd| start.elapsed() >= bd)
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
                    best_forest = build_forest(&pool, &sel, trees, n);
                    if trace {
                        eprintln!(
                            "[lagr] branch must-link({},{}) improved: best={}",
                            a, b, comps
                        );
                    }
                }
            }
            if trace {
                eprintln!(
                    "[lagr] branching-lite: tried {} pairs, best={} pool={}",
                    n_pairs, best_components, pool.len()
                );
            }
        }

        if trace {
            eprintln!(
                "[lagr] DONE reduced_n={} reduced_best={} lb={:.1} iters={} pool={} t={:.1}s",
                n, best_components, best_lb, iter, pool.len(), start.elapsed().as_secs_f64()
            );
        }
        // Expand the reduced-instance forest back to the original instance.
        let expanded = klados_core::kernelize::expand_solution(
            best_forest,
            &kern,
            &instance.trees[0],
            orig_n,
        );
        let (expanded, exploded) = repair_forest(expanded, &instance.trees, orig_n);
        if trace {
            eprintln!(
                "[lagr] expanded to {} components (orig n={}, repaired={})",
                expanded.len(), orig_n, exploded
            );
        }
        self.stats.upper_bound = Some(expanded.len());
        Some(expanded)
    }

    /// Warm-started exact-LP column generation (bp's `Rmp`). Each iteration
    /// solves the restricted-master LP exactly (→ exact duals), lazily
    /// separates node `≤1` rows, prices at those duals, and extracts an
    /// integral primal (MIP at convergence, greedy interim — both validated
    /// node-disjoint). This converges in B&P-class iteration counts.
    fn solve_rmp(
        &self,
        reduced: &Instance,
        deadline: Option<Instant>,
        trace: bool,
        start: Instant,
    ) -> (Vec<Tree>, bool) {
        let trees = &reduced.trees;
        let n = reduced.num_leaves;
        let nl = n as usize;
        let mut proved = false;

        let (_lo, _up, chen_sets) = chen_pair_agreement(&trees[0], &trees[1]);
        let mut best_forest = forest_from_partition(&chen_sets, trees, n);
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
        let mut add_labels =
            |labels: Vec<u32>, pool: &mut Vec<AfColumn>, seen: &mut ColumnSet, builder: &mut ColumnBuilder| {
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
        let num_seeds: u64 = if n <= 2_000 { 12 } else if n <= 6_000 { 5 } else { 2 };
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
        let window_max = std::env::var("KLADOS_LAGR_WINDOW")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(WINDOW_MAX_LEAVES);
        let mut windows: Vec<Window> = Vec::new();
        if !global_fits {
            for leaves in split_t0_windows(&trees[0], window_max) {
                if self.terminate.load(Ordering::Relaxed) {
                    break;
                }
                let mut keep = FixedBitSet::with_capacity(nl + 1);
                for &l in &leaves {
                    keep.insert(l as usize);
                }
                let (inst, rev) =
                    klados_core::kernelize::restrict_instance_simple(reduced, &keep);
                if inst.num_leaves < 2 || inst.num_trees() != 2 {
                    continue;
                }
                let img: Vec<Vec<u32>> = (0..2)
                    .map(|ti| node_images(&inst.trees[ti], &trees[ti], &rev))
                    .collect();
                let scratch_w = PricerScratch::new(&inst.trees);
                windows.push(Window {
                    inst,
                    rev,
                    img,
                    scratch: scratch_w,
                    seen: ColumnSet::new(),
                });
            }
        }
        // The RMP LP bound is a valid global lower bound only when pricing is
        // global (the DP sees the whole column space). Windowed pricing is
        // local, so its converged objective is not an optimality certificate.
        let global = windows.is_empty();
        if trace {
            eprintln!(
                "[lagr][rmp] n={} chen={} pool={} pricing={} ({:.0}ms)",
                n,
                best_components,
                pool.len(),
                if global { "global".to_string() } else { format!("windowed({})", windows.len()) },
                start.elapsed().as_secs_f64() * 1000.0
            );
        }

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
            if let Some((forest, comps)) = greedy_pack_af(&pool, &scores, trees, n) {
                if comps < best_components {
                    best_components = comps;
                    best_forest = forest;
                }
            }

            if trace && (iter <= 5 || iter % 10 == 0 || added == 0) {
                eprintln!(
                    "[lagr][rmp] iter={} cols={} +{} lp={:.2} best={} gap={:.1}% t={:.1}s",
                    iter, pool.len(), added, sol.objective, best_components,
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
                {
                    if let Ok(Some(mip)) = rmp.solve_mip_with_time_limit(0.5) {
                        if let Some((forest, comps)) =
                            forest_from_lp(&pool, &mip.column_values, trees, n)
                        {
                            if comps < best_components {
                                best_components = comps;
                                best_forest = forest;
                            }
                        }
                    }
                }
                // Global pricing with a complete pool ⇒ sol.objective is a valid
                // LB. best ≤ ⌈lb⌉ certifies optimality. Windowed never certifies.
                proved = global && best_components <= lb;
                if trace {
                    let status = if !global {
                        "(windowed)"
                    } else if proved {
                        "PROVED"
                    } else {
                        "(gap)"
                    };
                    eprintln!(
                        "[lagr][rmp] CG converged iter={} lp={:.3} best={} {}",
                        iter, sol.objective, best_components, status
                    );
                }
                break;
            }
        }
        (best_forest, proved)
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
            *best_forest = build_forest(pool, &selected, trees, n);
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
    #[allow(clippy::too_many_arguments)]
    fn volume_step(
        &self,
        trees: &[Tree],
        nl: usize,
        pool: &[Block],
        scores: &[f64],
        alpha: &mut [f64],
        beta: &mut [Vec<f64>],
        v_cov: &mut [f64],
        v_use: &mut [Vec<f64>],
        lambda: f64,
        avg_a: f64,
        ub_components: usize,
        init: bool,
    ) -> f64 {
        // Instantaneous Lagrangian subproblem solution x̂ (columns with rc<0).
        let mut cov = vec![0.0f64; nl + 1];
        let mut use_nodes: Vec<Vec<f64>> =
            trees.iter().map(|t| vec![0.0f64; t.num_nodes()]).collect();
        let mut sum_rc = 0.0f64;
        for (i, b) in pool.iter().enumerate() {
            if scores[i] > 1.0 {
                sum_rc += 1.0 - scores[i];
                for &l in &b.labels {
                    cov[l as usize] += 1.0;
                }
                for (t, nodes) in b.cover.iter().enumerate() {
                    for &v in nodes {
                        use_nodes[t][v as usize] += 1.0;
                    }
                }
            }
        }
        for l in 1..=nl {
            if alpha[l] > 1.0 {
                sum_rc += 1.0 - alpha[l];
                cov[l] += 1.0;
            }
        }
        let sum_alpha: f64 = alpha[1..=nl].iter().sum();
        let sum_beta: f64 = beta.iter().flat_map(|b| b.iter()).sum();
        let lagrangian = sum_alpha - sum_beta + sum_rc;

        // Primal estimate update: v ← a·x̂ + (1−a)·v. Seed v = x̂ on the first
        // step so there's no startup lag (a zero seed inflates α uniformly).
        let a = if init { 1.0 } else { avg_a };
        for l in 0..=nl {
            v_cov[l] = a * cov[l] + (1.0 - a) * v_cov[l];
        }
        for (t, tree) in trees.iter().enumerate() {
            for v in 0..tree.num_nodes() {
                v_use[t][v] = a * use_nodes[t][v] + (1.0 - a) * v_use[t][v];
            }
        }

        // Subgradient from the averaged primal estimate.
        let mut gnorm2 = 0.0f64;
        for l in 1..=nl {
            let g = 1.0 - v_cov[l];
            gnorm2 += g * g;
        }
        for (t, tree) in trees.iter().enumerate() {
            for v in 0..tree.num_nodes() {
                if tree.is_leaf(v as u32) {
                    continue;
                }
                let g = v_use[t][v] - 1.0;
                gnorm2 += g * g;
            }
        }
        if gnorm2 < 1e-12 {
            return lagrangian.max(0.0);
        }
        let target = (ub_components as f64 - lagrangian).max(0.5);
        let step = lambda * target / gnorm2;
        for l in 1..=nl {
            alpha[l] += step * (1.0 - v_cov[l]);
        }
        for (t, tree) in trees.iter().enumerate() {
            for v in 0..tree.num_nodes() {
                if tree.is_leaf(v as u32) {
                    continue;
                }
                let nv = beta[t][v] + step * (v_use[t][v] - 1.0);
                beta[t][v] = nv.max(0.0);
            }
        }
        lagrangian.max(0.0)
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
}

/// Split T₀ into leaf groups, each a subtree with ≤ `max_leaves` leaves.
/// Each group's leaves form a connected subtree in T₀, so any agreement
/// component fully inside the group is findable by the restricted DP.
fn split_t0_windows(tree: &Tree, max_leaves: usize) -> Vec<Vec<u32>> {
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
fn prune_pool(pool: &mut Vec<Block>, alpha: &[f64], beta: &[Vec<f64>], keep: usize) {
    pool.sort_unstable_by(|a, b| {
        block_score(b, alpha, beta)
            .total_cmp(&block_score(a, alpha, beta))
            .then_with(|| b.weight.cmp(&a.weight))
    });
    pool.truncate(keep);
}

fn groups_from_partition(partition: &[usize], nl: usize) -> Vec<Vec<u32>> {
    let mut by_comp: std::collections::HashMap<usize, Vec<u32>> = std::collections::HashMap::new();
    for (i, &comp) in partition.iter().enumerate().take(nl) {
        by_comp.entry(comp).or_default().push((i + 1) as u32);
    }
    by_comp.into_values().filter(|g| g.len() >= 2).collect()
}

fn forest_from_partition(sets: &[Vec<u32>], trees: &[Tree], n: u32) -> Vec<Tree> {
    let mut forest = Vec::with_capacity(sets.len());
    let mut covered = FixedBitSet::with_capacity(n as usize + 1);
    for s in sets {
        for &l in s {
            covered.insert(l as usize);
        }
        if s.len() == 1 {
            forest.push(Tree::singleton(s[0], n));
        } else {
            let mut bs = FixedBitSet::with_capacity(n as usize + 1);
            for &l in s {
                bs.insert(l as usize);
            }
            forest.push(Tree::component_from_leafset(&bs, &trees[0], n));
        }
    }
    for l in 1..=n {
        if !covered.contains(l as usize) {
            forest.push(Tree::singleton(l, n));
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
) -> Option<(Vec<Tree>, usize)> {
    let mut order: Vec<usize> = (0..pool.len()).filter(|&i| pool[i].labels().len() >= 2).collect();
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
                if used[t].contains(v as usize) {
                    continue 'cand;
                }
            }
        }
        for (t, nodes) in cov.iter_per_tree().enumerate() {
            for &v in nodes {
                used[t].insert(v as usize);
            }
        }
        let mut bs = FixedBitSet::with_capacity(n as usize + 1);
        for &l in pool[idx].labels() {
            bs.insert(l as usize);
            covered.insert(l as usize);
        }
        forest.push(Tree::component_from_leafset(&bs, &trees[0], n));
    }
    for l in 1..=n {
        if !covered.contains(l as usize) {
            forest.push(Tree::singleton(l, n));
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
                if used[t].contains(v as usize) {
                    return None; // node conflict ⇒ not a valid AF
                }
                used[t].insert(v as usize);
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
        forest.push(Tree::component_from_leafset(&bs, &trees[0], n));
    }
    for l in 1..=n {
        if !covered.contains(l as usize) {
            forest.push(Tree::singleton(l, n));
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
    let mut out: Vec<Tree> = Vec::with_capacity(forest.len());
    let mut forest: Vec<Option<Tree>> = forest.into_iter().map(Some).collect();
    let mut exploded = 0usize;

    'comp: for &i in &order {
        let labels = &comp_leaves[i];
        if labels.len() < 2 {
            out.push(forest[i].take().unwrap());
            continue;
        }
        // Must be a genuine agreement component, and node-disjoint from kept.
        let valid = is_valid_af_component(labels, trees);
        if valid {
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
                out.push(forest[i].take().unwrap());
                continue 'comp;
            }
        }
        // Offender: explode into singletons (always valid, never overlaps).
        for &l in labels {
            out.push(Tree::singleton(l, n));
        }
        exploded += 1;
    }
    (out, exploded)
}

fn build_forest(pool: &[Block], selected: &[usize], trees: &[Tree], n: u32) -> Vec<Tree> {
    let mut forest = Vec::with_capacity(selected.len());
    let mut covered = FixedBitSet::with_capacity(n as usize + 1);
    for &i in selected {
        let b = &pool[i];
        let mut bs = FixedBitSet::with_capacity(n as usize + 1);
        for &l in &b.labels {
            bs.insert(l as usize);
            covered.insert(l as usize);
        }
        forest.push(Tree::component_from_leafset(&bs, &trees[0], n));
    }
    for l in 1..=n {
        if !covered.contains(l as usize) {
            forest.push(Tree::singleton(l, n));
        }
    }
    forest
}

impl Default for LagrangianSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl HeuristicSolver for LagrangianSolver {
    fn name(&self) -> &'static str {
        "lagrangian"
    }

    fn description(&self) -> &'static str {
        "Dual-guided set-packing (Lagrangian column generation, anytime)"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("KLADOS_HEUR_TIME_MS", "wall-time budget in ms (default 290000)"),
            ("KLADOS_LAGR_TRACE", "print per-iteration diagnostics"),
        ]
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        LagrangianSolver::solve(self, instance)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }

    fn sigterm_handler(&self) {
        self.terminate.store(true, Ordering::SeqCst);
    }
}
