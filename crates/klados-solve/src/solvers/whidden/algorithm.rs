//! Core branch-and-bound for 2-tree rSPR distance.
//!
//! Faithful port of rspr's rSPR_branch_and_bound_hlpr.
//!
//! Flow:
//!   1. Process singletons (free: no k decrement)
//!   2. Find sibling pair in T1
//!   3. Case 2: pair matches in T2 → contract (free)
//!   4. Case 3: optional BB prune check, then 3-way branch (k-1 each)

use std::time::Instant;

use fixedbitset::FixedBitSet;
use klados_core::tree::{Label, NONE, NodeId, Tree};
use klados_core::{Instance, SolverStats};

use super::stats::{WhiddenProgressUpdate, WhiddenRuleStats};
use klados_core::lower_bound::{cherry_reduce_ub, red_blue_approx_detailed};
use klados_core::twin_tree::{T1, T2, TwinForest, UndoMachine};
use klados_core::twin_tree::{approx2, undo};

// ---------------------------------------------------------------------------
// Configuration — maps to rspr's optimization flags
// ---------------------------------------------------------------------------

/// Controls which rspr optimizations are active.
///
/// Flag names and semantics match rspr's globals.
/// `default()` enables the same set as rspr's `-allopt` (DEFAULT_OPTIMIZATIONS).
#[derive(Clone, Debug)]
pub struct BBConfig {
    // --- Approximation-based pruning ---
    /// BB: prune branches where 3-approx > 3k (rspr's `BB` flag).
    pub bb: bool,
    /// BB2: use Olver 2-approx dual LB instead of 3-approx for BB pruning.
    /// Prune when approx_2_lb(tf) > k (strictly tighter than approx_3 > 3k).
    pub bb_2approx: bool,

    // --- Branching reductions (reduce 3-way to fewer branches) ---
    /// COB: "cut one B" — skip branch A when T2_ab and T2_c are siblings.
    pub cut_one_b: bool,
    /// RCOB: "reverse cut one B" — skip branch C via uncle check.
    pub reverse_cut_one_b: bool,
    /// RCOB3: variant of reverse_cut_one_b (rspr's REVERSE_CUT_ONE_B_3).
    pub reverse_cut_one_b_3: bool,
    /// C2B: "cut two B" — uncle-sibling leaf check.
    pub cut_two_b: bool,
    /// CAB: "cut all B" — when safe, only branch on B.
    pub cut_all_b: bool,
    /// SC: "separate components" — cut A and C when they're in separate components.
    pub cut_ac_separate_components: bool,

    // --- Edge protection ---
    /// EP: protect edges in T2 from being cut (reduces branching).
    pub edge_protection: bool,
    /// EP2B: edge protection variant for two-B case.
    pub edge_protection_two_b: bool,

    // --- Sibling pair ordering ---
    /// DPO: prefer deepest protected sibling pair.
    pub deepest_protected_order: bool,
    /// DO: prefer deepest sibling pair (tiebreaker).
    pub deepest_order: bool,
    /// Near-preorder traversal for sibling pair enumeration.
    pub near_preorder_sibling_pairs: bool,

    // --- Leaf reduction ---
    /// LR: leaf reduction rule (rspr's LEAF_REDUCTION).
    pub leaf_reduction: bool,
    /// LR2: second leaf reduction rule (rspr's LEAF_REDUCTION2).
    pub leaf_reduction2: bool,

    // --- Approximation optimizations (used inside approx_3) ---
    /// Approx COB: cut-one-B inside 3-approximation.
    pub approx_cut_one_b: bool,
    /// Approx C2B: cut-two-B inside 3-approximation.
    pub approx_cut_two_b: bool,
    /// Approx RCOB: reverse-cut-one-B inside 3-approximation.
    pub approx_reverse_cut_one_b: bool,

    // --- Prefer rho (cluster-related) ---
    /// PREFER_RHO: prefer the rho component for sibling pair search.
    pub prefer_rho: bool,
    /// Prefer non-branching sibling pairs (Case 2 over Case 3).
    pub prefer_nonbranching: bool,

    // --- Transposition table ---
    /// TT: allocate and maintain transposition table.
    pub tt_enabled: bool,
    /// TT_PRUNE: actually prune on TT hit (vs observe-only).
    pub tt_prune: bool,
    /// TT size as power of 2 (default: 23 = 8M entries ≈ 96MB).
    pub tt_size_log2: u8,

    // --- Exact bound cache ---
    /// Cache exact approx_3 / approx_2_lb values by state hash.
    pub bound_cache_enabled: bool,
    /// Bound-cache size as power of 2.
    pub bound_cache_size_log2: u8,

    // --- Experimental rooted split-or-decompose rescue ---
    /// Apply a narrow version of Mestel et al. branching rule 6 when the
    /// current 3-component state matches its rooted overlap pattern.
    pub mestel_rule6: bool,

    /// Apply Mestel et al. (2024) full split-or-decompose framework
    /// before falling back to Whidden's case-based branching. When
    /// components have overlapping embeddings in T1, branch via SPLIT
    /// (Lemma 1's splitting core). When disjoint, apply DECOMPOSE
    /// (recursion rule). Off by default — gated for safety until
    /// validated.
    pub use_split_or_decompose: bool,
}

impl Default for BBConfig {
    /// Fastest currently-observed default on hard 2-tree instances:
    /// keep structural reductions on, but leave BB pruning off.
    fn default() -> Self {
        Self {
            bb: false,
            bb_2approx: false,
            cut_one_b: true,
            reverse_cut_one_b: true,
            reverse_cut_one_b_3: false,
            cut_two_b: true,
            cut_all_b: true,
            cut_ac_separate_components: true,
            edge_protection: true,
            edge_protection_two_b: true,
            deepest_protected_order: false,
            deepest_order: true,
            near_preorder_sibling_pairs: true,
            leaf_reduction: true,
            leaf_reduction2: true,
            approx_cut_one_b: true,
            approx_cut_two_b: true,
            approx_reverse_cut_one_b: true,
            prefer_rho: true,
            prefer_nonbranching: true,
            tt_enabled: false,
            tt_prune: false,
            tt_size_log2: 23,
            bound_cache_enabled: false,
            bound_cache_size_log2: 20,
            mestel_rule6: false,
            use_split_or_decompose: false,
        }
    }
}

impl BBConfig {
    /// No optimizations — pure 3-way branching (rspr's `-noopt`).
    #[allow(dead_code)]
    pub fn noopt() -> Self {
        Self {
            bb: false,
            bb_2approx: false,
            cut_one_b: false,
            reverse_cut_one_b: false,
            reverse_cut_one_b_3: false,
            cut_two_b: false,
            cut_all_b: false,
            cut_ac_separate_components: false,
            edge_protection: false,
            edge_protection_two_b: false,
            deepest_protected_order: false,
            deepest_order: false,
            near_preorder_sibling_pairs: false,
            leaf_reduction: false,
            leaf_reduction2: false,
            approx_cut_one_b: false,
            approx_cut_two_b: false,
            approx_reverse_cut_one_b: false,
            prefer_rho: false,
            prefer_nonbranching: false,
            tt_enabled: false,
            tt_prune: false,
            tt_size_log2: 23,
            bound_cache_enabled: false,
            bound_cache_size_log2: 20,
            mestel_rule6: false,
            use_split_or_decompose: false,
        }
    }

    /// Only BB pruning enabled (current baseline).
    #[allow(dead_code)]
    pub fn bb_only() -> Self {
        Self {
            bb: true,
            ..Self::noopt()
        }
    }
}

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct TTEntry {
    hash: u64,
    required_k_min: i16,
}

struct TranspositionTable {
    entries: Vec<TTEntry>,
    mask: usize,
}

impl TranspositionTable {
    fn new(size_log2: u8) -> Self {
        let size = 1usize << size_log2;
        Self {
            entries: vec![
                TTEntry {
                    hash: 0,
                    required_k_min: i16::MIN
                };
                size
            ],
            mask: size - 1,
        }
    }

    /// Probe the TT. Returns true if the state provably fails at budget k.
    #[inline]
    fn probe(&self, hash: u64, k: i32) -> bool {
        let idx = (hash as usize) & self.mask;
        let entry = &self.entries[idx];
        entry.hash == hash && k < entry.required_k_min as i32
    }

    /// Store a proven failure: state with this hash needs at least k+1 cuts.
    #[inline]
    fn store(&mut self, hash: u64, k: i32, rule_stats: &mut WhiddenRuleStats) {
        let idx = (hash as usize) & self.mask;
        let entry = &mut self.entries[idx];
        let new_min = (k + 1) as i16;
        if entry.hash == hash {
            if new_min > entry.required_k_min {
                entry.required_k_min = new_min;
                rule_stats.tt_overwrites += 1;
            }
        } else {
            entry.hash = hash;
            entry.required_k_min = new_min;
            rule_stats.tt_stores += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Bound cache — caches val3 and approx_2_lb per state hash
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct BoundEntry {
    hash: u64,
    val3: i16,       // -1 = not cached
    approx2_lb: i16, // -1 = not cached
}

struct BoundCache {
    entries: Vec<BoundEntry>,
    mask: usize,
}

impl BoundCache {
    fn new(size_log2: u8) -> Self {
        let size = 1usize << size_log2;
        Self {
            entries: vec![
                BoundEntry {
                    hash: 0,
                    val3: -1,
                    approx2_lb: -1
                };
                size
            ],
            mask: size - 1,
        }
    }

    /// Probe for cached val3. Returns Some(val3) on hit.
    #[inline]
    fn probe_val3(&self, hash: u64) -> Option<i32> {
        let idx = (hash as usize) & self.mask;
        let e = &self.entries[idx];
        if e.hash == hash && e.val3 >= 0 {
            Some(e.val3 as i32)
        } else {
            None
        }
    }

    /// Probe for cached approx_2_lb. Returns Some(lb) on hit.
    #[inline]
    fn probe_approx2(&self, hash: u64) -> Option<i32> {
        let idx = (hash as usize) & self.mask;
        let e = &self.entries[idx];
        if e.hash == hash && e.approx2_lb >= 0 {
            Some(e.approx2_lb as i32)
        } else {
            None
        }
    }

    /// Store val3 for a state.
    #[inline]
    fn store_val3(&mut self, hash: u64, val3: i32) {
        let idx = (hash as usize) & self.mask;
        let e = &mut self.entries[idx];
        if e.hash != hash {
            // New entry — evict old
            e.hash = hash;
            e.val3 = val3 as i16;
            e.approx2_lb = -1; // clear stale approx2
        } else {
            e.val3 = val3 as i16;
        }
    }

    /// Store approx_2_lb for a state (hash must already match from store_val3).
    #[inline]
    fn store_approx2(&mut self, hash: u64, lb: i32) {
        let idx = (hash as usize) & self.mask;
        let e = &mut self.entries[idx];
        if e.hash == hash {
            e.approx2_lb = lb as i16;
        }
    }
}

/// Propagated bound info from parent to child.
#[derive(Clone, Copy)]
struct ParentBounds {
    val3: i32,       // parent's val3 (-1 = unknown)
    approx2_lb: i32, // parent's approx_2_lb (-1 = unknown)
}

impl Default for ParentBounds {
    fn default() -> Self {
        Self {
            val3: -1,
            approx2_lb: -1,
        }
    }
}

/// Margin for parent approx_2_lb propagation.
/// Skip approx_2_lb if parent_2lb + MARGIN ≤ k.
const APPROX2_SKIP_MARGIN: i32 = 2;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn solve_with_rule_stats_and_progress<F>(
    instance: &Instance,
    stats: &mut SolverStats,
    rule_stats: &mut WhiddenRuleStats,
    config: &BBConfig,
    progress: Option<&mut F>,
) -> Option<Vec<Tree>>
where
    F: FnMut(WhiddenProgressUpdate) + ?Sized,
{
    solve_with_config_and_progress(instance, stats, rule_stats, config, progress)
}

fn solve_with_config_and_progress<F>(
    instance: &Instance,
    stats: &mut SolverStats,
    rule_stats: &mut WhiddenRuleStats,
    config: &BBConfig,
    mut progress: Option<&mut F>,
) -> Option<Vec<Tree>>
where
    F: FnMut(WhiddenProgressUpdate) + ?Sized,
{
    debug_assert!(instance.num_trees() == 2);
    let n = instance.num_leaves;
    if n <= 1 {
        return Some(vec![instance.trees[0].clone()]);
    }

    // Build TwinForest first — needed for both bounds and B&B.
    let mut tf = TwinForest::from_trees(&instance.trees[0], &instance.trees[1], n);
    let mut um = UndoMachine::new();

    // LB: Olver 2-approx dual on TwinForest (0.009ms, 69.9% tight).
    // This is the iterative deepening floor — skips k=0 through k=D-1.
    let lb_cuts = if n <= 128 {
        approx2::approx_2_lb(&tf).max(0) as usize
    } else {
        // The optimized TwinForest 2-approx uses u128 leaf masks internally.
        // Above 128 leaves, fall back to the original dynamic-bitset red/blue
        // lower bound on the reduced input trees.
        red_blue_approx_detailed(&instance.trees[0], &instance.trees[1]).dual_lb
    };
    let lb = lb_cuts + 1; // components space

    // UB: cherry reduction (cheap for 2-tree, tighter than approx_3).
    let ub_cuts = cherry_reduce_ub(&instance.trees[0], &instance.trees[1]);
    let ub = (ub_cuts + 1).min(n as usize);

    stats.lower_bound = lb;
    stats.upper_bound = Some(ub);

    let lb_k = lb.saturating_sub(1);
    let ub_k = ub.saturating_sub(1);
    rule_stats.lb_k = lb_k;
    rule_stats.ub_k = ub_k;

    // Allocate transposition table (persists across k iterations for cross-k reuse).
    let mut tt = if config.tt_enabled {
        Some(TranspositionTable::new(config.tt_size_log2))
    } else {
        None
    };

    // Bound cache: persists across k iterations for cross-k reuse of val3/approx2_lb.
    let mut bc = if config.bound_cache_enabled {
        Some(BoundCache::new(config.bound_cache_size_log2))
    } else {
        None
    };

    for k in lb_k..=ub_k {
        rule_stats.current_k = Some(k);
        rule_stats.k_attempts += 1;
        let k_start = Instant::now();

        if let Some(cb) = progress.as_mut() {
            cb(WhiddenProgressUpdate {
                current_k: k,
                lb_k,
                ub_k,
                k_attempts: rule_stats.k_attempts,
                k_elapsed_ms: 0.0,
                nodes_explored: stats.nodes_explored,
                solved: false,
            });
        }

        let cp = um.checkpoint();
        let result = branch_and_bound(
            &mut tf,
            k as i32,
            &mut um,
            stats,
            rule_stats,
            config,
            &mut tt,
            &mut bc,
            ParentBounds::default(),
        );
        let k_elapsed_ms = k_start.elapsed().as_secs_f64() * 1000.0;
        rule_stats.k_last_elapsed_ms = k_elapsed_ms;
        rule_stats.k_total_elapsed_ms += k_elapsed_ms;

        if let Some(cb) = progress.as_mut() {
            cb(WhiddenProgressUpdate {
                current_k: k,
                lb_k,
                ub_k,
                k_attempts: rule_stats.k_attempts,
                k_elapsed_ms,
                nodes_explored: stats.nodes_explored,
                solved: result.is_some(),
            });
        }

        if result.is_some() {
            rule_stats.k_success = Some(k);
            rule_stats.current_k = None;
            return Some(extract_components(&tf));
        }
        um.undo_to(cp, &mut tf);
    }

    rule_stats.current_k = None;
    None
}

// ---------------------------------------------------------------------------
// Branch-and-bound
// ---------------------------------------------------------------------------

fn branch_and_bound(
    tf: &mut TwinForest,
    mut k: i32,
    um: &mut UndoMachine,
    stats: &mut SolverStats,
    rule_stats: &mut WhiddenRuleStats,
    config: &BBConfig,
    tt: &mut Option<TranspositionTable>,
    bc: &mut Option<BoundCache>,
    parent_bounds: ParentBounds,
) -> Option<u32> {
    bb_inner(
        tf,
        &mut k,
        um,
        stats,
        rule_stats,
        config,
        tt,
        bc,
        parent_bounds,
        None,
    )
}

/// Inner B&B with optional forced pair (from CUT_ALL_B).
/// When `forced_pair` is Some, skip pair scanning and use it directly.
///
/// **Return semantics (optimization framing)**: `Some(c)` means a MAF was
/// found using `c` cuts within the budget `*k`. `None` means no feasible
/// MAF exists within the budget. Under the outer iterative-deepening loop
/// (which calls bb_inner with progressively larger budgets), the first
/// successful budget gives the optimum, so "first success" semantics is
/// equivalent to "min over branches" at that point.
///
/// The Mestel split-or-decompose rules (Phase 2/3) will take MIN over
/// branches explicitly when the recursion rule's sub-instances need exact
/// per-sub minimums.
fn bb_inner(
    tf: &mut TwinForest,
    k: &mut i32,
    um: &mut UndoMachine,
    stats: &mut SolverStats,
    rule_stats: &mut WhiddenRuleStats,
    config: &BBConfig,
    tt: &mut Option<TranspositionTable>,
    bc: &mut Option<BoundCache>,
    parent_bounds: ParentBounds,
    forced_pair: Option<(NodeId, NodeId)>,
) -> Option<u32> {
    stats.nodes_explored += 1;

    // SPLIT diagnostic — gated env var, sampled every 50 nodes to keep cost
    // bounded. Reads `KLADOS_WHIDDEN_SPLIT_DIAG` once via a lazy static.
    {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        let on = *ENABLED.get_or_init(|| std::env::var("KLADOS_WHIDDEN_SPLIT_DIAG").is_ok());
        if on && stats.nodes_explored % 50 == 0 {
            split_diag_check(tf, rule_stats);
        }
    }

    // Macro for TT store on failure.
    macro_rules! tt_store_fail {
        ($tt:expr, $tf:expr, $k:expr, $rs:expr) => {
            if let Some(t) = $tt.as_mut() {
                t.store($tf.state_hash, $k, $rs);
            }
        };
    }

    // CAB: if we have a forced pair from a previous B-cut, try it first.
    if let Some((t1_a, t1_c)) = forced_pair {
        // Process singletons first (some may have been created by the B-cut)
        if !process_singletons(tf, k, um, rule_stats) {
            tt_store_fail!(tt, tf, *k, rule_stats);
            return None;
        }
        if config.mestel_rule6 {
            match try_mestel_rule6(tf, k, um, rule_stats) {
                MestelRule6Result::Applied => {
                    return bb_inner(
                        tf,
                        k,
                        um,
                        stats,
                        rule_stats,
                        config,
                        tt,
                        bc,
                        parent_bounds,
                        forced_pair,
                    );
                }
                MestelRule6Result::ExhaustedBudget => {
                    rule_stats.prune_k_exhausted += 1;
                    tt_store_fail!(tt, tf, *k, rule_stats);
                    return None;
                }
                MestelRule6Result::NotApplicable => {}
            }
        }
        // Check if the forced pair is still valid (both still siblings in T1)
        let p_a = tf.parent[T1][t1_a as usize];
        let p_c = tf.parent[T1][t1_c as usize];
        if p_a != NONE && p_a == p_c && tf.is_leaf(T1, t1_a) && tf.is_leaf(T1, t1_c) {
            if let Some(result) = classify_pair(tf, p_a, t1_a, t1_c, config) {
                match result {
                    PairResult::Case2 {
                        t1_parent,
                        t2_parent,
                    } => {
                        do_case2_contract(tf, t1_parent, t2_parent, um);
                        rule_stats.forced_pair_case2 += 1;
                        rule_stats.action_case2_contracts += 1;
                        // Fall through to normal loop
                    }
                    PairResult::Case3 {
                        t1_a,
                        t1_c,
                        t2_a,
                        t2_b,
                        t2_c,
                        path_length,
                        ..
                    } => {
                        rule_stats.forced_pair_attempts += 1;
                        rule_stats.forced_pair_case3 += 1;
                        if *k <= 0 {
                            rule_stats.prune_k_exhausted += 1;
                            tt_store_fail!(tt, tf, *k, rule_stats);
                            return None;
                        }
                        let (should_prune, bounds) =
                            bb_should_prune(tf, um, *k, config, bc, parent_bounds, rule_stats);
                        if should_prune {
                            rule_stats.prune_bb_approx += 1;
                            tt_store_fail!(tt, tf, *k, rule_stats);
                            return None;
                        }
                        // Force cut_b_only for the CAB forced pair
                        let result = do_case3_branch(
                            tf,
                            *k,
                            um,
                            stats,
                            rule_stats,
                            config,
                            tt,
                            bc,
                            bounds,
                            t1_a,
                            t1_c,
                            t2_a,
                            t2_b,
                            t2_c,
                            true,
                            false,
                            false,
                            path_length,
                        );
                        if result.is_none() {
                            tt_store_fail!(tt, tf, *k, rule_stats);
                        }
                        return result;
                    }
                    _ => {}
                }
            }
        } else {
            rule_stats.forced_pair_attempts += 1;
            rule_stats.forced_pair_invalidated += 1;
        }
        // Forced pair no longer valid — fall through to normal B&B
    }

    loop {
        // --- Phase 1: Process singletons ---
        if !process_singletons(tf, k, um, rule_stats) {
            tt_store_fail!(tt, tf, *k, rule_stats);
            return None; // k went negative
        }

        // Exhaust common-cherry contractions before SoD. In Mestel's
        // terminology these are part of tidying-up, not a branching rule.
        // Running SPLIT before this contraction pass makes it branch on
        // overlap that the normal Whidden reductions erase for free.
        let pair_after_tidy = find_sibling_pair(tf, config, rule_stats);
        if let PairResult::Case2 {
            t1_parent,
            t2_parent,
        } = &pair_after_tidy
        {
            do_case2_contract(tf, *t1_parent, *t2_parent, um);
            rule_stats.action_case2_contracts += 1;
            continue;
        }

        // --- Mestel split-or-decompose entry point. When SPLIT matches it
        //     branches recursively and returns `Branched`; when DECOMPOSE
        //     matches it mutates T2 and returns `Applied`; otherwise normal
        //     Whidden case logic takes over.
        match try_split_or_decompose(
            tf,
            k,
            um,
            stats,
            rule_stats,
            config,
            tt,
            bc,
            parent_bounds,
        ) {
            SplitRuleResult::NotApplicable => {}
            SplitRuleResult::Branched(result) => {
                if result.is_none() {
                    tt_store_fail!(tt, tf, *k, rule_stats);
                }
                return result;
            }
            SplitRuleResult::Applied(None) => {
                // Budget exhausted inside SoD or sub-solver failed — prune.
                tt_store_fail!(tt, tf, *k, rule_stats);
                return None;
            }
            SplitRuleResult::Applied(Some(cuts)) => {
                if cuts == 0 {
                    // SoD claims success with zero cuts (all sub-MAFs are
                    // trivial). Don't loop — let Whidden's normal flow
                    // (case2 contracts, singleton processing) make
                    // tangible progress instead.
                } else {
                    rule_stats.split_rule_applied += 1;
                    // Charge the SoD cuts against the budget so the
                    // remainder of the search (process_singletons +
                    // normal branching, which projects T2 cuts into T1
                    // components) sees the correct remaining budget.
                    *k -= cuts as i32;
                    if *k < 0 {
                        rule_stats.prune_k_exhausted += 1;
                        tt_store_fail!(tt, tf, *k, rule_stats);
                        return None;
                    }
                    continue;
                }
            }
        }

        if config.mestel_rule6 {
            match try_mestel_rule6(tf, k, um, rule_stats) {
                MestelRule6Result::Applied => continue,
                MestelRule6Result::ExhaustedBudget => {
                    rule_stats.prune_k_exhausted += 1;
                    tt_store_fail!(tt, tf, *k, rule_stats);
                    return None;
                }
                MestelRule6Result::NotApplicable => {}
            }
        }

        // --- TT probe: after singletons, state is canonical ---
        if let Some(t) = tt.as_ref() {
            rule_stats.tt_lookups += 1;
            if t.probe(tf.state_hash, *k) {
                rule_stats.tt_hits += 1;
                if config.tt_prune {
                    rule_stats.tt_prunes += 1;
                    return None; // no store needed: already in TT
                }
            }
        }

        // --- Phase 2: Finish the already-classified sibling-pair state ---
        match pair_after_tidy {
            PairResult::NoPairs => {
                rule_stats.action_done += 1;
                // Phase 1: value is placeholder; under iterative deepening
                // first-success semantics is sufficient. Phase 2/3 will
                // replace with actual cut counts for SPLIT/DECOMPOSE rules.
                return Some(0);
            }
            PairResult::Case2 { .. } => unreachable!("case2 handled before SoD"),
            PairResult::Case3 {
                t1_a,
                t1_c,
                t2_a,
                t2_b,
                t2_c,
                cut_b_only,
                cut_c_only,
                cut_a_only,
                path_length,
            } => {
                rule_stats.action_case3_branches += 1;
                if *k <= 0 {
                    rule_stats.prune_k_exhausted += 1;
                    tt_store_fail!(tt, tf, *k, rule_stats);
                    return None;
                }
                let (should_prune, bounds) =
                    bb_should_prune(tf, um, *k, config, bc, parent_bounds, rule_stats);
                if should_prune {
                    rule_stats.prune_bb_approx += 1;
                    tt_store_fail!(tt, tf, *k, rule_stats);
                    return None;
                }
                let result = do_case3_branch(
                    tf,
                    *k,
                    um,
                    stats,
                    rule_stats,
                    config,
                    tt,
                    bc,
                    bounds,
                    t1_a,
                    t1_c,
                    t2_a,
                    t2_b,
                    t2_c,
                    cut_b_only,
                    cut_a_only,
                    cut_c_only,
                    path_length,
                );
                if result.is_none() {
                    tt_store_fail!(tt, tf, *k, rule_stats);
                }
                return result;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Singleton processing (Case 1)
// ---------------------------------------------------------------------------

/// Process all singletons. Returns false if k goes negative.
fn process_singletons(
    tf: &mut TwinForest,
    _k: &mut i32, // reserved for future singleton-charging optimizations
    um: &mut UndoMachine,
    rule_stats: &mut WhiddenRuleStats,
) -> bool {
    loop {
        let singleton = find_singleton(tf);
        if singleton == NONE {
            return true;
        }

        // singleton is a T2 leaf that is a component root (singleton in T2)
        let t2_node = singleton;
        let t1_node = tf.twin[T2][t2_node as usize];
        if t1_node == NONE {
            continue;
        }

        let t1_parent = tf.parent[T1][t1_node as usize];
        if t1_parent == NONE {
            // T1_a is already a component root — skip
            continue;
        }

        rule_stats.action_singleton_cuts += 1;
        undo::cut_parent(tf, T1, t1_node, um);
        undo::add_component(tf, T1, t1_node, um);

        // Contract the parent (may become degree-1 and get spliced)
        undo::contract(tf, T1, t1_parent, um);
    }
}

/// Find a singleton: a T2 component that is a single leaf.
/// Skip singletons whose twin is already a component root in T1.
fn find_singleton(tf: &TwinForest) -> NodeId {
    for &root in &tf.components[T2] {
        if tf.is_leaf(T2, root) {
            let twin = tf.twin[T2][root as usize];
            if twin != NONE && tf.parent[T1][twin as usize] != NONE {
                // Twin has a parent in T1 → can cut it
                return root;
            }
        }
    }
    NONE
}

// ---------------------------------------------------------------------------
// Sibling pair detection
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum PairResult {
    NoPairs,
    Case2 {
        t1_parent: NodeId,
        t2_parent: NodeId,
    },
    Case3 {
        t1_a: NodeId,
        t1_c: NodeId,
        t2_a: NodeId,
        t2_b: NodeId,
        t2_c: NodeId,
        /// COB: T2_ab and T2_c are siblings → only branch B needed.
        cut_b_only: bool,
        /// RCOB: uncle's twin is sibling of T2_a → only branch C needed.
        cut_c_only: bool,
        /// RCOB: uncle's twin is sibling of T2_c → only branch A needed.
        cut_a_only: bool,
        /// Path length from T2_a to T2_c through their LCA (for EP_TWO_B).
        path_length: u16,
    },
}

/// Find a sibling pair in T1 and classify it.
/// Walks from T1 component roots — only visits live nodes.
///
/// With `prefer_nonbranching`: prefers Case 2 or COB (1-branch) pairs over
/// full 3-way Case 3 pairs. With `deepest_order`: among Case 3 pairs, picks
/// the deepest (most constrained → prunes faster).
fn find_sibling_pair(
    tf: &TwinForest,
    config: &BBConfig,
    rule_stats: &mut WhiddenRuleStats,
) -> PairResult {
    if config.prefer_nonbranching || config.deepest_order {
        let mut fallback = PairResult::NoPairs;
        let mut best_depth = (0u16, 0u16);
        for &root in &tf.components[T1] {
            let result =
                find_preferred_pair(tf, root, config, rule_stats, &mut fallback, &mut best_depth);
            if !matches!(result, PairResult::NoPairs) {
                return result; // found a non-branching pair
            }
        }
        return fallback;
    }

    // No preference: return first pair found.
    for &root in &tf.components[T1] {
        let result = find_any_pair(tf, root, config);
        if !matches!(result, PairResult::NoPairs) {
            return result;
        }
    }
    PairResult::NoPairs
}

/// DFS to find any sibling pair (no preference).
fn find_any_pair(tf: &TwinForest, node: NodeId, config: &BBConfig) -> PairResult {
    let lc = tf.left[T1][node as usize];
    let rc = tf.right[T1][node as usize];

    if lc == NONE {
        return PairResult::NoPairs;
    }

    if rc != NONE && tf.is_leaf(T1, lc) && tf.is_leaf(T1, rc) {
        if let Some(result) = classify_pair(tf, node, lc, rc, config) {
            return result;
        }
    }

    if lc != NONE {
        let r = find_any_pair(tf, lc, config);
        if !matches!(r, PairResult::NoPairs) {
            return r;
        }
    }
    if rc != NONE {
        let r = find_any_pair(tf, rc, config);
        if !matches!(r, PairResult::NoPairs) {
            return r;
        }
    }
    PairResult::NoPairs
}

/// DFS to find the best sibling pair.
/// Returns immediately if a non-branching pair (Case 2 or COB/RCOB) is found.
/// Otherwise stores the best Case 3 in `fallback`:
///   - with deepest_order: prefer deepest pair (max depth in T2)
///   - without: take the first one found
fn find_preferred_pair(
    tf: &TwinForest,
    node: NodeId,
    config: &BBConfig,
    rule_stats: &mut WhiddenRuleStats,
    fallback: &mut PairResult,
    best_depth: &mut (u16, u16),
) -> PairResult {
    let lc = tf.left[T1][node as usize];
    let rc = tf.right[T1][node as usize];

    if lc == NONE {
        return PairResult::NoPairs;
    }

    if rc != NONE && tf.is_leaf(T1, lc) && tf.is_leaf(T1, rc) {
        if let Some(result) = classify_pair(tf, node, lc, rc, config) {
            match &result {
                PairResult::Case2 { .. } => return result,
                PairResult::Case3 {
                    t2_a,
                    t2_c,
                    cut_b_only,
                    cut_a_only,
                    cut_c_only,
                    ..
                } => {
                    if *cut_b_only || *cut_a_only || *cut_c_only {
                        if config.prefer_nonbranching {
                            rule_stats.prefer_nonbranching_hits += 1;
                        }
                        return result; // 1-way branching
                    }
                    if config.deepest_order {
                        // Score: max T2 depth of the pair (primary),
                        // min T2 depth (secondary tiebreak)
                        let da = depth_to_root(tf, T2, *t2_a);
                        let dc = depth_to_root(tf, T2, *t2_c);
                        let depth1 = da.max(dc);
                        let depth2 = da.min(dc);
                        if matches!(fallback, PairResult::NoPairs)
                            || depth1 > best_depth.0
                            || (depth1 == best_depth.0 && depth2 > best_depth.1)
                        {
                            *fallback = result;
                            *best_depth = (depth1, depth2);
                        }
                    } else if matches!(fallback, PairResult::NoPairs) {
                        *fallback = result;
                    }
                }
                _ => {}
            }
        }
    }

    if lc != NONE {
        let r = find_preferred_pair(tf, lc, config, rule_stats, fallback, best_depth);
        if !matches!(r, PairResult::NoPairs) {
            return r;
        }
    }
    if rc != NONE {
        let r = find_preferred_pair(tf, rc, config, rule_stats, fallback, best_depth);
        if !matches!(r, PairResult::NoPairs) {
            return r;
        }
    }
    PairResult::NoPairs
}

/// Classify a confirmed T1 sibling pair as Case 2 or Case 3.
fn classify_pair(
    tf: &TwinForest,
    t1_parent: NodeId,
    t1_a: NodeId,
    t1_c: NodeId,
    config: &BBConfig,
) -> Option<PairResult> {
    let t2_a = tf.twin[T1][t1_a as usize];
    let t2_c = tf.twin[T1][t1_c as usize];
    if t2_a == NONE || t2_c == NONE {
        return None;
    }

    // Case 2: T2_a and T2_c share a parent in T2
    let t2_a_parent = tf.parent[T2][t2_a as usize];
    let t2_c_parent = tf.parent[T2][t2_c as usize];
    if t2_a_parent != NONE && t2_a_parent == t2_c_parent {
        return Some(PairResult::Case2 {
            t1_parent,
            t2_parent: t2_a_parent,
        });
    }

    // Case 3: orient so T2_a is deeper (from current component root)
    let da = depth_to_root(tf, T2, t2_a);
    let dc = depth_to_root(tf, T2, t2_c);
    let (t1_a, t1_c, t2_a, t2_c) = if da >= dc {
        (t1_a, t1_c, t2_a, t2_c)
    } else {
        (t1_c, t1_a, t2_c, t2_a)
    };

    let t2_b = tf.sibling(T2, t2_a);
    if t2_b == NONE {
        return None;
    }

    // COB detection: T2_a.parent.parent == T2_c.parent means T2_ab and T2_c
    // are siblings, so only cutting B can resolve the pair.
    let t2_a_parent = tf.parent[T2][t2_a as usize];
    let t2_c_parent = tf.parent[T2][t2_c as usize];
    let mut cut_b_only = t2_a_parent != NONE && {
        let t2_ab_parent = tf.parent[T2][t2_a_parent as usize];
        t2_ab_parent != NONE && t2_ab_parent == t2_c_parent
    };

    // RCOB detection: uncle of sibling pair is a leaf whose T2 twin
    // constrains branching to a single direction.
    let mut cut_a_only = false;
    let mut cut_c_only = false;
    if !cut_b_only {
        let t1_ac_parent = tf.parent[T1][t1_parent as usize];
        if t1_ac_parent != NONE {
            let t1_s = tf.sibling(T1, t1_parent); // uncle
            if t1_s != NONE && tf.is_leaf(T1, t1_s) {
                let t2_s = tf.twin[T1][t1_s as usize];
                if t2_s != NONE {
                    let t2_s_parent = tf.parent[T2][t2_s as usize];
                    if t2_s_parent == t2_a_parent {
                        // Uncle's twin is sibling of T2_a → only cut C
                        cut_c_only = true;
                    } else if t2_s_parent == t2_c_parent {
                        // Uncle's twin is sibling of T2_c → only cut A
                        // (binary trees always have ≤ 2 children)
                        cut_a_only = true;
                    }
                }
            }
        }
    }

    // CUT_TWO_B: if the uncle's twin is sibling of the LCA of T2_a/T2_c,
    // then only cutting B resolves both the pair and the uncle.
    if config.cut_two_b && !cut_b_only {
        let t1_ac_parent = tf.parent[T1][t1_parent as usize];
        if t1_ac_parent != NONE {
            let t1_s = tf.sibling(T1, t1_parent); // uncle
            if t1_s != NONE && tf.is_leaf(T1, t1_s) {
                let t2_s = tf.twin[T1][t1_s as usize];
                if t2_s != NONE {
                    let t2_l = if t2_a_parent != NONE {
                        tf.parent[T2][t2_a_parent as usize]
                    } else {
                        NONE
                    };
                    if t2_l != NONE {
                        // Subcase 1: path_length 4 (balanced)
                        // T2_c.parent.parent == T2_l
                        if t2_c_parent != NONE && tf.parent[T2][t2_c_parent as usize] == t2_l {
                            if tf.sibling(T2, t2_l) == t2_s {
                                cut_b_only = true;
                            }
                        }
                        // Subcase 2: path_length 5
                        // T2_c.parent == T2_l.parent
                        if !cut_b_only {
                            let t2_l2 = tf.parent[T2][t2_l as usize];
                            if t2_l2 != NONE && t2_c_parent == t2_l2 {
                                if tf.sibling(T2, t2_l2) == t2_s {
                                    cut_b_only = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Path length: walk T2_a and T2_c up to their LCA, counting steps.
    let path_length = compute_path_length(tf, T2, t2_a, t2_c);

    Some(PairResult::Case3 {
        t1_a,
        t1_c,
        t2_a,
        t2_b,
        t2_c,
        cut_b_only,
        cut_c_only,
        cut_a_only,
        path_length,
    })
}

/// Distance from node to its component root (via parent pointers).
fn depth_to_root(tf: &TwinForest, ti: usize, mut node: NodeId) -> u16 {
    let mut d: u16 = 0;
    loop {
        let p = tf.parent[ti][node as usize];
        if p == NONE {
            return d;
        }
        d += 1;
        node = p;
    }
}

/// Compute path length from a to b through their LCA (rspr's same_component).
fn compute_path_length(tf: &TwinForest, ti: usize, mut a: NodeId, mut b: NodeId) -> u16 {
    let da = depth_to_root(tf, ti, a);
    let db = depth_to_root(tf, ti, b);
    let mut len: u16 = 0;
    // Level both to same depth
    let mut a_depth = da;
    let mut b_depth = db;
    while a_depth > b_depth {
        a = tf.parent[ti][a as usize];
        a_depth -= 1;
        len += 1;
    }
    while b_depth > a_depth {
        b = tf.parent[ti][b as usize];
        b_depth -= 1;
        len += 1;
    }
    // Walk both up until they meet
    while a != b {
        a = tf.parent[ti][a as usize];
        b = tf.parent[ti][b as usize];
        len += 2;
    }
    len
}

/// Walk parent pointers to find the component root.
#[inline]
fn find_root(tf: &TwinForest, ti: usize, mut node: NodeId) -> NodeId {
    loop {
        let p = tf.parent[ti][node as usize];
        if p == NONE {
            return node;
        }
        node = p;
    }
}

#[derive(Clone)]
struct ComponentShape {
    t2_root: NodeId,
    leafset: FixedBitSet,
    homeomorphic: bool,
    t1_edges: Vec<NodeId>,
}

#[inline]
fn leafset_capacity(tf: &TwinForest) -> usize {
    tf.num_leaves as usize + 1
}

fn collect_leafset_under(
    tf: &TwinForest,
    ti: usize,
    root: NodeId,
    restrict: Option<&FixedBitSet>,
) -> FixedBitSet {
    let mut out = FixedBitSet::with_capacity(leafset_capacity(tf));
    if root == NONE {
        return out;
    }

    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if tf.is_leaf(ti, node) {
            let lbl = tf.label[ti][node as usize];
            if lbl != 0
                && match restrict {
                    Some(keep) => keep.contains(lbl as usize),
                    None => true,
                }
            {
                out.insert(lbl as usize);
            }
            continue;
        }

        let rc = tf.right[ti][node as usize];
        if rc != NONE {
            stack.push(rc);
        }
        let lc = tf.left[ti][node as usize];
        if lc != NONE {
            stack.push(lc);
        }
    }

    out
}

fn common_root_for_leafset(tf: &TwinForest, ti: usize, leafset: &FixedBitSet) -> Option<NodeId> {
    let mut root: Option<NodeId> = None;
    for lbl in leafset.ones() {
        let node = tf.label_to_node[ti][lbl];
        if node == NONE {
            return None;
        }
        let this_root = find_root(tf, ti, node);
        match root {
            None => root = Some(this_root),
            Some(prev) if prev == this_root => {}
            Some(_) => return None,
        }
    }
    root
}

fn current_tree_canonical_for_labels(
    tf: &TwinForest,
    ti: usize,
    root: NodeId,
    labels: &FixedBitSet,
) -> u64 {
    fn build(tf: &TwinForest, ti: usize, node: NodeId, labels: &FixedBitSet) -> Option<u64> {
        if tf.is_leaf(ti, node) {
            let lbl = tf.label[ti][node as usize];
            if lbl != 0 && labels.contains(lbl as usize) {
                let mut h = lbl as u64;
                h = h.wrapping_mul(0x9e3779b97f4a7c15);
                h ^= h >> 30;
                Some(h)
            } else {
                None
            }
        } else {
            let lc = tf.left[ti][node as usize];
            let rc = tf.right[ti][node as usize];
            let left = if lc != NONE {
                build(tf, ti, lc, labels)
            } else {
                None
            };
            let right = if rc != NONE {
                build(tf, ti, rc, labels)
            } else {
                None
            };
            match (left, right) {
                (Some(a), Some(b)) => {
                    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                    let mut h = lo;
                    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
                    h ^= hi;
                    h = h.wrapping_mul(0x94d049bb133111eb);
                    h ^= h >> 31;
                    Some(h)
                }
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            }
        }
    }

    if root == NONE {
        0
    } else {
        build(tf, ti, root, labels).unwrap_or(0)
    }
}

fn collect_induced_edges(
    tf: &TwinForest,
    ti: usize,
    root: NodeId,
    labels: &FixedBitSet,
) -> Vec<NodeId> {
    fn dfs(
        tf: &TwinForest,
        ti: usize,
        node: NodeId,
        labels: &FixedBitSet,
        total: usize,
        out: &mut Vec<NodeId>,
    ) -> usize {
        if tf.is_leaf(ti, node) {
            let lbl = tf.label[ti][node as usize];
            return usize::from(lbl != 0 && labels.contains(lbl as usize));
        }

        let mut count = 0usize;
        let lc = tf.left[ti][node as usize];
        if lc != NONE {
            let left_count = dfs(tf, ti, lc, labels, total, out);
            if left_count > 0 && left_count < total {
                out.push(lc);
            }
            count += left_count;
        }
        let rc = tf.right[ti][node as usize];
        if rc != NONE {
            let right_count = dfs(tf, ti, rc, labels, total, out);
            if right_count > 0 && right_count < total {
                out.push(rc);
            }
            count += right_count;
        }
        count
    }

    let total = labels.count_ones(..);
    let mut out = Vec::new();
    if root != NONE && total >= 2 {
        dfs(tf, ti, root, labels, total, &mut out);
        out.sort_unstable();
    }
    out
}

fn shared_single_edge(a: &[NodeId], b: &[NodeId]) -> Option<NodeId> {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut found = None;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                if found.is_some() {
                    return None;
                }
                found = Some(a[i]);
                i += 1;
                j += 1;
            }
        }
    }

    found
}

fn have_shared_edge(a: &[NodeId], b: &[NodeId]) -> bool {
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

fn collect_component_shapes(tf: &TwinForest) -> Vec<ComponentShape> {
    tf.components[T2]
        .iter()
        .copied()
        .map(|t2_root| {
            let leafset = collect_leafset_under(tf, T2, t2_root, None);
            let t2_hash = current_tree_canonical_for_labels(tf, T2, t2_root, &leafset);
            let t1_root = common_root_for_leafset(tf, T1, &leafset);
            let t1_hash =
                t1_root.map(|root| current_tree_canonical_for_labels(tf, T1, root, &leafset));
            let homeomorphic = t1_hash == Some(t2_hash);
            let t1_edges = t1_root
                .map(|root| collect_induced_edges(tf, T1, root, &leafset))
                .unwrap_or_default();

            ComponentShape {
                t2_root,
                leafset,
                homeomorphic,
                t1_edges,
            }
        })
        .collect()
}

fn find_edge_with_descendant_leafset(
    tf: &TwinForest,
    ti: usize,
    root: NodeId,
    target: &FixedBitSet,
) -> Option<NodeId> {
    fn dfs(
        tf: &TwinForest,
        ti: usize,
        node: NodeId,
        target: &FixedBitSet,
        cap: usize,
        found: &mut Option<NodeId>,
    ) -> FixedBitSet {
        if tf.is_leaf(ti, node) {
            let mut out = FixedBitSet::with_capacity(cap);
            let lbl = tf.label[ti][node as usize];
            if lbl != 0 {
                out.insert(lbl as usize);
            }
            return out;
        }

        let mut out = FixedBitSet::with_capacity(cap);
        let lc = tf.left[ti][node as usize];
        if lc != NONE {
            let left = dfs(tf, ti, lc, target, cap, found);
            if *found == None && &left == target {
                *found = Some(lc);
            }
            out.union_with(&left);
        }
        let rc = tf.right[ti][node as usize];
        if rc != NONE {
            let right = dfs(tf, ti, rc, target, cap, found);
            if *found == None && &right == target {
                *found = Some(rc);
            }
            out.union_with(&right);
        }
        out
    }

    let mut found = None;
    if root != NONE && target.count_ones(..) > 0 {
        let _ = dfs(tf, ti, root, target, leafset_capacity(tf), &mut found);
    }
    found
}

/// Result of `detect_overlap`: the structural shape of `F'`'s components
/// relative to their embeddings in T1. Drives Mestel's split-or-decompose
/// rules.
pub(super) enum OverlapResult {
    /// Only one (or zero) components — neither SPLIT nor DECOMPOSE applies.
    SingleComponent,
    /// All components have pairwise disjoint T1-embeddings.
    /// DECOMPOSE rule applies: solve each sub-instance independently.
    AllDisjoint {
        components: Vec<ComponentShape>,
    },
    /// At least two components share a T1 edge.
    /// SPLIT rule applies: branch on which component "wins" the shared edge.
    Overlap {
        components: Vec<ComponentShape>,
        comp_a: usize,
        comp_b: usize,
        /// The shared T1 edge (identified by its child NodeId).
        shared_edge_t1: NodeId,
    },
}

/// Detect overlapping vs disjoint component embeddings in T1.
/// Deterministic choice of overlap pair (smallest shared edge, then
/// smallest comp_a, then smallest comp_b) for reproducibility.
pub(super) fn detect_overlap(tf: &TwinForest) -> OverlapResult {
    let components = collect_component_shapes(tf);
    if components.len() < 2 {
        return OverlapResult::SingleComponent;
    }

    // Map each T1 edge → list of component indices that claim it.
    use std::collections::HashMap;
    let mut edge_uses: HashMap<NodeId, Vec<usize>> = HashMap::new();
    for (idx, comp) in components.iter().enumerate() {
        for &edge in &comp.t1_edges {
            edge_uses.entry(edge).or_default().push(idx);
        }
    }

    // Find first overlap edge in deterministic order.
    let mut overlap_edges: Vec<_> = edge_uses
        .into_iter()
        .filter(|(_, comps)| comps.len() >= 2)
        .collect();
    if overlap_edges.is_empty() {
        return OverlapResult::AllDisjoint { components };
    }
    overlap_edges.sort_by_key(|(edge, _)| *edge);
    let (edge, mut comps) = overlap_edges.into_iter().next().unwrap();
    comps.sort_unstable();
    OverlapResult::Overlap {
        components,
        comp_a: comps[0],
        comp_b: comps[1],
        shared_edge_t1: edge,
    }
}

/// A single cut in a splitting core: a set of T1 edges whose simultaneous
/// removal splits T1 with respect to a bipartition. Each edge is
/// identified by its child NodeId (the edge from parent to that child).
pub(super) type SplittingCut = Vec<NodeId>;

/// Classify a subtree relative to the (Y, Z) bipartition.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SubtreeClass {
    Empty,
    PureY,
    PureZ,
    Mixed,
}

fn classify_labels(labels: &FixedBitSet, y_labels: &FixedBitSet) -> SubtreeClass {
    let mut has_y = false;
    let mut has_z = false;
    for lbl in labels.ones() {
        if y_labels.contains(lbl) {
            has_y = true;
        } else {
            has_z = true;
        }
        if has_y && has_z {
            return SubtreeClass::Mixed;
        }
    }
    match (has_y, has_z) {
        (true, false) => SubtreeClass::PureY,
        (false, true) => SubtreeClass::PureZ,
        (true, true) => SubtreeClass::Mixed,
        (false, false) => SubtreeClass::Empty,
    }
}

/// Collect labels in active that fall within v's subtree in T1.
fn active_labels_under_in(
    tf: &TwinForest,
    ti: usize,
    v: NodeId,
    active: &FixedBitSet,
) -> FixedBitSet {
    let mut out = FixedBitSet::with_capacity(active.len());
    let mut stack = vec![v];
    while let Some(n) = stack.pop() {
        if n == NONE {
            continue;
        }
        if tf.is_leaf(ti, n) {
            let lbl = tf.label[ti][n as usize] as usize;
            if lbl > 0 && active.contains(lbl) {
                out.insert(lbl);
            }
        } else {
            let lc = tf.left[ti][n as usize];
            let rc = tf.right[ti][n as usize];
            if lc != NONE {
                stack.push(lc);
            }
            if rc != NONE {
                stack.push(rc);
            }
        }
    }
    out
}

/// Find the "embedding-child" of `v` reached by descending into `child`
/// (which is one of v.left or v.right). Suppresses degree-2 embedding
/// nodes: keeps descending while exactly one of the current node's
/// children contains active labels.
///
/// Returns the NodeId of the embedding-child (the next embedding node
/// down that subtree, or a leaf if a single active label is reached).
/// Returns None if the subtree has no active labels.
fn embedding_descent(
    tf: &TwinForest,
    ti: usize,
    child: NodeId,
    active: &FixedBitSet,
) -> Option<NodeId> {
    if child == NONE {
        return None;
    }
    let mut cur = child;
    loop {
        if tf.is_leaf(ti, cur) {
            let lbl = tf.label[ti][cur as usize] as usize;
            if lbl > 0 && active.contains(lbl) {
                return Some(cur);
            }
            return None;
        }
        let lc = tf.left[ti][cur as usize];
        let rc = tf.right[ti][cur as usize];
        let left_has = lc != NONE
            && active_labels_under_in(tf, ti, lc, active).count_ones(..) > 0;
        let right_has = rc != NONE
            && active_labels_under_in(tf, ti, rc, active).count_ones(..) > 0;
        match (left_has, right_has) {
            (true, true) => return Some(cur),
            (true, false) => cur = lc,
            (false, true) => cur = rc,
            (false, false) => return None,
        }
    }
}

/// Splitting core (Mestel 2024, Lemma 1).
///
/// For a binary tree T (implicit: T1 restricted to `leafset`) and the
/// bipartition `(Y, Z)` of its labels where `Y = y_labels ∩ leafset`,
/// returns a set of cuts `C = {K_1, ..., K_n}` such that:
///   - every cut that splits the tree w.r.t. `(Y, Z)` refines some `K_i ∈ C`
///   - `Σ_{K ∈ C} 2^(-|K|) ≤ 1/2`
///
/// **Algorithm:**
/// - Base case: if T's labels split cleanly across a single embedding
///   edge `e` (one side all-Y, other all-Z), return `{{e}}`.
/// - Inductive case: find an embedding node `v` with three pendant
///   subtrees of classes `(PureY, PureZ, Mixed)`. Let the corresponding
///   edges be `e_1, e_2, e_3` respectively. Recurse on `T \ pure_Y_pendant`
///   and `T \ pure_Z_pendant`, get cores `C_1, C_2`. Return
///   `{K ∪ {e_1} | K ∈ C_1} ∪ {K ∪ {e_2} | K ∈ C_2}`.
///
/// **Edge identification:** an edge is named by its child-side NodeId in T1.
pub(super) fn splitting_core(
    tf: &TwinForest,
    embedding_root: NodeId,
    leafset: &FixedBitSet,
    y_labels: &FixedBitSet,
) -> Vec<SplittingCut> {
    splitting_core_in_tree(tf, T1, embedding_root, leafset, y_labels)
}

fn splitting_core_in_tree(
    tf: &TwinForest,
    ti: usize,
    embedding_root: NodeId,
    leafset: &FixedBitSet,
    y_labels: &FixedBitSet,
) -> Vec<SplittingCut> {
    splitting_core_impl(tf, ti, embedding_root, leafset, y_labels)
}

fn splitting_core_impl(
    tf: &TwinForest,
    ti: usize,
    embedding_root: NodeId,
    active: &FixedBitSet,
    y_labels: &FixedBitSet,
) -> Vec<SplittingCut> {
    // Trivial cases.
    if embedding_root == NONE || active.count_ones(..) <= 1 {
        return Vec::new();
    }
    // Step 1: base case — does a single embedding edge split the tree?
    if let Some(e) = single_edge_split(tf, ti, embedding_root, active, y_labels) {
        return vec![vec![e]];
    }
    // Step 2: inductive case — find a (PureY, PureZ, Mixed) pattern at
    // some embedding node, recurse.
    let Some(pat) = find_split_pattern(tf, ti, embedding_root, active, y_labels) else {
        // Shouldn't happen by the paper's claim, but be defensive.
        return Vec::new();
    };

    // Recurse on T \ pendant_y. The new active set excludes pendant_y's
    // leaves. The new tree's root is unchanged (the pendant being removed
    // is not the root). y_labels restricted to new active is computed.
    let new_active_1 = {
        let mut s = active.clone();
        s.difference_with(&pat.pendant_y_labels);
        s
    };
    let new_active_2 = {
        let mut s = active.clone();
        s.difference_with(&pat.pendant_z_labels);
        s
    };
    // Find the new embedding root in each recursive call. Removing a
    // pendant may make the original root degree-2 in the new embedding,
    // so we descend to find the new branching root.
    let new_root_1 = re_root_embedding(tf, ti, embedding_root, &new_active_1);
    let new_root_2 = re_root_embedding(tf, ti, embedding_root, &new_active_2);

    let core_1 = splitting_core_impl(tf, ti, new_root_1, &new_active_1, y_labels);
    let core_2 = splitting_core_impl(tf, ti, new_root_2, &new_active_2, y_labels);

    let mut out = Vec::with_capacity(core_1.len() + core_2.len());
    for mut k in core_1 {
        k.push(pat.edge_y);
        out.push(k);
    }
    for mut k in core_2 {
        k.push(pat.edge_z);
        out.push(k);
    }

    out
}

/// If a single embedding edge `e` cleanly bipartitions active labels into
/// all-Y on one side and all-Z on the other, return `e`. Otherwise None.
///
/// Iterates over embedding edges (which are exactly the t1_edges of the
/// component) and checks each.
fn single_edge_split(
    tf: &TwinForest,
    ti: usize,
    embedding_root: NodeId,
    active: &FixedBitSet,
    y_labels: &FixedBitSet,
) -> Option<NodeId> {
    // The embedding edges are nodes `c` such that c has some active
    // descendants and c is the child of some embedding-internal node.
    // We enumerate them by walking T1 from the embedding root and
    // checking each subtree's class.
    let mut found: Option<NodeId> = None;
    let mut walked_root_once = false;
    walk_embedding(tf, ti, embedding_root, active, &mut |c| {
        if c == embedding_root {
            // The root has no incoming embedding edge, skip.
            walked_root_once = true;
            return;
        }
        let below = active_labels_under_in(tf, ti, c, active);
        let below_class = classify_labels(&below, y_labels);
        // Above = active \ below
        let mut above = active.clone();
        above.difference_with(&below);
        let above_class = classify_labels(&above, y_labels);
        // A clean single-edge split: below is pure (Y or Z) and above is
        // pure of the opposite color.
        let clean = matches!(
            (below_class, above_class),
            (SubtreeClass::PureY, SubtreeClass::PureZ)
                | (SubtreeClass::PureZ, SubtreeClass::PureY)
        );
        if clean && found.is_none() {
            found = Some(c);
        }
    });
    let _ = walked_root_once;
    found
}

/// Walk all embedding nodes reachable from `root`. For each, call
/// `visit(node)`. An "embedding node" is one of:
///   - the embedding root,
///   - any T1 node with active labels in both its left and right subtrees,
///   - any T1 leaf whose label is active.
fn walk_embedding(
    tf: &TwinForest,
    ti: usize,
    root: NodeId,
    active: &FixedBitSet,
    visit: &mut impl FnMut(NodeId),
) {
    if root == NONE {
        return;
    }
    if tf.is_leaf(ti, root) {
        let lbl = tf.label[ti][root as usize] as usize;
        if lbl > 0 && active.contains(lbl) {
            visit(root);
        }
        return;
    }
    let lc = tf.left[ti][root as usize];
    let rc = tf.right[ti][root as usize];
    let left_has = lc != NONE && active_labels_under_in(tf, ti, lc, active).count_ones(..) > 0;
    let right_has =
        rc != NONE && active_labels_under_in(tf, ti, rc, active).count_ones(..) > 0;
    match (left_has, right_has) {
        (true, true) => {
            visit(root);
            walk_embedding(tf, ti, lc, active, visit);
            walk_embedding(tf, ti, rc, active, visit);
        }
        (true, false) => walk_embedding(tf, ti, lc, active, visit),
        (false, true) => walk_embedding(tf, ti, rc, active, visit),
        (false, false) => {}
    }
}

/// Re-root the embedding after a pendant has been removed. Descends from
/// the old root through degree-2 embedding nodes until reaching a true
/// branching node (or a leaf).
fn re_root_embedding(
    tf: &TwinForest,
    ti: usize,
    old_root: NodeId,
    active: &FixedBitSet,
) -> NodeId {
    let mut cur = old_root;
    while cur != NONE && !tf.is_leaf(ti, cur) {
        let lc = tf.left[ti][cur as usize];
        let rc = tf.right[ti][cur as usize];
        let left_has = lc != NONE && active_labels_under_in(tf, ti, lc, active).count_ones(..) > 0;
        let right_has =
            rc != NONE && active_labels_under_in(tf, ti, rc, active).count_ones(..) > 0;
        match (left_has, right_has) {
            (true, true) => return cur,
            (true, false) => cur = lc,
            (false, true) => cur = rc,
            (false, false) => return NONE,
        }
    }
    cur
}

struct SplitPattern {
    /// The T1 NodeId at the branching point with the pattern
    #[allow(dead_code)]
    v: NodeId,
    /// Embedding edge (child-side NodeId in T1) leading to pure-Y pendant
    edge_y: NodeId,
    /// Embedding edge leading to pure-Z pendant
    edge_z: NodeId,
    /// Embedding edge leading to mixed pendant (unused for recursion but
    /// useful for debugging / verification)
    #[allow(dead_code)]
    edge_mixed: NodeId,
    /// Labels in the pure-Y pendant
    pendant_y_labels: FixedBitSet,
    /// Labels in the pure-Z pendant
    pendant_z_labels: FixedBitSet,
}

/// Find an embedding node `v` whose three incident pendants are
/// `(PureY, PureZ, Mixed)` in some order. For rooted trees, the three
/// pendants at an internal node `v` are its left embedding-child,
/// its right embedding-child, and "outside `v`'s subtree" (= active \
/// labels_under(v)).
///
/// Returns the pattern info if found.
fn find_split_pattern(
    tf: &TwinForest,
    ti: usize,
    embedding_root: NodeId,
    active: &FixedBitSet,
    y_labels: &FixedBitSet,
) -> Option<SplitPattern> {
    let mut result: Option<SplitPattern> = None;
    walk_embedding(tf, ti, embedding_root, active, &mut |v| {
        if result.is_some() {
            return;
        }
        if tf.is_leaf(ti, v) {
            return;
        }
        // Embedding children of v.
        let lc = tf.left[ti][v as usize];
        let rc = tf.right[ti][v as usize];
        let lc_emb = embedding_descent(tf, ti, lc, active);
        let rc_emb = embedding_descent(tf, ti, rc, active);
        let (lc_emb, rc_emb) = match (lc_emb, rc_emb) {
            (Some(a), Some(b)) => (a, b),
            _ => return,
        };
        let left_labels = active_labels_under_in(tf, ti, lc_emb, active);
        let right_labels = active_labels_under_in(tf, ti, rc_emb, active);
        // Outside labels = active - labels under v.
        let under_v = active_labels_under_in(tf, ti, v, active);
        let mut outside = active.clone();
        outside.difference_with(&under_v);

        let left_cls = classify_labels(&left_labels, y_labels);
        let right_cls = classify_labels(&right_labels, y_labels);
        let outside_cls = classify_labels(&outside, y_labels);

        // Check for (PureY, PureZ, Mixed) pattern in any arrangement.
        let classes = [
            (left_cls, lc_emb, &left_labels),
            (right_cls, rc_emb, &right_labels),
            (outside_cls, NONE, &outside), // edge for outside has no NodeId
        ];
        let pure_y = classes.iter().find(|(c, _, _)| *c == SubtreeClass::PureY);
        let pure_z = classes.iter().find(|(c, _, _)| *c == SubtreeClass::PureZ);
        let mixed = classes.iter().find(|(c, _, _)| *c == SubtreeClass::Mixed);
        let (Some(y_info), Some(z_info), Some(m_info)) = (pure_y, pure_z, mixed) else {
            return;
        };
        // The outside slot has no edge NodeId — pattern requires the
        // pure-Y AND pure-Z to be on child sides (so we have edges to
        // them). Only the mixed slot may be "outside".
        if y_info.1 == NONE || z_info.1 == NONE {
            return;
        }
        result = Some(SplitPattern {
            v,
            edge_y: y_info.1,
            edge_z: z_info.1,
            edge_mixed: m_info.1,
            pendant_y_labels: y_info.2.clone(),
            pendant_z_labels: z_info.2.clone(),
        });
    });
    result
}

/// Verify the splitting core's inequality at runtime. Asserts in debug
/// builds, returns the inequality value (which must be ≤ 0.5).
#[cfg(test)]
pub(super) fn splitting_core_inequality(core: &[SplittingCut]) -> f64 {
    core.iter()
        .map(|k| 2f64.powi(-(k.len() as i32)))
        .sum::<f64>()
}

/// Result of the SPLIT rule entry point.
pub(super) enum SplitRuleResult {
    /// SPLIT rule not applicable (single component, or config gate off).
    NotApplicable,
    /// SPLIT branched recursively. The returned result is already the
    /// result of the recursive search and should be returned directly by
    /// the caller; unlike `Applied`, no further in-place loop progress is
    /// required at the current node.
    Branched(Option<u32>),
    /// Applied SPLIT (or DECOMPOSE in Phase 3) — caller should use this
    /// result instead of falling through to Whidden's case-based logic.
    /// `Some(c)` = MAF found with `c` cuts; `None` = no MAF within budget.
    Applied(Option<u32>),
}

/// Entry point for Mestel's split-or-decompose framework. Detects the
/// current state, and:
///   - On `Overlap`: applies SPLIT rule (Day 7+: actually branches).
///   - On `AllDisjoint`: applies DECOMPOSE rule (Day 7+: recursion).
///   - On `SingleComponent`: returns NotApplicable.
///
/// **Gated** by `BBConfig::use_split_or_decompose`.
pub(super) fn try_split_or_decompose(
    tf: &mut TwinForest,
    k: &mut i32,
    um: &mut UndoMachine,
    stats: &mut SolverStats,
    rule_stats: &mut WhiddenRuleStats,
    config: &BBConfig,
    tt: &mut Option<TranspositionTable>,
    bc: &mut Option<BoundCache>,
    parent_bounds: ParentBounds,
) -> SplitRuleResult {
    if !config.use_split_or_decompose {
        return SplitRuleResult::NotApplicable;
    }
    rule_stats.split_rule_checked += 1;
    match detect_overlap(tf) {
        OverlapResult::SingleComponent => SplitRuleResult::NotApplicable,
        OverlapResult::Overlap {
            components,
            comp_a,
            comp_b,
            shared_edge_t1,
        } => {
            rule_stats.split_rule_overlap_found += 1;
            match apply_split(
                tf,
                *k,
                um,
                stats,
                rule_stats,
                config,
                tt,
                bc,
                parent_bounds,
                &components,
                comp_a,
                comp_b,
                shared_edge_t1,
            ) {
                Some(result) => {
                    rule_stats.split_rule_applied += 1;
                    SplitRuleResult::Branched(result)
                }
                None => SplitRuleResult::NotApplicable,
            }
        }
        OverlapResult::AllDisjoint { components } => {
            rule_stats.split_rule_disjoint_found += 1;
            let substantive: Vec<&ComponentShape> = components
                .iter()
                .filter(|c| c.leafset.count_ones(..) >= 2)
                .collect();
            if substantive.len() < 2 {
                return SplitRuleResult::NotApplicable;
            }
            // Performance guard: don't bother decomposing when every
            // substantive component is small enough that Whidden's
            // case2/case3 will dispatch them quickly. Only fire SoD when
            // at least one component is genuinely large.
            let max_count = substantive.iter()
                .map(|c| c.leafset.count_ones(..))
                .max()
                .unwrap_or(0);
            // Cheap-out when no component is large enough for DECOMPOSE
            // to plausibly save work over Whidden's case2/case3. Whidden's
            // contracts already handle the trivial blocks fast; SoD only
            // pays off when sub-MAFs do non-trivial work, which requires
            // at least one substantive component of meaningful size.
            if max_count < 4 {
                return SplitRuleResult::NotApplicable;
            }
            match apply_decompose(tf, um, *k, &substantive) {
                Some(total_cuts) => SplitRuleResult::Applied(Some(total_cuts)),
                None => SplitRuleResult::Applied(None),
            }
        }
    }
}

/// Apply Mestel's SPLIT rule to one overlapping pair.
///
/// The shared edge lives in T1, so it induces a `(Y, Z)` bipartition on
/// whichever overlapping component loses the edge. To keep the live
/// Whidden state honest, we compute the splitting core directly on that
/// loser's current T2 component, then cut those T2 edges in each branch.
/// This is the same Lemma-1 branching argument, but it avoids the much
/// harder problem of translating a T1 core into equivalent live T2 cuts.
///
/// Returns:
/// - `None` if no usable SPLIT core could be constructed (caller should
///   fall back to normal Whidden branching);
/// - `Some(search_result)` if SPLIT was genuinely attempted.
fn apply_split(
    tf: &mut TwinForest,
    k_remaining: i32,
    um: &mut UndoMachine,
    stats: &mut SolverStats,
    rule_stats: &mut WhiddenRuleStats,
    config: &BBConfig,
    tt: &mut Option<TranspositionTable>,
    bc: &mut Option<BoundCache>,
    parent_bounds: ParentBounds,
    components: &[ComponentShape],
    comp_a: usize,
    comp_b: usize,
    shared_edge_t1: NodeId,
) -> Option<Option<u32>> {
    if k_remaining <= 0 {
        return Some(None);
    }

    let branch_specs = [comp_a, comp_b];
    let mut cores: Vec<(usize, Vec<SplittingCut>)> = Vec::with_capacity(2);
    for &loser_idx in &branch_specs {
        let loser = &components[loser_idx];
        let y_labels =
            active_labels_under_in(tf, T1, shared_edge_t1, &loser.leafset);
        if y_labels.count_ones(..) == 0 || y_labels == loser.leafset {
            return None;
        }
        let core = splitting_core_in_tree(
            tf,
            T2,
            loser.t2_root,
            &loser.leafset,
            &y_labels,
        );
        if core.is_empty() {
            return None;
        }
        rule_stats.split_rule_core_cutsets += core.len() as u64;
        rule_stats.split_rule_core_edges += core.iter().map(|k| k.len() as u64).sum::<u64>();
        rule_stats.split_rule_size1_cutsets +=
            core.iter().filter(|k| k.len() == 1).count() as u64;
        cores.push((loser_idx, core));
    }

    // Any size-1 cutset creates a unit-cost SPLIT branch. This is
    // theoretically fine, but in the hybrid Whidden solver these cheap
    // branches duplicate work the native case machinery already handles
    // well and are the dominant source of observed overfiring. Keep SPLIT
    // only when Lemma 1 gives a genuinely stronger branch family where
    // every recursive child spends at least two cuts immediately.
    if cores
        .iter()
        .any(|(_, core)| core.iter().any(|cutset| cutset.len() == 1))
    {
        return None;
    }

    for (_loser_idx, core) in cores {
        for cutset in core {
            if cutset.is_empty() || cutset.len() as i32 > k_remaining {
                continue;
            }
            let cp = um.checkpoint();
            let Some(applied) = apply_t2_cutset(tf, um, &cutset) else {
                um.undo_to(cp, tf);
                continue;
            };
            if applied == 0 || applied as i32 > k_remaining {
                um.undo_to(cp, tf);
                continue;
            }
            let result = branch_and_bound(
                tf,
                k_remaining - applied as i32,
                um,
                stats,
                rule_stats,
                config,
                tt,
                bc,
                parent_bounds,
            );
            if result.is_some() {
                return Some(result);
            }
            um.undo_to(cp, tf);
        }
    }

    Some(None)
}

/// Cut all child-side T2 edges in one SPLIT branch.
///
/// Cuts are applied deepest-first, contracting after each cut so the live
/// TwinForest remains a proper binary forest between operations. Applying
/// all cuts before contracting can leave 0-child internals behind when a
/// cutset contains siblings; later Whidden case3 logic is not prepared to
/// navigate those empty placeholders.
fn apply_t2_cutset(
    tf: &mut TwinForest,
    um: &mut UndoMachine,
    cutset: &[NodeId],
) -> Option<u32> {
    let mut unique = cutset.to_vec();
    unique.sort_unstable();
    unique.dedup();
    if unique.is_empty() {
        return None;
    }

    unique.sort_by_key(|&child| std::cmp::Reverse(depth_to_root(tf, T2, child)));
    for &child in &unique {
        let parent = tf.parent[T2][child as usize];
        if parent == NONE {
            return None;
        }
        // TwinForest deliberately keeps stale topology on dead nodes for
        // undo/hash reasons. A candidate from a stale embedding walk can
        // therefore have a parent pointer even though that parent no
        // longer references the child. Never feed such an edge to
        // `cut_parent`: it is not a live T2 edge.
        if tf.left[T2][parent as usize] != child && tf.right[T2][parent as usize] != child {
            return None;
        }
        undo::cut_parent(tf, T2, child, um);
        undo::add_component(tf, T2, child, um);
        undo::contract(tf, T2, parent, um);
    }

    Some(unique.len() as u32)
}

/// Translate a sub-MAF (one Tree per sub-component, with relabeled leaves)
/// into actual T2 cuts in `tf`. Each cut separates two adjacent
/// sub-components in the main `tf`'s T2 subtree rooted at `component_root`.
///
/// **Algorithm**:
/// 1. Color each leaf in main `tf` by which sub-MAF component it belongs to.
/// 2. DFS the T2 subtree bottom-up, deciding cuts: at each internal node
///    `v` with children `L` and `R` whose "anchor colors" (the color of
///    the leaves still connected to `v` via that side) differ, record one
///    cut (the edge to `L` or `R` — picked deterministically).
/// 3. Apply recorded cuts in deepest-first order (so each cut's child
///    edge is still alive when applied).
///
/// Returns the actual number of cuts applied. Each cut is registered with
/// `um` so backtracking works.
fn apply_sub_maf_cuts(
    tf: &mut TwinForest,
    um: &mut UndoMachine,
    component_root: NodeId,
    sub_maf: &[Tree],
    label_to_orig: &[klados_core::tree::Label],
) -> u32 {
    if sub_maf.len() <= 1 {
        return 0;
    }
    let n = tf.num_leaves as usize + 1;
    // leaf_color[orig_label] = sub-component id, or -1 if unassigned.
    //
    // sub_tree.num_leaves is the per-component count; iterating
    // 1..=sub_tree.num_leaves only covers low labels and misses any leaf
    // whose label exceeds the component's leaf count. Instead, iterate
    // over `label_to_orig`'s entire label space and check which sub-tree
    // claims that label via label_to_node[lbl] != NONE.
    let mut leaf_color: Vec<i32> = vec![-1; n];
    for new_lbl in 1..label_to_orig.len() {
        let orig_lbl = label_to_orig[new_lbl] as usize;
        if orig_lbl == 0 || orig_lbl >= n {
            continue;
        }
        for (sub_id, sub_tree) in sub_maf.iter().enumerate() {
            if new_lbl < sub_tree.label_to_node.len()
                && sub_tree.label_to_node[new_lbl] != NONE
            {
                leaf_color[orig_lbl] = sub_id as i32;
                break;
            }
        }
    }

    // Compute total per-color leaf counts in this T2 subtree.
    let num_colors = sub_maf.len();
    let mut total_count: Vec<u32> = vec![0; num_colors];
    {
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![component_root];
        while let Some(node) = stack.pop() {
            if node == NONE || !visited.insert(node) { continue; }
            if tf.is_leaf(T2, node) {
                let lbl = tf.label[T2][node as usize] as usize;
                if lbl != 0 && lbl < leaf_color.len() {
                    let c = leaf_color[lbl];
                    if c >= 0 && (c as usize) < num_colors {
                        total_count[c as usize] += 1;
                    }
                }
            } else {
                let l = tf.left[T2][node as usize];
                let r = tf.right[T2][node as usize];
                if l != NONE { stack.push(l); }
                if r != NONE && r != l { stack.push(r); }
            }
        }
    }

    // Pass 1: bottom-up DFS computing per-color leaf counts in each subtree.
    // Cut edge (parent, v) iff for every color c in subtree(v),
    // counts_v[c] == total_count[c] (all c-leaves are inside subtree(v)).
    // Equivalently: subtree(v) is a complete-and-separable color block.
    // Skip the root: we don't cut above the component root.
    let mut cuts_to_make: Vec<NodeId> = Vec::new();
    fn count_colors(
        tf: &TwinForest,
        node: NodeId,
        leaf_color: &[i32],
        total_count: &[u32],
        cuts_to_make: &mut Vec<NodeId>,
        is_root: bool,
    ) -> Vec<u32> {
        let nc = total_count.len();
        if node == NONE {
            return vec![0; nc];
        }
        if tf.is_leaf(T2, node) {
            let mut counts = vec![0u32; nc];
            let lbl = tf.label[T2][node as usize] as usize;
            if lbl != 0 && lbl < leaf_color.len() {
                let c = leaf_color[lbl];
                if c >= 0 && (c as usize) < nc {
                    counts[c as usize] = 1;
                }
            }
            // Even leaves can be "cut": isolating a single-leaf component.
            // But only if not root and the leaf's color is wholly contained.
            // For singletons (color appears once in T2 subtree), this is
            // always self-contained, so the leaf can be cut.
            if !is_root && counts.iter().enumerate().all(|(c, &n)| n == 0 || n == total_count[c]) {
                cuts_to_make.push(node);
            }
            return counts;
        }
        let lc = tf.left[T2][node as usize];
        let rc = tf.right[T2][node as usize];
        let mut counts = vec![0u32; nc];
        // Degenerate degree-1: `left == right` — recurse only once.
        if lc != NONE && lc == rc {
            let sub = count_colors(tf, lc, leaf_color, total_count, cuts_to_make, false);
            for i in 0..nc { counts[i] += sub[i]; }
        } else {
            if lc != NONE {
                let sub = count_colors(tf, lc, leaf_color, total_count, cuts_to_make, false);
                for i in 0..nc { counts[i] += sub[i]; }
            }
            if rc != NONE {
                let sub = count_colors(tf, rc, leaf_color, total_count, cuts_to_make, false);
                for i in 0..nc { counts[i] += sub[i]; }
            }
        }
        // Decide cut on the edge from this node up to its parent.
        // Cut iff subtree is "color-complete" (every color in this subtree
        // has all its leaves in this subtree). Skip for root.
        if !is_root {
            let any = counts.iter().any(|&n| n > 0);
            let complete = counts.iter().enumerate().all(|(c, &n)| n == 0 || n == total_count[c]);
            if any && complete {
                cuts_to_make.push(node);
            }
        }
        counts
    }
    let _ = count_colors(
        tf, component_root, &leaf_color, &total_count, &mut cuts_to_make, true,
    );

    // The bottom-up pass can record nested cuts (e.g., a leaf and its
    // parent when both are color-complete). Keep only the SHALLOWEST: a
    // cut at an ancestor makes descendant cuts redundant. Walk from each
    // queued cut up to the root; drop the cut if an ancestor is queued.
    let queued: std::collections::HashSet<NodeId> = cuts_to_make.iter().copied().collect();
    let cuts_to_make: Vec<NodeId> = cuts_to_make
        .iter()
        .copied()
        .filter(|&child| {
            let mut cur = tf.parent[T2][child as usize];
            while cur != NONE {
                if queued.contains(&cur) {
                    return false;
                }
                cur = tf.parent[T2][cur as usize];
            }
            true
        })
        .collect();

    // Pass 2: apply cuts in deepest-first order. Each cut: cut_parent +
    // add_component + contract on the parent (so the tree maintains
    // proper degree-2 internals). The contract may propagate up the
    // chain and, if it reaches a component root that was already in
    // tf.components[T2], splice that root out and replace it. Caller
    // (apply_decompose) re-resolves t2_roots by walking from leaves to
    // handle this case.
    let mut applied = 0u32;
    for &child in &cuts_to_make {
        let parent = tf.parent[T2][child as usize];
        if parent == NONE {
            continue;
        }
        undo::cut_parent(tf, T2, child, um);
        undo::add_component(tf, T2, child, um);
        undo::contract(tf, T2, parent, um);
        applied += 1;
    }
    applied
}

/// Find the current T2 component root that contains the given leaf label.
/// Walks up from the leaf's T2 node via parent pointers to find a node
/// with no parent (= a component root).
fn find_t2_component_root_for_leaf(tf: &TwinForest, leaf_label: usize) -> NodeId {
    let n = tf.label_to_node[T2][leaf_label];
    if n == NONE {
        return NONE;
    }
    let mut cur = n;
    loop {
        let p = tf.parent[T2][cur as usize];
        if p == NONE {
            return cur;
        }
        cur = p;
    }
}

/// Mestel DECOMPOSE rule: solve each disjoint component as an
/// independent sub-instance, sum the cuts.
///
/// Implementation: for each component, construct a sub-Instance with
/// T1 restricted to the component's leafset and T2 restricted to the
/// component's t2 subtree. Solve via a fresh `WhiddenSolver`. Sum the
/// cut counts (= forest size − 1).
///
/// Returns `Some(total_cuts)` if all sub-instances solved successfully,
/// `None` otherwise.
///
/// **Budget**: each sub-instance is bounded by the parent's remaining
/// budget `k_remaining`, simplified from the paper's two-phase recursion
/// rule. The paper's phase 1 (test each sub at depth `t=1` first) is a
/// performance optimization we can add later — this version is correct
/// but may do redundant work on subs that are trivially `0`.
fn apply_decompose(
    tf: &mut TwinForest,
    um: &mut UndoMachine,
    k_remaining: i32,
    substantive: &[&ComponentShape],
) -> Option<u32> {
    if k_remaining < 0 {
        return None;
    }
    let budget = k_remaining as u32;

    // Build the original T1 once; we'll relabel it per component.
    let t1_orig = tree_from_original(tf);

    let mut total_cuts: u32 = 0;
    for comp in substantive.iter() {
        let leafset = &comp.leafset;
        let count = leafset.count_ones(..);
        if count <= 1 {
            continue;
        }
        // Re-resolve this component's CURRENT t2_root by walking up from
        // any of its leaves. Earlier cuts may have contracted the cached
        // `comp.t2_root` out (replacing it with its child). Components are
        // disjoint in T2 so other components don't affect this lookup.
        let first_leaf = leafset.ones().next().unwrap_or(0);
        let current_t2_root = find_t2_component_root_for_leaf(tf, first_leaf);
        if current_t2_root == NONE {
            continue;
        }
        // Build relabel map: in-component leaves get new labels 1..count,
        // others get 0 (= drop). Also build the reverse map (new → orig).
        let mut label_map: Vec<klados_core::tree::Label> =
            vec![0; tf.num_leaves as usize + 1];
        let mut label_to_orig: Vec<klados_core::tree::Label> = vec![0; (count + 1) as usize];
        let mut new_label: u32 = 1;
        for old_lbl in leafset.ones() {
            label_map[old_lbl] = new_label as klados_core::tree::Label;
            label_to_orig[new_label as usize] = old_lbl as klados_core::tree::Label;
            new_label += 1;
        }
        let new_num_leaves = count as u32;
        let sub_t1 = t1_orig.relabel(&label_map, new_num_leaves);
        let t2_comp_tree = tree_from_t2_subtree(tf, current_t2_root);
        let sub_t2 = t2_comp_tree.relabel(&label_map, new_num_leaves);
        if sub_t1.root == NONE || sub_t2.root == NONE {
            continue;
        }
        let sub_instance = Instance::new(vec![sub_t1, sub_t2], new_num_leaves);
        if std::env::var("KLADOS_WHIDDEN_SOD_TRACE").is_ok() {
            trace_tree_shape("sub_t1", &sub_instance.trees[0]);
            trace_tree_shape("sub_t2", &sub_instance.trees[1]);
        }
        let mut sub_solver =
            crate::solvers::whidden::WhiddenSolver::new().with_split_or_decompose(false);
        let sub_solution = match crate::ExactSolver::solve(&mut sub_solver, &sub_instance) {
            Some(s) => s,
            None => {
                if std::env::var("KLADOS_WHIDDEN_SOD_TRACE").is_ok() {
                    eprintln!(
                        "[sod-trace] sub-solver returned None: n={} t1_root={} t1_nodes={} t1_num_leaves={} t2_root={} t2_nodes={} t2_num_leaves={}",
                        new_num_leaves,
                        sub_instance.trees[0].root,
                        sub_instance.trees[0].parent.len(),
                        sub_instance.trees[0].num_leaves,
                        sub_instance.trees[1].root,
                        sub_instance.trees[1].parent.len(),
                        sub_instance.trees[1].num_leaves,
                    );
                    // Print labels in each tree for debug
                    let t1_labels: Vec<u32> = (1..=new_num_leaves)
                        .filter(|&l| sub_instance.trees[0].label_to_node[l as usize] != NONE)
                        .collect();
                    let t2_labels: Vec<u32> = (1..=new_num_leaves)
                        .filter(|&l| sub_instance.trees[1].label_to_node[l as usize] != NONE)
                        .collect();
                    eprintln!(
                        "[sod-trace] t1_labels={:?} t2_labels={:?}",
                        t1_labels, t2_labels,
                    );
                }
                return None;
            }
        };
        let cuts_applied =
            apply_sub_maf_cuts(tf, um, current_t2_root, &sub_solution, &label_to_orig);
        total_cuts = total_cuts.saturating_add(cuts_applied);
        if total_cuts > budget {
            return None;
        }
    }
    Some(total_cuts)
}

fn trace_tree_shape(name: &str, tree: &Tree) {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![tree.root];
    let mut leaves = Vec::new();
    let mut bad = Vec::new();
    while let Some(node) = stack.pop() {
        if node == NONE || !seen.insert(node) {
            continue;
        }
        let l = tree.left[node as usize];
        let r = tree.right[node as usize];
        match (l, r, tree.label[node as usize]) {
            (NONE, NONE, lbl) if lbl != 0 => leaves.push(lbl),
            (NONE, NONE, lbl) => bad.push((node, l, r, lbl, "dead-leaf")),
            (NONE, _, lbl) | (_, NONE, lbl) => bad.push((node, l, r, lbl, "unary")),
            (_, _, lbl) if lbl != 0 => bad.push((node, l, r, lbl, "labeled-internal")),
            _ => {
                stack.push(l);
                stack.push(r);
            }
        }
    }
    leaves.sort_unstable();
    let expected: Vec<_> = (1..=tree.num_leaves).collect();
    if !bad.is_empty() || leaves != expected {
        eprintln!(
            "[sod-trace] {} invalid: root={} num_leaves={} reachable={} leaves={:?} bad={:?}",
            name,
            tree.root,
            tree.num_leaves,
            seen.len(),
            leaves,
            bad,
        );
    }
}

/// Reconstruct a single T2 component (rooted at `comp_root`) as a `Tree`.
///
/// **Important**: clones tf.label[T2] but zeros out labels of "dead"
/// nodes — those that have a label but are not the canonical owner in
/// `tf.label_to_node[T2]`. Dead leaves arise from contract_sibling_pair
/// and similar reductions: the parent absorbs a child's label, but the
/// orphaned children still have their old labels in `tf.label[T2]`. Left
/// uncleaned, those duplicate labels confuse `Tree::relabel`, which
/// would push them as additional leaves with the same new label.
fn tree_from_t2_subtree(tf: &TwinForest, comp_root: NodeId) -> Tree {
    let mut t = Tree::with_capacity(tf.num_leaves);
    t.parent = tf.parent[T2].clone();
    t.left = tf.left[T2].clone();
    t.right = tf.right[T2].clone();
    t.label = tf.label[T2].clone();
    // Zero out labels of dead leaves: keep only the canonical owner per
    // tf.label_to_node[T2].
    for node in 0..tf.num_nodes[T2] as NodeId {
        let lbl = t.label[node as usize];
        if lbl != 0 {
            let canonical = tf.label_to_node[T2][lbl as usize];
            if canonical != node {
                t.label[node as usize] = 0;
            }
        }
    }
    // Normalize degenerate degree-1 internals. TwinForest mutations can
    // leave either:
    //   - left == right == child, or
    //   - left == NONE, right == child.
    //
    // `Tree::is_leaf` uses `left == NONE`, so the right-only shape would
    // make `Tree::relabel` mistake an internal for a dead leaf and drop
    // the entire surviving subtree. Rewrite both cases to the one shape
    // `Tree::relabel` understands: left=child, right=NONE.
    for node in 0..tf.num_nodes[T2] as NodeId {
        let l = t.left[node as usize];
        let r = t.right[node as usize];
        if l != NONE && l == r {
            t.right[node as usize] = NONE;
        } else if l == NONE && r != NONE {
            t.left[node as usize] = r;
            t.right[node as usize] = NONE;
        }
    }
    t.label_to_node = tf.label_to_node[T2].clone();
    t.num_leaves = tf.num_leaves;
    t.root = comp_root;
    t.depth = vec![0; tf.num_nodes[T2]];
    t.subtree_size = vec![0; tf.num_nodes[T2]];
    t
}

/// SPLIT diagnostic: check whether the current TwinForest state would
/// trigger Mestel's SPLIT (overlapping component embeddings in T1) or
/// DECOMPOSE (disjoint instance with ≥ 2 components).
///
/// Gated by `KLADOS_WHIDDEN_SPLIT_DIAG=1`. Sampled every `SPLIT_DIAG_SAMPLE`
/// nodes by `nodes_explored` to keep cost bounded — `collect_component_shapes`
/// is O(n) per call and we don't want to inflate the search by 10×.
#[inline]
fn split_diag_check(tf: &TwinForest, rule_stats: &mut WhiddenRuleStats) {
    rule_stats.split_diag_nodes += 1;
    match detect_overlap(tf) {
        OverlapResult::SingleComponent => {
            rule_stats.split_diag_single_component += 1;
        }
        OverlapResult::Overlap { .. } => {
            rule_stats.split_diag_overlap += 1;
        }
        OverlapResult::AllDisjoint { components } => {
            rule_stats.split_diag_disjoint += 1;
            rule_stats.split_diag_disjoint_blocks_sum += components.len() as u64;
            let max_block = components
                .iter()
                .map(|c| c.leafset.count_ones(..))
                .max()
                .unwrap_or(0);
            rule_stats.split_diag_disjoint_max_block_sum += max_block as u64;
        }
    }
}

fn find_mestel_rule6_cut(tf: &TwinForest) -> Option<NodeId> {
    let shapes = collect_component_shapes(tf);
    if shapes.len() != 3 {
        return None;
    }

    for c in 0..3 {
        let others = match c {
            0 => [1usize, 2usize],
            1 => [0usize, 2usize],
            _ => [0usize, 1usize],
        };

        for (a, b) in [(others[0], others[1]), (others[1], others[0])] {
            let shape_a = &shapes[a];
            let shape_b = &shapes[b];
            let shape_c = &shapes[c];

            if !shape_a.homeomorphic || shape_b.homeomorphic || !shape_c.homeomorphic {
                continue;
            }
            if have_shared_edge(&shape_c.t1_edges, &shape_a.t1_edges)
                || have_shared_edge(&shape_c.t1_edges, &shape_b.t1_edges)
            {
                continue;
            }

            let Some(overlap_edge) = shared_single_edge(&shape_a.t1_edges, &shape_b.t1_edges)
            else {
                continue;
            };

            let below = collect_leafset_under(tf, T1, overlap_edge, Some(&shape_b.leafset));
            if below.count_ones(..) == 0 || below == shape_b.leafset {
                continue;
            }

            if let Some(cut_node) =
                find_edge_with_descendant_leafset(tf, T2, shape_b.t2_root, &below)
            {
                return Some(cut_node);
            }

            let mut rest = shape_b.leafset.clone();
            rest.difference_with(&below);
            if let Some(cut_node) =
                find_edge_with_descendant_leafset(tf, T2, shape_b.t2_root, &rest)
            {
                return Some(cut_node);
            }
        }
    }

    None
}

enum MestelRule6Result {
    NotApplicable,
    Applied,
    ExhaustedBudget,
}

fn try_mestel_rule6(
    tf: &mut TwinForest,
    k: &mut i32,
    um: &mut UndoMachine,
    rule_stats: &mut WhiddenRuleStats,
) -> MestelRule6Result {
    if tf.components[T2].len() != 3 {
        return MestelRule6Result::NotApplicable;
    }

    rule_stats.mestel6_checks += 1;

    let Some(cut_node) = find_mestel_rule6_cut(tf) else {
        return MestelRule6Result::NotApplicable;
    };

    if *k <= 0 {
        return MestelRule6Result::ExhaustedBudget;
    }

    let parent = tf.parent[T2][cut_node as usize];
    if parent == NONE {
        return MestelRule6Result::NotApplicable;
    }

    *k -= 1;
    undo::cut_parent(tf, T2, cut_node, um);
    undo::add_component(tf, T2, cut_node, um);
    undo::contract(tf, T2, parent, um);
    rule_stats.rule_mestel6_forced += 1;
    MestelRule6Result::Applied
}

// ---------------------------------------------------------------------------
// 3-approximation lower bound (rspr's BB pruning)
// ---------------------------------------------------------------------------

/// BB pruning decision: returns true if the current residual problem
/// provably has OPT > k, so this branch can be abandoned.
#[inline]
/// Compute bounds and decide whether to prune.
/// Returns (should_prune, bounds_for_children).
fn bb_should_prune(
    tf: &mut TwinForest,
    um: &mut UndoMachine,
    k: i32,
    config: &BBConfig,
    bc: &mut Option<BoundCache>,
    parent: ParentBounds,
    rule_stats: &mut WhiddenRuleStats,
) -> (bool, ParentBounds) {
    let mut bounds = ParentBounds {
        val3: -1,
        approx2_lb: -1,
    };

    if !config.bb {
        return (false, bounds);
    }

    // --- Phase 1: Parent-propagated skip for val3 ---
    // DISABLED: val3 can increase by more than MARGIN after one cut,
    // causing missed prunes that lead to incorrect results.
    // TODO: find a sound bound for val3 change per cut.
    // if parent.val3 >= 0 && parent.val3 + VAL3_SKIP_MARGIN <= 3 * k {
    //     rule_stats.bb_skipped_by_parent += 1;
    //     bounds.val3 = parent.val3;
    //     return (false, bounds);
    // }

    // --- Phase 2: Compute val3 (with optional bound cache) ---
    let hash = tf.state_hash;
    let val3;
    // TODO: bound cache disabled pending correctness investigation
    if let Some(bc) = bc.as_mut() {
        rule_stats.bc_lookups += 1;
        if let Some(cached) = bc.probe_val3(hash) {
            rule_stats.bc_hits += 1;
            val3 = cached;
        } else {
            rule_stats.bb_approx3_calls += 1;
            val3 = approx_3(tf, um);
            bc.store_val3(hash, val3);
            rule_stats.bc_stores += 1;
        }
    } else {
        rule_stats.bb_approx3_calls += 1;
        val3 = approx_3(tf, um);
    }
    bounds.val3 = val3;

    if val3 > 3 * k {
        return (true, bounds);
    }

    // --- Phase 3: Selective 2-approx (only near the pruning frontier) ---
    // Gate: val3 must be in the "gray zone" (close to 3k but not over)
    // AND parent's approx_2_lb must suggest we might prune
    if config.bb_2approx && k >= 3 && val3 > 3 * (k - 1) {
        // Parent skip: if parent's 2-approx was far from threshold, skip
        if parent.approx2_lb >= 0 && parent.approx2_lb + APPROX2_SKIP_MARGIN <= k {
            bounds.approx2_lb = parent.approx2_lb;
            return (false, bounds);
        }

        let lb2 = if let Some(bc) = bc.as_mut() {
            rule_stats.bc_lookups += 1;
            if let Some(cached) = bc.probe_approx2(hash) {
                rule_stats.bc_hits += 1;
                cached
            } else {
                rule_stats.bb_approx2_calls += 1;
                let lb2 = approx2::approx_2_lb(tf);
                bc.store_approx2(hash, lb2);
                rule_stats.bc_stores += 1;
                lb2
            }
        } else {
            rule_stats.bb_approx2_calls += 1;
            approx2::approx_2_lb(tf)
        };
        bounds.approx2_lb = lb2;

        if lb2 > k {
            rule_stats.bb_approx2_prunes += 1;
            return (true, bounds);
        }
    }

    (false, bounds)
}

/// Greedy 3-approximation of rSPR distance on the current forest state.
/// Faithful port of rspr's `rSPR_worse_3_approx_binary_hlpr`.
///
/// Each Case 3 round: cut T1_a, T1_c from T1; cut T2_a (and maybe T2_b,
/// T2_c) from T2; count 3. Resolves the pair in one round.
/// Guarantee: num_cut ≤ 3 × optimal.
///
/// Non-destructive: uses checkpoint/undo on the live TwinForest.
fn approx_3(tf: &mut TwinForest, um: &mut UndoMachine) -> i32 {
    let cp = um.checkpoint();
    let mut num_cut: i32 = 0;

    loop {
        // Process singletons (free — no contribution to num_cut)
        loop {
            let singleton = find_singleton(tf);
            if singleton == NONE {
                break;
            }
            let t2_node = singleton;
            let t1_node = tf.twin[T2][t2_node as usize];
            if t1_node == NONE {
                continue;
            }
            let t1_parent = tf.parent[T1][t1_node as usize];
            if t1_parent == NONE {
                continue;
            }
            undo::cut_parent(tf, T1, t1_node, um);
            undo::add_component(tf, T1, t1_node, um);
            undo::contract(tf, T1, t1_parent, um);
        }

        // Find a sibling pair in T1 (no preference in approx — just take first)
        let approx_config = BBConfig::noopt();
        let mut dummy_stats = WhiddenRuleStats::default();
        match find_sibling_pair(tf, &approx_config, &mut dummy_stats) {
            PairResult::NoPairs => break,
            PairResult::Case2 {
                t1_parent,
                t2_parent,
            } => {
                do_case2_contract(tf, t1_parent, t2_parent, um);
            }
            PairResult::Case3 {
                t1_a,
                t1_c,
                t2_a,
                t2_b: _,
                t2_c: _t2_c,
                cut_b_only,
                ..
            } => {
                let t1_parent = tf.parent[T1][t1_a as usize];
                let t2_a_parent = tf.parent[T2][t2_a as usize];
                let mut case_cuts: i32 = 0;

                if cut_b_only {
                    // COB: Only cut T2_b. Pair becomes Case 2 next iteration.
                    let t2_b = tf.sibling(T2, t2_a);
                    if t2_b != NONE {
                        let t2_b_parent = tf.parent[T2][t2_b as usize];
                        undo::cut_parent(tf, T2, t2_b, um);
                        undo::add_component(tf, T2, t2_b, um);
                        case_cuts += 1;
                        if t2_b_parent != NONE {
                            undo::contract(tf, T2, t2_b_parent, um);
                        }
                    }
                } else {
                    // Full Case 3: cut T1_a, T1_c from T1
                    undo::cut_parent(tf, T1, t1_a, um);
                    undo::add_component(tf, T1, t1_a, um);
                    if t1_parent != NONE {
                        undo::contract(tf, T1, t1_parent, um);
                    }
                    undo::cut_parent(tf, T1, t1_c, um);
                    undo::add_component(tf, T1, t1_c, um);
                    let t1_c_parent = tf.parent[T1][t1_c as usize];
                    if t1_c_parent != NONE {
                        undo::contract(tf, T1, t1_c_parent, um);
                    }

                    // Cut T2_a from T2
                    if t2_a_parent != NONE {
                        undo::cut_parent(tf, T2, t2_a, um);
                        undo::add_component(tf, T2, t2_a, um);
                        case_cuts += 1;
                        undo::contract(tf, T2, t2_a_parent, um);
                    }

                    // Cut T2_c from T2 (re-read twin in case it moved)
                    let t2_c_now = tf.twin[T1][t1_c as usize];
                    if t2_c_now != NONE {
                        let t2_c_p = tf.parent[T2][t2_c_now as usize];
                        if t2_c_p != NONE {
                            undo::cut_parent(tf, T2, t2_c_now, um);
                            undo::add_component(tf, T2, t2_c_now, um);
                            case_cuts += 1;
                            undo::contract(tf, T2, t2_c_p, um);
                        }
                    }
                }

                num_cut += case_cuts;
            }
        }
    }

    um.undo_to(cp, tf);
    num_cut
}

// ---------------------------------------------------------------------------
// Case 2: Contract matching sibling pair
// ---------------------------------------------------------------------------

fn do_case2_contract(
    tf: &mut TwinForest,
    t1_parent: NodeId,
    t2_parent: NodeId,
    um: &mut UndoMachine,
) {
    // Get children labels BEFORE detaching (right child's label is "kept")
    let t1_lc = tf.left[T1][t1_parent as usize];
    let t1_rc = tf.right[T1][t1_parent as usize];
    let kept_label = tf.label[T1][t1_rc as usize];
    let removed_label = tf.label[T1][t1_lc as usize];

    // Contract in T1: detach both children, parent becomes leaf
    undo::contract_sibling_pair(tf, T1, t1_parent, um);
    // Contract in T2: detach both children, parent becomes leaf
    undo::contract_sibling_pair(tf, T2, t2_parent, um);

    // Parent takes on "kept" label so it's visible to label-based operations
    undo::set_label(tf, T1, t1_parent, kept_label, um);
    undo::set_label(tf, T2, t2_parent, kept_label, um);
    // Update label_to_node: kept label → parent node
    undo::set_label_to_node(tf, T1, kept_label, t1_parent, um);
    undo::set_label_to_node(tf, T2, kept_label, t2_parent, um);
    // Removed label → NONE
    if removed_label != 0 {
        undo::set_label_to_node(tf, T1, removed_label, NONE, um);
        undo::set_label_to_node(tf, T2, removed_label, NONE, um);
        // Track: removed_label collapses into kept_label
        undo::set_collapsed(tf, removed_label, kept_label, um);
    }

    // Update twins: parents now represent the contracted pair
    undo::set_twin(tf, T1, t1_parent, t2_parent, um);
    undo::set_twin(tf, T2, t2_parent, t1_parent, um);
}

// ---------------------------------------------------------------------------
// Case 3: 3-way branching
// ---------------------------------------------------------------------------

fn do_case3_branch(
    tf: &mut TwinForest,
    k: i32,
    um: &mut UndoMachine,
    stats: &mut SolverStats,
    rule_stats: &mut WhiddenRuleStats,
    config: &BBConfig,
    tt: &mut Option<TranspositionTable>,
    bc: &mut Option<BoundCache>,
    bounds: ParentBounds,
    t1_a: NodeId,
    t1_c: NodeId,
    t2_a: NodeId,
    t2_b: NodeId,
    t2_c: NodeId,
    cut_b_only: bool,
    cut_a_only: bool,
    cut_c_only: bool,
    path_length: u16,
) -> Option<u32> {
    // Determine which branches to try based on optimization flags.
    // COB: cut_b_only → only B
    // RCOB: cut_a_only → only A; cut_c_only → only C
    let cob = config.cut_one_b && cut_b_only;
    let rcob_a = config.reverse_cut_one_b && cut_a_only && !cob;
    let rcob_c = config.reverse_cut_one_b && cut_c_only && !cob;

    let t2_a_parent = tf.parent[T2][t2_a as usize];
    let t2_c_parent = tf.parent[T2][t2_c as usize];
    let cob_structural = t2_a_parent != NONE && {
        let t2_ab_parent = tf.parent[T2][t2_a_parent as usize];
        t2_ab_parent != NONE && t2_ab_parent == t2_c_parent
    };
    if cob {
        rule_stats.rule_cob_fired += 1;
    }
    if rcob_a {
        rule_stats.rule_rcob_a_fired += 1;
    }
    if rcob_c {
        rule_stats.rule_rcob_c_fired += 1;
    }
    if config.cut_two_b && cut_b_only && !cob_structural {
        rule_stats.rule_cut_two_b_fired += 1;
    }

    // SC: if T2_a and T2_c are in different components, cutting B can't help.
    let separate_components = config.cut_ac_separate_components
        && !cob
        && !rcob_a
        && !rcob_c
        && find_root(tf, T2, t2_a) != find_root(tf, T2, t2_c);

    // EP: edge protection gates
    let ep = config.edge_protection;
    let ep_skip_a = ep && tf.protected[t2_a as usize];
    let ep_skip_b = ep && tf.protected[t2_b as usize];
    let ep_skip_c = ep && tf.protected[t2_c as usize];

    let skip_a = cob || rcob_c || ep_skip_a;
    let skip_b = rcob_a || rcob_c || separate_components || ep_skip_b;
    let skip_c = cob || rcob_a || ep_skip_c;

    if cob {
        rule_stats.skip_a_cob += 1;
        rule_stats.skip_c_cob += 1;
    }
    if rcob_c {
        rule_stats.skip_a_rcob_c += 1;
        rule_stats.skip_b_rcob_c += 1;
    }
    if rcob_a {
        rule_stats.skip_b_rcob_a += 1;
        rule_stats.skip_c_rcob_a += 1;
    }
    if ep_skip_a {
        rule_stats.skip_a_ep_protected += 1;
    }
    if ep_skip_b {
        rule_stats.skip_b_ep_protected += 1;
    }
    if ep_skip_c {
        rule_stats.skip_c_ep_protected += 1;
    }
    if separate_components {
        rule_stats.skip_b_separate_components += 1;
    }

    // --- Branch A: cut T2_a ---
    if !skip_a {
        let cp = um.checkpoint();
        let t2_a_parent = tf.parent[T2][t2_a as usize];
        undo::cut_parent(tf, T2, t2_a, um);
        undo::add_component(tf, T2, t2_a, um);
        if t2_a_parent != NONE {
            undo::contract(tf, T2, t2_a_parent, um);
        }

        // EP_TWO_B: when T2_c is protected and path_length==4, protect T2_b and T2_b2.
        if config.edge_protection_two_b
            && tf.protected[t2_c as usize]
            && !cut_a_only
            && path_length == 4
        {
            let balanced = t2_a_parent != NONE
                && tf.parent[T2][t2_a_parent as usize] != NONE
                && tf.parent[T2][t2_a_parent as usize]
                    == tf.parent[T2][tf.parent[T2][t2_c as usize] as usize];
            undo::protect_edge(tf, t2_b, um);
            let t2_b2 = if balanced {
                tf.sibling(T2, t2_c)
            } else {
                let bp = tf.parent[T2][t2_b as usize];
                if bp != NONE { tf.sibling(T2, bp) } else { NONE }
            };
            if t2_b2 != NONE {
                undo::protect_edge(tf, t2_b2, um);
            }
        }

        rule_stats.branch_a_attempts += 1;
        let result = branch_and_bound(tf, k - 1, um, stats, rule_stats, config, tt, bc, bounds);
        if result.is_some() {
            rule_stats.branch_a_successes += 1;
            return result;
        }
        um.undo_to(cp, tf);
    }

    // --- Branch B: cut T2_b ---
    if !skip_b {
        let cp = um.checkpoint();
        let t2_b_parent = tf.parent[T2][t2_b as usize];
        undo::cut_parent(tf, T2, t2_b, um);
        undo::add_component(tf, T2, t2_b, um);
        if t2_b_parent != NONE {
            undo::contract(tf, T2, t2_b_parent, um);
        }

        let mut k_b = k - 1;
        rule_stats.branch_b_attempts += 1;
        let result = if config.cut_all_b {
            rule_stats.rule_cut_all_b_forced += 1;
            bb_inner(
                tf,
                &mut k_b,
                um,
                stats,
                rule_stats,
                config,
                tt,
                bc,
                bounds,
                Some((t1_a, t1_c)),
            )
        } else {
            branch_and_bound(tf, k - 1, um, stats, rule_stats, config, tt, bc, bounds)
        };
        if result.is_some() {
            rule_stats.branch_b_successes += 1;
            return result;
        }
        um.undo_to(cp, tf);
    }

    // --- Branch C: cut T2_c ---
    if !skip_c {
        let cp = um.checkpoint();
        let t2_c_parent = tf.parent[T2][t2_c as usize];
        if t2_c_parent != NONE {
            undo::cut_parent(tf, T2, t2_c, um);
            undo::add_component(tf, T2, t2_c, um);
            undo::contract(tf, T2, t2_c_parent, um);
        }

        if ep && !cut_c_only {
            undo::protect_edge(tf, t2_a, um);
        }

        rule_stats.branch_c_attempts += 1;
        let result = branch_and_bound(tf, k - 1, um, stats, rule_stats, config, tt, bc, bounds);
        if result.is_some() {
            rule_stats.branch_c_successes += 1;
            return result;
        }
        um.undo_to(cp, tf);
    }

    if skip_a && skip_b && skip_c {
        rule_stats.prune_no_enabled_branches += 1;
    }

    None
}

// ---------------------------------------------------------------------------
// Solution extraction
// ---------------------------------------------------------------------------

/// Extract MAF components from the solved forest state.
fn extract_components(tf: &TwinForest) -> Vec<Tree> {
    let n = tf.num_leaves;

    // Resolve collapsed_into to final representatives (transitive closure)
    let mut collapsed: Vec<Label> = tf.collapsed_into[..=n as usize].to_vec();
    for _ in 0..n {
        let mut changed = false;
        for lbl in 1..=n {
            let target = collapsed[lbl as usize];
            if target != lbl && collapsed[target as usize] != target {
                collapsed[lbl as usize] = collapsed[target as usize];
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Collect label sets per component
    let orig_tree = tree_from_original(tf);
    let mut result = Vec::new();
    for &root in &tf.components[T1] {
        let mut current_labels = Vec::new();
        collect_labels(tf, root, &mut current_labels);
        if current_labels.is_empty() {
            continue;
        }

        // Expand: find all original labels whose representative is in this component
        let mut leafset = FixedBitSet::with_capacity(n as usize + 1);
        for &lbl in &current_labels {
            for orig in 1..=n {
                if collapsed[orig as usize] == lbl {
                    leafset.insert(orig as usize);
                }
            }
        }
        if leafset.count_ones(..) > 0 {
            let tree = Tree::component_from_leafset(&leafset, &orig_tree, n);
            result.push(tree);
        }
    }
    result
}

/// Collect current labels reachable from node.
fn collect_labels(tf: &TwinForest, node: NodeId, out: &mut Vec<Label>) {
    let lbl = tf.label[T1][node as usize];
    if lbl != 0 {
        out.push(lbl);
        return;
    }
    let lc = tf.left[T1][node as usize];
    let rc = tf.right[T1][node as usize];
    if lc != NONE {
        collect_labels(tf, lc, out);
    }
    if rc != NONE {
        collect_labels(tf, rc, out);
    }
}

/// Reconstruct original Tree from TwinForest's immutable orig_* arrays.
fn tree_from_original(tf: &TwinForest) -> Tree {
    let mut t = Tree::with_capacity(tf.num_leaves);
    t.parent = tf.orig_parent.clone();
    t.left = tf.orig_left.clone();
    t.right = tf.orig_right.clone();
    t.label = tf.orig_label.clone();
    t.label_to_node = vec![NONE; tf.num_leaves as usize + 1];
    for node in 0..tf.num_nodes[T1] as NodeId {
        let lbl = tf.orig_label[node as usize];
        if lbl != 0 && tf.orig_left[node as usize] == NONE {
            t.label_to_node[lbl as usize] = node;
        }
    }
    t.num_leaves = tf.num_leaves;
    t.root = tf.root[T1];
    t.depth = vec![0; tf.num_nodes[T1]];
    t.subtree_size = vec![0; tf.num_nodes[T1]];
    t
}

// ---------------------------------------------------------------------------
// Public accessor for the 2-approx dual lower bound (for testing/comparison)
// ---------------------------------------------------------------------------

/// Compute the Olver 2-approximation dual lower bound on an instance's
/// rSPR distance. Builds a TwinForest and calls `approx_2_lb`.
///
/// Returns D such that D ≤ OPT (rSPR distance).
pub fn approx_2_lb_for_instance(t1: &Tree, t2: &Tree, num_leaves: u32) -> i32 {
    let tf = TwinForest::from_trees(t1, t2, num_leaves);
    approx2::approx_2_lb(&tf)
}

/// Compute the 3-approximation value on an instance's rSPR distance.
/// Builds a TwinForest+UndoMachine, runs `approx_3`, and returns the raw
/// 3-approximation value (NOT divided by 3).
pub fn approx_3_for_instance(t1: &Tree, t2: &Tree, num_leaves: u32) -> i32 {
    let mut tf = TwinForest::from_trees(t1, t2, num_leaves);
    let mut um = UndoMachine::new();
    approx_3(&mut tf, &mut um)
}

// ---------------------------------------------------------------------------
// Unit tests for splitting core (Mestel 2024, Lemma 1)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod splitting_core_tests {
    use super::*;
    use klados_core::tree::{Label, NodeId};

    /// Build a tiny rooted binary tree from a manual construction.
    /// Returns the tree. Helper for constructing test cases.
    fn make_tree_balanced_4(labels: [Label; 4]) -> Tree {
        // ((l0, l1), (l2, l3))
        let mut t = Tree::with_capacity(4);
        let mk_leaf = |t: &mut Tree, lbl: Label| -> NodeId {
            let id = t.parent.len() as NodeId;
            t.parent.push(NONE);
            t.left.push(NONE);
            t.right.push(NONE);
            t.label.push(lbl);
            t.label_to_node[lbl as usize] = id;
            id
        };
        let mk_internal = |t: &mut Tree, l: NodeId, r: NodeId| -> NodeId {
            let id = t.parent.len() as NodeId;
            t.parent.push(NONE);
            t.left.push(l);
            t.right.push(r);
            t.label.push(0);
            t.parent[l as usize] = id;
            t.parent[r as usize] = id;
            id
        };
        let l0 = mk_leaf(&mut t, labels[0]);
        let l1 = mk_leaf(&mut t, labels[1]);
        let l2 = mk_leaf(&mut t, labels[2]);
        let l3 = mk_leaf(&mut t, labels[3]);
        let l01 = mk_internal(&mut t, l0, l1);
        let l23 = mk_internal(&mut t, l2, l3);
        let root = mk_internal(&mut t, l01, l23);
        t.root = root;
        t.compute_metadata();
        t
    }

    fn bits(labels: &[Label], capacity: usize) -> FixedBitSet {
        let mut s = FixedBitSet::with_capacity(capacity);
        for &l in labels {
            s.insert(l as usize);
        }
        s
    }

    /// **Test 1: Base case — single-edge clean split.**
    /// `((1,2),(3,4))` with Y={1,2}, Z={3,4}. The cherry `(1,2)` is
    /// pendant via one edge, opposite cherry `(3,4)` is pendant on the
    /// other side. Single edge cut suffices.
    #[test]
    fn splitting_core_base_case_clean_cherry_split() {
        let t1 = make_tree_balanced_4([1, 2, 3, 4]);
        // For this test we don't need an actual TwinForest with a paired
        // tree; we just need t1's structure and the active leafset.
        let t2 = make_tree_balanced_4([1, 2, 3, 4]); // same shape, irrelevant
        let tf = TwinForest::from_trees(&t1, &t2, 4);

        let active = bits(&[1, 2, 3, 4], 5);
        let y = bits(&[1, 2], 5);

        let core = splitting_core(&tf, t1.root, &active, &y);
        let ineq = splitting_core_inequality(&core);
        assert!(ineq <= 0.5 + 1e-9, "inequality violated: {}", ineq);
        // Expect a single cut of size 1.
        assert_eq!(core.len(), 1, "expected exactly one cut, got {:?}", core);
        assert_eq!(core[0].len(), 1, "expected size-1 cut, got {:?}", core);
    }

    /// **Test 2: Inductive case — interleaved Y/Z.**
    /// `((1,3),(2,4))` with Y={1,2}, Z={3,4}. The cherries mix Y and Z,
    /// so no single edge splits cleanly. Inductive case must fire.
    #[test]
    fn splitting_core_inductive_case_interleaved() {
        let t1 = make_tree_balanced_4([1, 3, 2, 4]);
        let t2 = make_tree_balanced_4([1, 3, 2, 4]);
        let tf = TwinForest::from_trees(&t1, &t2, 4);

        let active = bits(&[1, 2, 3, 4], 5);
        let y = bits(&[1, 2], 5);

        let core = splitting_core(&tf, t1.root, &active, &y);
        let ineq = splitting_core_inequality(&core);
        assert!(ineq <= 0.5 + 1e-9, "inequality violated: {} cuts={:?}", ineq, core);
        // The inductive case MUST produce a non-empty core. The paper
        // guarantees Lemma 1's existence.
        assert!(!core.is_empty(), "inductive case produced empty core for {:?}", core);
        // For ((1,3),(2,4)) with Y={1,2}: at v_left (cherry of 1,3) we
        // have (PureY={1}, PureZ={3}, Mixed_outside={2,4}). Recursion on
        // T \ {1} and T \ {3} should produce cores of size 1 each, and
        // each is prepended with e_1 or e_2 respectively. So we expect
        // two cuts, each of size 2: {edge_to_1, ...} and {edge_to_3, ...}.
        eprintln!("interleaved core: {:?}, inequality = {}", core, ineq);
    }

    /// **Test 3: Larger balanced tree.**
    /// Build a tree on 8 leaves with bipartition placing all "low" labels
    /// on Y and all "high" on Z. Verify the inequality.
    #[test]
    fn splitting_core_balanced_8() {
        // (((1,2),(3,4)),((5,6),(7,8))). Y = {1..4}, Z = {5..8}.
        // T[Y] and T[Z] are pendant, joined by a single edge → base case.
        // Build manually.
        let mut t = Tree::with_capacity(8);
        let mk_leaf = |t: &mut Tree, lbl: Label| -> NodeId {
            let id = t.parent.len() as NodeId;
            t.parent.push(NONE);
            t.left.push(NONE);
            t.right.push(NONE);
            t.label.push(lbl);
            t.label_to_node[lbl as usize] = id;
            id
        };
        let mk_internal = |t: &mut Tree, l: NodeId, r: NodeId| -> NodeId {
            let id = t.parent.len() as NodeId;
            t.parent.push(NONE);
            t.left.push(l);
            t.right.push(r);
            t.label.push(0);
            t.parent[l as usize] = id;
            t.parent[r as usize] = id;
            id
        };
        let l1 = mk_leaf(&mut t, 1);
        let l2 = mk_leaf(&mut t, 2);
        let l3 = mk_leaf(&mut t, 3);
        let l4 = mk_leaf(&mut t, 4);
        let l5 = mk_leaf(&mut t, 5);
        let l6 = mk_leaf(&mut t, 6);
        let l7 = mk_leaf(&mut t, 7);
        let l8 = mk_leaf(&mut t, 8);
        let l12 = mk_internal(&mut t, l1, l2);
        let l34 = mk_internal(&mut t, l3, l4);
        let l56 = mk_internal(&mut t, l5, l6);
        let l78 = mk_internal(&mut t, l7, l8);
        let l1234 = mk_internal(&mut t, l12, l34);
        let l5678 = mk_internal(&mut t, l56, l78);
        let root = mk_internal(&mut t, l1234, l5678);
        t.root = root;
        t.compute_metadata();
        let tf = TwinForest::from_trees(&t, &t, 8);

        let active = bits(&[1, 2, 3, 4, 5, 6, 7, 8], 9);
        let y = bits(&[1, 2, 3, 4], 9);
        let core = splitting_core(&tf, t.root, &active, &y);
        let ineq = splitting_core_inequality(&core);
        assert!(ineq <= 0.5 + 1e-9, "inequality violated: {} cuts={:?}", ineq, core);
        // Clean split: should find a single edge cut.
        assert_eq!(core.len(), 1, "expected 1 cut for clean split, got {:?}", core);
    }
}
