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
                solved: result >= 0,
            });
        }

        if result >= 0 {
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
) -> i32 {
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
) -> i32 {
    stats.nodes_explored += 1;

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
            return -1;
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
                    return -1;
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
                            return -1;
                        }
                        let (should_prune, bounds) =
                            bb_should_prune(tf, um, *k, config, bc, parent_bounds, rule_stats);
                        if should_prune {
                            rule_stats.prune_bb_approx += 1;
                            tt_store_fail!(tt, tf, *k, rule_stats);
                            return -1;
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
                        if result < 0 {
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
            return -1; // k went negative
        }

        if config.mestel_rule6 {
            match try_mestel_rule6(tf, k, um, rule_stats) {
                MestelRule6Result::Applied => continue,
                MestelRule6Result::ExhaustedBudget => {
                    rule_stats.prune_k_exhausted += 1;
                    tt_store_fail!(tt, tf, *k, rule_stats);
                    return -1;
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
                    return -1; // no store needed: already in TT
                }
            }
        }

        // --- Phase 2: Find sibling pair in T1 ---
        match find_sibling_pair(tf, config, rule_stats) {
            PairResult::NoPairs => {
                rule_stats.action_done += 1;
                return *k;
            }
            PairResult::Case2 {
                t1_parent,
                t2_parent,
            } => {
                do_case2_contract(tf, t1_parent, t2_parent, um);
                rule_stats.action_case2_contracts += 1;
                continue;
            }
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
                    return -1;
                }
                let (should_prune, bounds) =
                    bb_should_prune(tf, um, *k, config, bc, parent_bounds, rule_stats);
                if should_prune {
                    rule_stats.prune_bb_approx += 1;
                    tt_store_fail!(tt, tf, *k, rule_stats);
                    return -1;
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
                if result < 0 {
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
) -> i32 {
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
        if result >= 0 {
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
        if result >= 0 {
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
        if result >= 0 {
            rule_stats.branch_c_successes += 1;
            return result;
        }
        um.undo_to(cp, tf);
    }

    if skip_a && skip_b && skip_c {
        rule_stats.prune_no_enabled_branches += 1;
    }

    -1
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
