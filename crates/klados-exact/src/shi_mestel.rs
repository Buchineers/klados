//! Shi et al. (2018) parameterized algorithm for MAF on multiple rooted trees.
//!
//! Implements Alg-Maf from "A parameterized algorithm for the Maximum Agreement
//! Forest problem on multiple rooted multifurcating trees" (JCSS 97, 2018).
//!
//! Uses checkpoint/rollback on a SearchState to avoid cloning forests at each branch.

use fixedbitset::FixedBitSet;
use fxhash::{FxHashMap, FxHashSet};
use klados_core::{Instance, NodeId, SolverStats, Tree, NONE};

fn trace_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("SHI_MESTEL_TRACE").ok().as_deref() == Some("1"))
}

fn profile_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("SHI_MESTEL_PROFILE").ok().as_deref() == Some("1"))
}

thread_local! {
    static SPLIT_STATS: std::cell::RefCell<SplitStats> = std::cell::RefCell::new(SplitStats::default());
}

#[derive(Default)]
struct SplitStats {
    attempts: u64,
    triggered: u64,
    trees_scanned: u64,
    overlap_checks: u64,
    core_calls: u64,
    core_branches: u64,
    split_nanos: u128,
}

fn dump_split_stats() {
    if !profile_enabled() {
        return;
    }
    let line = SPLIT_STATS.with(|s| {
        let st = s.borrow();
        format!(
            "SPLIT stats: attempts={}, triggered={}, trees_scanned={}, overlap_checks={}, core_calls={}, core_branches={}, nanos={}",
            st.attempts,
            st.triggered,
            st.trees_scanned,
            st.overlap_checks,
            st.core_calls,
            st.core_branches,
            st.split_nanos
        )
    });
    eprintln!("{line}");
    if let Ok(path) = std::env::var("SHI_MESTEL_PROFILE_PATH") {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            use std::io::Write;
            let _ = writeln!(f, "{line}");
        }
    }
}

macro_rules! trace {
    ($($arg:tt)*) => {
        if trace_enabled() {
            eprintln!($($arg)*);
        }
    };
}

// ============================================================================
// XForest: forest representation (tree with cut edges)
// ============================================================================

#[derive(Clone, Debug)]
struct XForest {
    tree: Tree,
    cut_edges: FixedBitSet,
    full_leafsets: Vec<FixedBitSet>,
    live_leafsets: Vec<FixedBitSet>,
    component_roots: Vec<NodeId>,
}

impl XForest {
    fn from_tree(tree: Tree) -> Self {
        let num_nodes = tree.num_nodes();
        let num_leaves = tree.num_leaves;
        let mut full_leafsets = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let mut set = FixedBitSet::with_capacity(num_leaves as usize + 1);
            set.grow(num_leaves as usize + 1);
            full_leafsets.push(set);
        }
        for node in tree.post_order() {
            if let Some(lbl) = tree.leaf_label(node) {
                full_leafsets[node as usize].insert(lbl as usize);
            } else if let Some((l, r)) = tree.children(node) {
                let left = full_leafsets[l as usize].clone();
                let right = full_leafsets[r as usize].clone();
                full_leafsets[node as usize].union_with(&left);
                full_leafsets[node as usize].union_with(&right);
            }
        }
        let root = tree.root;
        let live_leafsets = full_leafsets.clone();
        Self {
            tree,
            cut_edges: FixedBitSet::with_capacity(num_nodes),
            full_leafsets,
            live_leafsets,
            component_roots: vec![root],
        }
    }

    fn is_cut(&self, node: NodeId) -> bool {
        self.cut_edges.contains(node as usize)
    }

    fn cut(&mut self, node: NodeId) {
        debug_assert!(node != self.tree.root, "Cannot cut above root");
        if !self.cut_edges.contains(node as usize) {
            self.cut_edges.insert(node as usize);
            self.component_roots.push(node);
            let removed = self.live_leafsets[node as usize].clone();
            let mut cur = self.tree.parent[node as usize];
            while cur != NONE {
                self.live_leafsets[cur as usize].difference_with(&removed);
                if self.is_cut(cur) {
                    break;
                }
                cur = self.tree.parent[cur as usize];
            }
        }
    }

    fn uncut(&mut self, node: NodeId) {
        debug_assert!(self.cut_edges.contains(node as usize));
        self.cut_edges.set(node as usize, false);
        if let Some(pos) = self.component_roots.iter().rposition(|&r| r == node) {
            self.component_roots.swap_remove(pos);
        }
        let restored = self.live_leafsets[node as usize].clone();
        let mut cur = self.tree.parent[node as usize];
        while cur != NONE {
            self.live_leafsets[cur as usize].union_with(&restored);
            if self.is_cut(cur) {
                break;
            }
            cur = self.tree.parent[cur as usize];
        }
    }

    fn reactivate_label(&mut self, lbl: u32) {
        let a_node = self.tree.label_to_node[lbl as usize];
        self.live_leafsets[a_node as usize].insert(lbl as usize);
        let mut cur = self.tree.parent[a_node as usize];
        while cur != NONE {
            self.live_leafsets[cur as usize].insert(lbl as usize);
            if self.is_cut(cur) {
                break;
            }
            cur = self.tree.parent[cur as usize];
        }
    }

    fn component_root(&self, mut node: NodeId) -> NodeId {
        while !self.is_cut(node) && self.tree.parent[node as usize] != NONE {
            node = self.tree.parent[node as usize];
        }
        node
    }
}

// ============================================================================
// SearchState: mutable search state with checkpoint/rollback
// ============================================================================

enum UndoEntry {
    Cut { forest_idx: usize, node: NodeId },
    Deactivate { label: u32 },
}

/// Tracks label collapses from Reduction Rule 2.
type Collapses = Vec<(u32, u32)>;

struct SearchState {
    forests: Vec<XForest>,
    collapses: Collapses,
    undo_log: Vec<UndoEntry>,
    checkpoint_stack: Vec<(usize, usize)>,
}

impl SearchState {
    fn new(forests: Vec<XForest>) -> Self {
        Self {
            forests,
            collapses: Vec::new(),
            undo_log: Vec::new(),
            checkpoint_stack: Vec::new(),
        }
    }

    fn checkpoint(&mut self) {
        self.checkpoint_stack
            .push((self.undo_log.len(), self.collapses.len()));
    }

    fn rollback(&mut self) {
        let (undo_target, collapses_target) = self.checkpoint_stack.pop().unwrap();
        while self.undo_log.len() > undo_target {
            match self.undo_log.pop().unwrap() {
                UndoEntry::Cut { forest_idx, node } => {
                    self.forests[forest_idx].uncut(node);
                }
                UndoEntry::Deactivate { label } => {
                    for f in &mut self.forests {
                        f.reactivate_label(label);
                    }
                }
            }
        }
        self.collapses.truncate(collapses_target);
    }

    fn cut_node(&mut self, forest_idx: usize, node: NodeId) {
        if node != self.forests[forest_idx].tree.root && !self.forests[forest_idx].is_cut(node) {
            self.forests[forest_idx].cut(node);
            self.undo_log.push(UndoEntry::Cut { forest_idx, node });
        }
    }

    fn add_collapse(&mut self, removed: u32, kept: u32) {
        self.collapses.push((removed, kept));
        // Deactivate the removed label in all forests
        for f in &mut self.forests {
            let a_node = f.tree.label_to_node[removed as usize];
            f.live_leafsets[a_node as usize].clear();
            let mut cur = f.tree.parent[a_node as usize];
            while cur != NONE {
                f.live_leafsets[cur as usize].set(removed as usize, false);
                if f.is_cut(cur) {
                    break;
                }
                cur = f.tree.parent[cur as usize];
            }
        }
        self.undo_log.push(UndoEntry::Deactivate { label: removed });
    }
}

// ============================================================================
// Zobrist hashing for transposition table
// ============================================================================

/// Pre-computed random values for Zobrist hashing.
/// We hash the component partition: for each label, XOR in a value
/// derived from its component ID.  Two states with identical component
/// partitions will hash identically regardless of operation order.
struct ZobristTable {
    /// label_keys[label] = random u64 for that label
    label_keys: Vec<u64>,
}

impl ZobristTable {
    fn new(num_labels: usize) -> Self {
        // Simple deterministic PRNG (splitmix64) seeded with a fixed value.
        // Determinism helps reproducibility; quality doesn't matter much.
        let mut state: u64 = 0xdeadbeef12345678;
        let mut label_keys = Vec::with_capacity(num_labels + 1);
        for _ in 0..=num_labels {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            label_keys.push(z);
        }
        Self { label_keys }
    }

    /// Hash a component partition.  The partition is a set of FixedBitSets
    /// where each label belongs to exactly one component.  We hash each
    /// component by XOR of its label keys, then XOR all component hashes
    /// together with a mixing step to distinguish different partitions that
    /// happen to have the same XOR.
    fn hash_partition(&self, components: &[FixedBitSet]) -> u64 {
        let mut h: u64 = 0;
        for comp in components {
            let mut comp_h: u64 = 0;
            for lbl in comp.ones() {
                if lbl < self.label_keys.len() {
                    comp_h ^= self.label_keys[lbl];
                }
            }
            // Mix the per-component hash before combining, so that
            // {A,B},{C} hashes differently from {A,C},{B}
            comp_h = comp_h.wrapping_mul(0x517cc1b727220a95);
            comp_h ^= comp_h >> 32;
            h ^= comp_h;
        }
        // Final mix
        h = h.wrapping_mul(0x2545f4914f6cdd1d);
        h ^= h >> 32;
        h
    }
}

/// Transposition table entry: records the minimum target_s at which
/// this state was proven infeasible.
#[derive(Clone, Copy)]
struct TTEntry {
    /// The state was proven infeasible (returned None) with this target_s.
    /// Any call with target_s <= this value can be pruned immediately.
    infeasible_at: usize,
}

/// Maximum number of entries in the transposition table to bound memory.
/// With 8GB competition limit, we can afford a larger TT.
const TT_MAX_ENTRIES: usize = 1 << 22; // ~4M entries, ~64MB

// ============================================================================
// ShiMestelSolver
// ============================================================================

pub struct ShiMestelSolver {
    stats: SolverStats,
}

impl ShiMestelSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
        }
    }

    pub fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        if instance.trees.is_empty() {
            return None;
        }
        if instance.num_trees() == 1 {
            return Some(vec![instance.trees[0].clone()]);
        }

        // --- Preprocessing: Bordewich-Semple Rule 1 (subtree reduction) ---
        let subtree_collapses = find_common_subtrees(&instance.trees, instance.num_leaves);
        if !subtree_collapses.is_empty() {
            let removed_count: usize = subtree_collapses.iter().map(|(_, r)| r.len()).sum();
            trace!(
                "subtree reduction: {} maximal common subtrees, removing {} labels ({} -> {} effective leaves)",
                subtree_collapses.len(),
                removed_count,
                instance.num_leaves,
                instance.num_leaves as usize - removed_count,
            );
            let (reduced, reverse_map) = reduce_instance(instance, &subtree_collapses);
            let reduced_result = self.solve_inner(&reduced);
            return reduced_result.map(|components| {
                expand_solution(
                    components,
                    &subtree_collapses,
                    &reverse_map,
                    &instance.trees[0],
                    instance.num_leaves,
                )
            });
        }

        self.solve_inner(instance)
    }

    /// Core solve logic (after preprocessing).
    fn solve_inner(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        let label_space = instance.num_leaves as usize;
        let forests: Vec<XForest> = instance
            .trees
            .iter()
            .map(|t| XForest::from_tree(t.clone()))
            .collect();

        let mut state = SearchState::new(forests);

        // Compute bounds on optimal component count via pairwise
        // 3-approximation. The lower bound lets us skip early fruitless
        // rounds; the upper bound (tight for 2-tree instances) lets us
        // stop early.
        let mut bounds = crate::lower_bound::maf_bounds(&instance.trees, instance.num_leaves);
        trace!("maf_bounds: lower={}, upper={}", bounds.lower, bounds.upper);

        // For multi-tree instances, compute exact pairwise distances to
        // tighten the lower bound.  Each pair is a 2-tree sub-problem
        // that we can solve quickly.  This avoids the 3x loss from the
        // approximation factor and typically saves many iterative-deepening
        // rounds on hard instances.
        if instance.num_trees() >= 3 && instance.num_leaves <= 50 {
            let exact_lb = exact_pairwise_lower_bound(
                &instance.trees,
                label_space,
                instance.num_leaves,
                bounds.lower,
                bounds.upper,
                &mut self.stats,
            );
            if exact_lb > bounds.lower {
                trace!(
                    "exact pairwise LB: {} -> {} (saved {} rounds)",
                    bounds.lower,
                    exact_lb,
                    exact_lb - bounds.lower
                );
                bounds.lower = exact_lb;
            }
        }

        // Transposition table persists across iterative deepening rounds
        let zobrist = ZobristTable::new(label_space);
        let mut tt: FxHashMap<u64, TTEntry> = FxHashMap::default();

        let solve_start = std::time::Instant::now();

        for target_s in bounds.lower..=bounds.upper {
            self.stats = SolverStats::default();
            let round_start = std::time::Instant::now();

            if let Some(result) = alg_maf(
                &mut state,
                target_s,
                label_space,
                instance.num_leaves,
                &mut self.stats,
                &zobrist,
                &mut tt,
            ) {
                let total_ms = solve_start.elapsed().as_millis();
                let round_ms = round_start.elapsed().as_millis();
                trace!(
                    "solution found: target_s={}, components={}, round={}ms, total={}ms, tt_size={}, nodes={}",
                    target_s,
                    result.len(),
                    round_ms,
                    total_ms,
                    tt.len(),
                    self.stats.nodes_explored,
                );
                dump_split_stats();
                return Some(result);
            }

            let round_ms = round_start.elapsed().as_millis();
            trace!(
                "target_s={} failed: {}ms, nodes={}, pruned={}, tt_size={}",
                target_s,
                round_ms,
                self.stats.nodes_explored,
                self.stats.branches_pruned,
                tt.len(),
            );
        }

        dump_split_stats();
        Some(trivial_forest(&instance.trees[0], instance.num_leaves))
    }
}

/// Compute a tight lower bound by solving each pair of trees exactly.
///
/// For each pair (Ti, Tj), we solve the 2-tree MAF problem exactly,
/// giving the exact rSPR distance d(Ti, Tj).  The optimal multi-tree
/// MAF cost is at least max over all pairs of d(Ti, Tj), so
/// LB_components = max(d(Ti, Tj)) + 1.
///
/// We use a per-pair time budget: if any pair takes too long, we fall
/// back to the approximation-based lower bound for that pair.
fn exact_pairwise_lower_bound(
    trees: &[Tree],
    label_space: usize,
    num_leaves: u32,
    approx_lb: usize,
    upper_bound: usize,
    _stats: &mut SolverStats,
) -> usize {
    let m = trees.len();
    let mut best_lb = approx_lb;

    // Collect all pairs with their approximate costs, sorted by cost
    // descending (most promising pairs first).
    let mut pairs: Vec<(usize, usize, usize)> = Vec::new(); // (i, j, approx_cost)
    for i in 0..m {
        for j in (i + 1)..m {
            let approx_cost = crate::lower_bound::approx_rspr_distance_pub(&trees[i], &trees[j]);
            pairs.push((i, j, approx_cost));
        }
    }
    // Sort by approx_cost descending: highest-distance pairs first
    pairs.sort_by(|a, b| b.2.cmp(&a.2));

    // Time budget: spend at most 3 seconds total
    let total_budget = std::time::Duration::from_secs(3);
    let start = std::time::Instant::now();
    let per_pair_budget = total_budget / (pairs.len() as u32).max(1);

    // Track exact distances for the additive bound (cuts = components - 1)
    let mut exact_dist: Vec<Vec<Option<usize>>> = vec![vec![None; m]; m];

    for &(i, j, approx_cost) in &pairs {
        if start.elapsed() >= total_budget {
            break;
        }

        // Skip pairs whose approximate UB (approx_cost + 1) can't improve best_lb
        // AND whose approx distance can't help the additive bound.
        // For the additive bound, we need exact distances, so we still want to
        // compute them even if they don't individually beat best_lb.
        // But skip if approx_cost is very small (won't contribute much).
        if approx_cost + 1 <= best_lb && approx_cost < 3 {
            exact_dist[i][j] = Some(approx_cost);
            exact_dist[j][i] = Some(approx_cost);
            continue;
        }

        // Create 2-tree sub-instance
        let sub_forests: Vec<XForest> = vec![
            XForest::from_tree(trees[i].clone()),
            XForest::from_tree(trees[j].clone()),
        ];
        let zobrist = ZobristTable::new(label_space);
        let mut tt: FxHashMap<u64, TTEntry> = FxHashMap::default();

        // Start iterative deepening from the approximation-based LB for this pair.
        let pair_lb_start = (approx_cost + 2) / 3;

        let pair_start = std::time::Instant::now();
        let mut pair_exact_components = pair_lb_start + 1; // fallback
        let mut timed_out = false;

        for target_s in (pair_lb_start + 1)..=(approx_cost + 1) {
            if pair_start.elapsed() >= per_pair_budget {
                timed_out = true;
                break;
            }

            let mut sub_state = SearchState::new(sub_forests.iter().map(|f| f.clone()).collect());
            let mut sub_stats = SolverStats::default();
            if alg_maf(
                &mut sub_state,
                target_s,
                label_space,
                num_leaves,
                &mut sub_stats,
                &zobrist,
                &mut tt,
            )
            .is_some()
            {
                pair_exact_components = target_s;
                trace!(
                    "exact pair ({},{}) = {} components",
                    i,
                    j,
                    pair_exact_components
                );
                break;
            }
            pair_exact_components = target_s + 1;
        }

        // Store exact distance (cuts = components - 1)
        let exact_cuts = pair_exact_components - 1;
        if !timed_out {
            exact_dist[i][j] = Some(exact_cuts);
            exact_dist[j][i] = Some(exact_cuts);
        } else {
            // Timed out: use the last proven lower bound (may be incomplete)
            exact_dist[i][j] = Some(exact_cuts);
            exact_dist[j][i] = Some(exact_cuts);
        }

        if pair_exact_components > best_lb {
            best_lb = pair_exact_components;
        }

        // Early termination: if we've matched the UB, no need to check more pairs
        if best_lb >= upper_bound {
            return best_lb;
        }
    }

    // Additive multi-tree lower bound using exact pairwise distances:
    // OPT_cuts >= sum_{j!=i} d(i,j) / (m-1) for any reference tree i.
    // This is tighter than the approximation-based additive bound because
    // we don't lose the factor-3 from the cherry approximation.
    if m >= 3 {
        for i in 0..m {
            let mut sum_d = 0usize;
            let mut all_known = true;
            for j in 0..m {
                if i == j {
                    continue;
                }
                if let Some(d) = exact_dist[i][j] {
                    sum_d += d;
                } else {
                    all_known = false;
                    // Use approximation-based lower bound for unknown pairs
                    let approx = crate::lower_bound::approx_rspr_distance_pub(&trees[i], &trees[j]);
                    sum_d += (approx + 2) / 3; // ceil(approx/3) as conservative estimate
                }
            }
            let denom = m - 1;
            let lb_cuts = (sum_d + denom - 1) / denom;
            let lb_components = lb_cuts + 1;
            if lb_components > best_lb {
                trace!(
                    "additive exact LB from ref {}: {} (sum_d={}, m-1={}, all_known={})",
                    i,
                    lb_components,
                    sum_d,
                    denom,
                    all_known
                );
                best_lb = lb_components;
            }
        }
    }

    best_lb
}

impl super::ExactSolver for ShiMestelSolver {
    fn name(&self) -> &'static str {
        "shi-mestel"
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        ShiMestelSolver::solve(self, instance)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

// ============================================================================
// Alg-Maf: Main recursive algorithm
// ============================================================================

fn max_order_from_cached(comp_sets: &[Vec<FixedBitSet>]) -> usize {
    comp_sets.iter().map(|cs| cs.len()).max().unwrap_or(1)
}

fn alg_maf(
    state: &mut SearchState,
    target_s: usize,
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    zobrist: &ZobristTable,
    tt: &mut FxHashMap<u64, TTEntry>,
) -> Option<Vec<Tree>> {
    stats.nodes_explored += 1;
    state.checkpoint();

    // Interleaved R1/R2 reduction to fixpoint
    let comp_sets = loop {
        // Apply Reduction Rule 1 exhaustively; returns fresh component sets
        let comp_sets = apply_reduction_rules_state(state, label_space);

        // Budget check after R1
        let cur_order = max_order_from_cached(&comp_sets);
        if cur_order > target_s {
            stats.branches_pruned += 1;
            state.rollback();
            return None;
        }

        // If not LSI, use BR-LSI (branch and recurse)
        if !all_pairs_lsi_cached(&comp_sets) {
            let result = br_lsi_step(
                state,
                target_s,
                label_space,
                num_leaves,
                stats,
                &comp_sets,
                zobrist,
                tt,
            );
            state.rollback();
            return result;
        }

        // LSI satisfied. Try one R2 collapse; if it fires, loop back to re-check R1
        if let Some((a, b)) = find_common_sibling_pair(&state.forests, label_space) {
            trace!("R2: collapsing common sibling-pair ({}, {})", a, b);
            state.add_collapse(a, b);
            continue; // R2 fired -> re-run R1 (may enable new cuts)
        }

        break comp_sets; // No R1 or R2 applicable, fully reduced
    };

    // --- Transposition table probe ---
    // After full reduction (R1 + LSI + R2 to fixpoint), the component
    // partition is canonical.  Hash it and check whether we've already
    // proven this state infeasible at this or higher target_s.
    let comps_f0 = &comp_sets[0];
    let tt_hash = zobrist.hash_partition(comps_f0);
    if let Some(entry) = tt.get(&tt_hash) {
        if target_s <= entry.infeasible_at {
            stats.branches_pruned += 1;
            state.rollback();
            return None;
        }
    }

    // Split-or-decompose: split overlapping components if any
    let (split_applied, split_result) = apply_split_branching_cached(
        state,
        target_s,
        label_space,
        num_leaves,
        stats,
        comps_f0,
        zobrist,
        tt,
    );
    if split_applied {
        if split_result.is_none() {
            tt_insert(tt, tt_hash, target_s);
        }
        state.rollback();
        return split_result;
    }

    // Classify components using pre-computed component sets
    let (all_comps, non_iso_comps) = classify_components_cached(&state.forests, comps_f0);

    if non_iso_comps.is_empty() {
        let result =
            extract_maf_components(&state.forests[0], &state.collapses, label_space, num_leaves);
        state.rollback();
        return Some(result);
    }

    // Budget check with fresh component count (fixes stale cur_order bug)
    let remaining = target_s.saturating_sub(all_comps.len());
    if remaining == 0 {
        stats.branches_pruned += 1;
        tt_insert(tt, tt_hash, target_s);
        state.rollback();
        return None;
    }

    // Component decomposition
    if non_iso_comps.len() >= 2 {
        let result = solve_decomposed(
            &state.forests,
            target_s,
            &state.collapses,
            label_space,
            num_leaves,
            &non_iso_comps,
            &all_comps,
            stats,
        );
        if result.is_none() {
            tt_insert(tt, tt_hash, target_s);
        }
        state.rollback();
        return result;
    }

    // Single non-iso component: MSS pair branching
    let (a, b) = match find_best_sibling_pair(&state.forests, label_space) {
        Some(pair) => pair,
        None => {
            tt_insert(tt, tt_hash, target_s);
            state.rollback();
            return None;
        }
    };

    trace!("MSS pair: a={}, b={}, remaining={}", a, b, remaining);
    let result = apply_case_2_branching(
        state,
        target_s,
        a,
        b,
        label_space,
        num_leaves,
        stats,
        zobrist,
        tt,
    );
    if result.is_none() {
        tt_insert(tt, tt_hash, target_s);
    }
    state.rollback();
    result
}

/// Insert or update a transposition table entry, respecting the size bound.
fn tt_insert(tt: &mut FxHashMap<u64, TTEntry>, hash: u64, target_s: usize) {
    if let Some(entry) = tt.get_mut(&hash) {
        // Keep the higher infeasible_at (more information)
        if target_s > entry.infeasible_at {
            entry.infeasible_at = target_s;
        }
    } else if tt.len() < TT_MAX_ENTRIES {
        tt.insert(
            hash,
            TTEntry {
                infeasible_at: target_s,
            },
        );
    }
    // If table is full and entry doesn't exist, we just don't insert.
    // A more sophisticated replacement policy could be added later.
}

// ============================================================================
// BR-LSI step (Section 3): Find LSI violation and branch
// ============================================================================

fn br_lsi_step(
    state: &mut SearchState,
    target_s: usize,
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    comp_sets: &[Vec<FixedBitSet>],
    zobrist: &ZobristTable,
    tt: &mut FxHashMap<u64, TTEntry>,
) -> Option<Vec<Tree>> {
    let (i, j) = match find_violating_pair_cached(comp_sets) {
        Some(pair) => pair,
        None => return None,
    };

    let (target_idx, v1, v2) = if let Some((_v, v1, v2)) =
        find_branching_vertex_cached(&state.forests[i], &state.forests[j], &comp_sets[j])
    {
        (i, v1, v2)
    } else if let Some((_v, v1, v2)) =
        find_branching_vertex_cached(&state.forests[j], &state.forests[i], &comp_sets[i])
    {
        (j, v1, v2)
    } else {
        trace!("no branching vertex found for pair ({}, {})", i, j);
        stats.branches_pruned += 1;
        return None;
    };

    trace!("BR1: forest={}, v1={}, v2={}", target_idx, v1, v2);

    // Branch 1: cut v1
    if v1 != state.forests[target_idx].tree.root && !state.forests[target_idx].is_cut(v1) {
        state.checkpoint();
        state.cut_node(target_idx, v1);
        if let Some(result) = alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    // Branch 2: cut v2
    if v2 != state.forests[target_idx].tree.root && !state.forests[target_idx].is_cut(v2) {
        state.checkpoint();
        state.cut_node(target_idx, v2);
        if let Some(result) = alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    stats.branches_pruned += 1;
    None
}

// ============================================================================
// Reduction Rule 1 (Section 3.1)
// ============================================================================

/// Apply R1 exhaustively, returning the final component sets for all forests.
///
/// Optimized to only recompute the component sets for the forest that was
/// modified on each iteration, rather than all forests.
fn apply_reduction_rules_state(
    state: &mut SearchState,
    label_space: usize,
) -> Vec<Vec<FixedBitSet>> {
    let mut scratch = FixedBitSet::with_capacity(label_space + 1);
    let nf = state.forests.len();

    // Initial computation of all component sets
    let mut comp_sets: Vec<Vec<FixedBitSet>> = state
        .forests
        .iter()
        .map(|f| component_leaf_sets_xf(f, label_space))
        .collect();

    loop {
        let mut changed = false;
        'outer: for i in 0..nf {
            for j in 0..nf {
                if i == j {
                    continue;
                }
                if let Some(node) =
                    find_r1_cut(&state.forests[i], &comp_sets[j], label_space, &mut scratch)
                {
                    trace!("R1: cut node {} in forest {}", node, i);
                    state.cut_node(i, node);
                    // Only recompute the modified forest's components
                    comp_sets[i] = component_leaf_sets_xf(&state.forests[i], label_space);
                    changed = true;
                    break 'outer;
                }
            }
        }
        if !changed {
            return comp_sets;
        }
    }
}

/// Find a node in forest_i that should be cut according to R1 (relative to fj_components).
///
/// R1 says: cut above `node` if the set of labels below `node` equals the
/// union of those fj-components whose intersection with the whole component
/// is fully contained in `node_ls`.  We check this without allocating by
/// doing two passes over fj_components with bitwise operations on slices.
fn find_r1_cut(
    forest_i: &XForest,
    fj_components: &[FixedBitSet],
    label_space: usize,
    scratch: &mut FixedBitSet,
) -> Option<NodeId> {
    for node in forest_i.tree.pre_order() {
        if forest_i.is_cut(node) || node == forest_i.tree.root {
            continue;
        }
        if forest_i.live_leafsets[node as usize].count_ones(..) == 0 {
            continue;
        }
        let parent = forest_i.tree.parent[node as usize];
        if parent == NONE {
            continue;
        }

        let node_ls = &forest_i.live_leafsets[node as usize];
        let node_count = node_ls.count_ones(..);
        let comp_root = forest_i.component_root(node);
        let comp_ls = &forest_i.live_leafsets[comp_root as usize];
        let comp_count = comp_ls.count_ones(..);

        if node_count == 0 || node_count >= comp_count {
            continue;
        }

        // Build union_matching in scratch without allocating
        scratch.clear();
        scratch.grow(label_space + 1);
        let mut union_count = 0usize;

        for fj_comp in fj_components {
            // Check: does fj_comp ∩ comp_ls ≠ ∅ and fj_comp ∩ comp_ls ⊆ node_ls?
            // We do this with raw slice ops to avoid allocating an intersection.
            let fj_sl = fj_comp.as_slice();
            let comp_sl = comp_ls.as_slice();
            let node_sl = node_ls.as_slice();
            let len = fj_sl.len().min(comp_sl.len());

            let mut has_inter = false;
            let mut subset_of_node = true;
            for k in 0..len {
                let inter_word = fj_sl[k] & comp_sl[k];
                if inter_word != 0 {
                    has_inter = true;
                    let node_word = if k < node_sl.len() { node_sl[k] } else { 0 };
                    if inter_word & !node_word != 0 {
                        subset_of_node = false;
                        break;
                    }
                }
            }

            if has_inter && subset_of_node {
                // Union the intersection into scratch
                let scratch_sl = scratch.as_mut_slice();
                for k in 0..len {
                    let inter_word = fj_sl[k] & comp_sl[k];
                    if k < scratch_sl.len() {
                        let old = scratch_sl[k];
                        scratch_sl[k] |= inter_word;
                        union_count += (scratch_sl[k].count_ones() - old.count_ones()) as usize;
                    }
                }
                // Early exit: if union already exceeds node_ls, this node won't match
                if union_count > node_count {
                    break;
                }
            }
        }

        if union_count == node_count && union_count > 0 {
            // Verify: scratch == node_ls (same bits set)
            if scratch.as_slice() == node_ls.as_slice()
                || (scratch.as_slice().len() >= node_ls.as_slice().len()
                    && scratch.as_slice()[..node_ls.as_slice().len()] == *node_ls.as_slice()
                    && scratch.as_slice()[node_ls.as_slice().len()..]
                        .iter()
                        .all(|&w| w == 0))
            {
                return Some(node);
            }
        }
    }
    None
}

// ============================================================================
// LSI checking (Section 2.2)
// ============================================================================

fn all_pairs_lsi_cached(comp_sets: &[Vec<FixedBitSet>]) -> bool {
    if comp_sets.len() <= 1 {
        return true;
    }
    // Compare sorted component partitions.  Instead of allocating key vecs,
    // we sort by the raw slice representation (which is deterministic for
    // FixedBitSets of the same capacity).
    let ref_sorted = sorted_partition_hashes(&comp_sets[0]);
    for cs in &comp_sets[1..] {
        if sorted_partition_hashes(cs) != ref_sorted {
            return false;
        }
    }
    true
}

/// Hash each component and return a sorted vec of hashes for comparison.
/// Much cheaper than allocating Vec<Vec<usize>> keys.
fn sorted_partition_hashes(components: &[FixedBitSet]) -> Vec<u64> {
    let mut hashes: Vec<u64> = components.iter().map(|s| hash_bitset(s)).collect();
    hashes.sort_unstable();
    hashes
}

/// Fast hash of a FixedBitSet (for partition comparison, not for TT).
fn hash_bitset(s: &FixedBitSet) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &word in s.as_slice() {
        h ^= word as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
        h ^= (word >> 32) as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn find_violating_pair_cached(comp_sets: &[Vec<FixedBitSet>]) -> Option<(usize, usize)> {
    for i in 0..comp_sets.len() {
        for j in (i + 1)..comp_sets.len() {
            if !lsi_pair(&comp_sets[i], &comp_sets[j]) {
                return Some((i, j));
            }
        }
    }
    None
}

fn lsi_pair(a: &[FixedBitSet], b: &[FixedBitSet]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    sorted_partition_hashes(a) == sorted_partition_hashes(b)
}

// ============================================================================
// Branching Rule 1: find branching vertex (Section 3.1, Case 1)
// ============================================================================

fn find_branching_vertex_cached(
    fi: &XForest,
    _fj: &XForest,
    fj_components: &[FixedBitSet],
) -> Option<(NodeId, NodeId, NodeId)> {
    for node in fi.tree.pre_order() {
        if fi.live_leafsets[node as usize].count_ones(..) == 0 {
            continue;
        }
        let children = active_children_xf(fi, node);
        if children.len() < 2 {
            continue;
        }

        let (c1, c2) = (children[0], children[1]);
        let ls1 = &fi.live_leafsets[c1 as usize];
        let ls2 = &fi.live_leafsets[c2 as usize];

        for comp in fj_components {
            let c1_inter = has_intersection(ls1, comp);
            let c2_inter = has_intersection(ls2, comp);

            if c1_inter && !c2_inter && is_subset(ls1, comp) {
                return Some((node, c1, c2));
            }
            if c2_inter && !c1_inter && is_subset(ls2, comp) {
                return Some((node, c2, c1));
            }
        }
    }
    None
}

// ============================================================================
// Reduction Rule 2 (Section 4): Common sibling-pair collapse
// ============================================================================

fn find_common_sibling_pair(forests: &[XForest], label_space: usize) -> Option<(u32, u32)> {
    if forests.is_empty() {
        return None;
    }
    let pairs = find_all_sibling_pairs(&forests[0], label_space);
    'outer: for (a, b) in &pairs {
        for forest in &forests[1..] {
            if !is_sibling_pair_in_forest(forest, *a, *b) {
                continue 'outer;
            }
        }
        return Some((*a, *b));
    }
    None
}

fn find_all_sibling_pairs(forest: &XForest, _label_space: usize) -> Vec<(u32, u32)> {
    let mut pairs = Vec::new();
    for node in forest.tree.pre_order() {
        if forest.live_leafsets[node as usize].count_ones(..) == 0 {
            continue;
        }
        if forest.tree.is_leaf(node) {
            continue;
        }
        let children = forest_children(forest, node);
        if children.len() == 2 {
            let c1 = children[0];
            let c2 = children[1];
            let c1_leaf = forest_is_leaf(forest, c1);
            let c2_leaf = forest_is_leaf(forest, c2);
            if c1_leaf && c2_leaf {
                let lbl1 = forest.tree.leaf_label(c1);
                let lbl2 = forest.tree.leaf_label(c2);
                if let (Some(l1), Some(l2)) = (lbl1, lbl2) {
                    pairs.push((l1.min(l2), l1.max(l2)));
                }
            }
        }
    }
    pairs
}

fn is_sibling_pair_in_forest(forest: &XForest, a: u32, b: u32) -> bool {
    let a_node = forest.tree.label_to_node[a as usize];
    let b_node = forest.tree.label_to_node[b as usize];
    if forest.live_leafsets[a_node as usize].count_ones(..) == 0
        || forest.live_leafsets[b_node as usize].count_ones(..) == 0
    {
        return false;
    }
    let pa = forest_parent_leaf(forest, a_node);
    let pb = forest_parent_leaf(forest, b_node);
    if pa == NONE || pa != pb {
        return false;
    }
    let children = forest_children(forest, pa);
    if children.len() != 2 {
        return false;
    }
    let c1_is_a = forest_resolves_to(forest, children[0], a_node);
    let c2_is_b = forest_resolves_to(forest, children[1], b_node);
    let c1_is_b = forest_resolves_to(forest, children[0], b_node);
    let c2_is_a = forest_resolves_to(forest, children[1], a_node);
    (c1_is_a && c2_is_b) || (c1_is_b && c2_is_a)
}

fn forest_resolves_to(forest: &XForest, start: NodeId, target: NodeId) -> bool {
    let mut cur = start;
    loop {
        if cur == target {
            return true;
        }
        if forest.tree.is_leaf(cur) {
            return cur == target;
        }
        let children = active_children_xf(forest, cur);
        if children.len() == 1 {
            cur = children[0];
        } else {
            return cur == target;
        }
    }
}

// ============================================================================
// Forest navigation (with forced contraction)
// ============================================================================

fn forest_children(forest: &XForest, node: NodeId) -> Children {
    let mut out = Children::new();
    if let Some((left, right)) = forest.tree.children(node) {
        if left != NONE
            && !forest.is_cut(left)
            && forest.live_leafsets[left as usize].count_ones(..) > 0
        {
            out.push(descend_to_effective(forest, left));
        }
        if right != NONE
            && !forest.is_cut(right)
            && forest.live_leafsets[right as usize].count_ones(..) > 0
        {
            out.push(descend_to_effective(forest, right));
        }
    }
    out
}

fn descend_to_effective(forest: &XForest, mut node: NodeId) -> NodeId {
    loop {
        if forest.tree.is_leaf(node) {
            return node;
        }
        let children = active_children_xf(forest, node);
        if children.len() == 1 {
            node = children[0];
        } else {
            return node;
        }
    }
}

fn forest_is_leaf(forest: &XForest, node: NodeId) -> bool {
    if forest.tree.is_leaf(node) {
        return true;
    }
    active_children_xf(forest, node).is_empty()
}

fn forest_parent_leaf(forest: &XForest, node: NodeId) -> NodeId {
    if node == forest.tree.root || forest.is_cut(node) {
        return NONE;
    }
    let mut cur = forest.tree.parent[node as usize];
    if cur == NONE {
        return NONE;
    }
    loop {
        let active = active_children_xf(forest, cur);
        if active.len() >= 2 {
            return cur;
        }
        if forest.is_cut(cur) {
            return NONE;
        }
        let p = forest.tree.parent[cur as usize];
        if p == NONE {
            return cur;
        }
        cur = p;
    }
}

fn forest_lca(forest: &XForest, mut a: NodeId, mut b: NodeId) -> NodeId {
    let depth = &forest.tree.depth;
    // Walk deeper node up to same depth, respecting cut edges
    while depth[a as usize] > depth[b as usize] {
        if forest.is_cut(a) {
            return NONE;
        }
        a = forest.tree.parent[a as usize];
        if a == NONE {
            return NONE;
        }
    }
    while depth[b as usize] > depth[a as usize] {
        if forest.is_cut(b) {
            return NONE;
        }
        b = forest.tree.parent[b as usize];
        if b == NONE {
            return NONE;
        }
    }
    // Walk both up until they meet
    while a != b {
        if forest.is_cut(a) || forest.is_cut(b) {
            return NONE;
        }
        a = forest.tree.parent[a as usize];
        b = forest.tree.parent[b as usize];
        if a == NONE || b == NONE {
            return NONE;
        }
    }
    a
}

// ============================================================================
// Split-or-decompose: SPLIT branching on overlapping components
// ============================================================================

fn apply_split_branching_cached(
    state: &mut SearchState,
    target_s: usize,
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    comps: &[FixedBitSet],
    zobrist: &ZobristTable,
    tt: &mut FxHashMap<u64, TTEntry>,
) -> (bool, Option<Vec<Tree>>) {
    let start = if profile_enabled() {
        Some(std::time::Instant::now())
    } else {
        None
    };
    if profile_enabled() {
        SPLIT_STATS.with(|s| s.borrow_mut().attempts += 1);
    }
    if comps.len() <= 1 {
        return (false, None);
    }

    if let Some((forest_idx, comp_a, comp_b, edge_child)) = find_best_overlap(&state.forests, comps)
    {
        if profile_enabled() {
            SPLIT_STATS.with(|s| s.borrow_mut().triggered += 1);
        }
        trace!(
            "SPLIT: forest={}, comp_a={}, comp_b={}, edge_child={}",
            forest_idx,
            comp_a,
            comp_b,
            edge_child
        );

        // Branch 1: comp_a does NOT get the edge -> split comp_a
        if let Some(result) = split_component_branch(
            state,
            target_s,
            label_space,
            num_leaves,
            stats,
            forest_idx,
            &comps[comp_a],
            edge_child,
            zobrist,
            tt,
        ) {
            if let Some(t0) = start {
                let dt = t0.elapsed().as_nanos();
                SPLIT_STATS.with(|s| s.borrow_mut().split_nanos += dt);
            }
            return (true, Some(result));
        }

        // Branch 2: comp_b does NOT get the edge -> split comp_b
        if let Some(result) = split_component_branch(
            state,
            target_s,
            label_space,
            num_leaves,
            stats,
            forest_idx,
            &comps[comp_b],
            edge_child,
            zobrist,
            tt,
        ) {
            if let Some(t0) = start {
                let dt = t0.elapsed().as_nanos();
                SPLIT_STATS.with(|s| s.borrow_mut().split_nanos += dt);
            }
            return (true, Some(result));
        }

        stats.branches_pruned += 1;
        if let Some(t0) = start {
            let dt = t0.elapsed().as_nanos();
            SPLIT_STATS.with(|s| s.borrow_mut().split_nanos += dt);
        }
        return (true, None);
    }

    if let Some(t0) = start {
        let dt = t0.elapsed().as_nanos();
        SPLIT_STATS.with(|s| s.borrow_mut().split_nanos += dt);
    }
    (false, None)
}

fn split_component_branch(
    state: &mut SearchState,
    target_s: usize,
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    forest_idx: usize,
    comp: &FixedBitSet,
    edge_child: NodeId,
    zobrist: &ZobristTable,
    tt: &mut FxHashMap<u64, TTEntry>,
) -> Option<Vec<Tree>> {
    let tree = &state.forests[forest_idx].tree;
    let full_leafsets = &state.forests[forest_idx].full_leafsets;

    let mut y = full_leafsets[edge_child as usize].clone();
    y.intersect_with(comp);
    if y.count_ones(..) == 0 {
        return None;
    }
    let mut z = comp.clone();
    z.difference_with(&y);
    if z.count_ones(..) == 0 {
        return None;
    }

    let core = splitting_core(tree, full_leafsets, comp, &y, &z);
    if core.is_empty() {
        return None;
    }
    if profile_enabled() {
        SPLIT_STATS.with(|s| {
            let mut st = s.borrow_mut();
            st.core_calls += 1;
            st.core_branches += core.len() as u64;
        });
    }

    for cut in core {
        state.checkpoint();
        for child in cut {
            state.cut_node(forest_idx, child);
        }
        if let Some(result) = alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    None
}

/// Scan all trees for overlapping components and select the overlap whose
/// splitting core is smallest (fewest branches = best branching factor).
fn find_best_overlap(
    forests: &[XForest],
    comps: &[FixedBitSet],
) -> Option<(usize, usize, usize, NodeId)> {
    if profile_enabled() {
        SPLIT_STATS.with(|s| s.borrow_mut().trees_scanned += forests.len() as u64);
    }

    let mut best: Option<(usize, usize, usize, NodeId)> = None;
    let mut best_score = usize::MAX;

    for (forest_idx, forest) in forests.iter().enumerate() {
        if let Some((a, b, edge_child)) = find_overlap_in_tree(forest, comps) {
            // Evaluate splitting core quality for both components
            let score = overlap_score(forest, comps, a, b, edge_child);
            if score < best_score {
                best_score = score;
                best = Some((forest_idx, a, b, edge_child));
                // A single-cut core is optimal; stop early
                if best_score <= 1 {
                    break;
                }
            }
        }
    }
    best
}

/// Score an overlap by the minimum splitting core size across both components.
/// Lower is better (fewer branches in the split).
fn overlap_score(
    forest: &XForest,
    comps: &[FixedBitSet],
    comp_a: usize,
    comp_b: usize,
    edge_child: NodeId,
) -> usize {
    let tree = &forest.tree;
    let full_leafsets = &forest.full_leafsets;

    let mut best = usize::MAX;
    for &comp_idx in &[comp_a, comp_b] {
        let comp = &comps[comp_idx];
        let mut y = full_leafsets[edge_child as usize].clone();
        y.intersect_with(comp);
        if y.count_ones(..) == 0 {
            continue;
        }
        let mut z = comp.clone();
        z.difference_with(&y);
        if z.count_ones(..) == 0 {
            continue;
        }
        let core = splitting_core(tree, full_leafsets, comp, &y, &z);
        if !core.is_empty() && core.len() < best {
            best = core.len();
        }
    }
    best
}

fn find_overlap_in_tree(forest: &XForest, comps: &[FixedBitSet]) -> Option<(usize, usize, NodeId)> {
    let num_nodes = forest.tree.num_nodes();
    let mut edge_owner: Vec<Option<usize>> = vec![None; num_nodes];
    let comp_sizes: Vec<usize> = comps.iter().map(|c| c.count_ones(..)).collect();

    for child in forest.tree.pre_order() {
        if child == forest.tree.root {
            continue;
        }
        if comp_sizes.iter().all(|&s| s <= 1) {
            break;
        }
        let child_ls = &forest.full_leafsets[child as usize];
        for (idx, comp) in comps.iter().enumerate() {
            if comp_sizes[idx] <= 1 {
                continue;
            }
            if profile_enabled() {
                SPLIT_STATS.with(|s| s.borrow_mut().overlap_checks += 1);
            }
            let inter = count_intersection(comp, child_ls);
            if inter == 0 || inter == comp_sizes[idx] {
                continue;
            }
            if let Some(other) = edge_owner[child as usize] {
                if other != idx {
                    return Some((other, idx, child));
                }
            } else {
                edge_owner[child as usize] = Some(idx);
            }
        }
    }
    None
}

fn splitting_core(
    tree: &Tree,
    full_leafsets: &[FixedBitSet],
    x: &FixedBitSet,
    y: &FixedBitSet,
    z: &FixedBitSet,
) -> Vec<Vec<NodeId>> {
    // Base case: find a single edge that separates Y and Z
    for child in tree.pre_order() {
        if child == tree.root {
            continue;
        }
        let mut side = full_leafsets[child as usize].clone();
        side.intersect_with(x);
        if side.count_ones(..) == 0 || side == *x {
            continue;
        }
        if side == *y || side == *z {
            return vec![vec![child]];
        }
    }

    // Find a node with one pure-Y side and one pure-Z side
    for v in tree.pre_order() {
        let mut pure_y_edge: Option<NodeId> = None;
        let mut pure_y_side: Option<FixedBitSet> = None;
        let mut pure_z_edge: Option<NodeId> = None;
        let mut pure_z_side: Option<FixedBitSet> = None;
        let mut has_mixed = false;

        for neighbor in neighbors(tree, v) {
            let side = side_leafset(tree, full_leafsets, x, v, neighbor);
            if side.count_ones(..) == 0 {
                continue;
            }
            if is_subset(&side, y) {
                if pure_y_edge.is_none() {
                    pure_y_edge = Some(edge_child(tree, v, neighbor));
                    pure_y_side = Some(side);
                }
            } else if is_subset(&side, z) {
                if pure_z_edge.is_none() {
                    pure_z_edge = Some(edge_child(tree, v, neighbor));
                    pure_z_side = Some(side);
                }
            } else {
                has_mixed = true;
            }
        }

        if pure_y_edge.is_some() && pure_z_edge.is_some() && has_mixed {
            let e1 = pure_y_edge.unwrap();
            let e2 = pure_z_edge.unwrap();
            let side_y = pure_y_side.unwrap();
            let side_z = pure_z_side.unwrap();

            let mut x1 = x.clone();
            x1.difference_with(&side_y);
            let mut y1 = y.clone();
            y1.difference_with(&side_y);
            let z1 = z.clone();

            let mut x2 = x.clone();
            x2.difference_with(&side_z);
            let y2 = y.clone();
            let mut z2 = z.clone();
            z2.difference_with(&side_z);

            let mut out = Vec::new();
            for mut k in splitting_core(tree, full_leafsets, &x1, &y1, &z1) {
                k.push(e1);
                out.push(k);
            }
            for mut k in splitting_core(tree, full_leafsets, &x2, &y2, &z2) {
                k.push(e2);
                out.push(k);
            }
            return out;
        }
    }

    Vec::new()
}

fn neighbors(tree: &Tree, node: NodeId) -> Vec<NodeId> {
    let mut out = Vec::with_capacity(3);
    let parent = tree.parent[node as usize];
    if parent != NONE {
        out.push(parent);
    }
    if let Some((l, r)) = tree.children(node) {
        if l != NONE {
            out.push(l);
        }
        if r != NONE {
            out.push(r);
        }
    }
    out
}

fn edge_child(tree: &Tree, from: NodeId, to: NodeId) -> NodeId {
    if tree.parent[to as usize] == from {
        to
    } else if tree.parent[from as usize] == to {
        from
    } else {
        to
    }
}

fn side_leafset(
    tree: &Tree,
    full_leafsets: &[FixedBitSet],
    x: &FixedBitSet,
    node: NodeId,
    neighbor: NodeId,
) -> FixedBitSet {
    if tree.parent[neighbor as usize] == node {
        let mut side = full_leafsets[neighbor as usize].clone();
        side.intersect_with(x);
        side
    } else if tree.parent[node as usize] == neighbor {
        let mut side = x.clone();
        side.difference_with(&full_leafsets[node as usize]);
        side
    } else {
        FixedBitSet::new()
    }
}

fn count_intersection(a: &FixedBitSet, b: &FixedBitSet) -> usize {
    let a_sl = a.as_slice();
    let b_sl = b.as_slice();
    let len = a_sl.len().min(b_sl.len());
    let mut total = 0usize;
    for i in 0..len {
        total += (a_sl[i] & b_sl[i]).count_ones() as usize;
    }
    total
}

// ============================================================================
// Case 2 branching (Section 4.1): sibling-pair case
// ============================================================================

fn find_best_sibling_pair(forests: &[XForest], label_space: usize) -> Option<(u32, u32)> {
    let mut seen = fxhash::FxHashSet::default();
    let mut all_pairs: Vec<(u32, u32)> = Vec::new();
    for forest in forests {
        for pair in find_all_sibling_pairs(forest, label_space) {
            if seen.insert(pair) {
                all_pairs.push(pair);
            }
        }
    }

    if all_pairs.len() <= 1 {
        return all_pairs.first().copied();
    }

    let mut best_pair = all_pairs[0];
    let mut best_score: i32 = i32::MIN;

    for &(a, b) in &all_pairs {
        let score = score_sibling_pair(forests, a, b);
        if score > best_score {
            best_score = score;
            best_pair = (a, b);
        }
    }

    Some(best_pair)
}

/// Score a sibling pair by the branching case it triggers.
/// Higher score = better (lower branching factor).
///
/// Case 2.2.1 (deterministic cut, branching factor 1): score 10000
/// BR 2.1 with |E_F| >= 3 (vector (1,1,>=3), factor ~2.21): score 5000 + |E_F|
/// BR 2.1 with |E_F| == 2 (vector (1,1,2), factor ~2.41): score 3000
/// BR 2.2.2 (vector (1,1,1), factor 3): score 0
/// No useful branching: score -1000
fn score_sibling_pair(forests: &[XForest], a: u32, b: u32) -> i32 {
    let e_sizes: Vec<usize> = forests.iter().map(|f| compute_e_f(f, a, b).len()).collect();
    let max_e = e_sizes.iter().copied().max().unwrap_or(0);

    let omega1: Vec<usize> = e_sizes
        .iter()
        .enumerate()
        .filter(|(_, e)| **e == 1)
        .map(|(i, _)| i)
        .collect();

    // Check for Case 2.2.1 (deterministic): all omega1 forests have same LCA leafset
    if !omega1.is_empty() {
        let lca_sets: Vec<FixedBitSet> = omega1
            .iter()
            .map(|&i| lca_leafset(&forests[i], a, b))
            .collect();
        let all_same_lca = lca_sets.windows(2).all(|w| w[0] == w[1]);
        if all_same_lca {
            return 10000; // Best: deterministic reduction, no branching
        }
    }

    // BR 2.1: 3-way branch (1, 1, |E_F|)
    if max_e >= 3 {
        return 5000 + max_e as i32;
    }
    if max_e >= 2 {
        return 3000;
    }

    // BR 2.2.2: 3-way branch (1, 1, 1) — worst branching factor
    if !omega1.is_empty() {
        return 0;
    }

    // No useful branching found
    -1000
}

fn compute_e_f(forest: &XForest, a: u32, b: u32) -> Vec<NodeId> {
    let a_node = forest.tree.label_to_node[a as usize];
    let b_node = forest.tree.label_to_node[b as usize];

    if forest.live_leafsets[a_node as usize].count_ones(..) == 0
        || forest.live_leafsets[b_node as usize].count_ones(..) == 0
    {
        return Vec::new();
    }

    let lca = forest_lca(forest, a_node, b_node);
    if lca == NONE {
        return Vec::new();
    }

    // Use a flat bool vector instead of HashSet for O(1) lookup without hashing.
    let n = forest.tree.num_nodes();
    let mut on_path = vec![false; n];
    // Collect path nodes in a small stack-vec for iteration.
    let mut path_nodes_buf = Vec::with_capacity(32);

    on_path[a_node as usize] = true;
    on_path[b_node as usize] = true;
    on_path[lca as usize] = true;
    path_nodes_buf.push(lca);

    let mut cur = a_node;
    while cur != lca {
        if forest.is_cut(cur) {
            break;
        }
        let p = forest.tree.parent[cur as usize];
        if p == NONE {
            break;
        }
        if !on_path[p as usize] {
            on_path[p as usize] = true;
            path_nodes_buf.push(p);
        }
        cur = p;
    }
    cur = b_node;
    while cur != lca {
        if forest.is_cut(cur) {
            break;
        }
        let p = forest.tree.parent[cur as usize];
        if p == NONE {
            break;
        }
        if !on_path[p as usize] {
            on_path[p as usize] = true;
            path_nodes_buf.push(p);
        }
        cur = p;
    }

    let mut e_f = Vec::new();
    for &path_node in &path_nodes_buf {
        if let Some((left, right)) = forest.tree.children(path_node) {
            if left != NONE
                && !forest.is_cut(left)
                && forest.live_leafsets[left as usize].count_ones(..) > 0
                && !on_path[left as usize]
            {
                e_f.push(left);
            }
            if right != NONE
                && !forest.is_cut(right)
                && forest.live_leafsets[right as usize].count_ones(..) > 0
                && !on_path[right as usize]
            {
                e_f.push(right);
            }
        }
    }
    e_f
}

fn lca_leafset(forest: &XForest, a: u32, b: u32) -> FixedBitSet {
    let a_node = forest.tree.label_to_node[a as usize];
    let b_node = forest.tree.label_to_node[b as usize];
    let lca = forest_lca(forest, a_node, b_node);
    if lca == NONE {
        FixedBitSet::new()
    } else {
        forest.live_leafsets[lca as usize].clone()
    }
}

fn apply_case_2_branching(
    state: &mut SearchState,
    target_s: usize,
    a: u32,
    b: u32,
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    zobrist: &ZobristTable,
    tt: &mut FxHashMap<u64, TTEntry>,
) -> Option<Vec<Tree>> {
    let e_sets: Vec<Vec<NodeId>> = state.forests.iter().map(|f| compute_e_f(f, a, b)).collect();
    let max_e = e_sets.iter().map(|e| e.len()).max().unwrap_or(0);

    if max_e >= 2 {
        return apply_branching_rule_2_1(
            state,
            target_s,
            a,
            b,
            &e_sets,
            label_space,
            num_leaves,
            stats,
            zobrist,
            tt,
        );
    }

    let omega1: Vec<usize> = e_sets
        .iter()
        .enumerate()
        .filter(|(_, e)| e.len() == 1)
        .map(|(i, _)| i)
        .collect();

    if omega1.is_empty() {
        return None;
    }

    let lca_sets: Vec<FixedBitSet> = omega1
        .iter()
        .map(|&i| lca_leafset(&state.forests[i], a, b))
        .collect();
    let all_same_lca = lca_sets.windows(2).all(|w| w[0] == w[1]);

    if all_same_lca {
        trace!("Case 2.2.1: reduction for ({}, {})", a, b);
        return apply_reduction_rule_2_2_1(
            state,
            target_s,
            &e_sets,
            label_space,
            num_leaves,
            stats,
            zobrist,
            tt,
        );
    }

    trace!("Case 2.2.2: branching for ({}, {})", a, b);
    apply_branching_rule_2_2_2(
        state,
        target_s,
        a,
        b,
        &e_sets,
        label_space,
        num_leaves,
        stats,
        zobrist,
        tt,
    )
}

/// BR 2.1: 3-way branch. [1] cut a, [2] cut b, [3] cut E_F
fn apply_branching_rule_2_1(
    state: &mut SearchState,
    target_s: usize,
    a: u32,
    b: u32,
    e_sets: &[Vec<NodeId>],
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    zobrist: &ZobristTable,
    tt: &mut FxHashMap<u64, TTEntry>,
) -> Option<Vec<Tree>> {
    trace!("BR 2.1: a={}, b={}", a, b);

    // Branch [1]: cut a in all forests
    {
        state.checkpoint();
        for idx in 0..state.forests.len() {
            let a_node = state.forests[idx].tree.label_to_node[a as usize];
            state.cut_node(idx, a_node);
        }
        if let Some(result) = alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    // Branch [2]: cut b in all forests
    {
        state.checkpoint();
        for idx in 0..state.forests.len() {
            let b_node = state.forests[idx].tree.label_to_node[b as usize];
            state.cut_node(idx, b_node);
        }
        if let Some(result) = alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    // Branch [3]: cut E_F for all forests
    {
        state.checkpoint();
        let mut any_cut = false;
        for (i, e_nodes) in e_sets.iter().enumerate() {
            for &node in e_nodes {
                if node != state.forests[i].tree.root && !state.forests[i].is_cut(node) {
                    state.cut_node(i, node);
                    any_cut = true;
                }
            }
        }
        if any_cut {
            if let Some(result) =
                alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
            {
                state.rollback();
                return Some(result);
            }
        }
        state.rollback();
    }

    None
}

/// Case 2.2.1: deterministic reduction (cut E_F)
fn apply_reduction_rule_2_2_1(
    state: &mut SearchState,
    target_s: usize,
    e_sets: &[Vec<NodeId>],
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    zobrist: &ZobristTable,
    tt: &mut FxHashMap<u64, TTEntry>,
) -> Option<Vec<Tree>> {
    for (i, e_nodes) in e_sets.iter().enumerate() {
        for &node in e_nodes {
            state.cut_node(i, node);
        }
    }
    alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
}

/// Case 2.2.2: 3-way branch
fn apply_branching_rule_2_2_2(
    state: &mut SearchState,
    target_s: usize,
    a: u32,
    b: u32,
    e_sets: &[Vec<NodeId>],
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
    zobrist: &ZobristTable,
    tt: &mut FxHashMap<u64, TTEntry>,
) -> Option<Vec<Tree>> {
    trace!("BR 2.2.2: a={}, b={}", a, b);

    // Branch [1]: cut a
    {
        state.checkpoint();
        for idx in 0..state.forests.len() {
            let a_node = state.forests[idx].tree.label_to_node[a as usize];
            state.cut_node(idx, a_node);
        }
        if let Some(result) = alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    // Branch [2]: cut b
    {
        state.checkpoint();
        for idx in 0..state.forests.len() {
            let b_node = state.forests[idx].tree.label_to_node[b as usize];
            state.cut_node(idx, b_node);
        }
        if let Some(result) = alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
        {
            state.rollback();
            return Some(result);
        }
        state.rollback();
    }

    // Branch [3]: cut E_F
    {
        state.checkpoint();
        let mut any_cut = false;
        for (i, e_nodes) in e_sets.iter().enumerate() {
            for &node in e_nodes {
                if node != state.forests[i].tree.root && !state.forests[i].is_cut(node) {
                    state.cut_node(i, node);
                    any_cut = true;
                }
            }
        }
        if any_cut {
            if let Some(result) =
                alg_maf(state, target_s, label_space, num_leaves, stats, zobrist, tt)
            {
                state.rollback();
                return Some(result);
            }
        }
        state.rollback();
    }

    None
}

// ============================================================================
// Isomorphism, decomposition, and MAF extraction
// ============================================================================

fn build_collapsed_into(collapses: &Collapses, num_leaves: u32) -> Vec<u32> {
    let mut collapsed_into: Vec<u32> = (0..=num_leaves).collect();
    for &(removed, kept) in collapses {
        collapsed_into[removed as usize] = kept;
    }
    for lbl in 1..=num_leaves {
        let mut cur = lbl;
        while collapsed_into[cur as usize] != cur {
            cur = collapsed_into[cur as usize];
        }
        collapsed_into[lbl as usize] = cur;
    }
    collapsed_into
}

fn expand_leafset(
    comp_ls: &FixedBitSet,
    collapsed_into: &[u32],
    num_leaves: u32,
    label_space: usize,
) -> FixedBitSet {
    let mut expanded = FixedBitSet::with_capacity(label_space + 1);
    expanded.grow(label_space + 1);
    for lbl in 1..=num_leaves {
        let target = collapsed_into[lbl as usize];
        if comp_ls.contains(target as usize) {
            expanded.insert(lbl as usize);
        }
    }
    expanded
}

fn build_component_tree(expanded: &FixedBitSet, reference_tree: &Tree, num_leaves: u32) -> Tree {
    if expanded.count_ones(..) == 1 {
        let lbl = expanded.ones().next().unwrap() as u32;
        make_singleton_tree(lbl, num_leaves)
    } else {
        reference_tree.prune_to_leafset(expanded)
    }
}

fn classify_components_cached(
    forests: &[XForest],
    all_comps: &[FixedBitSet],
) -> (Vec<FixedBitSet>, Vec<FixedBitSet>) {
    let all_comps = all_comps.to_vec();
    let mut non_iso = Vec::new();

    for comp_ls in &all_comps {
        if comp_ls.count_ones(..) <= 1 {
            continue;
        }
        let ref_canon = tree_canonical_for_labels(&forests[0].tree, comp_ls);
        let mut all_same = true;
        for forest in &forests[1..] {
            if tree_canonical_for_labels(&forest.tree, comp_ls) != ref_canon {
                all_same = false;
                break;
            }
        }
        if !all_same {
            non_iso.push(comp_ls.clone());
        }
    }

    // Sort non-iso components by size (smallest first).
    // In decomposition, solving small components first yields tighter
    // progressive budget bounds for the remaining larger components.
    non_iso.sort_by_key(|c| c.count_ones(..));

    (all_comps, non_iso)
}

/// Check if a component is pendant in a tree: its embedding shares no edges
/// with other components. Equivalent to: there exists a node v in the tree
/// such that the full leafset of v, intersected with the union of all active
/// labels, equals exactly the component's label set.
#[allow(dead_code)]
fn is_pendant_in_tree(
    tree: &Tree,
    full_leafsets: &[FixedBitSet],
    comp: &FixedBitSet,
    all_active: &FixedBitSet,
) -> bool {
    for node in tree.pre_order() {
        if tree.is_leaf(node) {
            continue;
        }
        let node_ls = &full_leafsets[node as usize];
        // Check: node_ls ∩ all_active == comp
        let node_sl = node_ls.as_slice();
        let active_sl = all_active.as_slice();
        let comp_sl = comp.as_slice();
        let len = node_sl.len().min(active_sl.len()).min(comp_sl.len());

        let mut matches = true;
        for k in 0..len {
            if (node_sl[k] & active_sl[k]) != comp_sl[k] {
                matches = false;
                break;
            }
        }
        if matches {
            // Check remaining words in comp_sl are zero
            for k in len..comp_sl.len() {
                if comp_sl[k] != 0 {
                    matches = false;
                    break;
                }
            }
        }
        if matches {
            // Check remaining words in (node ∩ active) beyond comp are zero
            for k in len..node_sl.len().min(active_sl.len()) {
                if node_sl[k] & active_sl[k] != 0 {
                    matches = false;
                    break;
                }
            }
        }
        if matches {
            return true;
        }
    }
    false
}

/// Check if a component is pendant in ALL trees.
#[allow(dead_code)]
fn is_pendant_in_all_trees(
    forests: &[XForest],
    comp: &FixedBitSet,
    all_active: &FixedBitSet,
) -> bool {
    forests
        .iter()
        .all(|f| is_pendant_in_tree(&f.tree, &f.full_leafsets, comp, all_active))
}

/// Solve independent component sub-problems (Recursion Rule with t=1 shallow probe)
fn solve_decomposed(
    forests: &[XForest],
    target_s: usize,
    collapses: &Collapses,
    label_space: usize,
    num_leaves: u32,
    non_iso_comps: &[FixedBitSet],
    all_comps: &[FixedBitSet],
    stats: &mut SolverStats,
) -> Option<Vec<Tree>> {
    let cur_order = all_comps.len();
    let remaining = target_s.saturating_sub(cur_order);

    trace!(
        "Decomposing: {} components, {} non-isomorphic, remaining={}",
        all_comps.len(),
        non_iso_comps.len(),
        remaining
    );

    let collapsed_into = build_collapsed_into(collapses, num_leaves);
    let nc = non_iso_comps.len();

    // Phase 1: Shallow probe (t=1) for ALL components to establish lower bounds.
    // Each component gets its own TT and Zobrist table that persist through
    // the deep solve phase (Phase 2), giving cross-round TT reuse.
    let mut sub_forest_cache: Vec<Vec<XForest>> = Vec::with_capacity(nc);
    let mut shallow_results: Vec<Option<Vec<Tree>>> = Vec::with_capacity(nc);
    let mut lower_bounds: Vec<usize> = Vec::with_capacity(nc);
    let mut total_lower_bound: usize = 0;
    let mut comp_zobrist: Vec<ZobristTable> = Vec::with_capacity(nc);
    let mut comp_tt: Vec<FxHashMap<u64, TTEntry>> = Vec::with_capacity(nc);

    for comp_ls in non_iso_comps {
        let sub_trees: Vec<Tree> = forests
            .iter()
            .map(|f| f.tree.prune_to_leafset(comp_ls))
            .collect();
        let sub_forests: Vec<XForest> = sub_trees
            .iter()
            .map(|t| XForest::from_tree(t.clone()))
            .collect();

        let zobrist = ZobristTable::new(label_space);
        let mut tt: FxHashMap<u64, TTEntry> = FxHashMap::default();

        // Use pairwise approximation to get a tighter initial lower bound
        // than the shallow probe alone (which only distinguishes 1 vs 2).
        let approx_lb = crate::lower_bound::maf_bounds(&sub_trees, num_leaves).lower;

        if approx_lb <= 1 {
            // Approximation says 1 component suffices -- but we know it's
            // non-isomorphic so it needs at least 2. Use shallow probe.
            let mut sub_state = SearchState::new(sub_forests.clone());
            if let Some(result) = alg_maf(
                &mut sub_state,
                2,
                label_space,
                num_leaves,
                stats,
                &zobrist,
                &mut tt,
            ) {
                shallow_results.push(Some(result));
                lower_bounds.push(1);
                total_lower_bound += 1;
            } else {
                shallow_results.push(None);
                lower_bounds.push(2);
                total_lower_bound += 2;
            }
        } else if approx_lb == 2 {
            // Try shallow probe at target_s=2
            let mut sub_state = SearchState::new(sub_forests.clone());
            if let Some(result) = alg_maf(
                &mut sub_state,
                2,
                label_space,
                num_leaves,
                stats,
                &zobrist,
                &mut tt,
            ) {
                shallow_results.push(Some(result));
                lower_bounds.push(1);
                total_lower_bound += 1;
            } else {
                shallow_results.push(None);
                lower_bounds.push(2);
                total_lower_bound += 2;
            }
        } else {
            // approx_lb >= 3: skip shallow probe, use approximation bound
            // The component needs at least approx_lb components = approx_lb - 1 cuts
            shallow_results.push(None);
            lower_bounds.push(approx_lb - 1); // cuts, not components
            total_lower_bound += approx_lb - 1;
        }
        sub_forest_cache.push(sub_forests);
        comp_zobrist.push(zobrist);
        comp_tt.push(tt);
    }

    if total_lower_bound > remaining {
        stats.branches_pruned += 1;
        return None;
    }

    // Phase 2: Deep solve with progressive budget tightening.
    // After solving each component, we replace its lower bound with the
    // actual cost, which tightens budgets for subsequent components.
    // TT from Phase 1 is reused so iterative deepening benefits from
    // prior infeasibility proofs.
    let mut total_cost: usize = 0;
    let mut component_results: Vec<Vec<Tree>> = Vec::with_capacity(nc);

    for (idx, _comp_ls) in non_iso_comps.iter().enumerate() {
        if let Some(ref result) = shallow_results[idx] {
            total_cost += 1;
            let mut trees = Vec::new();
            for sub_tree in result {
                let mut sub_ls = FixedBitSet::with_capacity(label_space + 1);
                sub_ls.grow(label_space + 1);
                for node in sub_tree.pre_order() {
                    if sub_tree.is_leaf(node) {
                        let lbl = sub_tree.label[node as usize];
                        if lbl > 0 {
                            sub_ls.insert(lbl as usize);
                        }
                    }
                }
                let expanded = expand_leafset(&sub_ls, &collapsed_into, num_leaves, label_space);
                trees.push(build_component_tree(
                    &expanded,
                    &forests[0].tree,
                    num_leaves,
                ));
            }
            component_results.push(trees);
        } else {
            // Compute budget using current (progressively tightened) lower bounds.
            // For already-solved components, lower_bounds[i] == actual cost.
            let other_lb: usize = lower_bounds
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != idx)
                .map(|(_, &lb)| lb)
                .sum();
            let budget = remaining.saturating_sub(other_lb);
            let comp_num_labels = non_iso_comps[idx].count_ones(..) as usize;

            let mut found = false;
            let start_cost = lower_bounds[idx].max(2);
            for cost in start_cost..=budget.min(comp_num_labels) {
                let mut sub_state = SearchState::new(sub_forest_cache[idx].clone());
                if let Some(result) = alg_maf(
                    &mut sub_state,
                    1 + cost,
                    label_space,
                    num_leaves,
                    stats,
                    &comp_zobrist[idx],
                    &mut comp_tt[idx],
                ) {
                    total_cost += cost;
                    // Progressive tightening: set this component's LB to actual cost
                    lower_bounds[idx] = cost;
                    let mut trees = Vec::new();
                    for sub_tree in &result {
                        let mut sub_ls = FixedBitSet::with_capacity(label_space + 1);
                        sub_ls.grow(label_space + 1);
                        for node in sub_tree.pre_order() {
                            if sub_tree.is_leaf(node) {
                                let lbl = sub_tree.label[node as usize];
                                if lbl > 0 {
                                    sub_ls.insert(lbl as usize);
                                }
                            }
                        }
                        let expanded =
                            expand_leafset(&sub_ls, &collapsed_into, num_leaves, label_space);
                        trees.push(build_component_tree(
                            &expanded,
                            &forests[0].tree,
                            num_leaves,
                        ));
                    }
                    component_results.push(trees);
                    found = true;
                    break;
                }
            }
            if !found {
                return None;
            }
        }
    }

    if total_cost > remaining {
        return None;
    }

    // Assemble result
    let mut result_trees: Vec<Tree> = Vec::new();

    // Pre-compute hashes of non-iso components for fast lookup
    let non_iso_hashes: Vec<u64> = non_iso_comps.iter().map(|c| hash_bitset(c)).collect();

    for comp_ls in all_comps {
        let h = hash_bitset(comp_ls);
        if non_iso_comps
            .iter()
            .zip(non_iso_hashes.iter())
            .any(|(c, &ch)| ch == h && c.as_slice() == comp_ls.as_slice())
        {
            continue;
        }
        let expanded = expand_leafset(comp_ls, &collapsed_into, num_leaves, label_space);
        result_trees.push(build_component_tree(
            &expanded,
            &forests[0].tree,
            num_leaves,
        ));
    }

    for trees in component_results {
        result_trees.extend(trees);
    }

    Some(result_trees)
}

fn extract_maf_components(
    forest: &XForest,
    collapses: &Collapses,
    label_space: usize,
    num_leaves: u32,
) -> Vec<Tree> {
    let collapsed_into = build_collapsed_into(collapses, num_leaves);
    let comps = component_leaf_sets_xf(forest, label_space);
    let mut result = Vec::new();
    for comp_ls in &comps {
        if comp_ls.count_ones(..) == 0 {
            continue;
        }
        let expanded = expand_leafset(comp_ls, &collapsed_into, num_leaves, label_space);
        result.push(build_component_tree(&expanded, &forest.tree, num_leaves));
    }
    result
}

// ============================================================================
// XForest navigation helpers
// ============================================================================

/// Stack-allocated list of 0-2 children (binary trees never have more).
#[derive(Clone, Copy)]
struct Children {
    nodes: [NodeId; 2],
    len: u8,
}

impl Children {
    #[inline(always)]
    fn new() -> Self {
        Self {
            nodes: [NONE, NONE],
            len: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, node: NodeId) {
        self.nodes[self.len as usize] = node;
        self.len += 1;
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len as usize
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl std::ops::Index<usize> for Children {
    type Output = NodeId;
    #[inline(always)]
    fn index(&self, idx: usize) -> &NodeId {
        &self.nodes[idx]
    }
}

fn active_children_xf(forest: &XForest, node: NodeId) -> Children {
    let tree = &forest.tree;
    let mut out = Children::new();
    if let Some((left, right)) = tree.children(node) {
        if left != NONE
            && !forest.is_cut(left)
            && forest.live_leafsets[left as usize].count_ones(..) > 0
        {
            out.push(left);
        }
        if right != NONE
            && !forest.is_cut(right)
            && forest.live_leafsets[right as usize].count_ones(..) > 0
        {
            out.push(right);
        }
    }
    out
}

/// Get the leaf set for each component in the forest.
///
/// A component root is either a cut node or the tree root.
/// Its `live_leafsets` already contains exactly the active labels in
/// that component, so we just clone those -- no tree walk needed.
fn component_leaf_sets_xf(forest: &XForest, _label_space: usize) -> Vec<FixedBitSet> {
    let mut components = Vec::new();
    // The tree root is always a component root (if it has live labels)
    let root_ls = &forest.live_leafsets[forest.tree.root as usize];
    if root_ls.count_ones(..) > 0 {
        components.push(root_ls.clone());
    }
    // Every cut node is a component root
    for node in forest.cut_edges.ones() {
        let ls = &forest.live_leafsets[node];
        if ls.count_ones(..) > 0 {
            components.push(ls.clone());
        }
    }
    components
}

// ============================================================================
// Preprocessing: Bordewich-Semple subtree reduction (Rule 1)
// ============================================================================

/// Compute a bottom-up Merkle hash for every node in a tree.
/// Leaf hash = mix(label), Internal hash = order-independent combine(left, right).
/// Returns a Vec indexed by NodeId.
fn subtree_hashes(tree: &Tree) -> Vec<u64> {
    let n = tree.num_nodes();
    let mut hashes = vec![0u64; n];
    for node in tree.post_order() {
        if tree.is_leaf(node) {
            let lbl = tree.label[node as usize] as u64;
            let mut h = lbl.wrapping_mul(0x9e3779b97f4a7c15);
            h ^= h >> 30;
            hashes[node as usize] = h;
        } else {
            let (l, r) = tree.children(node).unwrap();
            let lh = hashes[l as usize];
            let rh = hashes[r as usize];
            let (lo, hi) = if lh <= rh { (lh, rh) } else { (rh, lh) };
            let mut h = lo.wrapping_mul(0xbf58476d1ce4e5b9);
            h ^= hi;
            h = h.wrapping_mul(0x94d049bb133111eb);
            h ^= h >> 31;
            hashes[node as usize] = h;
        }
    }
    hashes
}

/// Find maximal common pendant subtrees across all trees and return the
/// labels to collapse.
///
/// Returns a list of (representative_label, Vec<removed_labels>) for each
/// maximal common subtree. The representative is kept; the removed labels
/// are pruned from the instance.
///
/// A subtree rooted at node v in the reference tree is "common" if:
/// 1. Its leaf set forms a clade (subtree) in every other tree.
/// 2. The restricted topology is identical (same Merkle hash).
///
/// Uses bottom-up hash comparison for efficiency. The expensive clade check
/// and hash verification are only done for internal nodes with ≥ 2 leaves.
fn find_common_subtrees(trees: &[Tree], num_leaves: u32) -> Vec<(u32, Vec<u32>)> {
    if trees.len() < 2 || num_leaves < 4 {
        return Vec::new();
    }

    let ref_tree = &trees[0];

    // Fast pre-check using sibling arrays: for each leaf label, record which
    // label is its sibling (if the sibling is also a leaf). If no sibling pair
    // from the ref tree is preserved in all other trees, skip full analysis.
    // Cost: O(n) per tree, no heap allocation beyond the sibling arrays.
    {
        // Build sibling array for ref tree: sibling_of[label] = sibling_label
        // (0 if sibling is not a leaf)
        let n = num_leaves as usize;
        let mut ref_sibling = vec![0u32; n + 1];
        for node in ref_tree.post_order() {
            if let Some((l, r)) = ref_tree.children(node) {
                if ref_tree.is_leaf(l) && ref_tree.is_leaf(r) {
                    let ll = ref_tree.label[l as usize];
                    let rl = ref_tree.label[r as usize];
                    if ll != 0 && rl != 0 {
                        ref_sibling[ll as usize] = rl;
                        ref_sibling[rl as usize] = ll;
                    }
                }
            }
        }

        // Check: does ref tree have any leaf sibling pairs at all?
        let has_any_pair = ref_sibling[1..].iter().any(|&s| s != 0);
        if !has_any_pair {
            return Vec::new();
        }

        // For each other tree, check if any ref sibling pair is also a
        // sibling pair there. Use a flat array for O(1) lookup.
        for other in &trees[1..] {
            let mut other_sibling = vec![0u32; n + 1];
            for node in other.post_order() {
                if let Some((l, r)) = other.children(node) {
                    if other.is_leaf(l) && other.is_leaf(r) {
                        let ll = other.label[l as usize];
                        let rl = other.label[r as usize];
                        if ll != 0 && rl != 0 {
                            other_sibling[ll as usize] = rl;
                            other_sibling[rl as usize] = ll;
                        }
                    }
                }
            }

            // Check if ANY ref pair survives in this tree
            let mut any_survives = false;
            for lbl in 1..=n {
                let rs = ref_sibling[lbl];
                if rs != 0 && other_sibling[lbl] == rs {
                    any_survives = true;
                    break;
                }
            }
            if !any_survives {
                return Vec::new();
            }
        }
    }

    // Full analysis: compute per-node hashes for ALL trees.
    let all_hashes: Vec<Vec<u64>> = trees.iter().map(|t| subtree_hashes(t)).collect();

    // For each other tree, build a set of (hash, subtree_size) pairs.
    let other_hash_sets: Vec<FxHashSet<(u64, u32)>> = trees[1..]
        .iter()
        .enumerate()
        .map(|(ti, other_tree)| {
            let mut set: FxHashSet<(u64, u32)> = FxHashSet::default();
            for node in other_tree.post_order() {
                if !other_tree.is_leaf(node) {
                    let h = all_hashes[ti + 1][node as usize];
                    let sz = other_tree.subtree_size[node as usize];
                    set.insert((h, sz));
                }
            }
            set
        })
        .collect();

    // Walk ref tree bottom-up. For each internal node with >= 2 leaves,
    // check if its (hash, subtree_size) appears in all other trees' sets.
    let num_nodes = ref_tree.num_nodes();
    let mut is_common = vec![false; num_nodes];

    for node in ref_tree.post_order() {
        if ref_tree.is_leaf(node) {
            is_common[node as usize] = true;
            continue;
        }

        let ref_size = ref_tree.subtree_size[node as usize];
        if ref_size < 2 || ref_size == num_leaves {
            continue;
        }

        let ref_hash = all_hashes[0][node as usize];
        let key = (ref_hash, ref_size);

        // Quick check: does this (hash, size) pair appear in every other tree?
        let all_match = other_hash_sets.iter().all(|set| set.contains(&key));

        is_common[node as usize] = all_match;
    }

    // Step 4: Find maximal common subtrees and collect their labels.
    // A common internal node is maximal if its parent is NOT common.
    let mut collapses = Vec::new();
    for node in ref_tree.post_order() {
        if ref_tree.is_leaf(node) || !is_common[node as usize] {
            continue;
        }
        let ref_size = ref_tree.subtree_size[node as usize];
        if ref_size < 2 {
            continue;
        }

        // Check if parent is also common (if so, this node is not maximal)
        let parent = ref_tree.parent[node as usize];
        if parent != NONE && is_common[parent as usize] {
            continue;
        }

        // This is a maximal common subtree. Collect its labels via DFS.
        let mut labels = Vec::with_capacity(ref_size as usize);
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            if ref_tree.is_leaf(n) {
                labels.push(ref_tree.label[n as usize]);
            } else if let Some((l, r)) = ref_tree.children(n) {
                stack.push(r);
                stack.push(l);
            }
        }
        labels.sort_unstable();

        let representative = labels[0];
        let removed: Vec<u32> = labels[1..].to_vec();
        if !removed.is_empty() {
            collapses.push((representative, removed));
        }
    }

    collapses
}

/// Build a reduced instance by collapsing common subtrees and relabeling
/// to a compact label space.
///
/// Returns (reduced_instance, reverse_label_map) where reverse_label_map[new_label] = old_label.
fn reduce_instance(instance: &Instance, collapses: &[(u32, Vec<u32>)]) -> (Instance, Vec<u32>) {
    if collapses.is_empty() {
        // Identity reverse map
        let reverse: Vec<u32> = (0..=instance.num_leaves).collect();
        return (instance.clone(), reverse);
    }

    // Build the set of labels to keep (representatives + non-collapsed labels)
    let label_space = instance.num_leaves as usize;
    let mut keep = FixedBitSet::with_capacity(label_space + 1);
    for lbl in 1..=instance.num_leaves {
        keep.insert(lbl as usize);
    }
    for (_, removed) in collapses {
        for &lbl in removed {
            keep.set(lbl as usize, false);
        }
    }

    // Build old->new label mapping (consecutive 1..=new_n)
    let kept_labels: Vec<u32> = keep.ones().map(|i| i as u32).collect();
    let new_num_leaves = kept_labels.len() as u32;

    // label_map[old_label] = new_label (0 means removed)
    let mut label_map = vec![0u32; label_space + 1];
    // reverse_map[new_label] = old_label (index 0 unused)
    let mut reverse_map = vec![0u32; new_num_leaves as usize + 1];

    for (new_idx, &old_lbl) in kept_labels.iter().enumerate() {
        let new_lbl = (new_idx + 1) as u32; // 1-indexed
        label_map[old_lbl as usize] = new_lbl;
        reverse_map[new_lbl as usize] = old_lbl;
    }

    // Build reduced + relabeled trees
    let reduced_trees: Vec<Tree> = instance
        .trees
        .iter()
        .map(|t| {
            // First prune removed labels, then relabel to compact space
            let pruned = t.prune_to_leafset(&keep);
            pruned.relabel(&label_map, new_num_leaves)
        })
        .collect();

    (Instance::new(reduced_trees, new_num_leaves), reverse_map)
}

/// Expand a solution from the reduced (relabeled) label space back to the original.
///
/// `reverse_map[new_label] = old_label` maps reduced labels back to originals.
/// Each component tree that contains a representative label gets all the
/// collapsed labels added back, and the component tree is rebuilt from
/// the original reference tree.
fn expand_solution(
    reduced_components: Vec<Tree>,
    collapses: &[(u32, Vec<u32>)],
    reverse_map: &[u32],
    original_ref_tree: &Tree,
    original_num_leaves: u32,
) -> Vec<Tree> {
    if collapses.is_empty() {
        return reduced_components;
    }

    let label_space = original_num_leaves as usize;

    // Build mapping: representative (old label) -> all labels (including representative)
    let mut rep_to_all: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for (rep, removed) in collapses {
        let mut all = vec![*rep];
        all.extend_from_slice(removed);
        rep_to_all.insert(*rep, all);
    }

    let mut result = Vec::with_capacity(reduced_components.len());
    for comp in &reduced_components {
        // Collect this component's labels, mapping new->old and expanding collapsed subtrees
        let mut expanded_ls = FixedBitSet::with_capacity(label_space + 1);
        for new_lbl in comp.leaves() {
            let old_lbl = reverse_map[new_lbl as usize];
            if let Some(all_labels) = rep_to_all.get(&old_lbl) {
                for &l in all_labels {
                    expanded_ls.insert(l as usize);
                }
            } else {
                expanded_ls.insert(old_lbl as usize);
            }
        }

        // Build component tree from original reference tree
        result.push(build_component_tree(
            &expanded_ls,
            original_ref_tree,
            original_num_leaves,
        ));
    }
    result
}

// ============================================================================
// Pure utility functions
// ============================================================================

/// Compute a canonical hash of a tree restricted to a label-set.
/// Uses a Merkle-style bottom-up hash: leaves hash their label,
/// internal nodes hash (min(left,right), max(left,right)) for
/// order-independence. Zero-allocation alternative to string canonicalization.
fn tree_canonical_for_labels(tree: &Tree, labels: &FixedBitSet) -> u64 {
    fn build(tree: &Tree, node: NodeId, labels: &FixedBitSet) -> Option<u64> {
        if tree.is_leaf(node) {
            let lbl = tree.label[node as usize];
            if labels.contains(lbl as usize) {
                // Hash the label with a mixing constant to avoid trivial collisions
                let mut h = lbl as u64;
                h = h.wrapping_mul(0x9e3779b97f4a7c15);
                h ^= h >> 30;
                return Some(h);
            } else {
                return None;
            }
        }
        if let Some((l, r)) = tree.children(node) {
            let left = build(tree, l, labels);
            let right = build(tree, r, labels);
            match (left, right) {
                (Some(a), Some(b)) => {
                    // Order-independent: sort the two hashes
                    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                    // Combine with a mixing step that distinguishes structure
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
        } else {
            None
        }
    }
    if tree.root == NONE {
        0
    } else {
        build(tree, tree.root, labels).unwrap_or(0)
    }
}

fn trivial_forest(reference: &Tree, num_leaves: u32) -> Vec<Tree> {
    let mut components = Vec::new();
    for lbl in 1..=num_leaves {
        components.push(make_singleton_tree(lbl, num_leaves));
    }
    if components.is_empty() && reference.root != NONE {
        components.push(reference.clone());
    }
    components
}

fn make_singleton_tree(lbl: u32, num_leaves: u32) -> Tree {
    let mut singleton = Tree::with_capacity(num_leaves);
    singleton.parent.push(NONE);
    singleton.left.push(NONE);
    singleton.right.push(NONE);
    singleton.label.push(lbl);
    singleton.label_to_node[lbl as usize] = 0;
    singleton.root = 0;
    singleton.compute_metadata();
    singleton
}

fn has_intersection(a: &FixedBitSet, b: &FixedBitSet) -> bool {
    let a_sl = a.as_slice();
    let b_sl = b.as_slice();
    let len = a_sl.len().min(b_sl.len());
    for i in 0..len {
        if a_sl[i] & b_sl[i] != 0 {
            return true;
        }
    }
    false
}

fn is_subset(a: &FixedBitSet, b: &FixedBitSet) -> bool {
    let a_sl = a.as_slice();
    let b_sl = b.as_slice();
    for i in 0..a_sl.len() {
        let b_word = if i < b_sl.len() { b_sl[i] } else { 0 };
        if a_sl[i] & !b_word != 0 {
            return false;
        }
    }
    true
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use klados_core::Instance;

    fn make_simple_tree() -> Tree {
        // Tree: ((1,2),3)
        let mut tree = Tree::with_capacity(3);
        tree.parent.push(3);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(1);
        tree.label_to_node[1] = 0;
        tree.parent.push(3);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(2);
        tree.label_to_node[2] = 1;
        tree.parent.push(4);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(3);
        tree.label_to_node[3] = 2;
        tree.parent.push(4);
        tree.left.push(0);
        tree.right.push(1);
        tree.label.push(0);
        tree.parent.push(NONE);
        tree.left.push(3);
        tree.right.push(2);
        tree.label.push(0);
        tree.root = 4;
        tree.compute_metadata();
        tree
    }

    #[test]
    fn test_identical_trees_single_component() {
        let t1 = make_simple_tree();
        let t2 = make_simple_tree();
        let instance = Instance::new(vec![t1, t2], 3);
        let mut solver = ShiMestelSolver::new();
        let components = solver.solve(&instance).expect("solution");
        assert_eq!(components.len(), 1);
    }

    #[test]
    fn test_xforest_from_tree() {
        let tree = make_simple_tree();
        let forest = XForest::from_tree(tree);
        assert_eq!(forest.cut_edges.count_ones(..), 0);
        assert_eq!(forest.component_roots.len(), 1);
    }

    #[test]
    fn test_xforest_cut_uncut() {
        let tree = make_simple_tree();
        let mut forest = XForest::from_tree(tree);
        let orig_leafsets: Vec<FixedBitSet> = forest.live_leafsets.clone();
        forest.cut(0);
        assert!(forest.is_cut(0));
        assert_eq!(forest.cut_edges.count_ones(..), 1);
        forest.uncut(0);
        assert!(!forest.is_cut(0));
        assert_eq!(forest.cut_edges.count_ones(..), 0);
        assert_eq!(forest.live_leafsets, orig_leafsets);
    }

    #[test]
    fn test_lsi_identical() {
        let t1 = make_simple_tree();
        let t2 = make_simple_tree();
        let f1 = XForest::from_tree(t1);
        let f2 = XForest::from_tree(t2);
        let forests = [f1, f2];
        let comp_sets: Vec<Vec<FixedBitSet>> = forests
            .iter()
            .map(|f| component_leaf_sets_xf(f, 3))
            .collect();
        assert!(all_pairs_lsi_cached(&comp_sets));
    }

    #[test]
    fn test_component_label_sets() {
        let tree = make_simple_tree();
        let forest = XForest::from_tree(tree);
        let sets = component_leaf_sets_xf(&forest, 3);
        assert_eq!(sets.len(), 1);
    }

    #[test]
    fn test_has_intersection() {
        let mut a = FixedBitSet::with_capacity(4);
        let mut b = FixedBitSet::with_capacity(4);
        a.insert(1);
        a.insert(2);
        b.insert(2);
        b.insert(3);
        assert!(has_intersection(&a, &b));
        b.clear();
        b.insert(3);
        assert!(!has_intersection(&a, &b));
    }

    #[test]
    fn test_is_subset() {
        let mut a = FixedBitSet::with_capacity(4);
        let mut b = FixedBitSet::with_capacity(4);
        a.insert(1);
        a.insert(2);
        b.insert(1);
        b.insert(2);
        b.insert(3);
        assert!(is_subset(&a, &b));
        assert!(!is_subset(&b, &a));
    }

    #[test]
    fn test_sibling_pair_detection() {
        let tree = make_simple_tree();
        let forest = XForest::from_tree(tree);
        let pairs = find_all_sibling_pairs(&forest, 3);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (1, 2));
    }

    #[test]
    fn test_e_f_siblings() {
        let tree = make_simple_tree();
        let forest = XForest::from_tree(tree);
        let e = compute_e_f(&forest, 1, 2);
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn test_search_state_checkpoint_rollback() {
        let t1 = make_simple_tree();
        let t2 = make_simple_tree();
        let mut state = SearchState::new(vec![XForest::from_tree(t1), XForest::from_tree(t2)]);
        let orig0: Vec<FixedBitSet> = state.forests[0].live_leafsets.clone();
        let orig1: Vec<FixedBitSet> = state.forests[1].live_leafsets.clone();

        state.checkpoint();
        state.cut_node(0, 0); // cut node 0 (leaf 1) in forest 0
        assert!(state.forests[0].is_cut(0));
        state.rollback();
        assert!(!state.forests[0].is_cut(0));
        assert_eq!(state.forests[0].live_leafsets, orig0);
        assert_eq!(state.forests[1].live_leafsets, orig1);
    }
}
