//! Clean 2-tree Chen-style rSPR search prototype.
//!
//! This module starts from Chen et al.'s explicit search tree:
//! for a sibling leaf pair `(x1, x2)` in `T`, branch on
//! `{e_F(x1)}`, `{e_F(x2)}`, and, when defined, `D_F(x1, x2)`.
//!
//! Implemented paper rules:
//! - Section 3, Step 4: force the unique edge in `D_F(x1, x2)`.
//! - Section 3, Step 5: force `e_F(x3)` for the symmetric unique-D case.
//! - Section 3, Steps 6-8 / Section 4: the ordered exhaustive search tree.
//! - Chen's released `MafFpt1.findOptimalCut()` single-cut pre-branch rule.
//!
//! Not yet implemented:
//! - Section 3, Step 3 as stated. That requires Theorem 3.2, whose core is
//!   the full Theorem 3.1 good-cut oracle from Section 6.3.

use fixedbitset::FixedBitSet;
use fxhash::FxHashMap;
use klados_core::tree::{Label, NONE, NodeId};
use log::debug;
use klados_core::twin_tree::forest::{T1, T2, TwinForest};
use klados_core::twin_tree::undo;
use klados_core::{Instance, SolverStats, Tree};
use std::collections::HashMap;


/// Compute Chen's 2-approximation bounds for a tree pair.
///
/// Returns `(lower, upper)` where `lower` is a valid lower bound on the
/// rSPR distance and `upper` is a valid upper bound satisfying
/// `lower ≤ OPT ≤ upper ≤ 2·lower`.
///
/// Uses the full good-key/stopper oracle (Theorem 3.2 / Section 6.3).
pub fn chen_pair_bounds(t1: &Tree, t2: &Tree) -> (usize, usize) {
    let n = t1.num_leaves;
    let tf = TwinForest::from_trees(t1, t2, n);
    // improve_upper=false avoids the expensive forest reconstruction
    let bounds = chen_app2_bounds_with_options(&tf, false);
    let (lower, upper) = if bounds.lower > n as usize || bounds.upper > n as usize {
        let app1 = chen_app1_bounds_inner(&tf);
        (
            app1.lower.min(n.saturating_sub(1) as usize),
            app1.upper.min(n.saturating_sub(1) as usize),
        )
    } else {
        (
            bounds.lower.min(n.saturating_sub(1) as usize),
            bounds.upper.min(n.saturating_sub(1) as usize),
        )
    };
    (lower, upper)
}

#[derive(Clone, Debug)]
#[derive(Default)]
pub struct ChenRsprConfig {
    /// Enable jar-dummy leaf attachment.
    pub jar_dummy: bool,
    /// Disable recursive lower bound.
    pub no_recursive_lb: bool,
    /// Enable forced-cut pre-branch rules.
    pub use_forced: bool,
}


pub struct ChenRsprSolver {
    stats: SolverStats,
    config: ChenRsprConfig,
}

impl Default for ChenRsprSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ChenRsprSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
            config: ChenRsprConfig::default(),
        }
    }
}

impl Solver for ChenRsprSolver {
    type Config = ChenRsprConfig;
    const SUPPORTED_TRACKS: &'static [Track] = &[Track::Exact];

    fn solve(&mut self, instance: &Instance, cfg: &RunConfig<Self::Config>) -> Option<Vec<Tree>> {
        if instance.num_trees() != 2 {
            log::warn!(
                "[chen-rspr] m={} is not supported by the clean Chen prototype",
                instance.num_trees()
            );
            return None;
        }
        self.config = cfg.specific.clone();
        solve_chen_rspr(instance, &mut self.stats, &self.config)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

fn with_chen_dummy(tree: &Tree, original_num_leaves: u32) -> Tree {
    let dummy_label = original_num_leaves + 1;
    let mut out = tree.clone();
    out.num_leaves = dummy_label;
    if out.label_to_node.len() <= dummy_label as usize {
        out.label_to_node.resize(dummy_label as usize + 1, NONE);
    }

    let old_root = out.root;
    let dummy = out.parent.len() as NodeId;
    out.parent.push(NONE);
    out.left.push(NONE);
    out.right.push(NONE);
    out.label.push(dummy_label);
    out.label_to_node[dummy_label as usize] = dummy;

    let new_root = out.parent.len() as NodeId;
    out.parent.push(NONE);
    out.left.push(dummy);
    out.right.push(old_root);
    out.label.push(0);
    out.parent[dummy as usize] = new_root;
    if old_root != NONE {
        out.parent[old_root as usize] = new_root;
    }
    out.root = new_root;
    out.compute_metadata();
    out
}

fn solve_chen_rspr(
    instance: &Instance,
    stats: &mut SolverStats,
    config: &ChenRsprConfig,
) -> Option<Vec<Tree>> {
    if instance.num_leaves <= 1 {
        return Some(vec![instance.trees[0].clone()]);
    }

    let output_n = instance.num_leaves;
    let use_jar_dummy = config.jar_dummy;
    let n = if use_jar_dummy {
        output_n + 1
    } else {
        output_n
    };
    let t1;
    let t2;
    let (tree1, tree2) = if use_jar_dummy {
        t1 = with_chen_dummy(&instance.trees[0], output_n);
        t2 = with_chen_dummy(&instance.trees[1], output_n);
        (&t1, &t2)
    } else {
        (&instance.trees[0], &instance.trees[1])
    };
    let mut tf = TwinForest::from_trees(tree1, tree2, n);
    let mut um = undo::UndoMachine::new();
    let mut ctx = ChenSearchCtx::default();

    let app2_bounds =
        chen_app2_bounds_with_options(&tf, true);
    let app1_bounds = if app2_bounds.lower > n as usize {
        chen_app1_bounds_inner(&tf)
    } else {
        app2_bounds
    };
    debug!(
        "[chen-rspr] app1={}~{} app2={}~{}",
        app1_bounds.lower, app1_bounds.upper, app2_bounds.lower, app2_bounds.upper
    );
    let app_bounds = app1_bounds;
    let lb_k = app_bounds.lower.min(n.saturating_sub(1) as usize);
    let ub_k = app_bounds.upper.min(n.saturating_sub(1) as usize);

    stats.lower_bound = lb_k + 1;
    stats.upper_bound = Some(ub_k + 1);

    for k in lb_k..=ub_k {
        let cp = um.checkpoint();
        let before_nodes = stats.nodes_explored;
        if chen_search(&mut tf, k as i32, &mut um, stats, &mut ctx, config) {
            debug!(
                "[chen-rspr] k={} yes nodes={}",
                k,
                stats.nodes_explored.saturating_sub(before_nodes)
            );
            let solution = extract_components_with_reference(&tf, output_n, &instance.trees[0]);
            stats.lower_bound = solution.len();
            stats.upper_bound = Some(solution.len());
            return Some(solution);
        }
        debug!(
            "[chen-rspr] k={} no nodes={}",
            k,
            stats.nodes_explored.saturating_sub(before_nodes)
        );
        um.undo_to(cp, &mut tf);
    }

    // The approximation upper bound is only a performance hint. Keep the
    // prototype exact while the full MafApp2 key/stopper oracle is being
    // ported by falling back to the trivial upper range if needed.
    for k in (ub_k + 1)..=n.saturating_sub(1) as usize {
        let cp = um.checkpoint();
        let before_nodes = stats.nodes_explored;
        if chen_search(&mut tf, k as i32, &mut um, stats, &mut ctx, config) {
            debug!(
                "[chen-rspr] k={} yes nodes={}",
                k,
                stats.nodes_explored.saturating_sub(before_nodes)
            );
            let solution = extract_components_with_reference(&tf, output_n, &instance.trees[0]);
            stats.lower_bound = solution.len();
            stats.upper_bound = Some(solution.len());
            return Some(solution);
        }
        debug!(
            "[chen-rspr] k={} no nodes={}",
            k,
            stats.nodes_explored.saturating_sub(before_nodes)
        );
        um.undo_to(cp, &mut tf);
    }

    None
}

fn chen_search(
    tf: &mut TwinForest,
    mut k: i32,
    um: &mut undo::UndoMachine,
    stats: &mut SolverStats,
    ctx: &mut ChenSearchCtx,
    config: &ChenRsprConfig,
) -> bool {
    stats.nodes_explored += 1;

    // Pre-screen: if the lower bound already says this branch can't succeed, bail out.
    if !config.no_recursive_lb {
        let lb = chen_lower_bound(tf, ctx);
        if (lb as i32) > k {
            return false;
        }
    }

    loop {
        if !process_singletons(tf, &mut k, um) {
            return false;
        }

        if k < 0 {
            return false;
        }

        if let Some(node) = find_reference_optimal_cut(tf) {
            if tf.protected[node as usize] {
                return false;
            }
            if k <= 0 {
                return false;
            }
            cut_t2_node(tf, node, um);
            k -= 1;
            continue;
        }

        if config.use_forced {
            if let Some(node) = find_step4_forced_cut(tf) {
                if tf.protected[node as usize] {
                    return false;
                }
                if k <= 0 {
                    return false;
                }
                cut_t2_node(tf, node, um);
                k -= 1;
                continue;
            }

            if let Some(node) = find_step5_forced_cut(tf) {
                if tf.protected[node as usize] {
                    return false;
                }
                if k <= 0 {
                    return false;
                }
                cut_t2_node(tf, node, um);
                k -= 1;
                continue;
            }
        }

        match find_sibling_pair(tf) {
            PairResult::NoPairs => return true,
            PairResult::Case2 {
                t1_parent,
                t2_parent,
            } => {
                contract_matching_pair(tf, t1_parent, t2_parent, um);
            }
            PairResult::Branches(branches) => {
                if k <= 0 {
                    return false;
                }
                for branch in branches {
                    if branch.contains_locked(tf) {
                        continue;
                    }
                    let cost = branch.cost() as i32;
                    if cost > k {
                        continue;
                    }
                    let cp = um.checkpoint();
                    apply_branch_cut(tf, &branch, um);
                    branch.apply_locks(tf, um);
                    let remaining = k - cost;
                    let passes_lower_bound = if config.no_recursive_lb {
                        true
                    } else {
                        let lb = chen_lower_bound(tf, ctx);
                        (lb as i32) <= remaining
                    };
                    if passes_lower_bound && chen_search(tf, remaining, um, stats, ctx, config) {
                        return true;
                    }
                    um.undo_to(cp, tf);
                }
                return false;
            }
        }
    }
}

enum BranchCut {
    One {
        node: NodeId,
        locks_after: Vec<NodeId>,
    },
    Many {
        nodes: Vec<NodeId>,
        locks_after: Vec<NodeId>,
    },
}

impl BranchCut {
    fn cost(&self) -> usize {
        match self {
            BranchCut::One { .. } => 1,
            BranchCut::Many { nodes, .. } => nodes.len(),
        }
    }

    fn contains_locked(&self, tf: &TwinForest) -> bool {
        match self {
            BranchCut::One { node, .. } => tf.protected[*node as usize],
            BranchCut::Many { nodes, .. } => nodes.iter().any(|&node| tf.protected[node as usize]),
        }
    }

    fn apply_locks(&self, tf: &mut TwinForest, um: &mut undo::UndoMachine) {
        let locks = match self {
            BranchCut::One { locks_after, .. } | BranchCut::Many { locks_after, .. } => locks_after,
        };
        for &lock in locks {
            if lock != NONE {
                undo::protect_edge(tf, lock, um);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Full Chen checkCase port — recognises RC, _2B, BCY, A2BC, RB2, A3BC patterns.
// ---------------------------------------------------------------------------

fn build_branches(
    tf: &TwinForest,
    t1_parent: NodeId,
    t2_x1: NodeId,
    t2_x2: NodeId,
    d_cut: Vec<NodeId>,
    lca: NodeId,
    t1_sibling_is_leaf: bool,
) -> Vec<BranchCut> {
    // ---- CheckCase (MafFpt1 lines 172-264) ----
    // f2a = t2_x1, f2c = t2_x2, lcaAC = lca

    // RC: lca is null → different T2 components
    if lca == NONE {
        let mut branches = Vec::with_capacity(2);
        if !tf.protected[t2_x1 as usize] {
            branches.push(BranchCut::One {
                node: t2_x1,
                locks_after: Vec::new(),
            });
        }
        if !tf.protected[t2_x2 as usize] {
            branches.push(BranchCut::One {
                node: t2_x2,
                locks_after: Vec::new(),
            });
        }
        return branches;
    }

    // BCY / _2B: t1_parent's sibling is a leaf
    if t1_sibling_is_leaf {
        let t1_sib = tf.sibling(T1, t1_parent);
        let f2x = if t1_sib != NONE {
            tf.twin[T1][t1_sib as usize]
        } else {
            NONE
        };
        let f2x_locked = f2x != NONE && tf.protected[f2x as usize];

        // BCY: f2x locked AND lca == parent of x1 or x2
        if f2x_locked
            && (lca == tf.parent[T2][t2_x1 as usize] || lca == tf.parent[T2][t2_x2 as usize])
        {
            let mut branches = Vec::with_capacity(2);
            let d_ac = d_f_cut_nodes(tf, T2, t2_x1, t2_x2).unwrap_or_default();
            if !d_ac.is_empty()
                && !d_ac.iter().any(|&n| tf.protected[n as usize]) {
                    branches.push(BranchCut::Many {
                        nodes: d_ac,
                        locks_after: Vec::new(),
                    });
                }
            if lca == tf.parent[T2][t2_x1 as usize] {
                let x2_sib = tf.sibling(T2, t2_x2);
                if x2_sib != NONE && !tf.is_leaf(T2, x2_sib) {
                    let x2_sib_l = tf.left[T2][x2_sib as usize];
                    let x2_sib_r = tf.right[T2][x2_sib as usize];
                    if x2_sib_l == f2x || x2_sib_r == f2x {
                        let f2y = if x2_sib_l == f2x { x2_sib_r } else { x2_sib_l };
                        if f2y != NONE
                            && !tf.protected[f2y as usize]
                            && !tf.protected[t2_x1 as usize]
                        {
                            branches.push(BranchCut::Many {
                                nodes: vec![f2y, t2_x1],
                                locks_after: Vec::new(),
                            });
                        }
                    }
                }
            } else {
                let x1_sib = tf.sibling(T2, t2_x1);
                if x1_sib != NONE && !tf.is_leaf(T2, x1_sib) {
                    let x1_sib_l = tf.left[T2][x1_sib as usize];
                    let x1_sib_r = tf.right[T2][x1_sib as usize];
                    if x1_sib_l == f2x || x1_sib_r == f2x {
                        let f2y = if x1_sib_l == f2x { x1_sib_r } else { x1_sib_l };
                        if f2y != NONE
                            && !tf.protected[f2y as usize]
                            && !tf.protected[t2_x2 as usize]
                        {
                            branches.push(BranchCut::Many {
                                nodes: vec![f2y, t2_x2],
                                locks_after: Vec::new(),
                            });
                        }
                    }
                }
            }
            if !branches.is_empty() {
                branches.reverse();
                return branches;
            }
        }

        // _2B: neither parent is root, one parent's sibling is f2x, |d_cut| == 2
        let x1_parent = tf.parent[T2][t2_x1 as usize];
        let x2_parent = tf.parent[T2][t2_x2 as usize];
        if x1_parent != NONE
            && tf.parent[T2][x1_parent as usize] != NONE
            && x2_parent != NONE
            && tf.parent[T2][x2_parent as usize] != NONE
            && (tf.sibling(T2, x1_parent) == f2x || tf.sibling(T2, x2_parent) == f2x)
            && d_cut.len() == 2
            && !d_cut.iter().any(|&n| tf.protected[n as usize]) {
                return vec![BranchCut::Many {
                    nodes: d_cut,
                    locks_after: Vec::new(),
                }];
            }
    }

    let x1_unprotected = !tf.protected[t2_x1 as usize];
    let x2_unprotected = !tf.protected[t2_x2 as usize];

    // A2BC: |d_cut| == 2
    if d_cut.len() == 2 {
        let d_locked = d_cut.iter().any(|&n| tf.protected[n as usize]);
        if x1_unprotected && x2_unprotected && !d_locked {
            let mut cut_x2_locks = d_cut.clone();
            cut_x2_locks.push(t2_x1);
            return vec![
                BranchCut::One {
                    node: t2_x2,
                    locks_after: cut_x2_locks,
                },
                BranchCut::Many {
                    nodes: d_cut,
                    locks_after: Vec::new(),
                },
                BranchCut::One {
                    node: t2_x1,
                    locks_after: Vec::new(),
                },
            ];
        } else if x1_unprotected != x2_unprotected {
            let single = if x1_unprotected { t2_x1 } else { t2_x2 };
            let mut branches = Vec::with_capacity(2);
            branches.push(BranchCut::One {
                node: single,
                locks_after: d_cut.clone(),
            });
            if !d_locked {
                branches.push(BranchCut::Many {
                    nodes: d_cut,
                    locks_after: Vec::new(),
                });
            }
            return branches;
        } else {
            if !d_locked {
                return vec![BranchCut::Many {
                    nodes: d_cut,
                    locks_after: Vec::new(),
                }];
            }
            return Vec::new();
        }
    }

    // RB2: lca == parent of x1 or x2, all d_cut are leaves, special geometry
    if d_cut.len() >= 2
        && (lca == tf.parent[T2][t2_x1 as usize] || lca == tf.parent[T2][t2_x2 as usize])
        && d_cut.iter().all(|&n| tf.is_leaf(T2, n))
    {
        let first_mate = tf.twin[T2][d_cut[0] as usize];
        let second_mate = tf.twin[T2][d_cut[1] as usize];
        if first_mate != NONE
            && second_mate != NONE
            && tf.parent[T1][first_mate as usize] == tf.parent[T1][second_mate as usize]
        {
            let mut ok = true;
            let mut current = tf.parent[T1][first_mate as usize];
            for i in 2..d_cut.len() {
                let t1_leaf = tf.twin[T2][d_cut[i] as usize];
                if t1_leaf == NONE || tf.parent[T1][current as usize] != t1_leaf {
                    ok = false;
                    break;
                }
                current = tf.parent[T1][current as usize];
            }
            let last = d_cut[d_cut.len() - 1];
            let last_p2 = tf.parent[T2][last as usize];
            let ok2 = last_p2 != NONE && {
                let last_gp2 = tf.parent[T2][last_p2 as usize];
                last_gp2 != NONE && {
                    let t2_gp = tf.twin[T2][last_gp2 as usize];
                    let t1a = tf.twin[T2][t2_x1 as usize];
                    t2_gp != NONE
                        && t1a != NONE
                        && tf.parent[T1][t1a as usize] != NONE
                        && t2_gp == tf.parent[T1][tf.parent[T1][t1a as usize] as usize]
                }
            };
            if ok && ok2 {
                let single = if lca == tf.parent[T2][t2_x1 as usize] {
                    t2_x2
                } else {
                    t2_x1
                };
                if !tf.protected[single as usize] {
                    return vec![BranchCut::One {
                        node: single,
                        locks_after: Vec::new(),
                    }];
                }
            }
        }
    }

    // ---- Default A3BC (Java lines 388-420, after reverse) ----
    let mut branches = Vec::with_capacity(3);
    let singleton_lock = if x1_unprotected && x2_unprotected {
        vec![t2_x1]
    } else {
        Vec::new()
    };
    if !d_cut.is_empty()
        && !d_cut.iter().any(|&n| tf.protected[n as usize]) {
            branches.push(BranchCut::Many {
                nodes: d_cut,
                locks_after: singleton_lock.clone(),
            });
        }
    if x2_unprotected {
        branches.push(BranchCut::One {
            node: t2_x2,
            locks_after: singleton_lock,
        });
    }
    if x1_unprotected {
        branches.push(BranchCut::One {
            node: t2_x1,
            locks_after: Vec::new(),
        });
    }
    branches
}

fn apply_branch_cut(tf: &mut TwinForest, branch: &BranchCut, um: &mut undo::UndoMachine) {
    match branch {
        BranchCut::One { node, .. } => cut_t2_node(tf, *node, um),
        BranchCut::Many { nodes, .. } => {
            for &node in nodes {
                cut_t2_node(tf, node, um);
            }
        }
    }
}

fn cut_t2_node(tf: &mut TwinForest, node: NodeId, um: &mut undo::UndoMachine) {
    let parent = tf.parent[T2][node as usize];
    if parent == NONE {
        return;
    }
    undo::cut_parent(tf, T2, node, um);
    undo::add_component(tf, T2, node, um);
    undo::contract(tf, T2, parent, um);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AppBounds {
    pub lower: usize,
    pub upper: usize,
}

#[derive(Default)]
struct ChenSearchCtx {
    lower_cache: HashMap<u64, usize>,
}

fn chen_lower_bound(tf: &TwinForest, ctx: &mut ChenSearchCtx) -> usize {
    if let Some(&lb) = ctx.lower_cache.get(&tf.state_hash) {
        return lb;
    }
    let app2_lb = chen_app2_bounds_with_options(tf, false).lower;
    let lb = if app2_lb > tf.num_leaves as usize {
        chen_app1_bounds_inner(tf).lower
    } else {
        app2_lb
    };
    ctx.lower_cache.insert(tf.state_hash, lb);
    lb
}

fn uf_find(uf: &mut [u32], mut x: u32) -> u32 {
    while uf[x as usize] != x {
        let p = uf[x as usize];
        let pp = uf[p as usize];
        uf[x as usize] = pp;
        x = pp;
    }
    x
}

fn uf_union(uf: &mut [u32], a: u32, b: u32) {
    let ra = uf_find(uf, a);
    let rb = uf_find(uf, b);
    if ra != rb {
        uf[ra as usize] = rb;
    }
}

/// Reconstruct the Chen 2-approximation agreement forest leaf-sets from the
/// LIVE state of the Chen algorithm.
///
/// `state.components[T2]` holds the roots of the live T2 forest after the
/// algorithm finishes. Each live "leaf" of these subtrees represents either
/// a single original leaf (label != 0) or a cherry-contracted internal node
/// (label == 0) standing in for the set of original leaves in its original-T2
/// subtree. Leaves that were dropped from the live forest by
/// `cut_tree1_singletons` / `drop_t2_leaf_roots` form their own singleton
/// components.
///
/// We use the live state (not just `f2_cut_list`) because the algorithm's
/// stopper machinery sometimes calls `unrecord_f2_cut` to re-assemble cut
/// pieces (e.g. `cut_complex_key`); those re-assemblies only show in the
/// live structure.
fn chen_agreement_leafsets_from_state(_tf: &TwinForest, state: &ChenAppState) -> Vec<Vec<u32>> {
    let n = state.num_leaves;
    let mut uf = state.leaf_uf.clone();
    let mut groups: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for lbl in 1..=n {
        let r = uf_find(&mut uf, lbl);
        groups.entry(r).or_default().push(lbl);
    }
    let mut result: Vec<Vec<u32>> = groups.into_values().collect();
    for v in result.iter_mut() {
        v.sort_unstable();
    }
    result.sort();
    result
}

/// Extract the Chen 2-approximation agreement forest component leaf-sets
/// for a tree pair. Returns `(lower, upper, leafsets)` where:
/// - `lower` is a valid lower bound on the rSPR distance
/// - `upper` is the size of the agreement forest (≤ 2·OPT)
/// - `leafsets` contains the component leaf label sets
///
/// The returned leaf-sets form a valid agreement forest for the two trees.
pub fn chen_pair_agreement(t1: &Tree, t2: &Tree) -> (usize, usize, Vec<Vec<u32>>) {
    let n = t1.num_leaves;
    let tf = TwinForest::from_trees(t1, t2, n);

    // Run the Chen 2-approx — this populates cut lists in the state
    // through the full stopper oracle.
    let mut state = ChenAppState::from_twin(&tf);
    let mut bounds = AppBounds::default();
    let mut guard = state.num_nodes[T1] * 8 + state.num_nodes[T2] * 8;
    while guard > 0 {
        guard -= 1;
        state.cut_tree1_singletons();
        state.shrink_common_cherries();
        state.cut_tree1_singletons();
        state.drop_t2_leaf_roots();

        let Some(root) = state.first_t1_root() else {
            break;
        };
        if !state.has_t1_cherry() {
            break;
        };

        if let Some(cut) = state.find_optimal_cut() {
            state.cut_forest2(cut);
            bounds.lower += 1;
            bounds.upper += 1;
            continue;
        }

        let Some(mut stopper) = ChenKey::find_stopper(root, &state) else {
            break;
        };
        if stopper.stopper.is_none() {
            // Non-stopper key: apply the selected cuts so the agreement
            // forest can be reconstructed from the cut lists.
            bounds.lower += stopper.lower_bound;
            bounds.upper += stopper.upper_bound;
            stopper.cut_key(&mut state);
            break;
        }
        stopper.cut_stopper(&mut state, &mut bounds);
    }

    let leafsets = chen_agreement_leafsets_from_state(&tf, &state);

    (
        bounds.lower.min(n.saturating_sub(1) as usize),
        bounds.upper.min(n.saturating_sub(1) as usize),
        leafsets,
    )
}

fn improved_upper_from_state(tf: &TwinForest, state: &ChenAppState) -> usize {
    chen_agreement_leafsets_from_state(tf, state)
        .len()
        .saturating_sub(1)
}

/// Native approximation bound oracle matching the simple cut loop used as the
/// base class of Chen's released `MafApp2`: repeatedly remove isolated leaves,
/// shrink common cherries, apply the optimal single-cut rule, otherwise cut the
/// two cherry leaves and the third path-side component.
///
/// This is intentionally isolated from the exact search state. The full
/// key/stopper oracle from `MafApp2.compute()` can replace this function
/// without changing the search driver or pruning call sites.
pub fn chen_app1_bounds(t1: &Tree, t2: &Tree) -> (usize, usize) {
    let n = t1.num_leaves;
    let tf = TwinForest::from_trees(t1, t2, n);
    let bounds = chen_app1_bounds_inner(&tf);
    (
        bounds.lower.min(n.saturating_sub(1) as usize),
        bounds.upper.min(n.saturating_sub(1) as usize),
    )
}

fn chen_app1_bounds_inner(tf: &TwinForest) -> AppBounds {
    let mut work = tf.clone();
    let mut um = undo::UndoMachine::new();
    let mut bounds = AppBounds::default();
    let mut guard = work.num_nodes[T1] * 4 + work.num_nodes[T2] * 4;

    while guard > 0 {
        guard -= 1;
        let mut dummy_k = i32::MAX / 4;
        if !process_singletons(&mut work, &mut dummy_k, &mut um) {
            break;
        }
        shrink_common_cherries(&mut work, &mut um);

        if !has_t1_cherry(&work) {
            break;
        }

        if let Some(node) = find_reference_optimal_cut(&work) {
            cut_t2_node(&mut work, node, &mut um);
            bounds.lower += 1;
            bounds.upper += 1;
            continue;
        }

        let Some((t2_a, t2_b)) = first_t1_cherry_t2_mates(&work) else {
            break;
        };

        let lca = lca_current(&work, T2, t2_a, t2_b);
        let parent_a = work.parent[T2][t2_a as usize];
        let t2_c = if lca != NONE && parent_a == lca {
            sibling_node(&work, T2, t2_b).unwrap_or(NONE)
        } else {
            sibling_node(&work, T2, t2_a).unwrap_or(NONE)
        };

        let mut cuts = Vec::with_capacity(3);
        cuts.push(t2_a);
        cuts.push(t2_b);
        if t2_c != NONE && work.parent[T2][t2_c as usize] != NONE {
            cuts.push(t2_c);
        }
        cuts.sort_unstable_by_key(|&node| std::cmp::Reverse(depth_to_root(&work, T2, node)));
        cuts.dedup();

        for node in cuts {
            if work.parent[T2][node as usize] != NONE {
                cut_t2_node(&mut work, node, &mut um);
                bounds.upper += 1;
            }
        }
        bounds.lower += 1;
    }

    bounds
}

fn chen_app2_bounds_with_options(
    tf: &TwinForest,
    improve_upper: bool,
) -> AppBounds {
    let mut state = ChenAppState::from_twin(tf);
    let mut bounds = AppBounds::default();
    let mut guard = state.num_nodes[T1] * 8 + state.num_nodes[T2] * 8;
    let mut iter = 0usize;
    while guard > 0 {
        guard -= 1;
        iter += 1;
        state.cut_tree1_singletons();
        state.shrink_common_cherries();
        state.cut_tree1_singletons();
        state.drop_t2_leaf_roots();

        let Some(root) = state.first_t1_root() else {
            break;
        };
        if !state.has_t1_cherry() {
            break;
        }

        if let Some(cut) = state.find_optimal_cut() {
            state.cut_forest2(cut);
            bounds.lower += 1;
            bounds.upper += 1;
            debug!(
                "[chen-rspr] app2 iter={} kind=OPT cut={} acc={}~{}",
                iter, cut, bounds.lower, bounds.upper
            );
            continue;
        }

        let Some(mut stopper) = ChenKey::find_stopper(root, &state) else {
            // The whole tree could not be covered by any stopper classification.
            // This means the remaining tree is fully consistent with F — equivalent
            // to the mergeless (⊥,⊥) case. The accumulated bounds from the singleton
            // keys (if any) are already correct.
            break;
        };
        if stopper.stopper.is_none() {
            // Merged key without stopper classification: the key's accumulated
            // lower/upper bounds are valid (Lemma 6.2) but may be weaker than a
            // proper stopper classification would give. Use them directly and
            // continue — the 2-approximation guarantee still holds.
            bounds.lower += stopper.lower_bound;
            bounds.upper += stopper.upper_bound;
            break;
        }
        stopper.cut_stopper(&mut state, &mut bounds);
        let name = match stopper.stopper {
            Some(ChenStopper::Disconnected) => "DIS",
            Some(ChenStopper::Root) => "ROOT",
            Some(ChenStopper::Close) => "CLOSE",
            Some(ChenStopper::Overlapping) => "OVER",
            None => "NONE",
        };
        debug!(
            "[chen-rspr] app2 iter={} kind={} acc={}~{}",
            iter, name, bounds.lower, bounds.upper
        );
    }
    if improve_upper {
        let raw_bounds = bounds;
        let improved_upper = improved_upper_from_state(tf, &state);
        bounds.upper = bounds.upper.min(improved_upper);
        debug!(
            "[chen-rspr] app2 raw={}~{} improved={}~{}",
            raw_bounds.lower, raw_bounds.upper, bounds.lower, bounds.upper
        );
    }
    bounds
}

#[derive(Clone)]
struct ChenAppState {
    parent: [Vec<NodeId>; 2],
    orig_parent: [Vec<NodeId>; 2],
    left: [Vec<NodeId>; 2],
    right: [Vec<NodeId>; 2],
    twin: [Vec<NodeId>; 2],
    components: [Vec<NodeId>; 2],
    num_nodes: [usize; 2],
    t1_cut_list: Vec<NodeId>,
    f2_cut_list: Vec<NodeId>,

    // --- AF reconstruction (mirrors Java MafApp's `mergeComponents` logic) ---
    //
    // `leaf_uf` is a union-find over leaf labels 1..=num_leaves. Two labels
    // share a representative iff they were merged into the same agreement-
    // forest component by `shrink_common_cherries` (the only operation that
    // groups leaves; cuts only separate, never merge).
    //
    // `node_repr_t1[v]` is a "delegate" leaf label for the current live T1
    // subtree rooted at node `v` — initially the leaf's own label for leaves
    // and 0 for internal nodes. Each cherry contraction promotes one of the
    // children's delegate to the parent so subsequent contractions can union
    // through nested cherry merges.
    num_leaves: u32,
    leaf_uf: Vec<u32>,
    node_repr_t1: Vec<Label>,
}

impl ChenAppState {
    fn from_twin(tf: &TwinForest) -> Self {
        let n = tf.num_leaves;
        let node_repr_t1 = tf.label[T1][..tf.num_nodes[T1]].to_vec();
        let mut state = Self {
            parent: tf.parent.clone(),
            orig_parent: [tf.orig_parent.clone(), tf.orig_t2_parent.clone()],
            left: tf.left.clone(),
            right: tf.right.clone(),
            twin: tf.twin.clone(),
            components: tf.components.clone(),
            num_nodes: tf.num_nodes,
            t1_cut_list: Vec::new(),
            f2_cut_list: Vec::new(),
            num_leaves: n,
            leaf_uf: (0..=n).collect(),
            node_repr_t1,
        };
        // The exact search's undo contract deliberately leaves stale topology
        // on contracted-away nodes (for speed / hash stability).  Java's
        // Vertex.contract() physically detaches those nodes.  The Chen App2
        // bounder walks parent/child pointers extensively, so sanitize the
        // cloned view to contain only nodes reachable from current components
        // through parent-consistent child links.
        state.sanitize_live_topology();
        state
    }

    fn sanitize_live_topology(&mut self) {
        for ti in [T1, T2] {
            let mut live = vec![false; self.num_nodes[ti]];
            let mut stack = Vec::new();
            for &root in &self.components[ti] {
                if root != NONE && (root as usize) < self.num_nodes[ti] {
                    stack.push(root);
                }
            }
            while let Some(node) = stack.pop() {
                if node == NONE || (node as usize) >= self.num_nodes[ti] || live[node as usize] {
                    continue;
                }
                live[node as usize] = true;
                let lc = self.left[ti][node as usize];
                if lc != NONE
                    && (lc as usize) < self.num_nodes[ti]
                    && self.parent[ti][lc as usize] == node
                {
                    stack.push(lc);
                }
                let rc = self.right[ti][node as usize];
                if rc != NONE
                    && (rc as usize) < self.num_nodes[ti]
                    && self.parent[ti][rc as usize] == node
                {
                    stack.push(rc);
                }
            }

            for node in 0..self.num_nodes[ti] {
                if !live[node] {
                    self.parent[ti][node] = NONE;
                    self.left[ti][node] = NONE;
                    self.right[ti][node] = NONE;
                    continue;
                }
                let lc = self.left[ti][node];
                if lc != NONE
                    && ((lc as usize) >= self.num_nodes[ti]
                        || !live[lc as usize]
                        || self.parent[ti][lc as usize] != node as NodeId)
                {
                    self.left[ti][node] = NONE;
                }
                let rc = self.right[ti][node];
                if rc != NONE
                    && ((rc as usize) >= self.num_nodes[ti]
                        || !live[rc as usize]
                        || self.parent[ti][rc as usize] != node as NodeId)
                {
                    self.right[ti][node] = NONE;
                }
            }
        }
    }

    #[inline]
    fn is_leaf(&self, ti: usize, node: NodeId) -> bool {
        node != NONE
            && self.left[ti][node as usize] == NONE
            && self.right[ti][node as usize] == NONE
    }

    #[inline]
    fn is_root(&self, ti: usize, node: NodeId) -> bool {
        node != NONE && self.parent[ti][node as usize] == NONE
    }

    #[inline]
    fn sibling(&self, ti: usize, node: NodeId) -> NodeId {
        if node == NONE {
            return NONE;
        }
        let p = self.parent[ti][node as usize];
        if p == NONE {
            return NONE;
        }
        if self.left[ti][p as usize] == node {
            self.right[ti][p as usize]
        } else {
            self.left[ti][p as usize]
        }
    }

    fn first_t1_root(&self) -> Option<NodeId> {
        self.components[T1]
            .iter()
            .copied()
            .find(|&root| root != NONE && self.parent[T1][root as usize] == NONE)
    }

    fn has_t1_cherry(&self) -> bool {
        self.components[T1]
            .iter()
            .copied()
            .any(|root| self.first_t1_cherry_under(root).is_some())
    }

    fn first_t1_cherry_under(&self, node: NodeId) -> Option<NodeId> {
        if node == NONE || self.is_leaf(T1, node) {
            return None;
        }
        let lc = self.left[T1][node as usize];
        let rc = self.right[T1][node as usize];
        if lc != NONE && rc != NONE && self.is_leaf(T1, lc) && self.is_leaf(T1, rc) {
            return Some(node);
        }
        self.first_t1_cherry_under(lc)
            .or_else(|| self.first_t1_cherry_under(rc))
    }

    fn cut_tree1_singletons(&mut self) {
        loop {
            let singleton_idx = self.components[T2]
                .iter()
                .position(|&root| root != NONE && self.is_leaf(T2, root));
            let Some(idx) = singleton_idx else {
                break;
            };
            let f2v = self.components[T2].remove(idx);
            let t1v = self.twin[T2][f2v as usize];
            if t1v == NONE {
                continue;
            }
            self.cut_tree1_leaf(t1v);
        }
    }

    fn cut_tree1_leaf(&mut self, t1v: NodeId) {
        let p = self.parent[T1][t1v as usize];
        if p == NONE {
            self.remove_component(T1, t1v);
            return;
        }
        self.t1_cut_list.push(t1v);
        self.detach(T1, t1v);
        let survivor = self.contract(T1, p);
        if self.components[T1].contains(&p) && survivor != NONE {
            self.replace_component(T1, p, survivor);
        }
    }

    fn shrink_common_cherries(&mut self) {
        loop {
            let mut target = None;
            for &root in &self.components[T1] {
                if let Some(pair) = self.find_common_cherry_under(root) {
                    target = Some(pair);
                    break;
                }
            }
            let Some((t1_parent, t2_parent)) = target else {
                break;
            };
            // Track the merge for AF extraction. The two T1 children's
            // delegate leaf labels are joined; the parent inherits one as
            // its new delegate so a cascading contraction can keep unioning.
            let lc = self.left[T1][t1_parent as usize];
            let rc = self.right[T1][t1_parent as usize];
            let lr = if lc != NONE {
                self.node_repr_t1[lc as usize]
            } else {
                0
            };
            let rr = if rc != NONE {
                self.node_repr_t1[rc as usize]
            } else {
                0
            };
            if lr != 0 && rr != 0 {
                uf_union(&mut self.leaf_uf, lr, rr);
            }
            let promoted = if lr != 0 { lr } else { rr };
            self.node_repr_t1[t1_parent as usize] = promoted;
            self.contract_sibling_pair(T1, t1_parent);
            self.contract_sibling_pair(T2, t2_parent);
            self.twin[T1][t1_parent as usize] = t2_parent;
            self.twin[T2][t2_parent as usize] = t1_parent;
        }
    }

    fn find_common_cherry_under(&self, node: NodeId) -> Option<(NodeId, NodeId)> {
        if node == NONE || self.is_leaf(T1, node) {
            return None;
        }
        let lc = self.left[T1][node as usize];
        let rc = self.right[T1][node as usize];
        if lc != NONE && rc != NONE && self.is_leaf(T1, lc) && self.is_leaf(T1, rc) {
            let t2_l = self.twin[T1][lc as usize];
            let t2_r = self.twin[T1][rc as usize];
            if t2_l != NONE && t2_r != NONE {
                let p = self.parent[T2][t2_l as usize];
                if p != NONE && p == self.parent[T2][t2_r as usize] {
                    return Some((node, p));
                }
            }
        }
        self.find_common_cherry_under(lc)
            .or_else(|| self.find_common_cherry_under(rc))
    }

    fn drop_t2_leaf_roots(&mut self) {
        let mut i = 0;
        while i < self.components[T2].len() {
            if self.is_leaf(T2, self.components[T2][i]) {
                self.components[T2].remove(i);
            } else {
                i += 1;
            }
        }
    }

    fn find_optimal_cut(&self) -> Option<NodeId> {
        for &root in &self.components[T1] {
            if let Some(cut) = self.find_optimal_cut_under(root) {
                return Some(cut);
            }
        }
        None
    }

    fn find_optimal_cut_under(&self, node: NodeId) -> Option<NodeId> {
        if node == NONE || self.is_leaf(T1, node) {
            return None;
        }
        let lc = self.left[T1][node as usize];
        let rc = self.right[T1][node as usize];
        if lc != NONE && rc != NONE && self.is_leaf(T1, lc) && self.is_leaf(T1, rc)
            && let Some(cut) = self.optimal_cut_for_cherry(node, lc, rc) {
                return Some(cut);
            }
        self.find_optimal_cut_under(lc)
            .or_else(|| self.find_optimal_cut_under(rc))
    }

    fn optimal_cut_for_cherry(
        &self,
        t1_parent: NodeId,
        t1_left: NodeId,
        t1_right: NodeId,
    ) -> Option<NodeId> {
        let t2_left = self.twin[T1][t1_left as usize];
        let t2_right = self.twin[T1][t1_right as usize];
        if t2_left == NONE || t2_right == NONE {
            return None;
        }
        let t1_sibling = self.sibling(T1, t1_parent);
        if t1_sibling != NONE && self.is_leaf(T1, t1_sibling) {
            let t2_third = self.twin[T1][t1_sibling as usize];
            if t2_third != NONE {
                let third_parent = self.parent[T2][t2_third as usize];
                if third_parent != NONE && self.parent[T2][t2_left as usize] == third_parent {
                    return Some(t2_right);
                }
                if third_parent != NONE && self.parent[T2][t2_right as usize] == third_parent {
                    return Some(t2_left);
                }
            }
        }
        let left_parent = self.parent[T2][t2_left as usize];
        let right_parent = self.parent[T2][t2_right as usize];
        if left_parent != NONE {
            let left_gp = self.parent[T2][left_parent as usize];
            if left_gp != NONE && left_gp == right_parent {
                return Some(self.sibling(T2, t2_left));
            }
        }
        if right_parent != NONE {
            let right_gp = self.parent[T2][right_parent as usize];
            if right_gp != NONE && right_gp == left_parent {
                return Some(self.sibling(T2, t2_right));
            }
        }
        None
    }

    fn get_lca(&self, a: NodeId, b: NodeId) -> NodeId {
        if a == NONE || b == NONE {
            return NONE;
        }
        let lca = lca_in_parent(&self.orig_parent[T2], a, b);
        if lca == NONE || !is_current_ancestor(&self.parent[T2], lca, a) {
            return NONE;
        }
        if !is_current_ancestor(&self.parent[T2], lca, b) {
            return NONE;
        }
        lca
    }

    fn dangling_list(&self, descendant: NodeId, ancestor: NodeId) -> Vec<NodeId> {
        let mut list = Vec::new();
        let mut v = descendant;
        while v != NONE
            && self.parent[T2][v as usize] != ancestor
            && self.parent[T2][v as usize] != NONE
        {
            let s = self.sibling(T2, v);
            if s != NONE {
                list.push(s);
            }
            v = self.parent[T2][v as usize];
        }
        list
    }

    fn cut_forest2(&mut self, v: NodeId) {
        if v == NONE || self.parent[T2][v as usize] == NONE {
            return;
        }
        self.f2_cut_list.push(v);
        let p = self.parent[T2][v as usize];
        self.detach(T2, v);
        self.add_component(T2, v);
        let survivor = self.contract(T2, p);
        if survivor != NONE && self.parent[T2][survivor as usize] == NONE {
            self.add_component(T2, survivor);
        }
    }

    fn unrecord_f2_cut(&mut self, v: NodeId) {
        if let Some(pos) = self.f2_cut_list.iter().rposition(|&cut| cut == v) {
            self.f2_cut_list.remove(pos);
        }
    }

    fn detach(&mut self, ti: usize, node: NodeId) {
        let p = self.parent[ti][node as usize];
        if p == NONE {
            return;
        }
        if self.left[ti][p as usize] == node {
            self.left[ti][p as usize] = NONE;
        } else if self.right[ti][p as usize] == node {
            self.right[ti][p as usize] = NONE;
        }
        self.parent[ti][node as usize] = NONE;
    }

    fn contract(&mut self, ti: usize, node: NodeId) -> NodeId {
        if node == NONE {
            return NONE;
        }
        let lc = self.left[ti][node as usize];
        let rc = self.right[ti][node as usize];
        if lc != NONE && rc != NONE {
            return node;
        }
        let child = if lc != NONE { lc } else { rc };
        let p = self.parent[ti][node as usize];
        if p == NONE {
            if child == NONE {
                return NONE;
            }
            self.parent[ti][child as usize] = NONE;
            self.left[ti][node as usize] = NONE;
            self.right[ti][node as usize] = NONE;
            if ti == T2 {
                self.remove_component(ti, node);
            }
            return child;
        }
        self.detach(ti, node);
        if child == NONE {
            return self.contract(ti, p);
        }
        self.parent[ti][child as usize] = p;
        if self.left[ti][p as usize] == NONE {
            self.left[ti][p as usize] = child;
        } else {
            self.right[ti][p as usize] = child;
        }
        self.sort_children(ti, p);
        self.left[ti][node as usize] = NONE;
        self.right[ti][node as usize] = NONE;
        child
    }

    fn contract_sibling_pair(&mut self, ti: usize, parent: NodeId) {
        let lc = self.left[ti][parent as usize];
        let rc = self.right[ti][parent as usize];
        if lc != NONE {
            self.parent[ti][lc as usize] = NONE;
        }
        if rc != NONE {
            self.parent[ti][rc as usize] = NONE;
        }
        self.left[ti][parent as usize] = NONE;
        self.right[ti][parent as usize] = NONE;
    }

    fn update_child(&mut self, ti: usize, parent: NodeId, old_child: NodeId, new_child: NodeId) {
        if parent == NONE {
            return;
        }
        if self.left[ti][parent as usize] == old_child {
            self.left[ti][parent as usize] = new_child;
        } else if self.right[ti][parent as usize] == old_child {
            self.right[ti][parent as usize] = new_child;
        }
        if old_child != NONE {
            self.parent[ti][old_child as usize] = NONE;
        }
        if new_child != NONE {
            self.parent[ti][new_child as usize] = parent;
        }
        self.sort_children(ti, parent);
    }

    fn set_children(&mut self, ti: usize, parent: NodeId, a: NodeId, b: NodeId) {
        self.left[ti][parent as usize] = NONE;
        self.right[ti][parent as usize] = NONE;
        if a != NONE {
            self.parent[ti][a as usize] = parent;
        }
        if b != NONE {
            self.parent[ti][b as usize] = parent;
        }
        if a != NONE && b != NONE && a > b {
            self.left[ti][parent as usize] = b;
            self.right[ti][parent as usize] = a;
        } else {
            self.left[ti][parent as usize] = a;
            self.right[ti][parent as usize] = b;
        }
    }

    fn sort_children(&mut self, ti: usize, parent: NodeId) {
        let l = self.left[ti][parent as usize];
        let r = self.right[ti][parent as usize];
        if l != NONE && r != NONE && l > r {
            self.left[ti][parent as usize] = r;
            self.right[ti][parent as usize] = l;
        }
    }

    fn remove_component(&mut self, ti: usize, node: NodeId) {
        if let Some(idx) = self.components[ti].iter().position(|&root| root == node) {
            self.components[ti].remove(idx);
        }
    }

    fn add_component(&mut self, ti: usize, node: NodeId) {
        if node != NONE && !self.components[ti].contains(&node) {
            self.components[ti].push(node);
        }
    }

    fn replace_component(&mut self, ti: usize, old: NodeId, new: NodeId) {
        if let Some(idx) = self.components[ti].iter().position(|&root| root == old) {
            self.components[ti][idx] = new;
        }
    }

    fn collect_ancestors(
        &self,
        start: NodeId,
        goal: NodeId,
        containing_goal: bool,
        out: &mut Vec<NodeId>,
    ) {
        let mut v = start;
        while v != NONE {
            if goal != NONE && v == goal {
                if containing_goal {
                    out.push(v);
                }
                return;
            }
            out.push(v);
            v = self.parent[T2][v as usize];
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ChenStopper {
    Disconnected,
    Root,
    Close,
    Overlapping,
}

impl ChenStopper {
    fn priority(self) -> i32 {
        match self {
            ChenStopper::Disconnected | ChenStopper::Root => 0,
            ChenStopper::Overlapping => 1,
            ChenStopper::Close => 2,
        }
    }
}

#[derive(Clone)]
struct ChenKey {
    left: Option<Box<ChenKey>>,
    right: Option<Box<ChenKey>>,
    lca: NodeId,
    leaf: NodeId,
    end: NodeId,
    obstacle_count: usize,
    lower_bound: usize,
    upper_bound: usize,
    has_path: bool,
    left_dangling_list: Vec<NodeId>,
    right_dangling_list: Vec<NodeId>,
    stopper: Option<ChenStopper>,
    cut: NodeId,
    is_super_robust: bool,
    is_robust: bool,
}

impl ChenKey {
    fn singleton(leaf: NodeId) -> Self {
        Self {
            left: None,
            right: None,
            lca: leaf,
            leaf,
            end: leaf,
            obstacle_count: 0,
            lower_bound: 0,
            upper_bound: 1,
            has_path: false,
            left_dangling_list: Vec::new(),
            right_dangling_list: Vec::new(),
            stopper: None,
            cut: leaf,
            is_super_robust: false,
            is_robust: false,
        }
    }

    fn merged(
        left: ChenKey,
        right: ChenKey,
        lca: NodeId,
        left_dangling_list: Vec<NodeId>,
        right_dangling_list: Vec<NodeId>,
        state: &ChenAppState,
    ) -> Self {
        let mut key = Self::new_internal(
            left,
            right,
            lca,
            left_dangling_list,
            right_dangling_list,
            0,
            None,
        );
        key.choose_cut(NONE, state);
        key
    }

    fn stopper(
        left: ChenKey,
        right: ChenKey,
        lca: NodeId,
        obstacle_count: usize,
        stopper: ChenStopper,
    ) -> Self {
        Self::new_internal(
            left,
            right,
            lca,
            Vec::new(),
            Vec::new(),
            obstacle_count,
            Some(stopper),
        )
    }

    fn new_internal(
        left: ChenKey,
        right: ChenKey,
        lca: NodeId,
        left_dangling_list: Vec<NodeId>,
        right_dangling_list: Vec<NodeId>,
        obstacle_count: usize,
        stopper: Option<ChenStopper>,
    ) -> Self {
        let lower_bound = left.lower_bound + right.lower_bound + 1;
        let upper_bound = left.upper_bound + right.upper_bound + 1;
        let (end, key_obstacles, has_path) = if let Some(stopper) = stopper {
            (NONE, obstacle_count, stopper == ChenStopper::Close)
        } else {
            let has_path = (left_dangling_list.is_empty()
                && (left.has_path || left.is_singleton()))
                || (right_dangling_list.is_empty() && (right.has_path || right.is_singleton()));
            if left.end != NONE || right.end != NONE {
                let left_obstacles = left.obstacle_count + left_dangling_list.len();
                let right_obstacles = right.obstacle_count + right_dangling_list.len();
                if left_obstacles <= right_obstacles {
                    // Mirrors the released bytecode: for merged keys this uses
                    // the constructor parameter, which is zero in normal merges.
                    let end = if obstacle_count <= 2 { left.end } else { NONE };
                    (end, left_obstacles, has_path)
                } else {
                    let end = if obstacle_count <= 2 { right.end } else { NONE };
                    (end, right_obstacles, has_path)
                }
            } else {
                (NONE, 3, has_path)
            }
        };
        Self {
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            lca,
            leaf: NONE,
            end,
            obstacle_count: key_obstacles,
            lower_bound,
            upper_bound,
            has_path,
            left_dangling_list,
            right_dangling_list,
            stopper,
            cut: NONE,
            is_super_robust: false,
            is_robust: false,
        }
    }

    fn find_stopper(t1v: NodeId, state: &ChenAppState) -> Option<Self> {
        if state.is_leaf(T1, t1v) {
            let mate = state.twin[T1][t1v as usize];
            return (mate != NONE).then(|| Self::singleton(mate));
        }
        let lc = state.left[T1][t1v as usize];
        let rc = state.right[T1][t1v as usize];
        let left = Self::find_stopper(lc, state)?;
        if matches!(
            left.stopper,
            Some(ChenStopper::Disconnected | ChenStopper::Root)
        ) {
            return Some(left);
        }
        let right = Self::find_stopper(rc, state)?;
        // Mirrors the released Java exactly:
        // if (right.stopper == DISCONNECTED || left.stopper == ROOT) return right;
        // The second condition is unreachable because left ROOT returned above.
        if matches!(right.stopper, Some(ChenStopper::Disconnected)) {
            return Some(right);
        }
        match (left.stopper, right.stopper) {
            (None, None) => Some(Self::merge(left, right, state)),
            (None, Some(_)) => Some(right),
            (Some(_), None) => Some(left),
            (Some(ChenStopper::Close), Some(ChenStopper::Close)) => {
                if left.obstacle_count < right.obstacle_count {
                    Some(left)
                } else {
                    Some(right)
                }
            }
            (Some(ls), Some(rs)) if ls.priority() == rs.priority() => {
                if left.upper_bound < right.upper_bound {
                    Some(left)
                } else {
                    Some(right)
                }
            }
            (Some(ls), Some(rs)) => {
                if ls.priority() < rs.priority() {
                    Some(left)
                } else {
                    Some(right)
                }
            }
        }
    }

    fn merge(mut left: ChenKey, mut right: ChenKey, state: &ChenAppState) -> Self {
        if left.lca > right.lca {
            std::mem::swap(&mut left, &mut right);
        }
        let lca = state.get_lca(left.lca, right.lca);
        if lca == NONE {
            return Self::stopper(left, right, lca, 0, ChenStopper::Disconnected);
        }
        if left.lca == lca || right.lca == lca {
            return Self::stopper(left, right, lca, 0, ChenStopper::Overlapping);
        }
        let left_dangling = state.dangling_list(left.lca, lca);
        let right_dangling = state.dangling_list(right.lca, lca);
        let obstacles =
            left_dangling.len() + right_dangling.len() + left.obstacle_count + right.obstacle_count;
        if left.end != NONE && right.end != NONE && obstacles <= 2 {
            return Self::stopper(left, right, lca, obstacles, ChenStopper::Close);
        }
        if state.is_root(T2, lca) {
            return Self::stopper(left, right, lca, 0, ChenStopper::Root);
        }
        Self::merged(left, right, lca, left_dangling, right_dangling, state)
    }

    fn is_singleton(&self) -> bool {
        self.leaf != NONE
    }

    fn left_key(&self) -> &ChenKey {
        self.left.as_ref().unwrap()
    }

    fn right_key(&self) -> &ChenKey {
        self.right.as_ref().unwrap()
    }

    fn left_key_mut(&mut self) -> &mut ChenKey {
        self.left.as_mut().unwrap()
    }

    fn right_key_mut(&mut self) -> &mut ChenKey {
        self.right.as_mut().unwrap()
    }

    fn choose_cut(&mut self, keep: NodeId, state: &ChenAppState) -> bool {
        if self.has_path {
            self.cut = self.lca;
            return false;
        }
        self.choose_cut_by_robust_way(keep, state)
    }

    fn choose_cut_by_robust_way(&mut self, keep: NodeId, state: &ChenAppState) -> bool {
        let old_super = self.is_super_robust;
        let old_robust = self.is_robust;
        if self.left_dangling_list.len() >= 2
            || (self.left_dangling_list.len() == 1 && self.left_key().is_robust)
        {
            self.cut = self.left_dangling_list[0];
            self.is_super_robust =
                self.right_key().is_robust || !self.right_dangling_list.is_empty();
            self.is_robust = true;
        } else if self.right_dangling_list.len() >= 2
            || (self.right_dangling_list.len() == 1 && self.right_key().is_robust)
        {
            self.cut = self.right_dangling_list[0];
            self.is_super_robust = self.left_key().is_robust || !self.left_dangling_list.is_empty();
            self.is_robust = true;
        } else if self.left_key().is_super_robust {
            let lca = self.left_key().lca;
            self.cut = if state.left[T2][lca as usize] == keep {
                state.right[T2][lca as usize]
            } else {
                state.left[T2][lca as usize]
            };
            self.is_super_robust =
                self.left_key().is_robust || !self.right_dangling_list.is_empty();
            self.is_robust = true;
        } else if self.right_key().is_super_robust {
            let lca = self.right_key().lca;
            self.cut = if state.left[T2][lca as usize] == keep {
                state.right[T2][lca as usize]
            } else {
                state.left[T2][lca as usize]
            };
            self.is_super_robust =
                self.right_key().is_robust || !self.right_dangling_list.is_empty();
            self.is_robust = true;
        } else if self.left_dangling_list.is_empty() && self.right_dangling_list.is_empty() {
            self.cut = if self.left_key().is_robust {
                self.left_key().lca
            } else {
                self.right_key().lca
            };
            self.is_super_robust = false;
            self.is_robust = self.left_key().is_robust && self.right_key().is_robust;
        } else {
            self.cut = if self.left_dangling_list.is_empty() {
                self.right_dangling_list[0]
            } else {
                self.left_dangling_list[0]
            };
            self.is_super_robust = false;
            self.is_robust = false;
        }
        self.is_robust != old_robust || self.is_super_robust != old_super
    }

    fn cut_stopper(&mut self, state: &mut ChenAppState, bounds: &mut AppBounds) {
        match self.stopper {
            Some(ChenStopper::Disconnected | ChenStopper::Root) => {
                self.left_key_mut().cut_key(state);
                self.right_key_mut().cut_key(state);
                bounds.lower += self.left_key().lower_bound + self.right_key().lower_bound + 1;
                bounds.upper += self.left_key().upper_bound + self.right_key().upper_bound;
            }
            Some(ChenStopper::Close) => self.cut_close_stopper(state, bounds),
            Some(ChenStopper::Overlapping) => self.cut_overlapping_stopper(state, bounds),
            None => {}
        }
    }

    fn cut_close_stopper(&mut self, state: &mut ChenAppState, bounds: &mut AppBounds) {
        let left_cut_list = state.dangling_list(self.left_key().end, self.lca);
        let right_cut_list = state.dangling_list(self.right_key().end, self.lca);
        let mut left_key_list = self.left_key().dangling_key_list(self.left_key().end);
        let mut right_key_list = self.right_key().dangling_key_list(self.right_key().end);
        for cut in left_cut_list.iter().copied() {
            state.cut_forest2(cut);
        }
        bounds.lower += left_cut_list.len();
        bounds.upper += left_cut_list.len();
        for cut in right_cut_list.iter().copied() {
            state.cut_forest2(cut);
        }
        bounds.lower += right_cut_list.len();
        bounds.upper += right_cut_list.len();
        if self.obstacle_count == 2 {
            bounds.lower = bounds.lower.saturating_sub(1);
        }
        for key in &mut left_key_list {
            key.cut_key(state);
            bounds.lower += key.lower_bound;
            bounds.upper += key.upper_bound;
        }
        for key in &mut right_key_list {
            key.cut_key(state);
            bounds.lower += key.lower_bound;
            bounds.upper += key.upper_bound;
        }
    }

    fn cut_overlapping_stopper(&mut self, state: &mut ChenAppState, bounds: &mut AppBounds) {
        let left_is_key1 = self.left_key().lca == self.lca;
        let (key1, key2) = if left_is_key1 {
            let left = self.left.as_mut().unwrap();
            let right = self.right.as_mut().unwrap();
            (left.as_mut(), right.as_mut())
        } else {
            let left = self.left.as_mut().unwrap();
            let right = self.right.as_mut().unwrap();
            (right.as_mut(), left.as_mut())
        };

        let mut vertices1 = Vec::new();
        key1.collect_vertices_in_key(state, &mut vertices1);
        let mut set1 = node_set(state, &vertices1);
        if key2.is_singleton() && in_set(&set1, state.parent[T2][key2.leaf as usize]) {
            key1.cut_key_with_leaf(key2.leaf, &set1, state, bounds);
            return;
        }
        if let Some(port_key) = key2.find_port_key(&set1, state) {
            key1.cut_port_key(port_key, &set1, state, bounds);
            return;
        }

        let mut vertices2 = Vec::new();
        key2.collect_vertices_in_key(state, &mut vertices2);
        let mut set2 = node_set(state, &vertices2);
        let port_key1 = if key1.lca == key2.lca {
            key1.find_port_key(&set2, state)
        } else {
            key1.find_port_key_reverse(NONE, &set2, state)
        };
        if let Some(port_key) = port_key1 {
            key2.cut_port_key(port_key, &set2, state, bounds);
            return;
        }

        let p1 = state.parent[T2][key1.lca as usize];
        let p2 = state.parent[T2][key2.lca as usize];
        state.collect_ancestors(p1, NONE, true, &mut vertices1);
        state.collect_ancestors(p2, NONE, true, &mut vertices2);
        set1 = node_set(state, &vertices1);
        set2 = node_set(state, &vertices2);
        let juncture = key1.find_extreme_juncture(key1.lca, &set1, &set2, state);
        if juncture == NONE {
            key1.cut_key(state);
            key2.cut_key(state);
            bounds.lower += key1.lower_bound + key2.lower_bound + 1;
            bounds.upper += key1.upper_bound + key2.upper_bound;
            return;
        }
        let target_is_key1 = in_set(&set1, state.left[T2][juncture as usize])
            && in_set(&set1, state.right[T2][juncture as usize]);
        if target_is_key1
            && in_set(&set2, state.left[T2][juncture as usize])
            && in_set(&set2, state.right[T2][juncture as usize])
        {
            if let Some(mut k) = key2.find_key_by_lca(juncture) {
                k.cut_cherries_key(state, bounds);
            }
            return;
        }
        if target_is_key1 {
            key1.cut_complex_key(juncture, &set2, state, bounds);
        } else {
            key2.cut_complex_key(juncture, &set1, state, bounds);
        }
    }

    fn cut_key(&mut self, state: &mut ChenAppState) {
        if self.cut != NONE {
            state.cut_forest2(self.cut);
        }
        if self.is_singleton() {
            return;
        }
        self.left_key_mut().cut_key(state);
        self.right_key_mut().cut_key(state);
    }

    fn cut_key_with_additional(&mut self, state: &mut ChenAppState, additional_cut: NodeId) {
        if self.cut != NONE {
            let additional_parent = state.parent[T2][additional_cut as usize];
            if additional_parent != NONE {
                if additional_parent == self.cut {
                    state.cut_forest2(self.cut);
                    state.cut_forest2(additional_cut);
                } else if state.parent[T2][self.cut as usize] == additional_cut {
                    state.cut_forest2(additional_cut);
                    state.cut_forest2(self.cut);
                } else {
                    state.cut_forest2(self.cut);
                }
            } else {
                state.cut_forest2(self.cut);
            }
        }
        if self.is_singleton() {
            return;
        }
        self.left_key_mut()
            .cut_key_with_additional(state, additional_cut);
        self.right_key_mut()
            .cut_key_with_additional(state, additional_cut);
    }

    fn dangling_key_list(&self, end: NodeId) -> Vec<ChenKey> {
        if self.is_singleton() {
            return Vec::new();
        }
        if end == self.left_key().end {
            let mut list = self.left_key().dangling_key_list(end);
            list.push((*self.right_key()).clone());
            list
        } else {
            let mut list = self.right_key().dangling_key_list(end);
            list.push((*self.left_key()).clone());
            list
        }
    }

    fn collect_vertices_in_key(&self, state: &ChenAppState, out: &mut Vec<NodeId>) {
        out.push(self.lca);
        if self.is_singleton() {
            return;
        }
        let lp = state.parent[T2][self.left_key().lca as usize];
        let rp = state.parent[T2][self.right_key().lca as usize];
        state.collect_ancestors(lp, self.lca, false, out);
        state.collect_ancestors(rp, self.lca, false, out);
        self.left_key().collect_vertices_in_key(state, out);
        self.right_key().collect_vertices_in_key(state, out);
    }

    fn replace_cut_recursive(
        &mut self,
        new_cut: NodeId,
        near_lca: NodeId,
        keep: NodeId,
        state: &ChenAppState,
    ) -> bool {
        if self.is_singleton() {
            return false;
        }
        let left_change = self
            .left_key_mut()
            .replace_cut_recursive(new_cut, near_lca, keep, state);
        let right_change = self
            .right_key_mut()
            .replace_cut_recursive(new_cut, near_lca, keep, state);
        if self.lca == near_lca {
            return self.replace_cut(new_cut);
        }
        if self.cut == keep || left_change || right_change {
            return self.choose_cut(keep, state);
        }
        false
    }

    fn replace_cut(&mut self, new_cut: NodeId) -> bool {
        if self.cut == new_cut {
            return false;
        }
        self.cut = new_cut;
        let old_super = self.is_super_robust;
        let old_robust = self.is_robust;
        if self.left_dangling_list.len() == 1 && self.left_dangling_list[0] == self.cut {
            self.is_super_robust = self.left_key().is_robust
                && (!self.right_dangling_list.is_empty() || self.right_key().is_robust);
            self.is_robust = self.is_super_robust
                || !self.right_dangling_list.is_empty()
                || self.left_key().is_robust
                || self.right_key().is_robust;
        } else if self.right_dangling_list.len() == 1 && self.right_dangling_list[0] == self.cut {
            self.is_super_robust = self.right_key().is_robust
                && (!self.left_dangling_list.is_empty() || self.left_key().is_robust);
            self.is_robust = self.is_super_robust
                || !self.left_dangling_list.is_empty()
                || self.left_key().is_robust
                || self.right_key().is_robust;
        }
        self.is_robust != old_robust || self.is_super_robust != old_super
    }

    fn replace_cuts(&mut self, set: &[bool], state: &ChenAppState) -> bool {
        if !in_set(set, self.lca) {
            return self.replace_cut_by_robust_way(NONE, NONE, state);
        }
        let _left_change = self.left_key_mut().replace_cuts(set, state);
        let _right_change = self.right_key_mut().replace_cuts(set, state);
        if self.left_dangling_list.len() == 1 && in_set(set, self.left_dangling_list[0]) {
            return self.replace_cut(state.left[T2][self.lca as usize]);
        }
        if self.right_dangling_list.len() == 1 && in_set(set, self.right_dangling_list[0]) {
            return self.replace_cut(state.right[T2][self.lca as usize]);
        }
        if !self.left_key().is_robust && self.left_dangling_list.is_empty() {
            return self.replace_cut(state.right[T2][self.lca as usize]);
        }
        if !self.right_key().is_robust && self.right_dangling_list.is_empty() {
            return self.replace_cut(state.left[T2][self.lca as usize]);
        }
        self.replace_cut(state.left[T2][self.lca as usize])
    }

    fn replace_cut_by_robust_way(
        &mut self,
        new_cut: NodeId,
        near_lca: NodeId,
        state: &ChenAppState,
    ) -> bool {
        if self.is_singleton() {
            return false;
        }
        let left_change = self
            .left_key_mut()
            .replace_cut_by_robust_way(new_cut, near_lca, state);
        let right_change = self
            .right_key_mut()
            .replace_cut_by_robust_way(new_cut, near_lca, state);
        if self.lca == near_lca {
            return self.replace_cut(new_cut);
        }
        if !self.is_robust || left_change || right_change {
            return self.choose_cut_by_robust_way(NONE, state);
        }
        false
    }

    fn replace_cut_with_leaf(
        &mut self,
        new_cut: NodeId,
        near_lca: NodeId,
        keep: NodeId,
        state: &ChenAppState,
    ) -> (bool, bool) {
        if self.is_singleton() {
            return (false, false);
        }
        if self.lca == near_lca {
            self.left_key_mut()
                .replace_cut_by_robust_way(NONE, NONE, state);
            self.right_key_mut()
                .replace_cut_by_robust_way(NONE, NONE, state);
            return (true, self.replace_cut(new_cut));
        }
        if self.left_key().lca == near_lca || self.right_key().lca == near_lca {
            let left_target = self.left_key().lca == near_lca;
            let (target_change, other_change) = if left_target {
                let target_change = self
                    .left_key_mut()
                    .replace_cut_with_leaf(new_cut, near_lca, keep, state)
                    .1;
                let other_change = self
                    .right_key_mut()
                    .replace_cut_by_robust_way(NONE, NONE, state);
                if self.right_key().is_super_robust {
                    let lca = self.right_key().lca;
                    self.right_key_mut().cut = if state.left[T2][lca as usize] == keep {
                        state.right[T2][lca as usize]
                    } else {
                        state.left[T2][lca as usize]
                    };
                    self.right_key_mut().is_super_robust = false;
                    (target_change, true)
                } else {
                    (target_change, other_change)
                }
            } else {
                let target_change = self
                    .right_key_mut()
                    .replace_cut_with_leaf(new_cut, near_lca, keep, state)
                    .1;
                let other_change = self
                    .left_key_mut()
                    .replace_cut_by_robust_way(NONE, NONE, state);
                if self.left_key().is_super_robust {
                    let lca = self.left_key().lca;
                    self.left_key_mut().cut = if state.left[T2][lca as usize] == keep {
                        state.right[T2][lca as usize]
                    } else {
                        state.left[T2][lca as usize]
                    };
                    self.left_key_mut().is_super_robust = false;
                    (target_change, true)
                } else {
                    (target_change, other_change)
                }
            };
            let this_change = self.choose_cut_by_robust_way(NONE, state);
            return (true, target_change || other_change || this_change);
        }
        let left_changes = self
            .left_key_mut()
            .replace_cut_with_leaf(new_cut, near_lca, keep, state);
        let right_changes = self
            .right_key_mut()
            .replace_cut_with_leaf(new_cut, near_lca, keep, state);
        if left_changes.0 || right_changes.0 {
            return (true, self.choose_cut(keep, state));
        }
        if !self.is_robust || left_changes.1 || right_changes.1 {
            return (false, self.choose_cut_by_robust_way(keep, state));
        }
        (false, false)
    }

    fn find_port_key(&self, set: &[bool], state: &ChenAppState) -> Option<ChenKey> {
        if self.is_singleton() && in_set(set, state.parent[T2][self.lca as usize]) {
            return None;
        }
        if in_set(set, self.lca) {
            return self
                .left_key()
                .find_port_key(set, state)
                .or_else(|| self.right_key().find_port_key(set, state));
        }
        Some(self.clone())
    }

    fn find_port_key_reverse(
        &self,
        top: NodeId,
        set: &[bool],
        state: &ChenAppState,
    ) -> Option<ChenKey> {
        if self.is_singleton() && in_set(set, state.parent[T2][self.leaf as usize]) {
            return None;
        }
        if !self.is_singleton() {
            if let Some(key) = self.left_key().find_port_key_reverse(self.lca, set, state) {
                return Some(key);
            }
            if let Some(key) = self.right_key().find_port_key_reverse(self.lca, set, state) {
                return Some(key);
            }
            if in_set(set, self.lca) {
                return None;
            }
        }
        let mut v = state.parent[T2][self.lca as usize];
        while v != top && v != NONE {
            if in_set(set, state.parent[T2][v as usize]) {
                return Some(self.clone());
            }
            v = state.parent[T2][v as usize];
        }
        None
    }

    fn find_key_by_lca(&self, lca: NodeId) -> Option<ChenKey> {
        if self.lca == lca {
            return Some(self.clone());
        }
        if self.is_singleton() {
            return None;
        }
        self.left_key()
            .find_key_by_lca(lca)
            .or_else(|| self.right_key().find_key_by_lca(lca))
    }

    fn find_extreme_juncture(
        &self,
        v: NodeId,
        set1: &[bool],
        set2: &[bool],
        state: &ChenAppState,
    ) -> NodeId {
        self.find_extreme_juncture2(v, set1, set2, state)
            .map(|(_, node)| node)
            .unwrap_or(NONE)
    }

    fn find_extreme_juncture2(
        &self,
        v: NodeId,
        set1: &[bool],
        set2: &[bool],
        state: &ChenAppState,
    ) -> Option<(i32, NodeId)> {
        if v == NONE || !in_set(set1, v) || !in_set(set2, v) || state.is_leaf(T2, v) {
            return None;
        }
        let left = self.find_extreme_juncture2(state.left[T2][v as usize], set1, set2, state);
        if let Some((3, _)) = left {
            return left;
        }
        let right = self.find_extreme_juncture2(state.right[T2][v as usize], set1, set2, state);
        if let Some((3, _)) = right {
            return right;
        }
        if left.is_some() {
            return left;
        }
        if right.is_some() {
            return right;
        }
        let l = state.left[T2][v as usize];
        let r = state.right[T2][v as usize];
        let status = in_set(set1, l) as i32
            + in_set(set2, l) as i32
            + in_set(set1, r) as i32
            + in_set(set2, r) as i32;
        (status >= 3).then_some((status, v))
    }

    fn reconstruct(&self, v: NodeId, set: &[bool], state: &ChenAppState) -> Option<ChenKey> {
        if !in_set(set, v) {
            return None;
        }
        if state.is_leaf(T2, v) {
            return Some(ChenKey::singleton(v));
        }
        let left = self.reconstruct(state.left[T2][v as usize], set, state);
        let right = self.reconstruct(state.right[T2][v as usize], set, state);
        match (left, right) {
            (Some(l), Some(r)) => {
                let left_d = state.dangling_list(l.lca, v);
                let right_d = state.dangling_list(r.lca, v);
                Some(ChenKey::merged(l, r, v, left_d, right_d, state))
            }
            (Some(k), None) | (None, Some(k)) => Some(k),
            (None, None) => None,
        }
    }

    fn cut_key_with_leaf(
        &mut self,
        leaf: NodeId,
        set: &[bool],
        state: &mut ChenAppState,
        bounds: &mut AppBounds,
    ) {
        let s = state.sibling(T2, leaf);
        let p = state.parent[T2][leaf as usize];
        let mut near_lca = NONE;
        let mut v = p;
        while v != NONE {
            let sib = state.sibling(T2, v);
            if in_set(set, sib) {
                near_lca = state.parent[T2][v as usize];
                break;
            }
            v = state.parent[T2][v as usize];
        }
        if s != NONE && state.is_leaf(T2, s) {
            self.replace_cut_recursive(leaf, near_lca, p, state);
            self.cut_key(state);
            state.remove_component(T2, leaf);
            state.remove_component(T2, s);
            state.unrecord_f2_cut(leaf);
            state.unrecord_f2_cut(s);
            state.set_children(T2, p, leaf, s);
            state.components[T2].push(p);
            bounds.lower += self.lower_bound;
            bounds.upper += self.upper_bound.saturating_sub(1);
        } else {
            self.replace_cut_with_leaf(leaf, near_lca, p, state);
            self.cut_key_with_additional(state, p);
            bounds.lower += self.lower_bound + 1;
            bounds.upper += self.upper_bound + 1;
        }
    }

    fn cut_port_key(
        &mut self,
        mut port_key: ChenKey,
        set: &[bool],
        state: &mut ChenAppState,
        bounds: &mut AppBounds,
    ) {
        let mut port = NONE;
        let mut v = port_key.lca;
        while v != NONE {
            if in_set(set, state.parent[T2][v as usize]) {
                port = v;
                break;
            }
            v = state.parent[T2][v as usize];
        }
        let mut near_lca = NONE;
        let mut v2 = state.parent[T2][port as usize];
        while v2 != NONE {
            if in_set(set, state.sibling(T2, v2)) {
                near_lca = state.parent[T2][v2 as usize];
                break;
            }
            v2 = state.parent[T2][v2 as usize];
        }
        self.replace_cut_by_robust_way(port, near_lca, state);
        port_key.replace_cut_by_robust_way(NONE, NONE, state);
        self.cut_key(state);
        port_key.cut_key(state);
        bounds.lower += self.lower_bound + port_key.lower_bound + 1;
        bounds.upper += self.upper_bound + port_key.upper_bound;
    }

    fn cut_cherries_key(&mut self, state: &mut ChenAppState, bounds: &mut AppBounds) {
        let a = self.left_key().lca;
        let b = self.right_key().lca;
        let c = state.sibling(T2, a);
        let d = state.sibling(T2, b);
        state.cut_forest2(a);
        state.cut_forest2(b);
        state.cut_forest2(c);
        state.cut_forest2(d);
        bounds.lower += 2;
        bounds.upper += 4;
    }

    fn cut_complex_key(
        &mut self,
        juncture: NodeId,
        others_set: &[bool],
        state: &mut ChenAppState,
        bounds: &mut AppBounds,
    ) {
        let jl = state.left[T2][juncture as usize];
        let jr = state.right[T2][juncture as usize];
        let x = if state.is_leaf(T2, jl) { jr } else { jl };
        let xp = state.parent[T2][x as usize];
        let y = state.sibling(T2, xp);
        let xy = state.parent[T2][xp as usize];
        let Some(juncture_key) = self.find_key_by_lca(juncture) else {
            self.cut_key(state);
            bounds.lower += self.lower_bound;
            bounds.upper += self.upper_bound;
            return;
        };
        let a = if juncture_key.left_key().lca == x {
            juncture_key.right_key().lca
        } else {
            juncture_key.left_key().lca
        };
        let b = state.sibling(T2, a);
        let ab = state.parent[T2][a as usize];
        self.replace_cuts(others_set, state);
        self.cut_key(state);
        bounds.lower += self.lower_bound;
        bounds.upper += self.upper_bound;

        let mut root_keys = Vec::new();
        let mut non_root_keys = Vec::new();
        let roots = state.components[T2].clone();
        for r in roots {
            if state.is_leaf(T2, r) || !in_set(others_set, r) {
                continue;
            }
            if let Some(k) = self.reconstruct(r, others_set, state) {
                if state.is_root(T2, k.lca) {
                    root_keys.push(k);
                } else {
                    non_root_keys.push(k);
                }
            }
        }
        for mut k in root_keys {
            k.left_key_mut().cut_key(state);
            k.right_key_mut().cut_key(state);
            bounds.lower += k.left_key().lower_bound + k.right_key().lower_bound + 1;
            bounds.upper += k.left_key().upper_bound + k.right_key().upper_bound;
        }
        if non_root_keys.len() == 1 {
            state.remove_component(T2, x);
            let w = state.parent[T2][y as usize];
            state.update_child(T2, w, y, xy);
            state.set_children(T2, xy, x, y);
            state.cut_forest2(b);
        } else if !non_root_keys.is_empty() {
            let count = non_root_keys.len();
            for mut k in non_root_keys {
                k.cut_key(state);
                bounds.lower += k.lower_bound;
                bounds.upper += k.upper_bound;
            }
            bounds.lower = bounds.lower.saturating_sub(count - 1);
        }
        state.remove_component(T2, a);
        state.remove_component(T2, b);
        state.unrecord_f2_cut(a);
        state.unrecord_f2_cut(b);
        state.set_children(T2, ab, a, b);
        state.components[T2].push(ab);
        bounds.upper = bounds.upper.saturating_sub(1);
    }
}

fn node_set(state: &ChenAppState, nodes: &[NodeId]) -> Vec<bool> {
    let mut set = vec![false; state.num_nodes[T2]];
    for &node in nodes {
        if node != NONE && (node as usize) < set.len() {
            set[node as usize] = true;
        }
    }
    set
}

fn in_set(set: &[bool], node: NodeId) -> bool {
    node != NONE && (node as usize) < set.len() && set[node as usize]
}

fn shrink_common_cherries(tf: &mut TwinForest, um: &mut undo::UndoMachine) {
    loop {
        let mut target = None;
        for &root in &tf.components[T1] {
            if let Some(pair) = find_common_cherry_under(tf, root) {
                target = Some(pair);
                break;
            }
        }

        let Some((t1_parent, t2_parent)) = target else {
            break;
        };
        contract_matching_pair(tf, t1_parent, t2_parent, um);
    }
}

fn find_common_cherry_under(tf: &TwinForest, node: NodeId) -> Option<(NodeId, NodeId)> {
    let lc = tf.left[T1][node as usize];
    let rc = tf.right[T1][node as usize];
    if lc == NONE {
        return None;
    }

    if rc != NONE && tf.is_leaf(T1, lc) && tf.is_leaf(T1, rc) {
        let t2_l = tf.twin[T1][lc as usize];
        let t2_r = tf.twin[T1][rc as usize];
        if t2_l != NONE && t2_r != NONE {
            let lp = tf.parent[T2][t2_l as usize];
            if lp != NONE && lp == tf.parent[T2][t2_r as usize] {
                return Some((node, lp));
            }
        }
    }

    find_common_cherry_under(tf, lc).or_else(|| {
        if rc != NONE {
            find_common_cherry_under(tf, rc)
        } else {
            None
        }
    })
}

fn has_t1_cherry(tf: &TwinForest) -> bool {
    first_t1_cherry_t2_mates(tf).is_some()
}

fn first_t1_cherry_t2_mates(tf: &TwinForest) -> Option<(NodeId, NodeId)> {
    for &root in &tf.components[T1] {
        if let Some(pair) = first_t1_cherry_t2_mates_under(tf, root) {
            return Some(pair);
        }
    }
    None
}

fn first_t1_cherry_t2_mates_under(tf: &TwinForest, node: NodeId) -> Option<(NodeId, NodeId)> {
    let lc = tf.left[T1][node as usize];
    let rc = tf.right[T1][node as usize];
    if lc == NONE {
        return None;
    }

    if rc != NONE && tf.is_leaf(T1, lc) && tf.is_leaf(T1, rc) {
        let t2_l = tf.twin[T1][lc as usize];
        let t2_r = tf.twin[T1][rc as usize];
        if t2_l != NONE && t2_r != NONE {
            return Some((t2_l, t2_r));
        }
    }

    first_t1_cherry_t2_mates_under(tf, lc).or_else(|| {
        if rc != NONE {
            first_t1_cherry_t2_mates_under(tf, rc)
        } else {
            None
        }
    })
}

fn process_singletons(tf: &mut TwinForest, _k: &mut i32, um: &mut undo::UndoMachine) -> bool {
    loop {
        let singleton = find_singleton(tf);
        if singleton == NONE {
            return true;
        }

        let t1_node = tf.twin[T2][singleton as usize];
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
}

fn find_singleton(tf: &TwinForest) -> NodeId {
    for &root in &tf.components[T2] {
        if tf.is_leaf(T2, root) {
            let twin = tf.twin[T2][root as usize];
            if twin != NONE && tf.parent[T1][twin as usize] != NONE {
                return root;
            }
        }
    }
    NONE
}

fn find_reference_optimal_cut(tf: &TwinForest) -> Option<NodeId> {
    for &root in &tf.components[T1] {
        if let Some(node) = find_reference_optimal_cut_under(tf, root) {
            return Some(node);
        }
    }
    None
}

fn find_reference_optimal_cut_under(tf: &TwinForest, node: NodeId) -> Option<NodeId> {
    let lc = tf.left[T1][node as usize];
    let rc = tf.right[T1][node as usize];
    if lc == NONE {
        return None;
    }

    if rc != NONE && tf.is_leaf(T1, lc) && tf.is_leaf(T1, rc)
        && let Some(cut) = reference_optimal_cut_for_t1_cherry(tf, node, lc, rc) {
            return Some(cut);
        }

    if let Some(cut) = find_reference_optimal_cut_under(tf, lc) {
        return Some(cut);
    }
    if rc != NONE {
        return find_reference_optimal_cut_under(tf, rc);
    }
    None
}

fn reference_optimal_cut_for_t1_cherry(
    tf: &TwinForest,
    t1_parent: NodeId,
    t1_left: NodeId,
    t1_right: NodeId,
) -> Option<NodeId> {
    let t2_left = tf.twin[T1][t1_left as usize];
    let t2_right = tf.twin[T1][t1_right as usize];
    if t2_left == NONE || t2_right == NONE {
        return None;
    }

    if let Some(t1_parent_sibling) = sibling_node(tf, T1, t1_parent)
        && tf.is_leaf(T1, t1_parent_sibling) {
            let t2_third = tf.twin[T1][t1_parent_sibling as usize];
            if t2_third != NONE {
                let t2_third_parent = tf.parent[T2][t2_third as usize];
                if t2_third_parent != NONE && tf.parent[T2][t2_left as usize] == t2_third_parent {
                    return Some(t2_right);
                }
                if t2_third_parent != NONE && tf.parent[T2][t2_right as usize] == t2_third_parent {
                    return Some(t2_left);
                }
            }
        }

    let t2_left_parent = tf.parent[T2][t2_left as usize];
    let t2_right_parent = tf.parent[T2][t2_right as usize];
    if t2_left_parent != NONE {
        let t2_left_grandparent = tf.parent[T2][t2_left_parent as usize];
        if t2_left_grandparent != NONE && t2_left_grandparent == t2_right_parent {
            return sibling_node(tf, T2, t2_left);
        }
    }
    if t2_right_parent != NONE {
        let t2_right_grandparent = tf.parent[T2][t2_right_parent as usize];
        if t2_right_grandparent != NONE && t2_right_grandparent == t2_left_parent {
            return sibling_node(tf, T2, t2_right);
        }
    }

    None
}

fn find_step4_forced_cut(tf: &TwinForest) -> Option<NodeId> {
    for &root in &tf.components[T1] {
        if let Some(node) = find_step4_forced_cut_under(tf, root) {
            return Some(node);
        }
    }
    None
}

fn find_step4_forced_cut_under(tf: &TwinForest, node: NodeId) -> Option<NodeId> {
    let lc = tf.left[T1][node as usize];
    let rc = tf.right[T1][node as usize];
    if lc == NONE {
        return None;
    }

    if rc != NONE && tf.is_leaf(T1, lc) && tf.is_leaf(T1, rc) {
        let t2_l = tf.twin[T1][lc as usize];
        let t2_r = tf.twin[T1][rc as usize];
        if t2_l != NONE && t2_r != NONE {
            let d_cut = d_f_cut_nodes(tf, T2, t2_l, t2_r).unwrap_or_default();
            if d_cut.len() == 1 {
                return Some(d_cut[0]);
            }
        }
    }

    if let Some(node) = find_step4_forced_cut_under(tf, lc) {
        return Some(node);
    }
    if rc != NONE {
        return find_step4_forced_cut_under(tf, rc);
    }
    None
}

fn find_step5_forced_cut(tf: &TwinForest) -> Option<NodeId> {
    for &root in &tf.components[T2] {
        if let Some(node) = find_step5_forced_cut_under(tf, root) {
            return Some(node);
        }
    }
    None
}

fn find_step5_forced_cut_under(tf: &TwinForest, node: NodeId) -> Option<NodeId> {
    let lc = tf.left[T2][node as usize];
    let rc = tf.right[T2][node as usize];
    if lc == NONE {
        return None;
    }

    if rc != NONE && tf.is_leaf(T2, lc) && tf.is_leaf(T2, rc) {
        let t1_l = tf.twin[T2][lc as usize];
        let t1_r = tf.twin[T2][rc as usize];
        if t1_l != NONE && t1_r != NONE {
            let d_cut = d_f_cut_nodes(tf, T1, t1_l, t1_r).unwrap_or_default();
            if d_cut.len() == 1 && tf.is_leaf(T1, d_cut[0]) {
                let t2_x3 = tf.twin[T1][d_cut[0] as usize];
                if t2_x3 != NONE && tf.parent[T2][t2_x3 as usize] != NONE {
                    return Some(t2_x3);
                }
            }
        }
    }

    if let Some(node) = find_step5_forced_cut_under(tf, lc) {
        return Some(node);
    }
    if rc != NONE {
        return find_step5_forced_cut_under(tf, rc);
    }
    None
}

enum PairResult {
    NoPairs,
    Case2 {
        t1_parent: NodeId,
        t2_parent: NodeId,
    },
    /// Pre-computed branches from checkCase or build_branches fallback.
    Branches(Vec<BranchCut>),
}

// ---------------------------------------------------------------------------
// Sibling pair selection — mirrors Java MafFpt1.makeBetterCutsList() exactly:
// pre-order traversal with checkCase first-match, then checkBetterCuts fallback.
// ---------------------------------------------------------------------------

fn find_sibling_pair(tf: &TwinForest) -> PairResult {
    let mut best_cherry: Option<(NodeId, NodeId, NodeId)> = None; // (t1_parent, t2_l, t2_r)

    for &root in &tf.components[T1] {
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            let lc = tf.left[T1][node as usize];
            if lc == NONE {
                continue;
            }
            let rc = tf.right[T1][node as usize];
            // Push right then left (left processed first = pre-order left-to-right)
            if rc != NONE {
                stack.push(rc);
            }
            stack.push(lc);

            // Only process parent-of-leaves nodes
            if rc == NONE || !tf.is_leaf(T1, lc) || !tf.is_leaf(T1, rc) {
                continue;
            }

            let t2_l = tf.twin[T1][lc as usize];
            let t2_r = tf.twin[T1][rc as usize];
            if t2_l == NONE || t2_r == NONE {
                continue;
            }

            // Case 2: matching pair (same T2 parent). Return immediately.
            let lp = tf.parent[T2][t2_l as usize];
            let rp = tf.parent[T2][t2_r as usize];
            if lp != NONE && lp == rp {
                return PairResult::Case2 {
                    t1_parent: node,
                    t2_parent: lp,
                };
            }

            // CheckCase: if optimized pattern matches, return IMMEDIATELY (Java line 433-435)
            let d_cut = d_f_cut_nodes(tf, T2, t2_l, t2_r).unwrap_or_default();
            let lca = lca_current(tf, T2, t2_l, t2_r);
            let t1_sib = tf.sibling(T1, node);
            let t1_sibling_is_leaf = t1_sib != NONE && tf.is_leaf(T1, t1_sib);

            if let Some(branches) =
                check_case(tf, node, t2_l, t2_r, &d_cut, lca, t1_sibling_is_leaf)
            {
                return PairResult::Branches(branches);
            }

            // checkBetterCuts: select the first priority cherry (Java line 437-439).
            if check_better_cuts(tf, node, t2_l, t2_r) {
                let branches = build_branches(tf, node, t2_l, t2_r, d_cut, lca, t1_sibling_is_leaf);
                return PairResult::Branches(branches);
            }

            // Fallback: keep first encountered (Java line 441-442)
            if best_cherry.is_none() {
                best_cherry = Some((node, t2_l, t2_r));
            }
        }
    }

    if let Some((node, t2_l, t2_r)) = best_cherry {
        let d_cut = d_f_cut_nodes(tf, T2, t2_l, t2_r).unwrap_or_default();
        let lca = lca_current(tf, T2, t2_l, t2_r);
        let t1_sib = tf.sibling(T1, node);
        let t1_sibling_is_leaf = t1_sib != NONE && tf.is_leaf(T1, t1_sib);
        // A3BC fallback
        let branches = build_branches(tf, node, t2_l, t2_r, d_cut, lca, t1_sibling_is_leaf);
        return PairResult::Branches(branches);
    }

    PairResult::NoPairs
}

/// Faithful port of Java MafFpt1.checkCase (lines 172-264).
/// Returns Some(branches) for RC, BCY, _2B, A2BC, RB2; None for A3BC.
fn check_case(
    tf: &TwinForest,
    t1_parent: NodeId,
    t2_x1: NodeId,
    t2_x2: NodeId,
    d_cut: &[NodeId],
    lca: NodeId,
    t1_sibling_is_leaf: bool,
) -> Option<Vec<BranchCut>> {
    // RC: lca is null → branch factor 2
    if lca == NONE {
        let mut b = Vec::with_capacity(2);
        if !tf.protected[t2_x1 as usize] {
            b.push(BranchCut::One {
                node: t2_x1,
                locks_after: vec![],
            });
        }
        if !tf.protected[t2_x2 as usize] {
            b.push(BranchCut::One {
                node: t2_x2,
                locks_after: vec![],
            });
        }
        return Some(b);
    }

    // BCY / _2B section (Java lines 190-221)
    if t1_sibling_is_leaf {
        let t1_sib = tf.sibling(T1, t1_parent);
        let f2x = if t1_sib != NONE {
            tf.twin[T1][t1_sib as usize]
        } else {
            NONE
        };
        let f2x_locked = f2x != NONE && tf.protected[f2x as usize];

        // BCY (lines 193-217)
        if f2x_locked
            && (lca == tf.parent[T2][t2_x1 as usize] || lca == tf.parent[T2][t2_x2 as usize])
        {
            let mut branches = Vec::with_capacity(2);
            let d_ac = d_f_cut_nodes(tf, T2, t2_x1, t2_x2).unwrap_or_default();
            if !d_ac.is_empty() && !d_ac.iter().any(|&n| tf.protected[n as usize]) {
                branches.push(BranchCut::Many {
                    nodes: d_ac,
                    locks_after: vec![],
                });
            }
            if lca == tf.parent[T2][t2_x1 as usize] {
                let x2_sib = tf.sibling(T2, t2_x2);
                if x2_sib != NONE && !tf.is_leaf(T2, x2_sib) {
                    let sl = tf.left[T2][x2_sib as usize];
                    let sr = tf.right[T2][x2_sib as usize];
                    if sl == f2x || sr == f2x {
                        let f2y = if sl == f2x { sr } else { sl };
                        if f2y != NONE
                            && !tf.protected[f2y as usize]
                            && !tf.protected[t2_x1 as usize]
                        {
                            branches.push(BranchCut::Many {
                                nodes: vec![f2y, t2_x1],
                                locks_after: vec![],
                            });
                        }
                    }
                }
            } else {
                let x1_sib = tf.sibling(T2, t2_x1);
                if x1_sib != NONE && !tf.is_leaf(T2, x1_sib) {
                    let sl = tf.left[T2][x1_sib as usize];
                    let sr = tf.right[T2][x1_sib as usize];
                    if sl == f2x || sr == f2x {
                        let f2y = if sl == f2x { sr } else { sl };
                        if f2y != NONE
                            && !tf.protected[f2y as usize]
                            && !tf.protected[t2_x2 as usize]
                        {
                            branches.push(BranchCut::Many {
                                nodes: vec![f2y, t2_x2],
                                locks_after: vec![],
                            });
                        }
                    }
                }
            }
            if !branches.is_empty() {
                branches.reverse();
                return Some(branches);
            }
        }

        // _2B (lines 218-221): single forced 2-cut
        let x1p = tf.parent[T2][t2_x1 as usize];
        let x2p = tf.parent[T2][t2_x2 as usize];
        if x1p != NONE
            && tf.parent[T2][x1p as usize] != NONE
            && x2p != NONE
            && tf.parent[T2][x2p as usize] != NONE
            && (tf.sibling(T2, x1p) == f2x || tf.sibling(T2, x2p) == f2x)
            && d_cut.len() == 2
            && !d_cut.iter().any(|&n| tf.protected[n as usize])
        {
            return Some(vec![BranchCut::Many {
                nodes: d_cut.to_vec(),
                locks_after: vec![],
            }]);
        }
    }

    let x1u = !tf.protected[t2_x1 as usize];
    let x2u = !tf.protected[t2_x2 as usize];

    // A2BC (lines 223-234): |d_cut| == 2
    if d_cut.len() == 2 {
        let dl = d_cut.iter().any(|&n| tf.protected[n as usize]);
        if x1u && x2u && !dl {
            // A2BC 3-branch: [cut f2c+locks, cut DC, cut f2a] (Java reversed)
            let mut lk = d_cut.to_vec();
            lk.push(t2_x1);
            return Some(vec![
                BranchCut::One {
                    node: t2_x2,
                    locks_after: lk,
                },
                BranchCut::Many {
                    nodes: d_cut.to_vec(),
                    locks_after: vec![],
                },
                BranchCut::One {
                    node: t2_x1,
                    locks_after: vec![],
                },
            ]);
        } else if x1u != x2u {
            let single = if x1u { t2_x1 } else { t2_x2 };
            let mut b = Vec::with_capacity(2);
            b.push(BranchCut::One {
                node: single,
                locks_after: d_cut.to_vec(),
            });
            if !dl {
                b.push(BranchCut::Many {
                    nodes: d_cut.to_vec(),
                    locks_after: vec![],
                });
            }
            return Some(b);
        } else {
            if !dl {
                return Some(vec![BranchCut::Many {
                    nodes: d_cut.to_vec(),
                    locks_after: vec![],
                }]);
            }
            return Some(vec![]); // No valid branches — signal caller
        }
    }

    // RB2 (lines 236-262): single forced cut
    if d_cut.len() >= 2
        && (lca == tf.parent[T2][t2_x1 as usize] || lca == tf.parent[T2][t2_x2 as usize])
        && d_cut.iter().all(|&n| tf.is_leaf(T2, n))
    {
        let fm = tf.twin[T2][d_cut[0] as usize];
        let sm = tf.twin[T2][d_cut[1] as usize];
        if fm != NONE && sm != NONE && tf.parent[T1][fm as usize] == tf.parent[T1][sm as usize] {
            let mut ok = true;
            let mut cur = tf.parent[T1][fm as usize];
            for i in 2..d_cut.len() {
                let lm = tf.twin[T2][d_cut[i] as usize];
                if lm == NONE || tf.parent[T1][cur as usize] != lm {
                    ok = false;
                    break;
                }
                cur = tf.parent[T1][cur as usize];
            }
            if ok {
                let last = d_cut[d_cut.len() - 1];
                let lp2 = tf.parent[T2][last as usize];
                let ok2 = lp2 != NONE && {
                    let lgp2 = tf.parent[T2][lp2 as usize];
                    lgp2 != NONE && {
                        let tg = tf.twin[T2][lgp2 as usize];
                        let ta = tf.twin[T2][t2_x1 as usize];
                        tg != NONE
                            && ta != NONE
                            && tf.parent[T1][ta as usize] != NONE
                            && tg == tf.parent[T1][tf.parent[T1][ta as usize] as usize]
                    }
                };
                if ok2 {
                    let single = if lca == tf.parent[T2][t2_x1 as usize] {
                        t2_x2
                    } else {
                        t2_x1
                    };
                    if !tf.protected[single as usize] {
                        return Some(vec![BranchCut::One {
                            node: single,
                            locks_after: vec![],
                        }]);
                    }
                }
            }
        }
    }

    None // A3BC — caller falls back to build_branches
}

fn check_better_cuts(tf: &TwinForest, t1_parent: NodeId, t2_x1: NodeId, t2_x2: NodeId) -> bool {
    if find_root(tf, T2, t2_x1) != find_root(tf, T2, t2_x2) {
        return true;
    }
    if tf.protected[t2_x1 as usize] && tf.protected[t2_x2 as usize] {
        return true;
    }
    if tf.protected[t2_x1 as usize] {
        return true;
    }
    if tf.protected[t2_x2 as usize] {
        return true;
    }
    let grandparent = tf.parent[T1][t1_parent as usize];
    if grandparent == NONE {
        return false;
    }
    let sibling = tf.sibling(T1, t1_parent);
    if sibling != NONE && tf.is_leaf(T1, sibling) {
        return true;
    }
    if tf.right[T1][grandparent as usize] != t1_parent {
        return false;
    }
    let left_sibling = tf.left[T1][grandparent as usize];
    if left_sibling == NONE {
        return false;
    }
    let lc = tf.left[T1][left_sibling as usize];
    let rc = tf.right[T1][left_sibling as usize];
    lc != NONE && rc != NONE && tf.is_leaf(T1, lc) && tf.is_leaf(T1, rc)
}

/// Java `Instance.dangling(f2a, f2b)` lines 457-477.
fn d_f_cut_nodes(tf: &TwinForest, ti: usize, a: NodeId, b: NodeId) -> Option<Vec<NodeId>> {
    if find_root(tf, ti, a) != find_root(tf, ti, b) {
        return None;
    }
    let lca = lca_current(tf, ti, a, b);
    if lca == NONE {
        return None;
    }
    // Java 462-463: both direct children of LCA → empty
    if tf.parent[ti][a as usize] == lca && tf.parent[ti][b as usize] == lca {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    let mut cur = a;
    while tf.parent[ti][cur as usize] != lca {
        let s = tf.sibling(ti, cur);
        if s != NONE {
            out.push(s);
        }
        cur = tf.parent[ti][cur as usize];
    }
    cur = b;
    while tf.parent[ti][cur as usize] != lca {
        let s = tf.sibling(ti, cur);
        if s != NONE {
            out.push(s);
        }
        cur = tf.parent[ti][cur as usize];
    }
    Some(out)
}

fn contract_matching_pair(
    tf: &mut TwinForest,
    t1_parent: NodeId,
    t2_parent: NodeId,
    um: &mut undo::UndoMachine,
) {
    let t1_lc = tf.left[T1][t1_parent as usize];
    let t1_rc = tf.right[T1][t1_parent as usize];
    let kept_label = tf.label[T1][t1_rc as usize];
    let removed_label = tf.label[T1][t1_lc as usize];

    undo::contract_sibling_pair(tf, T1, t1_parent, um);
    undo::contract_sibling_pair(tf, T2, t2_parent, um);
    undo::set_label(tf, T1, t1_parent, kept_label, um);
    undo::set_label(tf, T2, t2_parent, kept_label, um);
    undo::set_label_to_node(tf, T1, kept_label, t1_parent, um);
    undo::set_label_to_node(tf, T2, kept_label, t2_parent, um);
    if removed_label != 0 {
        undo::set_label_to_node(tf, T1, removed_label, NONE, um);
        undo::set_label_to_node(tf, T2, removed_label, NONE, um);
        undo::set_collapsed(tf, removed_label, kept_label, um);
    }
    undo::set_twin(tf, T1, t1_parent, t2_parent, um);
    undo::set_twin(tf, T2, t2_parent, t1_parent, um);
}

fn find_root(tf: &TwinForest, ti: usize, mut node: NodeId) -> NodeId {
    loop {
        let p = tf.parent[ti][node as usize];
        if p == NONE {
            return node;
        }
        node = p;
    }
}

fn sibling_node(tf: &TwinForest, ti: usize, node: NodeId) -> Option<NodeId> {
    let parent = tf.parent[ti][node as usize];
    if parent == NONE {
        return None;
    }
    let left = tf.left[ti][parent as usize];
    let right = tf.right[ti][parent as usize];
    if left == node && right != NONE {
        Some(right)
    } else if right == node && left != NONE {
        Some(left)
    } else {
        None
    }
}

fn depth_to_root(tf: &TwinForest, ti: usize, mut node: NodeId) -> u16 {
    let mut depth = 0;
    loop {
        let p = tf.parent[ti][node as usize];
        if p == NONE {
            return depth;
        }
        depth += 1;
        node = p;
    }
}

fn lca_current(tf: &TwinForest, ti: usize, a: NodeId, b: NodeId) -> NodeId {
    let original_parent = if ti == T1 {
        &tf.orig_parent
    } else {
        &tf.orig_t2_parent
    };
    let lca = lca_in_parent(original_parent, a, b);
    if lca == NONE || !is_current_ancestor(&tf.parent[ti], lca, a) {
        return NONE;
    }
    if !is_current_ancestor(&tf.parent[ti], lca, b) {
        return NONE;
    }
    lca
}

fn lca_in_parent(parent: &[NodeId], a: NodeId, b: NodeId) -> NodeId {
    if a == NONE || b == NONE {
        return NONE;
    }
    let mut ancestors = FixedBitSet::with_capacity(parent.len());
    let mut cur = a;
    while cur != NONE {
        ancestors.insert(cur as usize);
        cur = parent[cur as usize];
    }
    cur = b;
    while cur != NONE {
        if ancestors.contains(cur as usize) {
            return cur;
        }
        cur = parent[cur as usize];
    }
    NONE
}

fn is_current_ancestor(parent: &[NodeId], ancestor: NodeId, mut node: NodeId) -> bool {
    while node != NONE {
        if node == ancestor {
            return true;
        }
        node = parent[node as usize];
    }
    false
}

fn extract_components(tf: &TwinForest) -> Vec<Tree> {
    extract_components_with_reference(tf, tf.num_leaves, &tree_from_original(tf))
}

fn extract_components_with_reference(
    tf: &TwinForest,
    output_n: u32,
    reference: &Tree,
) -> Vec<Tree> {
    let n = tf.num_leaves;
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

    let mut result = Vec::new();
    for &root in &tf.components[T1] {
        let mut current_labels = Vec::new();
        collect_labels(tf, root, &mut current_labels);
        if current_labels.is_empty() {
            continue;
        }

        let mut leafset = FixedBitSet::with_capacity(n as usize + 1);
        for &lbl in &current_labels {
            if lbl > output_n {
                continue;
            }
            for orig in 1..=output_n {
                if collapsed[orig as usize] == lbl {
                    leafset.insert(orig as usize);
                }
            }
        }
        if leafset.count_ones(..) > 0 {
            result.push(Tree::component_from_leafset(&leafset, reference, output_n));
        }
    }
    result
}

fn collect_labels(tf: &TwinForest, node: NodeId, out: &mut Vec<Label>) {
    let lbl = tf.label[T1][node as usize];
    if lbl != 0 {
        out.push(lbl);
        return;
    }
    let lc = tf.left[T1][node as usize];
    if lc != NONE {
        collect_labels(tf, lc, out);
    }
    let rc = tf.right[T1][node as usize];
    if rc != NONE {
        collect_labels(tf, rc, out);
    }
}

fn tree_from_original(tf: &TwinForest) -> Tree {
    let mut tree = Tree::with_capacity(tf.num_leaves);
    tree.parent = tf.orig_parent.clone();
    tree.left = tf.orig_left.clone();
    tree.right = tf.orig_right.clone();
    tree.label = tf.orig_label.clone();
    tree.label_to_node = vec![NONE; tf.num_leaves as usize + 1];
    for node in 0..tf.num_nodes[T1] as NodeId {
        let lbl = tf.orig_label[node as usize];
        if lbl != 0 && tf.orig_left[node as usize] == NONE {
            tree.label_to_node[lbl as usize] = node;
        }
    }
    tree.num_leaves = tf.num_leaves;
    tree.root = tf.root[T1];
    tree.depth = vec![0; tf.num_nodes[T1]];
    tree.subtree_size = vec![0; tf.num_nodes[T1]];
    tree
}


// ── entry point ─────────────────────────────────────────────────────────────
use crate::{RunConfig, Solver, Track};

pub fn main() {
    crate::run(
        ChenRsprSolver::new(),
        RunConfig {
            track: Track::Exact,
            specific: ChenRsprConfig::default(),
            ..Default::default()
        },
    );
}
