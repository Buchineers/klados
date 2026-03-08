//! SAT-based MAF solver with CEGAR optimization.
//!
//! Key optimizations:
//! - Precomputed indices for O(1) lookups
//! - CEGAR for H4 (incompatible triples) - adds only violated constraints
//! - H3 path consistency with fast index access

use fixedbitset::FixedBitSet;
use klados_core::tree::{Label, NodeId, Tree, NONE};
use klados_core::{Instance, SolverStats};
use rustsat::encodings::card::{BoundUpper, Totalizer};
use rustsat::instances::{BasicVarManager, ManageVars};
use rustsat::solvers::{PhaseLit, Solve, SolveIncremental, SolverResult};
use rustsat::types::{Clause, Lit, TernaryVal, Var};
use rustsat_cadical::CaDiCaL;

use crate::kernelize::{self, KernelizeConfig};
use crate::lower_bound::{
    cherry_reduce_ub, greedy_multi_tree_partition, greedy_multi_tree_ub,
    greedy_multi_tree_ub_seeded, red_blue_approx,
};
use crate::ExactSolver;

// ═══════════════════════════════════════════════════════════════
// Timing / profiling
// ═══════════════════════════════════════════════════════════════

#[derive(Default)]
struct SolveProfile {
    n: usize,
    k: usize,
    m: usize,
    n_reduced: usize,
    cluster_splits: usize,
    num_vars: usize,
    h1_clauses: usize,
    h2_clauses: usize,
    h3_clauses: usize,
    h4_clauses: usize,
    h5_clauses: usize,
    sym_clauses: usize,
    sibling_clauses: usize,
    singleton_clauses: usize,
    encode_ms: f64,
    solve_ms: f64,
    cegar_ms: f64,
    sat_calls: usize,
    cegar_violations: usize,
    bounds_computed: (usize, usize), // (lb, ub)
    optimal_k: usize,
}

impl SolveProfile {
    fn total_clauses(&self) -> usize {
        self.h1_clauses
            + self.h2_clauses
            + self.h3_clauses
            + self.h4_clauses
            + self.h5_clauses
            + self.sym_clauses
            + self.sibling_clauses
            + self.singleton_clauses
    }

    fn report(&self) {
        eprintln!(
            "[profile] n={} n'={} k={} m={} splits={} vars={} clauses={} \
             (H1={} H2={} H3={} H4={} H5={} sym={} sib={} sing={}) \
             encode={:.1}ms solve={:.1}ms cegar={:.1}ms \
             sat_calls={} violations={} bounds=[{},{}] opt={}",
            self.n,
            self.n_reduced,
            self.k,
            self.m,
            self.cluster_splits,
            self.num_vars,
            self.total_clauses(),
            self.h1_clauses,
            self.h2_clauses,
            self.h3_clauses,
            self.h4_clauses,
            self.h5_clauses,
            self.sym_clauses,
            self.sibling_clauses,
            self.singleton_clauses,
            self.encode_ms,
            self.solve_ms,
            self.cegar_ms,
            self.sat_calls,
            self.cegar_violations,
            self.bounds_computed.0,
            self.bounds_computed.1,
            self.optimal_k,
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// Cluster Decomposition — split problem on common clusters
// ═══════════════════════════════════════════════════════════════

/// A cluster is a set of leaf labels that form a clade in a tree
#[derive(Clone, Debug)]
struct Cluster {
    leaves: FixedBitSet,
}

/// Find all non-trivial clusters (leaf sets below internal nodes) in a tree
fn find_tree_clusters(tree: &Tree, num_leaves: usize) -> Vec<Cluster> {
    let mut leaf_sets = vec![FixedBitSet::with_capacity(num_leaves + 1); tree.num_nodes()];
    let mut clusters = Vec::new();

    // Compute leaf sets bottom-up
    for node in tree.post_order() {
        if tree.is_leaf(node) {
            let lbl = tree.label[node as usize];
            if lbl > 0 && (lbl as usize) <= num_leaves {
                leaf_sets[node as usize].insert(lbl as usize);
            }
        } else if let Some((left, right)) = tree.children(node) {
            let mut set = leaf_sets[left as usize].clone();
            set.union_with(&leaf_sets[right as usize]);
            leaf_sets[node as usize] = set;
        }
    }

    // Extract non-trivial clusters (size >= 2 and < n)
    for node in 0..tree.num_nodes() as NodeId {
        if tree.is_leaf(node) {
            continue;
        }
        let leaf_count = leaf_sets[node as usize].count_ones(..);
        if leaf_count >= 2 && leaf_count < num_leaves {
            clusters.push(Cluster {
                leaves: leaf_sets[node as usize].clone(),
            });
        }
    }

    clusters
}

/// Find clusters that appear in ALL trees (regardless of internal topology)
/// Uses exact leaf-set comparison (no hash collisions)
fn find_common_clusters(trees: &[Tree], num_leaves: u32) -> Vec<Cluster> {
    if trees.len() < 2 || num_leaves < 6 {
        return Vec::new();
    }

    let n = num_leaves as usize;

    // Build clusters from reference tree
    let ref_clusters = find_tree_clusters(&trees[0], n);

    // Build lookup sets for other trees using exact leaf-set keys
    let other_sets: Vec<fxhash::FxHashSet<Vec<usize>>> = trees[1..]
        .iter()
        .map(|tree| {
            let clusters = find_tree_clusters(tree, n);
            clusters
                .into_iter()
                .map(|c| c.leaves.ones().collect::<Vec<usize>>())
                .collect()
        })
        .collect();

    // Keep only ref clusters present in ALL other trees
    let mut common: Vec<Cluster> = ref_clusters
        .into_iter()
        .filter(|c| {
            let key: Vec<usize> = c.leaves.ones().collect();
            other_sets.iter().all(|s| s.contains(&key))
        })
        .collect();

    // Sort by distance from n/2 (best splits first)
    let target = n / 2;
    common.sort_by_key(|c| {
        let sz = c.leaves.count_ones(..);
        (sz as isize - target as isize).unsigned_abs()
    });

    common
}

/// Find the best cluster to split on (closest to n/2 in size)
fn find_best_split_cluster(clusters: &[Cluster], num_leaves: usize) -> Option<&Cluster> {
    let target = num_leaves / 2;

    clusters
        .iter()
        .filter(|c| {
            let sz = c.leaves.count_ones(..);
            // Must leave at least 3 leaves on each side
            sz >= 3 && (num_leaves - sz) >= 3
        })
        .min_by_key(|c| {
            let sz = c.leaves.count_ones(..);
            (sz as isize - target as isize).abs() as usize
        })
}

/// Restrict an instance to only the leaves in the given set
fn restrict_instance(instance: &Instance, keep: &FixedBitSet) -> (Instance, Vec<u32>) {
    let kept_labels: Vec<u32> = keep.ones().map(|i| i as u32).collect();
    let new_n = kept_labels.len() as u32;

    // Build label mapping: old_label -> new_label
    let mut label_map = vec![0u32; instance.num_leaves as usize + 1];
    let mut reverse_map = vec![0u32; new_n as usize + 1];
    for (new_idx, &old_lbl) in kept_labels.iter().enumerate() {
        let new_lbl = (new_idx + 1) as u32;
        label_map[old_lbl as usize] = new_lbl;
        reverse_map[new_lbl as usize] = old_lbl;
    }

    // Create restricted trees
    let trees: Vec<Tree> = instance
        .trees
        .iter()
        .map(|t| {
            let pruned = t.prune_to_leafset(keep);
            pruned.relabel(&label_map, new_n)
        })
        .collect();

    (Instance::new(trees, new_n), reverse_map)
}

/// Collapse a cluster into a single representative leaf
fn collapse_cluster(
    instance: &Instance,
    cluster: &FixedBitSet,
    representative: u32,
) -> (Instance, Vec<u32>) {
    let n = instance.num_leaves as usize;

    // Keep all leaves NOT in cluster, plus the representative
    let mut keep = FixedBitSet::with_capacity(n + 1);
    for lbl in 1..=n as u32 {
        if !cluster.contains(lbl as usize) || lbl == representative {
            keep.insert(lbl as usize);
        }
    }

    restrict_instance(instance, &keep)
}

/// Combine cluster and rest solutions.
///
/// Cluster decomposition theorem: MAF(T1..Tm) = cluster_MAF + rest_MAF - 1.
///
/// In the rest_MAF, rep (the ghost leaf standing for the whole cluster) will
/// typically be a singleton component {rep}.  If it is, we substitute the
/// cluster_solution for that singleton.
///
/// If rep is grouped with other leaves {rep, X1, …, Xm} in the rest solution:
///   • cluster_solution has exactly 1 component → we can merge: replace the
///     component with {cluster_leaves ∪ X1 … Xm}.
///   • cluster_solution has > 1 component → cannot merge consistently; return
///     None so the caller can fall back to monolithic solving.  The theorem
///     guarantees that at the true optimum this case never arises, so returning
///     None is only a safety valve against a solver returning a sub-optimal
///     (or equal-cost but wrong-structure) solution.
fn combine_solutions(
    cluster_solution: &[Tree],
    rest_solution: &[Tree],
    rep: u32,
    cluster_reverse: &[u32],
    rest_reverse: &[u32],
    ref_tree: &Tree,
    num_leaves: u32,
) -> Option<Vec<Tree>> {
    let n = num_leaves as usize;
    let mut combined = Vec::new();

    // Find rep's label in the rest sub-instance label space.
    let rep_subproblem_label: u32 = (1..rest_reverse.len() as u32)
        .find(|&i| rest_reverse[i as usize] == rep)
        .expect("rep must be present in rest instance");

    let mut found_rep = false;

    for tree in rest_solution {
        let leaves: Vec<Label> = tree.leaves().collect();
        if leaves.contains(&rep_subproblem_label) {
            found_rep = true;
            if leaves.len() == 1 {
                // Singleton {rep} — substitute with all cluster components.
                for ct in cluster_solution {
                    let mut ls = FixedBitSet::with_capacity(n + 1);
                    for l in ct.leaves() {
                        let orig = cluster_reverse[l as usize];
                        if orig > 0 && (orig as usize) <= n {
                            ls.insert(orig as usize);
                        }
                    }
                    if ls.count_ones(..) > 0 {
                        combined.push(ref_tree.prune_to_leafset(&ls));
                    }
                }
            } else {
                // Rep is grouped with other rest leaves {rep, X1, …, Xm}.
                // Only valid when cluster has exactly 1 component (merge everything).
                if cluster_solution.len() != 1 {
                    return None;
                }
                let mut ls = FixedBitSet::with_capacity(n + 1);
                // Add cluster leaves.
                for ct in cluster_solution {
                    for l in ct.leaves() {
                        let orig = cluster_reverse[l as usize];
                        if orig > 0 && (orig as usize) <= n {
                            ls.insert(orig as usize);
                        }
                    }
                }
                // Add rest leaves (excluding rep).
                for &l in &leaves {
                    if l == rep_subproblem_label {
                        continue;
                    }
                    let orig = rest_reverse[l as usize];
                    if orig > 0 && (orig as usize) <= n {
                        ls.insert(orig as usize);
                    }
                }
                if ls.count_ones(..) > 0 {
                    combined.push(ref_tree.prune_to_leafset(&ls));
                }
            }
        } else {
            // Component doesn't contain rep — remap labels directly.
            let mut ls = FixedBitSet::with_capacity(n + 1);
            for &l in &leaves {
                let orig = rest_reverse[l as usize];
                if orig > 0 && (orig as usize) <= n {
                    ls.insert(orig as usize);
                }
            }
            if ls.count_ones(..) > 0 {
                combined.push(ref_tree.prune_to_leafset(&ls));
            }
        }
    }

    if !found_rep {
        // Rep not found in rest solution — shouldn't happen with a valid MAF.
        return None;
    }

    Some(combined)
}

pub struct MafSatSolver {
    stats: SolverStats,
    max_leaves: u32,
}

impl Default for MafSatSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl MafSatSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
            max_leaves: 500,
        }
    }
}

impl ExactSolver for MafSatSolver {
    fn name(&self) -> &'static str {
        "maf-sat"
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        if instance.num_leaves > self.max_leaves {
            eprintln!(
                "maf-sat: instance has {} leaves, exceeding limit of {}",
                instance.num_leaves, self.max_leaves
            );
            return None;
        }
        if instance.trees.is_empty() {
            return None;
        }
        if instance.num_trees() == 1 {
            return Some(instance.trees.clone());
        }
        if instance.num_leaves <= 1 {
            return Some(instance.trees[0..1].to_vec());
        }
        solve_sat(instance, &mut self.stats)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

// ═══════════════════════════════════════════════════════════════
// Precomputed data structures
// ═══════════════════════════════════════════════════════════════

struct TreePrecomp {
    internal_idx: Vec<usize>,
    num_internal: usize,
}

impl TreePrecomp {
    fn build(tree: &Tree) -> Self {
        let num_nodes = tree.num_nodes();
        let mut internal_idx = vec![usize::MAX; num_nodes];
        let mut count = 0;
        for v in 0..num_nodes as NodeId {
            if !tree.is_leaf(v) {
                internal_idx[v as usize] = count;
                count += 1;
            }
        }
        Self {
            internal_idx,
            num_internal: count,
        }
    }

    #[inline(always)]
    fn idx(&self, node: NodeId) -> usize {
        self.internal_idx[node as usize]
    }
}

struct NcaData {
    depths: Vec<Vec<Vec<u16>>>,
    m: usize,
}

impl NcaData {
    fn build(instance: &Instance, n: usize) -> Self {
        let m = instance.num_trees();
        let depths: Vec<Vec<Vec<u16>>> = (0..m)
            .map(|q| {
                let tree = &instance.trees[q];
                let mut d = vec![vec![0u16; n]; n];
                for a in 0..n {
                    let na = tree.node_by_label((a + 1) as Label);
                    for b in (a + 1)..n {
                        let nb = tree.node_by_label((b + 1) as Label);
                        let nca = tree.nearest_common_ancestor(na, nb);
                        let depth = tree.depth[nca as usize];
                        d[a][b] = depth;
                        d[b][a] = depth;
                    }
                }
                d
            })
            .collect();
        Self { depths, m }
    }

    #[inline]
    fn is_incompatible(&self, a: usize, b: usize, c: usize) -> bool {
        let mut first_odd: u8 = u8::MAX;
        for q in 0..self.m {
            let d_ab = self.depths[q][a][b];
            let d_ac = self.depths[q][a][c];
            let d_bc = self.depths[q][b][c];

            let odd = if d_ab > d_ac && d_ab > d_bc {
                2
            } else if d_ac > d_ab && d_ac > d_bc {
                1
            } else {
                0
            };

            if first_odd == u8::MAX {
                first_odd = odd;
            } else if odd != first_odd {
                return true;
            }
        }
        false
    }
}

// ═══════════════════════════════════════════════════════════════
// Encoding with w variables for H3, but no H2/t (not needed)
// ═══════════════════════════════════════════════════════════════

struct MafEncoding {
    l: Vec<Vec<Var>>,      // l[i][j]: leaf j in component i
    u: Vec<Var>,           // u[i]: component i used
    w: Vec<Vec<Vec<Var>>>, // w[q][i][idx]: internal node in component
    s: Vec<Vec<Var>>,      // ladder aux for H1b
    t: Vec<Vec<Vec<Var>>>, // ladder aux for H2
    // Phase 1: Local propagation variables (only used for large instances)
    d: Option<Vec<Vec<Vec<Var>>>>, // d[q][i][idx]: subtree has leaf of component i
    e: Option<Vec<Vec<Vec<Var>>>>, // e[q][i][idx]: leaf of component i exists outside
    n: usize,
    k: usize,
    m: usize,
}

fn create_variables(
    vm: &mut BasicVarManager,
    n: usize,
    k: usize,
    m: usize,
    precomps: &[TreePrecomp],
) -> MafEncoding {
    let alloc = |vm: &mut BasicVarManager, count: usize| -> Vec<Var> {
        (0..count).map(|_| vm.new_var()).collect()
    };

    let l: Vec<Vec<Var>> = (0..k).map(|_| alloc(vm, n)).collect();
    let u: Vec<Var> = alloc(vm, k);
    let s: Vec<Vec<Var>> = (0..k).map(|_| alloc(vm, n)).collect();

    let w: Vec<Vec<Vec<Var>>> = (0..m)
        .map(|q| {
            let ni = precomps[q].num_internal;
            (0..k).map(|_| alloc(vm, ni)).collect()
        })
        .collect();

    let t: Vec<Vec<Vec<Var>>> = (0..m)
        .map(|q| {
            let ni = precomps[q].num_internal;
            (0..k).map(|_| alloc(vm, ni)).collect()
        })
        .collect();

    MafEncoding {
        l,
        u,
        w,
        s,
        t,
        d: None,
        e: None,
        n,
        k,
        m,
    }
}

/// Tree navigation data for local propagation encoding.
/// Used by Phase 1 adaptive H3 encoding.
struct TreeNav {
    /// For each internal index: (left_child_node, right_child_node)
    children: Vec<(NodeId, NodeId)>,
    /// For each internal index: parent node (NONE for root)
    parent: Vec<NodeId>,
    /// For each internal index: sibling node (NONE for root)
    sibling: Vec<NodeId>,
    /// Post-order of internal indices
    post_order: Vec<usize>,
    /// Pre-order of internal indices
    pre_order: Vec<usize>,
    /// Root internal index
    root_idx: usize,
    /// For each node: Some(leaf_index) if leaf, None if internal
    leaf_idx: Vec<Option<usize>>,
    /// Is this node a leaf?
    is_leaf: Vec<bool>,
}

impl TreeNav {
    fn build(tree: &Tree, pc: &TreePrecomp, n: usize) -> Self {
        let num_nodes = tree.num_nodes();
        let ni = pc.num_internal;

        let mut children = vec![(NONE, NONE); ni];
        let mut parent = vec![NONE; ni];
        let mut sibling = vec![NONE; ni];
        let mut leaf_idx = vec![None; num_nodes];
        let mut is_leaf = vec![false; num_nodes];

        for v in 0..num_nodes as NodeId {
            if tree.is_leaf(v) {
                is_leaf[v as usize] = true;
                let lbl = tree.label[v as usize];
                if lbl > 0 && (lbl as usize) <= n {
                    leaf_idx[v as usize] = Some((lbl - 1) as usize);
                }
            } else {
                let ii = pc.idx(v);
                if let Some((l, r)) = tree.children(v) {
                    children[ii] = (l, r);
                }
                let p = tree.parent[v as usize];
                parent[ii] = p;
                if p != NONE {
                    let sib = if tree.left[p as usize] == v {
                        tree.right[p as usize]
                    } else {
                        tree.left[p as usize]
                    };
                    sibling[ii] = sib;
                }
            }
        }

        let post_order: Vec<usize> = tree
            .post_order()
            .filter(|v| !tree.is_leaf(*v))
            .map(|v| pc.idx(v))
            .collect();
        let pre_order: Vec<usize> = tree
            .pre_order()
            .filter(|v| !tree.is_leaf(*v))
            .map(|v| pc.idx(v))
            .collect();

        Self {
            children,
            parent,
            sibling,
            post_order,
            pre_order,
            root_idx: pc.idx(tree.root),
            leaf_idx,
            is_leaf,
        }
    }

}

fn add_clause(solver: &mut CaDiCaL, lits: &[Lit]) {
    solver.add_clause(Clause::from(lits)).unwrap();
}

/// Encode H3 using local propagation.
/// This is more efficient for large instances with many trees.
/// Requires d and e variables to be allocated in enc.
fn add_h3_local(
    solver: &mut CaDiCaL,
    instance: &Instance,
    precomps: &[TreePrecomp],
    navs: &[TreeNav],
    enc: &MafEncoding,
    profile: &mut SolveProfile,
) {
    let n = enc.n;
    let k = enc.k;
    let m = enc.m;
    let mut count = 0usize;

    // Get d and e variables (should be Some when this is called)
    let d_vars = enc.d.as_ref().expect("d variables not allocated");
    let e_vars = enc.e.as_ref().expect("e variables not allocated");

    for q in 0..m {
        let tree = &instance.trees[q];
        let pc = &precomps[q];
        let nav = &navs[q];

        for i in 0..k {
            // Bottom-up: d propagation
            // d[v] = OR(d[left], d[right])
            // Equivalent to: d[left] → d[v], d[right] → d[v], d[v] → (d[left] ∨ d[right])
            for &ii in &nav.post_order {
                let (c1, c2) = nav.children[ii];
                let d_ii = d_vars[q][i][ii];

                // Get d-literals for children
                let dc1 = if nav.is_leaf[c1 as usize] {
                    let j = nav.leaf_idx[c1 as usize].unwrap_or(0);
                    enc.l[i][j]
                } else {
                    d_vars[q][i][pc.idx(c1)]
                };
                let dc2 = if nav.is_leaf[c2 as usize] {
                    let j = nav.leaf_idx[c2 as usize].unwrap_or(0);
                    enc.l[i][j]
                } else {
                    d_vars[q][i][pc.idx(c2)]
                };

                // d[c1] → d[v]: ¬d[c1] ∨ d[v]
                add_clause(solver, &[dc1.neg_lit(), d_ii.pos_lit()]);
                // d[c2] → d[v]: ¬d[c2] ∨ d[v]
                add_clause(solver, &[dc2.neg_lit(), d_ii.pos_lit()]);
                // d[v] → d[c1] ∨ d[c2]: ¬d[v] ∨ d[c1] ∨ d[c2]
                add_clause(solver, &[d_ii.neg_lit(), dc1.pos_lit(), dc2.pos_lit()]);
                count += 3;
            }

            // Top-down: e propagation
            // e[v] = OR(e[parent], d[sibling])
            // Root has e[root] = false
            for &ii in &nav.pre_order {
                let e_ii = e_vars[q][i][ii];

                if ii == nav.root_idx {
                    // Root: e[root] = false
                    add_clause(solver, &[e_ii.neg_lit()]);
                    count += 1;
                } else {
                    let p = nav.parent[ii];
                    let sib = nav.sibling[ii];
                    let p_ii = pc.idx(p);
                    let e_p = e_vars[q][i][p_ii];

                    // Get d[sibling]
                    let d_sib = if nav.is_leaf[sib as usize] {
                        let j = nav.leaf_idx[sib as usize].unwrap_or(0);
                        enc.l[i][j]
                    } else {
                        d_vars[q][i][pc.idx(sib)]
                    };

                    // e[parent] → e[v]: ¬e[p] ∨ e[v]
                    add_clause(solver, &[e_p.neg_lit(), e_ii.pos_lit()]);
                    // d[sibling] → e[v]: ¬d[sib] ∨ e[v]
                    add_clause(solver, &[d_sib.neg_lit(), e_ii.pos_lit()]);
                    // e[v] → e[parent] ∨ d[sibling]: ¬e[v] ∨ e[p] ∨ d[sib]
                    add_clause(solver, &[e_ii.neg_lit(), e_p.pos_lit(), d_sib.pos_lit()]);
                    count += 3;
                }
            }

            // w definition: w[v] ↔ at_least_2(d[c1], d[c2], e[v])
            // This means w[v] is true iff at least 2 of {d[c1], d[c2], e[v]} are true
            // For 3 variables, at_least_2(a,b,c) = (a∧b) ∨ (a∧c) ∨ (b∧c)
            // In CNF: (¬a∨¬b∨w) ∧ (¬a∨¬c∨w) ∧ (¬b∨¬c∨w) ∧ (¬w∨a∨b) ∧ (¬w∨a∨c) ∧ (¬w∨b∨c)
            for &ii in &nav.post_order {
                let (c1, c2) = nav.children[ii];
                let w_ii = enc.w[q][i][ii];
                let e_ii = e_vars[q][i][ii];

                // Get d-literals for children
                let dc1 = if nav.is_leaf[c1 as usize] {
                    let j = nav.leaf_idx[c1 as usize].unwrap_or(0);
                    enc.l[i][j]
                } else {
                    d_vars[q][i][pc.idx(c1)]
                };
                let dc2 = if nav.is_leaf[c2 as usize] {
                    let j = nav.leaf_idx[c2 as usize].unwrap_or(0);
                    enc.l[i][j]
                } else {
                    d_vars[q][i][pc.idx(c2)]
                };

                // Forward: any pair → w
                // (¬d[c1] ∨ ¬d[c2] ∨ w)
                add_clause(solver, &[dc1.neg_lit(), dc2.neg_lit(), w_ii.pos_lit()]);
                // (¬d[c1] ∨ ¬e[v] ∨ w)
                add_clause(solver, &[dc1.neg_lit(), e_ii.neg_lit(), w_ii.pos_lit()]);
                // (¬d[c2] ∨ ¬e[v] ∨ w)
                add_clause(solver, &[dc2.neg_lit(), e_ii.neg_lit(), w_ii.pos_lit()]);
                // Backward: w → at least 2
                // (¬w ∨ d[c1] ∨ d[c2])
                add_clause(solver, &[w_ii.neg_lit(), dc1.pos_lit(), dc2.pos_lit()]);
                // (¬w ∨ d[c1] ∨ e[v])
                add_clause(solver, &[w_ii.neg_lit(), dc1.pos_lit(), e_ii.pos_lit()]);
                // (¬w ∨ d[c2] ∨ e[v])
                add_clause(solver, &[w_ii.neg_lit(), dc2.pos_lit(), e_ii.pos_lit()]);
                count += 6;
            }
        }
    }
    profile.h3_clauses = count;
}

/// Encode base clauses + H3. H4 added via CEGAR.
/// Uses local propagation H3 encoding for better clause efficiency.
fn add_structural_clauses(
    solver: &mut CaDiCaL,
    enc: &MafEncoding,
    profile: &mut SolveProfile,
    instance: &Instance,
    precomps: &[TreePrecomp],
    navs: &[TreeNav],
) {
    let n = enc.n;
    let k = enc.k;
    let m = enc.m;

    // H1a: every leaf in at least one component.
    // By O2+O3+H1, l[i][j]=FALSE for i>j (induction), so only components 0..=j
    // can hold leaf j. The clause need only span those components.
    for j in 0..n {
        let hi = j.min(k - 1);
        let clause: Vec<Lit> = (0..=hi).map(|i| enc.l[i][j].pos_lit()).collect();
        add_clause(solver, &clause);
    }
    profile.h1_clauses += n;

    // H1b: at most one component per leaf (ladder).
    // Same observation: ladder only needs to run up to component min(j, k-1).
    for j in 0..n {
        let jk = j.min(k - 1); // max component that can hold leaf j
        for i in 0..jk {
            add_clause(solver, &[enc.l[i][j].neg_lit(), enc.s[i][j].pos_lit()]);
        }
        for i in 1..jk {
            add_clause(solver, &[enc.s[i - 1][j].neg_lit(), enc.s[i][j].pos_lit()]);
        }
        for i in 1..=jk {
            add_clause(solver, &[enc.s[i - 1][j].neg_lit(), enc.l[i][j].neg_lit()]);
        }
    }
    profile.h1_clauses += n * (3 * k - 2);

    // H2: at most one component per internal node (ladder)
    for q in 0..m {
        let ni = enc.w[q][0].len();
        for idx in 0..ni {
            for i in 0..k - 1 {
                add_clause(
                    solver,
                    &[enc.w[q][i][idx].neg_lit(), enc.t[q][i][idx].pos_lit()],
                );
            }
            for i in 1..k - 1 {
                add_clause(
                    solver,
                    &[enc.t[q][i - 1][idx].neg_lit(), enc.t[q][i][idx].pos_lit()],
                );
            }
            for i in 1..k {
                add_clause(
                    solver,
                    &[enc.t[q][i - 1][idx].neg_lit(), enc.w[q][i][idx].neg_lit()],
                );
            }
        }
        profile.h2_clauses += ni * (3 * k - 2);
    }

    // H3: local propagation encoding (fewer clauses than path-based)
    add_h3_local(solver, instance, precomps, navs, enc, profile);

    // H5: usage tracking.
    // l[i][j]=FALSE for j<i (O2+O3+H1), so only need j>=i.
    let mut h5_count = 0;
    for i in 0..k {
        for j in i..n {
            add_clause(solver, &[enc.l[i][j].neg_lit(), enc.u[i].pos_lit()]);
            h5_count += 1;
        }
    }
    profile.h5_clauses = h5_count;

    // O1: ordered usage
    for i in 0..k - 1 {
        add_clause(solver, &[enc.u[i + 1].neg_lit(), enc.u[i].pos_lit()]);
    }
    // O2: leaf 0 in component 0
    add_clause(solver, &[enc.l[0][0].pos_lit()]);
    // O3: lex ordering
    let mut sym = k; // k-1 from O1, 1 from O2
    for i in 0..k - 1 {
        for j in 0..n {
            let mut lits = vec![enc.l[i + 1][j].neg_lit()];
            for jp in 0..j {
                lits.push(enc.l[i][jp].pos_lit());
            }
            add_clause(solver, &lits);
            sym += 1;
        }
    }
    profile.sym_clauses = sym;
}

fn seed_sibling_constraints(
    solver: &mut CaDiCaL,
    instance: &Instance,
    enc: &MafEncoding,
    profile: &mut SolveProfile,
    ghost_leaves: &[usize],
) {
    let ref_tree = &instance.trees[0];
    let k = enc.k;
    let mut count = 0;

    for node in ref_tree.post_order() {
        if let Some((left, right)) = ref_tree.children(node) {
            if ref_tree.is_leaf(left) && ref_tree.is_leaf(right) {
                let la = ref_tree.label[left as usize];
                let lb = ref_tree.label[right as usize];
                if la == 0 || lb == 0 {
                    continue;
                }

                let ja = (la - 1) as usize;
                let jb = (lb - 1) as usize;

                // Skip sibling constraints involving ghost leaves (cluster representatives).
                // A ghost leaf is an artificial stand-in for a whole cluster sub-instance;
                // it has no genuine cherry relationship in the original instance trees, so
                // forcing it to be co-component with its structural cherry partner is wrong
                // and will make the singleton assumption UNSAT.
                if ghost_leaves.contains(&ja) || ghost_leaves.contains(&jb) {
                    continue;
                }

                let is_common = instance.trees[1..].iter().all(|t| {
                    let na = t.node_by_label(la);
                    let nb = t.node_by_label(lb);
                    let pa = t.parent[na as usize];
                    let pb = t.parent[nb as usize];
                    pa != NONE && pa == pb
                });

                if is_common {
                    // l[i][j]=FALSE for i>j, so only need i<=min(ja,jb).
                    let hi = ja.min(jb).min(k - 1);
                    for i in 0..=hi {
                        add_clause(solver, &[enc.l[i][ja].neg_lit(), enc.l[i][jb].pos_lit()]);
                        add_clause(solver, &[enc.l[i][jb].neg_lit(), enc.l[i][ja].pos_lit()]);
                        count += 2;
                    }
                }
            }
        }
    }
    profile.sibling_clauses = count;
}

// ═══════════════════════════════════════════════════════════════
// CEGAR for H4 (incompatible triples)
// ═══════════════════════════════════════════════════════════════
// CEGAR for H4 (incompatible triples)
// ═══════════════════════════════════════════════════════════════

fn add_violated_triples(
    solver: &mut CaDiCaL,
    enc: &MafEncoding,
    components: &[Vec<usize>],
    nca_data: &NcaData,
    added_triples: &mut fxhash::FxHashSet<(usize, usize, usize)>,
) -> (usize, usize) {
    let k = enc.k;
    let mut violations = 0;
    let mut clauses = 0;

    for comp in components {
        if comp.len() < 3 {
            continue;
        }

        for ii in 0..comp.len() {
            for jj in (ii + 1)..comp.len() {
                for kk in (jj + 1)..comp.len() {
                    let a = comp[ii];
                    let b = comp[jj];
                    let c = comp[kk];

                    if nca_data.is_incompatible(a, b, c) {
                        let key = (a, b, c);
                        if added_triples.contains(&key) {
                            continue; // Already added, skip duplicate
                        }
                        added_triples.insert(key);

                        // l[i][x]=FALSE for i>x, so clause is trivially satisfied
                        // for i > min(a,b,c). Only add for i<=min(a,b,c).
                        let min_leaf = a.min(b).min(c);
                        let hi = min_leaf.min(k - 1);
                        for i in 0..=hi {
                            add_clause(
                                solver,
                                &[
                                    enc.l[i][a].neg_lit(),
                                    enc.l[i][b].neg_lit(),
                                    enc.l[i][c].neg_lit(),
                                ],
                            );
                        }
                        violations += 1;
                        clauses += hi + 1;
                    }
                }
            }
        }
    }

    (violations, clauses)
}

// ═══════════════════════════════════════════════════════════════
// Solver core
// ═══════════════════════════════════════════════════════════════

fn sat_solve_maf(
    instance: &Instance,
    k_max: usize,
    lb_components: usize,
    profile: &mut SolveProfile,
    preferred_singletons: &[usize],
) -> Option<Vec<Tree>> {
    let n = instance.num_leaves as usize;
    let k = k_max;
    let m = instance.num_trees();

    profile.n_reduced = n;
    profile.k = k;
    profile.m = m;

    let precomps: Vec<TreePrecomp> = instance
        .trees
        .iter()
        .map(|t| TreePrecomp::build(t))
        .collect();

    // Build TreeNav data for local propagation
    let t_nav = std::time::Instant::now();
    let navs: Vec<TreeNav> = (0..m)
        .map(|q| TreeNav::build(&instance.trees[q], &precomps[q], n))
        .collect();
    let nav_ms = t_nav.elapsed().as_secs_f64() * 1000.0;

    let nca_data = NcaData::build(instance, n);

    let mut solver = CaDiCaL::default();
    let mut vm = BasicVarManager::default();

    // Create variables, including d/e for local propagation
    let mut enc = create_variables(&mut vm, n, k, m, &precomps);
    let t_vars = std::time::Instant::now();
    let d: Vec<Vec<Vec<Var>>> = (0..m)
        .map(|q| {
            let ni = precomps[q].num_internal;
            (0..k)
                .map(|_| (0..ni).map(|_| vm.new_var()).collect())
                .collect()
        })
        .collect();
    let e: Vec<Vec<Vec<Var>>> = (0..m)
        .map(|q| {
            let ni = precomps[q].num_internal;
            (0..k)
                .map(|_| (0..ni).map(|_| vm.new_var()).collect())
                .collect()
        })
        .collect();
    enc.d = Some(d);
    enc.e = Some(e);
    let var_ms = t_vars.elapsed().as_secs_f64() * 1000.0;

    let t_enc = std::time::Instant::now();
    add_structural_clauses(&mut solver, &enc, profile, instance, &precomps, &navs);
    seed_sibling_constraints(&mut solver, instance, &enc, profile, preferred_singletons);

    // Hard singleton constraints for ghost leaves (cluster representatives).
    // For every ghost p: ~l[i][p] \/ ~l[i][j] for all i, all j != p.
    // This is a hard structural fact: a ghost leaf must be completely alone in its
    // component so that combine_solutions can substitute the cluster sub-solution.
    // Hard binary clauses (rather than a soft indicator variable + retry) ensure:
    //   (a) The solver finds the TRUE optimal k under the isolation requirement
    //       instead of a spuriously low k that groups the ghost with other leaves.
    //   (b) No O2 collision: the ghost is free to land in any component index,
    //       while O2 still pins leaf-index-0 to component 0.
    let mut singleton_clauses = 0usize;
    for &p in preferred_singletons {
        for i in 0..k {
            for q in 0..n {
                if q != p {
                    add_clause(&mut solver, &[enc.l[i][p].neg_lit(), enc.l[i][q].neg_lit()]);
                    singleton_clauses += 1;
                }
            }
        }
    }
    profile.singleton_clauses = singleton_clauses;

    profile.encode_ms = t_enc.elapsed().as_secs_f64() * 1000.0 + nav_ms + var_ms;
    profile.num_vars = vm.n_used() as usize;

    // Phase hints: bias the solver toward the greedy upper-bound partition.
    // CaDiCaL will override these via VSIDS if they are wrong, so the downside
    // is negligible while the upside is faster SAT proofs at the UB bound.
    {
        let (greedy_n_comps, greedy_partition) =
            greedy_multi_tree_partition(&instance.trees, 0, 0);
        if greedy_n_comps <= k {
            for j in 0..n {
                let comp_idx = greedy_partition[j];
                for i in 0..k {
                    let lit = if i == comp_idx {
                        enc.l[i][j].pos_lit()
                    } else {
                        enc.l[i][j].neg_lit()
                    };
                    let _ = solver.phase_lit(lit);
                }
            }
        }
    }

    let u_lits: Vec<Lit> = enc.u.iter().map(|v| v.pos_lit()).collect();
    let mut totalizer = Totalizer::default();
    for lit in u_lits {
        totalizer.extend([lit]);
    }

    let mut best_components: Option<Vec<Vec<usize>>> = None;

    // Pre-encode the full range once so enforce_ub can assume any bound in [lb_components, k].
    totalizer
        .encode_ub(lb_components..=k, &mut solver, &mut vm)
        .unwrap();

    // Track added H4 triples to avoid re-adding duplicates across CEGAR iterations.
    let mut added_triples: fxhash::FxHashSet<(usize, usize, usize)> = fxhash::FxHashSet::default();

    // Upfront H4: for small n add all incompatible triples before the CEGAR loop.
    // This gives the SAT solver full H4 propagation from the first call, and reduces
    // bound levels to exactly 1 SAT call each (no inner CEGAR iterations).
    // For larger n the clause count becomes prohibitive and lazy CEGAR is better.
    const UPFRONT_TRIPLE_THRESHOLD: usize = 40;
    if n <= UPFRONT_TRIPLE_THRESHOLD {
        for a in 0..n {
            for b in (a + 1)..n {
                for c in (b + 1)..n {
                    if nca_data.is_incompatible(a, b, c) {
                        added_triples.insert((a, b, c));
                        // a<b<c so a is min; l[i][a]=FALSE for i>a, making the
                        // clause trivially satisfied. Only add for i<=a.
                        let hi = a.min(k - 1);
                        for i in 0..=hi {
                            add_clause(
                                &mut solver,
                                &[
                                    enc.l[i][a].neg_lit(),
                                    enc.l[i][b].neg_lit(),
                                    enc.l[i][c].neg_lit(),
                                ],
                            );
                        }
                        profile.h4_clauses += hi + 1;
                    }
                }
            }
        }
    }

    // Linear descent from k down to lb_components. Learned clauses from SAT calls at
    // higher bounds accumulate and make the UNSAT proof at opt-1 significantly cheaper.
    for bound in (lb_components..=k).rev() {
        let assumps_vec = totalizer.enforce_ub(bound).unwrap();

        loop {
            profile.sat_calls += 1;

            let t_solve = std::time::Instant::now();
            let result = solver.solve_assumps(&assumps_vec).unwrap();
            profile.solve_ms += t_solve.elapsed().as_secs_f64() * 1000.0;

            match result {
                SolverResult::Sat => {
                    let t_cegar = std::time::Instant::now();
                    let comps = extract_components(&solver, &enc, n, k);
                    let (v, h4_new) = add_violated_triples(
                        &mut solver,
                        &enc,
                        &comps,
                        &nca_data,
                        &mut added_triples,
                    );
                    profile.cegar_ms += t_cegar.elapsed().as_secs_f64() * 1000.0;
                    profile.cegar_violations += v;
                    profile.h4_clauses += h4_new;

                    if v == 0 {
                        profile.optimal_k = bound;
                        best_components = Some(comps);
                        break;
                    }
                }
                SolverResult::Unsat => {
                    if best_components.is_some() {
                        profile.optimal_k = best_components.as_ref().map(|c| c.len()).unwrap_or(k);
                    }
                    profile.report();
                    return best_components
                        .map(|c| components_to_trees(&c, &instance.trees[0], instance.num_leaves));
                }
                SolverResult::Interrupted => {
                    profile.report();
                    return best_components
                        .map(|c| components_to_trees(&c, &instance.trees[0], instance.num_leaves));
                }
            }
        }
    }

    profile.optimal_k = best_components.as_ref().map(|c| c.len()).unwrap_or(k);
    profile.report();

    best_components.map(|c| components_to_trees(&c, &instance.trees[0], instance.num_leaves))
}

fn extract_components(solver: &CaDiCaL, enc: &MafEncoding, n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut components = Vec::new();
    for i in 0..k {
        let mut comp = Vec::new();
        for j in 0..n {
            if solver.var_val(enc.l[i][j]).unwrap() == TernaryVal::True {
                comp.push(j);
            }
        }
        if !comp.is_empty() {
            components.push(comp);
        }
    }
    components
}

fn components_to_trees(components: &[Vec<usize>], ref_tree: &Tree, num_leaves: u32) -> Vec<Tree> {
    let n = num_leaves as usize;
    components
        .iter()
        .map(|comp| {
            let mut leafset = FixedBitSet::with_capacity(n + 1);
            for &j in comp {
                leafset.insert(j + 1);
            }
            ref_tree.prune_to_leafset(&leafset)
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════
// Outer pipeline
// ═══════════════════════════════════════════════════════════════

fn solve_sat(instance: &Instance, stats: &mut SolverStats) -> Option<Vec<Tree>> {
    solve_sat_inner(instance, stats, vec![])
}

fn solve_sat_inner(
    instance: &Instance,
    stats: &mut SolverStats,
    preferred_singleton_labels: Vec<u32>,
) -> Option<Vec<Tree>> {
    solve_sat_inner_impl(instance, stats, preferred_singleton_labels, false)
}

fn solve_sat_inner_impl(
    instance: &Instance,
    stats: &mut SolverStats,
    preferred_singleton_labels: Vec<u32>,
    skip_cluster_decomp: bool,
) -> Option<Vec<Tree>> {
    let n = instance.num_leaves as usize;
    let mut profile = SolveProfile::default();
    profile.n = n;

    let kern_config = KernelizeConfig {
        subtree: true,
        chain: true,
        chain32: true,
        chain32_multi: true,
        protected_labels: preferred_singleton_labels.clone(),
    };
    let kern = kernelize::kernelize(instance, &kern_config);
    let reduced = &kern.instance;
    let param_reduction_32 = kern.param_reduction;

    let n_reduced = reduced.num_leaves as usize;
    if kern.stats.reduced_leaves < instance.num_leaves {
        let total = kern.stats.subtree_removed + kern.stats.chain_removed + kern.stats.chain32_removed;
        eprintln!(
            "[sat] Kernelized: {} → {} leaves ({} removed: {} subtree, {} chain, {} 3-2)",
            n, n_reduced, total,
            kern.stats.subtree_removed, kern.stats.chain_removed, kern.stats.chain32_removed,
        );
    }

    if n_reduced <= 1 {
        stats.lower_bound = 1 + param_reduction_32;
        stats.upper_bound = Some(1 + param_reduction_32);
        // The single surviving leaf is one component; expand back to original space.
        let trivial = vec![reduced.trees[0].clone()];
        let components = kernelize::expand_solution(
            trivial,
            &kern,
            &instance.trees[0],
            instance.num_leaves,
        );
        return Some(components);
    }

    // Try cluster decomposition
    let clusters = if skip_cluster_decomp {
        vec![]
    } else {
        find_common_clusters(&reduced.trees, reduced.num_leaves)
    };
    if let Some(split_cluster) = find_best_split_cluster(&clusters, n_reduced) {
        let cluster_size = split_cluster.leaves.count_ones(..);
        let rest_size = n_reduced - cluster_size + 1;

        profile.cluster_splits += 1;
        eprintln!(
            "[sat] Cluster decomposition: {} = {} + {}",
            n_reduced, cluster_size, rest_size
        );

        // Solve the cluster sub-instance.
        let (cluster_inst, cluster_rev) = restrict_instance(reduced, &split_cluster.leaves);
        let mut cluster_stats = SolverStats::default();
        let cluster_result = solve_sat_inner(&cluster_inst, &mut cluster_stats, vec![])?;

        // Pick representative (smallest label in cluster).
        let rep = split_cluster.leaves.ones().next().unwrap() as u32;

        // Build the rest instance (cluster collapsed to ghost rep).
        let (rest_inst, rest_rev) = collapse_cluster(reduced, &split_cluster.leaves, rep);

        let mut rest_stats = SolverStats::default();
        let rest_result = solve_sat_inner(&rest_inst, &mut rest_stats, vec![])?;

        if let Some(combined) = combine_solutions(
            &cluster_result,
            &rest_result,
            rep,
            &cluster_rev,
            &rest_rev,
            &reduced.trees[0],
            reduced.num_leaves,
        ) {
            let cluster_k =
                cluster_stats.lower_bound + rest_stats.lower_bound - 1 + param_reduction_32;
            stats.lower_bound = cluster_k;
            stats.upper_bound = match (cluster_stats.upper_bound, rest_stats.upper_bound) {
                (Some(a), Some(b)) => Some(a + b - 1 + param_reduction_32),
                _ => None,
            };
            let combined = kernelize::expand_solution(
                combined,
                &kern,
                &instance.trees[0],
                instance.num_leaves,
            );
            return Some(combined);
        }
        // Fall through to monolithic solving on the reduced instance.
        eprintln!("[sat] Cluster combine failed, falling back to monolithic");
    }

    // For m=2: compute red_blue once and derive both LB and UB from it,
    // then tighten UB with cherry_reduce.
    let two_tree_rb_dist = if reduced.num_trees() == 2 {
        Some(red_blue_approx(&reduced.trees[0], &reduced.trees[1]))
    } else {
        None
    };

    let ub_components = if let Some(rb_dist) = two_tree_rb_dist {
        let cherry_ub = cherry_reduce_ub(&reduced.trees[0], &reduced.trees[1]) + 1;
        cherry_ub.min(rb_dist + 1)
    } else {
        let mut best = usize::MAX;
        for ref_idx in 0..reduced.trees.len() {
            best = best.min(greedy_multi_tree_ub(&reduced.trees, ref_idx));
            for seed in 1..=3 {
                best = best.min(greedy_multi_tree_ub_seeded(&reduced.trees, ref_idx, seed));
            }
        }
        best
    };

    let lb_components = if let Some(rb_dist) = two_tree_rb_dist {
        rb_dist.div_ceil(2) + 1
    } else {
        let mut best_lb = 0;
        for i in 0..reduced.trees.len() {
            for j in (i + 1)..reduced.trees.len() {
                let lb = (red_blue_approx(&reduced.trees[i], &reduced.trees[j]).div_ceil(2)) + 1;
                best_lb = best_lb.max(lb);
            }
        }
        best_lb
    }
    .min(ub_components);

    eprintln!(
        "[sat] Bounds: LB={}, UB={} (n'={})",
        lb_components, ub_components, n_reduced
    );

    // Translate all preferred singleton labels into 0-based indices in the fully-reduced instance.
    let preferred_singletons_reduced: Vec<usize> = preferred_singleton_labels
        .iter()
        .filter_map(|&orig_lbl| {
            let found = (1..=reduced.num_leaves)
                .find(|&lbl| kern.reverse_map[lbl as usize] == orig_lbl)
                .map(|lbl| (lbl - 1) as usize);
            if found.is_none() {
                eprintln!(
                    "[sat] preferred_singleton_label={} not found after kernelization",
                    orig_lbl
                );
            }
            found
        })
        .collect();

    // Ghost leaf bounds adjustment.
    let g = preferred_singletons_reduced.len();
    let (lb_components, ub_components) = if g > 0 {
        let min_k_for_ghosts = if n_reduced > g { g + 1 } else { g };
        let lb = lb_components.max(min_k_for_ghosts);
        let ub = (ub_components + g).min(n_reduced);
        eprintln!(
            "[sat] Ghost adjustment: g={} min_k={} lb={} ub={}",
            g, min_k_for_ghosts, lb, ub
        );
        (lb, ub)
    } else {
        (lb_components, ub_components)
    };

    profile.bounds_computed = (lb_components, ub_components);
    stats.lower_bound = lb_components + param_reduction_32;
    stats.upper_bound = Some(ub_components + param_reduction_32);

    let components = sat_solve_maf(
        reduced,
        ub_components,
        lb_components,
        &mut profile,
        &preferred_singletons_reduced,
    )?;

    // Expand solution back to original label space.
    let components = kernelize::expand_solution(
        components,
        &kern,
        &instance.trees[0],
        instance.num_leaves,
    );

    Some(components)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExactSolver;
    use klados_core::tree::NONE;

    fn make_test_instance() -> Instance {
        let t1 = make_tree_1234_cherry12_34();
        let t2 = make_tree_1234_cherry13_24();
        Instance::new(vec![t1, t2], 4)
    }

    fn make_tree_1234_cherry12_34() -> Tree {
        let nodes = 7;
        let mut t = Tree::with_capacity(4);

        t.parent = vec![NONE; nodes];
        t.left = vec![NONE; nodes];
        t.right = vec![NONE; nodes];
        t.label = vec![0; nodes];

        t.label[0] = 1;
        t.label[1] = 2;
        t.label[3] = 3;
        t.label[4] = 4;

        t.left[2] = 0;
        t.right[2] = 1;
        t.parent[0] = 2;
        t.parent[1] = 2;

        t.left[5] = 3;
        t.right[5] = 4;
        t.parent[3] = 5;
        t.parent[4] = 5;

        t.left[6] = 2;
        t.right[6] = 5;
        t.parent[2] = 6;
        t.parent[5] = 6;

        t.root = 6;
        t.num_leaves = 4;
        t.label_to_node = vec![NONE; 5];
        t.label_to_node[1] = 0;
        t.label_to_node[2] = 1;
        t.label_to_node[3] = 3;
        t.label_to_node[4] = 4;

        t.compute_metadata();
        t
    }

    fn make_tree_1234_cherry13_24() -> Tree {
        let nodes = 7;
        let mut t = Tree::with_capacity(4);

        t.parent = vec![NONE; nodes];
        t.left = vec![NONE; nodes];
        t.right = vec![NONE; nodes];
        t.label = vec![0; nodes];

        t.label[0] = 1;
        t.label[1] = 3;
        t.label[3] = 2;
        t.label[4] = 4;

        t.left[2] = 0;
        t.right[2] = 1;
        t.parent[0] = 2;
        t.parent[1] = 2;

        t.left[5] = 3;
        t.right[5] = 4;
        t.parent[3] = 5;
        t.parent[4] = 5;

        t.left[6] = 2;
        t.right[6] = 5;
        t.parent[2] = 6;
        t.parent[5] = 6;

        t.root = 6;
        t.num_leaves = 4;
        t.label_to_node = vec![NONE; 5];
        t.label_to_node[1] = 0;
        t.label_to_node[2] = 3;
        t.label_to_node[3] = 1;
        t.label_to_node[4] = 4;

        t.compute_metadata();
        t
    }

    #[test]
    fn test_solve_identical_trees() {
        let t = make_tree_1234_cherry12_34();
        let instance = Instance::new(vec![t.clone(), t], 4);

        let mut solver = MafSatSolver::new();
        let result = solver.solve(&instance);
        assert!(result.is_some());
        let components = result.unwrap();
        assert_eq!(components.len(), 1);
    }

    #[test]
    fn test_solve_conflicting_trees() {
        let instance = make_test_instance();

        let mut solver = MafSatSolver::new();
        let result = solver.solve(&instance);
        assert!(result.is_some());
        let components = result.unwrap();
        assert_eq!(components.len(), 3);
    }
}
