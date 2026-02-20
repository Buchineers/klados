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
use rustsat::solvers::{Solve, SolveIncremental, SolverResult};
use rustsat::types::{Clause, Lit, TernaryVal, Var};
use rustsat_cadical::CaDiCaL;

use crate::lower_bound::{cherry_reduce_ub, greedy_multi_tree_ub, red_blue_approx};
use crate::shi_mestel::preprocessing::{expand_solution, find_common_subtrees, reduce_instance};
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
    }

    fn report(&self) {
        eprintln!(
            "[profile] n={} n'={} k={} m={} splits={} vars={} clauses={} \
             (H1={} H2={} H3={} H4={} H5={} sym={} sib={}) \
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
fn find_common_clusters(trees: &[Tree], num_leaves: u32) -> Vec<Cluster> {
    if trees.len() < 2 || num_leaves < 6 {
        return Vec::new();
    }

    let n = num_leaves as usize;

    // Get clusters for each tree
    let tree_clusters: Vec<Vec<Cluster>> = trees.iter().map(|t| find_tree_clusters(t, n)).collect();

    // Build a map from leaf set fingerprint to count
    let mut cluster_counts: std::collections::HashMap<u64, (Cluster, usize)> =
        std::collections::HashMap::new();

    for clusters in &tree_clusters {
        for cluster in clusters {
            // Use a simple hash of the leaf set
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            use std::hash::{Hash, Hasher};
            for leaf in cluster.leaves.ones() {
                leaf.hash(&mut hasher);
            }
            let fingerprint = hasher.finish();

            cluster_counts
                .entry(fingerprint)
                .and_modify(|(_, count)| *count += 1)
                .or_insert_with(|| (cluster.clone(), 1));
        }
    }

    // Keep only clusters present in ALL trees
    let mut common: Vec<Cluster> = cluster_counts
        .into_values()
        .filter(|(_, count)| *count == trees.len())
        .map(|(cluster, _)| cluster)
        .collect();

    // Sort by size descending for better splitting
    common.sort_by(|a, b| {
        let size_a = a.leaves.count_ones(..);
        let size_b = b.leaves.count_ones(..);
        size_b.cmp(&size_a)
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

/// Combine two solutions: expand the collapsed representative with the cluster solution
fn combine_solutions(
    cluster_solution: &[Tree],
    rest_solution: &[Tree],
    _cluster: &FixedBitSet,
    rep: u32,
    cluster_reverse: &[u32],
    rest_reverse: &[u32],
    ref_tree: &Tree,
    num_leaves: u32,
) -> Option<Vec<Tree>> {
    let n = num_leaves as usize;
    let mut combined = Vec::new();

    // Find what subproblem label the rep has in rest solution
    let rep_subproblem_label: u32 = (1..rest_reverse.len() as u32)
        .find(|&i| rest_reverse[i as usize] == rep)
        .unwrap_or(1);

    for tree in rest_solution {
        let leaves: Vec<Label> = tree.leaves().collect();
        let contains_rep = leaves.contains(&rep_subproblem_label);

        if contains_rep && leaves.len() > 1 {
            // The rep is grouped with other leaves - this breaks cluster decomposition
            // because the cluster leaves would need to merge with non-cluster leaves,
            // potentially creating invalid subtrees
            return None;
        }

        if contains_rep {
            // Rep is singleton - replace with cluster components
            for ct in cluster_solution {
                let orig_labels: Vec<Label> = ct
                    .leaves()
                    .map(|l| cluster_reverse[l as usize])
                    .filter(|&l| l > 0 && (l as usize) <= n)
                    .collect();
                let mut ls = FixedBitSet::with_capacity(n + 1);
                for &l in &orig_labels {
                    ls.insert(l as usize);
                }
                if ls.count_ones(..) > 0 {
                    combined.push(ref_tree.prune_to_leafset(&ls));
                }
            }
        } else {
            // Component doesn't contain rep - just remap labels
            let mut ls = FixedBitSet::with_capacity(n + 1);
            for &l in &leaves {
                let orig_l = rest_reverse[l as usize];
                if orig_l > 0 && (orig_l as usize) <= n {
                    ls.insert(orig_l as usize);
                }
            }
            if ls.count_ones(..) > 0 {
                combined.push(ref_tree.prune_to_leafset(&ls));
            }
        }
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
// Path Data - precomputed paths for H3
// ═══════════════════════════════════════════════════════════════

/// Precomputed path data for H3: for each pair (ja, jb), the list
/// of internal node indices on the path between them.
struct PathData {
    /// paths[pair_idx] = Vec of internal node indices
    paths: Vec<Vec<usize>>,
}

impl PathData {
    fn build(tree: &Tree, pc: &TreePrecomp, n: usize) -> Self {
        let num_pairs = n * (n - 1) / 2;
        let mut paths = Vec::with_capacity(num_pairs);

        for ja in 0..n {
            let node_a = tree.node_by_label((ja + 1) as Label);
            for jb in (ja + 1)..n {
                let node_b = tree.node_by_label((jb + 1) as Label);
                let nca = tree.nearest_common_ancestor(node_a, node_b);

                let mut path = Vec::with_capacity(16);

                let mut cur = node_a;
                while cur != nca {
                    if !tree.is_leaf(cur) {
                        path.push(pc.idx(cur));
                    }
                    cur = tree.parent[cur as usize];
                }

                cur = node_b;
                while cur != nca {
                    if !tree.is_leaf(cur) {
                        path.push(pc.idx(cur));
                    }
                    cur = tree.parent[cur as usize];
                }

                if !tree.is_leaf(nca) {
                    path.push(pc.idx(nca));
                }

                paths.push(path);
            }
        }

        Self { paths }
    }

    fn get(&self, ja: usize, jb: usize, n: usize) -> &[usize] {
        // Index into flat array matching the nested loop order
        let offset = ja * n - ja * (ja + 1) / 2 + (jb - ja - 1);
        &self.paths[offset]
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
        n,
        k,
        m,
    }
}

fn add_clause(solver: &mut CaDiCaL, lits: &[Lit]) {
    solver.add_clause(Clause::from(lits)).unwrap();
}

/// Encode base clauses + H3. H4 added via CEGAR.
fn add_structural_clauses(
    solver: &mut CaDiCaL,
    enc: &MafEncoding,
    path_data: &[PathData],
    profile: &mut SolveProfile,
) {
    let n = enc.n;
    let k = enc.k;
    let m = enc.m;

    // H1a: every leaf in at least one component
    for j in 0..n {
        let clause: Vec<Lit> = (0..k).map(|i| enc.l[i][j].pos_lit()).collect();
        add_clause(solver, &clause);
    }
    profile.h1_clauses += n;

    // H1b: at most one component per leaf (ladder)
    for j in 0..n {
        for i in 0..k - 1 {
            add_clause(solver, &[enc.l[i][j].neg_lit(), enc.s[i][j].pos_lit()]);
        }
        for i in 1..k - 1 {
            add_clause(solver, &[enc.s[i - 1][j].neg_lit(), enc.s[i][j].pos_lit()]);
        }
        for i in 1..k {
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

    // H3: path consistency — paths precomputed ONCE, shared across k
    let mut h3 = 0usize;
    for q in 0..m {
        let pd = &path_data[q];
        for ja in 0..n {
            for jb in (ja + 1)..n {
                let path = pd.get(ja, jb, n);
                for i in 0..k {
                    let neg_a = enc.l[i][ja].neg_lit();
                    let neg_b = enc.l[i][jb].neg_lit();
                    for &idx in path {
                        add_clause(solver, &[neg_a, neg_b, enc.w[q][i][idx].pos_lit()]);
                        h3 += 1;
                    }
                }
            }
        }
    }
    profile.h3_clauses = h3;

    // H5: usage tracking
    for i in 0..k {
        for j in 0..n {
            add_clause(solver, &[enc.l[i][j].neg_lit(), enc.u[i].pos_lit()]);
        }
    }
    profile.h5_clauses = n * k;

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

                let is_common = instance.trees[1..].iter().all(|t| {
                    let na = t.node_by_label(la);
                    let nb = t.node_by_label(lb);
                    let pa = t.parent[na as usize];
                    let pb = t.parent[nb as usize];
                    pa != NONE && pa == pb
                });

                if is_common {
                    let ja = (la - 1) as usize;
                    let jb = (lb - 1) as usize;
                    for i in 0..k {
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

fn add_violated_triples(
    solver: &mut CaDiCaL,
    enc: &MafEncoding,
    components: &[Vec<usize>],
    nca_data: &NcaData,
) -> usize {
    let k = enc.k;
    let mut violations = 0;

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
                        for i in 0..k {
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
                    }
                }
            }
        }
    }

    violations
}

// ═══════════════════════════════════════════════════════════════
// Solver core
// ═══════════════════════════════════════════════════════════════

fn sat_solve_maf(
    instance: &Instance,
    k_max: usize,
    lb_components: usize,
    profile: &mut SolveProfile,
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

    // Precompute ALL paths ONCE (not per k iteration)
    let t_path = std::time::Instant::now();
    let path_data: Vec<PathData> = (0..m)
        .map(|q| PathData::build(&instance.trees[q], &precomps[q], n))
        .collect();
    let path_ms = t_path.elapsed().as_secs_f64() * 1000.0;

    let nca_data = NcaData::build(instance, n);

    let mut solver = CaDiCaL::default();
    let mut vm = BasicVarManager::default();

    let enc = create_variables(&mut vm, n, k, m, &precomps);

    let t_enc = std::time::Instant::now();
    add_structural_clauses(&mut solver, &enc, &path_data, profile);
    seed_sibling_constraints(&mut solver, instance, &enc, profile);
    profile.encode_ms = t_enc.elapsed().as_secs_f64() * 1000.0 + path_ms;
    profile.num_vars = vm.n_used() as usize;

    let u_lits: Vec<Lit> = enc.u.iter().map(|v| v.pos_lit()).collect();
    let mut totalizer = Totalizer::default();
    for lit in u_lits {
        totalizer.extend([lit]);
    }

    let mut best_components: Option<Vec<Vec<usize>>> = None;

    for bound in (lb_components..=k).rev() {
        totalizer
            .encode_ub(bound..=bound, &mut solver, &mut vm)
            .unwrap();
        let assumps_vec = totalizer.enforce_ub(bound).unwrap();

        let t_bound = std::time::Instant::now();
        let mut cegar_iters = 0;

        loop {
            cegar_iters += 1;
            profile.sat_calls += 1;

            let t_solve = std::time::Instant::now();
            let result = solver.solve_assumps(&assumps_vec).unwrap();
            profile.solve_ms += t_solve.elapsed().as_secs_f64() * 1000.0;

            match result {
                SolverResult::Sat => {
                    let t_cegar = std::time::Instant::now();
                    let comps = extract_components(&solver, &enc, n, k);
                    let v = add_violated_triples(&mut solver, &enc, &comps, &nca_data);
                    profile.cegar_ms += t_cegar.elapsed().as_secs_f64() * 1000.0;
                    profile.cegar_violations += v;
                    profile.h4_clauses += v * k;

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
    let n = instance.num_leaves as usize;
    let mut profile = SolveProfile::default();
    profile.n = n;

    // Phase 0: Try cluster decomposition FIRST
    let clusters = find_common_clusters(&instance.trees, instance.num_leaves);
    if let Some(split_cluster) = find_best_split_cluster(&clusters, n) {
        let cluster_size = split_cluster.leaves.count_ones(..);
        let rest_size = n - cluster_size + 1;

        profile.cluster_splits += 1;
        eprintln!(
            "[sat] Cluster decomposition: {} = {} + {}",
            n, cluster_size, rest_size
        );

        // Solve the cluster sub-instance
        let (cluster_inst, cluster_rev) = restrict_instance(instance, &split_cluster.leaves);
        let mut cluster_stats = SolverStats::default();
        let cluster_result = solve_sat(&cluster_inst, &mut cluster_stats)?;

        // Pick representative (first label in cluster)
        let rep = split_cluster.leaves.ones().next().unwrap() as u32;

        // Solve the rest (cluster collapsed to representative)
        let (rest_inst, rest_rev) = collapse_cluster(instance, &split_cluster.leaves, rep);
        let mut rest_stats = SolverStats::default();
        let rest_result = solve_sat(&rest_inst, &mut rest_stats)?;

        // Combine solutions
        if let Some(combined) = combine_solutions(
            &cluster_result,
            &rest_result,
            &split_cluster.leaves,
            rep,
            &cluster_rev,
            &rest_rev,
            &instance.trees[0],
            instance.num_leaves,
        ) {
            stats.lower_bound = cluster_stats.lower_bound + rest_stats.lower_bound - 1;
            stats.upper_bound = match (cluster_stats.upper_bound, rest_stats.upper_bound) {
                (Some(a), Some(b)) => Some(a + b - 1),
                _ => None,
            };
            return Some(combined);
        }
        // If combine fails (rep was grouped), fall through to normal solving
        eprintln!("[sat] Cluster decomposition failed (rep grouped), falling back");
    }

    // Phase 1: Kernelize (subtree reduction)
    let collapses = find_common_subtrees(&instance.trees, instance.num_leaves);
    let (reduced, reverse_map) = if collapses.is_empty() {
        (instance.clone(), (0..=instance.num_leaves).collect())
    } else {
        reduce_instance(instance, &collapses)
    };

    let n_reduced = reduced.num_leaves as usize;
    if !collapses.is_empty() {
        eprintln!(
            "[sat] Kernelized: {} → {} leaves ({} subtrees collapsed)",
            n,
            n_reduced,
            collapses.len()
        );
    }

    if n_reduced <= 1 {
        stats.lower_bound = 1;
        stats.upper_bound = Some(1);
        return Some(vec![instance.trees[0].clone()]);
    }

    let ub_components = if reduced.num_trees() == 2 {
        cherry_reduce_ub(&reduced.trees[0], &reduced.trees[1]) + 1
    } else {
        let mut best = usize::MAX;
        for ref_idx in 0..reduced.trees.len() {
            best = best.min(greedy_multi_tree_ub(&reduced.trees, ref_idx));
        }
        best
    };

    let lb_components = if reduced.num_trees() == 2 {
        (red_blue_approx(&reduced.trees[0], &reduced.trees[1]).div_ceil(2)) + 1
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

    profile.bounds_computed = (lb_components, ub_components);
    stats.lower_bound = lb_components;
    stats.upper_bound = Some(ub_components);

    let components = sat_solve_maf(&reduced, ub_components, lb_components, &mut profile)?;

    if collapses.is_empty() {
        Some(components)
    } else {
        Some(expand_solution(
            components,
            &collapses,
            &reverse_map,
            &instance.trees[0],
            instance.num_leaves,
        ))
    }
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
