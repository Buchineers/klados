use fixedbitset::FixedBitSet;
use klados_core::tree::{Label, NodeId, Tree};
use klados_core::{Instance, SolverStats};
use rustsat::encodings::card::{BoundUpper, Totalizer};
use rustsat::instances::{BasicVarManager, ManageVars};
use rustsat::solvers::{Solve, SolveIncremental, SolverResult};
use rustsat::types::{Clause, Lit, TernaryVal, Var};
use rustsat_cadical::CaDiCaL;

use crate::cluster_reduction::{self, ClusterReductionResult};
use crate::kernelize::{self, KernelizeConfig};
use crate::lower_bound::maf_bounds;
use crate::ExactSolver;

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

pub struct MafSatSolver {
    stats: SolverStats,
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
        }
    }
}

impl ExactSolver for MafSatSolver {
    fn name(&self) -> &'static str {
        "maf-sat"
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
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

pub struct MafSatOlverSolver {
    stats: SolverStats,
}

impl Default for MafSatOlverSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl MafSatOlverSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
        }
    }
}

impl ExactSolver for MafSatOlverSolver {
    fn name(&self) -> &'static str {
        "maf-sat-olver"
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        if instance.trees.is_empty() {
            return None;
        }
        if instance.num_trees() == 1 {
            return Some(instance.trees.clone());
        }
        if instance.num_leaves <= 1 {
            return Some(instance.trees[0..1].to_vec());
        }
        if instance.num_trees() != 2 {
            eprintln!("[olver] Olver encoding requires exactly 2 trees, got {}", instance.num_trees());
            return None;
        }
        solve_sat_olver(instance, &mut self.stats)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

/// Solve a (typically 2-tree) instance with the SAT solver and return the component count.
/// Used as a callback for pairwise lower bound computation.
pub(crate) fn solve_pair_sat(instance: &Instance) -> Option<usize> {
    let mut stats = SolverStats::default();
    solve_sat(instance, &mut stats).map(|trees| trees.len())
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

fn add_clause(solver: &mut CaDiCaL, lits: &[Lit]) {
    solver.add_clause(Clause::from(lits)).unwrap();
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
// Cut-based encoding (k-independent!)
// ═══════════════════════════════════════════════════════════════

/// Compute the set of non-root nodes on the path from leaf a to leaf b in tree.
/// These are the nodes whose parent-edge, if cut, would disconnect a from b.
fn path_nodes(tree: &Tree, a: NodeId, b: NodeId) -> Vec<NodeId> {
    let nca = tree.nearest_common_ancestor(a, b);
    let mut nodes = Vec::new();
    let mut cur = a;
    while cur != nca {
        nodes.push(cur);
        cur = tree.parent[cur as usize];
    }
    cur = b;
    while cur != nca {
        nodes.push(cur);
        cur = tree.parent[cur as usize];
    }
    nodes
}

fn sat_solve_maf_cut(
    instance: &Instance,
    k_max: usize,
    lb_components: usize,
    profile: &mut SolveProfile,
) -> Option<Vec<Tree>> {
    let n = instance.num_leaves as usize;
    let m = instance.num_trees();

    profile.n_reduced = n;
    profile.k = k_max;
    profile.m = m;

    // Precompute NCA data for H4 incompatible triple detection.
    let nca_data = NcaData::build(instance, n);

    // Precompute paths between all leaf pairs for each tree.
    // paths[q][(a,b)] = list of non-root nodes on path from leaf a+1 to leaf b+1.
    let t_path = std::time::Instant::now();
    let mut paths: Vec<Vec<Vec<Vec<NodeId>>>> = Vec::with_capacity(m);
    for q in 0..m {
        let tree = &instance.trees[q];
        let mut pq = vec![vec![Vec::new(); n]; n];
        for a in 0..n {
            let na = tree.node_by_label((a + 1) as Label);
            for b in (a + 1)..n {
                let nb = tree.node_by_label((b + 1) as Label);
                let pnodes = path_nodes(tree, na, nb);
                pq[a][b] = pnodes;
            }
        }
        paths.push(pq);
    }
    let path_ms = t_path.elapsed().as_secs_f64() * 1000.0;

    let mut solver = CaDiCaL::default();
    let mut vm = BasicVarManager::default();

    // --- Variables ---
    // del[q][v]: edge from node v to parent(v) in tree q is cut.
    let num_nodes: Vec<usize> = instance.trees.iter().map(|t| t.num_nodes()).collect();
    let del: Vec<Vec<Option<Var>>> = (0..m)
        .map(|q| {
            let tree = &instance.trees[q];
            (0..num_nodes[q])
                .map(|v| {
                    if v as NodeId == tree.root {
                        None
                    } else {
                        Some(vm.new_var())
                    }
                })
                .collect()
        })
        .collect();
    let del_count: usize = del.iter().map(|d| d.iter().filter(|v| v.is_some()).count()).sum();

    // conn[a][b]: leaves a and b are in the same component (single set, not per-tree).
    // Cross-tree consistency is enforced via backward clauses on ALL trees' del vars.
    // Only allocate for a < b (upper triangle). conn[a][b] for a >= b is unused.
    let conn: Vec<Vec<Option<Var>>> = (0..n)
        .map(|a| {
            (0..n)
                .map(|b| if b > a { Some(vm.new_var()) } else { None })
                .collect()
        })
        .collect();
    let conn_count = n * (n - 1) / 2;

    eprintln!(
        "[cut] Variables: {} del + {} conn = {} (path precomp {:.1}ms)",
        del_count, conn_count, vm.n_used(), path_ms
    );

    // --- Clauses ---
    let t_enc = std::time::Instant::now();
    let mut backward_count = 0usize;
    let mut forward_count = 0usize;
    let mut h4_count = 0usize;

    // 1. Backward for ALL trees: conn[a][b] → ¬del[q][v].
    for q in 0..m {
        for a in 0..n {
            for b in (a + 1)..n {
                for &v in &paths[q][a][b] {
                    if let Some(dv) = del[q][v as usize] {
                        add_clause(
                            &mut solver,
                            &[conn[a][b].unwrap().neg_lit(), dv.neg_lit()],
                        );
                        backward_count += 1;
                    }
                }
            }
        }
    }

    // 2. Forward for ALL trees: (no cut on path_q) → conn[a][b].
    //    Needed for correctness: conn must be true only if connected in ALL trees.
    for q in 0..m {
        for a in 0..n {
            for b in (a + 1)..n {
                let mut clause = vec![conn[a][b].unwrap().pos_lit()];
                for &v in &paths[q][a][b] {
                    if let Some(dv) = del[q][v as usize] {
                        clause.push(dv.pos_lit());
                    }
                }
                add_clause(&mut solver, &clause);
                forward_count += 1;
            }
        }
    }

    // 3. H4: incompatible triples — all 3 binary clauses per triple.
    //    For integer solutions, at most one pair from an incompatible triple can
    //    be in the same component (s(a,b) + s(a,c) + s(b,c) ≤ 1).
    //    All 3 binary clauses maximize unit propagation in CaDiCaL.
    for a in 0..n {
        for b in (a + 1)..n {
            for c in (b + 1)..n {
                if nca_data.is_incompatible(a, b, c) {
                    add_clause(
                        &mut solver,
                        &[conn[a][b].unwrap().neg_lit(), conn[a][c].unwrap().neg_lit()],
                    );
                    add_clause(
                        &mut solver,
                        &[conn[a][b].unwrap().neg_lit(), conn[b][c].unwrap().neg_lit()],
                    );
                    add_clause(
                        &mut solver,
                        &[conn[a][c].unwrap().neg_lit(), conn[b][c].unwrap().neg_lit()],
                    );
                    h4_count += 3;
                }
            }
        }
    }

    // 5. Totalizer on del[0][*] to count cuts in tree 0.
    //    k components = k-1 cuts. Minimize cuts.
    let del_0_lits: Vec<Lit> = del[0]
        .iter()
        .filter_map(|v| v.map(|var| var.pos_lit()))
        .collect();
    let del_0_count = del_0_lits.len();

    let mut totalizer = Totalizer::default();
    for lit in del_0_lits {
        totalizer.extend([lit]);
    }
    let lb_cuts = if lb_components > 0 { lb_components - 1 } else { 0 };
    let ub_cuts = k_max - 1;
    totalizer
        .encode_ub(lb_cuts..=ub_cuts, &mut solver, &mut vm)
        .unwrap();

    let encode_ms = t_enc.elapsed().as_secs_f64() * 1000.0;
    let total_clauses = backward_count + forward_count + h4_count;
    profile.num_vars = vm.n_used() as usize;
    profile.encode_ms = path_ms + encode_ms;

    eprintln!(
        "[cut] Clauses: {} total ({} backward + {} forward + {} H4) in {:.1}ms",
        total_clauses, backward_count, forward_count, h4_count, encode_ms
    );
    eprintln!(
        "[cut] Total: {} vars, {} clauses. Totalizer over {} del vars.",
        profile.num_vars, total_clauses, del_0_count
    );

    // --- Descent with CEGAR for cross-tree consistency ---
    let t_total_solve = std::time::Instant::now();
    let mut best_components: Option<Vec<Vec<usize>>> = None;

    fn uf_find(p: &mut [usize], x: usize) -> usize {
        if p[x] != x {
            p[x] = uf_find(p, p[x]);
        }
        p[x]
    }

    for cuts_bound in (lb_cuts..=ub_cuts).rev() {
        let k_bound = cuts_bound + 1;
        let assumps_vec = totalizer.enforce_ub(cuts_bound).unwrap();

        profile.sat_calls += 1;
        let t_solve = std::time::Instant::now();
        let result = solver.solve_assumps(&assumps_vec).unwrap();
        let solve_ms = t_solve.elapsed().as_secs_f64() * 1000.0;
        profile.solve_ms += solve_ms;
        let cum_s = t_total_solve.elapsed().as_secs_f64();

        match result {
            SolverResult::Sat => {
                let mut uf: Vec<usize> = (0..n).collect();
                for a in 0..n {
                    for b in (a + 1)..n {
                        if solver.var_val(conn[a][b].unwrap()).unwrap() == TernaryVal::True {
                            let ra = uf_find(&mut uf, a);
                            let rb = uf_find(&mut uf, b);
                            if ra != rb {
                                uf[ra] = rb;
                            }
                        }
                    }
                }
                let mut comp_map: fxhash::FxHashMap<usize, Vec<usize>> =
                    fxhash::FxHashMap::default();
                for j in 0..n {
                    let r = uf_find(&mut uf, j);
                    comp_map.entry(r).or_default().push(j);
                }
                let comps: Vec<Vec<usize>> = comp_map.into_values().collect();
                let num_comps = comps.len();
                let max_sz = comps.iter().map(|c| c.len()).max().unwrap_or(0);
                eprintln!(
                    "[cut] k={} SAT {:.1}ms (cum {:.1}s) comps={} max_size={}",
                    k_bound, solve_ms, cum_s, num_comps, max_sz
                );
                best_components = Some(comps);
            }
            SolverResult::Unsat => {
                eprintln!(
                    "[cut] k={} UNSAT {:.1}ms (cum {:.1}s) — optimal={}",
                    k_bound, solve_ms, cum_s, k_bound + 1
                );
                profile.optimal_k =
                    best_components.as_ref().map(|c| c.len()).unwrap_or(k_max);
                profile.report();
                return best_components.map(|c| {
                    components_to_trees(&c, &instance.trees[0], instance.num_leaves)
                });
            }
            SolverResult::Interrupted => {
                profile.report();
                return best_components.map(|c| {
                    components_to_trees(&c, &instance.trees[0], instance.num_leaves)
                });
            }
        }
    }

    profile.optimal_k = best_components.as_ref().map(|c| c.len()).unwrap_or(k_max);
    profile.report();
    best_components.map(|c| components_to_trees(&c, &instance.trees[0], instance.num_leaves))
}

// ═══════════════════════════════════════════════════════════════
// Olver LP* formulation as SAT encoding (2-tree only)
// ═══════════════════════════════════════════════════════════════

/// Check whether `ancestor` is an ancestor of `descendant` in `tree`.
fn is_ancestor(tree: &Tree, ancestor: NodeId, descendant: NodeId) -> bool {
    let mut cur = descendant;
    while tree.depth[cur as usize] > tree.depth[ancestor as usize] {
        cur = tree.parent[cur as usize];
    }
    cur == ancestor
}

/// Pairwise at-most-one encoding: for every pair of literals, at most one can be true.
fn add_amo_pairwise(solver: &mut CaDiCaL, lits: &[Lit]) {
    for i in 0..lits.len() {
        for j in (i + 1)..lits.len() {
            add_clause(solver, &[!lits[i], !lits[j]]);
        }
    }
}

fn sat_solve_maf_olver(
    instance: &Instance,
    k_max: usize,
    lb_components: usize,
    profile: &mut SolveProfile,
) -> Option<Vec<Tree>> {
    let n = instance.num_leaves as usize;
    assert!(instance.num_trees() == 2, "Olver encoding requires exactly 2 trees");

    profile.n_reduced = n;
    profile.k = k_max;
    profile.m = 2;

    let t1 = &instance.trees[0];
    let t2 = &instance.trees[1];
    let trees_ref = [t1, t2];

    let t_enc = std::time::Instant::now();

    // --- Build DAG D = (Z, U1 ∪ U2) ---
    // Z = all pairs (a, b) with 0 ≤ a ≤ b < n (0-based leaf indices).
    // pair_to_idx(a, b) = b*(b+1)/2 + a
    let pair_to_idx = |a: usize, b: usize| -> usize {
        debug_assert!(a <= b && b < n);
        b * (b + 1) / 2 + a
    };
    let num_z = n * (n + 1) / 2;

    // Precompute LCA node for each Z-pair in both trees.
    // Using 0-based leaf indices, leaf a corresponds to label a+1.
    let mut lca_node: [Vec<NodeId>; 2] = [vec![0; num_z], vec![0; num_z]];
    for t in 0..2 {
        let tree = trees_ref[t];
        for a in 0..n {
            let na = tree.node_by_label((a + 1) as Label);
            for b in a..n {
                let idx = pair_to_idx(a, b);
                if a == b {
                    lca_node[t][idx] = na;
                } else {
                    let nb = tree.node_by_label((b + 1) as Label);
                    lca_node[t][idx] = tree.nearest_common_ancestor(na, nb);
                }
            }
        }
    }

    // Check if s is strictly below r in both trees.
    let is_strictly_below = |r_idx: usize, s_idx: usize| -> bool {
        for t in 0..2 {
            let r_node = lca_node[t][r_idx];
            let s_node = lca_node[t][s_idx];
            if r_node == s_node {
                return false;
            }
            let tree = trees_ref[t];
            if tree.depth[s_node as usize] <= tree.depth[r_node as usize] {
                return false;
            }
            if !is_ancestor(tree, r_node, s_node) {
                return false;
            }
        }
        true
    };

    // Build arc lists.
    // Arc (r, s) ∈ U1 if first indices match AND lca_t(s) strictly below lca_t(r) in both trees.
    // Arc (r, s) ∈ U2 if second indices match AND same condition.
    struct OlverArc {
        from: usize,
        to: usize,
    }

    let mut u1_arcs: Vec<OlverArc> = Vec::new();
    let mut u2_arcs: Vec<OlverArc> = Vec::new();

    // U1: arcs between Z-nodes sharing the same first index a.
    for a in 0..n {
        for b_r in a..n {
            let r_idx = pair_to_idx(a, b_r);
            for b_s in a..n {
                if b_r == b_s {
                    continue;
                }
                let s_idx = pair_to_idx(a, b_s);
                if is_strictly_below(r_idx, s_idx) {
                    u1_arcs.push(OlverArc { from: r_idx, to: s_idx });
                }
            }
        }
    }

    // U2: arcs between Z-nodes sharing the same second index b.
    for b in 0..n {
        for a_r in 0..=b {
            let r_idx = pair_to_idx(a_r, b);
            for a_s in 0..=b {
                if a_r == a_s {
                    continue;
                }
                let s_idx = pair_to_idx(a_s, b);
                if is_strictly_below(r_idx, s_idx) {
                    u2_arcs.push(OlverArc { from: r_idx, to: s_idx });
                }
            }
        }
    }

    // Precompute per-node incoming/outgoing arc indices.
    let mut out_u1: Vec<Vec<usize>> = vec![Vec::new(); num_z];
    let mut out_u2: Vec<Vec<usize>> = vec![Vec::new(); num_z];
    let mut in_arcs: Vec<Vec<(usize, bool)>> = vec![Vec::new(); num_z]; // (arc_idx, is_u1)

    for (ai, arc) in u1_arcs.iter().enumerate() {
        out_u1[arc.from].push(ai);
        in_arcs[arc.to].push((ai, true));
    }
    for (ai, arc) in u2_arcs.iter().enumerate() {
        out_u2[arc.from].push(ai);
        in_arcs[arc.to].push((ai, false));
    }

    // Identify leaf nodes in Z: (a, a) for each leaf a.
    let mut is_leaf_z = vec![false; num_z];
    for a in 0..n {
        is_leaf_z[pair_to_idx(a, a)] = true;
    }

    eprintln!(
        "[olver] DAG: {} Z-nodes, {} U1 arcs, {} U2 arcs, {} leaves",
        num_z,
        u1_arcs.len(),
        u2_arcs.len(),
        n
    );

    // --- SAT Variables ---
    let mut solver = CaDiCaL::default();
    let mut vm = BasicVarManager::default();

    // y_u1[i]: arc variable for U1 arc i
    let y_u1: Vec<Var> = (0..u1_arcs.len()).map(|_| vm.new_var()).collect();
    // y_u2[i]: arc variable for U2 arc i
    let y_u2: Vec<Var> = (0..u2_arcs.len()).map(|_| vm.new_var()).collect();
    // x[i]: leaf i is a singleton component
    let x: Vec<Var> = (0..n).map(|_| vm.new_var()).collect();

    let mut clause_count = 0usize;

    // --- SAT Constraints ---

    // LP*-2 (balance): For each non-leaf Z-node r, the number of active U1 out-arcs
    // must equal the number of active U2 out-arcs.
    // With binary variables + LP*-3 (flow conservation), each active non-leaf uses
    // exactly one U1 and one U2 arc.
    // Encode: AMO on out_u1, AMO on out_u2, (∨ out_u1) ↔ (∨ out_u2).
    for r in 0..num_z {
        if is_leaf_z[r] {
            continue;
        }
        let has_u1 = !out_u1[r].is_empty();
        let has_u2 = !out_u2[r].is_empty();

        if !has_u1 && !has_u2 {
            continue;
        }

        // AMO on out_u1 arcs
        if out_u1[r].len() > 1 {
            let lits: Vec<Lit> = out_u1[r].iter().map(|&ai| y_u1[ai].pos_lit()).collect();
            let amo_clauses = lits.len() * (lits.len() - 1) / 2;
            add_amo_pairwise(&mut solver, &lits);
            clause_count += amo_clauses;
        }

        // AMO on out_u2 arcs
        if out_u2[r].len() > 1 {
            let lits: Vec<Lit> = out_u2[r].iter().map(|&ai| y_u2[ai].pos_lit()).collect();
            let amo_clauses = lits.len() * (lits.len() - 1) / 2;
            add_amo_pairwise(&mut solver, &lits);
            clause_count += amo_clauses;
        }

        if has_u1 && has_u2 {
            // (∨ out_u1) ↔ (∨ out_u2)
            // Forward: each u1 arc active → some u2 arc active
            for &ai in &out_u1[r] {
                let mut clause: Vec<Lit> = vec![y_u1[ai].neg_lit()];
                for &aj in &out_u2[r] {
                    clause.push(y_u2[aj].pos_lit());
                }
                add_clause(&mut solver, &clause);
                clause_count += 1;
            }
            // Backward: each u2 arc active → some u1 arc active
            for &ai in &out_u2[r] {
                let mut clause: Vec<Lit> = vec![y_u2[ai].neg_lit()];
                for &aj in &out_u1[r] {
                    clause.push(y_u1[aj].pos_lit());
                }
                add_clause(&mut solver, &clause);
                clause_count += 1;
            }
        } else if has_u1 && !has_u2 {
            // No U2 out-arcs: force all U1 out-arcs to 0
            for &ai in &out_u1[r] {
                add_clause(&mut solver, &[y_u1[ai].neg_lit()]);
                clause_count += 1;
            }
        } else {
            // No U1 out-arcs: force all U2 out-arcs to 0
            for &ai in &out_u2[r] {
                add_clause(&mut solver, &[y_u2[ai].neg_lit()]);
                clause_count += 1;
            }
        }
    }
    let balance_clauses = clause_count;

    // LP*-3 (conservation): For each non-leaf Z-node r, for each incoming arc a_in:
    // y_a_in → ∨ y_out_u1(r)
    for r in 0..num_z {
        if is_leaf_z[r] {
            continue;
        }
        if out_u1[r].is_empty() {
            // If no outgoing U1 arcs, any incoming arc must be 0
            for &(ai, is_u1_arc) in &in_arcs[r] {
                let lit = if is_u1_arc {
                    y_u1[ai].neg_lit()
                } else {
                    y_u2[ai].neg_lit()
                };
                add_clause(&mut solver, &[lit]);
                clause_count += 1;
            }
        } else {
            for &(ai, is_u1_arc) in &in_arcs[r] {
                let in_lit = if is_u1_arc {
                    y_u1[ai].neg_lit()
                } else {
                    y_u2[ai].neg_lit()
                };
                let mut clause = vec![in_lit];
                for &oj in &out_u1[r] {
                    clause.push(y_u1[oj].pos_lit());
                }
                add_clause(&mut solver, &clause);
                clause_count += 1;
            }
        }
    }
    let conservation_clauses = clause_count - balance_clauses;

    // LP*-4 (leaf coverage): For each leaf (a, a):
    // x_a ∨ ∨ y_incoming (leaf must be singleton or have incoming flow)
    // ¬x_a ∨ ¬y_incoming_j for each incoming arc j (can't be both)
    for a in 0..n {
        let leaf_idx = pair_to_idx(a, a);
        let incoming = &in_arcs[leaf_idx];

        // Coverage: x_a ∨ ∨ y_incoming
        let mut clause = vec![x[a].pos_lit()];
        for &(ai, is_u1_arc) in incoming {
            clause.push(if is_u1_arc {
                y_u1[ai].pos_lit()
            } else {
                y_u2[ai].pos_lit()
            });
        }
        add_clause(&mut solver, &clause);
        clause_count += 1;

        // Exclusivity: ¬x_a ∨ ¬y_incoming_j for each j
        for &(ai, is_u1_arc) in incoming {
            let in_lit = if is_u1_arc {
                y_u1[ai].neg_lit()
            } else {
                y_u2[ai].neg_lit()
            };
            add_clause(&mut solver, &[x[a].neg_lit(), in_lit]);
            clause_count += 1;
        }
    }
    let leaf_clauses = clause_count - balance_clauses - conservation_clauses;

    // LP*-5 (capacity): For each internal node v in each tree t,
    // at most one Z-node r with lca_t(r) = v can have active outgoing U1 arcs.
    // This is an AMO constraint on the UNION of out_u1 arcs from all r where lca_t(r) = v.
    let mut capacity_clauses = 0usize;
    for t in 0..2 {
        let tree = trees_ref[t];
        for v in 0..tree.num_nodes() as NodeId {
            if tree.is_leaf(v) {
                continue;
            }
            // Collect all out_u1 arc literals from Z-nodes r where lca_t(r) = v.
            let mut lits: Vec<Lit> = Vec::new();
            for a in 0..n {
                for b in a..n {
                    let idx = pair_to_idx(a, b);
                    if lca_node[t][idx] == v {
                        for &ai in &out_u1[idx] {
                            lits.push(y_u1[ai].pos_lit());
                        }
                    }
                }
            }
            if lits.len() > 1 {
                let amo_count = lits.len() * (lits.len() - 1) / 2;
                add_amo_pairwise(&mut solver, &lits);
                capacity_clauses += amo_count;
            }
        }
    }
    clause_count += capacity_clauses;

    // Path-blocking constraints: when an arc (r→s) is active, its path in each tree
    // passes through intermediate internal nodes. No other Z-node at those intermediate
    // nodes can be active (otherwise edge-disjointness is violated).
    //
    // For each arc a, each tree t, walk from lca_t(a.to) up to lca_t(a.from).
    // For each intermediate node v (exclusive of endpoints):
    //   For each Z-node r' with lca_t(r') = v:
    //     For each out_u1 arc o from r':
    //       ¬y_a ∨ ¬y_o  (binary clause)
    let mut path_block_clauses = 0usize;

    // Precompute: for each tree t and internal node v, the set of out_u1 arc literals
    // from Z-nodes at v. Reuse from LP*-5.
    let mut node_out_u1_lits: [Vec<Vec<Lit>>; 2] = [
        vec![Vec::new(); t1.num_nodes()],
        vec![Vec::new(); t2.num_nodes()],
    ];
    for t in 0..2 {
        let tree = trees_ref[t];
        for a in 0..n {
            for b in a..n {
                let idx = pair_to_idx(a, b);
                let v = lca_node[t][idx];
                if !tree.is_leaf(v) {
                    for &ai in &out_u1[idx] {
                        node_out_u1_lits[t][v as usize].push(y_u1[ai].pos_lit());
                    }
                }
            }
        }
    }

    // Process all arcs (U1 and U2).
    let all_arcs: Vec<(Lit, usize, usize)> = u1_arcs
        .iter()
        .enumerate()
        .map(|(ai, arc)| (y_u1[ai].pos_lit(), arc.from, arc.to))
        .chain(
            u2_arcs
                .iter()
                .enumerate()
                .map(|(ai, arc)| (y_u2[ai].pos_lit(), arc.from, arc.to)),
        )
        .collect();

    for &(arc_lit, from_idx, to_idx) in &all_arcs {
        for t in 0..2 {
            let tree = trees_ref[t];
            let from_node = lca_node[t][from_idx];
            let to_node = lca_node[t][to_idx];
            // Walk from to_node up to from_node, collecting intermediate nodes.
            let mut cur = tree.parent[to_node as usize];
            while cur != from_node && cur != klados_core::tree::NONE {
                // cur is an intermediate node — block all out_u1 arcs from Z-nodes at cur.
                for out_lit in &node_out_u1_lits[t][cur as usize] {
                    add_clause(&mut solver, &[!arc_lit, !*out_lit]);
                    path_block_clauses += 1;
                }
                cur = tree.parent[cur as usize];
            }
        }
    }
    clause_count += path_block_clauses;

    // --- Component counting variables ---
    // A component is started by:
    // - A singleton leaf (x_a = 1), or
    // - An arborescence root: a non-leaf Z-node with active out-arcs but no incoming flow.
    //
    // For each non-leaf r with out-arcs, create root_r:
    //   root_r ↔ (∨ out_u1(r)) ∧ ¬(∨ in_arcs(r))
    // Which means:
    //   root_r → ∨ out_u1(r)          (root must have outgoing flow)
    //   root_r → ¬y_in_j for each j   (root has no incoming flow)
    //   (∨ out_u1(r)) ∧ (∧ ¬y_in_j) → root_r  (if outgoing and no incoming, then root)

    let mut component_lits: Vec<Lit> = Vec::new();

    // Singleton leaves
    for a in 0..n {
        component_lits.push(x[a].pos_lit());
    }

    // Arborescence root indicators
    let mut root_vars: Vec<(Var, usize)> = Vec::new(); // (var, z_node_idx)
    for r in 0..num_z {
        if is_leaf_z[r] {
            continue;
        }
        if out_u1[r].is_empty() {
            continue; // can't be a root without outgoing arcs
        }

        let root_r = vm.new_var();
        root_vars.push((root_r, r));
        component_lits.push(root_r.pos_lit());

        // root_r → ∨ out_u1(r)
        {
            let mut clause = vec![root_r.neg_lit()];
            for &ai in &out_u1[r] {
                clause.push(y_u1[ai].pos_lit());
            }
            add_clause(&mut solver, &clause);
            clause_count += 1;
        }

        // root_r → ¬y_in_j for each incoming arc j
        for &(ai, is_u1_arc) in &in_arcs[r] {
            let in_lit = if is_u1_arc {
                y_u1[ai].neg_lit()
            } else {
                y_u2[ai].neg_lit()
            };
            add_clause(&mut solver, &[root_r.neg_lit(), in_lit]);
            clause_count += 1;
        }

        // (∨ out_u1(r)) ∧ (∧ ¬y_in_j) → root_r
        // Contrapositive: ¬root_r → (∧ ¬out_u1(r)) ∨ (∨ y_in_j)
        // i.e., ¬root_r ∨ ¬out_u1_0 ∨ ... ∨ y_in_0 ∨ ...
        // But that's a single clause: we need one for each out_u1 arc.
        // Actually: (∨ out_u1) ∧ (∧ ¬in_j) → root_r
        // is equivalent to: for each out_u1 arc o_i:
        //   o_i ∧ (∧ ¬in_j) → root_r
        //   which is: ¬o_i ∨ (∨ in_j) ∨ root_r
        for &ai in &out_u1[r] {
            let mut clause = vec![y_u1[ai].neg_lit(), root_r.pos_lit()];
            for &(ij, is_u1_in) in &in_arcs[r] {
                clause.push(if is_u1_in {
                    y_u1[ij].pos_lit()
                } else {
                    y_u2[ij].pos_lit()
                });
            }
            add_clause(&mut solver, &clause);
            clause_count += 1;
        }
    }

    // Totalizer for component counting.
    let mut totalizer = Totalizer::default();
    for &lit in &component_lits {
        totalizer.extend([lit]);
    }

    let lb_comps = lb_components;
    let ub_comps = k_max;
    totalizer
        .encode_ub(lb_comps..=ub_comps, &mut solver, &mut vm)
        .unwrap();

    let encode_ms = t_enc.elapsed().as_secs_f64() * 1000.0;
    profile.num_vars = vm.n_used() as usize;
    profile.encode_ms = encode_ms;

    let root_clauses = clause_count - balance_clauses - conservation_clauses - leaf_clauses - capacity_clauses - path_block_clauses;
    eprintln!(
        "[olver] Clauses: {} total (balance={} conserv={} leaf={} capacity={} pathblock={} root={})",
        clause_count,
        balance_clauses,
        conservation_clauses,
        leaf_clauses,
        capacity_clauses,
        path_block_clauses,
        root_clauses
    );
    eprintln!(
        "[olver] Total: {} vars, {} component indicators, encode={:.1}ms",
        profile.num_vars,
        component_lits.len(),
        encode_ms
    );

    // --- Descent loop ---
    let t_total_solve = std::time::Instant::now();
    let mut best_components: Option<Vec<Vec<usize>>> = None;

    for comp_bound in (lb_comps..=ub_comps).rev() {
        let assumps_vec = totalizer.enforce_ub(comp_bound).unwrap();

        profile.sat_calls += 1;
        let t_solve = std::time::Instant::now();
        let result = solver.solve_assumps(&assumps_vec).unwrap();
        let solve_ms = t_solve.elapsed().as_secs_f64() * 1000.0;
        profile.solve_ms += solve_ms;
        let cum_s = t_total_solve.elapsed().as_secs_f64();

        match result {
            SolverResult::Sat => {
                // Extract components by tracing arborescences from roots to leaves.
                let mut leaf_to_comp: Vec<Option<usize>> = vec![None; n];
                let mut components: Vec<Vec<usize>> = Vec::new();

                // First, handle singletons.
                for a in 0..n {
                    if solver.var_val(x[a]).unwrap() == TernaryVal::True {
                        let comp_id = components.len();
                        components.push(vec![a]);
                        leaf_to_comp[a] = Some(comp_id);
                    }
                }

                // Then, trace arborescences from root nodes.
                for &(root_var, r) in &root_vars {
                    if solver.var_val(root_var).unwrap() != TernaryVal::True {
                        continue;
                    }
                    let comp_id = components.len();
                    components.push(Vec::new());

                    // BFS/DFS from r, following active arcs to collect leaves.
                    let mut stack = vec![r];
                    while let Some(node) = stack.pop() {
                        if is_leaf_z[node] {
                            // node = pair_to_idx(a, a); recover a.
                            // For (a, a): idx = a*(a+1)/2 + a = a*(a+3)/2.
                            // We can recover a by iterating or using the inverse.
                            // Simpler: check all leaves.
                            for a in 0..n {
                                if pair_to_idx(a, a) == node {
                                    if leaf_to_comp[a].is_none() {
                                        components[comp_id].push(a);
                                        leaf_to_comp[a] = Some(comp_id);
                                    }
                                    break;
                                }
                            }
                            continue;
                        }

                        // Follow active outgoing arcs (both U1 and U2).
                        for &ai in &out_u1[node] {
                            if solver.var_val(y_u1[ai]).unwrap() == TernaryVal::True {
                                stack.push(u1_arcs[ai].to);
                            }
                        }
                        for &ai in &out_u2[node] {
                            if solver.var_val(y_u2[ai]).unwrap() == TernaryVal::True {
                                stack.push(u2_arcs[ai].to);
                            }
                        }
                    }
                }

                // Any unassigned leaves become singletons (safety net).
                for a in 0..n {
                    if leaf_to_comp[a].is_none() {
                        components.push(vec![a]);
                    }
                }

                // Remove empty components.
                components.retain(|c| !c.is_empty());

                let num_comps = components.len();
                let max_sz = components.iter().map(|c| c.len()).max().unwrap_or(0);
                eprintln!(
                    "[olver] k={} SAT {:.1}ms (cum {:.1}s) comps={} max_size={}",
                    comp_bound, solve_ms, cum_s, num_comps, max_sz
                );
                best_components = Some(components);
            }
            SolverResult::Unsat => {
                eprintln!(
                    "[olver] k={} UNSAT {:.1}ms (cum {:.1}s) — optimal={}",
                    comp_bound, solve_ms, cum_s, comp_bound + 1
                );
                profile.optimal_k =
                    best_components.as_ref().map(|c| c.len()).unwrap_or(k_max);
                profile.report();
                return best_components.map(|c| {
                    components_to_trees(&c, &instance.trees[0], instance.num_leaves)
                });
            }
            SolverResult::Interrupted => {
                profile.report();
                return best_components.map(|c| {
                    components_to_trees(&c, &instance.trees[0], instance.num_leaves)
                });
            }
        }
    }

    profile.optimal_k = best_components.as_ref().map(|c| c.len()).unwrap_or(k_max);
    profile.report();
    best_components.map(|c| components_to_trees(&c, &instance.trees[0], instance.num_leaves))
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
    solve_sat_inner_impl(instance, stats, preferred_singleton_labels, false, false)
}

fn solve_sat_olver(instance: &Instance, stats: &mut SolverStats) -> Option<Vec<Tree>> {
    solve_sat_inner_impl(instance, stats, vec![], false, true)
}

fn solve_sat_inner_impl(
    instance: &Instance,
    stats: &mut SolverStats,
    preferred_singleton_labels: Vec<u32>,
    skip_cluster_decomp: bool,
    use_olver: bool,
) -> Option<Vec<Tree>> {
    let n = instance.num_leaves as usize;
    let mut profile = SolveProfile::default();
    profile.n = n;

    let kern_config = KernelizeConfig {
        protected_labels: preferred_singleton_labels.clone(),
        ..KernelizeConfig::default()
    };
    let kern = kernelize::kernelize_best(instance, &kern_config);
    let reduced = &kern.instance;
    let param_reduction_32 = kern.param_reduction;

    let n_reduced = reduced.num_leaves as usize;
    if kern.stats.reduced_leaves < instance.num_leaves {
        let total =
            kern.stats.subtree_removed() + kern.stats.chain_removed() + kern.stats.chain32_removed();
        eprintln!(
            "[sat] Kernelized: {} → {} leaves ({} removed: {} subtree, {} chain, {} 3-2)",
            n,
            n_reduced,
            total,
            kern.stats.subtree_removed(),
            kern.stats.chain_removed(),
            kern.stats.chain32_removed(),
        );
    }

    if n_reduced <= 1 {
        stats.lower_bound = 1 + param_reduction_32;
        stats.upper_bound = Some(1 + param_reduction_32);
        // The single surviving leaf is one component; expand back to original space.
        let trivial = vec![reduced.trees[0].clone()];
        let components =
            kernelize::expand_solution(trivial, &kern, &instance.trees[0], instance.num_leaves);
        return Some(components);
    }

    if !skip_cluster_decomp {
        match cluster_reduction::try_cluster_reduction(reduced, &mut |subinstance| {
            let mut sub_stats = SolverStats::default();
            solve_sat_inner(subinstance, &mut sub_stats, vec![])
        })? {
            ClusterReductionResult::NotApplicable => {}
            ClusterReductionResult::Solved(solution) => {
                profile.cluster_splits += 1;
                eprintln!(
                    "[sat] Cluster decomposition: {} = {} + {}",
                    n_reduced, solution.cluster_size, solution.rest_size
                );
                let exact_k = solution.components.len() + param_reduction_32;
                profile.bounds_computed = (exact_k, exact_k);
                profile.optimal_k = exact_k;
                profile.report();
                stats.lower_bound = exact_k;
                stats.upper_bound = Some(exact_k);
                let components = kernelize::expand_solution(
                    solution.components,
                    &kern,
                    &instance.trees[0],
                    instance.num_leaves,
                );
                return Some(components);
            }
        }
    }

    // Compute LB, UB, and best partition via the shared maf_bounds pipeline.
    // This uses 20 seeds per ref_idx for multi-tree (vs the old 4), and returns
    // the best greedy partition for use as SAT phase hints.
    let t_bounds = std::time::Instant::now();
    let bounds = maf_bounds(&reduced.trees, reduced.num_leaves);
    let ub_components = bounds.upper;
    let bounds_ms = t_bounds.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[sat] Greedy bounds: LB={}, UB={} in {:.1}ms (n'={}, m={})",
        bounds.lower, ub_components, bounds_ms, n_reduced, reduced.num_trees()
    );

    // For multi-tree instances, try to tighten LB via exact pairwise distances + additive formula.
    // Uses the SAT solver itself for pairwise solves (faster than FPT).
    let lb_components = if reduced.trees.len() >= 3 && bounds.upper > bounds.lower {
        let t_exact = std::time::Instant::now();
        let exact_lb = klados_core::lower_bound::exact_pairwise_lower_bound(
            &reduced.trees,
            reduced.num_leaves,
            bounds.lower,
            bounds.upper,
            std::time::Duration::from_secs(3),
            &mut |pair| {
                let mut sub_stats = SolverStats::default();
                solve_sat_inner(pair, &mut sub_stats, vec![]).map(|trees| trees.len())
            },
        );
        let exact_ms = t_exact.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[sat] Exact pairwise LB: {} (was {}) in {:.1}ms",
            exact_lb, bounds.lower, exact_ms
        );
        exact_lb
    } else {
        bounds.lower
    };

    eprintln!(
        "[sat] Final bounds: LB={}, UB={}, gap={}",
        lb_components, ub_components, ub_components - lb_components
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

    // Choose encoding: Olver LP* formulation for 2-tree instances, or cut-based encoding.
    let components = if use_olver && reduced.num_trees() == 2 {
        sat_solve_maf_olver(
            reduced,
            ub_components,
            lb_components,
            &mut profile,
        )?
    } else {
        sat_solve_maf_cut(
            reduced,
            ub_components,
            lb_components,
            &mut profile,
        )?
    };

    // Expand solution back to original label space.
    let components =
        kernelize::expand_solution(components, &kern, &instance.trees[0], instance.num_leaves);

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

    #[test]
    fn test_olver_solve_identical_trees() {
        let t = make_tree_1234_cherry12_34();
        let instance = Instance::new(vec![t.clone(), t], 4);

        let mut solver = MafSatOlverSolver::new();
        let result = solver.solve(&instance);
        assert!(result.is_some());
        let components = result.unwrap();
        assert_eq!(components.len(), 1);
    }

    #[test]
    fn test_olver_solve_conflicting_trees() {
        let instance = make_test_instance();

        let mut solver = MafSatOlverSolver::new();
        let result = solver.solve(&instance);
        assert!(result.is_some());
        let components = result.unwrap();
        assert_eq!(components.len(), 3);
    }
}
