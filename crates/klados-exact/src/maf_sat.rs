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
        let total =
            kern.stats.subtree_removed + kern.stats.chain_removed + kern.stats.chain32_removed;
        eprintln!(
            "[sat] Kernelized: {} → {} leaves ({} removed: {} subtree, {} chain, {} 3-2)",
            n,
            n_reduced,
            total,
            kern.stats.subtree_removed,
            kern.stats.chain_removed,
            kern.stats.chain32_removed,
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

    // Use cut-based encoding (k-independent, much smaller formula).
    let components = sat_solve_maf_cut(
        reduced,
        ub_components,
        lb_components,
        &mut profile,
    )?;

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
}
