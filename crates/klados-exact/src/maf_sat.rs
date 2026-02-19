//! In-process SAT-based exact MAF solver using rustsat-cadical.
//!
//! This approach eliminates process spawning, DIMACS serialization, and stdout parsing
//! by using the CaDiCaL SAT solver directly via FFI.
//!
//! The optimization uses iterative SAT calls with a totalizer encoding:
//! - Instead of general MaxSAT, we exploit that soft clauses are just K unit clauses (¬u_i)
//! - A totalizer encodes "at most B of the u_i are true" as O(K log K) clauses
//! - We linear search downward to find the minimum number of components

use fixedbitset::FixedBitSet;
use klados_core::tree::{Label, NodeId, Tree};
use klados_core::{Instance, SolverStats};
use rustsat::encodings::card::{BoundUpper, Totalizer};
use rustsat::instances::{BasicVarManager, ManageVars};
use rustsat::solvers::{Solve, SolveIncremental, SolverResult};
use rustsat::types::{Clause, Lit, TernaryVal, Var};
use rustsat_cadical::CaDiCaL;

use crate::ExactSolver;

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

struct MafEncoding {
    l: Vec<Vec<Var>>,
    u: Vec<Var>,
    w: Vec<Vec<Vec<Var>>>,
    s: Vec<Vec<Var>>,
    t: Vec<Vec<Vec<Var>>>,
    n: usize,
    k: usize,
    m: usize,
}

fn solve_sat(instance: &Instance, stats: &mut SolverStats) -> Option<Vec<Tree>> {
    let n = instance.num_leaves as usize;
    let k = n;
    let m = instance.num_trees();

    let mut solver = CaDiCaL::default();
    let mut vm = BasicVarManager::default();

    let encoding = create_variables(&mut vm, n, k, m);
    add_hard_clauses(&mut solver, instance, &encoding);

    let components = optimize_components(&mut solver, &mut vm, &encoding, n, k)?;

    let result = extract_forest(instance, &components, n);

    stats.lower_bound = components.len();
    stats.upper_bound = Some(components.len());

    Some(result)
}

fn create_variables(vm: &mut BasicVarManager, n: usize, k: usize, m: usize) -> MafEncoding {
    let mut l = vec![vec![Var::new(0); n]; k];
    for i in 0..k {
        for j in 0..n {
            l[i][j] = vm.new_var();
        }
    }

    let mut u = vec![Var::new(0); k];
    for i in 0..k {
        u[i] = vm.new_var();
    }

    let num_internal = n - 1;
    let mut w = vec![vec![vec![Var::new(0); num_internal]; k]; m];
    for q in 0..m {
        for i in 0..k {
            for v in 0..num_internal {
                w[q][i][v] = vm.new_var();
            }
        }
    }

    let mut s = vec![vec![Var::new(0); n]; k];
    for i in 0..k {
        for j in 0..n {
            s[i][j] = vm.new_var();
        }
    }

    let mut t = vec![vec![vec![Var::new(0); num_internal]; k]; m];
    for q in 0..m {
        for i in 0..k {
            for v in 0..num_internal {
                t[q][i][v] = vm.new_var();
            }
        }
    }

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

fn add_hard_clauses(solver: &mut CaDiCaL, instance: &Instance, enc: &MafEncoding) {
    let n = enc.n;
    let k = enc.k;
    let m = enc.m;
    let num_internal = n - 1;

    // H1a: Every leaf in at least one component
    for j in 0..n {
        let clause: Vec<Lit> = (0..k).map(|i| enc.l[i][j].pos_lit()).collect();
        add_clause(solver, &clause);
    }

    // H1b: Every leaf in at most one component (ladder encoding with auxiliary s)
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

    // H2: Every internal vertex belongs to at most one component (ladder encoding with t)
    for q in 0..m {
        for v in 0..num_internal {
            for i in 0..k - 1 {
                add_clause(
                    solver,
                    &[enc.w[q][i][v].neg_lit(), enc.t[q][i][v].pos_lit()],
                );
            }

            for i in 1..k - 1 {
                add_clause(
                    solver,
                    &[enc.t[q][i - 1][v].neg_lit(), enc.t[q][i][v].pos_lit()],
                );
            }

            for i in 1..k {
                add_clause(
                    solver,
                    &[enc.t[q][i - 1][v].neg_lit(), enc.w[q][i][v].neg_lit()],
                );
            }
        }
    }

    // H3: Path consistency
    for q in 0..m {
        let tree = &instance.trees[q];
        for i in 0..k {
            for j in 0..n {
                for j2 in (j + 1)..n {
                    let label_j = (j + 1) as Label;
                    let label_j2 = (j2 + 1) as Label;
                    let node_j = tree.node_by_label(label_j);
                    let node_j2 = tree.node_by_label(label_j2);
                    let nca = tree.nearest_common_ancestor(node_j, node_j2);

                    let mut cur = node_j;
                    while cur != nca {
                        if !tree.is_leaf(cur) {
                            let internal_idx = internal_node_index(tree, cur);
                            add_clause(
                                solver,
                                &[
                                    enc.l[i][j].neg_lit(),
                                    enc.l[i][j2].neg_lit(),
                                    enc.w[q][i][internal_idx].pos_lit(),
                                ],
                            );
                        }
                        cur = tree.parent[cur as usize];
                    }

                    cur = node_j2;
                    while cur != nca {
                        if !tree.is_leaf(cur) {
                            let internal_idx = internal_node_index(tree, cur);
                            add_clause(
                                solver,
                                &[
                                    enc.l[i][j].neg_lit(),
                                    enc.l[i][j2].neg_lit(),
                                    enc.w[q][i][internal_idx].pos_lit(),
                                ],
                            );
                        }
                        cur = tree.parent[cur as usize];
                    }
                }
            }
        }
    }

    // H4: Incompatible triples
    for a in 0..n {
        for b in (a + 1)..n {
            for c in (b + 1)..n {
                let label_a = (a + 1) as Label;
                let label_b = (b + 1) as Label;
                let label_c = (c + 1) as Label;

                if is_incompatible_triple(instance, label_a, label_b, label_c) {
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
                }
            }
        }
    }

    // H5: Usage tracking
    for i in 0..k {
        for j in 0..n {
            add_clause(solver, &[enc.l[i][j].neg_lit(), enc.u[i].pos_lit()]);
        }
    }

    // Symmetry breaking O1: u[i] >= u[i+1]
    for i in 0..k - 1 {
        add_clause(solver, &[enc.u[i + 1].neg_lit(), enc.u[i].pos_lit()]);
    }

    // O2: Leaf 1 is always in component 1
    add_clause(solver, &[enc.l[0][0].pos_lit()]);

    // O3: Lexicographic ordering of first leaf indices
    for i in 0..k - 1 {
        for j in 0..n {
            let mut lits = vec![enc.l[i + 1][j].neg_lit()];
            for jp in 0..j {
                lits.push(enc.l[i][jp].pos_lit());
            }
            add_clause(solver, &lits);
        }
    }
}

fn internal_node_index(tree: &Tree, node: NodeId) -> usize {
    let mut count = 0;
    for v in 0..tree.num_nodes() as NodeId {
        if !tree.is_leaf(v) && v < node {
            count += 1;
        } else if !tree.is_leaf(v) && v == node {
            return count;
        }
    }
    count
}

fn is_incompatible_triple(instance: &Instance, a: Label, b: Label, c: Label) -> bool {
    let mut first_odd: Option<Label> = None;

    for tree in &instance.trees {
        let node_a = tree.node_by_label(a);
        let node_b = tree.node_by_label(b);
        let node_c = tree.node_by_label(c);

        let nca_ab = tree.nearest_common_ancestor(node_a, node_b);
        let nca_ac = tree.nearest_common_ancestor(node_a, node_c);
        let nca_bc = tree.nearest_common_ancestor(node_b, node_c);

        let depth_ab = tree.depth[nca_ab as usize];
        let depth_ac = tree.depth[nca_ac as usize];
        let depth_bc = tree.depth[nca_bc as usize];

        let odd_leaf = if depth_ab > depth_ac && depth_ab > depth_bc {
            c
        } else if depth_ac > depth_ab && depth_ac > depth_bc {
            b
        } else {
            a
        };

        if let Some(first) = first_odd {
            if odd_leaf != first {
                return true;
            }
        } else {
            first_odd = Some(odd_leaf);
        }
    }

    false
}

fn optimize_components(
    solver: &mut CaDiCaL,
    vm: &mut BasicVarManager,
    enc: &MafEncoding,
    n: usize,
    k: usize,
) -> Option<Vec<Vec<usize>>> {
    let u_lits: Vec<Lit> = enc.u.iter().map(|v| v.pos_lit()).collect();

    let mut totalizer = Totalizer::default();
    for lit in u_lits {
        totalizer.extend([lit]);
    }

    let mut best_model: Option<Vec<Vec<usize>>> = None;

    for bound in (1..k).rev() {
        totalizer.encode_ub(bound..=bound, solver, vm).unwrap();

        let assumps = totalizer.enforce_ub(bound).unwrap();

        match solver.solve_assumps(&assumps).unwrap() {
            SolverResult::Sat => {
                best_model = Some(extract_components(solver, enc, n, k));
            }
            SolverResult::Unsat => break,
            SolverResult::Interrupted => break,
        }
    }

    best_model
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

fn extract_forest(instance: &Instance, components: &[Vec<usize>], n: usize) -> Vec<Tree> {
    let ref_tree = instance.reference_tree();

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
