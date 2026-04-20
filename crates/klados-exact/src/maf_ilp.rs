//! ILP-based exact solver for Maximum Agreement Forest using HiGHS.
//!
//! Formulation:
//!
//! **Variables:**
//! - `c_i[v]` ∈ {0,1} for each tree T_i, each non-root node v: is edge (parent(v),v) cut?
//! - `s[x,y]` ∈ {0,1} for each leaf pair x < y: are x,y in the same component?
//!
//! **Objective:** minimize Σ c_1[v] over non-root nodes of T_1
//!
//! **Constraints:**
//! - C1a: c_i[v] + s[x,y] ≤ 1  for each tree i, pair (x,y), node v on Path_i(x,y)
//! - C1b: Σ c_i[v on path] + s[x,y] ≥ 1  for each tree i, pair (x,y)
//! - C2:  s[x,y] + s[x,z] + s[y,z] ≤ 1  for each conflicting triple (x,y,z)
//!
//! C1a/C1b enforce non-overlap (convexity) in each tree.
//! C2 enforces compatibility (no incompatible triple within a component).
//! Transitivity of s is implied by C1a/C1b and is not explicitly added.

use fixedbitset::FixedBitSet;
use highs::{Col, HighsModelStatus, RowProblem, Sense};
use klados_core::tree::{Label, NONE, NodeId, Tree};
use klados_core::{Instance, SolverStats};

// ---------------------------------------------------------------------------
// Index helpers
// ---------------------------------------------------------------------------

/// Triangular index for unordered pair (x, y) with 1 ≤ x < y ≤ n.
/// Maps to a contiguous range 0..n*(n-1)/2.
/// (1,2)->0, (1,3)->1, (2,3)->2, (1,4)->3, (2,4)->4, (3,4)->5, ...
#[inline]
fn pair_index(x: Label, y: Label) -> usize {
    debug_assert!(x >= 1);
    debug_assert!(x < y);
    let y = y as usize;
    let x = x as usize;
    (y - 1) * (y - 2) / 2 + (x - 1)
}

/// Number of unordered pairs for n items (1-based labels 1..=n).
#[inline]
fn num_pairs(n: u32) -> usize {
    let n = n as usize;
    n * (n - 1) / 2
}

// ---------------------------------------------------------------------------
// LCA and path computation
// ---------------------------------------------------------------------------

/// Compute LCA of two nodes in a tree using naive depth-walk.
/// O(depth) time. Fine for balanced trees up to n~1000.
fn lca(tree: &Tree, mut a: NodeId, mut b: NodeId) -> NodeId {
    // Walk the deeper node up until both are at the same depth.
    while tree.depth[a as usize] > tree.depth[b as usize] {
        a = tree.parent[a as usize];
    }
    while tree.depth[b as usize] > tree.depth[a as usize] {
        b = tree.parent[b as usize];
    }
    // Now walk both up until they meet.
    while a != b {
        a = tree.parent[a as usize];
        b = tree.parent[b as usize];
    }
    a
}

/// Collect nodes on the path from leaf x to leaf y in `tree` whose edges,
/// when cut, would separate x from y. Each returned node v represents the
/// edge (parent(v), v).
///
/// Specifically: returns all nodes strictly between x and lca(x,y), and
/// between y and lca(x,y), but NOT the LCA itself. Cutting the edge above
/// the LCA separates {x,y} from the rest of the tree, but does not
/// separate x from y. Only edges below the LCA on the x-side or y-side
/// actually separate x from y.
///
/// Returns nodes in no particular order.
fn path_nodes(tree: &Tree, x: Label, y: Label) -> Vec<NodeId> {
    let a = tree.label_to_node[x as usize];
    let b = tree.label_to_node[y as usize];
    let l = lca(tree, a, b);

    let mut result = Vec::new();

    // Walk from a up to (but not including) l — collect each node we visit.
    let mut cur = a;
    while cur != l {
        result.push(cur);
        cur = tree.parent[cur as usize];
    }
    // Walk from b up to (but not including) l.
    cur = b;
    while cur != l {
        result.push(cur);
        cur = tree.parent[cur as usize];
    }
    // Do NOT include the LCA: cutting the edge above the LCA does not
    // separate x from y (they are both in the LCA's subtree).

    result
}

// ---------------------------------------------------------------------------
// Triplet topology and conflict detection
// ---------------------------------------------------------------------------

/// Determine the rooted triplet topology of (x, y, z) in `tree`.
/// Returns 0 if xy|z (x,y are the cherry), 1 if xz|y, 2 if yz|x,
/// or 3 if there's a polytomy (all three meet at one node — shouldn't
/// happen in binary trees, but handled for safety).
fn triplet_topology(tree: &Tree, x: Label, y: Label, z: Label) -> u8 {
    let nx = tree.label_to_node[x as usize];
    let ny = tree.label_to_node[y as usize];
    let nz = tree.label_to_node[z as usize];

    let lxy = lca(tree, nx, ny);
    let lxz = lca(tree, nx, nz);
    let lyz = lca(tree, ny, nz);

    let dxy = tree.depth[lxy as usize];
    let dxz = tree.depth[lxz as usize];
    let dyz = tree.depth[lyz as usize];

    // The deepest LCA corresponds to the cherry pair.
    if dxy > dxz && dxy > dyz {
        0 // xy|z
    } else if dxz > dxy && dxz > dyz {
        1 // xz|y
    } else if dyz > dxy && dyz > dxz {
        2 // yz|x
    } else {
        3 // polytomy / degenerate
    }
}

/// Find all conflicting triples: triples (x,y,z) with x < y < z where at
/// least two trees disagree on the topology.
fn find_conflict_triples(trees: &[Tree], n: u32) -> Vec<(Label, Label, Label)> {
    let mut conflicts = Vec::new();
    for x in 1..=n {
        for y in (x + 1)..=n {
            for z in (y + 1)..=n {
                let topo0 = triplet_topology(&trees[0], x, y, z);
                let mut conflict = false;
                for tree in &trees[1..] {
                    let topo = triplet_topology(tree, x, y, z);
                    if topo != topo0 {
                        conflict = true;
                        break;
                    }
                }
                if conflict {
                    conflicts.push((x, y, z));
                }
            }
        }
    }
    conflicts
}

// ---------------------------------------------------------------------------
// Union-Find for solution extraction
// ---------------------------------------------------------------------------

struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            self.parent[x as usize] = self.parent[self.parent[x as usize] as usize];
            x = self.parent[x as usize];
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra as usize] < self.rank[rb as usize] {
            self.parent[ra as usize] = rb;
        } else if self.rank[ra as usize] > self.rank[rb as usize] {
            self.parent[rb as usize] = ra;
        } else {
            self.parent[rb as usize] = ra;
            self.rank[ra as usize] += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// ILP builder and solver
// ---------------------------------------------------------------------------

/// Build and solve the MAF ILP for the given instance.
/// Returns `Some(components)` if an optimal solution is found, `None` otherwise.
fn solve_ilp(instance: &Instance) -> Option<(Vec<Tree>, SolverStats)> {
    let n = instance.num_leaves;
    let k = instance.num_trees();
    let trees = &instance.trees;

    // --- Precompute conflict triples ---
    let conflict_triples = find_conflict_triples(trees, n);

    // --- Precompute path nodes for all (tree, pair) combinations ---
    // paths[tree_idx][pair_index(x,y)] = Vec<NodeId>
    let mut paths: Vec<Vec<Vec<NodeId>>> = Vec::with_capacity(k);
    for tree in trees.iter() {
        let mut tree_paths = vec![Vec::new(); num_pairs(n)];
        for x in 1..=n {
            for y in (x + 1)..=n {
                tree_paths[pair_index(x, y)] = path_nodes(tree, x, y);
            }
        }
        paths.push(tree_paths);
    }

    // --- Build the ILP ---
    let mut pb = RowProblem::default();
    let mut col_count: usize = 0; // track column index for solution readback

    // 1. Create c_i[v] variables: binary, obj coeff = 1.0 for tree 0, else 0.0
    // cut_vars[tree_idx] maps node_id -> (Col, col_index) (only for non-root nodes)
    let mut cut_vars: Vec<Vec<Option<(Col, usize)>>> = Vec::with_capacity(k);
    for (tree_idx, tree) in trees.iter().enumerate() {
        let num_nodes = tree.num_nodes();
        let mut vars: Vec<Option<(Col, usize)>> = vec![None; num_nodes];
        for node in 0..num_nodes as u32 {
            if tree.parent[node as usize] == NONE {
                continue; // root — no edge above it
            }
            let obj = if tree_idx == 0 { 1.0 } else { 0.0 };
            let col = pb.add_integer_column(obj, 0.0..=1.0);
            vars[node as usize] = Some((col, col_count));
            col_count += 1;
        }
        cut_vars.push(vars);
    }

    // 2. Create s[x,y] variables: binary, obj coeff = 0.0
    let np = num_pairs(n);
    let mut same_vars: Vec<(Col, usize)> = Vec::with_capacity(np);
    for _ in 0..np {
        let col = pb.add_integer_column(0.0, 0.0..=1.0);
        same_vars.push((col, col_count));
        col_count += 1;
    }

    // 3. C1a constraints: c_i[v] + s[x,y] <= 1
    //    For each tree i, each pair (x,y), each node v on Path_i(x,y).
    for (tree_idx, tree_paths) in paths.iter().enumerate() {
        for x in 1..=n {
            for y in (x + 1)..=n {
                let pidx = pair_index(x, y);
                let (s_col, _) = same_vars[pidx];
                for &v in &tree_paths[pidx] {
                    if let Some((c_col, _)) = cut_vars[tree_idx][v as usize] {
                        pb.add_row(..=1.0, [(c_col, 1.0), (s_col, 1.0)]);
                    }
                }
            }
        }
    }

    // 4. C1b constraints: sum(c_i[v] on path) + s[x,y] >= 1
    //    For each tree i, each pair (x,y).
    for (tree_idx, tree_paths) in paths.iter().enumerate() {
        for x in 1..=n {
            for y in (x + 1)..=n {
                let pidx = pair_index(x, y);
                let (s_col, _) = same_vars[pidx];
                let path = &tree_paths[pidx];
                let mut coeffs: Vec<(Col, f64)> = Vec::with_capacity(path.len() + 1);
                for &v in path {
                    if let Some((c_col, _)) = cut_vars[tree_idx][v as usize] {
                        coeffs.push((c_col, 1.0));
                    }
                }
                coeffs.push((s_col, 1.0));
                pb.add_row(1.0.., &coeffs);
            }
        }
    }

    // 5. C2 constraints: s[x,y] + s[x,z] + s[y,z] <= 1
    //    For each conflicting triple.
    for &(x, y, z) in &conflict_triples {
        let (sxy, _) = same_vars[pair_index(x, y)];
        let (sxz, _) = same_vars[pair_index(x, z)];
        let (syz, _) = same_vars[pair_index(y, z)];
        pb.add_row(..=1.0, [(sxy, 1.0), (sxz, 1.0), (syz, 1.0)]);
    }

    // --- Configure and solve ---
    let mut model = pb.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_option("threads", 1_i32); // single-threaded for PACE
    model.set_option("presolve", "on");

    let solved = model.solve();

    let status = solved.status();
    if status != HighsModelStatus::Optimal {
        return None;
    }

    let solution = solved.get_solution();
    let col_values = solution.columns();

    // --- Extract solution: build partition from s[x,y] ---
    let mut uf = UnionFind::new((n + 1) as usize); // labels 1..=n
    for x in 1..=n {
        for y in (x + 1)..=n {
            let pidx = pair_index(x, y);
            let (_, s_idx) = same_vars[pidx];
            let s_val = col_values[s_idx];
            if s_val > 0.5 {
                uf.union(x, y);
            }
        }
    }

    // Group leaves by component root
    let mut component_map: std::collections::HashMap<u32, FixedBitSet> =
        std::collections::HashMap::new();
    for label in 1..=n {
        let root = uf.find(label);
        component_map
            .entry(root)
            .or_insert_with(|| FixedBitSet::with_capacity((n + 1) as usize))
            .insert(label as usize);
    }

    // Build output trees: prune reference tree to each component's leaf set
    let ref_tree = instance.reference_tree();
    let mut components: Vec<Tree> = component_map
        .values()
        .map(|leafset| ref_tree.prune_to_leafset(leafset))
        .collect();

    // Sort components so the one containing label 1 (usually the rho leaf) is first.
    // This follows the convention that the component containing the root-adjacent
    // leaf comes first.
    components.sort_by_key(|t| {
        // smallest label in this component
        (1..=n)
            .find(|&l| t.label_to_node[l as usize] != NONE)
            .unwrap_or(n + 1)
    });

    // Compute stats: count cuts in tree 0
    let num_cuts: usize = cut_vars[0]
        .iter()
        .filter_map(|entry| entry.as_ref())
        .filter(|(_, idx)| col_values[*idx] > 0.5)
        .count();

    let stats = SolverStats {
        nodes_explored: 0,
        branches_pruned: 0,
        lower_bound: num_cuts,
        upper_bound: Some(num_cuts),
    };

    Some((components, stats))
}

// ---------------------------------------------------------------------------
// LP relaxation
// ---------------------------------------------------------------------------

/// Solve the LP relaxation of the MAF ILP and return the fractional lower bound.
/// Uses continuous (not integer) variables and the tightened C2 ≤ 1 constraint.
/// Returns Some(fractional_cuts) where fractional_cuts is a lower bound on OPT cuts.
pub fn solve_lp_relaxation(instance: &Instance) -> Option<f64> {
    let n = instance.num_leaves;
    let k = instance.num_trees();
    let trees = &instance.trees;

    if n <= 1 {
        return Some(0.0);
    }

    // --- Precompute conflict triples ---
    let conflict_triples = find_conflict_triples(trees, n);

    // --- Precompute path nodes for all (tree, pair) combinations ---
    let mut paths: Vec<Vec<Vec<NodeId>>> = Vec::with_capacity(k);
    for tree in trees.iter() {
        let mut tree_paths = vec![Vec::new(); num_pairs(n)];
        for x in 1..=n {
            for y in (x + 1)..=n {
                tree_paths[pair_index(x, y)] = path_nodes(tree, x, y);
            }
        }
        paths.push(tree_paths);
    }

    // --- Build the LP (continuous relaxation) ---
    let mut pb = RowProblem::default();

    // 1. Create c_i[v] variables: continuous [0,1], obj coeff = 1.0 for tree 0
    let mut cut_vars: Vec<Vec<Option<Col>>> = Vec::with_capacity(k);
    for (tree_idx, tree) in trees.iter().enumerate() {
        let num_nodes = tree.num_nodes();
        let mut vars: Vec<Option<Col>> = vec![None; num_nodes];
        for node in 0..num_nodes as u32 {
            if tree.parent[node as usize] == NONE {
                continue;
            }
            let obj = if tree_idx == 0 { 1.0 } else { 0.0 };
            let col = pb.add_column(obj, 0.0..=1.0);
            vars[node as usize] = Some(col);
        }
        cut_vars.push(vars);
    }

    // 2. Create s[x,y] variables: continuous [0,1], obj coeff = 0.0
    let np = num_pairs(n);
    let mut same_vars: Vec<Col> = Vec::with_capacity(np);
    for _ in 0..np {
        let col = pb.add_column(0.0, 0.0..=1.0);
        same_vars.push(col);
    }

    // 3. C1a constraints: c_i[v] + s[x,y] <= 1
    for (tree_idx, tree_paths) in paths.iter().enumerate() {
        for x in 1..=n {
            for y in (x + 1)..=n {
                let pidx = pair_index(x, y);
                let s_col = same_vars[pidx];
                for &v in &tree_paths[pidx] {
                    if let Some(c_col) = cut_vars[tree_idx][v as usize] {
                        pb.add_row(..=1.0, [(c_col, 1.0), (s_col, 1.0)]);
                    }
                }
            }
        }
    }

    // 4. C1b constraints: sum(c_i[v] on path) + s[x,y] >= 1
    for (tree_idx, tree_paths) in paths.iter().enumerate() {
        for x in 1..=n {
            for y in (x + 1)..=n {
                let pidx = pair_index(x, y);
                let s_col = same_vars[pidx];
                let path = &tree_paths[pidx];
                let mut coeffs: Vec<(Col, f64)> = Vec::with_capacity(path.len() + 1);
                for &v in path {
                    if let Some(c_col) = cut_vars[tree_idx][v as usize] {
                        coeffs.push((c_col, 1.0));
                    }
                }
                coeffs.push((s_col, 1.0));
                pb.add_row(1.0.., &coeffs);
            }
        }
    }

    // 5. C2 constraints: s[x,y] + s[x,z] + s[y,z] <= 1
    for &(x, y, z) in &conflict_triples {
        let sxy = same_vars[pair_index(x, y)];
        let sxz = same_vars[pair_index(x, z)];
        let syz = same_vars[pair_index(y, z)];
        pb.add_row(..=1.0, [(sxy, 1.0), (sxz, 1.0), (syz, 1.0)]);
    }

    // --- Configure and solve ---
    let mut model = pb.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_option("threads", 1_i32);
    model.set_option("presolve", "on");

    let solved = model.solve();

    if solved.status() != HighsModelStatus::Optimal {
        return None;
    }

    Some(solved.objective_value())
}

// ---------------------------------------------------------------------------
// Public solver struct implementing ExactSolver
// ---------------------------------------------------------------------------

/// ILP-based exact MAF solver using HiGHS.
pub struct MafIlpSolver {
    stats: SolverStats,
    /// Maximum number of leaves the ILP solver will attempt.
    /// Beyond this, the constraint count becomes too large.
    max_leaves: u32,
}

impl Default for MafIlpSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl MafIlpSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
            max_leaves: 200,
        }
    }
}

impl super::ExactSolver for MafIlpSolver {
    fn name(&self) -> &'static str {
        "maf-ilp"
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        if instance.num_leaves > self.max_leaves {
            eprintln!(
                "maf-ilp: instance has {} leaves, exceeding limit of {}",
                instance.num_leaves, self.max_leaves
            );
            return None;
        }

        if instance.num_leaves <= 1 {
            // Trivial: single leaf or empty
            return Some(instance.trees[0..1].to_vec());
        }

        match solve_ilp(instance) {
            Some((components, stats)) => {
                self.stats = stats;
                Some(components)
            }
            None => {
                eprintln!("maf-ilp: solver did not find optimal solution");
                None
            }
        }
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExactSolver;

    /// Manually build tree ((1,2),(3,4)):
    ///       0 (root, node 4)
    ///      / \
    ///     3   4  (internal nodes 2, 3)
    ///    / \ / \
    ///   1  2 3  4  (leaves: nodes 0,1,2,3 with labels 1,2,3,4)
    fn tree_1234_cherry12_34(n: u32) -> Tree {
        // Arena layout: allocate nodes 0..6 for n=4 (7 nodes = 2*4-1)
        assert_eq!(n, 4);
        let mut t = Tree::with_capacity(4);

        // We need to push nodes into the arrays.
        // node 0: leaf label=1
        // node 1: leaf label=2
        // node 2: internal (parent of 0,1)
        // node 3: leaf label=3
        // node 4: leaf label=4
        // node 5: internal (parent of 3,4)
        // node 6: root (parent of 2,5)
        let nodes = 7;
        t.parent = vec![NONE; nodes];
        t.left = vec![NONE; nodes];
        t.right = vec![NONE; nodes];
        t.label = vec![0; nodes];

        // Leaves
        t.label[0] = 1;
        t.label[1] = 2;
        t.label[3] = 3;
        t.label[4] = 4;

        // Internal node 2: children 0, 1
        t.left[2] = 0;
        t.right[2] = 1;
        t.parent[0] = 2;
        t.parent[1] = 2;

        // Internal node 5: children 3, 4
        t.left[5] = 3;
        t.right[5] = 4;
        t.parent[3] = 5;
        t.parent[4] = 5;

        // Root node 6: children 2, 5
        t.left[6] = 2;
        t.right[6] = 5;
        t.parent[2] = 6;
        t.parent[5] = 6;

        t.root = 6;
        t.num_leaves = 4;
        t.label_to_node = vec![NONE; 5]; // index 0 unused
        t.label_to_node[1] = 0;
        t.label_to_node[2] = 1;
        t.label_to_node[3] = 3;
        t.label_to_node[4] = 4;

        t.compute_metadata();
        t
    }

    /// Build tree ((1,3),(2,4))
    fn tree_1234_cherry13_24(n: u32) -> Tree {
        assert_eq!(n, 4);
        let nodes = 7;
        let mut t = Tree::with_capacity(4);
        t.parent = vec![NONE; nodes];
        t.left = vec![NONE; nodes];
        t.right = vec![NONE; nodes];
        t.label = vec![0; nodes];

        // Leaves
        t.label[0] = 1;
        t.label[1] = 3;
        t.label[3] = 2;
        t.label[4] = 4;

        // Internal node 2: children 0(label=1), 1(label=3)
        t.left[2] = 0;
        t.right[2] = 1;
        t.parent[0] = 2;
        t.parent[1] = 2;

        // Internal node 5: children 3(label=2), 4(label=4)
        t.left[5] = 3;
        t.right[5] = 4;
        t.parent[3] = 5;
        t.parent[4] = 5;

        // Root node 6: children 2, 5
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
    fn test_pair_index() {
        assert_eq!(pair_index(1, 2), 0);
        assert_eq!(pair_index(1, 3), 1);
        assert_eq!(pair_index(2, 3), 2);
        assert_eq!(pair_index(1, 4), 3);
        assert_eq!(pair_index(2, 4), 4);
        assert_eq!(pair_index(3, 4), 5);
    }

    #[test]
    fn test_lca_simple() {
        let t = tree_1234_cherry12_34(4);
        // LCA(1,2) should be the internal node that is parent of both
        let n1 = t.label_to_node[1]; // node 0
        let n2 = t.label_to_node[2]; // node 1
        let l = lca(&t, n1, n2);
        assert_eq!(l, 2); // internal node 2

        // LCA(1,3) should be root
        let n3 = t.label_to_node[3]; // node 3
        let l = lca(&t, n1, n3);
        assert_eq!(l, 6); // root
    }

    #[test]
    fn test_triplet_topology() {
        let t1 = tree_1234_cherry12_34(4);
        // In T1 = ((1,2),(3,4)):
        // triple (1,2,3): lca(1,2) is deepest -> topology 0 (xy|z)
        assert_eq!(triplet_topology(&t1, 1, 2, 3), 0);
        // triple (1,3,4): lca(3,4) is deepest -> topology 2 (yz|x)
        assert_eq!(triplet_topology(&t1, 1, 3, 4), 2);

        let t2 = tree_1234_cherry13_24(4);
        // In T2 = ((1,3),(2,4)):
        // triple (1,2,3): lca(1,3) is deepest -> topology 1 (xz|y)
        assert_eq!(triplet_topology(&t2, 1, 2, 3), 1);
    }

    #[test]
    fn test_conflict_triples() {
        let t1 = tree_1234_cherry12_34(4);
        let t2 = tree_1234_cherry13_24(4);
        let conflicts = find_conflict_triples(&[t1, t2], 4);
        // All triples involving leaves from both cherries should conflict.
        // (1,2,3): T1 has 12|3, T2 has 13|2 -> conflict
        // (1,2,4): T1 has 12|4, T2 has 24|1 -> conflict
        // (1,3,4): T1 has 34|1, T2 has 13|4... wait let me check.
        // T2 = ((1,3),(2,4)): triple (1,3,4):
        //   lca_T2(1,3) = internal (deepest), so topology is 0 (xy|z = 13|4)
        // T1 = ((1,2),(3,4)): triple (1,3,4):
        //   lca_T1(3,4) = internal (deepest), so topology is 2 (yz|x = 34|1)
        // -> conflict
        // (2,3,4): T1 has 34|2, T2 has 24|3 -> conflict
        assert_eq!(conflicts.len(), 4);
    }

    #[test]
    fn test_solve_identical_trees() {
        // Two identical trees -> MAF = 1 component (no cuts needed)
        let t1 = tree_1234_cherry12_34(4);
        let t2 = tree_1234_cherry12_34(4);
        let instance = Instance::new(vec![t1, t2], 4);

        let mut solver = MafIlpSolver::new();
        let result = solver.solve(&instance);
        assert!(result.is_some());
        let components = result.unwrap();
        assert_eq!(components.len(), 1);
    }

    #[test]
    fn test_solve_conflicting_trees() {
        // T1 = ((1,2),(3,4)), T2 = ((1,3),(2,4))
        // These disagree on all triples. The MAF should have 2 components
        // (need to cut 1 edge, separating into 2 groups).
        let t1 = tree_1234_cherry12_34(4);
        let t2 = tree_1234_cherry13_24(4);
        let instance = Instance::new(vec![t1, t2], 4);

        let mut solver = MafIlpSolver::new();
        let result = solver.solve(&instance);
        assert!(result.is_some());
        let components = result.unwrap();
        // rSPR distance between ((1,2),(3,4)) and ((1,3),(2,4)) is 2.
        // No 2-component partition avoids overlap in both trees.
        // So MAF has 3 components.
        assert_eq!(components.len(), 3);
    }
}
