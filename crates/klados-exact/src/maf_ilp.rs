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
// Olver et al. LP* (compact formulation from "A Duality Based 2-Approximation")
// ---------------------------------------------------------------------------

/// Solve the Olver et al. LP* relaxation for a 2-tree MAF instance.
///
/// This is the compact formulation from Section 5 of the paper, which is
/// equivalent to the exponential LP and strictly stronger than Wu's formulation.
/// Integrality gap is between 1.25 and 2.
///
/// Returns Some(fractional_cuts) as a lower bound on OPT, or None if not applicable.
pub fn solve_olver_lp(instance: &Instance) -> Option<f64> {
    let n = instance.num_leaves as usize;
    if n <= 1 {
        return Some(0.0);
    }
    // LP* is defined for exactly 2 trees.
    if instance.num_trees() != 2 {
        return None;
    }

    let t1 = &instance.trees[0];
    let t2 = &instance.trees[1];

    // --- Build the DAG D = (Z, U1 ∪ U2) ---
    //
    // Z = all pairs (i1, i2) with 1 ≤ i1 ≤ i2 ≤ n.
    // Each pair r = (i1, i2) represents an internal node labeled by the
    // smallest leaves below each child subtree.
    // Z_L = {(i, i) : i ∈ L} are "leaf" nodes in the DAG.
    //
    // For r = (i1, i2), lca_t(r) = lca_t(i1, i2) in tree t.
    //
    // Arc (r, s) ∈ U1 if: i1 = j1 AND lca_t(s) ≺ lca_t(r) for t = 1, 2.
    // Arc (r, s) ∈ U2 if: i2 = j2 AND lca_t(s) ≺ lca_t(r) for t = 1, 2.

    // Map pair (i1, i2) to a flat index. i1, i2 are 1-based labels.
    let pair_to_idx = |i1: usize, i2: usize| -> usize {
        debug_assert!(i1 >= 1 && i2 >= 1 && i1 <= n && i2 <= n && i1 <= i2);
        // Map to 0-based: (i1-1, i2-1), then use triangular + diagonal indexing.
        let a = i1 - 1;
        let b = i2 - 1;
        b * (b + 1) / 2 + a
    };
    let num_z = n * (n + 1) / 2;

    // Precompute lca depths for each pair in both trees.
    // lca_node[t][idx] = node id of lca_t(i1, i2).
    let mut lca_node: [Vec<NodeId>; 2] = [vec![NONE; num_z], vec![NONE; num_z]];
    let mut lca_depth: [Vec<u16>; 2] = [vec![0; num_z], vec![0; num_z]];
    let trees_ref = [t1, t2];

    for t in 0..2 {
        let tree = trees_ref[t];
        for i1 in 1..=n {
            let node_i1 = tree.label_to_node[i1 as Label as usize];
            for i2 in i1..=n {
                let idx = pair_to_idx(i1, i2);
                if i1 == i2 {
                    lca_node[t][idx] = node_i1;
                    lca_depth[t][idx] = tree.depth[node_i1 as usize];
                } else {
                    let node_i2 = tree.label_to_node[i2 as Label as usize];
                    let l = lca(tree, node_i1, node_i2);
                    lca_node[t][idx] = l;
                    lca_depth[t][idx] = tree.depth[l as usize];
                }
            }
        }
    }

    // Check if s is strictly below r in both trees: lca_t(s) is a proper
    // descendant of lca_t(r) for t = 1, 2.
    let is_strictly_below = |r_idx: usize, s_idx: usize| -> bool {
        for t in 0..2 {
            let r_node = lca_node[t][r_idx];
            let s_node = lca_node[t][s_idx];
            if r_node == s_node {
                return false;
            }
            // s_node must be a descendant of r_node: depth(s) > depth(r)
            // and s is actually under r.
            if lca_depth[t][s_idx] <= lca_depth[t][r_idx] {
                return false;
            }
            // Verify ancestry: walk s_node up to depth of r_node.
            let tree = trees_ref[t];
            let mut cur = s_node;
            while tree.depth[cur as usize] > tree.depth[r_node as usize] {
                cur = tree.parent[cur as usize];
            }
            if cur != r_node {
                return false;
            }
        }
        true
    };

    // Build arc lists: include ALL arcs satisfying the paper's conditions.
    // (r, s) ∈ U1 if: i1(r) = i1(s) AND lca_t(s) ≺ lca_t(r) for t = 1, 2.
    // (r, s) ∈ U2 if: i2(r) = i2(s) AND lca_t(s) ≺ lca_t(r) for t = 1, 2.

    struct Arc {
        from: usize, // index in Z
        to: usize,   // index in Z
    }

    let mut u1_arcs: Vec<Arc> = Vec::new();
    let mut u2_arcs: Vec<Arc> = Vec::new();

    // For U1: arcs between Z-nodes sharing the same first index i1.
    for i1 in 1..=n {
        // r = (i1, i2_r), s = (i1, i2_s), where s is strictly below r in both trees.
        for i2_r in i1..=n {
            let r_idx = pair_to_idx(i1, i2_r);
            for i2_s in i1..=n {
                if i2_r == i2_s {
                    continue;
                }
                let s_idx = pair_to_idx(i1, i2_s);
                if is_strictly_below(r_idx, s_idx) {
                    u1_arcs.push(Arc {
                        from: r_idx,
                        to: s_idx,
                    });
                }
            }
        }
    }

    // For U2: arcs between Z-nodes sharing the same second index i2.
    for i2 in 1..=n {
        for i1_r in 1..=i2 {
            let r_idx = pair_to_idx(i1_r, i2);
            for i1_s in 1..=i2 {
                if i1_r == i1_s {
                    continue;
                }
                let s_idx = pair_to_idx(i1_s, i2);
                if is_strictly_below(r_idx, s_idx) {
                    u2_arcs.push(Arc {
                        from: r_idx,
                        to: s_idx,
                    });
                }
            }
        }
    }

    // --- Build LP ---
    let mut pb = RowProblem::default();

    // Variables: y_a for each arc a ∈ U1 ∪ U2.
    let y_u1: Vec<Col> = u1_arcs.iter().map(|_| pb.add_column(0.0, 0.0..)).collect();
    let y_u2: Vec<Col> = u2_arcs.iter().map(|_| pb.add_column(0.0, 0.0..)).collect();

    // Variables: x_i for each leaf i ∈ L.
    // These will appear in the objective with coefficient 1.0.
    let x: Vec<Col> = (0..n).map(|_| pb.add_column(1.0, 0.0..)).collect();

    // Precompute per-node incoming/outgoing arc indices.
    // out_u1[r] = indices into y_u1 where arc.from == r
    // in_u1[r]  = indices into y_u1 where arc.to == r (should be ≤ 1)
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

    // Objective: min Σ_{r ∈ Z \ Z_L} (y(δ⁺(r) ∩ U1) - y(δ⁻(r))) + Σ_i x_i - 1
    //
    // The x_i terms are already in the objective (coeff 1.0).
    // For the y terms: each y_u1 arc from r contributes +1 to the objective
    // for the "y(δ⁺(r) ∩ U1)" part. Each incoming arc to r (from U1 or U2)
    // contributes -1 for the "y(δ⁻(r))" part. But only for non-leaf r.
    //
    // We modify the objective coefficients by adjusting column costs.
    // HiGHS doesn't let us modify costs after creation, so we need to set them
    // correctly at creation time. Let me rebuild with correct costs.

    // Actually, let me compute the objective coefficients per arc:
    // For a U1 arc (r → s):
    //   +1 from "y(δ⁺(r) ∩ U1)" if r ∉ Z_L (which it isn't since it has a child)
    //   -1 from "y(δ⁻(s))" if s ∉ Z_L
    // For a U2 arc (r → s):
    //   +0 from outgoing (U2 arcs don't appear in the positive term)
    //   -1 from "y(δ⁻(s))" if s ∉ Z_L
    //
    // Wait, re-reading LP*-1 more carefully:
    // min Σ_{r ∈ Z\Z_L} [ y(δ⁺(r) ∩ U1) - y(δ⁻(r)) ] + Σ_i x_i - 1
    //
    // For each non-leaf r, the contribution is: (sum of y on outgoing U1 arcs from r)
    //                                          - (sum of y on ALL incoming arcs to r)
    //
    // For a U1 arc a = (r → s) with variable y_a:
    //   y_a appears in δ⁺(r) ∩ U1 → coefficient +1 (if r is non-leaf)
    //   y_a appears in δ⁻(s) → coefficient -1 (if s is non-leaf)
    //   Net coefficient: +1(if r non-leaf) - 1(if s non-leaf)
    //
    // For a U2 arc a = (r → s) with variable y_a:
    //   y_a does NOT appear in any δ⁺ ∩ U1 term
    //   y_a appears in δ⁻(s) → coefficient -1 (if s is non-leaf)
    //   Net coefficient: -1(if s non-leaf)

    // Rebuild LP with correct objective coefficients.
    pb = RowProblem::default();

    let is_leaf_node = |idx: usize| -> bool {
        // Leaf nodes are (i, i) for i ∈ 1..=n.
        // pair_to_idx(i, i) = (i-1)*(i)/2 + (i-1) = (i-1)*(i+2)/2... no.
        // Let me just check: for (i1, i2), it's a leaf if i1 == i2.
        // Given idx, we need to recover (i1, i2). Easier to precompute.
        false // placeholder, will use precomputed set
    };
    let _ = is_leaf_node; // suppress warning

    // Precompute which indices are leaf nodes.
    let mut is_leaf = vec![false; num_z];
    for i in 1..=n {
        is_leaf[pair_to_idx(i, i)] = true;
    }

    // Create y variables with correct objective coefficients.
    let y_u1: Vec<Col> = u1_arcs
        .iter()
        .map(|arc| {
            let from_contrib = if !is_leaf[arc.from] { 1.0 } else { 0.0 };
            let to_contrib = if !is_leaf[arc.to] { -1.0 } else { 0.0 };
            pb.add_column(from_contrib + to_contrib, 0.0..)
        })
        .collect();

    let y_u2: Vec<Col> = u2_arcs
        .iter()
        .map(|arc| {
            let to_contrib = if !is_leaf[arc.to] { -1.0 } else { 0.0 };
            pb.add_column(to_contrib, 0.0..)
        })
        .collect();

    // Create x_i variables (objective coefficient 1.0 each).
    let x: Vec<Col> = (0..n).map(|_| pb.add_column(1.0, 0.0..)).collect();

    // --- Constraints ---

    // LP*-2: y(δ⁺(r) ∩ U1) = y(δ⁺(r) ∩ U2)  for all r ∈ Z
    // i.e., total outgoing U1 flow = total outgoing U2 flow at every node.
    for r in 0..num_z {
        if out_u1[r].is_empty() && out_u2[r].is_empty() {
            continue;
        }
        // Σ y_u1[out_u1[r]] - Σ y_u2[out_u2[r]] = 0
        let mut coeffs: Vec<(Col, f64)> = Vec::new();
        for &ai in &out_u1[r] {
            coeffs.push((y_u1[ai], 1.0));
        }
        for &ai in &out_u2[r] {
            coeffs.push((y_u2[ai], -1.0));
        }
        if !coeffs.is_empty() {
            pb.add_row(0.0..=0.0, &coeffs);
        }
    }

    // LP*-3: y(δ⁺(r) ∩ U1) ≥ y(δ⁻(r))  for all r ∈ Z \ Z_L
    // (Lemma 16: cone constraints apply to non-leaf Z-nodes only.
    //  For leaves, this would force 0 ≥ incoming flow, making the LP trivial.)
    for r in 0..num_z {
        if is_leaf[r] {
            continue;
        }
        // Σ y_u1[out_u1[r]] - Σ y[incoming to r] ≥ 0
        let mut coeffs: Vec<(Col, f64)> = Vec::new();
        for &ai in &out_u1[r] {
            coeffs.push((y_u1[ai], 1.0));
        }
        for &(ai, is_u1_arc) in &in_arcs[r] {
            if is_u1_arc {
                coeffs.push((y_u1[ai], -1.0));
            } else {
                coeffs.push((y_u2[ai], -1.0));
            }
        }
        if !coeffs.is_empty() {
            pb.add_row(0.0.., &coeffs);
        }
    }

    // LP*-4: x_i = 1 - y(δ⁻(i,i))  for all i ∈ L
    // i.e., x_i + Σ y[incoming to (i,i)] = 1
    for i in 1..=n {
        let leaf_idx = pair_to_idx(i, i);
        let mut coeffs: Vec<(Col, f64)> = vec![(x[i - 1], 1.0)];
        for &(ai, is_u1_arc) in &in_arcs[leaf_idx] {
            if is_u1_arc {
                coeffs.push((y_u1[ai], 1.0));
            } else {
                coeffs.push((y_u2[ai], 1.0));
            }
        }
        pb.add_row(1.0..=1.0, &coeffs);
    }

    // LP*-5: Σ_{r ∈ lca⁻¹(v)} y(δ⁺(r) ∩ U1) ≤ 1  for all v ∈ V \ L
    // For each internal node v in either tree, the sum of outgoing U1 flow
    // from all Z-nodes r whose lca_t(r) = v must be ≤ 1.
    for t in 0..2 {
        let tree = trees_ref[t];
        for v in 0..tree.num_nodes() as NodeId {
            if tree.is_leaf(v) {
                continue;
            }
            // Find all Z-nodes r such that lca_t(r) = v.
            // Collect their outgoing U1 arcs.
            let mut coeffs: Vec<(Col, f64)> = Vec::new();
            for i1 in 1..=n {
                for i2 in i1..=n {
                    let idx = pair_to_idx(i1, i2);
                    if lca_node[t][idx] == v {
                        for &ai in &out_u1[idx] {
                            coeffs.push((y_u1[ai], 1.0));
                        }
                    }
                }
            }
            if !coeffs.is_empty() {
                pb.add_row(..=1.0, &coeffs);
            }
        }
    }

    // --- Solve ---
    // Objective is: Σ (obj coeffs on y arcs) + Σ x_i - 1.
    // HiGHS minimizes the sum of (column * obj_coeff). We set a constant offset of -1.
    let mut model = pb.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_option("threads", 1_i32);
    model.set_option("presolve", "on");

    let solved = model.solve();

    if solved.status() != HighsModelStatus::Optimal {
        return None;
    }

    // The LP value is Σ obj_coeffs * vars. We need to subtract 1 for the "-1" in LP*-1.
    Some(solved.objective_value() - 1.0)
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
