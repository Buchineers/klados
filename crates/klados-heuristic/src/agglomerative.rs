//! Bottom-up agglomerative heuristic for MAF.
//!
//! Starts from all-singletons partition and greedily merges compatible
//! components. Uses tree-guided candidate generation: walk each tree
//! bottom-up to find sibling components that can merge.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use fixedbitset::FixedBitSet;
use klados_core::tree::{NodeId, Tree, NONE};
use klados_core::{Instance, SolverStats};

use crate::HeuristicSolver;

// ---------------------------------------------------------------------------
// Union-Find for fast component tracking
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

    fn union(&mut self, a: u32, b: u32) -> u32 {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        if self.rank[ra as usize] < self.rank[rb as usize] {
            self.parent[ra as usize] = rb;
            rb
        } else if self.rank[ra as usize] > self.rank[rb as usize] {
            self.parent[rb as usize] = ra;
            ra
        } else {
            self.parent[rb as usize] = ra;
            self.rank[ra as usize] += 1;
            ra
        }
    }

    fn same(&mut self, a: u32, b: u32) -> bool {
        self.find(a) == self.find(b)
    }
}

// ---------------------------------------------------------------------------
// Tree precomp (LCA + leaf sets)
// ---------------------------------------------------------------------------

struct TreePrecomp {
    leaf_set: Vec<FixedBitSet>,
    parent: Vec<NodeId>,
    depth: Vec<u16>,
    label_to_node: Vec<NodeId>,
    left: Vec<NodeId>,
    right: Vec<NodeId>,
    num_nodes: usize,
    num_leaves: u32,
    root: NodeId,
    post_order: Vec<NodeId>,
    // Euler tour + sparse table for O(1) LCA
    first_occ: Vec<u32>,
    euler: Vec<NodeId>,
    euler_depth: Vec<u16>,
    sparse: Vec<Vec<u32>>,
    // label[node] -> leaf label (0 for internal)
    label: Vec<u32>,
}

impl TreePrecomp {
    fn build(tree: &Tree) -> Self {
        let n = tree.num_nodes();
        let nl = tree.num_leaves as usize;

        let post = tree.post_order_vec();
        let mut leaf_set = vec![FixedBitSet::with_capacity(nl + 1); n];
        for &node in &post {
            if tree.is_leaf(node) {
                let lbl = tree.label[node as usize];
                if lbl > 0 {
                    leaf_set[node as usize].insert(lbl as usize);
                }
            } else if let Some((l, r)) = tree.children(node) {
                let mut combined = leaf_set[l as usize].clone();
                combined.union_with(&leaf_set[r as usize]);
                leaf_set[node as usize] = combined;
            }
        }

        // Euler tour for LCA
        let tour_cap = 2 * n;
        let mut euler = Vec::with_capacity(tour_cap);
        let mut euler_depth = Vec::with_capacity(tour_cap);
        let mut first_occ = vec![u32::MAX; n];

        let mut stack: Vec<(NodeId, bool)> = vec![(tree.root, false)];
        while let Some((node, returning)) = stack.pop() {
            let pos = euler.len() as u32;
            euler.push(node);
            euler_depth.push(tree.depth[node as usize]);
            if first_occ[node as usize] == u32::MAX {
                first_occ[node as usize] = pos;
            }
            if !returning {
                if let Some((left, right)) = tree.children(node) {
                    stack.push((node, true));
                    stack.push((right, false));
                    stack.push((node, true));
                    stack.push((left, false));
                }
            }
        }

        let len = euler.len();
        let log_len = if len > 1 {
            (usize::BITS - (len - 1).leading_zeros()) as usize + 1
        } else {
            1
        };
        let mut sparse = Vec::with_capacity(log_len);
        let level0: Vec<u32> = (0..len as u32).collect();
        sparse.push(level0);
        for k in 1..log_len {
            let half = 1usize << (k - 1);
            let prev = &sparse[k - 1];
            let level_len = len.saturating_sub((1usize << k) - 1);
            let mut level = Vec::with_capacity(level_len);
            for i in 0..level_len {
                let a = prev[i];
                let b = prev[i + half];
                if euler_depth[a as usize] <= euler_depth[b as usize] {
                    level.push(a);
                } else {
                    level.push(b);
                }
            }
            sparse.push(level);
        }

        TreePrecomp {
            leaf_set,
            parent: tree.parent.clone(),
            depth: tree.depth.clone(),
            label_to_node: tree.label_to_node.clone(),
            left: tree.left.clone(),
            right: tree.right.clone(),
            num_nodes: n,
            num_leaves: tree.num_leaves,
            root: tree.root,
            post_order: post,
            first_occ,
            euler,
            euler_depth,
            sparse,
            label: tree.label.clone(),
        }
    }

    #[inline]
    fn lca(&self, u: NodeId, v: NodeId) -> NodeId {
        if u == NONE || v == NONE {
            return NONE;
        }
        if u == v {
            return u;
        }
        let mut l = self.first_occ[u as usize] as usize;
        let mut r = self.first_occ[v as usize] as usize;
        if l > r {
            std::mem::swap(&mut l, &mut r);
        }
        let len = r - l + 1;
        if len == 1 {
            return self.euler[l];
        }
        let k = (usize::BITS - len.leading_zeros() - 1) as usize;
        let a = self.sparse[k][l];
        let b = self.sparse[k][r + 1 - (1 << k)];
        if self.euler_depth[a as usize] <= self.euler_depth[b as usize] {
            self.euler[a as usize]
        } else {
            self.euler[b as usize]
        }
    }

    #[inline]
    fn is_leaf_node(&self, node: NodeId) -> bool {
        self.left[node as usize] == NONE
    }

    /// Get the leaf label at a leaf node (0 if internal).
    #[inline]
    fn leaf_label(&self, node: NodeId) -> u32 {
        self.label[node as usize]
    }
}

// ---------------------------------------------------------------------------
// Compatibility check: induced subtrees must be isomorphic
// ---------------------------------------------------------------------------

/// Check if a set of labels induces isomorphic subtrees in both trees.
fn is_compatible(tp1: &TreePrecomp, tp2: &TreePrecomp, labels: &[u32]) -> bool {
    if labels.len() <= 2 {
        return true;
    }
    canonical_form(tp1, labels) == canonical_form(tp2, labels)
}

/// Canonical string for the induced subtree on the given labels.
fn canonical_form(tp: &TreePrecomp, labels: &[u32]) -> String {
    let mut label_set = FixedBitSet::with_capacity(tp.num_leaves as usize + 1);
    for &l in labels {
        label_set.insert(l as usize);
    }
    let lca = {
        let mut r = NONE;
        for &l in labels {
            let node = tp.label_to_node[l as usize];
            r = if r == NONE { node } else { tp.lca(r, node) };
        }
        r
    };
    if lca == NONE {
        return String::new();
    }
    fn recurse(tp: &TreePrecomp, node: NodeId, ls: &FixedBitSet) -> Option<String> {
        if tp.is_leaf_node(node) {
            let lbl = tp.leaf_label(node);
            return if lbl > 0 && ls.contains(lbl as usize) {
                Some(format!("{}", lbl))
            } else {
                None
            };
        }
        let left = tp.left[node as usize];
        let right = tp.right[node as usize];
        let lh = !tp.leaf_set[left as usize].is_disjoint(ls);
        let rh = !tp.leaf_set[right as usize].is_disjoint(ls);
        match (lh, rh) {
            (false, false) => None,
            (true, false) => recurse(tp, left, ls),
            (false, true) => recurse(tp, right, ls),
            (true, true) => {
                let l = recurse(tp, left, ls)?;
                let r = recurse(tp, right, ls)?;
                let (a, b) = if l <= r { (l, r) } else { (r, l) };
                Some(format!("({},{})", a, b))
            }
        }
    }
    recurse(tp, lca, &label_set).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// V-set (internal nodes used by an agreement block) for node-disjointness
// ---------------------------------------------------------------------------

fn compute_vset(tp: &TreePrecomp, labels: &[u32]) -> FixedBitSet {
    let mut vset = FixedBitSet::with_capacity(tp.num_nodes);
    if labels.len() <= 1 {
        return vset;
    }
    let lca = {
        let mut r = NONE;
        for &l in labels {
            let node = tp.label_to_node[l as usize];
            r = if r == NONE { node } else { tp.lca(r, node) };
        }
        r
    };
    if lca == NONE {
        return vset;
    }
    for &lbl in labels {
        let mut cur = tp.label_to_node[lbl as usize];
        while cur != NONE && cur != lca && !vset.contains(cur as usize) {
            vset.insert(cur as usize);
            cur = tp.parent[cur as usize];
        }
        if cur == lca {
            vset.insert(lca as usize);
        }
    }
    vset
}

// ---------------------------------------------------------------------------
// Core solver
// ---------------------------------------------------------------------------

pub struct AgglomerativeSolver {
    terminate: Arc<AtomicBool>,
    stats: SolverStats,
}

impl AgglomerativeSolver {
    pub fn new() -> Self {
        Self {
            terminate: Arc::new(AtomicBool::new(false)),
            stats: SolverStats::default(),
        }
    }

    pub fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        let n = instance.num_leaves;
        if n <= 1 {
            return Some(
                instance.trees[0]
                    .leaves()
                    .map(|l| Tree::singleton(l, n))
                    .collect(),
            );
        }
        if instance.num_trees() < 2 {
            return Some(vec![instance.trees[0].clone()]);
        }

        let start = std::time::Instant::now();
        let t1 = &instance.trees[0];
        let t2 = &instance.trees[1];

        eprintln!("[agglo] Building tree precomps for n={}", n);
        let tp1 = TreePrecomp::build(t1);
        let tp2 = TreePrecomp::build(t2);
        eprintln!(
            "[agglo] Precomp done in {:.1}ms",
            start.elapsed().as_secs_f64() * 1000.0
        );

        // UF: labels are 1..=n, we use index = label (waste index 0)
        let mut uf = UnionFind::new(n as usize + 1);

        // Component data: comp_labels[root] = labels in component with UF root `root`
        // We only maintain this for active roots.
        let mut comp_labels: Vec<Vec<u32>> = vec![Vec::new(); n as usize + 1];
        for lbl in 1..=n {
            comp_labels[lbl as usize] = vec![lbl];
        }

        // Node-owner maps: for each internal node in T1/T2, which component root
        // (UF label) owns it. 0 = unowned. Allows O(|vset|) disjointness checks.
        let mut node_owner1: Vec<u32> = vec![0; tp1.num_nodes];
        let mut node_owner2: Vec<u32> = vec![0; tp2.num_nodes];

        let mut num_components = n as usize;

        // try_merge: check compatibility + incremental node-disjointness via owner maps.
        // Returns true on success and updates all state.
        fn try_merge(
            ar: u32,
            br: u32,
            uf: &mut UnionFind,
            comp_labels: &mut Vec<Vec<u32>>,
            node_owner1: &mut Vec<u32>,
            node_owner2: &mut Vec<u32>,
            tp1: &TreePrecomp,
            tp2: &TreePrecomp,
        ) -> bool {
            let ar = uf.find(ar);
            let br = uf.find(br);
            if ar == br {
                return false;
            }
            let labels_a = &comp_labels[ar as usize];
            let labels_b = &comp_labels[br as usize];
            if labels_a.is_empty() || labels_b.is_empty() {
                return false;
            }

            let mut merged: Vec<u32> = Vec::with_capacity(labels_a.len() + labels_b.len());
            merged.extend_from_slice(labels_a);
            merged.extend_from_slice(labels_b);

            // Check compatibility (isomorphic induced subtrees)
            if !is_compatible(tp1, tp2, &merged) {
                return false;
            }

            // Compute new V-sets and check disjointness via owner maps
            let new_vs1 = compute_vset(tp1, &merged);
            for node_idx in new_vs1.ones() {
                let owner = node_owner1[node_idx];
                if owner != 0 && owner != ar && owner != br {
                    // Owned by a different component — conflict
                    return false;
                }
            }
            let new_vs2 = compute_vset(tp2, &merged);
            for node_idx in new_vs2.ones() {
                let owner = node_owner2[node_idx];
                if owner != 0 && owner != ar && owner != br {
                    return false;
                }
            }

            // Commit merge
            let new_root = uf.union(ar, br);
            let other = if new_root == ar { br } else { ar };
            let other_labels = std::mem::take(&mut comp_labels[other as usize]);
            comp_labels[new_root as usize].extend_from_slice(&other_labels);

            // Update owner maps: claim all nodes in new V-sets
            for node_idx in new_vs1.ones() {
                node_owner1[node_idx] = new_root;
            }
            for node_idx in new_vs2.ones() {
                node_owner2[node_idx] = new_root;
            }

            true
        }

        // =================================================================
        // Phase 1: Common cherry merges only
        //
        // Only merge leaves that are siblings in BOTH trees. This is
        // conservative — preserves room for later non-local merges.
        // =================================================================
        let mut cherry_merges = 0usize;
        for &node in &tp1.post_order {
            if tp1.is_leaf_node(node) {
                continue;
            }
            let left = tp1.left[node as usize];
            let right = tp1.right[node as usize];
            if !tp1.is_leaf_node(left) || !tp1.is_leaf_node(right) {
                continue;
            }
            let lbl_a = tp1.leaf_label(left);
            let lbl_b = tp1.leaf_label(right);
            if lbl_a == 0 || lbl_b == 0 {
                continue;
            }
            // Check if also siblings in T2
            let node2a = tp2.label_to_node[lbl_a as usize];
            let node2b = tp2.label_to_node[lbl_b as usize];
            if tp2.parent[node2a as usize] == NONE
                || tp2.parent[node2a as usize] != tp2.parent[node2b as usize]
            {
                continue;
            }
            let ar = uf.find(lbl_a);
            let br = uf.find(lbl_b);
            if ar != br
                && try_merge(
                    ar, br, &mut uf, &mut comp_labels,
                    &mut node_owner1, &mut node_owner2, &tp1, &tp2,
                )
            {
                cherry_merges += 1;
                num_components -= 1;
            }
        }
        eprintln!(
            "[agglo] Phase 1 (common cherries): {} merges, {} components, {:.1}ms",
            cherry_merges, num_components, start.elapsed().as_secs_f64() * 1000.0,
        );

        // =================================================================
        // Phase 2: All-pairs scan sorted by LCA cost, no restart
        //
        // Build candidate list of all component pairs sorted by merge cost
        // (how far up the LCA jumps). Scan once, merging as we go.
        // After a full pass, rebuild and repeat until fixpoint.
        // Time-budgeted.
        // =================================================================
        let phase2_start = std::time::Instant::now();
        let mut phase2_merges = 0usize;
        let time_limit = std::time::Duration::from_secs(250);
        let mut pass = 0;

        loop {
            if self.terminate.load(Ordering::Relaxed) || start.elapsed() >= time_limit {
                break;
            }
            pass += 1;

            // Collect active roots
            let mut roots: Vec<u32> = Vec::new();
            {
                let mut seen = FixedBitSet::with_capacity(n as usize + 1);
                for lbl in 1..=n {
                    let r = uf.find(lbl);
                    if !seen.contains(r as usize) && !comp_labels[r as usize].is_empty() {
                        seen.insert(r as usize);
                        roots.push(r);
                    }
                }
            }
            if roots.len() <= 1 {
                break;
            }

            // Precompute LCA node per component in both trees
            let root_lca1: Vec<NodeId> = roots.iter().map(|&r| {
                let labels = &comp_labels[r as usize];
                let mut lca = NONE;
                for &l in labels {
                    let nd = tp1.label_to_node[l as usize];
                    lca = if lca == NONE { nd } else { tp1.lca(lca, nd) };
                }
                lca
            }).collect();
            let root_lca2: Vec<NodeId> = roots.iter().map(|&r| {
                let labels = &comp_labels[r as usize];
                let mut lca = NONE;
                for &l in labels {
                    let nd = tp2.label_to_node[l as usize];
                    lca = if lca == NONE { nd } else { tp2.lca(lca, nd) };
                }
                lca
            }).collect();

            // Build sorted candidate pairs by merge cost
            // Cost = depth drop in T1 + depth drop in T2 (lower = tighter merge)
            let mut candidates: Vec<(u32, usize, usize)> = Vec::new();
            for i in 0..roots.len() {
                for j in (i + 1)..roots.len() {
                    let merged_lca1 = tp1.lca(root_lca1[i], root_lca1[j]);
                    let merged_lca2 = tp2.lca(root_lca2[i], root_lca2[j]);
                    let cost1 = if merged_lca1 != NONE && root_lca1[i] != NONE && root_lca1[j] != NONE {
                        (tp1.depth[root_lca1[i] as usize] + tp1.depth[root_lca1[j] as usize])
                            .saturating_sub(2 * tp1.depth[merged_lca1 as usize])
                    } else {
                        u16::MAX
                    };
                    let cost2 = if merged_lca2 != NONE && root_lca2[i] != NONE && root_lca2[j] != NONE {
                        (tp2.depth[root_lca2[i] as usize] + tp2.depth[root_lca2[j] as usize])
                            .saturating_sub(2 * tp2.depth[merged_lca2 as usize])
                    } else {
                        u16::MAX
                    };
                    let cost = (cost1 as u32).saturating_add(cost2 as u32);
                    candidates.push((cost, i, j));
                }
            }
            candidates.sort_unstable_by_key(|&(c, _, _)| c);

            // Single pass — no restart
            let mut pass_merges = 0usize;
            for &(_, i, j) in &candidates {
                if self.terminate.load(Ordering::Relaxed) || start.elapsed() >= time_limit {
                    break;
                }
                let ri = uf.find(roots[i]);
                let rj = uf.find(roots[j]);
                if ri == rj {
                    continue;
                }
                if try_merge(
                    ri, rj, &mut uf, &mut comp_labels,
                    &mut node_owner1, &mut node_owner2, &tp1, &tp2,
                ) {
                    pass_merges += 1;
                    num_components -= 1;
                }
            }

            phase2_merges += pass_merges;
            eprintln!(
                "[agglo] Phase 2 pass {}: {} merges ({} total), {} components, {:.1}s",
                pass, pass_merges, phase2_merges, num_components,
                phase2_start.elapsed().as_secs_f64(),
            );

            if pass_merges == 0 {
                break;
            }
        }

        // Build output
        let total_time = start.elapsed().as_secs_f64();
        let total_merges = cherry_merges + phase2_merges;
        eprintln!(
            "[agglo] Final: {} components (score={}) in {:.2}s [{} p1 + {} p2 = {} merges]",
            num_components,
            num_components.saturating_sub(1),
            total_time,
            cherry_merges,
            phase2_merges,
            total_merges,
        );

        self.stats.upper_bound = Some(num_components);

        let mut forest: Vec<Tree> = Vec::with_capacity(num_components);
        let mut emitted = FixedBitSet::with_capacity(n as usize + 1);
        for lbl in 1..=n {
            let r = uf.find(lbl);
            if emitted.contains(r as usize) {
                continue;
            }
            emitted.insert(r as usize);
            let labels = &comp_labels[r as usize];
            if labels.is_empty() {
                continue;
            }
            if labels.len() == 1 {
                forest.push(Tree::singleton(labels[0], n));
            } else {
                let mut bs = FixedBitSet::with_capacity(n as usize + 1);
                for &l in labels {
                    bs.insert(l as usize);
                }
                forest.push(Tree::component_from_leafset(&bs, t1, n));
            }
        }

        Some(forest)
    }
}

/// Find the first (leftmost) leaf label under a node.
fn first_leaf_label(tp: &TreePrecomp, mut node: NodeId) -> u32 {
    loop {
        if tp.is_leaf_node(node) {
            return tp.leaf_label(node);
        }
        node = tp.left[node as usize];
        if node == NONE {
            return 0;
        }
    }
}

impl HeuristicSolver for AgglomerativeSolver {
    fn name(&self) -> &'static str {
        "agglomerative"
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        AgglomerativeSolver::solve(self, instance)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }

    fn sigterm_handler(&self) {
        self.terminate.store(true, Ordering::SeqCst);
    }
}
