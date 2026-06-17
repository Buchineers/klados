//! Bottom-up agglomerative heuristic for MAF.
//!
//! Starts from all-singletons partition and greedily merges compatible
//! components. Uses tree-guided candidate generation: walk each tree
//! bottom-up to find sibling components that can merge.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use fixedbitset::FixedBitSet;
use klados_core::kernelize::restrict_instance_simple;
use klados_core::tree::{NONE, NodeId, Tree};
use klados_core::{Instance, SolverStats};
use log::info;
use crate::solvers::maf_sat::MafSatSolver;
use crate::solvers::whidden::WhiddenSolver;


const REGION_REOPT_MAX_LABELS: usize = 96;
const REGION_REOPT_MAX_ROOTS: usize = 12;
const REGION_REOPT_MAX_NEIGHBORS: usize = 8;
const WHIDDEN_LOCAL_MAX_LEAVES: u32 = 80;
const PLATEAU_MAX_EPISODES: usize = 2;
const PLATEAU_MAX_STATES_PER_EPISODE: usize = 12;
const PLATEAU_MAX_ROOTS: usize = 10;
const PLATEAU_MAX_NEIGHBORS: usize = 6;
const PLATEAU_MAX_LABELS: usize = 72;

// ---------------------------------------------------------------------------
// Union-Find for fast component tracking
// ---------------------------------------------------------------------------

#[derive(Clone)]
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

#[derive(Clone, Copy)]
struct ComponentMeta {
    size: u16,
    min_label: u32,
    lca1: NodeId,
    lca2: NodeId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct CandidateKey {
    lca_jump_sum: u16,
    tree_hits_rank: u16,
    rise_sum: u16,
    balance_penalty: u16,
    size_sum: u16,
    tie_a: u32,
    tie_b: u32,
}

#[derive(Clone, Copy)]
struct Candidate {
    a: u32,
    b: u32,
    key: CandidateKey,
}

fn component_lca(tp: &TreePrecomp, labels: &[u32]) -> NodeId {
    let mut lca = NONE;
    for &l in labels {
        let nd = tp.label_to_node[l as usize];
        lca = if lca == NONE { nd } else { tp.lca(lca, nd) };
    }
    lca
}

fn try_merge(
    ar: u32,
    br: u32,
    uf: &mut UnionFind,
    comp_labels: &mut [Vec<u32>],
    node_owner1: &mut [u32],
    node_owner2: &mut [u32],
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

    if !is_compatible(tp1, tp2, &merged) {
        return false;
    }

    let new_vs1 = compute_vset(tp1, &merged);
    for node_idx in new_vs1.ones() {
        let owner = node_owner1[node_idx];
        if owner != 0 && owner != ar && owner != br {
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

    let new_root = uf.union(ar, br);
    let other = if new_root == ar { br } else { ar };
    let other_labels = std::mem::take(&mut comp_labels[other as usize]);
    comp_labels[new_root as usize].extend_from_slice(&other_labels);

    for node_idx in new_vs1.ones() {
        node_owner1[node_idx] = new_root;
    }
    for node_idx in new_vs2.ones() {
        node_owner2[node_idx] = new_root;
    }

    true
}

fn collect_nearby_roots(
    root: u32,
    leaf_to_root: &[u32],
    comp_labels: &[Vec<u32>],
    tp1: &TreePrecomp,
    tp2: &TreePrecomp,
    rise_limit: u16,
) -> Vec<(u32, u16, u16)> {
    let mut scores: HashMap<u32, (u16, u16)> = HashMap::new();
    let labels = &comp_labels[root as usize];

    for &tp in [tp1, tp2].iter() {
        for &leaf in labels {
            let mut cur = tp.label_to_node[leaf as usize];
            for rise in 1..=rise_limit {
                if cur == NONE || tp.parent[cur as usize] == NONE {
                    break;
                }
                let parent = tp.parent[cur as usize];
                let sibling = if tp.left[parent as usize] == cur {
                    tp.right[parent as usize]
                } else {
                    tp.left[parent as usize]
                };

                let mut found = HashSet::<u32>::new();
                let mut stack = vec![sibling];
                while let Some(node) = stack.pop() {
                    if tp.is_leaf_node(node) {
                        let lbl = tp.leaf_label(node);
                        let other = leaf_to_root[lbl as usize];
                        if other == 0 || other == root || comp_labels[other as usize].is_empty() {
                            continue;
                        }
                        if found.insert(other) {
                            let entry = scores.entry(other).or_insert((0, 0));
                            entry.0 = entry.0.saturating_add(1);
                            entry.1 = entry.1.saturating_add(rise);
                        }
                    } else {
                        let left = tp.left[node as usize];
                        let right = tp.right[node as usize];
                        if left != NONE {
                            stack.push(left);
                        }
                        if right != NONE {
                            stack.push(right);
                        }
                    }
                }
                cur = parent;
            }
        }
    }

    let mut out: Vec<(u32, u16, u16)> = scores
        .into_iter()
        .map(|(other, (hits, rise_sum))| (other, hits, rise_sum))
        .collect();
    out.sort_unstable_by_key(|&(other, hits, rise_sum)| {
        (
            u16::MAX - hits,
            rise_sum,
            comp_labels[other as usize].len() as u16,
            other,
        )
    });
    out
}

fn active_roots(uf: &mut UnionFind, comp_labels: &[Vec<u32>], n: u32) -> Vec<u32> {
    let mut roots: Vec<u32> = Vec::new();
    let mut seen = FixedBitSet::with_capacity(n as usize + 1);
    for lbl in 1..=n {
        let r = uf.find(lbl);
        if !seen.contains(r as usize) && !comp_labels[r as usize].is_empty() {
            seen.insert(r as usize);
            roots.push(r);
        }
    }
    roots
}

fn region_components_from_exact(
    exact_components: &[Tree],
    reverse_map: &[u32],
    global_n: u32,
) -> Vec<Vec<u32>> {
    let mut mapped: Vec<Vec<u32>> = Vec::new();
    for t in exact_components {
        let mut labels: Vec<u32> = Vec::new();
        for local in t.leaves() {
            if local == 0 || (local as usize) >= reverse_map.len() {
                continue;
            }
            let global = reverse_map[local as usize];
            if global > 0 && global <= global_n {
                labels.push(global);
            }
        }
        if labels.is_empty() {
            continue;
        }
        labels.sort_unstable();
        labels.dedup();
        mapped.push(labels);
    }
    mapped.sort_unstable_by_key(|labels| {
        (
            labels.first().copied().unwrap_or(u32::MAX),
            labels.len(),
            labels.last().copied().unwrap_or(u32::MAX),
        )
    });
    mapped
}

fn exact_local_solve(region_instance: &Instance) -> Option<(Vec<Tree>, &'static str)> {
    if region_instance.num_trees() == 2 && region_instance.num_leaves <= WHIDDEN_LOCAL_MAX_LEAVES {
        let mut whidden = WhiddenSolver::new();
        if let Some(sol) = crate::Solver::solve(&mut whidden, region_instance, &crate::RunConfig::default()) {
            return Some((sol, "whidden"));
        }
    }

    let mut sat = MafSatSolver::new();
    crate::Solver::solve(&mut sat, region_instance, &crate::RunConfig::default()).map(|sol| (sol, "sat"))
}

fn apply_exact_region_if_improves(
    uf: &mut UnionFind,
    comp_labels: &mut Vec<Vec<u32>>,
    node_owner1: &mut [u32],
    node_owner2: &mut [u32],
    tp1: &TreePrecomp,
    tp2: &TreePrecomp,
    n: u32,
    num_components: &mut usize,
    region_labels: &[u32],
    target_components: &[Vec<u32>],
    allow_equal: bool,
) -> Option<usize> {
    let mut region_roots = HashSet::<u32>::new();
    for &lbl in region_labels {
        region_roots.insert(uf.find(lbl));
    }

    let mut outside_components: Vec<Vec<u32>> = Vec::new();
    let mut seen = FixedBitSet::with_capacity(n as usize + 1);
    for lbl in 1..=n {
        let r = uf.find(lbl);
        if seen.contains(r as usize) || comp_labels[r as usize].is_empty() {
            continue;
        }
        seen.insert(r as usize);
        if !region_roots.contains(&r) {
            outside_components.push(comp_labels[r as usize].clone());
        }
    }

    let mut candidate_components = outside_components;
    let mut covered_region = FixedBitSet::with_capacity(n as usize + 1);
    for labels in target_components {
        if labels.is_empty() {
            continue;
        }
        for &lbl in labels {
            covered_region.insert(lbl as usize);
        }
        let mut c = labels.clone();
        c.sort_unstable();
        c.dedup();
        candidate_components.push(c);
    }
    for &lbl in region_labels {
        if !covered_region.contains(lbl as usize) {
            candidate_components.push(vec![lbl]);
        }
    }

    candidate_components.sort_unstable_by_key(|labels| {
        (
            labels.first().copied().unwrap_or(u32::MAX),
            labels.len(),
            labels.last().copied().unwrap_or(u32::MAX),
        )
    });

    let new_components = candidate_components.len();
    if allow_equal {
        if new_components > *num_components {
            return None;
        }
    } else if new_components >= *num_components {
        return None;
    }

    let mut all_labels = FixedBitSet::with_capacity(n as usize + 1);
    let mut new_owner1 = vec![0u32; tp1.num_nodes];
    let mut new_owner2 = vec![0u32; tp2.num_nodes];
    for labels in &candidate_components {
        if labels.is_empty() {
            return None;
        }
        for &lbl in labels {
            if lbl == 0 || lbl > n || all_labels.contains(lbl as usize) {
                return None;
            }
            all_labels.insert(lbl as usize);
        }
        if labels.len() <= 1 {
            continue;
        }
        if !is_compatible(tp1, tp2, labels) {
            return None;
        }
        let vs1 = compute_vset(tp1, labels);
        for node_idx in vs1.ones() {
            if new_owner1[node_idx] != 0 {
                return None;
            }
            new_owner1[node_idx] = labels[0];
        }
        let vs2 = compute_vset(tp2, labels);
        for node_idx in vs2.ones() {
            if new_owner2[node_idx] != 0 {
                return None;
            }
            new_owner2[node_idx] = labels[0];
        }
    }
    if all_labels.count_ones(..) != n as usize {
        return None;
    }

    let mut new_uf = UnionFind::new(n as usize + 1);
    let mut new_comp_labels = vec![Vec::new(); n as usize + 1];
    for labels in candidate_components {
        let mut labels = labels;
        labels.sort_unstable();
        labels.dedup();
        let root = labels[0];
        for &lbl in &labels {
            new_uf.parent[lbl as usize] = root;
        }
        new_uf.parent[root as usize] = root;
        new_uf.rank[root as usize] = 1;
        new_comp_labels[root as usize] = labels;
    }

    let merges_delta = num_components.saturating_sub(new_components);
    *uf = new_uf;
    *comp_labels = new_comp_labels;
    node_owner1.copy_from_slice(&new_owner1);
    node_owner2.copy_from_slice(&new_owner2);
    *num_components = new_components;
    Some(merges_delta)
}

#[derive(Default, Clone, Copy)]
struct RegionReoptStats {
    attempts: usize,
    accepted: usize,
    rejected: usize,
    merges: usize,
    best_delta: usize,
    whidden_attempts: usize,
    maf_sat_attempts: usize,
}

#[derive(Default, Clone, Copy)]
struct PlateauStats {
    episodes: usize,
    expanded: usize,
    deduped: usize,
    accepted: usize,
    best_delta: usize,
}

#[derive(Clone)]
struct PartitionState {
    uf: UnionFind,
    comp_labels: Vec<Vec<u32>>,
    node_owner1: Vec<u32>,
    node_owner2: Vec<u32>,
    num_components: usize,
}

impl PartitionState {
    fn capture(
        uf: &UnionFind,
        comp_labels: &[Vec<u32>],
        node_owner1: &[u32],
        node_owner2: &[u32],
        num_components: usize,
    ) -> Self {
        Self {
            uf: uf.clone(),
            comp_labels: comp_labels.to_vec(),
            node_owner1: node_owner1.to_vec(),
            node_owner2: node_owner2.to_vec(),
            num_components,
        }
    }

    fn restore(
        self,
        uf: &mut UnionFind,
        comp_labels: &mut Vec<Vec<u32>>,
        node_owner1: &mut [u32],
        node_owner2: &mut [u32],
        num_components: &mut usize,
    ) {
        *uf = self.uf;
        *comp_labels = self.comp_labels;
        node_owner1.copy_from_slice(&self.node_owner1);
        node_owner2.copy_from_slice(&self.node_owner2);
        *num_components = self.num_components;
    }
}

fn partition_hash(uf: &mut UnionFind, comp_labels: &[Vec<u32>], n: u32) -> u64 {
    let mut roots = active_roots(uf, comp_labels, n);
    let mut chunks: Vec<u64> = Vec::with_capacity(roots.len());
    for r in roots.drain(..) {
        let mut labels = comp_labels[r as usize].clone();
        labels.sort_unstable();
        let mut h: u64 = 1469598103934665603;
        for lbl in labels {
            h ^= lbl as u64;
            h = h.wrapping_mul(1099511628211);
        }
        chunks.push(h);
    }
    chunks.sort_unstable();
    let mut out: u64 = 1469598103934665603;
    for c in chunks {
        out ^= c;
        out = out.wrapping_mul(1099511628211);
    }
    out
}

fn deterministic_plateau_escape_pass(
    instance: &Instance,
    uf: &mut UnionFind,
    comp_labels: &mut Vec<Vec<u32>>,
    node_owner1: &mut [u32],
    node_owner2: &mut [u32],
    tp1: &TreePrecomp,
    tp2: &TreePrecomp,
    n: u32,
    terminate: &AtomicBool,
    started: std::time::Instant,
    time_limit: std::time::Duration,
    num_components: &mut usize,
) -> PlateauStats {
    let mut stats = PlateauStats::default();
    if terminate.load(Ordering::Relaxed) || started.elapsed() >= time_limit {
        return stats;
    }

    let initial_score = *num_components;
    let mut seen: HashSet<u64> = HashSet::new();
    let mut frontier: Vec<PartitionState> = vec![PartitionState::capture(
        uf,
        comp_labels,
        node_owner1,
        node_owner2,
        *num_components,
    )];

    for _episode in 0..PLATEAU_MAX_EPISODES {
        if terminate.load(Ordering::Relaxed) || started.elapsed() >= time_limit {
            break;
        }
        if frontier.is_empty() {
            break;
        }
        stats.episodes += 1;

        let mut next_frontier: Vec<PartitionState> = Vec::new();
        let current_frontier = std::mem::take(&mut frontier);
        for state in current_frontier
            .into_iter()
            .take(PLATEAU_MAX_STATES_PER_EPISODE)
        {
            if terminate.load(Ordering::Relaxed) || started.elapsed() >= time_limit {
                break;
            }

            state.restore(uf, comp_labels, node_owner1, node_owner2, num_components);
            let h = partition_hash(uf, comp_labels, n);
            if !seen.insert(h) {
                stats.deduped += 1;
                continue;
            }
            stats.expanded += 1;

            let mut leaf_to_root = vec![0u32; n as usize + 1];
            for lbl in 1..=n {
                leaf_to_root[lbl as usize] = uf.find(lbl);
            }

            let mut roots = active_roots(uf, comp_labels, n)
                .into_iter()
                .filter(|&r| comp_labels[r as usize].len() > 1)
                .collect::<Vec<_>>();
            if roots.is_empty() {
                continue;
            }

            roots.sort_unstable_by_key(|&r| {
                (
                    std::cmp::Reverse(comp_labels[r as usize].len()),
                    comp_labels[r as usize]
                        .iter()
                        .copied()
                        .min()
                        .unwrap_or(u32::MAX),
                    r,
                )
            });

            for &anchor in roots.iter().take(PLATEAU_MAX_ROOTS) {
                if terminate.load(Ordering::Relaxed) || started.elapsed() >= time_limit {
                    break;
                }

                let neighbors =
                    collect_nearby_roots(anchor, &leaf_to_root, comp_labels, tp1, tp2, 1);
                if neighbors.is_empty() {
                    continue;
                }

                let mut region_roots = vec![anchor];
                for (nb, _, _) in neighbors.into_iter().take(PLATEAU_MAX_NEIGHBORS) {
                    let rn = uf.find(nb);
                    if rn == anchor || comp_labels[rn as usize].len() <= 1 {
                        continue;
                    }
                    if !region_roots.contains(&rn) {
                        region_roots.push(rn);
                    }
                }
                if region_roots.len() <= 1 {
                    continue;
                }

                let mut keep = FixedBitSet::with_capacity(n as usize + 1);
                for &r in &region_roots {
                    for &lbl in &comp_labels[r as usize] {
                        keep.insert(lbl as usize);
                    }
                }
                let region_leaf_count = keep.count_ones(..);
                if region_leaf_count <= 1 || region_leaf_count > PLATEAU_MAX_LABELS {
                    continue;
                }

                let (region_instance, reverse_map) = restrict_instance_simple(instance, &keep);
                let Some((exact_solution, _solver_name)) = exact_local_solve(&region_instance)
                else {
                    continue;
                };
                let mapped = region_components_from_exact(&exact_solution, &reverse_map, n);
                let mut region_labels: Vec<u32> = Vec::new();
                for &r in &region_roots {
                    region_labels.extend_from_slice(&comp_labels[r as usize]);
                }
                region_labels.sort_unstable();
                region_labels.dedup();

                if apply_exact_region_if_improves(
                    uf,
                    comp_labels,
                    node_owner1,
                    node_owner2,
                    tp1,
                    tp2,
                    n,
                    num_components,
                    &region_labels,
                    &mapped,
                    true,
                )
                .is_some()
                {
                    let candidate = PartitionState::capture(
                        uf,
                        comp_labels,
                        node_owner1,
                        node_owner2,
                        *num_components,
                    );
                    let ch = partition_hash(uf, comp_labels, n);
                    if !seen.insert(ch) {
                        stats.deduped += 1;
                    } else {
                        next_frontier.push(candidate);
                        if *num_components < initial_score {
                            stats.accepted += 1;
                            stats.best_delta = stats
                                .best_delta
                                .max(initial_score.saturating_sub(*num_components));
                        }
                    }
                }

                if next_frontier.len() >= PLATEAU_MAX_STATES_PER_EPISODE {
                    break;
                }
            }

            if next_frontier.len() >= PLATEAU_MAX_STATES_PER_EPISODE {
                break;
            }
        }

        if next_frontier.is_empty() {
            break;
        }

        next_frontier.sort_by_key(|s| s.num_components);
        frontier = next_frontier;
    }

    if !frontier.is_empty() {
        frontier.sort_by_key(|s| s.num_components);
        if let Some(best) = frontier.into_iter().next() {
            best.restore(uf, comp_labels, node_owner1, node_owner2, num_components);
        }
    }

    stats
}

fn exact_region_reoptimization_pass(
    instance: &Instance,
    uf: &mut UnionFind,
    comp_labels: &mut Vec<Vec<u32>>,
    node_owner1: &mut [u32],
    node_owner2: &mut [u32],
    tp1: &TreePrecomp,
    tp2: &TreePrecomp,
    n: u32,
    terminate: &AtomicBool,
    started: std::time::Instant,
    time_limit: std::time::Duration,
    num_components: &mut usize,
) -> RegionReoptStats {
    let mut stats = RegionReoptStats::default();
    if terminate.load(Ordering::Relaxed) || started.elapsed() >= time_limit {
        return stats;
    }

    let mut roots = active_roots(uf, comp_labels, n)
        .into_iter()
        .filter(|&r| comp_labels[r as usize].len() > 1)
        .collect::<Vec<_>>();
    if roots.len() <= 1 {
        return stats;
    }

    let mut leaf_to_root = vec![0u32; n as usize + 1];
    for lbl in 1..=n {
        leaf_to_root[lbl as usize] = uf.find(lbl);
    }

    let mut anchor_rank: Vec<(u32, usize, u32, u32)> = Vec::with_capacity(roots.len());
    for &r in &roots {
        let neighbors = collect_nearby_roots(r, &leaf_to_root, comp_labels, tp1, tp2, 1);
        let mut conflict_score = 0u32;
        for (_, hits, _) in &neighbors {
            conflict_score = conflict_score.saturating_add(*hits as u32);
        }
        let min_label = comp_labels[r as usize]
            .iter()
            .copied()
            .min()
            .unwrap_or(u32::MAX);
        anchor_rank.push((r, neighbors.len(), conflict_score, min_label));
    }
    anchor_rank.sort_unstable_by_key(|&(r, deg, score, min_label)| {
        (
            std::cmp::Reverse(score),
            std::cmp::Reverse(deg),
            min_label,
            r,
        )
    });
    roots = anchor_rank.into_iter().map(|(r, _, _, _)| r).collect();

    let mut accepted_once = false;
    'anchor_loop: for &anchor in roots.iter().take(REGION_REOPT_MAX_ROOTS) {
        if terminate.load(Ordering::Relaxed) || started.elapsed() >= time_limit {
            break;
        }
        let anchor = uf.find(anchor);
        if comp_labels[anchor as usize].len() <= 1 {
            continue;
        }

        let neighbors = collect_nearby_roots(anchor, &leaf_to_root, comp_labels, tp1, tp2, 1);
        if neighbors.is_empty() {
            continue;
        }

        let mut region_roots: Vec<u32> = vec![anchor];
        for (neighbor, _, _) in neighbors.into_iter().take(REGION_REOPT_MAX_NEIGHBORS) {
            let rn = uf.find(neighbor);
            if rn == anchor || comp_labels[rn as usize].len() <= 1 {
                continue;
            }
            if !region_roots.contains(&rn) {
                region_roots.push(rn);
            }
        }
        if region_roots.len() <= 1 {
            continue;
        }

        let before = *num_components;
        let mut keep = FixedBitSet::with_capacity(n as usize + 1);
        for &r in &region_roots {
            for &lbl in &comp_labels[r as usize] {
                keep.insert(lbl as usize);
            }
        }

        let region_leaf_count = keep.count_ones(..);
        if region_leaf_count <= 1 || region_leaf_count > REGION_REOPT_MAX_LABELS {
            continue;
        }

        let (region_instance, reverse_map) = restrict_instance_simple(instance, &keep);
        stats.attempts += 1;

        let Some((exact_solution, solver_name)) = exact_local_solve(&region_instance) else {
            stats.rejected += 1;
            continue;
        };

        if solver_name == "whidden" {
            stats.whidden_attempts += 1;
        }
        if solver_name == "sat" {
            stats.maf_sat_attempts += 1;
        }

        let mapped = region_components_from_exact(&exact_solution, &reverse_map, n);
        let mut region_labels: Vec<u32> = Vec::new();
        for &r in &region_roots {
            region_labels.extend_from_slice(&comp_labels[r as usize]);
        }
        region_labels.sort_unstable();
        region_labels.dedup();

        if let Some(merges) = apply_exact_region_if_improves(
            uf,
            comp_labels,
            node_owner1,
            node_owner2,
            tp1,
            tp2,
            n,
            num_components,
            &region_labels,
            &mapped,
            false,
        ) {
            let delta = before.saturating_sub(*num_components);
            stats.accepted += 1;
            stats.merges += merges;
            stats.best_delta = stats.best_delta.max(delta);
            for lbl in 1..=n {
                leaf_to_root[lbl as usize] = uf.find(lbl);
            }
            accepted_once = true;
            break 'anchor_loop;
        } else {
            stats.rejected += 1;
        }
    }

    if !accepted_once {
        // no-op; explicit marker for deterministic pass behavior
    }

    stats
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

        info!("[agglo] Building tree precomps for n={}", n);
        let tp1 = TreePrecomp::build(t1);
        let tp2 = TreePrecomp::build(t2);
        info!(
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
                    ar,
                    br,
                    &mut uf,
                    &mut comp_labels,
                    &mut node_owner1,
                    &mut node_owner2,
                    &tp1,
                    &tp2,
                )
            {
                cherry_merges += 1;
                num_components -= 1;
            }
        }
        info!(
            "[agglo] Phase 1 (common cherries): {} merges, {} components, {:.1}ms",
            cherry_merges,
            num_components,
            start.elapsed().as_secs_f64() * 1000.0,
        );

        // =================================================================
        // Phase 2: single deterministic anytime improvement loop
        // =================================================================
        let phase2_start = std::time::Instant::now();
        let time_limit = std::time::Duration::MAX;
        let mut phase2_merges = 0usize;
        let mut region_stats_total = RegionReoptStats::default();
        let mut plateau_stats_total = PlateauStats::default();
        let mut cycle = 0u32;
        let mut no_improve_streak = 0usize;

        let max_depth = tp1
            .depth
            .iter()
            .chain(tp2.depth.iter())
            .copied()
            .max()
            .unwrap_or(1)
            .max(1);

        loop {
            if self.terminate.load(Ordering::Relaxed) || start.elapsed() >= time_limit {
                break;
            }
            cycle = cycle.saturating_add(1);
            let mut cycle_merges = 0usize;

            for rise_limit in 1..=max_depth {
                if self.terminate.load(Ordering::Relaxed) || start.elapsed() >= time_limit {
                    break;
                }

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

                let mut leaf_to_root = vec![0u32; n as usize + 1];
                for lbl in 1..=n {
                    leaf_to_root[lbl as usize] = uf.find(lbl);
                }

                let mut meta_by_root: HashMap<u32, ComponentMeta> =
                    HashMap::with_capacity(roots.len());
                for &r in &roots {
                    let labels = &comp_labels[r as usize];
                    if labels.is_empty() {
                        continue;
                    }
                    meta_by_root.insert(
                        r,
                        ComponentMeta {
                            size: labels.len() as u16,
                            min_label: labels.iter().copied().min().unwrap_or(u32::MAX),
                            lca1: component_lca(&tp1, labels),
                            lca2: component_lca(&tp2, labels),
                        },
                    );
                }

                let mut pair_seen: HashSet<(u32, u32)> = HashSet::with_capacity(roots.len() * 8);
                let mut pair_support: Vec<(u32, u32, u16, u16)> =
                    Vec::with_capacity(roots.len() * 8);
                for &r in &roots {
                    let rr = uf.find(r);
                    if rr != r || comp_labels[rr as usize].is_empty() {
                        continue;
                    }
                    let neighbors = collect_nearby_roots(
                        rr,
                        &leaf_to_root,
                        &comp_labels,
                        &tp1,
                        &tp2,
                        rise_limit,
                    );
                    for (other, hits, rise_sum) in neighbors {
                        let ro = uf.find(other);
                        if ro == rr || comp_labels[ro as usize].is_empty() {
                            continue;
                        }
                        let (a, b) = if rr < ro { (rr, ro) } else { (ro, rr) };
                        if pair_seen.insert((a, b)) {
                            pair_support.push((a, b, hits, rise_sum));
                        }
                    }
                }

                let mut candidates: Vec<Candidate> = Vec::with_capacity(pair_support.len());
                for (a, b, hits, rise_sum) in pair_support {
                    let Some(ma) = meta_by_root.get(&a).copied() else {
                        continue;
                    };
                    let Some(mb) = meta_by_root.get(&b).copied() else {
                        continue;
                    };

                    let merged_lca1 = tp1.lca(ma.lca1, mb.lca1);
                    let merged_lca2 = tp2.lca(ma.lca2, mb.lca2);
                    let jump1 = if merged_lca1 != NONE && ma.lca1 != NONE && mb.lca1 != NONE {
                        (tp1.depth[ma.lca1 as usize] + tp1.depth[mb.lca1 as usize])
                            .saturating_sub(2 * tp1.depth[merged_lca1 as usize])
                    } else {
                        u16::MAX
                    };
                    let jump2 = if merged_lca2 != NONE && ma.lca2 != NONE && mb.lca2 != NONE {
                        (tp2.depth[ma.lca2 as usize] + tp2.depth[mb.lca2 as usize])
                            .saturating_sub(2 * tp2.depth[merged_lca2 as usize])
                    } else {
                        u16::MAX
                    };

                    let size_sum = ma.size.saturating_add(mb.size);
                    let balance_penalty = ma.size.abs_diff(mb.size);
                    let tree_hits_rank = u16::MAX.saturating_sub(hits);
                    let key = CandidateKey {
                        lca_jump_sum: jump1.saturating_add(jump2),
                        tree_hits_rank,
                        rise_sum,
                        balance_penalty,
                        size_sum,
                        tie_a: ma.min_label.min(mb.min_label),
                        tie_b: ma.min_label.max(mb.min_label),
                    };
                    candidates.push(Candidate { a, b, key });
                }
                candidates.sort_unstable_by_key(|c| c.key);

                let generated = candidates.len();
                let mut pass_merges = 0usize;
                let mut attempts = 0usize;
                for c in candidates {
                    if self.terminate.load(Ordering::Relaxed) || start.elapsed() >= time_limit {
                        break;
                    }
                    let ra = uf.find(c.a);
                    let rb = uf.find(c.b);
                    if ra == rb
                        || comp_labels[ra as usize].is_empty()
                        || comp_labels[rb as usize].is_empty()
                    {
                        continue;
                    }
                    attempts += 1;
                    if try_merge(
                        ra,
                        rb,
                        &mut uf,
                        &mut comp_labels,
                        &mut node_owner1,
                        &mut node_owner2,
                        &tp1,
                        &tp2,
                    ) {
                        pass_merges += 1;
                        cycle_merges += 1;
                        phase2_merges += 1;
                        num_components -= 1;
                    }
                }

                info!(
                    "[agglo] cycle {} rise<={} cand={} attempts={} merges={} comps={} t={:.1}s",
                    cycle,
                    rise_limit,
                    generated,
                    attempts,
                    pass_merges,
                    num_components,
                    phase2_start.elapsed().as_secs_f64(),
                );

                if pass_merges == 0 {
                    break;
                }
            }

            if num_components <= 1 {
                break;
            }

            if cycle_merges == 0 {
                let before = num_components;
                let region_stats = exact_region_reoptimization_pass(
                    instance,
                    &mut uf,
                    &mut comp_labels,
                    &mut node_owner1,
                    &mut node_owner2,
                    &tp1,
                    &tp2,
                    n,
                    &self.terminate,
                    start,
                    time_limit,
                    &mut num_components,
                );

                region_stats_total.attempts += region_stats.attempts;
                region_stats_total.accepted += region_stats.accepted;
                region_stats_total.rejected += region_stats.rejected;
                region_stats_total.merges += region_stats.merges;
                region_stats_total.best_delta =
                    region_stats_total.best_delta.max(region_stats.best_delta);
                region_stats_total.whidden_attempts += region_stats.whidden_attempts;
                region_stats_total.maf_sat_attempts += region_stats.maf_sat_attempts;

                let improved = num_components < before;
                if improved {
                    no_improve_streak = 0;
                } else {
                    no_improve_streak = no_improve_streak.saturating_add(1);
                }

                info!(
                    "[agglo] cycle {} region-reopt: attempts={} accepted={} rejected={} delta={} best-delta={} whidden={} sat={} no-improve-streak={} comps={} t={:.1}s",
                    cycle,
                    region_stats.attempts,
                    region_stats.accepted,
                    region_stats.rejected,
                    before.saturating_sub(num_components),
                    region_stats.best_delta,
                    region_stats.whidden_attempts,
                    region_stats.maf_sat_attempts,
                    no_improve_streak,
                    num_components,
                    phase2_start.elapsed().as_secs_f64(),
                );

                if !improved {
                    let plateau_before = num_components;
                    let plateau_stats = deterministic_plateau_escape_pass(
                        instance,
                        &mut uf,
                        &mut comp_labels,
                        &mut node_owner1,
                        &mut node_owner2,
                        &tp1,
                        &tp2,
                        n,
                        &self.terminate,
                        start,
                        time_limit,
                        &mut num_components,
                    );
                    plateau_stats_total.episodes += plateau_stats.episodes;
                    plateau_stats_total.expanded += plateau_stats.expanded;
                    plateau_stats_total.deduped += plateau_stats.deduped;
                    plateau_stats_total.accepted += plateau_stats.accepted;
                    plateau_stats_total.best_delta =
                        plateau_stats_total.best_delta.max(plateau_stats.best_delta);

                    let plateau_improved = num_components < plateau_before;
                    info!(
                        "[agglo] cycle {} plateau: episodes={} expanded={} deduped={} accepted={} delta={} best-delta={} comps={} t={:.1}s",
                        cycle,
                        plateau_stats.episodes,
                        plateau_stats.expanded,
                        plateau_stats.deduped,
                        plateau_stats.accepted,
                        plateau_before.saturating_sub(num_components),
                        plateau_stats.best_delta,
                        num_components,
                        phase2_start.elapsed().as_secs_f64(),
                    );

                    if plateau_improved {
                        no_improve_streak = 0;
                        continue;
                    }
                    break;
                }
            }
        }

        let total_time = start.elapsed().as_secs_f64();
        let total_merges = cherry_merges + phase2_merges + region_stats_total.merges;
        info!(
            "[agglo] Final: {} components (score={}) in {:.2}s [{} p1 + {} p2 + {} region-merges = {} merges | region attempts={} accepted={} rejected={} best-delta={} whidden={} sat={} | plateau episodes={} expanded={} deduped={} accepted={} best-delta={}]",
            num_components,
            num_components.saturating_sub(1),
            total_time,
            cherry_merges,
            phase2_merges,
            region_stats_total.merges,
            total_merges,
            region_stats_total.attempts,
            region_stats_total.accepted,
            region_stats_total.rejected,
            region_stats_total.best_delta,
            region_stats_total.whidden_attempts,
            region_stats_total.maf_sat_attempts,
            plateau_stats_total.episodes,
            plateau_stats_total.expanded,
            plateau_stats_total.deduped,
            plateau_stats_total.accepted,
            plateau_stats_total.best_delta,
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

// ── Unified Solver impl + entry point ───────────────────────────────────────
use crate::{RunConfig, Solver, Track};

impl Solver for AgglomerativeSolver {
    type Config = ();
    const SUPPORTED_TRACKS: &'static [Track] = &[Track::Heuristic];
    fn solve(&mut self, inst: &Instance, _cfg: &RunConfig<Self::Config>) -> Option<Vec<Tree>> {
        AgglomerativeSolver::solve(self, inst)
    }
    fn stats(&self) -> &SolverStats {
        &self.stats
    }
    fn sigterm_handler(&self, _track: Track) -> Option<Box<dyn Fn() + Send + Sync>> {
        let flag = self.terminate.clone();
        Some(Box::new(move || flag.store(true, Ordering::Relaxed)))
    }
}

pub fn main() {
    crate::run(AgglomerativeSolver::new(), RunConfig { track: Track::Heuristic, ..Default::default() });
}
