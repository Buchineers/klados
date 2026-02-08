//! Shi et al. (2018) parameterized algorithm for MAF on multiple rooted trees.
//!
//! Implements Alg-Maf from "A parameterized algorithm for the Maximum Agreement
//! Forest problem on multiple rooted multifurcating trees" (JCSS 97, 2018).
//!
//! The algorithm interleaves:
//!   - BR-LSI: Achieves Label-Set Isomorphism via Reduction Rule 1 + Branching Rule 1
//!   - MSS branching (Section 4): Sibling-pair based branching on ALL forests
//!
//! Key insight: branching operations can be applied on DIFFERENT trees,
//! not just a fixed reference tree. This gives O(2.42^k m^3 n^4).

use fixedbitset::FixedBitSet;
use klados_core::{Instance, NodeId, SolverStats, Tree, NONE};

fn trace_enabled() -> bool {
    std::env::var("SHI_MESTEL_TRACE").ok().as_deref() == Some("1")
}

macro_rules! trace {
    ($($arg:tt)*) => {
        if trace_enabled() {
            eprintln!($($arg)*);
        }
    };
}

// ============================================================================
// XForest: forest representation (tree with cut edges)
// ============================================================================

#[derive(Clone, Debug)]
struct XForest {
    tree: Tree,
    cut_edges: FixedBitSet,
    live_leafsets: Vec<FixedBitSet>,
    component_roots: Vec<NodeId>,
}

impl XForest {
    fn from_tree(tree: Tree) -> Self {
        let num_nodes = tree.num_nodes();
        let num_leaves = tree.num_leaves;
        let mut leafsets = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let mut set = FixedBitSet::with_capacity(num_leaves as usize + 1);
            set.grow(num_leaves as usize + 1);
            leafsets.push(set);
        }
        for node in tree.post_order() {
            if let Some(lbl) = tree.leaf_label(node) {
                leafsets[node as usize].insert(lbl as usize);
            } else if let Some((l, r)) = tree.children(node) {
                let left = leafsets[l as usize].clone();
                let right = leafsets[r as usize].clone();
                leafsets[node as usize].union_with(&left);
                leafsets[node as usize].union_with(&right);
            }
        }
        let root = tree.root;
        Self {
            tree,
            cut_edges: FixedBitSet::with_capacity(num_nodes),
            live_leafsets: leafsets,
            component_roots: vec![root],
        }
    }

    fn is_cut(&self, node: NodeId) -> bool {
        self.cut_edges.contains(node as usize)
    }

    fn cut(&mut self, node: NodeId) {
        debug_assert!(node != self.tree.root, "Cannot cut above root");
        if !self.cut_edges.contains(node as usize) {
            self.cut_edges.insert(node as usize);
            self.component_roots.push(node);
            let removed = self.live_leafsets[node as usize].clone();
            let mut cur = self.tree.parent[node as usize];
            while cur != NONE {
                self.live_leafsets[cur as usize].difference_with(&removed);
                if self.is_cut(cur) {
                    break;
                }
                cur = self.tree.parent[cur as usize];
            }
        }
    }

    fn component_root(&self, mut node: NodeId) -> NodeId {
        while !self.is_cut(node) && self.tree.parent[node as usize] != NONE {
            node = self.tree.parent[node as usize];
        }
        node
    }

}

// ============================================================================
// ShiMestelSolver
// ============================================================================

pub struct ShiMestelSolver {
    stats: SolverStats,
}

impl ShiMestelSolver {
    pub fn new() -> Self {
        Self {
            stats: SolverStats::default(),
        }
    }

    pub fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        if instance.trees.is_empty() {
            return None;
        }
        if instance.num_trees() == 1 {
            return Some(vec![instance.trees[0].clone()]);
        }

        let label_space = instance.num_leaves as usize;

        let forests: Vec<XForest> = instance
            .trees
            .iter()
            .map(|t| XForest::from_tree(t.clone()))
            .collect();

        // Iterative deepening on target_s (target MAF size = number of components)
        for target_s in 1..=instance.num_leaves as usize {
            self.stats = SolverStats::default();
            trace!("trying target_s={}", target_s);

            let collapses: Collapses = Vec::new();
            if let Some(result) = alg_maf(
                forests.clone(),
                target_s,
                &collapses,
                label_space,
                instance.num_leaves,
                &mut self.stats,
            ) {
                trace!(
                    "solution found: target_s={}, components={}",
                    target_s,
                    result.len()
                );
                return Some(result);
            }
        }

        Some(trivial_forest(&instance.trees[0], instance.num_leaves))
    }
}

impl super::ExactSolver for ShiMestelSolver {
    fn name(&self) -> &'static str {
        "shi-mestel"
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        ShiMestelSolver::solve(self, instance)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

// ============================================================================
// Alg-Maf: Main recursive algorithm (Figure 3 of Shi et al. 2018)
// ============================================================================

/// Compute the maximum order (component count) across all forests.
fn max_order(forests: &[XForest], label_space: usize) -> usize {
    forests
        .iter()
        .map(|f| component_leaf_sets_xf(f, label_space).len())
        .max()
        .unwrap_or(1)
}

/// Tracks label collapses from Reduction Rule 2.
/// Each entry (removed, kept) means label `removed` was merged into `kept`.
type Collapses = Vec<(u32, u32)>;

fn alg_maf(
    mut forests: Vec<XForest>,
    target_s: usize,
    collapses: &Collapses,
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
) -> Option<Vec<Tree>> {
    stats.nodes_explored += 1;

    // Apply Reduction Rule 1 exhaustively
    apply_reduction_rules(&mut forests, label_space);

    // Budget check: max order must not exceed target
    let cur_order = max_order(&forests, label_space);
    if cur_order > target_s {
        stats.branches_pruned += 1;
        return None;
    }

    // Step 2: If F does not satisfy LSI, use BR-LSI
    if !all_pairs_lsi(&forests, label_space) {
        return br_lsi_step(&mut forests, target_s, collapses, label_space, num_leaves, stats);
    }

    // Step 3: LSI satisfied. Check if F1 is isomorphic to all others.
    // First apply Reduction Rule 2 exhaustively (common sibling-pairs)
    let mut collapses = collapses.clone();
    loop {
        if let Some((removed, kept)) = apply_reduction_rule_2(&mut forests, label_space) {
            collapses.push((removed, kept));
        } else {
            break;
        }
    }

    // Check if all forests are isomorphic
    if all_forests_isomorphic(&forests, label_space) {
        // Step 4: Return F1 as the MAF, expanding collapsed labels
        return Some(extract_maf_components(&forests[0], &collapses, label_space, num_leaves));
    }

    // Remaining budget
    let remaining = target_s - cur_order;
    if remaining == 0 {
        stats.branches_pruned += 1;
        return None;
    }

    // Find minimum sibling-pair in any forest and apply Case 2 branching
    let (a, b) = match find_minimum_sibling_pair(&forests, label_space) {
        Some(pair) => pair,
        None => {
            // No sibling pair found → can't make progress
            return None;
        }
    };

    trace!("MSS pair: a={}, b={}, remaining={}", a, b, remaining);
    apply_case_2_branching(&forests, target_s, &collapses, a, b, label_space, num_leaves, stats)
}

// ============================================================================
// BR-LSI step (Section 3): Find LSI violation and branch
// ============================================================================

fn br_lsi_step(
    forests: &mut Vec<XForest>,
    target_s: usize,
    collapses: &Collapses,
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
) -> Option<Vec<Tree>> {
    // Find violating pair (Fi, Fj)
    let (i, j) = match find_violating_pair(forests, label_space) {
        Some(pair) => pair,
        None => return None,
    };

    // Find branching vertex: try fi relative to fj, then fj relative to fi
    let (target_idx, v1, v2) =
        if let Some((_v, v1, v2)) = find_branching_vertex(&forests[i], &forests[j], label_space) {
            (i, v1, v2)
        } else if let Some((_v, v1, v2)) =
            find_branching_vertex(&forests[j], &forests[i], label_space)
        {
            (j, v1, v2)
        } else {
            trace!("no branching vertex found for pair ({}, {})", i, j);
            stats.branches_pruned += 1;
            return None;
        };

    trace!("BR1: forest={}, v1={}, v2={}", target_idx, v1, v2);

    // Branch 1: cut edge above v1
    if v1 != forests[target_idx].tree.root && !forests[target_idx].is_cut(v1) {
        let mut f1 = forests.clone();
        f1[target_idx].cut(v1);
        if let Some(result) = alg_maf(f1, target_s, collapses, label_space, num_leaves, stats) {
            return Some(result);
        }
    }

    // Branch 2: cut edge above v2
    if v2 != forests[target_idx].tree.root && !forests[target_idx].is_cut(v2) {
        let mut f2 = forests.clone();
        f2[target_idx].cut(v2);
        if let Some(result) = alg_maf(f2, target_s, collapses, label_space, num_leaves, stats) {
            return Some(result);
        }
    }

    stats.branches_pruned += 1;
    None
}

// ============================================================================
// Reduction Rule 1 (Section 3.1)
// ============================================================================

fn apply_reduction_rules(forests: &mut [XForest], label_space: usize) {
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..forests.len() {
            for j in 0..forests.len() {
                if i == j {
                    continue;
                }
                if apply_reduction_rule_1_pair(forests, i, j, label_space) {
                    changed = true;
                }
            }
        }
    }
}

fn apply_reduction_rule_1_pair(
    forests: &mut [XForest],
    i: usize,
    j: usize,
    label_space: usize,
) -> bool {
    let fj_components = component_leaf_sets_xf(&forests[j], label_space);

    for node in forests[i].tree.pre_order().collect::<Vec<_>>() {
        if forests[i].is_cut(node) || node == forests[i].tree.root {
            continue;
        }
        if forests[i].live_leafsets[node as usize].count_ones(..) == 0 {
            continue;
        }

        let parent = forests[i].tree.parent[node as usize];
        if parent == NONE {
            continue;
        }

        let node_ls = &forests[i].live_leafsets[node as usize];

        // Find which component of fi this node belongs to
        let comp_root = forests[i].component_root(node);
        let comp_ls = &forests[i].live_leafsets[comp_root as usize];

        // Check: is node_ls exactly a union of (fj_component ∩ comp_ls) sets?
        let mut union_matching = FixedBitSet::with_capacity(label_space + 1);
        union_matching.grow(label_space + 1);
        for fj_comp in &fj_components {
            let mut inter = fj_comp.clone();
            inter.intersect_with(comp_ls);
            if inter.count_ones(..) == 0 {
                continue;
            }
            if is_subset(&inter, node_ls) {
                union_matching.union_with(&inter);
            }
        }

        if union_matching == *node_ls
            && union_matching.count_ones(..) > 0
            && union_matching.count_ones(..) < comp_ls.count_ones(..)
        {
            trace!("R1: cut node {} in forest {}", node, i);
            forests[i].cut(node);
            return true;
        }
    }
    false
}

// ============================================================================
// LSI checking (Section 2.2)
// ============================================================================

fn all_pairs_lsi(forests: &[XForest], label_space: usize) -> bool {
    if forests.len() <= 1 {
        return true;
    }
    let sets: Vec<Vec<FixedBitSet>> = forests
        .iter()
        .map(|f| component_leaf_sets_xf(f, label_space))
        .collect();
    let keys: Vec<Vec<Vec<usize>>> = sets
        .iter()
        .map(|components| {
            let mut ks: Vec<Vec<usize>> = components.iter().map(|s| leafset_key(s)).collect();
            ks.sort();
            ks
        })
        .collect();
    keys.windows(2).all(|w| w[0] == w[1])
}

fn find_violating_pair(forests: &[XForest], label_space: usize) -> Option<(usize, usize)> {
    let sets: Vec<Vec<FixedBitSet>> = forests
        .iter()
        .map(|f| component_leaf_sets_xf(f, label_space))
        .collect();
    for i in 0..forests.len() {
        for j in (i + 1)..forests.len() {
            if !lsi_pair(&sets[i], &sets[j]) {
                return Some((i, j));
            }
        }
    }
    None
}

fn lsi_pair(a: &[FixedBitSet], b: &[FixedBitSet]) -> bool {
    let mut a_keys: Vec<Vec<usize>> = a.iter().map(|s| leafset_key(s)).collect();
    let mut b_keys: Vec<Vec<usize>> = b.iter().map(|s| leafset_key(s)).collect();
    a_keys.sort();
    b_keys.sort();
    a_keys == b_keys
}

// ============================================================================
// Branching Rule 1: find branching vertex (Section 3.1, Case 1)
// ============================================================================

fn find_branching_vertex(
    fi: &XForest,
    fj: &XForest,
    label_space: usize,
) -> Option<(NodeId, NodeId, NodeId)> {
    let fj_components = component_leaf_sets_xf(fj, label_space);

    for node in fi.tree.pre_order() {
        if fi.live_leafsets[node as usize].count_ones(..) == 0 {
            continue;
        }
        let children = active_children_xf(fi, node);
        if children.len() < 2 {
            continue;
        }

        let (c1, c2) = (children[0], children[1]);
        let ls1 = &fi.live_leafsets[c1 as usize];
        let ls2 = &fi.live_leafsets[c2 as usize];

        for comp in &fj_components {
            let c1_inter = has_intersection(ls1, comp);
            let c2_inter = has_intersection(ls2, comp);

            if c1_inter && !c2_inter && is_subset(ls1, comp) {
                return Some((node, c1, c2));
            }
            if c2_inter && !c1_inter && is_subset(ls2, comp) {
                return Some((node, c2, c1));
            }
        }
    }
    None
}

// ============================================================================
// Reduction Rule 2 (Section 4): Common sibling-pair collapse
// ============================================================================

/// If labels a,b are a sibling-pair in ALL forests, collapse them.
/// Returns Some((removed, kept)) if a collapse was made.
fn apply_reduction_rule_2(forests: &mut [XForest], label_space: usize) -> Option<(u32, u32)> {
    // Find a sibling-pair that exists in ALL forests
    let pair = find_common_sibling_pair(forests, label_space);
    if let Some((a, b)) = pair {
        trace!("R2: collapsing common sibling-pair ({}, {})", a, b);
        // In each forest, deactivate label a (keep b as representative)
        for forest in forests.iter_mut() {
            let a_node = forest.tree.label_to_node[a as usize];
            forest.live_leafsets[a_node as usize].clear();
            let mut cur = forest.tree.parent[a_node as usize];
            while cur != NONE {
                forest.live_leafsets[cur as usize].set(a as usize, false);
                if forest.is_cut(cur) {
                    break;
                }
                cur = forest.tree.parent[cur as usize];
            }
        }
        return Some((a, b)); // a was removed, b was kept
    }
    None
}

/// Find a sibling-pair {a,b} that is a sibling-pair in ALL forests.
fn find_common_sibling_pair(forests: &[XForest], label_space: usize) -> Option<(u32, u32)> {
    if forests.is_empty() {
        return None;
    }
    // Find all sibling-pairs in the first forest
    let pairs = find_all_sibling_pairs(&forests[0], label_space);
    // Check if any is also a sibling-pair in all other forests
    'outer: for (a, b) in &pairs {
        for forest in &forests[1..] {
            if !is_sibling_pair_in_forest(forest, *a, *b) {
                continue 'outer;
            }
        }
        return Some((*a, *b));
    }
    None
}

/// Find all sibling-pairs in a forest. A sibling-pair {a,b} means:
/// after forced contraction, a and b are the only two children of some node.
fn find_all_sibling_pairs(forest: &XForest, _label_space: usize) -> Vec<(u32, u32)> {
    let mut pairs = Vec::new();
    for node in forest.tree.pre_order() {
        if forest.live_leafsets[node as usize].count_ones(..) == 0 {
            continue;
        }
        if forest.tree.is_leaf(node) {
            continue;
        }
        // Get effective children (after forced contraction)
        let children = forest_children(forest, node);
        if children.len() == 2 {
            let c1 = children[0];
            let c2 = children[1];
            // Both must be leaves
            let c1_leaf = forest_is_leaf(forest, c1);
            let c2_leaf = forest_is_leaf(forest, c2);
            if c1_leaf && c2_leaf {
                let lbl1 = forest.tree.leaf_label(c1);
                let lbl2 = forest.tree.leaf_label(c2);
                if let (Some(l1), Some(l2)) = (lbl1, lbl2) {
                    pairs.push((l1.min(l2), l1.max(l2)));
                }
            }
        }
    }
    pairs
}

/// Check if labels a and b form a sibling-pair in the given forest.
fn is_sibling_pair_in_forest(forest: &XForest, a: u32, b: u32) -> bool {
    let a_node = forest.tree.label_to_node[a as usize];
    let b_node = forest.tree.label_to_node[b as usize];
    if forest.live_leafsets[a_node as usize].count_ones(..) == 0
        || forest.live_leafsets[b_node as usize].count_ones(..) == 0
    {
        return false;
    }
    // Find effective parent of a
    let pa = forest_parent_leaf(forest, a_node);
    let pb = forest_parent_leaf(forest, b_node);
    if pa == NONE || pa != pb {
        return false;
    }
    // Check that parent has exactly 2 effective children (both leaves a and b)
    let children = forest_children(forest, pa);
    if children.len() != 2 {
        return false;
    }
    // Check both children resolve to a_node and b_node
    let c1_is_a = forest_resolves_to(forest, children[0], a_node);
    let c2_is_b = forest_resolves_to(forest, children[1], b_node);
    let c1_is_b = forest_resolves_to(forest, children[0], b_node);
    let c2_is_a = forest_resolves_to(forest, children[1], a_node);
    (c1_is_a && c2_is_b) || (c1_is_b && c2_is_a)
}

/// Check if walking down from `start` through single-child chains reaches `target`.
fn forest_resolves_to(forest: &XForest, start: NodeId, target: NodeId) -> bool {
    let mut cur = start;
    loop {
        if cur == target {
            return true;
        }
        if forest.tree.is_leaf(cur) {
            return cur == target;
        }
        let children = active_children_xf(forest, cur);
        if children.len() == 1 {
            cur = children[0];
        } else {
            return cur == target;
        }
    }
}

// ============================================================================
// Forest navigation (with forced contraction)
// ============================================================================

/// Get effective children of a node (skip cut edges, contract single-child chains).
fn forest_children(forest: &XForest, node: NodeId) -> Vec<NodeId> {
    let mut out = Vec::with_capacity(2);
    if let Some((left, right)) = forest.tree.children(node) {
        if left != NONE
            && !forest.is_cut(left)
            && forest.live_leafsets[left as usize].count_ones(..) > 0
        {
            out.push(descend_to_effective(forest, left));
        }
        if right != NONE
            && !forest.is_cut(right)
            && forest.live_leafsets[right as usize].count_ones(..) > 0
        {
            out.push(descend_to_effective(forest, right));
        }
    }
    out
}

/// Descend through single-child internal nodes (forced contraction).
fn descend_to_effective(forest: &XForest, mut node: NodeId) -> NodeId {
    loop {
        if forest.tree.is_leaf(node) {
            return node;
        }
        let children = active_children_xf(forest, node);
        if children.len() == 1 {
            node = children[0];
        } else {
            return node;
        }
    }
}

/// Check if a node is an effective leaf (leaf or has no active children).
fn forest_is_leaf(forest: &XForest, node: NodeId) -> bool {
    if forest.tree.is_leaf(node) {
        return true;
    }
    active_children_xf(forest, node).is_empty()
}

/// Get effective parent of a leaf node (skip single-child ancestors).
fn forest_parent_leaf(forest: &XForest, node: NodeId) -> NodeId {
    if node == forest.tree.root || forest.is_cut(node) {
        return NONE;
    }
    let mut cur = forest.tree.parent[node as usize];
    if cur == NONE {
        return NONE;
    }
    loop {
        let active = active_children_xf(forest, cur);
        if active.len() >= 2 {
            return cur;
        }
        // cur has < 2 active children; if it's a component root, no valid parent
        if forest.is_cut(cur) {
            return NONE;
        }
        // Single-child: keep going up (forced contraction)
        let p = forest.tree.parent[cur as usize];
        if p == NONE {
            return cur; // tree root with 1 child
        }
        cur = p;
    }
}

/// Find the LCA of two nodes in the forest (respecting cuts).
fn forest_lca(forest: &XForest, a: NodeId, b: NodeId) -> NodeId {
    let mut ancestors = std::collections::HashSet::new();
    let mut cur = a;
    ancestors.insert(cur);
    loop {
        if forest.is_cut(cur) || forest.tree.parent[cur as usize] == NONE {
            break;
        }
        cur = forest.tree.parent[cur as usize];
        ancestors.insert(cur);
    }
    cur = b;
    if ancestors.contains(&cur) {
        return cur;
    }
    loop {
        if forest.is_cut(cur) || forest.tree.parent[cur as usize] == NONE {
            break;
        }
        cur = forest.tree.parent[cur as usize];
        if ancestors.contains(&cur) {
            return cur;
        }
    }
    NONE
}

// ============================================================================
// Case 2 branching (Section 4.1): |S| = 2 (sibling-pair case)
// ============================================================================

/// Find the minimum sibling-pair across all forests.
/// Returns (a, b) where a < b are leaf labels.
fn find_minimum_sibling_pair(forests: &[XForest], label_space: usize) -> Option<(u32, u32)> {
    // Try each forest to find any sibling-pair
    for forest in forests {
        let pairs = find_all_sibling_pairs(forest, label_space);
        if let Some(&(a, b)) = pairs.first() {
            return Some((a, b));
        }
    }
    None
}

/// Compute E_F(a, b): the set of "off-path" edges between labels a and b in forest F.
/// For binary trees: these are edges from internal nodes on the path a→lca→b
/// to children NOT on the path.
/// Returns the list of child nodes whose parent edges form E_F(a,b).
fn compute_e_f(forest: &XForest, a: u32, b: u32) -> Vec<NodeId> {
    let a_node = forest.tree.label_to_node[a as usize];
    let b_node = forest.tree.label_to_node[b as usize];

    if forest.live_leafsets[a_node as usize].count_ones(..) == 0
        || forest.live_leafsets[b_node as usize].count_ones(..) == 0
    {
        return Vec::new();
    }

    let lca = forest_lca(forest, a_node, b_node);
    if lca == NONE {
        return Vec::new(); // Different components
    }

    // Collect nodes on the path from a to lca and from b to lca
    let mut on_path = std::collections::HashSet::new();
    on_path.insert(a_node);
    on_path.insert(b_node);
    on_path.insert(lca);

    let mut cur = a_node;
    while cur != lca {
        if forest.is_cut(cur) {
            break;
        }
        let p = forest.tree.parent[cur as usize];
        if p == NONE {
            break;
        }
        on_path.insert(p);
        cur = p;
    }
    cur = b_node;
    while cur != lca {
        if forest.is_cut(cur) {
            break;
        }
        let p = forest.tree.parent[cur as usize];
        if p == NONE {
            break;
        }
        on_path.insert(p);
        cur = p;
    }

    // E1_F: for each internal node on path (excluding a, b, lca), collect off-path children
    // E2_F: for lca, collect children not on path and not parent edge
    let mut e_f = Vec::new();

    for &path_node in &on_path {
        if path_node == a_node || path_node == b_node {
            continue;
        }
        if let Some((left, right)) = forest.tree.children(path_node) {
            if left != NONE
                && !forest.is_cut(left)
                && forest.live_leafsets[left as usize].count_ones(..) > 0
                && !on_path.contains(&left)
            {
                e_f.push(left);
            }
            if right != NONE
                && !forest.is_cut(right)
                && forest.live_leafsets[right as usize].count_ones(..) > 0
                && !on_path.contains(&right)
            {
                e_f.push(right);
            }
        }
    }

    e_f
}

/// Compute L(lca_F(a, b)): the leaf-set of the LCA of a and b in forest F.
fn lca_leafset(forest: &XForest, a: u32, b: u32) -> FixedBitSet {
    let a_node = forest.tree.label_to_node[a as usize];
    let b_node = forest.tree.label_to_node[b as usize];
    let lca = forest_lca(forest, a_node, b_node);
    if lca == NONE {
        FixedBitSet::new()
    } else {
        forest.live_leafsets[lca as usize].clone()
    }
}

fn apply_case_2_branching(
    forests: &[XForest],
    target_s: usize,
    collapses: &Collapses,
    a: u32,
    b: u32,
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
) -> Option<Vec<Tree>> {
    // Compute E_Fi(a,b) for all forests
    let e_sets: Vec<Vec<NodeId>> = forests.iter().map(|f| compute_e_f(f, a, b)).collect();
    let max_e = e_sets.iter().map(|e| e.len()).max().unwrap_or(0);

    // Case 2.1: Some forest has |E_F(a,b)| >= 2
    if max_e >= 2 {
        return apply_branching_rule_2_1(forests, target_s, collapses, a, b, &e_sets, label_space, num_leaves, stats);
    }

    // Case 2.2: All forests have |E_F(a,b)| <= 1
    // Ω1 = forests where |E_F| = 1 (a and b are not siblings)
    let omega1: Vec<usize> = e_sets
        .iter()
        .enumerate()
        .filter(|(_, e)| e.len() == 1)
        .map(|(i, _)| i)
        .collect();

    // If Ω1 is empty, a and b are siblings everywhere → RR2 should have caught this
    if omega1.is_empty() {
        return None;
    }

    // Check Case 2.2.1 vs 2.2.2: do all forests in Ω1 have the same L(lca(a,b))?
    let lca_sets: Vec<FixedBitSet> = omega1.iter().map(|&i| lca_leafset(&forests[i], a, b)).collect();
    let all_same_lca = lca_sets.windows(2).all(|w| w[0] == w[1]);

    if all_same_lca {
        // Case 2.2.1: Reduction Rule 2.2.1 (deterministic, no branching)
        trace!("Case 2.2.1: reduction for ({}, {})", a, b);
        return apply_reduction_rule_2_2_1(forests, target_s, collapses, a, b, &e_sets, label_space, num_leaves, stats);
    }

    // Case 2.2.2: Different L(lca) values
    trace!("Case 2.2.2: branching for ({}, {})", a, b);
    apply_branching_rule_2_2_2(forests, target_s, collapses, a, b, &e_sets, &omega1, label_space, num_leaves, stats)
}

/// Branching Rule 2.1: 3-way branch.
/// [1] remove edge to a in ALL forests
/// [2] remove edge to b in ALL forests
/// [3] remove E_Fi(a,b) for all i (make a,b siblings everywhere)
fn apply_branching_rule_2_1(
    forests: &[XForest],
    target_s: usize,
    collapses: &Collapses,
    a: u32,
    b: u32,
    e_sets: &[Vec<NodeId>],
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
) -> Option<Vec<Tree>> {
    trace!("BR 2.1: a={}, b={}", a, b);

    // Branch [1]: remove edge incident to a in all forests
    {
        let mut next = forests.to_vec();
        for f in &mut next {
            let a_node = f.tree.label_to_node[a as usize];
            if a_node != f.tree.root && !f.is_cut(a_node) {
                f.cut(a_node);
            }
        }
        if let Some(result) = alg_maf(next, target_s, collapses, label_space, num_leaves, stats) {
            return Some(result);
        }
    }

    // Branch [2]: remove edge incident to b in all forests
    {
        let mut next = forests.to_vec();
        for f in &mut next {
            let b_node = f.tree.label_to_node[b as usize];
            if b_node != f.tree.root && !f.is_cut(b_node) {
                f.cut(b_node);
            }
        }
        if let Some(result) = alg_maf(next, target_s, collapses, label_space, num_leaves, stats) {
            return Some(result);
        }
    }

    // Branch [3]: remove E_Fi(a,b) for all i (make a,b siblings in all forests)
    {
        let mut next = forests.to_vec();
        let mut any_cut = false;
        for (i, e_nodes) in e_sets.iter().enumerate() {
            for &node in e_nodes {
                if node != next[i].tree.root && !next[i].is_cut(node) {
                    next[i].cut(node);
                    any_cut = true;
                }
            }
        }
        if any_cut {
            if let Some(result) = alg_maf(next, target_s, collapses, label_space, num_leaves, stats) {
                return Some(result);
            }
        }
    }

    None
}

/// Reduction Rule 2.2.1: Remove E_Fi(a,b) for all i (deterministic, no branching).
/// The order may increase, but target_s stays the same. The max_order check
/// in alg_maf will correctly handle the budget.
fn apply_reduction_rule_2_2_1(
    forests: &[XForest],
    target_s: usize,
    collapses: &Collapses,
    _a: u32,
    _b: u32,
    e_sets: &[Vec<NodeId>],
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
) -> Option<Vec<Tree>> {
    let mut next = forests.to_vec();
    for (i, e_nodes) in e_sets.iter().enumerate() {
        for &node in e_nodes {
            if node != next[i].tree.root && !next[i].is_cut(node) {
                next[i].cut(node);
            }
        }
    }
    alg_maf(next, target_s, collapses, label_space, num_leaves, stats)
}

/// Branching Rule 2.2.2: 3-way branch.
/// [1] remove edge to a in all forests
/// [2] remove edge to b in all forests
/// [3] remove E_Fi(a,b) for all i, then recurse (LSI will be violated, BR-LSI fires)
fn apply_branching_rule_2_2_2(
    forests: &[XForest],
    target_s: usize,
    collapses: &Collapses,
    a: u32,
    b: u32,
    e_sets: &[Vec<NodeId>],
    _omega1: &[usize],
    label_space: usize,
    num_leaves: u32,
    stats: &mut SolverStats,
) -> Option<Vec<Tree>> {
    trace!("BR 2.2.2: a={}, b={}", a, b);

    // Branch [1]: remove edge incident to a in all forests
    {
        let mut next = forests.to_vec();
        for f in &mut next {
            let a_node = f.tree.label_to_node[a as usize];
            if a_node != f.tree.root && !f.is_cut(a_node) {
                f.cut(a_node);
            }
        }
        if let Some(result) = alg_maf(next, target_s, collapses, label_space, num_leaves, stats) {
            return Some(result);
        }
    }

    // Branch [2]: remove edge incident to b in all forests
    {
        let mut next = forests.to_vec();
        for f in &mut next {
            let b_node = f.tree.label_to_node[b as usize];
            if b_node != f.tree.root && !f.is_cut(b_node) {
                f.cut(b_node);
            }
        }
        if let Some(result) = alg_maf(next, target_s, collapses, label_space, num_leaves, stats) {
            return Some(result);
        }
    }

    // Branch [3]: remove E_Fi(a,b) for all i, then recurse
    {
        let mut next = forests.to_vec();
        let mut any_cut = false;
        for (i, e_nodes) in e_sets.iter().enumerate() {
            for &node in e_nodes {
                if node != next[i].tree.root && !next[i].is_cut(node) {
                    next[i].cut(node);
                    any_cut = true;
                }
            }
        }
        if any_cut {
            if let Some(result) = alg_maf(next, target_s, collapses, label_space, num_leaves, stats) {
                return Some(result);
            }
        }
    }

    None
}

// ============================================================================
// Isomorphism check and MAF extraction
// ============================================================================

/// Check if all forests are isomorphic (same component structure).
fn all_forests_isomorphic(forests: &[XForest], label_space: usize) -> bool {
    if forests.len() <= 1 {
        return true;
    }
    // First check LSI (should already hold)
    if !all_pairs_lsi(forests, label_space) {
        return false;
    }

    // For each component (matched by label-set), check if the subtree
    // topology is the same across all forests.
    let ref_comps = component_leaf_sets_xf(&forests[0], label_space);

    for comp_ls in &ref_comps {
        if comp_ls.count_ones(..) <= 1 {
            continue;
        }
        let ref_sub = forests[0].tree.prune_to_leafset(comp_ls);
        let ref_canon = canonical_form(&ref_sub);
        for forest in &forests[1..] {
            let sub = forest.tree.prune_to_leafset(comp_ls);
            if canonical_form(&sub) != ref_canon {
                return false;
            }
        }
    }
    true
}

/// Extract MAF components from a forest (the components of F1).
fn extract_maf_components(
    forest: &XForest,
    collapses: &Collapses,
    label_space: usize,
    num_leaves: u32,
) -> Vec<Tree> {
    // Build a mapping: for each surviving label, collect all original labels
    // that were collapsed into it (transitively).
    let mut collapsed_into: Vec<u32> = (0..=num_leaves).collect(); // identity initially
    // Process collapses: each (removed, kept) means removed → kept
    // We need transitive closure: if a→b and b→c, then a→c
    for &(removed, kept) in collapses {
        collapsed_into[removed as usize] = kept;
    }
    // Resolve transitive chains
    for lbl in 1..=num_leaves {
        let mut cur = lbl;
        while collapsed_into[cur as usize] != cur {
            cur = collapsed_into[cur as usize];
        }
        collapsed_into[lbl as usize] = cur;
    }

    // For each component in the forest, expand its leaf-set with collapsed labels
    let comps = component_leaf_sets_xf(forest, label_space);
    let mut result = Vec::new();
    for comp_ls in &comps {
        if comp_ls.count_ones(..) == 0 {
            continue;
        }
        // Expand: for each original label, check if it collapsed into a label in comp_ls
        let mut expanded = FixedBitSet::with_capacity(label_space + 1);
        expanded.grow(label_space + 1);
        for lbl in 1..=num_leaves {
            let target = collapsed_into[lbl as usize];
            if comp_ls.contains(target as usize) {
                expanded.insert(lbl as usize);
            }
        }

        if expanded.count_ones(..) == 1 {
            let lbl = expanded.ones().next().unwrap() as u32;
            result.push(make_singleton_tree(lbl, num_leaves));
        } else {
            // Prune the ORIGINAL tree to this expanded leaf-set
            result.push(forest.tree.prune_to_leafset(&expanded));
        }
    }
    result
}

// ============================================================================
// XForest navigation helpers
// ============================================================================

fn active_children_xf(forest: &XForest, node: NodeId) -> Vec<NodeId> {
    let tree = &forest.tree;
    let mut out = Vec::with_capacity(2);
    if let Some((left, right)) = tree.children(node) {
        if left != NONE
            && !forest.is_cut(left)
            && forest.live_leafsets[left as usize].count_ones(..) > 0
        {
            out.push(left);
        }
        if right != NONE
            && !forest.is_cut(right)
            && forest.live_leafsets[right as usize].count_ones(..) > 0
        {
            out.push(right);
        }
    }
    out
}

fn component_leaf_sets_xf(forest: &XForest, label_space: usize) -> Vec<FixedBitSet> {
    let mut visited = vec![false; forest.tree.num_nodes()];
    let mut components = Vec::new();
    for node in forest.tree.pre_order() {
        if forest.live_leafsets[node as usize].count_ones(..) == 0 {
            continue;
        }
        let is_comp_root = if forest.is_cut(node) {
            true
        } else {
            let parent = forest.tree.parent[node as usize];
            parent == NONE
        };
        if !is_comp_root {
            continue;
        }
        if visited[node as usize] {
            continue;
        }
        let mut set = FixedBitSet::with_capacity(label_space + 1);
        set.grow(label_space + 1);
        let mut stack = vec![node];
        visited[node as usize] = true;
        while let Some(cur) = stack.pop() {
            if forest.tree.is_leaf(cur)
                && forest.live_leafsets[cur as usize].count_ones(..) > 0
            {
                set.union_with(&forest.live_leafsets[cur as usize]);
            }
            if let Some((left, right)) = forest.tree.children(cur) {
                if left != NONE && !forest.is_cut(left) && !visited[left as usize] {
                    visited[left as usize] = true;
                    stack.push(left);
                }
                if right != NONE && !forest.is_cut(right) && !visited[right as usize] {
                    visited[right as usize] = true;
                    stack.push(right);
                }
            }
        }
        if set.count_ones(..) > 0 {
            components.push(set);
        }
    }
    components
}

// ============================================================================
// Pure utility functions
// ============================================================================

fn leafset_key(set: &FixedBitSet) -> Vec<usize> {
    set.as_slice().to_vec()
}

fn canonical_form(tree: &Tree) -> String {
    fn build(tree: &Tree, node: NodeId) -> String {
        if tree.is_leaf(node) {
            return tree.label[node as usize].to_string();
        }
        let (l, r) = tree.children(node).unwrap();
        let mut a = build(tree, l);
        let mut b = build(tree, r);
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        format!("({},{})", a, b)
    }
    if tree.root == NONE {
        String::new()
    } else {
        build(tree, tree.root)
    }
}

fn trivial_forest(reference: &Tree, num_leaves: u32) -> Vec<Tree> {
    let mut components = Vec::new();
    for lbl in 1..=num_leaves {
        components.push(make_singleton_tree(lbl, num_leaves));
    }
    if components.is_empty() && reference.root != NONE {
        components.push(reference.clone());
    }
    components
}

fn make_singleton_tree(lbl: u32, num_leaves: u32) -> Tree {
    let mut singleton = Tree::with_capacity(num_leaves);
    singleton.parent.push(NONE);
    singleton.left.push(NONE);
    singleton.right.push(NONE);
    singleton.label.push(lbl);
    singleton.label_to_node[lbl as usize] = 0;
    singleton.root = 0;
    singleton.compute_metadata();
    singleton
}

fn has_intersection(a: &FixedBitSet, b: &FixedBitSet) -> bool {
    let a_sl = a.as_slice();
    let b_sl = b.as_slice();
    let len = a_sl.len().min(b_sl.len());
    for i in 0..len {
        if a_sl[i] & b_sl[i] != 0 {
            return true;
        }
    }
    false
}

fn is_subset(a: &FixedBitSet, b: &FixedBitSet) -> bool {
    let a_sl = a.as_slice();
    let b_sl = b.as_slice();
    for i in 0..a_sl.len() {
        let b_word = if i < b_sl.len() { b_sl[i] } else { 0 };
        if a_sl[i] & !b_word != 0 {
            return false;
        }
    }
    true
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use klados_core::Instance;

    fn make_simple_tree() -> Tree {
        // Tree: ((1,2),3)
        let mut tree = Tree::with_capacity(3);
        // Node 0: leaf 1
        tree.parent.push(3);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(1);
        tree.label_to_node[1] = 0;
        // Node 1: leaf 2
        tree.parent.push(3);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(2);
        tree.label_to_node[2] = 1;
        // Node 2: leaf 3
        tree.parent.push(4);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(3);
        tree.label_to_node[3] = 2;
        // Node 3: internal (1,2)
        tree.parent.push(4);
        tree.left.push(0);
        tree.right.push(1);
        tree.label.push(0);
        // Node 4: root ((1,2),3)
        tree.parent.push(NONE);
        tree.left.push(3);
        tree.right.push(2);
        tree.label.push(0);

        tree.root = 4;
        tree.compute_metadata();
        tree
    }

    #[test]
    fn test_identical_trees_single_component() {
        let t1 = make_simple_tree();
        let t2 = make_simple_tree();
        let instance = Instance::new(vec![t1, t2], 3);
        let mut solver = ShiMestelSolver::new();
        let components = solver.solve(&instance).expect("solution");
        assert_eq!(components.len(), 1);
    }

    #[test]
    fn test_xforest_from_tree() {
        let tree = make_simple_tree();
        let forest = XForest::from_tree(tree);
        assert_eq!(forest.cut_edges.count_ones(..), 0);
        assert_eq!(forest.component_roots.len(), 1);
    }

    #[test]
    fn test_xforest_cut() {
        let tree = make_simple_tree();
        let mut forest = XForest::from_tree(tree);
        forest.cut(0);
        assert_eq!(forest.cut_edges.count_ones(..), 1);
        assert!(forest.is_cut(0));
    }

    #[test]
    fn test_lsi_identical() {
        let t1 = make_simple_tree();
        let t2 = make_simple_tree();
        let f1 = XForest::from_tree(t1);
        let f2 = XForest::from_tree(t2);
        assert!(all_pairs_lsi(&[f1, f2], 3));
    }

    #[test]
    fn test_component_label_sets() {
        let tree = make_simple_tree();
        let forest = XForest::from_tree(tree);
        let sets = component_leaf_sets_xf(&forest, 3);
        assert_eq!(sets.len(), 1);
    }

    #[test]
    fn test_has_intersection() {
        let mut a = FixedBitSet::with_capacity(4);
        let mut b = FixedBitSet::with_capacity(4);
        a.insert(1);
        a.insert(2);
        b.insert(2);
        b.insert(3);
        assert!(has_intersection(&a, &b));
        b.clear();
        b.insert(3);
        assert!(!has_intersection(&a, &b));
    }

    #[test]
    fn test_is_subset() {
        let mut a = FixedBitSet::with_capacity(4);
        let mut b = FixedBitSet::with_capacity(4);
        a.insert(1);
        a.insert(2);
        b.insert(1);
        b.insert(2);
        b.insert(3);
        assert!(is_subset(&a, &b));
        assert!(!is_subset(&b, &a));
    }

    #[test]
    fn test_sibling_pair_detection() {
        let tree = make_simple_tree();
        let forest = XForest::from_tree(tree);
        let pairs = find_all_sibling_pairs(&forest, 3);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (1, 2));
    }

    #[test]
    fn test_e_f_siblings() {
        // When a,b are siblings, E_F(a,b) should be empty
        let tree = make_simple_tree();
        let forest = XForest::from_tree(tree);
        let e = compute_e_f(&forest, 1, 2);
        assert_eq!(e.len(), 0);
    }
}
