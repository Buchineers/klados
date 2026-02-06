use fixedbitset::FixedBitSet;
use fxhash::{FxHashMap, FxHashSet};
use klados_core::{Instance, NodeId, SolverConfig, SolverStats, Tree, NONE};

fn trace_enabled() -> bool {
    std::env::var("WHIDDEN_TRACE").ok().as_deref() == Some("1")
}

macro_rules! trace {
    ($($arg:tt)*) => {
        if trace_enabled() {
            eprintln!($($arg)*);
        }
    };
}

#[derive(Clone, Debug)]
struct FptForest {
    tree: Tree,
    cut_edges: FixedBitSet,
    base_leafsets: Vec<FixedBitSet>,
    live_leafsets: Vec<FixedBitSet>,
    component_roots: Vec<NodeId>,
    done_leafsets: Vec<FixedBitSet>,
}

impl FptForest {
    fn from_tree(tree: Tree) -> Self {
        let num_nodes = tree.num_nodes();
        let mut leafsets = vec![FixedBitSet::with_capacity(tree.num_leaves as usize + 1); num_nodes];
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
            base_leafsets: leafsets.clone(),
            live_leafsets: leafsets,
            component_roots: vec![root],
            done_leafsets: Vec::new(),
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

            // Update leafsets: remove subtree leafset from ancestors in this component.
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

/// Baseline Whidden-style FPT solver for two trees.
pub struct FptSolver {
    config: SolverConfig,
    stats: SolverStats,
    memo: FxHashMap<Vec<usize>, usize>,
}

impl FptSolver {
    pub fn new() -> Self {
        Self {
            config: SolverConfig::default(),
            stats: SolverStats::default(),
            memo: FxHashMap::default(),
        }
    }

    pub fn with_config(config: SolverConfig) -> Self {
        Self {
            config,
            stats: SolverStats::default(),
            memo: FxHashMap::default(),
        }
    }

    pub fn stats(&self) -> &SolverStats {
        &self.stats
    }

    pub fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        if instance.num_trees() != 2 {
            return Some(trivial_forest(&instance.trees[0], instance.num_leaves));
        }

        let t1 = &instance.trees[0];
        let t2 = &instance.trees[1];

        let f1 = FptForest::from_tree(t1.clone());
        let f2 = FptForest::from_tree(t2.clone());
        let max_depth = self
            .config
            .max_depth
            .unwrap_or(instance.num_leaves as usize);

        for k in 0..=max_depth {
            self.memo.clear();
            if let Some(solution) = self.search(f1.clone(), f2.clone(), k) {
                return Some(self.reconstruct_components(t1, &solution));
            }
        }

        Some(trivial_forest(&instance.trees[0], instance.num_leaves))
    }

    fn search(
        &mut self,
        mut f1: FptForest,
        mut f2: FptForest,
        budget: usize,
    ) -> Option<FptForest> {
        self.stats.nodes_explored += 1;
        trace!("search: budget={} cuts={}", budget, f2.cut_edges.count_ones(..));

        reduce_common_cherries(&mut f1, &mut f2);

        if active_label_count(&f1) <= 2 {
            trace!("agreement forest found via |R_t|<=2");
            return Some(f2);
        }

        if self.is_agreement_forest(&f1, &f2) {
            trace!("agreement forest found (cuts={})", f2.cut_edges.count_ones(..));
            return Some(f2);
        }

        if budget == 0 {
            return None;
        }

        let key = f2.cut_edges.as_slice().to_vec();

        let sig_map_f2 = signature_map(&f2);
        let mut sig_cache_f1 = FxHashMap::default();
        for node in f1.tree.pre_order() {
            if !in_main_component(&f1, node) {
                continue;
            }
            let children = effective_children(&f1, node);
            if children.len() != 2 {
                continue;
            }
            let l = children[0];
            let r = children[1];
            if !is_active_leaf(&f1, l) || !is_active_leaf(&f1, r) {
                continue;
            }
            let sig_a = node_signature(&f1, l, &mut sig_cache_f1);
            let sig_c = node_signature(&f1, r, &mut sig_cache_f1);
            let Some(&na) = sig_map_f2.get(&sig_a) else { continue };
            let Some(&nc) = sig_map_f2.get(&sig_c) else { continue };

            if are_siblings_in_f2(&f2, na, nc) {
                continue;
            }

            trace!(
                "branch on sibling pair: f1_nodes=({},{}), f2_nodes=({},{}), budget={}",
                l,
                r,
                na,
                nc,
                budget
            );

            let same_component = effective_component_root(&f2, na)
                == effective_component_root(&f2, nc);

            if !same_component {
                trace!("case 6.1 (different components)");
                let mut branches = Vec::new();
                if let Some(f) = try_cut(&f2, na) {
                    branches.push(f);
                }
                if let Some(f) = try_cut(&f2, nc) {
                    branches.push(f);
                }
                for branch in branches {
                    if let Some(sol) = self.search(f1.clone(), branch, budget - 1) {
                        return Some(sol);
                    }
                }
                continue;
            }

            let pendant_nodes = pendant_nodes_on_path(&f2, na, nc);
            if pendant_nodes.len() == 1 {
                trace!("case 6.2 (one pendant): node={}", pendant_nodes[0]);
                if let Some(next) = try_cut(&f2, pendant_nodes[0]) {
                    if let Some(sol) = self.search(f1.clone(), next, budget - 1) {
                        return Some(sol);
                    }
                }
                continue;
            }

            if pendant_nodes.len() >= 2 {
                trace!("case 6.3 ({} pendants)", pendant_nodes.len());
                let mut branches = Vec::new();
                if let Some(f) = try_cut(&f2, na) {
                    branches.push((1usize, f));
                }
                if let Some(f) = try_cut(&f2, nc) {
                    branches.push((1usize, f));
                }
                if pendant_nodes.len() <= budget {
                    if let Some(f) = try_cut_many(&f2, &pendant_nodes) {
                        branches.push((pendant_nodes.len(), f));
                    }
                }
                for (cost, branch) in branches {
                    if cost <= budget {
                        if let Some(sol) = self.search(f1.clone(), branch, budget - cost) {
                            return Some(sol);
                        }
                    }
                }
            }
        }

        self.stats.branches_pruned += 1;
        let _ = key;
        None
    }

    fn is_agreement_forest(&self, f1: &FptForest, f2: &FptForest) -> bool {
        let components = component_leaf_sets(f2, f1.tree.num_leaves as usize);
        let t1_clusters = build_cluster_keys_from_forest(f1);
        let t2_clusters = build_cluster_keys_from_forest(f2);

        components.iter().all(|set| {
            let key = leafset_key(set);
            if !t1_clusters.contains(&key) || !t2_clusters.contains(&key) {
                return false;
            }
            let t1_sub = f1.tree.prune_to_leafset(set);
            let t2_sub = f2.tree.prune_to_leafset(set);
            canonical_form(&t1_sub) == canonical_form(&t2_sub)
        })
    }

    fn reconstruct_components(&self, t1: &Tree, forest: &FptForest) -> Vec<Tree> {
        let components = component_leaf_sets(forest, t1.num_leaves as usize);
        components
            .iter()
            .map(|set| t1.prune_to_leafset(set))
            .collect()
    }
}

impl super::ExactSolver for FptSolver {
    fn name(&self) -> &'static str {
        "whidden"
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        FptSolver::solve(self, instance)
    }

    fn stats(&self) -> &SolverStats {
        self.stats()
    }
}

fn are_siblings_in_f2(forest: &FptForest, a: NodeId, c: NodeId) -> bool {
    let pa = effective_parent(forest, a);
    let pc = effective_parent(forest, c);
    if pa == NONE || pa != pc {
        return false;
    }
    let children = effective_children(forest, pa);
    children.len() == 2 && children.contains(&a) && children.contains(&c)
}

fn pendant_nodes_on_path(forest: &FptForest, a: NodeId, c: NodeId) -> Vec<NodeId> {
    let lca = effective_lca(forest, a, c);
    if lca == NONE {
        return Vec::new();
    }

    let mut on_path: FxHashMap<NodeId, Vec<NodeId>> = FxHashMap::default();

    let mut cur = a;
    while cur != lca {
        let p = effective_parent(forest, cur);
        if p == NONE {
            break;
        }
        on_path.entry(p).or_default().push(cur);
        cur = p;
    }

    cur = c;
    while cur != lca {
        let p = effective_parent(forest, cur);
        if p == NONE {
            break;
        }
        on_path.entry(p).or_default().push(cur);
        cur = p;
    }

    let mut pendants = Vec::new();
    for (node, path_children) in on_path {
        let children = effective_children(forest, node);
        for child in children {
            if !path_children.contains(&child) {
                pendants.push(child);
            }
        }
    }

    pendants.sort_unstable();
    pendants.dedup();
    pendants
}

fn try_cut(forest: &FptForest, node: NodeId) -> Option<FptForest> {
    if node == forest.tree.root || forest.is_cut(node) {
        return None;
    }
    let mut next = forest.clone();
    next.cut(node);
    Some(next)
}

fn try_cut_many(forest: &FptForest, nodes: &[NodeId]) -> Option<FptForest> {
    let mut next = forest.clone();
    for &node in nodes {
        if node == forest.tree.root || next.is_cut(node) {
            return None;
        }
        next.cut(node);
    }
    Some(next)
}

fn component_leaf_sets(forest: &FptForest, label_space: usize) -> Vec<FixedBitSet> {
    let mut visited = vec![false; forest.tree.num_nodes()];
    let mut components = Vec::new();

    for node in forest.tree.pre_order() {
        if forest.live_leafsets[node as usize].count_ones(..) == 0 {
            continue;
        }
        if effective_parent(forest, node) != NONE {
            continue;
        }
        if visited[node as usize] {
            continue;
        }

        let mut set = FixedBitSet::with_capacity(label_space + 1);
        let mut stack = vec![node];
        visited[node as usize] = true;
        while let Some(cur) = stack.pop() {
            if is_active_leaf(forest, cur) {
                set.union_with(&forest.live_leafsets[cur as usize]);
            }
            for child in active_children(forest, cur) {
                if !visited[child as usize] {
                    visited[child as usize] = true;
                    stack.push(child);
                }
            }
        }

        if set.count_ones(..) > 0 {
            components.push(set);
        }
    }

    for set in &forest.done_leafsets {
        if set.count_ones(..) > 0 {
            components.push(set.clone());
        }
    }

    components
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

fn leafset_key(set: &FixedBitSet) -> Vec<usize> {
    set.as_slice().to_vec()
}

fn build_cluster_keys_from_forest(forest: &FptForest) -> FxHashSet<Vec<usize>> {
    let mut set = FxHashSet::default();
    for node in 0..forest.tree.num_nodes() {
        let node = node as NodeId;
        if forest.live_leafsets[node as usize].count_ones(..) == 0 {
            continue;
        }
        set.insert(leafset_key(&forest.live_leafsets[node as usize]));
    }
    set
}

fn signature_map(forest: &FptForest) -> FxHashMap<String, NodeId> {
    let mut map = FxHashMap::default();
    let mut cache = FxHashMap::default();
    for node in 0..forest.tree.num_nodes() {
        let node = node as NodeId;
        if forest.live_leafsets[node as usize].count_ones(..) == 0 {
            continue;
        }
        let sig = node_signature(forest, node, &mut cache);
        if !sig.is_empty() {
            map.insert(sig, node);
        }
    }
    map
}

fn node_signature(
    forest: &FptForest,
    node: NodeId,
    cache: &mut FxHashMap<NodeId, String>,
) -> String {
    if let Some(sig) = cache.get(&node) {
        return sig.clone();
    }
    if forest.live_leafsets[node as usize].count_ones(..) == 0 {
        return String::new();
    }

    let children = active_children(forest, node);
    let sig = match children.len() {
        0 => {
            let key = leafset_key(&forest.live_leafsets[node as usize]);
            format!("L{:?}", key)
        }
        1 => node_signature(forest, children[0], cache),
        _ => {
            let mut a = node_signature(forest, children[0], cache);
            let mut b = node_signature(forest, children[1], cache);
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            format!("({},{})", a, b)
        }
    };

    cache.insert(node, sig.clone());
    sig
}

fn active_children(forest: &FptForest, node: NodeId) -> Vec<NodeId> {
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

fn contract_cherry_sets(
    f1: &mut FptForest,
    f2: &mut FptForest,
    p1: NodeId,
    p2: NodeId,
    a_set: &FixedBitSet,
    c_set: &FixedBitSet,
) -> bool {
    if p1 == NONE || p2 == NONE {
        return false;
    }

    let mut union = a_set.clone();
    union.union_with(c_set);

    contract_parent(f1, p1, &union);
    contract_parent(f2, p2, &union);
    true
}

fn contract_parent(forest: &mut FptForest, parent: NodeId, set: &FixedBitSet) {
    let mut stack = Vec::new();
    if let Some((left, right)) = forest.tree.children(parent) {
        if left != NONE {
            stack.push(left);
        }
        if right != NONE {
            stack.push(right);
        }
    }

    forest.tree.left[parent as usize] = NONE;
    forest.tree.right[parent as usize] = NONE;
    forest.tree.label[parent as usize] = 0;
    forest.base_leafsets[parent as usize] = set.clone();
    forest.live_leafsets[parent as usize] = set.clone();

    while let Some(node) = stack.pop() {
        forest.base_leafsets[node as usize].clear();
        forest.live_leafsets[node as usize].clear();
        if let Some((left, right)) = forest.tree.children(node) {
            if left != NONE {
                stack.push(left);
            }
            if right != NONE {
                stack.push(right);
            }
        }
    }
}

fn trivial_forest(reference: &Tree, num_leaves: u32) -> Vec<Tree> {
    let mut components = Vec::new();
    for lbl in 1..=num_leaves {
        let mut singleton = Tree::with_capacity(num_leaves);
        singleton.parent.push(NONE);
        singleton.left.push(NONE);
        singleton.right.push(NONE);
        singleton.label.push(lbl);
        singleton.label_to_node[lbl as usize] = 0;
        singleton.root = 0;
        singleton.compute_metadata();
        components.push(singleton);
    }
    if components.is_empty() && reference.root != NONE {
        components.push(reference.clone());
    }
    components
}

fn reduce_common_cherries(f1: &mut FptForest, f2: &mut FptForest) {
    loop {
        let mut reduced = false;
        if prune_agreeing_roots(f1, f2) {
            reduced = true;
        }
        let nodes: Vec<NodeId> = f1.tree.pre_order().collect();
        let sig_map_f2 = signature_map(f2);
        let mut sig_cache_f1 = FxHashMap::default();
        for node in nodes {
            if !in_main_component(f1, node) {
                continue;
            }
            let children = effective_children(f1, node);
            if children.len() != 2 {
                continue;
            }
            let l = children[0];
            let r = children[1];
            if !is_active_leaf(f1, l) || !is_active_leaf(f1, r) {
                continue;
            }
            let sig_a = node_signature(f1, l, &mut sig_cache_f1);
            let sig_c = node_signature(f1, r, &mut sig_cache_f1);
            let Some(&na) = sig_map_f2.get(&sig_a) else { continue };
            let Some(&nc) = sig_map_f2.get(&sig_c) else { continue };
            if are_siblings_in_f2(f2, na, nc) {
                let a_set = f1.live_leafsets[l as usize].clone();
                let c_set = f1.live_leafsets[r as usize].clone();
                let p2 = effective_parent(f2, na);
                trace!("reduce common cherry at f1 nodes ({},{})", l, r);
                if contract_cherry_sets(f1, f2, node, p2, &a_set, &c_set) {
                    reduced = true;
                    break;
                }
            }
        }
        if !reduced {
            break;
        }
    }
}

fn is_active_leaf(forest: &FptForest, node: NodeId) -> bool {
    forest.live_leafsets[node as usize].count_ones(..) > 0 && active_children(forest, node).is_empty()
}

fn in_main_component(forest: &FptForest, node: NodeId) -> bool {
    forest.component_root(node) == forest.tree.root
}

fn prune_agreeing_roots(f1: &mut FptForest, f2: &mut FptForest) -> bool {
    let sig_map_f2 = signature_map(f2);
    let mut sig_cache_f1 = FxHashMap::default();
    let mut changed = false;
    let nodes: Vec<NodeId> = f1.tree.pre_order().collect();
    for node in nodes {
        if !in_main_component(f1, node) {
            continue;
        }
        if !is_active_leaf(f1, node) {
            continue;
        }
        let sig = node_signature(f1, node, &mut sig_cache_f1);
        let Some(&other) = sig_map_f2.get(&sig) else { continue };
        if f2.is_cut(other) || other == f2.tree.root {
            let set = f2.live_leafsets[other as usize].clone();
            if set.count_ones(..) == 0 {
                continue;
            }
            trace!("prune agreeing root: f1_node={}, f2_node={}, leafset={:?}", node, other, leafset_key(&set));
            f1.done_leafsets.push(set.clone());
            f2.done_leafsets.push(set);
            deactivate_subtree(f1, node);
            deactivate_subtree(f2, other);
            changed = true;
        }
    }
    changed
}

fn deactivate_subtree(forest: &mut FptForest, node: NodeId) {
    if forest.live_leafsets[node as usize].count_ones(..) == 0 {
        return;
    }
    let removed = forest.live_leafsets[node as usize].clone();
    let mut cur = forest.tree.parent[node as usize];
    while cur != NONE {
        forest.live_leafsets[cur as usize].difference_with(&removed);
        cur = forest.tree.parent[cur as usize];
    }

    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        forest.live_leafsets[n as usize].clear();
        if let Some((left, right)) = forest.tree.children(n) {
            if left != NONE {
                stack.push(left);
            }
            if right != NONE {
                stack.push(right);
            }
        }
    }
}

fn active_label_count(forest: &FptForest) -> usize {
    let mut count = 0;
    for node in forest.tree.pre_order() {
        if !in_main_component(forest, node) {
            continue;
        }
        if is_active_leaf(forest, node) {
            count += 1;
        }
    }
    count
}

fn effective_parent(forest: &FptForest, node: NodeId) -> NodeId {
    let tree = &forest.tree;
    if node == tree.root || forest.is_cut(node) {
        return NONE;
    }

    let mut cur = node;
    let mut parent = tree.parent[cur as usize];
    if parent == NONE {
        return NONE;
    }

    loop {
        let (left, right) = match tree.children(parent) {
            Some(ch) => ch,
            None => return parent,
        };

        let left_active = left != NONE
            && !forest.is_cut(left)
            && forest.live_leafsets[left as usize].count_ones(..) > 0;
        let right_active = right != NONE
            && !forest.is_cut(right)
            && forest.live_leafsets[right as usize].count_ones(..) > 0;
        let active = left_active as u8 + right_active as u8;

        if active <= 1 {
            cur = parent;
            if forest.is_cut(cur) {
                return NONE;
            }
            parent = tree.parent[cur as usize];
            if parent == NONE {
                return NONE;
            }
            continue;
        }

        return parent;
    }
}

fn effective_children(forest: &FptForest, node: NodeId) -> Vec<NodeId> {
    let tree = &forest.tree;
    let mut out = Vec::with_capacity(2);
    if let Some((left, right)) = tree.children(node) {
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

fn descend_to_effective(forest: &FptForest, mut node: NodeId) -> NodeId {
    let tree = &forest.tree;
    loop {
        if tree.is_leaf(node) {
            return node;
        }
        let (left, right) = tree.children(node).unwrap();
        let left_active = left != NONE
            && !forest.is_cut(left)
            && forest.live_leafsets[left as usize].count_ones(..) > 0;
        let right_active = right != NONE
            && !forest.is_cut(right)
            && forest.live_leafsets[right as usize].count_ones(..) > 0;
        let active = left_active as u8 + right_active as u8;
        if active <= 1 {
            node = if left_active { left } else { right };
            continue;
        }
        return node;
    }
}

fn effective_lca(forest: &FptForest, a: NodeId, c: NodeId) -> NodeId {
    let mut ancestors = FxHashSet::default();
    let mut cur = a;
    ancestors.insert(cur);
    loop {
        let p = effective_parent(forest, cur);
        if p == NONE {
            break;
        }
        cur = p;
        ancestors.insert(cur);
    }

    cur = c;
    if ancestors.contains(&cur) {
        return cur;
    }
    loop {
        let p = effective_parent(forest, cur);
        if p == NONE {
            break;
        }
        cur = p;
        if ancestors.contains(&cur) {
            return cur;
        }
    }
    NONE
}

fn effective_component_root(forest: &FptForest, node: NodeId) -> NodeId {
    if forest.live_leafsets[node as usize].count_ones(..) == 0 {
        return NONE;
    }
    let mut cur = node;
    loop {
        let p = effective_parent(forest, cur);
        if p == NONE {
            return cur;
        }
        cur = p;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_tree() -> Tree {
        let mut tree = Tree::with_capacity(3);

        // leaf 1 (node 0)
        tree.parent.push(3);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(1);
        tree.label_to_node[1] = 0;

        // leaf 2 (node 1)
        tree.parent.push(3);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(2);
        tree.label_to_node[2] = 1;

        // leaf 3 (node 2)
        tree.parent.push(4);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(3);
        tree.label_to_node[3] = 2;

        // internal (1,2) (node 3)
        tree.parent.push(4);
        tree.left.push(0);
        tree.right.push(1);
        tree.label.push(0);

        // root (node 4)
        tree.parent.push(NONE);
        tree.left.push(3);
        tree.right.push(2);
        tree.label.push(0);

        tree.root = 4;
        tree.compute_metadata();
        tree
    }

    #[test]
    fn test_identical_trees_maf_is_single_component() {
        let t1 = make_simple_tree();
        let t2 = make_simple_tree();
        let instance = Instance::new(vec![t1, t2], 3);

        let mut solver = FptSolver::new();
        let components = solver.solve(&instance).expect("solution");
        assert_eq!(components.len(), 1);
    }
}
