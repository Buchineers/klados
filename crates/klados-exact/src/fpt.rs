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
    rt: Vec<FixedBitSet>,
    rt_keys: FxHashSet<Vec<usize>>,
}

impl FptForest {
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
        let mut rt = Vec::with_capacity(num_leaves as usize);
        let mut rt_keys = FxHashSet::default();
        for lbl in 1..=num_leaves {
            let mut set = FixedBitSet::with_capacity(num_leaves as usize + 1);
            set.grow(num_leaves as usize + 1);
            set.insert(lbl as usize);
            rt_keys.insert(leafset_key(&set));
            rt.push(set);
        }
        Self {
            tree,
            cut_edges: FixedBitSet::with_capacity(num_nodes),
            base_leafsets: leafsets.clone(),
            live_leafsets: leafsets,
            component_roots: vec![root],
            done_leafsets: Vec::new(),
            rt,
            rt_keys,
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

        if f1.rt.len() <= 2 {
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

        let map_f1 = dot_leafset_map(&f1);
        let map_f2 = dot_leafset_map(&f2);
        let sibling_pairs = collect_sibling_pairs(&f1, &map_f1);
        for (ka, kc) in sibling_pairs {
            let Some(&na) = map_f2.get(&ka) else { continue };
            let Some(&nc) = map_f2.get(&kc) else { continue };

            let na_dot = dot_rep(&f2, na);
            let nc_dot = dot_rep(&f2, nc);

            if are_siblings_in_f2(&f2, na_dot, nc_dot) {
                if grow_specific_cherry(&mut f1, &mut f2, &ka, &kc, &map_f1) {
                    return self.search(f1, f2, budget);
                }
            }

            trace!(
                "branch on sibling pair: f1_labels=({},{}), f2_nodes=({},{}), budget={}",
                ka.len(),
                kc.len(),
                na_dot,
                nc_dot,
                budget
            );

            let same_component = f2.component_root(na_dot) == f2.component_root(nc_dot);
            if !same_component
                || is_ancestor_in_forest(&f2, na_dot, nc_dot)
                || is_ancestor_in_forest(&f2, nc_dot, na_dot)
            {
                trace!("case 6.1 (ancestor or separate components)");
                let mut branches = Vec::new();
                if let Some(f) = try_cut(&f2, na_dot) {
                    branches.push(f);
                }
                if let Some(f) = try_cut(&f2, nc_dot) {
                    branches.push(f);
                }
                for branch in branches {
                    if let Some(sol) = self.search(f1.clone(), branch, budget - 1) {
                        return Some(sol);
                    }
                }
                return None;
            }

            let mut a_node = na_dot;
            let mut c_node = nc_dot;
            let da = effective_depth(&f2, a_node);
            let dc = effective_depth(&f2, c_node);
            if da < dc {
                std::mem::swap(&mut a_node, &mut c_node);
            }

            let pendant_nodes = pendant_nodes_on_path(&f2, a_node, c_node);
            if pendant_nodes.len() == 1 {
                trace!("case 6.2 (one pendant): node={}", pendant_nodes[0]);
                if let Some(next) = try_cut(&f2, pendant_nodes[0]) {
                    if let Some(sol) = self.search(f1.clone(), next, budget - 1) {
                        return Some(sol);
                    }
                }
                return None;
            }

            if pendant_nodes.len() >= 2 {
                trace!("case 6.3 ({} pendants)", pendant_nodes.len());
                let mut branches = Vec::new();
                if let Some(f) = try_cut(&f2, a_node) {
                    branches.push((1usize, f));
                }
                if let Some(f) = try_cut(&f2, c_node) {
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
                return None;
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
    let pa = dot_parent(forest, a);
    let pc = dot_parent(forest, c);
    if pa == NONE || pa != pc {
        return false;
    }
    let children = dot_children(forest, pa);
    if children.len() != 2 || !children.contains(&a) || !children.contains(&c) {
        return false;
    }
    let mut union = forest.live_leafsets[a as usize].clone();
    union.union_with(&forest.live_leafsets[c as usize]);
    union == forest.live_leafsets[pa as usize]
}

// (removed label-based sibling finder)

fn collect_sibling_pairs(
    f1: &FptForest,
    map_f1: &FxHashMap<Vec<usize>, NodeId>,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    if f1.rt.len() < 2 {
        return Vec::new();
    }
    let mut pairs = Vec::new();
    for node in f1.tree.pre_order() {
        if !is_dot_node(f1, node) {
            continue;
        }
        let children = dot_children(f1, node);
        if children.len() != 2 {
            continue;
        }
        let k1 = leafset_key(&f1.live_leafsets[children[0] as usize]);
        let k2 = leafset_key(&f1.live_leafsets[children[1] as usize]);
        if !f1.rt_keys.contains(&k1) || !f1.rt_keys.contains(&k2) {
            continue;
        }
        if map_f1.get(&k1).is_none() || map_f1.get(&k2).is_none() {
            continue;
        }
        pairs.push((k1, k2));
    }
    pairs
}

fn pendant_nodes_on_path(forest: &FptForest, a: NodeId, c: NodeId) -> Vec<NodeId> {
    let lca = forest_lca(forest, a, c);
    if lca == NONE {
        return Vec::new();
    }

    let mut on_path: FxHashMap<NodeId, Vec<NodeId>> = FxHashMap::default();

    let mut cur = a;
    while cur != lca {
        let p = forest_parent(forest, cur);
        if p == NONE {
            break;
        }
        on_path.entry(p).or_default().push(cur);
        cur = p;
    }

    cur = c;
    while cur != lca {
        let p = forest_parent(forest, cur);
        if p == NONE {
            break;
        }
        on_path.entry(p).or_default().push(cur);
        cur = p;
    }

    let mut pendants = Vec::new();
    for (node, path_children) in on_path {
        let children = active_children(forest, node);
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

fn forest_parent(forest: &FptForest, node: NodeId) -> NodeId {
    let tree = &forest.tree;
    if node == tree.root || forest.is_cut(node) {
        return NONE;
    }
    let p = tree.parent[node as usize];
    if p == NONE {
        return NONE;
    }
    p
}

fn dot_parent(forest: &FptForest, node: NodeId) -> NodeId {
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
        let active = active_children(forest, parent).len();
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

fn dot_children(forest: &FptForest, node: NodeId) -> Vec<NodeId> {
    let tree = &forest.tree;
    let mut out = Vec::with_capacity(2);
    if let Some((left, right)) = tree.children(node) {
        if left != NONE && !forest.is_cut(left) && forest.live_leafsets[left as usize].count_ones(..) > 0 {
            out.push(descend_to_dot(forest, left));
        }
        if right != NONE && !forest.is_cut(right) && forest.live_leafsets[right as usize].count_ones(..) > 0 {
            out.push(descend_to_dot(forest, right));
        }
    }
    out
}

fn dot_rep(forest: &FptForest, node: NodeId) -> NodeId {
    if node == NONE {
        return NONE;
    }
    let children = active_children(forest, node);
    if children.len() <= 1 {
        return descend_to_dot(forest, node);
    }
    node
}

fn descend_to_dot(forest: &FptForest, mut node: NodeId) -> NodeId {
    let tree = &forest.tree;
    loop {
        if tree.is_leaf(node) {
            return node;
        }
        let children = active_children(forest, node);
        if children.len() <= 1 {
            if let Some(next) = children.first().copied() {
                node = next;
                continue;
            }
        }
        return node;
    }
}

fn forest_lca(forest: &FptForest, a: NodeId, c: NodeId) -> NodeId {
    let mut ancestors = FxHashSet::default();
    let mut cur = a;
    ancestors.insert(cur);
    loop {
        let p = forest_parent(forest, cur);
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
        let p = forest_parent(forest, cur);
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
        if dot_parent(forest, node) != NONE {
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

fn rt_remove(forest: &mut FptForest, key: &Vec<usize>) {
    if !forest.rt_keys.remove(key) {
        return;
    }
    if let Some(pos) = forest
        .rt
        .iter()
        .position(|set| leafset_key(set) == *key)
    {
        forest.rt.swap_remove(pos);
    }
}

fn rt_add(forest: &mut FptForest, set: FixedBitSet) {
    let key = leafset_key(&set);
    if forest.rt_keys.insert(key) {
        forest.rt.push(set);
    }
}

fn dot_leafset_map(forest: &FptForest) -> FxHashMap<Vec<usize>, NodeId> {
    let mut map = FxHashMap::default();
    for node in forest.tree.pre_order() {
        if !is_dot_node(forest, node) {
            continue;
        }
        let key = leafset_key(&forest.live_leafsets[node as usize]);
        map.insert(key, node);
    }
    map
}

fn is_dot_node(forest: &FptForest, node: NodeId) -> bool {
    if forest.live_leafsets[node as usize].count_ones(..) == 0 {
        return false;
    }
    if forest.tree.is_leaf(node) {
        return true;
    }
    active_children(forest, node).len() == 2
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
        if prune_agreeing_roots(f1, f2) {
            continue;
        }
        if grow_agreeing_cherries(f1, f2) {
            continue;
        }
        break;
    }
}

fn is_active_leaf(forest: &FptForest, node: NodeId) -> bool {
    forest.live_leafsets[node as usize].count_ones(..) > 0 && active_children(forest, node).is_empty()
}

fn in_main_component(forest: &FptForest, node: NodeId) -> bool {
    forest.component_root(node) == forest.tree.root
}

fn prune_agreeing_roots(f1: &mut FptForest, f2: &mut FptForest) -> bool {
    let map_f1 = dot_leafset_map(f1);
    let map_f2 = dot_leafset_map(f2);
    for set in f1.rt.iter() {
        let key = leafset_key(set);
        let Some(&node_f2) = map_f2.get(&key) else { continue };
        let Some(&node_f1) = map_f1.get(&key) else { continue };
        if dot_parent(f2, node_f2) == NONE {
            trace!(
                "prune agreeing root: leafset={:?}, f1_node={}, f2_node={}",
                key,
                node_f1,
                node_f2
            );
            f1.done_leafsets.push(set.clone());
            f2.done_leafsets.push(set.clone());
            rt_remove(f1, &key);
            rt_remove(f2, &key);
            deactivate_subtree(f1, node_f1);
            deactivate_subtree(f2, node_f2);
            return true;
        }
    }
    false
}

fn grow_agreeing_cherries(f1: &mut FptForest, f2: &mut FptForest) -> bool {
    let map_f1 = dot_leafset_map(f1);
    let map_f2 = dot_leafset_map(f2);
    let nodes: Vec<NodeId> = f1.tree.pre_order().collect();
    for node in nodes {
        if !is_dot_node(f1, node) {
            continue;
        }
        let children = dot_children(f1, node);
        if children.len() != 2 {
            continue;
        }
        let key_a = leafset_key(&f1.live_leafsets[children[0] as usize]);
        let key_c = leafset_key(&f1.live_leafsets[children[1] as usize]);
        if !f1.rt_keys.contains(&key_a) || !f1.rt_keys.contains(&key_c) {
            continue;
        }
        let Some(&na) = map_f2.get(&key_a) else { continue };
        let Some(&nc) = map_f2.get(&key_c) else { continue };
        if !are_siblings_in_f2(f2, na, nc) {
            continue;
        }
        if grow_specific_cherry(f1, f2, &key_a, &key_c, &map_f1) {
            return true;
        }
    }
    false
}

fn grow_specific_cherry(
    f1: &mut FptForest,
    f2: &mut FptForest,
    ka: &Vec<usize>,
    kc: &Vec<usize>,
    map_f1: &FxHashMap<Vec<usize>, NodeId>,
) -> bool {
    let Some(&node_a) = map_f1.get(ka) else { return false };
    let Some(&node_c) = map_f1.get(kc) else { return false };
    let p1 = dot_parent(f1, node_a);
    if p1 == NONE || dot_parent(f1, node_c) != p1 {
        return false;
    }
    let mut union = f1.live_leafsets[node_a as usize].clone();
    union.union_with(&f1.live_leafsets[node_c as usize]);
    rt_remove(f1, ka);
    rt_remove(f1, kc);
    rt_remove(f2, ka);
    rt_remove(f2, kc);
    rt_add(f1, union.clone());
    rt_add(f2, union);
    trace!("grow agreeing cherry");
    true
}

fn deactivate_subtree(forest: &mut FptForest, node: NodeId) {
    if forest.base_leafsets[node as usize].count_ones(..) == 0 {
        return;
    }
    let removed = forest.live_leafsets[node as usize].clone();
    let mut cur = forest.tree.parent[node as usize];
    while cur != NONE {
        forest.live_leafsets[cur as usize].difference_with(&removed);
        if forest.is_cut(cur) {
            break;
        }
        cur = forest.tree.parent[cur as usize];
    }

    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        forest.live_leafsets[n as usize].clear();
        if let Some((left, right)) = forest.tree.children(n) {
            if left != NONE && !forest.is_cut(left) {
                stack.push(left);
            }
            if right != NONE && !forest.is_cut(right) {
                stack.push(right);
            }
        }
    }
}

fn is_ancestor_in_forest(forest: &FptForest, ancestor: NodeId, node: NodeId) -> bool {
    if ancestor == NONE || node == NONE {
        return false;
    }
    let mut cur = node;
    loop {
        if cur == ancestor {
            return true;
        }
        let p = dot_parent(forest, cur);
        if p == NONE {
            break;
        }
        cur = p;
    }
    false
}

fn effective_depth(forest: &FptForest, node: NodeId) -> usize {
    let mut depth = 0usize;
    let mut cur = node;
    loop {
        let p = dot_parent(forest, cur);
        if p == NONE {
            break;
        }
        depth += 1;
        cur = p;
    }
    depth
}

// (removed label-based helpers)

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
