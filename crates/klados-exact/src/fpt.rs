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

/// Whidden-style FPT solver generalized for multiple trees.
pub struct FptSolver {
    config: SolverConfig,
    stats: SolverStats,
    memo: FxHashMap<Vec<usize>, usize>,
    num_leaves: usize,
    multi_tree: bool,
    pendants_first: bool,
}

impl FptSolver {
    pub fn new() -> Self {
        Self {
            config: SolverConfig::default(),
            stats: SolverStats::default(),
            memo: FxHashMap::default(),
            num_leaves: 0,
            multi_tree: false,
            pendants_first: false,
        }
    }

    pub fn stats(&self) -> &SolverStats {
        &self.stats
    }

    pub fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        if instance.trees.is_empty() {
            return None;
        }

        // Single tree: the entire tree is the MAF
        if instance.num_trees() == 1 {
            return Some(vec![instance.trees[0].clone()]);
        }

        self.num_leaves = instance.num_leaves as usize;
        self.multi_tree = instance.num_trees() > 2;

        let max_depth = self
            .config
            .max_depth
            .unwrap_or(instance.num_leaves as usize);

        let f1 = FptForest::from_tree(instance.trees[0].clone());
        let forests: Vec<FptForest> = instance.trees[1..]
            .iter()
            .map(|t| FptForest::from_tree(t.clone()))
            .collect();

        // For multi-tree, try both branch orderings (pendants_first and original).
        let orderings: &[bool] = if self.multi_tree {
            &[true, false]
        } else {
            &[false]
        };

        let mut best: Option<Vec<Tree>> = None;
        let mut best_maf_size = usize::MAX;

        for k in 0..=max_depth {
            if k + 1 >= best_maf_size {
                break;
            }
            for &pf in orderings {
                self.pendants_first = pf;
                self.memo.clear();
                trace!("search: k={}, pendants_first={}", k, pf);
                if let Some(sol) = self.search(f1.clone(), forests.clone(), k) {
                    let components = self.reconstruct_components_multi(instance, &sol);
                    let maf_size = components.len();
                    trace!("found: k={}, pf={}, maf_size={}", k, pf, maf_size);
                    if maf_size < best_maf_size {
                        best_maf_size = maf_size;
                        best = Some(components);
                    }
                }
            }
            if best_maf_size <= k + 1 {
                break;
            }
        }

        best.or_else(|| Some(trivial_forest(&instance.trees[0], instance.num_leaves)))
    }

    fn search(
        &mut self,
        mut f1: FptForest,
        mut forests: Vec<FptForest>,
        max_cuts: usize,
    ) -> Option<Vec<FptForest>> {
        self.stats.nodes_explored += 1;

        // 1. Reduce: grow cherries and prune roots across ALL forests
        reduce_common_cherries_multi(&mut f1, &mut forests);

        // 2. Termination
        if f1.rt.len() <= 2 || self.is_agreement_forest_multi(&f1, &forests) {
            return Some(forests);
        }

        // 3. Budget check: can any forest still accept a cut?
        let any_budget = forests
            .iter()
            .any(|fi| fi.cut_edges.count_ones(..) < max_cuts);
        if !any_budget {
            return None;
        }

        // 4. Collect sibling pairs from F1
        let map_f1 = dot_leafset_map(&f1);
        let sibling_pairs = collect_sibling_pairs(&f1, &map_f1);

        for (ka, kc) in sibling_pairs {
            // 5. Check each non-reference forest for agreement on this pair
            let mut disagreements: Vec<(usize, DisagreementCase)> = Vec::new();

            for (i, fi) in forests.iter().enumerate() {
                let map_fi = dot_leafset_map(fi);
                let Some(&na) = map_fi.get(&ka) else {
                    continue;
                };
                let Some(&nc) = map_fi.get(&kc) else {
                    continue;
                };

                let na_dot = dot_rep(fi, na);
                let nc_dot = dot_rep(fi, nc);

                if are_siblings_in_f2(fi, na_dot, nc_dot) {
                    continue;
                }

                let case = classify_disagreement(fi, na_dot, nc_dot);
                disagreements.push((i, case));
            }

            // 6a. All forests agree: grow cherry in all forests, recurse
            if disagreements.is_empty() {
                if grow_specific_cherry_multi(&mut f1, &mut forests, &ka, &kc, &map_f1) {
                    return self.search(f1, forests, max_cuts);
                }
                continue;
            }

            // 6b. Disagreement: branch on best case (first-found).
            disagreements.sort_by_key(|(_, case)| disagreement_priority(case));

            for (forest_idx, case) in &disagreements {
                let forest_idx = *forest_idx;
                let remaining = max_cuts - forests[forest_idx].cut_edges.count_ones(..);

                if remaining == 0 {
                    continue;
                }

                trace!(
                    "branch on forest {} (of {}), remaining={}",
                    forest_idx,
                    forests.len(),
                    remaining
                );

                match case {
                    DisagreementCase::Case62 { pendant } => {
                        trace!("case 6.2 (one pendant): node={}", *pendant);
                        if let Some(cut_forest) = try_cut(&forests[forest_idx], *pendant) {
                            let mut next = forests.clone();
                            next[forest_idx] = cut_forest;
                            if let Some(sol) = self.search(f1.clone(), next, max_cuts) {
                                return Some(sol);
                            }
                        }
                    }
                    DisagreementCase::Case61 { branches } => {
                        trace!("case 6.1 ({} branches)", branches.len());
                        for node in branches {
                            if let Some(cut_forest) = try_cut(&forests[forest_idx], *node) {
                                let mut next = forests.clone();
                                next[forest_idx] = cut_forest;
                                if let Some(sol) = self.search(f1.clone(), next, max_cuts) {
                                    return Some(sol);
                                }
                            }
                        }
                    }
                    DisagreementCase::Case63 {
                        a_node,
                        c_node,
                        pendants,
                    } => {
                        trace!("case 6.3 ({} pendants)", pendants.len());
                        let fi = &forests[forest_idx];
                        let mut branch_cuts: Vec<(usize, FptForest)> = Vec::new();
                        if self.pendants_first {
                            if pendants.len() <= remaining {
                                if let Some(f) = try_cut_many(fi, pendants) {
                                    branch_cuts.push((pendants.len(), f));
                                }
                            }
                            if let Some(f) = try_cut(fi, *a_node) {
                                branch_cuts.push((1, f));
                            }
                            if let Some(f) = try_cut(fi, *c_node) {
                                branch_cuts.push((1, f));
                            }
                        } else {
                            if let Some(f) = try_cut(fi, *a_node) {
                                branch_cuts.push((1, f));
                            }
                            if let Some(f) = try_cut(fi, *c_node) {
                                branch_cuts.push((1, f));
                            }
                            if pendants.len() <= remaining {
                                if let Some(f) = try_cut_many(fi, pendants) {
                                    branch_cuts.push((pendants.len(), f));
                                }
                            }
                        }
                        for (cost, cut_forest) in branch_cuts {
                            if cost <= remaining {
                                let mut next = forests.clone();
                                next[forest_idx] = cut_forest;
                                if let Some(sol) = self.search(f1.clone(), next, max_cuts) {
                                    return Some(sol);
                                }
                            }
                        }
                    }
                }
            }
            return None;
        }

        self.stats.branches_pruned += 1;
        None
    }

    fn is_agreement_forest_multi(&self, f1: &FptForest, forests: &[FptForest]) -> bool {
        let label_space = f1.tree.num_leaves as usize;

        // Compute per-forest component leaf sets and refine
        let mut refined = component_leaf_sets(&forests[0], label_space);
        for fi in &forests[1..] {
            let partition = component_leaf_sets(fi, label_space);
            refined = refine_partitions(&refined, &partition);
        }

        let t1_clusters = build_cluster_keys_from_forest(f1);
        let all_fi_clusters: Vec<FxHashSet<Vec<usize>>> =
            forests.iter().map(build_cluster_keys_from_forest).collect();

        refined.iter().all(|set| {
            if set.count_ones(..) <= 1 {
                return true;
            }
            let key = leafset_key(set);
            if !t1_clusters.contains(&key) {
                return false;
            }
            for fi_clusters in &all_fi_clusters {
                if !fi_clusters.contains(&key) {
                    return false;
                }
            }
            let t1_sub = f1.tree.prune_to_leafset(set);
            let t1_canon = canonical_form(&t1_sub);
            for fi in forests {
                let ti_sub = fi.tree.prune_to_leafset(set);
                if canonical_form(&ti_sub) != t1_canon {
                    return false;
                }
            }
            true
        })
    }

    fn reconstruct_components_multi(
        &self,
        instance: &Instance,
        forests: &[FptForest],
    ) -> Vec<Tree> {
        let label_space = instance.num_leaves as usize;
        let t1 = &instance.trees[0];

        let mut partitions: Vec<Vec<FixedBitSet>> = forests
            .iter()
            .map(|fi| component_leaf_sets(fi, label_space))
            .collect();

        let mut refined = partitions.remove(0);
        for partition in &partitions {
            refined = refine_partitions(&refined, partition);
        }

        refined
            .iter()
            .filter(|set| set.count_ones(..) > 0)
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
        if left != NONE
            && !forest.is_cut(left)
            && forest.live_leafsets[left as usize].count_ones(..) > 0
        {
            out.push(descend_to_dot(forest, left));
        }
        if right != NONE
            && !forest.is_cut(right)
            && forest.live_leafsets[right as usize].count_ones(..) > 0
        {
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
    if let Some(pos) = forest.rt.iter().position(|set| leafset_key(set) == *key) {
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

fn is_active_leaf(forest: &FptForest, node: NodeId) -> bool {
    forest.live_leafsets[node as usize].count_ones(..) > 0
        && active_children(forest, node).is_empty()
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

// --- Multi-tree generalization ---

#[derive(Clone, Debug)]
enum DisagreementCase {
    /// Case 6.2: exactly one pendant on path. Deterministic cut (branching factor 1).
    Case62 { pendant: NodeId },
    /// Case 6.1: ancestor relationship or separate components. 2-way branch.
    Case61 { branches: Vec<NodeId> },
    /// Case 6.3: >= 2 pendants on path. Branch: cut a, cut c, or cut all pendants.
    Case63 {
        a_node: NodeId,
        c_node: NodeId,
        pendants: Vec<NodeId>,
    },
}

fn classify_disagreement(fi: &FptForest, na_dot: NodeId, nc_dot: NodeId) -> DisagreementCase {
    let same_component = fi.component_root(na_dot) == fi.component_root(nc_dot);

    if !same_component
        || is_ancestor_in_forest(fi, na_dot, nc_dot)
        || is_ancestor_in_forest(fi, nc_dot, na_dot)
    {
        let mut branches = Vec::new();
        if na_dot != fi.tree.root && !fi.is_cut(na_dot) {
            branches.push(na_dot);
        }
        if nc_dot != fi.tree.root && !fi.is_cut(nc_dot) {
            branches.push(nc_dot);
        }
        return DisagreementCase::Case61 { branches };
    }

    let mut a_node = na_dot;
    let mut c_node = nc_dot;
    let da = effective_depth(fi, a_node);
    let dc = effective_depth(fi, c_node);
    if da < dc {
        std::mem::swap(&mut a_node, &mut c_node);
    }

    let pendants = pendant_nodes_on_path(fi, a_node, c_node);

    if pendants.len() == 1 {
        DisagreementCase::Case62 {
            pendant: pendants[0],
        }
    } else {
        DisagreementCase::Case63 {
            a_node,
            c_node,
            pendants,
        }
    }
}

fn disagreement_priority(case: &DisagreementCase) -> (u8, usize) {
    match case {
        DisagreementCase::Case62 { .. } => (0, 0),
        DisagreementCase::Case61 { branches } => (1, branches.len()),
        DisagreementCase::Case63 { pendants, .. } => (2, pendants.len()),
    }
}

fn refine_partitions(p1: &[FixedBitSet], p2: &[FixedBitSet]) -> Vec<FixedBitSet> {
    let mut refined = Vec::new();
    for set1 in p1 {
        for set2 in p2 {
            let mut intersection = set1.clone();
            intersection.intersect_with(set2);
            if intersection.count_ones(..) > 0 {
                refined.push(intersection);
            }
        }
    }
    refined
}

fn reduce_common_cherries_multi(f1: &mut FptForest, forests: &mut [FptForest]) {
    loop {
        if prune_agreeing_roots_multi(f1, forests) {
            continue;
        }
        if grow_agreeing_cherries_multi(f1, forests) {
            continue;
        }
        break;
    }
}

fn prune_agreeing_roots_multi(f1: &mut FptForest, forests: &mut [FptForest]) -> bool {
    let map_f1 = dot_leafset_map(f1);
    let maps: Vec<FxHashMap<Vec<usize>, NodeId>> =
        forests.iter().map(|fi| dot_leafset_map(fi)).collect();

    // Collect R_t snapshot to avoid borrow conflict
    let rt_snapshot: Vec<FixedBitSet> = f1.rt.clone();

    for set in &rt_snapshot {
        let key = leafset_key(set);
        let Some(&node_f1) = map_f1.get(&key) else {
            continue;
        };

        let mut all_are_roots = true;
        let mut forest_nodes = Vec::with_capacity(forests.len());
        for (i, fi) in forests.iter().enumerate() {
            let Some(&node_fi) = maps[i].get(&key) else {
                all_are_roots = false;
                break;
            };
            if dot_parent(fi, node_fi) != NONE {
                all_are_roots = false;
                break;
            }
            forest_nodes.push(node_fi);
        }

        if !all_are_roots {
            continue;
        }

        trace!("prune agreeing root (multi): leafset={:?}", key);
        f1.done_leafsets.push(set.clone());
        rt_remove(f1, &key);
        deactivate_subtree(f1, node_f1);

        for (i, fi) in forests.iter_mut().enumerate() {
            fi.done_leafsets.push(set.clone());
            rt_remove(fi, &key);
            deactivate_subtree(fi, forest_nodes[i]);
        }

        return true;
    }
    false
}

fn grow_agreeing_cherries_multi(f1: &mut FptForest, forests: &mut [FptForest]) -> bool {
    let map_f1 = dot_leafset_map(f1);
    let maps: Vec<FxHashMap<Vec<usize>, NodeId>> =
        forests.iter().map(|fi| dot_leafset_map(fi)).collect();

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

        let mut all_siblings = true;
        for (i, fi) in forests.iter().enumerate() {
            let Some(&na) = maps[i].get(&key_a) else {
                all_siblings = false;
                break;
            };
            let Some(&nc) = maps[i].get(&key_c) else {
                all_siblings = false;
                break;
            };
            if !are_siblings_in_f2(fi, na, nc) {
                all_siblings = false;
                break;
            }
        }

        if !all_siblings {
            continue;
        }

        if grow_specific_cherry_multi(f1, forests, &key_a, &key_c, &map_f1) {
            return true;
        }
    }
    false
}

fn grow_specific_cherry_multi(
    f1: &mut FptForest,
    forests: &mut [FptForest],
    ka: &Vec<usize>,
    kc: &Vec<usize>,
    map_f1: &FxHashMap<Vec<usize>, NodeId>,
) -> bool {
    let Some(&node_a) = map_f1.get(ka) else {
        return false;
    };
    let Some(&node_c) = map_f1.get(kc) else {
        return false;
    };
    let p1 = dot_parent(f1, node_a);
    if p1 == NONE || dot_parent(f1, node_c) != p1 {
        return false;
    }

    let mut union = f1.live_leafsets[node_a as usize].clone();
    union.union_with(&f1.live_leafsets[node_c as usize]);

    rt_remove(f1, ka);
    rt_remove(f1, kc);
    rt_add(f1, union.clone());

    for fi in forests.iter_mut() {
        rt_remove(fi, ka);
        rt_remove(fi, kc);
        rt_add(fi, union.clone());
    }

    trace!("grow agreeing cherry (multi)");
    true
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
