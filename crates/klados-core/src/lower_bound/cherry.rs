//! Cherry-based heuristic for upper bounds.

use crate::tree::{Label, NONE, NodeId, Tree};

pub fn cherry_reduce_ub(t1: &Tree, t2: &Tree) -> usize {
    let mut best = cherry_reduce(t1, t2).min(cherry_reduce(t2, t1));

    // Try multiple seeded runs for better upper bound.
    // Each run is O(n^2) and takes <1 ms for n≤50, so increasing seeds is cheap.
    // More seeds give a better chance of finding the cherry ordering that minimizes cuts.
    for seed in 1..=20 {
        best = best.min(cherry_reduce_seeded(t1, t2, seed));
        best = best.min(cherry_reduce_seeded(t2, t1, seed));
    }
    best
}

struct MutableTree {
    parent: Vec<NodeId>,
    left: Vec<NodeId>,
    right: Vec<NodeId>,
    label: Vec<Label>,
    label_to_node: Vec<NodeId>,
    alive: Vec<bool>,
    num_alive_leaves: u32,
    root: NodeId,
}

impl MutableTree {
    fn from_tree(t: &Tree) -> Self {
        Self {
            parent: t.parent.clone(),
            left: t.left.clone(),
            right: t.right.clone(),
            label: t.label.clone(),
            label_to_node: t.label_to_node.clone(),
            alive: vec![true; t.num_nodes()],
            num_alive_leaves: t.num_leaves,
            root: t.root,
        }
    }

    #[inline]
    fn is_leaf(&self, node: NodeId) -> bool {
        self.left[node as usize] == NONE
    }

    #[inline]
    fn sibling(&self, node: NodeId) -> NodeId {
        let p = self.parent[node as usize];
        let l = self.left[p as usize];
        if l == node { self.right[p as usize] } else { l }
    }

    fn is_cherry(&self, a: Label, b: Label) -> bool {
        let na = self.label_to_node[a as usize];
        let nb = self.label_to_node[b as usize];
        if na == NONE || nb == NONE {
            return false;
        }
        if !self.alive[na as usize] || !self.alive[nb as usize] {
            return false;
        }
        let pa = self.parent[na as usize];
        let pb = self.parent[nb as usize];
        pa != NONE && pa == pb
    }

    fn contract_cherry(&mut self, keep: Label, remove: Label) {
        let keep_node = self.label_to_node[keep as usize];
        let remove_node = self.label_to_node[remove as usize];
        let parent = self.parent[keep_node as usize];
        self.alive[keep_node as usize] = false;
        self.alive[remove_node as usize] = false;
        self.left[parent as usize] = NONE;
        self.right[parent as usize] = NONE;
        self.label[parent as usize] = keep;
        self.label_to_node[keep as usize] = parent;
        self.label_to_node[remove as usize] = NONE;
        self.num_alive_leaves -= 1;
    }

    fn cut_leaf(&mut self, lbl: Label) {
        let node = self.label_to_node[lbl as usize];
        if node == NONE || !self.alive[node as usize] {
            return;
        }
        let parent = self.parent[node as usize];
        if parent == NONE {
            self.alive[node as usize] = false;
            self.label_to_node[lbl as usize] = NONE;
            self.num_alive_leaves -= 1;
            return;
        }
        let sib = self.sibling(node);
        let gp = self.parent[parent as usize];
        self.alive[node as usize] = false;
        self.label_to_node[lbl as usize] = NONE;
        self.num_alive_leaves -= 1;
        self.alive[parent as usize] = false;
        self.parent[sib as usize] = gp;
        if gp == NONE {
            self.root = sib;
        } else if self.left[gp as usize] == parent {
            self.left[gp as usize] = sib;
        } else {
            self.right[gp as usize] = sib;
        }
    }

    fn find_cherries(&self) -> Vec<(Label, Label)> {
        let mut cherries = Vec::new();
        for i in 0..self.parent.len() {
            let node = i as NodeId;
            if !self.alive[node as usize] || self.is_leaf(node) {
                continue;
            }
            let l = self.left[node as usize];
            let r = self.right[node as usize];
            if l != NONE
                && r != NONE
                && self.alive[l as usize]
                && self.alive[r as usize]
                && self.is_leaf(l)
                && self.is_leaf(r)
            {
                let la = self.label[l as usize];
                let lb = self.label[r as usize];
                if la > 0 && lb > 0 {
                    cherries.push((la.min(lb), la.max(lb)));
                }
            }
        }
        cherries
    }
}

fn cherry_reduce(ref_tree: &Tree, other_tree: &Tree) -> usize {
    cherry_reduce_seeded(ref_tree, other_tree, 0)
}

fn cherry_reduce_seeded(ref_tree: &Tree, other_tree: &Tree, seed: u64) -> usize {
    let mut m_ref = MutableTree::from_tree(ref_tree);
    let mut m_other = MutableTree::from_tree(other_tree);
    let mut cuts = 0;
    let mut step = 0u64;
    loop {
        if m_ref.num_alive_leaves <= 1 {
            break;
        }
        let cherries = m_ref.find_cherries();
        if cherries.is_empty() {
            break;
        }
        // Deterministic permutation based on seed + step
        let idx = if cherries.len() > 1 && seed != 0 {
            let h = seed
                .wrapping_mul(0x9e3779b97f4a7c15)
                .wrapping_add(step.wrapping_mul(0x517cc1b727220a95));
            (h as usize) % cherries.len()
        } else {
            0
        };
        step += 1;

        let (a, b) = cherries[idx];
        if m_other.is_cherry(a, b) {
            m_ref.contract_cherry(a, b);
            m_other.contract_cherry(a, b);
        } else {
            let cut = pick_leaf_to_cut(&m_other, a, b);
            m_ref.cut_leaf(cut);
            m_other.cut_leaf(cut);
            cuts += 1;
        }
    }
    cuts
}

fn pick_leaf_to_cut(t2: &MutableTree, a: Label, b: Label) -> Label {
    let na = t2.label_to_node[a as usize];
    let nb = t2.label_to_node[b as usize];
    if na == NONE {
        return a;
    }
    if nb == NONE {
        return b;
    }
    let da = depth_in_mtree(t2, na);
    let db = depth_in_mtree(t2, nb);
    if da >= db { a } else { b }
}

fn depth_in_mtree(t: &MutableTree, mut node: NodeId) -> u32 {
    let mut d = 0;
    while t.parent[node as usize] != NONE {
        node = t.parent[node as usize];
        d += 1;
    }
    d
}

pub fn greedy_multi_tree_ub(trees: &[Tree], ref_idx: usize) -> usize {
    greedy_multi_tree_ub_seeded(trees, ref_idx, 0)
}

pub fn greedy_multi_tree_ub_seeded(trees: &[Tree], ref_idx: usize, seed: u64) -> usize {
    let mut mtrees: Vec<MutableTree> = trees.iter().map(MutableTree::from_tree).collect();
    let mut cuts = 0;
    let mut step = 0u64;
    loop {
        if mtrees[ref_idx].num_alive_leaves <= 1 {
            break;
        }
        let cherries = mtrees[ref_idx].find_cherries();
        if cherries.is_empty() {
            break;
        }
        // Deterministic permutation based on seed + step
        let idx = if cherries.len() > 1 && seed != 0 {
            let h = seed
                .wrapping_mul(0x9e3779b97f4a7c15)
                .wrapping_add(step.wrapping_mul(0x517cc1b727220a95));
            (h as usize) % cherries.len()
        } else {
            0
        };
        step += 1;

        let (a, b) = cherries[idx];
        if mtrees.iter().all(|t| t.is_cherry(a, b)) {
            for t in &mut mtrees {
                t.contract_cherry(a, b);
            }
        } else {
            let cut = pick_multi_tree_cut(&mtrees, ref_idx, a, b);
            for t in &mut mtrees {
                t.cut_leaf(cut);
            }
            cuts += 1;
        }
    }
    cuts + 1
}

/// Run the greedy multi-tree cherry reduction and return the leaf partition.
///
/// Returns `(num_components, partition)` where `partition[j]` is the 0-based
/// component index for the leaf with label `j+1`. Component 0 contains all
/// leaves that were never cut (the "surviving" group); components 1.. each
/// hold a set of leaves that were cut together as one group.
///
/// The number of components equals the number of cuts + 1, matching
/// `greedy_multi_tree_ub_seeded`.
pub fn greedy_multi_tree_partition(
    trees: &[Tree],
    ref_idx: usize,
    seed: u64,
) -> (usize, Vec<usize>) {
    let n = trees[ref_idx].num_leaves as usize;
    let mut mtrees: Vec<MutableTree> = trees.iter().map(MutableTree::from_tree).collect();

    // Union-find over labels 1..=n (index 0 unused).
    let mut uf: Vec<usize> = (0..=n).collect();
    // Component assignment: 0 = surviving (default); 1.. = cut groups.
    let mut label_comp: Vec<usize> = vec![0; n + 1];
    let mut next_comp: usize = 1;

    let mut step = 0u64;
    loop {
        if mtrees[ref_idx].num_alive_leaves <= 1 {
            break;
        }
        let cherries = mtrees[ref_idx].find_cherries();
        if cherries.is_empty() {
            break;
        }
        let idx = if cherries.len() > 1 && seed != 0 {
            let h = seed
                .wrapping_mul(0x9e3779b97f4a7c15)
                .wrapping_add(step.wrapping_mul(0x517cc1b727220a95));
            (h as usize) % cherries.len()
        } else {
            0
        };
        step += 1;

        let (a, b) = cherries[idx];
        if mtrees.iter().all(|t| t.is_cherry(a, b)) {
            for t in &mut mtrees {
                t.contract_cherry(a, b);
            }
            // b merges into a: union their groups (a stays as representative).
            let ra = uf_find(&uf, a as usize);
            let rb = uf_find(&uf, b as usize);
            if ra != rb {
                uf[rb] = ra;
            }
        } else {
            let cut = pick_multi_tree_cut(&mtrees, ref_idx, a, b);
            for t in &mut mtrees {
                t.cut_leaf(cut);
            }
            // All leaves in cut's UF group get a new component ID.
            let r_cut = uf_find(&uf, cut as usize);
            let comp_idx = next_comp;
            next_comp += 1;
            for lbl in 1..=n {
                if uf_find(&uf, lbl) == r_cut {
                    label_comp[lbl] = comp_idx;
                }
            }
        }
    }

    // Map label 1..=n → component (0-indexed by leaf = label-1).
    let partition: Vec<usize> = (1..=n).map(|lbl| label_comp[lbl]).collect();
    (next_comp, partition)
}

fn uf_find(uf: &[usize], mut x: usize) -> usize {
    while uf[x] != x {
        x = uf[x];
    }
    x
}

fn pick_multi_tree_cut(mtrees: &[MutableTree], ref_idx: usize, a: Label, b: Label) -> Label {
    let mut a_deeper = 0i32;
    let mut total_diff: i32 = 0;
    for (i, t) in mtrees.iter().enumerate() {
        if i == ref_idx {
            continue;
        }
        let na = t.label_to_node[a as usize];
        let nb = t.label_to_node[b as usize];
        if na == NONE || nb == NONE {
            continue;
        }
        let da = depth_in_mtree(t, na) as i32;
        let db = depth_in_mtree(t, nb) as i32;
        total_diff += da - db;
        if da > db {
            a_deeper += 1;
        } else if db > da {
            a_deeper -= 1;
        }
    }
    if a_deeper > 0 || (a_deeper == 0 && total_diff >= 0) {
        a
    } else {
        b
    }
}

/// Compute a multi-tree UB by starting from the best pairwise partition
/// and refining it to be valid across all trees.
///
/// For each tree, any component that isn't a connected subtree is split
/// into its connected sub-components. Iterate until stable.
/// This is much tighter than the all-tree cherry heuristic for large m.
pub fn pairwise_refine_ub(trees: &[Tree], num_leaves: usize) -> (usize, Vec<usize>) {
    let m = trees.len();
    if m <= 1 {
        return (1, vec![0; num_leaves]);
    }

    // Find the best pairwise partition across all pairs and seeds.
    let mut best_ub = num_leaves;
    let mut best_partition: Vec<usize> = (0..num_leaves).collect(); // singletons

    for i in 0..m {
        for j in (i + 1)..m {
            let pair = [trees[i].clone(), trees[j].clone()];
            for seed in 0..=10u64 {
                let (k, part) = greedy_multi_tree_partition(&pair, 0, seed);
                if k < best_ub {
                    best_ub = k;
                    best_partition = part;
                }
                let (k, part) = greedy_multi_tree_partition(&pair, 1, seed);
                if k < best_ub {
                    best_ub = k;
                    best_partition = part;
                }
            }
        }
    }

    // Refine: split components that aren't connected in some tree.
    let mut partition = best_partition;
    let mut next_comp = *partition.iter().max().unwrap_or(&0) + 1;

    loop {
        let mut changed = false;
        for q in 0..m {
            let tree = &trees[q];
            let nn = tree.num_nodes();

            // Bottom-up: compute uniform component per subtree.
            let mut sub_comp = vec![usize::MAX; nn];
            for v in tree.post_order() {
                if tree.is_leaf(v) {
                    let lbl = tree.label[v as usize];
                    if lbl > 0 && (lbl as usize) <= num_leaves {
                        sub_comp[v as usize] = partition[(lbl - 1) as usize];
                    }
                } else {
                    let l = tree.left[v as usize];
                    let r = tree.right[v as usize];
                    if l != NONE && r != NONE {
                        let lc = sub_comp[l as usize];
                        let rc = sub_comp[r as usize];
                        sub_comp[v as usize] = if lc == rc { lc } else { usize::MAX };
                    }
                }
            }

            // UF: merge leaves that are in the same uniform subtree.
            let mut uf: Vec<usize> = (0..num_leaves).collect();
            fn find(p: &mut [usize], x: usize) -> usize {
                if p[x] != x {
                    p[x] = find(p, p[x]);
                }
                p[x]
            }

            for v in tree.post_order() {
                if tree.is_leaf(v) || sub_comp[v as usize] == usize::MAX {
                    continue;
                }
                // Uniform subtree: merge a leaf from left with a leaf from right.
                let l = tree.left[v as usize];
                let r = tree.right[v as usize];
                if l == NONE || r == NONE {
                    continue;
                }

                let la = find_any_leaf_label(tree, l, num_leaves);
                let ra = find_any_leaf_label(tree, r, num_leaves);
                if let (Some(a), Some(b)) = (la, ra) {
                    let fa = find(&mut uf, a);
                    let fb = find(&mut uf, b);
                    if fa != fb {
                        uf[fa] = fb;
                    }
                }
            }

            // Check: if any component is split by this tree, refine.
            // Group leaves by (input component, UF root).
            let mut groups: std::collections::HashMap<(usize, usize), Vec<usize>> =
                std::collections::HashMap::new();
            for j in 0..num_leaves {
                let comp = partition[j];
                let root = find(&mut uf, j);
                groups.entry((comp, root)).or_default().push(j);
            }

            // For each input component, if it has multiple UF groups, split.
            let mut comp_groups: std::collections::HashMap<usize, Vec<Vec<usize>>> =
                std::collections::HashMap::new();
            for ((comp, _root), leaves) in &groups {
                comp_groups.entry(*comp).or_default().push(leaves.clone());
            }
            for (_comp, sub_groups) in &comp_groups {
                if sub_groups.len() > 1 {
                    changed = true;
                    // Keep largest sub-group with original comp_id, assign new to rest.
                    let mut sorted = sub_groups.clone();
                    sorted.sort_by(|a, b| b.len().cmp(&a.len()));
                    for group in &sorted[1..] {
                        for &j in group {
                            partition[j] = next_comp;
                        }
                        next_comp += 1;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Phase 2: Split components that contain incompatible triples (H4).
    // An incompatible triple (a,b,c) has different "odd leaf" across trees.
    loop {
        let mut changed = false;
        // Group leaves by component.
        let mut comp_leaves: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for j in 0..num_leaves {
            comp_leaves.entry(partition[j]).or_default().push(j);
        }

        'outer: for (_comp, leaves) in &comp_leaves {
            if leaves.len() < 3 {
                continue;
            }
            // Check all triples within this component.
            for ii in 0..leaves.len() {
                for jj in (ii + 1)..leaves.len() {
                    for kk in (jj + 1)..leaves.len() {
                        let a = leaves[ii];
                        let b = leaves[jj];
                        let c = leaves[kk];
                        if is_incompatible_triple(trees, a, b, c) {
                            // Split: remove c from this component.
                            partition[c] = next_comp;
                            next_comp += 1;
                            changed = true;
                            break 'outer; // restart after any split
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Count components.
    let mut seen = std::collections::HashSet::new();
    for &c in &partition {
        seen.insert(c);
    }
    (seen.len(), partition)
}

/// Check if triple (a,b,c) is incompatible across trees (different odd leaf).
fn is_incompatible_triple(trees: &[Tree], a: usize, b: usize, c: usize) -> bool {
    let la = (a + 1) as Label;
    let lb = (b + 1) as Label;
    let lc = (c + 1) as Label;
    let mut first_odd = u8::MAX;
    for tree in trees {
        let na = tree.node_by_label(la);
        let nb = tree.node_by_label(lb);
        let nc = tree.node_by_label(lc);
        let nca_ab = tree.nearest_common_ancestor(na, nb);
        let nca_ac = tree.nearest_common_ancestor(na, nc);
        let nca_bc = tree.nearest_common_ancestor(nb, nc);
        let d_ab = tree.depth[nca_ab as usize];
        let d_ac = tree.depth[nca_ac as usize];
        let d_bc = tree.depth[nca_bc as usize];
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

/// Find any leaf label (0-indexed) in the subtree rooted at `node`.
fn find_any_leaf_label(tree: &Tree, node: NodeId, num_leaves: usize) -> Option<usize> {
    if node == NONE {
        return None;
    }
    if tree.is_leaf(node) {
        let lbl = tree.label[node as usize];
        if lbl > 0 && (lbl as usize) <= num_leaves {
            return Some((lbl - 1) as usize);
        }
        return None;
    }
    let l = tree.left[node as usize];
    if let Some(r) = find_any_leaf_label(tree, l, num_leaves) {
        return Some(r);
    }
    find_any_leaf_label(tree, tree.right[node as usize], num_leaves)
}
