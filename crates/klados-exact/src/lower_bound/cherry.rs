//! Cherry-based heuristic for upper bounds.

use klados_core::tree::{Label, NodeId, Tree, NONE};

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
        if l == node {
            self.right[p as usize]
        } else {
            l
        }
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
    if da >= db {
        a
    } else {
        b
    }
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
