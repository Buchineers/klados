//! Cherry-based heuristic for upper bounds.

use klados_core::tree::{Label, NodeId, Tree, NONE};


pub fn cherry_reduce_ub(t1: &Tree, t2: &Tree) -> usize {
    let c1 = cherry_reduce(t1, t2);
    let c2 = cherry_reduce(t2, t1);
    c1.min(c2)
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
    let mut m_ref = MutableTree::from_tree(ref_tree);
    let mut m_other = MutableTree::from_tree(other_tree);
    let mut cuts = 0;
    loop {
        if m_ref.num_alive_leaves <= 1 {
            break;
        }
        let cherries = m_ref.find_cherries();
        if cherries.is_empty() {
            break;
        }
        let (a, b) = cherries[0];
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
    let mut mtrees: Vec<MutableTree> = trees.iter().map(|t| MutableTree::from_tree(t)).collect();
    let mut cuts = 0;
    loop {
        if mtrees[ref_idx].num_alive_leaves <= 1 {
            break;
        }
        let cherries = mtrees[ref_idx].find_cherries();
        if cherries.is_empty() {
            break;
        }
        let (a, b) = cherries[0];
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
