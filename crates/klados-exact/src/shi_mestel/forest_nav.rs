//! Forest navigation helpers for traversing forests with cut edges.

use fixedbitset::FixedBitSet;
use klados_core::{NONE, NodeId, XForest};

#[derive(Clone, Copy)]
pub struct Children {
    nodes: [NodeId; 2],
    len: u8,
}

impl Children {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            nodes: [NONE, NONE],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, node: NodeId) {
        self.nodes[self.len as usize] = node;
        self.len += 1;
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl std::ops::Index<usize> for Children {
    type Output = NodeId;
    #[inline(always)]
    fn index(&self, idx: usize) -> &NodeId {
        &self.nodes[idx]
    }
}

pub fn active_children_xf(forest: &XForest, node: NodeId) -> Children {
    let tree = &forest.tree;
    let mut out = Children::new();
    if let Some((left, right)) = tree.children(node) {
        if left != NONE
            && !forest.is_cut(left)
            && forest.live_leaf_count[left as usize] > 0
        {
            out.push(left);
        }
        if right != NONE
            && !forest.is_cut(right)
            && forest.live_leaf_count[right as usize] > 0
        {
            out.push(right);
        }
    }
    out
}

pub fn forest_children(forest: &XForest, node: NodeId) -> Children {
    let mut out = Children::new();
    if let Some((left, right)) = forest.tree.children(node) {
        if left != NONE
            && !forest.is_cut(left)
            && forest.live_leaf_count[left as usize] > 0
        {
            out.push(descend_to_effective(forest, left));
        }
        if right != NONE
            && !forest.is_cut(right)
            && forest.live_leaf_count[right as usize] > 0
        {
            out.push(descend_to_effective(forest, right));
        }
    }
    out
}

pub fn descend_to_effective(forest: &XForest, mut node: NodeId) -> NodeId {
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

pub fn forest_is_leaf(forest: &XForest, node: NodeId) -> bool {
    if forest.tree.is_leaf(node) {
        return true;
    }
    active_children_xf(forest, node).is_empty()
}

pub fn forest_parent_leaf(forest: &XForest, node: NodeId) -> NodeId {
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
        if forest.is_cut(cur) {
            return NONE;
        }
        let p = forest.tree.parent[cur as usize];
        if p == NONE {
            return cur;
        }
        cur = p;
    }
}

pub fn forest_lca(forest: &XForest, mut a: NodeId, mut b: NodeId) -> NodeId {
    let depth = &forest.tree.depth;
    while depth[a as usize] > depth[b as usize] {
        if forest.is_cut(a) {
            return NONE;
        }
        a = forest.tree.parent[a as usize];
        if a == NONE {
            return NONE;
        }
    }
    while depth[b as usize] > depth[a as usize] {
        if forest.is_cut(b) {
            return NONE;
        }
        b = forest.tree.parent[b as usize];
        if b == NONE {
            return NONE;
        }
    }
    while a != b {
        if forest.is_cut(a) || forest.is_cut(b) {
            return NONE;
        }
        a = forest.tree.parent[a as usize];
        b = forest.tree.parent[b as usize];
        if a == NONE || b == NONE {
            return NONE;
        }
    }
    a
}

pub fn component_leaf_sets_xf(forest: &XForest, _label_space: usize) -> Vec<FixedBitSet> {
    let mut components = Vec::new();
    if forest.live_leaf_count[forest.tree.root as usize] > 0 {
        components.push(forest.live_leafsets[forest.tree.root as usize].clone());
    }
    for node in forest.cut_edges.ones() {
        if forest.live_leaf_count[node] > 0 {
            components.push(forest.live_leafsets[node].clone());
        }
    }
    components
}

pub fn forest_resolves_to(forest: &XForest, start: NodeId, target: NodeId) -> bool {
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
