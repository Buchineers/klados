//! Precomputed tree data for efficient LCA queries.

use fixedbitset::FixedBitSet;
use klados_core::tree::{NONE, NodeId, Tree};

pub struct TreeData {
    pub tree: Tree,
    pub leaf_set: Vec<FixedBitSet>,
    pub euler: Vec<NodeId>,
    pub euler_depth: Vec<u16>,
    pub first_occ: Vec<u32>,
    pub sparse: Vec<Vec<u32>>,
    pub post_order: Vec<NodeId>,
}

impl TreeData {
    pub fn build(tree: &Tree) -> Self {
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
            if !returning && let Some((left, right)) = tree.children(node) {
                stack.push((node, true));
                stack.push((right, false));
                stack.push((node, true));
                stack.push((left, false));
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

        TreeData {
            tree: tree.clone(),
            leaf_set,
            euler,
            euler_depth,
            first_occ,
            sparse,
            post_order: post,
        }
    }

    #[inline]
    pub fn lca(&self, u: NodeId, v: NodeId) -> NodeId {
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

    pub fn lca_of_labels(&self, labels: &FixedBitSet) -> NodeId {
        let mut result = NONE;
        for lbl in labels.ones() {
            if lbl >= self.tree.label_to_node.len() {
                continue;
            }
            let node = self.tree.label_to_node[lbl];
            if node == NONE {
                continue;
            }
            if result == NONE {
                result = node;
            } else {
                result = self.lca(result, node);
            }
        }
        result
    }
}
