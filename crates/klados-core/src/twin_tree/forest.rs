//! Twin-pointer forest pair for 2-tree rSPR branch-and-bound.
//!
//! Stores both T1 and T2 in a single struct with direct twin pointers
//! between corresponding leaves. Physical tree mutations (cut, contract,
//! splice) modify the topology arrays; the undo machine records old values.
//!
//! Optimized for the 2-tree case: no generic tree indexing,
//! twin lookups are O(1) array access.

use crate::tree::{Label, Tree, NONE, NodeId};

/// Index into the tree pair: T1 = 0, T2 = 1.
pub const T1: usize = 0;
pub const T2: usize = 1;

/// Paired forest for T1 and T2 with twin pointers.
#[derive(Clone)]
pub struct TwinForest {
    // --- Topology (mutable via undo) ---
    pub parent: [Vec<NodeId>; 2],
    pub left:   [Vec<NodeId>; 2],
    pub right:  [Vec<NodeId>; 2],

    // --- Labels ---
    pub label:         [Vec<Label>; 2],
    pub label_to_node: [Vec<NodeId>; 2],

    // --- Twin pointers: twin[T1][node] → T2 node, twin[T2][node] → T1 node ---
    pub twin: [Vec<NodeId>; 2],

    // --- Components ---
    pub components: [Vec<NodeId>; 2],

    // --- Contraction tracking (T1 only, for extraction) ---
    pub collapsed_into: Vec<Label>,

    // --- Immutable T1 original (for solution extraction) ---
    pub orig_parent: Vec<NodeId>,
    pub orig_left:   Vec<NodeId>,
    pub orig_right:  Vec<NodeId>,
    pub orig_label:  Vec<Label>,

    // --- Precomputed T2 depth (immutable, for Case 3 orientation) ---
    pub t2_depth: Vec<u16>,

    // --- Edge protection (T2 only, for branch pruning) ---
    pub protected: Vec<bool>,

    // --- Metadata ---
    pub num_nodes: [usize; 2],
    pub root:      [NodeId; 2],
    pub num_leaves: u32,
}

impl TwinForest {
    /// Build from two klados Trees, setting up twin pointers by label.
    pub fn from_trees(t1: &Tree, t2: &Tree, num_leaves: u32) -> Self {
        let n1 = t1.num_nodes();
        let n2 = t2.num_nodes();

        let mut tf = Self {
            parent: [t1.parent.clone(), t2.parent.clone()],
            left:   [t1.left.clone(),   t2.left.clone()],
            right:  [t1.right.clone(),  t2.right.clone()],
            label:         [t1.label.clone(),         t2.label.clone()],
            label_to_node: [t1.label_to_node.clone(), t2.label_to_node.clone()],
            twin: [vec![NONE; n1], vec![NONE; n2]],
            components: [vec![t1.root], vec![t2.root]],
            collapsed_into: (0..=num_leaves).collect(),
            orig_parent: t1.parent.clone(),
            orig_left:   t1.left.clone(),
            orig_right:  t1.right.clone(),
            orig_label:  t1.label.clone(),
            t2_depth: vec![0; n2],
            protected: vec![false; n2],
            num_nodes: [n1, n2],
            root: [t1.root, t2.root],
            num_leaves,
        };

        // Precompute T2 depth (distance from original root, immutable)
        Self::compute_depth(&tf.left[T2], &tf.right[T2], t2.root, &mut tf.t2_depth);

        // Set up twin pointers by matching leaf labels
        for lbl in 1..=num_leaves {
            let n1 = tf.label_to_node[T1][lbl as usize];
            let n2 = tf.label_to_node[T2][lbl as usize];
            if n1 != NONE && n2 != NONE {
                tf.twin[T1][n1 as usize] = n2;
                tf.twin[T2][n2 as usize] = n1;
            }
        }

        tf
    }

    /// Precompute depth via iterative DFS.
    fn compute_depth(left: &[NodeId], right: &[NodeId], root: NodeId, depth: &mut [u16]) {
        let mut stack = vec![(root, 0u16)];
        while let Some((node, d)) = stack.pop() {
            depth[node as usize] = d;
            let rc = right[node as usize];
            if rc != NONE { stack.push((rc, d + 1)); }
            let lc = left[node as usize];
            if lc != NONE { stack.push((lc, d + 1)); }
        }
    }

    // --- Predicates ---

    #[inline(always)]
    pub fn is_leaf(&self, ti: usize, node: NodeId) -> bool {
        self.left[ti][node as usize] == NONE
    }

    #[inline]
    #[allow(dead_code)] // utility for future optimizations
    pub fn is_sibling_pair(&self, ti: usize, node: NodeId) -> bool {
        let lc = self.left[ti][node as usize];
        let rc = self.right[ti][node as usize];
        lc != NONE && rc != NONE
            && self.is_leaf(ti, lc) && self.is_leaf(ti, rc)
    }

    // --- Navigation ---

    #[inline]
    pub fn sibling(&self, ti: usize, node: NodeId) -> NodeId {
        let p = self.parent[ti][node as usize];
        if p == NONE { return NONE; }
        let l = self.left[ti][p as usize];
        if l == node { self.right[ti][p as usize] } else { l }
    }

    #[inline]
    pub fn num_children(&self, ti: usize, node: NodeId) -> u8 {
        (self.left[ti][node as usize] != NONE) as u8
            + (self.right[ti][node as usize] != NONE) as u8
    }

    #[inline]
    pub fn only_child(&self, ti: usize, node: NodeId) -> NodeId {
        let lc = self.left[ti][node as usize];
        if lc != NONE { lc } else { self.right[ti][node as usize] }
    }
}
