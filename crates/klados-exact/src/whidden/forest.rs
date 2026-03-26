//! SoA forest with physical tree mutations matching rspr's semantics.
//!
//! Every structural change (cut, contract, splice) physically modifies the
//! parent/left/right arrays. The undo machine records old values for rollback.
//!
//! Key operations matching rspr:
//! - `cut_parent(n)`: detach n from its parent → n becomes a component root
//! - `contract(n)`: if n has 1 child, splice n out; if 0 children and is root,
//!   mark component dead
//! - `contract_sibling_pair(parent)`: detach both children → parent becomes leaf
//! - `is_leaf(n)`: left[n] == NONE (physical, not flag-based)

use klados_core::tree::{Label, Tree, NONE, NodeId};

/// A forest built from a Tree, supporting physical cut/contract/undo.
#[derive(Clone)]
pub struct Forest {
    // --- Topology (mutable via undo machine) ---
    pub parent: Vec<NodeId>,
    pub left: Vec<NodeId>,
    pub right: Vec<NodeId>,

    // --- Original topology (immutable, for solution extraction) ---
    pub orig_parent: Vec<NodeId>,
    pub orig_left: Vec<NodeId>,
    pub orig_right: Vec<NodeId>,

    // --- Labels ---
    pub label: Vec<Label>,         // current label (changes during contraction)
    pub orig_label: Vec<Label>,    // original label (immutable)
    pub label_to_node: Vec<NodeId>, // label → current node

    // --- Twin pointers (T1↔T2 by label) ---
    pub twin: Vec<NodeId>,

    // --- Components ---
    pub components: Vec<NodeId>,

    // --- Contraction tracking ---
    /// collapsed_into[lbl] = representative label after contractions.
    /// Updated during Case 2 contractions. Used for solution extraction.
    pub collapsed_into: Vec<Label>,

    // --- Metadata ---
    pub num_leaves: u32,
    pub root: NodeId,
    pub num_nodes: usize,
}

impl Forest {
    /// Build a Forest from a klados Tree.
    pub fn from_tree(tree: &Tree) -> Self {
        let n = tree.num_nodes();
        Self {
            parent: tree.parent.clone(),
            left: tree.left.clone(),
            right: tree.right.clone(),
            orig_parent: tree.parent.clone(),
            orig_left: tree.left.clone(),
            orig_right: tree.right.clone(),
            label: tree.label.clone(),
            orig_label: tree.label.clone(),
            label_to_node: tree.label_to_node.clone(),
            twin: vec![NONE; n],
            components: vec![tree.root],
            collapsed_into: (0..=tree.num_leaves).collect(),
            num_leaves: tree.num_leaves,
            root: tree.root,
            num_nodes: n,
        }
    }

    // --- Predicates ---

    #[inline(always)]
    pub fn is_leaf(&self, node: NodeId) -> bool {
        self.left[node as usize] == NONE
    }

    /// A sibling pair: internal node with two leaf children.
    #[inline]
    pub fn is_sibling_pair(&self, node: NodeId) -> bool {
        let lc = self.left[node as usize];
        let rc = self.right[node as usize];
        lc != NONE && rc != NONE
            && self.is_leaf(lc) && self.is_leaf(rc)
    }

    /// Is this node a singleton component (leaf that is a component root)?
    #[inline]
    pub fn is_singleton(&self, node: NodeId) -> bool {
        self.is_leaf(node) && self.parent[node as usize] == NONE
    }

    // --- Navigation ---

    /// Sibling: the other child of this node's parent.
    #[inline]
    pub fn sibling(&self, node: NodeId) -> NodeId {
        let p = self.parent[node as usize];
        if p == NONE { return NONE; }
        let l = self.left[p as usize];
        if l == node { self.right[p as usize] } else { l }
    }

    /// Number of live children (0, 1, or 2).
    #[inline]
    pub fn num_children(&self, node: NodeId) -> u8 {
        let l = self.left[node as usize] != NONE;
        let r = self.right[node as usize] != NONE;
        l as u8 + r as u8
    }

    /// Get the single child (when node has exactly 1 child).
    #[inline]
    pub fn only_child(&self, node: NodeId) -> NodeId {
        let lc = self.left[node as usize];
        let rc = self.right[node as usize];
        if lc != NONE { lc } else { rc }
    }
}

/// Set up twin pointers between two forests based on leaf labels.
pub fn sync_twins(f1: &mut Forest, f2: &mut Forest) {
    for lbl in 1..=f1.num_leaves {
        let n1 = f1.label_to_node[lbl as usize];
        let n2 = f2.label_to_node[lbl as usize];
        if n1 != NONE && n2 != NONE {
            f1.twin[n1 as usize] = n2;
            f2.twin[n2 as usize] = n1;
        }
    }
}
