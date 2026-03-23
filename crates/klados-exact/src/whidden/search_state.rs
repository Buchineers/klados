//! Search state with undo machine for Whidden's algorithm.
//!
//! Extends the basic checkpoint/rollback with contract and edge protection ops.

use fixedbitset::FixedBitSet;
use klados_core::{NONE, NodeId, XForest};

pub type Collapses = Vec<(u32, u32)>;

/// Operations that can be undone during backtracking.
pub enum UndoOp {
    /// Undo a cut (edge removal) in a forest.
    Cut { forest_idx: usize, node: NodeId },
    /// Reactivate a label that was deactivated (collapsed).
    Deactivate { label: u32 },
    /// Undo edge protection.
    Unprotect { forest_idx: usize, node: NodeId },
    /// Undo a push to the protected stack.
    PopProtectedStack,
    /// Undo a pop from the protected stack (re-push the node).
    PushProtectedStack { node: NodeId },
    /// Undo an unprotect (re-protect the node).
    Reprotect { forest_idx: usize, node: NodeId },
}

pub struct SearchState {
    pub forests: Vec<XForest>,
    /// Protected edges per forest — cannot be cut during search.
    pub protected: Vec<FixedBitSet>,
    pub collapses: Collapses,
    /// Stack of T2 nodes whose edges were protected (most recent last).
    /// Used by DEEPEST_PROTECTED_ORDER to constrain sibling pair selection.
    pub protected_stack: Vec<NodeId>,
    undo_log: Vec<UndoOp>,
    checkpoint_stack: Vec<(usize, usize)>,
}

impl SearchState {
    pub fn new(forests: Vec<XForest>) -> Self {
        let protected: Vec<FixedBitSet> = forests
            .iter()
            .map(|f| FixedBitSet::with_capacity(f.tree.num_nodes()))
            .collect();
        Self {
            forests,
            protected,
            collapses: Vec::new(),
            protected_stack: Vec::new(),
            undo_log: Vec::new(),
            checkpoint_stack: Vec::new(),
        }
    }

    pub fn checkpoint(&mut self) -> usize {
        let cp = (self.undo_log.len(), self.collapses.len());
        self.checkpoint_stack.push(cp);
        self.undo_log.len()
    }

    pub fn rollback(&mut self) {
        let (undo_target, collapses_target) = self.checkpoint_stack.pop().unwrap();
        while self.undo_log.len() > undo_target {
            self.undo_one();
        }
        self.collapses.truncate(collapses_target);
    }

    pub fn rollback_to(&mut self, target: usize) {
        while self.undo_log.len() > target {
            self.undo_one();
        }
    }

    fn undo_one(&mut self) {
        match self.undo_log.pop().unwrap() {
            UndoOp::Cut { forest_idx, node } => {
                self.forests[forest_idx].uncut(node);
            }
            UndoOp::Deactivate { label } => {
                for f in &mut self.forests {
                    f.reactivate_label(label);
                }
            }
            UndoOp::Unprotect { forest_idx, node } => {
                self.protected[forest_idx].set(node as usize, false);
            }
            UndoOp::PopProtectedStack => {
                self.protected_stack.pop();
            }
            UndoOp::PushProtectedStack { node } => {
                self.protected_stack.push(node);
            }
            UndoOp::Reprotect { forest_idx, node } => {
                self.protected[forest_idx].insert(node as usize);
            }
        }
    }

    /// Cut an edge in a forest (mark as removed).
    pub fn cut_node(&mut self, forest_idx: usize, node: NodeId) {
        let f = &self.forests[forest_idx];
        if node != f.tree.root && !f.is_cut(node) {
            self.forests[forest_idx].cut(node);
            self.undo_log.push(UndoOp::Cut { forest_idx, node });
        }
    }

    /// Collapse a sibling pair: deactivate `removed`, keep `kept`.
    pub fn add_collapse(&mut self, removed: u32, kept: u32) {
        self.collapses.push((removed, kept));
        for f in &mut self.forests {
            let a_node = f.tree.label_to_node[removed as usize];
            f.live_leafsets[a_node as usize].clear();
            let mut cur = f.tree.parent[a_node as usize];
            while cur != NONE {
                f.live_leafsets[cur as usize].set(removed as usize, false);
                if f.is_cut(cur) {
                    break;
                }
                cur = f.tree.parent[cur as usize];
            }
        }
        self.undo_log.push(UndoOp::Deactivate { label: removed });
    }

    /// Protect an edge (prevent it from being cut).
    #[inline]
    pub fn protect_edge(&mut self, forest_idx: usize, node: NodeId) {
        if !self.protected[forest_idx].contains(node as usize) {
            self.protected[forest_idx].insert(node as usize);
            self.undo_log.push(UndoOp::Unprotect { forest_idx, node });
        }
    }

    /// Push a T2 node onto the protected stack (for DEEPEST_PROTECTED_ORDER).
    pub fn push_protected_stack(&mut self, node: NodeId) {
        self.protected_stack.push(node);
        self.undo_log.push(UndoOp::PopProtectedStack);
    }

    /// Pop the protected stack (when a contraction consumes a protected node).
    /// Also clears the protection bit so EP no longer blocks cuts on this node.
    pub fn pop_protected_stack(&mut self) {
        if let Some(node) = self.protected_stack.pop() {
            self.undo_log.push(UndoOp::PushProtectedStack { node });
            // Also clear the protected bit — the node has been merged into a
            // larger entity, so EP on the original leaf no longer applies.
            self.unprotect_edge(1, node);
        }
    }

    /// Remove edge protection (undoable).
    pub fn unprotect_edge(&mut self, forest_idx: usize, node: NodeId) {
        if self.protected[forest_idx].contains(node as usize) {
            self.protected[forest_idx].set(node as usize, false);
            self.undo_log.push(UndoOp::Reprotect { forest_idx, node });
        }
    }

    /// Check if an edge is protected.
    #[inline]
    pub fn is_protected(&self, forest_idx: usize, node: NodeId) -> bool {
        self.protected[forest_idx].contains(node as usize)
    }
}
