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
    /// Incrementally maintained T1 sibling pairs stored as (label_a, label_b).
    /// Append-only; stale pairs are skipped during iteration.
    /// Checkpointed by length — rollback truncates appended entries.
    pub sibling_pairs: Vec<(u32, u32)>,
    /// Candidate singleton labels. Append-only, stale entries skipped on pop.
    /// A label is a candidate if its T2 component might have size 1.
    /// Checkpointed by length — rollback truncates.
    pub singleton_candidates: Vec<u32>,
    undo_log: Vec<UndoOp>,
    /// (undo_log_len, collapses_len, sibling_pairs_len, singletons_len)
    checkpoint_stack: Vec<(usize, usize, usize, usize)>,
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
            sibling_pairs: Vec::new(),
            singleton_candidates: Vec::new(),
            undo_log: Vec::new(),
            checkpoint_stack: Vec::new(),
        }
    }

    /// Initialize singleton candidates by scanning all labels.
    pub fn init_singletons(&mut self, label_space: usize) {
        self.singleton_candidates.clear();
        let f1 = &self.forests[0];
        let f2 = &self.forests[1];
        for lbl in 1..=label_space as u32 {
            if (lbl as usize) >= f1.tree.label_to_node.len() {
                continue;
            }
            let t1_node = f1.tree.label_to_node[lbl as usize];
            let t2_node = f2.tree.label_to_node[lbl as usize];
            if t1_node == NONE || t2_node == NONE {
                continue;
            }
            if f1.live_leaf_count[t1_node as usize] == 0 {
                continue;
            }
            let t2_comp_root = f2.component_root(t2_node);
            if f2.live_leaf_count[t2_comp_root as usize] == 1 {
                self.singleton_candidates.push(lbl);
            }
        }
    }

    pub fn checkpoint(&mut self) -> usize {
        let cp = (
            self.undo_log.len(),
            self.collapses.len(),
            self.sibling_pairs.len(),
            self.singleton_candidates.len(),
        );
        self.checkpoint_stack.push(cp);
        self.undo_log.len()
    }

    pub fn rollback(&mut self) {
        let (undo_target, collapses_target, pairs_target, singletons_target) =
            self.checkpoint_stack.pop().unwrap();
        while self.undo_log.len() > undo_target {
            self.undo_one();
        }
        self.collapses.truncate(collapses_target);
        self.sibling_pairs.truncate(pairs_target);
        self.singleton_candidates.truncate(singletons_target);
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
    /// When cutting in T2 (forest_idx=1), checks for new singleton candidates.
    pub fn cut_node(&mut self, forest_idx: usize, node: NodeId) {
        let f = &self.forests[forest_idx];
        if node != f.tree.root && !f.is_cut(node) {
            self.forests[forest_idx].cut(node);
            self.undo_log.push(UndoOp::Cut { forest_idx, node });

            // After a T2 cut, the cut node becomes a new component root.
            // Check if either the new component or the parent component became singleton-sized.
            if forest_idx == 1 {
                let f2 = &self.forests[1];
                // New component rooted at `node`
                if f2.live_leaf_count[node as usize] == 1 {
                    let lbl = f2.live_leafsets[node as usize].ones().next().unwrap() as u32;
                    self.singleton_candidates.push(lbl);
                }
                // Parent component may have shrunk
                let mut cur = f2.tree.parent[node as usize];
                while cur != NONE {
                    if f2.is_cut(cur) || f2.tree.parent[cur as usize] == NONE {
                        // cur is a component root
                        if f2.live_leaf_count[cur as usize] == 1 {
                            let lbl = f2.live_leafsets[cur as usize].ones().next().unwrap() as u32;
                            self.singleton_candidates.push(lbl);
                        }
                        break;
                    }
                    cur = f2.tree.parent[cur as usize];
                }
            }
        }
    }

    /// Collapse a sibling pair: deactivate `removed`, keep `kept`.
    /// Updates singleton candidates for the kept label's T2 component.
    pub fn add_collapse(&mut self, removed: u32, kept: u32) {
        self.collapses.push((removed, kept));
        for f in &mut self.forests {
            let a_node = f.tree.label_to_node[removed as usize];
            f.live_leafsets[a_node as usize].clear();
            f.live_leaf_count[a_node as usize] = 0;
            let mut cur = f.tree.parent[a_node as usize];
            while cur != NONE {
                f.live_leafsets[cur as usize].set(removed as usize, false);
                f.live_leaf_count[cur as usize] -= 1;
                if f.is_cut(cur) {
                    break;
                }
                cur = f.tree.parent[cur as usize];
            }
        }
        self.undo_log.push(UndoOp::Deactivate { label: removed });

        // After contraction, the kept label's T2 component may have become singleton.
        let f2 = &self.forests[1];
        let t2_kept = f2.tree.label_to_node[kept as usize];
        if t2_kept != NONE {
            let comp_root = f2.component_root(t2_kept);
            if f2.live_leaf_count[comp_root as usize] == 1 {
                self.singleton_candidates.push(kept);
            }
        }
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
