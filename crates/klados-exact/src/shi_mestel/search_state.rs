//! Search state management with checkpoint/rollback for efficient backtracking.

use klados_core::{NodeId, XForest};

pub type Collapses = Vec<(u32, u32)>;

pub enum UndoEntry {
    Cut { forest_idx: usize, node: NodeId },
    Deactivate { label: u32 },
}

pub struct SearchState {
    pub forests: Vec<XForest>,
    pub collapses: Collapses,
    pub undo_log: Vec<UndoEntry>,
    pub checkpoint_stack: Vec<(usize, usize)>,
}

impl SearchState {
    pub fn new(forests: Vec<XForest>) -> Self {
        Self {
            forests,
            collapses: Vec::new(),
            undo_log: Vec::new(),
            checkpoint_stack: Vec::new(),
        }
    }

    pub fn checkpoint(&mut self) {
        self.checkpoint_stack
            .push((self.undo_log.len(), self.collapses.len()));
    }

    pub fn rollback(&mut self) {
        let (undo_target, collapses_target) = self.checkpoint_stack.pop().unwrap();
        while self.undo_log.len() > undo_target {
            match self.undo_log.pop().unwrap() {
                UndoEntry::Cut { forest_idx, node } => {
                    self.forests[forest_idx].uncut(node);
                }
                UndoEntry::Deactivate { label } => {
                    for f in &mut self.forests {
                        f.reactivate_label(label);
                    }
                }
            }
        }
        self.collapses.truncate(collapses_target);
    }

    pub fn cut_node(&mut self, forest_idx: usize, node: NodeId) {
        if node != self.forests[forest_idx].tree.root && !self.forests[forest_idx].is_cut(node) {
            self.forests[forest_idx].cut(node);
            self.undo_log.push(UndoEntry::Cut { forest_idx, node });
        }
    }

    pub fn add_collapse(&mut self, removed: u32, kept: u32) {
        self.collapses.push((removed, kept));
        for f in &mut self.forests {
            let a_node = f.tree.label_to_node[removed as usize];
            f.live_leafsets[a_node as usize].clear();
            let mut cur = f.tree.parent[a_node as usize];
            while cur != klados_core::NONE {
                f.live_leafsets[cur as usize].set(removed as usize, false);
                if f.is_cut(cur) {
                    break;
                }
                cur = f.tree.parent[cur as usize];
            }
        }
        self.undo_log.push(UndoEntry::Deactivate { label: removed });
    }
}
