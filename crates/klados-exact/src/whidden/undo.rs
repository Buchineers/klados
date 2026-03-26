//! Undo machine for branch-and-bound backtracking.
//!
//! Every physical mutation to the forest pair is recorded as an UndoOp.
//! `undo_to(checkpoint)` replays ops in reverse to restore state.

use klados_core::tree::{NodeId, NONE};
use super::forest::Forest;

#[derive(Clone, Copy)]
pub enum UndoOp {
    SetParent      { fi: u8, node: NodeId, old: NodeId },
    SetLeft        { fi: u8, node: NodeId, old: NodeId },
    SetRight       { fi: u8, node: NodeId, old: NodeId },
    SetLabel       { fi: u8, node: NodeId, old: u32 },
    SetLabelToNode { fi: u8, label: u32, old: NodeId },
    SetTwin        { fi: u8, node: NodeId, old: NodeId },
    SetCollapsed   { fi: u8, label: u32, old: u32 },
    AddComponent   { fi: u8 },                          // undo: pop
    ReplaceComponent { fi: u8, idx: usize, old: NodeId }, // undo: restore old at idx
}

pub struct UndoMachine {
    ops: Vec<UndoOp>,
}

impl UndoMachine {
    pub fn new() -> Self {
        Self { ops: Vec::with_capacity(1024) }
    }

    #[inline(always)]
    pub fn checkpoint(&self) -> usize { self.ops.len() }

    #[inline(always)]
    pub fn push(&mut self, op: UndoOp) { self.ops.push(op); }

    pub fn undo_to(&mut self, cp: usize, forests: &mut [Forest; 2]) {
        while self.ops.len() > cp {
            match self.ops.pop().unwrap() {
                UndoOp::SetParent { fi, node, old } =>
                    forests[fi as usize].parent[node as usize] = old,
                UndoOp::SetLeft { fi, node, old } =>
                    forests[fi as usize].left[node as usize] = old,
                UndoOp::SetRight { fi, node, old } =>
                    forests[fi as usize].right[node as usize] = old,
                UndoOp::SetLabel { fi, node, old } =>
                    forests[fi as usize].label[node as usize] = old,
                UndoOp::SetLabelToNode { fi, label, old } =>
                    forests[fi as usize].label_to_node[label as usize] = old,
                UndoOp::SetTwin { fi, node, old } =>
                    forests[fi as usize].twin[node as usize] = old,
                UndoOp::SetCollapsed { fi, label, old } =>
                    forests[fi as usize].collapsed_into[label as usize] = old,
                UndoOp::AddComponent { fi } => {
                    forests[fi as usize].components.pop();
                }
                UndoOp::ReplaceComponent { fi, idx, old } => {
                    forests[fi as usize].components[idx] = old;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Physical tree operations (matching rspr)
// ---------------------------------------------------------------------------

/// Cut node from its parent (rspr's `cut_parent()`).
/// Does NOT add to components — caller must do that.
pub fn cut_parent(f: &mut Forest, fi: u8, node: NodeId, um: &mut UndoMachine) {
    let p = f.parent[node as usize];
    if p == NONE { return; }

    // Remove from parent's children
    if f.left[p as usize] == node {
        um.push(UndoOp::SetLeft { fi, node: p, old: f.left[p as usize] });
        f.left[p as usize] = NONE;
    } else {
        debug_assert_eq!(f.right[p as usize], node);
        um.push(UndoOp::SetRight { fi, node: p, old: f.right[p as usize] });
        f.right[p as usize] = NONE;
    }

    // Detach
    um.push(UndoOp::SetParent { fi, node, old: p });
    f.parent[node as usize] = NONE;
}

/// Add node as a forest component.
pub fn add_component(f: &mut Forest, fi: u8, node: NodeId, um: &mut UndoMachine) {
    f.components.push(node);
    um.push(UndoOp::AddComponent { fi });
}

/// Contract degree-1 node: splice it out, child takes its place.
/// If node has 0 children and is component root, leave it (dead component).
/// Returns the node that "replaced" it (for sibling-pair checks).
///
/// Matches rspr's `Node::contract()` for binary trees.
pub fn contract(f: &mut Forest, fi: u8, node: NodeId, um: &mut UndoMachine) -> NodeId {
    let nc = f.num_children(node);
    if nc != 1 {
        return node; // nothing to splice (0 or 2 children)
    }

    let child = f.only_child(node);
    let gp = f.parent[node as usize];

    if gp != NONE {
        // Splice: replace node with child in grandparent
        if f.left[gp as usize] == node {
            um.push(UndoOp::SetLeft { fi, node: gp, old: f.left[gp as usize] });
            f.left[gp as usize] = child;
        } else {
            um.push(UndoOp::SetRight { fi, node: gp, old: f.right[gp as usize] });
            f.right[gp as usize] = child;
        }
        um.push(UndoOp::SetParent { fi, node: child, old: f.parent[child as usize] });
        f.parent[child as usize] = gp;
    } else {
        // Node is a component root — child becomes root
        um.push(UndoOp::SetParent { fi, node: child, old: f.parent[child as usize] });
        f.parent[child as usize] = NONE;

        // Replace in components list
        if let Some(idx) = f.components.iter().position(|&c| c == node) {
            um.push(UndoOp::ReplaceComponent { fi, idx, old: node });
            f.components[idx] = child;
        }
    }

    // Disconnect node itself
    um.push(UndoOp::SetParent { fi, node, old: f.parent[node as usize] });
    f.parent[node as usize] = NONE;
    if f.left[node as usize] != NONE {
        um.push(UndoOp::SetLeft { fi, node, old: f.left[node as usize] });
        f.left[node as usize] = NONE;
    }
    if f.right[node as usize] != NONE {
        um.push(UndoOp::SetRight { fi, node, old: f.right[node as usize] });
        f.right[node as usize] = NONE;
    }

    // Recurse: grandparent might now be degree-1 too
    if gp != NONE {
        contract(f, fi, gp, um)
    } else {
        child
    }
}

/// Contract a sibling pair: detach both leaf children from parent.
/// Parent becomes a leaf (is_leaf == true). Called on the parent node.
///
/// Matches rspr's `contract_sibling_pair_undoable()`.
pub fn contract_sibling_pair(f: &mut Forest, fi: u8, parent: NodeId, um: &mut UndoMachine) {
    let lc = f.left[parent as usize];
    let rc = f.right[parent as usize];
    debug_assert!(lc != NONE && rc != NONE, "need 2 children");
    debug_assert!(f.is_leaf(lc) && f.is_leaf(rc), "children must be leaves");

    // Detach right child
    um.push(UndoOp::SetRight { fi, node: parent, old: rc });
    f.right[parent as usize] = NONE;
    um.push(UndoOp::SetParent { fi, node: rc, old: parent });
    f.parent[rc as usize] = NONE;

    // Detach left child
    um.push(UndoOp::SetLeft { fi, node: parent, old: lc });
    f.left[parent as usize] = NONE;
    um.push(UndoOp::SetParent { fi, node: lc, old: parent });
    f.parent[lc as usize] = NONE;
}

/// Set twin pointer with undo.
pub fn set_twin(f: &mut Forest, fi: u8, node: NodeId, twin: NodeId, um: &mut UndoMachine) {
    um.push(UndoOp::SetTwin { fi, node, old: f.twin[node as usize] });
    f.twin[node as usize] = twin;
}

/// Set label with undo.
pub fn set_label(f: &mut Forest, fi: u8, node: NodeId, label: u32, um: &mut UndoMachine) {
    um.push(UndoOp::SetLabel { fi, node, old: f.label[node as usize] });
    f.label[node as usize] = label;
}

/// Set collapsed_into with undo.
pub fn set_collapsed(f: &mut Forest, fi: u8, label: u32, target: u32, um: &mut UndoMachine) {
    um.push(UndoOp::SetCollapsed { fi, label, old: f.collapsed_into[label as usize] });
    f.collapsed_into[label as usize] = target;
}

/// Set label_to_node with undo.
pub fn set_label_to_node(f: &mut Forest, fi: u8, label: u32, node: NodeId, um: &mut UndoMachine) {
    um.push(UndoOp::SetLabelToNode { fi, label, old: f.label_to_node[label as usize] });
    f.label_to_node[label as usize] = node;
}
