//! Undo machine for branch-and-bound backtracking.
//!
//! Every physical mutation to the twin forest is recorded as an UndoOp.
//! `undo_to(checkpoint)` replays ops in reverse to restore state.
//!
//! UndoOp is 12 bytes (idx: u16 keeps ReplaceComponent small).

use klados_core::tree::{NodeId, NONE};
use super::forest::TwinForest;

#[derive(Clone, Copy)]
pub enum UndoOp {
    SetParent      { ti: u8, node: NodeId, old: NodeId },
    SetLeft        { ti: u8, node: NodeId, old: NodeId },
    SetRight       { ti: u8, node: NodeId, old: NodeId },
    SetLabel       { ti: u8, node: NodeId, old: u32 },
    SetLabelToNode { ti: u8, label: u32, old: NodeId },
    SetTwin        { ti: u8, node: NodeId, old: NodeId },
    SetCollapsed   { label: u32, old: u32 },
    AddComponent   { ti: u8 },
    ReplaceComponent { ti: u8, idx: u16, old: NodeId }, // u16: components < 65536
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

    pub fn undo_to(&mut self, cp: usize, tf: &mut TwinForest) {
        while self.ops.len() > cp {
            match self.ops.pop().unwrap() {
                UndoOp::SetParent { ti, node, old } =>
                    tf.parent[ti as usize][node as usize] = old,
                UndoOp::SetLeft { ti, node, old } =>
                    tf.left[ti as usize][node as usize] = old,
                UndoOp::SetRight { ti, node, old } =>
                    tf.right[ti as usize][node as usize] = old,
                UndoOp::SetLabel { ti, node, old } =>
                    tf.label[ti as usize][node as usize] = old,
                UndoOp::SetLabelToNode { ti, label, old } =>
                    tf.label_to_node[ti as usize][label as usize] = old,
                UndoOp::SetTwin { ti, node, old } =>
                    tf.twin[ti as usize][node as usize] = old,
                UndoOp::SetCollapsed { label, old } =>
                    tf.collapsed_into[label as usize] = old,
                UndoOp::AddComponent { ti } => {
                    tf.components[ti as usize].pop();
                }
                UndoOp::ReplaceComponent { ti, idx, old } => {
                    tf.components[ti as usize][idx as usize] = old;
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
#[inline]
pub fn cut_parent(tf: &mut TwinForest, ti: usize, node: NodeId, um: &mut UndoMachine) {
    let p = tf.parent[ti][node as usize];
    if p == NONE { return; }

    // Remove from parent's children
    if tf.left[ti][p as usize] == node {
        um.push(UndoOp::SetLeft { ti: ti as u8, node: p, old: node });
        tf.left[ti][p as usize] = NONE;
    } else {
        debug_assert_eq!(tf.right[ti][p as usize], node);
        um.push(UndoOp::SetRight { ti: ti as u8, node: p, old: node });
        tf.right[ti][p as usize] = NONE;
    }

    // Detach
    um.push(UndoOp::SetParent { ti: ti as u8, node, old: p });
    tf.parent[ti][node as usize] = NONE;
}

/// Add node as a forest component.
#[inline]
pub fn add_component(tf: &mut TwinForest, ti: usize, node: NodeId, um: &mut UndoMachine) {
    tf.components[ti].push(node);
    um.push(UndoOp::AddComponent { ti: ti as u8 });
}

/// Contract degree-1 node: splice it out, child takes its place.
/// Iterates up the tree while ancestors remain degree-1.
/// Returns the final node that ended the chain (the surviving replacement).
///
/// Matches rspr's `Node::contract()` for binary trees.
/// Dead node's own pointers are NOT zeroed — callers walk from component
/// roots, so dead nodes are unreachable. This saves 1-3 undo ops per contract.
pub fn contract(tf: &mut TwinForest, ti: usize, mut node: NodeId, um: &mut UndoMachine) -> NodeId {
    let ti8 = ti as u8;

    loop {
        let nc = tf.num_children(ti, node);
        if nc != 1 {
            return node; // nothing to splice (0 or 2 children)
        }

        let child = tf.only_child(ti, node);
        let gp = tf.parent[ti][node as usize];

        if gp != NONE {
            // Splice: replace node with child in grandparent
            if tf.left[ti][gp as usize] == node {
                um.push(UndoOp::SetLeft { ti: ti8, node: gp, old: node });
                tf.left[ti][gp as usize] = child;
            } else {
                um.push(UndoOp::SetRight { ti: ti8, node: gp, old: node });
                tf.right[ti][gp as usize] = child;
            }
            um.push(UndoOp::SetParent { ti: ti8, node: child, old: node });
            tf.parent[ti][child as usize] = gp;

            // Continue: grandparent might now be degree-1 too
            node = gp;
        } else {
            // Node is a component root — child becomes root
            um.push(UndoOp::SetParent { ti: ti8, node: child, old: node });
            tf.parent[ti][child as usize] = NONE;

            if let Some(idx) = tf.components[ti].iter().position(|&c| c == node) {
                um.push(UndoOp::ReplaceComponent { ti: ti8, idx: idx as u16, old: node });
                tf.components[ti][idx] = child;
            }
            return child;
        }
    }
}

/// Contract a sibling pair: detach both leaf children from parent.
/// Parent becomes a leaf (is_leaf == true). Called on the parent node.
///
/// Matches rspr's `contract_sibling_pair_undoable()`.
#[inline]
pub fn contract_sibling_pair(tf: &mut TwinForest, ti: usize, parent: NodeId, um: &mut UndoMachine) {
    let lc = tf.left[ti][parent as usize];
    let rc = tf.right[ti][parent as usize];
    debug_assert!(lc != NONE && rc != NONE, "need 2 children");
    debug_assert!(tf.is_leaf(ti, lc) && tf.is_leaf(ti, rc), "children must be leaves");

    let ti8 = ti as u8;

    // Detach right child
    um.push(UndoOp::SetRight { ti: ti8, node: parent, old: rc });
    tf.right[ti][parent as usize] = NONE;
    um.push(UndoOp::SetParent { ti: ti8, node: rc, old: parent });
    tf.parent[ti][rc as usize] = NONE;

    // Detach left child
    um.push(UndoOp::SetLeft { ti: ti8, node: parent, old: lc });
    tf.left[ti][parent as usize] = NONE;
    um.push(UndoOp::SetParent { ti: ti8, node: lc, old: parent });
    tf.parent[ti][lc as usize] = NONE;
}

/// Set twin pointer with undo.
#[inline]
pub fn set_twin(tf: &mut TwinForest, ti: usize, node: NodeId, twin: NodeId, um: &mut UndoMachine) {
    um.push(UndoOp::SetTwin { ti: ti as u8, node, old: tf.twin[ti][node as usize] });
    tf.twin[ti][node as usize] = twin;
}

/// Set label with undo.
#[inline]
pub fn set_label(tf: &mut TwinForest, ti: usize, node: NodeId, label: u32, um: &mut UndoMachine) {
    um.push(UndoOp::SetLabel { ti: ti as u8, node, old: tf.label[ti][node as usize] });
    tf.label[ti][node as usize] = label;
}

/// Set collapsed_into with undo (T1 only).
#[inline]
pub fn set_collapsed(tf: &mut TwinForest, label: u32, target: u32, um: &mut UndoMachine) {
    um.push(UndoOp::SetCollapsed { label, old: tf.collapsed_into[label as usize] });
    tf.collapsed_into[label as usize] = target;
}

/// Set label_to_node with undo.
#[inline]
pub fn set_label_to_node(tf: &mut TwinForest, ti: usize, label: u32, node: NodeId, um: &mut UndoMachine) {
    um.push(UndoOp::SetLabelToNode { ti: ti as u8, label, old: tf.label_to_node[ti][label as usize] });
    tf.label_to_node[ti][label as usize] = node;
}
