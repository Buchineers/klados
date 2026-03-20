//! Tree helper functions used by reduction rules.

use crate::Tree;

pub fn num_leaf_children(tree: &Tree, node: u32) -> u32 {
    if let Some((l, r)) = tree.children(node) {
        let mut count = 0;
        if tree.is_leaf(l) {
            count += 1;
        }
        if tree.is_leaf(r) {
            count += 1;
        }
        count
    } else {
        0
    }
}

pub fn non_leaf_child(tree: &Tree, node: u32) -> Option<u32> {
    if let Some((l, r)) = tree.children(node) {
        if !tree.is_leaf(l) {
            Some(l)
        } else if !tree.is_leaf(r) {
            Some(r)
        } else {
            None
        }
    } else {
        None
    }
}

pub fn leaf_child(tree: &Tree, node: u32) -> Option<u32> {
    if let Some((l, r)) = tree.children(node) {
        if tree.is_leaf(l) {
            Some(l)
        } else if tree.is_leaf(r) {
            Some(r)
        } else {
            None
        }
    } else {
        None
    }
}

pub fn has_leaf_child_with_label(tree: &Tree, node: u32, label: u32) -> bool {
    if let Some((l, r)) = tree.children(node) {
        (tree.is_leaf(l) && tree.label[l as usize] == label)
            || (tree.is_leaf(r) && tree.label[r as usize] == label)
    } else {
        false
    }
}
