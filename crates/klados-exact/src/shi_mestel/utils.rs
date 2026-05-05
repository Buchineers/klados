//! Utility functions for bitset operations and hashing.

use fixedbitset::FixedBitSet;
use klados_core::{NONE, NodeId, Tree};

pub fn has_intersection(a: &FixedBitSet, b: &FixedBitSet) -> bool {
    let a_sl = a.as_slice();
    let b_sl = b.as_slice();
    let len = a_sl.len().min(b_sl.len());
    for i in 0..len {
        if a_sl[i] & b_sl[i] != 0 {
            return true;
        }
    }
    false
}

pub fn is_subset(a: &FixedBitSet, b: &FixedBitSet) -> bool {
    let a_sl = a.as_slice();
    let b_sl = b.as_slice();
    for i in 0..a_sl.len() {
        let b_word = if i < b_sl.len() { b_sl[i] } else { 0 };
        if a_sl[i] & !b_word != 0 {
            return false;
        }
    }
    true
}

pub fn count_intersection(a: &FixedBitSet, b: &FixedBitSet) -> usize {
    let a_sl = a.as_slice();
    let b_sl = b.as_slice();
    let len = a_sl.len().min(b_sl.len());
    let mut total = 0usize;
    for i in 0..len {
        total += (a_sl[i] & b_sl[i]).count_ones() as usize;
    }
    total
}

pub fn hash_bitset(s: &FixedBitSet) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &word in s.as_slice() {
        h ^= word as u64;
        h = h.wrapping_mul(0x100000001b3);
        h ^= (word >> 32) as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn sorted_partition_hashes(components: &[FixedBitSet]) -> Vec<u64> {
    let mut hashes: Vec<u64> = components.iter().map(hash_bitset).collect();
    hashes.sort_unstable();
    hashes
}

pub fn tree_canonical_for_labels(tree: &Tree, labels: &FixedBitSet) -> u64 {
    fn build(tree: &Tree, node: NodeId, labels: &FixedBitSet) -> Option<u64> {
        if tree.is_leaf(node) {
            let lbl = tree.label[node as usize];
            return if labels.contains(lbl as usize) {
                let mut h = lbl as u64;
                h = h.wrapping_mul(0x9e3779b97f4a7c15);
                h ^= h >> 30;
                Some(h)
            } else {
                None
            };
        }
        if let Some((l, r)) = tree.children(node) {
            let left = build(tree, l, labels);
            let right = build(tree, r, labels);
            match (left, right) {
                (Some(a), Some(b)) => {
                    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                    let mut h = lo;
                    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
                    h ^= hi;
                    h = h.wrapping_mul(0x94d049bb133111eb);
                    h ^= h >> 31;
                    Some(h)
                }
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            }
        } else {
            None
        }
    }
    if tree.root == NONE {
        0
    } else {
        build(tree, tree.root, labels).unwrap_or(0)
    }
}

pub fn make_singleton_tree(lbl: u32, num_leaves: u32) -> Tree {
    Tree::singleton(lbl, num_leaves)
}

pub fn trivial_forest(reference: &Tree, num_leaves: u32) -> Vec<Tree> {
    let mut components = Vec::new();
    for lbl in 1..=num_leaves {
        components.push(make_singleton_tree(lbl, num_leaves));
    }
    if components.is_empty() && reference.root != NONE {
        components.push(reference.clone());
    }
    components
}
