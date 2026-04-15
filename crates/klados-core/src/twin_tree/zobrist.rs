//! Zobrist hashing for TwinForest transposition table.
//!
//! Maintains a 64-bit incremental hash over the search-critical topology state:
//! parent/left/right arrays for both trees. Non-search fields (labels, twins,
//! collapsed_into, protected) are excluded — they don't affect feasibility.
//!
//! The hash is self-inverse under XOR: undo operations automatically restore it.

use crate::tree::NodeId;

/// Splitmix64 finalizer — mixes a u64 into a well-distributed hash.
#[inline(always)]
fn splitmix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

/// Per-atom Zobrist contribution: combines a position salt with a value.
/// Different (salt, value) pairs produce well-distributed, independent results.
#[inline(always)]
pub fn zobrist_atom(salt: u64, value: NodeId) -> u64 {
    // +1 offset so value=0 and value=NONE (u32::MAX) don't accidentally cancel
    splitmix64(salt ^ (value as u64).wrapping_add(1))
}

/// Field indices for the salt table.
/// 6 search-critical fields: parent/left/right × T1/T2.
const FIELD_PARENT_T1: usize = 0;
const FIELD_LEFT_T1: usize = 1;
const FIELD_RIGHT_T1: usize = 2;
const FIELD_PARENT_T2: usize = 3;
const FIELD_LEFT_T2: usize = 4;
const FIELD_RIGHT_T2: usize = 5;
const NUM_FIELDS: usize = 6;

/// Pre-computed random salts for Zobrist hashing.
///
/// `salts[field_idx][node_id]` is a random u64 assigned to that (field, node) pair.
/// Dynamically sized to match the actual tree node count.
#[derive(Clone)]
pub struct ZobristSalts {
    /// salts[node_id * NUM_FIELDS + field_idx]
    data: Vec<u64>,
    stride: usize, // == NUM_FIELDS
}

impl ZobristSalts {
    /// Generate salts for trees with up to `max_nodes` nodes.
    /// Uses a deterministic PRNG seeded from a fixed value for reproducibility.
    pub fn new(max_nodes: usize) -> Self {
        let total = max_nodes * NUM_FIELDS;
        let mut data = Vec::with_capacity(total);
        // Deterministic seed — chosen arbitrarily, just needs to be fixed.
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        for _ in 0..total {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            data.push(splitmix64(state));
        }
        Self {
            data,
            stride: NUM_FIELDS,
        }
    }

    #[inline(always)]
    fn salt(&self, field: usize, node: NodeId) -> u64 {
        self.data[(node as usize) * self.stride + field]
    }

    #[inline(always)]
    pub fn parent(&self, ti: usize, node: NodeId) -> u64 {
        self.salt(
            if ti == 0 {
                FIELD_PARENT_T1
            } else {
                FIELD_PARENT_T2
            },
            node,
        )
    }

    #[inline(always)]
    pub fn left(&self, ti: usize, node: NodeId) -> u64 {
        self.salt(
            if ti == 0 {
                FIELD_LEFT_T1
            } else {
                FIELD_LEFT_T2
            },
            node,
        )
    }

    #[inline(always)]
    pub fn right(&self, ti: usize, node: NodeId) -> u64 {
        self.salt(
            if ti == 0 {
                FIELD_RIGHT_T1
            } else {
                FIELD_RIGHT_T2
            },
            node,
        )
    }
}

/// Compute the full Zobrist hash from scratch for a TwinForest's topology arrays.
/// Used at initialization and for debug verification.
pub fn compute_full_hash(
    parent: &[Vec<NodeId>; 2],
    left: &[Vec<NodeId>; 2],
    right: &[Vec<NodeId>; 2],
    salts: &ZobristSalts,
) -> u64 {
    let mut h: u64 = 0;
    for ti in 0..2 {
        for node in 0..parent[ti].len() {
            let n = node as NodeId;
            h ^= zobrist_atom(salts.parent(ti, n), parent[ti][node]);
            h ^= zobrist_atom(salts.left(ti, n), left[ti][node]);
            h ^= zobrist_atom(salts.right(ti, n), right[ti][node]);
        }
    }
    h
}

/// Apply a single field update to the hash: XOR out old, XOR in new.
#[inline(always)]
pub fn hash_update(hash: &mut u64, salt: u64, old_value: NodeId, new_value: NodeId) {
    *hash ^= zobrist_atom(salt, old_value);
    *hash ^= zobrist_atom(salt, new_value);
}
