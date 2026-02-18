//! Transposition table using Zobrist hashing for state deduplication.

use fixedbitset::FixedBitSet;
use fxhash::FxHashMap;


pub const TT_MAX_ENTRIES: usize = 1 << 22;

#[derive(Clone, Copy)]
pub struct TTEntry {
    pub infeasible_at: usize,
}

pub struct ZobristTable {
    label_keys: Vec<u64>,
}

impl ZobristTable {
    pub fn new(num_labels: usize) -> Self {
        let mut state: u64 = 0xdeadbeef12345678;
        let mut label_keys = Vec::with_capacity(num_labels + 1);
        for _ in 0..=num_labels {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            label_keys.push(z);
        }
        Self { label_keys }
    }

    pub fn hash_partition(&self, components: &[FixedBitSet]) -> u64 {
        let mut h: u64 = 0;
        for comp in components {
            let mut comp_h: u64 = 0;
            for lbl in comp.ones() {
                if lbl < self.label_keys.len() {
                    comp_h ^= self.label_keys[lbl];
                }
            }
            comp_h = comp_h.wrapping_mul(0x517cc1b727220a95);
            comp_h ^= comp_h >> 32;
            h ^= comp_h;
        }
        h = h.wrapping_mul(0x2545f4914f6cdd1d);
        h ^= h >> 32;
        h
    }
}

pub fn tt_insert(tt: &mut FxHashMap<u64, TTEntry>, hash: u64, target_s: usize) {
    if let Some(entry) = tt.get_mut(&hash) {
        if target_s > entry.infeasible_at {
            entry.infeasible_at = target_s;
        }
    } else if tt.len() < TT_MAX_ENTRIES {
        tt.insert(
            hash,
            TTEntry {
                infeasible_at: target_s,
            },
        );
    }
}
