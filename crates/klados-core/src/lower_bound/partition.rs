//! Partition of labels into components.

use crate::tree::Label;
use fixedbitset::FixedBitSet;

pub struct Partition {
    pub comp: Vec<u32>,
    pub members: Vec<FixedBitSet>,
    pub n: u32,
}

impl Partition {
    pub fn new_single(n: u32) -> Self {
        let mut all = FixedBitSet::with_capacity(n as usize + 1);
        for i in 1..=n as usize {
            all.insert(i);
        }
        Partition {
            comp: vec![0; n as usize + 1],
            members: vec![all],
            n,
        }
    }

    #[inline]
    pub fn component_of(&self, label: Label) -> u32 {
        self.comp[label as usize]
    }

    pub fn split_off(&mut self, subset: &FixedBitSet) -> u32 {
        if subset.count_ones(..) == 0 {
            return u32::MAX;
        }
        let new_id = self.members.len() as u32;
        let mut new_set = FixedBitSet::with_capacity(self.n as usize + 1);

        let old_id = self.comp[subset.ones().next().unwrap()];

        for lbl in subset.ones() {
            debug_assert_eq!(self.comp[lbl], old_id);
            self.comp[lbl] = new_id;
            self.members[old_id as usize].set(lbl, false);
            new_set.insert(lbl);
        }
        self.members.push(new_set);
        new_id
    }

    pub fn merge(&mut self, keep: u32, remove: u32) {
        if keep == remove {
            return;
        }
        let remove_members: Vec<usize> = self.members[remove as usize].ones().collect();
        for lbl in remove_members {
            self.comp[lbl] = keep;
            self.members[keep as usize].insert(lbl);
            self.members[remove as usize].set(lbl, false);
        }
    }

    pub fn count_components(&self) -> usize {
        self.members.iter().filter(|m| m.count_ones(..) > 0).count()
    }

    pub fn active_component_ids(&self) -> Vec<u32> {
        (0..self.members.len() as u32)
            .filter(|&id| self.members[id as usize].count_ones(..) > 0)
            .collect()
    }
}
