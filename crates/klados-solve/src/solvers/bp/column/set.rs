//! Dedup set for column labelsets. Opaque so we can swap the underlying
//! representation later (FxHashSet now; could become a Bloom prefilter +
//! fingerprint table if profiling demands).

use fxhash::FxHashSet;

#[derive(Default)]
pub struct ColumnSet {
    seen: FxHashSet<Vec<u32>>,
}

impl ColumnSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if `labels` was newly inserted.
    pub fn insert(&mut self, labels: Vec<u32>) -> bool {
        self.seen.insert(labels)
    }

    pub fn contains(&self, labels: &[u32]) -> bool {
        self.seen.contains(labels)
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}
