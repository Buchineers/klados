//! Cached-positive-column reuse for the m=2 leaf-pair-DP pricer.
//!
//! See `papers/ours/Pricing Skip Theorem - Exact Caching.md` for the
//! mathematical foundation. In practice the stability-gap certificate was
//! too loose/expensive for the current pricer, and the available gap was not
//! exact enough to support sound skips. The cache therefore uses only the
//! robust part of the idea: a cached column is re-scored exactly under the
//! current duals and emitted if it is still positive, unseen, and
//! branch-feasible. It does **not** certify optimality and it does **not**
//! skip non-improving anchors.
//!
//! Phase 1 scope: no tree-static precompute; the cache is just label-pair
//! indexing plus cached column supports.
//!
//! Gated by `BpConfig.use_anchor_cache`.
//!
//! ## Indexing
//!
//! Leaf pairs are indexed by **leaf labels** `(la, lb)` with `1 ≤ la <
//! lb ≤ num_leaves`, not by active-label position. Labels are
//! tree-invariant; positions shift between CG iterations when the
//! active set changes (e.g. duals drop a leaf below the threshold).
//! Label indexing makes the cache survive minor active-set changes
//! without rebuilding.
//!
//! ## Memory
//!
//! Memory is proportional only to cached positive columns; never-priced or
//! non-improving pairs occupy one `Option` slot.

use klados_core::Tree;

/// Cached entry per leaf-pair anchor.
#[derive(Clone)]
pub struct AnchorEntry {
    /// Optimum column's leaves, sorted ascending.
    pub column_leaves: Vec<u32>,
    /// Internal node IDs the column uses in tree 0.
    pub column_nodes_t0: Vec<u32>,
    /// Internal node IDs the column uses in tree 1.
    pub column_nodes_t1: Vec<u32>,
    /// Score of the cached optimum at compute time.
    pub score: f64,
    /// Diagnostic gap value from the producer. Not used for correctness.
    pub gap: f64,
}

/// Positive-column reuse cache for leaf-pair anchors.
pub struct AnchorCache {
    /// Number of leaves in the instance. Cache is sized for all
    /// `num_leaves²/2` possible label pairs; only used pairs occupy
    /// memory beyond a `None`.
    num_leaves: usize,
    /// True once `refresh_static` has initialized the entry table.
    built: bool,
    /// Total label pairs: `num_leaves * (num_leaves - 1) / 2`.
    n_pairs: usize,
    /// Per-pair cache entry, `None` if never priced or invalidated.
    entries: Vec<Option<AnchorEntry>>,
    // Stats (for monitoring).
    pub hits: u64,
    pub skips: u64,
    pub misses: u64,
    pub stales: u64,
    pub refreshes: u64,
    pub gap_zero_refreshes: u64,
    pub gap_positive_refreshes: u64,
    pub gap_sum: f64,
}

/// Result of trying to emit from the cache for one anchor.
///
/// This cache is intentionally a **positive-column reuse** cache, not a
/// convergence certificate. A cached column does not need to remain
/// anchor-optimal for column generation to make progress; it only needs to
/// be a valid column with positive reduced cost under the current duals.
/// If that cheap test fails, the caller must fall through to the DP.
pub enum CacheResult {
    /// Cached column has score > 1+ε under current duals. Emit it if the
    /// caller's seen/branch checks also accept it. Score is exact.
    Emit { score: f64 },
    /// Cache entry exists but the cached column is no longer improving.
    /// The DP must run to either confirm no improving column or find one.
    Stale,
    /// No cache entry. DP must run.
    Miss,
}

impl AnchorCache {
    /// Construct an empty cache for the given instance.
    pub fn new(num_leaves: usize) -> Self {
        Self {
            num_leaves,
            built: false,
            n_pairs: 0,
            entries: Vec::new(),
            hits: 0,
            skips: 0,
            misses: 0,
            stales: 0,
            refreshes: 0,
            gap_zero_refreshes: 0,
            gap_positive_refreshes: 0,
            gap_sum: 0.0,
        }
    }

    /// True once `refresh_static(trees)` has initialized label-pair storage.
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// Clear all entries (e.g. on B&P branch).
    pub fn clear(&mut self) {
        self.built = false;
        self.n_pairs = 0;
        self.entries.clear();
        // stats persist across clears for cumulative monitoring
    }

    /// Get a cache entry (if any) by label pair.
    pub fn entry_for(&self, la: u32, lb: u32) -> Option<&AnchorEntry> {
        let (la, lb) = if la < lb { (la, lb) } else { (lb, la) };
        let idx = Self::pair_idx_labels(la, lb, self.num_leaves);
        self.entries.get(idx).and_then(|x| x.as_ref())
    }

    /// Map label pair (la, lb) with 1 ≤ la < lb ≤ num_leaves to a flat
    /// index. The indexing covers all unordered pairs in the leaf
    /// universe (not just active labels).
    #[inline]
    pub fn pair_idx_labels(la: u32, lb: u32, num_leaves: usize) -> usize {
        debug_assert!(la >= 1 && lb >= 1);
        debug_assert!(la < lb);
        debug_assert!((lb as usize) <= num_leaves);
        // We use 0-based labels in the index calculation: la' = la-1,
        // lb' = lb-1. There are C(num_leaves, 2) total pairs.
        let i = (la - 1) as usize;
        let j = (lb - 1) as usize;
        let n = num_leaves;
        i * (2 * n - i - 1) / 2 + (j - i - 1)
    }

    /// Total number of label pairs.
    #[inline]
    pub fn num_pairs_for(num_leaves: usize) -> usize {
        num_leaves.saturating_mul(num_leaves.saturating_sub(1)) / 2
    }

    /// Initialize storage for all label pairs. This intentionally does not
    /// precompute `R(a,b)`/`V(a,b)`; positive-column reuse needs only a flat
    /// label-pair index and cached column supports.
    pub fn refresh_static(&mut self, trees: &[Tree]) {
        debug_assert_eq!(trees.len(), 2);

        let n_pairs = Self::num_pairs_for(self.num_leaves);
        self.n_pairs = n_pairs;
        self.built = true;

        self.entries.clear();
        self.entries.resize(n_pairs, None);
    }

    /// Try to serve anchor `(la, lb)` (leaf labels, `la < lb`) from
    /// the cache.
    ///
    /// `alpha` is indexed by label id (1..=num_leaves).
    /// `beta_t0`, `beta_t1` are indexed by node id within each tree.
    /// `epsilon` is the pricing tolerance for the improving threshold.
    pub fn try_emit(
        &mut self,
        la: u32,
        lb: u32,
        alpha: &[f64],
        beta_t0: &[f64],
        beta_t1: &[f64],
        epsilon: f64,
    ) -> CacheResult {
        if !self.built {
            return CacheResult::Miss;
        }
        let (la, lb) = if la < lb { (la, lb) } else { (lb, la) };
        let idx = Self::pair_idx_labels(la, lb, self.num_leaves);
        let Some(entry) = self.entries[idx].as_ref() else {
            self.misses += 1;
            return CacheResult::Miss;
        };

        let new_score = exact_current_score(entry, alpha, beta_t0, beta_t1);
        if new_score > 1.0 + epsilon {
            self.hits += 1;
            CacheResult::Emit { score: new_score }
        } else {
            self.stales += 1;
            CacheResult::Stale
        }
    }

    /// Insert or replace a cache entry for label pair `(la, lb)`.
    ///
    /// Caller provides the cached column (leaves, node sets) and a
    /// diagnostic gap-like value. Dual slices are accepted to keep the call
    /// site stable but are not used by positive-column reuse.
    pub fn refresh(
        &mut self,
        la: u32,
        lb: u32,
        column_leaves: Vec<u32>,
        column_nodes_t0: Vec<u32>,
        column_nodes_t1: Vec<u32>,
        score: f64,
        gap: f64,
        _alpha: &[f64],
        _beta_t0: &[f64],
        _beta_t1: &[f64],
    ) {
        let (la, lb) = if la < lb { (la, lb) } else { (lb, la) };
        let idx = Self::pair_idx_labels(la, lb, self.num_leaves);

        self.refreshes += 1;
        let gap = gap.max(0.0);
        if gap <= 1.0e-9 {
            self.gap_zero_refreshes += 1;
        } else {
            self.gap_positive_refreshes += 1;
            self.gap_sum += gap;
        }
        self.entries[idx] = Some(AnchorEntry {
            column_leaves,
            column_nodes_t0,
            column_nodes_t1,
            score,
            gap,
        });
    }
}

/// Exact current reduced-cost score for a cached column. No snapshots, no
/// stability assumptions, and no f32 arithmetic are involved.
fn exact_current_score(
    entry: &AnchorEntry,
    alpha: &[f64],
    beta_t0: &[f64],
    beta_t1: &[f64],
) -> f64 {
    let leaf_gain: f64 = entry.column_leaves.iter().map(|&l| alpha[l as usize]).sum();
    let node_penalty_t0: f64 = entry
        .column_nodes_t0
        .iter()
        .map(|&v| beta_t0[v as usize])
        .sum();
    let node_penalty_t1: f64 = entry
        .column_nodes_t1
        .iter()
        .map(|&v| beta_t1[v as usize])
        .sum();
    leaf_gain - node_penalty_t0 - node_penalty_t1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_idx_roundtrip() {
        // Verify pair_idx_labels is injective and consistent.
        let n = 10;
        let n_pairs = AnchorCache::num_pairs_for(n);
        let mut seen = vec![false; n_pairs];
        for la in 1..=n as u32 {
            for lb in (la + 1)..=n as u32 {
                let idx = AnchorCache::pair_idx_labels(la, lb, n);
                assert!(!seen[idx], "duplicate index {} for ({}, {})", idx, la, lb);
                seen[idx] = true;
            }
        }
        assert!(seen.iter().all(|&b| b), "all indices covered");
    }
}
