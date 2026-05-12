//! Column representation and validity-by-construction.
//!
//! [`AfColumn`] is the type-level guarantee that every column reaching the LP
//! is a valid agreement-forest component (its restricted shape is the same in
//! every tree). The fields are private and the constructor is `pub(super)`,
//! so only code inside the `bp` module can build one. External code goes
//! through [`ColumnBuilder::try_build`] which validates the leafset.
//!
//! Pricers within `bp` may use the unchecked path because they construct
//! validity by design — e.g., a pair-DP only emits states that correspond to
//! consistent topologies across trees.

pub mod coverage;
pub mod set;

pub use coverage::{ColumnCoverage, Scratch};
pub use set::ColumnSet;

use klados_core::Tree;

/// A leafset that is guaranteed to be a valid AF component for the trees
/// passed to its constructor.
#[derive(Clone, Debug)]
pub struct AfColumn {
    labels: Vec<u32>,
    coverage: ColumnCoverage,
}

impl AfColumn {
    /// Sorted, deduplicated leaf labels.
    pub fn labels(&self) -> &[u32] {
        &self.labels
    }

    pub fn coverage(&self) -> &ColumnCoverage {
        &self.coverage
    }

    pub fn size(&self) -> usize {
        self.labels.len()
    }

    /// Reduced cost relative to LP duals: `Σ α_l − Σ β_{t,v}`.
    /// The pricer adds a column iff this exceeds `1 + ε`.
    pub fn pricing_score(&self, alpha: &[f64], beta: &[Vec<f64>]) -> f64 {
        let leaf_gain: f64 = self.labels.iter().map(|&l| alpha[l as usize]).sum();
        let node_penalty: f64 = self
            .coverage
            .iter_per_tree()
            .enumerate()
            .map(|(ti, nodes)| nodes.iter().map(|&v| beta[ti][v]).sum::<f64>())
            .sum();
        leaf_gain - node_penalty
    }

    /// Module-internal constructor. Caller is responsible for ensuring
    /// `labels` are sorted+deduped and that the leafset is a valid AF
    /// component (or accepts that as a precondition documented at the call
    /// site, e.g., "any |L|≤2 leafset is trivially valid").
    pub(super) fn from_parts(labels: Vec<u32>, coverage: ColumnCoverage) -> Self {
        Self { labels, coverage }
    }
}

/// External entry point for constructing columns. Validates the leafset's AF
/// membership and only returns `Some(...)` on success.
///
/// The builder owns reusable scratch memory but does **not** borrow trees —
/// trees are passed at each call. This lets a single builder be shared
/// across multiple pricers in [`crate::bp::pricer::PricerScratch`].
pub struct ColumnBuilder {
    scratch: Scratch,
    num_trees: usize,
}

impl ColumnBuilder {
    pub fn new(trees: &[Tree]) -> Self {
        Self {
            scratch: Scratch::new(trees),
            num_trees: trees.len(),
        }
    }

    pub fn try_build(&mut self, mut labels: Vec<u32>, trees: &[Tree]) -> Option<AfColumn> {
        labels.sort_unstable();
        labels.dedup();
        self.try_build_with_violation(labels, trees).ok()
    }

    pub fn try_build_with_violation(
        &mut self,
        mut labels: Vec<u32>,
        trees: &[Tree],
    ) -> Result<AfColumn, ViolatingTriplet> {
        labels.sort_unstable();
        labels.dedup();
        match validate_component_with_triplet(&labels, trees) {
            ComponentValidation::Valid => Ok(self.build_unchecked(labels, trees)),
            ComponentValidation::Invalid(v) => Err(v),
        }
    }

    /// Construct without validation. Available to bp internals only via
    /// `pub(super)`. Use only when validity is guaranteed by construction.
    pub(super) fn build_unchecked(&mut self, mut labels: Vec<u32>, trees: &[Tree]) -> AfColumn {
        debug_assert_eq!(trees.len(), self.num_trees);
        labels.sort_unstable();
        labels.dedup();
        let coverage = if labels.len() < 2 {
            ColumnCoverage::empty(trees.len())
        } else {
            ColumnCoverage::from_marker_paths(&labels, trees, &mut self.scratch)
        };
        AfColumn::from_parts(labels, coverage)
    }
}

/// True if the leafset induces the same rooted topology in every tree.
///
/// For `|L| ≤ 2` or single-tree instances this is trivially true; otherwise
/// every leaf triplet must have the same outgroup in every tree (rooted
/// triplets uniquely determine a rooted binary tree's topology).
pub fn is_valid_af_component(labels: &[u32], trees: &[Tree]) -> bool {
    matches!(
        validate_component_with_triplet(labels, trees),
        ComponentValidation::Valid
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViolatingTriplet {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub ref_tree: usize,
    pub other_tree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentValidation {
    Valid,
    Invalid(ViolatingTriplet),
}

/// Validate a component and return the first discordant triplet if it fails.
///
/// This intentionally starts with the simple rooted-triplet characterization
/// used by [`is_valid_af_component`]. It is asymptotically heavier than the
/// future induced-topology walk, but it is exact, deterministic, and gives the
/// DSSR pricer the concrete triplet needed to cut off an invalid relaxed DP
/// state.
pub fn validate_component_with_triplet(labels: &[u32], trees: &[Tree]) -> ComponentValidation {
    if labels.len() <= 2 || trees.len() <= 1 {
        return ComponentValidation::Valid;
    }
    let n = labels.len();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let (a, b, c) = (labels[i], labels[j], labels[k]);
                let og0 = triplet_outgroup(&trees[0], a, b, c);
                for (ti, tree) in trees.iter().enumerate().skip(1) {
                    let og = triplet_outgroup(tree, a, b, c);
                    if og != og0 {
                        return ComponentValidation::Invalid(ViolatingTriplet {
                            a,
                            b,
                            c,
                            ref_tree: 0,
                            other_tree: ti,
                        });
                    }
                }
            }
        }
    }
    ComponentValidation::Valid
}

fn triplet_outgroup(tree: &Tree, a: u32, b: u32, c: u32) -> u32 {
    let na = tree.node_by_label(a);
    let nb = tree.node_by_label(b);
    let nc = tree.node_by_label(c);
    let nab = tree.nearest_common_ancestor(na, nb);
    let nac = tree.nearest_common_ancestor(na, nc);
    let nbc = tree.nearest_common_ancestor(nb, nc);
    if nab == nac {
        a
    } else if nab == nbc {
        b
    } else {
        c
    }
}
