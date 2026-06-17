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

use klados_core::{NONE, Tree};

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
/// across multiple pricers in [`crate::solvers::bp::pricer::PricerScratch`].
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
/// Fast path: build the canonical topology of the restricted tree `T|labels`
/// in each input tree and compare those strings. This avoids the old
/// `O(m·|L|³)` all-triplets scan for valid components. If a mismatch is
/// detected, we then do a focused triplet scan between `T₀` and the first
/// disagreeing tree to produce the concrete triplet DSSR needs.
pub fn validate_component_with_triplet(labels: &[u32], trees: &[Tree]) -> ComponentValidation {
    if labels.len() <= 2 || trees.len() <= 1 {
        return ComponentValidation::Valid;
    }
    // For the overwhelmingly common tiny components (pairs/triples/quartets),
    // the direct triplet test is faster than constructing induced signatures.
    // Use the signature path only when it can amortize its setup cost.
    if labels.len() <= 8 {
        return validate_component_by_triplets(labels, trees);
    }
    let sig0 = induced_signature(labels, &trees[0]);
    for (ti, tree) in trees.iter().enumerate().skip(1) {
        let sig = induced_signature(labels, tree);
        if sig != sig0 {
            return ComponentValidation::Invalid(
                find_violating_triplet_between(labels, &trees[0], tree, ti).unwrap_or(
                    ViolatingTriplet {
                        a: labels[0],
                        b: labels[1],
                        c: labels[2],
                        ref_tree: 0,
                        other_tree: ti,
                    },
                ),
            );
        }
    }
    ComponentValidation::Valid
}

fn validate_component_by_triplets(labels: &[u32], trees: &[Tree]) -> ComponentValidation {
    let n = labels.len();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let (a, b, c) = (labels[i], labels[j], labels[k]);
                let og0 = triplet_outgroup(&trees[0], a, b, c);
                for (ti, tree) in trees.iter().enumerate().skip(1) {
                    if triplet_outgroup(tree, a, b, c) != og0 {
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

fn induced_signature(labels: &[u32], tree: &Tree) -> String {
    debug_assert!(!labels.is_empty());

    if labels.len() == 1 {
        return labels[0].to_string();
    }

    let mut lca = tree.node_by_label(labels[0]);
    for &lbl in &labels[1..] {
        lca = tree.nearest_common_ancestor(lca, tree.node_by_label(lbl));
    }

    induced_signature_rec(tree, lca, labels)
}

fn induced_signature_rec(tree: &Tree, lca: u32, labels: &[u32]) -> String {
    if labels.len() == 1 {
        return labels[0].to_string();
    }

    if tree.is_leaf(lca) {
        return tree.label[lca as usize].to_string();
    }

    let (left_child, right_child) = tree.children_pair(lca);
    let mut left_labels = Vec::new();
    let mut right_labels = Vec::new();

    for &lbl in labels {
        let side = child_below(tree, lca, lbl);
        if side == left_child {
            left_labels.push(lbl);
        } else if side == right_child {
            right_labels.push(lbl);
        }
    }

    match (!left_labels.is_empty(), !right_labels.is_empty()) {
        (true, true) => {
            let ls = induced_signature(&left_labels, tree);
            let rs = induced_signature(&right_labels, tree);
            let (a, b) = if ls <= rs { (ls, rs) } else { (rs, ls) };
            let mut out = String::with_capacity(a.len() + b.len() + 3);
            out.push('(');
            out.push_str(&a);
            out.push(',');
            out.push_str(&b);
            out.push(')');
            out
        }
        (true, false) => induced_signature(&left_labels, tree),
        (false, true) => induced_signature(&right_labels, tree),
        (false, false) => String::new(),
    }
}

fn child_below(tree: &Tree, ancestor: u32, label: u32) -> u32 {
    let mut cur = tree.node_by_label(label);
    while cur != NONE {
        let parent = tree.parent[cur as usize];
        if parent == ancestor {
            return cur;
        }
        if cur == ancestor {
            return cur;
        }
        cur = parent;
    }
    NONE
}

fn find_violating_triplet_between(
    labels: &[u32],
    ref_tree: &Tree,
    other_tree: &Tree,
    other_tree_idx: usize,
) -> Option<ViolatingTriplet> {
    let n = labels.len();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let (a, b, c) = (labels[i], labels[j], labels[k]);
                if triplet_outgroup(ref_tree, a, b, c) != triplet_outgroup(other_tree, a, b, c) {
                    return Some(ViolatingTriplet {
                        a,
                        b,
                        c,
                        ref_tree: 0,
                        other_tree: other_tree_idx,
                    });
                }
            }
        }
    }
    None
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
