//! Agreement-forest validator.
//!
//! Given an `Instance` (m phylogenies on the same leaf set L) and a list of
//! components (each a `Tree` on a subset of L), check whether the components
//! form a valid agreement forest:
//!
//! 1. The component leaf-sets are a partition of L (each label appears in
//!    exactly one component).
//! 2. For each component X and every input tree T_i, the restriction of T_i
//!    to X (pruned, then suppressing degree-1 internal nodes) is the same
//!    tree across all i (up to leaf-label-respecting isomorphism).
//!
//! This is the *definition* of an agreement forest. It is the only honest
//! way to confirm a solver's output is correct; leaf-coverage checks are
//! necessary but not sufficient.

use fixedbitset::FixedBitSet;

use crate::Instance;
use crate::tree::{Label, NONE, NodeId, Tree};

/// Result of validating a candidate agreement forest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AfValidation {
    Ok,
    /// A leaf appears in zero or multiple components.
    BadPartition(String),
    /// Some component's restriction differs between two input trees.
    NotIsomorphic {
        component_index: usize,
        tree_a: usize,
        tree_b: usize,
        canonical_a: String,
        canonical_b: String,
    },
    /// Two components' LCAs are comparable in some input tree, so the components
    /// can't simultaneously be sibling-disjoint subtrees of any forest derived
    /// from that tree. This is the "no nesting" condition that makes the
    /// partition an actual AF, not just a set of isomorphic restrictions.
    ComponentsNest {
        tree_index: usize,
        component_a: usize,
        component_b: usize,
    },
    /// A component contains a label that is not in the instance.
    LabelOutOfRange {
        component_index: usize,
        label: Label,
    },
}

impl AfValidation {
    pub fn is_ok(&self) -> bool {
        matches!(self, AfValidation::Ok)
    }
}

/// Validate that `components` form an agreement forest of `instance`.
pub fn validate_agreement_forest(instance: &Instance, components: &[Tree]) -> AfValidation {
    let n = instance.num_leaves;

    // Step 1: partition check.
    let mut seen = FixedBitSet::with_capacity(n as usize + 1);
    for (ci, comp) in components.iter().enumerate() {
        for lbl in comp.leaves() {
            if lbl == 0 || lbl > n {
                return AfValidation::LabelOutOfRange {
                    component_index: ci,
                    label: lbl,
                };
            }
            if seen.contains(lbl as usize) {
                return AfValidation::BadPartition(format!(
                    "leaf {} appears in multiple components (component {})",
                    lbl, ci
                ));
            }
            seen.insert(lbl as usize);
        }
    }
    for lbl in 1..=n {
        if !seen.contains(lbl as usize) {
            return AfValidation::BadPartition(format!("leaf {} missing from all components", lbl));
        }
    }

    // Step 2: isomorphism check on each component's restriction.
    // For each component X and each input tree T_i, take the restriction T_i|X
    // and reduce it to a canonical leaf-labeled string; compare across i.
    for (ci, comp) in components.iter().enumerate() {
        let leafset = component_leafset(comp, n);
        if leafset.count_ones(..) <= 1 {
            // Singleton or empty components are trivially OK across all trees
            // (they always restrict to a single leaf or nothing).
            continue;
        }
        let mut canon_per_tree: Vec<String> = Vec::with_capacity(instance.num_trees());
        for t in &instance.trees {
            let restricted = t.prune_to_leafset(&leafset);
            canon_per_tree.push(canonical_newick(&restricted));
        }
        // Also check the component's *own* topology matches the restricted
        // input-tree topology. Without this, a solver could return arbitrary
        // shapes and pass the isomorphism check among themselves.
        let comp_canon = canonical_newick(comp);
        for (i, c) in canon_per_tree.iter().enumerate() {
            if c != &comp_canon {
                return AfValidation::NotIsomorphic {
                    component_index: ci,
                    tree_a: i,
                    tree_b: usize::MAX, // means: vs the component itself
                    canonical_a: c.clone(),
                    canonical_b: comp_canon,
                };
            }
        }
    }

    // Step 3: spanned-subtree disjointness (Bordewich-Semple AF condition).
    //
    // For each input tree T_i, compute the minimal subtree of T_i that spans
    // each component's leaves (= union of all paths from each component-leaf
    // up to the component's LCA). The spanned subtrees of distinct components
    // must be pairwise NODE-disjoint in every T_i.
    let mut leafsets: Vec<FixedBitSet> = Vec::with_capacity(components.len());
    for comp in components {
        leafsets.push(component_leafset(comp, n));
    }
    for (ti, tree) in instance.trees.iter().enumerate() {
        // owner[node] = component index that claims this node, or -1 (free).
        let mut owner: Vec<i32> = vec![-1; tree.num_nodes()];
        for (ci, ls) in leafsets.iter().enumerate() {
            if ls.count_ones(..) == 0 {
                continue;
            }
            let lca = lca_of_labels(tree, ls);
            for lbl in ls.ones() {
                let mut cur = tree.label_to_node[lbl as Label as usize];
                while cur != NONE {
                    let prev = owner[cur as usize];
                    if prev == -1 {
                        owner[cur as usize] = ci as i32;
                    } else if prev != ci as i32 {
                        return AfValidation::ComponentsNest {
                            tree_index: ti,
                            component_a: prev as usize,
                            component_b: ci,
                        };
                    }
                    if cur == lca {
                        break;
                    }
                    cur = tree.parent[cur as usize];
                }
            }
        }
    }

    AfValidation::Ok
}

fn lca_of_labels(tree: &Tree, leafset: &FixedBitSet) -> NodeId {
    let mut iter = leafset.ones();
    let first = match iter.next() {
        Some(l) => l as Label,
        None => return NONE,
    };
    let mut acc = tree.label_to_node[first as usize];
    for lbl in iter {
        let n = tree.label_to_node[lbl as Label as usize];
        if n == NONE {
            continue;
        }
        acc = tree.nearest_common_ancestor(acc, n);
    }
    acc
}

fn leaves_under(tree: &Tree, node: NodeId, n: u32) -> FixedBitSet {
    let mut s = FixedBitSet::with_capacity(n as usize + 1);
    if node == NONE {
        return s;
    }
    let mut stack = vec![node];
    while let Some(v) = stack.pop() {
        if tree.is_leaf(v) {
            let lbl = tree.label[v as usize];
            if lbl > 0 && lbl <= n {
                s.insert(lbl as usize);
            }
        } else if let Some((l, r)) = tree.children(v) {
            stack.push(l);
            stack.push(r);
        }
    }
    s
}

/// True if `a` and `b` lie on a single root-to-leaf path (one is the other's
/// ancestor, possibly equal).
fn comparable(tree: &Tree, a: NodeId, b: NodeId) -> bool {
    descendant_or_equal(tree, a, b) || descendant_or_equal(tree, b, a)
}

/// True if `desc` is `anc` or a descendant of `anc`.
fn descendant_or_equal(tree: &Tree, desc: NodeId, anc: NodeId) -> bool {
    if desc == NONE || anc == NONE {
        return false;
    }
    let mut cur = desc;
    while cur != NONE {
        if cur == anc {
            return true;
        }
        cur = tree.parent[cur as usize];
    }
    false
}

fn component_leafset(comp: &Tree, n: u32) -> FixedBitSet {
    let mut s = FixedBitSet::with_capacity(n as usize + 1);
    for lbl in comp.leaves() {
        if lbl > 0 && lbl <= n {
            s.insert(lbl as usize);
        }
    }
    s
}

/// Canonical Newick for a leaf-labeled binary tree:
/// - leaves: their integer label
/// - internal: "(left,right)" with left <= right by lexicographic comparison
/// - empty tree: "_"
pub fn canonical_newick(tree: &Tree) -> String {
    if tree.root == NONE {
        return "_".to_string();
    }
    let mut buf = String::new();
    canon_rec(tree, tree.root, &mut buf);
    buf
}

fn canon_rec(tree: &Tree, node: NodeId, buf: &mut String) {
    if tree.is_leaf(node) {
        buf.push_str(&tree.label[node as usize].to_string());
        return;
    }
    let (l, r) = tree.children(node).unwrap();
    let mut lb = String::new();
    canon_rec(tree, l, &mut lb);
    let mut rb = String::new();
    canon_rec(tree, r, &mut rb);
    let (a, b) = if lb <= rb { (lb, rb) } else { (rb, lb) };
    buf.push('(');
    buf.push_str(&a);
    buf.push(',');
    buf.push_str(&b);
    buf.push(')');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_leaf(t: &mut Tree, lbl: Label) -> NodeId {
        let id = t.parent.len() as NodeId;
        t.parent.push(NONE);
        t.left.push(NONE);
        t.right.push(NONE);
        t.label.push(lbl);
        t.label_to_node[lbl as usize] = id;
        id
    }
    fn push_internal(t: &mut Tree, l: NodeId, r: NodeId) -> NodeId {
        let id = t.parent.len() as NodeId;
        t.parent.push(NONE);
        t.left.push(l);
        t.right.push(r);
        t.label.push(0);
        t.parent[l as usize] = id;
        t.parent[r as usize] = id;
        id
    }

    /// (((1,2),(3,4)),(5,6))
    fn balanced_6() -> Tree {
        let mut t = Tree::with_capacity(6);
        let l1 = push_leaf(&mut t, 1);
        let l2 = push_leaf(&mut t, 2);
        let l3 = push_leaf(&mut t, 3);
        let l4 = push_leaf(&mut t, 4);
        let l5 = push_leaf(&mut t, 5);
        let l6 = push_leaf(&mut t, 6);
        let n12 = push_internal(&mut t, l1, l2);
        let n34 = push_internal(&mut t, l3, l4);
        let n1234 = push_internal(&mut t, n12, n34);
        let n56 = push_internal(&mut t, l5, l6);
        let root = push_internal(&mut t, n1234, n56);
        t.root = root;
        t.compute_metadata();
        t
    }

    /// ((1,2),((3,4),(5,6)))  — same leaves, different topology
    fn alt_6() -> Tree {
        let mut t = Tree::with_capacity(6);
        let l1 = push_leaf(&mut t, 1);
        let l2 = push_leaf(&mut t, 2);
        let l3 = push_leaf(&mut t, 3);
        let l4 = push_leaf(&mut t, 4);
        let l5 = push_leaf(&mut t, 5);
        let l6 = push_leaf(&mut t, 6);
        let n12 = push_internal(&mut t, l1, l2);
        let n34 = push_internal(&mut t, l3, l4);
        let n56 = push_internal(&mut t, l5, l6);
        let n3456 = push_internal(&mut t, n34, n56);
        let root = push_internal(&mut t, n12, n3456);
        t.root = root;
        t.compute_metadata();
        t
    }

    #[test]
    fn canonical_newick_is_canonical() {
        let t = balanced_6();
        let c = canonical_newick(&t);
        assert!(c.contains('1'));
        assert!(c.contains('6'));
        // canonical form sorts subtrees lexically
        assert_eq!(c, canonical_newick(&t));
    }

    #[test]
    fn identical_trees_single_component_is_valid() {
        let t = balanced_6();
        let inst = Instance::new(vec![t.clone(), t.clone()], 6);
        let comps = vec![t.clone()];
        assert_eq!(validate_agreement_forest(&inst, &comps), AfValidation::Ok);
    }

    #[test]
    fn missing_leaf_rejected() {
        let t = balanced_6();
        let inst = Instance::new(vec![t.clone(), t.clone()], 6);
        // Component missing leaf 6.
        let mut keep = FixedBitSet::with_capacity(7);
        for lbl in 1..=5 {
            keep.insert(lbl);
        }
        let bad = t.prune_to_leafset(&keep);
        let result = validate_agreement_forest(&inst, &[bad]);
        assert!(matches!(result, AfValidation::BadPartition(_)));
    }

    #[test]
    fn duplicate_leaf_rejected() {
        let t = balanced_6();
        let inst = Instance::new(vec![t.clone(), t.clone()], 6);
        let mut keep_a = FixedBitSet::with_capacity(7);
        for lbl in 1..=4 {
            keep_a.insert(lbl);
        }
        let mut keep_b = FixedBitSet::with_capacity(7);
        for lbl in 3..=6 {
            keep_b.insert(lbl);
        }
        let a = t.prune_to_leafset(&keep_a);
        let b = t.prune_to_leafset(&keep_b);
        let result = validate_agreement_forest(&inst, &[a, b]);
        assert!(matches!(result, AfValidation::BadPartition(_)));
    }

    #[test]
    fn non_isomorphic_partition_rejected() {
        // T1 = balanced_6, T2 = alt_6. The single-component partition
        // {all 6 leaves} cannot be an AF because the two trees disagree.
        let t1 = balanced_6();
        let t2 = alt_6();
        let inst = Instance::new(vec![t1.clone(), t2.clone()], 6);
        // Submit T1 as the lone component — its restriction matches T1 (Ok)
        // but not T2 (mismatch).
        let result = validate_agreement_forest(&inst, &[t1.clone()]);
        assert!(
            matches!(result, AfValidation::NotIsomorphic { .. }),
            "got {:?}",
            result
        );
    }

    #[test]
    fn singleton_components_always_valid() {
        let t1 = balanced_6();
        let t2 = alt_6();
        let inst = Instance::new(vec![t1, t2], 6);
        let comps: Vec<Tree> = (1..=6).map(|lbl| Tree::singleton(lbl, 6)).collect();
        assert_eq!(validate_agreement_forest(&inst, &comps), AfValidation::Ok);
    }
}
