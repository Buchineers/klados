//! Brute-force MAF oracle.
//!
//! Enumerates set partitions of the leaf set L and finds the smallest one that
//! is a valid agreement forest of the input instance. Used as ground truth in
//! tests; not for production solving.
//!
//! Cost: B(n) partitions where B is the Bell number. B(8)=4140, B(9)=21147,
//! B(10)=115975. Each partition costs O(m·n²) to validate. Practical up to
//! n≈9; very slow at n=10. Refuses to run for n>10 by default.
//!
//! Returns the optimal *number of components*, plus an example optimal
//! partition expressed as a Vec<Vec<Label>>.

use fixedbitset::FixedBitSet;

use crate::Instance;
use crate::af_validator::{AfValidation, validate_agreement_forest};
use crate::tree::{Label, NONE, NodeId, Tree};

pub const BRUTE_MAF_MAX_N: u32 = 10;

#[derive(Debug, Clone)]
pub struct BruteMafResult {
    pub num_components: usize,
    pub partition: Vec<Vec<Label>>,
}

/// Brute-force optimum for `instance`. Returns `None` if `n > BRUTE_MAF_MAX_N`.
pub fn brute_force_maf(instance: &Instance) -> Option<BruteMafResult> {
    let n = instance.num_leaves;
    if n > BRUTE_MAF_MAX_N {
        return None;
    }
    if n == 0 {
        return Some(BruteMafResult {
            num_components: 0,
            partition: vec![],
        });
    }
    if n == 1 {
        return Some(BruteMafResult {
            num_components: 1,
            partition: vec![vec![1]],
        });
    }

    let mut best: Option<BruteMafResult> = None;

    enumerate_partitions(n, &mut |part| {
        // Early prune: only consider partitions strictly smaller than best.
        let k = part.iter().filter(|p| !p.is_empty()).count();
        if let Some(ref b) = best {
            if k >= b.num_components {
                return;
            }
        }

        // Materialize partition as component trees: each component's tree is
        // the restriction of T0 to the component's leaves.
        let t0 = &instance.trees[0];
        let comps: Vec<Tree> = part
            .iter()
            .filter(|p| !p.is_empty())
            .map(|labels| {
                let mut keep = FixedBitSet::with_capacity(n as usize + 1);
                for &lbl in labels {
                    keep.insert(lbl as usize);
                }
                t0.prune_to_leafset(&keep)
            })
            .collect();

        if matches!(
            validate_agreement_forest(instance, &comps),
            AfValidation::Ok
        ) {
            best = Some(BruteMafResult {
                num_components: k,
                partition: part.iter().filter(|p| !p.is_empty()).cloned().collect(),
            });
        }
    });

    best
}

/// Enumerate set partitions of {1, .., n} via the standard recursive scheme:
/// each new element joins one of the existing parts or starts a new one.
fn enumerate_partitions<F: FnMut(&[Vec<Label>])>(n: u32, visit: &mut F) {
    let mut current: Vec<Vec<Label>> = vec![];
    rec(1, n, &mut current, visit);
}

fn rec<F: FnMut(&[Vec<Label>])>(
    next_label: u32,
    n: u32,
    current: &mut Vec<Vec<Label>>,
    visit: &mut F,
) {
    if next_label > n {
        visit(current);
        return;
    }
    // Place next_label into each existing part.
    for i in 0..current.len() {
        current[i].push(next_label as Label);
        rec(next_label + 1, n, current, visit);
        current[i].pop();
    }
    // Or start a new part.
    current.push(vec![next_label as Label]);
    rec(next_label + 1, n, current, visit);
    current.pop();
}

// Re-export NodeId/NONE consumers don't strictly need them, but keep them in
// scope so the file compiles standalone.
#[allow(dead_code)]
fn _scope_check(_n: NodeId) -> NodeId {
    NONE
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
    fn balanced_4() -> Tree {
        // ((1,2),(3,4))
        let mut t = Tree::with_capacity(4);
        let l1 = push_leaf(&mut t, 1);
        let l2 = push_leaf(&mut t, 2);
        let l3 = push_leaf(&mut t, 3);
        let l4 = push_leaf(&mut t, 4);
        let n12 = push_internal(&mut t, l1, l2);
        let n34 = push_internal(&mut t, l3, l4);
        let root = push_internal(&mut t, n12, n34);
        t.root = root;
        t.compute_metadata();
        t
    }
    fn ladder_4() -> Tree {
        // (1,(2,(3,4)))
        let mut t = Tree::with_capacity(4);
        let l1 = push_leaf(&mut t, 1);
        let l2 = push_leaf(&mut t, 2);
        let l3 = push_leaf(&mut t, 3);
        let l4 = push_leaf(&mut t, 4);
        let n34 = push_internal(&mut t, l3, l4);
        let n234 = push_internal(&mut t, l2, n34);
        let root = push_internal(&mut t, l1, n234);
        t.root = root;
        t.compute_metadata();
        t
    }

    #[test]
    fn brute_identical_trees_one_component() {
        let t = balanced_4();
        let inst = Instance::new(vec![t.clone(), t.clone()], 4);
        let r = brute_force_maf(&inst).unwrap();
        assert_eq!(r.num_components, 1);
    }

    #[test]
    fn brute_balanced_vs_ladder_4() {
        // Two distinct topologies on the same 4 leaves. The MAF size is small
        // and known to be 2 (one cut suffices).
        let t1 = balanced_4();
        let t2 = ladder_4();
        let inst = Instance::new(vec![t1, t2], 4);
        let r = brute_force_maf(&inst).unwrap();
        assert_eq!(r.num_components, 2, "got partition {:?}", r.partition);
    }

    #[test]
    fn brute_too_large_refuses() {
        // Build an 11-leaf instance to confirm refusal.
        let mut t = Tree::with_capacity(11);
        let mut prev = push_leaf(&mut t, 1);
        for lbl in 2..=11u32 {
            let leaf = push_leaf(&mut t, lbl as Label);
            prev = push_internal(&mut t, prev, leaf);
        }
        t.root = prev;
        t.compute_metadata();
        let inst = Instance::new(vec![t.clone(), t], 11);
        assert!(brute_force_maf(&inst).is_none());
    }
}
