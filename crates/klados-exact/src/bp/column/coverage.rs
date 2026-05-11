//! Per-tree internal-node coverage of a column.
//!
//! For a column with leaves `L`, the coverage in tree `T_i` is the set of
//! internal nodes lying on the union of leaf→LCA(L) paths. This is exactly
//! the set of internal nodes of `T_i|L`, and these are the nodes the column
//! "uses" in the LP's per-tree `≤1` cover constraint.

use klados_core::Tree;

/// Per-tree LCA-path internal nodes. `nodes_per_tree[t]` is sorted, deduped.
#[derive(Clone, Debug)]
pub struct ColumnCoverage {
    nodes_per_tree: Vec<Vec<usize>>,
}

impl ColumnCoverage {
    pub fn nodes(&self, tree_idx: usize) -> &[usize] {
        &self.nodes_per_tree[tree_idx]
    }

    pub fn iter_per_tree(&self) -> impl Iterator<Item = &[usize]> {
        self.nodes_per_tree.iter().map(|v| v.as_slice())
    }

    pub fn total_count(&self) -> usize {
        self.nodes_per_tree.iter().map(|v| v.len()).sum()
    }

    pub(super) fn empty(num_trees: usize) -> Self {
        Self {
            nodes_per_tree: vec![Vec::new(); num_trees],
        }
    }

    pub(super) fn from_marker_paths(
        labels: &[u32],
        trees: &[Tree],
        scratch: &mut Scratch,
    ) -> Self {
        let nodes_per_tree = trees
            .iter()
            .enumerate()
            .map(|(ti, tree)| scratch.mark_lca_path(ti, tree, labels))
            .collect();
        Self { nodes_per_tree }
    }
}

/// Reusable per-tree mark vectors used while constructing coverage. Lets
/// repeated column construction avoid reallocation.
pub struct Scratch {
    marks: Vec<Vec<u32>>,
    epochs: Vec<u32>,
}

impl Scratch {
    pub fn new(trees: &[Tree]) -> Self {
        Self {
            marks: trees.iter().map(|t| vec![0; t.num_nodes()]).collect(),
            epochs: vec![1; trees.len()],
        }
    }

    fn next_stamp(&mut self, ti: usize) -> u32 {
        let stamp = self.epochs[ti];
        if stamp == u32::MAX {
            self.marks[ti].fill(0);
            self.epochs[ti] = 2;
            1
        } else {
            self.epochs[ti] += 1;
            stamp
        }
    }

    fn mark_lca_path(&mut self, ti: usize, tree: &Tree, labels: &[u32]) -> Vec<usize> {
        debug_assert!(labels.len() >= 2, "coverage is empty for |L|<2");
        let stamp = self.next_stamp(ti);
        let marks = &mut self.marks[ti];

        let mut lca = tree.node_by_label(labels[0]);
        for &lbl in &labels[1..] {
            lca = tree.nearest_common_ancestor(lca, tree.node_by_label(lbl));
        }

        let mut covered = Vec::new();
        for &lbl in labels {
            let mut cur = tree.node_by_label(lbl);
            loop {
                let idx = cur as usize;
                if marks[idx] == stamp {
                    break;
                }
                marks[idx] = stamp;
                if !tree.is_leaf(cur) {
                    covered.push(idx);
                }
                if cur == lca {
                    break;
                }
                cur = tree.parent[idx];
            }
        }
        covered.sort_unstable();
        covered
    }
}
