//! Greedy packing of edge-disjoint discordant rooted triples.
//!
//! Each discordant triple `{a,b,c}` induces a necessary cut constraint on a
//! reference tree: if all edges in the minimal reference subtree spanning the
//! three leaves are preserved, the three leaves remain in one component whose
//! rooted triple topology disagrees with some input tree. Therefore every valid
//! agreement forest must cut at least one edge in that subtree. A set of such
//! triples with pairwise edge-disjoint induced subtrees gives a sound cut lower
//! bound by packing.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::tree::{Label, NONE, NodeId, Tree};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiscordantTriplePackingResult {
    /// Lower bound in component count, i.e. `cut_lower_bound + 1`.
    pub lower_bound: usize,
    /// Number of edge-disjoint discordant triples selected.
    pub cut_lower_bound: usize,
    /// All triples inspected by the greedy scan.
    pub triples_scanned: usize,
    /// Discordant triples encountered before edge-disjoint filtering.
    pub conflicts_seen: usize,
    /// Reference tree used for cut regions.
    pub reference_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiscordantTripleFractionalPackingResult {
    /// Lower bound in component count, i.e. `ceil(dual_cut_bound) + 1`.
    pub lower_bound: usize,
    /// Fractional lower bound on the number of cuts.
    pub dual_cut_bound: f64,
    /// Number of discordant triples kept in the sampled dual.
    pub conflicts_used: usize,
    /// All triples inspected while collecting the sample.
    pub triples_scanned: usize,
    /// Discordant triples encountered while collecting the sample.
    pub conflicts_seen: usize,
    /// Reference tree used for cut regions.
    pub reference_index: usize,
    /// Whether collection stopped because `max_conflicts` was reached.
    pub capped: bool,
}

impl Default for DiscordantTripleFractionalPackingResult {
    fn default() -> Self {
        Self {
            lower_bound: 1,
            dual_cut_bound: 0.0,
            conflicts_used: 0,
            triples_scanned: 0,
            conflicts_seen: 0,
            reference_index: 0,
            capped: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DiscordantTripleFractionalPackingConfig {
    /// Maximum sampled discordant triples per reference tree.
    pub max_conflicts: usize,
    /// Maximum triples to inspect per reference tree before returning the bound.
    pub max_triples_scanned: usize,
    /// Number of sampled reference trees. Tree 0 is always included.
    pub reference_limit: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct DiscordantTripleSortedPackingConfig {
    /// Maximum sampled discordant triples per reference tree.
    pub max_conflicts: usize,
    /// Maximum triples to inspect per reference tree before packing the sample.
    pub max_triples_scanned: usize,
    /// Number of sampled reference trees. Tree 0 is always included.
    pub reference_limit: usize,
}

impl Default for DiscordantTripleSortedPackingConfig {
    fn default() -> Self {
        Self {
            max_conflicts: 100_000,
            max_triples_scanned: 5_000_000,
            reference_limit: 3,
        }
    }
}

impl Default for DiscordantTripleFractionalPackingConfig {
    fn default() -> Self {
        Self {
            max_conflicts: 100_000,
            max_triples_scanned: 5_000_000,
            reference_limit: 3,
        }
    }
}

struct PairLca {
    n: usize,
    lca: Vec<NodeId>,
    depth: Vec<u16>,
}

struct TripleConflictCandidate {
    edges: Vec<usize>,
}

struct HeapCandidate {
    edge_count: usize,
    serial: usize,
    edges: Vec<usize>,
}

impl PartialEq for HeapCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.edge_count == other.edge_count && self.serial == other.serial
    }
}

impl Eq for HeapCandidate {}

impl PartialOrd for HeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.edge_count
            .cmp(&other.edge_count)
            .then_with(|| self.serial.cmp(&other.serial))
    }
}

impl PairLca {
    fn build(tree: &Tree, n: usize) -> Self {
        let stride = n + 1;
        let mut lca = vec![NONE; stride * stride];
        for a in 1..=n {
            let na = tree.node_by_label(a as Label);
            lca[a * stride + a] = na;
            for b in (a + 1)..=n {
                let nb = tree.node_by_label(b as Label);
                let x = tree.nearest_common_ancestor(na, nb);
                lca[a * stride + b] = x;
                lca[b * stride + a] = x;
            }
        }
        Self {
            n,
            lca,
            depth: tree.depth.clone(),
        }
    }

    #[inline]
    fn lca(&self, a: usize, b: usize) -> NodeId {
        self.lca[a * (self.n + 1) + b]
    }

    #[inline]
    fn depth(&self, node: NodeId) -> u16 {
        self.depth[node as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TripleTopology {
    Ab,
    Ac,
    Bc,
}

/// Greedy lower bound from edge-disjoint discordant triples using tree 0 as the
/// reference. Returns a lower bound on the MAF component count.
pub fn discordant_triple_packing_lower_bound(trees: &[Tree]) -> usize {
    discordant_triple_packing_detailed(trees).lower_bound
}

/// Best currently available discordant-triple packing lower bound. This combines
/// the full lexicographic edge-disjoint scan with the bounded shortest-region
/// scan. It is intended for analysis/bounds comparison; solver hot paths should
/// use an explicitly budgeted variant.
pub fn discordant_triple_best_packing_lower_bound(trees: &[Tree]) -> usize {
    discordant_triple_packing_lower_bound(trees)
        .max(discordant_triple_sorted_packing_lower_bound(trees))
}

/// Detailed version of [`discordant_triple_packing_lower_bound`].
pub fn discordant_triple_packing_detailed(trees: &[Tree]) -> DiscordantTriplePackingResult {
    discordant_triple_packing_with_reference(trees, 0)
}

/// Greedy lower bound using `reference_index` as the tree whose physical edges
/// define conflict regions. The returned lower bound is in component count.
pub fn discordant_triple_packing_with_reference(
    trees: &[Tree],
    reference_index: usize,
) -> DiscordantTriplePackingResult {
    if trees.len() <= 1 || trees.is_empty() || reference_index >= trees.len() {
        return DiscordantTriplePackingResult {
            lower_bound: 1,
            reference_index,
            ..Default::default()
        };
    }
    let n = trees[reference_index].num_leaves as usize;
    if n < 3 {
        return DiscordantTriplePackingResult {
            lower_bound: 1,
            reference_index,
            ..Default::default()
        };
    }

    let lcas: Vec<PairLca> = trees.iter().map(|t| PairLca::build(t, n)).collect();
    let reference = &trees[reference_index];
    let ref_lca = &lcas[reference_index];
    let mut used_edges = vec![false; reference.num_nodes()];
    let mut selected = 0usize;
    let mut conflicts_seen = 0usize;
    let mut triples_scanned = 0usize;

    for a in 1..=(n - 2) {
        for b in (a + 1)..=(n - 1) {
            for c in (b + 1)..=n {
                triples_scanned += 1;
                let ref_topology = triple_topology(ref_lca, a, b, c);
                if !lcas.iter().enumerate().any(|(ti, ix)| {
                    ti != reference_index && triple_topology(ix, a, b, c) != ref_topology
                }) {
                    continue;
                }
                conflicts_seen += 1;

                let root = triple_root(ref_lca, a, b, c);
                if region_touches_used_edge(reference, a, b, c, root, &used_edges) {
                    continue;
                }
                mark_region_edges(reference, a, b, c, root, &mut used_edges);
                selected += 1;
            }
        }
    }

    DiscordantTriplePackingResult {
        lower_bound: selected + 1,
        cut_lower_bound: selected,
        triples_scanned,
        conflicts_seen,
        reference_index,
    }
}

/// Fractional dual lower bound from sampled discordant triple cut regions.
///
/// For each sampled discordant triple region `R`, the valid primal constraint is
/// `sum_{e in R} x_e >= 1`. We build a feasible dual solution by assigning each
/// sampled conflict `c` weight `min_{e in R(c)} 1 / count(e)`, where `count(e)`
/// is the number of sampled conflicts containing edge `e`. Therefore, for every
/// edge `e`, the total dual weight of conflicts containing `e` is at most 1,
/// giving a sound lower bound on the number of cuts.
pub fn discordant_triple_fractional_packing_lower_bound(trees: &[Tree]) -> usize {
    discordant_triple_fractional_packing_detailed(trees).lower_bound
}

/// Detailed version of [`discordant_triple_fractional_packing_lower_bound`].
pub fn discordant_triple_fractional_packing_detailed(
    trees: &[Tree],
) -> DiscordantTripleFractionalPackingResult {
    discordant_triple_fractional_packing_with_config(
        trees,
        DiscordantTripleFractionalPackingConfig::default(),
    )
}

/// Fractional sampled lower bound with explicit collection limits.
pub fn discordant_triple_fractional_packing_with_config(
    trees: &[Tree],
    config: DiscordantTripleFractionalPackingConfig,
) -> DiscordantTripleFractionalPackingResult {
    if trees.len() <= 1 || trees.is_empty() {
        return DiscordantTripleFractionalPackingResult::default();
    }

    let mut best = DiscordantTripleFractionalPackingResult::default();
    for reference_index in sampled_reference_indices(trees.len(), config.reference_limit) {
        let result =
            discordant_triple_fractional_packing_with_reference(trees, reference_index, config);
        if result.lower_bound > best.lower_bound
            || (result.lower_bound == best.lower_bound
                && result.dual_cut_bound > best.dual_cut_bound)
        {
            best = result;
        }
    }
    best
}

/// Fractional sampled lower bound using a specific reference tree.
pub fn discordant_triple_fractional_packing_with_reference(
    trees: &[Tree],
    reference_index: usize,
    config: DiscordantTripleFractionalPackingConfig,
) -> DiscordantTripleFractionalPackingResult {
    if trees.len() <= 1 || trees.is_empty() || reference_index >= trees.len() {
        return DiscordantTripleFractionalPackingResult {
            reference_index,
            ..Default::default()
        };
    }
    let n = trees[reference_index].num_leaves as usize;
    if n < 3 || config.max_conflicts == 0 || config.max_triples_scanned == 0 {
        return DiscordantTripleFractionalPackingResult {
            reference_index,
            ..Default::default()
        };
    }

    let lcas: Vec<PairLca> = trees.iter().map(|t| PairLca::build(t, n)).collect();
    let reference = &trees[reference_index];
    let ref_lca = &lcas[reference_index];
    let mut candidates: Vec<TripleConflictCandidate> =
        Vec::with_capacity(config.max_conflicts.min(16_384));
    let mut triples_scanned = 0usize;
    let mut conflicts_seen = 0usize;
    let mut capped = false;

    'outer: for a in 1..=(n - 2) {
        for b in (a + 1)..=(n - 1) {
            for c in (b + 1)..=n {
                triples_scanned += 1;
                if triples_scanned > config.max_triples_scanned {
                    capped = true;
                    break 'outer;
                }

                let ref_topology = triple_topology(ref_lca, a, b, c);
                if !lcas.iter().enumerate().any(|(ti, ix)| {
                    ti != reference_index && triple_topology(ix, a, b, c) != ref_topology
                }) {
                    continue;
                }
                conflicts_seen += 1;

                let root = triple_root(ref_lca, a, b, c);
                let edges = region_edges(reference, a, b, c, root);
                if !edges.is_empty() {
                    candidates.push(TripleConflictCandidate { edges });
                    if candidates.len() >= config.max_conflicts {
                        capped = true;
                        break 'outer;
                    }
                }
            }
        }
    }

    let mut edge_counts = vec![0u32; reference.num_nodes()];
    for candidate in &candidates {
        for &edge in &candidate.edges {
            edge_counts[edge] += 1;
        }
    }

    let mut dual = 0.0f64;
    for candidate in &candidates {
        let mut weight = f64::INFINITY;
        for &edge in &candidate.edges {
            let count = edge_counts[edge];
            debug_assert!(count > 0);
            weight = weight.min(1.0 / count as f64);
        }
        if weight.is_finite() {
            dual += weight;
        }
    }

    DiscordantTripleFractionalPackingResult {
        lower_bound: (dual - 1.0e-9).ceil().max(0.0) as usize + 1,
        dual_cut_bound: dual,
        conflicts_used: candidates.len(),
        triples_scanned,
        conflicts_seen,
        reference_index,
        capped,
    }
}

/// Greedy edge-disjoint packing over a bounded sample, selecting shortest
/// reference-tree conflict regions first. This is still a sound packing bound;
/// the cap only weakens it by considering fewer valid conflicts.
pub fn discordant_triple_sorted_packing_lower_bound(trees: &[Tree]) -> usize {
    discordant_triple_sorted_packing_detailed(trees).lower_bound
}

/// Detailed version of [`discordant_triple_sorted_packing_lower_bound`].
pub fn discordant_triple_sorted_packing_detailed(trees: &[Tree]) -> DiscordantTriplePackingResult {
    discordant_triple_sorted_packing_with_config(
        trees,
        DiscordantTripleSortedPackingConfig::default(),
    )
}

/// Sorted sampled packing with explicit collection limits.
pub fn discordant_triple_sorted_packing_with_config(
    trees: &[Tree],
    config: DiscordantTripleSortedPackingConfig,
) -> DiscordantTriplePackingResult {
    if trees.len() <= 1 || trees.is_empty() {
        return DiscordantTriplePackingResult::default();
    }

    let mut best = DiscordantTriplePackingResult::default();
    for reference_index in sampled_reference_indices(trees.len(), config.reference_limit) {
        let result =
            discordant_triple_sorted_packing_with_reference(trees, reference_index, config);
        if result.lower_bound > best.lower_bound {
            best = result;
        }
    }
    best
}

/// Sorted sampled packing using a specific reference tree.
pub fn discordant_triple_sorted_packing_with_reference(
    trees: &[Tree],
    reference_index: usize,
    config: DiscordantTripleSortedPackingConfig,
) -> DiscordantTriplePackingResult {
    if trees.len() <= 1 || trees.is_empty() || reference_index >= trees.len() {
        return DiscordantTriplePackingResult {
            lower_bound: 1,
            reference_index,
            ..Default::default()
        };
    }
    let n = trees[reference_index].num_leaves as usize;
    if n < 3 || config.max_conflicts == 0 || config.max_triples_scanned == 0 {
        return DiscordantTriplePackingResult {
            lower_bound: 1,
            reference_index,
            ..Default::default()
        };
    }

    let lcas: Vec<PairLca> = trees.iter().map(|t| PairLca::build(t, n)).collect();
    let reference = &trees[reference_index];
    let ref_lca = &lcas[reference_index];
    let mut candidates: BinaryHeap<HeapCandidate> =
        BinaryHeap::with_capacity(config.max_conflicts.min(16_384));
    let mut triples_scanned = 0usize;
    let mut conflicts_seen = 0usize;
    let mut serial = 0usize;

    'outer: for a in 1..=(n - 2) {
        for b in (a + 1)..=(n - 1) {
            for c in (b + 1)..=n {
                triples_scanned += 1;
                if triples_scanned > config.max_triples_scanned {
                    break 'outer;
                }

                let ref_topology = triple_topology(ref_lca, a, b, c);
                if !lcas.iter().enumerate().any(|(ti, ix)| {
                    ti != reference_index && triple_topology(ix, a, b, c) != ref_topology
                }) {
                    continue;
                }
                conflicts_seen += 1;

                let root = triple_root(ref_lca, a, b, c);
                let edges = region_edges(reference, a, b, c, root);
                if !edges.is_empty() {
                    let edge_count = edges.len();
                    if candidates.len() < config.max_conflicts {
                        candidates.push(HeapCandidate {
                            edge_count,
                            serial,
                            edges,
                        });
                        serial += 1;
                    } else if candidates
                        .peek()
                        .is_some_and(|worst| edge_count < worst.edge_count)
                    {
                        candidates.pop();
                        candidates.push(HeapCandidate {
                            edge_count,
                            serial,
                            edges,
                        });
                        serial += 1;
                    }
                }
            }
        }
    }

    let mut candidates: Vec<TripleConflictCandidate> = candidates
        .into_iter()
        .map(|candidate| TripleConflictCandidate {
            edges: candidate.edges,
        })
        .collect();
    candidates.sort_unstable_by_key(|candidate| candidate.edges.len());
    let mut used_edges = vec![false; reference.num_nodes()];
    let mut selected = 0usize;
    for candidate in candidates {
        if candidate.edges.iter().any(|&edge| used_edges[edge]) {
            continue;
        }
        for edge in candidate.edges {
            used_edges[edge] = true;
        }
        selected += 1;
    }

    DiscordantTriplePackingResult {
        lower_bound: selected + 1,
        cut_lower_bound: selected,
        triples_scanned,
        conflicts_seen,
        reference_index,
    }
}

fn sampled_reference_indices(m: usize, limit: usize) -> Vec<usize> {
    let limit = limit.max(1).min(m);
    if limit == m {
        return (0..m).collect();
    }
    let mut refs = Vec::with_capacity(limit);
    for slot in 0..limit {
        let idx = slot * (m - 1) / (limit - 1).max(1);
        if refs.last().copied() != Some(idx) {
            refs.push(idx);
        }
    }
    refs
}

fn triple_topology(ix: &PairLca, a: usize, b: usize, c: usize) -> TripleTopology {
    let ab = ix.lca(a, b);
    let ac = ix.lca(a, c);
    let bc = ix.lca(b, c);
    let dab = ix.depth(ab);
    let dac = ix.depth(ac);
    let dbc = ix.depth(bc);
    if dab >= dac && dab >= dbc {
        TripleTopology::Ab
    } else if dac >= dbc {
        TripleTopology::Ac
    } else {
        TripleTopology::Bc
    }
}

fn triple_root(ix: &PairLca, a: usize, b: usize, c: usize) -> NodeId {
    let ab = ix.lca(a, b);
    let ac = ix.lca(a, c);
    let bc = ix.lca(b, c);
    let dab = ix.depth(ab);
    let dac = ix.depth(ac);
    let dbc = ix.depth(bc);
    if dab <= dac && dab <= dbc {
        ab
    } else if dac <= dbc {
        ac
    } else {
        bc
    }
}

fn region_touches_used_edge(
    tree: &Tree,
    a: usize,
    b: usize,
    c: usize,
    root: NodeId,
    used_edges: &[bool],
) -> bool {
    path_touches_used_edge(tree, a, root, used_edges)
        || path_touches_used_edge(tree, b, root, used_edges)
        || path_touches_used_edge(tree, c, root, used_edges)
}

fn path_touches_used_edge(tree: &Tree, label: usize, stop: NodeId, used_edges: &[bool]) -> bool {
    let mut cur = tree.node_by_label(label as Label);
    while cur != NONE && cur != stop {
        if used_edges[cur as usize] {
            return true;
        }
        cur = tree.parent[cur as usize];
    }
    false
}

fn mark_region_edges(
    tree: &Tree,
    a: usize,
    b: usize,
    c: usize,
    root: NodeId,
    used_edges: &mut [bool],
) {
    mark_path_edges(tree, a, root, used_edges);
    mark_path_edges(tree, b, root, used_edges);
    mark_path_edges(tree, c, root, used_edges);
}

fn mark_path_edges(tree: &Tree, label: usize, stop: NodeId, used_edges: &mut [bool]) {
    let mut cur = tree.node_by_label(label as Label);
    while cur != NONE && cur != stop {
        used_edges[cur as usize] = true;
        cur = tree.parent[cur as usize];
    }
}

fn region_edges(tree: &Tree, a: usize, b: usize, c: usize, root: NodeId) -> Vec<usize> {
    let mut out = Vec::new();
    append_path_edges(tree, a, root, &mut out);
    append_path_edges(tree, b, root, &mut out);
    append_path_edges(tree, c, root, &mut out);
    out
}

fn append_path_edges(tree: &Tree, label: usize, stop: NodeId, out: &mut Vec<usize>) {
    let mut cur = tree.node_by_label(label as Label);
    while cur != NONE && cur != stop {
        out.push(cur as usize);
        cur = tree.parent[cur as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::Tree;

    enum Shape {
        Leaf(Label),
        Join(Box<Shape>, Box<Shape>),
    }

    fn leaf(label: Label) -> Shape {
        Shape::Leaf(label)
    }

    fn join(left: Shape, right: Shape) -> Shape {
        Shape::Join(Box::new(left), Box::new(right))
    }

    fn build_tree(shape: Shape, n: u32) -> Tree {
        fn build_rec(tree: &mut Tree, shape: Shape, parent: NodeId) -> NodeId {
            let id = tree.parent.len() as NodeId;
            tree.parent.push(parent);
            match shape {
                Shape::Leaf(label) => {
                    tree.left.push(NONE);
                    tree.right.push(NONE);
                    tree.label.push(label);
                    tree.label_to_node[label as usize] = id;
                }
                Shape::Join(left, right) => {
                    tree.left.push(NONE);
                    tree.right.push(NONE);
                    tree.label.push(0);
                    let l = build_rec(tree, *left, id);
                    let r = build_rec(tree, *right, id);
                    tree.left[id as usize] = l;
                    tree.right[id as usize] = r;
                }
            }
            id
        }

        let mut tree = Tree::with_capacity(n);
        let root = build_rec(&mut tree, shape, NONE);
        tree.root = root;
        tree.compute_metadata();
        tree
    }

    #[test]
    fn identical_trees_have_trivial_bound() {
        let t = build_tree(join(join(leaf(1), leaf(2)), leaf(3)), 3);
        let result = discordant_triple_packing_detailed(&[t.clone(), t]);
        assert_eq!(result.lower_bound, 1);
        assert_eq!(result.cut_lower_bound, 0);
    }

    #[test]
    fn single_discordant_triple_gives_one_cut_bound() {
        let t1 = build_tree(join(join(leaf(1), leaf(2)), leaf(3)), 3);
        let t2 = build_tree(join(join(leaf(1), leaf(3)), leaf(2)), 3);
        let result = discordant_triple_packing_detailed(&[t1, t2]);
        assert_eq!(result.lower_bound, 2);
        assert_eq!(result.cut_lower_bound, 1);
        assert_eq!(result.conflicts_seen, 1);
    }

    #[test]
    fn edge_disjoint_discordant_triples_pack_together() {
        let t1 = build_tree(
            join(
                join(join(leaf(1), leaf(2)), leaf(3)),
                join(join(leaf(4), leaf(5)), leaf(6)),
            ),
            6,
        );
        let t2 = build_tree(
            join(
                join(join(leaf(1), leaf(3)), leaf(2)),
                join(join(leaf(4), leaf(6)), leaf(5)),
            ),
            6,
        );
        let result = discordant_triple_packing_detailed(&[t1, t2]);
        assert_eq!(result.lower_bound, 3);
        assert_eq!(result.cut_lower_bound, 2);
    }
}
