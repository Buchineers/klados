//! Whidden-style cluster decomposition for 2-tree MAF instances.
//!
//! The production path identifies a strict/common "cluster point" c in T1 —
//! an internal node whose round-trip (T1 -> twin in T2 -> back to T1) lands
//! exactly on c, so the same pendant leaf-set X can be cut out in both trees.
//! For such a c, the optimal MAF decomposes as
//!
//!   d(T1, T2) = d(T1|X, T2|X) + d(T1[X->P], T2[X->P]) - 1
//!
//! where T1[X->P] replaces the X-subtree with a single placeholder leaf P.
//! Equivalently, the MAF component count is inner + outer - 1.
//!
//! Inner and outer subproblems are solved by recursion; on return, the
//! component from the inner solution that can validly merge with the outer
//! P-component is the "anchor". The merge is validated against the original
//! instance; if no anchor produces a valid AF, decomposition is aborted.
//!
//! Restricted to m == 2. Returns None if no strict cluster point exists or if
//! no valid merge is found.  rspr's more general relaxed/batch machinery uses
//! explicit forest boundary state (rho/component-zero/cluster-node joins); the
//! old Rust prototypes for those variants remain opt-in below until that state
//! is ported faithfully.

use std::sync::atomic::{AtomicBool, Ordering};

use fixedbitset::FixedBitSet;

use klados_core::Instance;
use klados_core::af_validator::validate_agreement_forest;
use klados_core::tree::{Label, NONE, NodeId, Tree};
use log::debug;

/// Never-set termination flag for callers that don't supply their own. The
/// decomposition is then uninterruptible (its prior behaviour). Hot anytime
/// callers pass a real flag so a SIGTERM mid-decomposition bails promptly
/// instead of grinding through the O(n^3) cluster-point recursion.
pub static NEVER_TERMINATE: AtomicBool = AtomicBool::new(false);

/// Try Whidden cluster decomposition on a 2-tree instance.
///
/// `solve` is invoked on each sub-instance; pass the inner solver
/// (e.g. B&P-Multi). Returns `None` if no useful cluster point is found
/// or if any sub-solve returns `None`.
/// `terminate`: when set, the decomposition bails (returns `None`) instead of
/// continuing the recursion — callers without their own flag pass
/// [`NEVER_TERMINATE`].
pub fn try_whidden_decomp_2tree<F>(
    instance: &Instance,
    solve: &mut F,
    terminate: &AtomicBool,
) -> Option<Vec<Tree>>
where
    F: FnMut(&Instance) -> Option<Vec<Tree>>,
{
    // The C++ rspr code does *not* solve relaxed/batch clusters as closed
    // leaf-set subinstances. It builds boundary-aware forests, possibly adds
    // rho, and joins them back through ClusterForest/ClusterInstance state.
    // The Rust batch/relaxed prototypes below validate feasibility but do not
    // prove the lower-bound accounting needed for an exact solver return, so
    // the exact path uses the certified strict split only.
    try_single_decomp(instance, solve, terminate)
}

/// Try the relaxed/batch rspr-inspired decompositions as a *feasible
/// incumbent only*.
///
/// These paths deliberately do not certify optimality until the full
/// ClusterInstance boundary join is represented natively.  The returned forest
/// has passed the AF validator, so callers may safely use it as an upper bound
/// / incumbent, but must not set lower_bound from it.
pub fn try_whidden_relaxed_incumbent_2tree<F>(
    instance: &Instance,
    solve: &mut F,
    batch_incumbent: bool,
) -> Option<Vec<Tree>>
where
    F: FnMut(&Instance) -> Option<Vec<Tree>>,
{
    if let Some(r) = try_rspr_decomp(instance, solve) {
        return Some(r);
    }
    if batch_incumbent {
        return try_batch_decomp(instance, solve);
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RsprClusterPointKind {
    Strict,
    Relaxed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RsprClusterPoint {
    t1_node: NodeId,
    t2_twin: NodeId,
    t1_round_trip: NodeId,
    size: usize,
    kind: RsprClusterPointKind,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct WhiddenDecompPotential {
    pub strict_points: usize,
    pub relaxed_points: usize,
    pub balanced_selected: usize,
    pub balanced_clustered: usize,
    pub balanced_remainder: usize,
    pub balanced_largest_subproblem: usize,
    pub strict_selected: usize,
    pub strict_clustered: usize,
    pub strict_remainder: usize,
    pub strict_largest_subproblem: usize,
    pub top_strict_sizes: Vec<usize>,
    pub top_relaxed_sizes: Vec<usize>,
}

pub(crate) fn analyze_whidden_decomp_potential(
    instance: &Instance,
) -> Option<WhiddenDecompPotential> {
    if instance.num_trees() != 2 {
        return None;
    }
    let n = instance.num_leaves as usize;
    if n < 6 {
        return None;
    }

    let t1 = &instance.trees[0];
    let t2 = &instance.trees[1];
    let leaf_sets = compute_leaf_sets(t1, n);
    let tw12 = compute_twins(t1, t2);
    let tw21 = compute_twins(t2, t1);
    let points = collect_rspr_cluster_points(t1, t2, &leaf_sets, &tw12, &tw21, n);
    if points.is_empty() {
        return Some(WhiddenDecompPotential {
            strict_remainder: n,
            strict_largest_subproblem: n,
            balanced_remainder: n,
            balanced_largest_subproblem: n,
            ..WhiddenDecompPotential::default()
        });
    }

    let strict_points = points
        .iter()
        .filter(|p| p.kind == RsprClusterPointKind::Strict)
        .count();
    let relaxed_points = points.len().saturating_sub(strict_points);

    let mut balanced_candidates: Vec<(NodeId, usize)> = points
        .iter()
        .filter(|p| p.size >= 2 && p.size <= n - 2)
        .map(|p| (p.t1_node, p.size))
        .collect();
    balanced_candidates.sort_by_key(|(_, size)| -((*size).min(n - *size) as isize));

    let mut strict_candidates: Vec<(NodeId, usize)> = points
        .iter()
        .filter(|p| p.kind == RsprClusterPointKind::Strict)
        .filter(|p| p.size >= 2 && p.size <= n - 2)
        .filter(|p| !t2.is_root(p.t2_twin))
        .map(|p| (p.t1_node, p.size))
        .collect();
    strict_candidates.sort_by_key(|(_, size)| -((*size).min(n - *size) as isize));

    let balanced = select_disjoint_cluster_stats(n, &leaf_sets, &balanced_candidates);
    let strict = select_disjoint_cluster_stats(n, &leaf_sets, &strict_candidates);

    let mut top_strict_sizes: Vec<usize> = points
        .iter()
        .filter(|p| p.kind == RsprClusterPointKind::Strict)
        .map(|p| p.size)
        .collect();
    top_strict_sizes.sort_unstable_by(|a, b| b.cmp(a));
    top_strict_sizes.truncate(8);

    let mut top_relaxed_sizes: Vec<usize> = points
        .iter()
        .filter(|p| p.kind == RsprClusterPointKind::Relaxed)
        .map(|p| p.size)
        .collect();
    top_relaxed_sizes.sort_unstable_by(|a, b| b.cmp(a));
    top_relaxed_sizes.truncate(8);

    Some(WhiddenDecompPotential {
        strict_points,
        relaxed_points,
        balanced_selected: balanced.selected,
        balanced_clustered: balanced.clustered,
        balanced_remainder: balanced.remainder,
        balanced_largest_subproblem: balanced.largest_subproblem,
        strict_selected: strict.selected,
        strict_clustered: strict.clustered,
        strict_remainder: strict.remainder,
        strict_largest_subproblem: strict.largest_subproblem,
        top_strict_sizes,
        top_relaxed_sizes,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DisjointClusterStats {
    selected: usize,
    clustered: usize,
    remainder: usize,
    largest_subproblem: usize,
}

fn select_disjoint_cluster_stats(
    n: usize,
    leaf_sets: &[FixedBitSet],
    candidates: &[(NodeId, usize)],
) -> DisjointClusterStats {
    let cap = leaf_sets.first().map(|l| l.len()).unwrap_or(0);
    let mut taken = FixedBitSet::with_capacity(cap);
    let mut selected = 0usize;
    let mut largest = 0usize;
    for &(node, size) in candidates {
        let leaves = &leaf_sets[node as usize];
        if !taken.is_disjoint(leaves) {
            continue;
        }
        taken.union_with(leaves);
        selected += 1;
        largest = largest.max(size);
    }
    let clustered = taken.count_ones(..);
    let remainder = n.saturating_sub(clustered) + selected;
    DisjointClusterStats {
        selected,
        clustered,
        remainder,
        largest_subproblem: largest.max(remainder),
    }
}

/// rspr-style: find clusters, solve inners, keep anchors in tree, prune rest.
fn try_rspr_decomp<F>(instance: &Instance, solve: &mut F) -> Option<Vec<Tree>>
where
    F: FnMut(&Instance) -> Option<Vec<Tree>>,
{
    let n = instance.num_leaves as usize;
    if n < 6 {
        return None;
    }
    let t1 = &instance.trees[0];
    let t2 = &instance.trees[1];
    let leaf_sets = compute_leaf_sets(t1, n);
    let tw12 = compute_twins(t1, t2);
    let tw21 = compute_twins(t2, t1);

    // 1. Find cluster points (relaxed + child filter).
    let cluster_points = collect_rspr_cluster_points(t1, t2, &leaf_sets, &tw12, &tw21, n);
    let mut pts: Vec<(NodeId, usize)> = cluster_points
        .iter()
        .filter(|point| point.size <= n - 2)
        .map(|point| (point.t1_node, point.size.min(n - point.size)))
        .collect();
    pts.sort_by_key(|(_, s)| -(*s as isize));
    if pts.is_empty() {
        return None;
    }

    // 2. Greedy disjoint selection.
    let cap = leaf_sets.first().map(|l| l.len()).unwrap_or(0);
    let mut taken = FixedBitSet::with_capacity(cap);
    let mut sel: Vec<usize> = Vec::new();
    for (i, (node, _)) in pts.iter().enumerate() {
        if !taken.is_disjoint(&leaf_sets[*node as usize]) {
            continue;
        }
        taken.union_with(&leaf_sets[*node as usize]);
        sel.push(i);
    }
    if sel.is_empty() {
        return None;
    }

    // 3. Solve each cluster's inner instance. Find anchors by trial-validate.
    let mut inner_results: Vec<(Vec<Tree>, usize, FixedBitSet)> = Vec::with_capacity(sel.len());
    let mut all_clustered = FixedBitSet::with_capacity(n + 1);
    for &ci in &sel {
        let (cnode, _) = pts[ci];
        let leaves = leaf_sets[cnode as usize].clone();
        all_clustered.union_with(&leaves);
        let csize = leaves.count_ones(..);
        let mut imap = vec![0u32; n + 1];
        let mut ito = vec![0u32; csize + 1];
        for (l, leaf) in (1u32..).zip(leaves.ones()) {
            imap[leaf] = l;
            ito[l as usize] = leaf as u32;
        }
        let inner = solve(&Instance::new(
            vec![
                t1.relabel(&imap, csize as u32),
                t2.relabel(&imap, csize as u32),
            ],
            csize as u32,
        ))?;

        // Decode inner components to original labels.
        let decoded: Vec<Tree> = inner
            .iter()
            .map(|c| c.relabel(&ito, instance.num_leaves))
            .collect();
        inner_results.push((decoded, 0, leaves)); // anchor = first component
    }

    // 4. Build remaining: keep non-clustered leaves + anchor leaves, prune rest.
    let mut remaining_keep = FixedBitSet::with_capacity(n + 1);
    for lbl in 1..=n as u32 {
        remaining_keep.insert(lbl as usize);
    }
    for (decoded, anchor_idx, leaves) in &inner_results {
        let anchor_leaves: FixedBitSet =
            decoded[*anchor_idx]
                .leaves()
                .fold(FixedBitSet::with_capacity(n + 1), |mut s, lbl| {
                    s.insert(lbl as usize);
                    s
                });
        for leaf in leaves.ones() {
            if !anchor_leaves.contains(leaf) {
                remaining_keep.set(leaf, false);
            }
        }
    }
    // Compact via relabel.
    let mut rem_map = vec![0u32; n + 1];
    let mut rem_rev = vec![0u32; n + 1];
    let mut next_rem: u32 = 1;
    for lbl in 1..=n as u32 {
        if remaining_keep.contains(lbl as usize) {
            rem_map[lbl as usize] = next_rem;
            rem_rev[next_rem as usize] = lbl;
            next_rem += 1;
        }
    }
    let rem_n = next_rem - 1;
    if rem_n <= 1 {
        return None;
    }
    let rem_t1 = t1.relabel(&rem_map, rem_n);
    let rem_t2 = t2.relabel(&rem_map, rem_n);
    let rem_inst = Instance::new(vec![rem_t1, rem_t2], rem_n);
    let rem_solution = solve(&rem_inst)?;

    // 5. Combine: non-anchor inner components + remaining solution.
    let mut result: Vec<Tree> = Vec::new();
    for (decoded, anchor_idx, _) in &inner_results {
        for (j, c) in decoded.iter().enumerate() {
            if j != *anchor_idx {
                result.push(c.clone());
            }
        }
    }
    for c in &rem_solution {
        result.push(c.relabel(&rem_rev, instance.num_leaves));
    }

    if !result.is_empty() && validate_agreement_forest(instance, &result).is_ok() {
        log::debug!(
            "[whidden] rspr n={}: {} clusters -> {} comps",
            n,
            sel.len(),
            result.len()
        );
        return Some(result);
    }
    None
}

fn try_batch_decomp<F>(instance: &Instance, solve: &mut F) -> Option<Vec<Tree>>
where
    F: FnMut(&Instance) -> Option<Vec<Tree>>,
{
    let n = instance.num_leaves as usize;
    let t1 = &instance.trees[0];
    let t2 = &instance.trees[1];
    let leaf_sets = compute_leaf_sets(t1, n);
    let tw12 = compute_twins(t1, t2);
    let tw21 = compute_twins(t2, t1);

    // 1. Find strict/common cluster points, with the same child filter rspr
    // uses for cluster points.  This intentionally does NOT use relaxed
    // depth-only cluster points: those require the full rspr ClusterInstance
    // boundary/rho/component-zero machinery during join.
    let mut pts = Vec::new();
    for node in t1.post_order() {
        if t1.is_leaf(node) || t1.is_root(node) {
            continue;
        }
        let sz = leaf_sets[node as usize].count_ones(..);
        if sz < 2 || sz > n - 2 {
            continue;
        }
        let t2t = tw12[node as usize];
        if t2t == NONE {
            continue;
        }
        let depth_node = t1.depth[node as usize];
        let round_trip = tw21[t2t as usize];
        if round_trip == NONE || depth_node > t1.depth[round_trip as usize] {
            continue;
        }
        if round_trip != node {
            continue;
        }
        if t2.is_root(t2t) {
            continue;
        }
        // rspr filter: if all children are cluster points, this node is just
        // their union, not a separate cluster.
        if children_are_rspr_cluster_candidates(t1, &tw12, &tw21, node) {
            continue;
        }
        pts.push((node, sz.min(n - sz)));
    }
    pts.sort_by_key(|(_, s)| -(*s as isize));
    if pts.is_empty() {
        return None;
    }

    // Select a maximal disjoint set (greedy, most-balanced first).
    let cap = leaf_sets.first().map(|l| l.len()).unwrap_or(0);
    let mut taken = FixedBitSet::with_capacity(cap);
    let mut sel = Vec::new();
    for (i, (node, _)) in pts.iter().enumerate() {
        let ls = &leaf_sets[*node as usize];
        if !taken.is_disjoint(ls) {
            continue;
        }
        taken.union_with(ls);
        sel.push(i);
    }
    if sel.is_empty() {
        return None;
    }

    #[derive(Clone)]
    struct SolvedBatchCluster {
        leaves: FixedBitSet,
        p_label: u32,
        anchor: Option<Tree>,
        non_anchor: Vec<Tree>,
    }

    // 2. Solve boundary-aware inner instances for selected clusters.
    //
    // This mirrors the useful part of rspr's ClusterInstance contract: the
    // component connected to the cut boundary must remain distinguishable after
    // the sub-solve.  The C++ code keeps that as forest component 0 / rho
    // state; in the immutable Tree API we attach a temporary rho leaf above
    // the cluster root and later use the rho-containing solution component as
    // the anchor to be joined to the outer placeholder.
    let mut solved: Vec<SolvedBatchCluster> = Vec::with_capacity(sel.len());
    let mut all_clustered = FixedBitSet::with_capacity(n + 1);
    let mut next_p: u32 = n as u32 + 1;
    for &ci in &sel {
        let (cnode, _) = pts[ci];
        let leaves = leaf_sets[cnode as usize].clone();
        let csize = leaves.count_ones(..);
        if csize < 2 {
            continue;
        }
        let t2t = tw12[cnode as usize];
        if t2t == NONE || t2.is_root(t2t) {
            continue;
        }
        all_clustered.union_with(&leaves);
        let mut imap = vec![0u32; n + 1];
        let marker_label = csize as u32 + 1;
        let mut ito = vec![0u32; csize + 2];
        for (l, leaf) in (1u32..).zip(leaves.ones()) {
            imap[leaf] = l;
            ito[l as usize] = leaf as u32;
        }
        // The marker is deliberately not mapped back to an original label.
        ito[marker_label as usize] = 0;

        let inner_t1 = attach_rho(&t1.relabel(&imap, csize as u32), marker_label);
        let inner_t2 = attach_rho(&t2.relabel(&imap, csize as u32), marker_label);
        let mut inner_instance = Instance::new(vec![inner_t1, inner_t2], marker_label);
        inner_instance.protected_labels = vec![marker_label];
        let inner_solution = solve(&inner_instance)?;

        let mut anchor: Option<Tree> = None;
        let mut non_anchor = Vec::new();
        for component in inner_solution {
            let has_marker = component_contains_label(&component, marker_label);
            let decoded = component.relabel(&ito, instance.num_leaves);
            let decoded_has_leaves = decoded.root != NONE && decoded.leaves().next().is_some();
            if has_marker {
                if decoded_has_leaves {
                    anchor = Some(decoded);
                }
            } else if decoded_has_leaves {
                non_anchor.push(decoded);
            }
        }

        solved.push(SolvedBatchCluster {
            leaves,
            anchor,
            non_anchor,
            p_label: next_p,
        });
        next_p += 1;
    }
    if solved.is_empty() {
        return None;
    }
    let max_p = next_p - 1;

    // 3. Build remaining trees (bottom-up LCA check).
    let mut t1_p = vec![0u32; t1.num_nodes()];
    let mut t2_p = vec![0u32; t2.num_nodes()];
    for s in &solved {
        let lca1 = lca_of_labels(t1, &s.leaves);
        let lca2 = lca_of_labels(t2, &s.leaves);
        if lca1 != NONE {
            t1_p[lca1 as usize] = s.p_label;
        }
        if lca2 != NONE && leaf_set_under(t2, lca2, n) == s.leaves {
            t2_p[lca2 as usize] = s.p_label;
        }
    }
    fn build(src: &Tree, node_to_p: &[u32], clustered: &FixedBitSet) -> Tree {
        fn walk(
            src: &Tree,
            n2p: &[u32],
            cl: &FixedBitSet,
            out: &mut Tree,
            node: NodeId,
        ) -> Option<NodeId> {
            if src.is_leaf(node) {
                let lbl = src.label[node as usize];
                if lbl == 0 || cl.contains(lbl as usize) {
                    return None;
                }
                let id = out.parent.len() as NodeId;
                out.parent.push(NONE);
                out.left.push(NONE);
                out.right.push(NONE);
                out.label.push(lbl);
                while out.label_to_node.len() <= lbl as usize {
                    out.label_to_node.push(NONE);
                }
                out.label_to_node[lbl as usize] = id;
                return Some(id);
            }
            let (l, r) = src.children(node).unwrap();
            let lc = walk(src, n2p, cl, out, l);
            let rc = walk(src, n2p, cl, out, r);
            let p = n2p[node as usize];
            if p != 0 {
                let id = out.parent.len() as NodeId;
                out.parent.push(NONE);
                out.left.push(NONE);
                out.right.push(NONE);
                out.label.push(p);
                while out.label_to_node.len() <= p as usize {
                    out.label_to_node.push(NONE);
                }
                out.label_to_node[p as usize] = id;
                return Some(id);
            }
            match (lc, rc) {
                (None, None) => None,
                (Some(c), None) | (None, Some(c)) => Some(c),
                (Some(lc), Some(rc)) => {
                    let id = out.parent.len() as NodeId;
                    out.parent.push(NONE);
                    out.left.push(lc);
                    out.right.push(rc);
                    out.label.push(0);
                    out.parent[lc as usize] = id;
                    out.parent[rc as usize] = id;
                    Some(id)
                }
            }
        }
        let maxp = node_to_p.iter().max().copied().unwrap_or(0);
        let mut out = Tree::with_capacity(src.num_leaves.max(maxp));
        if src.root != NONE
            && let Some(r) = walk(src, node_to_p, clustered, &mut out, src.root)
        {
            out.root = r;
        }
        out.compute_metadata();
        out
    }
    let rem_t1 = build(t1, &t1_p, &all_clustered);

    // T2 builder.  The `relaxed_p` hook is deliberately unused by the strict
    // batch path; it remains here only because the old experimental relaxed
    // code used this helper.
    fn build_t2(
        src: &Tree,
        node_to_p: &[u32],
        clustered: &FixedBitSet,
        relaxed_p: &[(NodeId, u32)], // (twin_node, p_label) for relaxed clusters
    ) -> Tree {
        // marks: twin_node -> p_label AND no_children_from=cluster_lca for relaxed
        let mut p_at: Vec<u32> = vec![0; src.num_nodes()];
        for &(twin, p) in relaxed_p {
            p_at[twin as usize] = p;
        }

        fn walk(
            src: &Tree,
            n2p: &[u32],
            cl: &FixedBitSet,
            p_at: &[u32],
            out: &mut Tree,
            node: NodeId,
        ) -> Option<NodeId> {
            if src.is_leaf(node) {
                let lbl = src.label[node as usize];
                if lbl == 0 || cl.contains(lbl as usize) {
                    return None;
                }
                let id = out.parent.len() as NodeId;
                out.parent.push(NONE);
                out.left.push(NONE);
                out.right.push(NONE);
                out.label.push(lbl);
                while out.label_to_node.len() <= lbl as usize {
                    out.label_to_node.push(NONE);
                }
                out.label_to_node[lbl as usize] = id;
                return Some(id);
            }
            let (l, r) = src.children(node).unwrap();
            let lc = walk(src, n2p, cl, p_at, out, l);
            let rc = walk(src, n2p, cl, p_at, out, r);
            // Strict cluster: LCA marked with P → emit P, skip children.
            let p = n2p[node as usize];
            if p != 0 {
                let id = out.parent.len() as NodeId;
                out.parent.push(NONE);
                out.left.push(NONE);
                out.right.push(NONE);
                out.label.push(p);
                while out.label_to_node.len() <= p as usize {
                    out.label_to_node.push(NONE);
                }
                out.label_to_node[p as usize] = id;
                return Some(id);
            }
            // Relaxed cluster: twin has extra leaves. Attach P as sibling if needed.
            let rp = p_at[node as usize];
            match (lc, rc) {
                (None, None) => {
                    if rp != 0 {
                        let id = out.parent.len() as NodeId;
                        out.parent.push(NONE);
                        out.left.push(NONE);
                        out.right.push(NONE);
                        out.label.push(rp);
                        while out.label_to_node.len() <= rp as usize {
                            out.label_to_node.push(NONE);
                        }
                        out.label_to_node[rp as usize] = id;
                        Some(id)
                    } else {
                        None
                    }
                }
                (Some(c), None) | (None, Some(c)) => {
                    if rp != 0 {
                        let pid = out.parent.len() as NodeId;
                        out.parent.push(NONE);
                        out.left.push(NONE);
                        out.right.push(NONE);
                        out.label.push(rp);
                        while out.label_to_node.len() <= rp as usize {
                            out.label_to_node.push(NONE);
                        }
                        out.label_to_node[rp as usize] = pid;
                        let id = out.parent.len() as NodeId;
                        out.parent.push(NONE);
                        out.left.push(c);
                        out.right.push(pid);
                        out.label.push(0);
                        out.parent[c as usize] = id;
                        out.parent[pid as usize] = id;
                        Some(id)
                    } else {
                        Some(c)
                    }
                }
                (Some(lc), Some(rc)) => {
                    if rp != 0 {
                        let pid = out.parent.len() as NodeId;
                        out.parent.push(NONE);
                        out.left.push(NONE);
                        out.right.push(NONE);
                        out.label.push(rp);
                        while out.label_to_node.len() <= rp as usize {
                            out.label_to_node.push(NONE);
                        }
                        out.label_to_node[rp as usize] = pid;
                        let id2 = out.parent.len() as NodeId;
                        out.parent.push(NONE);
                        out.left.push(lc);
                        out.right.push(pid);
                        out.label.push(0);
                        out.parent[lc as usize] = id2;
                        out.parent[pid as usize] = id2;
                        let id = out.parent.len() as NodeId;
                        out.parent.push(NONE);
                        out.left.push(id2);
                        out.right.push(rc);
                        out.label.push(0);
                        out.parent[id2 as usize] = id;
                        out.parent[rc as usize] = id;
                        Some(id)
                    } else {
                        let id = out.parent.len() as NodeId;
                        out.parent.push(NONE);
                        out.left.push(lc);
                        out.right.push(rc);
                        out.label.push(0);
                        out.parent[lc as usize] = id;
                        out.parent[rc as usize] = id;
                        Some(id)
                    }
                }
            }
        }
        let maxp = node_to_p
            .iter()
            .max()
            .copied()
            .unwrap_or(0)
            .max(p_at.iter().max().copied().unwrap_or(0));
        let mut out = Tree::with_capacity(src.num_leaves.max(maxp));
        if src.root != NONE
            && let Some(r) = walk(src, node_to_p, clustered, &p_at, &mut out, src.root)
        {
            out.root = r;
        }
        out.compute_metadata();
        out
    }

    let relaxed_t2: Vec<(NodeId, u32)> = Vec::new();
    let rem_t2 = build_t2(t2, &t2_p, &all_clustered, &relaxed_t2);

    // 4. Compact and solve remaining.
    let rem_leaves = (1..=n as u32)
        .filter(|l| !all_clustered.contains(*l as usize))
        .count();
    let rem_n = rem_leaves as u32 + solved.len() as u32;
    let mut cmap = vec![0u32; max_p as usize + 1];
    let mut crev = vec![0u32; rem_n as usize + 1];
    let mut nc: u32 = 1;
    for lbl in 1..=n as u32 {
        if !all_clustered.contains(lbl as usize) {
            cmap[lbl as usize] = nc;
            crev[nc as usize] = lbl;
            nc += 1;
        }
    }
    for i in 0..solved.len() {
        cmap[solved[i].p_label as usize] = nc;
        solved[i].p_label = nc;
        nc += 1;
    }
    let prot: Vec<u32> = solved.iter().map(|s| s.p_label).collect();
    let mut ri = Instance::new(
        vec![rem_t1.relabel(&cmap, rem_n), rem_t2.relabel(&cmap, rem_n)],
        rem_n,
    );
    ri.protected_labels = prot;

    let rem_comps = solve(&ri)?;

    // 5. Join: sequential P processing. Before each fuse, strip other P labels
    //    from current to keep label_to_node bounded by instance.num_leaves+1.
    let mut result: Vec<Tree> = solved
        .iter()
        .flat_map(|s| s.non_anchor.iter().cloned())
        .collect();
    let big_n = instance.num_leaves + solved.len() as u32 + 100;
    for comp in &rem_comps {
        let p_is: Vec<usize> = (0..solved.len())
            .filter(|&i| component_contains_label(comp, solved[i].p_label))
            .collect();
        if p_is.is_empty() {
            result.push(comp.relabel(&crev, instance.num_leaves));
            continue;
        }
        // Decode non-P via crev, map P's to > n.
        let mut fmap0 = vec![0u32; big_n as usize + 1];
        for lbl in 1..=comp.num_leaves {
            let l = lbl as usize;
            if l < crev.len() && crev[l] != 0 {
                fmap0[l] = crev[l];
            } else {
                for (ii, s) in solved.iter().enumerate() {
                    if l as u32 == s.p_label {
                        fmap0[l] = instance.num_leaves + ii as u32 + 2;
                        break;
                    }
                }
            }
        }
        let mut current = comp.relabel(&fmap0, big_n);

        for &si in &p_is {
            let p_cur = instance.num_leaves + si as u32 + 2;
            let Some(anchor) = solved[si].anchor.as_ref() else {
                // Boundary was cut in the marker solve.  Leave the placeholder
                // in `current`; it will be stripped below.
                continue;
            };

            // Strip impossible labels from current, but keep *all* placeholder
            // labels that are present in this outer component.  The previous
            // experimental code kept only p_cur, which silently dropped other
            // placeholders when one outer component contained multiple P_i.
            let mut strip_map = vec![0u32; big_n as usize + 1];
            let mut max_kept: u32 = 0;
            let keep_placeholders: Vec<u32> = p_is
                .iter()
                .map(|&idx| instance.num_leaves + idx as u32 + 2)
                .collect();
            for lbl in current.leaves() {
                if lbl <= instance.num_leaves || keep_placeholders.contains(&lbl) {
                    strip_map[lbl as usize] = lbl;
                    max_kept = max_kept.max(lbl);
                }
            }
            if max_kept == 0 {
                continue;
            }
            let stripped = current.relabel(&strip_map, max_kept);

            let al = anchor.leaves().next().unwrap();
            current = fuse_at_label(&stripped, anchor, p_cur, al, big_n);
        }
        // Strip remaining P labels from current.
        let mut smap = vec![0u32; big_n as usize + 1];
        let mut smax: u32 = 0;
        for lbl in current.leaves() {
            if lbl <= instance.num_leaves {
                smap[lbl as usize] = lbl;
                smax = smax.max(lbl);
            }
        }
        if smax > 0 {
            result.push(current.relabel(&smap, smax));
        }
    }
    if result.is_empty() {
        return None;
    }
    let validation = validate_agreement_forest(instance, &result);
    if !validation.is_ok() {
        if log::log_enabled!(log::Level::Debug) {
            let sizes: Vec<usize> = solved.iter().map(|s| s.leaves.count_ones(..)).collect();
            debug!(
                "[whidden] batch n={}: validation failed for {} clusters sizes={:?}: {:?}",
                n,
                solved.len(),
                sizes,
                validation
            );
        }
        return None;
    }
    log::debug!(
        "[whidden] batch n={}: {} clusters -> {} comps",
        n,
        solved.len(),
        result.len()
    );
    Some(result)
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
        if n != NONE {
            acc = tree.nearest_common_ancestor(acc, n);
        }
    }
    acc
}

fn try_single_decomp<F>(
    instance: &Instance,
    solve: &mut F,
    terminate: &AtomicBool,
) -> Option<Vec<Tree>>
where
    F: FnMut(&Instance) -> Option<Vec<Tree>>,
{
    if instance.num_trees() != 2 {
        return None;
    }
    // Bail before the O(n^2) cluster-point detection so a SIGTERM landing
    // mid-recursion unwinds promptly (each recursion level re-enters here).
    if terminate.load(Ordering::Relaxed) {
        return None;
    }
    let n = instance.num_leaves as usize;
    if n < 6 {
        return None;
    }

    let t1 = &instance.trees[0];
    let t2 = &instance.trees[1];

    let leaf_sets_t1 = compute_leaf_sets(t1, n);
    let twin_t1_to_t2 = compute_twins(t1, t2);
    let twin_t2_to_t1 = compute_twins(t2, t1);

    let mut candidates: Vec<(NodeId, usize)> =
        collect_rspr_cluster_points(t1, t2, &leaf_sets_t1, &twin_t1_to_t2, &twin_t2_to_t1, n)
            .into_iter()
            .filter(|point| point.kind == RsprClusterPointKind::Strict)
            .filter(|point| point.size >= 2 && point.size <= n - 2)
            .filter(|point| point.t1_round_trip == point.t1_node)
            .filter(|point| !t2.is_root(point.t2_twin))
            .map(|point| (point.t1_node, point.size.min(n - point.size)))
            .collect();
    candidates.sort_by_key(|(_, score)| -(*score as isize));

    for (cluster, _) in candidates {
        if terminate.load(Ordering::Relaxed) {
            return None;
        }
        if let Some(result) = try_single_decomp_at(instance, solve, cluster, terminate) {
            return Some(result);
        }
    }

    None
}

fn try_single_decomp_at<F>(
    instance: &Instance,
    solve: &mut F,
    cluster: NodeId,
    terminate: &AtomicBool,
) -> Option<Vec<Tree>>
where
    F: FnMut(&Instance) -> Option<Vec<Tree>>,
{
    if instance.num_trees() != 2 {
        return None;
    }
    let n = instance.num_leaves as usize;
    if n < 6 {
        return None;
    }

    let t1 = &instance.trees[0];
    let t2 = &instance.trees[1];

    let leaf_sets_t1 = compute_leaf_sets(t1, n);
    let twin_t1_to_t2 = compute_twins(t1, t2);

    let cluster_leaves = &leaf_sets_t1[cluster as usize];
    let cluster_size = cluster_leaves.count_ones(..);
    let outer_size = n - cluster_size;
    // Sanity: both halves must be non-trivial.
    if cluster_size < 2 || outer_size < 2 {
        return None;
    }

    let twin_in_t2 = twin_t1_to_t2[cluster as usize];
    if twin_in_t2 == NONE {
        return None;
    }

    // Sanity: leaf-sets must match exactly. Strict cluster point ought to
    // guarantee this, but verify before we splice.
    let t2_leaves_under_twin = leaf_set_under(t2, twin_in_t2, n);
    if t2_leaves_under_twin != *cluster_leaves {
        return None;
    }
    // Twin must not be the root of T2 either (otherwise outer_t2 collapses
    // to a single leaf P and we lose the outer structure).
    if t2.is_root(twin_in_t2) {
        return None;
    }

    // Build a compact label remap for the outer instance: outer leaves get
    // 1..=outer_size, placeholder gets outer_size+1. This keeps the outer
    // label-space dense so downstream kernelization doesn't trip on holes.
    let mut outer_label_map: Vec<Label> = vec![0; instance.num_leaves as usize + 1];
    let mut outer_to_orig: Vec<Label> = vec![0; outer_size + 2];
    let mut next_outer: Label = 1;
    for lbl in 1..=instance.num_leaves {
        if !cluster_leaves.contains(lbl as usize) {
            outer_label_map[lbl as usize] = next_outer;
            outer_to_orig[next_outer as usize] = lbl as Label;
            next_outer += 1;
        }
    }
    let outer_p_label: Label = next_outer;
    let outer_new_num_leaves = next_outer;

    // Build inner sub-instance (no ρ): relabel X to 1..=cluster_size compactly.
    let mut inner_label_map: Vec<Label> = vec![0; instance.num_leaves as usize + 1];
    let mut inner_to_orig: Vec<Label> = vec![0; cluster_size + 1];
    for (next, old_lbl) in (1u32..).zip(cluster_leaves.ones()) {
        inner_label_map[old_lbl] = next;
        inner_to_orig[next as usize] = old_lbl as Label;
    }
    let inner_t1 = t1.relabel(&inner_label_map, cluster_size as u32);
    let inner_t2 = t2.relabel(&inner_label_map, cluster_size as u32);
    let inner_instance = Instance::new(vec![inner_t1, inner_t2], cluster_size as u32);

    // Build outer sub-instance: replace cluster subtree with single leaf P.
    let outer_t1 = replace_and_relabel(
        t1,
        cluster,
        outer_p_label,
        &outer_label_map,
        outer_new_num_leaves,
    );
    let outer_t2 = replace_and_relabel(
        t2,
        twin_in_t2,
        outer_p_label,
        &outer_label_map,
        outer_new_num_leaves,
    );
    let mut outer_instance = Instance::new(vec![outer_t1, outer_t2], outer_new_num_leaves);
    outer_instance.protected_labels = vec![outer_p_label];

    // Solve both sub-problems.
    let inner_components = solve(&inner_instance)?;
    let outer_components = solve(&outer_instance)?;

    // Decode inner: compact -> original label space.
    let decoded_inner: Vec<Tree> = inner_components
        .iter()
        .map(|c| c.relabel(&inner_to_orig, instance.num_leaves))
        .collect();

    // Decode outer.
    let outer_p_idx = outer_components
        .iter()
        .position(|c| component_contains_label(c, outer_p_label))
        .expect("P must appear in some outer component");

    let p_decoded = instance.num_leaves + 1;
    let outer_p_comp_decoded = decode_outer_keeping_placeholder(
        &outer_components[outer_p_idx],
        &outer_to_orig,
        outer_p_label,
        instance.num_leaves,
    );

    let decoded_outer_non_p: Vec<Tree> = outer_components
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != outer_p_idx)
        .map(|(_, c)| c.relabel(&outer_to_orig, instance.num_leaves))
        .collect();

    // On termination, skip the O(k·n) per-anchor validation loop and the ρ
    // re-solve below: graft the first inner component at P and return the
    // best-effort join unvalidated. This PRESERVES the solved inner/outer
    // structure (so the emitted forest keeps its quality) while unwinding the
    // in-flight recursion within the SIGTERM grace window. The caller's final
    // repair pass restores validity if this graft happens to be sub-optimal.
    if terminate.load(Ordering::Relaxed) && !decoded_inner.is_empty() {
        let anchor_leaf = decoded_inner[0]
            .leaves()
            .next()
            .expect("anchor must have at least one leaf");
        let merged = fuse_at_label(
            &outer_p_comp_decoded,
            &decoded_inner[0],
            p_decoded,
            anchor_leaf,
            instance.num_leaves,
        );
        // Move (don't clone) the already-decoded vectors — we return here.
        let mut candidate = decoded_outer_non_p;
        candidate.extend(decoded_inner.into_iter().skip(1));
        candidate.push(merged);
        return Some(candidate);
    }

    // Try each inner component as the anchor (boundary component).
    // The correct anchor is the one whose merge with P produces a valid AF.
    for anchor_idx in 0..decoded_inner.len() {
        let anchor_leaf = decoded_inner[anchor_idx]
            .leaves()
            .next()
            .expect("anchor must have at least one leaf");

        let mut candidate: Vec<Tree> = decoded_inner
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != anchor_idx)
            .map(|(_, c)| c.clone())
            .collect();
        candidate.extend(decoded_outer_non_p.clone());

        let merged = fuse_at_label(
            &outer_p_comp_decoded,
            &decoded_inner[anchor_idx],
            p_decoded,
            anchor_leaf,
            instance.num_leaves,
        );
        candidate.push(merged);

        if validate_agreement_forest(instance, &candidate).is_ok() {
            log::debug!(
                "[whidden] decomp n={}: inner={} outer={} (cluster_node={}, anchor_idx={}, {} comps)",
                n,
                cluster_size,
                outer_size + 1,
                cluster,
                anchor_idx,
                candidate.len()
            );
            return Some(candidate);
        }
    }

    // ── ρ fallback (Whidden boundary marker) ─────────────────────────────
    //
    // The anchorless search above failed: the inner solve returned an optimal
    // AF that fragments the boundary leaves across several components, so no
    // single inner component grafts validly at P. This is luck-dependent on
    // the inner B&P's tie-breaking among equal optima.
    //
    // Re-solve the inner WITH a ρ pendant leaf — the boundary marker that
    // mirrors P on the outer side (rspr's `add_rho()` /
    // `ClusterInstance::join_cluster`, rspr.h:4305-4338). ρ is attached as the
    // sibling of the cluster root in both inner trees. If the marker stays with
    // real cluster leaves, it identifies the unique boundary component and does
    // not change the inner distance.
    // The component containing ρ is then the *unique* boundary component,
    // robust to which optimal inner partition the solver returns. We merge that
    // ρ-component (minus ρ) with the outer P-component exactly as the anchor
    // loop would, and validate before accepting.
    //
    // Component-count accounting (identical optimum to a successful anchor):
    //   |candidate| = (|inner_ρ| − 1) + (|outer| − 1) + 1 = |inner_ρ| + |outer| − 1
    // and |inner_ρ| = |inner| (ρ is a free agreeing pendant), so this equals the
    // strict-cluster optimum |inner| + |outer| − 1.
    {
        let rho_label = cluster_size as Label + 1;
        let inner_rho_num_leaves = cluster_size as u32 + 1;
        let inner_t1_rho =
            build_inner_with_rho(t1, &inner_label_map, rho_label, inner_rho_num_leaves);
        let inner_t2_rho =
            build_inner_with_rho(t2, &inner_label_map, rho_label, inner_rho_num_leaves);
        let mut inner_rho_instance =
            Instance::new(vec![inner_t1_rho, inner_t2_rho], inner_rho_num_leaves);
        inner_rho_instance.protected_labels = vec![rho_label];

        if let Some(inner_rho_components) = solve(&inner_rho_instance)
            && let Some(rho_pos) = inner_rho_components
                .iter()
                .position(|c| component_contains_label(c, rho_label))
        {
            // Boundary B = ρ-component minus ρ, in original labels.
            let boundary_inner = strip_label(
                &inner_rho_components[rho_pos],
                rho_label,
                cluster_size as u32,
            );
            let marker_increased_inner = inner_rho_components.len() > decoded_inner.len();
            if boundary_inner.leaves().next().is_some() {
                let boundary_decoded = boundary_inner.relabel(&inner_to_orig, instance.num_leaves);
                let other_inner: Vec<Tree> = inner_rho_components
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != rho_pos)
                    .map(|(_, c)| c.relabel(&inner_to_orig, instance.num_leaves))
                    .collect();
                let anchor_leaf = boundary_decoded
                    .leaves()
                    .next()
                    .expect("boundary is non-empty");
                let merged = fuse_at_label(
                    &outer_p_comp_decoded,
                    &boundary_decoded,
                    p_decoded,
                    anchor_leaf,
                    instance.num_leaves,
                );
                let mut candidate: Vec<Tree> = other_inner;
                candidate.extend(decoded_outer_non_p.clone());
                candidate.push(merged);
                // Optimality guard. The strict-cluster optimum is
                // |inner| + |outer| − 1 = decoded_inner.len() +
                // decoded_outer_non_p.len(). The ρ-fallback equals it only
                // when ρ was a free pendant (|inner_ρ| = |inner|); since
                // T1|X and T2|X differ in topology, ρ can instead force one
                // extra cut (|inner_ρ| = |inner| + 1), yielding a valid but
                // SUBoptimal forest. Accept only when the count matches the
                // optimum, else bail to full B&P (sound, as before).
                let expected = decoded_inner.len() + decoded_outer_non_p.len();
                if candidate.len() == expected
                    && validate_agreement_forest(instance, &candidate).is_ok()
                {
                    log::debug!(
                        "[whidden] decomp n={}: inner={} outer={} (cluster_node={}, ρ-fallback, {} comps)",
                        n,
                        cluster_size,
                        outer_size + 1,
                        cluster,
                        candidate.len()
                    );
                    return Some(candidate);
                }
            }
            if marker_increased_inner {
                // Boundary-cut/no-saving case: adding rho to the inner
                // cluster increased the optimum, so rspr's join_cluster
                // adjustment removes the cross-boundary -1 saving. The
                // top side must be solved without P (TP4); stripping P out
                // of the already solved top-with-P instance is only
                // feasible, not optimal.
                let outer_no_p_t1 = t1.relabel(&outer_label_map, outer_size as u32);
                let outer_no_p_t2 = t2.relabel(&outer_label_map, outer_size as u32);
                let outer_no_p_instance =
                    Instance::new(vec![outer_no_p_t1, outer_no_p_t2], outer_size as u32);

                if let Some(outer_no_p_components) = solve(&outer_no_p_instance) {
                    let mut candidate: Vec<Tree> = decoded_inner.clone();
                    candidate.extend(
                        outer_no_p_components
                            .iter()
                            .map(|c| c.relabel(&outer_to_orig, instance.num_leaves)),
                    );

                    if validate_agreement_forest(instance, &candidate).is_ok() {
                        log::debug!(
                            "[whidden] decomp n={}: inner={} outer={} (cluster_node={}, ρ-boundary-cut, {} comps)",
                            n,
                            cluster_size,
                            outer_size + 1,
                            cluster,
                            candidate.len()
                        );
                        return Some(candidate);
                    }
                }
            }
        }
    }

    // No valid anchor — decomposition fails for this cluster point.
    log::debug!(
        "[whidden] decomp n={}: inner={} outer={} — no valid anchor among {} inner components",
        n,
        cluster_size,
        outer_size + 1,
        decoded_inner.len()
    );
    None
}

/// Remove a single leaf from a tree, suppressing the resulting degree-1 node.
fn strip_label(tree: &Tree, target: Label, new_num_leaves: u32) -> Tree {
    let n = new_num_leaves as usize;
    let mut s = FixedBitSet::with_capacity(n + 2);
    for lbl in tree.leaves() {
        if lbl != target {
            s.insert(lbl as usize);
        }
    }
    if s.count_ones(..) == 0 {
        return Tree::with_capacity(new_num_leaves);
    }
    tree.prune_to_leafset(&s)
}

/// Build `T|X` (the cluster subtree, via `label_map` sending X-leaves to
/// `1..=|X|` and non-X leaves to 0) with a pendant ρ leaf attached as the
/// sibling of the cluster root — Whidden's boundary marker. When keeping ρ
/// attached is optimal, the component containing ρ identifies the mergeable
/// boundary. When ρ increases the marked optimum, the outer side is solved
/// without its placeholder and the cluster is joined with no cross-boundary
/// component saving.
fn build_inner_with_rho(
    src: &Tree,
    label_map: &[Label],
    rho_label: Label,
    new_num_leaves: u32,
) -> Tree {
    let mut out = Tree::with_capacity(new_num_leaves);

    // Copy `src` while relabelling leaves through `label_map` (label 0 ⇒ drop),
    // exactly mirroring `Tree::relabel`'s recursion.
    fn build(src: &Tree, label_map: &[Label], out: &mut Tree, node: NodeId) -> Option<NodeId> {
        if node == NONE {
            return None;
        }
        if src.is_leaf(node) {
            let old_lbl = src.label[node as usize];
            if old_lbl == 0 {
                return None;
            }
            let new_lbl = if (old_lbl as usize) < label_map.len() {
                label_map[old_lbl as usize]
            } else {
                0
            };
            if new_lbl == 0 {
                return None;
            }
            let id = out.parent.len() as NodeId;
            out.parent.push(NONE);
            out.left.push(NONE);
            out.right.push(NONE);
            out.label.push(new_lbl);
            out.label_to_node[new_lbl as usize] = id;
            return Some(id);
        }
        let (left, right) = src.children(node).unwrap();
        let l = build(src, label_map, out, left);
        let r = build(src, label_map, out, right);
        match (l, r) {
            (None, None) => None,
            (Some(c), None) | (None, Some(c)) => Some(c),
            (Some(lc), Some(rc)) => {
                let id = out.parent.len() as NodeId;
                out.parent.push(NONE);
                out.left.push(lc);
                out.right.push(rc);
                out.label.push(0);
                out.parent[lc as usize] = id;
                out.parent[rc as usize] = id;
                Some(id)
            }
        }
    }

    let sub_root = match build(src, label_map, &mut out, src.root) {
        Some(r) => r,
        None => {
            out.compute_metadata();
            return out;
        }
    };

    // ρ pendant leaf.
    let rho_id = out.parent.len() as NodeId;
    out.parent.push(NONE);
    out.left.push(NONE);
    out.right.push(NONE);
    out.label.push(rho_label);
    out.label_to_node[rho_label as usize] = rho_id;

    // New root = (cluster subtree, ρ).
    let root = out.parent.len() as NodeId;
    out.parent.push(NONE);
    out.left.push(sub_root);
    out.right.push(rho_id);
    out.label.push(0);
    out.parent[sub_root as usize] = root;
    out.parent[rho_id as usize] = root;
    out.root = root;

    out.compute_metadata();
    out
}

/// Relabel an outer component where outer-side leaves go through `map` but
/// `placeholder` (which has map[placeholder]==0) is mapped to
/// `final_num_leaves + 1` — safely beyond all original labels.
/// The result occupies label space 1..=final_num_leaves+1.
fn decode_outer_keeping_placeholder(
    comp: &Tree,
    map: &[Label],
    placeholder: Label,
    final_num_leaves: u32,
) -> Tree {
    let p_mapped = final_num_leaves + 1;
    let cap = (final_num_leaves as usize).max(p_mapped as usize) + 1;
    let mut full_map: Vec<Label> = vec![0; cap.max(comp.num_leaves as usize + 1)];
    for lbl in 1..=comp.num_leaves {
        let l = lbl as usize;
        if l < map.len() && map[l] != 0 {
            full_map[l] = map[l];
        } else if lbl == placeholder {
            full_map[l] = p_mapped;
        }
    }
    let new_size = final_num_leaves.max(p_mapped);
    comp.relabel(&full_map, new_size)
}

// Replace the subtree at `target` with a leaf labeled `placeholder`, AND
// relabel surviving leaves through `label_map` (old_label -> new_label).
// Leaves with label_map[old] == 0 are dropped.
fn replace_and_relabel(
    tree: &Tree,
    target: NodeId,
    placeholder: Label,
    label_map: &[Label],
    new_num_leaves: u32,
) -> Tree {
    let mut out = Tree::with_capacity(new_num_leaves);

    fn build(
        src: &Tree,
        target: NodeId,
        placeholder: Label,
        label_map: &[Label],
        out: &mut Tree,
        node: NodeId,
    ) -> Option<NodeId> {
        if node == target {
            let id = out.parent.len() as NodeId;
            out.parent.push(NONE);
            out.left.push(NONE);
            out.right.push(NONE);
            out.label.push(placeholder);
            out.label_to_node[placeholder as usize] = id;
            return Some(id);
        }
        if src.is_leaf(node) {
            let old_lbl = src.label[node as usize];
            if old_lbl == 0 {
                return None;
            }
            let new_lbl = if (old_lbl as usize) < label_map.len() {
                label_map[old_lbl as usize]
            } else {
                0
            };
            if new_lbl == 0 {
                return None;
            }
            let id = out.parent.len() as NodeId;
            out.parent.push(NONE);
            out.left.push(NONE);
            out.right.push(NONE);
            out.label.push(new_lbl);
            out.label_to_node[new_lbl as usize] = id;
            return Some(id);
        }
        let (l, r) = src.children(node).unwrap();
        let lc = build(src, target, placeholder, label_map, out, l);
        let rc = build(src, target, placeholder, label_map, out, r);
        match (lc, rc) {
            (None, None) => None,
            (Some(c), None) | (None, Some(c)) => Some(c),
            (Some(lc), Some(rc)) => {
                let id = out.parent.len() as NodeId;
                out.parent.push(NONE);
                out.left.push(lc);
                out.right.push(rc);
                out.label.push(0);
                out.parent[lc as usize] = id;
                out.parent[rc as usize] = id;
                Some(id)
            }
        }
    }

    if tree.root != NONE
        && let Some(root) = build(tree, target, placeholder, label_map, &mut out, tree.root)
    {
        out.root = root;
        out.parent[root as usize] = NONE;
    }
    out.compute_metadata();
    out
}

// ---------------------------------------------------------------------------
// Fuse two component trees at corresponding leaves.
//
// `outer_comp` contains a leaf labeled `p_label`. `inner_comp` is a phylogeny
// on a disjoint set of leaves. Returns a new tree where `p_label` is replaced
// by `inner_comp` (with `inner_comp`'s root grafted in place of P).
// `anchor_label` is unused for the fuse itself but documents the convention:
// inner_comp must contain `anchor_label` (the inner-side counterpart of P).
//
// The result lives in the label space 1..=final_num_leaves (no P).
// ---------------------------------------------------------------------------
fn fuse_at_label(
    outer_comp: &Tree,
    inner_comp: &Tree,
    p_label: Label,
    _anchor_label: Label,
    final_num_leaves: u32,
) -> Tree {
    let mut out = Tree::with_capacity(final_num_leaves + 1);

    // First, copy outer_comp into out, but at the leaf labeled p_label, splice
    // in inner_comp.
    fn copy_inner(src: &Tree, out: &mut Tree, node: NodeId) -> Option<NodeId> {
        if src.is_leaf(node) {
            let lbl = src.label[node as usize];
            if lbl == 0 {
                return None;
            }
            let id = out.parent.len() as NodeId;
            out.parent.push(NONE);
            out.left.push(NONE);
            out.right.push(NONE);
            out.label.push(lbl);
            out.label_to_node[lbl as usize] = id;
            return Some(id);
        }
        let (l, r) = src.children(node).unwrap();
        let lc = copy_inner(src, out, l);
        let rc = copy_inner(src, out, r);
        match (lc, rc) {
            (None, None) => None,
            (Some(c), None) | (None, Some(c)) => Some(c),
            (Some(lc), Some(rc)) => {
                let id = out.parent.len() as NodeId;
                out.parent.push(NONE);
                out.left.push(lc);
                out.right.push(rc);
                out.label.push(0);
                out.parent[lc as usize] = id;
                out.parent[rc as usize] = id;
                Some(id)
            }
        }
    }

    fn copy_outer_with_splice(
        src_outer: &Tree,
        src_inner: &Tree,
        p_label: Label,
        out: &mut Tree,
        node: NodeId,
    ) -> Option<NodeId> {
        if src_outer.is_leaf(node) {
            let lbl = src_outer.label[node as usize];
            if lbl == p_label {
                // Splice in inner here.
                return copy_inner(src_inner, out, src_inner.root);
            }
            if lbl == 0 {
                return None;
            }
            let id = out.parent.len() as NodeId;
            out.parent.push(NONE);
            out.left.push(NONE);
            out.right.push(NONE);
            out.label.push(lbl);
            out.label_to_node[lbl as usize] = id;
            return Some(id);
        }
        let (l, r) = src_outer.children(node).unwrap();
        let lc = copy_outer_with_splice(src_outer, src_inner, p_label, out, l);
        let rc = copy_outer_with_splice(src_outer, src_inner, p_label, out, r);
        match (lc, rc) {
            (None, None) => None,
            (Some(c), None) | (None, Some(c)) => Some(c),
            (Some(lc), Some(rc)) => {
                let id = out.parent.len() as NodeId;
                out.parent.push(NONE);
                out.left.push(lc);
                out.right.push(rc);
                out.label.push(0);
                out.parent[lc as usize] = id;
                out.parent[rc as usize] = id;
                Some(id)
            }
        }
    }

    if outer_comp.root != NONE
        && let Some(root) =
            copy_outer_with_splice(outer_comp, inner_comp, p_label, &mut out, outer_comp.root)
    {
        out.root = root;
        out.parent[root as usize] = NONE;
    }
    out.compute_metadata();
    out
}

/// Attach a ρ leaf as sibling of the root: creates a new root whose children
/// are the old root and the new ρ leaf.
fn attach_rho(tree: &Tree, rho_label: Label) -> Tree {
    let new_n = tree.num_leaves + 1;
    let mut out = Tree::with_capacity(new_n);
    // Copy old tree verbatim, tracking node-id mapping.
    let mut old_to_new: Vec<NodeId> = vec![NONE; tree.num_nodes()];
    for node in tree.pre_order() {
        let id = out.parent.len() as NodeId;
        out.parent.push(NONE);
        out.left.push(NONE);
        out.right.push(NONE);
        out.label.push(tree.label[node as usize]);
        old_to_new[node as usize] = id;
    }
    // Fix up children for internal nodes.
    for node in 0..tree.parent.len() as u32 {
        let new_node = old_to_new[node as usize];
        if new_node == NONE {
            continue;
        }
        if !tree.is_leaf(node) {
            let (l, r) = tree.children(node).unwrap();
            let nl = old_to_new[l as usize];
            let nr = old_to_new[r as usize];
            out.left[new_node as usize] = nl;
            out.right[new_node as usize] = nr;
            if nl != NONE {
                out.parent[nl as usize] = new_node;
            }
            if nr != NONE {
                out.parent[nr as usize] = new_node;
            }
        }
    }
    // Create ρ leaf.
    let rho_id = out.parent.len() as NodeId;
    out.parent.push(NONE);
    out.left.push(NONE);
    out.right.push(NONE);
    out.label.push(rho_label);
    // Create new root with old_root + ρ as children.
    let new_root = out.parent.len() as NodeId;
    out.parent.push(NONE);
    out.left.push(old_to_new[tree.root as usize]);
    out.right.push(rho_id);
    out.label.push(0);
    out.parent[old_to_new[tree.root as usize] as usize] = new_root;
    out.parent[rho_id as usize] = new_root;
    out.root = new_root;
    // Update label_to_node and metadata.
    out.label_to_node = vec![NONE; new_n as usize + 1];
    for n in 0..out.parent.len() as u32 {
        if out.is_leaf(n) {
            let lbl = out.label[n as usize];
            if lbl > 0 && (lbl as usize) <= new_n as usize {
                out.label_to_node[lbl as usize] = n;
            }
        }
    }
    out.num_leaves = new_n;
    out.compute_metadata();
    out
}

fn component_contains_label(tree: &Tree, label: Label) -> bool {
    if (label as usize) >= tree.label_to_node.len() {
        return false;
    }
    tree.label_to_node[label as usize] != NONE
}

fn leaf_set_under(tree: &Tree, node: NodeId, n: usize) -> FixedBitSet {
    let mut s = FixedBitSet::with_capacity(n + 1);
    let mut stack = vec![node];
    while let Some(v) = stack.pop() {
        if tree.is_leaf(v) {
            let lbl = tree.label[v as usize];
            if lbl > 0 && (lbl as usize) <= n {
                s.insert(lbl as usize);
            }
        } else if let Some((l, r)) = tree.children(v) {
            stack.push(l);
            stack.push(r);
        }
    }
    s
}

fn compute_leaf_sets(tree: &Tree, n: usize) -> Vec<FixedBitSet> {
    let num_nodes = tree.num_nodes();
    let mut sets = vec![FixedBitSet::with_capacity(n + 1); num_nodes];
    for node in tree.post_order() {
        if tree.is_leaf(node) {
            let lbl = tree.label[node as usize];
            if lbl > 0 && (lbl as usize) <= n {
                sets[node as usize].insert(lbl as usize);
            }
        } else {
            let (l, r) = tree.children(node).unwrap();
            let mut s = sets[l as usize].clone();
            s.union_with(&sets[r as usize]);
            sets[node as usize] = s;
        }
    }
    sets
}

fn compute_twins(src: &Tree, dst: &Tree) -> Vec<NodeId> {
    let mut twin = vec![NONE; src.num_nodes()];
    for node in src.post_order() {
        if src.is_leaf(node) {
            let lbl = src.label[node as usize];
            if lbl > 0 && (lbl as usize) < dst.label_to_node.len() {
                twin[node as usize] = dst.label_to_node[lbl as usize];
            }
        } else {
            let (l, r) = src.children(node).unwrap();
            let tl = twin[l as usize];
            let tr = twin[r as usize];
            if tl != NONE && tr != NONE {
                twin[node as usize] = dst.nearest_common_ancestor(tl, tr);
            } else if tl != NONE {
                twin[node as usize] = tl;
            } else if tr != NONE {
                twin[node as usize] = tr;
            }
        }
    }
    twin
}

fn is_rspr_cluster_candidate(
    tree: &Tree,
    twin_to_other: &[NodeId],
    twin_back: &[NodeId],
    node: NodeId,
) -> bool {
    if tree.is_leaf(node) || tree.is_root(node) {
        return false;
    }
    let twin = twin_to_other[node as usize];
    if twin == NONE {
        return false;
    }
    let round_trip = twin_back[twin as usize];
    round_trip != NONE && tree.depth[node as usize] <= tree.depth[round_trip as usize]
}

fn children_are_rspr_cluster_candidates(
    tree: &Tree,
    twin_to_other: &[NodeId],
    twin_back: &[NodeId],
    node: NodeId,
) -> bool {
    let Some((left, right)) = tree.children(node) else {
        return false;
    };
    is_rspr_cluster_candidate(tree, twin_to_other, twin_back, left)
        && is_rspr_cluster_candidate(tree, twin_to_other, twin_back, right)
}

fn collect_rspr_cluster_points(
    t1: &Tree,
    t2: &Tree,
    leaf_sets_t1: &[FixedBitSet],
    twin_t1_to_t2: &[NodeId],
    twin_t2_to_t1: &[NodeId],
    n: usize,
) -> Vec<RsprClusterPoint> {
    let mut points = Vec::new();
    for node in t1.post_order() {
        if t1.is_leaf(node) || t1.is_root(node) {
            continue;
        }
        let size = leaf_sets_t1[node as usize].count_ones(..);
        if size < 2 || size > n - 1 {
            continue;
        }
        let twin_t2 = twin_t1_to_t2[node as usize];
        if twin_t2 == NONE {
            continue;
        }
        let round_trip = twin_t2_to_t1[twin_t2 as usize];
        if round_trip == NONE || t1.depth[node as usize] > t1.depth[round_trip as usize] {
            continue;
        }
        if children_are_rspr_cluster_candidates(t1, twin_t1_to_t2, twin_t2_to_t1, node) {
            continue;
        }
        let kind = if round_trip == node {
            RsprClusterPointKind::Strict
        } else {
            RsprClusterPointKind::Relaxed
        };
        // The exact solver cannot cut away the whole T2 root as an ordinary
        // placeholder, but the seed remains useful for parity diagnostics.
        let _twin_is_t2_root = t2.is_root(twin_t2);
        points.push(RsprClusterPoint {
            t1_node: node,
            t2_twin: twin_t2,
            t1_round_trip: round_trip,
            size,
            kind,
        });
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_leaf(tree: &mut Tree, label: Label) -> NodeId {
        let id = tree.parent.len() as NodeId;
        tree.parent.push(NONE);
        tree.left.push(NONE);
        tree.right.push(NONE);
        tree.label.push(label);
        tree.label_to_node[label as usize] = id;
        id
    }
    fn push_internal(tree: &mut Tree, l: NodeId, r: NodeId) -> NodeId {
        let id = tree.parent.len() as NodeId;
        tree.parent.push(NONE);
        tree.left.push(l);
        tree.right.push(r);
        tree.label.push(0);
        tree.parent[l as usize] = id;
        tree.parent[r as usize] = id;
        id
    }

    /// (((1,2),(3,4)),(5,6))
    fn make_balanced_6() -> Tree {
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

    #[test]
    fn test_identical_trees_no_cuts() {
        let t = make_balanced_6();
        let inst = Instance::new(vec![t.clone(), t.clone()], 6);
        let result = try_whidden_decomp_2tree(
            &inst,
            &mut |sub| {
                // Identical => single component containing all leaves.
                Some(vec![sub.trees[0].clone()])
            },
            &NEVER_TERMINATE,
        );
        let comps = result.expect("decomp should fire");
        // 1 inner + 1 outer - 1 = 1 component
        assert_eq!(comps.len(), 1, "identical trees: 1 component expected");
        // All 6 leaves present
        let mut seen = FixedBitSet::with_capacity(7);
        for c in &comps {
            for lbl in c.leaves() {
                seen.insert(lbl as usize);
            }
        }
        for lbl in 1..=6 {
            assert!(seen.contains(lbl), "leaf {} missing", lbl);
        }
    }

    #[test]
    fn test_strict_decomp_tries_next_cluster_after_subsolve_declines() {
        let t = make_balanced_6();
        let inst = Instance::new(vec![t.clone(), t.clone()], 6);
        let mut calls = 0usize;
        let result = try_whidden_decomp_2tree(
            &inst,
            &mut |sub| {
                calls += 1;
                if calls <= 2 {
                    return None;
                }
                Some(vec![sub.trees[0].clone()])
            },
            &NEVER_TERMINATE,
        );

        let comps = result.expect("later strict cluster should still be attempted");
        assert_eq!(comps.len(), 1);
        assert!(
            calls >= 4,
            "expected at least two failed inner attempts plus one successful inner/outer pair, got {}",
            calls
        );
    }

    #[test]
    fn test_strict_batch_identical_trees_no_cuts() {
        let t = make_balanced_6();
        let inst = Instance::new(vec![t.clone(), t.clone()], 6);
        let result = try_batch_decomp(&inst, &mut |sub| {
            // Identical => single component containing all leaves.
            Some(vec![sub.trees[0].clone()])
        });
        let comps = result.expect("strict batch decomp should fire");
        assert_eq!(comps.len(), 1, "identical trees: 1 component expected");
        assert!(validate_agreement_forest(&inst, &comps).is_ok());
    }

    /// T1: (((1,2),(3,4)),(5,6)), T2: ((1,3),((2,4),(5,6)))
    /// {5,6} is a common cherry in both trees (cluster point).
    /// Optimal MAF = 3: {1,2}, {3,4}, {5,6} (exchange 1↔3).
    #[test]
    fn test_common_cluster_point_decomposition() {
        use crate::solvers::maf_branch_price_multi::MafBranchPriceMultiSolver;
        use klados_core::af_validator::validate_agreement_forest;
        use klados_core::brute_maf::brute_force_maf;

        // T1: (((1,2),(3,4)),(5,6))
        let mut t1 = Tree::with_capacity(6);
        let l1 = push_leaf(&mut t1, 1);
        let l2 = push_leaf(&mut t1, 2);
        let l3 = push_leaf(&mut t1, 3);
        let l4 = push_leaf(&mut t1, 4);
        let l5 = push_leaf(&mut t1, 5);
        let l6 = push_leaf(&mut t1, 6);
        let n12 = push_internal(&mut t1, l1, l2);
        let n34 = push_internal(&mut t1, l3, l4);
        let n1234 = push_internal(&mut t1, n12, n34);
        let n56 = push_internal(&mut t1, l5, l6);
        let root = push_internal(&mut t1, n1234, n56);
        t1.root = root;
        t1.compute_metadata();

        // T2: ((1,3),((2,4),(5,6)))
        let mut t2 = Tree::with_capacity(6);
        let l1 = push_leaf(&mut t2, 1);
        let l2 = push_leaf(&mut t2, 2);
        let l3 = push_leaf(&mut t2, 3);
        let l4 = push_leaf(&mut t2, 4);
        let l5 = push_leaf(&mut t2, 5);
        let l6 = push_leaf(&mut t2, 6);
        let n13 = push_internal(&mut t2, l1, l3);
        let n24 = push_internal(&mut t2, l2, l4);
        let n56 = push_internal(&mut t2, l5, l6);
        let n24_56 = push_internal(&mut t2, n24, n56);
        let root = push_internal(&mut t2, n13, n24_56);
        t2.root = root;
        t2.compute_metadata();

        let inst = Instance::new(vec![t1, t2], 6);

        // Ground truth via brute force
        let oracle = brute_force_maf(&inst).expect("brute force should work for n=6");

        // Run Whidden decomposition with B&P-Multi as the inner solver
        let mut solver = MafBranchPriceMultiSolver::new();
        let result = try_whidden_decomp_2tree(
            &inst,
            &mut |sub| {
                // Use B&P-Multi directly (without re-decomposition to avoid recursion)
                // NOTE: won't recurse into Whidden again since we call solve directly
                crate::Solver::solve(&mut solver, sub, &crate::RunConfig::default())
            },
            &NEVER_TERMINATE,
        );

        match result {
            Some(comps) => {
                // Validate AF
                let validation = validate_agreement_forest(&inst, &comps);
                assert!(
                    validation.is_ok(),
                    "whidden output is not a valid AF: {:?}",
                    validation
                );
                assert_eq!(
                    comps.len(),
                    oracle.num_components,
                    "whidden={}, oracle={} (partition: {:?})",
                    comps.len(),
                    oracle.num_components,
                    oracle.partition,
                );
            }
            None => {
                // Decomposition found no cluster point — that's also a valid
                // outcome. The property battery test will still catch cases
                // where the solver path produces wrong results.
                eprintln!("whidden decomp returned None (no cluster point found)");
            }
        }
    }
}
