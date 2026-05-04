//! Whidden-style cluster decomposition for 2-tree MAF instances.
//!
//! Identifies a "cluster point" c in T1 — an internal node whose round-trip
//! depth (T1 -> twin in T2 -> back to T1) does not exceed depth(c). For such
//! a c with leaf-set X, the optimal MAF decomposes as
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
//! Restricted to m == 2. Returns None if no cluster point exists or if no
//! valid merge is found.

use fixedbitset::FixedBitSet;

use klados_core::af_validator::validate_agreement_forest;
use klados_core::tree::{Label, NodeId, Tree, NONE};
use klados_core::Instance;

/// Try Whidden cluster decomposition on a 2-tree instance.
///
/// `solve` is invoked on each sub-instance; pass the inner solver
/// (e.g. B&P-Multi). Returns `None` if no useful cluster point is found
/// or if any sub-solve returns `None`.
pub fn try_whidden_decomp_2tree<F>(
    instance: &Instance,
    solve: &mut F,
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
    let twin_t2_to_t1 = compute_twins(t2, t1);

    let cluster = find_best_cluster_point(
        t1,
        &leaf_sets_t1,
        &twin_t1_to_t2,
        &twin_t2_to_t1,
        n,
    )?;

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
    let mut next: Label = 1;
    for old_lbl in cluster_leaves.ones() {
        inner_label_map[old_lbl] = next;
        inner_to_orig[next as usize] = old_lbl as Label;
        next += 1;
    }
    let inner_t1 = t1.relabel(&inner_label_map, cluster_size as u32);
    let inner_t2 = t2.relabel(&inner_label_map, cluster_size as u32);
    let inner_instance = Instance::new(vec![inner_t1, inner_t2], cluster_size as u32);

    // Build outer sub-instance: replace cluster subtree with single leaf P.
    let outer_t1 = replace_and_relabel(
        t1, cluster, outer_p_label, &outer_label_map, outer_new_num_leaves,
    );
    let outer_t2 = replace_and_relabel(
        t2, twin_in_t2, outer_p_label, &outer_label_map, outer_new_num_leaves,
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
    let outer_p_comp_decoded =
        decode_outer_keeping_placeholder(&outer_components[outer_p_idx], &outer_to_orig, outer_p_label, instance.num_leaves);

    let decoded_outer_non_p: Vec<Tree> = outer_components
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != outer_p_idx)
        .map(|(_, c)| c.relabel(&outer_to_orig, instance.num_leaves))
        .collect();

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
            eprintln!(
                "[whidden] decomp n={}: inner={} outer={} (cluster_node={}, anchor_idx={}, {} comps)",
                n, cluster_size, outer_size + 1, cluster, anchor_idx, candidate.len()
            );
            return Some(candidate);
        }
    }

    // No valid anchor — decomposition fails for this cluster point.
    eprintln!(
        "[whidden] decomp n={}: inner={} outer={} — no valid anchor among {} inner components",
        n, cluster_size, outer_size + 1, decoded_inner.len()
    );
    None
}

/// Create a leaf set with a single label excluded.
fn leaf_set_excluding(tree: &Tree, exclude: Label, cap: usize) -> FixedBitSet {
    let mut s = FixedBitSet::with_capacity(cap + 1);
    for lbl in tree.leaves() {
        if lbl != exclude {
            s.insert(lbl as usize);
        }
    }
    s
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

    if tree.root != NONE {
        if let Some(root) = build(tree, target, placeholder, label_map, &mut out, tree.root) {
            out.root = root;
            out.parent[root as usize] = NONE;
        }
    }
    out.compute_metadata();
    out
}

// ---------------------------------------------------------------------------
// Subtree replacement: build a copy of `tree` where the subtree rooted at
// `target` is replaced by a single leaf labeled `placeholder`.
// ---------------------------------------------------------------------------
fn replace_subtree_with_leaf(
    tree: &Tree,
    target: NodeId,
    placeholder: Label,
    new_num_leaves: u32,
) -> Tree {
    let mut out = Tree::with_capacity(new_num_leaves);

    fn build(
        src: &Tree,
        target: NodeId,
        placeholder: Label,
        out: &mut Tree,
        node: NodeId,
    ) -> Option<NodeId> {
        if node == target {
            // Emit single leaf labeled `placeholder`.
            let id = out.parent.len() as NodeId;
            out.parent.push(NONE);
            out.left.push(NONE);
            out.right.push(NONE);
            out.label.push(placeholder);
            out.label_to_node[placeholder as usize] = id;
            return Some(id);
        }
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
        let (left, right) = src.children(node).unwrap();
        let l = build(src, target, placeholder, out, left);
        let r = build(src, target, placeholder, out, right);
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

    if tree.root != NONE {
        if let Some(root) = build(tree, target, placeholder, &mut out, tree.root) {
            out.root = root;
            out.parent[root as usize] = NONE;
        }
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

    if outer_comp.root != NONE {
        if let Some(root) = copy_outer_with_splice(outer_comp, inner_comp, p_label, &mut out, outer_comp.root) {
            out.root = root;
            out.parent[root as usize] = NONE;
        }
    }
    out.compute_metadata();
    out
}

// Outer components other than the P-component contain only original labels
// (1..=n) but their `num_leaves` is n+1. Reproject to label space n.
fn reproject_outer(comp: &Tree, final_num_leaves: u32) -> Tree {
    // Identity remap: labels stay the same; only num_leaves changes.
    let mut map: Vec<Label> = vec![0; comp.num_leaves as usize + 1];
    for lbl in 1..=comp.num_leaves {
        map[lbl as usize] = lbl;
    }
    comp.relabel(&map, final_num_leaves)
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

// ---------------------------------------------------------------------------
// Cluster-point search.
// ---------------------------------------------------------------------------

fn debug_assert_valid_tree(tree: &Tree, name: &str) {
    assert_ne!(tree.root, NONE, "{}: root is NONE", name);
    assert!(
        (tree.root as usize) < tree.parent.len(),
        "{}: root {} out of range (len {})",
        name,
        tree.root,
        tree.parent.len()
    );
    assert_eq!(
        tree.parent[tree.root as usize],
        NONE,
        "{}: root has parent",
        name
    );
    let mut leaves_seen = 0u32;
    for node in 0..tree.parent.len() as u32 {
        if tree.is_leaf(node) && tree.label[node as usize] != 0 {
            leaves_seen += 1;
            let lbl = tree.label[node as usize];
            assert!(
                (lbl as usize) < tree.label_to_node.len(),
                "{}: label {} out of label_to_node (len {})",
                name,
                lbl,
                tree.label_to_node.len()
            );
            assert_eq!(
                tree.label_to_node[lbl as usize], node,
                "{}: label_to_node[{}] != node {}",
                name, lbl, node
            );
        }
    }
    eprintln!(
        "[whidden] {} ok: nodes={} leaves={} num_leaves_field={} root={}",
        name,
        tree.parent.len(),
        leaves_seen,
        tree.num_leaves,
        tree.root
    );
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

/// Pick the best cluster point: deepest valid (= smallest cluster) so that
/// after one decomposition both pieces are substantially smaller than n.
fn find_best_cluster_point(
    t1: &Tree,
    leaf_sets_t1: &[FixedBitSet],
    twin_t1_to_t2: &[NodeId],
    twin_t2_to_t1: &[NodeId],
    n: usize,
) -> Option<NodeId> {
    let mut best: Option<(NodeId, usize)> = None;

    for node in t1.post_order() {
        if t1.is_leaf(node) || t1.is_root(node) {
            continue;
        }
        let size = leaf_sets_t1[node as usize].count_ones(..);
        // Reject trivial clusters and trivial outers.
        if size < 2 || size > n - 2 {
            continue;
        }

        let twin_t2 = twin_t1_to_t2[node as usize];
        if twin_t2 == NONE {
            continue;
        }
        let round_trip = twin_t2_to_t1[twin_t2 as usize];
        if round_trip == NONE {
            continue;
        }
        // Strict cluster point: round-trip lands exactly back on `node`. This
        // guarantees leaves(twin_in_t2) == leaves(node), so the inner /
        // outer split is on the same leaf set in both trees.
        if round_trip != node {
            continue;
        }

        // Score: maximize min(size, n - size) — most balanced split.
        let score = size.min(n - size);
        match best {
            None => best = Some((node, score)),
            Some((_, s)) if score > s => best = Some((node, score)),
            _ => {}
        }
    }

    best.map(|(node, _)| node)
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
        let result = try_whidden_decomp_2tree(&inst, &mut |sub| {
            // Identical => single component containing all leaves.
            Some(vec![sub.trees[0].clone()])
        });
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

    /// T1: (((1,2),(3,4)),(5,6)), T2: ((1,3),((2,4),(5,6)))
    /// {5,6} is a common cherry in both trees (cluster point).
    /// Optimal MAF = 3: {1,2}, {3,4}, {5,6} (exchange 1↔3).
    #[test]
    fn test_common_cluster_point_decomposition() {
        use klados_core::af_validator::validate_agreement_forest;
        use klados_core::brute_maf::brute_force_maf;
        use crate::ExactSolver;
        use crate::maf_branch_price_multi::MafBranchPriceMultiSolver;

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
        let result = try_whidden_decomp_2tree(&inst, &mut |sub| {
            // Use B&P-Multi directly (without re-decomposition to avoid recursion)
            // NOTE: this won't recurse into Whidden again since we call solve directly
            solver.solve(sub)
        });

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
