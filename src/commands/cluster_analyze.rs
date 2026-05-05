//! Analyze cluster decomposition potential.
//!
//! This is intentionally diagnostic-only.  The production solver has several
//! different decomposition paths:
//! - Kelk/common-cluster four-subinstance reduction (`cluster_reduction`)
//! - generic `cluster_decomposition::find_clusters`
//! - 2-tree Whidden-style relaxed cluster points in `whidden_cluster`
//!
//! The timeout instances are mostly huge 2-tree cases, so this command reports
//! the relaxed Whidden cluster-point structure in more detail than the current
//! solver logs: all candidates, child-filtered candidates, disjoint selections
//! under both balanced and postorder policies, and the placeholder remainder
//! size we would expect from an rspr-style "solve clusters first, join later"
//! pipeline.

use fixedbitset::FixedBitSet;
use klados_core::Instance;
use klados_core::cluster_decomposition;
use klados_core::cluster_reduction;
use klados_core::kernelize::{self, KernelizeConfig};
use klados_core::tree::{NONE, NodeId, Tree};
use std::time::Instant;

// Re-use the cluster-finding logic from whidden_cluster.
// (Avoids duplicating the code; calls into the same functions.)

pub fn run(instance: &Instance) -> Result<(), Box<dyn std::error::Error>> {
    let m = instance.num_trees();
    let n = instance.num_leaves;

    let raw = analyze_whidden(instance);

    let kern_config = KernelizeConfig::default();
    let kern = kernelize::kernelize_best(instance, &kern_config);
    let reduced = &kern.instance;
    let n_red = reduced.num_leaves;
    let kern_removed = n - n_red;

    eprintln!("Instance: m={} n={} → kernelized={}", m, n, n_red);

    if m == 2 {
        eprintln!("Raw 2-tree Whidden-style structure:");
        print_whidden("  raw", &raw);
    }

    // --- rspr reference (existing cluster_decomposition::find_clusters) ---
    let t0 = Instant::now();
    let rspr_clusters = cluster_decomposition::find_clusters(reduced);
    let rspr_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (rspr_count, rspr_max, rspr_remainder) = match &rspr_clusters {
        Some(cs) => {
            let sizes: Vec<usize> = cs.iter().map(|c| c.leaves.count_ones(..)).collect();
            let rem = n_red as usize - sizes.iter().sum::<usize>();
            (cs.len(), sizes.iter().copied().max().unwrap_or(0), rem)
        }
        None => (0, 0, n_red as usize),
    };

    // --- Whidden pipeline (relaxed + child filter, disjoint selection) ---
    let t1 = Instant::now();
    let whidden = analyze_whidden(reduced);
    let whidden_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // --- Kelk common-cluster ---
    let t2 = Instant::now();
    let kelk = cluster_reduction::find_best_common_cluster(reduced);
    let kelk_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let (kelk_found, kelk_size) = match &kelk {
        Some(c) => (true, c.leaves.count_ones(..)),
        None => (false, 0),
    };

    // Report
    eprintln!(
        "Kelk common-cluster: {} (size={}) in {:.1}ms",
        if kelk_found { "FOUND" } else { "none" },
        kelk_size,
        kelk_ms
    );
    eprintln!(
        "rspr (cluster_decomposition): {} clusters, max={}, remainder={} in {:.1}ms",
        rspr_count, rspr_max, rspr_remainder, rspr_ms
    );
    eprintln!(
        "Whidden relaxed diagnostics on kernel in {:.1}ms:",
        whidden_ms
    );
    print_whidden("  kernel", &whidden);

    // Machine-readable line
    println!(
        "m={} n={} n_red={} removed={} kelk_found={} kelk_size={} strict={} strict_balanced_selected={} strict_balanced_placeholder_rem={} relaxed_raw={} relaxed_child_filtered={} balanced_selected={} balanced_placeholder_rem={} postorder_selected={} postorder_placeholder_rem={} largest_balanced_subproblem={} rspr_ms={:.1} whidden_ms={:.1} kelk_ms={:.1}",
        m,
        n,
        n_red,
        kern_removed,
        if kelk_found { 1 } else { 0 },
        kelk_size,
        whidden.strict_common,
        whidden.strict_balanced.selected,
        whidden.strict_balanced.placeholder_remainder,
        whidden.relaxed_raw,
        whidden.relaxed_child_filtered,
        whidden.balanced.selected,
        whidden.balanced.placeholder_remainder,
        whidden.postorder.selected,
        whidden.postorder.placeholder_remainder,
        whidden.balanced.largest_subproblem,
        rspr_ms,
        whidden_ms,
        kelk_ms
    );

    Ok(())
}

#[derive(Default)]
struct SelectionStats {
    selected: usize,
    clustered_leaves: usize,
    closed_remainder: usize,
    placeholder_remainder: usize,
    largest_subproblem: usize,
    sizes: Vec<usize>,
}

#[derive(Default)]
struct WhiddenStats {
    strict_common: usize,
    relaxed_raw: usize,
    relaxed_child_filtered: usize,
    strict_balanced: SelectionStats,
    balanced: SelectionStats,
    postorder: SelectionStats,
    top_strict_sizes: Vec<usize>,
    top_relaxed_sizes: Vec<usize>,
    top_raw_relaxed_sizes: Vec<usize>,
}

fn print_whidden(prefix: &str, stats: &WhiddenStats) {
    eprintln!(
        "{}: strict_common={} relaxed_raw={} relaxed_child_filtered={}",
        prefix, stats.strict_common, stats.relaxed_raw, stats.relaxed_child_filtered
    );
    eprintln!(
        "{} balanced-select: selected={} clustered={} closed_rem={} placeholder_rem={} largest_subproblem={}",
        prefix,
        stats.balanced.selected,
        stats.balanced.clustered_leaves,
        stats.balanced.closed_remainder,
        stats.balanced.placeholder_remainder,
        stats.balanced.largest_subproblem
    );
    eprintln!(
        "{} strict-balanced-select: selected={} clustered={} closed_rem={} placeholder_rem={} largest_subproblem={}",
        prefix,
        stats.strict_balanced.selected,
        stats.strict_balanced.clustered_leaves,
        stats.strict_balanced.closed_remainder,
        stats.strict_balanced.placeholder_remainder,
        stats.strict_balanced.largest_subproblem
    );
    eprintln!(
        "{} postorder-select: selected={} clustered={} closed_rem={} placeholder_rem={} largest_subproblem={}",
        prefix,
        stats.postorder.selected,
        stats.postorder.clustered_leaves,
        stats.postorder.closed_remainder,
        stats.postorder.placeholder_remainder,
        stats.postorder.largest_subproblem
    );
    if !stats.top_strict_sizes.is_empty() {
        eprintln!(
            "{} top strict/common sizes: {:?}",
            prefix, stats.top_strict_sizes
        );
    }
    if !stats.top_raw_relaxed_sizes.is_empty() {
        eprintln!(
            "{} top relaxed raw sizes: {:?}",
            prefix, stats.top_raw_relaxed_sizes
        );
    }
    if !stats.top_relaxed_sizes.is_empty() {
        eprintln!(
            "{} top relaxed child-filtered sizes: {:?}",
            prefix, stats.top_relaxed_sizes
        );
    }
    if !stats.balanced.sizes.is_empty() {
        eprintln!(
            "{} balanced selected sizes: {:?}",
            prefix, stats.balanced.sizes
        );
    }
    if !stats.strict_balanced.sizes.is_empty() {
        eprintln!(
            "{} strict-balanced selected sizes: {:?}",
            prefix, stats.strict_balanced.sizes
        );
    }
    if !stats.postorder.sizes.is_empty() {
        eprintln!(
            "{} postorder selected sizes: {:?}",
            prefix, stats.postorder.sizes
        );
    }
}

fn analyze_whidden(instance: &Instance) -> WhiddenStats {
    let n = instance.num_leaves as usize;
    if n < 6 || instance.num_trees() != 2 {
        return WhiddenStats::default();
    }
    let t1 = &instance.trees[0];
    let t2 = &instance.trees[1];

    // Compute leaf sets and twins (same as whidden_cluster.rs).
    let leaf_sets = compute_leaf_sets(t1, n);
    let tw12 = compute_twins(t1, t2);
    let tw21 = compute_twins(t2, t1);

    // Find cluster points.  Whidden's test is relaxed:
    // depth(n) <= depth(twin(twin(n))).  The strict/common case is equality
    // of the round trip node; it is the subset that can be treated as a closed
    // common cluster without rho/boundary state.
    let mut is_cl = vec![false; t1.num_nodes()];
    let mut raw_relaxed = Vec::new();
    let mut pts = Vec::new();
    let mut strict_pts = Vec::new();
    let mut strict_sizes = Vec::new();
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
        let round_trip = tw21[t2t as usize];
        if round_trip == NONE {
            continue;
        }
        if round_trip == node {
            strict_sizes.push(sz);
            strict_pts.push((node, sz));
        }
        if t1.depth[node as usize] > t1.depth[round_trip as usize] {
            continue;
        }
        raw_relaxed.push((node, sz));
        if let Some((l, r)) = t1.children(node) {
            if is_cl[l as usize] && is_cl[r as usize] {
                continue;
            }
        }
        is_cl[node as usize] = true;
        pts.push((node, sz));
    }

    let mut top_strict_sizes = strict_sizes.clone();
    top_strict_sizes.sort_unstable_by(|a, b| b.cmp(a));
    top_strict_sizes.truncate(16);

    let mut top_raw_relaxed_sizes: Vec<_> = raw_relaxed.iter().map(|(_, size)| *size).collect();
    top_raw_relaxed_sizes.sort_unstable_by(|a, b| b.cmp(a));
    top_raw_relaxed_sizes.truncate(16);

    let mut top_relaxed_sizes: Vec<_> = pts.iter().map(|(_, size)| *size).collect();
    top_relaxed_sizes.sort_unstable_by(|a, b| b.cmp(a));
    top_relaxed_sizes.truncate(16);

    // Greedy disjoint selection, most balanced first.  This mirrors the
    // current fast rspr-ish path in `whidden_cluster.rs`.
    let mut balanced_order = pts.clone();
    balanced_order.sort_by_key(|(_, size)| -((*size).min(n - *size) as isize));
    let balanced = select_disjoint(n, &leaf_sets, &balanced_order);

    let mut strict_balanced_order = strict_pts;
    strict_balanced_order.sort_by_key(|(_, size)| -((*size).min(n - *size) as isize));
    let strict_balanced = select_disjoint(n, &leaf_sets, &strict_balanced_order);

    // Postorder/deep-first selection.  This mirrors
    // `cluster_decomposition::select_disjoint_clusters`.
    let postorder = select_disjoint(n, &leaf_sets, &pts);

    WhiddenStats {
        strict_common: strict_sizes.len(),
        relaxed_raw: raw_relaxed.len(),
        relaxed_child_filtered: pts.len(),
        strict_balanced,
        balanced,
        postorder,
        top_strict_sizes,
        top_relaxed_sizes,
        top_raw_relaxed_sizes,
    }
}

fn select_disjoint(
    n: usize,
    leaf_sets: &[FixedBitSet],
    candidates: &[(NodeId, usize)],
) -> SelectionStats {
    let cap = leaf_sets.first().map(|l| l.len()).unwrap_or(0);
    let mut taken = FixedBitSet::with_capacity(cap);
    let mut sizes = Vec::new();
    for &(node, size) in candidates {
        let leaves = &leaf_sets[node as usize];
        if !taken.is_disjoint(leaves) {
            continue;
        }
        taken.union_with(leaves);
        sizes.push(size);
    }
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let clustered_leaves = taken.count_ones(..);
    let closed_remainder = n.saturating_sub(clustered_leaves);
    let selected = sizes.len();
    // In an rspr-style join pipeline, the remaining outer instance keeps one
    // placeholder/boundary object per solved cluster, not an arbitrary "anchor
    // half" of the cluster.  This is the useful target size to compare with the
    // current large B&P kernels.
    let placeholder_remainder = closed_remainder + selected;
    let largest_cluster = sizes.iter().copied().max().unwrap_or(0);
    let largest_subproblem = largest_cluster.max(placeholder_remainder);
    let mut sample_sizes = sizes;
    sample_sizes.truncate(32);
    SelectionStats {
        selected,
        clustered_leaves,
        closed_remainder,
        placeholder_remainder,
        largest_subproblem,
        sizes: sample_sizes,
    }
}

fn compute_leaf_sets(tree: &Tree, n: usize) -> Vec<FixedBitSet> {
    let mut sets = vec![FixedBitSet::with_capacity(n + 1); tree.num_nodes()];
    for node in tree.post_order() {
        if tree.is_leaf(node) {
            let lbl = tree.label[node as usize];
            if lbl > 0 && (lbl as usize) <= n {
                sets[node as usize].insert(lbl as usize);
            }
        } else if let Some((l, r)) = tree.children(node) {
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
        } else if let Some((l, r)) = src.children(node) {
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
