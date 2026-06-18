//! Cluster decomposition analysis.
//!
//! Reports cluster decomposition potential under multiple strategies:
//! - Kelk common-cluster (4-subinstance reduction)
//! - rSPR-style cluster points (strict/common)
//! - Whidden relaxed cluster points (balanced, postorder, strict-balanced selection)

use fixedbitset::FixedBitSet;
use klados_core::Instance;
use klados_core::cluster_decomposition;
use klados_core::cluster_reduction;
use klados_core::kernelize::{self, KernelizeConfig};
use klados_core::tree::{NONE, NodeId, Tree};
use std::time::Instant;

pub fn run(instance: &Instance) -> Result<(), Box<dyn std::error::Error>> {
    let m = instance.num_trees();
    let n = instance.num_leaves;

    let t_kern = Instant::now();
    let kern_config = KernelizeConfig::default();
    let kern = kernelize::kernelize_best(instance, &kern_config);
    let reduced = &kern.instance;
    let n_red = reduced.num_leaves;
    let kern_removed = n.saturating_sub(n_red);
    let kern_ms = t_kern.elapsed().as_secs_f64() * 1000.0;

    eprintln!();
    eprintln!(
        "  trees={}  leaves={} → kernelized={}  ({:.0}% removed by kernel, {:.1}ms)",
        m,
        n,
        n_red,
        if n > 0 {
            100.0 * kern_removed as f64 / n as f64
        } else {
            0.0
        },
        kern_ms,
    );
    eprintln!();

    // 2-tree only for Whidden diagnostics
    let raw = if m == 2 {
        Some(analyze_whidden(reduced))
    } else {
        None
    };

    // Kelk common-cluster
    let t = Instant::now();
    let kelk = cluster_reduction::find_best_common_cluster(reduced);
    let kelk_ms = t.elapsed().as_secs_f64() * 1000.0;
    let kelk_size = kelk.as_ref().map(|c| c.leaves.count_ones(..)).unwrap_or(0);

    // rSPR cluster decomposition
    let t = Instant::now();
    let rspr = cluster_decomposition::find_clusters(reduced);
    let rspr_ms = t.elapsed().as_secs_f64() * 1000.0;
    let (rspr_count, rspr_max, rspr_clustered, rspr_rem) = match &rspr {
        Some(cs) => {
            let sizes: Vec<usize> = cs.iter().map(|c| c.leaves.count_ones(..)).collect();
            let clustered = sizes.iter().sum::<usize>();
            (
                cs.len(),
                sizes.iter().copied().max().unwrap_or(0),
                clustered,
                n_red as usize - clustered,
            )
        }
        None => (0, 0, 0, n_red as usize),
    };

    // Header
    eprintln!(
        "  {:<32} {:>8} {:>10} {:>10} {:>10} {:>8}",
        "method", "clusters", "clustered", "remainder", "largest", "ms"
    );
    eprintln!(
        "  {:-<32} {:-<8} {:-<10} {:-<10} {:-<10} {:-<8}",
        "", "", "", "", "", ""
    );

    // Kelk row
    eprintln!(
        "  {:<32} {:>8} {:>10} {:>10} {:>10} {:>7.1}",
        "Kelk common-cluster",
        if kelk.is_some() { "1" } else { "—" },
        kelk_size,
        format_label(kelk_size, n_red as usize),
        "—",
        kelk_ms,
    );

    // rSPR row
    eprintln!(
        "  {:<32} {:>8} {:>10} {:>10} {:>10} {:>7.1}",
        "rSPR cluster points",
        if rspr_count > 0 {
            rspr_count.to_string()
        } else {
            "—".into()
        },
        rspr_clustered.max(1),
        if rspr_count > 0 {
            rspr_rem.to_string()
        } else {
            "—".into()
        },
        if rspr_count > 0 {
            rspr_max.to_string()
        } else {
            "—".into()
        },
        rspr_ms,
    );

    // Whidden rows (2-tree only)
    if let Some(ref w) = raw {
        eprintln!();
        eprintln!("  Whidden cluster points (relaxed):");
        eprintln!(
            "    strict/common: {}  relaxed raw: {}  child-filtered: {}",
            w.strict_common, w.relaxed_raw, w.relaxed_child_filtered
        );

        if !w.top_strict_sizes.is_empty() {
            eprintln!("    top strict sizes:   {}", fmt_sizes(&w.top_strict_sizes));
        }
        if !w.top_raw_relaxed_sizes.is_empty() {
            eprintln!(
                "    top relaxed sizes:  {}",
                fmt_sizes(&w.top_raw_relaxed_sizes)
            );
        }
        if !w.top_relaxed_sizes.is_empty() {
            eprintln!(
                "    top child-filtered: {}",
                fmt_sizes(&w.top_relaxed_sizes)
            );
        }

        eprintln!();
        eprintln!(
            "  {:<28} {:>8} {:>10} {:>10} {:>10}",
            "selection strategy", "selected", "clustered", "remainder", "largest-subproblem"
        );
        eprintln!(
            "  {:-<28} {:-<8} {:-<10} {:-<10} {:-<10}",
            "", "", "", "", ""
        );

        print_sel("balanced", &w.balanced);
        print_sel("strict-balanced", &w.strict_balanced);
        print_sel("postorder", &w.postorder);
    }

    // Machine-readable line
    println!(
        "m={}\tn={}\tn_red={}\tremoved={}\tkelk_size={}\tkelk_ms={:.1}\trspr_count={}\trspr_max={}\trspr_rem={}\trspr_ms={:.1}",
        m, n, n_red, kern_removed, kelk_size, kelk_ms, rspr_count, rspr_max, rspr_rem, rspr_ms,
    );
    if let Some(ref w) = raw {
        println!(
            "strict={}\trelaxed_raw={}\tchild_filtered={}\tbal_sel={}\tbal_clustered={}\tbal_rem={}\tpost_sel={}\tpost_clustered={}\tpost_rem={}",
            w.strict_common,
            w.relaxed_raw,
            w.relaxed_child_filtered,
            w.balanced.selected,
            w.balanced.clustered_leaves,
            w.balanced.placeholder_remainder,
            w.postorder.selected,
            w.postorder.clustered_leaves,
            w.postorder.placeholder_remainder,
        );
    }

    Ok(())
}

fn print_sel(name: &str, s: &SelectionStats) {
    eprintln!(
        "  {:<28} {:>8} {:>10} {:>10} {:>10}",
        name, s.selected, s.clustered_leaves, s.placeholder_remainder, s.largest_subproblem
    );
    if !s.sizes.is_empty() {
        eprintln!("    sizes: {}", fmt_sizes(&s.sizes));
    }
}

fn fmt_sizes(sizes: &[usize]) -> String {
    let items: Vec<String> = sizes.iter().take(10).map(|s| s.to_string()).collect();
    if sizes.len() > 10 {
        format!("[{} ... ({} total)]", items.join(", "), sizes.len())
    } else {
        format!("[{}]", items.join(", "))
    }
}

fn format_label(count: usize, total: usize) -> String {
    if count == 0 {
        "—".into()
    } else if total > 0 {
        format!("{}+{}", count, total - count)
    } else {
        "—".into()
    }
}

// ── Whidden stats types ────────────────────────────────────────────────────

#[derive(Default)]
struct SelectionStats {
    selected: usize,
    clustered_leaves: usize,
    placeholder_remainder: usize,
    largest_subproblem: usize,
    sizes: Vec<usize>,
}

#[derive(Default)]
struct WhiddenStats {
    strict_common: usize,
    relaxed_raw: usize,
    relaxed_child_filtered: usize,
    balanced: SelectionStats,
    strict_balanced: SelectionStats,
    postorder: SelectionStats,
    top_strict_sizes: Vec<usize>,
    top_relaxed_sizes: Vec<usize>,
    top_raw_relaxed_sizes: Vec<usize>,
}

fn analyze_whidden(instance: &Instance) -> WhiddenStats {
    let n = instance.num_leaves as usize;
    if n < 6 || instance.num_trees() != 2 {
        return WhiddenStats::default();
    }
    let t1 = &instance.trees[0];
    let t2 = &instance.trees[1];

    let leaf_sets = compute_leaf_sets(t1, n);
    let tw12 = compute_twins(t1, t2);
    let tw21 = compute_twins(t2, t1);

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
        if let Some((l, r)) = t1.children(node)
            && is_cl[l as usize] && is_cl[r as usize] {
                continue;
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

    let mut balanced_order = pts.clone();
    balanced_order.sort_by_key(|(_, size)| -((*size).min(n - *size) as isize));
    let balanced = select_disjoint(n, &leaf_sets, &balanced_order);

    let mut strict_balanced_order = strict_pts;
    strict_balanced_order.sort_by_key(|(_, size)| -((*size).min(n - *size) as isize));
    let strict_balanced = select_disjoint(n, &leaf_sets, &strict_balanced_order);

    let postorder = select_disjoint(n, &leaf_sets, &pts);

    WhiddenStats {
        strict_common: strict_sizes.len(),
        relaxed_raw: raw_relaxed.len(),
        relaxed_child_filtered: pts.len(),
        balanced,
        strict_balanced,
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
    let selected = sizes.len();
    let placeholder_remainder = (n.saturating_sub(clustered_leaves)) + selected;
    let largest = sizes.iter().copied().max().unwrap_or(0);
    let largest_subproblem = largest.max(placeholder_remainder);
    let mut sample = sizes;
    sample.truncate(32);
    SelectionStats {
        selected,
        clustered_leaves,
        placeholder_remainder,
        largest_subproblem,
        sizes: sample,
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
