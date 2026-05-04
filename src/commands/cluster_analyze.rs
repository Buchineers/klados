//! Analyze cluster decomposition: rspr reference vs our pipeline.

use klados_core::Instance;
use klados_core::cluster_decomposition;
use klados_core::cluster_reduction;
use klados_core::kernelize::{self, KernelizeConfig};
use klados_core::tree::{NONE, NodeId, Tree};
use fixedbitset::FixedBitSet;
use std::time::Instant;

// Re-use the cluster-finding logic from whidden_cluster.
// (Avoids duplicating the code; calls into the same functions.)

pub fn run(instance: &Instance) -> Result<(), Box<dyn std::error::Error>> {
    let m = instance.num_trees();
    let n = instance.num_leaves;

    let kern_config = KernelizeConfig::default();
    let kern = kernelize::kernelize_best(instance, &kern_config);
    let reduced = &kern.instance;
    let n_red = reduced.num_leaves;
    let kern_removed = n - n_red;

    eprintln!("Instance: m={} n={} → kernelized={}", m, n, n_red);

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
    eprintln!("Kelk common-cluster: {} (size={}) in {:.1}ms",
        if kelk_found { "FOUND" } else { "none" }, kelk_size, kelk_ms);
    eprintln!("rspr (cluster_decomposition): {} clusters, max={}, remainder={} in {:.1}ms",
        rspr_count, rspr_max, rspr_remainder, rspr_ms);
    eprintln!("Whidden (forest+single): {} found → {} disjoint → {} selected, remainder={} in {:.1}ms",
        whidden.found, whidden.disjoint, whidden.selected, whidden.remainder, whidden_ms);
    if !whidden.sizes.is_empty() {
        eprintln!("  cluster sizes: {:?}", whidden.sizes);
    }
    if !whidden.anchor_sizes.is_empty() {
        eprintln!("  anchor sizes (est): {:?}", whidden.anchor_sizes);
    }

    // Machine-readable line
    println!("{} {} {} {} {} {} {} {} {} {} {:.1} {:.1} {:.1}",
        m, n, n_red, kern_removed,
        if kelk_found {1}else{0}, kelk_size,
        whidden.found, whidden.disjoint, whidden.selected, whidden.remainder,
        rspr_ms, whidden_ms, kelk_ms);

    Ok(())
}

struct WhiddenStats { found: usize, disjoint: usize, selected: usize, remainder: usize, sizes: Vec<usize>, anchor_sizes: Vec<usize> }

fn analyze_whidden(instance: &Instance) -> WhiddenStats {
    let n = instance.num_leaves as usize;
    if n < 6 || instance.num_trees() != 2 {
        return WhiddenStats { found:0, disjoint:0, selected:0, remainder:n, sizes:vec![], anchor_sizes:vec![] };
    }
    let t1 = &instance.trees[0];
    let t2 = &instance.trees[1];

    // Compute leaf sets and twins (same as whidden_cluster.rs).
    let leaf_sets = compute_leaf_sets(t1, n);
    let tw12 = compute_twins(t1, t2);
    let tw21 = compute_twins(t2, t1);

    // Find cluster points (relaxed + child filter).
    let mut is_cl = vec![false; t1.num_nodes()];
    let mut pts = Vec::new();
    for node in t1.post_order() {
        if t1.is_leaf(node) || t1.is_root(node) { continue; }
        let sz = leaf_sets[node as usize].count_ones(..);
        if sz < 2 || sz > n - 2 { continue; }
        let t2t = tw12[node as usize];
        if t2t == NONE { continue; }
        if t1.depth[node as usize] > t1.depth[tw21[t2t as usize] as usize] { continue; }
        if let Some((l, r)) = t1.children(node) {
            if is_cl[l as usize] && is_cl[r as usize] { continue; }
        }
        is_cl[node as usize] = true;
        pts.push((node, sz.min(n - sz)));
    }
    let found = pts.len();
    pts.sort_by_key(|(_, s)| -(*s as isize));

    // Greedy disjoint selection.
    let cap = leaf_sets.first().map(|l| l.len()).unwrap_or(0);
    let mut taken = FixedBitSet::with_capacity(cap);
    let mut sel: Vec<usize> = Vec::new();
    for (i, (node, _)) in pts.iter().enumerate() {
        let ls = &leaf_sets[*node as usize];
        if !taken.is_disjoint(ls) { continue; }
        taken.union_with(ls); sel.push(i);
    }
    let disjoint = sel.len();

    // For each selected cluster, estimate anchor size (first inner component).
    // Without actually solving, use the cluster itself as proxy.
    let mut sizes = Vec::new();
    let mut anchor_sizes = Vec::new();
    let mut all_clustered = FixedBitSet::with_capacity(n + 1);
    for &ci in &sel {
        let (cnode, _) = pts[ci];
        let leaves = &leaf_sets[cnode as usize];
        let csize = leaves.count_ones(..);
        sizes.push(csize);
        // Anchor is typically ~half the cluster (rough estimate).
        anchor_sizes.push(csize / 2);
        all_clustered.union_with(leaves);
    }
    let remainder = (1..=n as u32).filter(|l| !all_clustered.contains(*l as usize)).count();

    // Also count anchor leaves remaining.
    let mut anchored_remainder = remainder;
    for &ci in &sel {
        let (cnode, _) = pts[ci];
        let leaves = &leaf_sets[cnode as usize];
        let csize = leaves.count_ones(..);
        let anchor_size = csize / 2; // rough
        anchored_remainder += anchor_size; // anchor stays
    }

    WhiddenStats {
        found, disjoint, selected: sel.len(), remainder: anchored_remainder,
        sizes, anchor_sizes,
    }
}

fn compute_leaf_sets(tree: &Tree, n: usize) -> Vec<FixedBitSet> {
    let mut sets = vec![FixedBitSet::with_capacity(n + 1); tree.num_nodes()];
    for node in tree.post_order() {
        if tree.is_leaf(node) {
            let lbl = tree.label[node as usize];
            if lbl > 0 && (lbl as usize) <= n { sets[node as usize].insert(lbl as usize); }
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
            let tl = twin[l as usize]; let tr = twin[r as usize];
            if tl != NONE && tr != NONE {
                twin[node as usize] = dst.nearest_common_ancestor(tl, tr);
            } else if tl != NONE { twin[node as usize] = tl; }
            else if tr != NONE { twin[node as usize] = tr; }
        }
    }
    twin
}
