//! Analyze cluster decomposition potential (Kelk common-cluster + rspr-style).

use klados_core::Instance;
use klados_core::cluster_decomposition;
use klados_core::cluster_reduction;
use klados_core::kernelize::{self, KernelizeConfig};

pub fn run(instance: &Instance) -> Result<(), Box<dyn std::error::Error>> {
    let m = instance.num_trees();
    let n = instance.num_leaves;

    // Kernelize first (same as solver pipeline).
    let kern_config = KernelizeConfig::default();
    let kern = kernelize::kernelize_best(instance, &kern_config);
    let reduced = &kern.instance;
    let n_red = reduced.num_leaves;

    let kern_removed = n - n_red;

    // Kelk common-cluster detection.
    let t_kelk = std::time::Instant::now();
    let kelk_cluster = cluster_reduction::find_best_common_cluster(reduced);
    let kelk_ms = t_kelk.elapsed().as_secs_f64() * 1000.0;

    let (kelk_found, kelk_size) = match &kelk_cluster {
        Some(c) => (true, c.leaves.count_ones(..)),
        None => (false, 0),
    };

    // rspr-style cluster decomposition (generalized to any m >= 2).
    let t_rspr = std::time::Instant::now();
    let rspr_clusters = cluster_decomposition::find_clusters(reduced);
    let rspr_ms = t_rspr.elapsed().as_secs_f64() * 1000.0;

    let (rspr_count, rspr_sizes, rspr_remainder) = match &rspr_clusters {
        Some(clusters) => {
            let sizes: Vec<usize> = clusters.iter().map(|c| c.leaves.count_ones(..)).collect();
            let total_clustered: usize = sizes.iter().sum();
            let remainder = n_red as usize - total_clustered;
            (clusters.len(), sizes, remainder)
        }
        None => (0, vec![], n_red as usize),
    };

    // Output single-line summary for easy parsing by justfile script.
    // Format: m n n_red kern_removed kelk_found kelk_size rspr_count rspr_max rspr_remainder kelk_ms rspr_ms
    let rspr_max = rspr_sizes.iter().copied().max().unwrap_or(0);

    println!(
        "{} {} {} {} {} {} {} {} {} {:.1} {:.1}",
        m,
        n,
        n_red,
        kern_removed,
        if kelk_found { 1 } else { 0 },
        kelk_size,
        rspr_count,
        rspr_max,
        rspr_remainder,
        kelk_ms,
        rspr_ms,
    );

    // Verbose output to stderr.
    eprintln!("Instance: m={} n={}", m, n);
    eprintln!("Kernelized: {} → {} ({} removed)", n, n_red, kern_removed);
    eprintln!(
        "Kelk common-cluster: {} (size={}) in {:.1}ms",
        if kelk_found { "FOUND" } else { "none" },
        kelk_size,
        kelk_ms
    );
    eprintln!(
        "rspr clusters: {} clusters, max_size={}, remainder={} in {:.1}ms",
        rspr_count, rspr_max, rspr_remainder, rspr_ms
    );
    if !rspr_sizes.is_empty() {
        let mut sorted = rspr_sizes.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let sizes_str: Vec<String> = sorted.iter().map(|s| s.to_string()).collect();
        eprintln!("  cluster sizes (desc): [{}]", sizes_str.join(", "));
    }

    Ok(())
}
