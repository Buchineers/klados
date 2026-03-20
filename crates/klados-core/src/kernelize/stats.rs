//! Kernelization statistics and output.

use std::collections::BTreeMap;
use fxhash::FxHashMap;

/// Statistics from a kernelization run.
#[derive(Clone, Debug, Default)]
pub struct KernelizeStats {
    pub original_leaves: u32,
    pub reduced_leaves: u32,
    /// Per-rule firing counts: rule_name -> count.
    pub rule_counts: BTreeMap<&'static str, usize>,
    /// Labels deleted by parameter-reducing rules, in original label space.
    pub deleted_labels: Vec<u32>,
    /// For each surviving label in the reduced instance: (reduced_label, original_labels_it_represents)
    pub surviving_taxa: Vec<(u32, Vec<u32>)>,
}

impl KernelizeStats {
    /// Number of leaves removed by cherry/subtree reduction.
    pub fn subtree_removed(&self) -> usize {
        *self.rule_counts.get("cherry").unwrap_or(&0)
    }

    /// Number of leaves removed by chain reduction.
    pub fn chain_removed(&self) -> usize {
        *self.rule_counts.get("chain").unwrap_or(&0)
    }

    /// Number of leaves removed by 3-2 chain reduction.
    pub fn chain32_removed(&self) -> usize {
        *self.rule_counts.get("chain-3-2").unwrap_or(&0)
    }

    /// Total leaves removed across all rules.
    pub fn total_removed(&self) -> usize {
        self.rule_counts.values().sum()
    }
}

/// Print kernelization results.
/// Maintains backward-compatible stderr format for `just kernelize-test` parser.
pub fn print_stats(stats: &KernelizeStats) {
    let total_removed = stats.total_removed();
    let pct = if stats.original_leaves > 0 {
        100.0 * total_removed as f64 / stats.original_leaves as f64
    } else {
        0.0
    };
    eprintln!("// --- Kernelization is finished.");
    eprintln!("// Leaves in original instance: {}", stats.original_leaves);
    eprintln!("// Leaves in reduced instance: {}", stats.reduced_leaves);
    eprintln!(
        "// So {} leaves removed ({:.1}% reduction). Breakdown is as follows:",
        total_removed, pct
    );
    // Fixed-name rules in backward-compatible format
    eprintln!("// --- Due to subtree reduction: {}", stats.subtree_removed());
    eprintln!("// --- Due to chain reduction: {}", stats.chain_removed());
    eprintln!(
        "// --- Due to 3-2 chain reduction: {}",
        stats.chain32_removed()
    );
    // Any additional rules beyond the core three
    for (name, &count) in &stats.rule_counts {
        if *name != "cherry" && *name != "chain" && *name != "chain-3-2" && count > 0 {
            eprintln!("// --- Due to {} reduction: {}", name, count);
        }
    }
}

/// Print detailed taxon info matching Kelk's verbose output.
pub fn print_taxa_detail(stats: &KernelizeStats) {
    eprintln!("// Explaining the semantics of the surviving and deleted taxa:");
    for (rep, group) in &stats.surviving_taxa {
        let labels: Vec<String> = group.iter().map(|l| l.to_string()).collect();
        eprintln!(
            "// Taxon {} represents the following subset of taxa from the original trees: {}",
            rep,
            labels.join(",")
        );
    }
    for &del in &stats.deleted_labels {
        eprintln!("// Parameter-reducing rule deleted taxon {}", del);
    }
}

/// Build info about which original taxa each surviving label represents.
pub fn build_surviving_taxa(
    composite_rev: &[u32],
    all_collapses: &[(u32, Vec<u32>)],
    reduced_leaves: u32,
    _original_leaves: u32,
) -> Vec<(u32, Vec<u32>)> {
    let mut label_groups: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for reduced_lbl in 1..=reduced_leaves {
        let orig = composite_rev[reduced_lbl as usize];
        label_groups.entry(orig).or_default();
    }
    for (rep, removed) in all_collapses {
        let group = label_groups.entry(*rep).or_default();
        if group.is_empty() {
            group.push(*rep);
        }
        for &r in removed {
            if !group.contains(&r) {
                group.push(r);
            }
        }
    }
    for reduced_lbl in 1..=reduced_leaves {
        let orig = composite_rev[reduced_lbl as usize];
        let group = label_groups.entry(orig).or_default();
        if group.is_empty() {
            group.push(orig);
        } else if !group.contains(&orig) {
            group.insert(0, orig);
        }
    }

    let mut result: Vec<(u32, Vec<u32>)> = label_groups.into_iter().collect();
    result.sort_by_key(|(k, _)| *k);
    result
}
