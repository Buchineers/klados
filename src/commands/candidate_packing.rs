//! Exact packing over a mined non-singleton component pool.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Write;
use std::path::Path;

use fixedbitset::FixedBitSet;
use highs::{Col, HighsModelStatus, RowProblem, Sense};
use klados_core::kernelize::{self, KernelizeConfig};
use klados_core::{Instance, NodeId, Tree};
use pace26io::newick::NewickWriter;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum MasterMode {
    Pairwise,
    Nodepack,
}

#[derive(Debug, Deserialize)]
struct CandidatePoolFile {
    all_candidates: Option<Vec<MinedCandidate>>,
    top_candidates: Vec<MinedCandidate>,
}

#[derive(Debug, Clone, Deserialize)]
struct MinedCandidate {
    labels: Vec<u32>,
    size: usize,
    seen_count: usize,
    modes_seen: Vec<String>,
    max_depth_from_ub: Option<isize>,
    mean_depth_from_ub: Option<f64>,
}

#[derive(Debug, Clone)]
struct PackedCandidate {
    original_labels: Vec<u32>,
    reduced_labels: Vec<u32>,
    weight: usize,
    seen_count: usize,
    modes_seen: Vec<String>,
    max_depth_from_ub: Option<isize>,
    mean_depth_from_ub: Option<f64>,
    leafset: FixedBitSet,
    node_covers: Vec<FixedBitSet>,
    source: CandidateSource,
    introduced_round: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "kebab-case")]
enum CandidateSource {
    Mined,
    Priced,
    Subset,
    AddOne,
}

#[derive(Debug, Serialize)]
struct PackingReport {
    master_mode: MasterMode,
    input_leaves: u32,
    reduced_leaves: u32,
    kernel_removed: u32,
    kernel_subtree_removed: usize,
    kernel_chain_removed: usize,
    kernel_chain32_removed: usize,
    param_reduction: usize,
    candidates_requested: usize,
    candidates_loaded: usize,
    candidates_used_final: usize,
    candidates_dropped_missing_after_kernel: usize,
    candidates_dropped_too_small: usize,
    candidates_dropped_incompatible: usize,
    candidates_deduplicated_initial: usize,
    rounds: Vec<RoundSummary>,
    total_savings: usize,
    reduced_component_count: usize,
    projected_original_component_count: usize,
    selected_histogram: Vec<SizeCount>,
    selected_candidates: Vec<SelectedCandidate>,
}

#[derive(Debug, Serialize)]
struct RoundSummary {
    round: usize,
    candidate_count: usize,
    leaf_conflicts: usize,
    tree_conflicts: usize,
    graph_components: Vec<usize>,
    total_savings: usize,
    reduced_component_count: usize,
    selected_count: usize,
    generated_priced: usize,
    generated_subsets: usize,
    generated_add_one: usize,
    generated_total: usize,
    skipped_expansion_due_to_priced: bool,
    best_savings_so_far: usize,
    consecutive_stall_rounds: usize,
    master_comparison: Option<MasterComparison>,
    priced_component: Option<PricedComponentReport>,
}

#[derive(Debug, Serialize)]
struct MasterComparison {
    leaf_rows: usize,
    pairwise_tree_rows: usize,
    nodepack_rows: usize,
    pairwise_ilp: MasterSolveReport,
    pairwise_lp: MasterSolveReport,
    nodepack_ilp: MasterSolveReport,
    nodepack_lp: MasterSolveReport,
}

#[derive(Debug, Serialize)]
struct MasterSolveReport {
    objective: f64,
    selected_count: Option<usize>,
    nonzero_vars: usize,
    fractional_vars: usize,
    max_fractionality: f64,
}

#[derive(Debug, Serialize)]
struct PricedComponentReport {
    dual_sign_flipped: bool,
    score: f64,
    reduced_cost: f64,
    best_in_pool_score: f64,
    best_in_pool_reduced_cost: f64,
    beats_best_in_pool_by: f64,
    size: usize,
    reduced_labels: Vec<u32>,
    original_labels: Vec<u32>,
    already_in_pool: bool,
}

#[derive(Debug, Serialize)]
struct SizeCount {
    size: usize,
    count: usize,
}

#[derive(Debug, Serialize)]
struct SelectedCandidate {
    original_labels: Vec<u32>,
    reduced_labels: Vec<u32>,
    size: usize,
    weight: usize,
    seen_count: usize,
    modes_seen: Vec<String>,
    max_depth_from_ub: Option<isize>,
    mean_depth_from_ub: Option<f64>,
    source: CandidateSource,
    introduced_round: usize,
}

#[derive(Debug)]
struct ConflictData {
    adjacency: Vec<Vec<usize>>,
    tree_conflicts: Vec<(usize, usize)>,
    leaf_conflicts: usize,
}

pub fn run(
    instance: &Instance,
    candidates_file: &Path,
    solution_out: Option<&Path>,
    limit: usize,
    min_seen: usize,
    min_size: usize,
    rounds: usize,
    expand_subsets: bool,
    expand_add_one: bool,
    add_one_max_base_size: usize,
    add_one_max_rise: u16,
    add_one_top_k: usize,
    generated_cap: usize,
    master_mode: MasterMode,
    compare_masters: bool,
    price_missing: bool,
    price_add: bool,
    defer_expansion_while_priced: bool,
    stall_patience: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let pool_text = std::fs::read_to_string(candidates_file)?;
    let pool: CandidatePoolFile = serde_json::from_str(&pool_text)?;

    let raw_candidates = pool.all_candidates.unwrap_or(pool.top_candidates);
    let requested = if limit == 0 {
        raw_candidates.len()
    } else {
        raw_candidates.len().min(limit)
    };

    let kern = kernelize::kernelize_best(instance, &KernelizeConfig::default());
    let reduced = &kern.instance;
    let n_reduced = reduced.num_leaves as usize;
    let trees = &reduced.trees;

    let mut original_to_reduced = vec![0u32; instance.num_leaves as usize + 1];
    for reduced_lbl in 1..=reduced.num_leaves {
        let original_lbl = kern.reverse_map[reduced_lbl as usize];
        if (original_lbl as usize) < original_to_reduced.len() {
            original_to_reduced[original_lbl as usize] = reduced_lbl;
        }
    }

    let mut dedup = HashMap::<Vec<u32>, usize>::new();
    let mut kept = Vec::<PackedCandidate>::new();
    let mut dropped_missing = 0usize;
    let mut dropped_too_small = 0usize;
    let mut dropped_incompatible = 0usize;
    let mut deduplicated = 0usize;

    for candidate in raw_candidates.into_iter().take(requested) {
        if candidate.seen_count < min_seen || candidate.size < min_size {
            dropped_too_small += 1;
            continue;
        }
        let mut reduced_labels = Vec::with_capacity(candidate.labels.len());
        let mut missing = false;
        for &orig in &candidate.labels {
            let mapped = original_to_reduced.get(orig as usize).copied().unwrap_or(0);
            if mapped == 0 {
                missing = true;
                break;
            }
            reduced_labels.push(mapped);
        }
        if missing {
            dropped_missing += 1;
            continue;
        }
        reduced_labels.sort_unstable();
        reduced_labels.dedup();
        match build_candidate(
            &reduced_labels,
            &kern.reverse_map,
            trees,
            n_reduced,
            candidate.seen_count,
            candidate.modes_seen,
            candidate.max_depth_from_ub,
            candidate.mean_depth_from_ub,
            CandidateSource::Mined,
            0,
            &mut dedup,
            &mut kept,
        ) {
            BuildResult::Inserted => {}
            BuildResult::TooSmall => dropped_too_small += 1,
            BuildResult::Incompatible => dropped_incompatible += 1,
            BuildResult::Duplicate => deduplicated += 1,
        }
    }

    let mut round_summaries = Vec::new();
    let mut final_selected = Vec::new();
    let mut final_savings = 0usize;
    let mut best_savings_so_far = 0usize;
    let mut consecutive_stall_rounds = 0usize;

    for round in 0..=rounds {
        let conflicts = build_conflicts(&kept, n_reduced);
        let graph_components = connected_components(&conflicts.adjacency);
        let selected = solve_packing_ilp(&kept, &conflicts, trees, n_reduced, master_mode)?;
        validate_selection(&selected, &kept, trees)?;
        let total_savings = selected.iter().map(|&idx| kept[idx].weight).sum::<usize>();
        let reduced_component_count = n_reduced.saturating_sub(total_savings);
        if round == 0 || total_savings > best_savings_so_far {
            best_savings_so_far = total_savings;
            consecutive_stall_rounds = 0;
        } else {
            consecutive_stall_rounds += 1;
        }
        let master_comparison = if compare_masters {
            Some(compare_masters_on_pool(
                &kept, &conflicts, trees, n_reduced,
            )?)
        } else {
            None
        };
        let priced_component = if (price_missing || price_add) && trees.len() == 2 {
            price_missing_component(&kept, trees, n_reduced, &kern.reverse_map)?
        } else {
            None
        };

        let mut generated_priced = 0usize;
        let mut generated_subsets = 0usize;
        let mut generated_add_one = 0usize;
        let mut skipped_expansion_due_to_priced = false;
        if round < rounds {
            if price_add {
                if let Some(priced) = &priced_component {
                    if !priced.already_in_pool && priced.reduced_cost > 1e-6 {
                        if matches!(
                            build_candidate(
                                &priced.reduced_labels,
                                &kern.reverse_map,
                                trees,
                                n_reduced,
                                0,
                                vec!["generated-priced".to_string()],
                                None,
                                None,
                                CandidateSource::Priced,
                                round + 1,
                                &mut dedup,
                                &mut kept,
                            ),
                            BuildResult::Inserted
                        ) {
                            generated_priced = 1;
                        }
                    }
                }
            }
            let expansion = if defer_expansion_while_priced && generated_priced > 0 {
                skipped_expansion_due_to_priced = true;
                ExpansionCounts {
                    generated_subsets: 0,
                    generated_add_one: 0,
                    generated_total: 0,
                }
            } else {
                expand_candidates(
                    &selected,
                    &mut kept,
                    &mut dedup,
                    &kern.reverse_map,
                    trees,
                    n_reduced,
                    round + 1,
                    expand_subsets,
                    expand_add_one,
                    add_one_max_base_size,
                    add_one_max_rise,
                    add_one_top_k,
                    generated_cap,
                )
            };
            generated_subsets = expansion.generated_subsets;
            generated_add_one = expansion.generated_add_one;
            if generated_priced + expansion.generated_total == 0 {
                round_summaries.push(RoundSummary {
                    round,
                    candidate_count: kept.len(),
                    leaf_conflicts: conflicts.leaf_conflicts,
                    tree_conflicts: conflicts.tree_conflicts.len(),
                    graph_components: graph_components.iter().map(Vec::len).collect(),
                    total_savings,
                    reduced_component_count,
                    selected_count: selected.len(),
                    generated_priced,
                    generated_subsets,
                    generated_add_one,
                    generated_total: generated_priced,
                    skipped_expansion_due_to_priced,
                    best_savings_so_far,
                    consecutive_stall_rounds,
                    master_comparison,
                    priced_component,
                });
                final_selected = selected;
                final_savings = total_savings;
                break;
            }
        }

        round_summaries.push(RoundSummary {
            round,
            candidate_count: kept.len(),
            leaf_conflicts: conflicts.leaf_conflicts,
            tree_conflicts: conflicts.tree_conflicts.len(),
            graph_components: graph_components.iter().map(Vec::len).collect(),
            total_savings,
            reduced_component_count,
            selected_count: selected.len(),
            generated_priced,
            generated_subsets,
            generated_add_one,
            generated_total: generated_priced + generated_subsets + generated_add_one,
            skipped_expansion_due_to_priced,
            best_savings_so_far,
            consecutive_stall_rounds,
            master_comparison,
            priced_component,
        });

        final_selected = selected;
        final_savings = total_savings;

        if stall_patience != 0 && consecutive_stall_rounds >= stall_patience {
            break;
        }
    }

    let reduced_component_count = n_reduced.saturating_sub(final_savings);
    let projected_original_component_count = reduced_component_count + kern.param_reduction;
    let expanded_components = build_expanded_solution(
        &final_selected,
        &kept,
        reduced,
        n_reduced,
        &kern,
        &instance.trees[0],
        instance.num_leaves,
    );
    if expanded_components.len() != projected_original_component_count {
        return Err(format!(
            "expanded solution size {} != projected {}",
            expanded_components.len(),
            projected_original_component_count
        )
        .into());
    }
    let mut hist = BTreeMap::<usize, usize>::new();
    let mut selected_candidates = Vec::new();
    for &idx in &final_selected {
        let cand = &kept[idx];
        *hist.entry(cand.reduced_labels.len()).or_default() += 1;
        selected_candidates.push(SelectedCandidate {
            original_labels: cand.original_labels.clone(),
            reduced_labels: cand.reduced_labels.clone(),
            size: cand.reduced_labels.len(),
            weight: cand.weight,
            seen_count: cand.seen_count,
            modes_seen: cand.modes_seen.clone(),
            max_depth_from_ub: cand.max_depth_from_ub,
            mean_depth_from_ub: cand.mean_depth_from_ub,
            source: cand.source.clone(),
            introduced_round: cand.introduced_round,
        });
    }
    selected_candidates.sort_by(|a, b| {
        b.size
            .cmp(&a.size)
            .then_with(|| b.seen_count.cmp(&a.seen_count))
            .then_with(|| a.original_labels.cmp(&b.original_labels))
    });

    let report = PackingReport {
        master_mode,
        input_leaves: instance.num_leaves,
        reduced_leaves: reduced.num_leaves,
        kernel_removed: kern
            .stats
            .original_leaves
            .saturating_sub(kern.stats.reduced_leaves),
        kernel_subtree_removed: kern.stats.subtree_removed(),
        kernel_chain_removed: kern.stats.chain_removed(),
        kernel_chain32_removed: kern.stats.chain32_removed(),
        param_reduction: kern.param_reduction,
        candidates_requested: requested,
        candidates_loaded: kept.len()
            + dropped_missing
            + dropped_too_small
            + dropped_incompatible
            + deduplicated,
        candidates_used_final: kept.len(),
        candidates_dropped_missing_after_kernel: dropped_missing,
        candidates_dropped_too_small: dropped_too_small,
        candidates_dropped_incompatible: dropped_incompatible,
        candidates_deduplicated_initial: deduplicated,
        rounds: round_summaries,
        total_savings: final_savings,
        reduced_component_count,
        projected_original_component_count,
        selected_histogram: hist
            .into_iter()
            .map(|(size, count)| SizeCount { size, count })
            .collect(),
        selected_candidates,
    };

    if let Some(path) = solution_out {
        write_solution_file(path, &expanded_components)?;
    }

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

enum BuildResult {
    Inserted,
    TooSmall,
    Incompatible,
    Duplicate,
}

fn build_candidate(
    reduced_labels: &[u32],
    reverse_map: &[u32],
    trees: &[Tree],
    n_reduced: usize,
    seen_count: usize,
    modes_seen: Vec<String>,
    max_depth_from_ub: Option<isize>,
    mean_depth_from_ub: Option<f64>,
    source: CandidateSource,
    introduced_round: usize,
    dedup: &mut HashMap<Vec<u32>, usize>,
    kept: &mut Vec<PackedCandidate>,
) -> BuildResult {
    if reduced_labels.len() < 2 {
        return BuildResult::TooSmall;
    }
    if dedup.contains_key(reduced_labels) {
        return BuildResult::Duplicate;
    }
    if !is_set_compatible_all(trees, reduced_labels) {
        return BuildResult::Incompatible;
    }
    let original_labels = reduced_labels
        .iter()
        .map(|&lbl| reverse_map[lbl as usize])
        .collect::<Vec<_>>();
    let leafset = make_leafset(reduced_labels, n_reduced);
    let node_covers = trees
        .iter()
        .map(|tree| mark_component_nodes(tree, reduced_labels))
        .collect::<Vec<_>>();

    let packed = PackedCandidate {
        original_labels,
        reduced_labels: reduced_labels.to_vec(),
        weight: reduced_labels.len() - 1,
        seen_count,
        modes_seen,
        max_depth_from_ub,
        mean_depth_from_ub,
        leafset,
        node_covers,
        source,
        introduced_round,
    };
    dedup.insert(reduced_labels.to_vec(), kept.len());
    kept.push(packed);
    BuildResult::Inserted
}

struct ExpansionCounts {
    generated_subsets: usize,
    generated_add_one: usize,
    generated_total: usize,
}

fn expand_candidates(
    selected: &[usize],
    kept: &mut Vec<PackedCandidate>,
    dedup: &mut HashMap<Vec<u32>, usize>,
    reverse_map: &[u32],
    trees: &[Tree],
    n_reduced: usize,
    round: usize,
    expand_subsets: bool,
    expand_add_one: bool,
    add_one_max_base_size: usize,
    add_one_max_rise: u16,
    add_one_top_k: usize,
    generated_cap: usize,
) -> ExpansionCounts {
    let mut generated_subsets = 0usize;
    let mut generated_add_one = 0usize;

    for &idx in selected {
        if generated_subsets + generated_add_one >= generated_cap {
            break;
        }
        let base = kept[idx].reduced_labels.clone();
        if expand_subsets && base.len() > 2 {
            let subset_masks = 1usize << base.len();
            for mask in 1usize..subset_masks {
                if generated_subsets + generated_add_one >= generated_cap {
                    break;
                }
                let subset_size = mask.count_ones() as usize;
                if subset_size < 2 || subset_size == base.len() {
                    continue;
                }
                let subset = base
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &lbl)| {
                        if (mask & (1usize << i)) != 0 {
                            Some(lbl)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                if matches!(
                    build_candidate(
                        &subset,
                        reverse_map,
                        trees,
                        n_reduced,
                        0,
                        vec!["generated-subset".to_string()],
                        None,
                        None,
                        CandidateSource::Subset,
                        round,
                        dedup,
                        kept,
                    ),
                    BuildResult::Inserted
                ) {
                    generated_subsets += 1;
                }
            }
        }

        if expand_add_one && base.len() <= add_one_max_base_size {
            let extra_labels =
                rank_add_one_extras(&base, trees, n_reduced, add_one_max_rise, add_one_top_k);
            for extra in extra_labels {
                if generated_subsets + generated_add_one >= generated_cap {
                    break;
                }
                let mut sup = base.clone();
                match sup.binary_search(&extra) {
                    Ok(_) => continue,
                    Err(pos) => sup.insert(pos, extra),
                }
                if matches!(
                    build_candidate(
                        &sup,
                        reverse_map,
                        trees,
                        n_reduced,
                        0,
                        vec!["generated-add-one".to_string()],
                        None,
                        None,
                        CandidateSource::AddOne,
                        round,
                        dedup,
                        kept,
                    ),
                    BuildResult::Inserted
                ) {
                    generated_add_one += 1;
                }
            }
        }
    }

    ExpansionCounts {
        generated_subsets,
        generated_add_one,
        generated_total: generated_subsets + generated_add_one,
    }
}

fn rank_add_one_extras(
    base: &[u32],
    trees: &[Tree],
    n_reduced: usize,
    max_rise: u16,
    top_k: usize,
) -> Vec<u32> {
    if base.is_empty() {
        return Vec::new();
    }
    let base_lcas = trees
        .iter()
        .map(|tree| lca_of_labels(tree, base))
        .collect::<Vec<_>>();
    let mut scored = Vec::<(u16, u16, u32)>::new();
    for extra in 1..=n_reduced as u32 {
        if base.binary_search(&extra).is_ok() {
            continue;
        }
        let mut total_rise = 0u16;
        let mut max_tree_rise = 0u16;
        let mut ok = true;
        for (tree, &base_lca) in trees.iter().zip(&base_lcas) {
            let join = tree.nearest_common_ancestor(base_lca, tree.node_by_label(extra));
            let rise = tree.depth[base_lca as usize] - tree.depth[join as usize];
            if rise > max_rise {
                ok = false;
                break;
            }
            total_rise = total_rise.saturating_add(rise);
            max_tree_rise = max_tree_rise.max(rise);
        }
        if ok {
            scored.push((total_rise, max_tree_rise, extra));
        }
    }
    scored.sort_unstable();
    if top_k != 0 && scored.len() > top_k {
        scored.truncate(top_k);
    }
    scored.into_iter().map(|(_, _, extra)| extra).collect()
}

fn lca_of_labels(tree: &Tree, labels: &[u32]) -> u32 {
    let mut iter = labels.iter().copied();
    let first = iter.next().expect("base must be non-empty");
    let mut lca = tree.node_by_label(first);
    for lbl in iter {
        lca = tree.nearest_common_ancestor(lca, tree.node_by_label(lbl));
    }
    lca
}

fn make_leafset(labels: &[u32], n: usize) -> FixedBitSet {
    let mut bits = FixedBitSet::with_capacity(n + 1);
    for &lbl in labels {
        bits.insert(lbl as usize);
    }
    bits
}

fn is_set_compatible_all(trees: &[Tree], labels: &[u32]) -> bool {
    if labels.len() <= 2 {
        return true;
    }
    for i in 0..labels.len() {
        for j in (i + 1)..labels.len() {
            for k in (j + 1)..labels.len() {
                let a = labels[i];
                let b = labels[j];
                let c = labels[k];
                let topo0 = triplet_topology(&trees[0], a, b, c);
                if trees[1..]
                    .iter()
                    .any(|tree| triplet_topology(tree, a, b, c) != topo0)
                {
                    return false;
                }
            }
        }
    }
    true
}

fn triplet_topology(tree: &Tree, x: u32, y: u32, z: u32) -> u8 {
    let nx = tree.node_by_label(x);
    let ny = tree.node_by_label(y);
    let nz = tree.node_by_label(z);
    let lxy = tree.nearest_common_ancestor(nx, ny);
    let lxz = tree.nearest_common_ancestor(nx, nz);
    let lyz = tree.nearest_common_ancestor(ny, nz);
    let dxy = tree.depth[lxy as usize];
    let dxz = tree.depth[lxz as usize];
    let dyz = tree.depth[lyz as usize];
    if dxy > dxz && dxy > dyz {
        0
    } else if dxz > dxy && dxz > dyz {
        1
    } else {
        2
    }
}

fn mark_component_nodes(tree: &Tree, labels: &[u32]) -> FixedBitSet {
    let mut bits = FixedBitSet::with_capacity(tree.num_nodes());
    if labels.is_empty() {
        return bits;
    }
    let mut lca_node = tree.node_by_label(labels[0]);
    for &lbl in &labels[1..] {
        lca_node = tree.nearest_common_ancestor(lca_node, tree.node_by_label(lbl));
    }
    for &lbl in labels {
        let mut cur = tree.node_by_label(lbl);
        loop {
            bits.insert(cur as usize);
            if cur == lca_node {
                break;
            }
            cur = tree.parent[cur as usize];
        }
    }
    bits
}

fn bitsets_intersect(a: &FixedBitSet, b: &FixedBitSet) -> bool {
    let aslice = a.as_slice();
    let bslice = b.as_slice();
    let len = aslice.len().min(bslice.len());
    for i in 0..len {
        if aslice[i] & bslice[i] != 0 {
            return true;
        }
    }
    false
}

fn candidates_conflict(a: &PackedCandidate, b: &PackedCandidate) -> (bool, bool) {
    let leaf_conflict = bitsets_intersect(&a.leafset, &b.leafset);
    let tree_conflict = a
        .node_covers
        .iter()
        .zip(&b.node_covers)
        .any(|(na, nb)| bitsets_intersect(na, nb));
    (leaf_conflict, tree_conflict)
}

fn build_conflicts(candidates: &[PackedCandidate], n_reduced: usize) -> ConflictData {
    let mut adjacency = vec![Vec::new(); candidates.len()];
    let mut tree_conflicts = Vec::new();
    let mut leaf_conflicts = 0usize;

    let mut by_leaf = vec![Vec::<usize>::new(); n_reduced + 1];
    for (idx, cand) in candidates.iter().enumerate() {
        for lbl in cand.leafset.ones() {
            by_leaf[lbl].push(idx);
        }
    }

    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            let (leaf_conflict, tree_conflict) =
                candidates_conflict(&candidates[i], &candidates[j]);
            if leaf_conflict {
                leaf_conflicts += 1;
                adjacency[i].push(j);
                adjacency[j].push(i);
            } else if tree_conflict {
                tree_conflicts.push((i, j));
                adjacency[i].push(j);
                adjacency[j].push(i);
            }
        }
    }

    // Add remaining leaf-conflict neighbors from incidence, avoiding duplicates.
    let _ = by_leaf;

    ConflictData {
        adjacency,
        tree_conflicts,
        leaf_conflicts,
    }
}

fn connected_components(adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adjacency.len();
    let mut seen = vec![false; n];
    let mut components = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut comp = Vec::new();
        let mut queue = VecDeque::from([start]);
        seen[start] = true;
        while let Some(v) = queue.pop_front() {
            comp.push(v);
            for &u in &adjacency[v] {
                if !seen[u] {
                    seen[u] = true;
                    queue.push_back(u);
                }
            }
        }
        components.push(comp);
    }
    components
}

fn solve_packing_ilp(
    candidates: &[PackedCandidate],
    conflicts: &ConflictData,
    trees: &[Tree],
    n_reduced: usize,
    master_mode: MasterMode,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let result = solve_master(
        candidates,
        Some(conflicts),
        trees,
        n_reduced,
        master_mode,
        true,
    )?;
    Ok(result
        .selected
        .expect("integer master solve should return a selection"))
}

struct SolveResult {
    objective: f64,
    selected: Option<Vec<usize>>,
    nonzero_vars: usize,
    fractional_vars: usize,
    max_fractionality: f64,
}

struct PaperRmpLpSolution {
    alpha: Vec<f64>,
    beta: Vec<Vec<f64>>,
}

#[derive(Clone, Copy)]
enum SplitChoice {
    None,
    Straight,
    Cross,
}

#[derive(Clone, Copy)]
enum VChoice {
    None,
    LeafMatch(u32),
    UseRooted,
    SkipLeftU,
    SkipRightU,
    SkipLeftV,
    SkipRightV,
}

#[derive(Clone, Copy)]
enum MChoice {
    None,
    LeafMatch(u32),
    UseRooted,
    SkipLeftU,
    SkipRightU,
    SkipLeftV,
    SkipRightV,
}

fn compare_masters_on_pool(
    candidates: &[PackedCandidate],
    conflicts: &ConflictData,
    trees: &[Tree],
    n_reduced: usize,
) -> Result<MasterComparison, Box<dyn std::error::Error>> {
    let leaf_rows = count_leaf_rows(candidates, n_reduced);
    let nodepack_rows = count_nodepack_rows(candidates, trees);

    let pairwise_ilp = solve_master(
        candidates,
        Some(conflicts),
        trees,
        n_reduced,
        MasterMode::Pairwise,
        true,
    )?;
    let pairwise_lp = solve_master(
        candidates,
        Some(conflicts),
        trees,
        n_reduced,
        MasterMode::Pairwise,
        false,
    )?;
    let nodepack_ilp = solve_master(
        candidates,
        Some(conflicts),
        trees,
        n_reduced,
        MasterMode::Nodepack,
        true,
    )?;
    let nodepack_lp = solve_master(
        candidates,
        Some(conflicts),
        trees,
        n_reduced,
        MasterMode::Nodepack,
        false,
    )?;

    if let Some(selected) = &nodepack_ilp.selected {
        validate_selection(selected, candidates, trees)?;
    }

    Ok(MasterComparison {
        leaf_rows,
        pairwise_tree_rows: conflicts.tree_conflicts.len(),
        nodepack_rows,
        pairwise_ilp: to_master_report(pairwise_ilp),
        pairwise_lp: to_master_report(pairwise_lp),
        nodepack_ilp: to_master_report(nodepack_ilp),
        nodepack_lp: to_master_report(nodepack_lp),
    })
}

fn to_master_report(result: SolveResult) -> MasterSolveReport {
    MasterSolveReport {
        objective: result.objective,
        selected_count: result.selected.as_ref().map(Vec::len),
        nonzero_vars: result.nonzero_vars,
        fractional_vars: result.fractional_vars,
        max_fractionality: result.max_fractionality,
    }
}

fn count_leaf_rows(candidates: &[PackedCandidate], n_reduced: usize) -> usize {
    let mut rows = 0usize;
    for leaf in 1..=n_reduced {
        let count = candidates
            .iter()
            .filter(|cand| cand.leafset.contains(leaf))
            .count();
        if count >= 2 {
            rows += 1;
        }
    }
    rows
}

fn count_nodepack_rows(candidates: &[PackedCandidate], trees: &[Tree]) -> usize {
    let mut rows = 0usize;
    for (ti, tree) in trees.iter().enumerate() {
        for node in 0..tree.num_nodes() as u32 {
            if tree.is_leaf(node) {
                continue;
            }
            let count = candidates
                .iter()
                .filter(|cand| cand.node_covers[ti].contains(node as usize))
                .count();
            if count >= 2 {
                rows += 1;
            }
        }
    }
    rows
}

fn solve_master(
    candidates: &[PackedCandidate],
    conflicts: Option<&ConflictData>,
    trees: &[Tree],
    n_reduced: usize,
    kind: MasterMode,
    integer: bool,
) -> Result<SolveResult, Box<dyn std::error::Error>> {
    let mut pb = RowProblem::default();
    let vars: Vec<Col> = candidates
        .iter()
        .map(|cand| {
            if integer {
                pb.add_integer_column(cand.weight as f64, 0.0..=1.0)
            } else {
                pb.add_column(cand.weight as f64, 0.0..=1.0)
            }
        })
        .collect();

    for leaf in 1..=n_reduced {
        let mut cols = Vec::new();
        for (idx, cand) in candidates.iter().enumerate() {
            if cand.leafset.contains(leaf) {
                cols.push((vars[idx], 1.0));
            }
        }
        if cols.len() >= 2 {
            pb.add_row(..=1.0, &cols);
        }
    }

    match kind {
        MasterMode::Pairwise => {
            let conflicts = conflicts.expect("pairwise master requires conflict data");
            for &(i, j) in &conflicts.tree_conflicts {
                pb.add_row(..=1.0, [(vars[i], 1.0), (vars[j], 1.0)]);
            }
        }
        MasterMode::Nodepack => {
            for (ti, tree) in trees.iter().enumerate() {
                for node in 0..tree.num_nodes() as u32 {
                    if tree.is_leaf(node) {
                        continue;
                    }
                    let mut cols = Vec::new();
                    for (idx, cand) in candidates.iter().enumerate() {
                        if cand.node_covers[ti].contains(node as usize) {
                            cols.push((vars[idx], 1.0));
                        }
                    }
                    if cols.len() >= 2 {
                        pb.add_row(..=1.0, &cols);
                    }
                }
            }
        }
    }

    let mut model = pb.optimise(Sense::Maximise);
    model.make_quiet();
    model.set_option("threads", 1_i32);
    model.set_option("presolve", "on");
    let solved = model.solve();
    if solved.status() != HighsModelStatus::Optimal {
        return Err(format!("candidate master status: {:?}", solved.status()).into());
    }
    let solution = solved.get_solution();
    let columns = solution.columns();
    let mut selected = None;
    if integer {
        selected = Some(
            columns
                .iter()
                .enumerate()
                .filter_map(|(idx, &value)| if value > 0.5 { Some(idx) } else { None })
                .collect(),
        );
    }
    let mut nonzero_vars = 0usize;
    let mut fractional_vars = 0usize;
    let mut max_fractionality = 0.0f64;
    for &value in columns {
        if value > 1e-9 {
            nonzero_vars += 1;
        }
        let nearest = if value <= 0.5 { 0.0 } else { 1.0 };
        let frac = (value - nearest).abs();
        if frac > 1e-6 && frac < 1.0 - 1e-6 {
            fractional_vars += 1;
            if frac > max_fractionality {
                max_fractionality = frac;
            }
        }
    }

    Ok(SolveResult {
        objective: solved.objective_value(),
        selected,
        nonzero_vars,
        fractional_vars,
        max_fractionality,
    })
}

fn solve_paper_rmp_lp(
    candidates: &[PackedCandidate],
    trees: &[Tree],
    n_reduced: usize,
) -> Result<PaperRmpLpSolution, Box<dyn std::error::Error>> {
    let mut pb = RowProblem::default();
    let singleton_vars: Vec<Col> = (1..=n_reduced)
        .map(|_| pb.add_column(1.0, 0.0..=1.0))
        .collect();
    let candidate_vars: Vec<Col> = candidates
        .iter()
        .map(|_| pb.add_column(1.0, 0.0..=1.0))
        .collect();

    let mut leaf_row_idx = vec![None; n_reduced + 1];
    let mut node_row_idx: Vec<Vec<Option<usize>>> = trees
        .iter()
        .map(|tree| vec![None; tree.num_nodes()])
        .collect();
    let mut next_row = 0usize;

    for leaf in 1..=n_reduced {
        let mut cols = vec![(singleton_vars[leaf - 1], 1.0)];
        for (idx, cand) in candidates.iter().enumerate() {
            if cand.leafset.contains(leaf) {
                cols.push((candidate_vars[idx], 1.0));
            }
        }
        pb.add_row(1.0.., &cols);
        leaf_row_idx[leaf] = Some(next_row);
        next_row += 1;
    }

    for (ti, tree) in trees.iter().enumerate() {
        for node in 0..tree.num_nodes() as u32 {
            if tree.is_leaf(node) {
                continue;
            }
            let mut cols = Vec::new();
            for (idx, cand) in candidates.iter().enumerate() {
                if cand.node_covers[ti].contains(node as usize) {
                    cols.push((candidate_vars[idx], 1.0));
                }
            }
            if cols.len() >= 2 {
                pb.add_row(..=1.0, &cols);
                node_row_idx[ti][node as usize] = Some(next_row);
                next_row += 1;
            }
        }
    }

    let mut model = pb.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_option("threads", 1_i32);
    model.set_option("presolve", "on");
    let solved = model.solve();
    if solved.status() != HighsModelStatus::Optimal {
        return Err(format!("pricing LP status: {:?}", solved.status()).into());
    }

    let solution = solved.get_solution();
    let mut raw_cover = vec![0.0; n_reduced + 1];
    let mut raw_pack: Vec<Vec<f64>> = trees
        .iter()
        .map(|tree| vec![0.0; tree.num_nodes()])
        .collect();
    for leaf in 1..=n_reduced {
        if let Some(row) = leaf_row_idx[leaf] {
            raw_cover[leaf] = solution.dual_rows()[row];
        }
    }
    for (ti, tree) in trees.iter().enumerate() {
        for node in 0..tree.num_nodes() {
            if let Some(row) = node_row_idx[ti][node] {
                raw_pack[ti][node] = solution.dual_rows()[row];
            }
        }
    }

    // HiGHS row duals follow the active row-bound sign in minimization:
    // lower-bound rows are nonnegative, upper-bound rows nonpositive.
    // Cover rows are >= 1, so alpha is direct. Packing rows are <= 1, so beta = -row_dual.
    let alpha = raw_cover.into_iter().map(clean_dual).collect::<Vec<_>>();
    let beta = raw_pack
        .into_iter()
        .map(|row| row.into_iter().map(|v| clean_dual(-v)).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    Ok(PaperRmpLpSolution { alpha, beta })
}

fn price_missing_component(
    candidates: &[PackedCandidate],
    trees: &[Tree],
    n_reduced: usize,
    reverse_map: &[u32],
) -> Result<Option<PricedComponentReport>, Box<dyn std::error::Error>> {
    if trees.len() != 2 {
        return Ok(None);
    }

    let lp = solve_paper_rmp_lp(candidates, trees, n_reduced)?;
    let priced = run_rooted_paper_pricer(&trees[0], &trees[1], &lp.alpha, &lp.beta)?;
    let Some((score, reduced_labels)) = priced else {
        return Ok(None);
    };

    let best_in_pool_score = candidates
        .iter()
        .map(|cand| candidate_paper_score(cand, &lp.alpha, &lp.beta))
        .fold(f64::NEG_INFINITY, f64::max);

    let mut original_labels = reduced_labels
        .iter()
        .map(|&lbl| reverse_map[lbl as usize])
        .collect::<Vec<_>>();
    original_labels.sort_unstable();

    let already_in_pool = candidates
        .iter()
        .any(|cand| cand.reduced_labels == reduced_labels);

    if reduced_labels.len() < 2 {
        return Ok(None);
    }

    Ok(Some(PricedComponentReport {
        dual_sign_flipped: false,
        score,
        reduced_cost: score - 1.0,
        best_in_pool_score,
        best_in_pool_reduced_cost: best_in_pool_score - 1.0,
        beats_best_in_pool_by: score - best_in_pool_score,
        size: reduced_labels.len(),
        reduced_labels,
        original_labels,
        already_in_pool,
    }))
}

fn clean_dual(value: f64) -> f64 {
    if value.abs() <= 1.0e-9 { 0.0 } else { value }
}

fn candidate_paper_score(cand: &PackedCandidate, alpha: &[f64], beta: &[Vec<f64>]) -> f64 {
    let leaf_sum = cand
        .reduced_labels
        .iter()
        .map(|&lbl| alpha[lbl as usize])
        .sum::<f64>();
    let node_sum = cand
        .node_covers
        .iter()
        .enumerate()
        .map(|(ti, cover)| cover.ones().map(|node| beta[ti][node]).sum::<f64>())
        .sum::<f64>();
    leaf_sum - node_sum
}

fn run_rooted_paper_pricer(
    t1: &Tree,
    t2: &Tree,
    alpha: &[f64],
    beta: &[Vec<f64>],
) -> Result<Option<(f64, Vec<u32>)>, Box<dyn std::error::Error>> {
    const NEG_INF: f64 = -1.0e100;

    let n2 = t2.num_nodes();
    let idx = |u: NodeId, v: NodeId| -> usize { u as usize * n2 + v as usize };
    let mut v_score = vec![NEG_INF; t1.num_nodes() * n2];
    let mut v_choice = vec![VChoice::None; t1.num_nodes() * n2];
    let mut m_score = vec![0.0; t1.num_nodes() * n2];
    let mut m_choice = vec![MChoice::None; t1.num_nodes() * n2];
    let mut split_choice = vec![SplitChoice::None; t1.num_nodes() * n2];
    let post1 = t1.post_order_vec();
    let post2 = t2.post_order_vec();

    for &u in &post1 {
        for &v in &post2 {
            let pair = idx(u, v);
            match (t1.children(u), t2.children(v)) {
                (None, None) => {
                    if t1.label[u as usize] == t2.label[v as usize] {
                        let lbl = t1.label[u as usize];
                        let score = alpha[lbl as usize];
                        v_score[pair] = score;
                        v_choice[pair] = VChoice::LeafMatch(lbl);
                        m_score[pair] = score.max(0.0);
                        m_choice[pair] = if score > 0.0 {
                            MChoice::LeafMatch(lbl)
                        } else {
                            MChoice::None
                        };
                    } else {
                        v_score[pair] = NEG_INF;
                        v_choice[pair] = VChoice::None;
                        m_score[pair] = 0.0;
                        m_choice[pair] = MChoice::None;
                    }
                }
                (Some((ul, ur)), None) => {
                    let left = -beta[0][u as usize] + v_score[idx(ul, v)];
                    let right = -beta[0][u as usize] + v_score[idx(ur, v)];
                    if left >= right {
                        v_score[pair] = left;
                        v_choice[pair] = VChoice::SkipLeftU;
                    } else {
                        v_score[pair] = right;
                        v_choice[pair] = VChoice::SkipRightU;
                    }
                    let ml = m_score[idx(ul, v)];
                    let mr = m_score[idx(ur, v)];
                    if ml >= mr && ml > 0.0 {
                        m_score[pair] = ml;
                        m_choice[pair] = m_choice[idx(ul, v)];
                    } else if mr > 0.0 {
                        m_score[pair] = mr;
                        m_choice[pair] = m_choice[idx(ur, v)];
                    } else {
                        m_score[pair] = 0.0;
                        m_choice[pair] = MChoice::None;
                    }
                }
                (None, Some((vl, vr))) => {
                    let left = -beta[1][v as usize] + v_score[idx(u, vl)];
                    let right = -beta[1][v as usize] + v_score[idx(u, vr)];
                    if left >= right {
                        v_score[pair] = left;
                        v_choice[pair] = VChoice::SkipLeftV;
                    } else {
                        v_score[pair] = right;
                        v_choice[pair] = VChoice::SkipRightV;
                    }
                    let ml = m_score[idx(u, vl)];
                    let mr = m_score[idx(u, vr)];
                    if ml >= mr && ml > 0.0 {
                        m_score[pair] = ml;
                        m_choice[pair] = m_choice[idx(u, vl)];
                    } else if mr > 0.0 {
                        m_score[pair] = mr;
                        m_choice[pair] = m_choice[idx(u, vr)];
                    } else {
                        m_score[pair] = 0.0;
                        m_choice[pair] = MChoice::None;
                    }
                }
                (Some((ul, ur)), Some((vl, vr))) => {
                    let straight = v_score[idx(ul, vl)] + v_score[idx(ur, vr)];
                    let cross = v_score[idx(ul, vr)] + v_score[idx(ur, vl)];
                    let (best_split, split_pick) = if straight >= cross {
                        (straight, SplitChoice::Straight)
                    } else {
                        (cross, SplitChoice::Cross)
                    };
                    split_choice[pair] = if best_split > NEG_INF / 2.0 {
                        split_pick
                    } else {
                        SplitChoice::None
                    };

                    let rooted = if best_split > NEG_INF / 2.0 {
                        -beta[0][u as usize] - beta[1][v as usize] + best_split
                    } else {
                        NEG_INF
                    };

                    let mut best_v = rooted;
                    let mut best_v_choice = VChoice::UseRooted;
                    for (cand, branch_pick) in [
                        (
                            -beta[0][u as usize] + v_score[idx(ul, v)],
                            VChoice::SkipLeftU,
                        ),
                        (
                            -beta[0][u as usize] + v_score[idx(ur, v)],
                            VChoice::SkipRightU,
                        ),
                        (
                            -beta[1][v as usize] + v_score[idx(u, vl)],
                            VChoice::SkipLeftV,
                        ),
                        (
                            -beta[1][v as usize] + v_score[idx(u, vr)],
                            VChoice::SkipRightV,
                        ),
                    ] {
                        if cand > best_v {
                            best_v = cand;
                            best_v_choice = branch_pick;
                        }
                    }
                    v_score[pair] = best_v;
                    v_choice[pair] = if best_v > NEG_INF / 2.0 {
                        best_v_choice
                    } else {
                        VChoice::None
                    };

                    let mut best_m = 0.0;
                    let mut best_m_choice = MChoice::None;
                    for (cand, branch_pick) in [
                        (rooted, MChoice::UseRooted),
                        (m_score[idx(ul, v)], MChoice::SkipLeftU),
                        (m_score[idx(ur, v)], MChoice::SkipRightU),
                        (m_score[idx(u, vl)], MChoice::SkipLeftV),
                        (m_score[idx(u, vr)], MChoice::SkipRightV),
                    ] {
                        if cand > best_m {
                            best_m = cand;
                            best_m_choice = branch_pick;
                        }
                    }
                    m_score[pair] = best_m;
                    m_choice[pair] = best_m_choice;
                }
            }
        }
    }

    let root_score = m_score[idx(t1.root, t2.root)];
    if root_score <= 1e-9 {
        return Ok(None);
    }

    let mut reduced_labels = Vec::new();
    collect_m_labels(
        t1,
        t2,
        &m_choice,
        &v_choice,
        &split_choice,
        t1.root,
        t2.root,
        &mut reduced_labels,
    );
    reduced_labels.sort_unstable();
    reduced_labels.dedup();
    Ok(Some((root_score, reduced_labels)))
}

fn collect_m_labels(
    t1: &Tree,
    t2: &Tree,
    m_choice: &[MChoice],
    v_choice: &[VChoice],
    split_choice: &[SplitChoice],
    u: NodeId,
    v: NodeId,
    out: &mut Vec<u32>,
) {
    let n2 = t2.num_nodes();
    let pair = u as usize * n2 + v as usize;
    match m_choice[pair] {
        MChoice::None => {}
        MChoice::LeafMatch(lbl) => out.push(lbl),
        MChoice::UseRooted => collect_split_labels(t1, t2, v_choice, split_choice, u, v, out),
        MChoice::SkipLeftU => {
            let (ul, _) = t1.children(u).expect("skip-left-u requires internal u");
            collect_m_labels(t1, t2, m_choice, v_choice, split_choice, ul, v, out);
        }
        MChoice::SkipRightU => {
            let (_, ur) = t1.children(u).expect("skip-right-u requires internal u");
            collect_m_labels(t1, t2, m_choice, v_choice, split_choice, ur, v, out);
        }
        MChoice::SkipLeftV => {
            let (vl, _) = t2.children(v).expect("skip-left-v requires internal v");
            collect_m_labels(t1, t2, m_choice, v_choice, split_choice, u, vl, out);
        }
        MChoice::SkipRightV => {
            let (_, vr) = t2.children(v).expect("skip-right-v requires internal v");
            collect_m_labels(t1, t2, m_choice, v_choice, split_choice, u, vr, out);
        }
    }
}

fn collect_v_labels(
    t1: &Tree,
    t2: &Tree,
    v_choice: &[VChoice],
    split_choice: &[SplitChoice],
    u: NodeId,
    v: NodeId,
    out: &mut Vec<u32>,
) {
    let n2 = t2.num_nodes();
    let pair = u as usize * n2 + v as usize;
    match v_choice[pair] {
        VChoice::None => {}
        VChoice::LeafMatch(lbl) => out.push(lbl),
        VChoice::UseRooted => collect_split_labels(t1, t2, v_choice, split_choice, u, v, out),
        VChoice::SkipLeftU => {
            let (ul, _) = t1.children(u).expect("skip-left-u requires internal u");
            collect_v_labels(t1, t2, v_choice, split_choice, ul, v, out);
        }
        VChoice::SkipRightU => {
            let (_, ur) = t1.children(u).expect("skip-right-u requires internal u");
            collect_v_labels(t1, t2, v_choice, split_choice, ur, v, out);
        }
        VChoice::SkipLeftV => {
            let (vl, _) = t2.children(v).expect("skip-left-v requires internal v");
            collect_v_labels(t1, t2, v_choice, split_choice, u, vl, out);
        }
        VChoice::SkipRightV => {
            let (_, vr) = t2.children(v).expect("skip-right-v requires internal v");
            collect_v_labels(t1, t2, v_choice, split_choice, u, vr, out);
        }
    }
}

fn collect_split_labels(
    t1: &Tree,
    t2: &Tree,
    v_choice: &[VChoice],
    split_choice: &[SplitChoice],
    u: NodeId,
    v: NodeId,
    out: &mut Vec<u32>,
) {
    let n2 = t2.num_nodes();
    let pair = u as usize * n2 + v as usize;
    let (ul, ur) = t1.children(u).expect("split requires internal u");
    let (vl, vr) = t2.children(v).expect("split requires internal v");
    match split_choice[pair] {
        SplitChoice::Straight => {
            collect_v_labels(t1, t2, v_choice, split_choice, ul, vl, out);
            collect_v_labels(t1, t2, v_choice, split_choice, ur, vr, out);
        }
        SplitChoice::Cross => {
            collect_v_labels(t1, t2, v_choice, split_choice, ul, vr, out);
            collect_v_labels(t1, t2, v_choice, split_choice, ur, vl, out);
        }
        SplitChoice::None => {}
    }
}

fn validate_selection(
    chosen: &[usize],
    candidates: &[PackedCandidate],
    trees: &[Tree],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut leaf_cover = FixedBitSet::with_capacity(
        candidates
            .iter()
            .map(|c| c.leafset.len())
            .max()
            .unwrap_or(0),
    );
    let mut node_covers: Vec<FixedBitSet> = trees
        .iter()
        .map(|tree| FixedBitSet::with_capacity(tree.num_nodes()))
        .collect();

    for &idx in chosen {
        let cand = &candidates[idx];
        if bitsets_intersect(&leaf_cover, &cand.leafset) {
            return Err(format!(
                "selected candidates overlap in leaves: {:?}",
                cand.original_labels
            )
            .into());
        }
        leaf_cover.union_with(&cand.leafset);
        for (cover, cand_cover) in node_covers.iter_mut().zip(&cand.node_covers) {
            if bitsets_intersect(cover, cand_cover) {
                return Err(format!(
                    "selected candidates overlap in a tree: {:?}",
                    cand.original_labels
                )
                .into());
            }
            cover.union_with(cand_cover);
        }
    }
    Ok(())
}

fn build_expanded_solution(
    chosen: &[usize],
    candidates: &[PackedCandidate],
    reduced: &Instance,
    n_reduced: usize,
    kern: &kernelize::KernelizeResult,
    original_ref_tree: &Tree,
    original_num_leaves: u32,
) -> Vec<Tree> {
    let mut covered = FixedBitSet::with_capacity(n_reduced + 1);
    let mut reduced_components = Vec::with_capacity(chosen.len() + n_reduced);
    for &idx in chosen {
        let cand = &candidates[idx];
        covered.union_with(&cand.leafset);
        reduced_components.push(Tree::component_from_leafset(
            &cand.leafset,
            &reduced.trees[0],
            reduced.num_leaves,
        ));
    }
    for lbl in 1..=reduced.num_leaves {
        if !covered.contains(lbl as usize) {
            reduced_components.push(Tree::singleton(lbl, reduced.num_leaves));
        }
    }
    kernelize::expand_solution(
        reduced_components,
        kern,
        original_ref_tree,
        original_num_leaves,
    )
}

fn write_solution_file(path: &Path, components: &[Tree]) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    for tree in components {
        tree.cursor().write_newick(&mut out)?;
        writeln!(out)?;
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pace26io::binary_tree::IndexedBinTreeBuilder;
    use pace26io::pace::simplified::Instance as PaceInstance;
    use std::io::BufReader;

    fn instance_from_newick(newicks: &[&str]) -> Instance {
        let num_trees = newicks.len();
        let num_leaves: usize = newicks[0]
            .split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<usize>().unwrap())
            .max()
            .unwrap_or(0);

        let tree_lines: Vec<String> = newicks.iter().map(|nw| format!("{nw};")).collect();
        let input = format!("#p {} {}\n{}", num_trees, num_leaves, tree_lines.join("\n"));

        let mut builder = IndexedBinTreeBuilder::default();
        let reader = BufReader::new(input.as_bytes());
        let pace = PaceInstance::try_read(reader, &mut builder).unwrap();

        let n = pace.num_leaves as u32;
        let trees: Vec<Tree> = pace
            .trees
            .iter()
            .map(|t| Tree::from_cursor(t.top_down(), n))
            .collect();

        Instance::new(trees, n)
    }

    fn compatible_in_both(t1: &Tree, t2: &Tree, labels: &[u32]) -> bool {
        if labels.len() <= 2 {
            return true;
        }
        for i in 0..labels.len() {
            for j in (i + 1)..labels.len() {
                for k in (j + 1)..labels.len() {
                    let a = labels[i];
                    let b = labels[j];
                    let c = labels[k];
                    if triplet_topology(t1, a, b, c) != triplet_topology(t2, a, b, c) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn subset_score(labels: &[u32], trees: &[Tree], alpha: &[f64], beta: &[Vec<f64>]) -> f64 {
        let leaf_sum = labels.iter().map(|&lbl| alpha[lbl as usize]).sum::<f64>();
        let node_sum = trees
            .iter()
            .enumerate()
            .map(|(ti, tree)| {
                mark_component_nodes(tree, labels)
                    .ones()
                    .map(|node| beta[ti][node])
                    .sum::<f64>()
            })
            .sum::<f64>();
        leaf_sum - node_sum
    }

    fn brute_force_best_priced_set(
        t1: &Tree,
        t2: &Tree,
        alpha: &[f64],
        beta: &[Vec<f64>],
    ) -> Option<(f64, Vec<u32>)> {
        let n = t1.num_leaves as usize;
        let trees = [t1.clone(), t2.clone()];
        let mut best_score = f64::NEG_INFINITY;
        let mut best_labels = Vec::new();

        for mask in 1usize..(1usize << n) {
            let mut labels = Vec::new();
            for idx in 0..n {
                if (mask >> idx) & 1 == 1 {
                    labels.push((idx + 1) as u32);
                }
            }
            if !compatible_in_both(t1, t2, &labels) {
                continue;
            }
            let score = subset_score(&labels, &trees, alpha, beta);
            if score > best_score + 1e-9
                || ((score - best_score).abs() <= 1e-9 && labels < best_labels)
            {
                best_score = score;
                best_labels = labels;
            }
        }

        if best_score > 1e-9 {
            Some((best_score, best_labels))
        } else {
            None
        }
    }

    fn make_weights(inst: &Instance, case_idx: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = inst.num_leaves as usize;
        let mut alpha = vec![0.0; n + 1];
        for lbl in 1..=n {
            alpha[lbl] = match case_idx {
                0 => 0.35 + 0.55 * lbl as f64,
                1 => 0.2 + (((lbl * 37) % 17) as f64) / 4.0,
                _ => {
                    let parity_bonus = if lbl % 2 == 0 { 1.7 } else { 0.45 };
                    parity_bonus + 0.18 * lbl as f64
                }
            };
        }

        let beta = inst
            .trees
            .iter()
            .enumerate()
            .map(|(ti, tree)| {
                (0..tree.num_nodes())
                    .map(|node| {
                        let node_id = node as u32;
                        if tree.is_leaf(node_id) {
                            0.0
                        } else {
                            match case_idx {
                                0 => {
                                    0.08 * (tree.depth[node] as f64 + 1.0)
                                        + 0.003 * node as f64
                                        + 0.01 * ti as f64
                                }
                                1 => {
                                    let root_bonus = if node_id == tree.root { 0.55 } else { 0.0 };
                                    root_bonus
                                        + 0.04 * tree.subtree_size[node] as f64
                                        + 0.01 * tree.depth[node] as f64
                                }
                                _ => {
                                    0.03 * (tree.subtree_size[node] as f64 - 1.0)
                                        + 0.06 * (tree.depth[node] as f64 + 1.0)
                                        + 0.005 * ti as f64
                                }
                            }
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        (alpha, beta)
    }

    #[test]
    fn rooted_paper_pricer_matches_bruteforce_under_current_master_scoring() {
        let instances = [
            ["((1,2),(3,4))", "((1,3),(2,4))"],
            ["(((1,2),3),(4,5))", "((1,(2,4)),(3,5))"],
            ["(((5,6),(3,4)),(1,2))", "(((((4,2),1),5),3),6)"],
            ["((((1,2),3),4),(5,6))", "(((1,5),(2,6)),(3,4))"],
        ];

        for newicks in instances {
            let inst = instance_from_newick(&newicks);
            for case_idx in 0..3 {
                let (alpha, beta) = make_weights(&inst, case_idx);
                let brute =
                    brute_force_best_priced_set(&inst.trees[0], &inst.trees[1], &alpha, &beta);
                let priced =
                    run_rooted_paper_pricer(&inst.trees[0], &inst.trees[1], &alpha, &beta).unwrap();

                match (brute, priced) {
                    (None, None) => {}
                    (Some((best_score, _)), Some((priced_score, labels))) => {
                        let realized = subset_score(&labels, &inst.trees, &alpha, &beta);
                        assert!(
                            (priced_score - best_score).abs() <= 1e-6,
                            "DP score mismatch for {:?}, case {}: brute={}, dp={}",
                            newicks,
                            case_idx,
                            best_score,
                            priced_score
                        );
                        assert!(
                            (realized - best_score).abs() <= 1e-6,
                            "DP labels realize wrong score for {:?}, case {}: brute={}, labels={:?}, realized={}",
                            newicks,
                            case_idx,
                            best_score,
                            labels,
                            realized
                        );
                        assert!(
                            compatible_in_both(&inst.trees[0], &inst.trees[1], &labels),
                            "DP labels incompatible for {:?}, case {}: {:?}",
                            newicks,
                            case_idx,
                            labels
                        );
                    }
                    (None, Some((priced_score, labels))) => panic!(
                        "DP returned positive score {} for {:?}, case {} with labels {:?}, but brute force found none",
                        priced_score, newicks, case_idx, labels
                    ),
                    (Some((best_score, labels)), None) => panic!(
                        "DP returned none for {:?}, case {} but brute force found score {} with labels {:?}",
                        newicks, case_idx, best_score, labels
                    ),
                }
            }
        }
    }
}
