//! Scan a reduced instance for small feasible agreement components.

use std::collections::HashSet;
use std::time::Instant;

use klados_core::kernelize::{self, KernelizeConfig};
use klados_core::lower_bound::pairwise_refine_ub;
use klados_core::{Instance, Tree};
use serde::Serialize;

type Label3 = [u16; 3];
type Label4 = [u16; 4];
type Label5 = [u16; 5];

#[derive(Serialize)]
struct ComponentScanReport {
    input_trees: usize,
    input_leaves: u32,
    kernelized: bool,
    reduced_leaves: u32,
    kernel_removed: u32,
    kernel_subtree_removed: usize,
    kernel_chain_removed: usize,
    kernel_chain32_removed: usize,
    max_size: usize,
    candidate_cap: usize,
    sample_limit: usize,
    common_cherries: usize,
    common_cherry_samples_original_labels: Vec<[u32; 2]>,
    feasible_triples: usize,
    feasible_triples_with_common_cherry_pair: usize,
    feasible_triple_samples_original_labels: Vec<[u32; 3]>,
    supported_pairs: usize,
    max_pair_support: usize,
    mean_pair_support_nonzero: f64,
    triple_scan_ms: f64,
    pairwise_refine_partition: PartitionSummary,
    size4: Option<SizeStageReport4>,
    size5: Option<SizeStageReport5>,
}

#[derive(Serialize)]
struct SizeStageReport4 {
    generation: &'static str,
    raw_unique_candidates: usize,
    feasible_candidates: usize,
    truncated: bool,
    generation_ms: f64,
    feasible_samples_original_labels: Vec<[u32; 4]>,
}

#[derive(Serialize)]
struct SizeStageReport5 {
    generation: &'static str,
    raw_unique_candidates: usize,
    feasible_candidates: usize,
    truncated: bool,
    skipped_due_to_prior_truncation: bool,
    generation_ms: f64,
    feasible_samples_original_labels: Vec<[u32; 5]>,
}

#[derive(Serialize)]
struct PartitionSummary {
    components: usize,
    singleton_components: usize,
    nontrivial_components: usize,
    max_component_size: usize,
    total_savings: usize,
    size_histogram: Vec<SizeCount>,
    largest_components_original_labels: Vec<Vec<u32>>,
}

#[derive(Serialize)]
struct SizeCount {
    size: usize,
    count: usize,
}

pub fn run(
    instance: &Instance,
    kernelize_first: bool,
    max_size: usize,
    sample_limit: usize,
    candidate_cap: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if max_size < 3 || max_size > 5 {
        return Err(format!(
            "component-scan currently supports 3 <= max-size <= 5, got {max_size}"
        )
        .into());
    }

    let (
        scan_instance,
        reverse_map,
        kernel_removed,
        subtree_removed,
        chain_removed,
        chain32_removed,
    ) = if kernelize_first {
        let kern = kernelize::kernelize_best(instance, &KernelizeConfig::default());
        let stats = &kern.stats;
        (
            kern.instance,
            kern.reverse_map,
            stats.original_leaves.saturating_sub(stats.reduced_leaves),
            *stats.rule_counts.get("cherry").unwrap_or(&0),
            *stats.rule_counts.get("chain").unwrap_or(&0),
            *stats.rule_counts.get("chain-3-2").unwrap_or(&0)
                + *stats.rule_counts.get("triple").unwrap_or(&0),
        )
    } else {
        (
            instance.clone(),
            (0..=instance.num_leaves).collect(),
            0,
            0,
            0,
            0,
        )
    };

    let n = scan_instance.num_leaves as usize;
    if n > u16::MAX as usize {
        return Err(format!("component-scan requires at most {}, got {}", u16::MAX, n).into());
    }

    let stride = n + 1;
    let t0 = Instant::now();

    let mut common_cherry = vec![false; stride * stride];
    let mut common_cherries = 0usize;
    let mut common_cherry_samples = Vec::new();
    for a in 1..=n {
        for b in (a + 1)..=n {
            if scan_instance
                .trees
                .iter()
                .all(|tree| is_cherry(tree, a as u32, b as u32))
            {
                common_cherry[pair_index(stride, a as u16, b as u16)] = true;
                common_cherries += 1;
                push_sample_pair(
                    &mut common_cherry_samples,
                    a as u32,
                    b as u32,
                    sample_limit,
                    &reverse_map,
                );
            }
        }
    }

    let mut pair_to_thirds: Vec<Vec<u16>> = vec![Vec::new(); stride * stride];
    let mut feasible_triples = 0usize;
    let mut feasible_triples_with_common_cherry_pair = 0usize;
    let mut feasible_triple_samples = Vec::new();

    for a in 1..=n {
        for b in (a + 1)..=n {
            for c in (b + 1)..=n {
                let topo0 = triplet_topology(&scan_instance.trees[0], a as u32, b as u32, c as u32);
                if scan_instance.trees[1..]
                    .iter()
                    .all(|tree| triplet_topology(tree, a as u32, b as u32, c as u32) == topo0)
                {
                    feasible_triples += 1;
                    let au = a as u16;
                    let bu = b as u16;
                    let cu = c as u16;
                    pair_to_thirds[pair_index(stride, au, bu)].push(cu);
                    pair_to_thirds[pair_index(stride, au, cu)].push(bu);
                    pair_to_thirds[pair_index(stride, bu, cu)].push(au);
                    if common_cherry[pair_index(stride, au, bu)]
                        || common_cherry[pair_index(stride, au, cu)]
                        || common_cherry[pair_index(stride, bu, cu)]
                    {
                        feasible_triples_with_common_cherry_pair += 1;
                    }
                    push_sample_triple(
                        &mut feasible_triple_samples,
                        [a as u32, b as u32, c as u32],
                        sample_limit,
                        &reverse_map,
                    );
                }
            }
        }
    }

    let mut supported_pairs = 0usize;
    let mut max_pair_support = 0usize;
    let mut total_pair_support = 0usize;
    for a in 1..=n {
        for b in (a + 1)..=n {
            let list = &mut pair_to_thirds[pair_index(stride, a as u16, b as u16)];
            if !list.is_empty() {
                supported_pairs += 1;
                max_pair_support = max_pair_support.max(list.len());
                total_pair_support += list.len();
            }
            list.sort_unstable();
        }
    }
    let triple_scan_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mean_pair_support_nonzero = if supported_pairs == 0 {
        0.0
    } else {
        total_pair_support as f64 / supported_pairs as f64
    };

    let (pr_components, pr_partition) = pairwise_refine_ub(&scan_instance.trees, n);
    let pairwise_refine_partition =
        summarize_partition(&pr_partition, pr_components, sample_limit, &reverse_map);

    let (size4, feasible_size4) = if max_size >= 4 {
        generate_size4(
            &pair_to_thirds,
            stride,
            candidate_cap,
            sample_limit,
            &reverse_map,
        )
    } else {
        (None, Vec::new())
    };

    let size5 = if max_size >= 5 {
        generate_size5(
            &pair_to_thirds,
            stride,
            candidate_cap,
            sample_limit,
            &reverse_map,
            &feasible_size4,
            size4.as_ref().is_some_and(|report| report.truncated),
        )
    } else {
        None
    };

    let report = ComponentScanReport {
        input_trees: instance.num_trees(),
        input_leaves: instance.num_leaves,
        kernelized: kernelize_first,
        reduced_leaves: scan_instance.num_leaves,
        kernel_removed,
        kernel_subtree_removed: subtree_removed,
        kernel_chain_removed: chain_removed,
        kernel_chain32_removed: chain32_removed,
        max_size,
        candidate_cap,
        sample_limit,
        common_cherries,
        common_cherry_samples_original_labels: common_cherry_samples,
        feasible_triples,
        feasible_triples_with_common_cherry_pair,
        feasible_triple_samples_original_labels: feasible_triple_samples,
        supported_pairs,
        max_pair_support,
        mean_pair_support_nonzero,
        triple_scan_ms,
        pairwise_refine_partition,
        size4,
        size5,
    };

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn summarize_partition(
    partition: &[usize],
    components: usize,
    sample_limit: usize,
    reverse_map: &[u32],
) -> PartitionSummary {
    let max_comp = partition.iter().copied().max().unwrap_or(0);
    let mut groups: Vec<Vec<u32>> = vec![Vec::new(); max_comp + 1];
    for (idx, &comp) in partition.iter().enumerate() {
        let reduced_label = (idx + 1) as u32;
        let original_label = reverse_map
            .get(reduced_label as usize)
            .copied()
            .unwrap_or(reduced_label);
        groups[comp].push(original_label);
    }

    let mut sizes = Vec::new();
    for group in groups {
        if !group.is_empty() {
            sizes.push(group);
        }
    }

    let mut histogram_map = std::collections::BTreeMap::<usize, usize>::new();
    let mut singleton_components = 0usize;
    let mut nontrivial_components = 0usize;
    let mut max_component_size = 0usize;
    let mut total_savings = 0usize;
    for group in &sizes {
        let size = group.len();
        *histogram_map.entry(size).or_default() += 1;
        max_component_size = max_component_size.max(size);
        if size == 1 {
            singleton_components += 1;
        } else {
            nontrivial_components += 1;
            total_savings += size - 1;
        }
    }

    let mut largest = sizes;
    largest.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    largest.truncate(sample_limit);

    PartitionSummary {
        components,
        singleton_components,
        nontrivial_components,
        max_component_size,
        total_savings,
        size_histogram: histogram_map
            .into_iter()
            .map(|(size, count)| SizeCount { size, count })
            .collect(),
        largest_components_original_labels: largest,
    }
}

fn generate_size4(
    pair_to_thirds: &[Vec<u16>],
    stride: usize,
    candidate_cap: usize,
    sample_limit: usize,
    reverse_map: &[u32],
) -> (Option<SizeStageReport4>, Vec<Label4>) {
    let start = Instant::now();
    let n = stride - 1;
    let mut raw_seen: HashSet<Label4> = HashSet::new();
    let mut feasible = Vec::new();
    let mut samples = Vec::new();
    let mut truncated = false;

    'outer: for a in 1..=n {
        for b in (a + 1)..=n {
            let list = &pair_to_thirds[pair_index(stride, a as u16, b as u16)];
            if list.len() < 2 {
                continue;
            }
            for i in 0..list.len() {
                for j in (i + 1)..list.len() {
                    let key = sort4(a as u16, b as u16, list[i], list[j]);
                    if raw_seen.insert(key) {
                        if raw_seen.len() > candidate_cap {
                            truncated = true;
                            break 'outer;
                        }
                        if feasible4(pair_to_thirds, stride, &key) {
                            feasible.push(key);
                            if samples.len() < sample_limit {
                                samples.push(map4(key, reverse_map));
                            }
                        }
                    }
                }
            }
        }
    }

    let report = SizeStageReport4 {
        generation: "union_of_two_feasible_triples_sharing_a_pair",
        raw_unique_candidates: raw_seen.len(),
        feasible_candidates: feasible.len(),
        truncated,
        generation_ms: start.elapsed().as_secs_f64() * 1000.0,
        feasible_samples_original_labels: samples,
    };

    (Some(report), feasible)
}

fn generate_size5(
    pair_to_thirds: &[Vec<u16>],
    stride: usize,
    candidate_cap: usize,
    sample_limit: usize,
    reverse_map: &[u32],
    feasible_size4: &[Label4],
    prior_truncation: bool,
) -> Option<SizeStageReport5> {
    if prior_truncation {
        return Some(SizeStageReport5 {
            generation: "extend_feasible_size4_by_locally_supported_leaf",
            raw_unique_candidates: 0,
            feasible_candidates: 0,
            truncated: false,
            skipped_due_to_prior_truncation: true,
            generation_ms: 0.0,
            feasible_samples_original_labels: Vec::new(),
        });
    }

    let start = Instant::now();
    let n = stride - 1;
    let mut raw_seen: HashSet<Label5> = HashSet::new();
    let mut feasible_count = 0usize;
    let mut samples = Vec::new();
    let mut truncated = false;
    let mut mark = vec![false; n + 1];

    'outer: for quad in feasible_size4 {
        mark.fill(false);
        for &leaf in quad {
            mark[leaf as usize] = true;
        }
        let pairs = [
            (quad[0], quad[1]),
            (quad[0], quad[2]),
            (quad[0], quad[3]),
            (quad[1], quad[2]),
            (quad[1], quad[3]),
            (quad[2], quad[3]),
        ];
        for &(a, b) in &pairs {
            let list = &pair_to_thirds[pair_index(stride, a, b)];
            for &x in list {
                if mark[x as usize] {
                    continue;
                }
                let key = sort5([quad[0], quad[1], quad[2], quad[3], x]);
                if raw_seen.insert(key) {
                    if raw_seen.len() > candidate_cap {
                        truncated = true;
                        break 'outer;
                    }
                    if feasible5(pair_to_thirds, stride, &key) {
                        feasible_count += 1;
                        if samples.len() < sample_limit {
                            samples.push(map5(key, reverse_map));
                        }
                    }
                }
            }
        }
    }

    Some(SizeStageReport5 {
        generation: "extend_feasible_size4_by_locally_supported_leaf",
        raw_unique_candidates: raw_seen.len(),
        feasible_candidates: feasible_count,
        truncated,
        skipped_due_to_prior_truncation: false,
        generation_ms: start.elapsed().as_secs_f64() * 1000.0,
        feasible_samples_original_labels: samples,
    })
}

#[inline]
fn pair_index(stride: usize, a: u16, b: u16) -> usize {
    debug_assert!(a < b);
    a as usize * stride + b as usize
}

#[inline]
fn is_cherry(tree: &Tree, a: u32, b: u32) -> bool {
    let na = tree.node_by_label(a);
    let nb = tree.node_by_label(b);
    let pa = tree.parent[na as usize];
    let pb = tree.parent[nb as usize];
    pa != klados_core::NONE && pa == pb
}

#[inline]
fn triplet_topology(tree: &Tree, x: u32, y: u32, z: u32) -> u8 {
    let nx = tree.node_by_label(x);
    let ny = tree.node_by_label(y);
    let nz = tree.node_by_label(z);
    let nca_xy = tree.nearest_common_ancestor(nx, ny);
    let nca_xz = tree.nearest_common_ancestor(nx, nz);
    let nca_yz = tree.nearest_common_ancestor(ny, nz);
    let d_xy = tree.depth[nca_xy as usize];
    let d_xz = tree.depth[nca_xz as usize];
    let d_yz = tree.depth[nca_yz as usize];
    if d_xy > d_xz && d_xy > d_yz {
        0
    } else if d_xz > d_xy && d_xz > d_yz {
        1
    } else {
        2
    }
}

#[inline]
fn contains_triple(pair_to_thirds: &[Vec<u16>], stride: usize, a: u16, b: u16, c: u16) -> bool {
    let [x, y, z] = sort3(a, b, c);
    pair_to_thirds[pair_index(stride, x, y)]
        .binary_search(&z)
        .is_ok()
}

#[inline]
fn feasible4(pair_to_thirds: &[Vec<u16>], stride: usize, quad: &Label4) -> bool {
    contains_triple(pair_to_thirds, stride, quad[0], quad[1], quad[2])
        && contains_triple(pair_to_thirds, stride, quad[0], quad[1], quad[3])
        && contains_triple(pair_to_thirds, stride, quad[0], quad[2], quad[3])
        && contains_triple(pair_to_thirds, stride, quad[1], quad[2], quad[3])
}

fn feasible5(pair_to_thirds: &[Vec<u16>], stride: usize, leaves: &Label5) -> bool {
    for i in 0..5 {
        for j in (i + 1)..5 {
            for k in (j + 1)..5 {
                if !contains_triple(pair_to_thirds, stride, leaves[i], leaves[j], leaves[k]) {
                    return false;
                }
            }
        }
    }
    true
}

#[inline]
fn sort3(a: u16, b: u16, c: u16) -> Label3 {
    let mut xs = [a, b, c];
    xs.sort_unstable();
    xs
}

#[inline]
fn sort4(a: u16, b: u16, c: u16, d: u16) -> Label4 {
    let mut xs = [a, b, c, d];
    xs.sort_unstable();
    xs
}

#[inline]
fn sort5(mut xs: Label5) -> Label5 {
    xs.sort_unstable();
    xs
}

#[inline]
fn map_label(label: u16, reverse_map: &[u32]) -> u32 {
    reverse_map
        .get(label as usize)
        .copied()
        .unwrap_or(label as u32)
}

fn map4(labels: Label4, reverse_map: &[u32]) -> [u32; 4] {
    [
        map_label(labels[0], reverse_map),
        map_label(labels[1], reverse_map),
        map_label(labels[2], reverse_map),
        map_label(labels[3], reverse_map),
    ]
}

fn map5(labels: Label5, reverse_map: &[u32]) -> [u32; 5] {
    [
        map_label(labels[0], reverse_map),
        map_label(labels[1], reverse_map),
        map_label(labels[2], reverse_map),
        map_label(labels[3], reverse_map),
        map_label(labels[4], reverse_map),
    ]
}

fn push_sample_pair(
    samples: &mut Vec<[u32; 2]>,
    a: u32,
    b: u32,
    limit: usize,
    reverse_map: &[u32],
) {
    if samples.len() < limit {
        samples.push([
            reverse_map.get(a as usize).copied().unwrap_or(a),
            reverse_map.get(b as usize).copied().unwrap_or(b),
        ]);
    }
}

fn push_sample_triple(
    samples: &mut Vec<[u32; 3]>,
    labels: [u32; 3],
    limit: usize,
    reverse_map: &[u32],
) {
    if samples.len() < limit {
        samples.push([
            reverse_map
                .get(labels[0] as usize)
                .copied()
                .unwrap_or(labels[0]),
            reverse_map
                .get(labels[1] as usize)
                .copied()
                .unwrap_or(labels[1]),
            reverse_map
                .get(labels[2] as usize)
                .copied()
                .unwrap_or(labels[2]),
        ]);
    }
}
