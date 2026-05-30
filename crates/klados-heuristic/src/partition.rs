use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::time::Instant;

use fixedbitset::FixedBitSet;
use highs::{Col, ColProblem, HighsModelStatus, Model, Row, RowProblem, Sense};
use klados_core::kernelize::{self, KernelizeConfig};
use klados_core::lower_bound::greedy_multi_tree_partition;
use klados_core::{Instance, SolverStats, Tree};

use crate::HeuristicSolver;

pub struct PartitionHeuristicSolver {
    terminate_requested: Arc<AtomicBool>,
    stats: SolverStats,
}

pub(crate) struct PartitionBlock {
    pub(crate) weight: usize,
    pub(crate) labels: Vec<u32>,
}

pub(crate) struct PaperLpSolution {
    pub(crate) alpha: Vec<f64>,
    pub(crate) beta: Vec<Vec<f64>>,
    pub(crate) candidate_columns: Vec<f64>,
}

struct GlobalPricingResult {
    generated: usize,
    lp_basis: Vec<usize>,
}

struct GlobalPricingWorkspace {
    reduced_candidates: Vec<PartitionBlock>,
    reduced_to_original: Vec<usize>,
    reduced_seen: HashSet<Vec<u32>>,
    synced_original_candidates: usize,
    primal_model: Model,
    primal_leaf_rows: Vec<Option<Row>>,
    primal_node_rows: Vec<Vec<Option<Row>>>,
    dual_model: Model,
    dual_penalty_vars: Vec<Vec<Option<Col>>>,
    num_singletons: usize,
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

impl PartitionHeuristicSolver {
    pub fn greedy_union_add_one() -> Self {
        Self::new()
    }

    fn new() -> Self {
        Self {
            terminate_requested: Arc::new(AtomicBool::new(false)),
            stats: SolverStats::default(),
        }
    }

    fn mode_name(&self) -> &'static str {
        "greedy-partition-union-addone"
    }

    fn solve_partitions(
        &self,
        trees: &[Tree],
        num_leaves: usize,
        testing_mode: bool,
    ) -> PartitionOutcome {
        const INITIAL_UNION_SEED_CAP_NORMAL: u64 = 15;
        const INITIAL_UNION_SEED_CAP_TEST: u64 = 80;
        const PARTITION_SOFT_BUDGET_NORMAL: Duration = Duration::from_secs(2);
        const PARTITION_SOFT_BUDGET_TEST: Duration = Duration::from_secs(25);

        let union_seed_cap = if testing_mode {
            INITIAL_UNION_SEED_CAP_TEST
        } else {
            INITIAL_UNION_SEED_CAP_NORMAL
        };
        let partition_soft_budget = if testing_mode {
            PARTITION_SOFT_BUDGET_TEST
        } else {
            PARTITION_SOFT_BUDGET_NORMAL
        };

        let total_runs_cap = trees.len() * (union_seed_cap as usize + 1);
        let mut best = (num_leaves.max(1), None);
        let mut partitions = Vec::with_capacity(total_runs_cap);
        let mut runs = 0usize;
        let t0 = Instant::now();
        for ref_idx in 0..trees.len() {
            for seed in 0..=union_seed_cap {
                let (components, partition) = greedy_multi_tree_partition(trees, ref_idx, seed);
                if components < best.0 {
                    best = (components, Some((ref_idx, seed)));
                }
                partitions.push(partition);
                runs += 1;
                if t0.elapsed() >= partition_soft_budget
                    || self.terminate_requested.load(Ordering::Relaxed)
                {
                    break;
                }
            }
            if t0.elapsed() >= partition_soft_budget
                || self.terminate_requested.load(Ordering::Relaxed)
            {
                break;
            }
        }
        PartitionOutcome {
            best_components: best.0,
            best_meta: best.1,
            partitions,
            total_runs: runs,
        }
    }

    fn extend_with_partition_batch(
        &self,
        candidates: &mut Vec<PartitionBlock>,
        seen_leafsets: &mut HashSet<Vec<u32>>,
        kern: &kernelize::KernelizeResult,
        original: &Instance,
        rep_to_all: &HashMap<u32, Vec<u32>, fxhash::FxBuildHasher>,
        start_seed: u64,
        batch_size: u64,
    ) -> usize {
        let mut generated = 0usize;
        for ref_idx in 0..kern.instance.trees.len() {
            for seed in start_seed..(start_seed + batch_size) {
                if self.terminate_requested.load(Ordering::Relaxed) {
                    return generated;
                }
                let (_components, partition) =
                    greedy_multi_tree_partition(&kern.instance.trees, ref_idx, seed);
                let groups = partition_groups(&partition);
                for reduced_labels in groups {
                    let expanded_leafset = expand_reduced_leafset(
                        &reduced_labels,
                        &kern.reverse_map,
                        rep_to_all,
                        original.num_leaves,
                    );
                    let expanded_labels: Vec<u32> =
                        expanded_leafset.ones().map(|label| label as u32).collect();
                    if expanded_labels.len() < 2 {
                        continue;
                    }
                    if !seen_leafsets.insert(expanded_labels.clone()) {
                        continue;
                    }
                    if !is_set_compatible_all(&original.trees, &expanded_labels) {
                        continue;
                    }
                    candidates.push(PartitionBlock {
                        weight: expanded_labels.len() - 1,
                        labels: expanded_labels,
                    });
                    generated += 1;
                }
            }
        }
        generated
    }

    fn build_repaired_components(
        &self,
        partitions: &[Vec<usize>],
        kern: &kernelize::KernelizeResult,
        original: &Instance,
    ) -> Result<(Vec<Tree>, RepairStats), Box<dyn std::error::Error>> {
        let rep_to_all = kernelize::build_rep_to_all(&kern.collapses_original);
        let (pool_soft_limit, pool_hard_limit) =
            candidate_pool_limits(original.num_leaves, &original.trees);
        let testing_mode = heuristic_testing_mode();
        let defer_incumbent = !testing_mode;

        let mut candidates = Vec::new();
        let mut raw_nontrivial_blocks = 0usize;
        let mut seen_leafsets: HashSet<Vec<u32>> = HashSet::new();
        for partition in partitions {
            let groups = partition_groups(partition);
            for reduced_labels in groups {
                let expanded_leafset = expand_reduced_leafset(
                    &reduced_labels,
                    &kern.reverse_map,
                    &rep_to_all,
                    original.num_leaves,
                );
                let expanded_labels: Vec<u32> =
                    expanded_leafset.ones().map(|label| label as u32).collect();
                if expanded_labels.len() < 2 {
                    continue;
                }
                raw_nontrivial_blocks += 1;
                if !seen_leafsets.insert(expanded_labels.clone()) {
                    continue;
                }
                if !is_set_compatible_all(&original.trees, &expanded_labels) {
                    continue;
                }
                candidates.push(PartitionBlock {
                    weight: expanded_labels.len() - 1,
                    labels: expanded_labels,
                });
                if candidates.len() > pool_hard_limit {
                    let before = candidates.len();
                    prune_candidate_pool(&mut candidates, None, pool_soft_limit);
                    eprintln!(
                        "[heur:{}] pruned initial candidate pool {} -> {} blocks",
                        self.mode_name(),
                        before,
                        candidates.len(),
                    );
                }
            }
        }

        let mut forbidden_labels = HashSet::<u32>::default();
        for &deleted in &kern.stats.deleted_labels {
            forbidden_labels.insert(deleted);
            if let Some(all_labels) = rep_to_all.get(&deleted) {
                for &label in all_labels {
                    forbidden_labels.insert(label);
                }
            }
        }
        let reduced = &kern.instance;
        let original_to_reduced =
            build_original_to_reduced_map(kern, &rep_to_all, original.num_leaves);
        let reduced_forbidden = build_reduced_forbidden_labels(
            &forbidden_labels,
            &original_to_reduced,
            reduced.num_leaves,
        );

        let mut selected = if defer_incumbent {
            greedy_nodepack_selection(&candidates, &original.trees, None)
        } else {
            solve_nodepack_selection(&candidates, &original.trees)?
        };
        if candidates.len() > pool_soft_limit {
            let before = candidates.len();
            selected = prune_candidate_pool(&mut candidates, Some(&selected), pool_soft_limit)
                .unwrap_or(selected);
            eprintln!(
                "[heur:{}] pruned selected candidate pool {} -> {} blocks",
                self.mode_name(),
                before,
                candidates.len(),
            );
        }
        let mut local_search_merges = 0usize;
        let (mut components, mut selected_savings, mut incumbent_component_count) =
            if defer_incumbent {
                let (mut components, _) =
                    materialize_components(&selected, &candidates, &rep_to_all, kern, original);
                if let Some((improved_components, merges)) =
                    self.improve_components_locally(&components, original, &forbidden_labels)
                {
                    components = improved_components;
                    local_search_merges += merges;
                }
                let savings = original.num_leaves as usize - components.len();
                eprintln!(
                    "[heur:{}] initial greedy incumbent: selected={} savings={} components={} local_merges={}",
                    self.mode_name(),
                    selected.len(),
                    savings,
                    components.len(),
                    local_search_merges,
                );
                let count = components.len();
                (components, savings, count)
            } else {
                let (mut components, _) =
                    materialize_components(&selected, &candidates, &rep_to_all, kern, original);
                if let Some((improved_components, merges)) =
                    self.improve_components_locally(&components, original, &forbidden_labels)
                {
                    components = improved_components;
                    local_search_merges += merges;
                }
                let savings = original.num_leaves as usize - components.len();
                eprintln!(
                    "[heur:{}] initial pack selected={} savings={} components={} local_merges={}",
                    self.mode_name(),
                    selected.len(),
                    savings,
                    components.len(),
                    local_search_merges,
                );
                let count = components.len();
                (components, savings, count)
            };
        let mut best_selected_nontrivial_blocks =
            count_nontrivial_label_sets(&component_label_sets(&components));

        const MAX_ROUNDS: usize = 32;
        const STALL_PATIENCE: usize = 6;
        const IMPROVEMENT_SOFT_BUDGET: Duration = Duration::from_secs(240);
        const GLOBAL_PRICING_BURST_NORMAL: usize = 8;
        const GLOBAL_PRICING_BURST_TEST: usize = 4;
        const GLOBAL_PRICING_TIME_BUDGET_NORMAL: Duration = Duration::from_millis(1200);
        const LOW_YIELD_DIVERSIFY_TRIGGER: usize = 6;
        const DIVERSIFY_SEED_BATCH: u64 = 8;
        const INCUMBENT_REFRESH_ADDONE_ROUNDS: usize = 2;
        let mut total_add_one = 0usize;
        let mut total_priced = 0usize;
        let mut stalled_rounds = 0usize;
        let mut low_yield_rounds = 0usize;
        let t_improve = Instant::now();
        let mut next_diversify_seed = if testing_mode { 81u64 } else { 16u64 };
        let mut round = 1usize;
        let mut addone_rounds_since_refresh = 0usize;
        let mut lp_basis_cache = selected.clone();
        let mut force_lp_refresh = true;
        let mut search_exhausted = false;
        let mut exhausted_rounds = 0usize;
        let mut global_pricing_workspace = None::<GlobalPricingWorkspace>;

        eprintln!(
            "[heur:{}] adaptive improvement loop (leaves={}, reduced_leaves={}, rounds={}, patience={}, test_mode={})",
            self.mode_name(),
            original.num_leaves,
            kern.instance.num_leaves,
            if testing_mode {
                MAX_ROUNDS.to_string()
            } else {
                "unbounded".to_string()
            },
            if testing_mode {
                STALL_PATIENCE.to_string()
            } else {
                "disabled".to_string()
            },
            testing_mode,
        );

        loop {
            if self.terminate_requested.load(Ordering::Relaxed) {
                break;
            }
            if testing_mode && round > MAX_ROUNDS {
                eprintln!(
                    "[heur:{}] stopping adaptive loop on test round cap {}",
                    self.mode_name(),
                    MAX_ROUNDS,
                );
                break;
            }
            if testing_mode && t_improve.elapsed() >= IMPROVEMENT_SOFT_BUDGET {
                eprintln!(
                    "[heur:{}] stopping adaptive loop on soft budget after {:.1}ms",
                    self.mode_name(),
                    t_improve.elapsed().as_secs_f64() * 1000.0,
                );
                break;
            }

            let mut working_basis = if lp_basis_cache.is_empty() {
                selected.clone()
            } else {
                lp_basis_cache.clone()
            };

            let mut generated_global_priced = 0usize;
            let mut generated_local_priced = 0usize;
            let mut generated_add_one = 0usize;
            if !search_exhausted {
                let global_pricing = self.add_global_priced_blocks(
                    &mut candidates,
                    &mut seen_leafsets,
                    reduced,
                    &original_to_reduced,
                    &reduced_forbidden,
                    &mut global_pricing_workspace,
                    &kern.reverse_map,
                    &rep_to_all,
                    original,
                    &forbidden_labels,
                    if testing_mode {
                        GLOBAL_PRICING_BURST_TEST
                    } else {
                        GLOBAL_PRICING_BURST_NORMAL
                    },
                    if testing_mode {
                        None
                    } else {
                        Some(GLOBAL_PRICING_TIME_BUDGET_NORMAL)
                    },
                )?;
                generated_global_priced = global_pricing.generated;
                if !global_pricing.lp_basis.is_empty() {
                    lp_basis_cache = global_pricing.lp_basis;
                    force_lp_refresh = false;
                    working_basis = lp_basis_cache.clone();
                }
                total_priced += generated_global_priced;
            }

            if !search_exhausted {
                if force_lp_refresh || lp_basis_cache.is_empty() {
                    let lp = solve_paper_rmp_lp_blocks(
                        &candidates,
                        &original.trees,
                        original.num_leaves as usize,
                    )?;
                    let basis = lp_active_candidates(&lp.candidate_columns, &candidates);
                    lp_basis_cache = if basis.is_empty() {
                        selected.clone()
                    } else {
                        basis
                    };
                    force_lp_refresh = false;
                }
                working_basis = if lp_basis_cache.is_empty() {
                    selected.clone()
                } else {
                    lp_basis_cache.clone()
                };

                if generated_global_priced <= 1 {
                    generated_local_priced = self.add_local_priced_blocks(
                        &mut candidates,
                        &working_basis,
                        &mut seen_leafsets,
                        original,
                        &forbidden_labels,
                    )?;
                }
                total_priced += generated_local_priced;

                generated_add_one += self.add_incumbent_absorption_blocks(
                    &mut candidates,
                    &components,
                    &mut seen_leafsets,
                    original,
                    &forbidden_labels,
                );
                generated_add_one += self.add_incumbent_pair_merge_blocks(
                    &mut candidates,
                    &components,
                    &mut seen_leafsets,
                    original,
                    &forbidden_labels,
                );

                if generated_global_priced + generated_local_priced <= 1 {
                    generated_add_one += self.add_local_add_one_blocks(
                        &mut candidates,
                        &working_basis,
                        &mut seen_leafsets,
                        original,
                        &forbidden_labels,
                    );
                    total_add_one += generated_add_one;
                }
            }

            if candidates.len() > pool_hard_limit {
                let before = candidates.len();
                selected = prune_candidate_pool(&mut candidates, Some(&selected), pool_soft_limit)
                    .unwrap_or(selected);
                lp_basis_cache.clear();
                force_lp_refresh = true;
                global_pricing_workspace = None;
                eprintln!(
                    "[heur:{}] pruned adaptive candidate pool {} -> {} blocks",
                    self.mode_name(),
                    before,
                    candidates.len(),
                );
            }

            let generated_priced = generated_local_priced + generated_global_priced;
            let mut generated_diversified = 0usize;
            let mut round_generated = generated_add_one + generated_priced;
            if !testing_mode
                && (round_generated == 0 || low_yield_rounds >= LOW_YIELD_DIVERSIFY_TRIGGER)
            {
                generated_diversified = self.extend_with_partition_batch(
                    &mut candidates,
                    &mut seen_leafsets,
                    kern,
                    original,
                    &rep_to_all,
                    next_diversify_seed,
                    DIVERSIFY_SEED_BATCH,
                );
                next_diversify_seed += DIVERSIFY_SEED_BATCH;
                round_generated += generated_diversified;
                low_yield_rounds = 0;
            }
            if candidates.len() > pool_hard_limit {
                let before = candidates.len();
                selected = prune_candidate_pool(&mut candidates, Some(&selected), pool_soft_limit)
                    .unwrap_or(selected);
                lp_basis_cache.clear();
                force_lp_refresh = true;
                global_pricing_workspace = None;
                eprintln!(
                    "[heur:{}] pruned diversified candidate pool {} -> {} blocks",
                    self.mode_name(),
                    before,
                    candidates.len(),
                );
            }
            if round_generated != 0 {
                force_lp_refresh = true;
                search_exhausted = false;
                exhausted_rounds = 0;
            }

            let should_refresh_incumbent = if defer_incumbent {
                round_generated != 0
            } else {
                round_generated != 0
                    && (generated_priced != 0
                        || generated_diversified != 0
                        || addone_rounds_since_refresh >= INCUMBENT_REFRESH_ADDONE_ROUNDS)
            };
            let round_selected = if should_refresh_incumbent {
                if defer_incumbent {
                    greedy_nodepack_selection(&candidates, &original.trees, Some(&working_basis))
                } else {
                    solve_nodepack_selection(&candidates, &original.trees)?
                }
            } else {
                selected.clone()
            };
            let previous_components = incumbent_component_count;
            let mut current_savings = selected_savings;
            let mut current_component_count = incumbent_component_count;
            let mut refreshed_components = None;
            let mut refreshed_local_merges = 0usize;
            if should_refresh_incumbent {
                let (mut current_components, _) = materialize_components(
                    &round_selected,
                    &candidates,
                    &rep_to_all,
                    kern,
                    original,
                );
                if let Some((improved_components, merges)) = self.improve_components_locally(
                    &current_components,
                    original,
                    &forbidden_labels,
                ) {
                    current_components = improved_components;
                    refreshed_local_merges = merges;
                }
                current_component_count = current_components.len();
                current_savings = original.num_leaves as usize - current_component_count;
                refreshed_components = Some(current_components);
            }
            selected = round_selected;
            let improved = current_component_count < previous_components;
            let round_gain = if improved {
                previous_components - current_component_count
            } else {
                0
            };

            if should_refresh_incumbent {
                addone_rounds_since_refresh = 0;
                if improved {
                    local_search_merges += refreshed_local_merges;
                    components = refreshed_components.expect("refreshed incumbent");
                    selected_savings = current_savings;
                    best_selected_nontrivial_blocks =
                        count_nontrivial_label_sets(&component_label_sets(&components));
                    incumbent_component_count = components.len();
                    stalled_rounds = 0;
                    low_yield_rounds = 0;
                    eprintln!(
                        "[heur:{}] round {} improved: add_one={} priced_local={} priced_global={} diversified={} lp_basis={} savings={} components={} gain={} local_merges={}",
                        self.mode_name(),
                        round,
                        generated_add_one,
                        generated_local_priced,
                        generated_global_priced,
                        generated_diversified,
                        working_basis.len(),
                        selected_savings,
                        incumbent_component_count,
                        round_gain,
                        local_search_merges,
                    );
                } else {
                    stalled_rounds += 1;
                    if generated_diversified == 0
                        && generated_priced <= 1
                        && generated_add_one <= 64
                    {
                        low_yield_rounds += 1;
                    } else {
                        low_yield_rounds = 0;
                    }
                    eprintln!(
                        "[heur:{}] round {} refreshed-no-improve: add_one={} priced_local={} priced_global={} diversified={} lp_basis={} savings={} components={} local_merges={}",
                        self.mode_name(),
                        round,
                        generated_add_one,
                        generated_local_priced,
                        generated_global_priced,
                        generated_diversified,
                        working_basis.len(),
                        current_savings,
                        current_component_count,
                        refreshed_local_merges,
                    );
                }
            } else {
                addone_rounds_since_refresh = if generated_add_one != 0 {
                    addone_rounds_since_refresh + 1
                } else {
                    0
                };
                eprintln!(
                    "[heur:{}] round {} pool-expanded: add_one={} priced_local={} priced_global={} diversified={} lp_basis={} incumbent_components={} refresh_in={}",
                    self.mode_name(),
                    round,
                    generated_add_one,
                    generated_local_priced,
                    generated_global_priced,
                    generated_diversified,
                    working_basis.len(),
                    incumbent_component_count,
                    if defer_incumbent {
                        0
                    } else {
                        INCUMBENT_REFRESH_ADDONE_ROUNDS.saturating_sub(addone_rounds_since_refresh)
                    },
                );
            }

            if round_generated == 0 {
                search_exhausted = true;
                exhausted_rounds += 1;
                if testing_mode {
                    eprintln!(
                        "[heur:{}] stopping adaptive loop after exhausted pricing phase at round {}",
                        self.mode_name(),
                        round,
                    );
                    break;
                } else {
                    eprintln!(
                        "[heur:{}] round {} exhausted current search phase; idling until SIGTERM (dry_rounds={})",
                        self.mode_name(),
                        round,
                        exhausted_rounds,
                    );
                    let sleep_ms = match exhausted_rounds {
                        0 | 1 => 100,
                        2 => 250,
                        3 => 500,
                        _ => 1000,
                    };
                    std::thread::sleep(Duration::from_millis(sleep_ms));
                }
            }

            if testing_mode && stalled_rounds >= STALL_PATIENCE {
                eprintln!(
                    "[heur:{}] stopping adaptive loop after {} stalled rounds",
                    self.mode_name(),
                    stalled_rounds,
                );
                break;
            }
            round += 1;
        }

        let (generated_add_one_blocks, generated_priced_blocks) = (total_add_one, total_priced);

        if defer_incumbent {
            const FINAL_NODEPACK_TIME_LIMIT_SECS: f64 = 8.0;
            const FINAL_NODEPACK_MAX_CANDIDATES: usize = 3000;
            if candidates.len() > FINAL_NODEPACK_MAX_CANDIDATES {
                eprintln!(
                    "[heur:{}] finalized incumbent from runtime greedy node-pack over {} candidates: selected={} savings={} components={} local_merges={} (exact fallback skipped: candidate pool too large)",
                    self.mode_name(),
                    candidates.len(),
                    best_selected_nontrivial_blocks,
                    selected_savings,
                    incumbent_component_count,
                    local_search_merges,
                );
            } else if let Ok(exact_selected) = solve_nodepack_selection_with_limit(
                &candidates,
                &original.trees,
                Some(FINAL_NODEPACK_TIME_LIMIT_SECS),
            ) {
                let (mut exact_components, _) = materialize_components(
                    &exact_selected,
                    &candidates,
                    &rep_to_all,
                    kern,
                    original,
                );
                let mut exact_local_merges = 0usize;
                if let Some((improved_components, merges)) =
                    self.improve_components_locally(&exact_components, original, &forbidden_labels)
                {
                    exact_components = improved_components;
                    exact_local_merges = merges;
                }
                let exact_savings = original.num_leaves as usize - exact_components.len();
                if exact_components.len() < incumbent_component_count {
                    local_search_merges += exact_local_merges;
                    best_selected_nontrivial_blocks =
                        count_nontrivial_label_sets(&component_label_sets(&exact_components));
                    selected_savings = exact_savings;
                    incumbent_component_count = exact_components.len();
                    components = exact_components;
                    eprintln!(
                        "[heur:{}] finalized incumbent from exact node-pack over {} candidates: selected={} savings={} components={} local_merges={} (time_limit={:.1}s)",
                        self.mode_name(),
                        candidates.len(),
                        best_selected_nontrivial_blocks,
                        selected_savings,
                        incumbent_component_count,
                        local_search_merges,
                        FINAL_NODEPACK_TIME_LIMIT_SECS,
                    );
                } else {
                    eprintln!(
                        "[heur:{}] finalized incumbent from runtime greedy node-pack over {} candidates: selected={} savings={} components={} local_merges={} (exact fallback kept greedy)",
                        self.mode_name(),
                        candidates.len(),
                        best_selected_nontrivial_blocks,
                        selected_savings,
                        incumbent_component_count,
                        local_search_merges,
                    );
                }
            } else {
                eprintln!(
                    "[heur:{}] finalized incumbent from runtime greedy node-pack over {} candidates: selected={} savings={} components={} local_merges={} (exact fallback unavailable)",
                    self.mode_name(),
                    candidates.len(),
                    best_selected_nontrivial_blocks,
                    selected_savings,
                    incumbent_component_count,
                    local_search_merges,
                );
            }
        }

        Ok((
            components,
            RepairStats {
                raw_nontrivial_blocks,
                unique_nontrivial_blocks: seen_leafsets.len(),
                compatible_nontrivial_blocks: candidates.len(),
                selected_nontrivial_blocks: best_selected_nontrivial_blocks,
                selected_savings,
                generated_add_one_blocks,
                generated_priced_blocks,
                local_search_merges,
            },
        ))
    }

    fn add_local_add_one_blocks(
        &self,
        candidates: &mut Vec<PartitionBlock>,
        selected: &[usize],
        seen_leafsets: &mut HashSet<Vec<u32>>,
        original: &Instance,
        forbidden_labels: &HashSet<u32>,
    ) -> usize {
        const MAX_BASE_SIZE: usize = 12;
        const MAX_RISE: u16 = 2;
        const TOP_K: usize = 8;
        const MAX_GENERATED: usize = 20_000;
        let (_, hard_limit) = candidate_pool_limits(original.num_leaves, &original.trees);
        if candidates.len() >= hard_limit {
            return 0;
        }

        let mut generated = 0usize;
        for &idx in selected {
            if generated >= MAX_GENERATED
                || candidates.len() >= hard_limit
                || self.terminate_requested.load(Ordering::Relaxed)
            {
                break;
            }

            let base_labels: Vec<u32> = candidates[idx].labels.clone();
            if base_labels.len() < 2 || base_labels.len() > MAX_BASE_SIZE {
                continue;
            }

            let extras = rank_add_one_extras(
                &base_labels,
                &original.trees,
                original.num_leaves,
                MAX_RISE,
                TOP_K,
            );
            for extra in extras {
                if generated >= MAX_GENERATED || candidates.len() >= hard_limit {
                    break;
                }
                if forbidden_labels.contains(&extra) {
                    continue;
                }
                let mut sup = base_labels.clone();
                match sup.binary_search(&extra) {
                    Ok(_) => continue,
                    Err(pos) => sup.insert(pos, extra),
                }
                if !seen_leafsets.insert(sup.clone()) {
                    continue;
                }
                if !is_set_compatible_all(&original.trees, &sup) {
                    continue;
                }
                candidates.push(PartitionBlock {
                    weight: sup.len() - 1,
                    labels: sup,
                });
                generated += 1;
            }
        }

        generated
    }

    fn add_incumbent_absorption_blocks(
        &self,
        candidates: &mut Vec<PartitionBlock>,
        components: &[Tree],
        seen_leafsets: &mut HashSet<Vec<u32>>,
        original: &Instance,
        forbidden_labels: &HashSet<u32>,
    ) -> usize {
        const MAX_COMPONENT_SIZE: usize = 48;
        const MAX_GENERATED: usize = 4_096;
        let (_, hard_limit) = candidate_pool_limits(original.num_leaves, &original.trees);
        if candidates.len() >= hard_limit {
            return 0;
        }

        let component_labels = component_label_sets(components);
        let active = vec![true; component_labels.len()];
        let mut leaf_to_component = vec![usize::MAX; original.num_leaves as usize + 1];
        let mut component_forbidden = vec![false; component_labels.len()];
        for (idx, labels) in component_labels.iter().enumerate() {
            component_forbidden[idx] = labels.iter().any(|label| forbidden_labels.contains(label));
            for &label in labels {
                leaf_to_component[label as usize] = idx;
            }
        }

        let mut generated = 0usize;
        for (singleton_idx, labels) in component_labels.iter().enumerate() {
            if generated >= MAX_GENERATED
                || candidates.len() >= hard_limit
                || self.terminate_requested.load(Ordering::Relaxed)
            {
                break;
            }
            if component_forbidden[singleton_idx] || labels.len() != 1 {
                continue;
            }
            let leaf = labels[0];
            let neighbors = collect_nearby_component_ids(
                leaf,
                &original.trees,
                &component_labels,
                &leaf_to_component,
                &active,
                singleton_idx,
            );
            for component_idx in neighbors {
                if generated >= MAX_GENERATED || candidates.len() >= hard_limit {
                    break;
                }
                if component_idx == singleton_idx
                    || component_forbidden[component_idx]
                    || component_labels[component_idx].len() < 2
                    || component_labels[component_idx].len() >= MAX_COMPONENT_SIZE
                {
                    continue;
                }
                let mut merged = component_labels[component_idx].clone();
                match merged.binary_search(&leaf) {
                    Ok(_) => continue,
                    Err(pos) => merged.insert(pos, leaf),
                }
                if !seen_leafsets.insert(merged.clone()) {
                    continue;
                }
                if !is_set_compatible_all(&original.trees, &merged) {
                    continue;
                }
                candidates.push(PartitionBlock {
                    weight: merged.len() - 1,
                    labels: merged,
                });
                generated += 1;
            }
        }

        generated
    }

    fn add_incumbent_pair_merge_blocks(
        &self,
        candidates: &mut Vec<PartitionBlock>,
        components: &[Tree],
        seen_leafsets: &mut HashSet<Vec<u32>>,
        original: &Instance,
        forbidden_labels: &HashSet<u32>,
    ) -> usize {
        const MAX_COMPONENT_SIZE: usize = 16;
        const MAX_MERGED_SIZE: usize = 32;
        const MAX_LEAVES_PER_BASE: usize = 8;
        const MAX_GENERATED: usize = 2_048;
        let (_, hard_limit) = candidate_pool_limits(original.num_leaves, &original.trees);
        if candidates.len() >= hard_limit {
            return 0;
        }

        let component_labels = component_label_sets(components);
        let active = vec![true; component_labels.len()];
        let mut leaf_to_component = vec![usize::MAX; original.num_leaves as usize + 1];
        let mut component_forbidden = vec![false; component_labels.len()];
        for (idx, labels) in component_labels.iter().enumerate() {
            component_forbidden[idx] = labels.iter().any(|label| forbidden_labels.contains(label));
            for &label in labels {
                leaf_to_component[label as usize] = idx;
            }
        }

        let mut generated = 0usize;
        for (component_idx, labels) in component_labels.iter().enumerate() {
            if generated >= MAX_GENERATED
                || candidates.len() >= hard_limit
                || self.terminate_requested.load(Ordering::Relaxed)
            {
                break;
            }
            if component_forbidden[component_idx]
                || labels.len() < 2
                || labels.len() > MAX_COMPONENT_SIZE
            {
                continue;
            }

            let mut neighbors = HashSet::<usize>::default();
            for &leaf in labels.iter().take(MAX_LEAVES_PER_BASE) {
                for neighbor in collect_nearby_component_ids(
                    leaf,
                    &original.trees,
                    &component_labels,
                    &leaf_to_component,
                    &active,
                    component_idx,
                ) {
                    neighbors.insert(neighbor);
                }
            }

            for neighbor_idx in neighbors {
                if generated >= MAX_GENERATED || candidates.len() >= hard_limit {
                    break;
                }
                if neighbor_idx <= component_idx
                    || component_forbidden[neighbor_idx]
                    || component_labels[neighbor_idx].len() < 2
                    || component_labels[neighbor_idx].len() > MAX_COMPONENT_SIZE
                {
                    continue;
                }
                if labels.len() + component_labels[neighbor_idx].len() > MAX_MERGED_SIZE {
                    continue;
                }

                let mut merged = labels.clone();
                merged.extend_from_slice(&component_labels[neighbor_idx]);
                merged.sort_unstable();
                merged.dedup();
                if merged.len() < 4 || merged.len() > MAX_MERGED_SIZE {
                    continue;
                }
                if !seen_leafsets.insert(merged.clone()) {
                    continue;
                }
                if !is_set_compatible_all(&original.trees, &merged) {
                    continue;
                }
                candidates.push(PartitionBlock {
                    weight: merged.len() - 1,
                    labels: merged,
                });
                generated += 1;
            }
        }

        generated
    }

    fn improve_components_locally(
        &self,
        components: &[Tree],
        original: &Instance,
        forbidden_labels: &HashSet<u32>,
    ) -> Option<(Vec<Tree>, usize)> {
        const MAX_COMPONENT_SIZE: usize = 48;

        let mut component_labels = component_label_sets(components);
        let mut active = vec![true; component_labels.len()];
        let mut leaf_to_component = vec![usize::MAX; original.num_leaves as usize + 1];
        let mut component_forbidden = vec![false; component_labels.len()];
        let mut component_covers = component_labels
            .iter()
            .map(|labels| {
                original
                    .trees
                    .iter()
                    .map(|tree| internal_cover_nodes(tree, labels))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut used_nodes = original
            .trees
            .iter()
            .map(|tree| vec![0u16; tree.num_nodes()])
            .collect::<Vec<_>>();
        for (idx, labels) in component_labels.iter().enumerate() {
            component_forbidden[idx] = labels.iter().any(|label| forbidden_labels.contains(label));
            for &label in labels {
                leaf_to_component[label as usize] = idx;
            }
            for (tree_idx, nodes) in component_covers[idx].iter().enumerate() {
                for &node in nodes {
                    used_nodes[tree_idx][node] += 1;
                }
            }
        }

        let mut total_merges = 0usize;
        loop {
            if self.terminate_requested.load(Ordering::Relaxed) {
                break;
            }

            let mut merged_this_pass = 0usize;
            for singleton_idx in 0..component_labels.len() {
                if !active[singleton_idx]
                    || component_forbidden[singleton_idx]
                    || component_labels[singleton_idx].len() != 1
                {
                    continue;
                }
                let leaf = component_labels[singleton_idx][0];
                if leaf_to_component[leaf as usize] != singleton_idx {
                    continue;
                }

                let neighbors = collect_nearby_component_ids(
                    leaf,
                    &original.trees,
                    &component_labels,
                    &leaf_to_component,
                    &active,
                    singleton_idx,
                );
                for component_idx in neighbors {
                    if !active[component_idx]
                        || component_forbidden[component_idx]
                        || component_labels[component_idx].len() < 2
                        || component_labels[component_idx].len() >= MAX_COMPONENT_SIZE
                    {
                        continue;
                    }
                    let mut merged = component_labels[component_idx].clone();
                    match merged.binary_search(&leaf) {
                        Ok(_) => continue,
                        Err(pos) => merged.insert(pos, leaf),
                    }
                    if !is_set_compatible_all(&original.trees, &merged) {
                        continue;
                    }
                    let merged_covers = original
                        .trees
                        .iter()
                        .map(|tree| internal_cover_nodes(tree, &merged))
                        .collect::<Vec<_>>();
                    let mut cover_ok = true;
                    for (tree_idx, nodes) in merged_covers.iter().enumerate() {
                        for &node in nodes {
                            let mut used_elsewhere = used_nodes[tree_idx][node];
                            if component_covers[component_idx][tree_idx].contains(&node) {
                                used_elsewhere = used_elsewhere.saturating_sub(1);
                            }
                            if used_elsewhere != 0 {
                                cover_ok = false;
                                break;
                            }
                        }
                        if !cover_ok {
                            break;
                        }
                    }
                    if !cover_ok {
                        continue;
                    }
                    for (tree_idx, nodes) in component_covers[component_idx].iter().enumerate() {
                        for &node in nodes {
                            used_nodes[tree_idx][node] =
                                used_nodes[tree_idx][node].saturating_sub(1);
                        }
                    }
                    component_labels[component_idx] = merged;
                    component_covers[component_idx] = merged_covers;
                    for (tree_idx, nodes) in component_covers[component_idx].iter().enumerate() {
                        for &node in nodes {
                            used_nodes[tree_idx][node] += 1;
                        }
                    }
                    active[singleton_idx] = false;
                    leaf_to_component[leaf as usize] = component_idx;
                    merged_this_pass += 1;
                    break;
                }
            }

            if merged_this_pass == 0 {
                break;
            }
            total_merges += merged_this_pass;
        }

        if total_merges == 0 {
            return None;
        }

        let kept_labels = component_labels
            .into_iter()
            .enumerate()
            .filter_map(|(idx, labels)| active[idx].then_some(labels))
            .collect::<Vec<_>>();
        Some((
            materialize_label_components(&kept_labels, original),
            total_merges,
        ))
    }

    fn build_reduced_singletons(&self, reduced: &Instance) -> Vec<Tree> {
        (1..=reduced.num_leaves)
            .map(|label| Tree::singleton(label, reduced.num_leaves))
            .collect()
    }

    fn add_local_priced_blocks(
        &self,
        candidates: &mut Vec<PartitionBlock>,
        selected: &[usize],
        seen_leafsets: &mut HashSet<Vec<u32>>,
        original: &Instance,
        forbidden_labels: &HashSet<u32>,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        const MAX_BASE_BLOCKS: usize = 48;
        const MAX_BASE_SIZE: usize = 14;
        const MAX_RISE: u16 = 3;
        const TOP_K: usize = 24;
        let (_, hard_limit) = candidate_pool_limits(original.num_leaves, &original.trees);

        if original.num_trees() != 2 || selected.is_empty() || candidates.len() >= hard_limit {
            return Ok(0);
        }

        let mut generated = 0usize;
        for &idx in selected.iter().take(MAX_BASE_BLOCKS) {
            if self.terminate_requested.load(Ordering::Relaxed) || candidates.len() >= hard_limit {
                break;
            }

            let base_labels: Vec<u32> = candidates[idx].labels.clone();
            if base_labels.len() < 2 || base_labels.len() > MAX_BASE_SIZE {
                continue;
            }

            let extras = rank_add_one_extras(
                &base_labels,
                &original.trees,
                original.num_leaves,
                MAX_RISE,
                TOP_K,
            );
            let mut keep = make_leafset(&base_labels, original.num_leaves);
            for extra in extras {
                if !forbidden_labels.contains(&extra) {
                    keep.insert(extra as usize);
                }
            }
            if keep.ones().count() <= base_labels.len() {
                continue;
            }

            let (local_instance, reverse_map) =
                kernelize::restrict_instance_simple(original, &keep);
            if local_instance.num_trees() != 2 || local_instance.num_leaves < 3 {
                continue;
            }

            let mut orig_to_local = vec![0u32; original.num_leaves as usize + 1];
            for local_lbl in 1..=local_instance.num_leaves {
                orig_to_local[reverse_map[local_lbl as usize] as usize] = local_lbl;
            }

            let mut local_candidates = Vec::new();
            let mut local_seen = HashSet::<Vec<u32>>::default();
            for cand in candidates.iter() {
                let mut local_labels = Vec::new();
                let mut inside = true;
                for &leaf in &cand.labels {
                    let local = orig_to_local[leaf as usize];
                    if local == 0 {
                        inside = false;
                        break;
                    }
                    local_labels.push(local);
                }
                if !inside || local_labels.len() < 2 {
                    continue;
                }
                local_labels.sort_unstable();
                if !local_seen.insert(local_labels.clone()) {
                    continue;
                }
                local_candidates.push(PartitionBlock {
                    weight: local_labels.len() - 1,
                    labels: local_labels,
                });
            }

            let lp = solve_paper_rmp_lp_blocks(
                &local_candidates,
                &local_instance.trees,
                local_instance.num_leaves as usize,
            )?;
            let priced = run_rooted_paper_pricer(
                &local_instance.trees[0],
                &local_instance.trees[1],
                &lp.alpha,
                &lp.beta,
            )?;
            let Some((score, local_labels)) = priced else {
                continue;
            };
            if score <= 1.0 + 1e-9 || local_labels.len() < 2 {
                continue;
            }

            let mut original_labels = local_labels
                .iter()
                .map(|&lbl| reverse_map[lbl as usize])
                .collect::<Vec<_>>();
            original_labels.sort_unstable();
            if original_labels
                .iter()
                .any(|label| forbidden_labels.contains(label))
            {
                continue;
            }
            if !seen_leafsets.insert(original_labels.clone()) {
                continue;
            }
            if !is_set_compatible_all(&original.trees, &original_labels) {
                continue;
            }

            candidates.push(PartitionBlock {
                weight: original_labels.len() - 1,
                labels: original_labels,
            });
            generated += 1;
        }

        Ok(generated)
    }

    fn add_global_priced_blocks(
        &self,
        candidates: &mut Vec<PartitionBlock>,
        seen_leafsets: &mut HashSet<Vec<u32>>,
        reduced: &Instance,
        original_to_reduced: &[u32],
        reduced_forbidden: &[u32],
        workspace: &mut Option<GlobalPricingWorkspace>,
        reverse_map: &[u32],
        rep_to_all: &HashMap<u32, Vec<u32>, fxhash::FxBuildHasher>,
        original: &Instance,
        forbidden_labels: &HashSet<u32>,
        max_per_round: usize,
        time_budget: Option<Duration>,
    ) -> Result<GlobalPricingResult, Box<dyn std::error::Error>> {
        let (_, hard_limit) = candidate_pool_limits(original.num_leaves, &original.trees);
        if reduced.num_trees() != 2 || candidates.len() >= hard_limit {
            return Ok(GlobalPricingResult {
                generated: 0,
                lp_basis: Vec::new(),
            });
        }
        let t_global = Instant::now();

        if workspace.is_none() {
            *workspace = Some(build_global_pricing_workspace(
                candidates,
                reduced,
                original_to_reduced,
            )?);
        }
        let workspace = workspace
            .as_mut()
            .expect("global pricing workspace initialized");
        sync_global_pricing_workspace(workspace, candidates, reduced, original_to_reduced)?;

        let mut generated = 0usize;
        let mut last_basis = Vec::new();
        loop {
            if self.terminate_requested.load(Ordering::Relaxed) || candidates.len() >= hard_limit {
                break;
            }
            if max_per_round != 0 && generated >= max_per_round {
                break;
            }
            if generated != 0
                && time_budget
                    .map(|budget| t_global.elapsed() >= budget)
                    .unwrap_or(false)
            {
                break;
            }

            let dual_model =
                std::mem::replace(&mut workspace.dual_model, Model::new(ColProblem::default()));
            let dual_solved = dual_model.solve();
            if dual_solved.status() != HighsModelStatus::Optimal {
                return Err(format!(
                    "global pricing stabilized dual LP status: {:?}",
                    dual_solved.status()
                )
                .into());
            }
            let (mut alpha, beta) = extract_stabilized_dual_solution(
                dual_solved.get_solution(),
                &workspace.dual_penalty_vars,
                &reduced.trees,
                reduced.num_leaves as usize,
            );
            workspace.dual_model = Model::from(dual_solved);
            for &label in reduced_forbidden {
                if (label as usize) < alpha.len() {
                    alpha[label as usize] = -1.0e12;
                }
            }

            let priced =
                run_rooted_paper_pricer(&reduced.trees[0], &reduced.trees[1], &alpha, &beta)?;
            let Some((score, reduced_labels)) = priced else {
                break;
            };
            if score <= 1.0 + 1e-9 {
                break;
            }

            let expanded = expand_reduced_leafset(
                &reduced_labels,
                reverse_map,
                rep_to_all,
                original.num_leaves,
            );
            let mut labels = expanded
                .ones()
                .filter(|&leaf| leaf != 0)
                .map(|leaf| leaf as u32)
                .collect::<Vec<_>>();
            labels.sort_unstable();
            labels.dedup();
            if labels.len() < 2 {
                break;
            }
            if labels.iter().any(|label| forbidden_labels.contains(label)) {
                break;
            }
            if !seen_leafsets.insert(labels.clone()) {
                break;
            }
            if !is_set_compatible_all(&original.trees, &labels) {
                continue;
            }

            candidates.push(PartitionBlock {
                weight: labels.len() - 1,
                labels,
            });
            let original_idx = candidates.len() - 1;
            let reduced_candidate = PartitionBlock {
                weight: reduced_labels.len() - 1,
                labels: reduced_labels,
            };
            let _ = add_reduced_candidate_to_primal_model(
                &mut workspace.primal_model,
                &workspace.primal_leaf_rows,
                &workspace.primal_node_rows,
                &reduced_candidate,
                &reduced.trees,
            );
            add_reduced_candidate_to_stabilized_dual_model(
                &mut workspace.dual_model,
                &workspace.dual_penalty_vars,
                &reduced_candidate,
                &reduced.trees,
            );
            workspace.reduced_to_original.push(original_idx);
            workspace
                .reduced_seen
                .insert(reduced_candidate.labels.clone());
            workspace.reduced_candidates.push(reduced_candidate);
            workspace.synced_original_candidates = candidates.len();
            generated += 1;
        }

        if !workspace.reduced_candidates.is_empty() {
            let primal_model = std::mem::replace(
                &mut workspace.primal_model,
                Model::new(ColProblem::default()),
            );
            let primal_solved = primal_model.solve();
            if primal_solved.status() != HighsModelStatus::Optimal {
                return Err(format!(
                    "global pricing primal LP status: {:?}",
                    primal_solved.status()
                )
                .into());
            }
            let primal_solution = primal_solved.get_solution();
            workspace.primal_model = Model::from(primal_solved);
            let candidate_columns = primal_solution.columns()[workspace.num_singletons
                ..workspace.num_singletons + workspace.reduced_candidates.len()]
                .to_vec();
            let reduced_basis =
                lp_active_candidates(&candidate_columns, &workspace.reduced_candidates);
            last_basis = reduced_basis
                .into_iter()
                .filter_map(|idx| workspace.reduced_to_original.get(idx).copied())
                .collect();
        }

        Ok(GlobalPricingResult {
            generated,
            lp_basis: last_basis,
        })
    }

    fn build_original_singletons(&self, instance: &Instance) -> Vec<Tree> {
        (1..=instance.num_leaves)
            .map(|label| Tree::singleton(label, instance.num_leaves))
            .collect()
    }

    pub fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        if instance.trees.is_empty() {
            return None;
        }

        if instance.num_trees() == 1 {
            self.stats.upper_bound = Some(1);
            return Some(instance.trees.clone());
        }

        let t_total = Instant::now();
        let kern_config = KernelizeConfig::default();
        let t_kernel = Instant::now();
        let kern = kernelize::kernelize_best(instance, &kern_config);
        let reduced = &kern.instance;
        let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "[heur:{}] kernelized {} -> {} leaves in {:.1}ms (param_reduction={})",
            self.mode_name(),
            instance.num_leaves,
            reduced.num_leaves,
            kernel_ms,
            kern.param_reduction,
        );

        if reduced.num_leaves <= 1 {
            let reduced_components = if reduced.num_leaves == 0 {
                Vec::new()
            } else {
                vec![reduced.reference_tree().clone()]
            };
            let components = kernelize::expand_solution(
                reduced_components,
                &kern,
                instance.reference_tree(),
                instance.num_leaves,
            );
            self.stats.upper_bound = Some(components.len());
            return Some(components);
        }

        if self.terminate_requested.load(Ordering::Relaxed) {
            let reduced_components = self.build_reduced_singletons(reduced);
            let components = kernelize::expand_solution(
                reduced_components,
                &kern,
                instance.reference_tree(),
                instance.num_leaves,
            );
            self.stats.upper_bound = Some(components.len());
            return Some(components);
        }

        let t_partition = Instant::now();
        let partition_outcome = self.solve_partitions(
            &reduced.trees,
            reduced.num_leaves as usize,
            heuristic_testing_mode(),
        );
        let partition_ms = t_partition.elapsed().as_secs_f64() * 1000.0;

        match partition_outcome.best_meta {
            Some((ref_idx, seed)) => eprintln!(
                "[heur:{}] reduced partition best={} components in {:.1}ms (ref={}, seed={}, runs={})",
                self.mode_name(),
                partition_outcome.best_components,
                partition_ms,
                ref_idx,
                seed,
                partition_outcome.total_runs,
            ),
            None => eprintln!(
                "[heur:{}] reduced partition best={} components in {:.1}ms (runs={})",
                self.mode_name(),
                partition_outcome.best_components,
                partition_ms,
                partition_outcome.total_runs,
            ),
        }

        let t_repair = Instant::now();
        let (components, repair) =
            match self.build_repaired_components(&partition_outcome.partitions, &kern, instance) {
                Ok(result) => result,
                Err(err) => {
                    eprintln!(
                        "[heur:{}] repair failed: {}. Falling back to reduced singletons.",
                        self.mode_name(),
                        err
                    );
                    (
                        self.build_original_singletons(instance),
                        RepairStats {
                            raw_nontrivial_blocks: 0,
                            unique_nontrivial_blocks: 0,
                            compatible_nontrivial_blocks: 0,
                            selected_nontrivial_blocks: 0,
                            selected_savings: 0,
                            generated_add_one_blocks: 0,
                            generated_priced_blocks: 0,
                            local_search_merges: 0,
                        },
                    )
                }
            };
        let repair_ms = t_repair.elapsed().as_secs_f64() * 1000.0;

        self.stats.upper_bound = Some(components.len());
        eprintln!(
            "[heur:{}] repair kept {}/{} compatible blocks (raw_nontrivial={}, unique_nontrivial={}, generated_add_one={}, generated_priced={}, local_merges={}), savings={}, repair_ms={:.1}",
            self.mode_name(),
            repair.selected_nontrivial_blocks,
            repair.compatible_nontrivial_blocks,
            repair.raw_nontrivial_blocks,
            repair.unique_nontrivial_blocks,
            repair.generated_add_one_blocks,
            repair.generated_priced_blocks,
            repair.local_search_merges,
            repair.selected_savings,
            repair_ms,
        );
        eprintln!(
            "[heur:{}] expanded to {} components in {:.1}ms total",
            self.mode_name(),
            components.len(),
            t_total.elapsed().as_secs_f64() * 1000.0,
        );

        Some(components)
    }
}

struct RepairStats {
    raw_nontrivial_blocks: usize,
    unique_nontrivial_blocks: usize,
    compatible_nontrivial_blocks: usize,
    selected_nontrivial_blocks: usize,
    selected_savings: usize,
    generated_add_one_blocks: usize,
    generated_priced_blocks: usize,
    local_search_merges: usize,
}

struct PartitionOutcome {
    best_components: usize,
    best_meta: Option<(usize, u64)>,
    partitions: Vec<Vec<usize>>,
    total_runs: usize,
}

fn partition_groups(partition: &[usize]) -> Vec<Vec<u32>> {
    let max_comp = partition.iter().copied().max().unwrap_or(0);
    let mut groups = vec![Vec::new(); max_comp + 1];
    for (leaf_idx, &comp) in partition.iter().enumerate() {
        groups[comp].push((leaf_idx + 1) as u32);
    }
    groups
        .into_iter()
        .filter(|group| !group.is_empty())
        .collect()
}

fn expand_reduced_leafset(
    reduced_labels: &[u32],
    reverse_map: &[u32],
    rep_to_all: &HashMap<u32, Vec<u32>, fxhash::FxBuildHasher>,
    original_num_leaves: u32,
) -> FixedBitSet {
    let mut bits = FixedBitSet::with_capacity(original_num_leaves as usize + 1);
    for &reduced_label in reduced_labels {
        let original_label = reverse_map[reduced_label as usize];
        if let Some(all_labels) = rep_to_all.get(&original_label) {
            for &label in all_labels {
                bits.insert(label as usize);
            }
        } else {
            bits.insert(original_label as usize);
        }
    }
    bits
}

fn build_original_to_reduced_map(
    kern: &kernelize::KernelizeResult,
    rep_to_all: &HashMap<u32, Vec<u32>, fxhash::FxBuildHasher>,
    original_num_leaves: u32,
) -> Vec<u32> {
    let mut original_to_reduced = vec![0u32; original_num_leaves as usize + 1];
    for reduced_label in 1..=kern.instance.num_leaves {
        let original_rep = kern.reverse_map[reduced_label as usize];
        original_to_reduced[original_rep as usize] = reduced_label;
        if let Some(all_labels) = rep_to_all.get(&original_rep) {
            for &label in all_labels {
                original_to_reduced[label as usize] = reduced_label;
            }
        }
    }
    original_to_reduced
}

fn build_reduced_forbidden_labels(
    forbidden_labels: &HashSet<u32>,
    original_to_reduced: &[u32],
    reduced_num_leaves: u32,
) -> Vec<u32> {
    let mut reduced_forbidden = FixedBitSet::with_capacity(reduced_num_leaves as usize + 1);
    for &label in forbidden_labels {
        let reduced = original_to_reduced
            .get(label as usize)
            .copied()
            .unwrap_or(0);
        if reduced != 0 {
            reduced_forbidden.insert(reduced as usize);
        }
    }
    reduced_forbidden.ones().map(|idx| idx as u32).collect()
}

fn project_candidate_to_reduced(
    candidate: &PartitionBlock,
    reduced: &Instance,
    original_to_reduced: &[u32],
) -> Option<PartitionBlock> {
    let mut reduced_labels = candidate
        .labels
        .iter()
        .filter_map(|&leaf| {
            let reduced = original_to_reduced.get(leaf as usize).copied().unwrap_or(0);
            (reduced != 0).then_some(reduced)
        })
        .collect::<Vec<_>>();
    reduced_labels.sort_unstable();
    reduced_labels.dedup();
    if reduced_labels.len() < 2 {
        return None;
    }
    if !is_set_compatible_all(&reduced.trees, &reduced_labels) {
        return None;
    }
    Some(PartitionBlock {
        weight: reduced_labels.len() - 1,
        labels: reduced_labels,
    })
}

fn project_candidates_to_reduced(
    candidates: &[PartitionBlock],
    reduced: &Instance,
    original_to_reduced: &[u32],
) -> (Vec<PartitionBlock>, Vec<usize>) {
    let mut reduced_candidates = Vec::with_capacity(candidates.len());
    let mut reduced_to_original = Vec::with_capacity(candidates.len());
    let mut seen = HashSet::<Vec<u32>>::default();
    for (original_idx, cand) in candidates.iter().enumerate() {
        let Some(reduced_candidate) =
            project_candidate_to_reduced(cand, reduced, original_to_reduced)
        else {
            continue;
        };
        if !seen.insert(reduced_candidate.labels.clone()) {
            continue;
        }
        reduced_candidates.push(reduced_candidate);
        reduced_to_original.push(original_idx);
    }
    (reduced_candidates, reduced_to_original)
}

fn make_leafset(labels: &[u32], num_leaves: u32) -> FixedBitSet {
    let mut bits = FixedBitSet::with_capacity(num_leaves as usize + 1);
    for &label in labels {
        bits.insert(label as usize);
    }
    bits
}

fn heuristic_testing_mode() -> bool {
    matches!(
        std::env::var("KLADOS_HEURISTIC_TEST_MODE").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

fn candidate_pool_limits(_num_leaves: u32, _trees: &[Tree]) -> (usize, usize) {
    (100_000, 125_000)
}

fn compare_block_priority(a: &PartitionBlock, b: &PartitionBlock) -> std::cmp::Ordering {
    b.weight
        .cmp(&a.weight)
        .then_with(|| a.labels.len().cmp(&b.labels.len()))
}

fn prune_candidate_pool(
    candidates: &mut Vec<PartitionBlock>,
    selected: Option<&[usize]>,
    target_limit: usize,
) -> Option<Vec<usize>> {
    if candidates.len() <= target_limit {
        return selected.map(|sel| sel.to_vec());
    }

    let mut pinned = vec![false; candidates.len()];
    let pinned_count = if let Some(sel) = selected {
        for &idx in sel {
            if idx < pinned.len() {
                pinned[idx] = true;
            }
        }
        sel.len()
    } else {
        0
    };
    let keep_limit = target_limit.max(pinned_count);

    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_unstable_by(|&i, &j| {
        pinned[j]
            .cmp(&pinned[i])
            .then_with(|| compare_block_priority(&candidates[i], &candidates[j]))
    });

    let mut keep = vec![false; candidates.len()];
    for idx in order.into_iter().take(keep_limit) {
        keep[idx] = true;
    }

    let old = std::mem::take(candidates);
    let mut remap = vec![usize::MAX; old.len()];
    for (old_idx, block) in old.into_iter().enumerate() {
        if keep[old_idx] {
            remap[old_idx] = candidates.len();
            candidates.push(block);
        }
    }

    selected.map(|sel| {
        sel.iter()
            .filter_map(|&idx| {
                let mapped = remap.get(idx).copied().unwrap_or(usize::MAX);
                (mapped != usize::MAX).then_some(mapped)
            })
            .collect()
    })
}

fn materialize_components(
    selected: &[usize],
    candidates: &[PartitionBlock],
    rep_to_all: &HashMap<u32, Vec<u32>, fxhash::FxBuildHasher>,
    kern: &kernelize::KernelizeResult,
    original: &Instance,
) -> (Vec<Tree>, usize) {
    let mut covered = FixedBitSet::with_capacity(original.num_leaves as usize + 1);
    let mut components = Vec::with_capacity(selected.len() + original.num_leaves as usize);
    let mut selected_savings = 0usize;

    for &idx in selected {
        let block = &candidates[idx];
        let leafset = make_leafset(&block.labels, original.num_leaves);
        covered.union_with(&leafset);
        selected_savings += block.weight;
        components.push(Tree::component_from_leafset(
            &leafset,
            original.reference_tree(),
            original.num_leaves,
        ));
    }

    for &original_label in &kern.stats.deleted_labels {
        let mut deleted_leafset = FixedBitSet::with_capacity(original.num_leaves as usize + 1);
        if let Some(all_labels) = rep_to_all.get(&original_label) {
            for &label in all_labels {
                deleted_leafset.insert(label as usize);
            }
        } else {
            deleted_leafset.insert(original_label as usize);
        }
        covered.union_with(&deleted_leafset);
        components.push(Tree::component_from_leafset(
            &deleted_leafset,
            original.reference_tree(),
            original.num_leaves,
        ));
    }

    for label in 1..=original.num_leaves {
        if !covered.contains(label as usize) {
            components.push(Tree::singleton(label, original.num_leaves));
        }
    }

    (components, selected_savings)
}

fn materialize_label_components(label_sets: &[Vec<u32>], original: &Instance) -> Vec<Tree> {
    label_sets
        .iter()
        .map(|labels| {
            if labels.len() == 1 {
                Tree::singleton(labels[0], original.num_leaves)
            } else {
                Tree::component_from_leafset(
                    &make_leafset(labels, original.num_leaves),
                    original.reference_tree(),
                    original.num_leaves,
                )
            }
        })
        .collect()
}

fn component_label_sets(components: &[Tree]) -> Vec<Vec<u32>> {
    components
        .iter()
        .map(|component| component.leaves().collect::<Vec<_>>())
        .collect()
}

fn count_nontrivial_label_sets(label_sets: &[Vec<u32>]) -> usize {
    label_sets.iter().filter(|labels| labels.len() > 1).count()
}

fn collect_nearby_component_ids(
    leaf: u32,
    trees: &[Tree],
    component_labels: &[Vec<u32>],
    leaf_to_component: &[usize],
    active: &[bool],
    self_component: usize,
) -> Vec<usize> {
    const MAX_RISE: usize = 2;
    const MAX_COMPONENTS_PER_SUBTREE: usize = 6;
    const MAX_TOTAL_COMPONENTS: usize = 16;

    let mut scores = HashMap::<usize, (usize, usize, usize)>::new();
    for tree in trees {
        let mut cur = tree.node_by_label(leaf);
        for rise in 1..=MAX_RISE {
            if tree.is_root(cur) {
                break;
            }
            let parent = tree.parent[cur as usize];
            let sibling = if tree.left[parent as usize] == cur {
                tree.right[parent as usize]
            } else {
                tree.left[parent as usize]
            };
            let mut found = HashSet::<usize>::default();
            let mut stack = vec![sibling];
            while let Some(node) = stack.pop() {
                if tree.is_leaf(node) {
                    let label = tree.label[node as usize];
                    let component = leaf_to_component[label as usize];
                    if component == usize::MAX || component == self_component || !active[component]
                    {
                        continue;
                    }
                    if found.insert(component) {
                        let entry = scores.entry(component).or_insert((
                            0usize,
                            0usize,
                            component_labels[component].len(),
                        ));
                        entry.0 += 1;
                        entry.1 += rise;
                        if scores.len() >= MAX_TOTAL_COMPONENTS
                            && found.len() >= MAX_COMPONENTS_PER_SUBTREE
                        {
                            break;
                        }
                    }
                } else {
                    stack.push(tree.left[node as usize]);
                    stack.push(tree.right[node as usize]);
                }
                if found.len() >= MAX_COMPONENTS_PER_SUBTREE {
                    break;
                }
            }
            cur = parent;
            if scores.len() >= MAX_TOTAL_COMPONENTS {
                break;
            }
        }
        if scores.len() >= MAX_TOTAL_COMPONENTS {
            break;
        }
    }

    let mut ranked = scores.into_iter().collect::<Vec<_>>();
    ranked.sort_unstable_by(
        |(ci, (tree_hits_i, rise_i, size_i)), (cj, (tree_hits_j, rise_j, size_j))| {
            tree_hits_j
                .cmp(tree_hits_i)
                .then_with(|| rise_i.cmp(rise_j))
                .then_with(|| size_i.cmp(size_j))
                .then_with(|| ci.cmp(cj))
        },
    );
    ranked.into_iter().map(|(component, _)| component).collect()
}

fn mark_component_nodes(tree: &Tree, labels: &[u32]) -> FixedBitSet {
    let mut bits = FixedBitSet::with_capacity(tree.num_nodes());
    if labels.is_empty() {
        return bits;
    }
    let mut lca_node = tree.node_by_label(labels[0]);
    for &label in &labels[1..] {
        lca_node = tree.nearest_common_ancestor(lca_node, tree.node_by_label(label));
    }
    for &label in labels {
        let mut cur = tree.node_by_label(label);
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

fn internal_cover_nodes(tree: &Tree, labels: &[u32]) -> Vec<usize> {
    mark_component_nodes(tree, labels)
        .ones()
        .filter(|&node| !tree.is_leaf(node as u32))
        .collect()
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

fn lca_of_labels(tree: &Tree, labels: &[u32]) -> u32 {
    let mut iter = labels.iter().copied();
    let first = iter.next().expect("non-empty label set");
    let mut lca = tree.node_by_label(first);
    for label in iter {
        lca = tree.nearest_common_ancestor(lca, tree.node_by_label(label));
    }
    lca
}

fn rank_add_one_extras(
    base: &[u32],
    trees: &[Tree],
    num_leaves: u32,
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
    for extra in 1..=num_leaves {
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

fn build_leaf_to_candidates(candidates: &[PartitionBlock], num_leaves: usize) -> Vec<Vec<usize>> {
    let mut leaf_to_candidates = vec![Vec::new(); num_leaves + 1];
    for (idx, cand) in candidates.iter().enumerate() {
        for &leaf in &cand.labels {
            if (leaf as usize) < leaf_to_candidates.len() {
                leaf_to_candidates[leaf as usize].push(idx);
            }
        }
    }
    leaf_to_candidates
}

fn build_node_to_candidates(candidates: &[PartitionBlock], trees: &[Tree]) -> Vec<Vec<Vec<usize>>> {
    let mut node_to_candidates = trees
        .iter()
        .map(|tree| vec![Vec::new(); tree.num_nodes()])
        .collect::<Vec<_>>();
    for (idx, cand) in candidates.iter().enumerate() {
        for (ti, tree) in trees.iter().enumerate() {
            let cover = mark_component_nodes(tree, &cand.labels);
            for node in cover.ones() {
                if !tree.is_leaf(node as u32) {
                    node_to_candidates[ti][node].push(idx);
                }
            }
        }
    }
    node_to_candidates
}

fn greedy_nodepack_selection(
    candidates: &[PartitionBlock],
    trees: &[Tree],
    preferred: Option<&[usize]>,
) -> Vec<usize> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut preferred_rank = vec![usize::MAX; candidates.len()];
    if let Some(preferred) = preferred {
        for (rank, &idx) in preferred.iter().enumerate() {
            if idx < preferred_rank.len() && preferred_rank[idx] == usize::MAX {
                preferred_rank[idx] = rank;
            }
        }
    }

    let mut covers = Vec::with_capacity(candidates.len());
    for cand in candidates {
        let mut per_tree = Vec::with_capacity(trees.len());
        let mut total_span = 0usize;
        for tree in trees {
            let cover = mark_component_nodes(tree, &cand.labels);
            let internal_nodes = cover
                .ones()
                .filter(|&node| !tree.is_leaf(node as u32))
                .collect::<Vec<_>>();
            total_span += internal_nodes.len();
            per_tree.push(internal_nodes);
        }
        covers.push((per_tree, total_span));
    }

    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_unstable_by(|&i, &j| {
        let pi = preferred_rank[i];
        let pj = preferred_rank[j];
        pi.cmp(&pj)
            .then_with(|| candidates[j].weight.cmp(&candidates[i].weight))
            .then_with(|| covers[i].1.cmp(&covers[j].1))
            .then_with(|| candidates[i].labels.len().cmp(&candidates[j].labels.len()))
    });

    let mut used_nodes = trees
        .iter()
        .map(|tree| FixedBitSet::with_capacity(tree.num_nodes()))
        .collect::<Vec<_>>();
    let mut selected = Vec::new();

    'candidate: for idx in order {
        for (tree_idx, nodes) in covers[idx].0.iter().enumerate() {
            if nodes
                .iter()
                .any(|&node| used_nodes[tree_idx].contains(node))
            {
                continue 'candidate;
            }
        }
        for (tree_idx, nodes) in covers[idx].0.iter().enumerate() {
            for &node in nodes {
                used_nodes[tree_idx].insert(node);
            }
        }
        selected.push(idx);
    }

    selected
}

fn solve_nodepack_selection(
    candidates: &[PartitionBlock],
    trees: &[Tree],
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    solve_nodepack_selection_with_limit(candidates, trees, None)
}

fn solve_nodepack_selection_with_limit(
    candidates: &[PartitionBlock],
    trees: &[Tree],
    time_limit_secs: Option<f64>,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let mut pb = RowProblem::default();
    let vars: Vec<Col> = candidates
        .iter()
        .map(|cand| pb.add_integer_column(cand.weight as f64, 0.0..=1.0))
        .collect();
    let node_to_candidates = build_node_to_candidates(candidates, trees);

    for (tree_idx, tree) in trees.iter().enumerate() {
        for node in 0..tree.num_nodes() as u32 {
            if tree.is_leaf(node) {
                continue;
            }
            let cols = node_to_candidates[tree_idx][node as usize]
                .iter()
                .map(|&idx| (vars[idx], 1.0))
                .collect::<Vec<_>>();
            if cols.len() >= 2 {
                pb.add_row(..=1.0, &cols);
            }
        }
    }

    let mut model = pb.optimise(Sense::Maximise);
    model.make_quiet();
    model.set_option("threads", 1_i32);
    model.set_option("presolve", "on");
    if let Some(time_limit_secs) = time_limit_secs {
        model.set_option("time_limit", time_limit_secs.max(0.001));
    }
    let solved = model.solve();
    let status = solved.status();
    let feasible = solved.primal_solution_status() == highs::HighsSolutionStatus::Feasible;
    if status != HighsModelStatus::Optimal && !feasible {
        return Err(format!(
            "partition nodepack status: {:?} / feasible={}",
            status, feasible
        )
        .into());
    }

    Ok(solved
        .get_solution()
        .columns()
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| if value > 0.5 { Some(idx) } else { None })
        .collect())
}

fn build_global_pricing_workspace(
    candidates: &[PartitionBlock],
    reduced: &Instance,
    original_to_reduced: &[u32],
) -> Result<GlobalPricingWorkspace, Box<dyn std::error::Error>> {
    let (reduced_candidates, reduced_to_original) =
        project_candidates_to_reduced(candidates, reduced, original_to_reduced);

    let mut primal_model = Model::new(ColProblem::default());
    primal_model.make_quiet();
    primal_model.set_option("threads", 1_i32);
    primal_model.set_option("presolve", "on");
    primal_model.set_option("solver", "simplex");

    let mut primal_leaf_rows = vec![None; reduced.num_leaves as usize + 1];
    for leaf in 1..=reduced.num_leaves as usize {
        primal_leaf_rows[leaf] = Some(primal_model.add_row(1.0..=1.0, Vec::new()));
    }
    let mut primal_node_rows = reduced
        .trees
        .iter()
        .map(|tree| vec![None; tree.num_nodes()])
        .collect::<Vec<_>>();
    for (ti, tree) in reduced.trees.iter().enumerate() {
        for node in 0..tree.num_nodes() as u32 {
            if tree.is_leaf(node) {
                continue;
            }
            primal_node_rows[ti][node as usize] = Some(primal_model.add_row(..=1.0, Vec::new()));
        }
    }

    for leaf in 1..=reduced.num_leaves as usize {
        let row = primal_leaf_rows[leaf].expect("leaf row exists");
        let _ = primal_model.add_col(1.0, 0.0..=1.0, vec![(row, 1.0)]);
    }
    for cand in &reduced_candidates {
        let _ = add_reduced_candidate_to_primal_model(
            &mut primal_model,
            &primal_leaf_rows,
            &primal_node_rows,
            cand,
            &reduced.trees,
        );
    }

    let mut dual_model = Model::new(ColProblem::default());
    dual_model.make_quiet();
    dual_model.set_option("threads", 1_i32);
    dual_model.set_option("presolve", "on");
    dual_model.set_option("solver", "simplex");

    let mut dual_penalty_vars: Vec<Vec<Option<Col>>> = reduced
        .trees
        .iter()
        .map(|tree| vec![None; tree.num_nodes()])
        .collect();
    for (ti, tree) in reduced.trees.iter().enumerate() {
        for node in 0..tree.num_nodes() as u32 {
            if tree.is_leaf(node) {
                continue;
            }
            dual_penalty_vars[ti][node as usize] =
                Some(dual_model.add_col(1000.0, 0.0.., Vec::new()));
        }
    }
    for cand in &reduced_candidates {
        add_reduced_candidate_to_stabilized_dual_model(
            &mut dual_model,
            &dual_penalty_vars,
            cand,
            &reduced.trees,
        );
    }

    let reduced_seen = reduced_candidates
        .iter()
        .map(|candidate| candidate.labels.clone())
        .collect::<HashSet<_>>();

    Ok(GlobalPricingWorkspace {
        reduced_candidates,
        reduced_to_original,
        reduced_seen,
        synced_original_candidates: candidates.len(),
        primal_model,
        primal_leaf_rows,
        primal_node_rows,
        dual_model,
        dual_penalty_vars,
        num_singletons: reduced.num_leaves as usize,
    })
}

fn sync_global_pricing_workspace(
    workspace: &mut GlobalPricingWorkspace,
    candidates: &[PartitionBlock],
    reduced: &Instance,
    original_to_reduced: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    if workspace.synced_original_candidates > candidates.len() {
        return Err("global pricing workspace is out of sync with candidate pool".into());
    }

    for original_idx in workspace.synced_original_candidates..candidates.len() {
        let Some(reduced_candidate) =
            project_candidate_to_reduced(&candidates[original_idx], reduced, original_to_reduced)
        else {
            continue;
        };
        if !workspace
            .reduced_seen
            .insert(reduced_candidate.labels.clone())
        {
            continue;
        }
        let _ = add_reduced_candidate_to_primal_model(
            &mut workspace.primal_model,
            &workspace.primal_leaf_rows,
            &workspace.primal_node_rows,
            &reduced_candidate,
            &reduced.trees,
        );
        add_reduced_candidate_to_stabilized_dual_model(
            &mut workspace.dual_model,
            &workspace.dual_penalty_vars,
            &reduced_candidate,
            &reduced.trees,
        );
        workspace.reduced_to_original.push(original_idx);
        workspace.reduced_candidates.push(reduced_candidate);
    }
    workspace.synced_original_candidates = candidates.len();
    Ok(())
}

fn add_reduced_candidate_to_primal_model(
    primal_model: &mut Model,
    leaf_rows: &[Option<Row>],
    node_rows: &[Vec<Option<Row>>],
    cand: &PartitionBlock,
    trees: &[Tree],
) -> Col {
    let mut rows = cand
        .labels
        .iter()
        .map(|&leaf| (leaf_rows[leaf as usize].expect("leaf row exists"), 1.0))
        .collect::<Vec<_>>();
    for (ti, tree) in trees.iter().enumerate() {
        let cover = mark_component_nodes(tree, &cand.labels);
        for node in cover.ones() {
            if tree.is_leaf(node as u32) {
                continue;
            }
            if let Some(row) = node_rows[ti][node] {
                rows.push((row, 1.0));
            }
        }
    }
    primal_model.add_col(1.0, 0.0..=1.0, rows)
}

fn add_reduced_candidate_to_stabilized_dual_model(
    dual_model: &mut Model,
    penalty_vars: &[Vec<Option<Col>>],
    cand: &PartitionBlock,
    trees: &[Tree],
) {
    let mut cover_penalties = Vec::new();
    for (ti, tree) in trees.iter().enumerate() {
        let cover = mark_component_nodes(tree, &cand.labels);
        for node in cover.ones() {
            if tree.is_leaf(node as u32) {
                continue;
            }
            if let Some(var) = penalty_vars[ti][node] {
                cover_penalties.push(var);
            }
        }
    }
    if cover_penalties.is_empty() {
        return;
    }

    dual_model.add_row(
        (cand.weight as f64)..,
        cover_penalties.iter().copied().map(|var| (var, 1.0)),
    );

    if cand.labels.len() > 1 {
        let max_rows = cover_penalties
            .iter()
            .copied()
            .map(|var| dual_model.add_row(..=0.0, vec![(var, 1.0)]))
            .collect::<Vec<_>>();
        let _ = dual_model.add_col(1.0, 0.0.., max_rows.into_iter().map(|row| (row, -1.0)));
    }
}

pub(crate) fn solve_paper_rmp_lp_blocks(
    candidates: &[PartitionBlock],
    trees: &[Tree],
    num_leaves: usize,
) -> Result<PaperLpSolution, Box<dyn std::error::Error>> {
    let mut pb = RowProblem::default();
    let singleton_vars: Vec<Col> = (0..num_leaves)
        .map(|_| pb.add_column(1.0, 0.0..=1.0))
        .collect();
    let candidate_vars: Vec<Col> = candidates
        .iter()
        .map(|_| pb.add_column(1.0, 0.0..=1.0))
        .collect();
    let leaf_to_candidates = build_leaf_to_candidates(candidates, num_leaves);
    let node_to_candidates = build_node_to_candidates(candidates, trees);

    for leaf in 1..=num_leaves {
        let mut cols = vec![(singleton_vars[leaf - 1], 1.0)];
        for &idx in &leaf_to_candidates[leaf] {
            cols.push((candidate_vars[idx], 1.0));
        }
        pb.add_row(1.0..=1.0, &cols);
    }

    for (ti, tree) in trees.iter().enumerate() {
        for node in 0..tree.num_nodes() as u32 {
            if tree.is_leaf(node) {
                continue;
            }
            let cols = node_to_candidates[ti][node as usize]
                .iter()
                .map(|&idx| (candidate_vars[idx], 1.0))
                .collect::<Vec<_>>();
            if cols.len() >= 2 {
                pb.add_row(..=1.0, &cols);
            }
        }
    }

    let mut model = pb.optimise(Sense::Minimise);
    model.make_quiet();
    model.set_option("threads", 1_i32);
    model.set_option("presolve", "on");
    model.set_option("solver", "simplex");
    let solved = model.solve();
    if solved.status() != HighsModelStatus::Optimal {
        return Err(format!("local paper LP status: {:?}", solved.status()).into());
    }

    let solution = solved.get_solution();
    let candidate_columns = solution.columns()[num_leaves..].to_vec();
    let (alpha, beta) = solve_stabilized_dual_blocks(candidates, trees, num_leaves)?;

    Ok(PaperLpSolution {
        alpha,
        beta,
        candidate_columns,
    })
}

fn lp_active_candidates(candidate_columns: &[f64], candidates: &[PartitionBlock]) -> Vec<usize> {
    const ACTIVE_EPS: f64 = 1.0e-6;
    const BASIS_LIMIT: usize = 64;

    let mut active = candidate_columns
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value > ACTIVE_EPS).then_some((idx, value)))
        .collect::<Vec<_>>();
    active.sort_unstable_by(|&(ia, va), &(ib, vb)| {
        vb.total_cmp(&va)
            .then_with(|| candidates[ib].weight.cmp(&candidates[ia].weight))
            .then_with(|| {
                candidates[ia]
                    .labels
                    .len()
                    .cmp(&candidates[ib].labels.len())
            })
    });
    active.truncate(BASIS_LIMIT);
    active.into_iter().map(|(idx, _)| idx).collect()
}

fn clean_dual(value: f64) -> f64 {
    if value.abs() <= 1.0e-9 { 0.0 } else { value }
}

fn solve_stabilized_dual_blocks(
    candidates: &[PartitionBlock],
    trees: &[Tree],
    num_leaves: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), Box<dyn std::error::Error>> {
    if candidates.is_empty() {
        return Ok((
            (0..=num_leaves)
                .map(|leaf| if leaf == 0 { 0.0 } else { 1.0 })
                .collect::<Vec<_>>(),
            trees
                .iter()
                .map(|tree| vec![0.0; tree.num_nodes()])
                .collect::<Vec<_>>(),
        ));
    }

    let mut dual_model = Model::new(ColProblem::default());
    dual_model.make_quiet();
    dual_model.set_option("threads", 1_i32);
    dual_model.set_option("presolve", "on");
    dual_model.set_option("solver", "simplex");

    let mut penalty_vars: Vec<Vec<Option<Col>>> = trees
        .iter()
        .map(|tree| vec![None; tree.num_nodes()])
        .collect();
    for (ti, tree) in trees.iter().enumerate() {
        for node in 0..tree.num_nodes() as u32 {
            if tree.is_leaf(node) {
                continue;
            }
            penalty_vars[ti][node as usize] = Some(dual_model.add_col(1000.0, 0.0.., Vec::new()));
        }
    }
    for cand in candidates {
        add_reduced_candidate_to_stabilized_dual_model(&mut dual_model, &penalty_vars, cand, trees);
    }

    let solved = dual_model.solve();
    if solved.status() != HighsModelStatus::Optimal {
        return Err(format!("stabilized dual LP status: {:?}", solved.status()).into());
    }

    Ok(extract_stabilized_dual_solution(
        solved.get_solution(),
        &penalty_vars,
        trees,
        num_leaves,
    ))
}

fn extract_stabilized_dual_solution(
    solution: highs::Solution,
    penalty_vars: &[Vec<Option<Col>>],
    trees: &[Tree],
    num_leaves: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let alpha = (0..=num_leaves)
        .map(|leaf| if leaf == 0 { 0.0 } else { 1.0 })
        .collect::<Vec<_>>();
    let mut beta = trees
        .iter()
        .map(|tree| vec![0.0; tree.num_nodes()])
        .collect::<Vec<_>>();
    for (ti, tree) in trees.iter().enumerate() {
        for node in 0..tree.num_nodes() {
            if let Some(var) = penalty_vars[ti][node] {
                beta[ti][node] = clean_dual(solution[var]);
            }
        }
    }
    (alpha, beta)
}

fn run_rooted_paper_pricer(
    t1: &Tree,
    t2: &Tree,
    alpha: &[f64],
    beta: &[Vec<f64>],
) -> Result<Option<(f64, Vec<u32>)>, Box<dyn std::error::Error>> {
    const NEG_INF: f64 = -1.0e100;

    let n2 = t2.num_nodes();
    let idx = |u: u32, v: u32| -> usize { u as usize * n2 + v as usize };
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
                        m_score[pair] = 0.0;
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

    let mut labels = Vec::new();
    collect_m_labels(
        t1,
        t2,
        &m_choice,
        &v_choice,
        &split_choice,
        t1.root,
        t2.root,
        &mut labels,
    );
    labels.sort_unstable();
    labels.dedup();
    Ok(Some((root_score, labels)))
}

fn collect_m_labels(
    t1: &Tree,
    t2: &Tree,
    m_choice: &[MChoice],
    v_choice: &[VChoice],
    split_choice: &[SplitChoice],
    u: u32,
    v: u32,
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
    u: u32,
    v: u32,
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
    u: u32,
    v: u32,
    out: &mut Vec<u32>,
) {
    let n2 = t2.num_nodes();
    let pair = u as usize * n2 + v as usize;
    match split_choice[pair] {
        SplitChoice::None => {}
        SplitChoice::Straight => {
            let (ul, ur) = t1.children(u).expect("straight split requires internal u");
            let (vl, vr) = t2.children(v).expect("straight split requires internal v");
            collect_v_labels(t1, t2, v_choice, split_choice, ul, vl, out);
            collect_v_labels(t1, t2, v_choice, split_choice, ur, vr, out);
        }
        SplitChoice::Cross => {
            let (ul, ur) = t1.children(u).expect("cross split requires internal u");
            let (vl, vr) = t2.children(v).expect("cross split requires internal v");
            collect_v_labels(t1, t2, v_choice, split_choice, ul, vr, out);
            collect_v_labels(t1, t2, v_choice, split_choice, ur, vl, out);
        }
    }
}

impl HeuristicSolver for PartitionHeuristicSolver {
    fn name(&self) -> &'static str {
        self.mode_name()
    }

    fn description(&self) -> &'static str {
        "Greedy partition heuristic with union-add-one refinement"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("KLADOS_HEURISTIC_TEST_MODE", "enable extended search budget (set to 1)"),
        ]
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        PartitionHeuristicSolver::solve(self, instance)
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }

    fn sigterm_handler(&self) {
        self.terminate_requested.store(true, Ordering::SeqCst);
    }
}

// ===========================================================================
// Dual-guided set-packing GAP EXPERIMENT (design doc §6, de-risking)
//
// Measures, on a fixed column pool, whether LP duals improve a greedy packing
// and how large the integrality gap is. Reuses this module's master LP
// (`solve_paper_rmp_lp_blocks` → leaf+node duals), the rooted dual pricer
// (`run_rooted_paper_pricer`), greedy node-packing, and the exact node-pack
// ILP. Size-aware: heavy O(n²) steps (pricer/LP) are gated so the run stays
// inside memory/time on the largest instances.
// ===========================================================================

/// One row of experiment output.
pub struct GapExperimentResult {
    pub n: usize,
    pub best_known: usize,
    pub pool_size: usize,
    pub cg_columns_added: usize,
    pub lp_components: Option<f64>,      // fractional master objective over pool
    pub greedy_priceless: Option<usize>, // (i) weight-first greedy packing
    pub greedy_dual: Option<usize>,      // (ii) LP-basis-guided greedy packing
    pub pool_ip: Option<usize>,          // (iii) exact node-pack ILP over pool
    pub pool_ip_timed_out: bool,
    pub timings_ms: Vec<(&'static str, f64)>,
}

fn comps_from_selection(n: usize, candidates: &[PartitionBlock], sel: &[usize]) -> usize {
    let savings: usize = sel.iter().map(|&i| candidates[i].weight).sum();
    n - savings
}

/// Run the §6 de-risking experiment on a single 2-tree instance.
pub fn run_packing_gap_experiment(instance: &Instance, best_known: usize) -> GapExperimentResult {
    use std::collections::HashSet;
    use std::time::Instant;

    let trees = &instance.trees;
    let n = instance.num_leaves as usize;
    let t1 = &trees[0];
    let t2 = &trees[1];

    // Size gates (node products to bound memory of the O(n1*n2) DP / LP).
    let node_product = t1.num_nodes() * t2.num_nodes();
    let allow_cg = node_product <= 30_000_000;
    let allow_lp = n <= 8_000;
    let num_seeds: u64 = if n <= 200 { 40 } else if n <= 2_000 { 16 } else if n <= 6_000 { 6 } else { 3 };

    let mut timings = Vec::new();
    let mut seen: HashSet<Vec<u32>> = HashSet::new();
    let mut pool: Vec<PartitionBlock> = Vec::new();

    let mut push_block = |labels: Vec<u32>, pool: &mut Vec<PartitionBlock>, seen: &mut HashSet<Vec<u32>>| {
        if labels.len() < 2 { return; }
        let mut l = labels; l.sort_unstable(); l.dedup();
        if l.len() < 2 || !seen.insert(l.clone()) { return; }
        if !is_set_compatible_all(trees, &l) { return; }
        pool.push(PartitionBlock { weight: l.len() - 1, labels: l });
    };

    // ----- Pool: Chen 2-approx forest blocks -----
    let t = Instant::now();
    let (_lo, _up, chen_sets) = klados_exact::chen_rspr::chen_pair_agreement(t1, t2);
    for s in chen_sets { push_block(s, &mut pool, &mut seen); }
    timings.push(("chen_pool", t.elapsed().as_secs_f64() * 1000.0));

    // ----- Pool: multi-seed greedy cherry partitions (overlapping blocks) -----
    let t = Instant::now();
    for ref_idx in 0..trees.len() {
        for seed in 0..num_seeds {
            let (_k, part) = klados_core::lower_bound::greedy_multi_tree_partition(trees, ref_idx, seed);
            for g in partition_groups(&part) { push_block(g, &mut pool, &mut seen); }
        }
    }
    timings.push(("seed_pool", t.elapsed().as_secs_f64() * 1000.0));

    // ----- Column generation enrichment via the rooted dual pricer -----
    let mut cg_added = 0usize;
    let mut last_lp: Option<PaperLpSolution> = None;
    if allow_cg && allow_lp {
        let t = Instant::now();
        let cg_budget = std::time::Duration::from_secs(if n <= 200 { 30 } else { 60 });
        let cg_start = Instant::now();
        let max_rounds = if n <= 200 { 400 } else { 150 };
        for _round in 0..max_rounds {
            if cg_start.elapsed() >= cg_budget { break; }
            let lp = match solve_paper_rmp_lp_blocks(&pool, trees, n) {
                Ok(lp) => lp,
                Err(_) => break,
            };
            let priced = run_rooted_paper_pricer(t1, t2, &lp.alpha, &lp.beta);
            last_lp = Some(lp);
            match priced {
                Ok(Some((score, labels))) if score > 1.0 + 1e-6 => {
                    let before = pool.len();
                    push_block(labels, &mut pool, &mut seen);
                    if pool.len() == before { break; } // nothing new -> converged
                    cg_added += 1;
                }
                _ => break,
            }
        }
        timings.push(("column_gen", t.elapsed().as_secs_f64() * 1000.0));
    }

    // ----- Final LP solve for reported duals + fractional objective -----
    let mut lp_components = None;
    let mut dual_basis: Option<Vec<usize>> = None;
    if allow_lp {
        let t = Instant::now();
        let lp = if last_lp.is_some() && cg_added == 0 {
            last_lp
        } else {
            solve_paper_rmp_lp_blocks(&pool, trees, n).ok()
        };
        if let Some(lp) = lp {
            let savings: f64 = pool
                .iter()
                .zip(lp.candidate_columns.iter())
                .map(|(b, &x)| x * b.weight as f64)
                .sum();
            lp_components = Some(n as f64 - savings);
            dual_basis = Some(lp_active_candidates(&lp.candidate_columns, &pool));
        }
        timings.push(("final_lp", t.elapsed().as_secs_f64() * 1000.0));
    }

    // ----- (i) priceless weight-first greedy -----
    let t = Instant::now();
    let sel_priceless = greedy_nodepack_selection(&pool, trees, None);
    let greedy_priceless = Some(comps_from_selection(n, &pool, &sel_priceless));
    timings.push(("greedy_priceless", t.elapsed().as_secs_f64() * 1000.0));

    // ----- (ii) dual-guided greedy (LP basis as preference) -----
    let greedy_dual = dual_basis.as_ref().map(|basis| {
        let t = Instant::now();
        let sel = greedy_nodepack_selection(&pool, trees, Some(basis));
        let r = comps_from_selection(n, &pool, &sel);
        timings.push(("greedy_dual", t.elapsed().as_secs_f64() * 1000.0));
        r
    });

    // ----- (iii) exact node-pack ILP over the pool (best achievable from pool) -----
    let t = Instant::now();
    let ip_limit = if n <= 200 { 20.0 } else { 30.0 };
    let (pool_ip, pool_ip_timed_out) =
        match solve_nodepack_selection_with_limit(&pool, trees, Some(ip_limit)) {
            Ok(sel) => {
                let comps = comps_from_selection(n, &pool, &sel);
                let elapsed = t.elapsed().as_secs_f64();
                (Some(comps), elapsed >= ip_limit * 0.95)
            }
            Err(_) => (None, false),
        };
    timings.push(("pool_ip", t.elapsed().as_secs_f64() * 1000.0));

    GapExperimentResult {
        n,
        best_known,
        pool_size: pool.len(),
        cg_columns_added: cg_added,
        lp_components,
        greedy_priceless,
        greedy_dual,
        pool_ip,
        pool_ip_timed_out,
        timings_ms: timings,
    }
}
