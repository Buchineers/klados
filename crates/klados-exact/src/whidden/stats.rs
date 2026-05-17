use std::ops::AddAssign;

/// Per-run rule/pruning telemetry for Whidden branch-and-bound.
#[derive(Clone, Debug, Default)]
pub struct WhiddenRuleStats {
    // Search envelope
    pub lb_k: usize,
    pub ub_k: usize,
    pub current_k: Option<usize>,
    pub k_attempts: u64,
    pub k_success: Option<usize>,
    pub k_last_elapsed_ms: f64,
    pub k_total_elapsed_ms: f64,

    // Action flow
    pub action_singleton_cuts: u64,
    pub action_case2_contracts: u64,
    pub action_case3_branches: u64,
    pub action_done: u64,

    // Forced-pair / preference behavior
    pub forced_pair_attempts: u64,
    pub forced_pair_invalidated: u64,
    pub forced_pair_case2: u64,
    pub forced_pair_case3: u64,
    pub prefer_nonbranching_hits: u64,

    // Rule fires (effective)
    pub rule_cob_fired: u64,
    pub rule_rcob_a_fired: u64,
    pub rule_rcob_c_fired: u64,
    pub rule_cut_two_b_fired: u64,
    pub rule_cut_all_b_forced: u64,
    pub rule_mestel6_forced: u64,

    // Branch skips by reason
    pub skip_a_cob: u64,
    pub skip_a_rcob_c: u64,
    pub skip_a_ep_protected: u64,

    pub skip_b_rcob_a: u64,
    pub skip_b_rcob_c: u64,
    pub skip_b_separate_components: u64,
    pub skip_b_ep_protected: u64,

    pub skip_c_cob: u64,
    pub skip_c_rcob_a: u64,
    pub skip_c_ep_protected: u64,

    // Branch attempt/success
    pub branch_a_attempts: u64,
    pub branch_a_successes: u64,
    pub branch_b_attempts: u64,
    pub branch_b_successes: u64,
    pub branch_c_attempts: u64,
    pub branch_c_successes: u64,

    // Pruning / fail reasons
    pub prune_k_exhausted: u64,
    pub prune_bb_approx: u64,
    pub prune_no_enabled_branches: u64,

    // Transposition table
    pub tt_lookups: u64,
    pub tt_hits: u64,
    pub tt_prunes: u64,
    pub tt_stores: u64,
    pub tt_overwrites: u64,

    // Experimental rooted split-or-decompose rescue
    pub mestel6_checks: u64,

    // SPLIT diagnostic: per-branching-node overlap/decomposition check.
    // Counts whether the current TwinForest state, before branching, has
    // overlapping component embeddings in T1 (Mestel's SPLIT triggers) or
    // pairwise disjoint embeddings (Mestel's DECOMPOSE triggers).
    pub split_diag_nodes: u64,           // # nodes where the check was run
    pub split_diag_overlap: u64,         // # nodes with at least one overlapping pair
    pub split_diag_disjoint: u64,        // # nodes where all pairs are disjoint (≥2 comps)
    pub split_diag_single_component: u64,// # nodes with only one component (vacuous)
    pub split_diag_disjoint_blocks_sum: u64,    // sum of block counts at disjoint nodes
    pub split_diag_disjoint_max_block_sum: u64, // sum of max block sizes at disjoint nodes

    // SPLIT rule (Mestel 2024) firing counters. Active when
    // BBConfig::use_split_or_decompose is on.
    pub split_rule_checked: u64,        // # times SPLIT rule entry point was hit
    pub split_rule_overlap_found: u64,  // # times overlap detected → SPLIT would fire
    pub split_rule_disjoint_found: u64, // # times disjoint → DECOMPOSE would fire
    pub split_rule_applied: u64,        // # times SPLIT actually applied (Day 7+)
    pub split_rule_core_cutsets: u64,   // total cutsets across constructed SPLIT cores
    pub split_rule_core_edges: u64,     // total edge multiplicity across those cutsets
    pub split_rule_size1_cutsets: u64,  // cutsets with exactly one edge

    // Bound cache & propagation
    pub bc_lookups: u64,
    pub bc_hits: u64,
    pub bc_stores: u64,
    pub bb_skipped_by_parent: u64,
    pub bb_approx3_calls: u64,
    pub bb_approx2_calls: u64,
    pub bb_approx2_prunes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct WhiddenProgressUpdate {
    pub current_k: usize,
    pub lb_k: usize,
    pub ub_k: usize,
    pub k_attempts: u64,
    pub k_elapsed_ms: f64,
    pub nodes_explored: u64,
    pub solved: bool,
}

impl AddAssign<&WhiddenRuleStats> for WhiddenRuleStats {
    fn add_assign(&mut self, rhs: &WhiddenRuleStats) {
        self.k_attempts += rhs.k_attempts;
        self.k_last_elapsed_ms = rhs.k_last_elapsed_ms;
        self.k_total_elapsed_ms += rhs.k_total_elapsed_ms;

        self.action_singleton_cuts += rhs.action_singleton_cuts;
        self.action_case2_contracts += rhs.action_case2_contracts;
        self.action_case3_branches += rhs.action_case3_branches;
        self.action_done += rhs.action_done;

        self.forced_pair_attempts += rhs.forced_pair_attempts;
        self.forced_pair_invalidated += rhs.forced_pair_invalidated;
        self.forced_pair_case2 += rhs.forced_pair_case2;
        self.forced_pair_case3 += rhs.forced_pair_case3;
        self.prefer_nonbranching_hits += rhs.prefer_nonbranching_hits;

        self.rule_cob_fired += rhs.rule_cob_fired;
        self.rule_rcob_a_fired += rhs.rule_rcob_a_fired;
        self.rule_rcob_c_fired += rhs.rule_rcob_c_fired;
        self.rule_cut_two_b_fired += rhs.rule_cut_two_b_fired;
        self.rule_cut_all_b_forced += rhs.rule_cut_all_b_forced;
        self.rule_mestel6_forced += rhs.rule_mestel6_forced;

        self.skip_a_cob += rhs.skip_a_cob;
        self.skip_a_rcob_c += rhs.skip_a_rcob_c;
        self.skip_a_ep_protected += rhs.skip_a_ep_protected;

        self.skip_b_rcob_a += rhs.skip_b_rcob_a;
        self.skip_b_rcob_c += rhs.skip_b_rcob_c;
        self.skip_b_separate_components += rhs.skip_b_separate_components;
        self.skip_b_ep_protected += rhs.skip_b_ep_protected;

        self.skip_c_cob += rhs.skip_c_cob;
        self.skip_c_rcob_a += rhs.skip_c_rcob_a;
        self.skip_c_ep_protected += rhs.skip_c_ep_protected;

        self.branch_a_attempts += rhs.branch_a_attempts;
        self.branch_a_successes += rhs.branch_a_successes;
        self.branch_b_attempts += rhs.branch_b_attempts;
        self.branch_b_successes += rhs.branch_b_successes;
        self.branch_c_attempts += rhs.branch_c_attempts;
        self.branch_c_successes += rhs.branch_c_successes;

        self.prune_k_exhausted += rhs.prune_k_exhausted;
        self.prune_bb_approx += rhs.prune_bb_approx;
        self.prune_no_enabled_branches += rhs.prune_no_enabled_branches;

        self.tt_lookups += rhs.tt_lookups;
        self.tt_hits += rhs.tt_hits;
        self.tt_prunes += rhs.tt_prunes;
        self.tt_stores += rhs.tt_stores;
        self.tt_overwrites += rhs.tt_overwrites;
        self.mestel6_checks += rhs.mestel6_checks;

        self.split_diag_nodes += rhs.split_diag_nodes;
        self.split_diag_overlap += rhs.split_diag_overlap;
        self.split_diag_disjoint += rhs.split_diag_disjoint;
        self.split_diag_single_component += rhs.split_diag_single_component;
        self.split_diag_disjoint_blocks_sum += rhs.split_diag_disjoint_blocks_sum;
        self.split_diag_disjoint_max_block_sum += rhs.split_diag_disjoint_max_block_sum;

        self.split_rule_checked += rhs.split_rule_checked;
        self.split_rule_overlap_found += rhs.split_rule_overlap_found;
        self.split_rule_disjoint_found += rhs.split_rule_disjoint_found;
        self.split_rule_applied += rhs.split_rule_applied;
        self.split_rule_core_cutsets += rhs.split_rule_core_cutsets;
        self.split_rule_core_edges += rhs.split_rule_core_edges;
        self.split_rule_size1_cutsets += rhs.split_rule_size1_cutsets;

        self.bc_lookups += rhs.bc_lookups;
        self.bc_hits += rhs.bc_hits;
        self.bc_stores += rhs.bc_stores;
        self.bb_skipped_by_parent += rhs.bb_skipped_by_parent;
        self.bb_approx3_calls += rhs.bb_approx3_calls;
        self.bb_approx2_calls += rhs.bb_approx2_calls;
        self.bb_approx2_prunes += rhs.bb_approx2_prunes;
    }
}

#[derive(Clone, Debug, Default)]
pub struct WhiddenRunStats {
    pub rules: WhiddenRuleStats,
}
