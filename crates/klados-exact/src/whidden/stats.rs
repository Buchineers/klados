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
    }
}

#[derive(Clone, Debug, Default)]
pub struct WhiddenRunStats {
    pub rules: WhiddenRuleStats,
}
