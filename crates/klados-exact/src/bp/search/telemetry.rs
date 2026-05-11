//! Pure observation. The search loop never reads telemetry; removing this
//! module would not change behaviour. Stored separately from [`SearchState`]
//! so it can be serialised/dumped for analysis without interfering with
//! the algorithm.

use std::time::Duration;

#[derive(Default, Debug)]
pub struct Telemetry {
    pub cg_iters: usize,
    pub nodes_explored: usize,
    pub columns_added: usize,
    pub incumbent_updates: usize,
    pub bound_prunes: usize,
    pub cuts_added: usize,
    pub timings: Timings,
    /// Set by the search loop when a pricer reports `Exhausted`. If any node
    /// terminated CG without a `Converged` proof, the LP-bound prune used at
    /// that node was technically heuristic. Useful for soundness audits.
    pub had_unsound_prune: bool,
}

#[derive(Default, Debug)]
pub struct Timings {
    pub lp_solve: Duration,
    pub pricing: Duration,
    pub branching: Duration,
    pub bounds_apply: Duration,
    pub cut_separation: Duration,
}
