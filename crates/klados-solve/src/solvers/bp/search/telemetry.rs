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
    /// True if the pricer ever returned `Converged` (proved no positive-RC
    /// columns exist) at any node.  Indicates the LP-bound prune was backed
    /// by a pricing optimality proof at least once.
    pub had_converged: bool,
}

#[derive(Default, Debug)]
pub struct Timings {
    pub lp_solve: Duration,
    pub pricing: Duration,
    pub branching: Duration,
    pub bounds_apply: Duration,
    pub cut_separation: Duration,
}
