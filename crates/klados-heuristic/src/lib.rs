//! klados-heuristic: Heuristic FPT solvers for Maximum Agreement Forest

pub mod agglomerative;
pub mod dive_bp;
pub mod lagrangian;
pub mod max_sat;
pub mod partition;

use klados_core::{Instance, SolverStats, Tree};

use crate::agglomerative::AgglomerativeSolver;
use crate::dive_bp::DiveBpSolver;
use crate::lagrangian::LagrangianSolver;
use crate::max_sat::MaxSatSolver;
use crate::partition::PartitionHeuristicSolver;

pub use crate::partition::{GapExperimentResult, run_packing_gap_experiment};

/// Trait for heuristic solver approaches.
pub trait HeuristicSolver {
    /// Short name used for CLI selection.
    fn name(&self) -> &'static str;

    /// One-line description.
    fn description(&self) -> &'static str { "" }

    /// Configurable options: `[(name, description)]`. Empty if none.
    fn options(&self) -> &'static [(&'static str, &'static str)] { &[] }

    /// Solve the instance, returning the components of the agreement forest.
    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>>;
    /// Access statistics.
    fn stats(&self) -> &SolverStats;
    /// Handle SIGTERM signal.
    fn sigterm_handler(&self);

    /// Best complete, ready-to-emit forest found so far (original labels).
    /// Returned by an anytime solver so a watcher can emit it the instant a
    /// SIGTERM arrives, WITHOUT waiting for `solve` to unwind. `None` ⇒ the
    /// solver doesn't maintain a live incumbent (the harness then waits for
    /// `solve` to return normally).
    fn snapshot(&self) -> Option<Vec<Tree>> {
        None
    }
}

/// Return all available heuristic solvers.
pub fn available_solvers() -> Vec<Box<dyn HeuristicSolver>> {
    vec![
        Box::new(PartitionHeuristicSolver::greedy_union_add_one()),
        Box::new(AgglomerativeSolver::new()),
    ]
}

pub fn solver_by_name(name: &str) -> Option<Box<dyn HeuristicSolver>> {
    match name {
        "greedy-partition-union-addone" => {
            Some(Box::new(PartitionHeuristicSolver::greedy_union_add_one()))
        }
        "agglomerative" => Some(Box::new(AgglomerativeSolver::new())),
        "lagrangian" => Some(Box::new(LagrangianSolver::new())),
        "divebp" => Some(Box::new(DiveBpSolver::new())),
        // Keep the legacy path addressable explicitly without advertising it.
        "maxsat" => Some(Box::new(MaxSatSolver::new())),
        _ => None,
    }
}
