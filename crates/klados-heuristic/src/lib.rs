//! klados-heuristic: Heuristic FPT solvers for Maximum Agreement Forest

pub mod max_sat;

use klados_core::{Instance, SolverStats, Tree};

/// Trait for heuristic solver approaches.
pub trait HeuristicSolver {
    /// Short name used for CLI selection.
    fn name(&self) -> &'static str;
    /// Solve the instance, returning the components of the agreement forest.
    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>>;
    /// Access statistics.
    fn stats(&self) -> &SolverStats;
    /// Handle SIGTERM signal. Called by signal handler when process receives SIGTERM.
    fn sigterm_handler(&self);
}

/// Run all available heuristc solvers.
pub fn available_solvers() -> Vec<Box<dyn HeuristicSolver>> {
    vec![]
}

pub fn solver_by_name(name: &str) -> Option<Box<dyn HeuristicSolver>> {
    available_solvers().into_iter().find(|s| s.name() == name)
}
