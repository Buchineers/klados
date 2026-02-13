//! klados-exact: Exact FPT solvers for Maximum Agreement Forest
//!
//! Each approach is implemented as its own solver and registered here.

mod fpt;
pub mod lower_bound;
mod shi_mestel;

use klados_core::{Instance, SolverStats, Tree};

/// Trait for exact solver approaches.
pub trait ExactSolver {
    /// Short name used for CLI selection.
    fn name(&self) -> &'static str;
    /// Solve the instance, returning the components of the agreement forest.
    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>>;
    /// Access statistics.
    fn stats(&self) -> &SolverStats;
}

/// Return all available exact solvers.
pub fn available_solvers() -> Vec<Box<dyn ExactSolver>> {
    vec![
        Box::new(fpt::FptSolver::new()),
        Box::new(shi_mestel::ShiMestelSolver::new()),
    ]
}

/// Lookup solver by name (case-insensitive).
pub fn solver_by_name(name: &str) -> Option<Box<dyn ExactSolver>> {
    let name = name.trim().to_ascii_lowercase();
    if name == "fpt" {
        return Some(Box::new(fpt::FptSolver::new()));
    }
    if name == "shi-mestel" || name == "shimestel" {
        return Some(Box::new(shi_mestel::ShiMestelSolver::new()));
    }
    for solver in available_solvers() {
        if solver.name().eq_ignore_ascii_case(&name) {
            return Some(solver);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_fpt() {
        let solver = solver_by_name("whidden");
        assert!(solver.is_some());
    }

    #[test]
    fn test_registry_has_shi_mestel() {
        let solver = solver_by_name("shi-mestel");
        assert!(solver.is_some());
    }
}
