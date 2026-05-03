//! klados-exact: Exact FPT solvers for Maximum Agreement Forest
//!
//! Each approach is implemented as its own solver and registered here.

pub use klados_core::cluster_reduction;
// kernelize and lower_bound now live in klados-core; re-export for backward compatibility
pub use klados_core::kernelize;
pub use klados_core::lower_bound;
pub mod chen_rspr;
pub mod maf_branch_price;
pub mod maf_branch_price_multi;
pub mod maf_ilp;
pub mod maf_sat;
pub mod shi_mestel;
pub mod whidden;

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
        Box::new(shi_mestel::ShiMestelSolver::new()),
        Box::new(maf_ilp::MafIlpSolver::new()),
        Box::new(maf_sat::MafSatSolver::new()),
        Box::new(maf_sat::MafSatOlverSolver::new()),
        Box::new(chen_rspr::ChenRsprSolver::new()),
        Box::new(whidden::WhiddenSolver::new()),
        Box::new(maf_branch_price::MafBranchPriceSolver::new()),
        Box::new(maf_branch_price_multi::MafBranchPriceMultiSolver::new()),
    ]
}

/// Lookup solver by name (case-insensitive).
pub fn solver_by_name(name: &str) -> Option<Box<dyn ExactSolver>> {
    let name = name.trim().to_ascii_lowercase();
    available_solvers()
        .into_iter()
        .find(|solver| solver.name().eq_ignore_ascii_case(&name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_shi_mestel() {
        let solver = solver_by_name("shi-mestel");
        assert!(solver.is_some());
    }

    #[test]
    fn test_registry_has_maf_ilp() {
        let solver = solver_by_name("maf-ilp");
        assert!(solver.is_some());
    }

    #[test]
    fn test_registry_has_maf_sat() {
        let solver = solver_by_name("maf-sat");
        assert!(solver.is_some());
    }

    #[test]
    fn test_registry_has_maf_sat_olver() {
        let solver = solver_by_name("maf-sat-olver");
        assert!(solver.is_some());
    }

    #[test]
    fn test_registry_has_maf_bp() {
        let solver = solver_by_name("maf-bp");
        assert!(solver.is_some());
    }

    #[test]
    fn test_registry_has_maf_bp_multi() {
        let solver = solver_by_name("maf-bp-multi");
        assert!(solver.is_some());
    }

    #[test]
    fn test_registry_has_chen_rspr() {
        let solver = solver_by_name("chen-rspr");
        assert!(solver.is_some());
    }
}
