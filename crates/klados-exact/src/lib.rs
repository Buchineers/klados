//! klados-exact: Exact FPT solvers for Maximum Agreement Forest
//!
//! Each approach is implemented as its own solver and registered here.

pub use klados_core::cluster_reduction;
// kernelize and lower_bound now live in klados-core; re-export for backward compatibility
pub use klados_core::kernelize;
pub use klados_core::lower_bound;
pub mod bp;
pub mod chen_rspr;
pub mod corridor;
pub mod maf_branch_price_multi;
pub mod maf_ilp;
pub mod maf_sat;
pub mod overlay_exchange;
pub mod root_corridor;
pub mod root_pool;
pub mod whidden;
pub mod whidden_cluster;

use klados_core::{Instance, SolverStats, Tree};

/// Trait for exact solver approaches.
pub trait ExactSolver {
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

    /// Handle SIGTERM. Solvers that support early termination should
    /// store this notification and gracefully return the best solution
    /// found so far from `solve`.
    #[cfg(feature = "early-termination")]
    fn sigterm_handler(&self) {}
}

/// Return all available exact solvers.
pub fn available_solvers() -> Vec<Box<dyn ExactSolver>> {
    vec![
        Box::new(maf_ilp::MafIlpSolver::new()),
        Box::new(maf_sat::MafSatSolver::new()),
        Box::new(maf_sat::MafSatOlverSolver::new()),
        Box::new(chen_rspr::ChenRsprSolver::new()),
        Box::new(whidden::WhiddenSolver::new()),
        Box::new(maf_branch_price_multi::MafBranchPriceMultiSolver::new()),
        Box::new(bp::BpSolver::new()),
        Box::new(overlay_exchange::OverlayExchangeSolver::new()),
        Box::new(root_corridor::RootCorridorSolver::new()),
        Box::new(root_pool::RootPoolSolver::new()),
        Box::new(corridor::CorridorSolver::new()),
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
    fn test_registry_has_maf_ilp() {
        let solver = solver_by_name("ilp");
        assert!(solver.is_some());
    }

    #[test]
    fn test_registry_has_maf_sat() {
        let solver = solver_by_name("sat");
        assert!(solver.is_some());
    }

    #[test]
    fn test_registry_has_maf_sat_olver() {
        let solver = solver_by_name("sat-olver");
        assert!(solver.is_some());
    }

    #[test]
    fn test_registry_has_maf_bp_multi() {
        let solver = solver_by_name("bp-multi");
        assert!(solver.is_some());
    }

    #[test]
    fn test_registry_has_bp() {
        let solver = solver_by_name("bp");
        assert!(solver.is_some());
    }

    #[test]
    fn test_registry_has_chen_rspr() {
        let solver = solver_by_name("chen-rspr");
        assert!(solver.is_some());
    }

    #[test]
    fn test_registry_has_overlay_exchange() {
        let solver = solver_by_name("overlay-exchange");
        assert!(solver.is_some());
    }

    #[test]
    fn test_registry_has_root_pool() {
        let solver = solver_by_name("root-pool");
        assert!(solver.is_some());
    }

    #[test]
    fn test_registry_has_root_corridor() {
        let solver = solver_by_name("root-corridor");
        assert!(solver.is_some());
    }
}
