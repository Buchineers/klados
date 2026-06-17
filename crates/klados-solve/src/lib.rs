//! klados-solve: merged exact + heuristic solvers for Maximum Agreement Forest.
//!
//! Transitional state: still exposes the legacy `ExactSolver` / `HeuristicSolver`
//! traits and registries. The unified `Solver` trait + `run()` harness are
//! layered on in a following step.

pub use klados_core::cluster_reduction;
// kernelize and lower_bound live in klados-core; re-export for convenience.
pub use klados_core::kernelize;
pub use klados_core::lower_bound;

// ── the solvers + decomposition primitives ──────────────────────────────────
pub mod decomp;
pub mod solvers;

pub use crate::solvers::partition::{GapExperimentResult, run_packing_gap_experiment};

use klados_core::{Instance, SolverStats, Tree};

// ── Exact solver trait + registry ───────────────────────────────────────────

/// Trait for exact solver approaches.
pub trait ExactSolver {
    /// Short name used for CLI selection.
    fn name(&self) -> &'static str;
    /// One-line description.
    fn description(&self) -> &'static str {
        ""
    }
    /// Configurable options: `[(name, description)]`. Empty if none.
    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }
    /// Solve the instance, returning the components of the agreement forest.
    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>>;
    /// Access statistics.
    fn stats(&self) -> &SolverStats;
    /// Handle SIGTERM. Solvers that support early termination should store this
    /// notification and gracefully return the best solution found so far.
    #[cfg(feature = "early-termination")]
    fn sigterm_handler(&self) {}
}

/// Return all available exact solvers.
pub fn available_solvers() -> Vec<Box<dyn ExactSolver>> {
    vec![
        Box::new(solvers::maf_ilp::MafIlpSolver::new()),
        Box::new(solvers::maf_sat::MafSatSolver::new()),
        Box::new(solvers::maf_sat::MafSatOlverSolver::new()),
        Box::new(solvers::chen_rspr::ChenRsprSolver::new()),
        Box::new(solvers::whidden::WhiddenSolver::new()),
        Box::new(solvers::maf_branch_price_multi::MafBranchPriceMultiSolver::new()),
        Box::new(solvers::bp::BpSolver::new()),
        Box::new(solvers::overlay_exchange::OverlayExchangeSolver::new()),
        Box::new(solvers::root_corridor::RootCorridorSolver::new()),
        Box::new(solvers::root_pool::RootPoolSolver::new()),
        Box::new(solvers::corridor::CorridorSolver::new()),
    ]
}

/// Lookup an exact solver by name (case-insensitive).
pub fn solver_by_name(name: &str) -> Option<Box<dyn ExactSolver>> {
    let name = name.trim().to_ascii_lowercase();
    available_solvers()
        .into_iter()
        .find(|solver| solver.name().eq_ignore_ascii_case(&name))
}

// ── Heuristic solver trait + registry ───────────────────────────────────────

use crate::solvers::agglomerative::AgglomerativeSolver;
use crate::solvers::lagrangian::LagrangianSolver;
use crate::solvers::max_sat::MaxSatSolver;
use crate::solvers::partition::PartitionHeuristicSolver;

/// Trait for heuristic solver approaches.
pub trait HeuristicSolver {
    /// Short name used for CLI selection.
    fn name(&self) -> &'static str;
    /// One-line description.
    fn description(&self) -> &'static str {
        ""
    }
    /// Configurable options: `[(name, description)]`. Empty if none.
    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }
    /// Solve the instance, returning the components of the agreement forest.
    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>>;
    /// Access statistics.
    fn stats(&self) -> &SolverStats;
    /// Handle SIGTERM signal.
    fn sigterm_handler(&self);
    /// Best complete, ready-to-emit forest found so far (original labels).
    fn snapshot(&self) -> Option<Vec<Tree>> {
        None
    }
}

/// Return all advertised heuristic solvers.
pub fn available_heuristic_solvers() -> Vec<Box<dyn HeuristicSolver>> {
    vec![
        Box::new(PartitionHeuristicSolver::greedy_union_add_one()),
        Box::new(AgglomerativeSolver::new()),
    ]
}

/// Lookup a heuristic solver by name.
pub fn heuristic_solver_by_name(name: &str) -> Option<Box<dyn HeuristicSolver>> {
    match name {
        "greedy-partition-union-addone" => {
            Some(Box::new(PartitionHeuristicSolver::greedy_union_add_one()))
        }
        "agglomerative" => Some(Box::new(AgglomerativeSolver::new())),
        "lagrangian" => Some(Box::new(LagrangianSolver::new())),
        // Legacy path, addressable but not advertised.
        "maxsat" => Some(Box::new(MaxSatSolver::new())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_maf_ilp() {
        assert!(solver_by_name("ilp").is_some());
    }
    #[test]
    fn test_registry_has_maf_sat() {
        assert!(solver_by_name("sat").is_some());
    }
    #[test]
    fn test_registry_has_maf_sat_olver() {
        assert!(solver_by_name("sat-olver").is_some());
    }
    #[test]
    fn test_registry_has_maf_bp_multi() {
        assert!(solver_by_name("bp-multi").is_some());
    }
    #[test]
    fn test_registry_has_bp() {
        assert!(solver_by_name("bp").is_some());
    }
    #[test]
    fn test_registry_has_chen_rspr() {
        assert!(solver_by_name("chen-rspr").is_some());
    }
    #[test]
    fn test_registry_has_overlay_exchange() {
        assert!(solver_by_name("overlay-exchange").is_some());
    }
    #[test]
    fn test_registry_has_root_pool() {
        assert!(solver_by_name("root-pool").is_some());
    }
    #[test]
    fn test_registry_has_root_corridor() {
        assert!(solver_by_name("root-corridor").is_some());
    }
}
