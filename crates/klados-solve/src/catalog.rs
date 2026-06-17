//! Central solver catalog: name + description + in-process entry point.
//!
//! This is the single list the dev CLI uses to print the available solvers and
//! to dispatch `klados solve <name>` in-process. Each `run` points at the
//! solver module's entry fn (the same one its `klados-<name>` binary calls), so
//! `klados solve <name>` and the binary are identical.

use crate::Solver;

/// Static metadata + entry point for one solver.
pub struct SolverInfo {
    pub name: &'static str,
    pub description: &'static str,
    /// Runs the solver end-to-end (stdin -> solve -> stdout), with its config.
    pub run: fn(),
    /// `(env-var, description)` knobs the solver still reads — see
    /// [`Solver::OPTIONS`]. Surfaced by `klados solve` (no-arg).
    pub options: fn() -> &'static [(&'static str, &'static str)],
}

/// Monomorphized accessor for a solver's [`Solver::OPTIONS`], so the catalog can
/// carry it without an instance.
fn opts<S: Solver>() -> &'static [(&'static str, &'static str)] {
    S::OPTIONS
}

/// All solvers, in listing order.
pub fn catalog() -> &'static [SolverInfo] {
    use crate::solvers::*;
    &[
        SolverInfo { name: "bp", description: "Branch & Price for multi-tree MAF (exact, default)", run: bp::main, options: opts::<bp::BpSolver> },
        SolverInfo { name: "bp-multi", description: "Branch & Price, legacy multi-tree variant (exact)", run: maf_branch_price_multi::main, options: opts::<maf_branch_price_multi::MafBranchPriceMultiSolver> },
        SolverInfo { name: "ilp", description: "Integer Linear Programming via HiGHS", run: maf_ilp::main, options: opts::<maf_ilp::MafIlpSolver> },
        SolverInfo { name: "sat", description: "SAT encoding via rustsat/cadical", run: maf_sat::main, options: opts::<maf_sat::MafSatSolver> },
        SolverInfo { name: "sat-olver", description: "SAT with Olver 2-approx LB seeding", run: maf_sat::olver_main, options: opts::<maf_sat::MafSatOlverSolver> },
        SolverInfo { name: "chen-rspr", description: "Chen rSPR branch-and-bound (2-tree only)", run: chen_rspr::main, options: opts::<chen_rspr::ChenRsprSolver> },
        SolverInfo { name: "whidden", description: "Whidden 3-way branch-and-bound (2-tree only)", run: whidden::main, options: opts::<whidden::WhiddenSolver> },
        SolverInfo { name: "corridor", description: "Reduced-cost corridor solver (m=2 native)", run: corridor::main, options: opts::<corridor::CorridorSolver> },
        SolverInfo { name: "root-corridor", description: "Certified root-corridor probe with B&P fallback", run: root_corridor::main, options: opts::<root_corridor::RootCorridorSolver> },
        SolverInfo { name: "root-pool", description: "Root column generation + integer pool cover (prototype)", run: root_pool::main, options: opts::<root_pool::RootPoolSolver> },
        SolverInfo { name: "overlay-exchange", description: "Incumbent-overlay replacement (prototype)", run: overlay_exchange::main, options: opts::<overlay_exchange::OverlayExchangeSolver> },
        SolverInfo { name: "lagrangian", description: "Dual-guided set-packing, Lagrangian column generation (anytime)", run: lagrangian::main, options: opts::<lagrangian::LagrangianSolver> },
        SolverInfo { name: "agglomerative", description: "Agglomerative clustering heuristic", run: agglomerative::main, options: opts::<agglomerative::AgglomerativeSolver> },
        SolverInfo { name: "greedy-partition", description: "Greedy partition heuristic with union-add-one refinement", run: partition::main, options: opts::<partition::PartitionHeuristicSolver> },
        SolverInfo { name: "maxsat", description: "MaxSAT via open-wbo (legacy)", run: max_sat::main, options: opts::<max_sat::MaxSatSolver> },
        SolverInfo { name: "lower", description: "Lower-bound track racer: fastest #a-bounded forest", run: lower::main, options: opts::<lower::LowerSolver> },
    ]
}
