//! Central solver catalog: name + description + in-process entry point.
//!
//! This is the single list the dev CLI uses to print the available solvers and
//! to dispatch `klados solve <name>` in-process. Each `run` points at the
//! solver module's entry fn (the same one its `klados-<name>` binary calls), so
//! `klados solve <name>` and the binary are identical.

/// Static metadata + entry point for one solver.
pub struct SolverInfo {
    pub name: &'static str,
    pub description: &'static str,
    /// Runs the solver end-to-end (stdin -> solve -> stdout), with its config.
    pub run: fn(),
}

/// All solvers, in listing order.
pub fn catalog() -> &'static [SolverInfo] {
    use crate::solvers::*;
    &[
        SolverInfo {
            name: "dispatch",
            description: "Feature-based portfolio: routes each instance to bp or sat by (m, n)",
            run: dispatch::main,
        },
        SolverInfo {
            name: "bp",
            description: "Branch & Price for multi-tree MAF (exact)",
            run: bp::main,
        },
        SolverInfo {
            name: "bp-multi",
            description: "Branch & Price, legacy multi-tree variant (exact)",
            run: maf_branch_price_multi::main,
        },
        SolverInfo {
            name: "ilp",
            description: "Integer Linear Programming via HiGHS",
            run: maf_ilp::main,
        },
        SolverInfo {
            name: "sat",
            description: "SAT encoding via rustsat/cadical",
            run: maf_sat::main,
        },
        SolverInfo {
            name: "sat-olver",
            description: "SAT with Olver 2-approx LB seeding",
            run: maf_sat::olver_main,
        },
        SolverInfo {
            name: "chen-rspr",
            description: "Chen rSPR branch-and-bound (2-tree only)",
            run: chen_rspr::main,
        },
        SolverInfo {
            name: "whidden",
            description: "Whidden 3-way branch-and-bound (2-tree only)",
            run: whidden::main,
        },
        SolverInfo {
            name: "corridor",
            description: "Reduced-cost corridor solver (m=2 native)",
            run: corridor::main,
        },
        SolverInfo {
            name: "root-corridor",
            description: "Certified root-corridor probe with B&P fallback",
            run: root_corridor::main,
        },
        SolverInfo {
            name: "root-pool",
            description: "Root column generation + integer pool cover (prototype)",
            run: root_pool::main,
        },
        SolverInfo {
            name: "collapse",
            description: "Coarsen-Collapse: CG + column-crossing tree-DP master (fast, anytime)",
            run: root_pool::collapse_main,
        },
        SolverInfo {
            name: "overlay-exchange",
            description: "Incumbent-overlay replacement (prototype)",
            run: overlay_exchange::main,
        },
        SolverInfo {
            name: "lagrangian",
            description: "Dual-guided set-packing, Lagrangian column generation (anytime)",
            run: lagrangian::main,
        },
        SolverInfo {
            name: "agglomerative",
            description: "Agglomerative clustering heuristic",
            run: agglomerative::main,
        },
        SolverInfo {
            name: "greedy-partition",
            description: "Greedy partition heuristic with union-add-one refinement",
            run: partition::main,
        },
        SolverInfo {
            name: "maxsat",
            description: "MaxSAT via open-wbo (legacy)",
            run: max_sat::main,
        },
        SolverInfo {
            name: "lower",
            description: "Lower-bound track racer: fastest #a-bounded forest",
            run: lower::main,
        },
    ]
}
