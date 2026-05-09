//! Unified solver registry.
//!
//! `SolverChoice` is the single source of truth — each variant carries its
//! display name, kind, description, and a constructor.

use klados_core::Instance;
use klados_core::Tree;
use clap::ValueEnum;
use pace26io::newick::NewickWriter;
use std::io::{self, Write};

// ── Solver choice enum ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
pub enum SolverChoice {
    #[value(name = "bp-multi")]
    BpMulti,
    #[value(name = "chen-rspr")]
    ChenRspr,
    #[value(name = "sat")]
    Sat,
    #[value(name = "ilp")]
    ILP,
    #[value(name = "sat-olver")]
    MafSatOlver,
    #[value(name = "whidden")]
    Whidden,
    #[value(name = "greedy-partition-union-addone")]
    GreedyPartition,
    #[value(name = "agglomerative")]
    Agglomerative,
}

impl SolverChoice {
    pub fn kind(&self) -> SolverKind {
        use SolverChoice::*;
        match self {
            GreedyPartition | Agglomerative => SolverKind::Heuristic,
            _ => SolverKind::Exact,
        }
    }

    pub fn description(&self) -> &'static str {
        use SolverChoice::*;
        match self {
            ILP => "Integer Linear Programming via HiGHS",
            Sat => "SAT-based encoding via rustsat/cadical",
            MafSatOlver => "SAT-based with Olver 2-approx LB seeding",
            ChenRspr => "Chen-style rSPR branch-and-bound (2-tree only)",
            Whidden => "Whidden 3-way branch-and-bound (2-tree only)",
            BpMulti => "Branch & Price for multi-tree MAF (default)",
            GreedyPartition => "Greedy partition heuristic with union-add-one refinement",
            Agglomerative => "Agglomerative clustering heuristic",
        }
    }

    pub fn options(&self) -> &'static [(&'static str, &'static str)] {
        use SolverChoice::*;
        match self {
            ILP => &[],
            Sat => &[
                ("KLADOS_MAF_SAT_H4", "H4 mode: full, lazy, seeded-lazy, staged"),
                ("KLADOS_MAF_SAT_COMPONENT_TRACE", "enable component trace"),
            ],
            MafSatOlver => &[
                ("KLADOS_MAF_SAT_H4", "H4 mode: full, lazy, seeded-lazy, staged"),
                ("KLADOS_MAF_SAT_H4_PROMOTE_MS", "promote MaxSAT solution"),
            ],
            ChenRspr => &[
                ("KLADOS_CHEN_JAR_DUMMY", "enable jar-dummy leaf"),
                ("KLADOS_CHEN_USE_FORCED", "enable forced-cut pre-branch"),
                ("KLADOS_CHEN_NO_RECURSIVE_LB", "disable recursive LB"),
                ("KLADOS_CHEN_BOUNDS", "print bound details"),
                ("KLADOS_CHEN_TRACE_K", "trace k-search"),
                ("KLADOS_CHEN_STOPPERS", "trace stopper classifications"),
            ],
            Whidden => &[
                ("KLADOS_WHIDDEN_BATCH_STRICT", "strict cluster points"),
                ("KLADOS_WHIDDEN_RSPR_GREEDY", "greedy rSPR decomposition"),
                ("KLADOS_WHIDDEN_DEBUG", "enable debug output"),
            ],
            BpMulti => &[],
            GreedyPartition => &[
                ("KLADOS_HEURISTIC_TEST_MODE", "extended search budget"),
            ],
            Agglomerative => &[],
        }
    }

    pub fn build(&self) -> Box<dyn AnySolver> {
        match self {
            SolverChoice::ILP => from_exact(klados_exact::maf_ilp::MafIlpSolver::new()),
            SolverChoice::Sat => from_exact(klados_exact::maf_sat::MafSatSolver::new()),
            SolverChoice::MafSatOlver => from_exact(klados_exact::maf_sat::MafSatOlverSolver::new()),
            SolverChoice::ChenRspr => from_exact(klados_exact::chen_rspr::ChenRsprSolver::new()),
            SolverChoice::Whidden => from_exact(klados_exact::whidden::WhiddenSolver::new()),
            SolverChoice::BpMulti => from_exact(klados_exact::maf_branch_price_multi::MafBranchPriceMultiSolver::new()),
            SolverChoice::GreedyPartition => from_heuristic(klados_heuristic::partition::PartitionHeuristicSolver::greedy_union_add_one()),
            SolverChoice::Agglomerative => from_heuristic(klados_heuristic::agglomerative::AgglomerativeSolver::new()),
        }
    }
}

// ── Solver kind ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverKind {
    Exact,
    Heuristic,
}

impl std::fmt::Display for SolverKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverKind::Exact => write!(f, "exact"),
            SolverKind::Heuristic => write!(f, "heuristic"),
        }
    }
}

// ── Listing ────────────────────────────────────────────────────────────────

pub fn list_solvers() {
    eprintln!("{:=<72}", "");
    eprintln!("  {:<6}  {:<30}  {}", "TYPE", "NAME", "DESCRIPTION");
    eprintln!("{:=<72}", "");
    for choice in SolverChoice::value_variants() {
        let name = choice.to_possible_value().unwrap().get_name().to_owned();
        eprintln!("  {:<6}  {:<30}  {}",
            choice.kind().to_string(), name, choice.description());
        for (opt, desc) in choice.options() {
            eprintln!("  {:>6}    {:>28}: {}", "", opt, desc);
        }
    }
    eprintln!("{:=<72}", "");
}

// ── AnySolver trait ────────────────────────────────────────────────────────

pub trait AnySolver {
    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>>;
}

struct ExactWrapper(Box<dyn klados_exact::ExactSolver>);
struct HeuristicWrapper(Box<dyn klados_heuristic::HeuristicSolver>);

impl AnySolver for ExactWrapper {
    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        self.0.solve(instance)
    }
}

impl AnySolver for HeuristicWrapper {
    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        self.0.solve(instance)
    }
}

fn from_exact(s: impl klados_exact::ExactSolver + 'static) -> Box<dyn AnySolver> {
    Box::new(ExactWrapper(Box::new(s)))
}

fn from_heuristic(s: impl klados_heuristic::HeuristicSolver + 'static) -> Box<dyn AnySolver> {
    Box::new(HeuristicWrapper(Box::new(s)))
}

// ── Solve + print ──────────────────────────────────────────────────────────

pub fn solve_and_print(instance: &Instance, choice: SolverChoice) -> Result<(), Box<dyn std::error::Error>> {
    log::info!(
        "solving: {} trees, {} leaves, solver={}",
        instance.num_trees(),
        instance.num_leaves,
        choice.to_possible_value().unwrap().get_name(),
    );

    let mut solver = choice.build();
    let components = solver.solve(instance).expect("failed to find solution");

    let mut stdout = io::stdout().lock();
    for tree in &components {
        tree.cursor().write_newick(&mut stdout)?;
        writeln!(stdout)?;
    }

    Ok(())
}
