//! Unified solver registry.
//!
//! `SolverChoice` is the single source of truth — each variant carries its
//! display name, kind, description, and a constructor.

use clap::ValueEnum;
use klados_core::Instance;
use klados_core::Tree;
use pace26io::newick::NewickWriter;
use std::io::{self, Write};
#[cfg(feature = "early-termination")]
use std::sync::Arc;
#[cfg(feature = "early-termination")]
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

// ── Solver choice enum ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
pub enum SolverChoice {
    #[value(name = "bp-multi")]
    BpMulti,
    #[value(name = "bp")]
    Bp,
    #[value(name = "chen-rspr")]
    ChenRspr,
    #[value(name = "sat")]
    Sat,
    #[value(name = "ilp")]
    ILP,
    #[value(name = "sat-olver")]
    MafSatOlver,
    #[value(name = "maxhs-probe")]
    MaxhsProbe,
    #[value(name = "reach-refute")]
    ReachRefute,
    #[value(name = "whidden")]
    Whidden,
    #[value(name = "greedy-partition-union-addone")]
    GreedyPartition,
    #[value(name = "agglomerative")]
    Agglomerative,
    #[value(name = "lagrangian")]
    Lagrangian,
    #[value(name = "overlay-exchange")]
    OverlayExchange,
    #[value(name = "root-pool")]
    RootPool,
    #[value(name = "root-corridor")]
    RootCorridor,
    #[value(name = "corridor")]
    Corridor,
    #[value(name = "lower")]
    Lower,
}

impl SolverChoice {
    pub fn kind(&self) -> SolverKind {
        use SolverChoice::*;
        match self {
            GreedyPartition | Agglomerative | Lagrangian | OverlayExchange | RootPool => {
                SolverKind::Heuristic
            }
            RootCorridor | Corridor => SolverKind::Exact,
            _ => SolverKind::Exact,
        }
    }

    /// Whether this solver consumes the `#a` approximation parameters.
    pub fn is_lower_track(&self) -> bool {
        matches!(self, SolverChoice::Lower)
    }

    pub fn description(&self) -> &'static str {
        use SolverChoice::*;
        match self {
            ILP => "Integer Linear Programming via HiGHS",
            Sat => "SAT-based encoding via rustsat/cadical",
            MafSatOlver => "SAT-based with Olver 2-approx LB seeding",
            MaxhsProbe => "Core-guided implicit-hitting-set lower-bound probe (measurement)",
            ReachRefute => "Bound-bracketed refutation on static reach encoding (emits forest)",
            ChenRspr => "Chen-style rSPR branch-and-bound (2-tree only)",
            Whidden => "Whidden 3-way branch-and-bound (2-tree only)",
            BpMulti => "Branch & Price for multi-tree MAF (default)",
            Bp => "Branch & Price (rewrite, in progress)",
            GreedyPartition => "Greedy partition heuristic with union-add-one refinement",
            Agglomerative => "Agglomerative clustering heuristic",
            Lagrangian => "Dual-guided set-packing (Lagrangian column generation, anytime)",
            OverlayExchange => "Incumbent-overlay replacement prototype",
            RootPool => "Root column generation + integer pool cover prototype",
            RootCorridor => "Certified root-corridor probe with exact B&P fallback",
            Corridor => "Reduced-cost corridor solver (m=2 native; m≥3 routes to B&P)",
            Lower => "Lower-bound track racer: fastest #a-bounded forest (Chen 2-approx, bp fallback)",
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
            MaxhsProbe => &[
                ("KLADOS_PROBE_BUDGET_S", "wall budget in seconds (default 120)"),
                ("KLADOS_PROBE_OPT", "known optimum for gap reporting"),
            ],
            ReachRefute => &[("KLADOS_REACH_BUDGET_S", "wall budget in seconds (default 1700)")],
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
            Bp => &[],
            GreedyPartition => &[("KLADOS_HEURISTIC_TEST_MODE", "extended search budget")],
            Agglomerative => &[],
            Lagrangian => &[
                ("KLADOS_HEUR_TIME_MS", "wall-time budget in ms (default 290000)"),
                ("KLADOS_LAGR_TRACE", "print per-iteration diagnostics"),
            ],
            OverlayExchange => &[
                ("KLADOS_OVERLAY_MAX_H", "maximum incumbent neighborhood size"),
                ("KLADOS_OVERLAY_MAX_ROUNDS", "maximum improvement rounds"),
                ("KLADOS_OVERLAY_LOCAL_LEAF_CAP", "skip split neighborhoods above this many leaves"),
                ("KLADOS_OVERLAY_MAX_NEIGHBORHOODS", "neighborhood checks per round cap"),
                ("KLADOS_OVERLAY_GEN_CAP", "per-neighborhood local generation cap"),
                ("KLADOS_OVERLAY_TRACE", "print diagnostics"),
            ],
            RootPool => &[
                ("KLADOS_ROOT_POOL_MAX_CG", "maximum root column-generation iterations"),
                ("KLADOS_ROOT_POOL_MAX_MS", "soft root-pool wall budget in milliseconds"),
                ("KLADOS_ROOT_POOL_MIP_PASSES", "lazy-cut MIP repair passes"),
                ("KLADOS_ROOT_POOL_MIP_TIME_LIMIT", "HiGHS MIP time limit per pass in seconds"),
                ("KLADOS_ROOT_POOL_SEEDS", "randomized incumbent seed budget"),
                ("KLADOS_ROOT_POOL_TRACE", "print diagnostics"),
            ],
            RootCorridor => &[
                ("KLADOS_ROOT_CORRIDOR_MAX_PROBE_LEAVES", "skip root probe above this leaf count"),
                ("KLADOS_ROOT_CORRIDOR_PROBE_MS", "soft root probe wall budget in milliseconds"),
                ("KLADOS_ROOT_CORRIDOR_MIP_TIME_LIMIT", "HiGHS pool-MIP time limit per probe pass"),
                ("KLADOS_ROOT_CORRIDOR_TRACE", "print diagnostics"),
            ],
            Corridor => &[
                ("KLADOS_CORRIDOR_MAX_CG", "max root CG iterations per outer iter"),
                ("KLADOS_CORRIDOR_MAX_OUTER", "max outer (γ-shrink) iterations"),
                ("KLADOS_CORRIDOR_MAX_MS", "soft wall-time budget in milliseconds"),
                ("KLADOS_CORRIDOR_MIP_TIME_LIMIT", "per-MIP HiGHS time limit in seconds"),
                ("KLADOS_CORRIDOR_SEEDS", "primal-seed budget for randomized cherry partitions"),
                ("KLADOS_CORRIDOR_TRACE", "print per-iteration diagnostics"),
            ],
            Lower => &[("KLADOS_LOWER_TRACE", "print per-instance tier decisions")],
        }
    }

    pub fn build(&self) -> Box<dyn AnySolver> {
        match self {
            SolverChoice::ILP => from_exact(klados_exact::maf_ilp::MafIlpSolver::new()),
            SolverChoice::Sat => from_exact(klados_exact::maf_sat::MafSatSolver::new()),
            SolverChoice::MafSatOlver => {
                from_exact(klados_exact::maf_sat::MafSatOlverSolver::new())
            }
            SolverChoice::MaxhsProbe => {
                from_exact(klados_exact::maf_sat::MaxhsProbeSolver::new())
            }
            SolverChoice::ReachRefute => {
                from_exact(klados_exact::maf_sat::ReachRefuteSolver::new())
            }
            SolverChoice::ChenRspr => from_exact(klados_exact::chen_rspr::ChenRsprSolver::new()),
            SolverChoice::Whidden => from_exact(klados_exact::whidden::WhiddenSolver::new()),
            SolverChoice::BpMulti => {
                from_exact(klados_exact::maf_branch_price_multi::MafBranchPriceMultiSolver::new())
            }
            SolverChoice::Bp => from_exact(klados_exact::bp::BpSolver::new()),
            SolverChoice::OverlayExchange => {
                from_exact(klados_exact::overlay_exchange::OverlayExchangeSolver::new())
            }
            SolverChoice::RootPool => from_exact(klados_exact::root_pool::RootPoolSolver::new()),
            SolverChoice::RootCorridor => {
                from_exact(klados_exact::root_corridor::RootCorridorSolver::new())
            }
            SolverChoice::Corridor => from_exact(klados_exact::corridor::CorridorSolver::new()),
            SolverChoice::GreedyPartition => from_heuristic(
                klados_heuristic::partition::PartitionHeuristicSolver::greedy_union_add_one(),
            ),
            SolverChoice::Agglomerative => {
                from_heuristic(klados_heuristic::agglomerative::AgglomerativeSolver::new())
            }
            SolverChoice::Lagrangian => {
                from_heuristic(klados_heuristic::lagrangian::LagrangianSolver::new())
            }
            SolverChoice::Lower => Box::new(LowerWrapper),
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
        eprintln!(
            "  {:<6}  {:<30}  {}",
            choice.kind().to_string(),
            name,
            choice.description()
        );
        for (opt, desc) in choice.options() {
            eprintln!("  {:>6}    {:>28}: {}", "", opt, desc);
        }
    }
    eprintln!("{:=<72}", "");
}

// ── AnySolver trait ────────────────────────────────────────────────────────

pub trait AnySolver {
    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>>;
    fn sigterm_handler(&self);
    /// Best ready-to-emit forest so far, for the SIGTERM watcher. `None` ⇒ no
    /// live incumbent (wait for `solve` to return).
    fn snapshot(&self) -> Option<Vec<Tree>> {
        None
    }
}

struct ExactWrapper(Box<dyn klados_exact::ExactSolver>);
struct HeuristicWrapper(Box<dyn klados_heuristic::HeuristicSolver>);

impl AnySolver for ExactWrapper {
    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        self.0.solve(instance)
    }
    fn sigterm_handler(&self) {
        #[cfg(feature = "early-termination")]
        self.0.sigterm_handler()
    }
}

impl AnySolver for HeuristicWrapper {
    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        self.0.solve(instance)
    }
    fn sigterm_handler(&self) {
        self.0.sigterm_handler()
    }
    fn snapshot(&self) -> Option<Vec<Tree>> {
        self.0.snapshot()
    }
}

/// Lower-bound track racer. Not an `ExactSolver`/`HeuristicSolver`: it
/// orchestrates several engines under the `#a` threshold, so it implements
/// `AnySolver` directly.
struct LowerWrapper;

impl AnySolver for LowerWrapper {
    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        crate::lower::solve_lower(instance)
    }
    fn sigterm_handler(&self) {}
}

fn from_exact(s: impl klados_exact::ExactSolver + 'static) -> Box<dyn AnySolver> {
    Box::new(ExactWrapper(Box::new(s)))
}

fn from_heuristic(s: impl klados_heuristic::HeuristicSolver + 'static) -> Box<dyn AnySolver> {
    Box::new(HeuristicWrapper(Box::new(s)))
}

// ── Solve + print ──────────────────────────────────────────────────────────

#[cfg(feature = "early-termination")]
static SOLVER_DATA: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

#[cfg(feature = "early-termination")]
static SOLVER_VTABLE: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

#[cfg(feature = "early-termination")]
fn request_graceful_shutdown() {
    let data = SOLVER_DATA.load(Ordering::SeqCst);
    let vtable = SOLVER_VTABLE.load(Ordering::SeqCst);
    if !data.is_null() && !vtable.is_null() {
        let solver: &dyn AnySolver = unsafe { std::mem::transmute((data, vtable)) };
        solver.sigterm_handler();
    }
}

/// Set true by whichever path writes the solution first (the normal return, or
/// the SIGTERM watcher) so it is emitted exactly once.
#[cfg(feature = "early-termination")]
static EMITTED: AtomicBool = AtomicBool::new(false);

/// Guaranteed-fast response after SIGTERM. The solver is given this long to
/// return its own (best) result and emit it; if it is still unwinding past it
/// (the decomposition recursion can take seconds), the watcher emits the live
/// incumbent and force-exits. Well inside the challenge's 10 s SIGTERM→SIGKILL
/// grace, so a slow unwind can never cost the whole instance (a score-0
/// timeout).
#[cfg(feature = "early-termination")]
const WATCHDOG_EMIT_DELAY: std::time::Duration = std::time::Duration::from_secs(2);

#[cfg(feature = "early-termination")]
fn emit_forest(forest: &[Tree]) {
    let mut stdout = io::stdout().lock();
    for tree in forest {
        let _ = tree.cursor().write_newick(&mut stdout);
        let _ = writeln!(stdout);
    }
    let _ = stdout.flush();
}

/// Background watcher: once SIGTERM/SIGINT fires, give the solver a brief window
/// to return on its own; if it hasn't emitted by then, emit its live incumbent
/// and force-exit so the harness always gets a forest within the grace window.
#[cfg(feature = "early-termination")]
fn spawn_emit_watchdog(terminated: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        while !terminated.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        std::thread::sleep(WATCHDOG_EMIT_DELAY);
        // Only act for solvers that expose a live incumbent (anytime heuristic).
        // Exact solvers return None — then the watchdog must NOT exit, or it
        // would kill the process before the exact solve emits its result.
        let data = SOLVER_DATA.load(Ordering::SeqCst);
        let vtable = SOLVER_VTABLE.load(Ordering::SeqCst);
        if data.is_null() || vtable.is_null() {
            return;
        }
        let solver: &dyn AnySolver = unsafe { std::mem::transmute((data, vtable)) };
        let Some(forest) = solver.snapshot() else {
            return;
        };
        // We have an incumbent: emit it (unless the normal path beat us) and
        // force-exit so a still-unwinding `solve` can't blow the grace window.
        if !EMITTED.swap(true, Ordering::SeqCst) {
            emit_forest(&forest);
        }
        std::process::exit(0);
    });
}

pub fn solve_and_print(
    instance: &Instance,
    choice: SolverChoice,
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!(
        "solving: {} trees, {} leaves, solver={}",
        instance.num_trees(),
        instance.num_leaves,
        choice.to_possible_value().unwrap().get_name(),
    );

    let mut solver = choice.build();

    #[cfg(feature = "early-termination")]
    let terminated = {
        let solver_ref: &dyn AnySolver = &*solver;
        let fat_ptr: (*const (), *const ()) = unsafe { std::mem::transmute(solver_ref) };
        SOLVER_DATA.store(fat_ptr.0 as *mut (), Ordering::SeqCst);
        SOLVER_VTABLE.store(fat_ptr.1 as *mut (), Ordering::SeqCst);

        let terminated = Arc::new(AtomicBool::new(false));
        let t_sigterm = Arc::clone(&terminated);
        let t_sigint = Arc::clone(&terminated);
        let sigterm_id = unsafe {
            signal_hook::low_level::register(signal_hook::consts::SIGTERM, move || {
                t_sigterm.store(true, Ordering::SeqCst);
                request_graceful_shutdown();
            })?
        };
        let sigint_id = unsafe {
            signal_hook::low_level::register(signal_hook::consts::SIGINT, move || {
                t_sigint.store(true, Ordering::SeqCst);
                request_graceful_shutdown();
            })?
        };
        let _ = sigterm_id;
        let _ = sigint_id;
        spawn_emit_watchdog(Arc::clone(&terminated));
        terminated
    };

    // `None` means "no solution to emit". For the lower-bound track that is the
    // safe response when no forest can be certified within the `#a` bound:
    // emitting nothing scores 0 on the instance but, unlike an oversized forest,
    // never risks disqualification. An empty forest emits zero newick lines.
    let components = match solver.solve(instance) {
        Some(c) => c,
        None => {
            log::info!("solver returned no solution — emitting nothing");
            Vec::new()
        }
    };

    #[cfg(feature = "early-termination")]
    {
        let signalled = terminated.load(Ordering::SeqCst);
        if signalled {
            log::info!(
                "terminated by signal: {} components (best-effort)",
                components.len()
            );
        } else {
            log::info!("solution: {} components", components.len());
        }
        // Emit exactly once. The watcher may already have written the incumbent
        // (if `solve` was still unwinding past the grace window); claim the flag
        // before clearing the solver pointer so a watcher that wakes meanwhile
        // can still read a valid snapshot if it wins the race.
        let we_emit = !EMITTED.swap(true, Ordering::SeqCst);
        SOLVER_DATA.store(std::ptr::null_mut(), Ordering::SeqCst);
        SOLVER_VTABLE.store(std::ptr::null_mut(), Ordering::SeqCst);
        if !we_emit {
            return Ok(()); // the watcher already wrote a forest
        }
    }
    #[cfg(not(feature = "early-termination"))]
    log::info!("solution: {} components", components.len());

    let mut stdout = io::stdout().lock();
    for tree in &components {
        tree.cursor().write_newick(&mut stdout)?;
        writeln!(stdout)?;
    }

    Ok(())
}
