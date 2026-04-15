//! klados-heuristic: Heuristic FPT solvers for Maximum Agreement Forest

pub mod agglomerative;
pub mod max_sat;
pub mod partition;

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use klados_core::{Instance, SolverStats, Tree};

use crate::agglomerative::AgglomerativeSolver;
use crate::max_sat::MaxSatSolver;
use crate::partition::PartitionHeuristicSolver;

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

const ACTIVE_NONE: u8 = 0;
const ACTIVE_GREEDY: u8 = 1;

pub struct AutoHeuristicSolver {
    active: Arc<AtomicU8>,
    greedy: PartitionHeuristicSolver,
    stats: SolverStats,
}

impl AutoHeuristicSolver {
    pub fn new() -> Self {
        Self {
            active: Arc::new(AtomicU8::new(ACTIVE_NONE)),
            greedy: PartitionHeuristicSolver::greedy_union_add_one(),
            stats: SolverStats::default(),
        }
    }
}

impl HeuristicSolver for AutoHeuristicSolver {
    fn name(&self) -> &'static str {
        "auto"
    }

    fn solve(&mut self, instance: &Instance) -> Option<Vec<Tree>> {
        self.active.store(ACTIVE_GREEDY, Ordering::SeqCst);
        eprintln!("[heur:auto] using unified heuristic path");
        eprintln!("[heur:auto] mode={}", self.greedy.name());
        let result = self.greedy.solve(instance);
        self.stats = self.greedy.stats().clone();
        self.active.store(ACTIVE_NONE, Ordering::SeqCst);
        result
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }

    fn sigterm_handler(&self) {
        match self.active.load(Ordering::SeqCst) {
            ACTIVE_GREEDY => self.greedy.sigterm_handler(),
            _ => {}
        }
    }
}

/// Run all available heuristc solvers.
pub fn available_solvers() -> Vec<Box<dyn HeuristicSolver>> {
    vec![
        Box::new(AutoHeuristicSolver::new()),
        Box::new(PartitionHeuristicSolver::greedy_union_add_one()),
        Box::new(AgglomerativeSolver::new()),
    ]
}

pub fn solver_by_name(name: &str) -> Option<Box<dyn HeuristicSolver>> {
    match name {
        "auto" => Some(Box::new(AutoHeuristicSolver::new())),
        "greedy-partition-union-addone" => {
            Some(Box::new(PartitionHeuristicSolver::greedy_union_add_one()))
        }
        "agglomerative" => Some(Box::new(AgglomerativeSolver::new())),
        // Keep the legacy path addressable explicitly without advertising it.
        "maxsat" => Some(Box::new(MaxSatSolver::new())),
        _ => None,
    }
}
