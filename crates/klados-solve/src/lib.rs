//! klados-solve: merged exact + heuristic solvers for Maximum Agreement Forest.
//!
//! The unified [`Solver`] trait + [`run`] harness are the public surface. Each
//! solver lives in [`solvers`] and implements [`Solver`] directly; decomposition
//! primitives that are not solvers live in [`decomp`].

pub use klados_core::cluster_reduction;
// kernelize and lower_bound live in klados-core; re-export for convenience.
pub use klados_core::kernelize;
pub use klados_core::lower_bound;

// ── the solvers (each implements `Solver`, enumerated by `catalog()`) ────────
pub mod solvers;

// ── decomposition primitives (not solvers) ──────────────────────────────────
pub mod decomp;

// ── solver catalog (name -> description -> in-process entry point) ───────────
mod catalog;
pub use catalog::{SolverInfo, catalog};

pub use crate::solvers::partition::{GapExperimentResult, run_packing_gap_experiment};

use klados_core::{Instance, SolverStats, Tree};

// ── Unified solver abstraction ──────────────────────────────────────────────
mod config;
mod run;
pub use config::{RunConfig, Track};
pub use run::run;

/// The unified solver trait. Every solver implements this; the per-solver
/// `main()` runs it via [`run`].
pub trait Solver {
    /// Solver-specific knobs.
    type Config: Default;

    /// Tracks this solver can run. `run()` refuses any other track (e.g.
    /// `agglomerative` is heuristic-only; `max_sat` can't do the lower-bound
    /// track since it can't tell open-wbo the `#a` budget).
    const SUPPORTED_TRACKS: &'static [Track];

    /// Polls its own stop flag + the `cfg.budget` deadline; returns its best
    /// forest per `cfg.track` (exact returns one only if proven optimal).
    fn solve(&mut self, inst: &Instance, cfg: &RunConfig<Self::Config>) -> Option<Vec<Tree>>;

    /// Statistics gathered during the last `solve`.
    fn stats(&self) -> &SolverStats;

    /// Optional async-signal-safe SIGTERM action, **track-aware**: exact may
    /// ignore SIGTERM (a partial result isn't proven optimal), while heuristic
    /// flips a stop flag or kills a child to emit its best. Default: none.
    fn sigterm_handler(&self, track: Track) -> Option<Box<dyn Fn() + Send + Sync>> {
        let _ = track;
        None
    }
}

