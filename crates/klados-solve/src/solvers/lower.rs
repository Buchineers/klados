//! Lower-bound track racer.
//!
//! The Lower-bound track (PACE 2026) ships each 2-tree instance with a `#a a b`
//! line: a forest of size `k` scores iff `k <= floor(a * k*) + b`, where `k*` is
//! the unknown optimum. Points are purely speed-based, so the goal is the
//! *fastest valid* forest, not the optimum.
//!
//! CRITICAL — disqualification: emitting an infeasible forest, OR a forest whose
//! size exceeds the threshold, on ANY instance disqualifies the WHOLE solver. So
//! we emit a forest only when we can *prove* it clears the bound — i.e.
//! `k <= floor(a * LB) + b` for a sound lower bound `LB <= k*` (then
//! `floor(a*LB)+b <= floor(a*k*)+b`, so the forest is genuinely within bound).
//! When nothing can be certified we emit NOTHING: that scores 0 on the instance
//! (a "did not finish"), which is safe — never a disqualification.
//!
//! Pipeline:
//!   1. Lagrangian (our best heuristic), made track-aware: it seeds from Chen's
//!      2-approx (returning it immediately if that already clears the bound),
//!      drives the incumbent down, raises a tight dual LB, and early-aborts the
//!      instant its forest clears the bound against that LB (speed bonus). We
//!      emit its forest iff it is certified by its own (sound) dual LB.
//!   2. Exact `bp`, deadline-bounded, only for small `n`. Its optimum `k*`
//!      trivially clears the bound, so it is always emittable.
//!   3. Otherwise: emit nothing.

use klados_core::af_validator::validate_agreement_forest;
use klados_core::{Instance, Tree};
use log::debug;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Above this leaf count, exact `bp` cannot finish inside the per-instance
/// budget, so it is skipped and the whole budget goes to Lagrangian.
const BP_MAX_LEAVES: u32 = 800;

/// `true` iff `forest` is a valid agreement forest AND provably within the track
/// bound: `|forest| <= floor(a * lb) + b` with `lb` a SOUND lower bound on `k*`.
fn certifiable(instance: &Instance, forest: &[Tree], a: f64, b: usize, lb: usize) -> bool {
    forest.len() <= (a * lb as f64).floor() as usize + b
        && validate_agreement_forest(instance, forest).is_ok()
}

/// Solve under the Lower-bound track. Returns a forest only when it is provably
/// within the `#a` bound; returns `None` (emit nothing) otherwise. `approx` is
/// absent only off-track, where we demand the exact optimum (`a = 1, b = 0`).
///
/// `budget` is the wall limit (`cfg.budget` ← `STRIDE_TIMEOUT`); the racer
/// subdivides it (lagrangian then exact `bp`) and leaves a 3 s flush margin.
pub fn solve_lower(instance: &Instance, budget: Duration) -> Option<Vec<Tree>> {
    let (a, b) = instance.approx.unwrap_or((1.0, 0));
    let start = Instant::now();
    let wall_secs = budget.as_secs_f64();
    let deadline = start + Duration::from_secs_f64((wall_secs - 3.0).max(1.0));

    // ── Lagrangian (track-aware): seeds Chen, raises a tight dual LB, and
    //    early-aborts the instant its forest clears the bound. ──
    {
        let mut lagr = crate::solvers::lagrangian::LagrangianSolver::new();
        lagr.set_approx_target(a, b);
        // Reserve a slice for exact bp only when bp can actually run.
        let bp_reserve = if instance.num_leaves <= BP_MAX_LEAVES {
            deadline
                .saturating_duration_since(Instant::now())
                .mul_f64(0.3)
        } else {
            Duration::ZERO
        };
        let budget = deadline
            .saturating_duration_since(Instant::now())
            .saturating_sub(bp_reserve)
            .max(Duration::from_secs(1));
        lagr.set_budget(budget);

        let forest = lagr.solve(instance);
        // The dual LB lagrangian exposes is sound (LP weak duality); it is the
        // tightest bound we have, so it certifies the most forests.
        let lb = crate::Solver::stats(&lagr).lower_bound;
        if let Some(forest) = forest {
            let ok = certifiable(instance, &forest, a, b, lb);
            debug!(
                "[lower] lagrangian: k={} lb={lb} T={} certified={ok}",
                forest.len(),
                (a * lb as f64).floor() as usize + b,
            );
            if ok {
                return Some(forest);
            }
        }
    }

    // ── Exact bp (small n, deadline-bounded). The optimum always clears T. ──
    if instance.num_leaves <= BP_MAX_LEAVES && Instant::now() < deadline {
        let terminate = Arc::new(AtomicBool::new(false));
        let tw = Arc::clone(&terminate);
        let remaining = deadline.saturating_duration_since(Instant::now());
        std::thread::spawn(move || {
            std::thread::sleep(remaining);
            tw.store(true, Ordering::Relaxed);
        });
        debug!(
            "[lower] exact bp (n={}, {:.0}s left)",
            instance.num_leaves,
            remaining.as_secs_f64()
        );
        if let Some(forest) = crate::solvers::bp::bp_solve_capped(instance, &terminate) {
            // bp returns the proven optimum; validate defensively before trusting.
            if validate_agreement_forest(instance, &forest).is_ok() {
                return Some(forest);
            }
        }
    }

    // ── Nothing certifiable: emit NOTHING (0 points, never a disqualification). ──
    debug!("[lower] no certifiable forest within bound -> emit nothing");
    None
}

// ── Unified Solver wrapper + entry point ────────────────────────────────────
use crate::{RunConfig, Solver, Track};
use klados_core::SolverStats;

#[derive(Default)]
pub struct LowerSolver {
    stats: SolverStats,
}

impl LowerSolver {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Solver for LowerSolver {
    type Config = ();
    const SUPPORTED_TRACKS: &'static [Track] = &[Track::LowerBound];
    fn solve(&mut self, inst: &Instance, cfg: &RunConfig<()>) -> Option<Vec<Tree>> {
        solve_lower(inst, cfg.budget.unwrap_or(Duration::from_secs(600)))
    }
    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

pub fn main() {
    crate::run(
        LowerSolver::new(),
        RunConfig {
            track: Track::LowerBound,
            ..Default::default()
        },
    );
}
