//! Feature-based solver dispatch (portfolio-by-selection).
//!
//! [`Dispatch`] reads cheap instance features (`m`, `n`) and routes each
//! instance to **one** inner solver that runs end-to-end — not a hybrid, there
//! is no cooperation between engines. No single engine wins the exact track
//! everywhere: `sat` proves the high-`m`/small-`n` band bp's loose LP can't,
//! but its `O(n²)` + `O(n³)` encoding explodes as `n` grows, while `bp` owns
//! large-`n` and `m=2`. Dispatch sends each instance to whichever engine owns
//! its region.
//!
//! ## Swapping / adding solvers
//!
//! The routing table lives in one place: [`default_routes`]. Each [`Route`] is
//! `{ label, when, build }` — a predicate over [`Features`] and a factory for a
//! boxed solver. Routes are tried in order, first match wins, and the last
//! route is the catch-all fallback. Any [`Solver`] works as-is (the blanket
//! [`ErasedSolver`] impl runs it with `Config::default()`). To retune, edit
//! [`default_routes`] and nothing else.

use std::time::Duration;

use klados_core::{Instance, SolverStats, Tree};

use crate::solvers::bp::BpSolver;
use crate::solvers::maf_sat::MafSatSolver;
use crate::{RunConfig, Solver, Track};

const LOG_TARGET: &str = "klados::dispatch";

/// Cheap, solver-independent instance features used for routing. Extend this
/// (and [`Features::of`]) with any signal a route needs — it's the only place
/// routing predicates read from.
#[derive(Clone, Copy, Debug)]
pub struct Features {
    /// Number of input trees.
    pub m: usize,
    /// Number of leaves.
    pub n: usize,
}

impl Features {
    pub fn of(inst: &Instance) -> Self {
        Features {
            m: inst.num_trees(),
            n: inst.num_leaves as usize,
        }
    }
}

/// Object-safe, config-erased solver handle. Lets heterogeneous solvers (each
/// with its own [`Solver::Config`]) live in one routing table behind
/// `Box<dyn ErasedSolver>`. Implemented for every [`Solver`] via a blanket impl
/// that runs it with `Config::default()`; routes needing non-default knobs can
/// box a hand-constructed solver instead.
pub trait ErasedSolver {
    fn solve(
        &mut self,
        inst: &Instance,
        track: Track,
        budget: Option<Duration>,
    ) -> Option<Vec<Tree>>;
    fn stats(&self) -> &SolverStats;
    fn supported_tracks(&self) -> &'static [Track];
    fn sigterm_handler(&self, track: Track) -> Option<Box<dyn Fn() + Send + Sync>>;
}

impl<S: Solver> ErasedSolver for S {
    fn solve(
        &mut self,
        inst: &Instance,
        track: Track,
        budget: Option<Duration>,
    ) -> Option<Vec<Tree>> {
        let cfg = RunConfig {
            track,
            budget,
            specific: <S::Config as Default>::default(),
        };
        <S as Solver>::solve(self, inst, &cfg)
    }
    fn stats(&self) -> &SolverStats {
        <S as Solver>::stats(self)
    }
    fn supported_tracks(&self) -> &'static [Track] {
        S::SUPPORTED_TRACKS
    }
    fn sigterm_handler(&self, track: Track) -> Option<Box<dyn Fn() + Send + Sync>> {
        <S as Solver>::sigterm_handler(self, track)
    }
}

/// Factory that constructs a fresh boxed solver on demand (so only the engine a
/// route actually selects is ever built).
type Factory = Box<dyn Fn() -> Box<dyn ErasedSolver>>;

/// One routing rule: a human label, a predicate over [`Features`], and a
/// factory for the solver to run when it matches.
pub struct Route {
    pub label: &'static str,
    pub when: Box<dyn Fn(Features) -> bool>,
    pub build: Factory,
}

impl Route {
    pub fn new(
        label: &'static str,
        when: impl Fn(Features) -> bool + 'static,
        build: impl Fn() -> Box<dyn ErasedSolver> + 'static,
    ) -> Self {
        Route {
            label,
            when: Box::new(when),
            build: Box::new(build),
        }
    }
}

/// The single edit point for routing policy: an ordered [`Route`] list whose
/// last entry is the catch-all fallback (first match wins).
///
/// Calibrated 2026-06-23 on exact_pub instances in n∈[80,160]: bp and sat are
/// complementary (7 unique wins each) — sat-only wins cluster at n≤112, bp-only
/// at n≥115. A single n-cutoff captures 22 of the 23 "run both" wins, so
/// dispatch (pick-one), not a time-split portfolio, is the right tool. The
/// cutoff is set conservatively to 110 (n>110 → bp, n≤110 → sat); m=2 → bp.
pub fn default_routes() -> Vec<Route> {
    const N_SAT_LIMIT: usize = 110;
    vec![
        // m=2: bp's specialized 2-tree path dominates; sat errored on large
        // 2-tree instances (m=2/n=891 SolverError).
        Route::new("m2-bp", |f| f.m == 2, || Box::new(BpSolver::new())),
        // n>110: bp wins the exact_pub bp-only band and keeps working on
        // large/very-high-m instances where sat's encoding explodes.
        Route::new(
            "large-n-bp",
            |f| f.n > N_SAT_LIMIT,
            || Box::new(BpSolver::new()),
        ),
        // Fallback (n≤110): sat's CEGAR closes bp's LP-gap on the high-m
        // instances bp cannot prove (sat-only wins cluster at n≤112).
        Route::new("sat", |_| true, || Box::new(MafSatSolver::new())),
    ]
}

/// Feature-routing meta-solver. Picks one inner solver per instance from a
/// [`Route`] table and delegates to it. Itself a [`Solver`], so it plugs into
/// the catalog, the `run()` harness, and the track machinery unchanged.
pub struct Dispatch {
    routes: Vec<Route>,
    /// The inner solver actually run (kept so `stats()` can forward to it).
    inner: Option<Box<dyn ErasedSolver>>,
    empty_stats: SolverStats,
}

impl Default for Dispatch {
    fn default() -> Self {
        Self::new()
    }
}

impl Dispatch {
    pub fn new() -> Self {
        Self::with_routes(default_routes())
    }

    /// Construct with a custom routing table (tests / alternate policies). The
    /// last route should be a catch-all (`when` = `|_| true`).
    pub fn with_routes(routes: Vec<Route>) -> Self {
        Dispatch {
            routes,
            inner: None,
            empty_stats: SolverStats::default(),
        }
    }

    /// First route matching `f` whose solver supports `track`; if none does,
    /// the last route unconditionally. Returns the built solver and its label.
    fn select(&self, f: Features, track: Track) -> (&'static str, Box<dyn ErasedSolver>) {
        for route in &self.routes {
            if (route.when)(f) {
                let built = (route.build)();
                if built.supported_tracks().contains(&track) {
                    return (route.label, built);
                }
            }
        }
        let last = self.routes.last().expect("default_routes is non-empty");
        (last.label, (last.build)())
    }
}

impl Solver for Dispatch {
    type Config = ();
    // Union of what routed solvers can do; the per-track check happens in
    // `select` against the chosen inner solver.
    const SUPPORTED_TRACKS: &'static [Track] = &[Track::Exact, Track::Heuristic, Track::LowerBound];

    fn solve(&mut self, inst: &Instance, cfg: &RunConfig<()>) -> Option<Vec<Tree>> {
        let f = Features::of(inst);
        let (label, mut inner) = self.select(f, cfg.track);
        log::info!(target: LOG_TARGET, "m={} n={} {:?} -> '{}'", f.m, f.n, cfg.track, label);

        // Install the chosen solver's SIGTERM action now that the engine is
        // known — our own `sigterm_handler` returns None so `run()` deferred
        // this to us. No-op on the exact track (every solver ignores SIGTERM
        // there; the harness SIGKILLs at the deadline), relevant only to
        // anytime tracks.
        install_sigterm(inner.sigterm_handler(cfg.track));

        let out = inner.solve(inst, cfg.track, cfg.budget);
        self.inner = Some(inner);
        out
    }

    fn stats(&self) -> &SolverStats {
        match &self.inner {
            Some(s) => s.stats(),
            None => &self.empty_stats,
        }
    }

    // Intentionally None: the real handler is installed in `solve` once the
    // routed solver is known (see the comment there).
    fn sigterm_handler(&self, _track: Track) -> Option<Box<dyn Fn() + Send + Sync>> {
        None
    }
}

/// Register `action` for SIGTERM/SIGINT, mirroring the `run()` harness. No-op
/// when `action` is None.
fn install_sigterm(action: Option<Box<dyn Fn() + Send + Sync>>) {
    let Some(action) = action else { return };
    let action: std::sync::Arc<dyn Fn() + Send + Sync> = std::sync::Arc::from(action);
    for sig in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
        let a = action.clone();
        // SAFETY: same contract as `run()` — the closure does only
        // async-signal-safe work (atomic store / libc::kill).
        unsafe {
            let _ = signal_hook::low_level::register(sig, move || a());
        }
    }
}

pub fn main() {
    crate::run(
        Dispatch::new(),
        RunConfig {
            track: Track::Exact,
            ..Default::default()
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feats(m: usize, n: usize) -> Features {
        Features { m, n }
    }

    // A trivial solver to observe which route fires without running real engines.
    struct Tag(SolverStats);
    impl ErasedSolver for Tag {
        fn solve(&mut self, _: &Instance, _: Track, _: Option<Duration>) -> Option<Vec<Tree>> {
            None
        }
        fn stats(&self) -> &SolverStats {
            &self.0
        }
        fn supported_tracks(&self) -> &'static [Track] {
            &[Track::Exact]
        }
        fn sigterm_handler(&self, _: Track) -> Option<Box<dyn Fn() + Send + Sync>> {
            None
        }
    }

    fn tag_dispatch() -> Dispatch {
        fn tag() -> Box<dyn ErasedSolver> {
            Box::new(Tag(SolverStats::default()))
        }
        Dispatch::with_routes(vec![
            Route::new("m2", |f| f.m == 2, tag),
            Route::new("largen", |f| f.n > 150, tag),
            Route::new("fallback", |_| true, tag),
        ])
    }

    #[test]
    fn routes_first_match_wins() {
        let d = tag_dispatch();
        // m=2 takes priority even when n is also large.
        assert_eq!(d.select(feats(2, 999), Track::Exact).0, "m2");
        assert_eq!(d.select(feats(10, 200), Track::Exact).0, "largen");
        assert_eq!(d.select(feats(17, 75), Track::Exact).0, "fallback");
    }

    #[test]
    fn unsupported_track_falls_through() {
        let d = tag_dispatch();
        // Tag only supports Exact; a Heuristic request skips matching routes and
        // lands on the last route unconditionally.
        assert_eq!(d.select(feats(2, 10), Track::Heuristic).0, "fallback");
    }
}
