//! General (solver-independent) run configuration.

use std::time::Duration;

/// Competition track. Governs what a solver is allowed to emit on early stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Track {
    /// Emit only a provably-optimal forest.
    #[default]
    Exact,
    /// Emit the best forest found so far (anytime).
    Heuristic,
    /// Emit a forest only if proven within the `#a` bound.
    LowerBound,
}

/// Configuration passed to [`crate::Solver::solve`]: general fields plus a
/// solver-specific `specific` payload. Constructed in each solver's `main()`;
/// `run()` overrides `budget` from `STRIDE_TIMEOUT` at startup.
#[derive(Default)]
pub struct RunConfig<C> {
    pub track: Track,
    /// Wall-time budget. `None` means "until SIGTERM".
    pub budget: Option<Duration>,
    /// Solver-specific knobs.
    pub specific: C,
}
