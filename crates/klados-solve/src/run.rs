//! The shared binary harness: stdin → solve → stdout, plus SIGTERM wiring.
//!
//! Every per-solver `main()` is `run(SomeSolver::new(), cfg)`. The dev CLI's
//! `klados solve <name>` calls the same `main()` in-process.

use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

use klados_core::Instance;
use pace26io::newick::NewickWriter;

use crate::{RunConfig, Solver};

/// Run a solver end-to-end: parse the instance from stdin, install the solver's
/// SIGTERM action, solve, and write the resulting forest as newick to stdout.
///
/// `budget` is taken from the `STRIDE_TIMEOUT` environment variable when set
/// (the only environment read in the codebase); otherwise the in-config default
/// (set by the calling binary) applies.
pub fn run<S: Solver>(mut solver: S, mut cfg: RunConfig<S::Config>) {
    let _ = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn"),
    )
    .try_init();

    if !S::SUPPORTED_TRACKS.contains(&cfg.track) {
        log::error!("solver does not support the {:?} track", cfg.track);
        return;
    }

    if let Ok(t) = std::env::var("STRIDE_TIMEOUT") {
        // Guard `from_secs_f64`, which panics on negative / NaN / inf / overflow.
        if let Some(secs) = t.trim().parse::<f64>().ok().filter(|s| s.is_finite() && *s > 0.0) {
            cfg.budget = Some(Duration::from_secs_f64(secs));
        }
    }

    if let Some(action) = solver.sigterm_handler(cfg.track) {
        let action: Arc<dyn Fn() + Send + Sync> = Arc::from(action);
        for sig in [
            signal_hook::consts::SIGTERM,
            signal_hook::consts::SIGINT,
        ] {
            let a = action.clone();
            // SAFETY: the registered closure only performs async-signal-safe
            // work (an atomic store, or `libc::kill`); it allocates nothing,
            // takes no locks, and never panics.
            unsafe {
                let _ = signal_hook::low_level::register(sig, move || a());
            }
        }
    }

    let inst = match Instance::from_stdin() {
        Ok(i) => i,
        Err(e) => {
            log::error!("failed to parse instance: {e}");
            return;
        }
    };

    if let Some(forest) = solver.solve(&inst, &cfg) {
        let mut out = std::io::stdout().lock();
        for tree in &forest {
            let _ = tree.cursor().write_newick(&mut out);
            let _ = writeln!(out);
        }
        let _ = out.flush();
    }
}
