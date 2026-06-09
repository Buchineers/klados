use clap::ValueEnum;
use klados::solver::{solve_and_print, SolverChoice};
use klados_core::Instance;

// Lower-bound track: emit an agreement forest of size k <= floor(a*k*) + b (the
// `#a` line), as fast as possible — NOT a bound value. The `lower` racer reads
// the approximation parameters off the instance, races Chen/Lagrangian to a
// certified-within-bound forest, and emits nothing when it cannot certify one
// (safe: scored 0, never disqualified). See klados/src/lower.rs.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .target(env_logger::Target::Stderr)
        .init();

    let name = option_env!("KLADOS_LB_SOLVER").unwrap_or("lower");
    let choice = SolverChoice::from_str(name, true)
        .unwrap_or_else(|_| panic!("unknown solver: {}", name));
    let instance = Instance::from_stdin()?;
    solve_and_print(&instance, choice)
}
