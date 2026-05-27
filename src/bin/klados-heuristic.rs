use clap::ValueEnum;
use klados::solver::{solve_and_print, SolverChoice};
use klados_core::Instance;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .target(env_logger::Target::Stderr)
        .init();

    let name = option_env!("KLADOS_HEURISTIC_SOLVER").unwrap_or("greedy-partition-union-addone");
    let choice = SolverChoice::from_str(name, true)
        .unwrap_or_else(|_| panic!("unknown solver: {}", name));
    let instance = Instance::from_stdin()?;
    solve_and_print(&instance, choice)
}
