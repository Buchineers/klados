use clap::ValueEnum;
use klados::solver::{SolverChoice, solve_and_print};
use klados_core::Instance;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .target(env_logger::Target::Stderr)
        .init();

    let name = option_env!("KLADOS_EXACT_SOLVER").unwrap_or("bp");
    let choice =
        SolverChoice::from_str(name, true).unwrap_or_else(|_| panic!("unknown solver: {}", name));
    let instance = Instance::from_stdin()?;
    solve_and_print(&instance, choice)
}
