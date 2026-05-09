use klados_core::Instance;
use pace26io::newick::NewickWriter;
use std::io::{self, Write};

fn default_solver() -> &'static str {
    option_env!("KLADOS_EXACT_SOLVER").unwrap_or("bp-multi")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .target(env_logger::Target::Stderr)
        .init();

    let solver_name = default_solver();
    let instance = Instance::from_stdin()?;
    log::info!(
        "{} trees, {} leaves, solver={}",
        instance.num_trees(),
        instance.num_leaves,
        solver_name
    );

    let mut solver = klados_exact::solver_by_name(solver_name)
        .unwrap_or_else(|| panic!("unknown solver: {}", solver_name));

    let components = solver.solve(&instance).expect("failed to find solution");

    let mut stdout = io::stdout().lock();
    for tree in &components {
        tree.cursor().write_newick(&mut stdout)?;
        writeln!(stdout)?;
    }

    Ok(())
}
