//! Exact solver subcommand.

use std::io::{self, Write};

use klados_core::Instance;
use pace26io::newick::NewickWriter;

pub fn run(
    instance: &Instance,
    approach: &str,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut solver = klados_exact::solver_by_name(approach)
        .unwrap_or_else(|| panic!("Unknown approach: {}", approach));
    let components = solver.solve(instance).expect("Failed to find solution");

    if verbose {
        eprintln!("Solution: {} components", components.len());
    }

    let mut stdout = io::stdout().lock();
    for tree in &components {
        tree.cursor().write_newick(&mut stdout)?;
        writeln!(stdout)?;
    }

    Ok(())
}
