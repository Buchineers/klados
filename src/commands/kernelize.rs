//! Kernelize subcommand — apply reduction rules and report statistics.

use klados_core::Instance;
use klados_exact::kernelize::{self, KernelizeConfig};
use pace26io::newick::NewickWriter;
use std::io::{self, Write};

pub fn run(
    instance: &Instance,
    subtree: bool,
    chain: bool,
    chain32: bool,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = KernelizeConfig {
        subtree,
        chain,
        chain32,
        protected_labels: Vec::new(),
    };

    let result = kernelize::kernelize(instance, &config);

    // Always print summary stats to stderr
    kernelize::print_stats(&result.stats);

    if verbose {
        kernelize::print_taxa_detail(&result.stats);
    }

    // Print reduced trees to stdout (Newick), followed by param reduction
    let mut stdout = io::stdout().lock();
    for tree in &result.instance.trees {
        tree.cursor().write_newick(&mut stdout)?;
        writeln!(stdout)?;
    }
    writeln!(stdout, "{}", result.param_reduction)?;

    Ok(())
}
