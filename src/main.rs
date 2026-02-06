//! Klados - PACE 2026 Maximum Agreement Forest Solver
//!
//! κλάδος (klados) - Ancient Greek for "branch"

use clap::{Parser, Subcommand};
use klados_core::{Instance, Tree};
use klados_exact::solver_by_name;
use pace26io::binary_tree::IndexedBinTreeBuilder;
use pace26io::newick::NewickWriter;
use pace26io::pace::simplified::Instance as PaceInstance;
use std::io::{self, BufReader, Write};

#[derive(Parser)]
#[command(name = "klados")]
#[command(author, version, about = "PACE 2026 Maximum Agreement Forest solver")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run exact solver (default)
    Exact {
        /// Exact approach name (e.g., fpt)
        #[arg(long, default_value = "whidden")]
        approach: String,
    },
    /// Show instance info
    Info,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Read instance from stdin using pace26io directly
    let stdin = io::stdin();
    let reader = BufReader::new(stdin.lock());
    let mut builder = IndexedBinTreeBuilder::default();
    let pace = PaceInstance::try_read(reader, &mut builder)?;

    let num_leaves = pace.num_leaves as u32;
    let trees: Vec<Tree> = pace
        .trees
        .iter()
        .map(|t| Tree::from_cursor(t.top_down(), num_leaves))
        .collect();
    let instance = Instance::new(trees, num_leaves);

    if cli.verbose {
        eprintln!(
            "Instance: {} trees, {} leaves",
            instance.num_trees(),
            instance.num_leaves
        );
    }

    match cli.command {
        Some(Commands::Info) => {
            println!("Trees: {}", instance.num_trees());
            println!("Leaves: {}", instance.num_leaves);
            for (i, tree) in instance.trees.iter().enumerate() {
                println!("Tree {}: {} nodes", i + 1, tree.num_nodes());
            }
        }
        Some(Commands::Exact { approach }) => {
            // Run exact solver
            let mut solver = solver_by_name(&approach)
                .unwrap_or_else(|| panic!("Unknown approach: {}", approach));
            let components = solver
                .solve(&instance)
                .expect("Failed to find solution");

            if cli.verbose {
                eprintln!("Solution: {} components", components.len());
            }

            // Write solution to stdout (Newick trees, one per line, no header)
            let mut stdout = io::stdout().lock();
            for tree in &components {
                tree.cursor().write_newick(&mut stdout)?;
                writeln!(stdout)?;
            }
        }
        None => {
            let mut solver = solver_by_name("whidden").expect("Missing default solver");
            let components = solver
                .solve(&instance)
                .expect("Failed to find solution");

            if cli.verbose {
                eprintln!("Solution: {} components", components.len());
            }

            let mut stdout = io::stdout().lock();
            for tree in &components {
                tree.cursor().write_newick(&mut stdout)?;
                writeln!(stdout)?;
            }
        }
    }

    Ok(())
}
