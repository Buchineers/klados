//! Klados - PACE 2026 Maximum Agreement Forest Solver
//!
//! κλάδος (klados) - Ancient Greek for "branch"

mod commands;

use clap::{Parser, Subcommand};
use klados_core::{Instance, Tree};
use pace26io::binary_tree::IndexedBinTreeBuilder;
use pace26io::pace::simplified::Instance as PaceInstance;
use std::io::{self, BufReader};

#[derive(Parser)]
#[command(name = "klados")]
#[command(author, version, about = "PACE 2026 Maximum Agreement Forest solver")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    Exact {
        #[arg(long, default_value = "shi-mestel")]
        approach: String,
    },
    Heuristic {
        #[arg(long)]
        solver: String,
    },
    Info,
    Bounds,
    ValidateBounds {
        #[arg(value_name = "FILE")]
        scores_file: std::path::PathBuf,
    },
    CheckBounds {
        #[arg(value_name = "FILE")]
        list_file: std::path::PathBuf,
    },
    CompareBounds {
        #[arg(value_name = "FILE")]
        scores_file: std::path::PathBuf,
        #[arg(long, default_value = "50")]
        max_leaves: u32,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::ValidateBounds { scores_file }) => {
            commands::validate::run(&scores_file)?;
            return Ok(());
        }
        Some(Commands::CheckBounds { list_file }) => {
            commands::check::run(&list_file)?;
            return Ok(());
        }
        Some(Commands::CompareBounds {
            scores_file,
            max_leaves,
        }) => {
            commands::compare::run(&scores_file, max_leaves)?;
            return Ok(());
        }
        _ => {
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
                Some(Commands::Bounds) => {
                    commands::bounds::run(&instance, cli.verbose)?;
                }
                Some(Commands::Exact { approach }) => {
                    commands::exact::run(&instance, &approach, cli.verbose)?;
                }
                Some(Commands::Heuristic { solver }) => {
                    commands::heuristic::run(&instance, &solver, cli.verbose)?;
                }
                None => {
                    commands::exact::run(&instance, "shi-mestel", cli.verbose)?;
                }
                _ => unreachable!(),
            }
        }
    }

    Ok(())
}
