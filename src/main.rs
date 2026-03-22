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
    /// Apply kernelization rules and report reduction statistics.
    Kernelize {
        /// Enable subtree reduction (default: all enabled)
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        subtree: bool,
        /// Enable chain reduction
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        chain: bool,
        /// Enable 3-2 chain reduction
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        chain32: bool,
        /// Enable 3-2 chain reduction for multi-tree (proven for all t >= 2)
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        chain32_multi: bool,
        /// Victim selection strategy for parameter-reducing rules: first, last, max-cascade
        #[arg(long, default_value = "first")]
        strategy: String,
        /// Max distinct cherry partners for multi-tree 3-2 (2=classic, 0=unlimited)
        #[arg(long, default_value = "0")]
        max_partners: usize,
    },
    /// Diagnose kernelization gaps: find singletons missed by reduction rules.
    KernelizeDiag,
    /// Delete a specific leaf and output the reduced instance.
    DeleteLeaf {
        #[arg(long)]
        leaf: u32,
    },
    Info,
    Bounds,
    RedBlueUB,
    /// Print detailed bounds comparison (red-blue, LP relaxation, Olver LP*).
    BoundsDetail,
    ValidateBounds {
        #[arg(value_name = "FILE")]
        scores_file: std::path::PathBuf,
        #[arg(long, default_value = "0")]
        top_n: usize,
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
    AnalyzeRun {
        #[arg(value_name = "FILE")]
        summary_file: std::path::PathBuf,
        #[arg(long, default_value = "10")]
        top_n: usize,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::ValidateBounds { scores_file, top_n }) => {
            commands::validate::run(&scores_file, top_n)?;
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
        Some(Commands::AnalyzeRun {
            summary_file,
            top_n,
        }) => {
            commands::analyze::run(&summary_file, top_n)?;
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
                Some(Commands::BoundsDetail) => {
                    commands::bounds_detail::run(&instance)?;
                }
                Some(Commands::Exact { approach }) => {
                    commands::exact::run(&instance, &approach, cli.verbose)?;
                }
                Some(Commands::Kernelize {
                    subtree,
                    chain,
                    chain32,
                    chain32_multi,
                    strategy,
                    max_partners,
                }) => {
                    commands::kernelize::run(
                        &instance,
                        subtree,
                        chain,
                        chain32,
                        chain32_multi,
                        &strategy,
                        if max_partners == 0 { usize::MAX } else { max_partners },
                        cli.verbose,
                    )?;
                }
                Some(Commands::KernelizeDiag) => {
                    commands::kernelize_diag::run(&instance, cli.verbose)?;
                }
                Some(Commands::DeleteLeaf { leaf }) => {
                    commands::delete_leaf::run(&instance, leaf)?;
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
