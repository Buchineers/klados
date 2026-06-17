//! Klados - PACE 2026 Maximum Agreement Forest Solver
//!
//! κλάδος (klados) - Ancient Greek for "branch"

mod commands;

use clap::{Parser, Subcommand, ValueEnum};
use commands::bounds::BoundsAlgo;
use klados_core::Instance;
use klados_core::kernelize::VictimStrategy;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "klados")]
#[command(
    author,
    version,
    about = "κλάδος - PACE 2026 Maximum Agreement Forest solver"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Solve an instance (reads from stdin). Run without a name to list solvers.
    Solve {
        /// Solver name; omit to list available solvers.
        solver: Option<String>,
        #[arg(short, long)]
        verbose: bool,
    },

    /// Compute bounds. Single instance from stdin, or batch with --list.
    /// Specify multiple --algo for side-by-side comparison.
    Bounds {
        /// Bound algorithm(s). Repeat for comparison: --algo chen-pair --algo red-blue
        #[arg(long = "algo", short = 'a', value_enum, action = clap::ArgAction::Append)]
        algos: Vec<BoundsAlgo>,
        /// Instance list file (.lst) for batch mode.
        #[arg(long, short = 'l', value_name = "FILE")]
        list: Option<PathBuf>,
        /// Known scores JSON for validation.
        #[arg(long, short = 's', value_name = "FILE")]
        scores: Option<PathBuf>,
        #[arg(short, long)]
        verbose: bool,
    },

    /// Apply kernelization rules.
    Kernelize {
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        subtree: bool,
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        chain: bool,
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        chain32: bool,
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        chain32_multi: bool,
        #[arg(long, value_enum, default_value_t = VictimStrategy::default())]
        strategy: VictimStrategy,
        #[arg(long, default_value = "0")]
        max_partners: usize,
        #[arg(short, long)]
        verbose: bool,
    },

    /// Diagnose kernelization gaps.
    KernelizeDiag,

    /// Cluster decomposition analysis.
    ClusterAnalyze,

    /// Print instance info.
    Info,
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .target(env_logger::Target::Stderr)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Solve {
            solver,
            verbose: _verbose,
        } => match solver {
            None => {
                for info in klados_solve::catalog() {
                    println!("{:<22}  {}", info.name, info.description);
                    for (var, desc) in (info.options)() {
                        println!("    {var:<38}  {desc}");
                    }
                }
            }
            Some(name) => match klados_solve::catalog().iter().find(|i| i.name == name) {
                Some(info) => (info.run)(),
                None => {
                    eprintln!("unknown solver: {name}");
                    eprintln!("run `klados solve` (no argument) to list available solvers");
                }
            },
        },

        Commands::Bounds {
            algos,
            list,
            scores,
            verbose,
        } => {
            if algos.is_empty() {
                eprintln!("bounds: specify at least one --algo. Options:");
                for v in BoundsAlgo::value_variants() {
                    eprintln!("  {:?}", v);
                }
                return Ok(());
            }
            commands::bounds::run(&algos, list.as_deref(), scores.as_deref(), verbose)?;
        }

        Commands::Kernelize {
            subtree,
            chain,
            chain32,
            chain32_multi,
            strategy,
            max_partners,
            verbose,
        } => {
            let instance = Instance::from_stdin()?;
            commands::kernelize::run(
                &instance,
                subtree,
                chain,
                chain32,
                chain32_multi,
                strategy,
                if max_partners == 0 {
                    usize::MAX
                } else {
                    max_partners
                },
                verbose,
            )?;
        }

        Commands::KernelizeDiag => {
            let instance = Instance::from_stdin()?;
            commands::kernelize_diag::run(&instance)?;
        }

        Commands::ClusterAnalyze => {
            let instance = Instance::from_stdin()?;
            commands::cluster_analyze::run(&instance)?;
        }

        Commands::Info => {
            let instance = Instance::from_stdin()?;
            eprintln!("Trees: {}", instance.num_trees());
            eprintln!("Leaves: {}", instance.num_leaves);
            for (i, tree) in instance.trees.iter().enumerate() {
                eprintln!("Tree {}: {} nodes", i + 1, tree.num_nodes());
            }
        }
    }

    Ok(())
}
