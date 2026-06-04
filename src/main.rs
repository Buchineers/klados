//! Klados - PACE 2026 Maximum Agreement Forest Solver
//!
//! κλάδος (klados) - Ancient Greek for "branch"

// glibc malloc retains freed pages on this alloc-heavy workload (millions of
// short-lived column vectors), letting RSS climb toward the 8 GB instance limit
// on large cluster-free cores. mimalloc reclaims aggressively and fragments far
// less, cutting peak RSS substantially.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod commands;
mod solver;

use clap::{Parser, Subcommand, ValueEnum};
use commands::bounds::BoundsAlgo;
use klados_core::Instance;
use klados_core::kernelize::VictimStrategy;
use solver::SolverChoice;
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
    /// Solve an instance. Run without arguments to list available solvers.
    Solve {
        #[arg(value_enum)]
        solver: Option<SolverChoice>,
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
        } => {
            if solver.is_none() {
                solver::list_solvers();
                return Ok(());
            }
            let instance = Instance::from_stdin()?;
            solver::solve_and_print(&instance, solver.unwrap())?;
        }

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
