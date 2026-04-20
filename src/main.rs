//! Klados - PACE 2026 Maximum Agreement Forest Solver
//!
//! κλάδος (klados) - Ancient Greek for "branch"

mod commands;

use clap::{Parser, Subcommand};
use klados_core::lower_bound::red_blue_approx_detailed;
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
        #[arg(long, default_value = "maf-bp-multi")]
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
    /// Analyze cluster decomposition potential (Kelk + rspr).
    ClusterAnalyze,
    /// Solve an exact packing problem over a mined component pool.
    CandidatePacking {
        #[arg(value_name = "FILE")]
        candidates_file: std::path::PathBuf,
        /// Optional path to write the reconstructed forest as Newick lines.
        #[arg(long)]
        solution_out: Option<std::path::PathBuf>,
        /// Use at most this many ranked candidates from the pool (0 = all available).
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// Drop candidates seen fewer than this many times in the mining runs.
        #[arg(long, default_value_t = 1)]
        min_seen: usize,
        /// Drop candidates smaller than this size.
        #[arg(long, default_value_t = 2)]
        min_size: usize,
        /// Number of enrichment rounds after the initial pack solve.
        #[arg(long, default_value_t = 1)]
        rounds: usize,
        /// Add all non-singleton proper subsets of selected larger blocks.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        expand_subsets: bool,
        /// Add exact-compatible one-leaf supersets of selected blocks.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        expand_add_one: bool,
        /// Only add-one expand selected blocks up to this size.
        #[arg(long, default_value_t = 6)]
        add_one_max_base_size: usize,
        /// In each tree, only consider add-one leaves whose join sits at most
        /// this many ancestors above the base block LCA.
        #[arg(long, default_value_t = 2)]
        add_one_max_rise: u16,
        /// Keep at most this many structurally closest add-one leaves per base
        /// block (0 = keep all leaves that pass the rise filter).
        #[arg(long, default_value_t = 16)]
        add_one_top_k: usize,
        /// Stop generating new candidates in one enrichment round after this cap.
        #[arg(long, default_value_t = 5000)]
        generated_cap: usize,
        /// Restricted master formulation to use for the actual pack solve.
        #[arg(long, value_enum, default_value_t = commands::candidate_packing::MasterMode::Nodepack)]
        master: commands::candidate_packing::MasterMode,
        /// Report side-by-side results for pairwise-conflict vs node-packing masters.
        #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
        compare_masters: bool,
        /// Run a reporting-only rooted pricing probe against the current node-packing LP.
        #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
        price_missing: bool,
        /// Add at most one LP-priced component per round before local expansion.
        #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
        price_add: bool,
        /// When pricing inserts a new column in a round, skip heuristic expansion
        /// and let the next re-solve react to pricing first.
        #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
        defer_expansion_while_priced: bool,
        /// Stop after this many consecutive rounds without improving the incumbent
        /// savings (0 = disabled).
        #[arg(long, default_value_t = 0)]
        stall_patience: usize,
    },
    /// Scan the kernelized instance for feasible small agreement components.
    ComponentScan {
        /// Kernelize before scanning (recommended).
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        kernelize: bool,
        /// Largest locally-grown candidate size to report (currently supports up to 5).
        #[arg(long, default_value_t = 5)]
        max_size: usize,
        /// Max number of feasible candidates to keep as samples per size.
        #[arg(long, default_value_t = 12)]
        sample_limit: usize,
        /// Stop local candidate generation once raw unique candidates exceed this cap.
        #[arg(long, default_value_t = 1_000_000)]
        candidate_cap: usize,
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
    /// Compare LB algorithms (red-blue dual, Olver TF, 3-approx) on 2-tree instances.
    CompareLb {
        #[arg(value_name = "FILE")]
        scores_file: std::path::PathBuf,
        #[arg(long, default_value = "4294967295")]
        max_leaves: u32,
    },
    /// Compare Red-Blue approximation scores against public heuristic best-known scores.
    CompareHeurApprox {
        #[arg(value_name = "FILE")]
        scores_file: std::path::PathBuf,
        #[arg(long, default_value = "4294967295")]
        max_leaves: u32,
        #[arg(long, default_value_t = 0)]
        limit: usize,
    },
    AnalyzeRun {
        #[arg(value_name = "FILE")]
        summary_file: std::path::PathBuf,
        #[arg(long, default_value = "10")]
        top_n: usize,
    },
    WhiddenStats {
        #[arg(value_name = "FILE")]
        list_file: std::path::PathBuf,
        #[arg(long)]
        max_instances: Option<usize>,
        #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
        show_instances: bool,
        #[arg(long, value_enum, default_value_t = commands::whidden_stats::OutputFormat::Human)]
        format: commands::whidden_stats::OutputFormat,
        #[arg(long, value_enum, default_value_t = commands::whidden_stats::ProgressMode::Auto)]
        progress: commands::whidden_stats::ProgressMode,
        #[arg(long, default_value_t = 250)]
        log_interval_ms: u64,
        #[arg(long)]
        output: Option<std::path::PathBuf>,
        /// Enable BB pruning via approx_3.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        bb: bool,
        /// Use Olver 2-approx dual LB for BB pruning instead of 3-approx.
        #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
        bb_2approx: bool,
        /// Enable the Whidden transposition table.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        tt_enabled: bool,
        /// Prune on TT hits instead of observe-only accounting.
        #[arg(long, default_value_t = false, action = clap::ArgAction::SetTrue)]
        tt_prune: bool,
        /// TT size as log2(entry_count).
        #[arg(long, default_value_t = 23)]
        tt_size_log2: u8,
        /// Enable the narrow rooted Mestel rule-6 rescue.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        mestel_rule6: bool,
        /// Enable exact bound-cache reuse for approx_3 / approx_2_lb.
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        bound_cache_enabled: bool,
        /// Bound-cache size as log2(entry_count).
        #[arg(long, default_value_t = 20)]
        bound_cache_size_log2: u8,
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
        Some(Commands::CompareLb {
            scores_file,
            max_leaves,
        }) => {
            commands::compare_lb::run(&scores_file, max_leaves)?;
            return Ok(());
        }
        Some(Commands::CompareHeurApprox {
            scores_file,
            max_leaves,
            limit,
        }) => {
            commands::compare_heur_approx::run(&scores_file, max_leaves, limit)?;
            return Ok(());
        }
        Some(Commands::AnalyzeRun {
            summary_file,
            top_n,
        }) => {
            commands::analyze::run(&summary_file, top_n)?;
            return Ok(());
        }
        Some(Commands::WhiddenStats {
            list_file,
            max_instances,
            show_instances,
            format,
            progress,
            log_interval_ms,
            output,
            bb,
            bb_2approx,
            tt_enabled,
            tt_prune,
            tt_size_log2,
            mestel_rule6,
            bound_cache_enabled,
            bound_cache_size_log2,
        }) => {
            commands::whidden_stats::run(
                &list_file,
                max_instances,
                show_instances,
                format,
                progress,
                log_interval_ms,
                output,
                bb,
                bb_2approx,
                tt_enabled,
                tt_prune,
                tt_size_log2,
                mestel_rule6,
                bound_cache_enabled,
                bound_cache_size_log2,
            )?;
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
                Some(Commands::RedBlueUB) => {
                    if instance.num_trees() != 2 {
                        return Err("red-blue-ub requires exactly 2 trees".into());
                    }
                    let t0 = std::time::Instant::now();
                    let rb = red_blue_approx_detailed(&instance.trees[0], &instance.trees[1]);
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!(
                        "Red-Blue: cuts={} components={} dual_lb={} ({:.1}ms)",
                        rb.ub,
                        rb.ub + 1,
                        rb.dual_lb,
                        ms
                    );
                    println!("{}", rb.ub + 1);
                }
                Some(Commands::ClusterAnalyze) => {
                    commands::cluster_analyze::run(&instance)?;
                }
                Some(Commands::CandidatePacking {
                    candidates_file,
                    solution_out,
                    limit,
                    min_seen,
                    min_size,
                    rounds,
                    expand_subsets,
                    expand_add_one,
                    add_one_max_base_size,
                    add_one_max_rise,
                    add_one_top_k,
                    generated_cap,
                    master,
                    compare_masters,
                    price_missing,
                    price_add,
                    defer_expansion_while_priced,
                    stall_patience,
                }) => {
                    commands::candidate_packing::run(
                        &instance,
                        &candidates_file,
                        solution_out.as_deref(),
                        limit,
                        min_seen,
                        min_size,
                        rounds,
                        expand_subsets,
                        expand_add_one,
                        add_one_max_base_size,
                        add_one_max_rise,
                        add_one_top_k,
                        generated_cap,
                        master,
                        compare_masters,
                        price_missing,
                        price_add,
                        defer_expansion_while_priced,
                        stall_patience,
                    )?;
                }
                Some(Commands::ComponentScan {
                    kernelize,
                    max_size,
                    sample_limit,
                    candidate_cap,
                }) => {
                    commands::component_scan::run(
                        &instance,
                        kernelize,
                        max_size,
                        sample_limit,
                        candidate_cap,
                    )?;
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
                        if max_partners == 0 {
                            usize::MAX
                        } else {
                            max_partners
                        },
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
