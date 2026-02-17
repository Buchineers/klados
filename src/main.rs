//! Klados - PACE 2026 Maximum Agreement Forest Solver
//!
//! κλάδος (klados) - Ancient Greek for "branch"

use clap::{Parser, Subcommand};
use klados_core::{Instance, Tree};
use klados_exact::lower_bound::maf_bounds;
use klados_exact::solver_by_name;
use pace26io::binary_tree::IndexedBinTreeBuilder;
use pace26io::newick::NewickWriter;
use pace26io::pace::simplified::Instance as PaceInstance;
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

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
        /// Exact approach name (shi-mestel or ilp)
        #[arg(long, default_value = "shi-mestel")]
        approach: String,
    },
    /// Show instance info
    Info,
    /// Compute bounds and output trivial MAF with upper bound components
    Bounds,
    /// Validate bounds against best known scores from a JSON file
    ValidateBounds {
        /// Path to best_known_scores.json file
        #[arg(value_name = "FILE")]
        scores_file: PathBuf,
    },
    /// Check bounds on all instances from a list file
    CheckBounds {
        /// Path to instance list file (e.g., all.lst)
        #[arg(value_name = "FILE")]
        list_file: PathBuf,
    },
    /// Compare LP relaxation bounds vs Olver 2-approx bounds
    CompareBounds {
        /// Path to best_known_scores.json file
        #[arg(value_name = "FILE")]
        scores_file: PathBuf,
        /// Max leaves to test with LP (LP is slower for large instances)
        #[arg(long, default_value = "50")]
        max_leaves: u32,
    },
}

#[derive(serde::Deserialize)]
struct ScoreEntry {
    best_known: usize,
    our_score: usize,
    leaves: usize,
    trees: usize,
    #[serde(default)]
    name: String,
    path: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::ValidateBounds { scores_file }) => {
            run_validate_bounds(&scores_file)?;
            return Ok(());
        }
        Some(Commands::CheckBounds { list_file }) => {
            run_check_bounds(&list_file)?;
            return Ok(());
        }
        Some(Commands::CompareBounds {
            scores_file,
            max_leaves,
        }) => {
            run_compare_bounds(&scores_file, max_leaves)?;
            return Ok(());
        }
        _ => {
            // Other commands read from stdin
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
                    let bounds = maf_bounds(&instance.trees, instance.num_leaves);
                    if cli.verbose {
                        eprintln!("Bounds: lower={}, upper={}", bounds.lower, bounds.upper);
                    }
                    let mut stdout = io::stdout().lock();
                    let n = instance.num_leaves as usize;
                    let k = bounds.upper;

                    for i in 0..(k - 1).min(n) {
                        writeln!(stdout, "{};", i + 1)?;
                    }

                    if k <= n {
                        let remaining: Vec<String> =
                            ((k - 1 + 1)..=n).map(|i| i.to_string()).collect();
                        if !remaining.is_empty() {
                            if remaining.len() == 1 {
                                writeln!(stdout, "{};", remaining[0])?;
                            } else {
                                writeln!(stdout, "({});", remaining.join(","))?;
                            }
                        }
                    }
                }
                Some(Commands::Exact { approach }) => {
                    let mut solver = solver_by_name(&approach)
                        .unwrap_or_else(|| panic!("Unknown approach: {}", approach));
                    let components = solver.solve(&instance).expect("Failed to find solution");

                    if cli.verbose {
                        eprintln!("Solution: {} components", components.len());
                    }

                    let mut stdout = io::stdout().lock();
                    for tree in &components {
                        tree.cursor().write_newick(&mut stdout)?;
                        writeln!(stdout)?;
                    }
                }
                None => {
                    let mut solver = solver_by_name("shi-mestel").expect("Missing default solver");
                    let components = solver.solve(&instance).expect("Failed to find solution");

                    if cli.verbose {
                        eprintln!("Solution: {} components", components.len());
                    }

                    let mut stdout = io::stdout().lock();
                    for tree in &components {
                        tree.cursor().write_newick(&mut stdout)?;
                        writeln!(stdout)?;
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    Ok(())
}

fn run_validate_bounds(scores_file: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // Read the scores file
    let content = fs::read_to_string(scores_file)?;
    let entries: HashMap<String, ScoreEntry> = serde_json::from_str(&content)?;

    println!("Validating bounds against {} instances...\n", entries.len());
    println!(
        "{:<40} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "Instance", "Leaves", "Trees", "Best", "Lower", "Upper", "Gap", "Time(ms)"
    );
    println!("{}", "-".repeat(100));

    let mut total_time_ms = 0.0f64;
    let mut upper_violations = 0usize;
    let mut lower_violations = 0usize;
    let mut total_gap = 0usize;
    let mut violation_list: Vec<(String, usize, usize, usize, usize, String)> = Vec::new(); // (digest, lower, upper, best, gap, type)
    let mut processed = 0usize;
    let mut errors = 0usize;

    for (digest, entry) in entries.iter() {
        let path = &entry.path;

        if !std::path::Path::new(path).exists() {
            println!(
                "{:<40} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8}",
                &digest[..20],
                entry.leaves,
                entry.trees,
                entry.best_known,
                "N/A",
                "FILE",
                "N/A"
            );
            errors += 1;
            continue;
        }

        let start = Instant::now();

        // Read and parse instance
        let file_content = match fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => {
                println!(
                    "{:<40} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8}",
                    &digest[..20],
                    entry.leaves,
                    entry.trees,
                    entry.best_known,
                    "N/A",
                    "READ",
                    "N/A"
                );
                errors += 1;
                continue;
            }
        };

        let reader = BufReader::new(file_content.as_bytes());
        let mut builder = IndexedBinTreeBuilder::default();
        let pace = match PaceInstance::try_read(reader, &mut builder) {
            Ok(p) => p,
            Err(_) => {
                println!(
                    "{:<40} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8}",
                    &digest[..20],
                    entry.leaves,
                    entry.trees,
                    entry.best_known,
                    "N/A",
                    "PARSE",
                    "N/A"
                );
                errors += 1;
                continue;
            }
        };

        let num_leaves = pace.num_leaves as u32;
        let trees: Vec<Tree> = pace
            .trees
            .iter()
            .map(|t| Tree::from_cursor(t.top_down(), num_leaves))
            .collect();
        let instance = Instance::new(trees, num_leaves);

        // Compute bounds
        let bounds = maf_bounds(&instance.trees, instance.num_leaves);
        let elapsed = start.elapsed().as_micros() as f64;
        total_time_ms += elapsed / 1000.0;

        // Check for BOTH upper and lower bound violations
        let reference = entry.best_known.min(entry.our_score);

        let gap;
        if bounds.upper < reference {
            // Upper bound violation: our upper < optimal
            upper_violations += 1;
            gap = reference - bounds.upper;
            violation_list.push((
                digest.clone(),
                bounds.lower,
                bounds.upper,
                reference,
                gap,
                "UPPER".to_string(),
            ));
        } else if bounds.lower > reference {
            // Lower bound violation: our lower > optimal
            lower_violations += 1;
            gap = bounds.lower - reference;
            violation_list.push((
                digest.clone(),
                bounds.lower,
                bounds.upper,
                reference,
                gap,
                "LOWER".to_string(),
            ));
        } else {
            gap = bounds.upper - reference;
        }

        total_gap += gap;
        processed += 1;

        let name = if entry.name.is_empty() {
            digest[..20].to_string()
        } else {
            format!(
                "{} ({})",
                &entry.name[..entry.name.len().min(15)],
                &digest[..8]
            )
        };

        println!(
            "{:<40} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8.3}",
            name,
            entry.leaves,
            entry.trees,
            reference,
            bounds.lower,
            bounds.upper,
            gap,
            elapsed / 1000.0
        );
    }

    let total_violations = upper_violations + lower_violations;

    println!("{}", "-".repeat(100));
    println!("\n=== SUMMARY ===");
    println!("Total instances:     {}", entries.len());
    println!("Processed:           {}", processed);
    println!("Errors:              {}", errors);
    println!(
        "Upper violations:    {} (our upper < best)",
        upper_violations
    );
    println!(
        "Lower violations:    {} (our lower > best)",
        lower_violations
    );
    println!("Total violations:    {}", total_violations);
    println!(
        "Success rate:        {:.2}%",
        100.0 * (processed - total_violations) as f64 / processed as f64
    );
    println!("Total gap:           {}", total_gap);
    println!(
        "Mean gap:            {:.2}",
        if processed > 0 {
            total_gap as f64 / processed as f64
        } else {
            0.0
        }
    );
    println!("Total time:          {:.2} ms", total_time_ms);
    println!(
        "Mean time:           {:.2} ms",
        if processed > 0 {
            total_time_ms / processed as f64
        } else {
            0.0
        }
    );

    if total_violations > 0 {
        println!("\n=== ALL VIOLATIONS ===");
        println!(
            "{:<20} {:>6} {:>6} {:>6} {:>8} {:>8}",
            "Instance", "Lower", "Upper", "Best", "Gap", "Type"
        );
        println!("{}", "-".repeat(70));
        for (digest, lower, upper, best, gap, vtype) in &violation_list {
            println!(
                "{:<20} {:>6} {:>6} {:>6} {:>8} {:>8}",
                &digest[..20],
                lower,
                upper,
                best,
                gap,
                vtype
            );
        }
        println!("\n⚠️  WARNING: Found {} total violations", total_violations);
    } else if processed > 0 {
        println!("\n✓ All bounds valid");
    }

    Ok(())
}

fn run_check_bounds(list_file: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // Read the list file
    let content = fs::read_to_string(list_file)?;
    let lines: Vec<&str> = content.lines().collect();

    println!(
        "Checking bounds on {} instances from {}...\n",
        lines.len(),
        list_file.display()
    );
    println!(
        "{:<20} {:>6} {:>6} {:>6} {:>8} {:>8} {:>10}",
        "Instance", "Trees", "Leaves", "Lower", "Upper", "Time(ms)", "Status"
    );
    println!("{}", "-".repeat(80));

    let mut total_time_ms = 0.0f64;
    let mut processed = 0usize;
    let mut errors = 0usize;
    let mut slowest = (String::new(), 0.0f64, usize::MAX, usize::MAX); // (id, time, leaves, trees)

    let total_instances = lines.len();
    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let digest = if line.starts_with("s:") {
            &line[2..]
        } else {
            line
        };

        // Build path from digest
        let path = format!(
            "stride-downloads/{}/{}/{}",
            &digest[..2],
            &digest[2..4],
            &digest[4..]
        );

        if !std::path::Path::new(&path).exists() {
            println!(
                "{:<20} {:>6} {:>6} {:>6} {:>8} {:>8} {:>10}",
                &digest[..20.min(digest.len())],
                "-",
                "-",
                "-",
                "-",
                "-",
                "NOT_FOUND"
            );
            errors += 1;
            continue;
        }

        let start = Instant::now();

        // Read and parse instance
        let file_content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => {
                println!(
                    "{:<20} {:>6} {:>6} {:>6} {:>8} {:>8} {:>10}",
                    &digest[..20.min(digest.len())],
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "READ_ERR"
                );
                errors += 1;
                continue;
            }
        };

        let reader = BufReader::new(file_content.as_bytes());
        let mut builder = IndexedBinTreeBuilder::default();
        let pace = match PaceInstance::try_read(reader, &mut builder) {
            Ok(p) => p,
            Err(_) => {
                println!(
                    "{:<20} {:>6} {:>6} {:>6} {:>8} {:>8} {:>10}",
                    &digest[..20.min(digest.len())],
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "PARSE_ERR"
                );
                errors += 1;
                continue;
            }
        };

        let num_leaves = pace.num_leaves as u32;
        let trees: Vec<Tree> = pace
            .trees
            .iter()
            .map(|t| Tree::from_cursor(t.top_down(), num_leaves))
            .collect();
        let instance = Instance::new(trees, num_leaves);

        // Compute bounds
        let bounds = maf_bounds(&instance.trees, instance.num_leaves);
        let elapsed = start.elapsed().as_micros() as u64;
        let elapsed_ms = elapsed as f64 / 1000.0;
        total_time_ms += elapsed_ms;
        processed += 1;

        // Track slowest
        if elapsed_ms > slowest.1 {
            slowest = (
                digest.to_string(),
                elapsed_ms,
                num_leaves as usize,
                instance.num_trees(),
            );
        }

        let status = "OK";

        println!(
            "{:<20} {:>6} {:>6} {:>6} {:>8} {:>8.3} {:>10}",
            &digest[..20.min(digest.len())],
            instance.num_trees(),
            num_leaves,
            bounds.lower,
            bounds.upper,
            elapsed as f64 / 1000.0,
            status
        );
    }

    println!("{}", "-".repeat(80));
    println!("\n=== SUMMARY ===");
    println!("Total instances:     {}", total_instances);
    println!("Processed:           {}", processed);
    println!("Errors:              {}", errors);
    println!("Total time:          {:.2} ms", total_time_ms);
    println!(
        "Mean time:           {:.2} ms",
        if processed > 0 {
            total_time_ms / processed as f64
        } else {
            0.0
        }
    );

    if slowest.1 > 0.0 {
        println!("\nSlowest instance:");
        println!(
            "  {}: {:.3} ms ({} leaves, {} trees)",
            slowest.0, slowest.1, slowest.2, slowest.3
        );
    }

    if errors == 0 && processed > 0 {
        println!("\n✓ All {} instances processed successfully", processed);
    }

    Ok(())
}

fn run_compare_bounds(
    scores_file: &PathBuf,
    max_leaves: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read the scores file
    let content = fs::read_to_string(scores_file)?;
    let entries: HashMap<String, ScoreEntry> = serde_json::from_str(&content)?;

    // Filter to multi-tree instances with <= max_leaves
    let test_entries: Vec<_> = entries
        .iter()
        .filter(|(_, entry)| entry.trees >= 3 && entry.leaves <= max_leaves as usize)
        .collect();

    println!(
        "Analyzing bounds gaps on {} multi-tree instances (<= {} leaves)...\n",
        test_entries.len(),
        max_leaves
    );
    println!(
        "{:<40} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "Instance", "Leaves", "Trees", "Best", "Olver LB", "Gap", "Time(ms)", "Gap/Best%"
    );
    println!("{}", "-".repeat(100));

    let mut total_time_ms = 0.0f64;
    let mut processed = 0usize;
    let mut errors = 0usize;
    let mut total_gap = 0usize;
    let mut gap_histogram: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();

    for (digest, entry) in test_entries {
        let path = &entry.path;

        if !std::path::Path::new(path).exists() {
            println!(
                "{:<40} {:>6} {:>6} FILE NOT FOUND",
                &digest[..20],
                entry.leaves,
                entry.trees
            );
            errors += 1;
            continue;
        }

        // Read and parse instance
        let file_content = match fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => {
                println!(
                    "{:<40} {:>6} {:>6} READ ERROR",
                    &digest[..20],
                    entry.leaves,
                    entry.trees
                );
                errors += 1;
                continue;
            }
        };

        let reader = BufReader::new(file_content.as_bytes());
        let mut builder = IndexedBinTreeBuilder::default();
        let pace = match PaceInstance::try_read(reader, &mut builder) {
            Ok(p) => p,
            Err(_) => {
                println!(
                    "{:<40} {:>6} {:>6} PARSE ERROR",
                    &digest[..20],
                    entry.leaves,
                    entry.trees
                );
                errors += 1;
                continue;
            }
        };

        let num_leaves = pace.num_leaves as u32;
        let trees: Vec<Tree> = pace
            .trees
            .iter()
            .map(|t| Tree::from_cursor(t.top_down(), num_leaves))
            .collect();
        let instance = Instance::new(trees, num_leaves);

        // Time bounds computation
        let start = Instant::now();
        let bounds = maf_bounds(&instance.trees, instance.num_leaves);
        let elapsed_ms = start.elapsed().as_micros() as f64 / 1000.0;
        total_time_ms += elapsed_ms;

        let best = entry.best_known;
        let olver_lb = bounds.lower;
        let gap = best.saturating_sub(olver_lb);
        total_gap += gap;
        *gap_histogram.entry(gap).or_insert(0) += 1;

        let gap_pct = if best > 0 {
            (gap as f64 / best as f64) * 100.0
        } else {
            0.0
        };

        let name = if entry.name.is_empty() {
            digest[..20].to_string()
        } else {
            format!(
                "{} ({})",
                &entry.name[..entry.name.len().min(15)],
                &digest[..8]
            )
        };

        println!(
            "{:<40} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10.2} {:>9.1}%",
            name, entry.leaves, entry.trees, best, olver_lb, gap, elapsed_ms, gap_pct
        );

        processed += 1;
    }

    println!("{}", "-".repeat(100));
    println!("\n=== GAP ANALYSIS SUMMARY ===");
    println!("Instances tested:    {}", processed);
    println!("Errors:              {}", errors);
    println!("Total gap:           {}", total_gap);
    println!(
        "Mean gap:            {:.2}",
        if processed > 0 {
            total_gap as f64 / processed as f64
        } else {
            0.0
        }
    );
    println!(
        "Mean time:           {:.2} ms",
        total_time_ms / processed as f64
    );

    // Show gap distribution
    println!("\nGap distribution:");
    let mut gaps: Vec<_> = gap_histogram.iter().collect();
    gaps.sort_by_key(|(g, _)| **g);
    for (gap, count) in gaps.iter().take(15) {
        let bar = "█".repeat(*count / processed.max(1) * 50 + 1);
        println!("  gap={:>2}: {:>4} instances {}", gap, count, bar);
    }

    Ok(())
}
