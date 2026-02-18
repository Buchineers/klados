//! Check bounds on all instances from a list file.

use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use klados_core::{Instance, Tree};
use klados_exact::lower_bound::maf_bounds;
use pace26io::binary_tree::IndexedBinTreeBuilder;
use pace26io::pace::simplified::Instance as PaceInstance;

pub fn run(list_file: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(list_file)?;
    let lines: Vec<&str> = content.lines().collect();
    let base_dir = list_file.parent().unwrap_or(std::path::Path::new("."));

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
    let mut slowest = (String::new(), 0.0f64, usize::MAX, usize::MAX);

    let total_instances = lines.len();
    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let digest = line.strip_prefix("s:").unwrap_or(line);

        let path = base_dir.join(format!(
            "stride-downloads/{}/{}/{}",
            &digest[..2],
            &digest[2..4],
            &digest[4..]
        ));

        if !path.exists() {
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

        let bounds = maf_bounds(&instance.trees, instance.num_leaves);
        let elapsed = start.elapsed().as_micros() as u64;
        let elapsed_ms = elapsed as f64 / 1000.0;
        total_time_ms += elapsed_ms;
        processed += 1;

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
