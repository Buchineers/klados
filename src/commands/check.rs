//! Check bounds on all instances from a list file.
//!
//! Kernelizes each instance first (like the actual solvers do), then computes
//! bounds on the reduced instance and reports both raw and kernelized stats.

use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use klados_core::{Instance, Tree};
use klados_exact::kernelize::{self, KernelizeConfig};
use klados_exact::lower_bound::maf_bounds;
use pace26io::binary_tree::IndexedBinTreeBuilder;
use pace26io::pace::simplified::Instance as PaceInstance;

pub fn run(list_file: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(list_file)?;
    let lines: Vec<&str> = content.lines().collect();
    let base_dir = list_file.parent().unwrap_or(std::path::Path::new("."));

    let total_instances = lines.iter().filter(|l| {
        let l = l.trim();
        !l.is_empty() && !l.starts_with('#')
    }).count();

    eprintln!(
        "Checking bounds on {} instances from {}\n",
        total_instances,
        list_file.display()
    );

    // Header
    println!(
        "{:<16} {:>2} {:>4} {:>4} {:>4} {:>3} {:>3} {:>4} {:>4} {:>4} {:>8}",
        "Instance", "m", "n", "kern", "sub", "chn", "c32", "LB", "UB", "gap", "ms"
    );
    println!("{}", "-".repeat(72));

    let mut total_time_ms = 0.0f64;
    let mut processed = 0usize;
    let mut errors = 0usize;
    let mut total_gap = 0usize;
    let mut max_gap = 0usize;
    let mut total_lb = 0usize;
    let mut total_ub = 0usize;

    let kern_config = KernelizeConfig::default();

    for line in &lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let digest = line.strip_prefix("s:").unwrap_or(line);

        // Try relative to list file first, then relative to cwd
        let rel = format!(
            "stride-downloads/{}/{}/{}",
            &digest[..2],
            &digest[2..4],
            &digest[4..]
        );
        let path = {
            let p = base_dir.join(&rel);
            if p.exists() { p } else { PathBuf::from(&rel) }
        };

        let short = &digest[..16.min(digest.len())];

        if !path.exists() {
            println!("{:<16} NOT_FOUND", short);
            errors += 1;
            continue;
        }

        let file_content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => {
                println!("{:<16} READ_ERR", short);
                errors += 1;
                continue;
            }
        };

        let reader = BufReader::new(file_content.as_bytes());
        let mut builder = IndexedBinTreeBuilder::default();
        let pace = match PaceInstance::try_read(reader, &mut builder) {
            Ok(p) => p,
            Err(_) => {
                println!("{:<16} PARSE_ERR", short);
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
        let m = instance.num_trees();

        let start = Instant::now();

        // Kernelize first
        let kern = kernelize::kernelize(&instance, &kern_config);
        let reduced = &kern.instance;
        let n_kern = reduced.num_leaves;
        let param_red = kern.param_reduction;

        // Compute bounds on the reduced instance
        let bounds = maf_bounds(&reduced.trees, reduced.num_leaves);

        // Adjust for parameter reduction from 3-2 chain
        let lb = bounds.lower + param_red;
        let ub = bounds.upper + param_red;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        total_time_ms += elapsed_ms;
        processed += 1;

        let gap = ub.saturating_sub(lb);
        total_gap += gap;
        total_lb += lb;
        total_ub += ub;
        if gap > max_gap {
            max_gap = gap;
        }

        println!(
            "{:<16} {:>2} {:>4} {:>4} {:>4} {:>3} {:>3} {:>4} {:>4} {:>4} {:>8.1}",
            short,
            m,
            num_leaves,
            n_kern,
            kern.stats.subtree_removed(),
            kern.stats.chain_removed(),
            kern.stats.chain32_removed(),
            lb,
            ub,
            gap,
            elapsed_ms,
        );
    }

    println!("{}", "-".repeat(72));

    if processed > 0 {
        let avg_gap = total_gap as f64 / processed as f64;
        let avg_time = total_time_ms / processed as f64;
        println!(
            "\n{} instances | avg gap: {:.1} | max gap: {} | total LB: {} | total UB: {}",
            processed, avg_gap, max_gap, total_lb, total_ub,
        );
        println!(
            "total time: {:.1}s | avg: {:.1}ms | errors: {}",
            total_time_ms / 1000.0,
            avg_time,
            errors,
        );
    }

    Ok(())
}
