//! Kernelization diagnostic — find leaves that are always singletons in optimal MAFs
//! but not detected by current reduction rules.
//!
//! This identifies the "gap" between our kernelization and the theoretical optimum
//! for each specific instance.

use std::time::Instant;

use klados_core::{Instance, Tree, NONE};
use klados_exact::kernelize::{self, KernelizeConfig};

pub fn run(instance: &Instance, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    let m = instance.num_trees();
    let n = instance.num_leaves;

    // Step 1: Kernelize
    let kern_start = Instant::now();
    let config = KernelizeConfig::default();
    let kern = kernelize::kernelize(instance, &config);
    let kern_time = kern_start.elapsed();

    let n_kern = kern.instance.num_leaves;
    let param_red = kern.param_reduction;

    eprintln!("Instance: {} trees, {} leaves", m, n);
    eprintln!("After kernelization: {} leaves, param_reduction={} ({:.1}% removed, {:.2}ms)",
        n_kern, param_red,
        100.0 * (n - n_kern) as f64 / n as f64,
        kern_time.as_secs_f64() * 1000.0);

    // Step 2: Solve the original instance exactly
    let solve_start = Instant::now();
    let mut solver = klados_exact::solver_by_name("maf-sat")
        .expect("maf-sat solver not available");
    let components = match solver.solve(instance) {
        Some(c) => c,
        None => {
            eprintln!("ERROR: Solver failed to find a solution");
            return Ok(());
        }
    };
    let solve_time = solve_start.elapsed();

    let maf_size = components.len();
    eprintln!("Optimal MAF: {} components (solved in {:.2}ms)", maf_size, solve_time.as_secs_f64() * 1000.0);

    // Step 3: Identify singleton leaves in the optimal solution
    let mut singleton_labels: Vec<u32> = Vec::new();
    for comp in &components {
        let leaves: Vec<u32> = comp.leaves().collect();
        if leaves.len() == 1 {
            singleton_labels.push(leaves[0]);
        }
    }
    singleton_labels.sort();

    eprintln!("Singletons in optimal MAF: {} leaves: {:?}", singleton_labels.len(), singleton_labels);

    // Step 4: Which singletons were already found by kernelization?
    let kern_deleted: Vec<u32> = kern.stats.deleted_labels.clone();
    let mut already_found: Vec<u32> = Vec::new();
    let mut missed: Vec<u32> = Vec::new();

    for &s in &singleton_labels {
        if kern_deleted.contains(&s) {
            already_found.push(s);
        } else {
            missed.push(s);
        }
    }

    eprintln!();
    eprintln!("=== Kernelization Gap Analysis ===");
    eprintln!("Singletons detected by 3-2 rule: {} {:?}", already_found.len(), already_found);
    eprintln!("Singletons MISSED by kernelization: {} {:?}", missed.len(), missed);

    if missed.is_empty() {
        eprintln!("PERFECT: All singletons in the optimal MAF were detected by kernelization.");
    } else {
        eprintln!();
        eprintln!("Missed singletons — local structure:");
        for &label in &missed {
            eprintln!("  Leaf {}:", label);
            for (t_idx, tree) in instance.trees.iter().enumerate() {
                let node = tree.node_by_label(label);
                if node == NONE {
                    eprintln!("    T{}: not present", t_idx + 1);
                    continue;
                }
                let sib = tree.sibling(node);
                let sib_info = match sib {
                    Some(s) if tree.is_leaf(s) => format!("leaf sibling={}", tree.label[s as usize]),
                    Some(_) => "internal sibling".to_string(),
                    None => "no sibling (root)".to_string(),
                };
                let depth = tree.depth[node as usize];
                let in_cherry = match sib {
                    Some(s) => tree.is_leaf(s),
                    None => false,
                };
                eprintln!("    T{}: depth={}, {}{}", t_idx + 1, depth, sib_info,
                    if in_cherry { " [cherry]" } else { "" });
            }
        }

        eprintln!();
        eprintln!("Potential kernel improvement: {} → {} leaves ({} more leaves removable)",
            n_kern,
            n_kern as i32 - missed.len() as i32,
            missed.len());
    }

    // Print summary line suitable for batch processing
    // n  m  n_kern  maf  singletons  found  missed
    println!("{}\t{}\t{}\t{}\t{}\t{}\t{}",
        n, m, n_kern, maf_size, singleton_labels.len(), already_found.len(), missed.len());

    Ok(())
}
