//! Bounds computation subcommand.

use std::io::{self, Write};

use klados_core::Instance;
use klados_exact::lower_bound::maf_bounds;

pub fn run(instance: &Instance, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    let bounds = maf_bounds(&instance.trees, instance.num_leaves);
    if verbose {
        eprintln!("Bounds: lower={}, upper={}", bounds.lower, bounds.upper);
    }
    let mut stdout = io::stdout().lock();
    let n = instance.num_leaves as usize;
    let k = bounds.upper;

    for i in 0..(k - 1).min(n) {
        writeln!(stdout, "{};", i + 1)?;
    }

    if k <= n {
        let remaining: Vec<String> = ((k - 1 + 1)..=n).map(|i| i.to_string()).collect();
        if !remaining.is_empty() {
            if remaining.len() == 1 {
                writeln!(stdout, "{};", remaining[0])?;
            } else {
                writeln!(stdout, "({});", remaining.join(","))?;
            }
        }
    }

    Ok(())
}
