//! Kernelize subcommand — apply reduction rules and report statistics.

use klados_core::Instance;
use klados_solve::kernelize::{self, KernelizeConfig, VictimStrategy};
use pace26io::newick::NewickWriter;
use std::io::{self, Write};
use std::time::Instant;

pub fn run(
    instance: &Instance,
    subtree: bool,
    chain: bool,
    chain32: bool,
    chain32_multi: bool,
    victim_strategy: VictimStrategy,
    max_partners: usize,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = KernelizeConfig {
        subtree,
        chain,
        chain32,
        chain32_multi,
        protected_labels: Vec::new(),
        victim_strategy,
        max_partners,
    };

    let start = Instant::now();
    let result = kernelize::kernelize(instance, &config);
    let total_duration = start.elapsed();

    // Always print summary stats to stderr
    kernelize::print_stats(&result.stats);

    // Always print timing
    let total_us = total_duration.as_micros();
    if total_us > 1000 {
        eprintln!(
            "// Time: {:.2} ms ({} rule applications)",
            total_us as f64 / 1000.0,
            result.trace.len()
        );
    } else {
        eprintln!(
            "// Time: {} us ({} rule applications)",
            total_us,
            result.trace.len()
        );
    }

    // Per-rule timing breakdown
    if !result.trace.is_empty() {
        let mut rule_times: std::collections::BTreeMap<&str, (u128, usize)> =
            std::collections::BTreeMap::new();
        for event in &result.trace {
            let entry = rule_times.entry(event.rule_name).or_default();
            entry.0 += event.duration.as_micros();
            entry.1 += 1;
        }
        for (name, (us, count)) in &rule_times {
            eprintln!(
                "//   {}: {} us ({} calls, {:.1} us/call)",
                name,
                us,
                count,
                *us as f64 / *count as f64
            );
        }
    }

    if verbose {
        kernelize::print_taxa_detail(&result.stats);

        // Print rule trace
        if !result.trace.is_empty() {
            eprintln!("//");
            eprintln!("// Rule application trace ({} steps):", result.trace.len());
            for (i, event) in result.trace.iter().enumerate() {
                let us = event.duration.as_micros();
                let action_str = match &event.action {
                    kernelize::ReductionAction::Collapse { keep, remove } => {
                        format!("collapse({} <- {})", keep, remove)
                    }
                    kernelize::ReductionAction::Delete { victim } => {
                        format!("delete({})", victim)
                    }
                };
                eprintln!(
                    "//  {:>3}. {:<12} {:<22} -> {} leaves  ({} us)",
                    i + 1,
                    event.rule_name,
                    action_str,
                    event.leaves_after,
                    us,
                );
            }
        }
    }

    // Print reduced trees to stdout (Newick), followed by param reduction
    let mut stdout = io::stdout().lock();
    for tree in &result.instance.trees {
        tree.cursor().write_newick(&mut stdout)?;
        writeln!(stdout)?;
    }
    writeln!(stdout, "{}", result.param_reduction)?;

    Ok(())
}
