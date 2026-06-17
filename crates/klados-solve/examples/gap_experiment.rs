//! De-risking experiment for the dual-guided set-packing design (design doc §6).
//!
//! Usage: cargo run --release --example gap_experiment -- <best_known> <file> [<best_known> <file> ...]

use klados_core::Instance;
use klados_solve::run_packing_gap_experiment;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 || args.len() % 2 != 0 {
        eprintln!("usage: gap_experiment <best_known> <file> [<best_known> <file> ...]");
        std::process::exit(2);
    }

    println!(
        "{:>7} {:>9} {:>8} {:>6} {:>10} {:>9} {:>9} {:>9} {:>8}",
        "n", "best_kn", "pool", "cg+", "lp_frac", "greedy0", "greedyD", "poolIP", "ip_to?"
    );

    let mut i = 0;
    while i < args.len() {
        let best_known: usize = args[i].parse()?;
        let path = &args[i + 1];
        i += 2;

        let instance = Instance::from_file(Path::new(path))?;
        let t0 = std::time::Instant::now();
        let r = run_packing_gap_experiment(&instance, best_known);
        let wall = t0.elapsed().as_secs_f64();

        let fmt_opt_usize =
            |o: Option<usize>| o.map(|v| v.to_string()).unwrap_or_else(|| "-".into());
        let fmt_opt_f64 =
            |o: Option<f64>| o.map(|v| format!("{:.1}", v)).unwrap_or_else(|| "-".into());

        println!(
            "{:>7} {:>9} {:>8} {:>6} {:>10} {:>9} {:>9} {:>9} {:>8}",
            r.n,
            r.best_known,
            r.pool_size,
            r.cg_columns_added,
            fmt_opt_f64(r.lp_components),
            fmt_opt_usize(r.greedy_priceless),
            fmt_opt_usize(r.greedy_dual),
            fmt_opt_usize(r.pool_ip),
            if r.pool_ip_timed_out { "yes" } else { "no" },
        );

        // Detail line: ratios vs best_known and step timings.
        let ratio = |v: Option<usize>| {
            v.map(|x| format!("{:.3}", x as f64 / best_known as f64))
                .unwrap_or_else(|| "-".into())
        };
        eprintln!(
            "  [{}] wall={:.1}s  vs best_known={}: greedy0={} greedyD={} poolIP={} | timings_ms={:?}",
            path,
            wall,
            best_known,
            ratio(r.greedy_priceless),
            ratio(r.greedy_dual),
            ratio(r.pool_ip),
            r.timings_ms,
        );
    }

    Ok(())
}
