//! Bounds computation — single instance (stdin) or batch (--list).
//!
//! Supports multiple --algo flags for side-by-side comparison.
//! Reports per-algo timing, gap vs known optimum, and approximation guarantees.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use klados_core::Instance;
use klados_core::lower_bound::{maf_bounds, red_blue_approx_detailed};

// ── Algorithm selection ────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
pub enum BoundsAlgo {
    #[value(name = "maf-bounds")]
    MafBounds,
    #[value(name = "chen-pair")]
    ChenPair,
    #[value(name = "chen-app1")]
    ChenApp1,
    #[value(name = "red-blue")]
    RedBlue,
}

impl BoundsAlgo {
    fn display(&self) -> &'static str {
        match self {
            BoundsAlgo::MafBounds => "maf-bounds",
            BoundsAlgo::ChenPair => "chen-pair",
            BoundsAlgo::ChenApp1 => "chen-app1",
            BoundsAlgo::RedBlue => "red-blue",
        }
    }

    fn guarantee(&self) -> &'static str {
        match self {
            BoundsAlgo::MafBounds => "none",
            BoundsAlgo::ChenPair => "2-approx",
            BoundsAlgo::ChenApp1 => "2-approx",
            BoundsAlgo::RedBlue => "2-approx",
        }
    }
}

// ── Result ─────────────────────────────────────────────────────────────────

struct AlgoResult {
    lower: usize,
    upper: usize,
    ms: f64,
    err: Option<String>,
}

fn compute(algo: BoundsAlgo, instance: &Instance) -> AlgoResult {
    let t0 = Instant::now();

    let pair_err = |a: BoundsAlgo| AlgoResult {
        lower: 0, upper: 0, ms: 0.0,
        err: Some(format!("{} requires 2 trees, got {}", a.display(), instance.num_trees())),
    };

    let (lower, upper) = match algo {
        BoundsAlgo::MafBounds => {
            let b = maf_bounds(&instance.trees, instance.num_leaves);
            (b.lower, b.upper)
        }
        BoundsAlgo::ChenPair => {
            if instance.num_trees() != 2 { return pair_err(algo); }
            let (l, u) = klados_exact::chen_rspr::chen_pair_bounds(
                &instance.trees[0], &instance.trees[1],
            );
            (l + 1, u + 1)
        }
        BoundsAlgo::ChenApp1 => {
            if instance.num_trees() != 2 { return pair_err(algo); }
            let (l, u) = klados_exact::chen_rspr::chen_app1_bounds(
                &instance.trees[0], &instance.trees[1],
            );
            (l + 1, u + 1)
        }
        BoundsAlgo::RedBlue => {
            if instance.num_trees() != 2 { return pair_err(algo); }
            let rb = red_blue_approx_detailed(&instance.trees[0], &instance.trees[1]);
            (rb.dual_lb + 1, rb.ub + 1)
        }
    };

    AlgoResult { lower, upper, ms: t0.elapsed().as_secs_f64() * 1000.0, err: None }
}

// ── Entry point ────────────────────────────────────────────────────────────

pub fn run(
    algos: &[BoundsAlgo],
    list: Option<&Path>,
    scores_file: Option<&Path>,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let scores = if let Some(path) = scores_file {
        Some(load_scores(path)?)
    } else {
        None
    };

    if let Some(list_file) = list {
        run_batch(algos, list_file, scores.as_ref(), verbose)
    } else {
        run_single(algos, scores.as_ref(), verbose)
    }
}

// ── Single instance ────────────────────────────────────────────────────────

fn run_single(
    algos: &[BoundsAlgo],
    scores: Option<&HashMap<String, usize>>,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let instance = Instance::from_stdin()?;
    let n = instance.num_leaves;
    let m = instance.num_trees();

    eprintln!("trees={}  leaves={}", m, n);
    for &algo in algos {
        eprintln!("-- {} ({}) --", algo.display(), algo.guarantee());
        let r = compute(algo, &instance);
        if let Some(ref e) = r.err {
            eprintln!("  ERROR: {}", e);
        } else {
            let lb_ub_gap = if r.lower > 0 { format!("{:.2}x", r.upper as f64 / r.lower as f64) } else { "-".into() };
            eprintln!("  LB={}  UB={}  UB/LB={}  {:.1}ms", r.lower, r.upper, lb_ub_gap, r.ms);
            if let Some(scores) = scores {
                if let Some(name) = &instance.name {
                    if let Some(&opt) = scores.get(&name.to_uppercase()) {
                        let gap = if opt > 0 { r.upper as f64 / opt as f64 } else { 0.0 };
                        let ok = r.lower <= opt && opt <= r.upper;
                        eprintln!("  OPT={}  UB/OPT={:.2}x  {}", opt, gap, if ok { "✓" } else { "✗ VIOLATION" });
                    }
                }
            }
        }
    }
    Ok(())
}

// ── Batch mode ─────────────────────────────────────────────────────────────

struct BatchStats {
    times: Vec<f64>,
    gaps: Vec<f64>,
    violations: Vec<(String, usize, usize, usize)>, // (digest, LB, UB, OPT)
    errors: usize,
}

fn run_batch(
    algos: &[BoundsAlgo],
    list_file: &Path,
    scores: Option<&HashMap<String, usize>>,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (entries, _) = klados_core::parse_list_file(list_file)?;
    let has_scores = scores.is_some();

    let mut stats: Vec<BatchStats> = (0..algos.len())
        .map(|_| BatchStats { times: Vec::new(), gaps: Vec::new(), violations: Vec::new(), errors: 0 })
        .collect();

    // Header
    let col_w = 10usize;
    let hdr: Vec<String> = algos.iter()
        .map(|a| format!("{:>width$}", &a.display()[..a.display().len().min(col_w)], width = col_w))
        .collect();
    eprintln!("{:>64} {:>3} {:>5}  {:>5} | {} | {} | {}",
        "DIGEST", "m", "n", "OPT",
        hdr.iter().map(|s| format!("{} {:<5} {:<5} {:<5} {:<5}",
            s, "LB", "UB", "gap", "ms")).collect::<Vec<_>>().join(" | "),
        if algos.len() > 1 { "best-gap" } else { "" },
        "ok"
    );
    eprintln!("{}", "-".repeat(180));

    for entry in &entries {
        let instance = match Instance::from_file(&entry.path) {
            Ok(i) => i,
            Err(_) => {
                for s in &mut stats { s.errors += 1; }
                continue;
            }
        };

        let digest = &entry.digest;
        let opt = scores.and_then(|s| s.get(&entry.digest.to_uppercase()).copied());
        eprint!("{:>64} {:>3} {:>5}  {:>5}",
            digest,
            instance.num_trees(),
            instance.num_leaves,
            opt.map_or("-".into(), |o| o.to_string())
        );

        eprint!(" |");
        let mut best_gap = f64::MAX;
        let mut all_ok = true;

        for (i, &algo) in algos.iter().enumerate() {
            let r = compute(algo, &instance);
            if let Some(ref e) = r.err {
                stats[i].errors += 1;
                eprint!(" {:>col_w$} {:>5} {:>5} {:>5} {:>5}",
                    format!("{:>width$}", e.split_whitespace().next().unwrap_or("ERR"), width = col_w),
                    "-", "-", "-", "-");
                continue;
            }

            stats[i].times.push(r.ms);

            let gap_str = if let Some(opt) = opt {
                if opt > 0 {
                    let g = r.upper as f64 / opt as f64;
                    stats[i].gaps.push(g);
                    if g < best_gap { best_gap = g; }
                    let ok = r.lower <= opt && opt <= r.upper;
                    if !ok {
                        all_ok = false;
                        stats[i].violations.push((entry.digest.clone(), r.lower, r.upper, opt));
                    }
                    format!("{:.2}x", g)
                } else { "-".into() }
            } else { "-".into() };

            eprint!(" {:>col_w$} {:>5} {:>5} {:>5} {:>5.1}",
                format!("{:>width$}", algo.display(), width = col_w),
                r.lower, r.upper, gap_str, r.ms);
        }

        // Best gap column
        if algos.len() > 1 {
            eprint!(" | {:>8}", if best_gap < f64::MAX { format!("{:.2}x", best_gap) } else { "-".into() });
        }
        eprint!(" | {:>4}", if has_scores { if all_ok { "✓" } else { "✗" } } else { "-" });
        eprintln!();
    }

    // ── Summary ─────────────────────────────────────────────────────────────

    eprintln!();
    eprintln!("══ Summary ══");

    // Per-algo stats table
    for (i, &algo) in algos.iter().enumerate() {
        let s = &stats[i];
        let n = s.times.len();
        eprintln!();
        eprintln!("  {} ({}) — {} instances, {} errors",
            algo.display(), algo.guarantee(), n, s.errors);

        if n > 0 {
            let mut ts = s.times.clone();
            ts.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let avg_t = ts.iter().sum::<f64>() / n as f64;
            let (min_t, max_t) = (ts[0], ts[n - 1]);
            let med_t = if n % 2 == 0 { (ts[n/2 - 1] + ts[n/2]) / 2.0 } else { ts[n/2] };
            let p95_t = ts[(n as f64 * 0.95) as usize];
            eprintln!("    time    avg={:>6.1}ms  min={:>6.1}ms  max={:>6.1}ms  med={:>6.1}ms  p95={:>6.1}ms",
                avg_t, min_t, max_t, med_t, p95_t);
        }

        if !s.gaps.is_empty() {
            let mut gs = s.gaps.clone();
            gs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let n_g = gs.len();
            let avg_g = gs.iter().sum::<f64>() / n_g as f64;
            let (min_g, max_g) = (gs[0], gs[n_g - 1]);
            eprintln!("    UB/OPT  avg={:>5.2}x  min={:>5.2}x  max={:>5.2}x",
                avg_g, min_g, max_g);
        }

        if !s.violations.is_empty() {
            eprintln!("    violations  {}", s.violations.len());
        } else {
            eprintln!("    violations  0 ✓");
        }
    }

    // Violations detail — always
    let total_violations: usize = stats.iter().map(|s| s.violations.len()).sum();
    if total_violations > 0 {
        eprintln!();
        eprintln!("══ Violations ══");
        for (i, &algo) in algos.iter().enumerate() {
            for (digest, lb, ub, opt) in &stats[i].violations {
                eprintln!("  {:42}  {:<12}  LB={:<4} UB={:<4} OPT={}",
                    digest, algo.display(), lb, ub, opt);
            }
        }
    }

    Ok(())
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn load_scores(path: &Path) -> Result<HashMap<String, usize>, Box<dyn std::error::Error>> {
    let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let arr = data.as_array().ok_or("scores file is not a JSON array")?;
    let mut map = HashMap::new();
    for entry in arr {
        if let (Some(id), Some(score)) = (
            entry["idigest"].as_str(),
            entry["best_known_score"].as_u64(),
        ) {
            map.insert(id.to_uppercase(), score as usize);
        }
    }
    Ok(map)
}
