//! Validate Chen 2-approx and Whidden 3-approx bounds against known optimal
//! scores from pace_summary.json. Computes per-method timing, tightness vs OPT,
//! and reports any violations (LB > OPT or UB < OPT).

use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use klados_core::Instance;
use klados_exact::chen_rspr::chen_pair_bounds;
use klados_exact::kernelize::{self, KernelizeConfig};
use klados_exact::whidden::approx_3_for_instance;
use pace26io::binary_tree::IndexedBinTreeBuilder;
use pace26io::pace::simplified::Instance as PaceInstance;

pub fn run(list_file: &PathBuf, scores_file: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // Load known scores
    let scores_data: serde_json::Value = serde_json::from_str(&fs::read_to_string(scores_file)?)?;
    let scores_arr = scores_data
        .as_array()
        .ok_or("pace_summary.json is not an array")?;
    let mut known_scores: HashMap<String, usize> = HashMap::new();
    for entry in scores_arr {
        if let (Some(id), Some(score)) = (
            entry["idigest"].as_str(),
            entry["best_known_score"].as_u64(),
        ) {
            // Match on first 32 hex chars (full hash), case-insensitive
            known_scores.insert(id.to_uppercase(), score as usize);
        }
    }

    let content = fs::read_to_string(list_file)?;
    let lines: Vec<&str> = content.lines().collect();
    let base_dir = list_file.parent().unwrap_or(std::path::Path::new("."));

    let total = lines
        .iter()
        .filter(|l| {
            let l = l.trim();
            !l.is_empty() && !l.starts_with('#')
        })
        .count();

    eprintln!("Validating bounds on {} instances\n", total);

    // Header
    println!(
        "{:>16} {:>5} {:>5} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5} | {:>5} {:>5}",
        "Instance", "n", "kern", "w3LB", "w3UB", "t_ms", "chLB", "chUB", "t_ms", "opt", "ok?"
    );
    println!("{}", "-".repeat(90));

    let mut processed = 0;
    let mut errors = 0;
    let mut w3_total_ms = 0.0;
    let mut ch_total_ms = 0.0;
    let mut ch_violations = 0;
    let mut w3_violations = 0;
    let mut ch_ratios = Vec::new();
    let mut w3_ratios = Vec::new();

    let kern_config = KernelizeConfig::default();

    for line in &lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let digest = line.strip_prefix("s:").unwrap_or(line);
        let short = &digest[..16.min(digest.len())];

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
        if !path.exists() {
            errors += 1;
            continue;
        }
        let fc = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => {
                errors += 1;
                continue;
            }
        };
        let reader = BufReader::new(fc.as_bytes());
        let mut builder = IndexedBinTreeBuilder::default();
        let pace = match PaceInstance::try_read(reader, &mut builder) {
            Ok(p) => p,
            Err(_) => {
                errors += 1;
                continue;
            }
        };

        let num_leaves = pace.num_leaves as u32;
        let trees: Vec<_> = pace
            .trees
            .iter()
            .map(|t| klados_core::Tree::from_cursor(t.top_down(), num_leaves))
            .collect();
        let instance = Instance::new(trees, num_leaves);
        let m = instance.num_trees();

        // Kernelize
        let kern = kernelize::kernelize(&instance, &kern_config);
        let reduced = &kern.instance;
        let n_kern = reduced.num_leaves;
        let pr = kern.param_reduction;

        let opt_score = known_scores.get(&digest.to_uppercase()).copied();

        if m != 2 {
            continue;
        }

        // -- Whidden 3-approx --
        let t0 = Instant::now();
        let w3_cuts =
            approx_3_for_instance(&reduced.trees[0], &reduced.trees[1], reduced.num_leaves)
                as usize;
        let w3_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let w3_lb = w3_cuts.div_ceil(3) + 1 + pr; // ceil(cuts/3) + 1 = MAF LB
        let w3_ub = w3_cuts + 1 + pr; // cuts + 1 = MAF UB

        // -- Chen 2-approx --
        let t0 = Instant::now();
        let (cl, cu) = chen_pair_bounds(&reduced.trees[0], &reduced.trees[1]);
        let ch_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ch_lb = cl.saturating_add(1) + pr;
        let ch_ub = cu.saturating_add(1) + pr;

        w3_total_ms += w3_ms;
        ch_total_ms += ch_ms;

        // Validate
        let ok_w3 = if let Some(opt) = opt_score {
            let v = w3_lb <= opt && opt <= w3_ub;
            if !v {
                w3_violations += 1;
            }
            if opt > 0 {
                w3_ratios.push(w3_ub as f64 / opt as f64);
            }
            v
        } else {
            true
        }; // no known optimum to check

        let ok_ch = if let Some(opt) = opt_score {
            let v = ch_lb <= opt && opt <= ch_ub;
            if !v {
                ch_violations += 1;
            }
            if opt > 0 {
                ch_ratios.push(ch_ub as f64 / opt as f64);
            }
            v
        } else {
            true
        };

        processed += 1;

        let ok_str = match (opt_score, ok_w3, ok_ch) {
            (None, _, _) => "?".to_string(),
            (Some(_), true, true) => "✓".to_string(),
            _ => format!(
                "!{}{}",
                if !ok_w3 { "w" } else { "" },
                if !ok_ch { "c" } else { "" }
            ),
        };

        println!(
            "{:>16} {:>5} {:>5} | {:>5} {:>5} {:>5.0} | {:>5} {:>5} {:>5.0} | {:>5} {:>5}",
            short,
            num_leaves,
            n_kern,
            w3_lb,
            w3_ub,
            w3_ms,
            ch_lb,
            ch_ub,
            ch_ms,
            opt_score.unwrap_or(0),
            ok_str,
        );
    }

    println!("{}", "-".repeat(90));

    // Summary
    if processed > 0 {
        let avg_w3 = w3_total_ms / processed as f64;
        let avg_ch = ch_total_ms / processed as f64;
        println!(
            "\n{:>16} {:>5} {:>5} | {:>5} {:>5} {:>5.0} | {:>5} {:>5} {:>5.0} |",
            "AVG (ms)", "", "", "", "", avg_w3, "", "", avg_ch
        );
        println!("{} instances | errors: {}", processed, errors);
        if w3_violations > 0 {
            println!("WARNING: Whidden 3-approx VIOLATIONS: {}", w3_violations);
        }
        if ch_violations > 0 {
            println!("WARNING: Chen 2-approx VIOLATIONS: {}", ch_violations);
        }
        if !ch_ratios.is_empty() {
            let avg_r = ch_ratios.iter().sum::<f64>() / ch_ratios.len() as f64;
            let max_r = ch_ratios.iter().cloned().fold(0.0, f64::max);
            println!("Chen tightness vs OPT: avg={:.2}x max={:.2}x", avg_r, max_r);
        }
        if !w3_ratios.is_empty() {
            let avg_r = w3_ratios.iter().sum::<f64>() / w3_ratios.len() as f64;
            let max_r = w3_ratios.iter().cloned().fold(0.0, f64::max);
            println!(
                "Whidden tightness vs OPT: avg={:.2}x max={:.2}x",
                avg_r, max_r
            );
        }
    }

    Ok(())
}
