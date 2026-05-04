//! Validate Chen 2-approx and Whidden 3-approx bounds against known optimal
//! scores from pace_summary.json. Computes per-method timing, tightness vs OPT,
//! and reports any violations (LB > OPT or UB < OPT).

use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use klados_core::Instance;
use klados_exact::chen_rspr::{chen_app1_bounds, chen_pair_bounds};
use klados_exact::kernelize::{self, KernelizeConfig};
use klados_exact::whidden::approx_3_for_instance;
use pace26io::binary_tree::IndexedBinTreeBuilder;
use pace26io::pace::simplified::Instance as PaceInstance;

pub fn run(list_file: &PathBuf, scores_file: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // Load known scores
    let scores_data: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(scores_file)?)?;
    let scores_arr = scores_data.as_array().ok_or("pace_summary.json is not an array")?;
    let mut known_scores: HashMap<String, usize> = HashMap::new();
    for entry in scores_arr {
        if let (Some(id), Some(score)) = (
            entry["idigest"].as_str(),
            entry["best_known_score"].as_u64(),
        ) {
            known_scores.insert(id.to_uppercase(), score as usize);
        }
    }

    let content = fs::read_to_string(list_file)?;
    let lines: Vec<&str> = content.lines().collect();
    let base_dir = list_file.parent().unwrap_or(std::path::Path::new("."));

    let total = lines.iter().filter(|l| {
        let l = l.trim();
        !l.is_empty() && !l.starts_with('#')
    }).count();

    eprintln!("Validating bounds on {} instances vs {}\n", total, scores_file.display());

    println!("{:>16} {:>5} {:>5} | {:>5} {:>6} | {:>5} {:>5} {:>6} | {:>5} {:>5} {:>6} | {:>5} {:>4}",
        "Instance", "n", "kern", "w3LB", "t_ms", "a1LB", "a1UB", "t_ms", "chLB", "chUB", "t_ms", "OPT", "ok");
    println!("{}", "-".repeat(96));

    let mut processed = 0;
    let mut errors = 0;
    let mut w3_total_ms = 0.0;
    let mut a1_total_ms = 0.0;
    let mut ch_total_ms = 0.0;
    let mut ch_viol = 0;
    let mut a1_viol = 0;
    let mut w3_viol = 0;
    let mut ch_ratios = Vec::new();
    let mut a1_ratios = Vec::new();

    let kern_config = KernelizeConfig::default();

    for line in &lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let digest = line.strip_prefix("s:").unwrap_or(line);
        let short = &digest[..16.min(digest.len())];

        let rel = format!(
            "stride-downloads/{}/{}/{}",
            &digest[..2], &digest[2..4], &digest[4..]
        );
        let path: PathBuf = { let p = base_dir.join(&rel); if p.exists() { p } else { PathBuf::from(&rel) } };
        if !path.exists() { errors += 1; continue; }
        let fc = match fs::read_to_string(&path) { Ok(c) => c, Err(_) => { errors += 1; continue; } };
        let reader = BufReader::new(fc.as_bytes());
        let mut builder = IndexedBinTreeBuilder::default();
        let pace = match PaceInstance::try_read(reader, &mut builder) { Ok(p) => p, Err(_) => { errors += 1; continue; } };

        let num_leaves = pace.num_leaves as u32;
        let trees: Vec<_> = pace.trees.iter()
            .map(|t| klados_core::Tree::from_cursor(t.top_down(), num_leaves))
            .collect();
        let instance = Instance::new(trees, num_leaves);
        let m = instance.num_trees();
        if m != 2 { continue; }

        let kern = kernelize::kernelize(&instance, &kern_config);
        let reduced = &kern.instance;
        let n_kern = reduced.num_leaves;
        let pr = kern.param_reduction;

        let opt = known_scores.get(&digest.to_uppercase()).copied();

        // Whidden 3-approx (LB only)
        let t0 = Instant::now();
        let w3_cuts = approx_3_for_instance(&reduced.trees[0], &reduced.trees[1], reduced.num_leaves) as usize;
        let w3_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let w3_lb = w3_cuts.div_ceil(3) + 1 + pr;

        // Chen app1 — simple cut-loop (LB + UB)
        let t0 = Instant::now();
        let (a1l, a1u) = chen_app1_bounds(&reduced.trees[0], &reduced.trees[1]);
        let a1_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let a1_lb = a1l.saturating_add(1) + pr;
        let a1_ub = a1u.saturating_add(1) + pr;

        // Chen app2 — full stopper oracle (LB + UB)
        let t0 = Instant::now();
        let (cl, cu) = chen_pair_bounds(&reduced.trees[0], &reduced.trees[1]);
        let ch_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ch_lb = cl.saturating_add(1) + pr;
        let ch_ub = cu.saturating_add(1) + pr;

        w3_total_ms += w3_ms;
        a1_total_ms += a1_ms;
        ch_total_ms += ch_ms;

        let ok_w3 = if let Some(o) = opt { let v = w3_lb <= o; if !v { w3_viol += 1; } v } else { true };
        let ok_a1 = if let Some(o) = opt { let v = a1_lb <= o && o <= a1_ub; if !v { a1_viol += 1; } if o > 0 { a1_ratios.push(a1_ub as f64 / o as f64); } v } else { true };
        let ok_ch = if let Some(o) = opt { let v = ch_lb <= o && o <= ch_ub; if !v { ch_viol += 1; } if o > 0 { ch_ratios.push(ch_ub as f64 / o as f64); } v } else { true };

        processed += 1;

        let ok = if ok_w3 && ok_a1 && ok_ch { "✓".to_string() }
            else { format!("!{}{}{}", if !ok_w3 { "w" } else { "" }, if !ok_a1 { "a" } else { "" }, if !ok_ch { "c" } else { "" }) };

        println!("{:>16} {:>5} {:>5} | {:>5} {:>5.0} | {:>5} {:>5} {:>5.0} | {:>5} {:>5} {:>5.0} | {:>5} {:>4}",
            short, num_leaves, n_kern,
            w3_lb, w3_ms,
            a1_lb, a1_ub, a1_ms,
            ch_lb, ch_ub, ch_ms,
            opt.unwrap_or(0), ok,
        );
    }

    println!("{}", "-".repeat(96));
    if processed > 0 {
        let avg_w3 = w3_total_ms / processed as f64;
        let avg_a1 = a1_total_ms / processed as f64;
        let avg_ch = ch_total_ms / processed as f64;
        println!("\n{:>16} {:>5} {:>5} | {:>5} {:>5.0} | {:>5} {:>5} {:>5.0} | {:>5} {:>5} {:>5.0} |",
            "AVG (ms)", "", "", "", avg_w3, "", "", avg_a1, "", "", avg_ch);
        println!("{} instances | errors: {}", processed, errors);
        if w3_viol > 0 { println!("W3  violations: {}", w3_viol); } else { println!("W3:  all valid ✓"); }
        if a1_viol > 0 { println!("App1 violations: {}", a1_viol); } else { println!("App1: all valid ✓"); }
        if ch_viol > 0 { println!("Chen violations: {}", ch_viol); } else { println!("Chen: all valid ✓"); }
        if !a1_ratios.is_empty() {
            let avg = a1_ratios.iter().sum::<f64>() / a1_ratios.len() as f64;
            let max = a1_ratios.iter().cloned().fold(0.0, f64::max);
            println!("App1 UB/OPT: avg={:.2} max={:.2}", avg, max);
        }
        if !ch_ratios.is_empty() {
            let avg = ch_ratios.iter().sum::<f64>() / ch_ratios.len() as f64;
            let max = ch_ratios.iter().cloned().fold(0.0, f64::max);
            println!("Chen UB/OPT: avg={:.2} max={:.2} (2-approx ≤2.0{})",
                avg, max, if max <= 2.0 { " ✓" } else { " ✗" });
        }
    }

    Ok(())
}
