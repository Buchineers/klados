//! Compare lower bound algorithms: red-blue dual, Olver TwinForest 2-approx, and 3-approx.
//!
//! For each 2-tree instance, computes all three LBs and reports quality/performance.

use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use klados_core::Tree;
use klados_core::lower_bound::red_blue_approx_detailed;
use klados_exact::whidden::{approx_2_lb_for_instance, approx_3_for_instance};
use pace26io::binary_tree::IndexedBinTreeBuilder;
use pace26io::pace::simplified::Instance as PaceInstance;

#[derive(serde::Deserialize)]
struct ScoreEntry {
    best_known: usize,
    #[allow(dead_code)]
    our_score: usize,
    leaves: usize,
    trees: usize,
    #[serde(default)]
    #[allow(dead_code)]
    name: String,
    path: String,
}

// STRIDE summary.json format
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct SummaryEntry {
    #[serde(rename = "s_key")]
    key: String,
    #[serde(rename = "s_idigest")]
    idigest: String,
    #[serde(rename = "s_num_trees")]
    num_trees: u32,
    #[serde(rename = "s_num_leaves")]
    num_leaves: u32,
    #[serde(rename = "s_prev_best")]
    prev_best: Option<u32>,
    #[serde(rename = "s_score")]
    score: u32,
    #[serde(rename = "s_path")]
    path: String,
}

// PACE public database format
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct PaceSummaryEntry {
    track: String,
    inst_num: usize,
    name: String,
    trees: usize,
    leaves: usize,
    idigest: String,
    best_known_score: usize,
    #[serde(default)]
    num_best: usize,
    #[serde(default)]
    num_valid: usize,
    #[serde(default)]
    avg_best_compute_time_secs: f64,
}

pub fn run(scores_file: &PathBuf, max_leaves: u32) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(scores_file)?;

    let first_line = content.lines().next().unwrap_or("");
    let is_summary_format = first_line.contains("s_key") || first_line.contains("s_prev_best");
    let is_pace_format = first_line.trim_start().starts_with('[');

    let entries: HashMap<String, ScoreEntry> = if is_pace_format {
        let pace_entries: Vec<PaceSummaryEntry> = serde_json::from_str(&content)?;
        let base = scores_file.parent().unwrap_or(std::path::Path::new("."));
        let mut entries = HashMap::new();
        for pe in &pace_entries {
            if pe.track != "exact" || pe.trees != 2 {
                continue;
            }
            let h = pe.idigest.to_lowercase();
            let path = base
                .join("stride-downloads")
                .join(&h[..2])
                .join(&h[2..4])
                .join(&h[4..]);
            entries.insert(
                h.clone(),
                ScoreEntry {
                    best_known: pe.best_known_score,
                    our_score: pe.best_known_score,
                    leaves: pe.leaves,
                    trees: pe.trees,
                    name: pe.name.clone(),
                    path: path.to_string_lossy().into_owned(),
                },
            );
        }
        entries
    } else if is_summary_format {
        let mut entries = HashMap::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(summary) = serde_json::from_str::<SummaryEntry>(line) {
                if summary.num_trees != 2 {
                    continue;
                }
                let optimal = if let Some(db_best) = summary.prev_best {
                    (summary.score as usize).min(db_best as usize)
                } else {
                    summary.score as usize
                };
                entries.insert(
                    summary.idigest.clone(),
                    ScoreEntry {
                        best_known: optimal,
                        our_score: optimal,
                        leaves: summary.num_leaves as usize,
                        trees: summary.num_trees as usize,
                        name: String::new(),
                        path: summary.path,
                    },
                );
            }
        }
        entries
    } else {
        let all: HashMap<String, ScoreEntry> = serde_json::from_str(&content)?;
        all.into_iter().filter(|(_, e)| e.trees == 2).collect()
    };

    let base_dir = scores_file.parent().unwrap_or(std::path::Path::new("."));

    let mut sorted_entries: Vec<_> = entries.iter().collect();
    sorted_entries.sort_by(|a, b| (a.1.leaves, a.1.trees).cmp(&(b.1.leaves, b.1.trees)));

    if max_leaves < u32::MAX {
        sorted_entries.retain(|(_, e)| e.leaves <= max_leaves as usize);
    }

    eprintln!(
        "Comparing LB algorithms on {} 2-tree instances (leaves <= {})...\n",
        sorted_entries.len(),
        max_leaves
    );

    // Header
    println!(
        "{:<12} {:>5} {:>5} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>5} {:>5}",
        "Instance", "n", "Best", "RB_D", "Olv_D", "A3/3", "RB_ms", "Olv_ms", "A3_ms", "O>R", "O>A3"
    );
    println!("{}", "-".repeat(110));

    let mut processed = 0usize;
    let mut errors = 0usize;

    // Accumulators
    let mut total_rb_ms = 0.0f64;
    let mut total_olv_ms = 0.0f64;
    let mut total_a3_ms = 0.0f64;

    let mut olv_gt_rb = 0usize; // Olver strictly > red-blue dual
    let mut olv_lt_rb = 0usize; // Olver strictly < red-blue dual
    let mut olv_eq_rb = 0usize;

    let mut olv_gt_a3 = 0usize; // Olver strictly > approx3/3
    let mut olv_lt_a3 = 0usize;
    let mut olv_eq_a3 = 0usize;

    let mut rb_violations = 0usize; // rb dual > best_known
    let mut olv_violations = 0usize; // olver lb > best_known

    let mut rb_sum = 0i64;
    let mut olv_sum = 0i64;
    let mut a3_sum = 0i64;
    let mut best_sum = 0i64;

    // Per-instance delta tracking for summary
    let mut olv_minus_rb: Vec<i32> = Vec::new();
    let mut olv_minus_a3: Vec<i32> = Vec::new();

    for (digest, entry) in &sorted_entries {
        let entry_path = std::path::Path::new(&entry.path);
        let full_path = if entry_path.is_absolute() {
            // Try absolute path first; if missing, remap stride-downloads/ relative to base_dir
            if entry_path.exists() {
                entry_path.to_path_buf()
            } else if let Some(idx) = entry.path.find("stride-downloads/") {
                base_dir.join(&entry.path[idx..])
            } else {
                entry_path.to_path_buf()
            }
        } else {
            base_dir.join(entry_path)
        };

        if !full_path.exists() {
            errors += 1;
            continue;
        }

        let file_content = match fs::read_to_string(&full_path) {
            Ok(c) => c,
            Err(_) => {
                errors += 1;
                continue;
            }
        };

        let reader = BufReader::new(file_content.as_bytes());
        let mut builder = IndexedBinTreeBuilder::default();
        let pace = match PaceInstance::try_read(reader, &mut builder) {
            Ok(p) => p,
            Err(_) => {
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

        if trees.len() != 2 {
            continue;
        }

        // 1. Red-blue dual LB
        let t0 = Instant::now();
        let rb = red_blue_approx_detailed(&trees[0], &trees[1]);
        let rb_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let rb_dual = rb.dual_lb as i32;

        // 2. Olver TwinForest 2-approx dual LB
        let t0 = Instant::now();
        let olv_lb = approx_2_lb_for_instance(&trees[0], &trees[1], num_leaves);
        let olv_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // 3. 3-approximation via TwinForest (reuse the instance)
        let t0 = Instant::now();
        let a3_val = approx_3_for_instance(&trees[0], &trees[1], num_leaves);
        let a3_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let a3_lb = (a3_val + 2) / 3; // ceil(a3/3) — a3 is a 3-approx so a3/3 <= OPT

        let best = entry.best_known as i32;

        // Track violations
        if rb_dual > best {
            rb_violations += 1;
        }
        if olv_lb > best {
            olv_violations += 1;
        }

        // Comparisons
        let d_olv_rb = olv_lb - rb_dual;
        let d_olv_a3 = olv_lb - a3_lb;
        olv_minus_rb.push(d_olv_rb);
        olv_minus_a3.push(d_olv_a3);

        if olv_lb > rb_dual {
            olv_gt_rb += 1;
        } else if olv_lb < rb_dual {
            olv_lt_rb += 1;
        } else {
            olv_eq_rb += 1;
        }

        if olv_lb > a3_lb {
            olv_gt_a3 += 1;
        } else if olv_lb < a3_lb {
            olv_lt_a3 += 1;
        } else {
            olv_eq_a3 += 1;
        }

        total_rb_ms += rb_ms;
        total_olv_ms += olv_ms;
        total_a3_ms += a3_ms;

        rb_sum += rb_dual as i64;
        olv_sum += olv_lb as i64;
        a3_sum += a3_lb as i64;
        best_sum += best as i64;

        processed += 1;

        // Markers for improvements
        let olv_rb_mark = if olv_lb > rb_dual {
            "+"
        } else if olv_lb < rb_dual {
            "-"
        } else {
            ""
        };
        let olv_a3_mark = if olv_lb > a3_lb {
            "+"
        } else if olv_lb < a3_lb {
            "-"
        } else {
            ""
        };

        let short_name = if digest.len() >= 12 {
            &digest[..12]
        } else {
            digest.as_str()
        };

        println!(
            "{:<12} {:>5} {:>5} {:>6} {:>6} {:>6} {:>7.2} {:>7.2} {:>7.2} {:>5} {:>5}",
            short_name,
            num_leaves,
            best,
            rb_dual,
            olv_lb,
            a3_lb,
            rb_ms,
            olv_ms,
            a3_ms,
            olv_rb_mark,
            olv_a3_mark,
        );
    }

    println!("{}", "-".repeat(110));

    if processed == 0 {
        println!("No instances processed.");
        return Ok(());
    }

    // === SUMMARY ===
    println!(
        "\n=== SUMMARY ({} instances, {} errors) ===",
        processed, errors
    );

    println!("\n--- Correctness ---");
    println!("RB dual violations (> best):    {}", rb_violations);
    println!("Olver TF violations (> best):   {}", olv_violations);
    if olv_violations > 0 {
        println!(
            "WARNING: Olver LB exceeds best_known on {} instances!",
            olv_violations
        );
    } else {
        println!("OK: Olver LB never exceeds best_known");
    }

    println!("\n--- Quality: Olver vs Red-Blue Dual ---");
    println!(
        "Olver > RB dual:  {:>6} ({:.1}%)",
        olv_gt_rb,
        100.0 * olv_gt_rb as f64 / processed as f64
    );
    println!(
        "Olver = RB dual:  {:>6} ({:.1}%)",
        olv_eq_rb,
        100.0 * olv_eq_rb as f64 / processed as f64
    );
    println!(
        "Olver < RB dual:  {:>6} ({:.1}%)",
        olv_lt_rb,
        100.0 * olv_lt_rb as f64 / processed as f64
    );
    println!("Mean RB dual:     {:.2}", rb_sum as f64 / processed as f64);
    println!("Mean Olver LB:    {:.2}", olv_sum as f64 / processed as f64);
    println!(
        "Mean best_known:  {:.2}",
        best_sum as f64 / processed as f64
    );

    olv_minus_rb.sort();
    let olv_rb_median = olv_minus_rb[processed / 2];
    let olv_rb_mean = olv_minus_rb.iter().map(|&x| x as f64).sum::<f64>() / processed as f64;
    let olv_rb_total: i64 = olv_minus_rb.iter().map(|&x| x as i64).sum();
    println!(
        "Delta (Olv-RB):   total={} mean={:.2} median={}",
        olv_rb_total, olv_rb_mean, olv_rb_median
    );

    println!("\n--- Quality: Olver vs 3-Approx/3 ---");
    println!(
        "Olver > A3/3:     {:>6} ({:.1}%)",
        olv_gt_a3,
        100.0 * olv_gt_a3 as f64 / processed as f64
    );
    println!(
        "Olver = A3/3:     {:>6} ({:.1}%)",
        olv_eq_a3,
        100.0 * olv_eq_a3 as f64 / processed as f64
    );
    println!(
        "Olver < A3/3:     {:>6} ({:.1}%)",
        olv_lt_a3,
        100.0 * olv_lt_a3 as f64 / processed as f64
    );
    println!("Mean A3/3 LB:     {:.2}", a3_sum as f64 / processed as f64);

    olv_minus_a3.sort();
    let olv_a3_median = olv_minus_a3[processed / 2];
    let olv_a3_mean = olv_minus_a3.iter().map(|&x| x as f64).sum::<f64>() / processed as f64;
    let olv_a3_total: i64 = olv_minus_a3.iter().map(|&x| x as i64).sum();
    println!(
        "Delta (Olv-A3/3): total={} mean={:.2} median={}",
        olv_a3_total, olv_a3_mean, olv_a3_median
    );

    println!("\n--- Performance ---");
    println!(
        "Total RB time:    {:.1} ms ({:.3} ms/inst)",
        total_rb_ms,
        total_rb_ms / processed as f64
    );
    println!(
        "Total Olver time: {:.1} ms ({:.3} ms/inst)",
        total_olv_ms,
        total_olv_ms / processed as f64
    );
    println!(
        "Total A3 time:    {:.1} ms ({:.3} ms/inst)",
        total_a3_ms,
        total_a3_ms / processed as f64
    );
    let speedup_rb = total_rb_ms / total_olv_ms;
    let speedup_a3 = total_a3_ms / total_olv_ms;
    println!("RB/Olver ratio:   {:.2}x", speedup_rb);
    println!("A3/Olver ratio:   {:.2}x", speedup_a3);

    // Tightness vs best_known
    println!("\n--- Tightness (LB / best_known) ---");
    let rb_tightness = if best_sum > 0 {
        rb_sum as f64 / best_sum as f64 * 100.0
    } else {
        0.0
    };
    let olv_tightness = if best_sum > 0 {
        olv_sum as f64 / best_sum as f64 * 100.0
    } else {
        0.0
    };
    let a3_tightness = if best_sum > 0 {
        a3_sum as f64 / best_sum as f64 * 100.0
    } else {
        0.0
    };
    println!("RB dual:          {:.1}%", rb_tightness);
    println!("Olver TF:         {:.1}%", olv_tightness);
    println!("A3/3:             {:.1}%", a3_tightness);

    Ok(())
}
