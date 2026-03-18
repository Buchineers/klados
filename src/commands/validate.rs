//! Validate bounds against best known scores.

use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use klados_core::{Instance, Tree};
use klados_exact::lower_bound::maf_bounds;
use pace26io::binary_tree::IndexedBinTreeBuilder;
use pace26io::pace::simplified::Instance as PaceInstance;

#[derive(serde::Deserialize)]
struct ScoreEntry {
    best_known: usize,
    our_score: usize,
    leaves: usize,
    trees: usize,
    #[serde(default)]
    name: String,
    path: String,
}

// STRIDE summary.json format (one JSON object per line)
#[derive(serde::Deserialize, Debug)]
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

// PACE public database format (JSON array)
#[derive(serde::Deserialize, Debug)]
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

pub fn run(scores_file: &PathBuf, top_n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(scores_file)?;

    // Detect file format
    let first_line = content.lines().next().unwrap_or("");
    let is_summary_format = first_line.contains("s_key") || first_line.contains("s_prev_best");

    // Try pace_summary.json format first (JSON array with "idigest", "best_known_score")
    let is_pace_format = first_line.trim_start().starts_with('[');

    let entries: HashMap<String, ScoreEntry> = if is_pace_format {
        // Parse PACE public database format: JSON array of objects.
        // Resolve instance files via stride-downloads/<h[0:2]>/<h[2:4]>/<h[4:]>
        let pace_entries: Vec<PaceSummaryEntry> = serde_json::from_str(&content)?;
        let mut entries = HashMap::new();
        let base = scores_file.parent().unwrap_or(std::path::Path::new("."));
        for pe in &pace_entries {
            if pe.track != "exact" {
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
        // Parse summary.json format (one JSON object per line)
        let mut entries = HashMap::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(summary) = serde_json::from_str::<SummaryEntry>(line) {
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
        // Original best_known_scores.json format
        serde_json::from_str(&content)?
    };
    let base_dir = scores_file.parent().unwrap_or(std::path::Path::new("."));

    let mut sorted_entries: Vec<_> = entries.iter().collect();
    sorted_entries.sort_by(|a, b| (a.1.leaves, a.1.trees).cmp(&(b.1.leaves, b.1.trees)));

    if top_n > 0 {
        sorted_entries.truncate(top_n);
    }

    println!(
        "Validating bounds against {} instances...\n",
        sorted_entries.len()
    );
    println!(
        "{:<40} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5} {:>7} {:>8}",
        "Instance", "Leaves", "Trees", "Best", "LB", "UB", "Gap", "Tight%", "Time(ms)"
    );
    println!("{}", "-".repeat(100));

    let mut total_time_ms = 0.0f64;
    let mut upper_violations = 0usize;
    let mut lower_violations = 0usize;
    let mut violation_list: Vec<(String, usize, usize, usize, usize, String)> = Vec::new();
    let mut processed = 0usize;
    let mut errors = 0usize;

    let mut gaps: Vec<usize> = Vec::new();
    let mut tightness_pcts: Vec<f64> = Vec::new();

    for (digest, entry) in sorted_entries {
        let entry_path = std::path::Path::new(&entry.path);
        let full_path = if entry_path.is_absolute() {
            entry_path.to_path_buf()
        } else {
            base_dir.join(entry_path)
        };

        if !full_path.exists() {
            println!(
                "{:<40} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5} {:>7} {:>8}",
                &digest[..20],
                entry.leaves,
                entry.trees,
                entry.best_known,
                "N/A",
                "FILE",
                "-",
                "-",
                "-"
            );
            errors += 1;
            continue;
        }

        let start = Instant::now();

        let file_content = match fs::read_to_string(&full_path) {
            Ok(c) => c,
            Err(_) => {
                println!(
                    "{:<40} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5} {:>7} {:>8}",
                    &digest[..20],
                    entry.leaves,
                    entry.trees,
                    entry.best_known,
                    "N/A",
                    "READ",
                    "-",
                    "-",
                    "-"
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
                    "{:<40} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5} {:>7} {:>8}",
                    &digest[..20],
                    entry.leaves,
                    entry.trees,
                    entry.best_known,
                    "N/A",
                    "PARSE",
                    "-",
                    "-",
                    "-"
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
        let elapsed = start.elapsed().as_micros() as f64;
        total_time_ms += elapsed / 1000.0;

        let reference = entry.best_known.min(entry.our_score);
        let lb = bounds.lower;
        let ub = bounds.upper;

        let tightness_pct = if reference > 0 {
            (lb as f64 / reference as f64) * 100.0
        } else {
            100.0
        };
        tightness_pcts.push(tightness_pct);

        let gap;
        if ub < reference {
            upper_violations += 1;
            gap = reference - ub;
            violation_list.push((digest.clone(), lb, ub, reference, gap, "UPPER".to_string()));
        } else if lb > reference {
            lower_violations += 1;
            gap = lb - reference;
            violation_list.push((digest.clone(), lb, ub, reference, gap, "LOWER".to_string()));
        } else {
            gap = ub - reference;
        }

        gaps.push(gap);
        processed += 1;

        let name = if entry.name.is_empty() {
            digest[..20].to_string()
        } else {
            format!(
                "{} ({})",
                &entry.name[..entry.name.len().min(15)],
                &digest[..8]
            )
        };

        println!(
            "{:<40} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5} {:>6.1}% {:>8.2}",
            name,
            entry.leaves,
            entry.trees,
            reference,
            lb,
            ub,
            gap,
            tightness_pct,
            elapsed / 1000.0
        );
    }

    let total_violations = upper_violations + lower_violations;

    println!("{}", "-".repeat(100));

    if gaps.is_empty() {
        println!("No instances processed.");
        return Ok(());
    }

    gaps.sort_unstable();
    tightness_pcts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let total_gap: usize = gaps.iter().sum();
    let mean_gap = total_gap as f64 / processed as f64;
    let median_gap = gaps[processed / 2];
    let max_gap = *gaps.last().unwrap();
    let min_gap = *gaps.first().unwrap();

    let tight_count = gaps.iter().filter(|&&g| g == 0).count();
    let tight_pct = (tight_count as f64 / processed as f64) * 100.0;

    let mean_tightness: f64 = tightness_pcts.iter().sum::<f64>() / processed as f64;
    let median_tightness = tightness_pcts[processed / 2];
    let min_tightness = tightness_pcts.first().unwrap();

    println!("\n=== SUMMARY ===");
    println!("Total instances:     {}", entries.len());
    println!("Processed:           {}", processed);
    println!("Errors:              {}", errors);
    println!(
        "Total time:          {:.2} ms ({:.3} ms/instance)",
        total_time_ms,
        total_time_ms / processed as f64
    );

    println!("\n--- Violations ---");
    println!("Upper violations:    {} (our UB < best)", upper_violations);
    println!("Lower violations:    {} (our LB > best)", lower_violations);
    println!("Total violations:    {}", total_violations);
    println!(
        "Success rate:        {:.2}%",
        100.0 * (processed - total_violations) as f64 / processed as f64
    );

    println!("\n--- Gap Statistics (UB - Best) ---");
    println!("Total gap:           {}", total_gap);
    println!("Mean gap:            {:.2}", mean_gap);
    println!("Median gap:          {}", median_gap);
    println!("Min gap:             {}", min_gap);
    println!("Max gap:             {}", max_gap);

    println!("\n--- Tightness (LB/Best %) ---");
    println!("Mean tightness:      {:.1}%", mean_tightness);
    println!("Median tightness:    {:.1}%", median_tightness);
    println!("Min tightness:       {:.1}%", min_tightness);
    println!("Tight (gap=0):       {} ({:.1}%)", tight_count, tight_pct);

    println!("\n--- Gap Distribution ---");
    let mut gap_histogram: HashMap<usize, usize> = HashMap::new();
    for &g in &gaps {
        *gap_histogram.entry(g).or_insert(0) += 1;
    }
    let mut sorted_gaps: Vec<_> = gap_histogram.iter().collect();
    sorted_gaps.sort_by_key(|(g, _)| **g);

    let max_count = sorted_gaps.iter().map(|(_, c)| **c).max().unwrap_or(1);
    for (gap, count) in sorted_gaps.iter() {
        let pct = (**count as f64 / processed as f64) * 100.0;
        let bar_len = (**count as f64 / max_count as f64 * 40.0) as usize;
        let bar = "█".repeat(bar_len.max(1));
        println!(
            "  gap={:>2}: {:>4} instances ({:>5.1}%) {}",
            gap, count, pct, bar
        );
    }

    println!("\n--- Tightness Distribution ---");
    let mut tightness_buckets: [usize; 5] = [0; 5];
    for &t in &tightness_pcts {
        if t < 20.0 {
            tightness_buckets[0] += 1;
        } else if t < 40.0 {
            tightness_buckets[1] += 1;
        } else if t < 60.0 {
            tightness_buckets[2] += 1;
        } else if t < 80.0 {
            tightness_buckets[3] += 1;
        } else {
            tightness_buckets[4] += 1;
        }
    }

    let bucket_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"];
    let max_bucket = *tightness_buckets.iter().max().unwrap_or(&1);
    for (i, (label, &count)) in bucket_labels
        .iter()
        .zip(tightness_buckets.iter())
        .enumerate()
    {
        let pct = (count as f64 / processed as f64) * 100.0;
        let bar_len = (count as f64 / max_bucket as f64 * 30.0) as usize;
        let bar = "█".repeat(bar_len.max(1));
        let marker = if i == 4 { " ← tight" } else { "" };
        println!(
            "  {}: {:>4} instances ({:>5.1}%) {}{}",
            label, count, pct, bar, marker
        );
    }

    if total_violations > 0 {
        println!("\n=== ALL VIOLATIONS ===");
        println!(
            "{:<20} {:>6} {:>6} {:>6} {:>8} {:>8}",
            "Instance", "Lower", "Upper", "Best", "Gap", "Type"
        );
        println!("{}", "-".repeat(70));
        for (digest, lower, upper, best, gap, vtype) in &violation_list {
            println!(
                "{:<20} {:>6} {:>6} {:>6} {:>8} {:>8}",
                &digest[..20],
                lower,
                upper,
                best,
                gap,
                vtype
            );
        }
        println!("\nWARNING: Found {} total violations", total_violations);
    } else if processed > 0 {
        println!("\nOK: All bounds valid");
    }

    Ok(())
}
