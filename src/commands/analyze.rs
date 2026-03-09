//! Analyze STRIDE run results from summary.json files.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(serde::Deserialize, Debug)]
#[allow(dead_code)]
struct SummaryEntry {
    s_key: String,
    s_idigest: String,
    s_num_trees: u32,
    s_num_leaves: u32,
    s_prev_best: Option<u32>,
    s_heuristic_score: Option<f64>,
    s_result: String,
    s_score: Option<u32>,
    s_wtime: Option<f64>,
    s_utime: Option<f64>,
    s_stime: Option<f64>,
    s_maxrss: Option<u64>,
}

pub fn run(summary_file: &PathBuf, top_n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(summary_file)?;
    let mut entries: Vec<SummaryEntry> = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<SummaryEntry>(line) {
            Ok(entry) => entries.push(entry),
            Err(e) => eprintln!("Warning: Failed to parse line: {}", e),
        }
    }

    if entries.is_empty() {
        println!("No entries found in summary file");
        return Ok(());
    }

    println!("=== STRIDE Run Analysis ===\n");
    println!("Summary file: {}", summary_file.display());
    println!("Total entries: {}\n", entries.len());

    // Overall statistics
    let total = entries.len();
    let valid: Vec<_> = entries.iter().filter(|e| e.s_result == "Valid").collect();
    let timeout: Vec<_> = entries.iter().filter(|e| e.s_result == "Timeout").collect();
    let error: Vec<_> = entries
        .iter()
        .filter(|e| !matches!(e.s_result.as_str(), "Valid" | "Timeout"))
        .collect();

    println!("--- Result Distribution ---");
    println!(
        "Valid:     {:>6} ({:.1}%)",
        valid.len(),
        100.0 * valid.len() as f64 / total as f64
    );
    println!(
        "Timeout:   {:>6} ({:.1}%)",
        timeout.len(),
        100.0 * timeout.len() as f64 / total as f64
    );
    println!(
        "Error:     {:>6} ({:.1}%)",
        error.len(),
        100.0 * error.len() as f64 / total as f64
    );

    // Valid entries analysis
    if !valid.is_empty() {
        println!("\n--- Valid Instances ---");

        // Score statistics
        let scores: Vec<u32> = valid.iter().filter_map(|e| e.s_score).collect();
        let total_score: u32 = scores.iter().sum();
        let mean_score = total_score as f64 / valid.len() as f64;
        let max_score = *scores.iter().max().unwrap_or(&0);
        let min_score = *scores.iter().min().unwrap_or(&0);

        println!("Total score: {}", total_score);
        println!("Mean score:  {:.2}", mean_score);
        println!("Score range: {}-{}", min_score, max_score);

        // Optimality check
        let optimal: Vec<_> = valid
            .iter()
            .filter(|e| e.s_prev_best.map_or(false, |best| e.s_score == Some(best)))
            .collect();
        let suboptimal: Vec<_> = valid
            .iter()
            .filter(|e| e.s_prev_best.map_or(false, |best| e.s_score > Some(best)))
            .collect();
        let new_best: Vec<_> = valid
            .iter()
            .filter(|e| e.s_prev_best.map_or(false, |best| e.s_score < Some(best)))
            .collect();
        let no_ref: Vec<_> = valid.iter().filter(|e| e.s_prev_best.is_none()).collect();

        println!("\n--- Optimality ---");
        println!(
            "Optimal:     {:>6} ({:.1}%)",
            optimal.len(),
            100.0 * optimal.len() as f64 / valid.len() as f64
        );
        println!(
            "Suboptimal:  {:>6} ({:.1}%)",
            suboptimal.len(),
            100.0 * suboptimal.len() as f64 / valid.len() as f64
        );
        println!(
            "New best:    {:>6} ({:.1}%)",
            new_best.len(),
            100.0 * new_best.len() as f64 / valid.len() as f64
        );
        println!(
            "No ref:      {:>6} ({:.1}%)",
            no_ref.len(),
            100.0 * no_ref.len() as f64 / valid.len() as f64
        );

        // Time statistics
        let times: Vec<f64> = valid
            .iter()
            .filter_map(|e| e.s_utime.or(e.s_wtime))
            .collect();
        let total_time: f64 = times.iter().sum();
        let mean_time = total_time / valid.len() as f64;
        let max_time = times.iter().cloned().fold(0.0, f64::max);

        let mut sorted_times = times.clone();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p50 = sorted_times[valid.len() / 2];
        let p95 = sorted_times[((valid.len() as f64 * 0.95) as usize).min(valid.len() - 1)];

        println!("\n--- Timing (user time) ---");
        println!("Total:       {:.2}s", total_time);
        println!("Mean:        {:.3}s", mean_time);
        println!("Median:      {:.3}s", p50);
        println!("P95:         {:.3}s", p95);
        println!("Max:         {:.3}s", max_time);

        // Distribution by tree count
        let mut tree_dist: HashMap<u32, usize> = HashMap::new();
        let mut leaf_dist: HashMap<u32, usize> = HashMap::new();
        for e in &valid {
            *tree_dist.entry(e.s_num_trees).or_insert(0) += 1;
            *leaf_dist.entry(e.s_num_leaves).or_insert(0) += 1;
        }

        println!("\n--- Tree Count Distribution ---");
        let mut tree_counts: Vec<_> = tree_dist.iter().collect();
        tree_counts.sort_by_key(|(k, _)| *k);
        for (trees, count) in tree_counts {
            println!(
                "{:>2} trees: {:>6} ({:.1}%)",
                trees,
                count,
                100.0 * *count as f64 / valid.len() as f64
            );
        }

        println!("\n--- Leaf Count Distribution ---");
        let mut leaf_counts: Vec<_> = leaf_dist.iter().collect();
        leaf_counts.sort_by_key(|(k, _)| *k);
        for (leaves, count) in leaf_counts.iter().take(20) {
            println!(
                "{:>3} leaves: {:>6} ({:.1}%)",
                leaves,
                count,
                100.0 * **count as f64 / valid.len() as f64
            );
        }

        // Top N slowest instances
        let mut with_time: Vec<_> = valid
            .iter()
            .filter_map(|e| Some((e, e.s_utime.or(e.s_wtime)?)))
            .collect();
        with_time.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!("\n--- Top {} Slowest Instances ---", top_n);
        println!(
            "{:^32} {:>6} {:>6} {:>8} {:>8}",
            "Digest", "Trees", "Leaves", "Score", "Time(s)"
        );
        println!("{}", "-".repeat(70));
        for (entry, time) in with_time.iter().take(top_n) {
            println!(
                "{:32} {:>6} {:>6} {:>8} {:>8.3}",
                entry.s_idigest,
                entry.s_num_trees,
                entry.s_num_leaves,
                entry.s_score.unwrap_or(0),
                time
            );
        }
    }

    // Error analysis
    if !error.is_empty() {
        println!("\n--- Error Analysis ---");
        let mut error_types: HashMap<String, usize> = HashMap::new();
        for e in &error {
            *error_types.entry(e.s_result.clone()).or_insert(0) += 1;
        }

        let mut sorted_errors: Vec<_> = error_types.iter().collect();
        sorted_errors.sort_by(|a, b| b.1.cmp(a.1));

        for (err_type, count) in sorted_errors.iter().take(10) {
            println!("{:20}: {:>4} instances", err_type, count);
        }
    }

    Ok(())
}
