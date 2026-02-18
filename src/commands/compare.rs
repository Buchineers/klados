//! Compare LP relaxation bounds vs Olver 2-approx bounds.

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
    #[allow(dead_code)]
    our_score: usize,
    leaves: usize,
    trees: usize,
    #[serde(default)]
    name: String,
    path: String,
}

pub fn run(scores_file: &PathBuf, max_leaves: u32) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(scores_file)?;
    let entries: HashMap<String, ScoreEntry> = serde_json::from_str(&content)?;
    let base_dir = scores_file.parent().unwrap_or(std::path::Path::new("."));

    let mut test_entries: Vec<_> = entries
        .iter()
        .filter(|(_, entry)| entry.trees >= 3 && entry.leaves <= max_leaves as usize)
        .collect();
    test_entries.sort_by(|a, b| (a.1.leaves, a.1.trees).cmp(&(b.1.leaves, b.1.trees)));

    println!(
        "Analyzing bounds gaps on {} multi-tree instances (<= {} leaves)...\n",
        test_entries.len(),
        max_leaves
    );
    println!(
        "{:<40} {:>6} {:>6} {:>6} {:>8} {:>8} {:>6} {:>8} {:>8}",
        "Instance", "Leaves", "Trees", "Best", "LB", "Gap", "Gap%", "Tight%", "Time(ms)"
    );
    println!("{}", "-".repeat(110));

    let mut total_time_ms = 0.0f64;
    let mut processed = 0usize;
    let mut errors = 0usize;
    let mut gaps: Vec<usize> = Vec::new();
    let mut tightness_pcts: Vec<f64> = Vec::new();
    let mut gap_histogram: HashMap<usize, usize> = HashMap::new();

    for (digest, entry) in test_entries {
        let full_path = base_dir.join(&entry.path);

        if !full_path.exists() {
            println!(
                "{:<40} {:>6} {:>6} FILE NOT FOUND",
                &digest[..20],
                entry.leaves,
                entry.trees
            );
            errors += 1;
            continue;
        }

        let file_content = match fs::read_to_string(&full_path) {
            Ok(c) => c,
            Err(_) => {
                println!(
                    "{:<40} {:>6} {:>6} READ ERROR",
                    &digest[..20],
                    entry.leaves,
                    entry.trees
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
                    "{:<40} {:>6} {:>6} PARSE ERROR",
                    &digest[..20],
                    entry.leaves,
                    entry.trees
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

        let start = Instant::now();
        let bounds = maf_bounds(&instance.trees, instance.num_leaves);
        let elapsed_ms = start.elapsed().as_micros() as f64 / 1000.0;
        total_time_ms += elapsed_ms;

        let best = entry.best_known;
        let lb = bounds.lower;
        let gap = best.saturating_sub(lb);
        gaps.push(gap);

        let tightness_pct = if best > 0 {
            (lb as f64 / best as f64) * 100.0
        } else {
            100.0
        };
        tightness_pcts.push(tightness_pct);

        let gap_pct = if best > 0 {
            (gap as f64 / best as f64) * 100.0
        } else {
            0.0
        };

        *gap_histogram.entry(gap).or_insert(0) += 1;

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
            "{:<40} {:>6} {:>6} {:>6} {:>8} {:>8} {:>5.0}% {:>7.1}% {:>8.2}",
            name, entry.leaves, entry.trees, best, lb, gap, gap_pct, tightness_pct, elapsed_ms
        );

        processed += 1;
    }

    println!("{}", "-".repeat(110));

    if gaps.is_empty() {
        println!("No instances processed.");
        return Ok(());
    }

    gaps.sort_unstable();
    tightness_pcts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let total_gap: usize = gaps.iter().sum();
    let mean_gap = total_gap as f64 / processed as f64;
    let median_gap = gaps[gaps.len() / 2];
    let max_gap = *gaps.last().unwrap();
    let min_gap = *gaps.first().unwrap();

    let tight_count = gaps.iter().filter(|&&g| g == 0).count();
    let tight_pct = (tight_count as f64 / processed as f64) * 100.0;

    let mean_tightness: f64 = tightness_pcts.iter().sum::<f64>() / processed as f64;
    let median_tightness = tightness_pcts[processed / 2];
    let min_tightness = tightness_pcts.first().unwrap();

    println!("\n=== GAP ANALYSIS SUMMARY ===");
    println!("Instances processed: {}", processed);
    println!("Errors:              {}", errors);
    println!(
        "Total time:          {:.2} ms ({:.2} ms/instance)",
        total_time_ms,
        total_time_ms / processed as f64
    );

    println!("\n--- Gap Statistics ---");
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
    let mut tightness_buckets: [usize; 5] = [0; 5]; // 0-20, 20-40, 40-60, 60-80, 80-100
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

    Ok(())
}
