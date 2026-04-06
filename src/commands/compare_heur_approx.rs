//! Compare simple heuristic approximations against public heuristic best-known scores.

use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;

use klados_core::Tree;
use klados_core::lower_bound::red_blue_approx_detailed;
use pace26io::binary_tree::IndexedBinTreeBuilder;
use pace26io::pace::simplified::Instance as PaceInstance;

#[derive(serde::Deserialize)]
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

pub fn run(
    scores_file: &PathBuf,
    max_leaves: u32,
    limit: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(scores_file)?;
    let pace_entries: Vec<PaceSummaryEntry> = serde_json::from_str(&content)?;
    let base = scores_file.parent().unwrap_or(Path::new("."));

    let mut entries: Vec<_> = pace_entries
        .into_iter()
        .filter(|e| e.track == "heuristic" && e.trees == 2 && e.leaves <= max_leaves as usize)
        .collect();
    entries.sort_by(|a, b| (a.leaves, a.inst_num).cmp(&(b.leaves, b.inst_num)));
    if limit > 0 && entries.len() > limit {
        entries.truncate(limit);
    }

    eprintln!(
        "Comparing Red-Blue UB against {} heuristic 2-tree instances (leaves <= {})",
        entries.len(),
        max_leaves
    );

    println!(
        "{:<8} {:>6} {:>7} {:>7} {:>7} {:>7} {:>8}",
        "Inst", "n", "Best", "RB", "Gap", "Gap%", "RB_ms"
    );
    println!("{}", "-".repeat(64));

    let mut processed = 0usize;
    let mut errors = 0usize;
    let mut ties = 0usize;
    let mut wins = 0usize;
    let mut within_5 = 0usize;
    let mut within_10 = 0usize;
    let mut within_25 = 0usize;
    let mut total_rb_ms = 0.0f64;
    let mut total_best = 0usize;
    let mut total_rb = 0usize;
    let mut gaps: Vec<usize> = Vec::new();
    let mut gap_pcts: Vec<f64> = Vec::new();

    for entry in entries {
        let h = entry.idigest.to_lowercase();
        let path = base
            .join("stride-downloads")
            .join(&h[..2])
            .join(&h[2..4])
            .join(&h[4..]);

        if !path.exists() {
            errors += 1;
            continue;
        }

        let file_content = match fs::read_to_string(&path) {
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

        let t0 = Instant::now();
        let rb = red_blue_approx_detailed(&trees[0], &trees[1]);
        let rb_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let rb_components = rb.ub + 1;
        let best = entry.best_known_score;
        let gap = rb_components.saturating_sub(best);
        let gap_pct = if best > 0 {
            (gap as f64 / best as f64) * 100.0
        } else {
            0.0
        };

        if rb_components < best {
            wins += 1;
        } else if rb_components == best {
            ties += 1;
        }
        if gap_pct <= 5.0 {
            within_5 += 1;
        }
        if gap_pct <= 10.0 {
            within_10 += 1;
        }
        if gap_pct <= 25.0 {
            within_25 += 1;
        }

        total_rb_ms += rb_ms;
        total_best += best;
        total_rb += rb_components;
        gaps.push(gap);
        gap_pcts.push(gap_pct);
        processed += 1;

        println!(
            "{:<8} {:>6} {:>7} {:>7} {:>7} {:>6.1}% {:>8.1}",
            entry.name, entry.leaves, best, rb_components, gap, gap_pct, rb_ms
        );
    }

    println!("{}", "-".repeat(64));

    if processed == 0 {
        println!("No instances processed.");
        return Ok(());
    }

    gaps.sort_unstable();
    gap_pcts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean_gap = gaps.iter().sum::<usize>() as f64 / processed as f64;
    let median_gap = gaps[processed / 2];
    let mean_gap_pct = gap_pcts.iter().sum::<f64>() / processed as f64;
    let median_gap_pct = gap_pcts[processed / 2];
    let mean_rb_ms = total_rb_ms / processed as f64;
    let rb_vs_best_pct = if total_best > 0 {
        total_rb as f64 / total_best as f64 * 100.0
    } else {
        0.0
    };

    println!();
    println!(
        "Processed={} errors={} total_time_ms={:.1} mean_ms={:.1}",
        processed, errors, total_rb_ms, mean_rb_ms
    );
    println!(
        "Mean gap={:.1} median_gap={} mean_gap_pct={:.1}% median_gap_pct={:.1}%",
        mean_gap, median_gap, mean_gap_pct, median_gap_pct
    );
    println!(
        "Aggregate RB/best = {:.1}% | ties={} wins={} within_5%={} within_10%={} within_25%={}",
        rb_vs_best_pct, ties, wins, within_5, within_10, within_25
    );

    Ok(())
}
